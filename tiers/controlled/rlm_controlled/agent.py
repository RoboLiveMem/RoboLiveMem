"""Agent track: a frontier model chooses RoboMME multi-choice commands from camera frames, paused or live.

Observation channel (same in both modes): frames rendered by the simulator are sampled at `obs_hz` into a perception
buffer; each decision receives the buffered frames since the previous decision (last `max_new_frames`), each stamped
with its simulated time, plus the current frame with numbered marks on the candidate objects (set-of-marks grounding:
mark k is the object's projected pixel at observation time; the command resolves that pixel against the scene when it
executes). Demonstration-video tasks also get `max_demo_frames` keyframes in the first turn.

Clock: paused charges 0 s per decision; live charges the measured wall-clock latency of the model call (mode "measured")
or a fixed value (mode "fixed", for controlled sweeps). The agent may also choose to wait (observe without acting).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from typing import Optional

import numpy as np

from rlm_controlled.live import LiveEpisode, CONTROL_HZ, _last, render_now, privileged_oracle
from robolivemem.models import parse_json_reply

SYSTEM_BASE = """You control a Franka robot arm in a tabletop simulator through high-level commands.
Each turn you receive camera frames from a fixed front camera, stamped with simulated time in seconds, and the list of
commands available now. Commands that act on an object need a target: the newest frame has numbered marks on the
candidate objects; give the mark number. Marks are renumbered every turn by left-to-right position; they do not follow
object identity, so use your own memory of what happened in earlier frames.
Reply with one JSON object only, no other text:
{"choice": "<command label, or wait>", "mark": <number or null>, "wait": <seconds, only for wait>, "note": "<at most 40 words: what you remember that matters for the rest of the task>"}
Use "choice": "wait" with "wait": s to let s seconds of simulated time pass while you keep observing.
The episode ends when the task succeeds, fails, or the time limit of {budget_s:.0f} s of simulated time runs out."""

LIVE_RULE = """
IMPORTANT: the world does not pause while you think. The time you take to reply passes in the simulator before your
command starts; the robot holds still meanwhile, but other things in the scene keep moving and events keep happening.
Frames recorded while you were thinking are shown at your next turn."""

FRAME_HINT = """
Directions in commands and target names (left/right, forward/backward) are in the robot's frame. The front camera faces
the robot: "forward" moves away from the robot base, toward the camera (down in the image), and the robot's left is on
the right side of the image."""

TIMING_HINT = """
Command timing: when the gripper is already above a button, "press" commands lower it onto the button, and the button
goes down about 1.5 s after the command starts."""

WATCH_RULE = """
Use "choice": "watch" with "wait": s to wait for at most s seconds but get your next turn as soon as something in the
scene changes (an object appears, moves or lights up): useful when you are waiting for something to happen."""

COMMIT_RULE = """
Safety check: a command that acts on a mark is executed only if the image around that mark did not change while you were
thinking; otherwise it is cancelled and you get a new turn with the new frames."""

CHUNK_RULE = """
You may queue up to {n_more} more commands after the first one by adding "then": [{{"choice": ..., "mark": ...}}, ...] to
your reply. Queued commands run one after another as soon as the previous one finishes, without a new turn and without
extra thinking time. A mark in "then" refers to the newest frame of this turn: the command acts on whatever object is
at that marked spot when it starts. The queue stops early if a queued command is not available when its turn comes."""

PAUSED_RULE = """
The simulator is paused while you think: no simulated time passes between the newest frame and the start of your command."""


def draw_marks(img: np.ndarray, marks: list[tuple[int, int, int]]) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    for k, x, y in marks:
        r = 7
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((x - 3 if k < 10 else x - 6, y - 6), str(k), fill=(200, 0, 0))
    return np.asarray(im)


FRAMEDENSE_MAX = 60   # 30 s at 2 Hz: covers a 20 s clock plus the longest command


def _changed(a, b, box=None, thr=40, min_px=20):
    """Pixel-level change between two front frames (a static scene renders identically): at least min_px pixels whose
    largest channel difference exceeds thr, optionally inside box = (x0, y0, x1, y1). Used by the "watch" and "commit"
    harness primitives; it reads only images, never simulator state."""
    a = np.asarray(a, dtype=np.int16)
    b = np.asarray(b, dtype=np.int16)
    if box is not None:
        h, w = a.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    d = np.abs(a - b).max(axis=-1) if a.ndim == 3 else np.abs(a - b)
    return int((d > thr).sum()) >= min_px

LATE_EVENT = {
    "HighlightLate": "some cubes will be highlighted briefly at some point after the button press; watch carefully. ",
    "RevealLate": "the containers will be lifted briefly at some point after the button press, showing the cubes; "
                  "watch carefully. ",
    "InterruptSwap": "after the cubes are covered, someone may swap two containers at any time, also while you are "
                     "deciding; watch carefully. ",
}
# "evwindow": the event's actual schedule (10-20 s after the start, whatever the robot is doing); "after the button
# press" holds when paused but not when a slow clock delays the press past the event
LATE_EVENT_WINDOW = {
    "HighlightLate": "some cubes will be highlighted briefly, once, at a time between 10 s and 20 s after the start "
                     "(whether or not the button has been pressed yet); watch carefully. ",
    "RevealLate": "the containers will be lifted briefly, once, at a time between 10 s and 20 s after the start "
                  "(whether or not the button has been pressed yet), showing the cubes; watch carefully. ",
    "InterruptSwap": "after the cubes are covered, someone may swap two containers at any time, also while you are "
                     "deciding; watch carefully. ",
}

# 2026-11-15 (standard / minimal protocol): what a visible twin's visual cue means. Part of the twin's definition (applied
# with the "twinnote" option, also under "noevent"); says nothing about when or whether anything happens.
TWIN_NOTES = {}


class AgentEpisode:
    def __init__(self, task, ep, client, mode="paused", fixed_latency=0.0, budget_steps=1300, max_env_steps=6000,
                 obs_hz=2.0, max_new_frames=10, max_demo_frames=12, max_images=24, max_decisions=30,
                 wait_cap_s=10.0, dataset="test", sparse_render=True, max_chunk=1, hints=()):
        self.task, self.ep, self.client, self.mode = task, ep, client, mode
        self.fixed_latency = fixed_latency
        self.max_chunk = max(1, int(max_chunk))
        self.hints = tuple(hints or ())
        self.budget_steps = budget_steps
        self.obs_hz, self.max_new_frames, self.max_demo_frames = obs_hz, max_new_frames, max_demo_frames
        self.max_images, self.max_decisions, self.wait_cap_s = max_images, max_decisions, wait_cap_s
        if "demodense" in self.hints:   # demo keyframes are always kept; keep room for the newest frames too
            self.max_images = max_images + 40 - max_demo_frames
        if "framedense" in self.hints:
            # every frame of the 2 Hz grid since the last turn (up to FRAMEDENSE_MAX), so the visual sampling rate does
            # not fall as the clock slows: with 10 frames per turn, a 10 s clock spaces them ~1.5 s apart and a <1 s
            # event (RevealLate's lift, a swap) can fall between two frames
            self.max_new_frames = FRAMEDENSE_MAX
            self.max_images = self.max_images + FRAMEDENSE_MAX - max_new_frames
        self.stride = max(1, int(round(CONTROL_HZ / obs_hz)))
        self.E = LiveEpisode(task, ep, dataset=dataset, budget_steps=None, keep_frames=True, fast=False,
                             max_env_steps=max_env_steps, render_every=self.stride if sparse_render else 1)
        self.turns: list = []       # (role, parts)
        self.log: list = []
        self.cursor = 0             # index into E.frames_front already delivered

    # ------------------------------------------------------------------ marks / candidates
    def _candidates(self):
        """Per option label -> list of (mark, x, y) in the current frame, renumbered by x."""
        from robomme.robomme_env.utils.vqa_options import get_vqa_options
        from robomme.robomme_env.utils.choice_action_mapping import project_world_to_pixel, extract_actor_position_xyz
        mc = self.E._mc_wrapper()
        tgt = {"obj": None, "name": None, "seg_id": None, "position": None, "match_distance": None,
               "selection_mode": None}
        opts = get_vqa_options(mc.env, mc.planner, tgt, self.task)
        out = {}
        for o in opts:
            av = o.get("available")
            if not av:
                continue
            seen, pts = set(), []
            for a in (av if isinstance(av, (list, tuple)) else [av]):
                if id(a) in seen:
                    continue
                seen.add(id(a))
                pos = extract_actor_position_xyz(a)
                px = project_world_to_pixel(pos, mc._front_camera_intrinsic_cv, mc._front_camera_extrinsic_cv,
                                            mc._front_rgb_shape) if pos is not None else None
                if px is not None:
                    pts.append((int(px[0]), int(px[1])))
            pts.sort()
            out[o["label"].lower()] = [(i + 1, x, y) for i, (x, y) in enumerate(pts)]
        return out

    def _resolve(self, label, xy):
        """Analysis only (never shown to the agent): which object a point would select now, and whether it is the
        ground-truth target of the current subgoal."""
        try:
            from robomme.robomme_env.utils.vqa_options import get_vqa_options
            from robomme.robomme_env.utils.choice_action_mapping import project_world_to_pixel, \
                extract_actor_position_xyz
            mc = self.E._mc_wrapper()
            U = self.E.env.unwrapped
            tgt = {"obj": None, "name": None, "seg_id": None, "position": None, "match_distance": None,
                   "selection_mode": None}
            opts = get_vqa_options(mc.env, mc.planner, tgt, self.task)
            opt = next((o for o in opts if o.get("label", "").lower() == label), None)
            av = (opt or {}).get("available") or []
            best, bd = None, 1e18
            for a in (av if isinstance(av, (list, tuple)) else [av]):
                pos = extract_actor_position_xyz(a)
                px = project_world_to_pixel(pos, mc._front_camera_intrinsic_cv, mc._front_camera_extrinsic_cv,
                                            mc._front_rgb_shape) if pos is not None else None
                if px is None:
                    continue
                d = (px[0] - xy[0]) ** 2 + (px[1] - xy[1]) ** 2
                if d < bd:
                    best, bd = a, d
            truth = getattr(U, "current_segment", None)
            if isinstance(truth, (list, tuple)):
                truth = truth[0] if truth else None
            return {"obj": getattr(best, "name", None), "truth": getattr(truth, "name", None),
                    "correct": (best is truth) if (best is not None and truth is not None) else None}
        except Exception as exc:  # analysis must never break a run
            return {"error": repr(exc)[:120]}

    def _mark_objects(self, marks):
        """2026-11-15, analysis only: the object nearest to each mark (by projected pixel) and the current subgoal's
        ground-truth target(s)."""
        try:
            from robomme.robomme_env.utils.vqa_options import get_vqa_options
            from robomme.robomme_env.utils.choice_action_mapping import project_world_to_pixel, \
                extract_actor_position_xyz
            mc = self.E._mc_wrapper()
            U = self.E.env.unwrapped
            tgt = {"obj": None, "name": None, "seg_id": None, "position": None, "match_distance": None,
                   "selection_mode": None}
            pts = []
            for o in get_vqa_options(mc.env, mc.planner, tgt, self.task):
                av = o.get("available") or []
                for a in (av if isinstance(av, (list, tuple)) else [av]):
                    pos = extract_actor_position_xyz(a)
                    px = project_world_to_pixel(pos, mc._front_camera_intrinsic_cv, mc._front_camera_extrinsic_cv,
                                                mc._front_rgb_shape) if pos is not None else None
                    if px is not None:
                        pts.append((float(px[0]), float(px[1]), getattr(a, "name", None)))
            mo = {}
            for k, x, y in marks:
                if pts:
                    mo[int(k)] = min(pts, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)[2]
            truth = getattr(U, "current_segment", None)
            tl = truth if isinstance(truth, (list, tuple)) else [truth]
            return {"mark_objs": mo, "truth": [getattr(t, "name", None) for t in tl if t is not None],
                    "subgoal": getattr(U, "current_task_name", None)}
        except Exception as exc:
            return {"error": repr(exc)[:160]}

    def _obs_record(self):
        """2026-11-15: indices of the images now in the conversation (after trimming), each image stored once with the
        text part that precedes it; save_obs() writes them per episode for the offline probe."""
        st = self.__dict__.setdefault("_obs_store", {"imgs": [], "labels": [], "ids": {}})
        out = []
        for role, ps in self.turns:
            lab = ""
            for p in ps:
                if isinstance(p, str):
                    lab = p if not p.startswith("[image omitted]") else lab
                    continue
                key = id(p)
                if key not in st["ids"]:
                    st["ids"][key] = len(st["imgs"])
                    st["imgs"].append(p)            # keeps a reference, so the id is never reused
                    st["labels"].append(lab[:120])
                out.append(st["ids"][key])
        return out

    def save_obs(self, path):
        st = getattr(self, "_obs_store", None)
        if not st or not st["imgs"]:
            return None
        shapes = sorted({np.asarray(i).shape for i in st["imgs"]})
        arr = {f"shape{j}": np.stack([np.asarray(i, dtype=np.uint8) for i in st["imgs"] if np.asarray(i).shape == sh])
               for j, sh in enumerate(shapes)}
        cnt, where = {}, []
        for i in st["imgs"]:
            j = shapes.index(np.asarray(i).shape)
            where.append([j, cnt.get(j, 0)])
            cnt[j] = cnt.get(j, 0) + 1
        meta = {"labels": st["labels"], "where": where}
        np.savez_compressed(path, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arr)
        return path

    def _changed_since(self, t_obs, box=None):
        """Did the front image (inside box, if given) change after the decision's view at t_obs? Every frame recorded
        since then (2 Hz grid) is compared with that view, and the current one too, so a change that undoes itself
        (two identical containers trading places) is still seen. Images only."""
        ref = getattr(self, "_obs_frame", None)
        if ref is None:
            return False
        E = self.E
        for i in range(len(E.frames_front) - 1, -1, -1):
            s = E.frame_steps[i]
            if s is None or s < 0:
                continue
            if s <= t_obs:
                break
            if _changed(ref, E.frames_front[i], box):
                return True
        return _changed(ref, render_now(E.env.unwrapped), box)

    # ------------------------------------------------------------------ observation packet
    def _t(self, step):
        return (step - self.E.t0) / CONTROL_HZ

    def _new_frames(self):
        E = self.E
        idx = list(range(self.cursor, len(E.frames_front)))
        self.cursor = len(E.frames_front)
        live = [(i, E.frame_steps[i]) for i in idx if E.frame_steps[i] is not None and E.frame_steps[i] >= 0]
        # frames on the render grid (fresh images), one per step, at obs_hz
        sampled, seen = [], set()
        for i, s in live:
            if s % self.stride == 0 and s not in seen:
                sampled.append((i, s))
                seen.add(s)
        if len(sampled) > self.max_new_frames:   # fixed visual budget per turn: spread over the whole interval
            pick = np.linspace(0, len(sampled) - 1, self.max_new_frames).round().astype(int)
            sampled = [sampled[j] for j in pick]
        return sampled

    def _packet(self, first: bool, last_meta: Optional[dict]):
        E = self.E
        parts = []
        if first:
            demo = [i for i, s in enumerate(E.frame_steps) if s == -1]
            full_demo = demo
            demo = [i for k, i in enumerate(demo) if k % self.stride == 0] or demo
            if "demotail" in self.hints and full_demo and demo[-1] != full_demo[-1]:
                # protocol v2: always show the final demonstration frame (the stride grid can miss up to 0.45 s of it)
                demo = demo + [full_demo[-1]]
            if demo:
                # "demodense": every frame of the 2 Hz grid (up to 40) instead of 12 keyframes; 12 keyframes leave 1-2
                # frames per 2.5 s swap of identical cubes (VideoRepick, RepickStream), too few to tell where they went
                nmax = 40 if "demodense" in self.hints else self.max_demo_frames
                pick = np.linspace(0, len(demo) - 1, min(nmax, len(demo))).round().astype(int)
                parts.append(f"Demonstration video ({len(pick)} keyframes, in order):")
                for j in pick:
                    parts.append(E.frames_front[demo[j]])
        frames = self._new_frames()
        now = E.elapsed()
        cands = self._candidates()
        fb = getattr(self, "_feedback", None)
        if fb:
            parts.append(fb)
        if last_meta is not None:
            shown = last_meta.get("charged_s", 0) if self.mode == "fixed" else last_meta.get("latency_s", 0)
            parts.append(f"Your previous reply took {shown:.1f} s of wall-clock time"
                         + (f"; {last_meta.get('charged_s', 0):.1f} s of simulated time passed before your command started."
                            if self.mode != "paused" else "; the simulator was paused meanwhile."))
        frames = [(i, s) for i, s in frames if s < now]
        self._sent_steps = [s - E.t0 for _, s in frames] + [now - E.t0]
        if frames:
            gap = (frames[1][1] - frames[0][1]) / CONTROL_HZ if len(frames) > 1 else 0
            parts.append(f"Frames since your last turn ({len(frames)}"
                         + (f", about {gap:.1f} s apart" if gap else "") + "):")
            for i, s in frames:
                parts.append(f"t={self._t(s):.1f}s")
                parts.append(E.frames_front[i])
        cur = render_now(E.env.unwrapped)
        self._obs_frame = cur          # "commit": the view the decision is based on
        allm = sorted({(x, y) for v in cands.values() for (_, x, y) in v})
        marks = [(i + 1, x, y) for i, (x, y) in enumerate(allm)]
        if "privlog" in self.hints:      # 2026-11-15: analysis only (never shown): which object each mark is on, the truth
            self._priv = self._mark_objects(marks)
        if "cleanframe" in self.hints:
            # protocol v2: the same newest frame without marks, so marks do not hide small objects' colours
            parts.append(f"Newest frame without marks, t={self._t(now):.1f}s:")
            parts.append(cur)
        parts.append(f"Newest frame, t={self._t(now):.1f}s (time left {max(0, self.budget_steps - E.used()) / CONTROL_HZ:.1f}s), with marks:")
        parts.append(draw_marks(cur, marks))
        opts = E.info.get("available_multi_choices") or []
        lines = []
        for o in opts:
            lab = o["label"].lower()
            tag = " (needs a mark)" if o.get("need_parameter") else ""
            lines.append(f"  {lab}: {o['action']}{tag}")
        lines.append("  wait: let time pass and keep observing (give \"wait\": seconds)")
        if "watch" in self.hints:
            lines.append("  watch: wait until something in the scene changes, at most \"wait\" seconds; your next turn "
                         "starts as soon as a change is seen")
        goal = E.task_goal
        if "fullgoal" in self.hints:
            # RoboMME's last paraphrase can drop a step its success check requires: VideoRepick (and RepickStream) with
            # one repeat says "pick up the same cube ... again, finally press the button", but pressing while holding the
            # cube fails; another paraphrase states the put-down, so use that one
            gs = E.info.get("task_goal")
            gs = [g for g in (gs if isinstance(gs, list) else [gs]) if isinstance(g, str)]
            full = [g for g in gs if "put it down" in g]
            if full and "put" not in goal and "place down" not in goal:
                goal = full[0]
            # live-native tasks whose event comes late inherit a goal that does not say an event is coming;
            # PatternLockLate's goal says so ("the buttons will light up one by one"), so say it for these two as well
            # "noevent" (hard split): events are not announced, as in the world; the goal is the task's own
            if "noevent" not in self.hints:
                goal = (LATE_EVENT_WINDOW if "evwindow" in self.hints else LATE_EVENT).get(self.task, "") + goal
        if "twinnote" in self.hints:
            goal = TWIN_NOTES.get(self.task, "") + goal
        parts.append("Task: " + goal + "\nAvailable commands:\n" + "\n".join(lines) + "\nReply with the JSON object.")
        return parts, {m[0]: (m[1], m[2]) for m in marks}

    def _trim_images(self):
        """Keep the demo keyframes and the most recent images within max_images."""
        n = sum(1 for _, ps in self.turns for p in ps if not isinstance(p, str))
        if n <= self.max_images:
            return
        drop = n - self.max_images
        for ti, (role, ps) in enumerate(self.turns):
            if drop <= 0:
                break
            if ti == 0:
                # first turn: drop its non-demo frames first, keep demo keyframes
                pass
            new = []
            in_demo = ti == 0 and any(isinstance(p, str) and p.startswith("Demonstration video") for p in ps)
            demo_done = False
            for p in ps:
                if isinstance(p, str):
                    if p.startswith("Frames since") or p.startswith("Newest frame"):
                        demo_done = True
                    new.append(p)
                    continue
                if in_demo and not demo_done:
                    new.append(p)
                    continue
                if drop > 0:
                    drop -= 1
                    new.append("[image omitted]")
                else:
                    new.append(p)
            self.turns[ti] = (role, new)

    # ------------------------------------------------------------------ main loop
    def run(self):
        E = self.E
        t_wall = time.time()
        E.reset()
        system = SYSTEM_BASE.replace("{budget_s:.0f}", f"{self.budget_steps / CONTROL_HZ:.0f}") + (
            LIVE_RULE if self.mode != "paused" else PAUSED_RULE) + (
            CHUNK_RULE.format(n_more=self.max_chunk - 1) if self.max_chunk > 1 else "") + (
            FRAME_HINT if "frame" in self.hints else "") + (
            TIMING_HINT if "timing" in self.hints else "")
        if "watch" in self.hints:
            system = system.replace('"<command label, or wait>"', '"<command label, wait or watch>"').replace(
                "<seconds, only for wait>", "<seconds, only for wait or watch>") + WATCH_RULE
        if "commit" in self.hints and self.mode != "paused":
            system += COMMIT_RULE
        last_meta = None
        first = True
        for k in range(self.max_decisions):
            if E.used() >= self.budget_steps:
                E.status = "timeout"
                break
            parts, markmap = self._packet(first, last_meta)
            first = False
            self.turns.append(("user", parts))
            self._trim_images()
            ctx_imgs = self._obs_record() if getattr(self, "save_obs_path", None) else None
            t_obs = E.elapsed()
            U_ = E.env.unwrapped
            injected = U_.rlm_on_observe(t_obs) if hasattr(U_, "rlm_on_observe") else None   # InterruptSwap
            oracle_obs = _oracle_choice(E)        # logged for attribution only; never shown to the agent
            text, meta = self.client(system, self.turns)
            reply = parse_json_reply(text)
            self.turns.append(("assistant", [text if text else "{}"]))
            lat = meta.get("latency_s", 0.0) or 0.0
            charged = 0.0 if self.mode == "paused" else (self.fixed_latency if self.mode == "fixed" else lat)
            meta["charged_s"] = charged
            entry = {"k": k, "t_obs": t_obs - E.t0, "reply": text[-600:], "parsed": reply, "meta": meta,
                     **({"priv": getattr(self, "_priv", None)} if "privlog" in self.hints else {}),
                     **({"ctx_imgs": ctx_imgs} if ctx_imgs is not None else {}),
                     "frames_sent": getattr(self, "_sent_steps", []),
                     "subgoal_truth": getattr(E.env.unwrapped, "current_task_name", None), "oracle_obs": oracle_obs,
                     "injected_at": (injected - E.t0) if injected is not None else None}
            choice = (reply or {}).get("choice")
            if isinstance(choice, str):
                choice = choice.strip().lower().rstrip(":.)")
            opts = {o["label"].lower(): o for o in (E.info.get("available_multi_choices") or [])}
            special = ("wait", "watch") if "watch" in self.hints else ("wait",)
            if isinstance(choice, str) and choice not in opts and choice not in special:
                # accept the command text itself (e.g. "move forward") or "label: text" in place of the label
                byact = {(o.get("action") or "").lower().strip(): l for l, o in opts.items()}
                head = choice.split(":", 1)[0].strip()
                choice = byact.get(choice.strip(), head if head in opts else choice)
            mk = (reply or {}).get("mark")
            try:
                mk = int(mk) if mk is not None else None
            except Exception:
                mk = None
            self._feedback = None
            if not reply or not choice or (choice not in special and choice not in opts):
                entry["action"] = "invalid"
                self._feedback = ("Your last reply was not a valid command (reply with one JSON object whose "
                                  "choice is one of the listed labels or wait). Nothing was executed.")
                E.hold(int(math.ceil(charged * CONTROL_HZ)))
            elif choice == "watch":
                # event-triggered wait: after the thinking time, poll the front camera on the 2 Hz grid and return
                # as soon as the image changes (or after the given time)
                w = float(reply.get("wait") or 5.0)
                w = max(0.5, min(self.wait_cap_s, w))
                E.hold(int(math.ceil(charged * CONTROL_HZ)))
                U_w = E.env.unwrapped
                # compared with the view the decision was based on, so a change during the thinking time counts too
                ref = getattr(self, "_obs_frame", None)
                if ref is None:
                    ref = render_now(U_w)
                n_total, waited, seen = int(math.ceil(w * CONTROL_HZ)), 0, None
                if E.status == "ongoing" and self._changed_since(t_obs):
                    seen = E.elapsed()
                    n_total = 0
                while waited < n_total and E.status == "ongoing":
                    n = min(self.stride, n_total - waited)
                    E.hold(n)
                    waited += n
                    if _changed(ref, render_now(U_w)):
                        seen = E.elapsed()
                        break
                entry["action"] = f"watch {w:.1f}s"
                entry["wait_s"] = waited / CONTROL_HZ
                entry["watch_change_at"] = (seen - E.t0) if seen is not None else None
                self._feedback = (f"You watched for {waited / CONTROL_HZ:.1f} s; " +
                                  (f"a change was seen at t={self._t(seen):.1f}s." if seen is not None
                                   else "nothing changed."))
            elif choice != "wait" and opts[choice].get("need_parameter") and (mk is None or mk not in markmap):
                entry["action"] = "invalid: missing mark"
                self._feedback = (f"Command {choice} needs a target mark from the newest frame; none was given, "
                                  "so nothing was executed.")
                E.hold(int(math.ceil(charged * CONTROL_HZ)))
            elif choice == "wait":
                w = float(reply.get("wait") or 1.0)
                w = max(0.5, min(self.wait_cap_s, w))
                entry["action"] = f"wait {w:.1f}s"
                entry["wait_s"] = w
                E.hold(int(math.ceil((charged + w) * CONTROL_HZ)))
                self._feedback = f"You waited {w:.1f} s."
            else:
                cmd = {"choice": choice.upper()}
                if mk is not None and mk in markmap and opts[choice].get("need_parameter"):
                    x, y = markmap[mk]
                    cmd["point"] = [y, x]
                entry["action"] = cmd
                if "point" in cmd:
                    entry["target_at_obs"] = self._resolve(choice, (x, y))
                d = int(math.ceil(charged * CONTROL_HZ))
                E.hold(d)                                   # the world runs while the agent was thinking
                if "point" in cmd and E.status == "ongoing":
                    entry["target_at_exec"] = self._resolve(choice, (x, y))
                    a, b = entry["target_at_obs"].get("correct"), entry["target_at_exec"].get("correct")
                    entry["stale"] = bool(a and b is False)
                entry["oracle_exec"] = _oracle_choice(E)
                if ("commit" in self.hints and "point" in cmd and d > 0 and E.status == "ongoing"
                        and self._changed_since(t_obs, (x - 24, y - 24, x + 24, y + 24))):
                    # commit check: the image around the chosen mark changed while the agent was deciding, so the
                    # command may act on a stale view; it is not executed and the agent decides again
                    entry["action"] = {"aborted": cmd}
                    entry["commit_abort"] = True
                    self._feedback = (f"Your command ({choice} on mark {mk}) was NOT executed: the scene near mark {mk} "
                                      "changed while you were deciding. Look at the new frames and decide again.")
                    entry["status"] = E.status
                    self.log.append(entry)
                    last_meta = meta
                    if E.status != "ongoing":
                        break
                    continue
                rec = E.act(cmd, 0.0, entry["subgoal_truth"])
                rec.t_decide = t_obs
                entry["t_exec"] = rec.t_exec - E.t0
                entry["t_end"] = rec.t_end - E.t0
                entry["subgoal_after"] = getattr(E.env.unwrapped, "current_task_name", None)
                act_txt = opts[choice].get("action", choice) + (f" on mark {mk}" if "point" in cmd else "")
                self._feedback = (f"Your last command ({choice}: {act_txt}) started at "
                                  f"t={(rec.t_exec - E.t0) / CONTROL_HZ:.1f}s and finished at "
                                  f"t={(rec.t_end - E.t0) / CONTROL_HZ:.1f}s.")
                if self.max_chunk > 1 and isinstance((reply or {}).get("then"), list) and E.status == "ongoing":
                    entry["status"] = E.status
                    self.log.append(entry)
                    entry = self._run_queue(k, reply["then"][: self.max_chunk - 1], markmap, entry)
            entry["status"] = E.status
            self.log.append(entry)
            last_meta = meta
            if E.status != "ongoing":
                break
        if E.status == "ongoing":
            E.status = "timeout" if E.used() >= self.budget_steps else "max_decisions"
        U = E.env.unwrapped
        events = [(int(a) - E.t0, int(b) - E.t0, kk) for a, b, kk in U.rlm_events()] if hasattr(U, "rlm_events") else []
        res = {"task": self.task, "ep": self.ep, "mode": self.mode, "fixed_latency": self.fixed_latency,
               "events": events,
               "model": getattr(self.client, "short", type(self.client).__name__), "status": E.status,
               "success": E.status == "success", "steps": E.used(), "within_budget": E.used() <= self.budget_steps,
               "n_calls": len(self.log), "sum_latency_s": round(sum((l["meta"].get("latency_s") or 0) for l in self.log), 2),
               "goal": E.task_goal, "wall_s": round(time.time() - t_wall, 1), "log": self.log}
        return res

    def _run_queue(self, k, queue, markmap, first_entry):
        """Execute queued commands of decision k back to back (no latency charge). Each executed command is logged as
        its own entry with "chunk": j (j >= 1) and charged_s 0, so replay.py re-applies it unchanged. Returns the last
        entry (not yet appended); the caller appends it."""
        E = self.E
        fb = [self._feedback]
        last = None
        for j, q in enumerate(queue, start=1):
            if E.status != "ongoing" or not isinstance(q, dict):
                break
            ch = q.get("choice")
            ch = ch.strip().lower().rstrip(":.)") if isinstance(ch, str) else None
            opts = {o["label"].lower(): o for o in (E.info.get("available_multi_choices") or [])}
            if isinstance(ch, str) and ch not in opts:
                byact = {(o.get("action") or "").lower().strip(): l for l, o in opts.items()}
                ch = byact.get(ch.strip(), ch)
            try:
                mk = int(q.get("mark")) if q.get("mark") is not None else None
            except Exception:
                mk = None
            if ch not in opts or (opts[ch].get("need_parameter") and (mk is None or mk not in markmap)):
                fb.append(f"Queued command {j} ({q.get('choice')}) was not available when its turn came; the queue stopped.")
                break
            if last is not None:
                last["status"] = E.status
                self.log.append(last)
            cmd = {"choice": ch.upper()}
            ent = {"k": k, "chunk": j, "t_obs": first_entry["t_obs"], "reply": "", "parsed": q,
                   "meta": {"latency_s": 0.0, "charged_s": 0.0}, "frames_sent": [],
                   "subgoal_truth": getattr(E.env.unwrapped, "current_task_name", None), "oracle_obs": _oracle_choice(E)}
            if mk is not None and opts[ch].get("need_parameter"):
                x, y = markmap[mk]
                cmd["point"] = [y, x]
                ent["target_at_exec"] = self._resolve(ch, (x, y))
            ent["action"] = cmd
            ent["oracle_exec"] = ent["oracle_obs"]
            rec = E.act(cmd, 0.0, ent["subgoal_truth"])
            rec.t_decide = first_entry["t_obs"]
            ent["t_exec"], ent["t_end"] = rec.t_exec - E.t0, rec.t_end - E.t0
            ent["subgoal_after"] = getattr(E.env.unwrapped, "current_task_name", None)
            act_txt = opts[ch].get("action", ch) + (f" on mark {mk}" if "point" in cmd else "")
            fb.append(f"Queued command {j} ({ch}: {act_txt}) ran from t={(rec.t_exec - E.t0) / CONTROL_HZ:.1f}s to "
                      f"t={(rec.t_end - E.t0) / CONTROL_HZ:.1f}s.")
            last = ent
        self._feedback = " ".join(fb)
        if last is None:          # nothing queued ran: the first entry was already appended; return a no-op marker
            self.log.pop()
            return first_entry
        return last

    def save_video(self, path, fps=20, stride=2):
        import imageio.v2 as imageio
        fr = [f for f, s in zip(self.E.frames_front, self.E.frame_steps) if s is not None and s >= 0][::stride]
        if fr:
            imageio.mimsave(path, fr, fps=fps // stride if fps // stride > 0 else 10)


def _oracle_choice(E):
    try:
        c = privileged_oracle(E)
        return c.get("choice") if c else None
    except Exception:
        return None


class OracleClient:
    """Plumbing test: a 'model' that returns the privileged answer after a simulated delay (no API)."""
    short = "oracle-sim"

    def __init__(self, episode_ref, sim_latency=0.0):
        self.ref = episode_ref
        self.sim_latency = sim_latency

    def __call__(self, system, turns):
        A = self.ref
        cmd = privileged_oracle(A.E)
        meta = {"latency_s": self.sim_latency}
        if cmd is None:
            return json.dumps({"choice": "wait", "wait": 1}), meta
        out = {"choice": cmd["choice"].lower(), "mark": None}
        if "point" in cmd:
            # nearest mark to the oracle point
            y, x = cmd["point"]
            cands = A._candidates()
            allm = sorted({(cx, cy) for v in cands.values() for (_, cx, cy) in v})
            if allm:
                d = [((cx - x) ** 2 + (cy - y) ** 2, i + 1) for i, (cx, cy) in enumerate(allm)]
                out["mark"] = min(d)[1]
        return json.dumps(out), meta
