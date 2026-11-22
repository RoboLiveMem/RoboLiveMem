"""Run agents on RoboLiveMem part H (ESE household scenes).

  python run_lm2.py --jobs jobs.jsonl --shard 0/8 --out results/x_s0.jsonl --media media/x
A job: {"family", "member", "seed", "clock": paused|live|fixed, "L": seconds (fixed), "policy": model short name or
scripted:oracle / scripted:naive, "diag": none|look|oracle, "frames": buffer|latest, "tag"}.

One decision = one model call, stateless: the task, the text history (commands, results, the agent's own notes), the
camera frames recorded since the previous decision (2 Hz, at most --max-frames, evenly thinned, newest kept) and the
current front image with numbered marks on what can be seen and acted on. Clock: paused (the world waits), live (the
world runs for the call's measured latency before the command runs), fixed (it runs for L). Diagnostics on a clock
with time: `look` asks again after the think time with a fresh image (no time charged for the second answer);
`oracle` does the same and adds one privileged line stating the true current value of what the task needs remembered.

2026-11-06: job field "protocol": "orig" (default; livemem2.py, unchanged behaviour) or "strict" (livemem2_strict.py:
change at a world time after hand-over while the robot stands still, event / look-alike / null seeds by seed % 3, optional
static person, strict heat scoring). --protocol sets the default for jobs without the field. Strict scripted policies:
scripted:oracle (privileged state at decision time, acts after the think time without re-checking = stale command with
perfect memory), scripted:naive (frame-blind stale reference: current image + its own earlier decisions only),
scripted:aware (latency-aware: privileged, waits while a person's motion is pending or running), scripted:always (assumes
the change always happened; mugs / heat / apple only).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import livemem2 as LM  # noqa: E402

SYSTEM = """You control a one-arm robot (a UR5e with a two-finger gripper) at a kitchen counter, through a camera that \
looks at the counter from the front. People work in the same kitchen and may move or change things at any time, also \
while you are thinking: the time you take to answer may pass in the kitchen before your command runs.

Each decision you get: your task; the commands you gave so far, their results and your notes; the camera frames recorded \
since your previous decision (small images, two per second of world time, oldest first; they show what happened while you \
were deciding or while the robot moved); and the current camera image, with numbered white marks on things you can act on.

Commands:
- pick: grasp the thing at a mark (the hand must be empty)
- place: put what the hand holds on or at a mark
- press: press the button at a mark
- wait: hold still and keep watching for some seconds (0.5 to 5)
- done: the task is finished (the episode ends)

Reply with ONE JSON object and nothing else:
{"note": "<short: what you need to remember>", "tool": "pick" | "place" | "press" | "wait" | "done", "mark": <a mark number in the CURRENT image, for pick, place and press>, "seconds": <for wait>}"""

REASK = ("Before your command runs, here is a fresh camera image of the scene as it is now (the marks are renumbered). "
         "Keep your command or change it.")


def thin(frames, k):
    if len(frames) <= k:
        return frames
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, k)).astype(int))
    return [frames[i] for i in idx]


def held_text(ep):
    h = ep.held()
    if h is not None and h in getattr(ep, "lm_held_names", {}):      # 2026-11-06: e.g. the SUGAR jar (kind salt_jar)
        return "the " + ep.lm_held_names[h]
    return "nothing" if h is None else ("a " + LM.kind_of(ep, h))


def result_text(out):
    if not out.get("ok"):
        return "failed: " + str(out.get("error", ""))[:160]
    r = out.get("result") or {}
    bits = []
    if isinstance(r, dict):
        st = r.get("status")
        if st:
            bits.append(str(st))
        if r.get("reason"):
            bits.append(str(r["reason"])[:80])
        if "holding" in r:
            bits.append("holding something" if r["holding"] else "hand empty")
    return ", ".join(bits) or "ok"


def marks_text(marks):
    return "; ".join(f"{m['id']} = {m['label']}" for m in marks) or "(no marks)"


def build_turn(task, k, ep, hist, frames, cur_img, marks, extra=None, reask=False):
    parts = [f"Task: {task}\n\nDecision {k}. World time now: {ep.world.time:.1f} s. The robot's hand holds {held_text(ep)}.\n"
             f"Your commands so far (oldest first):\n" + ("\n".join(hist) if hist else "(none yet)")]
    if reask:
        parts.append(REASK)
    elif frames:
        parts.append(f"Camera frames since your previous decision ({len(frames)}, world time {frames[0][0]:.1f} s to {frames[-1][0]:.1f} s):")
        for t, img in frames:
            parts.append(f"t = {t:.1f} s")
            parts.append(img)
    if extra:
        parts.append(extra)
    parts.append(f"Current camera image (world time {ep.world.time:.1f} s). Marks: {marks_text(marks)}.")
    parts.append(LM.draw_marks(cur_img, marks))
    parts.append("Reply with the JSON object.")
    return parts


# ------------------------------------------------------------------------------------------------- policies
class ModelPolicy:
    def __init__(self, short, max_tokens=None):
        from robolivemem.models import Chat as BedrockChat
        mt = 20000 if "luna" in short else (4096 if short.startswith("gpt") else 1024)
        if "think" in short:
            mt = 16000
        if max_tokens:                      # 2026-11-15: job field "max_tokens" (main tables: Claude 4096, Astra 16000)
            mt = int(max_tokens)
        self.chat = BedrockChat(short, max_tokens=mt, timeout_s=300.0)
        self.short = short

    def decide(self, parts, ctx):
        from robolivemem.models import parse_json_reply
        tries, lat_total = 0, 0.0
        while True:
            tries += 1
            text, meta = self.chat(SYSTEM + (("\n\n" + self.hint) if getattr(self, "hint", "") else ""), [("user", parts)])
            lat_total += float(meta.get("latency_s") or 0.0)
            if meta.get("error") and tries < 4 and ("throttl" in str(meta.get("error")).lower() or meta.get("http_status") in (429, 500, 502, 503, 504, None)):
                time.sleep(3 * tries + random.random() * 3)
                lat_total = 0.0        # a refused request is not charged to the clock (the service did not answer)
                continue
            break
        cmd = parse_json_reply(text) if text else None
        meta = {k: meta.get(k) for k in ("latency_s", "in_tokens", "out_tokens", "reasoning_tokens", "stop", "error", "http_status")}
        meta["tries"] = tries
        return cmd, text, meta


class ScriptedPolicy:
    """oracle: privileged, true state at decision time. naive: no memory of what happened (beliefs from the start or
    from the current image only)."""

    def __init__(self, kind):
        self.kind = kind
        self.rng = random.Random(0)

    def decide(self, parts, ctx):
        sc, ep, marks = ctx["sc"], ctx["ep"], ctx["marks"]
        cmd = getattr(self, "_" + sc.name)(sc, ep, marks, ctx)
        return cmd, json.dumps(cmd), {"latency_s": 0.0}

    @staticmethod
    def mark(marks, obj, label=None):
        for m in marks:
            if m["obj"] == obj and (label is None or m["label"] == label):
                return m["id"]
        return None

    def _mugs(self, sc, ep, marks, ctx):
        h = ep.held()
        tray = self.mark(marks, "tray")
        spot = self.mark(marks, "table")
        if self.kind == "oracle":
            if h == "mug_a":
                return {"tool": "place", "mark": tray}
            if h == "mug_b":
                return {"tool": "place", "mark": spot}
            if sc.success(ep):
                return {"tool": "done"}
            m = self.mark(marks, "mug_a")
            return {"tool": "pick", "mark": m} if m else {"tool": "wait", "seconds": 1}
        # naive: Anna's mug is the one at the left start position (by pixel), or the red one when Anna's is red
        if h is not None:
            if ctx["mem"].get("placed"):
                return {"tool": "done"}
            ctx["mem"]["placed"] = True
            return {"tool": "place", "mark": tray}
        if ctx["mem"].get("placed"):
            return {"tool": "done"}
        mugs = [m for m in marks if m["label"] == "mug"]
        if sc.member == "visible":
            m = self.mark(marks, "mug_a")
        else:
            pl = ep.world.project("front", sc.L + np.array([0, -0.02, 0.07]))
            m = min(mugs, key=lambda q: abs(q["px"][0] - pl[0]) + abs(q["px"][1] - pl[1]))["id"] if mugs else None
        return {"tool": "pick", "mark": m}

    def _knife(self, sc, ep, marks, ctx):
        h, w = ep.held(), sc.where(ep)
        plate, km = self.mark(marks, "plate"), self.mark(marks, "knife")
        if h == "knife":
            return {"tool": "place", "mark": plate}
        if h in ("cover_l", "cover_r"):
            return {"tool": "place", "mark": self.mark(marks, "table")}
        if w == "plate":
            return {"tool": "done"}
        if km:
            return {"tool": "pick", "mark": km}
        if self.kind == "oracle":
            c = w if w.startswith("cover") else None
        else:                      # naive: no memory of where it went: lift the covers in a random order
            tried = ctx["mem"].setdefault("tried", [])
            left = [c for c in ("cover_l", "cover_r") if c not in tried]
            c = self.rng.choice(left) if left else None
            if c:
                tried.append(c)
        if c is None:
            return {"tool": "wait", "seconds": 1}
        return {"tool": "pick", "mark": self.mark(marks, c)}

    def _apple(self, sc, ep, marks, ctx):
        h = ep.held()
        if h == "jar":
            return {"tool": "place", "mark": self.mark(marks, "tray")}
        if h in ("cover_a", "cover_b"):
            return {"tool": "place", "mark": self.mark(marks, "table")}
        if h == "apple":
            return {"tool": "place", "mark": self.mark(marks, "bowl")}
        if sc.success(ep):
            return {"tool": "done"}
        jt = sc.on(ep, "jar", sc.tray, 0.13, 0.11)
        if not jt:
            return {"tool": "pick", "mark": self.mark(marks, "jar")}
        am = self.mark(marks, "apple")
        if am:
            return {"tool": "pick", "mark": am}
        if self.kind == "oracle":
            c = sc.under(ep)
        else:                      # naive: the apple is under the cover that stands where the left one stood at the start
            pl = ep.world.project("front", sc.L + np.array([0, 0, 0.15]))
            covers = [m for m in marks if m["obj"] in ("cover_a", "cover_b")]
            c = min(covers, key=lambda q: abs(q["px"][0] - pl[0]))["obj"] if covers else None
            if sc.member == "visible":
                c = "cover_a"
        return {"tool": "pick", "mark": self.mark(marks, c)} if c else {"tool": "wait", "seconds": 1}

    def _heat(self, sc, ep, marks, ctx):
        h = ep.held()
        if sc.success(ep):
            return {"tool": "done"}
        if h is None:
            return {"tool": "pick", "mark": self.mark(marks, "bowl")}
        if self.kind == "oracle":
            hot = sc.heated
        else:                      # naive: hot only if the plate looks on in the current image
            hot = bool(ep.state.devices["hotplate"].state.get("on"))
        return {"tool": "place", "mark": self.mark(marks, "tray" if hot else "hotplate")}

    def _point(self, sc, ep, marks, ctx):
        h = ep.held()
        tgt = f"jar{sc.target}"
        if sc.success(ep):
            return {"tool": "done"}
        if h is not None:
            return {"tool": "place", "mark": self.mark(marks, "board")}
        if self.kind == "oracle":
            known = sc.member == "static" or ep.world.time >= sc.cue_t
        else:                      # naive: knows only what the current image shows (the finger over a jar now)
            t = ep.world.time - sc.cue_t
            known = ctx["mem"].get("known") or (sc.member == "visible" and t >= 0) or (0 <= t < sc.hold_s)
            ctx["mem"]["known"] = known
            if not known and ep.world.time > 8.0:
                tgt = f"jar{self.rng.randrange(3)}"
                known = True
        if not known:
            return {"tool": "wait", "seconds": 1}
        return {"tool": "pick", "mark": self.mark(marks, tgt)}


class StrictScriptedPolicy(ScriptedPolicy):
    """Scripted references for the strict protocol (see the module docstring)."""

    def decide(self, parts, ctx):
        sc, ep = ctx["sc"], ctx["ep"]
        kind = self.kind
        if kind == "aware":
            if sc.name == "point":
                if not sc.cue_happened(ep) and not sc.success(ep):
                    return self._w(1.0)
            elif self.person_pending(sc, ep, float(ctx.get("L", 0.0))):
                return self._w(3.0)
            kind = "oracle"
        cmd = getattr(self, "_" + sc.name + "_" + kind, None)
        if cmd is None and kind in ("oracle", "naive"):
            cmd = getattr(self, "_" + sc.name)
            self_kind = self.kind
            self.kind = kind
            try:
                out = cmd(sc, ep, ctx["marks"], ctx)
            finally:
                self.kind = self_kind
        elif cmd is None:
            out = {"tool": "done"}
        else:
            out = cmd(sc, ep, ctx["marks"], ctx)
        return out, json.dumps(out), {"latency_s": 0.0}

    @staticmethod
    def _w(s):
        out = {"tool": "wait", "seconds": s}
        return out, json.dumps(out), {"latency_s": 0.0}

    @staticmethod
    def person_pending(sc, ep, L):
        if getattr(sc, "kind", "none") == "none" or sc.done:
            return False
        if sc.busy(ep):
            return True
        return sc.t0 is None and sc.t_due is not None and sc.ready(ep) and sc.t_due <= ep.world.time + L + 0.3

    def _point_oracle(self, sc, ep, marks, ctx):
        h = ep.held()
        if sc.success(ep):
            return {"tool": "done"}
        if h is not None:
            return {"tool": "place", "mark": self.mark(marks, "board")}
        if not sc.cue_happened(ep):
            return {"tool": "wait", "seconds": 1}
        return {"tool": "pick", "mark": self.mark(marks, f"jar{sc.target}")}

    def _point_naive(self, sc, ep, marks, ctx):
        h = ep.held()
        if sc.success(ep):
            return {"tool": "done"}
        if h is not None:
            return {"tool": "place", "mark": self.mark(marks, "board")}
        mem = ctx["mem"]
        if mem.get("tgt") is None and sc.cue_visible_now(ep):
            mem["tgt"] = sc.target                    # the finger is over this jar in the current image
        if mem.get("tgt") is None:
            if ep.world.time > (sc.t_ho or 0.0) + 8.0:
                mem["tgt"] = self.rng.randrange(3)
            else:
                return {"tool": "wait", "seconds": 1}
        return {"tool": "pick", "mark": self.mark(marks, f"jar{mem['tgt']}")}

    # sugar (round 2): count of cubes in the tea; oracle = privileged count at decision time, naive = what the current
    # image shows (0 for the hidden member: dissolved cubes; the cubes on the saucer for the visible twin)
    @staticmethod
    def _spot(marks, ep, p):
        sp = [m for m in marks if m["obj"] == "table"]
        if not sp:
            return None
        q = ep.world.project("front", p)
        if q is None:
            return sp[0]["id"]
        return min(sp, key=lambda m: abs(m["px"][0] - q[0]) + abs(m["px"][1] - q[1]))["id"]

    def _sugar_policy(self, sc, ep, marks, n):
        h = ep.held()
        ready = n >= sc.NEED
        if h == "jar":
            return {"tool": "place", "mark": self._spot(marks, ep, sc.far if ready else sc.near)}
        if h == "tea":
            return {"tool": "place", "mark": self.mark(marks, "tray") if ready else self._spot(marks, ep, sc.far)}
        if sc.success(ep):
            return {"tool": "done"}
        if ready:
            if sc.tea_served(ep):
                return {"tool": "done"}
            return {"tool": "pick", "mark": self.mark(marks, "tea")}
        if sc.jar_near_tea(ep):
            return {"tool": "done"}
        return {"tool": "pick", "mark": self.mark(marks, "jar")}

    def _sugar(self, sc, ep, marks, ctx):
        if self.kind == "oracle":
            return self._sugar_policy(sc, ep, marks, sc.count)
        return self._sugar_policy(sc, ep, marks, sc.visible_count(ep))

    def _sugar_always(self, sc, ep, marks, ctx):
        return self._sugar_policy(sc, ep, marks, sc.NEED)

    # "always": assumes the change happened in every episode (frame-blind otherwise)
    def _mugs_always(self, sc, ep, marks, ctx):
        h, mem = ep.held(), ctx["mem"]
        if h is not None:
            if mem.get("placed"):
                return {"tool": "done"}
            mem["placed"] = True
            return {"tool": "place", "mark": self.mark(marks, "tray")}
        if mem.get("placed"):
            return {"tool": "done"}
        if sc.member == "visible":
            return {"tool": "pick", "mark": self.mark(marks, "mug_a")}
        mugs = [m for m in marks if m["label"] == "mug"]
        pr = ep.world.project("front", sc.R + np.array([0, -0.02, 0.07]))
        m = min(mugs, key=lambda q: abs(q["px"][0] - pr[0]) + abs(q["px"][1] - pr[1]))["id"] if mugs else None
        return {"tool": "pick", "mark": m}

    def _heat_always(self, sc, ep, marks, ctx):
        if sc.success(ep):
            return {"tool": "done"}
        if ep.held() is None:
            if ctx["mem"].get("placed"):
                return {"tool": "done"}
            return {"tool": "pick", "mark": self.mark(marks, "bowl")}
        ctx["mem"]["placed"] = True
        return {"tool": "place", "mark": self.mark(marks, "tray")}

    def _apple_always(self, sc, ep, marks, ctx):
        h = ep.held()
        if h == "jar":
            return {"tool": "place", "mark": self.mark(marks, "tray")}
        if h in ("cover_a", "cover_b"):
            return {"tool": "place", "mark": self.mark(marks, "table")}
        if h == "apple":
            return {"tool": "place", "mark": self.mark(marks, "bowl")}
        if sc.success(ep):
            return {"tool": "done"}
        if not sc.on(ep, "jar", sc.tray, 0.13, 0.11):
            return {"tool": "pick", "mark": self.mark(marks, "jar")}
        am = self.mark(marks, "apple")
        if am:
            return {"tool": "pick", "mark": am}
        if ctx["mem"].get("lifted_once"):
            return {"tool": "done"}
        ctx["mem"]["lifted_once"] = True
        pr = ep.world.project("front", sc.R + np.array([0, 0, 0.15]))
        covers = [m for m in marks if m["obj"] in ("cover_a", "cover_b")]
        c = min(covers, key=lambda q: abs(q["px"][0] - pr[0]))["obj"] if covers else None
        if sc.member == "visible":
            c = "cover_b"
        return {"tool": "pick", "mark": self.mark(marks, c)} if c else {"tool": "wait", "seconds": 1}


# ------------------------------------------------------------------------------------------------- one episode
def to_call(cmd, marks):
    tool = cmd.get("tool") if isinstance(cmd, dict) else None
    if tool == "wait":
        try:
            s = float(cmd.get("seconds", 1.0))
        except (TypeError, ValueError):
            s = 1.0
        return {"tool": "wait", "args": {"seconds": min(max(s, 0.5), 5.0)}}, None
    if tool in ("pick", "place", "press"):
        try:
            mid = int(cmd.get("mark"))
        except (TypeError, ValueError):
            return None, "no valid mark"
        m = next((q for q in marks if q["id"] == mid), None)
        if m is None:
            return None, f"mark {mid} is not in the image"
        u, v = m["px"]
        return {"tool": tool, "args": {"camera": "front", "u": u, "v": v}}, m
    if tool == "done":
        return {"tool": "done"}, None
    return None, "unknown command"


PROTOCOL_DEFAULT = "orig"


class ObsStore:
    """2026-11-15: keeps the images each decision received (frames since the previous decision with their world times, and
    the current image with its marks), deduplicated, and saves them as one compressed npz per episode."""

    def __init__(self):
        self.imgs, self.key, self.dec = [], {}, {}

    def _idx(self, key, img):
        if key not in self.key:
            self.key[key] = len(self.imgs)
            self.imgs.append(np.asarray(img, dtype=np.uint8))
        return self.key[key]

    def add(self, k, frames, cur, marks):
        fr = [(float(t), self._idx(("f", float(t)), img)) for t, img in frames]
        ci = self._idx(("cur", k), cur)
        self.dec[int(k)] = {"frames": fr, "cur": ci, "marks": [(m["id"], m["label"], m["obj"], list(map(int, m["px"]))) for m in marks]}

    def save(self, d, job_id):
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, job_id + ".npz")
        small = [i for i in self.imgs if i.shape == self.imgs[0].shape] if self.imgs else []
        shapes = sorted({i.shape for i in self.imgs})
        arr = {f"shape{j}": np.stack([i for i in self.imgs if i.shape == sh]) for j, sh in enumerate(shapes)}
        where = [[shapes.index(i.shape), sum(1 for x in self.imgs[:n] if x.shape == i.shape)] for n, i in enumerate(self.imgs)]
        np.savez_compressed(p, meta=np.frombuffer(json.dumps({"dec": self.dec, "where": where}).encode(), dtype=np.uint8), **arr)
        del small
        return p

HINTS = {"swap": "Watch the frames closely: people may swap, move, cover or use things while you think, and identical "
                 "objects can trade places; before acting on what you remember, check whether the frames show such a change."}


def run_episode(job, policy, media_dir=None, max_turns=12, max_frames=10, max_world_s=240.0):
    fam, member, seed, clock = job["family"], job["member"], int(job["seed"]), job["clock"]
    max_frames = int(job.get("max_frames", max_frames))
    policy.hint = HINTS.get(job.get("hint", ""), "")
    diag, fmode = job.get("diag", "none"), job.get("frames", "buffer")
    L = float(job.get("L", 0.0))
    strict = job.get("protocol", PROTOCOL_DEFAULT) == "strict"
    if strict:
        import livemem2_strict as LMS
        sc, ep = LMS.make(fam, member, seed, clock, person=job.get("person"))
        if job.get("held_names", True):
            ep.lm_held_names = LMS.HELD_NAMES.get(sc.name, {})
    else:
        sc, ep = LM.make(fam, member, seed, clock)
    task = sc.goal_text()
    hist, turns, mem = [], [], {}
    shown_t = -1.0
    end_reason = "max_turns"
    t_start = time.time()
    first_img = None
    wrong_drawer = 0
    obs_store = ObsStore() if job.get("save_obs") else None
    for k in range(1, max_turns + 1):
        if ep.world.time > max_world_s:
            end_reason = "world_time"
            break
        marks = sc.marks(ep)
        cur = ep.world.render("front")
        ep.lm_obs_t = ep.world.time
        if first_img is None:
            first_img = cur
        new = [f for f in ep.lm_frames if f[0] > shown_t + 1e-6 and f[0] < ep.world.time - 1e-6]
        frames = [] if fmode == "latest" else thin(new, max_frames)
        shown_t = ep.world.time
        parts = build_turn(task, k, ep, hist, frames, cur, marks)
        ctx = {"sc": sc, "ep": ep, "marks": marks, "mem": mem, "L": L if clock == "fixed" else 0.0}
        t_dec = ep.world.time
        cmd, text, meta = policy.decide(parts, ctx)
        lat = float(meta.get("latency_s") or 0.0)
        think = lat if clock == "live" else (L if clock == "fixed" else 0.0)
        rec = {"k": k, "t": round(t_dec, 2), "n_frames": len(frames), "n_new": len(new), "marks": [(m["id"], m["label"], m["obj"]) for m in marks],
               "cmd": cmd, "reply": (text or "")[:1500], "meta": meta, "think_s": round(think, 3)}
        if job.get("priv_log"):             # 2026-11-15: privileged truth at observation time (analysis only, never shown)
            try:
                rec["priv"] = sc.oracle_text(ep)
            except Exception as e:  # noqa: BLE001
                rec["priv"] = f"error: {e}"[:120]
        if obs_store is not None:           # 2026-11-15: the exact images of this decision (for the offline probe)
            obs_store.add(k, frames, cur, marks)
        if diag in ("look", "oracle") and clock != "paused":
            ep.execute({"tool": "observe", "args": {"images": False}}, think_s=think)
            ep._seen = None
            think = 0.0
            marks2 = sc.marks(ep)
            cur2 = ep.world.render("front")
            ep.lm_obs_t = ep.world.time
            note = sc.oracle_text(ep) if diag == "oracle" else None
            h2 = hist + [f"(decision {k}, not run yet) you chose: {json.dumps(cmd) if cmd else 'an invalid reply'}"]
            parts2 = build_turn(task, k, ep, h2, [], cur2, marks2, extra=note, reask=True)
            ctx2 = {"sc": sc, "ep": ep, "marks": marks2, "mem": mem}
            cmd2, text2, meta2 = policy.decide(parts2, ctx2)
            rec.update({"reask_t": round(ep.world.time, 2), "cmd2": cmd2, "reply2": (text2 or "")[:1000], "meta2": meta2,
                        "marks2": [(m["id"], m["label"], m["obj"]) for m in marks2]})
            cmd, marks = cmd2, marks2
        call, m = to_call(cmd, marks)
        if call is None:
            out = ep.execute({"tool": "observe", "args": {"images": False}}, think_s=think)
            ep._seen = None
            hist.append(f"t={t_dec:.1f} s: invalid reply ({m}); nothing done. note: {(cmd or {}).get('note', '') if isinstance(cmd, dict) else ''}")
            rec["invalid"] = m
            turns.append(rec)
            continue
        if call["tool"] == "done":
            hist.append(f"t={t_dec:.1f} s: done")
            rec["done"] = True
            turns.append(rec)
            end_reason = "done"
            break
        lifted_before = list(getattr(sc, "lifted", []))
        out = ep.execute(call, think_s=think)
        ep._seen = None
        tr = ep.trace[-1] if ep.trace else {}
        if sc.name in ("knife", "apple") and call["tool"] == "pick" and str(tr.get("object")).startswith("cover") \
                and tr.get("object") != sc.under_at_cmd and tr.get("object") not in lifted_before:
            wrong_drawer += 1
        desc = call["tool"] + (f" mark {m['id']} ({m['label']})" if m else f" {call['args'].get('seconds', '')} s")
        note = cmd.get("note", "") if isinstance(cmd, dict) else ""
        hist.append(f"t={t_dec:.1f} s: {desc} -> {result_text(out)}" + (f". note: {str(note)[:200]}" if note else ""))
        rec.update({"call": call, "obj": tr.get("object"), "ok": out.get("ok"), "result": result_text(out), "t_end": round(ep.world.time, 2),
                    "held_after": ep.held()})
        turns.append(rec)
    ok = bool(sc.success(ep))
    res = {**job, "success": ok, "end": end_reason, "turns": len(turns), "world_s": round(ep.world.time, 2),
           "changed": sc.t0 is not None, "change_t": None if sc.t0 is None else round(sc.t0, 2), "outcome": sc.outcome(ep),
           "wrong_cover": wrong_drawer, "safety_hold_s": round(ep.lm_hold_ticks * LM.TICK, 2), "wall_s": round(time.time() - t_start, 1),
           "lat_mean": round(float(np.mean([t["meta"].get("latency_s") or 0 for t in turns])) if turns else 0.0, 2),
           "trace": turns}
    if strict:
        res["protocol"] = "strict"
        res["strict"] = sc.strict_info()
    if obs_store is not None:
        try:
            res["obs_file"] = obs_store.save(job["save_obs"], job["job_id"])
        except Exception as e:  # noqa: BLE001
            res["obs_error"] = str(e)[:200]
    if sc.name in ("knife", "apple"):
        res["success_raw"] = ok
        res["success"] = ok and wrong_drawer == 0     # the knife fetched from where it was put, without a search
    if media_dir:
        os.makedirs(media_dir, exist_ok=True)
        stem = f"{job.get('tag', 'x')}_{fam}_{member}_{clock}{'' if diag == 'none' else '_' + diag}_s{seed}"
        try:
            from PIL import Image
            ims = [Image.fromarray(f[1]) for f in ep.lm_frames]
            if ims and int(job["seed"]) < int(os.environ.get("LM2_MEDIA_SEEDS", "3")):
                ims[0].save(os.path.join(media_dir, stem + ".gif"), save_all=True, append_images=ims[1:], duration=125, loop=0)
            Image.fromarray(first_img).save(os.path.join(media_dir, stem + "_first.jpg"), quality=85)
            Image.fromarray(ep.world.render("front")).save(os.path.join(media_dir, stem + "_last.jpg"), quality=85)
            res["media"] = stem
        except Exception as e:  # noqa: BLE001
            res["media_error"] = str(e)[:200]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--media", default=None)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--protocol", default="orig", choices=["orig", "strict"],
                    help="default protocol for jobs without a 'protocol' field (2026-11-06)")
    a = ap.parse_args()
    global PROTOCOL_DEFAULT
    PROTOCOL_DEFAULT = a.protocol
    i, n = map(int, a.shard.split("/"))
    jobs = [json.loads(x) for x in open(a.jobs) if x.strip()]
    jobs = [j for idx, j in enumerate(jobs) if idx % n == i]
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            try:
                r = json.loads(x)
                done.add(r["job_id"])
            except Exception:
                pass
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pols = {}
    for j in jobs:
        if j["job_id"] in done:
            continue
        p = j["policy"]
        st = j.get("protocol", PROTOCOL_DEFAULT) == "strict"
        pk = (p, st, j.get("max_tokens"))
        if pk not in pols:
            if p.startswith("scripted:"):
                pols[pk] = (StrictScriptedPolicy if st else ScriptedPolicy)(p.split(":", 1)[1])
            else:
                pols[pk] = ModelPolicy(p, j.get("max_tokens"))
        t0 = time.time()
        try:
            r = run_episode(j, pols[pk], a.media, a.max_turns, a.max_frames)
        except Exception as e:  # noqa: BLE001
            r = {**j, "success": None, "error": f"{type(e).__name__}: {e}"[:500], "tb": traceback.format_exc()[-2000:]}
        with open(a.out, "a") as f:
            f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        print(f"{j['job_id']} success={r.get('success')} turns={r.get('turns')} world={r.get('world_s')} wall={time.time() - t0:.0f}s "
              f"{r.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
