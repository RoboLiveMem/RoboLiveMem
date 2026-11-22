"""Tabletop: a tiny, dependency-light re-creation of the "Grandpa's mug" family, to try the protocol in seconds.

Three identical mugs stand at marks 1-3. The robot must bring Grandpa's mug to his tray. A person (the son) may reach in
while the robot is still and swap mugs. Seeds follow the paper's design (L1-L8):

    seed % 3 == 0   event   the son swaps Grandpa's mug with a look-alike          (the decisive variable z changes)
    seed % 3 == 1   decoy   the son swaps the two other mugs, same motion          (z stays)
    seed % 3 == 2   null    nobody touches the table

The change starts 0.5-1.5 s after hand-over and only while the robot holds still (L3). Under the paused clock the robot
is still only while it explicitly waits; under the live clock it is also still while the model thinks. A safety
interlock refuses a pick while the person is at the table (no budget used), so answering fast does not beat the change.
Conditions: P (hidden, paused), L (hidden, live), V (visible twin, live: Grandpa's mug carries a red sticker).

This is a teaching-scale environment (2-D rendering with NumPy and Pillow). The paper's results come from the three
simulator tiers in tiers/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import commit as commit_check
from .clock import Clock, PAUSED, LIVE, parse as parse_clock

W, H = 320, 240
FPS = 2.0                     # frames recorded per world second
MARKS = {1: (90.0, 122.0), 2: (160.0, 122.0), 3: (230.0, 122.0)}
STAGE = (160.0, 66.0)         # where the son puts a mug down mid-swap
ENTRY = (200.0, -40.0)        # where his hand comes from
TRAY = (276.0, 196.0)
PICK_S = 2.0                  # duration of the robot's pick-and-place
DEADLINE_S = 40.0
# the son's motion: (time from start, hand position, what happens there)
SWAP_KEYS = [(0.0, "E", None), (0.5, "a", "grab_a"), (1.2, "S", "drop"), (1.6, "b", "grab_b"), (2.2, "a", "drop"),
             (2.6, "S", "grab_s"), (3.2, "b", "drop"), (3.7, "E", None)]
SWAP_S = SWAP_KEYS[-1][0]


# ---------------------------------------------------------------------- world
@dataclass
class Instance:
    seed: int
    variant: str               # event | decoy | null
    g0: int                    # Grandpa's mark at hand-over
    a: int | None              # marks the son swaps
    b: int | None
    start_s: float             # scheduled start of the son's motion

    @property
    def instance_id(self):
        return f"mug/{self.seed}"


def make_instance(seed: int) -> Instance:
    rng = random.Random(1000 + seed)
    variant = ("event", "decoy", "null")[seed % 3]
    g0 = rng.choice([1, 2, 3])
    others = [m for m in (1, 2, 3) if m != g0]
    if variant == "event":
        a, b = g0, rng.choice(others)
    elif variant == "decoy":
        a, b = others
    else:
        a = b = None
    return Instance(seed, variant, g0, a, b, round(rng.uniform(0.5, 1.5), 2))


class World:
    """World state as a function of time. Mug 0 is Grandpa's; mug i starts at mark ((g0 - 1 + i) % 3) + 1."""

    def __init__(self, inst: Instance, visible: bool):
        self.inst, self.visible = inst, visible
        self.home = {i: ((inst.g0 - 1 + i) % 3) + 1 for i in range(3)}   # mug -> mark at hand-over
        self.actor_start: float | None = None                            # set when the son actually starts
        self.robot: dict | None = None                                   # {"t0", "mark"} while picking

    # -- the son
    def maybe_start_actor(self, t_from: float, t_to: float, robot_still: bool):
        if self.inst.variant != "null" and self.actor_start is None and robot_still and t_to >= self.inst.start_s:
            self.actor_start = max(t_from, self.inst.start_s)

    def actor_busy(self, t: float) -> bool:
        return self.actor_start is not None and self.actor_start <= t < self.actor_start + SWAP_S

    @property
    def tau_e(self):
        return None if self.actor_start is None else self.actor_start + SWAP_S

    def _pt(self, key):
        return {"E": ENTRY, "S": STAGE, "a": MARKS.get(self.inst.a), "b": MARKS.get(self.inst.b)}[key]

    def _hand(self, u: float):
        for (t0, k0, _), (t1, k1, _) in zip(SWAP_KEYS, SWAP_KEYS[1:]):
            if t0 <= u <= t1:
                s = (u - t0) / (t1 - t0)
                s = s * s * (3 - 2 * s)
                p0, p1 = self._pt(k0), self._pt(k1)
                return (p0[0] + (p1[0] - p0[0]) * s, p0[1] + (p1[1] - p0[1]) * s)
        return ENTRY

    def mugs(self, t: float):
        """mug -> (x, y, lifted) at time t."""
        pos = {i: (*MARKS[m], False) for i, m in self.home.items()}
        if self.actor_start is None or t < self.actor_start:
            return pos
        u = t - self.actor_start
        mug_at = {m: i for i, m in self.home.items()}
        ma, mb = mug_at[self.inst.a], mug_at[self.inst.b]
        hand = self._hand(min(u, SWAP_S))
        # mug a: at a -> carried (0.5-1.2) -> stage (1.2-2.6) -> carried (2.6-3.2) -> at b
        if u < 0.5:
            pa = (*MARKS[self.inst.a], False)
        elif u < 1.2:
            pa = (*hand, True)
        elif u < 2.6:
            pa = (*STAGE, False)
        elif u < 3.2:
            pa = (*hand, True)
        else:
            pa = (*MARKS[self.inst.b], False)
        # mug b: at b -> carried (1.6-2.2) -> at a
        if u < 1.6:
            pb = (*MARKS[self.inst.b], False)
        elif u < 2.2:
            pb = (*hand, True)
        else:
            pb = (*MARKS[self.inst.a], False)
        pos[ma], pos[mb] = pa, pb
        return pos

    def mug_at_mark(self, t: float, mark: int):
        x, y = MARKS[mark]
        for i, (mx, my, lifted) in self.mugs(t).items():
            if not lifted and abs(mx - x) < 1 and abs(my - y) < 1:
                return i
        return None

    def z(self, t: float):
        """The decisive variable: the mark holding Grandpa's mug (None while it is off the marks)."""
        x, y, lifted = self.mugs(t)[0]
        for m, (mx, my) in MARKS.items():
            if not lifted and abs(mx - x) < 1 and abs(my - y) < 1:
                return m
        return None

    # -- rendering
    def render(self, t: float) -> np.ndarray:
        img = Image.new("RGB", (W, H), (233, 228, 220))
        d = ImageDraw.Draw(img)
        d.polygon([(0, 44), (W, 44), (W, H), (0, H)], fill=(214, 190, 160))
        d.line([(0, 44), (W, 44)], fill=(170, 146, 116), width=2)
        d.rectangle([TRAY[0] - 30, TRAY[1] - 20, TRAY[0] + 30, TRAY[1] + 20], fill=(96, 140, 196), outline=(52, 86, 140), width=2)
        font = _font(11)
        for m, (x, y) in MARKS.items():
            d.ellipse([x - 9, y + 22, x + 9, y + 40], fill=(255, 255, 255), outline=(40, 40, 40), width=2)
            d.text((x - 3, y + 24), str(m), fill=(20, 20, 20), font=font)
        mugs = self.mugs(t)
        robot_mug = None
        if self.robot is not None:
            robot_mug, gp = self._robot_state(t)
        for i, (x, y, lifted) in mugs.items():
            if i != robot_mug and not lifted:
                _mug(d, x, y, lifted, sticker=(self.visible and i == 0))
        if self.actor_busy(t):
            hx, hy = self._hand(t - self.actor_start)
            d.line([(hx + 34, -10), (hx + 4, hy - 22)], fill=(205, 160, 125), width=14)
            d.ellipse([hx - 10, hy - 34, hx + 14, hy - 12], fill=(226, 182, 144), outline=(170, 120, 90))
        for i, (x, y, lifted) in mugs.items():
            if i != robot_mug and lifted:
                _mug(d, x, y, lifted, sticker=(self.visible and i == 0))
        if self.robot is not None:
            gx, gy = gp
            d.line([(20, H + 10), (gx, gy + 14)], fill=(110, 112, 118), width=12)
            if robot_mug is not None:
                _mug(d, gx, gy, True, sticker=(self.visible and robot_mug == 0))
            d.rectangle([gx - 15, gy + 6, gx - 11, gy + 20], fill=(60, 62, 66))
            d.rectangle([gx + 11, gy + 6, gx + 15, gy + 20], fill=(60, 62, 66))
        d.text((6, 4), f"t = {t:4.1f} s", fill=(60, 56, 70), font=font)
        return np.asarray(img)

    def _robot_state(self, t):
        """(mug carried, gripper position) during the pick-and-place."""
        u = (t - self.robot["t0"]) / PICK_S
        x0, y0 = 40.0, 230.0
        mx, my = MARKS[self.robot["mark"]]
        if u < 0.4:
            s = u / 0.4
            return None, (x0 + (mx - x0) * s, y0 + (my - 4 - y0) * s)
        s = min(1.0, (u - 0.4) / 0.6)
        return self.robot["mug"], (mx + (TRAY[0] - mx) * s, my + (TRAY[1] - 14 - my) * s)


def _mug(d, x, y, lifted, sticker=False):
    if lifted:
        d.ellipse([x - 14, y + 8, x + 14, y + 16], fill=(180, 160, 132))
        y -= 12
    else:
        d.ellipse([x - 16, y + 6, x + 16, y + 16], fill=(190, 168, 138))
    d.arc([x + 6, y - 10, x + 24, y + 6], start=-90, end=90, fill=(110, 110, 112), width=4)
    d.rectangle([x - 13, y - 14, x + 13, y + 10], fill=(210, 210, 212), outline=(110, 110, 112))
    d.ellipse([x - 13, y + 4, x + 13, y + 14], fill=(210, 210, 212), outline=(110, 110, 112))
    d.ellipse([x - 13, y - 20, x + 13, y - 8], fill=(150, 150, 154), outline=(110, 110, 112))
    if sticker:
        d.ellipse([x - 6, y - 5, x + 6, y + 7], fill=(214, 40, 40))


_FONTS: dict = {}


def _font(size):
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def contact_sheet(frames, times, cols=4, scale=0.5) -> np.ndarray:
    """Frames recorded since the last decision, tiled with their world times."""
    tw, th = int(W * scale), int(H * scale)
    rows = max(1, math.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * tw, rows * th), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for k, (f, t) in enumerate(zip(frames, times)):
        x, y = (k % cols) * tw, (k // cols) * th
        sheet.paste(Image.fromarray(f).resize((tw, th)), (x, y))
        d.rectangle([x, y, x + 54, y + 13], fill=(255, 255, 255))
        d.text((x + 2, y + 1), f"{t:.1f} s", fill=(0, 0, 0), font=_font(10))
    return np.asarray(sheet)


# ---------------------------------------------------------------------- episode
@dataclass
class Obs:
    t: float
    image: np.ndarray
    new_frames: list
    new_times: list
    message: str = ""
    first: bool = False


@dataclass
class Episode:
    inst: Instance
    condition: str                       # P | L | V
    clock: Clock
    commit: bool = False
    max_decisions: int = 10
    world: World = None
    t: float = 0.0
    frames: list = field(default_factory=list)   # (t, image)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.world = World(self.inst, visible=(self.condition == "V"))
        self._record(0.0)

    def _record(self, t):
        self.frames.append((t, self.world.render(t)))

    def _advance(self, t_to, robot_still=True):
        if t_to <= self.t:
            return
        self.world.maybe_start_actor(self.t, t_to, robot_still)
        k = math.floor(self.t * FPS) + 1
        while k / FPS <= t_to + 1e-9:
            self._record(k / FPS)
            k += 1
        self.t = t_to

    def task_text(self):
        s = (f"Task: bring Grandpa's mug to his tray (the blue tray). When you took over, Grandpa's mug was the one at "
             f"mark {self.inst.g0}. The three mugs look the same. Your son is setting this table.")
        if self.condition == "V":
            s += " Grandpa's mug carries a red sticker."
        return s

    def run(self, policy, verbose=False) -> dict:
        seen_t = -1.0                   # time of the last view the agent saw
        msg, decisive = "", None
        status = "timeout"
        n_calls = tokens = 0
        for i in range(self.max_decisions):
            new = [(t, f) for t, f in self.frames if seen_t + 1e-9 < t <= self.t + 1e-9]
            cur = self.world.render(self.t)
            obs = Obs(self.t, cur, [f for _, f in new], [t for t, _ in new], msg, first=(i == 0))
            seen_t, seen_img = self.t, cur
            t_o = self.t
            cmd, latency, meta = policy(self, obs)
            n_calls += 1
            tokens += int((meta or {}).get("in_tokens") or 0) + int((meta or {}).get("out_tokens") or 0)
            t_c = t_o + self.clock.kappa(latency)
            self._advance(t_c)
            rec = {"i": i, "t_o": round(t_o, 2), "latency_s": round(latency, 3), "t_c": round(t_c, 2), "cmd": cmd}
            act = (cmd or {}).get("action")
            msg = ""
            if act == "pick" and cmd.get("mark") in MARKS:
                m = cmd["mark"]
                if self.world.actor_busy(t_c):
                    rec["result"] = msg = "refused: someone is working at the table; nothing was executed"
                elif self.commit and commit_check.changed(
                        seen_img, [f for t, f in self.frames if t > seen_t] + [self.world.render(t_c)], MARKS[m]):
                    rec["result"] = msg = commit_check.MESSAGE.format(mark=m)
                else:
                    mug = self.world.mug_at_mark(t_c, m)
                    self.world.robot = {"t0": t_c, "mark": m, "mug": mug}
                    self._advance(t_c + PICK_S, robot_still=False)
                    status = "success" if mug == 0 else "fail"
                    rec["result"] = f"delivered the mug from mark {m}"
                    decisive = rec
                    self.log.append(rec)
                    break
            elif act == "wait":
                s = float(min(10.0, max(0.5, cmd.get("seconds", 1.0))))
                self._advance(t_c + s)
                rec["result"] = f"waited {s:g} s"
            elif act == "done":
                status, rec["result"] = "fail", "done without delivering"
                self.log.append(rec)
                break
            else:
                rec["result"] = msg = "invalid command; use pick, wait or done"
                self._advance(t_c + 0.5)
            self.log.append(rec)
            if verbose:
                print(f"    {rec}", file=sys.stderr)
            if self.t >= DEADLINE_S:
                break
        tau_e = self.world.tau_e
        stale = lead = None
        if decisive is not None and self.inst.variant != "null":
            if tau_e is not None and self.world.actor_start <= decisive["t_c"]:
                stale, lead = decisive["t_o"] < tau_e, round(decisive["t_o"] - tau_e, 2)
            else:
                stale = False
        return {"instance": self.inst.instance_id, "seed": self.inst.seed, "variant": self.inst.variant,
                "condition": self.condition, "clock": self.clock.label, "commit": self.commit,
                "success": status == "success", "status": status, "stale": stale, "lead_s": lead,
                "tau_s": self.world.actor_start, "tau_e": tau_e, "decisions": len(self.log), "calls": n_calls,
                "tokens": tokens, "world_s": round(self.t, 2), "log": self.log}

    def gif(self, path, ms=250):
        ims = [Image.fromarray(f) for _, f in self.frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0)


# ---------------------------------------------------------------------- policies
class Scripted:
    """Reference policies (paper Sec. 4, Validation). `think_s` is charged as their thinking time."""

    def __init__(self, kind: str, think_s: float = 3.0):
        assert kind in ("privileged", "stale", "assume-change"), kind
        self.kind, self.think_s = kind, think_s

    def __call__(self, ep: Episode, obs: Obs):
        g0 = ep.inst.g0
        if self.kind == "privileged":       # reads z_t, waits while the person is at the table
            z = ep.world.z(obs.t)
            cmd = {"action": "wait", "seconds": 0.5} if (ep.world.actor_busy(obs.t) or z is None) else \
                {"action": "pick", "mark": z}
        elif self.kind == "stale":          # plan fixed at the first observation
            cmd = {"action": "pick", "mark": g0}
        else:                               # always assumes Grandpa's mug was swapped
            cmd = {"action": "wait", "seconds": 6} if obs.first else {"action": "pick", "mark": (g0 % 3) + 1}
        return cmd, self.think_s, {}


SYSTEM = """You control a robot arm at a kitchen table, seen through the robot's camera. Three numbered marks (1, 2, 3)
are drawn in front of three mugs; the blue rectangle is Grandpa's tray. The world time is printed in the corner.

Each turn you get the current camera image and, when there are any, the frames recorded since your previous turn
(a contact sheet with their world times). Commands:
  {"action": "pick", "mark": 2}       pick up the mug at mark 2 and put it on Grandpa's tray (this ends the episode)
  {"action": "wait", "seconds": 3}    hold still for 0.5 to 10 seconds while the camera keeps recording
  {"action": "done"}                  stop
Write at most a few sentences of reasoning, then exactly one JSON command on the last line."""


class ModelPolicy:
    """Any chat model (robolivemem.models.Chat); latency = the measured time of the model call."""

    def __init__(self, spec: str, max_tokens: int = 1024, max_frames: int = 12, retries: int = 3, **kw):
        from .models import Chat, parse_json_reply
        self.chat, self.parse = Chat(spec, max_tokens=max_tokens, **kw), parse_json_reply
        self.max_frames, self.retries = max_frames, retries
        self.turns = []

    def __call__(self, ep: Episode, obs: Obs):
        if obs.first:
            self.turns = []
        parts = [ep.task_text()] if obs.first else []
        if obs.message:
            parts.append(f"Result of your last command: {obs.message}")
        fr, ts = obs.new_frames, obs.new_times
        if len(fr) > 1:
            idx = np.linspace(0, len(fr) - 1, min(len(fr), self.max_frames)).round().astype(int)
            parts += [f"Frames recorded since your previous turn ({ts[0]:.1f}-{ts[-1]:.1f} s):",
                      contact_sheet([fr[i] for i in idx], [ts[i] for i in idx])]
        parts += [f"Current view (t = {obs.t:.1f} s):", obs.image]
        self.turns.append(("user", parts))
        # keep the dialogue short: text of earlier turns, images only from the latest one
        turns = [(r, [p for p in ps if isinstance(p, str)]) for r, ps in self.turns[:-1]] + [self.turns[-1]]
        meta = {}
        for _ in range(self.retries):
            text, meta = self.chat(SYSTEM, turns)
            if not meta.get("error"):
                break
        else:
            raise RuntimeError(f"model call failed: {meta.get('error')}")
        self.turns.append(("assistant", [text or "{}"]))
        cmd = self.parse(text) or {}
        if "mark" in cmd:
            try:
                cmd["mark"] = int(cmd["mark"])
            except (TypeError, ValueError):
                pass
        return cmd, float(meta.get("latency_s", 0.0)), meta


# ---------------------------------------------------------------------- CLI
CONDITIONS = {"P": PAUSED, "L": LIVE, "V": LIVE}


def run(policy_factory, seeds, conditions=("P", "L", "V"), clock_override=None, commit=False, verbose=False,
        gif_dir=None):
    rows = []
    for seed in seeds:
        inst = make_instance(seed)
        for c in conditions:
            clock = PAUSED if c == "P" else (clock_override or CONDITIONS[c])
            ep = Episode(inst, c, clock, commit=commit)
            row = ep.run(policy_factory(), verbose=verbose)
            rows.append(row)
            if gif_dir:
                Path(gif_dir).mkdir(parents=True, exist_ok=True)
                ep.gif(Path(gif_dir) / f"seed{seed}_{c}.gif")
            if verbose:
                print(f"  seed {seed:2d} {inst.variant:5s} {c} -> {row['status']:7s} stale={row['stale']}",
                      file=sys.stderr)
    return rows


def parse_seeds(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    from .metrics import report
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", default="privileged,stale,assume-change",
                    help="scripted policies (privileged, stale, assume-change), comma-separated")
    ap.add_argument("--model", help="a chat model instead, e.g. anthropic:claude-haiku-5-5 or openai:<model>")
    ap.add_argument("--think", type=float, default=3.0, help="thinking time charged to scripted policies (s)")
    ap.add_argument("--clock", help="replace the live clock of L and V, e.g. fixed:2")
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--conditions", default="P,L,V")
    ap.add_argument("--commit", action="store_true", help="enable the commit check")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--effort", default=None, help="reasoning effort passed to the model, if it supports one")
    ap.add_argument("--out", help="write one JSON row per episode here")
    ap.add_argument("--gif", help="save an animated GIF of every episode in this folder")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    seeds, conds = parse_seeds(a.seeds), tuple(a.conditions.split(","))
    clock = parse_clock(a.clock) if a.clock else None
    jobs = [(a.model, lambda: ModelPolicy(a.model, max_tokens=a.max_tokens, effort=a.effort))] if a.model else \
        [(p, (lambda p=p: Scripted(p, a.think))) for p in a.policy.split(",")]
    all_rows = []
    for name, factory in jobs:
        t0 = time.time()
        rows = run(factory, seeds, conds, clock, a.commit, a.verbose, a.gif)
        for r in rows:
            r["policy"] = name
        all_rows += rows
        print(report(rows, f"{name}  ({len(seeds)} seeds: event/decoy/null; {time.time() - t0:.1f} s)"))
        for v in ("event", "decoy", "null"):
            cells = []
            for c in conds:
                sub = [r for r in rows if r["variant"] == v and r["condition"] == c]
                if sub:
                    cells.append(f"{c} {sum(r['success'] for r in sub)}/{len(sub)}")
            print(f"    {v:5s}  " + "   ".join(cells))
        print()
    if a.out:
        with open(a.out, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
    return all_rows


def cli():
    """Console entry point (`robolivemem ...`)."""
    main()


if __name__ == "__main__":
    cli()
