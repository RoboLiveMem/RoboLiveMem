"""RoboLiveMem v3: realistic live-memory scenarios.

What changes against v2:
- People act in the scene for their own reasons (mannequin proxies that walk, carry, pour, open and close).
- The agent's robot is busy with a chore when the episode starts (a scripted prelude of `prelude_s` world seconds). Its
  head camera keeps recording; the agent receives that recording with its first observation and decides afterwards.
- Every scenario has three seeded variants: `event` (the change that matters), `decoy` (a look-alike action that does
  not change the decisive state) and `null` (nobody changes anything). Variant = VARIANTS[seed % 3].
- `visible` control: the same episode, but the decisive state stays observable at decision time (a tag, an ajar lid, a
  marker on the item, a low screen instead of a tall one).
- Success is two-sided: acting on a stale memory fails, and so does over-caution (acting twice, never acting, acting on
  the wrong item, missing a deadline).

Templates (one class each): DoseOnce (DONE), TrackOne (WHICH), Stash (WHERE), ZoneClear (STILL).
Instances are small modules in scenarios/v3_*.py that set geometry, assets and wording.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

MQ_MODEL = "nphsfp"           # BEHAVIOR-1K mannequin (lying in its native frame; stood up with a 90 deg pitch)
MQ_FWD_YAW = globals().get("MQ_FWD_YAW", 0.0)
_Q_UP = _Rot.from_euler("y", 90, degrees=True)
TOKEN_RGBA = [0.86, 0.12, 0.1, 1]


import os as _os
LIVE4 = _os.environ.get("RLM_LIVE4", "1") == "1"   # main: on by default
LIVE4_PRELUDE = float(_os.environ.get("RLM_LIVE4_PRELUDE", "10"))
LIVE4_WINDOW = (0.3, 2.5)    # s after the hand-over at which the person's scheduled action becomes due

def person_cfg(name, k, scale=1.0):
    return {"type": "DatasetObject", "name": name, "category": "mannequin", "model": MQ_MODEL, "kinematic_only": True,
            "visual_only": True, "position": park(300 + k), "scale": [scale, scale, scale]}


def ds(name, category, model, k):
    """A kinematic, visual-only BEHAVIOR object parked out of the way until a scenario poses it."""
    return {"type": "DatasetObject", "name": name, "category": category, "model": model, "kinematic_only": True,
            "visual_only": True, "position": park(k)}


def _lerp(a, b, u):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * u


def _ang(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- people
class Person:
    """A person (mannequin proxy) posed kinematically: walks with a slight bob, turns, carries one item in front of
    the body at waist height, puts it down, pours from it, opens / closes lids."""

    def __init__(self, sc, name, scale=1.0, k=0):
        self.sc, self.name, self.s, self.k = sc, name, scale, k
        self.zoff = 0.88 * scale; self.height = 1.76 * scale
        self.reset()

    def reset(self):
        self.x = self.y = 0.0; self.yaw = 0.0; self.present = False; self.held = None; self.held_q = None
        self.bob = 0.0; self.t = 0.0; self.reach = 0.0; self.tilt = 0.0
        self.park_pos = np.asarray(park(300 + self.k), float)

    def calibrate(self):
        o = obj(self.name)
        set_pose(o, np.array([0.0, 0.0, 10.0]), self.quat(0.0))
        og.sim.step()
        lo, hi = [npf(v) for v in o.aabb]
        self.zoff = 10.0 - float(lo[2]); self.height = float(hi[2] - lo[2])
        set_pose(o, self.park_pos, self.quat(0.0))

    def quat(self, yaw):
        return (_Rot.from_euler("z", yaw + MQ_FWD_YAW) * _Q_UP).as_quat()

    def fwd(self):
        return np.array([math.cos(self.yaw), math.sin(self.yaw)])

    def hand(self):
        """Where a carried item is: in front of the body at waist height (reach > 0 extends it toward a target)."""
        f = self.fwd()
        xy = np.array([self.x, self.y]) + f * (0.30 * self.s + self.reach)
        return np.array([xy[0], xy[1], max(0.98 * self.s, getattr(self.sc, "h", 0.0) + 0.12)])

    def g_reach(self, target_xy=None, seconds=0.4):
        """Extend the hand toward a point on the table (or retract it when target_xy is None)."""
        if target_xy is None:
            r1 = 0.0
        else:
            d = float(np.dot(np.asarray(target_xy[:2], float) - np.array([self.x, self.y]), self.fwd()))
            r1 = float(np.clip(d - 0.30 * self.s - 0.04, 0.0, 0.75))
        r0 = self.reach; n = max(1, int(seconds * SIM_HZ))
        for i in range(1, n + 1):
            self.reach = r0 + (r1 - r0) * ease(i / n); self.apply()
            yield

    def apply(self):
        sc = self.sc; o = obj(self.name)
        if not self.present:
            sc.pins.move(o, self.park_pos, self.quat(0.0))
            return
        sc.pins.move(o, np.array([self.x, self.y, self.zoff + self.bob]), self.quat(self.yaw))
        if self.held is not None:
            q = self.held_q if self.held_q is not None else get_pose(obj(self.held))[1]
            if self.tilt:
                q = (_Rot.from_euler("z", self.yaw) * _Rot.from_euler("y", self.tilt) * _Rot.from_euler("z", -self.yaw)
                     * _Rot.from_quat(q)).as_quat()
            sc.pins.move(obj(self.held), self._held_pos(), q)

    def _held_pos(self):
        return self.hand() if getattr(self, "_held_override", None) is None else self._held_override

    # ---- generators (one yield per simulation tick)
    def g_enter(self, x, y, yaw):
        self.present = True; self.x, self.y, self.yaw = float(x), float(y), float(yaw); self.apply()
        yield

    def g_leave(self):
        self.present = False; self.apply()
        yield

    def g_turn(self, yaw, rate=2.6):
        while True:
            d = _ang(yaw - self.yaw)
            if abs(d) < 0.02:
                self.yaw = yaw; self.apply(); return
            self.yaw += max(-rate * DT, min(rate * DT, d)); self.apply()
            yield

    def g_walk(self, pts, speed=0.85, face=None):
        for p in pts:
            p = np.asarray(p[:2], float)
            while True:
                d = p - np.array([self.x, self.y]); dist = float(np.linalg.norm(d))
                if dist < 1e-3:
                    break
                want = math.atan2(d[1], d[0]); dy = _ang(want - self.yaw)
                self.yaw += max(-3.0 * DT, min(3.0 * DT, dy))
                step = min(dist, speed * DT * (0.35 if abs(dy) > 1.0 else 1.0))
                self.x += d[0] / dist * step; self.y += d[1] / dist * step
                self.t += DT; self.bob = 0.014 * abs(math.sin(self.t * 5.5))
                self.apply()
                yield
        self.bob = 0.0
        if face is not None:
            yield from self.g_turn(face)
        self.apply()

    def g_wait(self, seconds):
        for _ in range(max(1, int(seconds * SIM_HZ))):
            yield

    def g_take(self, name, seconds=0.8):
        """Reach to an object (wherever it is pinned) and bring it to the hand."""
        sc = self.sc; o = obj(name)
        p0 = sc.pins.pos(o) if o.name in sc.pins.p else get_pose(o)[0]
        q0 = sc.pins.p[o.name][2] if o.name in sc.pins.p else get_pose(o)[1]
        yield from self.g_reach(p0)
        n = max(1, int(seconds * SIM_HZ))
        for i in range(1, n + 1):
            u = ease(i / n)
            sc.pins.move(o, _lerp(p0, self.hand(), u), q0)
            yield
        self.held, self.held_q = name, q0; self.apply()
        yield from self.g_reach(None)

    def g_put(self, pos, quat=None, seconds=0.8):
        sc = self.sc; name = self.held; o = obj(name)
        yield from self.g_reach(pos)
        p0 = self.hand(); q = self.held_q if quat is None else quat
        self.held = None
        n = max(1, int(seconds * SIM_HZ))
        for i in range(1, n + 1):
            sc.pins.move(o, _lerp(p0, pos, ease(i / n)), q)
            yield
        sc.pins.move(o, np.asarray(pos, float), q)
        yield from self.g_reach(None)

    def g_pour(self, above, seconds=2.2, tilt=1.9):
        """Move the held item over `above`, tip it, hold, bring it back (salt, pills, water, reagent)."""
        yield from self.g_reach(above)
        n = max(1, int(0.7 * SIM_HZ)); start = self.hand()
        for i in range(1, n + 1):
            self._held_override = _lerp(start, above, ease(i / n)); self.apply()
            yield
        m = max(1, int(0.5 * SIM_HZ))
        for i in range(1, m + 1):
            self.tilt = tilt * ease(i / m); self.apply()
            yield
        for _ in range(max(1, int((seconds - 1.9) * SIM_HZ))):
            self.tilt = tilt + 0.15 * math.sin(self.t * 9); self.t += DT; self.apply()
            yield
        for i in range(1, m + 1):
            self.tilt = tilt * (1 - ease(i / m)); self.apply()
            yield
        self.tilt = 0.0
        for i in range(1, n + 1):
            self._held_override = _lerp(above, self.hand(), ease(i / n)); self.apply()
            yield
        self._held_override = None; self.apply()
        yield from self.g_reach(None)

    def g_lift_lid(self, lid, closed_pos, open_pos, quat, seconds=0.8):
        """Open (closed -> open) or close (open -> closed) a lid by hand."""
        o = obj(lid); n = max(1, int(seconds * SIM_HZ))
        yield from self.g_reach(closed_pos, 0.3)
        for i in range(1, n + 1):
            u = ease(i / n)
            p = _lerp(closed_pos, open_pos, u)
            p[2] += 0.06 * math.sin(math.pi * u)
            self.sc.pins.move(o, p, quat)
            yield
        if self.held is None:
            yield from self.g_reach(None, 0.3)


def g_seq(*gens):
    for g in gens:
        yield from g


# ----------------------------------------------------------------------------- base
class V3Scenario(StationScenario):
    VARIANTS = ("event", "decoy", "null")
    PEOPLE = ()                  # ((name, scale), ...); objects come from people_objects()
    prelude_s = 30.0             # seconds of the robot's chore before the agent takes over
    chore = "wiping the table"
    deadline = 300.0
    budget_ends_episode = True
    B_START = None

    def __init__(self):
        super().__init__()
        self.people = {n: Person(self, n, s, k=self.PEOPLE_K0 + i) for i, (n, s) in enumerate(self.PEOPLE)}
        self.visible = False; self.variant = "null"; self.runners = []

    PEOPLE_K0 = 0                # parking index of the first person (unique per scene)
    CHORE = None                 # ((df, dl), ...) wiping points in the station frame; default: in front, right
    wait_actors = True           # agent skills wait (world time runs) until people finish what they are doing
    SHEET_DT = 1.5               # contact-sheet spacing (s) of the recording delivered with the first observation

    @classmethod
    def people_objects(cls):
        return [person_cfg(n, cls.PEOPLE_K0 + i, s) for i, (n, s) in enumerate(cls.PEOPLE)]

    HEAD_FOCAL = 11.0            # wide head camera (about 87 deg horizontal), so people beside the table stay in view

    def build(self):
        super().build()
        try:
            self.head.focal_length = self.HEAD_FOCAL
        except Exception:
            pass
        for p in self.people.values():
            p.calibrate()

    # ---- episode
    HARD_PRELUDE = 95.0          # "busy day": a longer chore with several similar actions, at most one of which matters
    hard = False

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}; self.containers = {}; self.hidden = {}; self.runners = []
        self.live4 = LIVE4 and not self.hard and "ZoneClear" not in [c.__name__ for c in type(self).__mro__]
        self.prelude_s = self.HARD_PRELUDE if self.hard else (LIVE4_PRELUDE if self.live4 else type(self).prelude_s)
        self.place_robots()
        for p in self.people.values():
            p.reset(); p.apply()
        self.variant = self.VARIANTS[ep.seed % len(self.VARIANTS)]
        self.ep = ep
        if self.hard:
            self.setup_hard(ep, rng)
        else:
            self.setup(ep, rng)
        if self.live4:
            t_ev = self.prelude_s + rng.uniform(*LIVE4_WINDOW)
            for r in self.runners:
                if not r[2]:
                    r[0] = t_ev; r[3] = True
            self.hidden.update(protocol="v4_live", t_event_sched=round(t_ev, 2), t_person=round(t_ev, 2))
        self.hidden.update(variant=self.variant, visible=bool(self.visible), hard=bool(self.hard))

    def setup_hard(self, ep, rng):
        raise NotImplementedError

    def g_until(self, t):
        while self.ep.world_t < t:
            yield

    def g_visits(self, visits, t_first, gap=(3.0, 8.0), rng=None):
        """Run person visits one after another: the first at t_first, then each after a random pause."""
        first = True
        for g in visits:
            if not first:
                yield from self.g_until(self.ep.world_t + rng.uniform(*gap))
            first = False
            yield from g

    def setup(self, ep, rng):
        raise NotImplementedError

    def run(self, gen, at=0.0, bg=False, gated=False):
        """Schedule an actor script (generator) to start at world time `at` (bg: decoration that never ends).
        gated: start only while the agent's robot is not executing a skill (v4 live protocol)."""
        self.runners.append([float(at), gen, bg, gated, False])

    def world(self, ep):
        keep = []
        for r in self.runners:
            if ep.world_t + 1e-9 < r[0]:
                keep.append(r); continue
            if r[3] and not r[4] and getattr(self, "A_busy", False):
                keep.append(r); continue
            if not r[4]:
                r[4] = True
                if r[3]:
                    self.hidden["t_person_started"] = round(ep.world_t, 2)
                    ep.ev("person_started")
            try:
                next(r[1]); keep.append(r)
            except StopIteration:
                pass
        self.runners = keep

    def actors_busy(self):
        return any((not r[2]) and r[4] for r in self.runners)

    def _busy(self, gen):
        n = 0
        while self.wait_actors and self.actors_busy() and n < 60 * SIM_HZ:
            n += 1
            yield
        if n:
            self.ep.ev("agent_waited_for_people", seconds=round(n * DT, 2))
        if getattr(self, "_commit_ep", None) is not None:
            self._commit_ep.commit_check()     # ablation A only
        self.A_busy = True
        try:
            yield from gen
        finally:
            self.A_busy = False

    def skill_place(self, ep, slot, target):
        """Place with the hand that holds the item (slots name a default arm only)."""
        held = list(self.A.held.values())
        arm = held[0][1] if held else None
        saved = self.slots[slot]
        if arm is not None and saved.get("arm", "right") != arm:
            self.slots[slot] = dict(saved, arm=arm)
        try:
            yield from super().skill_place(ep, slot, target)
        finally:
            self.slots[slot] = saved

    # ---- the agent robot's chore (prelude): small wiping strokes in front of the robot
    def chore_points(self):
        if self.CART is not None and not self.CHORE:
            cdf, cdl = self.CART
            return [np.array(self.fr.p(cdf + a, cdl + b, 0.80)) for a, b in ((0.02, 0.12), (0.04, 0.0), (0.0, -0.08), (0.04, 0.04))]
        z = getattr(self, "chore_z", None)
        z = (self.h + 0.04) if z is None else z
        return [np.array(self.fr.p(df, dl, z)) for df, dl in self.CHORE]

    WORK_D = 0.66                # the robot works from a base this far behind an item (it watches from further back)
    CART = (0.18, -0.62)         # side cart (station frame) where the robot does its chore; None: chore over the table

    def wbase(self, df, dl, arm):
        return self.fr.base(df - self.WORK_D, dl + (-0.12 if arm == "right" else 0.12))

    @classmethod
    def cart_objects(cls, fr, prefix):
        if cls.CART is None:
            return []
        cdf, cdl = cls.CART
        out = table(f"{prefix}cart", fr, cdf, cdl, 0.34, 0.42, 0.72, rgba=[0.55, 0.57, 0.60, 1], leg=0.03)
        out += [box(f"{prefix}cart_tray", fr, cdf, cdl, 0.735, 0.30, 0.36, 0.03, [0.80, 0.80, 0.78, 1])]
        for i, (a, b) in enumerate([(-0.06, -0.10), (0.04, 0.0), (-0.02, 0.10)]):
            out += [box(f"{prefix}cart_item{i}", fr, cdf + a, cdl + b, 0.765, 0.16, 0.025, 0.02, [0.70, 0.72, 0.75, 1])]
        return out

    def chore_q(self, arm, p, q_init=None):
        key = ("chore", tuple(np.round(self.baseA, 3)), arm, tuple(np.round(p, 3)))
        c = self.armA.cache
        if key not in c:
            b = self.A; b.place_base(*self.baseA)
            f = np.array([math.cos(self.baseA[2]), math.sin(self.baseA[2])])
            q, err = b.ik(arm, np.asarray(p, float), approach=[0.3 * f[0], 0.3 * f[1], -1.0], use_trunk=False,
                          q_init=self.qA0 if q_init is None else q_init)
            c[key] = q
        return c[key]

    def g_prelude(self, ep):
        self.A_busy = True
        try:
            pts = self.chore_points(); arm = "right"; q_prev = None; qs = []
            for p in pts:
                q = self.chore_q(arm, p, q_prev); q_prev = q; qs.append(q)
            t_end = ep.world_t + self.prelude_s - 1.5
            i = 0
            while ep.world_t < t_end:
                yield from self.armA.g_to(qs[i % len(qs)], 0.9, arm, closed=True)
                i += 1
            yield from self.armA.g_to(self.qA0, 1.2, arm, closed=False)
        finally:
            self.A_busy = False

    def intro(self):
        return (f"Your robot has just spent the last {self.prelude_s:.0f} seconds {self.chore}; its head camera kept "
                f"recording the whole time, and your first observation includes that recording. ")

    def instruction(self):
        return self.intro() + self.task_text() + ((" " + self.visible_text()) if self.visible and self.visible_text() else "")

    def visible_text(self):
        return ""

    def task_text(self):
        raise NotImplementedError

    # helpers
    def surface_point(self, df, dl, dz=0.0):
        return np.asarray(self.fr.p(df, dl, self.h + dz), float)


# ----------------------------------------------------------------------------- DONE: dose exactly once
class DoseOnce(V3Scenario):
    """A receptacle (mark 1) must receive exactly one dose. A person may dose it (event), dose a different receptacle
    (decoy, mark 2) or not dose anything (null). The agent doses with `dose <mark>` using the source item (mark 3)."""
    skills = ("dose", "wait", "done")
    budget = 2
    RECEPTACLES = ()      # ((obj name, df, dl, label), ...): first is the target
    SOURCE = None         # (obj name, df, dl, label)
    STAND = {}            # receptacle index -> (df, dl) where the person stands to dose it; "source" -> (df, dl)
    ENTRY = (0, 0); EXIT = (0, 0); WALKBY = ()
    dose_word = "dose"
    extra_help = {"dose": "dose <mark>: give one dose to the receptacle at the numbered place, using the item at the "
                          "source place (one manipulation action)"}

    def build_scenario(self):
        fr = self.fr
        for i, (n, df, dl, lab) in enumerate(self.RECEPTACLES):
            self.hz[n] = self._hz(n)
            self.slots[f"rec{i}"] = dict(pos=self.surface_point(df, dl), label=lab, reach=False)
        n, df, dl, lab = self.SOURCE
        self.hz[n] = self._hz(n)
        self.slots["source"] = dict(pos=self.surface_point(df, dl), label=lab, arm="right", grasp_z=self.hz[n] * 1.3,
                                    base=self.wbase(df, dl, "right"))
        self.hz["token"] = 0.02

    def _hz(self, n):
        lo, hi = [npf(v) for v in obj(n).aabb]
        return float(hi[2] - lo[2]) / 2 + 0.002

    def legend(self):
        out = {}
        for i, (n, df, dl, lab) in enumerate(self.RECEPTACLES):
            out[str(i + 1)] = lab
        out[str(len(self.RECEPTACLES) + 1)] = self.SOURCE[3]
        return out

    def mark_points(self):
        pts = [self.surface_point(df, dl, self.hz[n] * 2) for n, df, dl, _ in self.RECEPTACLES]
        n, df, dl, _ = self.SOURCE
        return pts + [self.surface_point(df, dl, self.hz[n] * 2)]

    def marked(self):
        return [dict(pos=p, label=l) for p, l in zip(self.mark_points(), self.legend().values())]

    def slot_by_mark(self, target):
        try:
            k = int(str(target).strip()) - 1
        except Exception:
            return None
        R = len(self.RECEPTACLES)
        if 0 <= k < R:
            return f"rec{k}"
        return "source" if k == R else None

    def describe(self, name):
        return self.SOURCE[3]

    def setup(self, ep, rng):
        fr = self.fr
        self.count = {n: 0 for n, *_ in self.RECEPTACLES}; self.agent_doses = []
        self.pin_home(self.PREFIX)
        for n, df, dl, _ in self.RECEPTACLES:
            self.pins.move(obj(n), self.surface_point(df, dl, self.hz[n]), fr.q())
        sn, sdf, sdl, _ = self.SOURCE
        self.src_home = self.surface_point(sdf, sdl, self.hz[sn]); self.pins.move(obj(sn), self.src_home, fr.q())
        self.pins.move(obj(self.TOKEN), park(290), fr.q())
        self.setup_extra(ep, rng)
        p = list(self.people.values())[0]
        t0 = rng.uniform(3.0, 8.0)
        self.hidden.update(t_person=round(t0, 2))
        if self.variant == "null":
            self.run(self.g_walkby(p), t0)
        else:
            k = 0 if self.variant == "event" else 1
            self.run(self.g_dose_by_person(p, k), t0)

    def setup_extra(self, ep, rng):
        pass

    def setup_hard(self, ep, rng):
        """Busy day: the person comes by two or three times (dose the target, dose the other receptacle, pick up the
        source and put it back unused, or just stop by), in random order; only the event variant doses the target."""
        self.setup_common_dose(ep, rng)
        p = list(self.people.values())[0]
        kinds = {"event": ["dose0", "dose1", "handle"], "decoy": ["dose1", "handle", "walkby"], "null": ["handle", "walkby"]}[self.variant]
        rng.shuffle(kinds)
        gens = [self.g_dose_by_person(p, 0) if k == "dose0" else self.g_dose_by_person(p, 1) if k == "dose1"
                else self.g_handle_source(p) if k == "handle" else self.g_walkby(p) for k in kinds]
        t0 = rng.uniform(5.0, 12.0)
        self.hidden.update(visits=kinds, t_person=round(t0, 2))
        self.run(self.g_visits(gens, t0, rng=rng), t0)

    def setup_common_dose(self, ep, rng):
        fr = self.fr
        self.count = {n: 0 for n, *_ in self.RECEPTACLES}; self.agent_doses = []
        self.pin_home(self.PREFIX)
        for n, df, dl, _ in self.RECEPTACLES:
            self.pins.move(obj(n), self.surface_point(df, dl, self.hz[n]), fr.q())
        sn, sdf, sdl, _ = self.SOURCE
        self.src_home = self.surface_point(sdf, sdl, self.hz[sn]); self.pins.move(obj(sn), self.src_home, fr.q())
        self.pins.move(obj(self.TOKEN), park(290), fr.q())

    def g_handle_source(self, p):
        """Look-alike: the person picks the source up, holds it for a moment and puts it back without dosing."""
        fr = self.fr; sn = self.SOURCE[0]
        e = np.asarray(fr.p(*self.ENTRY, 0)[:2]); x = np.asarray(fr.p(*self.EXIT, 0)[:2]); s_xy = self.stand("source")
        yield from p.g_enter(e[0], e[1], math.atan2(s_xy[1] - e[1], s_xy[0] - e[0]))
        yield from p.g_walk([s_xy], face=self.face_to(s_xy, self.SOURCE[1:3]))
        yield from p.g_take(sn)
        self.ep.ev("person_handled_source")
        yield from p.g_wait(2.0)
        yield from p.g_put(self.src_home)
        yield from p.g_wait(0.6)
        yield from p.g_walk([x])
        yield from p.g_leave()

    def visible_text(self):
        return f"House rule: whoever gives a {self.dose_word} puts a red token next to that {self.rec_word}."

    rec_word = "cup"

    def stand(self, key):
        df, dl = self.STAND[key]
        return np.asarray(self.fr.p(df, dl, 0)[:2])

    def face_to(self, xy, target_df_dl):
        t = np.asarray(self.fr.p(*target_df_dl, 0)[:2])
        return math.atan2(t[1] - xy[1], t[0] - xy[0])

    def g_dose_by_person(self, p, k):
        fr = self.fr; sn = self.SOURCE[0]
        e = np.asarray(fr.p(*self.ENTRY, 0)[:2]); x = np.asarray(fr.p(*self.EXIT, 0)[:2])
        s_xy = self.stand("source"); r_xy = self.stand(k)
        rn, rdf, rdl, _ = self.RECEPTACLES[k]
        yield from p.g_enter(e[0], e[1], math.atan2(s_xy[1] - e[1], s_xy[0] - e[0]))
        yield from p.g_walk(self.path_to(e, s_xy), face=self.face_to(s_xy, self.SOURCE[1:3]))
        yield from p.g_take(sn)
        yield from p.g_walk(self.path_to(s_xy, r_xy), face=self.face_to(r_xy, (rdf, rdl)))
        above = self.surface_point(rdf, rdl, self.hz[rn] * 2 + 0.16)
        self.ep.ev("person_dose_start", receptacle=k + 1)
        yield from p.g_pour(above)
        self.count[rn] += 1
        self.ep.ev("person_dosed", receptacle=k + 1)
        if self.visible:
            yield from p.g_wait(0.3)
            self.pins.move(obj(self.TOKEN), self.surface_point(rdf - 0.02, rdl - 0.10, 0.007), fr.q())
        yield from p.g_walk(self.path_to(r_xy, s_xy), face=self.face_to(s_xy, self.SOURCE[1:3]))
        yield from p.g_put(self.src_home)
        yield from p.g_wait(0.6)
        yield from p.g_walk(self.path_to(s_xy, x))
        yield from p.g_leave()

    def g_walkby(self, p):
        """Null variant: the same person comes by, stops at the table for a moment without dosing, and leaves."""
        e = np.asarray(self.fr.p(*self.ENTRY, 0)[:2]); x = np.asarray(self.fr.p(*self.EXIT, 0)[:2])
        mid = self.stand("source")
        yield from p.g_enter(e[0], e[1], math.atan2(mid[1] - e[1], mid[0] - e[0]))
        yield from p.g_walk(self.path_to(e, mid), face=self.face_to(mid, self.RECEPTACLES[0][1:3]))
        yield from p.g_wait(3.5)
        yield from p.g_walk(self.path_to(mid, x))
        yield from p.g_leave()

    def path_to(self, a, b):
        return [b]

    def skill_dose(self, ep, slot, target):
        if slot is None or not slot.startswith("rec"):
            ep.result = f"mark {target} is not a receptacle"; return
        k = int(slot[3:]); rn, rdf, rdl, _ = self.RECEPTACLES[k]
        sn = self.SOURCE[0]; o = obj(sn); s = self.slots["source"]; wb = s["base"]
        yield from g_base(self.A, [wb[:2]], final_yaw=wb[2], speed=0.35)
        yield from self.armA.g_pick(wb, "right", self.grasp_point("source"), obj=o, on_grasp=lambda: self.pins.unpin(o),
                                    keep_quat=self.fr.q())
        above = self.surface_point(rdf, rdl, self.hz[rn] * 2 + 0.22)
        pts = [above, above + np.array([0, 0, 0.04]), above]
        yield from self.armA.g_gesture(wb, "right", pts, seconds_each=0.6)
        self.count[rn] += 1; self.agent_doses.append(k + 1)
        yield from self.armA.g_place(wb, "right", self.grasp_point("source"), obj=o, quat=self.fr.q(), pins=self.pins)
        yield from g_base(self.A, [self.baseA[:2]], final_yaw=self.baseA[2], speed=0.35)
        ep.result = f"gave one {self.dose_word} to mark {target}"

    def privileged(self):
        return dict(count={k: v for k, v in getattr(self, "count", {}).items()}, agent_doses=list(getattr(self, "agent_doses", [])))

    def verify(self, final):
        tgt = self.RECEPTACLES[0][0]; c = final["count"].get(tgt, 0)
        wrong = [k for k in final["agent_doses"] if k != 1]
        if wrong:
            return False, f"the robot dosed the wrong receptacle (mark {wrong[0]})"
        if c == 1:
            return True, f"exactly one {self.dose_word}"
        if c == 0:
            return False, f"no {self.dose_word} was given"
        return False, f"{c} doses: one too many"


# ----------------------------------------------------------------------------- WHICH: deliver the one that was at mark 1
class TrackOne(V3Scenario):
    """Identical items at places 1..K (the target starts at 1) plus free places; a person may move the target and put a
    look-alike where it was (event), move only a look-alike (decoy) or just pass by (null). Deliver the target."""
    skills = ("pick", "place", "wait", "done")
    budget = 2
    ITEMS = ()            # object names, identical; ITEMS[0] is the target
    PLACES = ()           # ((df, dl, label), ...): first len(ITEMS) start occupied, the rest free
    DEST = None           # (df, dl, label)
    STAND = {}            # place index -> (df, dl)
    ENTRY = (0, 0); EXIT = (0, 0)
    item_word = "item"

    def build_scenario(self):
        n0 = self.ITEMS[0]
        lo, hi = [npf(v) for v in obj(n0).aabb]
        hz = float(hi[2] - lo[2]) / 2 + 0.002
        for n in self.ITEMS:
            self.hz[n] = hz
        for i, (df, dl, lab) in enumerate(self.PLACES):
            a = self.arm_for(dl)
            self.slots[f"p{i}"] = dict(pos=self.surface_point(df, dl), label=lab, arm=a, grasp_z=hz * 1.1, base=self.wbase(df, dl, a))
        df, dl, lab = self.DEST
        a = self.arm_for(dl)
        self.slots["dest"] = dict(pos=self.surface_point(df, dl, self.dest_z), label=lab, arm=a, grasp_z=hz * 1.1, base=self.wbase(df, dl, a))

    dest_z = 0.0

    def arm_for(self, dl):
        return "left" if dl > 0.05 else "right"

    def setup(self, ep, rng):
        fr = self.fr
        self.pin_home(self.PREFIX)
        for i, n in enumerate(self.ITEMS):
            self.put(n, f"p{i}")
        for i in range(len(self.ITEMS), len(self.PLACES)):
            self.occ[f"p{i}"] = None
        self.occ["dest"] = None
        self.pins.move(obj(self.MARKER), park(291), fr.q())
        p = list(self.people.values())[0]
        t0 = rng.uniform(3.0, 7.0)
        free = list(range(len(self.ITEMS), len(self.PLACES)))
        if self.variant == "event":
            a = int(rng.choice(free)); other = int(rng.integers(1, len(self.ITEMS))) if hasattr(rng, "integers") else rng.randrange(1, len(self.ITEMS))
            moves = [(0, a), (other, 0)]
        elif self.variant == "decoy":
            a = int(rng.choice(free)); other = rng.randrange(1, len(self.ITEMS))
            moves = [(other, a)]
        else:
            moves = []
        self.hidden.update(moves=[(m[0] + 1, m[1] + 1) for m in moves], t_person=round(t0, 2))
        if self.visible:
            self.run(self.g_marker(), 0.0, bg=True)
        self.run(self.g_person(p, moves), t0)

    def setup_hard(self, ep, rng):
        """Busy day: two visits; the person rearranges three items per visit; only in the event variant does one of the
        moves take the target (it goes to a free spot and a look-alike later takes its place)."""
        fr = self.fr
        self.pin_home(self.PREFIX)
        for i, n in enumerate(self.ITEMS):
            self.put(n, f"p{i}")
        for i in range(len(self.ITEMS), len(self.PLACES)):
            self.occ[f"p{i}"] = None
        self.occ["dest"] = None
        self.pins.move(obj(self.MARKER), park(291), fr.q())
        p = list(self.people.values())[0]
        occ = {i: (i if i < len(self.ITEMS) else None) for i in range(len(self.PLACES))}   # place -> item index
        moves_all = []
        target_moved = False
        for visit in range(2):
            moves = []
            for _ in range(2 if visit == 0 else 1):
                src_choices = [i for i, it in occ.items() if it is not None and it != 0 and not (self.variant == "event" and i == 0)]
                if self.variant == "event" and not target_moved and (visit == 1 or rng.random() < 0.5):
                    src = [i for i, it in occ.items() if it == 0][0]; target_moved = True
                else:
                    src = rng.choice(src_choices)
                free = [i for i, it in occ.items() if it is None]
                dst = rng.choice(free)
                moves.append((src, dst)); occ[dst] = occ[src]; occ[src] = None
            if self.variant == "event" and visit == 1:
                # a look-alike goes to where the target stood at the start
                home = 0
                if occ[home] is None:
                    src = rng.choice([i for i, it in occ.items() if it not in (None, 0)])
                    moves.append((src, home)); occ[home] = occ[src]; occ[src] = None
            moves_all.append(moves)
        if self.variant == "null":
            moves_all = [[], []]
        t0 = rng.uniform(5.0, 12.0)
        self.hidden.update(moves=[[(a + 1, b + 1) for a, b in m] for m in moves_all], t_person=round(t0, 2))
        if self.visible:
            self.run(self.g_marker(), 0.0, bg=True)
        self.run(self.g_visits([self.g_person(p, m) for m in moves_all], t0, gap=(8.0, 16.0), rng=rng), t0)

    def visible_text(self):
        return f"Today the {self.target_word} carries a small red sticker on top."

    target_word = "target item"

    def g_marker(self):
        """Visible twin: a red sticker rides on the target wherever it goes."""
        tgt = obj(self.ITEMS[0]); m = obj(self.MARKER)
        while True:
            p = self.pins.pos(tgt) if tgt.name in self.pins.p else get_pose(tgt)[0]
            self.pins.move(m, p + np.array([0, 0, self.hz[self.ITEMS[0]] + 0.012]), self.fr.q())
            yield

    def stand(self, i):
        df, dl = self.STAND[i]
        return np.asarray(self.fr.p(df, dl, 0)[:2])

    def g_person(self, p, moves):
        fr = self.fr
        e = np.asarray(fr.p(*self.ENTRY, 0)[:2]); x = np.asarray(fr.p(*self.EXIT, 0)[:2])
        first = self.stand(moves[0][0]) if moves else self.stand(0)
        yield from p.g_enter(e[0], e[1], math.atan2(first[1] - e[1], first[0] - e[0]))
        if not moves:
            yield from p.g_walk([first], face=self.face(first, 0))
            yield from p.g_wait(3.5)
        for src, dst in moves:
            s_xy = self.stand(src); d_xy = self.stand(dst)
            name = self.occ[f"p{src}"]
            yield from p.g_walk([s_xy], face=self.face(s_xy, src))
            self.occ[f"p{src}"] = None
            yield from p.g_take(name)
            yield from p.g_walk([d_xy], face=self.face(d_xy, dst))
            df, dl, _ = self.PLACES[dst]
            yield from p.g_put(self.surface_point(df, dl, self.hz[name]))
            self.occ[f"p{dst}"] = name
            self.ep.ev("person_moved", src=src + 1, dst=dst + 1)
            yield from p.g_wait(0.4)
        yield from p.g_walk([x])
        yield from p.g_leave()

    def face(self, xy, i):
        df, dl, _ = self.PLACES[i]
        t = np.asarray(self.fr.p(df, dl, 0)[:2])
        return math.atan2(t[1] - xy[1], t[0] - xy[0])

    def legend(self):
        out = {str(i + 1): lab for i, (_, _, lab) in enumerate(self.PLACES)}
        out[str(len(self.PLACES) + 1)] = self.DEST[2]
        return out

    def marked(self):
        return [dict(pos=self.slots[f"p{i}"]["pos"], label=l) for i, (_, _, l) in enumerate(self.PLACES)] + \
               [dict(pos=self.slots["dest"]["pos"], label=self.DEST[2])]

    def slot_by_mark(self, target):
        try:
            k = int(str(target).strip()) - 1
        except Exception:
            return None
        if 0 <= k < len(self.PLACES):
            return f"p{k}"
        if k == len(self.PLACES):
            return "dest"
        return None

    def describe(self, name):
        return self.item_word

    def privileged(self):
        return dict(target_at=self.where(self.ITEMS[0]), dest=self.occ.get("dest"))

    def verify(self, final):
        if final["target_at"] == "dest":
            return True, f"the right {self.item_word} was delivered"
        if final["dest"]:
            return False, f"a look-alike {self.item_word} was delivered"
        return False, f"nothing delivered (target at {final['target_at']})"


# ----------------------------------------------------------------------------- WHERE: which closed box is it in
class Stash(V3Scenario):
    """An item (place K+1) and K closed lidded boxes (1..K). A person puts the item into one box (event), puts it into one
    box and later moves it to another (decoy: only the last box counts) or leaves it where it is (null). Bring the item to
    the destination (K+2). Opening, picking and placing each cost one action (budget 3): at most one box can be opened."""
    skills = ("open", "close", "pick", "place", "wait", "done")
    budget = 3
    ITEM = None           # (name, df, dl, label)
    BOXES = ()            # ((body prefix, df, dl, label), ...): bodies built as primitives, lids named <prefix>_lid
    DEST = None           # (df, dl, label)
    BOX = (0.26, 0.22, 0.16)   # sf, sl, sz of a box
    LID_OPEN = (-0.15, 0.0)    # an open lid is slid toward the robot, hanging over the near edge (station-frame offset)
    LID_AJAR = (-0.05, 0.0)    # visible twin: the last lid the person touched is left slightly open
    STAND = {}            # "item" or box index -> (df, dl)
    ENTRY = (0, 0); EXIT = (0, 0)
    ITEM_MARK_D = None    # (d_forward, d_left) offset of the item's mark from the item (None: on the item)

    @classmethod
    def box_objects(cls, fr, h, rgba, lid_rgba):
        out = []
        sf, sl, sz = cls.BOX
        for pre, df, dl, _ in cls.BOXES:
            out += open_bin(pre, fr, df, dl, h, sf, sl, sz, rgba)
            out += [box(f"{pre}_lid", fr, df, dl, h + sz + 0.012, sf + 0.02, sl + 0.02, 0.024, lid_rgba)]
        return out

    def build_scenario(self):
        n, df, dl, lab = self.ITEM
        lo, hi = [npf(v) for v in obj(n).aabb]
        self.hz[n] = float(hi[2] - lo[2]) / 2 + 0.002
        sf, sl, sz = self.BOX
        self.lids = {}
        for i, (pre, bdf, bdl, blab) in enumerate(self.BOXES):
            a = self.arm_for(bdl)
            self.slots[f"box{i}"] = dict(pos=self.surface_point(bdf, bdl, 0.02), label=blab, arm=a, grasp_z=self.hz[n] * 1.2,
                                         mark_pos=self.surface_point(bdf, bdl, sz + 0.04), base=self.wbase(bdf, bdl, a))
            closed = self.surface_point(bdf, bdl, sz + 0.012)
            opened = self.surface_point(bdf + self.LID_OPEN[0], bdl + self.LID_OPEN[1], sz + 0.012 + 0.03)
            self.lids[i] = LidBox(self, f"box{i}", f"{pre}_lid", closed, opened)
        self.slots["item"] = dict(pos=self.surface_point(df, dl), label=lab, arm=self.arm_for(dl), grasp_z=self.hz[n] * 1.2,
                                  base=self.wbase(df, dl, self.arm_for(dl)))
        if self.ITEM_MARK_D:   # main_1115: draw the item's mark beside the item instead of on top of it
            self.slots["item"]["mark_pos"] = self.surface_point(df + self.ITEM_MARK_D[0], dl + self.ITEM_MARK_D[1])
        ddf, ddl, dlab = self.DEST
        self.slots["dest"] = dict(pos=self.surface_point(ddf, ddl, self.dest_z), label=dlab, arm=self.arm_for(ddl), grasp_z=self.hz[n] * 1.2,
                                  base=self.wbase(ddf, ddl, self.arm_for(ddl)))

    dest_z = 0.0

    def arm_for(self, dl):
        return "left" if dl > 0.05 else "right"

    def legend(self):
        out = {str(i + 1): b[3] for i, b in enumerate(self.BOXES)}
        out[str(len(self.BOXES) + 1)] = self.ITEM[3]
        out[str(len(self.BOXES) + 2)] = self.DEST[2]
        return out

    def marked(self):
        return [dict(pos=self.slots[f"box{i}"]["mark_pos"], label=b[3]) for i, b in enumerate(self.BOXES)] + \
               [dict(pos=self.slots["item"].get("mark_pos", self.slots["item"]["pos"]), label=self.ITEM[3]), dict(pos=self.slots["dest"]["pos"], label=self.DEST[2])]

    def slot_by_mark(self, target):
        try:
            k = int(str(target).strip()) - 1
        except Exception:
            return None
        K = len(self.BOXES)
        if 0 <= k < K:
            return f"box{k}"
        return {K: "item", K + 1: "dest"}.get(k)

    def setup(self, ep, rng):
        fr = self.fr
        self.pin_home(self.PREFIX)
        for i, lb in self.lids.items():
            self.containers[f"box{i}"] = {"open": False, "box": lb}; lb.cur_open = lb.open_pos.copy(); lb.set(False)
            self.occ[f"box{i}"] = None
        self.put(self.ITEM[0], "item"); self.occ["dest"] = None; self.ajar_k = None
        p = list(self.people.values())[0]
        K = len(self.BOXES); t0 = rng.uniform(3.0, 6.0)
        k = rng.randrange(K)
        if self.variant == "event":
            plan = [k]
        elif self.variant == "decoy":
            j = rng.choice([i for i in range(K) if i != k]); plan = [j, k]
        else:
            plan = []
        self.hidden.update(boxes=[b + 1 for b in plan], t_person=round(t0, 2))
        self.run(self.g_person(p, plan), t0)

    def setup_hard(self, ep, rng):
        """Busy day: the person also looks into other boxes (open, look, close) during two visits; the item goes into one
        box (event), into one and later into another (decoy) or stays where it is (null)."""
        self.setup(ep, rng)          # sets plan and the single visit; replace the visit with the busy-day version
        self.runners = []
        p = list(self.people.values())[0]
        K = len(self.BOXES); plan = [b - 1 for b in self.hidden["boxes"]]
        peeks = [rng.randrange(K) for _ in range(3)]
        t0 = rng.uniform(5.0, 12.0)
        self.hidden.update(peeks=[b + 1 for b in peeks], t_person=round(t0, 2))
        first = self.g_peeks(p, peeks[:2])
        second = self.g_person(p, plan, peek_before=peeks[2])
        self.run(self.g_visits([first, second], t0, gap=(6.0, 14.0), rng=rng), t0)

    def g_peeks(self, p, boxes):
        """The person looks into boxes (open, look, close) without taking or leaving anything."""
        fr = self.fr
        e = np.asarray(fr.p(*self.ENTRY, 0)[:2]); x = np.asarray(fr.p(*self.EXIT, 0)[:2])
        b0 = self.stand(boxes[0])
        yield from p.g_enter(e[0], e[1], math.atan2(b0[1] - e[1], b0[0] - e[0]))
        for k in boxes:
            b_xy = self.stand(k)
            yield from p.g_walk([b_xy], face=self.face_xy(b_xy, *self.BOXES[k][1:3]))
            yield from self.g_person_lid(p, k, True)
            self.ep.ev("person_peeked", box=k + 1)
            yield from p.g_wait(1.5)
            yield from self.g_person_lid(p, k, False)
        yield from p.g_walk([x])
        yield from p.g_leave()

    def stand(self, key):
        df, dl = self.STAND[key]
        return np.asarray(self.fr.p(df, dl, 0)[:2])

    def face_xy(self, xy, df, dl):
        t = np.asarray(self.fr.p(df, dl, 0)[:2])
        return math.atan2(t[1] - xy[1], t[0] - xy[0])

    def g_person(self, p, plan, peek_before=None):
        fr = self.fr; name = self.ITEM[0]
        e = np.asarray(fr.p(*self.ENTRY, 0)[:2]); x = np.asarray(fr.p(*self.EXIT, 0)[:2])
        i_xy = self.stand("item")
        yield from p.g_enter(e[0], e[1], math.atan2(i_xy[1] - e[1], i_xy[0] - e[0]))
        if peek_before is not None:
            b_xy = self.stand(peek_before)
            yield from p.g_walk([b_xy], face=self.face_xy(b_xy, *self.BOXES[peek_before][1:3]))
            yield from self.g_person_lid(p, peek_before, True)
            self.ep.ev("person_peeked", box=peek_before + 1)
            yield from p.g_wait(1.5)
            yield from self.g_person_lid(p, peek_before, False)
        yield from p.g_walk([i_xy], face=self.face_xy(i_xy, *self.ITEM[1:3]))
        if not plan:
            yield from p.g_wait(3.5)
        else:
            self.occ["item"] = None
            yield from p.g_take(name)
            src = None
            for step, k in enumerate(plan):
                if src is not None:     # take it out of the previous box again
                    s_xy = self.stand(src)
                    yield from p.g_walk([s_xy], face=self.face_xy(s_xy, *self.BOXES[src][1:3]))
                    yield from self.g_person_lid(p, src, True)
                    yield from p.g_take(name)
                    self.occ[f"box{src}"] = None
                    yield from self.g_person_lid(p, src, False)
                    self.ep.ev("person_took_out", box=src + 1)
                b_xy = self.stand(k)
                yield from p.g_walk([b_xy], face=self.face_xy(b_xy, *self.BOXES[k][1:3]))
                yield from self.g_person_lid(p, k, True)
                yield from p.g_put(self.surface_point(self.BOXES[k][1], self.BOXES[k][2], 0.02 + self.hz[name]))
                self.occ[f"box{k}"] = name
                ajar = self.visible and step == len(plan) - 1
                yield from self.g_person_lid(p, k, False, ajar=ajar)
                self.ep.ev("person_put_in", box=k + 1)
                src = k
                if step < len(plan) - 1:          # steps back, hesitates, changes their mind
                    back = np.asarray(b_xy) + (np.asarray(b_xy) - np.asarray(self.fr.p(*self.BOXES[k][1:3], 0)[:2])) * 0.8
                    yield from p.g_walk([back]); yield from p.g_wait(2.0)
        yield from p.g_walk([x])
        yield from p.g_leave()

    def g_container(self, bot, arm_, slot, open_, base):
        k = int(slot[3:]); lb = self.lids[k]
        if open_ and getattr(self, "ajar_k", None) == k:
            self.pins.move(obj(lb.lid), lb.closed_pos, self.fr.q())
        if base != self.baseA:
            yield from g_base(bot, [base[:2]], final_yaw=base[2], speed=0.35)
        yield from lb.g_move(bot, arm_, base, open_, arm=self.slots[slot].get("arm", "right"))
        if base != self.baseA:
            yield from g_base(bot, [self.baseA[:2]], final_yaw=self.baseA[2], speed=0.35)

    def describe(self, name):
        return self.item_word

    def g_person_lid(self, p, k, open_, ajar=False):
        lb = self.lids[k]; q = self.fr.q()
        closed = lb.closed_pos; opened = lb.open_pos
        if ajar:
            opened_target = closed + (np.asarray(self.fr.p(*self.LID_AJAR, 0.01)) - np.asarray(self.fr.p(0, 0, 0)))
            yield from p.g_lift_lid(lb.lid, opened, opened_target, q)
            self.containers[f"box{k}"]["open"] = False; self.ajar_k = k
            return
        if open_:
            yield from p.g_lift_lid(lb.lid, closed, opened, q)
        else:
            yield from p.g_lift_lid(lb.lid, opened, closed, q)
        self.containers[f"box{k}"]["open"] = open_

    def privileged(self):
        return dict(item_at=self.where(self.ITEM[0]), open_boxes=[i + 1 for i in range(len(self.BOXES))
                                                                   if self.containers.get(f"box{i}", {}).get("open")])

    def verify(self, final):
        if final["item_at"] == "dest":
            return True, f"the {self.item_word} were delivered" if self.item_word.endswith("s") else f"the {self.item_word} was delivered"
        return False, f"not delivered (the {self.item_word} is at {final['item_at']})"

    item_word = "item"


# ----------------------------------------------------------------------------- STILL: is someone still in the zone
class ZoneClear(V3Scenario):
    """A zone with a tall screen. A person may walk in behind the screen and stay until some time after the agent takes
    over (event), walk in and leave again before (decoy), or nobody comes (null). The agent must `start` the machine
    for the zone, never while someone is inside, and no later than `grace_s` after the zone is empty."""
    skills = ("start", "wait", "done")
    budget = 1
    grace_s = 30.0
    wait_actors = False
    EVENT_STAY = (25.0, 70.0)    # event: the person leaves this long after the agent takes over
    ENTRY = (0, 0); INSIDE = (0, 0); EXIT = (0, 0); PATH_IN = (); PATH_OUT = ()
    machine_word = "start the machine"
    extra_help = {"start": "start: start the machine for the zone (one action; it cannot be undone)"}

    def build_scenario(self):
        self.slots["machine"] = dict(pos=np.asarray(self.MACHINE_POS, float), label=self.MACHINE_LABEL, reach=False)

    def legend(self):
        return {"1": self.MACHINE_LABEL, "2": self.ZONE_LABEL}

    def marked(self):
        return [dict(pos=np.asarray(self.MACHINE_POS, float), label=self.MACHINE_LABEL),
                dict(pos=np.asarray(self.ZONE_MARK, float), label=self.ZONE_LABEL)]

    def slot_by_mark(self, target):
        return "machine"

    def execute(self, ep, skill, target):
        if skill == "start":
            return self._busy(self.skill_start(ep))
        return super().execute(ep, skill, target)

    def setup(self, ep, rng):
        self.pin_home(self.PREFIX)
        self.screen_visible()
        p = list(self.people.values())[0]
        self.inside = False; self.started_at = None; self.started_occupied = None; self.empty_since = 0.0
        t0 = rng.uniform(3.0, 8.0)
        if self.variant == "event":
            t_out = self.prelude_s + rng.uniform(*self.EVENT_STAY)
        elif self.variant == "decoy":
            t_out = rng.uniform(t0 + 12.0, self.prelude_s - 10.0)
        else:
            t_out = None
        self.hidden.update(t_in=round(t0, 2), t_out=None if t_out is None else round(t_out, 2))
        if t_out is not None:
            self.run(self.g_person(p, t0, t_out), t0)
        else:
            self.run(self.g_passby(p), t0)

    def setup_hard(self, ep, rng):
        """Busy day: the person also walks past the cell; event: passes by, later goes in and is still inside at the
        takeover; decoy: goes in, comes out, later passes by again; null: passes by twice."""
        self.pin_home(self.PREFIX)
        self.screen_visible()
        p = list(self.people.values())[0]
        self.inside = False; self.started_at = None; self.started_occupied = None; self.empty_since = 0.0
        t0 = rng.uniform(5.0, 12.0)
        if self.variant == "event":
            t_in = rng.uniform(35.0, 55.0); t_out = self.prelude_s + rng.uniform(*self.EVENT_STAY)
            plan = [("pass", t0), ("in", t_in, t_out)]
        elif self.variant == "decoy":
            t_out = t0 + rng.uniform(25.0, 40.0)
            plan = [("in", t0, t_out), ("pass", t_out + rng.uniform(18.0, 26.0))]
        else:
            plan = [("pass", t0), ("pass", rng.uniform(40.0, 60.0))]
        self.hidden.update(plan=[list(x) for x in plan], t_out=(plan[-1][2] if plan[-1][0] == "in" else (plan[0][2] if plan[0][0] == "in" else None)))
        self.run(self.g_plan(p, plan), t0)

    def g_plan(self, p, plan):
        for step in plan:
            yield from self.g_until(step[1])
            if step[0] == "pass":
                yield from self.g_passby(p)
            else:
                yield from self.g_person(p, step[1], step[2])

    def screen_visible(self):
        """Visible twin: the tall screen is swapped for a low one (the person stays in view)."""
        for n, (pz, q) in self.home.items():
            if n.startswith(self.PREFIX + "screen"):
                p = pz.copy()
                if self.visible:
                    p[2] -= self.screen_drop
                self.pins.move(obj(n), p, q)

    screen_drop = 1.0

    def g_person(self, p, t0, t_out):
        fr = self.fr
        e = np.asarray(fr.p(*self.ENTRY, 0)[:2]); ins = np.asarray(fr.p(*self.INSIDE, 0)[:2]); x = np.asarray(fr.p(*self.EXIT, 0)[:2])
        pin = [np.asarray(fr.p(*q, 0)[:2]) for q in self.PATH_IN]; pout = [np.asarray(fr.p(*q, 0)[:2]) for q in self.PATH_OUT]
        yield from p.g_enter(e[0], e[1], math.atan2((pin[0] if pin else ins)[1] - e[1], (pin[0] if pin else ins)[0] - e[0]))
        yield from p.g_walk(pin + [ins], speed=0.75)
        self.inside = True; self.ep.ev("person_entered_zone")
        while self.ep.world_t < t_out:
            yield
        yield from p.g_walk(pout + [x], speed=0.75)
        self.inside = False; self.empty_since = self.ep.world_t; self.ep.ev("person_left_zone")
        yield from p.g_leave()

    def g_passby(self, p):
        fr = self.fr
        e = np.asarray(fr.p(*self.ENTRY, 0)[:2]); x = np.asarray(fr.p(*getattr(self, "PASS_EXIT", self.EXIT), 0)[:2])
        mid = np.asarray(fr.p(*self.PASSBY, 0)[:2])
        yield from p.g_enter(e[0], e[1], math.atan2(mid[1] - e[1], mid[0] - e[0]))
        yield from p.g_walk([mid, x], speed=0.85)
        yield from p.g_leave()

    def skill_start(self, ep):
        b = self.baseA
        q, _ = self.armA.reach(b, "right", np.asarray(self.BUTTON_POS, float))
        yield from self.armA.g_to(q, 1.2, "right", closed=True)
        self.started_at = ep.world_t; self.started_occupied = bool(self.inside)
        ep.ev("machine_started", occupied=self.started_occupied)
        self.on_start(ep)
        yield from self.armA.g_to(self.qA0, 1.0, "right", closed=False)
        ep.result = f"{self.machine_word}: started"

    def on_start(self, ep):
        pass

    def privileged(self):
        return dict(occupied=bool(getattr(self, "inside", False)), started_at=getattr(self, "started_at", None),
                    started_occupied=getattr(self, "started_occupied", None), empty_since=getattr(self, "empty_since", 0.0))

    def verify(self, final):
        if final["started_at"] is None:
            return False, "never started"
        if final["started_occupied"]:
            return False, "started while a person was inside the zone"
        late = final["started_at"] - max(final["empty_since"] or 0.0, self.prelude_s)
        if late > self.grace_s:
            return False, f"started {late:.0f} s after the zone was clear (limit {self.grace_s:.0f} s)"
        return True, "started with the zone clear and in time"
