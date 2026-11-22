"""StationScenario: logical slot occupancy + generic pick / place / open / close for the agent robot.

The logical state (which object occupies which slot, which lids are open, counters) is the source of
truth; object poses are derived from it (pins) or animated by movers. The verifier reads the logical
state, so success never depends on fragile contact physics.
"""
import math
import numpy as np


def look_yaw(fr, df, dl, tdf, tdl):
    """Yaw (world) of a robot at (df, dl) in frame fr that faces the frame point (tdf, tdl)."""
    a = np.asarray(fr.p(tdf, tdl, 0))[:2] - np.asarray(fr.p(df, dl, 0))[:2]
    return math.atan2(a[1], a[0])


def robot_by_name(name):
    return [r for r in env.robots if r.name == name][0]


def head_sensor(r):
    for k, s in r.sensors.items():
        if "eyes" in k:
            return s
    return list(r.sensors.values())[0]


class StationScenario(Scenario):
    ANCHOR_KEY = None
    CAM = None              # (eye, target, focal) in world coords, or callable(fr) -> tuple
    B_START = None          # station-frame (df, dl, dyaw) where the helper waits, or None (no helper)
    A_BASE = (0.0, 0.0, 0.0)
    hand_z = 0.95

    def __init__(self):
        super().__init__()
        a = SCENE_SPEC["anchors"][self.ANCHOR_KEY]
        self.fr = Frame(*((a.get("surface") or a.get("base")) if isinstance(a, dict) else a))
        self.occ = {}; self.hz = {}; self.containers = {}
        self.hidden = {}; self.log_state = {}
        self.A_busy = False

    # ------------------------------------------------------------------ setup
    def make_cams(self):
        eye, tgt, focal = self.CAM(self.fr) if callable(self.CAM) else self.CAM
        self.main_cam = make_cam(f"{self.sid}_main", 1280, 720, eye, tgt, focal)

    def build(self):
        self.Ar = robot_by_name("robot"); self.A = Bot(self.Ar); self.armA = Arm(self.A, hand_z=self.hand_z, tag="A")
        self.head = head_sensor(self.Ar)
        disable_robot_cams(self.Ar, keep=[k for k in self.Ar.sensors if "eyes" in k])
        if self.B_START is not None:
            self.Br = robot_by_name("helper"); self.B = Bot(self.Br); self.armB = Arm(self.B, hand_z=self.hand_z, tag="B")
            disable_robot_cams(self.Br)
        else:
            self.B = None
        self.baseA = tuple(self.A_WORLD) if getattr(self, "A_WORLD", None) else self.fr.base(*self.A_BASE)
        self.qA0 = self.armA.neutral(self.baseA)
        if self.B is not None:
            self.baseB0 = tuple(self.B_WORLD) if getattr(self, "B_WORLD", None) else self.fr.base(*self.B_START)
            self.qB0 = self.armB.neutral(self.baseB0)
        self.build_scenario()
        self.home_poses()
        self.find_held_joints()
        self.precompute()

    def build_scenario(self):
        pass

    def home_poses(self):
        """Remember every own object's configured pose and switch its gravity off (props are posed kinematically)."""
        self.home = {}
        for cfg in self.OBJECTS:
            o = obj(cfg["name"])
            self.home[cfg["name"]] = (np.asarray(cfg.get("position", [0, 0, 0]), float), np.asarray(cfg.get("orientation", [0, 0, 0, 1]), float))
            for link in getattr(o, "links", {}).values():
                try:
                    link.disable_gravity()
                except Exception:
                    pass

    def find_held_joints(self, radius=4.0):
        """Scene furniture with joints (doors, drawers) near the station: held at their initial positions so that a robot
        brushing past cannot swing them open."""
        own = {c["name"] for c in self.OBJECTS}
        cx, cy = self.baseA[0], self.baseA[1]
        self.held_joints = []
        for o in env.scene.objects:
            if o.name in own or not getattr(o, "joints", None) or o in env.robots:
                continue
            try:
                lo, hi = [npf(v) for v in o.aabb]
                c = (lo + hi) / 2
                if math.hypot(c[0] - cx, c[1] - cy) > radius:
                    continue
                q = o.get_joint_positions().clone()
                if len(q):
                    self.held_joints.append((o, q))
            except Exception:
                pass

    def hold(self):
        for o, q in getattr(self, "held_joints", []):
            try:
                o.set_joint_positions(q); o.set_joint_velocities(th.zeros(len(q)))
            except Exception:
                pass

    def pin_home(self, prefix):
        """Pin every own object whose name starts with prefix at its configured pose."""
        for n, (p, q) in self.home.items():
            if n.startswith(prefix):
                self.pins.move(obj(n), p, q)

    def precompute(self):
        """Warm the IK cache for every slot (pick and place heights)."""
        for name, s in self.slots.items():
            if s.get("reach") is False:
                continue
            base = s.get("base", self.baseA)
            arm = s.get("arm", "right")
            gp = np.asarray(s["pos"]) + np.array([0, 0, s.get("grasp_z", 0.04)])
            if s.get("pre") is None:
                for dz in (0.14, 0.02, 0.03):
                    self.armA.reach(base, arm, gp, dz=dz, approach=s.get("approach"))
            else:
                self.armA.reach(base, arm, gp + np.asarray(s["pre"]), approach=s.get("approach"))

    def place_robots(self):
        self.A.place_base(*self.baseA); self.A.q = self.qA0.clone(); self.A.held = {}; self.A.fixed_quat = {}
        if self.B is not None:
            self.B.place_base(*self.baseB0); self.B.q = self.qB0.clone(); self.B.held = {}; self.B.fixed_quat = {}
        elif getattr(self, "_park_helper", True):
            try:
                hb = Bot(robot_by_name("helper")); hb.place_base(200.0, 200.0, 0.0); hb.apply()
                self._parked = hb
            except Exception:
                pass

    def put(self, objname, slot):
        """Logically and physically put an object at a slot."""
        self.occ[slot] = objname
        o = obj(objname)
        p = np.asarray(self.slots[slot]["pos"], float) + np.array([0, 0, self.hz.get(objname, 0.0)])
        self.pins.move(o, p, self.slots[slot].get("quat", self.fr.q()))

    def where(self, objname):
        for s, o in self.occ.items():
            if o == objname:
                return s
        if self.A is not None and objname in self.A.holding():
            return "agent_hand"
        if self.B is not None and objname in self.B.holding():
            return "helper_hand"
        return None

    def world(self, ep):
        pass          # robots are pushed to the simulator in Episode.sync (only when a frame is rendered)

    # ------------------------------------------------------------------ agent skills
    def execute(self, ep, skill, target):
        self._ep = ep
        slot = self.slot_by_mark(target) if target is not None else None
        if skill in ("pick", "place", "open", "close") and slot is None:
            ep.result = f"'{target}' is not a numbered mark (marks: {', '.join(self.legend())})"
            return iter(())
        fn = getattr(self, f"skill_{skill}")
        return self._busy(fn(ep, slot, target))

    # ------------------------------------------------------------------ v2s strict live-memory protocol
    VARIANTS = ("event", "decoy", "null")
    V2S_WINDOW = (0.3, 2.5)          # world seconds after the hand-over (world 0) at which the other actor becomes due

    def v2s_setup(self, ep, rng):
        """Variant = VARIANTS[seed % 3]; the actor's action is due at world U(0.3, 2.5) s and starts at the first tick
        after that at which the agent's robot stands still (A_busy False)."""
        self.variant = self.VARIANTS[ep.seed % len(self.VARIANTS)]
        self.t_ev = rng.uniform(*self.V2S_WINDOW)
        self.A_busy = False
        self.hidden.update(protocol="v2s", variant=self.variant, visible=bool(getattr(self, "visible", False)),
                           t_event_sched=round(self.t_ev, 2))

    def actor_due(self, ep):
        return ep.world_t >= self.t_ev and not self.A_busy

    def actor_started(self, ep, **kw):
        self.hidden["t_actor_started"] = round(ep.world_t, 2)
        ep.ev("actor_started", variant=self.variant, **kw)

    def workspace_busy(self):
        h = getattr(self, "helper", None)
        if h is not None and h.gen is not None and getattr(self, "helper_blocks", True):
            return f"another robot is working in the shared workspace ({h.status})"
        return None

    def _busy(self, gen):
        # v2s: a robot never reaches into a place another robot, machine or person is handling right now: the command
        # first waits (world time runs) until they are done. Silent: the agent gets no message about it.
        n = 0
        while self.workspace_busy() and n < 120 * SIM_HZ:
            n += 1
            yield
        if n and getattr(self, "_ep", None) is not None:
            self._ep.ev("agent_waited_for_actor", seconds=round(n * DT, 2))
            self.hidden["agent_waited_s"] = round(self.hidden.get("agent_waited_s", 0.0) + n * DT, 2)
        if getattr(self, "_commit_ep", None) is not None:
            self._commit_ep.commit_check()     # ablation A only
        self.A_busy = True
        try:
            yield from gen
        finally:
            self.A_busy = False

    def grasp_point(self, slot):
        s = self.slots[slot]
        return np.asarray(s["pos"], float) + np.array([0, 0, s.get("grasp_z", 0.04)])

    def skill_pick(self, ep, slot, target):
        s = self.slots[slot]; base = s.get("base", self.baseA); arm = s.get("arm", "right")
        if self.A.holding():
            ep.result = "the hand is full; place the item first"; return
        c = self.containers.get(slot)
        if c is not None and not c["open"]:
            yield from self.armA.g_pick(base, arm, self.grasp_point(slot) + np.array([0, 0, 0.12]))
            ep.result = f"mark {target} is closed; nothing grasped"; return
        name = self.occ.get(slot)
        via = [tuple(v) for v in s.get("via", [])]
        if base != self.baseA:
            yield from g_base(self.A, via + [base[:2]], final_yaw=base[2], speed=0.35)
        o = obj(name) if name else None

        def on_grasp():
            if name:
                self.occ[slot] = None; self.pins.unpin(o)
        yield from self.armA.g_pick(base, arm, self.grasp_point(slot), obj=o, on_grasp=on_grasp,
                                    keep_quat=None if o is None else self.pins.p.get(name, (None, None, self.fr.q()))[2],
                                    pre=s.get("pre"), approach=s.get("approach"))
        if base != self.baseA:
            yield from g_base(self.A, list(reversed(via)) + [self.baseA[:2]], final_yaw=self.baseA[2], speed=0.35)
        ep.result = f"grasped one item at mark {target}" if name else f"nothing to grasp at mark {target}"
        if name:
            self.on_agent_pick(ep, name, slot)

    def skill_place(self, ep, slot, target):
        s = self.slots[slot]; base = s.get("base", self.baseA); arm = s.get("arm", "right")
        held = self.A.holding()
        if not held:
            ep.result = "you are not holding anything"; return
        c = self.containers.get(slot)
        if c is not None and not c["open"]:
            ep.result = f"mark {target} is closed; open it first"; return
        if self.occ.get(slot) and not s.get("stack", False):
            ep.result = f"mark {target} is occupied"; return
        name = held[0]; o = obj(name)
        via = [tuple(v) for v in s.get("via", [])]
        if base != self.baseA:
            yield from g_base(self.A, via + [base[:2]], final_yaw=base[2], speed=0.35)
        p = np.asarray(s["pos"], float)

        def release():
            # the item is let go when the gripper opens, before the arm retreats (it must not ride back with the hand)
            self.A.detach(o)
            if s.get("stack", False):
                self.occ.setdefault(slot + "_items", []).append(name)
                self.pins.move(o, p + np.array([0, 0, self.hz.get(name, 0.0)]) + np.asarray(s.get("jitter", [0, 0, 0])) * len(self.occ[slot + "_items"]))
            else:
                self.put(name, slot)
        yield from self.armA.g_place(base, arm, p + np.array([0, 0, s.get("grasp_z", 0.04)]), pre=s.get("pre"), approach=s.get("approach"),
                                     on_release=release)
        if base != self.baseA:
            yield from g_base(self.A, list(reversed(via)) + [self.baseA[:2]], final_yaw=self.baseA[2], speed=0.35)
        ep.result = f"placed the item at mark {target}"
        self.on_agent_place(ep, name, slot)

    def skill_open(self, ep, slot, target):
        c = self.containers.get(slot)
        if c is None:
            ep.result = f"mark {target} has nothing to open"; return
        if c["open"]:
            ep.result = f"mark {target} is already open"; return
        yield from self.g_container(self.A, self.armA, slot, True, self.slots[slot].get("base", self.baseA))
        ep.result = f"opened mark {target}"
        self.on_agent_open(ep, slot)

    def skill_close(self, ep, slot, target):
        c = self.containers.get(slot)
        if c is None:
            ep.result = f"mark {target} has nothing to close"; return
        if not c["open"]:
            ep.result = f"mark {target} is already closed"; return
        yield from self.g_container(self.A, self.armA, slot, False, self.slots[slot].get("base", self.baseA))
        ep.result = f"closed mark {target}"

    def g_container(self, bot, arm_, slot, open_, base):
        raise NotImplementedError

    # hooks
    def on_agent_pick(self, ep, name, slot):
        pass

    def on_agent_place(self, ep, name, slot):
        pass

    def on_agent_open(self, ep, slot):
        pass


class LidBox:
    """A lidded box: body pinned; the lid is a separate object that is lifted off and set down beside the box."""

    def __init__(self, sc, slot, lid_name, lid_closed_pos, lid_open_pos):
        self.sc, self.slot, self.lid = sc, slot, lid_name
        self.closed_pos = np.asarray(lid_closed_pos, float); self.open_pos = np.asarray(lid_open_pos, float)
        self.cur_open = self.open_pos.copy()
        sc.containers[slot] = {"open": False, "box": self}

    def set(self, open_):
        self.sc.containers[self.slot]["open"] = open_
        self.sc.pins.move(obj(self.lid), self.cur_open if open_ else self.closed_pos, self.sc.fr.q())

    def g_move(self, bot, arm_, base, open_, arm="right", open_pos=None):
        """Grasp the lid and move it to the open pose (or open_pos) / back onto the box."""
        sc = self.sc; o = obj(self.lid)
        if open_ and open_pos is not None:
            self.cur_open = np.asarray(open_pos, float)
        src = self.closed_pos if open_ else self.cur_open
        dst = self.cur_open if open_ else self.closed_pos
        yield from arm_.g_pick(base, arm, src, obj=o, on_grasp=lambda: sc.pins.unpin(o), keep_quat=sc.fr.q(), lift=0.12)
        yield from arm_.g_place(base, arm, dst, obj=o, quat=sc.fr.q(), pins=sc.pins, lift=0.12)
        sc.containers[self.slot]["open"] = open_


class Hinged:
    """A door (or lid) that rotates about a hinge axis; posed every tick from its angle."""

    def __init__(self, sc, slot, name, hinge, axis, half_extent, angle_open, sz_center=None):
        self.sc, self.slot, self.name = sc, slot, name
        self.hinge = np.asarray(hinge, float)          # world point on the hinge axis
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)
        self.rel = np.asarray(half_extent, float)      # closed: door centre = hinge + rel
        self.angle_open = angle_open; self.angle = 0.0
        sc.containers[slot] = {"open": False, "door": self}
        self.q0 = np.asarray(sc.fr.q(), float)

    def pose(self):
        R = Rot.from_rotvec(self.axis * self.angle)
        c = self.hinge + R.apply(self.rel)
        q = (R * Rot.from_quat(self.q0)).as_quat()
        return c, q

    def handle(self, frac=0.9):
        R = Rot.from_rotvec(self.axis * self.angle)
        return self.hinge + R.apply(self.rel * 2 * frac)

    def apply(self):
        c, q = self.pose(); self.sc.pins.move(obj(self.name), c, q)

    def set(self, open_):
        self.angle = self.angle_open if open_ else 0.0
        self.sc.containers[self.slot]["open"] = open_
        self.apply()

    def g_swing(self, bot, arm_, base, open_, arm="right", seconds=1.6, steps=5):
        """Hand goes to the handle and follows it while the door swings."""
        a0 = self.angle; a1 = self.angle_open if open_ else 0.0
        h = self.handle()
        yield from arm_.g_to(arm_.reach(base, arm, h, approach=None, dz=0.06)[0], 1.0, arm, closed=False)
        yield from arm_.g_to(arm_.reach(base, arm, h, approach=None)[0], 0.6, arm, closed=True)
        n = int(seconds * SIM_HZ)
        keys = []
        for k in range(steps + 1):
            self.angle = a0 + (a1 - a0) * k / steps
            keys.append(arm_.reach(base, arm, self.handle(), approach=None)[0])
        self.angle = a0
        for i in range(1, n + 1):
            u = ease(i / n); self.angle = a0 + (a1 - a0) * u
            k = min(steps - 1, int(u * steps)); w = u * steps - k
            q = keys[k] + (keys[k + 1] - keys[k]) * w
            bot.q = q.clone(); bot.q[bot.GRIP[arm]] = 0.0
            self.apply(); yield
        self.angle = a1; self.apply()
        self.sc.containers[self.slot]["open"] = open_
        yield from arm_.g_to(arm_.neutral(base), 1.0, arm, closed=False)


class PathMover:
    """Moves objects along polylines at a fixed speed (conveyors, vacuum, turntables are separate)."""

    def __init__(self):
        self.jobs = {}

    def add(self, name, pts, speed, t_start, quat=None, on_done=None):
        pts = [np.asarray(p, float) for p in pts]
        L = np.cumsum([0] + [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)])
        self.jobs[name] = dict(pts=pts, L=L, v=speed, t0=t_start, quat=quat, done=False, on_done=on_done)

    def pos(self, j, t):
        s = max(0.0, (t - j["t0"]) * j["v"])
        if s >= j["L"][-1]:
            return j["pts"][-1], True
        i = int(np.searchsorted(j["L"], s, side="right") - 1)
        u = (s - j["L"][i]) / max(1e-9, j["L"][i + 1] - j["L"][i])
        return j["pts"][i] + (j["pts"][i + 1] - j["pts"][i]) * u, False

    def step(self, sc, t):
        for name, j in self.jobs.items():
            if j["done"] or t < j["t0"]:
                continue
            p, fin = self.pos(j, t)
            sc.pins.move(obj(name), p, j["quat"])
            if fin:
                j["done"] = True
                if j["on_done"]:
                    j["on_done"]()

    def busy(self):
        return any(not j["done"] for j in self.jobs.values())
