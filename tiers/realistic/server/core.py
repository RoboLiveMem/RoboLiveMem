"""RoboLiveMem-Bench runtime (exec'd inside the OmniGibson process by bench_server.py).

Everything an agent can change, and everything another robot or machine changes, is kinematic:
objects are pinned to poses that the scenario logic sets every tick, robots are posed by
numeric IK and attach-on-grasp. Physics only steps the scene; nothing can fly away.

Clocks
  paused : the world advances only while the robot executes a skill (or `wait`).
  live   : in addition, the wall time the agent spends between receiving a response and
           sending its next request is simulated before that request is served
           (lazy real time: identical outcome to a world that kept running, because the
           robot holds still between requests).

Stream
  The robot head camera is recorded at STREAM_HZ whenever the world advances. Every
  response delivers the frames recorded since the previous delivery, so changes that
  happened while the agent was thinking are observable after the fact.
"""
import math, json, time, io, base64, random, os
import numpy as np
import torch as th
import cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

SIM_HZ = 30
DT = 1.0 / SIM_HZ
STREAM_HZ = 5          # head-camera frames kept for the agent's stream
MAIN_HZ_ACT = 10       # spectator frames while the robot acts
MAIN_HZ_THINK = 5      # spectator frames while the agent deliberates (live clock)
MAX_CATCHUP = 900.0    # seconds of world time simulated for one request at most
# ablation A (commit check): window around the target mark, per-pixel change threshold, changed-pixel fraction
SETTLE_N = 6           # main_1115: fresh renders after the last change before images are reused
COMMIT_HALF = 64       # px at head_res 720 (window 128 x 128)
COMMIT_PIX = 24        # grey levels after a 5x5 blur
COMMIT_FRAC = 0.02


class CommitAbort(Exception):
    pass


# ----------------------------------------------------------------------------- helpers
def npf(x):
    return np.asarray(x.tolist() if hasattr(x, "tolist") else x, dtype=float)


def yq(yaw):
    return th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=th.float32)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def T_of(pos, quat):
    T = np.eye(4); T[:3, :3] = Rot.from_quat(list(quat)).as_matrix(); T[:3, 3] = list(pos); return T


def look_at_quat(eye, tgt):
    eye, tgt = np.array(eye, float), np.array(tgt, float)
    f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return Rot.from_matrix(np.stack([r, u, -f], axis=1)).as_quat()


def get_pose(o):
    p, q = o.get_position_orientation()
    return npf(p), npf(q)


def set_pose(o, pos, quat=None):
    if quat is None:
        quat = get_pose(o)[1]
    o.set_position_orientation(position=th.tensor(np.asarray(pos, float), dtype=th.float32),
                               orientation=th.tensor(np.asarray(quat, float), dtype=th.float32))
    try:
        o.set_linear_velocity(th.zeros(3)); o.set_angular_velocity(th.zeros(3))
    except Exception:
        pass


def rgb_of(sensor):
    if sensor is None:
        x = og.sim.viewer_camera.get_obs()[0]["rgb"]
    else:
        x = sensor.get_obs()[0]["rgb"]
    x = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return np.ascontiguousarray(x[..., :3], dtype=np.uint8)


def jpeg_b64(rgb, q=85):
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def decode_jpeg(b):
    return np.ascontiguousarray(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1])


def make_cam(name, w, h, eye, tgt, focal=None):
    from omnigibson.sensors import create_sensor
    s = create_sensor(sensor_type="VisionSensor", relative_prim_path=f"/{name}", name=name, modalities=["rgb"],
                      sensor_kwargs={"image_width": w, "image_height": h})
    s.load(env.scene); s.initialize()
    s.set_position_orientation(position=th.tensor(eye, dtype=th.float32), orientation=th.tensor(look_at_quat(eye, tgt), dtype=th.float32))
    if focal:
        s.focal_length = focal
    return s


def aim_cam(s, eye, tgt, focal=None):
    s.set_position_orientation(position=th.tensor(eye, dtype=th.float32), orientation=th.tensor(look_at_quat(eye, tgt), dtype=th.float32))
    if focal:
        s.focal_length = focal


def cam_project(sensor, pts, w, h):
    """World points -> pixel coordinates for an OmniGibson VisionSensor (USD camera: -Z forward, +Y up)."""
    pos, quat = get_pose(sensor)
    R = Rot.from_quat(quat).as_matrix()
    try:
        K = npf(sensor.intrinsic_matrix)
    except Exception:
        f = float(sensor.focal_length); ap = float(sensor.horizontal_aperture)
        fx = w * f / ap; K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
    out = []
    for p in pts:
        c = R.T @ (np.asarray(p, float) - pos)
        if c[2] >= -1e-3:
            out.append(None); continue
        u = K[0, 0] * (c[0] / -c[2]) + K[0, 2]
        v = K[1, 2] - K[1, 1] * (c[1] / -c[2])
        out.append((float(u), float(v)))
    return out


def disable_robot_cams(r, keep=()):
    for k, s in r.sensors.items():
        if k in keep:
            continue
        try:
            s.enabled = False
        except Exception:
            pass


# ----------------------------------------------------------------------------- kinematic robot
class Bot:
    def __init__(self, r):
        self.r = r
        self.ARM = {a: [int(i) for i in npf(r.arm_control_idx[a])] for a in ("left", "right")}
        self.TRUNK = [int(i) for i in npf(r.trunk_control_idx)]
        self.GRIP = {a: [int(i) for i in npf(r.gripper_control_idx[a])] for a in ("left", "right")}
        self.q = r.get_joint_positions().clone().float()
        p, o = r.get_position_orientation()
        self.x, self.y = float(p[0]), float(p[1])
        self.z = 0.0048
        self.yaw = float(Rot.from_quat(npf(o)).as_euler("xyz")[2])
        js = list(r.joints.values())
        self.lo = th.tensor([float(j.lower_limit) for j in js]); self.hi = th.tensor([float(j.upper_limit) for j in js])
        self.idx = th.arange(6, len(self.q))
        self.held = {}
        self.fixed_quat = {}

    def set_q(self, q):
        self.r.set_joint_positions(q[6:], indices=self.idx)

    def place_base(self, x, y, yaw):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)

    def apply(self):
        r = self.r
        r.set_position_orientation(position=th.tensor([self.x, self.y, self.z], dtype=th.float32), orientation=yq(self.yaw))
        self.set_q(self.q)
        try:
            r.set_joint_positions(self.q[6:], indices=self.idx, drive=True)
        except Exception:
            pass
        r.set_joint_velocities(th.zeros(len(self.q)))
        for name, (o, arm, T) in self.held.items():
            Tw = self.eef_T(arm) @ T
            quat = self.fixed_quat[name] if name in self.fixed_quat else Rot.from_matrix(Tw[:3, :3]).as_quat()
            set_pose(o, Tw[:3, 3], quat)

    def eef_T(self, arm):
        return T_of(npf(self.r.get_eef_position(arm)), npf(self.r.get_eef_orientation(arm)))

    def ik(self, arm, target, approach=None, use_trunk=True, w_ap=0.04, iters=80, q_init=None, reg=2e-3, base=None):
        if base is not None:
            self.place_base(*base)
        ids = (self.TRUNK if use_trunk else []) + self.ARM[arm]
        q0 = (q_init if q_init is not None else self.q).clone()
        x0 = npf(q0[ids]); lo = npf(self.lo[ids]); hi = npf(self.hi[ids])
        x0 = np.clip(x0, lo + 1e-3, hi - 1e-3)
        tgt = np.asarray(target, float)
        ap = None if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
        self.r.set_position_orientation(position=th.tensor([self.x, self.y, self.z], dtype=th.float32), orientation=yq(self.yaw))

        def f(x):
            q = q0.clone(); q[ids] = th.tensor(x, dtype=th.float32); self.set_q(q)
            p = npf(self.r.get_eef_position(arm)); c = float(np.sum((p - tgt) ** 2))
            if ap is not None:
                z = Rot.from_quat(npf(self.r.get_eef_orientation(arm))).as_matrix()[:, 2]
                c += w_ap * (1.0 - float(np.dot(z, ap)))
            return c + reg * float(np.sum((x - x0) ** 2))
        res = minimize(f, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": iters, "eps": 1e-3})
        q = q0.clone(); q[ids] = th.tensor(res.x, dtype=th.float32)
        self.set_q(q); err = float(np.linalg.norm(npf(self.r.get_eef_position(arm)) - tgt))
        self.set_q(self.q)
        return q, err

    def grip(self, q, arm, closed):
        q = q.clone(); q[self.GRIP[arm]] = 0.0 if closed else 0.05; return q

    def attach(self, o, arm, fixed_quat=None):
        self.apply()
        p, qq = get_pose(o)
        self.held[o.name] = (o, arm, np.linalg.inv(self.eef_T(arm)) @ T_of(p, qq))
        if fixed_quat is not None:
            self.fixed_quat[o.name] = np.asarray(fixed_quat, float)

    def detach(self, o):
        self.held.pop(o.name, None); self.fixed_quat.pop(o.name, None)

    def holding(self):
        return list(self.held.keys())


def g_joints(b, q_target, seconds):
    q0 = b.q.clone(); n = max(1, int(seconds * SIM_HZ))
    for i in range(1, n + 1):
        b.q = q0 + (q_target - q0) * ease(i / n); yield


def g_base(b, waypoints, final_yaw=None, speed=0.45, turn=1.3):
    def rot_to(yaw):
        d = (yaw - b.yaw + math.pi) % (2 * math.pi) - math.pi
        n = max(1, int(abs(d) / turn / DT)); y0 = b.yaw
        for i in range(1, n + 1):
            b.yaw = y0 + d * ease(i / n); yield
    for (wx, wy) in waypoints:
        dx, dy = wx - b.x, wy - b.y; dist = math.hypot(dx, dy)
        if dist < 1e-3:
            continue
        yield from rot_to(math.atan2(dy, dx))
        n = max(1, int(dist / speed / DT)); x0, y0 = b.x, b.y
        for i in range(1, n + 1):
            u = ease(i / n); b.x = x0 + dx * u; b.y = y0 + dy * u; yield
    if final_yaw is not None:
        yield from rot_to(final_yaw)


def g_still(seconds):
    for _ in range(max(0, int(round(seconds * SIM_HZ)))):
        yield


def g_seq(*gens):
    for g in gens:
        yield from g


def g_call(fn):
    fn()
    yield


# ----------------------------------------------------------------------------- pinned objects
class Pins:
    """Objects held at a target pose (unless a robot holds them). Writes to the simulator are lazy: only targets that
    changed since the last sync are pushed (`apply`), right before a frame is rendered."""
    def __init__(self):
        self.p = {}; self.dirty = set()

    def pin(self, o, pos=None, quat=None):
        if pos is None or quat is None:
            p0, q0 = get_pose(o)
        self.p[o.name] = (o, np.asarray(p0 if pos is None else pos, float), np.asarray(q0 if quat is None else quat, float))
        self.dirty.add(o.name)

    def move(self, o, pos, quat=None):
        if quat is None:
            quat = self.p[o.name][2] if o.name in self.p else get_pose(o)[1]
        self.p[o.name] = (o, np.asarray(pos, float), np.asarray(quat, float))
        self.dirty.add(o.name)

    def unpin(self, o):
        self.p.pop(o.name, None); self.dirty.discard(o.name)

    def pos(self, o):
        return self.p[o.name][1].copy()

    def apply(self, skip=(), all_=False):
        names = list(self.p) if all_ else list(self.dirty)
        for n in names:
            if n in skip or n not in self.p:
                continue
            o, p, q = self.p[n]
            set_pose(o, p, q)
        self.dirty.clear()


# ----------------------------------------------------------------------------- scenario base
class Scenario:
    """Subclass per scenario. Positions are world coordinates in the scenario's scene."""
    sid = "base"
    title = ""
    setting = ""
    budget = 3            # manipulation skills (wait / observe are free)
    budget_ends_episode = True
    deadline = 600.0      # world seconds
    skills = ("pick", "place", "wait", "done")
    head_res = 720

    def __init__(self):
        self.slots = {}   # name -> dict(pos, mark, label)
        self.pins = Pins()

    # --- hooks
    def build(self):
        pass

    def reset(self, ep, rng):
        raise NotImplementedError

    def world(self, ep):
        pass

    def instruction(self):
        raise NotImplementedError

    def privileged(self):
        return {}

    def verify(self, final):
        raise NotImplementedError

    def execute(self, ep, skill, target):
        """Generator yielding once per tick; sets ep.result."""
        raise NotImplementedError

    def workspace_busy(self):
        """Reason string while another robot or a machine works in the shared workspace (a manipulation command first
        waits silently until it is free; nothing is reported to the agent)."""
        return None

    SKILL_HELP = {
        "pick": "pick <mark>: grasp whatever is at the numbered place and lift it (one item at a time)",
        "place": "place <mark>: put the item you are holding at the numbered place",
        "open": "open <mark>: open the lid / door / drawer at the numbered place",
        "close": "close <mark>: close the lid / door / drawer at the numbered place",
        "wait": "wait <seconds>: hold still and let the world run (free; 0.5 to 60 s)",
        "done": "done: you are finished; ends the episode",
    }
    extra_help = {}

    def skill_help(self):
        h = dict(self.SKILL_HELP); h.update(self.extra_help)
        return {k: h[k] for k in self.skills if k in h}

    def holding_text(self):
        held = self.A.holding()
        return "nothing" if not held else f"one item ({self.describe(held[0])})"

    def describe(self, name):
        return "an object"

    # --- shared
    def legend(self):
        return {str(i + 1): s["label"] for i, s in enumerate(self.marked())}

    def marked(self):
        return [dict(name=k, **v) for k, v in self.slots.items() if v.get("mark", True)]

    def slot_by_mark(self, m):
        ms = self.marked()
        try:
            i = int(m) - 1
        except Exception:
            return None
        return ms[i]["name"] if 0 <= i < len(ms) else None


SCENARIOS = {}


def register(cls):
    SCENARIOS[cls.sid] = cls
    return cls


# ----------------------------------------------------------------------------- episode
class Episode:
    def __init__(self, sc, seed, clock, out, record=True, privileged=False, visible=False, hard=False, commit=False):
        assert clock in ("paused", "live")
        self.commit = bool(commit); self.commit_n = 0; self.commit_log = []
        self.sc, self.seed, self.clock, self.out = sc, int(seed), clock, out
        self.privileged_mode = privileged
        self.rng = random.Random(1000 + self.seed)
        self.world_t = 0.0; self.agent_t = 0.0; self.tick_i = 0
        self.started = False; self.over = False; self.over_reason = ""
        self.used = 0; self.result = "episode started"; self.actions = []
        self.stream = []; self.delivered = 0   # (world_t, jpeg bytes) of every head-camera frame
        self.t_resp = None; self.phase = "idle"; self.label = ""
        self.record = record
        out.mkdir(parents=True, exist_ok=True)
        (out / "head").mkdir(exist_ok=True)
        self.timeline = []; self.events = []
        self.vid = None; self.vid_head = None
        if record:
            self.vid = Video(out / "main.mp4", fps=MAIN_HZ_ACT)
            self.vid_head = Video(out / "head.mp4", fps=STREAM_HZ)
            self.flog = open(out / "frames.jsonl", "w")
        self._ros = getattr(og.sim, "_render_on_step", None)
        if self._ros is not None:
            og.sim._render_on_step = False
        sc.visible = bool(visible); sc.hard = bool(hard)
        sc.reset(self, self.rng)
        sc.pins.apply(all_=True)
        for _ in range(SETTLE_N + 2):   # robot bases teleport across the scene: let the cameras catch up before the first frame
            self.sync()
            og.sim.render()
        self.capture(force=True)
        self.ev("reset", seed=self.seed, clock=clock, visible=bool(visible))
        if getattr(sc, "prelude_s", 0) and hasattr(sc, "g_prelude"):
            # v3: the robot is busy with a chore while the world goes on; its camera records; the agent takes over after
            self.run_skill(sc.g_prelude(self), getattr(sc, "chore", "chore"))
            self.phase = "idle"; self.label = ""
            self.result = f"your robot finished {getattr(sc, 'chore', 'its chore')}; you are in control now"
            self.ev("prelude_done")

    # ---- logging
    def ev(self, kind, **kw):
        self.events.append(dict(kind=kind, world_t=round(self.world_t, 3), agent_t=round(self.agent_t, 3), **kw))

    # ---- simulation
    def capture(self, force=False, reuse=False):
        """Record the head-camera stream (STREAM_HZ) and the spectator video. With reuse=True nothing in the scene changed
        since the last render, so the previous images are recorded again (no simulator calls)."""
        k_stream = SIM_HZ // STREAM_HZ
        k_main = SIM_HZ // (MAIN_HZ_THINK if self.phase == "think" else MAIN_HZ_ACT)
        want_stream = force or self.tick_i % k_stream == 0
        want_main = self.record and (force or self.tick_i % k_main == 0)
        if not (want_stream or want_main):
            return
        if reuse and want_stream and getattr(self, "_stream_need", False):
            reuse = False      # main_1115 fix: the last render after a change went to the spectator only
        if not reuse:
            og.sim.render()
            if not want_stream:
                self._stream_need = True
        if want_stream:
            if reuse and self.stream:
                jb, px = self.stream[-1][1], self.stream[-1][2]
                self.stream.append((round(self.world_t, 2), jb, px))
                if self.record:
                    self.vid_head.add(self._last_head)
            else:
                img = rgb_of(self.sc.head)
                self._stream_need = False
                ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.stream.append((round(self.world_t, 2), buf.tobytes(), self.mark_px()))
                self._last_head = img
                if self.record:
                    self.vid_head.add(img)
        if want_main:
            if reuse and getattr(self, "_last_main", None) is not None:
                img, hp = self._last_main, self._last_hp
            else:
                img = rgb_of(self.sc.main_cam)
                try:
                    hp = cam_project(self.sc.main_cam, [get_pose(self.sc.head)[0] + np.array([0, 0, 0.32])], 1280, 720)[0]
                except Exception:
                    hp = None
                self._last_main, self._last_hp = img, hp
            self.vid.add(img)
            row = dict(i=self.vid.n - 1, world_t=round(self.world_t, 3), agent_t=round(self.agent_t, 3), phase=self.phase,
                       label=self.label, clock=self.clock, head_px=None if hp is None else [round(hp[0], 1), round(hp[1], 1)],
                       state=self.sc.privileged())
            self.flog.write(json.dumps(row) + "\n")

    def scene_sig(self):
        """Signature of everything that moves: robot bases and joints, held objects; plus any pin moved since the last sync."""
        sc = self.sc
        parts = []
        for b in (sc.A, getattr(sc, "B", None)):
            if b is not None:
                parts.append((round(b.x, 4), round(b.y, 4), round(b.yaw, 4), tuple(np.round(b.q.numpy(), 4).tolist()), tuple(sorted(b.held))))
        return hash(tuple(parts))

    def sync(self):
        """Push robot joints / bases, held objects and changed pins to the simulator, then step once."""
        sc = self.sc
        sc.A.apply()
        if getattr(sc, "B", None) is not None:
            sc.B.apply()
        if getattr(sc, "_parked", None) is not None:
            sc._parked.apply()
        held = set(sc.A.holding()) | (set(sc.B.holding()) if getattr(sc, "B", None) is not None else set())
        sc.pins.apply(skip=held, all_=True)
        if hasattr(sc, "hold"):
            sc.hold()
        og.sim.step()
        self._sig = self.scene_sig()

    def changed(self):
        return bool(self.sc.pins.dirty) or getattr(self, "_sig", None) != self.scene_sig()

    def capture_due(self):
        k_stream = SIM_HZ // STREAM_HZ
        k_main = SIM_HZ // (MAIN_HZ_THINK if self.phase == "think" else MAIN_HZ_ACT)
        return self.tick_i % k_stream == 0 or (self.record and self.tick_i % k_main == 0)

    def cap_step(self):
        """v2s fix: a pose written in one sync is rendered only after the next simulator step. After the last change of
        a burst, sync once more before reusing images; otherwise the stream (and current_view) would keep showing the
        state before the last change (e.g. a lamp that just switched never appears switched)."""
        ch = self.changed()
        if ch:
            self._settle = SETTLE_N
        if ch or getattr(self, "_settle", 0) > 0:
            self.sync()
            self.capture()
            if not ch:
                self._settle = int(self._settle) - 1
        else:
            self.capture(reuse=True)

    def settle_end(self):
        """End of an advance / skill: make the final state visible in the last recorded frame."""
        if self.changed() or getattr(self, "_settle", 0) or getattr(self, "_stream_need", False):
            for _ in range(SETTLE_N):
                self.sync(); og.sim.render()
            self.capture(force=True)
            self._settle = 0

    def commit_window(self, k):
        """Image window (y0, y1, x0, x1) around mark k (0-based) in the last frame the agent received, or None."""
        if not self.delivered:
            return None
        px = self.stream[self.delivered - 1][2]
        if px is None or not (0 <= k < len(px)) or px[k] is None:
            return None
        u, v = float(px[k][0]), float(px[k][1]); R = self.sc.head_res
        if not (0 <= u < R and 0 <= v < R):
            return None
        h = COMMIT_HALF
        return (max(0, int(v) - h), min(R, int(v) + h), max(0, int(u) - h), min(R, int(u) + h))

    def commit_check(self):
        """Ablation A: raise CommitAbort when the window around the target mark changed in any frame recorded since the
        agent's last observation (plus a fresh frame of the state now)."""
        tgt = getattr(self, "_commit_target", None)
        try:
            k = int(str(tgt).strip()) - 1
        except Exception:
            self.commit_log.append(dict(world_t=round(self.world_t, 2), target=tgt, result="no_mark")); return
        win = self.commit_window(k)
        if win is None:
            self.commit_log.append(dict(world_t=round(self.world_t, 2), target=tgt, result="not_in_view")); return
        for _ in range(SETTLE_N):
            self.sync(); og.sim.render()
        self.capture(force=True)
        y0, y1, x0, x1 = win

        def g(jb):
            im = decode_jpeg(jb)[y0:y1, x0:x1]
            return cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.int16)
        ref_jb = self.stream[self.delivered - 1][1]
        ref = g(ref_jb); worst = 0.0; t_w = None; last = ref_jb; n = 0
        for (t, jb, _) in self.stream[self.delivered:]:
            if jb is last:
                continue
            last = jb; n += 1
            frac = float((np.abs(g(jb) - ref) > COMMIT_PIX).mean())
            if frac > worst:
                worst, t_w = frac, t
        changed = worst > COMMIT_FRAC
        self.commit_log.append(dict(world_t=round(self.world_t, 2), target=tgt, frac=round(worst, 4), t_max=t_w, n_new=n,
                                    result="abort" if changed else "pass"))
        if changed:
            raise CommitAbort(tgt)

    def track_actors(self):
        """main_1115: log when the other actors become busy / idle (logging only)."""
        sc = self.sc
        try:
            b = bool(sc.actors_busy()) if hasattr(sc, "actors_busy") else bool(sc.workspace_busy())
        except Exception:
            return
        if b != getattr(self, "_actors_b", False):
            self._actors_b = b
            self.ev("actors_busy" if b else "actors_idle")

    def tick(self, robot_gen=None):
        sc = self.sc
        if robot_gen is not None:
            try:
                next(robot_gen)
            except StopIteration:
                robot_gen = None
        sc.world(self)
        self.track_actors()
        self.world_t += DT; self.agent_t += DT; self.tick_i += 1
        if self.capture_due():
            self.cap_step()
        return robot_gen

    def advance(self, seconds, phase, label=""):
        self.phase, self.label = phase, label
        n = int(round(min(seconds, MAX_CATCHUP) * SIM_HZ))
        for _ in range(n):
            self.tick()
            if self.world_t >= self.sc.deadline:
                break
        self.settle_end()

    def run_skill(self, gen, label):
        self.phase, self.label = "act", label
        while True:
            try:
                next(gen)
            except StopIteration:
                break
            self.sc.world(self)
            self.track_actors()
            self.world_t += DT; self.agent_t += DT; self.tick_i += 1
            if self.capture_due():
                self.cap_step()
        self.sync()
        self._settle = True
        self.settle_end()

    # ---- clock
    def charge_latency(self, override=None):
        """Called at the start of every request after the first observation. `override` fixes the charged
        latency (scripted agents, fixed-latency clock) instead of the measured wall time."""
        if not self.started or self.t_resp is None or self.over:
            return 0.0
        lat = max(0.0, time.time() - self.t_resp) if override is None else float(override)
        t0w, t0a = self.world_t, self.agent_t
        if self.clock == "live":
            self.advance(lat, "think", "agent deliberating")
        else:
            self.agent_t += lat
        self.timeline.append(dict(kind="think", latency=round(lat, 3), world_t0=round(t0w, 3), world_t1=round(self.world_t, 3),
                                  agent_t0=round(t0a, 3), agent_t1=round(self.agent_t, 3)))
        if self.world_t >= self.sc.deadline:
            self.finish("deadline")
        return lat

    def responded(self):
        self.t_resp = time.time()

    # ---- agent API
    def stream_payload(self):
        new = self.stream[self.delivered:]
        self.delivered = len(self.stream)
        if not new:
            return dict(n_frames=0)
        return dict(n_frames=len(new), t_from=new[0][0], t_to=new[-1][0], **self.sheet(new))

    def sheet(self, frames, max_tiles=None):
        """Contact sheet: evenly spaced frames, at most ~3 s apart (8 to 24 tiles)."""
        span = frames[-1][0] - frames[0][0] if frames else 0.0
        if max_tiles is None:
            max_tiles = int(min(24, max(8, math.ceil(span / getattr(self.sc, "SHEET_DT", 3.0)) + 1)))
        if len(frames) > max_tiles:
            idx = np.linspace(0, len(frames) - 1, max_tiles).round().astype(int)
            frames = [frames[i] for i in idx]
        cols = 4 if len(frames) <= 12 else 6
        tile = 320 if cols == 4 else 288
        rows = (len(frames) + cols - 1) // cols
        canvas = np.full((rows * (tile + 26), cols * tile, 3), 255, np.uint8)
        for i, (t, jb, px) in enumerate(frames):
            im = self.draw_marks(decode_jpeg(jb), px)
            im = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
            r_, c_ = divmod(i, cols)
            y0 = r_ * (tile + 26); x0 = c_ * tile
            canvas[y0 + 26:y0 + 26 + tile, x0:x0 + tile] = im
            cv2.putText(canvas, f"t = {t:6.1f} s", (x0 + 8, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        return dict(sheet=jpeg_b64(canvas, 80), sheet_times=[f[0] for f in frames])

    def mark_px(self):
        sc = self.sc
        pts = [m.get("mark_pos", m["pos"]) for m in sc.marked()]
        try:
            return cam_project(sc.head, pts, sc.head_res, sc.head_res)
        except Exception:
            return [None] * len(pts)

    def draw_marks(self, img, px=None):
        h, w = img.shape[:2]
        sc = self.sc
        if px is None:
            px = self.mark_px()
        s = w / sc.head_res
        for i, p in enumerate(px):
            if p is None:
                continue
            u, v = int(p[0] * s), int(p[1] * s)
            if not (0 <= u < w and 0 <= v < h):
                continue
            rad = max(9, int(14 * s))
            cv2.circle(img, (u, v), rad, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (u, v), rad, (30, 30, 30), 2, cv2.LINE_AA)
            txt = str(i + 1); fs = 0.55 * s * 1.3
            (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.putText(img, txt, (u - tw // 2, v + th_ // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 2, cv2.LINE_AA)
        return img

    def obs(self):
        if not self.started:
            self.started = True; self.ev("first_observation")
        cur, cur_px = (decode_jpeg(self.stream[-1][1]), self.stream[-1][2]) if self.stream else (rgb_of(self.sc.head), None)
        payload = dict(world_time=round(self.world_t, 2), clock=self.clock, holding=self.sc.holding_text(),
                       last_result=self.result, manipulation_actions_left=self.sc.budget - self.used,
                       episode_over=self.over, over_reason=self.over_reason,
                       current_view=jpeg_b64(self.draw_marks(cur.copy(), cur_px)), stream=self.stream_payload(),
                       marks=self.sc.legend())
        if self.privileged_mode:
            payload["privileged"] = self.sc.privileged()
        return payload

    def act(self, skill, target=None, seconds=None):
        if self.over:
            self.result = f"episode is over ({self.over_reason})"; return
        if skill not in self.sc.skills:
            self.result = f"unknown skill '{skill}'; allowed: {', '.join(self.sc.skills)}"; return
        rec = dict(skill=skill, target=target, world_t=round(self.world_t, 2), agent_t=round(self.agent_t, 2),
                   state_before=self.sc.privileged())
        if skill == "done":
            self.actions.append(rec); self.finish("agent_done"); self.result = "episode ended by the agent"; return
        if skill == "wait":
            s = float(seconds if seconds is not None else (target if target is not None else 5.0))
            s = max(0.5, min(s, 60.0))
            t0 = self.world_t
            self.advance(s, "wait", f"waiting {s:.0f} s")
            self.result = f"waited {self.world_t - t0:.1f} s"
            rec["seconds"] = s
        else:
            if self.used >= self.sc.budget:
                self.result = "no manipulation actions left"; return
            # v2s: no refusal. A command into a place another robot, machine or person is handling first waits silently
            # (world time runs) until they are done; the scenario's execute() wraps the skill in that wait (_busy).
            self.used += 1
            t0w = self.world_t
            self.sc._commit_ep = self if self.commit else None
            self._commit_target = target
            gen = self.sc.execute(self, skill, target)
            try:
                self.run_skill(gen, f"{skill} {target}")
                self.timeline.append(dict(kind="act", skill=skill, target=target, world_t0=round(t0w, 3), world_t1=round(self.world_t, 3)))
            except CommitAbort:
                self.used -= 1; self.commit_n += 1
                self.sync(); self._settle = True; self.settle_end()
                self.result = (f"not executed: the camera view around mark {target} changed since your last observation; "
                               f"no manipulation action was used")
                rec["aborted"] = True
                self.timeline.append(dict(kind="abort", skill=skill, target=target, world_t0=round(t0w, 3), world_t1=round(self.world_t, 3)))
            finally:
                self.sc._commit_ep = None
        rec["result"] = self.result; rec["state_after"] = self.sc.privileged()
        self.actions.append(rec); self.ev("act", skill=skill, target=target, result=self.result)
        if self.world_t >= self.sc.deadline:
            self.finish("deadline")
        elif self.used >= self.sc.budget and self.sc.budget_ends_episode:
            # let the world settle briefly so a consequence is recorded, then end
            self.advance(2.0, "act", "")
            self.finish("budget_used")

    def finish(self, reason):
        if self.over:
            return
        self.over = True; self.over_reason = reason
        final = self.sc.privileged()
        ok, why = self.sc.verify(final)
        self.final = dict(scenario=self.sc.sid, seed=self.seed, clock=self.clock, success=bool(ok), reason=why,
                          end=reason, world_t=round(self.world_t, 2), agent_t=round(self.agent_t, 2),
                          manipulation_actions=self.used, final_state=final, hidden=getattr(self.sc, "hidden", {}),
                          actions=self.actions, timeline=self.timeline, events=self.events,
                          commit=self.commit, commit_aborts=self.commit_n, commit_log=self.commit_log)
        self.ev("finish", reason=reason, success=bool(ok))

    def close(self):
        if not self.over:
            self.finish("closed")
        (self.out / "result.json").write_text(json.dumps(self.final, indent=1, default=float))
        if self.record:
            self.vid.close(); self.vid_head.close(); self.flog.close()
        if self._ros is not None:
            og.sim._render_on_step = self._ros
        return self.final


class Video:
    def __init__(self, path, fps=10, crf=20):
        import av
        self.c = av.open(str(path), mode="w"); self.s = None; self.fps = fps; self.crf = crf; self.n = 0

    def add(self, rgb):
        import av
        if self.s is None:
            h, w = rgb.shape[:2]
            self.s = self.c.add_stream("libx264", rate=self.fps); self.s.width, self.s.height = w, h
            self.s.pix_fmt = "yuv420p"; self.s.options = {"crf": str(self.crf), "preset": "veryfast"}
        for p in self.s.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self.c.mux(p)
        self.n += 1

    def close(self):
        if self.s is not None:
            for p in self.s.encode():
                self.c.mux(p)
        self.c.close()
