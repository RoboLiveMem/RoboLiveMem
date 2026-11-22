"""Shared manipulation primitives for the agent robot (R1) and the helper robot (R1 Pro).

All motions are kinematic joint interpolations between IK solutions; grasped objects are attached
to the gripper and, when released, pinned exactly at the target pose (no drops, no tipping).
"""
import math
import numpy as np

IK_CACHE = globals().get("IK_CACHE") or {}      # survives a reload job (re-exec of this file keeps the solved poses)


def fwd_left(yaw):
    return np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([-math.sin(yaw), math.cos(yaw), 0.0])


def rel_point(base, dx, dl, z):
    x, y, yaw = base
    f, l = fwd_left(yaw)
    return [x + dx * f[0] + dl * l[0], y + dx * f[1] + dl * l[1], z]


class Arm:
    """IK cache + motion generators for one robot at one or more base poses."""

    def __init__(self, bot, hand_z=0.95, tag=""):
        self.b = bot; self.hand_z = hand_z; self.tag = tag
        self.cache = IK_CACHE.setdefault(bot.r.name, {})

    def neutral(self, base, z=None):
        key = ("neutral", tuple(np.round(base, 3)), z)
        if key in self.cache:
            return self.cache[key]
        z = self.hand_z if z is None else z
        f, _ = fwd_left(base[2])
        b = self.b; b.place_base(*base)
        q = b.q.clone()
        q, e1 = b.ik("left", rel_point(base, 0.18, 0.27, z - 0.12), approach=[0.3 * f[0], 0.3 * f[1], -1.0], use_trunk=False, q_init=q)
        q, e2 = b.ik("right", rel_point(base, 0.18, -0.27, z - 0.12), approach=[0.3 * f[0], 0.3 * f[1], -1.0], use_trunk=False, q_init=q)
        for a in ("left", "right"):
            q[b.GRIP[a]] = 0.05
        self.cache[key] = q
        return q

    def reach(self, base, arm, pos, approach=None, dz=0.0, q_init=None):
        pos = np.asarray(pos, float) + np.array([0, 0, dz])
        key = ("reach", tuple(np.round(base, 3)), arm, tuple(np.round(pos, 3)), None if approach is None else tuple(np.round(approach, 2)))
        if key in self.cache:
            return self.cache[key]
        b = self.b; b.place_base(*base)
        f, _ = fwd_left(base[2])
        if approach is None:
            approach = [0.45 * f[0], 0.45 * f[1], -1.0]
        q0 = self.neutral(base) if q_init is None else q_init
        q, err = b.ik(arm, pos, approach=approach, use_trunk=True, q_init=q0)
        if err > 0.03:
            q2, err2 = b.ik(arm, pos, approach=None, use_trunk=True, q_init=q0, iters=300)
            if err2 < err:
                q, err = q2, err2
        self.cache[key] = (q, err)
        return q, err

    # ------------------------------------------------------------- generators
    def g_to(self, q, seconds, arm=None, closed=None):
        b = self.b
        q = q.clone()
        if arm is not None and closed is not None:
            q[b.GRIP[arm]] = 0.0 if closed else 0.05
        # keep the other arm's gripper as it is
        for a in ("left", "right"):
            if a != arm:
                q[b.GRIP[a]] = b.q[b.GRIP[a]]
        yield from g_joints(b, q, seconds)

    def g_pick(self, base, arm, pos, obj=None, on_grasp=None, lift=0.14, speed=1.0, keep_quat=None, pre=None, approach=None):
        """Reach the pre-grasp point (above pos, or pos + pre), move in, close, attach obj (if any), back out, neutral."""
        b = self.b
        pos = np.asarray(pos, float)
        if pre is None:
            qa, _ = self.reach(base, arm, pos, dz=lift, approach=approach)
            qb, _ = self.reach(base, arm, pos, dz=0.02, q_init=qa, approach=approach)
        else:
            qa, _ = self.reach(base, arm, pos + np.asarray(pre, float), approach=approach)
            qb, _ = self.reach(base, arm, pos, q_init=qa, approach=approach)
        yield from self.g_to(qa, 1.2 / speed, arm, closed=False)
        yield from self.g_to(qb, 0.8 / speed, arm, closed=False)
        q = b.q.clone(); q[b.GRIP[arm]] = 0.0
        yield from g_joints(b, q, 0.4)
        if obj is not None:
            b.attach(obj, arm, fixed_quat=keep_quat)
        if on_grasp:
            on_grasp()
        yield from self.g_to(qa, 0.8 / speed, arm, closed=True)
        yield from self.g_to(self.neutral(base), 1.0 / speed, arm, closed=True)

    def g_place(self, base, arm, pos, obj=None, quat=None, pins=None, on_release=None, lift=0.14, speed=1.0, obj_offset=0.0,
                pre=None, approach=None):
        """Carry to the pre-place point, move in, release obj pinned at pos (+obj_offset in z), retreat to neutral."""
        b = self.b
        pos = np.asarray(pos, float)
        if pre is None:
            qa, _ = self.reach(base, arm, pos, dz=lift + obj_offset, approach=approach)
            qb, _ = self.reach(base, arm, pos, dz=0.03 + obj_offset, q_init=qa, approach=approach)
        else:
            qa, _ = self.reach(base, arm, pos + np.asarray(pre, float), approach=approach)
            qb, _ = self.reach(base, arm, pos + np.array([0, 0, 0.01]), q_init=qa, approach=approach)
        yield from self.g_to(qa, 1.2 / speed, arm, closed=True)
        yield from self.g_to(qb, 0.8 / speed, arm, closed=True)
        if obj is not None:
            b.detach(obj)
            if pins is not None:
                pins.move(obj, np.asarray(pos, float) + np.array([0, 0, obj_offset]), quat)
        if on_release:
            on_release()
        q = b.q.clone(); q[b.GRIP[arm]] = 0.05
        yield from g_joints(b, q, 0.4)
        yield from self.g_to(qa, 0.7 / speed, arm, closed=False)
        yield from self.g_to(self.neutral(base), 1.0 / speed, arm, closed=False)

    def g_gesture(self, base, arm, pts, seconds_each=0.5, closed=True):
        """Move the hand through a list of points (e.g. shaking a salt shaker, wiping)."""
        q_prev = None
        for p in pts:
            q, _ = self.reach(base, arm, p, q_init=q_prev)
            q_prev = q
            yield from self.g_to(q, seconds_each, arm, closed=closed)


def upright_quat(yaw=0.0):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


class Helper:
    """Runs one scripted behaviour generator for the helper robot (or a machine) inside the world tick."""

    def __init__(self):
        self.gen = None; self.status = "idle"; self.started = False

    def start(self, gen, status="working"):
        self.gen = gen; self.status = status; self.started = True

    def step(self):
        if self.gen is None:
            return
        try:
            next(self.gen)
        except StopIteration:
            self.gen = None
