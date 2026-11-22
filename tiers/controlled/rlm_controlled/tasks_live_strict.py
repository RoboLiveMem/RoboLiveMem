"""Strict-definition variants of live-native tasks (added 2026-11-06; nothing in tasks_live.py changes).

InterruptSwapWorld: InterruptSwap with the swaps scheduled at WORLD times, independent of the policy's decisions
(InterruptSwap schedules a swap 0.6-2 s after each observation, i.e. relative to the decision start, which makes the event
target the agent). Here, after the cubes are covered (step 70), swap k becomes due at a world time drawn once per seed:
due_0 = 70 + U(6, 50) steps (0.3-2.5 s), due_{k+1} = due_k + 50 + U(6, 50). A due swap starts only while the robot stands
still (no command executing; rlm_busy False) and only when no other swap is running; it is postponed (not cancelled)
while the robot moves. None starts after a container has been lifted. With the paused clock thinking takes no world time,
so a due swap can only start during an explicit wait; that is intended (the person waits for the robot to stand still).
Same seeds/scenes as InterruptSwap (seed slot 6). No decision-time hook (rlm_on_observe does nothing).
"""
from __future__ import annotations

import json
import os

from rlm_controlled import tasks_live
from rlm_controlled.tasks_live import InterruptSwap, SwapStream, _gen, _ri, _step, _register
from robomme.robomme_env.utils import vqa_options


@_register("InterruptSwapWorld", "ButtonUnmaskSwap")
class InterruptSwapWorld(InterruptSwap):
    T_COVER = 70
    GAP = (6, 50)          # 0.3-2.5 s at 20 Hz

    def _refresh_swap_schedule(self):
        super()._refresh_swap_schedule()
        g = _gen(self.seed, 17)
        n = self.LEVELS[self.live_level]["n"]
        d, dues = self.T_COVER, []
        for _ in range(n):
            d = d + _ri(g, *self.GAP)
            dues.append(d)
            d = d + self.SWAP_LEN
        self.rlm_dues = dues
        self.rlm_fired = [None] * n        # step at which swap k started (None: not yet)

    def rlm_on_observe(self, t):
        return None                        # no decision-triggered events

    def _active(self, t):
        return any(s <= t <= e for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs) if p not in ("skip", "cancel"))

    def rlm_events(self):
        end = self.rlm_stop_step if self.rlm_stop_step is not None else 10 ** 9
        return [(s, e, "swap") for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs) if s < end and p not in ("skip", "cancel")]

    def rlm_safe_step(self, t, extra=0):
        """Latency-aware oracle: earliest step >= t at which a container command observed at that step is safe, assuming
        the robot stands still from t on (so a due swap starts as soon as it is due and no other swap runs)."""
        if self.rlm_stop_step is not None or "container" not in (getattr(self, "current_task_name", "") or ""):
            return t
        need = extra + self.NEED
        ev = [(s, e) for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs) if p not in ("skip", "cancel")]
        prev_end = max([e for _, e in ev], default=-1)
        for k, d in enumerate(self.rlm_dues):
            if self.rlm_fired[k] is None:
                s = max(d, t, prev_end + 1)
                ev.append((s, s + self.SWAP_LEN))
                prev_end = s + self.SWAP_LEN
        cand = t
        for s, e in sorted(ev):
            if e <= cand:
                continue
            if s >= cand + need:
                break
            cand = e + 1
        return cand

    def step(self, action):
        t = _step(self)
        if self.rlm_stop_step is None and not self.rlm_busy and not self._active(t):
            for k, d in enumerate(self.rlm_dues):
                if self.rlm_fired[k] is None and d <= t:
                    self.rlm_fired[k] = t
                    self.rlm_schedule.append((t, t + self.SWAP_LEN))
                    self.rlm_pairs.append(None)
                    break
        return SwapStream.step(self, action)     # skip InterruptSwap.step (its cancel-while-busy rule)


tasks_live.TASK_SEED_SLOT.setdefault("InterruptSwapWorld", tasks_live.TASK_SEED_SLOT["InterruptSwap"])
if "ButtonUnmaskSwap" in vqa_options.OPTION_BUILDERS:
    vqa_options.OPTION_BUILDERS["InterruptSwapWorld"] = vqa_options.OPTION_BUILDERS["ButtonUnmaskSwap"]


def ensure_metadata():
    """Metadata for InterruptSwapWorld = InterruptSwap's records (same seeds and scenes), written once."""
    d = os.path.join(os.path.dirname(os.path.abspath(tasks_live.__file__)), "metadata")
    dst = os.path.join(d, "record_dataset_InterruptSwapWorld_metadata.json")
    if os.path.exists(dst):
        return dst
    src = json.load(open(os.path.join(d, "record_dataset_InterruptSwap_metadata.json")))
    src["env_id"] = "InterruptSwapWorld"
    for r in src["records"]:
        r["task"] = "InterruptSwapWorld"
    tmp = dst + f".tmp{os.getpid()}"
    json.dump(src, open(tmp, "w"), indent=1)
    os.replace(tmp, dst)
    return dst


ensure_metadata()


# =====================================================================================================================
# Round 2 (2026-11-06): world-time versions of HighlightLate / RevealLate / SwapStream / RepickStream with seed types.
# Seed type by episode: KIND = ("event", "lookalike", "null")[(seed // 100) % 3]  (seed = base + slot*10000 + ep*100).
# Hand-over = the first step of agent control (t0 = 1 for these tasks; RepickStream: the end of the demo video).
# A due motion starts at the first step >= due at which the robot stands still (rlm_busy False, set by the harness while a
# command executes); with the paused clock only during explicit waits.
#   swap tasks:  event = n world-scheduled swaps; look-alike = n "feints" (two objects move part-way towards each other and
#                back: visible motion, no change); null = nothing.
#   cue tasks:   event = the only cue (highlight / reveal) comes after hand-over at the due time; look-alike / null = the cue
#                is shown at hand-over (visible in the first decision's observation, off when the first command starts);
#                look-alike = the same cue is repeated later at the due time (redundant), null = nothing later.
# rlm_cue_visible(t) is a privileged "the cue is visible in an observation at t" used by the frame-blind stale oracle.
# =====================================================================================================================
import math  # noqa: E402

import numpy as np  # noqa: E402

from rlm_controlled.tasks_live import HighlightLate, RevealLate, RepickStream  # noqa: E402
from robomme.robomme_env.utils.statechange import (highlight_obj, lift_and_drop_objects_back_to_original,  # noqa: E402
                                                    lift_and_drop_objectA_onto_objectB, swap_flat_two_lane)
from robomme.robomme_env.utils.subgoal_evaluate_func import is_bin_pickup  # noqa: E402
from mani_skill.envs.sapien_env import BaseEnv  # noqa: E402

KINDS = ("event", "lookalike", "null")


def _kind_of(seed):
    return KINDS[(int(seed) // 100) % 3]


def _xyzq(actor):
    import torch
    p, q = actor.pose.p, actor.pose.q
    if isinstance(p, torch.Tensor):
        p, q = p.detach().cpu().numpy(), q.detach().cpu().numpy()
    return np.asarray(p, dtype=np.float64).reshape(-1)[:3], np.asarray(q, dtype=np.float32).reshape(-1)[:4]


def feint_two(env, a, b, s, e, t, amp=0.35, lane=0.06):
    """a and b move part-way (amp of the distance) towards each other on two lanes and back to where they were."""
    import sapien
    if t < s or t > e:
        return
    cache = env.__dict__.setdefault("_rlm_feints", {})
    key = (id(a), id(b), int(s), int(e))
    if key not in cache:
        cache[key] = (_xyzq(a), _xyzq(b))
    (pa, qa), (pb, qb) = cache[key]
    u = (t - s) / max(1, e - s)
    f = amp * math.sin(math.pi * u)
    d = pb - pa
    n = np.array([-d[1], d[0], 0.0])
    n = n / (np.linalg.norm(n) + 1e-9)
    off = n * lane * math.sin(math.pi * u)
    if t >= e:
        a.set_pose(sapien.Pose(p=pa.tolist(), q=qa))
        b.set_pose(sapien.Pose(p=pb.tolist(), q=qb))
        cache.pop(key, None)
        return
    a.set_pose(sapien.Pose(p=(pa + d * f + off).tolist(), q=qa))
    b.set_pose(sapien.Pose(p=(pb - d * f - off).tolist(), q=qb))


class _WorldSwapMixin:
    """Shared by SwapStreamW / RepickStreamW: world-scheduled dues, firing while idle, feints for look-alike seeds."""
    GAP = (6, 50)

    def _w_init_dues(self, anchor, n):
        g = _gen(self.seed, 19)
        self.rlm_kind = _kind_of(self.seed)
        d, dues = anchor, []
        for _ in range(n if self.rlm_kind != "null" else 0):
            d = d + _ri(g, *self.GAP)
            dues.append(d)
            d = d + self.SWAP_LEN
        self.rlm_dues = dues
        self.rlm_fired = [None] * len(dues)
        self.rlm_feints = []          # [(s, e, (a, b))]

    def _w_active(self, t):
        sw = any(s <= t <= e for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs) if p not in ("skip", "cancel"))
        return sw or any(s <= t <= e for s, e, _ in self.rlm_feints)

    def _w_fire(self, t, can_fire, pick_pair):
        if not can_fire or self._w_active(t):
            return
        for k, d in enumerate(self.rlm_dues):
            if self.rlm_fired[k] is None and d <= t:
                self.rlm_fired[k] = t
                if self.rlm_kind == "event":
                    self.rlm_schedule.append((t, t + self.SWAP_LEN))
                    self.rlm_pairs.append(None)
                else:
                    pr = pick_pair()
                    if pr is not None:
                        self.rlm_feints.append((t, t + self.SWAP_LEN, pr))
                break

    def _w_events(self):
        ev = [(s, e, "swap") for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs) if p not in ("skip", "cancel")]
        return ev + [(s, e, "feint") for s, e, _ in self.rlm_feints]

    def _w_safe(self, t, extra):
        need = extra + self.NEED
        ev = [(s, e) for s, e, _ in self._w_events()]
        prev_end = max([e for _, e in ev], default=-1)
        for k, d in enumerate(self.rlm_dues):
            if self.rlm_fired[k] is None:
                s = max(d, t, prev_end + 1)
                ev.append((s, s + self.SWAP_LEN))
                prev_end = s + self.SWAP_LEN
        cand = t
        for s, e in sorted(ev):
            if e <= cand:
                continue
            if s >= cand + need:
                break
            cand = e + 1
        return cand


@_register("SwapStreamW", "ButtonUnmaskSwap")
class SwapStreamW(_WorldSwapMixin, SwapStream):
    """SwapStream, strict: swaps (event) / feints (look-alike) / nothing (null) at world times after the cubes are covered,
    starting only while the robot stands still; none after a container is lifted. n = 3 (moderate) / 4 (tight)."""
    T_COVER = 70

    def __init__(self, *args, **kwargs):
        self.rlm_busy = False
        super().__init__(*args, **kwargs)

    def _refresh_swap_schedule(self):
        self.rlm_schedule, self.rlm_pairs = [], []
        self._rlm_swap_rng = _gen(self.seed, 12)
        self.rlm_stop_step = None
        self.swap_schedule = []
        self._w_init_dues(self.T_COVER, {"moderate": 3, "tight": 4}[self.live_level])

    def rlm_events(self):
        return self._w_events()

    def rlm_safe_step(self, t, extra=0):
        if self.rlm_stop_step is not None or "container" not in (getattr(self, "current_task_name", "") or ""):
            return t
        return self._w_safe(t, extra)

    def _pick_pair(self):
        free = list(self.spawned_bins)
        pos_ = [self._get_actor_position(b) for b in free]
        pr = self._select_swap_pair_from_positions(pos_, self._rlm_swap_rng)
        return None if pr is None else (free[pr["idx1"]], free[pr["idx2"]])

    def step(self, action):
        t = _step(self)
        if self.rlm_stop_step is None and t > 70 and any(bool(is_bin_pickup(self, obj=b).any()) for b in self.spawned_bins):
            self.rlm_stop_step = t
        self._w_fire(t, self.rlm_stop_step is None and not self.rlm_busy, self._pick_pair)
        for b in getattr(self, "spawned_bins", []):
            lift_and_drop_objects_back_to_original(self, obj=b, start_step=0, end_step=64, cur_step=t)
        for k, (s, e) in enumerate(self.rlm_schedule):
            if not (s <= t <= e):
                continue
            if self.rlm_pairs[k] is None:
                free = [b for b in self.spawned_bins if b is not self.rlm_reserved]
                pos_ = [self._get_actor_position(b) for b in free]
                pr = self._select_swap_pair_from_positions(pos_, self._rlm_swap_rng)
                self.rlm_pairs[k] = "skip" if pr is None else (free[pr["idx1"]], free[pr["idx2"]])
            if self.rlm_pairs[k] in ("skip", "cancel"):
                continue
            a, b = self.rlm_pairs[k]
            _unpin_local(self, a, b, s, e, [self.rlm_reserved])
            swap_flat_two_lane(self, cube_a=a, cube_b=b, start_step=s, end_step=e, cur_step=t, lane_offset=0.07,
                               smooth=True, keep_upright=True,
                               other_cube=[x for x in self.spawned_bins if x is not a and x is not b
                                           and x is not self.rlm_reserved])
        for s, e, (a, b) in self.rlm_feints:
            if s <= t <= e:
                feint_two(self, a, b, s, e, t)
        ends = [e for (s, e) in self.rlm_schedule] + [e for s, e, _ in self.rlm_feints]
        hide_end = max([64] + ends)
        if self.rlm_stop_step is not None:
            hide_end = min(hide_end, self.rlm_stop_step)
        for cube_actor, bin_actor in getattr(self, "cube_bin_pairs", []):
            if cube_actor is None or bin_actor is None:
                continue
            lift_and_drop_objectA_onto_objectB(self, obj_a=cube_actor, obj_b=bin_actor, start_step=64,
                                               end_step=max(hide_end, t + 1) if self.rlm_stop_step is None else hide_end,
                                               cur_step=t)
        return BaseEnv.step(self, action)


def _unpin_local(env, a, b, s, e, free_objs):
    from rlm_controlled.tasks_live import _unpin
    _unpin(env, a, b, s, e, free_objs)


@_register("RepickStreamW", "VideoRepick")
class RepickStreamW(_WorldSwapMixin, RepickStream):
    """RepickStream, strict: after the demo video (hand-over), swaps (event) / feints (look-alike) / nothing (null) of the
    identical cubes at world times, starting only while the robot stands still and no cube is held. n = 3 / 4."""

    def __init__(self, *args, **kwargs):
        self.rlm_busy = False
        self.rlm_dues, self.rlm_fired, self.rlm_feints, self.rlm_kind = [], [], [], _kind_of(kwargs.get("seed", 0))
        super().__init__(*args, **kwargs)

    def _rlm_build(self, anchor):
        self.rlm_schedule, self.rlm_pairs = [], []
        self._rlm_swap_rng = _gen(self.seed, 52)
        self._w_init_dues(anchor, {"moderate": 3, "tight": 4}[self.live_level])

    def rlm_events(self):
        return self._w_events()

    def rlm_safe_step(self, t, extra=0):
        name = getattr(self, "current_task_name", "") or ""
        if "pick up" not in name or self.rlm_anchor is None:
            return t
        return self._w_safe(t, extra)

    def _pick_pair(self):
        free = [c for c in self.spawned_cubes]
        pos_ = [self._get_actor_position(c) for c in free]
        pr = self._select_swap_pair_from_positions(pos_, self._rlm_swap_rng)
        return None if pr is None else (free[pr["idx1"]], free[pr["idx2"]])

    def step(self, action):
        t = _step(self)
        if self.rlm_anchor is None and getattr(self, "current_task_demonstration", True) is False and t > 0:
            self.rlm_anchor = t
            self._rlm_build(t)
        if self.rlm_anchor is not None:
            self._w_fire(t, not self.rlm_busy and not self._held(), self._pick_pair)
        for k, (s, e) in enumerate(self.rlm_schedule):
            if not (s <= t <= e) or self.rlm_pairs[k] == "skip":
                continue
            if self.rlm_pairs[k] is None:
                free = [c for c in self.spawned_cubes if c is not self.rlm_reserved]
                pos_ = [self._get_actor_position(c) for c in free]
                pr = self._select_swap_pair_from_positions(pos_, self._rlm_swap_rng)
                if pr is None:
                    self.rlm_pairs[k] = "skip"
                    continue
                self.rlm_pairs[k] = (free[pr["idx1"]], free[pr["idx2"]])
            a, b = self.rlm_pairs[k]
            held = self._held()
            _unpin_local(self, a, b, s, e, [self.rlm_reserved] + held)
            swap_flat_two_lane(self, cube_a=a, cube_b=b, start_step=s, end_step=e, cur_step=t, lane_offset=0.07,
                               smooth=True, keep_upright=True,
                               other_cube=[x for x in self.spawned_cubes if x is not a and x is not b
                                           and x is not self.rlm_reserved and not any(x is h for h in held)])
        for s, e, (a, b) in self.rlm_feints:
            if s <= t <= e:
                feint_two(self, a, b, s, e, t)
        return VideoRepick_step(self, action)


def VideoRepick_step(env, action):
    from robomme.robomme_env.VideoRepick import VideoRepick
    return VideoRepick.step(env, action)


class _WorldCueMixin:
    """Shared by HighlightLateW / RevealLateW. Intervals of the cue: self.rlm_cues = [[s, e or None, tag]]."""
    GAP = (6, 50)

    def _c_defaults(self, seed):
        """Placeholders before the parent constructor (which may step the env once); _c_init sets the real values."""
        self.rlm_busy = False
        self.rlm_kind = _kind_of(seed)
        self.rlm_cues, self.rlm_due, self.rlm_due_fired, self.rlm_seen_busy, self.rlm_dur = [], 10 ** 9, None, False, 1

    def _c_init(self, dur):
        g = _gen(self.seed, 23)
        self.rlm_kind = _kind_of(self.seed)
        self.rlm_dur = dur
        self.rlm_due = 1 + _ri(g, *self.GAP)
        self.rlm_due_fired = None
        self.rlm_cues = []
        self.rlm_seen_busy = False
        if self.rlm_kind in ("lookalike", "null"):
            self.rlm_cues.append([0, None, "handover"])     # on until the first command starts

    def _c_tick(self, t):
        if self.rlm_busy:
            self.rlm_seen_busy = True
        for c in self.rlm_cues:
            if c[1] is None and c[2] == "handover" and self.rlm_seen_busy:
                c[1] = t
        want = self.rlm_kind in ("event", "lookalike")
        if want and self.rlm_due_fired is None and t >= self.rlm_due and not self.rlm_busy \
                and all(c[1] is not None and c[1] < t for c in self.rlm_cues):
            self.rlm_due_fired = t
            self.rlm_cues.append([t, t + self.rlm_dur, "due"])

    def _c_intervals(self, t):
        return [(s, (e if e is not None else t + 1), tag) for s, e, tag in self.rlm_cues]

    def rlm_cue_visible(self, t):
        return any(s <= t <= e for s, e, _ in self._c_intervals(t))

    def rlm_cue_pending(self, t):
        """The answer's cue can still come (privileged): an event cue not yet over."""
        if self.rlm_kind != "event":
            return False
        return self.rlm_due_fired is None or t <= self.rlm_due_fired + self.rlm_dur

    def _c_info_end(self, t):
        """End of the first cue (the information is complete after it); predicted if not yet fired."""
        if self.rlm_cues:
            s, e, _ = self._c_intervals(t)[0]
            return e if self.rlm_cues[0][1] is not None else t + 1
        return max(self.rlm_due, t) + self.rlm_dur

    def rlm_events(self):
        return [(s, e, "cue") for s, e, _ in self.rlm_cues if e is not None]


@_register("HighlightLateW", "PickHighlight")
class HighlightLateW(_WorldCueMixin, HighlightLate):
    """HighlightLate, strict (see the round-2 header)."""

    def __init__(self, *args, **kwargs):
        self._c_defaults(kwargs.get("seed", 0))
        super().__init__(*args, **kwargs)
        self._c_init(self.rlm_hl_end - self.rlm_hl_start)

    def rlm_events(self):
        return [(s, e, "highlight") for s, e, _ in self.rlm_cues if e is not None]

    def rlm_needs_cue(self):
        return "highlighted" in (getattr(self, "current_task_name", "") or "")

    def rlm_safe_step(self, t, extra=0):
        name = getattr(self, "current_task_name", "") or ""
        if "highlighted" in name:
            return max(t, self._c_info_end(t) + 1)
        return t

    def step(self, action):
        t = _step(self)
        self._c_tick(t)
        target_cubes = getattr(self, "target_cubes", [])
        n = min(self.configs[self.difficulty]["pickup"], len(target_cubes))
        for s, e, _ in self._c_intervals(t):
            for i in range(n):
                highlight_obj(self, target_cubes[i], start_step=s, end_step=e, cur_step=t)
        return BaseEnv.step(self, action)


@_register("RevealLateW", "ButtonUnmask")
class RevealLateW(_WorldCueMixin, RevealLate):
    """RevealLate, strict (see the round-2 header). The hand-over cue is ButtonUnmask's own opening reveal (containers up,
    dropped back over the cubes by step 64); the due cue is a reveal of `dur` steps."""

    def __init__(self, *args, **kwargs):
        self._c_defaults(kwargs.get("seed", 0))
        self._rlm_skip = set()
        super().__init__(*args, **kwargs)
        self._c_init(self.rlm_rv_end - self.rlm_rv_start)
        if self.rlm_kind in ("lookalike", "null"):
            self.rlm_cues = [[0, 32, "handover"]]          # cubes visible until the containers are half-way down

    def rlm_events(self):
        return [(s, e, "reveal") for s, e, _ in self.rlm_cues if e is not None]

    def rlm_needs_cue(self):
        return "container" in (getattr(self, "current_task_name", "") or "")

    def rlm_safe_step(self, t, extra=0):
        name = getattr(self, "current_task_name", "") or ""
        if "container" in name:
            return max(t, self._c_info_end(t) + 1)
        return t

    def _reveal_windows(self, t):
        out = []
        for s, e, tag in self._c_intervals(t):
            if tag == "handover":
                out.append((0, 64))
            else:
                out.append((s, e))
        return out

    def evaluate(self, solve_complete_eval=False):
        # the opening reveal (hand-over cue) is ButtonUnmask's own and is evaluated as there; a later reveal lifts every
        # container, which must not count as the robot picking one up (as in RevealLate)
        t = _step(self)
        if any(s <= t <= e + 5 for s, e, tag in self._c_intervals(t) if tag != "handover"):
            return {"success": torch.tensor([False]), "fail": torch.tensor([False])}
        return ButtonUnmask_evaluate(self, solve_complete_eval)

    def rlm_block_until(self, t):
        for s, e in self._reveal_windows(t):
            drop = s + max(1, (e - s) // 2)
            if s <= t <= drop:
                return drop + 1
        return t

    def step(self, action):
        t = _step(self)
        self._c_tick(t)
        for s, e in self._reveal_windows(t):
            for i in range(15):
                if hasattr(self, f"bin_{i}"):
                    b = getattr(self, f"bin_{i}")
                    if t == s and b is self.rlm_reserved:
                        self._rlm_skip.add((id(b), s))
                    if (id(b), s) in self._rlm_skip:
                        continue
                    lift_and_drop_objects_back_to_original(self, obj=b, start_step=s, end_step=e, cur_step=t)
        return BaseEnv.step(self, action)


def ButtonUnmask_evaluate(env, solve_complete_eval):
    from robomme.robomme_env.ButtonUnmask import ButtonUnmask
    return ButtonUnmask.evaluate(env, solve_complete_eval=solve_complete_eval)


import torch  # noqa: E402

for _n, _src in (("SwapStreamW", "SwapStream"), ("RepickStreamW", "RepickStream"), ("HighlightLateW", "HighlightLate"),
                 ("RevealLateW", "RevealLate")):
    tasks_live.TASK_SEED_SLOT.setdefault(_n, tasks_live.TASK_SEED_SLOT[_src])
    _b = tasks_live.LIVE_TASKS[_n]
    if _b in vqa_options.OPTION_BUILDERS:
        vqa_options.OPTION_BUILDERS[_n] = vqa_options.OPTION_BUILDERS[_b]


def ensure_metadata_w():
    """Metadata of each W task = its source task's records (same seeds and scenes), written once."""
    d = os.path.join(os.path.dirname(os.path.abspath(tasks_live.__file__)), "metadata")
    for n, src in (("SwapStreamW", "SwapStream"), ("RepickStreamW", "RepickStream"), ("HighlightLateW", "HighlightLate"),
                   ("RevealLateW", "RevealLate")):
        dst = os.path.join(d, f"record_dataset_{n}_metadata.json")
        if os.path.exists(dst):
            continue
        m = json.load(open(os.path.join(d, f"record_dataset_{src}_metadata.json")))
        m["env_id"] = n
        for r in m["records"]:
            r["task"] = n
        tmp = dst + f".tmp{os.getpid()}"
        json.dump(m, open(tmp, "w"), indent=1)
        os.replace(tmp, dst)


ensure_metadata_w()


W_NOTES = {
    "HighlightLateW": "some cubes are highlighted briefly; this can happen at the start or at any later time, also while you "
                      "are deciding, and it may be shown more than once; watch carefully. ",
    "RevealLateW": "the containers are lifted briefly, showing the cubes; this can happen at the start or at any later time, "
                   "also while you are deciding, and more than once; watch carefully. ",
    "SwapStreamW": "after the cubes are covered, two containers may be moved at any time, also while you are deciding: "
                   "sometimes they swap places, sometimes they only move a little and come back; watch carefully. ",
    "RepickStreamW": "after the video, two cubes may be moved at any time, also while you are deciding: sometimes they swap "
                     "places, sometimes they only move a little and come back; watch carefully. ",
}


def _install_agent_notes():
    """Task notes for the agent harness (agent.py's LATE_EVENT dicts), only if the agent module is already loaded."""
    import sys
    ag = sys.modules.get("rlm_controlled.agent")
    if ag is None:
        return
    for d in ("LATE_EVENT", "LATE_EVENT_WINDOW"):
        if hasattr(ag, d):
            getattr(ag, d).update(W_NOTES)


_install_agent_notes()


# =====================================================================================================================
# Round 3 (2026-11-06): visible twins of the strict swap tasks (L7 control). Same seeds, schedule and motions; the decisive
# identity stays observable at decision time through a small visual-only tag (no collision) that follows the object:
#   SwapStreamWV  - each container that hides a cube carries a tag of that cube's colour on top
#   RepickStreamWV - the cube that was picked in the video carries a yellow tag on top
# =====================================================================================================================
_RGBA = {"red": (1.0, 0.0, 0.0, 1.0), "green": (0.0, 0.9, 0.0, 1.0), "blue": (0.0, 0.2, 1.0, 1.0), "yellow": (1.0, 0.85, 0.0, 1.0)}
TAG_DZ = {"bin": 0.055, "cube": None}


def _visual_tag(env, name, rgba, half=(0.016, 0.016, 0.003)):
    import sapien
    b = env.scene.create_actor_builder()
    b.add_box_visual(half_size=list(half), material=sapien.render.RenderMaterial(base_color=list(rgba)))
    b.initial_pose = sapien.Pose(p=[5.0, 5.0, 5.0])
    return b.build_kinematic(name=name)


def _follow(env, tag, actor, dz):
    import sapien
    p, _ = _xyzq(actor)
    tag.set_pose(sapien.Pose(p=[float(p[0]), float(p[1]), float(p[2]) + dz], q=[1, 0, 0, 0]))


@_register("SwapStreamWV", "ButtonUnmaskSwap")
class SwapStreamWV(SwapStreamW):
    def _load_scene(self, options):
        super()._load_scene(options)
        self.rlm_tags = []
        for i, (cube, b) in enumerate(getattr(self, "cube_bin_pairs", [])):
            if cube is None or b is None:
                continue
            col = self.color_names[i] if i < len(self.color_names) else "yellow"
            self.rlm_tags.append((_visual_tag(self, f"rlm_tag_{i}", _RGBA.get(col, _RGBA["yellow"])), b))

    def step(self, action):
        out = super().step(action)
        for tag, b in getattr(self, "rlm_tags", []):
            _follow(self, tag, b, TAG_DZ["bin"])
        return out


@_register("RepickStreamWV", "VideoRepick")
class RepickStreamWV(RepickStreamW):
    def _load_scene(self, options):
        super()._load_scene(options)
        self.rlm_tag = _visual_tag(self, "rlm_tag_target", _RGBA["yellow"])

    def step(self, action):
        out = super().step(action)
        tgt = getattr(self, "target_cube_1", None)
        if tgt is not None and getattr(self, "rlm_tag", None) is not None:
            _follow(self, self.rlm_tag, tgt, float(self.cube_half_size) + 0.004)
        return out


for _n, _src in (("SwapStreamWV", "SwapStreamW"), ("RepickStreamWV", "RepickStreamW")):
    tasks_live.TASK_SEED_SLOT.setdefault(_n, tasks_live.TASK_SEED_SLOT[_src.rstrip("W")])
    _b = tasks_live.LIVE_TASKS[_n]
    if _b in vqa_options.OPTION_BUILDERS:
        vqa_options.OPTION_BUILDERS[_n] = vqa_options.OPTION_BUILDERS[_b]


def ensure_metadata_wv():
    d = os.path.join(os.path.dirname(os.path.abspath(tasks_live.__file__)), "metadata")
    for n, src in (("SwapStreamWV", "SwapStream"), ("RepickStreamWV", "RepickStream")):
        dst = os.path.join(d, f"record_dataset_{n}_metadata.json")
        if os.path.exists(dst):
            continue
        m = json.load(open(os.path.join(d, f"record_dataset_{src}_metadata.json")))
        m["env_id"] = n
        for r in m["records"]:
            r["task"] = n
        tmp = dst + f".tmp{os.getpid()}"
        json.dump(m, open(tmp, "w"), indent=1)
        os.replace(tmp, dst)


ensure_metadata_wv()

W_NOTES.update({
    "SwapStreamWV": "each container that hides a cube carries a small tag of that cube's colour on top, which moves with it, "
                    "so you can always see which container hides which cube. After the cubes are covered, two containers may "
                    "be moved at any time, also while you are deciding: sometimes they swap places, sometimes they only move a "
                    "little and come back. ",
    "RepickStreamWV": "the cube that was picked in the video carries a small yellow tag on top, which stays on it, so you can "
                      "always see which cube it is. After the video, two cubes may be moved at any time, also while you are "
                      "deciding: sometimes they swap places, sometimes they only move a little and come back. ",
})
_install_agent_notes()


# =====================================================================================================================
# Round 4 (2026-11-15, main tables): InterruptSwapW (InterruptSwapWorld with event / look-alike / null seeds) and cheap
# visible twins of the remaining strict tasks. Nothing above changes.
#   InterruptSwapW  - InterruptSwap's scenes and seeds (slot 6) with SwapStreamW's mechanics: n = 3 / 4 world-scheduled
#                     swaps (event), feints (look-alike) or nothing (null) after the cubes are covered, starting only while
#                     the robot stands still; dues drawn with InterruptSwapWorld's salt (17).
#   InterruptSwapWV - + a tag of the hidden cube's colour on each container that hides a cube (as SwapStreamWV)
#   HighlightLateWV - the cue stays visible: from the first moment cubes are highlighted, each highlighted cube keeps a
#                     small yellow tag on top
#   RevealLateWV    - the cue stays visible: from the first moment the cubes are shown, each container that hides a cube
#                     keeps a small tag of that cube's colour on top
# Agent notes for the twins (agent.TWIN_NOTES, used with the "twinnote" option) state only what a tag means.
# =====================================================================================================================
@_register("InterruptSwapW", "ButtonUnmaskSwap")
class InterruptSwapW(SwapStreamW):
    """InterruptSwapWorld with seed types (see the round-4 header)."""

    def _w_init_dues(self, anchor, n):
        super()._w_init_dues(anchor, n)
        g = _gen(self.seed, 17)
        d, dues = anchor, []
        for _ in range(n if self.rlm_kind != "null" else 0):
            d = d + _ri(g, *self.GAP)
            dues.append(d)
            d = d + self.SWAP_LEN
        self.rlm_dues = dues
        self.rlm_fired = [None] * len(dues)


class _BinTagMixin:
    """Tags of the hidden cubes' colours on the containers that hide them, following the containers."""

    def _load_scene(self, options):
        super()._load_scene(options)
        self.rlm_tags = []
        for i, (cube, b) in enumerate(getattr(self, "cube_bin_pairs", [])):
            if cube is None or b is None:
                continue
            col = self.color_names[i] if i < len(self.color_names) else "yellow"
            self.rlm_tags.append((_visual_tag(self, f"rlm_tag_{i}", _RGBA.get(col, _RGBA["yellow"])), b))

    def step(self, action):
        out = super().step(action)
        for tag, b in getattr(self, "rlm_tags", []):
            _follow(self, tag, b, TAG_DZ["bin"])
        return out


@_register("InterruptSwapWV", "ButtonUnmaskSwap")
class InterruptSwapWV(_BinTagMixin, InterruptSwapW):
    pass


def _cue_started(env, t):
    return any(s <= t for s, e, _ in env._c_intervals(t))


@_register("HighlightLateWV", "PickHighlight")
class HighlightLateWV(HighlightLateW):
    def _load_scene(self, options):
        super()._load_scene(options)
        tc = getattr(self, "target_cubes", [])
        n = min(self.configs[self.difficulty]["pickup"], len(tc))
        self.rlm_htags = [_visual_tag(self, f"rlm_htag_{i}", _RGBA["yellow"]) for i in range(n)]

    def step(self, action):
        out = super().step(action)
        t = _step(self)
        tc = getattr(self, "target_cubes", [])
        on = bool(self.rlm_cues) and _cue_started(self, t)
        for i, tag in enumerate(self.rlm_htags):
            if on:
                _follow(self, tag, tc[i], float(self.cube_half_size) + 0.004)
        return out


@_register("RevealLateWV", "ButtonUnmask")
class RevealLateWV(RevealLateW):
    def _load_scene(self, options):
        super()._load_scene(options)
        self.rlm_rtags = []
        for i, b in enumerate(list(getattr(self, "spawned_bins", []))[:3]):
            col = self.color_names[i] if i < len(getattr(self, "color_names", [])) else "yellow"
            self.rlm_rtags.append((_visual_tag(self, f"rlm_rtag_{i}", _RGBA.get(col, _RGBA["yellow"])), b))

    def step(self, action):
        out = super().step(action)
        t = _step(self)
        on = bool(self.rlm_cues) and _cue_started(self, t)
        for tag, b in getattr(self, "rlm_rtags", []):
            if on:
                _follow(self, tag, b, TAG_DZ["bin"])
        return out


for _n, _slot_src in (("InterruptSwapW", "InterruptSwap"), ("InterruptSwapWV", "InterruptSwap"),
                      ("HighlightLateWV", "HighlightLate"), ("RevealLateWV", "RevealLate")):
    tasks_live.TASK_SEED_SLOT.setdefault(_n, tasks_live.TASK_SEED_SLOT[_slot_src])
    _b = tasks_live.LIVE_TASKS[_n]
    if _b in vqa_options.OPTION_BUILDERS:
        vqa_options.OPTION_BUILDERS[_n] = vqa_options.OPTION_BUILDERS[_b]


def ensure_metadata_r4():
    d = os.path.join(os.path.dirname(os.path.abspath(tasks_live.__file__)), "metadata")
    for n, src in (("InterruptSwapW", "InterruptSwap"), ("InterruptSwapWV", "InterruptSwap"),
                   ("HighlightLateWV", "HighlightLateW"), ("RevealLateWV", "RevealLateW")):
        dst = os.path.join(d, f"record_dataset_{n}_metadata.json")
        if os.path.exists(dst):
            continue
        m = json.load(open(os.path.join(d, f"record_dataset_{src}_metadata.json")))
        m["env_id"] = n
        for r in m["records"]:
            r["task"] = n
        tmp = dst + f".tmp{os.getpid()}"
        json.dump(m, open(tmp, "w"), indent=1)
        os.replace(tmp, dst)


ensure_metadata_r4()

W_NOTES.update({"InterruptSwapW": W_NOTES["SwapStreamW"], "InterruptSwapWV": W_NOTES["SwapStreamWV"]})
TWIN_NOTES_R4 = {
    "SwapStreamWV": "Each container that hides a cube carries a small tag of that cube's colour on top, which moves with "
                    "the container. ",
    "InterruptSwapWV": "Each container that hides a cube carries a small tag of that cube's colour on top, which moves "
                       "with the container. ",
    "RepickStreamWV": "The cube that was picked up in the video carries a small yellow tag on top, which stays on it. ",
    "HighlightLateWV": "Once cubes have been highlighted, each highlighted cube keeps a small yellow tag on top. ",
    "RevealLateWV": "Once the cubes under the containers have been shown, each container that hides a cube keeps a small "
                    "tag of that cube's colour on top. ",
}


def _install_twin_notes():
    import sys
    ag = sys.modules.get("rlm_controlled.agent")
    if ag is not None and hasattr(ag, "TWIN_NOTES"):
        ag.TWIN_NOTES.update(TWIN_NOTES_R4)


_install_agent_notes()
_install_twin_notes()
