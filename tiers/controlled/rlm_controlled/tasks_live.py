"""Live-native RoboLiveMem tasks: RoboMME tasks re-timed so that memory-relevant events happen while the agent is
operating (and therefore, under the live clock, while it is thinking).

Every task exposes
  rlm_events()            -> list of (start_step, end_step, kind) of world events (for staleness analysis)
  rlm_safe_step(t, need)  -> earliest env step >= t at which an action needing `need` quiet steps can start
                             (used by the latency-aware privileged oracle; it encodes "do not act before the needed
                             information is revealed" and "do not act on objects that are being moved")
Live knob: `live_level` in {"moderate", "tight"} (constructor kwarg, default chosen from the seed).

Import this module before creating environments; it registers the env ids and the option/goal tables.
"""
from __future__ import annotations

import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

from robomme.robomme_env.ButtonUnmask import ButtonUnmask
from robomme.robomme_env.ButtonUnmaskSwap import ButtonUnmaskSwap
from robomme.robomme_env.PickHighlight import PickHighlight
from robomme.robomme_env.StopCube import StopCube
from robomme.robomme_env.VideoRepick import VideoRepick
from robomme.robomme_env.PatternLock import PatternLock
from robomme.robomme_env.utils.statechange import (swap_flat_two_lane, highlight_obj,
                                                    lift_and_drop_objects_back_to_original,
                                                    lift_and_drop_objectA_onto_objectB)
from robomme.robomme_env.utils.subgoal_evaluate_func import is_bin_pickup
from robomme.robomme_env.utils import vqa_options, task_goal

LIVE_TASKS = {}
# fixed seed slots so adding tasks never changes existing instances
TASK_SEED_SLOT = {"SwapStream": 0, "HighlightLate": 1, "StopCubeFast": 2, "RevealLate": 3, "RepickStream": 4,
                  "PatternLockLate": 5, "InterruptSwap": 6, "CueRecall": 7}
GOAL_OVERRIDE = {
    "CueRecall": "first press the button; later a cube will appear beside the table: as soon as it appears, pick up the "
                 "container that hides the cube of the same color (quickly, you have only a few seconds)",
    "PatternLockLate": "the buttons on the table will light up one by one; watch carefully, then use the stick attached "
                       "to the robot to retrace the same pattern, starting from the button the stick is on",
}


def _register(name, base_id):
    def deco(cls):
        register_env(name)(cls)
        LIVE_TASKS[name] = base_id
        return cls
    return deco


def _gen(seed, salt):
    g = torch.Generator()
    g.manual_seed(int(seed) * 7919 + salt)
    return g


def _ri(g, lo, hi):  # inclusive
    return int(torch.randint(lo, hi + 1, (1,), generator=g).item())


def _step(env):
    return int(env.elapsed_steps.reshape(-1)[0].item()) if isinstance(env.elapsed_steps, torch.Tensor) else int(env.elapsed_steps)


def _unpin(env, a, b, s, e, free_objs):
    """Drop objects from the poses an active swap is pinning (swap_flat_two_lane caches them at swap start)."""
    cache = getattr(env, "_two_lane_swaps", {}).get((id(a), id(b), int(s), int(e)))
    if cache and free_objs:
        cache["other_bins_poses"] = [p for p in cache["other_bins_poses"]
                                     if not any(p["bin"] is o for o in free_objs if o is not None)]


# ----------------------------------------------------------------------------------------------------------- swaps
@_register("SwapStream", "ButtonUnmaskSwap")
class SwapStream(ButtonUnmaskSwap):
    """Containers hide cubes (as ButtonUnmaskSwap) and then keep being swapped in bursts for ~40 s.
    Swap k takes 50 steps; gaps between swaps are drawn per level. Swapping stops once a container is lifted.
    Paused agents can pick in a gap; agents slower than the gap must track every swap until the stream ends."""
    LEVELS = {"moderate": dict(gap=(150, 220)), "tight": dict(gap=(100, 130))}
    STREAM_END = 800       # no swap starts after this step (40 s)
    SWAP_LEN = 50

    def __init__(self, *args, live_level=None, **kwargs):
        seed = kwargs.get("seed", 0)
        self.live_level = live_level or ("moderate" if (int(seed) // 100) % 2 == 0 else "tight")
        self.rlm_reserved = None   # set by the live clock to the object the current command acts on
        super().__init__(*args, **kwargs)

    def _refresh_swap_schedule(self):
        g = _gen(self.seed, 11)
        lo, hi = self.LEVELS[self.live_level]["gap"]
        sched, t = [], 64
        while t < self.STREAM_END:
            sched.append((t, t + self.SWAP_LEN))
            t = t + self.SWAP_LEN + _ri(g, lo, hi)
        self.rlm_schedule = sched
        self.rlm_pairs = [None] * len(sched)
        self._rlm_swap_rng = _gen(self.seed, 12)
        self.rlm_stop_step = None
        # keep the parent's attribute meaningful for code that reads it
        self.swap_schedule = [(None, None, s, e) for s, e in sched]

    def rlm_events(self):
        end = self.rlm_stop_step if self.rlm_stop_step is not None else 10 ** 9
        return [(s, e, "swap") for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs) if s < end and p != "skip"]

    NEED = 10   # the object a command acts on is never swapped once the command starts (rlm_reserved)

    def rlm_safe_step(self, t, extra=0):
        # observe at t, act on a container only if no swap is active within [t, t + extra + NEED)
        need = extra + self.NEED
        if self.rlm_stop_step is not None or "container" not in (getattr(self, "current_task_name", "") or ""):
            return t
        cand = t
        for s, e in self.rlm_schedule:
            if e <= cand:
                continue
            if s >= cand + need:
                break
            cand = e + 1
        return cand

    def step(self, action):
        t = _step(self)
        for b in getattr(self, "spawned_bins", []):
            lift_and_drop_objects_back_to_original(self, obj=b, start_step=0, end_step=64, cur_step=t)
        if self.rlm_stop_step is None and t > 70 and any(bool(is_bin_pickup(self, obj=b).any()) for b in self.spawned_bins):
            self.rlm_stop_step = t
        last_end = self.rlm_schedule[-1][1] if self.rlm_schedule else 64
        for k, (s, e) in enumerate(self.rlm_schedule):
            if not (s <= t <= e):
                continue
            if self.rlm_stop_step is not None and s >= self.rlm_stop_step:
                continue
            if self.rlm_pairs[k] is None:
                free = [b for b in self.spawned_bins if b is not self.rlm_reserved]
                pos = [self._get_actor_position(b) for b in free]
                pr = self._select_swap_pair_from_positions(pos, self._rlm_swap_rng)
                if pr is None:
                    self.rlm_pairs[k] = "skip"
                    continue
                self.rlm_pairs[k] = (free[pr["idx1"]], free[pr["idx2"]])
            if self.rlm_pairs[k] in ("skip", "cancel"):
                continue
            a, b = self.rlm_pairs[k]
            # other containers are pinned during a swap (no collisions), except the one a command is acting on
            _unpin(self, a, b, s, e, [self.rlm_reserved])
            swap_flat_two_lane(self, cube_a=a, cube_b=b, start_step=s, end_step=e, cur_step=t, lane_offset=0.07,
                               smooth=True, keep_upright=True,
                               other_cube=[x for x in self.spawned_bins if x is not a and x is not b
                                           and x is not self.rlm_reserved])
        hide_end = min(last_end, self.rlm_stop_step) if self.rlm_stop_step is not None else last_end
        for cube_actor, bin_actor in getattr(self, "cube_bin_pairs", []):
            if cube_actor is None or bin_actor is None:
                continue
            lift_and_drop_objectA_onto_objectB(self, obj_a=cube_actor, obj_b=bin_actor, start_step=64,
                                               end_step=hide_end, cur_step=t)
        return BaseEnv.step(self, action)


# ------------------------------------------------------------------------------------------------- interruptions
@_register("InterruptSwap", "ButtonUnmaskSwap")
class InterruptSwap(SwapStream):
    """Containers hide cubes, as in ButtonUnmaskSwap; afterwards two containers swap only while the robot is idle.
    Each time the policy observes the scene (a decision starts; the harness calls rlm_on_observe), a swap of two
    containers is scheduled 0.6-2 s later (none before the cubes are covered, none after a container is lifted, one at a
    time, a few per episode). A person moves things only while the robot stands still: if the robot is executing a
    command when the swap is due (rlm_busy, set by the harness), the swap is cancelled and does not count. A policy that
    decides faster than the delay is already moving and never sees a swap; a policy that thinks longer finds two
    containers swapped while it was deciding, so the command it then issues was computed for the scene before the swap.
    A latency-aware policy re-observes before committing (the latency-aware oracle waits for the swap to end); this is
    the real-world case of a person moving things while the robot deliberates, made reproducible by tying the event to
    the policy's own observation time."""
    LEVELS = {"moderate": dict(delay=(24, 40), n=3), "tight": dict(delay=(12, 24), n=4)}

    def __init__(self, *args, **kwargs):
        self.rlm_busy = False      # set by the harness while a command executes
        self.rlm_cancelled = []
        super().__init__(*args, **kwargs)

    def _refresh_swap_schedule(self):
        self.rlm_schedule, self.rlm_pairs = [], []
        self._rlm_swap_rng = _gen(self.seed, 12)
        self._rlm_inject_rng = _gen(self.seed, 13)
        self._rlm_n_inject = 0
        self.rlm_stop_step = None
        self.rlm_cancelled = []
        self.swap_schedule = []

    def rlm_on_observe(self, t):
        lv = self.LEVELS[self.live_level]
        if (self._rlm_n_inject >= lv["n"] or self.rlm_stop_step is not None or t < 70
                or any(e >= t for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs) if p != "cancel")):
            return None
        s = t + _ri(self._rlm_inject_rng, *lv["delay"])
        self.rlm_schedule.append((s, s + self.SWAP_LEN))
        self.rlm_pairs.append(None)
        self._rlm_n_inject += 1
        return s

    def rlm_events(self):
        end = self.rlm_stop_step if self.rlm_stop_step is not None else 10 ** 9
        return [(s, e, "swap") for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs)
                if s < end and p not in ("skip", "cancel")]

    def rlm_safe_step(self, t, extra=0):
        if self.rlm_stop_step is not None or "container" not in (getattr(self, "current_task_name", "") or ""):
            return t
        need, cand = extra + self.NEED, t
        for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs):
            if p == "cancel" or e <= cand:
                continue
            if s >= cand + need:
                break
            cand = e + 1
        return cand

    def step(self, action):
        t = _step(self)
        for k, (s, e) in enumerate(self.rlm_schedule):
            if s <= t and self.rlm_pairs[k] is None and self.rlm_busy:
                # due while the robot is moving: no swap (and it does not count toward the episode's swaps)
                self.rlm_pairs[k] = "cancel"
                self.rlm_cancelled.append((s, t))
                self._rlm_n_inject -= 1
        return super().step(action)


# ------------------------------------------------------------------------------------------------------ cue recall
@_register("CueRecall", "ButtonUnmask")
class CueRecall(ButtonUnmask):
    """React from memory. The cubes are shown and covered by containers (as in ButtonUnmask); the robot presses the
    button; 5-15 s later a cue cube of one of the colors appears beside the table, and the robot must then pick up the
    container hiding the cube of that color, within a deadline after the cue appears (15 s moderate, 10 s tight). The
    answer is in memory (which container hides which color, seen only at the start); the question arrives late and must
    be answered quickly. Picking up a container before the cue, a wrong one, or after the deadline fails.
    Paused, the policy can deliberate at the cue for free; live, it must notice the cue and recall the answer in time."""
    LEVELS = {"moderate": dict(delay=(100, 300), deadline=300), "tight": dict(delay=(100, 300), deadline=200)}
    CUE_XY = (0.12, 0.30)     # beside the containers' region, in the front camera's view

    def __init__(self, *args, live_level=None, **kwargs):
        seed = kwargs.get("seed", 0)
        self.live_level = live_level or ("moderate" if (int(seed) // 100) % 2 == 0 else "tight")
        g = _gen(seed, 71)
        lv = self.LEVELS[self.live_level]
        self.rlm_cue_delay = _ri(g, *lv["delay"])
        # RLM_CUE_DEADLINE=<steps> (reaction-window sweep): one deadline for every episode instead of the level's
        import os
        self.rlm_deadline = int(os.environ.get("RLM_CUE_DEADLINE") or lv["deadline"])
        self.rlm_press_step = None
        self.rlm_cue_start = None
        self.rlm_first_pick = None
        self.rlm_verdict = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options):
        super()._load_scene(options)
        # one pick only (the cue names one color); the first container hides color_names[0]
        self.task_list = self.task_list[:2]
        from robomme.robomme_env.utils.object_generation import spawn_fixed_cube
        rgba = {"red": (1, 0, 0, 1), "green": (0, 1, 0, 1), "blue": (0, 0, 1, 1)}[self.color_names[0]]
        self.rlm_cue = spawn_fixed_cube(self, position=[10.0, 10.0, 10.0], half_size=self.cube_half_size / 1.2,
                                        color=rgba, name_prefix=f"cue_cube_{self.color_names[0]}", dynamic=False)

    def rlm_events(self):
        if self.rlm_cue_start is None:
            return []
        return [(self.rlm_cue_start, self.rlm_cue_start + self.rlm_deadline, "cue")]

    def rlm_safe_step(self, t, extra=0):
        name = getattr(self, "current_task_name", "") or ""
        if "container" in name:
            if self.rlm_cue_start is None:
                return t + 400
            return max(t, self.rlm_cue_start + 1)
        return t

    def _set_cue(self, on):
        import sapien
        hs = self.cube_half_size / 1.2
        p = [self.CUE_XY[0], self.CUE_XY[1], hs] if on else [10.0, 10.0, 10.0]
        self.rlm_cue.set_pose(sapien.Pose(p=p, q=[1, 0, 0, 0]))

    def evaluate(self, solve_complete_eval=False):
        out = super().evaluate(solve_complete_eval=solve_complete_eval)
        t = _step(self)
        if self.rlm_verdict is not None:
            return {"success": torch.tensor([self.rlm_verdict]), "fail": torch.tensor([not self.rlm_verdict])}
        if self.rlm_first_pick is None and t > 70 and any(bool(is_bin_pickup(self, obj=b).any()) for b in self.spawned_bins):
            self.rlm_first_pick = t
        early = self.rlm_first_pick is not None and (self.rlm_cue_start is None or self.rlm_first_pick < self.rlm_cue_start)
        late = self.rlm_cue_start is not None and t > self.rlm_cue_start + self.rlm_deadline
        if early or (late and not bool(out["success"].any())):
            self.rlm_verdict = False
        elif bool(out["success"].any()):
            self.rlm_verdict = True
        elif bool(out["fail"].any()):
            self.rlm_verdict = False
        if self.rlm_verdict is None:
            return {"success": torch.tensor([False]), "fail": torch.tensor([False])}
        return {"success": torch.tensor([self.rlm_verdict]), "fail": torch.tensor([not self.rlm_verdict])}

    def step(self, action):
        t = _step(self)
        if self.rlm_press_step is None and t > 64:
            from robomme.robomme_env.utils.subgoal_evaluate_func import is_button_pressed
            if is_button_pressed(self, obj=self.button_left):
                self.rlm_press_step = t
                self.rlm_cue_start = t + self.rlm_cue_delay
        if self.rlm_cue_start is not None:
            on = self.rlm_cue_start <= t <= self.rlm_cue_start + self.rlm_deadline and self.rlm_first_pick is None
            if on != getattr(self, "_rlm_cue_on", False):
                self._set_cue(on)
                self._rlm_cue_on = on
        return super().step(action)


# ------------------------------------------------------------------------------------------------------- highlight
@_register("HighlightLate", "PickHighlight")
class HighlightLate(PickHighlight):
    """PickHighlight with the highlight shown once, briefly, at a random time after the button press (10-20 s),
    instead of during the first 5 s. Agents must keep watching and remember which cubes were highlighted."""
    LEVELS = {"moderate": dict(start=(200, 400), dur=60), "tight": dict(start=(200, 400), dur=30)}

    def __init__(self, *args, live_level=None, **kwargs):
        seed = kwargs.get("seed", 0)
        self.live_level = live_level or ("moderate" if (int(seed) // 100) % 2 == 0 else "tight")
        g = _gen(seed, 21)
        lv = self.LEVELS[self.live_level]
        self.rlm_hl_start = _ri(g, *lv["start"])
        self.rlm_hl_end = self.rlm_hl_start + lv["dur"]
        super().__init__(*args, **kwargs)

    def rlm_events(self):
        return [(self.rlm_hl_start, self.rlm_hl_end, "highlight")]

    def rlm_safe_step(self, t, extra=0):
        # the highlight must have been shown before the observation at t
        name = getattr(self, "current_task_name", "") or ""
        if "highlighted" in name:
            return max(t, self.rlm_hl_end + 1)
        return t

    def step(self, action):
        t = _step(self)
        target_cubes = getattr(self, "target_cubes", [])
        n = min(self.configs[self.difficulty]["pickup"], len(target_cubes))
        for i in range(n):
            highlight_obj(self, target_cubes[i], start_step=self.rlm_hl_start, end_step=self.rlm_hl_end, cur_step=t)
        return BaseEnv.step(self, action)


# ---------------------------------------------------------------------------------------------------------- reveal
@_register("RevealLate", "ButtonUnmask")
class RevealLate(ButtonUnmask):
    """ButtonUnmask with the containers lifted (revealing the cubes) once, at a random time after the button press,
    instead of at the start. The reveal is short; agents must keep watching and remember what was under which."""
    LEVELS = {"moderate": dict(start=(200, 400), dur=64), "tight": dict(start=(200, 400), dur=36)}

    def __init__(self, *args, live_level=None, **kwargs):
        seed = kwargs.get("seed", 0)
        self.live_level = live_level or ("moderate" if (int(seed) // 100) % 2 == 0 else "tight")
        g = _gen(seed, 41)
        lv = self.LEVELS[self.live_level]
        self.rlm_rv_start = _ri(g, *lv["start"])
        self.rlm_rv_end = self.rlm_rv_start + lv["dur"]
        # the containers are away during the first half of the reveal and dropped back at its midpoint
        self.rlm_rv_drop = min(self.rlm_rv_end, self.rlm_rv_start + max(1, lv["dur"] // 2))
        self._rlm_skip = set()
        super().__init__(*args, **kwargs)
        self.rlm_reserved = None   # set by the live clock to the object the current command acts on

    def rlm_events(self):
        return [(self.rlm_rv_start, self.rlm_rv_end, "reveal")]

    def rlm_safe_step(self, t, extra=0):
        # the reveal must have happened before the observation at t
        name = getattr(self, "current_task_name", "") or ""
        if "container" in name:
            return max(t, self.rlm_rv_end + 1)
        return t

    def evaluate(self, solve_complete_eval=False):
        # the reveal lifts every container; do not count that as the robot picking one up
        t = _step(self)
        if self.rlm_rv_start <= t <= self.rlm_rv_end + 5:
            return {"success": torch.tensor([False]), "fail": torch.tensor([False])}
        return super().evaluate(solve_complete_eval=solve_complete_eval)

    def rlm_block_until(self, t):
        # a command cannot act on the containers while they are away; it starts when they are back
        if self.rlm_rv_start <= t <= self.rlm_rv_drop:
            return self.rlm_rv_drop + 1
        return t

    def step(self, action):
        t = _step(self)
        for i in range(15):
            if hasattr(self, f"bin_{i}"):
                b = getattr(self, f"bin_{i}")
                # a container the robot is already acting on when the reveal starts stays where it is
                if t == self.rlm_rv_start and b is self.rlm_reserved:
                    self._rlm_skip.add(id(b))
                if id(b) in self._rlm_skip:
                    continue
                lift_and_drop_objects_back_to_original(self, obj=b, start_step=self.rlm_rv_start,
                                                       end_step=self.rlm_rv_end, cur_step=t)
        return BaseEnv.step(self, action)


# -------------------------------------------------------------------------------------------------------- stopcube
@_register("StopCubeFast", "StopCube")
class StopCubeFast(StopCube):
    """StopCube with a faster cube (one pass every 2-3 s instead of 3-6 s): the press must land in a shorter window,
    and the agent must count passes it partly observes while thinking."""
    LEVELS = {"moderate": [60], "tight": [50]}

    def __init__(self, *args, live_level=None, **kwargs):
        seed = kwargs.get("seed", 0)
        self.live_level = live_level or ("moderate" if (int(seed) // 100) % 2 == 0 else "tight")
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        from robomme.robomme_env.utils.subgoal_evaluate_func import before_absTimestep
        from robomme.robomme_env.utils.subgoal_planner_func import solve_hold_obj_absTimestep
        g = _gen(self.seed, 31)
        opts = self.LEVELS[self.live_level]
        self.move_interval = opts[_ri(g, 0, len(opts) - 1)]
        stop_time = _ri(g, 3, 5)
        self.stop_time = stop_time
        self.steps_press = self.move_interval * stop_time - self.move_interval / 2
        self.stop_time_range = (self.move_interval * (stop_time - 1), self.move_interval * stop_time)
        hover, press = self.task_list[0], self.task_list[-1]
        final_abs = self.steps_press - self.interval
        checkpoints = list(range(100, int(final_abs), 100))
        if not checkpoints or checkpoints[-1] != final_abs:
            checkpoints.append(final_abs)
        mids = [{"func": (lambda ts=ts: before_absTimestep(self, absTimestep=ts)), "name": "remain static",
                 "subgoal_segment": "remain static", "choice_label": "remain static", "demonstration": False,
                 "failure_func": None, "specialflag": "swap",
                 "solve": (lambda env, planner, ts=ts: solve_hold_obj_absTimestep(env, planner, absTimestep=ts))}
                for ts in checkpoints]
        self.task_list = [hover] + mids + [press]

    def rlm_events(self):
        return [(self.move_interval * k, self.move_interval * (k + 1), "pass") for k in range(5)]

    def step(self, action):
        # same as StopCube.step but with the faster interval (StopCube.step reads self.move_interval)
        return StopCube.step(self, action)


# ------------------------------------------------------------------------------------------------------- repick
@_register("RepickStream", "VideoRepick")
class RepickStream(VideoRepick):
    """VideoRepick (watch which cube is picked in the video, then pick that cube N times) with the cubes swapped in
    bursts while the robot works: identical-looking cubes keep trading places for ~35 s after control starts.
    A swap only starts when no cube is held."""
    LEVELS = {"moderate": dict(gap=(150, 220)), "tight": dict(gap=(100, 130))}
    STREAM_LEN = 700
    SWAP_LEN = 50
    NEED = 10

    def __init__(self, *args, live_level=None, **kwargs):
        seed = kwargs.get("seed", 0)
        self.live_level = live_level or ("moderate" if (int(seed) // 100) % 2 == 0 else "tight")
        self.rlm_anchor = None
        self.rlm_schedule = []
        self.rlm_pairs = []
        self.rlm_reserved = None
        super().__init__(*args, **kwargs)

    def _rlm_build(self, anchor):
        g = _gen(self.seed, 51)
        lo, hi = self.LEVELS[self.live_level]["gap"]
        sched, t = [], anchor + 40
        while t < anchor + self.STREAM_LEN:
            sched.append((t, t + self.SWAP_LEN))
            t = t + self.SWAP_LEN + _ri(g, lo, hi)
        self.rlm_schedule, self.rlm_pairs = sched, [None] * len(sched)
        self._rlm_swap_rng = _gen(self.seed, 52)

    def _held(self):
        return [c for c in self.spawned_cubes if float(c.pose.p[0, 2]) > self.cube_half_size * 2 + 0.02]

    def rlm_events(self):
        return [(s, e, "swap") for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs) if p != "skip"]

    def rlm_safe_step(self, t, extra=0):
        name = getattr(self, "current_task_name", "") or ""
        if "pick up" not in name or not self.rlm_schedule:
            return t
        need, cand = extra + self.NEED, t
        for (s, e), p in zip(self.rlm_schedule, self.rlm_pairs):
            if p == "skip" or e <= cand:
                continue
            if s >= cand + need:
                break
            cand = e + 1
        return cand

    def step(self, action):
        t = _step(self)
        if self.rlm_anchor is None and getattr(self, "current_task_demonstration", True) is False and t > 0:
            self.rlm_anchor = t
            self._rlm_build(t)
        for k, (s, e) in enumerate(self.rlm_schedule):
            if not (s <= t <= e) or self.rlm_pairs[k] == "skip":
                continue
            if self.rlm_pairs[k] is None:
                held = self._held()
                if held:
                    self.rlm_pairs[k] = "skip"
                    continue
                free = [c for c in self.spawned_cubes if c is not self.rlm_reserved]
                pos = [self._get_actor_position(c) for c in free]
                pr = self._select_swap_pair_from_positions(pos, self._rlm_swap_rng)
                if pr is None:
                    self.rlm_pairs[k] = "skip"
                    continue
                self.rlm_pairs[k] = (free[pr["idx1"]], free[pr["idx2"]])
            a, b = self.rlm_pairs[k]
            held = self._held()
            _unpin(self, a, b, s, e, [self.rlm_reserved] + held)
            swap_flat_two_lane(self, cube_a=a, cube_b=b, start_step=s, end_step=e, cur_step=t, lane_offset=0.07,
                               smooth=True, keep_upright=True,
                               other_cube=[x for x in self.spawned_cubes if x is not a and x is not b
                                           and x is not self.rlm_reserved and not any(x is h for h in held)])
        return VideoRepick.step(self, action)


# ---------------------------------------------------------------------------------------------------- patternlock
@_register("PatternLockLate", "PatternLock")
class PatternLockLate(PatternLock):
    """PatternLock without the demonstration video: the stick starts on the first button, and at a random time 5-15 s
    into the episode the pattern's buttons light up one by one, once. The agent must then retrace the pattern."""
    LEVELS = {"moderate": dict(on=30, gap=10), "tight": dict(on=16, gap=6)}

    def __init__(self, *args, live_level=None, **kwargs):
        seed = kwargs.get("seed", 0)
        self.live_level = live_level or ("moderate" if (int(seed) // 100) % 2 == 0 else "tight")
        self.rlm_anchor = None
        self.rlm_show = []
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        # keep only "move to the start button" (demonstration) and the agent's moves
        first = self.task_list[0]
        agent = [t for t in self.task_list if t.get("demonstration") is False]
        self.task_list = [first] + agent
        self.rlm_anchor, self.rlm_show = None, []

    def _rlm_build(self, anchor):
        g = _gen(self.seed, 61)
        lv = self.LEVELS[self.live_level]
        start = anchor + _ri(g, 100, 300)
        self.rlm_show = [(b, start + i * (lv["on"] + lv["gap"]), start + i * (lv["on"] + lv["gap"]) + lv["on"])
                         for i, b in enumerate(self.selected_buttons)]

    def rlm_events(self):
        return [(s, e, "highlight") for _, s, e in self.rlm_show]

    def rlm_safe_step(self, t, extra=0):
        name = getattr(self, "current_task_name", "") or ""
        if name.startswith("move") and self.rlm_show:
            return max(t, self.rlm_show[-1][2] + 1)
        if name.startswith("move") and not self.rlm_show:
            return t + 400  # pattern not scheduled yet
        return t

    def step(self, action):
        t = _step(self)
        if self.rlm_anchor is None and getattr(self, "current_task_demonstration", True) is False and t > 0:
            self.rlm_anchor = t
            self._rlm_build(t)
            self.after_demo = True      # start recording touched buttons (RoboMME sets this in its post-demo reset)
            self.achieved_list = []
        for b, s0, e0 in self.rlm_show:
            if s0 - 1 <= t <= e0 + 1:
                highlight_obj(self, b, start_step=s0, end_step=e0, cur_step=t, disk_radius=0.02 * 1.002,
                              disk_half_length=0.01 * 2 * 1.002, highlight_color=[1.0, 0.0, 0.0, 1.0],
                              use_target_style=True)
        return PatternLock.step(self, action)


# ----------------------------------------------------------------------------------------------- table registration
def _install_tables():
    for name, base in LIVE_TASKS.items():
        if base in vqa_options.OPTION_BUILDERS:
            vqa_options.OPTION_BUILDERS[name] = vqa_options.OPTION_BUILDERS[base]
    orig = task_goal.get_language_goal
    if getattr(orig, "_rlm", False):
        return

    def get_language_goal(env, env_id, *a, **k):
        if env_id in GOAL_OVERRIDE:
            return [GOAL_OVERRIDE[env_id]]
        return orig(env, LIVE_TASKS.get(env_id, env_id), *a, **k)
    get_language_goal._rlm = True
    task_goal.get_language_goal = get_language_goal
    # modules that imported the function by name
    import robomme.env_record_wrapper.DemonstrationWrapper as dw
    if hasattr(dw, "task_goal"):
        dw.task_goal.get_language_goal = get_language_goal


_install_tables()


# Instances whose default seed makes RoboMME's own scene loader crash (found by the oracle sweeps); the seed is bumped by
# the given amount, which keeps the instance's live level ((seed // 100) % 2) unchanged.
SEED_FIX = {("SwapStream", 19): 1}


def write_metadata(out_dir, n=50, base_seed=880000):
    """RoboMME-format metadata for the live-native tasks (seeds disjoint from RoboMME's)."""
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    diffs = ["easy", "medium", "hard"]
    for name in sorted(LIVE_TASKS):
        ti = TASK_SEED_SLOT[name]
        recs = [{"task": name, "episode": e, "seed": base_seed + ti * 10000 + e * 100 + SEED_FIX.get((name, e), 0),
                 "difficulty": diffs[e % 3]} for e in range(n)]
        with open(os.path.join(out_dir, f"record_dataset_{name}_metadata.json"), "w") as f:
            json.dump({"env_id": name, "record_count": n, "records": recs}, f, indent=1)


# 2026-11-06: with RLM_STRICT_TASKS=1 in the environment, also register the strict world-time variants (tasks_live_strict.py);
# without it nothing changes.
import os as _os_rlm
if _os_rlm.environ.get("RLM_STRICT_TASKS"):
    from rlm_controlled import tasks_live_strict  # noqa: E402,F401
