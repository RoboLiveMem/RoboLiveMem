"""RoboLiveMem: the live (latency-charged) clock on top of RoboMME's oracle-planner (multi-choice) interface.

A decision that takes `latency` seconds advances the simulator ceil(latency * f) control steps while the robot holds its
current joint targets; then the decided command executes. Frames rendered during the hold are recorded and handed to the
next decision. latency = 0 everywhere is the paused protocol (current practice).

The command is RoboMME's multi-choice command {"choice": label, "point": [y, x]}; the point is resolved to the nearest
candidate object *when the command executes*, so a point chosen from a stale observation can select the wrong object.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import planner_denseStep

CONTROL_HZ = 20  # ManiSkill default control rate; RoboMME does not override it
DEFAULT_BUDGET = 1300  # RoboMME's max_steps


def _scalar(x):
    if isinstance(x, torch.Tensor):
        return x.reshape(-1)[-1].item() if x.numel() else None
    if isinstance(x, np.ndarray):
        return x.reshape(-1)[-1].item() if x.size else None
    if isinstance(x, list):
        return _scalar(x[-1]) if x else None
    return x


def _last(v):
    return v[-1] if isinstance(v, list) and v else v


@dataclass
class StepRecord:
    t_decide: int            # env step at which the decision started (observation time)
    latency_s: float
    hold_steps: int
    command: Any
    t_exec: int              # env step at which the command started executing
    t_end: int
    status: str
    subgoal_at_decide: Optional[str] = None
    subgoal_at_exec: Optional[str] = None


@dataclass
class LiveEpisode:
    env_id: str
    episode_idx: int
    dataset: str = "test"
    budget_steps: Optional[int] = DEFAULT_BUDGET   # None = unlimited (holds and execution never time out)
    control_hz: int = CONTROL_HZ
    keep_frames: bool = True
    frame_stride: int = 1
    max_env_steps: int = 4000   # hard cap so a stuck run ends
    fast: bool = False          # skip camera rendering (sensor data cached); for privileged-oracle sweeps only
    render_every: int = 1       # render cameras only every k steps (agents sample frames on this grid)
    env: Any = None
    frames_front: list = field(default_factory=list)
    frames_wrist: list = field(default_factory=list)
    frame_steps: list = field(default_factory=list)
    demo_len: int = 0
    log: list = field(default_factory=list)
    status: str = "ongoing"
    info: dict = field(default_factory=dict)
    task_goal: str = ""

    # ------------------------------------------------------------------ setup
    def reset(self):
        from rlm_controlled import tasks_live
        override = None
        if self.env_id in tasks_live.LIVE_TASKS:
            import os
            override = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata")
            if not os.path.exists(os.path.join(override, f"record_dataset_{self.env_id}_metadata.json")):
                tasks_live.write_metadata(override)
        b = BenchmarkEnvBuilder(env_id=self.env_id, dataset=self.dataset, action_space="multi_choice",
                                gui_render=False, max_steps=self.max_env_steps, override_metadata_path=override)
        if self.fast:
            _install_render_cache()
        elif self.render_every > 1:
            _install_sparse_render(self.render_every)
        self.env = b.make_env_for_episode(self.episode_idx, include_available_multi_choices=True)
        obs, info = self.env.reset()
        self.info = info
        self.task_goal = _last(info.get("task_goal")) or ""
        n = len(obs.get("front_rgb_list", []))
        self.demo_len = max(0, n - 1)  # frames before the last are the demonstration video (if any)
        self._ingest(obs, info, demo=True)
        self.status = "ongoing"
        self.t0 = self.elapsed()  # steps used by a demonstration video (if any) are not charged to the budget
        return obs, info

    def used(self) -> int:
        return self.elapsed() - getattr(self, "t0", 0)

    def elapsed(self) -> int:
        try:
            return int(_scalar(self.env.unwrapped.elapsed_steps))
        except Exception:
            return int(_scalar(self.info.get("elapsed_steps")) or 0)

    # ------------------------------------------------------------------ frames
    def _ingest(self, obs, info, demo=False):
        if not obs:
            return
        fr = obs.get("front_rgb_list") or []
        wr = obs.get("wrist_rgb_list") or []
        es = info.get("elapsed_steps") if isinstance(info, dict) else None
        if not (isinstance(es, list) and len(es) == len(fr)):
            # flat info (last step only): the batch holds consecutive steps ending now
            end = self.elapsed()
            es = [end - (len(fr) - 1 - i) for i in range(len(fr))]
        for i in range(len(fr)):
            if not self.keep_frames:
                break
            if (len(self.frames_front) % self.frame_stride) == 0 or demo:
                self.frames_front.append(np.asarray(fr[i].cpu() if hasattr(fr[i], "cpu") else fr[i]).astype(np.uint8))
                if i < len(wr):
                    self.frames_wrist.append(np.asarray(wr[i].cpu() if hasattr(wr[i], "cpu") else wr[i]).astype(np.uint8))
                step_i = _scalar(es[i]) if isinstance(es, list) and i < len(es) else None
                self.frame_steps.append(-1 if demo and i < len(fr) - 1 else (step_i if step_i is not None else self.elapsed()))

    # ------------------------------------------------------------------ status
    def _update_status(self, info) -> str:
        st = _last(info.get("status")) if isinstance(info, dict) else None
        if st in ("success", "fail", "error"):
            return st
        if self.budget_steps is not None and self.used() >= self.budget_steps:
            return "timeout"
        if self.used() >= self.max_env_steps:
            return "timeout"
        return "ongoing"

    # ------------------------------------------------------------------ hold
    def _hold_action(self):
        mc = self._mc_wrapper()
        planner = mc.planner
        U = self.env.unwrapped
        q = U.agent.robot.get_qpos()
        q = q.cpu().numpy().reshape(-1) if hasattr(q, "cpu") else np.asarray(q).reshape(-1)
        if self.env_id in ("PatternLock", "RouteStick", "PatternLockLate"):
            return q[:7].astype(np.float32)
        g = getattr(planner, "gripper_state", 1.0)
        return np.concatenate([q[:7], [g]]).astype(np.float32)

    def _mc_wrapper(self):
        e = self.env
        for _ in range(12):
            if type(e).__name__ == "OraclePlannerDemonstrationWrapper":
                return e
            e = e.env
        raise RuntimeError("OraclePlannerDemonstrationWrapper not found")

    def hold(self, n_steps: int):
        """Advance the world n_steps with the robot holding its current targets."""
        if n_steps <= 0 or self.status != "ongoing":
            return
        mc = self._mc_wrapper()
        planner = mc.planner
        steps_left = n_steps
        if self.budget_steps is not None:
            steps_left = min(steps_left, max(0, self.budget_steps - self.used()))
        steps_left = min(steps_left, max(0, self.max_env_steps - self.used()))
        a = self._hold_action()

        def _do():
            for _ in range(steps_left):
                out = planner.env.step(a)
                info = out[-1] if isinstance(out, tuple) else {}
                st = _last(info.get("status")) if isinstance(info, dict) else None
                if st in ("success", "fail"):
                    break
            return None

        batch = planner_denseStep._run_with_dense_collection(planner, _do)
        if isinstance(batch, tuple) and len(batch) == 5:
            obs, _, _, _, info = batch
            info_flat = {k: _last(v) for k, v in (info or {}).items()}
            self._ingest(obs, info)
            self.info.update({k: v for k, v in info_flat.items() if v is not None})
            self.status = self._update_status(info_flat)
        if self.status == "ongoing" and ((self.budget_steps is not None and self.used() >= self.budget_steps)
                                         or self.used() >= self.max_env_steps):
            self.status = "timeout"

    # ------------------------------------------------------------------ act
    def act(self, command: dict, latency_s: float = 0.0, subgoal_at_decide: Optional[str] = None) -> StepRecord:
        t_dec = self.elapsed()
        d = int(math.ceil(max(0.0, latency_s) * self.control_hz))
        self.hold(d)
        U = self.env.unwrapped
        if self.status == "ongoing" and hasattr(U, "rlm_block_until"):
            # e.g. RevealLate: the containers are away for half of the reveal; a command waits until they are back
            w = int(U.rlm_block_until(self.elapsed())) - self.elapsed()
            if w > 0:
                self.hold(w)
        t_exec = self.elapsed()
        sub_exec = _last(self.info.get("simple_subgoal_online"))
        if self.status == "ongoing" and hasattr(U, "rlm_reserved"):
            # tasks with moving objects never move the object the robot is about to manipulate (resolved now, at
            # execution time, exactly as the planner will resolve it)
            U.rlm_reserved = resolve_command_target(self, command)
        if self.status == "ongoing":
            if hasattr(U, "rlm_busy"):
                U.rlm_busy = True        # InterruptSwap: nobody moves things while the robot is moving
            try:
                obs, _, term, trunc, info = self.env.step(command)
                self._ingest(obs, info)
                self.info.update({k: v for k, v in info.items() if v is not None})
                self.status = self._update_status(info)
            except Exception as exc:  # planner failure etc.
                self.status = "error"
                self.info["error_message"] = repr(exc)[:300]
            finally:
                if hasattr(U, "rlm_busy"):
                    U.rlm_busy = False
        if hasattr(U, "rlm_reserved"):
            U.rlm_reserved = None
        rec = StepRecord(t_decide=t_dec, latency_s=latency_s, hold_steps=d, command=command, t_exec=t_exec,
                         t_end=self.elapsed(), status=self.status, subgoal_at_decide=subgoal_at_decide,
                         subgoal_at_exec=sub_exec)
        self.log.append(rec)
        return rec

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


_FAST_PATCHED = False


def _install_render_cache(U=None):
    """Never render (process-wide): sensor data become zero images of the right shapes. Physics, task logic,
    success/fail checks and target resolution (projection of object poses) do not read rendered images, so a privileged
    oracle is unaffected; images and segmentation-filled subgoal text are meaningless in this mode. Patched on the class
    so the render inside gym.make's initial reset is skipped too."""
    global _FAST_PATCHED
    if _FAST_PATCHED:
        return
    from mani_skill.envs.sapien_env import BaseEnv

    def patched(self, *a, **k):
        cache = getattr(self, "_rlm_dummy_sensor", None)
        if cache is None:
            cache = {}
            for name, sensor in self._sensors.items():
                cfg = sensor.config
                H, W = int(cfg.height), int(cfg.width)
                cache[name] = {"rgb": torch.zeros((1, H, W, 3), dtype=torch.uint8),
                               "depth": torch.zeros((1, H, W, 1), dtype=torch.int16),
                               "segmentation": torch.zeros((1, H, W, 1), dtype=torch.int16)}
            self._rlm_dummy_sensor = cache
        return cache

    BaseEnv._get_obs_sensor_data = patched
    if not os.environ.get("RLM_NO_LIGHT"):
        _install_light_collection()
    _FAST_PATCHED = True


def _install_light_collection():
    """No-render mode only: RoboMME's planner wrappers keep the full observation of every simulator step of a command
    (dense collection), which in long oracle episodes grows to several GB. Keep every step's info but only the last
    step's observation, shared by all steps of the batch (observations are zero images in this mode anyway)."""
    from robomme.robomme_env.utils import planner_denseStep as pds

    def _collect_light(planner, fn):
        collected = []
        last_obs = [None]
        original_step = planner.env.step

        def _step(action):
            out = original_step(action)
            for o, r, te, tr, inf in pds._output_to_steps(out):
                last_obs[0] = o
                collected.append([None, r, te, tr, inf])
            return out

        planner.env.step = _step
        try:
            result = fn()
            if result == -1:
                return -1
            for c in collected:
                c[0] = last_obs[0]
            return [tuple(c) for c in collected]
        finally:
            planner.env.step = original_step

    pds._collect_dense_steps = _collect_light


_SPARSE_PATCHED = None


def _install_sparse_render(every: int):
    """Render cameras only on steps that are multiples of `every` (and whenever env._rlm_force_render is set);
    other steps reuse the last rendered images. Agents sample frames on the same grid, so they see fresh images;
    this makes CPU rendering `every` times cheaper."""
    global _SPARSE_PATCHED
    if _SPARSE_PATCHED is not None or every <= 1:
        return
    from mani_skill.envs.sapien_env import BaseEnv
    orig = BaseEnv._get_obs_sensor_data

    def patched(self, *a, **k):
        try:
            t = int(self.elapsed_steps.reshape(-1)[0].item()) if isinstance(self.elapsed_steps, torch.Tensor) \
                else int(self.elapsed_steps)
        except Exception:
            t = 0
        cache = getattr(self, "_rlm_sparse_cache", None)
        if cache is None or t % every == 0 or getattr(self, "_rlm_force_render", False):
            cache = orig(self, *a, **k)
            self._rlm_sparse_cache = cache
            self._rlm_sparse_step = t
        return cache

    BaseEnv._get_obs_sensor_data = patched
    _SPARSE_PATCHED = every


def render_now(U) -> np.ndarray:
    """Fresh front-camera RGB of the current state (no simulation step)."""
    U._rlm_force_render = True
    try:
        sd = U._get_obs_sensor_data()
    finally:
        U._rlm_force_render = False
    img = sd["base_camera"]["rgb"][0]
    return (img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)).astype(np.uint8)


def resolve_command_target(E, command):
    """The object a multi-choice command's point selects in the current scene (None if no point / no candidates)."""
    if not isinstance(command, dict) or "point" not in command:
        return None
    try:
        from robomme.robomme_env.utils.vqa_options import get_vqa_options
        from robomme.robomme_env.utils.choice_action_mapping import project_world_to_pixel, extract_actor_position_xyz
        mc = E._mc_wrapper()
        tgt = {"obj": None, "name": None, "seg_id": None, "position": None, "match_distance": None,
               "selection_mode": None}
        opts = get_vqa_options(mc.env, mc.planner, tgt, E.env_id)
        label = str(command.get("choice", "")).lower()
        opt = next((o for o in opts if o.get("label", "").lower() == label), None)
        av = (opt or {}).get("available") or []
        y, x = command["point"]
        best, bd = None, 1e18
        for a in (av if isinstance(av, (list, tuple)) else [av]):
            pos = extract_actor_position_xyz(a)
            px = project_world_to_pixel(pos, mc._front_camera_intrinsic_cv, mc._front_camera_extrinsic_cv,
                                        mc._front_rgb_shape) if pos is not None else None
            if px is None:
                continue
            dd = (px[0] - x) ** 2 + (px[1] - y) ** 2
            if dd < bd:
                best, bd = a, dd
        return best
    except Exception:
        return None


# ---------------------------------------------------------------------- oracle
_POINT = re.compile(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>")


def oracle_command(info: dict) -> Optional[dict]:
    """The privileged oracle: the current ground-truth subgoal mapped to a multi-choice command.

    It reads RoboMME's online grounded subgoal (privileged) at decision time; with latency the command executes later,
    possibly against a changed scene. Returns None if no option matches.
    """
    simple = _last(info.get("simple_subgoal_online")) or ""
    grounded = _last(info.get("grounded_subgoal_online")) or ""
    opts = info.get("available_multi_choices") or []
    simple_l = simple.lower().strip()
    best = None
    for o in opts:
        act = (o.get("action") or "").lower().strip()
        if not act:
            continue
        if simple_l.startswith(act) or act in simple_l:
            if best is None or len(act) > len(best.get("action", "")):
                best = o
    if best is None:
        return None
    cmd = {"choice": best["label"].upper()}
    m = _POINT.search(grounded)
    if m:
        cmd["point"] = [int(m.group(1)), int(m.group(2))]
    return cmd


def _match_option(name: str, opts) -> Optional[dict]:
    name_l = (name or "").lower().strip()
    best = None
    for o in opts or []:
        act = (o.get("action") or "").lower().strip()
        if act and (name_l.startswith(act) or act in name_l):
            if best is None or len(act) > len(best.get("action", "")):
                best = o
    return best


def privileged_oracle(E: "LiveEpisode") -> Optional[dict]:
    """Oracle from simulator state at decision time: current subgoal -> option label; current target object's
    position projected into the front camera -> point [y, x]. Works in fast (no-render) mode."""
    from robomme.robomme_env.utils.choice_action_mapping import project_world_to_pixel, extract_actor_position_xyz
    U = E.env.unwrapped
    mc = E._mc_wrapper()
    opts = E.info.get("available_multi_choices") or []
    label = (getattr(U, "current_choice_label", None) or "").lower().strip()
    opt = next((o for o in opts if (o.get("action") or "").lower().strip() == label), None) if label else None
    if opt is None:
        name = getattr(U, "current_task_name", None) or _last(E.info.get("simple_subgoal_online"))
        opt = _match_option(name, opts)
    if opt is None:
        return None
    cmd = {"choice": opt["label"].upper()}
    if opt.get("need_parameter"):
        seg = getattr(U, "current_segment", None)
        if isinstance(seg, (list, tuple)):
            seg = seg[0] if seg else None
        if seg is not None:
            pos = extract_actor_position_xyz(seg)
            px = project_world_to_pixel(pos, mc._front_camera_intrinsic_cv, mc._front_camera_extrinsic_cv,
                                        mc._front_rgb_shape) if pos is not None else None
            if px is not None:
                cmd["point"] = [int(px[1]), int(px[0])]
    return cmd
