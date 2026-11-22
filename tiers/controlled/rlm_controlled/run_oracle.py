"""Oracle latency sweep: privileged oracle, every decision charged a fixed latency L (seconds).

Runs with an unlimited budget (up to --max-steps) and records the step of success, so success@K for any K is derived
offline: success and success_step - t0 <= K.

usage: python -m rlm_controlled.run_oracle --tasks StopCube,PickXtimes --eps 0-4 --lat 0,1,2,4,8 --out results/oracle.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math  # noqa: E402
from rlm_controlled.live import LiveEpisode, privileged_oracle, _last, CONTROL_HZ  # noqa: E402
from rlm_controlled import tasks_live  # noqa: E402,F401  (registers live-native tasks)

LIVE_NATIVE = ["SwapStream", "HighlightLate", "RevealLate", "StopCubeFast", "RepickStream", "PatternLockLate"]
ALL_TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask",
             "ButtonUnmaskSwap", "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube",
             "InsertPeg", "PatternLock", "RouteStick"]


def parse_eps(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


class _Stuck(Exception):
    pass


def _alarm(signum, frame):
    raise _Stuck()


def run_one(task, ep, L, max_steps, fast=True, max_decisions=60, oracle="aware", wall_limit_s=180):
    """One oracle episode. A wall-clock watchdog ends episodes in which the motion planner loops (rare: e.g. the naive
    oracle grasping a container while it is being revealed); they are recorded as status "stuck" (a failure)."""
    import signal
    t_wall = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(wall_limit_s)
    E = LiveEpisode(task, ep, budget_steps=None, keep_frames=False, fast=fast, max_env_steps=max_steps)
    rec = {"task": task, "ep": ep, "latency_s": L, "fast": fast, "oracle": oracle}
    try:
        E.reset()
        rec["t0"] = E.t0
        rec["goal"] = E.task_goal
        decisions = []
        U = E.env.unwrapped
        dl = int(math.ceil(L * CONTROL_HZ))
        for k in range(max_decisions):
            if hasattr(U, "rlm_on_observe"):
                U.rlm_on_observe(E.elapsed())      # InterruptSwap: the scene reacts to every observation
            if oracle == "aware" and hasattr(U, "steps_press") and hasattr(U, "interval"):
                # StopCube family: anticipate. The press must start executing at press_at; decide dl earlier.
                now = E.elapsed()
                press_at = int(U.steps_press - U.interval)
                name = getattr(U, "current_task_name", "") or ""
                if name.startswith("remain static") or name.startswith("press"):
                    if now + 2 * dl < press_at:          # time for a wait decision first
                        w = max(1, press_at - now - 2 * dl)
                        E.hold(dl + w)
                        decisions.append({"sub": name, "cmd": {"wait_steps": w}, "t_dec": now - E.t0,
                                          "t_end": E.elapsed() - E.t0, "status": E.status})
                        if E.status != "ongoing":
                            break
                        continue
                    opts = E.info.get("available_multi_choices") or []
                    press = next((o for o in opts if o.get("action", "").startswith("press")), None)
                    if press is not None:
                        r = E.act({"choice": press["label"].upper()}, L, name)
                        decisions.append({"sub": name, "cmd": {"choice": press["label"].upper()},
                                          "t_dec": r.t_decide - E.t0, "t_exec": r.t_exec - E.t0,
                                          "t_end": r.t_end - E.t0, "status": r.status})
                        if E.status != "ongoing":
                            break
                        continue
            if oracle == "aware" and hasattr(U, "rlm_safe_step"):
                # latency-aware: the decision observes the world now and acts dl steps later; it is safe only if the
                # needed information is already revealed now and nothing relevant moves in [now, now + dl + need).
                # Otherwise spend this decision on a wait so that the next one (made dl + w later) is safe.
                now = E.elapsed()
                if U.rlm_safe_step(now, dl) > now:
                    nxt = U.rlm_safe_step(now + dl, dl)
                    w = max(1, nxt - now - dl)
                    E.hold(dl + w)
                    decisions.append({"sub": getattr(U, "current_task_name", None), "cmd": {"wait_steps": w},
                                      "t_dec": now - E.t0, "t_end": E.elapsed() - E.t0, "status": E.status})
                    if E.status != "ongoing":
                        break
                    continue
            cmd = privileged_oracle(E)
            sub = getattr(E.env.unwrapped, "current_task_name", None)
            if cmd is None:
                rec["note"] = f"no matching option for subgoal {sub!r}"
                E.status = "no_option"
                break
            r = E.act(cmd, L, sub)
            decisions.append({"sub": sub, "cmd": cmd, "t_dec": r.t_decide - E.t0, "t_exec": r.t_exec - E.t0,
                              "t_end": r.t_end - E.t0, "status": r.status,
                              "sub_exec": getattr(E.env.unwrapped, "current_task_name", None)})
            if E.status != "ongoing":
                break
        rec["status"] = E.status
        rec["steps"] = E.used()
        rec["success"] = E.status == "success"
        rec["decisions"] = decisions
        rec["n_decisions"] = len(decisions)
        if E.status == "error":
            rec["error"] = E.info.get("error_message")
    except _Stuck:
        rec["status"] = "stuck"
        rec["success"] = False
        try:
            rec["steps"] = E.used()
        except Exception:
            pass
    except Exception as exc:
        rec["status"] = "crash"
        rec["success"] = False
        rec["error"] = traceback.format_exc()[-600:]
    finally:
        signal.alarm(0)
        E.close()
    rec["wall_s"] = round(time.time() - t_wall, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--eps", default="0-4")
    ap.add_argument("--lat", default="0,1,2,4,8,16")
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slow", action="store_true", help="render every step (needed for image agents, not the oracle)")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--oracle", default="aware", choices=["aware", "naive"])
    ap.add_argument("--max-jobs", type=int, default=0,
                    help="exit after this many jobs (the caller restarts the process; bounds memory growth)")
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    eps = parse_eps(a.eps)
    lats = [float(x) for x in a.lat.split(",")]
    si, sn = [int(x) for x in a.shard.split("/")]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["task"], r["ep"], r["latency_s"]))
            except Exception:
                pass
    jobs = [(t, e, L) for t in tasks for e in eps for L in lats]
    jobs = [j for i, j in enumerate(jobs) if i % sn == si and j not in done]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"{len(jobs)} jobs", flush=True)
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    for (t, e, L) in jobs:
        r = run_one(t, e, L, a.max_steps, fast=not a.slow, oracle=a.oracle)
        with open(a.out, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(f"{t} ep{e} L={L}: {r['status']} steps={r.get('steps')} dec={r.get('n_decisions')} wall={r['wall_s']}s",
              flush=True)


if __name__ == "__main__":
    main()
