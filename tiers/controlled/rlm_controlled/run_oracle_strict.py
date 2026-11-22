"""Clock-sensitivity test for the strict live-memory definition (2026-11-06). run_oracle.py is unchanged.

Three scripted policies, each at a fixed decision latency L (seconds; L=0 is the paused clock):
  naive  - run_oracle.py's naive oracle: privileged subgoal/target read at decision time, acts at once (no waits).
           For late-event tasks it is clairvoyant (it knows the target before the event is shown).
  stale  - the stale reference policy of the strict definition: it acts only on what it has seen. If the information the
           subgoal needs has not been shown yet, or the relevant objects are visibly moving at the observation, it spends the
           decision on a wait; otherwise it plans from the observation it decided on and acts after L, never re-checking.
           It does not anticipate events it cannot see. (= naive on tasks without rlm_safe_step.)
  aware  - run_oracle.py's latency-aware oracle (anticipates: acts only if nothing relevant moves in [now, now+L+need)).
Outputs the same records as run_oracle.py plus "events" (world events of the episode) for staleness analysis.

usage: python -m rlm_controlled.run_oracle_strict --tasks SwapStream --eps 0-19 --lat 0,3 --oracle stale --out x.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlm_controlled.live import LiveEpisode, privileged_oracle, CONTROL_HZ  # noqa: E402
from rlm_controlled import tasks_live  # noqa: E402,F401
from rlm_controlled import tasks_live_strict  # noqa: E402,F401  (registers InterruptSwapWorld)
from rlm_controlled.run_oracle import parse_eps, _Stuck, _alarm  # noqa: E402

SWAP_TASKS = {"SwapStream", "InterruptSwap", "InterruptSwapWorld", "RepickStream", "SwapStreamW", "RepickStreamW"}
SWAP_TASKS |= {"InterruptSwapW", "InterruptSwapWV", "SwapStreamWV", "RepickStreamWV"}   # 2026-11-15
BLIND_WAIT = 10       # steps (0.5 s) of a frame-blind wait decision


def stale_ready_at(E, U, now):
    """Earliest step >= now at which the stale reference would act (what it can see at `now`)."""
    if not hasattr(U, "rlm_safe_step"):
        return now
    r0 = int(U.rlm_safe_step(now, 0))
    if r0 <= now:
        return now
    if E.env_id in SWAP_TASKS:
        ev = U.rlm_events() if hasattr(U, "rlm_events") else []
        act = [e for s, e, k in ev if s <= now <= e]
        return (max(act) + 1) if act else now      # a swap it cannot see yet is not waited for
    return r0                                       # needed information not shown yet: wait for it


def run_one(task, ep, L, max_steps, fast=True, max_decisions=80, oracle="aware", wall_limit_s=240):
    t_wall = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(wall_limit_s)
    E = LiveEpisode(task, ep, budget_steps=None, keep_frames=False, fast=fast, max_env_steps=max_steps)
    rec = {"task": task, "ep": ep, "latency_s": L, "fast": fast, "oracle": oracle}
    decisions = []
    blind_seen = False
    try:
        E.reset()
        rec["t0"] = E.t0
        rec["goal"] = E.task_goal
        U = E.env.unwrapped
        rec["live_level"] = getattr(U, "live_level", None)
        dl = int(math.ceil(L * CONTROL_HZ))
        for k in range(max_decisions):
            if hasattr(U, "rlm_on_observe"):
                U.rlm_on_observe(E.elapsed())      # InterruptSwap: the scene reacts to every observation
            now = E.elapsed()
            if oracle == "aware" and hasattr(U, "steps_press") and hasattr(U, "interval"):
                press_at = int(U.steps_press - U.interval)
                name = getattr(U, "current_task_name", "") or ""
                if name.startswith("remain static") or name.startswith("press"):
                    if now + 2 * dl < press_at:
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
                if U.rlm_safe_step(now, dl) > now:
                    nxt = U.rlm_safe_step(now + dl, dl)
                    w = max(1, nxt - now - dl)
                    E.hold(dl + w)
                    decisions.append({"sub": getattr(U, "current_task_name", None), "cmd": {"wait_steps": w},
                                      "t_dec": now - E.t0, "t_end": E.elapsed() - E.t0, "status": E.status})
                    if E.status != "ongoing":
                        break
                    continue
            if oracle == "blind" and hasattr(U, "rlm_cue_visible"):
                # frame-blind stale reference (round 2): knows the cue only if it is visible in the observation of one of
                # its own decisions (no frames recorded while it thinks or moves); waits in short chunks while the cue can
                # still come; if the cue is over and it never saw it, it cannot answer (counted as a failure; a random
                # guess would be right with probability 1 / number of candidates)
                if U.rlm_cue_visible(now):
                    blind_seen = True
                if U.rlm_needs_cue() and not blind_seen:
                    if U.rlm_cue_pending(now) or not getattr(U, "rlm_cues", None):
                        E.hold(dl + BLIND_WAIT)
                        decisions.append({"sub": getattr(U, "current_task_name", None), "cmd": {"wait_steps": BLIND_WAIT},
                                          "t_dec": now - E.t0, "t_end": E.elapsed() - E.t0, "status": E.status})
                        if E.status != "ongoing":
                            break
                        continue
                    rec["missed_cue"] = True
                    E.status = "missed_cue"
                    break
            if oracle in ("stale", "blind"):
                ra = stale_ready_at(E, U, now)
                if ra > now:
                    w = max(1, ra - now - dl)
                    E.hold(dl + w)
                    decisions.append({"sub": getattr(U, "current_task_name", None), "cmd": {"wait_steps": w},
                                      "t_dec": now - E.t0, "t_end": E.elapsed() - E.t0, "status": E.status})
                    if E.status != "ongoing":
                        break
                    continue
            cmd = privileged_oracle(E)
            sub = getattr(U, "current_task_name", None)
            if cmd is None:
                rec["note"] = f"no matching option for subgoal {sub!r}"
                E.status = "no_option"
                break
            r = E.act(cmd, L, sub)
            decisions.append({"sub": sub, "cmd": cmd, "t_dec": r.t_decide - E.t0, "t_exec": r.t_exec - E.t0,
                              "t_end": r.t_end - E.t0, "status": r.status,
                              "sub_exec": getattr(U, "current_task_name", None)})
            if E.status != "ongoing":
                break
        rec["status"] = E.status
        rec["steps"] = E.used()
        rec["success"] = E.status == "success"
        if E.status == "error":
            rec["error"] = E.info.get("error_message")
    except _Stuck:
        rec["status"] = "stuck"
        rec["success"] = False
        try:
            rec["steps"] = E.used()
        except Exception:
            pass
    except Exception:
        rec["status"] = "crash"
        rec["success"] = False
        rec["error"] = traceback.format_exc()[-600:]
    finally:
        signal.alarm(0)
        try:
            U = E.env.unwrapped
            rec["events"] = [list(map(lambda x: x if isinstance(x, str) else int(x), ev)) for ev in U.rlm_events()] \
                if hasattr(U, "rlm_events") else None
            if hasattr(U, "rlm_cancelled"):
                rec["cancelled"] = [list(map(int, c)) for c in U.rlm_cancelled]
            if hasattr(U, "rlm_dues"):
                rec["dues"] = [int(d) for d in U.rlm_dues]
                rec["fired"] = U.rlm_fired
            if hasattr(U, "rlm_kind"):
                rec["kind"] = U.rlm_kind
            if hasattr(U, "rlm_cues"):
                rec["cues"] = [[c[0], c[1], c[2]] for c in U.rlm_cues]
            if hasattr(U, "rlm_feints"):
                rec["feints"] = [[int(f[0]), int(f[1])] for f in U.rlm_feints]
        except Exception:
            pass
        E.close()
    rec["decisions"] = decisions
    rec["n_decisions"] = len(decisions)
    rec["wall_s"] = round(time.time() - t_wall, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--eps", default="0-19")
    ap.add_argument("--lat", default="0,3")
    ap.add_argument("--oracle", default="stale", help="comma list of naive,stale,aware,blind")
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-jobs", type=int, default=0)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    eps = parse_eps(a.eps)
    lats = [float(x) for x in a.lat.split(",")]
    oracles = a.oracle.split(",")
    si, sn = [int(x) for x in a.shard.split("/")]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["task"], r["ep"], r["latency_s"], r["oracle"]))
            except Exception:
                pass
    jobs = [(t, e, L, o) for t in tasks for e in eps for L in lats for o in oracles]
    jobs = [j for i, j in enumerate(jobs) if i % sn == si and j not in done]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"{len(jobs)} jobs", flush=True)
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    for (t, e, L, o) in jobs:
        r = run_one(t, e, L, a.max_steps, oracle=o)
        with open(a.out, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(f"{t} ep{e} L={L} {o}: {r['status']} steps={r.get('steps')} dec={r.get('n_decisions')} wall={r['wall_s']}s",
              flush=True)
    if not jobs:
        print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
