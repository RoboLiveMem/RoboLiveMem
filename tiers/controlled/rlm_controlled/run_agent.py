"""Run the agent track: model x task x episode x mode (paused / live).

usage:
  python -m rlm_controlled.run_agent --model anthropic:claude-sonnet-5-5 --tasks SwapStreamW --eps 0-2 --modes paused,live \
      --out results/agent.jsonl --video-dir videos
  python -m rlm_controlled.run_agent --model oracle-sim --sim-latency 4 ...   # plumbing test without an API
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlm_controlled.agent import AgentEpisode, OracleClient  # noqa: E402
from rlm_controlled import tasks_live  # noqa: E402,F401
from rlm_controlled.run_oracle import ALL_TASKS, parse_eps  # noqa: E402


class _Retry:
    """2026-11-15: retry a request the service refused (throttling, 5xx, connection error); only the answered attempt's
    latency is returned (and charged), as in part H's harness."""

    def __init__(self, inner):
        self.inner = inner
        self.short = getattr(inner, "short", "api")

    def __call__(self, system, turns):
        import random as _r
        for k in range(4):
            text, meta = self.inner(system, turns)
            err = str(meta.get("error") or "").lower()
            refused = meta.get("error") and (meta.get("http_status") in (429, 500, 502, 503, 504, None) or "throttl" in err)
            if not refused or k == 3:
                meta["tries"] = k + 1
                return text, meta
            time.sleep(3 * (k + 1) + _r.random() * 3)
        return text, meta


def make_client(name, max_tokens, tile=0, retry=False):
    if name == "oracle-sim":
        return None
    if name == "echo-sim":   # plumbing test: records what would be sent, always waits 1 s
        from rlm_controlled.tiled import TiledClient

        class _Echo:
            short = "echo-sim"

            def __call__(self, system, turns):
                n = sum(1 for _, ps in turns for p in ps if not isinstance(p, str))
                return '{"choice": "wait", "wait": 1, "note": "echo"}', {"latency_s": 0.0, "images_in_request": n}
        return TiledClient(_Echo(), max_images=tile) if tile else _Echo()
    from robolivemem.models import Chat as BedrockChat
    c = BedrockChat(name, max_tokens=max_tokens)
    if retry:
        c = _Retry(c)
    if tile:
        from rlm_controlled.tiled import TiledClient
        c = TiledClient(c, max_images=tile)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--eps", default="0-2")
    ap.add_argument("--modes", default="paused,live")
    ap.add_argument("--fixed-latency", type=float, default=0.0, help="for mode 'fixed'")
    ap.add_argument("--sim-latency", type=float, default=0.0, help="oracle-sim only")
    ap.add_argument("--budget", type=int, default=1300)
    ap.add_argument("--max-env-steps", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-images", type=int, default=24, help="images per request (provider limit)")
    ap.add_argument("--max-new-frames", type=int, default=10)
    ap.add_argument("--max-demo-frames", type=int, default=12)
    ap.add_argument("--max-decisions", type=int, default=25)
    ap.add_argument("--hints", default="", help="comma list of protocol options, e.g. the final protocol "
                    "frame,timing,demotail,cleanframe,demodense,fullgoal,framedense,evwindow")
    ap.add_argument("--split", default=None, help="JSON file with 'ids' (rlm-<fam>-<Task>-<ep>); replaces --tasks/--eps")
    ap.add_argument("--tile", type=int, default=0, help="tile frames as the file interface does; max images per request")
    ap.add_argument("--single", default=None, help="internal: run one TASK:EP:MODE in this process")
    ap.add_argument("--save-obs", default=None, help="2026-11-15: directory; save the images each live decision had "
                    "(npz per episode) for the offline probe")
    ap.add_argument("--retry", action="store_true", help="2026-11-15: retry throttled / 5xx / failed requests (up to 3 "
                    "times); a refused request is not charged to the clock")
    a = ap.parse_args()
    hints = tuple(h for h in a.hints.split(",") if h)
    if a.single:
        t, e, m = a.single.split(":")
        return run_one(a, t, int(e), m, hints)
    si, sn = [int(x) for x in a.shard.split("/")]
    done = set()
    import glob as _glob
    import re as _re
    # every shard file of this tag counts as done, so a tag can be relaunched with a different number of shards
    for _f in sorted(set(_glob.glob(_re.sub(r"_s\d+\.jsonl$", "_s*.jsonl", a.out))) | {a.out}):
        if not os.path.exists(_f):
            continue
        for line in open(_f):
            try:
                r = json.loads(line)
                done.add((r["task"], r["ep"], r["mode"]))
            except Exception:
                pass
    if a.split:
        ids = json.load(open(a.split))["ids"]
        pairs = [(i.split("-")[2], int(i.split("-")[3])) for i in ids]
        jobs = [(t, e, m) for t, e in pairs for m in a.modes.split(",")]
    else:
        jobs = [(t, e, m) for t in a.tasks.split(",") for e in parse_eps(a.eps) for m in a.modes.split(",")]
    jobs = [j for i, j in enumerate(jobs) if i % sn == si and j not in done]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.video_dir:
        os.makedirs(a.video_dir, exist_ok=True)
    print(f"{len(jobs)} jobs", flush=True)
    import subprocess
    base = [a for a in sys.argv[1:]]
    for (t, e, m) in jobs:
        # every episode in a fresh process: some RoboMME task state carries over between episodes of one process
        subprocess.run([sys.executable, "-m", "rlm_controlled.run_agent"] + base + ["--single", f"{t}:{e}:{m}"],
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_one(a, t, e, m, hints):
    client = make_client(a.model, a.max_tokens, a.tile, retry=a.retry)
    # modes: paused | live (measured latency) | fixed<L> (e.g. fixed4: every decision charged 4 s)
    fl = a.fixed_latency
    if m.startswith("fixed") and len(m) > 5:
        fl = float(m[5:])
    mode = "measured" if m == "live" else ("fixed" if m.startswith("fixed") else m)
    A = AgentEpisode(t, e, client, mode=mode, fixed_latency=fl, budget_steps=a.budget,
                     max_env_steps=a.max_env_steps, max_images=a.max_images, max_new_frames=a.max_new_frames,
                     max_demo_frames=a.max_demo_frames, max_decisions=a.max_decisions, hints=hints)
    if a.model == "oracle-sim":
        A.client = OracleClient(A, a.sim_latency)
    if a.save_obs and m != "paused":
        A.save_obs_path = os.path.join(a.save_obs, f"{a.model}_{t}_ep{e}_{m}.npz")
    try:
        r = A.run()
    except Exception:
        r = {"task": t, "ep": e, "mode": m, "model": a.model, "status": "crash", "success": False,
             "error": traceback.format_exc()[-1500:], "log": A.log}
    r["mode"] = m
    r["model"] = a.model
    r["hints"] = list(hints)
    r["max_chunk"] = 1
    r["tile"] = a.tile
    r["host"] = os.uname().nodename
    if a.video_dir:
        try:
            A.save_video(os.path.join(a.video_dir, f"{a.model}_{t}_ep{e}_{m}.mp4"))
        except Exception as exc:
            r["video_error"] = repr(exc)[:200]
    if getattr(A, "save_obs_path", None):
        try:
            os.makedirs(os.path.dirname(A.save_obs_path), exist_ok=True)
            r["obs_file"] = A.save_obs(A.save_obs_path)
        except Exception as exc:
            r["obs_error"] = repr(exc)[:200]
    A.E.close()
    with open(a.out, "a") as f:
        f.write(json.dumps(r, default=str) + "\n")
    print(f"{a.model} {t} ep{e} {m}: {r['status']} steps={r.get('steps')} calls={r.get('n_calls')} "
          f"lat={r.get('sum_latency_s')} wall={r.get('wall_s')}", flush=True)

if __name__ == "__main__":
    main()
