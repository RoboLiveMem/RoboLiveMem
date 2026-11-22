"""Controlled tier (ManiSkill, RoboMME scenes) as an environment server (robolivemem.protocol).

    RLM_STRICT_TASKS=1 python server.py --port 8802

Each /reset plays one episode in a fresh worker process (RoboMME task state can carry over between episodes of one
process), with the paper's settings: up to 30 new frames and 30 demonstration frames per decision, frames tiled into at
most 24 images, 25 decisions, a 1300-step budget, and the protocol options fullgoal, noevent, cleanframe, twinnote
(+ commit for the commit check). At every decision the agent receives the new user turn (obs["content"]) and the
dialogue the paper's models received (obs["dialogue"]); it answers with one command:
{"choice": "<a listed command label>" | "wait", "mark": n, "wait": s}.

Reference policy for validation: /reset with "policy": "oracle" (privileged: reads the simulator and answers at once;
"think": <s> charges it a thinking time, which makes it stale).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]

from robolivemem.benchmark import TIERS  # noqa: E402
from robolivemem.protocol import Aborted, Bridge, CONDITIONS, VARIANTS, parse_clock, to_content  # noqa: E402

HINTS = ("fullgoal", "noevent", "cleanframe", "twinnote")
SETTINGS = dict(budget_steps=1300, max_env_steps=6000, max_images=62, max_new_frames=30, max_demo_frames=30,
                max_decisions=25)
TILE = 24


class RemoteClient:
    """The agent episode's model client, answered by the remote agent through the parent process."""
    short = "remote"

    def __init__(self, conn):
        self.conn = conn

    def __call__(self, system, turns):
        user = [p for r, p in turns if r == "user"][-1]
        self.conn.send({"type": "obs", "obs": {"system": system, "content": to_content(user),
                                               "dialogue": [[r, to_content(p)] for r, p in turns]}})
        cmd, text, lat, src = self.conn.recv()
        return (text or json.dumps(cmd or {})), {"latency_s": lat, "latency_source": src}


def worker(job, conn):
    os.environ.setdefault("RLM_STRICT_TASKS", "1")
    try:
        from rlm_controlled.agent import AgentEpisode, OracleClient
        from rlm_controlled.tiled import TiledClient
        member, _ = CONDITIONS[job.get("condition", "L")]
        clock, L = parse_clock("paused" if job.get("condition") == "P" else job.get("clock", "live"))
        task = job["task"] + ("WV" if member == "visible" else "W")
        hints = HINTS + (("commit",) if job.get("commit") else ())
        mode = {"paused": "paused", "live": "measured", "fixed": "fixed"}[clock]
        A = AgentEpisode(task, int(job["seed"]), None, mode=mode, fixed_latency=L, hints=hints, **SETTINGS)
        if job.get("policy"):
            A.client = OracleClient(A, float(job.get("think", 0.0)) if clock == "live" else 0.0)
        else:
            A.client = TiledClient(RemoteClient(conn), max_images=TILE)
        r = A.run()
        kind = getattr(A.E.env.unwrapped, "rlm_kind", None)
        A.E.close()
        conn.send({"type": "result", "result": {
            "success": bool(r.get("success")), "status": r.get("status"), "variant": VARIANTS.get(kind, kind),
            "steps": r.get("steps"), "decisions": r.get("n_calls"), "world_s": round((r.get("steps") or 0) / 20.0, 2),
            "commit_aborts": sum(1 for e in r.get("log", []) if e.get("commit_abort"))}})
    except Exception as e:  # noqa: BLE001
        conn.send({"type": "result", "result": {"success": False, "status": "crash",
                                                "error": f"{type(e).__name__}: {e}"[:400],
                                                "traceback": traceback.format_exc()[-2000:]}})


def runner(job: dict, ask) -> dict:
    if job["task"] not in TIERS["controlled"]["tasks"]:
        raise ValueError(f"unknown task {job['task']}")
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=worker, args=(job, child), daemon=True)
    p.start()
    try:
        while True:
            if not parent.poll(3600):
                raise RuntimeError("worker silent for an hour")
            msg = parent.recv()
            if msg["type"] == "obs":
                parent.send(ask(msg["obs"]))
            else:
                return msg["result"]
    except (Aborted, EOFError):
        p.terminate()
        raise
    finally:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()


INFO = {"tier": "controlled", "tasks": list(TIERS["controlled"]["tasks"]), "conditions": ["P", "L", "V"],
        "clocks": ["paused", "live", "fixed:<s>"], "commit": True, "policies": ["oracle"],
        "commands": ['{"choice": "<label>", "mark": n}', '{"choice": "wait", "wait": s}']}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8802)
    a = ap.parse_args()
    os.environ.setdefault("RLM_STRICT_TASKS", "1")
    Bridge(runner, INFO).run(a.host, a.port)


if __name__ == "__main__":
    main()
