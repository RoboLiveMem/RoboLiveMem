"""Semi-realistic tier (MuJoCo kitchen) as an environment server (robolivemem.protocol).

    python server.py --port 8801            # inside the venv made by install_sim.sh; MUJOCO_GL=egl or osmesa

Each /reset plays one episode of the paper's protocol (strict families, a person in the kitchen, up to 30 frames per
decision at 2 Hz, 12 decisions, 240 s of world time). At every decision the agent receives exactly what the paper's
models received (obs["content"]; obs["system"] holds the instructions and the reply format) and answers with one
command: {"tool": "pick" | "place" | "press" | "wait" | "done", "mark": n, "seconds": s, "note": "..."}.

Reference policies for validation (no agent needed): /reset with "policy": "scripted:aware" (privileged, waits for the
person), "scripted:naive" (stale plan) or "scripted:always" (assumes a change), "think" = their charged seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]

import run_lm2 as R  # noqa: E402
from robolivemem.benchmark import TIERS  # noqa: E402
from robolivemem.protocol import Bridge, CONDITIONS, parse_clock, to_content  # noqa: E402

R.PROTOCOL_DEFAULT = "strict"


class RemotePolicy:
    """run_lm2's policy interface (decide(parts, ctx) -> (cmd, text, meta)) answered by a remote agent."""

    def __init__(self, ask):
        self.ask, self.hint = ask, ""

    def decide(self, parts, ctx):
        content = to_content(parts)
        cmd, text, lat, src = self.ask({"system": R.SYSTEM, "content": content, "dialogue": [["user", content]],
                                        "world_time": round(float(ctx["ep"].world.time), 2)})
        if (not isinstance(cmd, dict) or not cmd) and text:
            from robolivemem.models import parse_json_reply
            cmd = parse_json_reply(text)
        return cmd, text or json.dumps(cmd), {"latency_s": lat, "latency_source": src}


def runner(job: dict, ask) -> dict:
    fam = job["task"]
    if fam not in TIERS["semi"]["tasks"]:
        raise ValueError(f"unknown family {fam}")
    if job.get("commit"):
        raise ValueError("the commit check is not part of the semi-realistic tier")
    member, _ = CONDITIONS[job.get("condition", "L")]
    clock, L = parse_clock("paused" if job.get("condition") == "P" else job.get("clock", "live"))
    seed = int(job["seed"])
    j = dict(job_id=f"remote_{fam}_{member}_{clock}_s{seed}", family=fam, member=member, seed=seed, clock=clock, L=L,
             policy=job.get("policy") or "remote", diag="none", frames="buffer", protocol="strict", person=True,
             max_frames=30, harness="minimal", tag="remote")
    pol = job.get("policy")
    if pol:
        policy = R.StrictScriptedPolicy(pol.split(":", 1)[1] if ":" in pol else pol)
        if clock == "live":          # scripted policies answer instantly: charge their nominal thinking time
            j["clock"], j["L"] = "fixed", float(job.get("think", 3.0))
    else:
        policy = RemotePolicy(ask)
    res = R.run_episode(j, policy, max_turns=12, max_frames=30)
    info = res.get("strict") or {}
    return {"success": bool(res.get("success")), "status": res.get("end"), "variant": info.get("kind"),
            "world_s": res.get("world_s"), "turns": res.get("turns"), "changed": res.get("changed"),
            "change_t": res.get("change_t"), "outcome": res.get("outcome")}


INFO = {"tier": "semi", "tasks": list(TIERS["semi"]["tasks"]), "conditions": ["P", "L", "V"],
        "clocks": ["paused", "live", "fixed:<s>"], "commit": False,
        "policies": ["scripted:aware", "scripted:naive", "scripted:always"],
        "commands": ['{"tool": "pick", "mark": n}', '{"tool": "place", "mark": n}', '{"tool": "press", "mark": n}',
                     '{"tool": "wait", "seconds": s}', '{"tool": "done"}']}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8801)
    a = ap.parse_args()
    Bridge(runner, INFO).run(a.host, a.port)


if __name__ == "__main__":
    main()
