"""Run an agent on the benchmark and report the results.

    python -m robolivemem.eval --tier semi --env http://127.0.0.1:8801 --agent chat:anthropic:claude-haiku-5-5 \
        --out results/semi_haiku.jsonl
    python -m robolivemem.eval --tier realistic --env house_v3=http://127.0.0.1:8701,hall_v3=http://127.0.0.1:8702 ...
    python -m robolivemem.eval --tier tabletop --env http://127.0.0.1:8800 --agent my_agent.py:MyAgent --seeds 0-5

Every episode becomes one JSON row in --out (runs resume: finished episodes are skipped). At the end the per-task
table and the tier summary are printed and written next to --out (.md and .json); robolivemem.report combines files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import benchmark as B
from .agents import Agent, CommandAgent, ServerPolicy, load_agent
from .env import connect, normalize_result


def _key(j):
    return (j["task"], int(j["seed"]), j["condition"], j["clock"], bool(j.get("commit")))


def run_job(env, agent: Agent, job: dict, max_steps: int = 200) -> dict:
    t0 = time.time()
    body = {k: job[k] for k in ("task", "seed", "condition", "clock", "commit")}
    if isinstance(agent, ServerPolicy):
        body["policy"] = agent.policy
    if isinstance(agent, CommandAgent):
        body["defer_clock"] = True
    texts = []
    try:
        obs = env.reset(body)
        if isinstance(agent, CommandAgent):
            agent.run_episode(env.tier, getattr(env, "url", ""))
            obs = env.observe()
            if not obs.get("done"):
                obs = env.end()
        else:
            if not obs.get("done") and not isinstance(agent, ServerPolicy):
                agent.reset(obs)
            for _ in range(max_steps):
                if obs.get("done"):
                    break
                out = agent.act(obs)
                cmd, text, lat = (out if isinstance(out, tuple) else (out, None, None))
                if text:
                    texts.append(text[-400:])
                obs = env.act(cmd if isinstance(cmd, dict) else {}, latency=lat, text=text)
                if "error" in obs and not obs.get("done"):
                    raise RuntimeError(obs["error"])
            if not obs.get("done"):
                obs = env.end()
        res = normalize_result(obs.get("result") or {})
    except Exception as e:  # noqa: BLE001
        res = {"success": None, "status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
    keep = ("success", "status", "variant", "world_s", "decisions", "mean_latency_s", "latency_source", "error",
            "reason", "commit_aborts", "stale")
    row = dict(job, agent=agent.name, **{k: res.get(k) for k in keep if k in res})
    row["valid"] = row.get("success") is not None and row.get("status") not in ("crash", "error", "aborted")
    row["wall_s"] = round(time.time() - t0, 1)
    if texts:
        row["last_reply"] = texts[-1]
    return row


def parse_seeds(s: str | None):
    if not s:
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def run_tier(tier, env_url, agent, out: Path, tasks=None, seeds=None, conditions=("P", "L", "V"), clock="live",
             commit=False, limit=0):
    """Run the paper's episodes of one tier (resuming), appending one row per episode to `out`."""
    env = connect(tier, env_url)
    if tier == "realistic" and not tasks:
        tasks = [t for t, i in B.TIERS["realistic"]["tasks"].items() if i["scene"] in env.scenes()]
    jobs = B.jobs(tier, tasks, seeds, tuple(conditions), None if clock == "live" else clock, commit)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("valid") and r.get("agent") == agent.name:
                done.add(_key(r))
    todo = [j for j in jobs if _key(j) not in done]
    if limit:
        todo = todo[:limit]
    print(f"{tier}: {len(jobs)} episodes in the protocol, {len(jobs) - len([j for j in jobs if _key(j) not in done])} "
          f"done, running {len(todo)}", flush=True)
    for i, job in enumerate(todo, 1):
        row = run_job(env, agent, job)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        status = "success" if row.get("success") else (
            "fail" if row.get("valid") else "INVALID " + str(row.get("error") or row.get("status")))
        print(f"[{i}/{len(todo)}] {job['task']} seed {job['seed']} {job['condition']}: {status} "
              f"({row.get('variant')}, {row['wall_s']} s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate an agent on RoboLiveMem (one tier, or all tiers).")
    ap.add_argument("--tier", required=True, choices=list(B.TIERS) + ["all"])
    ap.add_argument("--env", help="server URL (realistic: scene=url,scene=url,...)")
    ap.add_argument("--envs", help="--tier all: JSON file {\"semi\": url, \"controlled\": url, "
                                   "\"realistic\": {\"house_v3\": url, ...}, \"tabletop\": url}")
    ap.add_argument("--agent", required=True, help="chat:<model>, server:<policy>, file.py:Class or cmd:<command>")
    ap.add_argument("--out", required=True, help="episode rows (jsonl; --tier all: one file per tier with this stem); "
                                                 "the report is written next to it")
    ap.add_argument("--tasks", help="comma-separated subset of the tier's tasks")
    ap.add_argument("--seeds", help="e.g. 0-11 (default: the paper's seeds)")
    ap.add_argument("--conditions", default="P,L,V", help="P (hidden, paused), L (hidden, live), V (visible twin, live)")
    ap.add_argument("--clock", default="live", help="clock of L and V: live or fixed:<seconds>")
    ap.add_argument("--commit", action="store_true", help="enable the commit check (realistic, controlled, tabletop)")
    ap.add_argument("--name", help="name of this agent in the report (default: the --agent spec)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--keep-images", type=int, default=12)
    ap.add_argument("--wall-clock", action="store_true",
                    help="chat agents: let the server charge the full wall time instead of declaring the model time")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many episodes per tier (smoke tests)")
    a = ap.parse_args(argv)

    kw = dict(max_tokens=a.max_tokens, effort=a.effort, keep_images=a.keep_images, declare=not a.wall_clock) \
        if a.agent.startswith("chat:") else {}
    agent = load_agent(a.agent, **kw)
    if a.name:
        agent.name = a.name
    out = Path(a.out)
    common = dict(tasks=a.tasks.split(",") if a.tasks else None, seeds=parse_seeds(a.seeds),
                  conditions=a.conditions.split(","), clock=a.clock, commit=a.commit, limit=a.limit)
    if a.tier == "all":
        if not a.envs:
            ap.error("--tier all needs --envs")
        envs = json.loads(Path(a.envs).read_text())
        files = []
        for tier in [t for t in ("realistic", "semi", "controlled", "tabletop") if t in envs]:
            url = envs[tier]
            if isinstance(url, dict):
                url = ",".join(f"{k}={v}" for k, v in url.items())
            f = out.with_name(f"{out.stem}_{tier}.jsonl")
            run_tier(tier, url, agent, f, **dict(common, tasks=None))
            files.append(f)
        stem = out.with_name(f"{out.stem}_report")
    else:
        if not a.env:
            ap.error("--env is required")
        run_tier(a.tier, a.env, agent, out, **common)
        files, stem = [out], out.with_suffix("")
    agent.close()
    from .report import write_report
    print()
    print(write_report(files, stem, agents=[agent.name]))


if __name__ == "__main__":
    sys.exit(main())
