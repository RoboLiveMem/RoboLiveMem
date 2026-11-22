#!/usr/bin/env python3
"""Aggregate RoboLiveMem-Bench results: scripted baselines and model runs -> markdown table + json.

    python3 aggregate.py <root>   (root = your working folder)
"""
import json, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
rows = []; excluded = []
for f in sorted((root / "evals").glob("models_*/summary.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("valid"):
            # harness fault: the Codex MCP client gave up on a slow skill call (300 s default; raised to 3600 s at 09:00)
            vb = r["backend"] + ("_" + r["variant"] if r.get("variant") else "")
            ep = f.parent / "episodes" / f"{vb}_{r['scenario']}_{r['clock']}_s{r['seed']}_a{r['attempt']}" / "agent.jsonl"
            if ep.exists() and "timed out awaiting tools/call" in ep.read_text():
                excluded.append(r); continue
            if r.get("variant") and r["variant"] != "rerun":      # "rerun": same condition, re-run after a harness fault
                r["model"] = f"{r['model']} ({r['variant']})"
            rows.append(r)
# one row per job (a job that was re-run keeps its last valid result)
_last = {}
for r in rows:
    _last[(r["backend"], r["model"], r["scenario"], r["seed"], r["clock"])] = r
rows = list(_last.values())
scripted = {}
for f in sorted((root / "evals" / "scripted").glob("*_stale.json")):
    try:
        r = json.loads(f.read_text())
        scripted[(r["scenario"], r["clock"])] = r
    except Exception:
        pass
sids = sorted({r["scenario"] for r in rows} | {k[0] for k in scripted})
models = sorted({r["model"] for r in rows})
cell = defaultdict(list)
for r in rows:
    cell[(r["scenario"], r["model"], r["clock"])].append(r)
lines = ["| Scenario | stale policy paused | stale policy live | " + " | ".join(f"{m} paused | {m} live" for m in models) + " |",
         "|---" * (3 + 2 * len(models)) + "|"]
tot = defaultdict(lambda: [0, 0])
for sid in sids:
    c = [sid]
    for clock in ("paused", "live"):
        s = scripted.get((sid, clock))
        c.append("-" if s is None else ("✓" if s["success"] else "✗"))
    for m in models:
        for clock in ("paused", "live"):
            rs = cell[(sid, m, clock)]
            if not rs:
                c.append("-"); continue
            k = sum(1 for r in rs if r["success"]); n = len(rs)
            tot[(m, clock)][0] += k; tot[(m, clock)][1] += n
            c.append(f"{k}/{n}")
    lines.append("| " + " | ".join(c) + " |")
c = ["**all**", "", ""]
for m in models:
    for clock in ("paused", "live"):
        k, n = tot[(m, clock)]
        c.append(f"**{k}/{n}**" + (f" ({100 * k / n:.0f}%)" if n else ""))
lines.append("| " + " | ".join(c) + " |")
detail = ["", "| model | scenario | clock | seed | success | reason | agent s | world s | decisions | latencies (s) |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["model"], r["scenario"], r["clock"], r["seed"])):
    lat = r.get("latencies") or []
    detail.append(f"| {r['model']} | {r['scenario']} | {r['clock']} | {r['seed']} | {'✓' if r['success'] else '✗'} | {r['reason']} | "
                  f"{r.get('agent_t')} | {r.get('world_t')} | {len(lat)} | {', '.join(str(x) for x in lat[:8])} |")
# behaviour per model and clock: decision latency, interlock refusals, frame look-ups, waits
import statistics as st
beh = ["", "| model | clock | episodes | median decision latency (s) | refused commands / ep | robot_frames / ep | robot_wait / ep | world time / ep (s) |",
       "|---|---|---|---|---|---|---|---|"]
for m in models:
    for clock in ("paused", "live"):
        rs = [r for r in rows if r["model"] == m and r["clock"] == clock]
        if not rs:
            continue
        lat = [x for r in rs for x in (r.get("latencies") or [])]
        nref = nfr = nwait = 0
        for r in rs:
            vb = r["backend"] + ("_" + r["variant"] if r.get("variant") else "")
            for d in (root / "evals").glob(f"models_*/episodes/{vb}_{r['scenario']}_{r['clock']}_s{r['seed']}_a{r['attempt']}"):
                try:
                    res = json.loads((d / "result.json").read_text())
                    nref += sum(1 for a in res.get("actions", []) if a.get("refused"))
                    for l in (d / "tools.jsonl").read_text().splitlines():
                        t = json.loads(l).get("tool")
                        nfr += t == "robot_frames"; nwait += t == "robot_wait"
                except Exception:
                    pass
        n = len(rs)
        beh.append(f"| {m} | {clock} | {n} | {st.median(lat):.1f} | {nref / n:.1f} | {nfr / n:.1f} | {nwait / n:.1f} | "
                   f"{st.mean([r.get('world_t') or 0 for r in rs]):.0f} |" if lat else f"| {m} | {clock} | {n} | - | - | - | - | - |")
exc = ["", f"Excluded (harness fault, Codex MCP tool-call timeout): " + (", ".join(f"{r['model']} {r['scenario']} {r['clock']} s{r['seed']}" for r in excluded) or "none")]
out = "\n".join(lines + beh + exc + detail)
(root / "evals" / "RESULTS.md").write_text(out + "\n")
(root / "evals" / "results.json").write_text(json.dumps(dict(rows=rows, scripted={f"{k[0]}|{k[1]}": v for k, v in scripted.items()}), indent=1))
print(out)
