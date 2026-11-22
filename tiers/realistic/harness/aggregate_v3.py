"""Aggregate v3 model runs: success per scenario x model x condition (hidden event / decoy / null, visible).

    python aggregate_v3.py <eval root> [<eval root> ...]  -> prints tables, writes <first root>/v3_results.json
"""
import json, sys, collections
from pathlib import Path

rows = []
for root in map(Path, sys.argv[1:]):
    s = root / "summary.jsonl"
    if not s.exists():
        continue
    for line in s.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("valid"):
            continue
        tag = {"claude-haiku-5-5": "haiku", "claude-sonnet-5-5": "sonnet", "gpt-6-luna": "luna", "gpt-6.1-sol": "sol"}.get(r["model"], "")
        name = f"{r['backend']}{'_' + tag if tag else ''}{'_' + r['variant'] if r.get('variant') else ''}_{r['scenario']}_{r['clock']}_s{r['seed']}_a{r['attempt']}"
        res = root / "episodes" / name / "result.json"
        try:
            fin = json.loads(res.read_text())
        except Exception:
            fin = {}
        acts = [a for a in fin.get("actions", []) if a.get("skill") not in ("wait",)] if fin.get("scenario") == r["scenario"] else []
        rows.append(dict(scenario=r["scenario"], seed=r["seed"], model=r["model"], cond={"visible": "visible", "hard": "hard", "nostream": "nostream"}.get(r.get("variant") or "", "hidden"),
                         variant=("event", "decoy", "null")[r["seed"] % 3], success=bool(r.get("success")), reason=r.get("reason"), end=r.get("end"),
                         world_t=fin.get("world_t"), lat=r.get("latencies", []), n_frames_calls=r.get("n_frames", None),
                         actions=[(a.get("skill"), a.get("target")) for a in acts], name=name, root=str(root)))
# keep one row per (scenario, seed, model, cond): the last valid one
uniq = {}
for r in rows:
    uniq[(r["scenario"], r["seed"], r["model"], r["cond"])] = r
rows = list(uniq.values())
models = sorted({r["model"] for r in rows})
sids = sorted({r["scenario"] for r in rows})
tab = collections.defaultdict(lambda: [0, 0])
for r in rows:
    for key in [(r["scenario"], r["model"], r["cond"], r["variant"]), (r["scenario"], r["model"], r["cond"], "all"),
                ("ALL", r["model"], r["cond"], r["variant"]), ("ALL", r["model"], r["cond"], "all")]:
        tab[key][0] += r["success"]; tab[key][1] += 1


def cell(k):
    s, n = tab.get(k, [0, 0])
    return f"{s}/{n}" if n else "-"


for cond in ("hidden", "hard", "nostream"):
    print(f"\n[{cond}]")
    print("| scenario | model | event | decoy | null | all | visible |")
    print("|---|---|---|---|---|---|---|")
    for sid in sids + ["ALL"]:
        for m in models:
            if not tab.get((sid, m, cond, "all")):
                continue
            print(f"| {sid} | {m} | {cell((sid, m, cond, 'event'))} | {cell((sid, m, cond, 'decoy'))} | {cell((sid, m, cond, 'null'))} | "
                  f"{cell((sid, m, cond, 'all'))} | {cell((sid, m, 'visible', 'all')) if cond == 'hidden' else ''} |")
print()
for r in sorted(rows, key=lambda r: (r["scenario"], r["model"], r["cond"], r["seed"])):
    if not r["success"]:
        print(f"FAIL {r['scenario']} s{r['seed']} {r['model']} {r['cond']} {r['variant']}: {r['reason']} | {r['actions']} | lat {r['lat'][:6]}")
out = Path(sys.argv[1]) / "v3_results.json"
out.write_text(json.dumps(dict(rows=rows, table={"|".join(map(str, k)): v for k, v in tab.items()}), indent=1, default=str))
