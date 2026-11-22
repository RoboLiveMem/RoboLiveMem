"""Part H (semi-realistic tier) -> one compact row per episode (2026-11-15 main tables).
usage: python rows_h.py OUT.jsonl TAG[,TAG...]   (reads main_1115/results/TAG_s*.jsonl)"""
import os
import glob, json, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attr_h import classify
EFFORT = {"gpt-6-astra": "low", "gpt-6-astra-med": "medium", "gpt-6-luna": "low"}
rows = {}
for tag in sys.argv[2].split(","):
    for f in sorted(glob.glob(os.path.join(os.environ.get("RESULTS_DIR", "results"), f"{tag}_s*.jsonl"))):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("success") is None:
                continue
            rows[r["job_id"]] = r
out = open(sys.argv[1], "w")
for jid, r in sorted(rows.items()):
    s = r.get("strict") or {}
    tr = r.get("trace") or []
    lat = [float((t.get("meta") or {}).get("latency_s") or 0) for t in tr]
    cls, i, end = classify(r)
    row = {"tier": "semi", "part": "H", "family": r["family"], "task": r["family"], "seed": r["seed"], "ep": None,
           "variant": s.get("kind"), "model": r["policy"], "effort": EFFORT.get(r["policy"]),
           "max_tokens": r.get("max_tokens"), "clock": r["clock"], "condition": r["member"], "harness": "minimal",
           "round": "r4", "success": bool(r["success"]), "reason": r.get("end"), "outcome": r.get("outcome"),
           "changed": r.get("changed"), "change_t": r.get("change_t"), "change_end_t": end, "person": s.get("person"),
           "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
           "latency_median_s": round(statistics.median(lat), 2) if lat else None, "n_calls": len(lat),
           "latency_sum_s": round(sum(lat), 1), "attribution": cls,
           "decisive": None if i is None else {"k": tr[i].get("k"), "t_obs_s": tr[i].get("t"), "think_s": tr[i].get("think_s"),
                                               "cmd": tr[i].get("cmd"), "obj": tr[i].get("obj"), "priv": tr[i].get("priv")},
           "n_empty": sum(1 for t in tr if not t.get("reply")), "n_maxtok": sum(1 for t in tr if (t.get("meta") or {}).get("stop") in ("max_tokens", "incomplete")),
           "obs_file": r.get("obs_file"), "source": r["job_id"]}
    out.write(json.dumps(row) + "\n")
print(len(rows), "rows")
