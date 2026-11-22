"""Write the v3 model-evaluation queue for one scene: hidden seeds 0-2, then visible seeds 0-1, then hidden seeds 3-5."""
import json, sys
scene, out = sys.argv[1], sys.argv[2]
SIDS = {"house": ["v3_sugar", "v3_mugs", "v3_keys"], "hall": ["v3_totes", "v3_bins", "v3_reagent"]}[scene]
MODELS = [dict(backend="claude", model="claude-opus-5-5"), dict(backend="codex", model="gpt-6-astra", effort="medium")]
rows = []
for seeds, variant in (([0, 1, 2], ""), ([0, 1], "visible"), ([3, 4, 5], "")):
    for seed in seeds:
        for sid in SIDS:
            for m in MODELS:
                rows.append(dict(scenario=sid, seed=seed, clock="live", **m, **({"variant": variant} if variant else {})))
open(out, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
print(len(rows), "jobs ->", out)
