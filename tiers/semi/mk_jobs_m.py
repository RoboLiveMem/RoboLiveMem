"""Main-table jobs for part H (2026-11-15, standard protocol): strict, person model, 30 frames per decision, no hints,
hidden paused / hidden live / visible live, seeds by seed % 3 = event / look-alike / null.
env: FAMS, SEEDS ("a-b"), CONDS (full|hp|vl), TAG (default m15), SAVE_OBS (1 = save the images of every live episode).
usage: python mk_jobs_m.py MODEL[:MAXTOK][,MODEL[:MAXTOK]...] OUT.jsonl
(MODEL: a robolivemem.models spec such as anthropic:claude-sonnet-5-5, a paper alias, or scripted:<policy>)"""
import json, os, sys
FAMS = os.environ.get("FAMS", "mugs,knife,apple,point,heatw,heatdur,sugar2,heat").split(",")
a, b = map(int, os.environ.get("SEEDS", "0-35").split("-"))
CONDS = {"full": [("hidden", "paused"), ("hidden", "live"), ("visible", "live")],
         "hp": [("hidden", "paused"), ("hidden", "live")], "vl": [("visible", "live")]}[os.environ.get("CONDS", "full")]
TAG = os.environ.get("TAG", "m15")
OBS = os.environ.get("OBS_DIR", "obs")
J = []
for s in range(a, b + 1):
    for f in FAMS:
        for spec in sys.argv[1].split(","):
            head, _, tail = spec.rpartition(":")
            m, mt = (head, tail) if (head and tail.isdigit()) else (spec, "")
            for mem, clk in CONDS:
                j = dict(job_id=f"{TAG}_{m}_{f}_{mem}_{clk}_f30_s{s}", family=f, member=mem, seed=s, clock=clk, L=0.0,
                         policy=m, diag="none", frames="buffer", tag=TAG, protocol="strict", person=True, max_frames=30,
                         harness="minimal", priv_log=True)
                if mt:
                    j["max_tokens"] = int(mt)
                if clk == "live" and os.environ.get("SAVE_OBS", "1") == "1" and not m.startswith("scripted"):
                    j["save_obs"] = f"{OBS}/{TAG}"
                J.append(j)
open(sys.argv[2], "w").write("\n".join(json.dumps(j) for j in J) + "\n")
print(len(J), "jobs ->", sys.argv[2])
