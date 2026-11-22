"""Jobs for the strict protocol of part H (2026-11-06; see livemem2_strict.py). Presets:
  check          scripted oracle / naive / aware (+ always on mugs, heat, apple), hidden seeds 0-5, paused and fixed L=3;
                 plus visible seeds 0-5 at fixed L=3 (oracle, naive, aware)
  models MODEL   hidden seeds 0-5 paused + live, visible seeds 0-5 live (5 families)
All jobs: protocol strict, person true (ignored for point).
usage: python mk_jobs_strict.py check jobs_strict_check.jsonl | python mk_jobs_strict.py models opus-5.5 jobs_x.jsonl
"""
import json
import os
import sys

FAMS = ["mugs", "knife", "heat", "point", "apple"]
if os.environ.get("FAMS"):
    FAMS = os.environ["FAMS"].split(",")
SEEDS = range(*map(int, os.environ.get("SEEDR", "0,6").split(",")))


def job(fam, mem, seed, clock, pol, L=0.0, tag="strict"):
    short = pol.replace("scripted:", "")
    jid = f"{tag}_{short}_{fam}_{mem}_{clock}{'' if clock != 'fixed' else int(L)}_s{seed}"
    return dict(job_id=jid, family=fam, member=mem, seed=seed, clock=clock, L=L, policy=pol, diag="none", frames="buffer",
                tag=tag, protocol="strict", person=True)


def main():
    preset, out = sys.argv[1], sys.argv[-1]
    J = []
    if preset == "check":
        for s in SEEDS:
            for f in FAMS:
                pols = ["oracle", "naive", "aware"] + (["always"] if f in ("mugs", "heat", "apple", "heatw", "heatdur", "sugar") else [])
                for p in pols:
                    J.append(job(f, "hidden", s, "paused", "scripted:" + p, tag="schk"))
                    J.append(job(f, "hidden", s, "fixed", "scripted:" + p, L=3.0, tag="schk"))
                for p in ["oracle", "naive", "aware"]:
                    J.append(job(f, "visible", s, "fixed", "scripted:" + p, L=3.0, tag="schk"))
    elif preset == "models":
        model = sys.argv[2]
        for s in SEEDS:
            for f in FAMS:
                J.append(job(f, "hidden", s, "paused", model, tag="smod"))
                J.append(job(f, "hidden", s, "live", model, tag="smod"))
                J.append(job(f, "visible", s, "live", model, tag="smod"))
    with open(out, "w") as fh:
        fh.write("\n".join(json.dumps(j) for j in J) + "\n")
    print(len(J), "jobs ->", out)


if __name__ == "__main__":
    main()
