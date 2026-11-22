"""Write a jobs file. Presets:
  val            scripted oracle (paused, and fixed L=4 with the oracle re-ask) and naive (fixed L=4), seeds 0-1
  main MODEL     4 families x 3 members x {paused, live}, seeds 0-9
  diag MODEL     live with look-again and oracle re-ask (hidden, visible), seeds 0-9; latest-frame-only (hidden)
  fixed MODEL    fixed L in {2, 4, 8, 16} on hidden and visible (scripted or model)
"""
import json
import sys

FAMS = ["mugs", "knife", "heat", "point", "apple"]
MEMS = ["hidden", "visible", "static"]


def job(fam, mem, seed, clock, pol, diag="none", frames="buffer", L=0.0, tag="x", **extra):
    short = pol.replace("scripted:", "")
    jid = f"{tag}_{short}_{fam}_{mem}_{clock}{'' if clock != 'fixed' else int(L)}_{diag}_{frames}_s{seed}"
    jid += "".join(f"_{k}{v}" for k, v in sorted(extra.items()))
    return dict(job_id=jid, family=fam, member=mem, seed=seed, clock=clock, L=L, policy=pol, diag=diag, frames=frames, tag=tag,
                **extra)


def main():
    preset, out = sys.argv[1], sys.argv[-1]
    model = sys.argv[2] if len(sys.argv) > 3 else None
    seeds = range(int(sys.argv[3]) if len(sys.argv) > 4 else 10)
    J = []
    if preset == "val":
        for f in FAMS:
            for m in MEMS:
                for s in (0, 1):
                    J.append(job(f, m, s, "paused", "scripted:oracle", tag="val"))
                    J.append(job(f, m, s, "fixed", "scripted:oracle", diag="oracle", L=4.0, tag="val"))
                    J.append(job(f, m, s, "fixed", "scripted:naive", L=4.0, tag="val"))
    elif preset == "main":
        for s in seeds:
            for f in FAMS:
                for m in MEMS:
                    for c in ("paused", "live"):
                        J.append(job(f, m, s, c, model, tag="main"))
    elif preset == "diag":
        for s in seeds:
            for f in FAMS:
                for m in ("hidden", "visible"):
                    J.append(job(f, m, s, "live", model, diag="look", tag="diag"))
                    J.append(job(f, m, s, "live", model, diag="oracle", tag="diag"))
                J.append(job(f, "hidden", s, "live", model, frames="latest", tag="diag"))
    elif preset == "fixed":
        for s in seeds:
            for f in FAMS:
                for m in ("hidden", "visible"):
                    for L in (2.0, 4.0, 8.0, 16.0):
                        J.append(job(f, m, s, "fixed", model, L=L, tag="fixed"))
    with open(out, "w") as fh:
        fh.write("\n".join(json.dumps(j) for j in J) + "\n")
    print(len(J), "jobs ->", out)


if __name__ == "__main__":
    main()
