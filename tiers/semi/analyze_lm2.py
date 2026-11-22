"""Summarize RoboLiveMem part H runs.

  python analyze_lm2.py 'results/main_*.jsonl' 'results/diag_*.jsonl' ... --out summary.json --tex macros.tex

Cells: (policy, family, member, clock, diag, frames). Per cell: n, successes. Derived per policy and family:
drops S_P - S_L for each member, the twin gap Delta_mem = (S_P - S_L)_hidden - (S_P - S_L)_visible, and the hidden-member
decomposition by the diagnostics (look-again repairs = execution, oracle beyond look-again = memory).
Each failed episode gets one cause (see `cause`).
"""
import argparse
import collections
import glob
import json
import math

import numpy as np

FAMS = ["mugs", "knife", "heat", "point", "apple"]
MEMS = ["hidden", "visible", "static"]
SHORT = {"gpt-6-astra": "Astra", "opus-5.5": "Opus", "opus-5.5-think": "Opus-think", "gpt-6-luna": "Luna",
         "gpt-6.1-sol": "Sol", "haiku-5.5": "Haiku", "sonnet-5.5": "Sonnet", "scripted:oracle": "Oracle",
         "scripted:naive": "Naive"}


def load(patterns):
    rows = []
    for p in patterns:
        for f in sorted(glob.glob(p)):
            for x in open(f):
                if x.strip():
                    r = json.loads(x)
                    if r.get("success") is None:
                        continue
                    rows.append(r)
    seen, out = set(), []
    for r in rows:                    # one row per job id (a resumed shard may repeat one)
        if r["job_id"] in seen:
            continue
        seen.add(r["job_id"])
        out.append(r)
    return out


def motions(r):
    return [t for t in r.get("trace", []) if isinstance(t.get("call"), dict) and t["call"]["tool"] in ("pick", "place", "press")]


def first_after_change(r):
    """The first motion command that ran after the change started (its decision began before the change)."""
    ct = r.get("change_t")
    if ct is None:
        return None
    for t in motions(r):
        if t["t"] <= ct <= t.get("t_end", 1e9):
            return t
    return None


def stale(r):
    """Did the command decided before the change act on a now-wrong target?"""
    t = first_after_change(r)
    if t is None:
        return False
    fam, obj, tool = r["family"], t.get("obj"), t["call"]["tool"]
    if fam == "mugs":
        return tool == "pick" and obj == "mug_b"
    if fam == "knife":
        return tool == "pick" and obj != "knife"
    if fam == "heat":
        return tool == "place" and obj == "hotplate"
    return False


def cause(r):
    """One cause per failed episode."""
    if r["success"]:
        return "success"
    fam, o = r["family"], r.get("outcome", {})
    tr = r.get("trace", [])
    fails = [t for t in motions(r) if t.get("ok") is False or "failed" in str(t.get("result", ""))]
    if fam in ("knife", "apple") and r.get("wrong_cover", 0) > 0:
        return "search" if r.get("success_raw") or fam == "knife" else "wrong cover"
    if fam == "heat" and not o.get("heated") and o.get("on_tray"):
        return "over-cautious"
    if fam == "heat" and o.get("plastic_on_hot_plate_at_end"):
        return "stale, not repaired" if stale(r) else "missed write"
    if fam == "mugs" and o.get("ben_on_tray"):
        return "stale, not repaired" if stale(r) else "wrong identity"
    if fam == "point" and o.get("on_board") and r["outcome"].get("target") not in o.get("on_board"):
        return "missed write"
    if fam == "knife" and o.get("knife_at", "").startswith("cover"):
        return "lost (under cover)"
    if r.get("end") == "max_turns":
        return "turn budget" if not fails else "manipulation"
    if fails and r.get("end") != "done":
        return "manipulation"
    if r.get("end") == "done":
        return "premature done"
    return "other"


def rate(c):
    return c[1] / c[0] if c[0] else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex", default=None)
    ap.add_argument("--rows", default=None, help="write LaTeX rows of the main table (tab_home_rows.tex)")
    ap.add_argument("--famrows", default=None, help="write LaTeX rows of the per-family table")
    a = ap.parse_args()
    rows = load(a.patterns)
    cells = collections.defaultdict(lambda: [0, 0])
    causes = collections.defaultdict(collections.Counter)
    lat = collections.defaultdict(list)
    nfr = collections.defaultdict(list)
    stale_ct = collections.defaultdict(lambda: [0, 0, 0])   # (changed, stale, stale & success)
    hold = collections.defaultdict(list)
    for r in rows:
        key = (r["policy"], r["family"], r["member"], r["clock"], r.get("diag", "none"), r.get("frames", "buffer"))
        cells[key][0] += 1
        cells[key][1] += int(bool(r["success"]))
        causes[key][cause(r)] += 1
        for t in r.get("trace", []):
            if (t.get("meta") or {}).get("latency_s"):
                lat[r["policy"]].append(t["meta"]["latency_s"])
            if r["clock"] == "live":
                nfr[r["policy"]].append(t.get("n_new", 0))
        if r["clock"] == "live" and r.get("changed"):
            s = stale_ct[key]
            s[0] += 1
            if stale(r):
                s[1] += 1
                s[2] += int(bool(r["success"]))
        hold[r["policy"]].append(r.get("safety_hold_s", 0.0))
    pols = sorted({k[0] for k in cells}, key=lambda p: list(SHORT).index(p) if p in SHORT else 99)
    summ = {"cells": {"|".join(map(str, k)): v for k, v in cells.items()},
            "causes": {"|".join(map(str, k)): dict(v) for k, v in causes.items()},
            "stale": {"|".join(map(str, k)): v for k, v in stale_ct.items()},
            "latency": {p: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
                        for p, v in lat.items() if v},
            "frames_per_decision_live": {p: float(np.mean(v)) for p, v in nfr.items() if v}}
    print(f"{len(rows)} episodes")
    for p in pols:
        print(f"\n=== {SHORT.get(p, p)}  latency {summ['latency'].get(p, {})}")
        print(f"{'family':7s} {'member':8s} {'paused':>8s} {'live':>8s} {'look':>8s} {'oracle':>8s} {'latest':>8s}")
        tot = collections.defaultdict(lambda: [0, 0])
        for f in FAMS:
            for m in MEMS:
                def g(clock, diag="none", frames="buffer"):
                    c = cells.get((p, f, m, clock, diag, frames), [0, 0])
                    tot[(m, clock, diag, frames)][0] += c[0]
                    tot[(m, clock, diag, frames)][1] += c[1]
                    return f"{c[1]}/{c[0]}" if c[0] else "-"
                print(f"{f:7s} {m:8s} {g('paused'):>8s} {g('live'):>8s} {g('live', 'look'):>8s} {g('live', 'oracle'):>8s} {g('live', 'none', 'latest'):>8s}")
        for m in MEMS:
            def gt(clock, diag="none", frames="buffer"):
                c = tot[(m, clock, diag, frames)]
                return f"{c[1]}/{c[0]}" if c[0] else "-"
            print(f"{'ALL':7s} {m:8s} {gt('paused'):>8s} {gt('live'):>8s} {gt('live', 'look'):>8s} {gt('live', 'oracle'):>8s} {gt('live', 'none', 'latest'):>8s}")
        # twin gap per family and overall
        for f in FAMS + ["ALL"]:
            def s(m, clock):
                if f == "ALL":
                    c = tot[(m, clock, "none", "buffer")]
                else:
                    c = cells.get((p, f, m, clock, "none", "buffer"), [0, 0])
                return rate(c)
            dh, dv, ds = s("hidden", "paused") - s("hidden", "live"), s("visible", "paused") - s("visible", "live"), s("static", "paused") - s("static", "live")
            print(f"  {f:6s} drop hidden {dh:+.2f}  visible {dv:+.2f}  static {ds:+.2f}  Delta_mem {dh - dv:+.2f}")
        cc = collections.Counter()
        for k, v in causes.items():
            if k[0] == p and k[3] == "live" and k[4] == "none" and k[5] == "buffer":
                for c, n in v.items():
                    if c != "success":
                        cc[(k[2], c)] += n
        print("  live failure causes:", dict(sorted(cc.items())))
        for f in FAMS:
            for m in ("hidden", "visible"):
                s_ = stale_ct.get((p, f, m, "live", "none", "buffer"))
                if s_:
                    print(f"  stale {f} {m}: changed {s_[0]}, stale {s_[1]}, repaired {s_[2]}")
    if a.out:
        json.dump(summ, open(a.out, "w"), indent=1)
    if a.tex:
        lines = ["% generated by analyze_lm2.py (RoboLiveMem part H)"]
        TAGS = (("paused", "none", "buffer", "P"), ("live", "none", "buffer", "L"), ("live", "look", "buffer", "LA"),
                ("live", "oracle", "buffer", "OR"), ("live", "none", "latest", "LT"))
        def mac(name, val):
            lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
        for p in pols:
            P = SHORT.get(p, p).replace("-", "")
            for m in MEMS:
                for clock, diag, frames, tag in TAGS:
                    tn, tk = 0, 0
                    for f in FAMS:
                        c = cells.get((p, f, m, clock, diag, frames))
                        if c:
                            mac(f"h{P}{f.capitalize()}{m.capitalize()}{tag}", c[1])
                            mac(f"h{P}{f.capitalize()}{m.capitalize()}{tag}N", c[0])
                            tn += c[0]
                            tk += c[1]
                    if tn:
                        mac(f"h{P}All{m.capitalize()}{tag}", tk)
                        mac(f"h{P}All{m.capitalize()}{tag}N", tn)
                        mac(f"h{P}All{m.capitalize()}{tag}Pct", f"{100.0 * tk / tn:.0f}")
            # twin gap over all families (percentage points)
            def r_(m, clock):
                n = sum(cells.get((p, f, m, clock, "none", "buffer"), [0, 0])[0] for f in FAMS)
                k = sum(cells.get((p, f, m, clock, "none", "buffer"), [0, 0])[1] for f in FAMS)
                return k / n if n else float("nan")
            dh = r_("hidden", "paused") - r_("hidden", "live")
            dv = r_("visible", "paused") - r_("visible", "live")
            ds = r_("static", "paused") - r_("static", "live")
            if not math.isnan(dh):
                mac(f"h{P}DropHidden", f"{100 * dh:.0f}")
                mac(f"h{P}DropVisible", f"{100 * dv:.0f}")
                mac(f"h{P}DropStatic", f"{100 * ds:.0f}")
                mac(f"h{P}DeltaMem", f"{100 * (dh - dv):.0f}")
            L = summ["latency"].get(p)
            if L:
                mac(f"h{P}Lat", f"{L['median']:.1f}")
                mac(f"h{P}LatNinety", f"{L['p90']:.1f}")
            fr = summ["frames_per_decision_live"].get(p)
            if fr:
                mac(f"h{P}Frames", f"{fr:.1f}")
            for m in ("hidden", "visible"):
                ch = st = rp = 0
                for f in FAMS:
                    v = stale_ct.get((p, f, m, "live", "none", "buffer"))
                    if v:
                        ch, st, rp = ch + v[0], st + v[1], rp + v[2]
                mac(f"h{P}Changed{m.capitalize()}", ch)
                mac(f"h{P}Stale{m.capitalize()}", st)
                mac(f"h{P}Repaired{m.capitalize()}", rp)
        open(a.tex, "w").write("\n".join(lines) + "\n")
    NAMES = {"gpt-6-astra": "GPT-6 Astra", "fable-5.1": "Fable 5.1", "opus-5.5": "Opus 5.5", "opus-5.5-think": "Opus 5.5 think", "gpt-6-luna": "GPT-6 Luna",
             "gpt-6.1-sol": "GPT-6.1 Sol", "haiku-5.5": "Haiku 5.5"}
    def cnt(p, fams, m, clock, diag="none", frames="buffer"):
        n = sum(cells.get((p, f, m, clock, diag, frames), [0, 0])[0] for f in fams)
        k = sum(cells.get((p, f, m, clock, diag, frames), [0, 0])[1] for f in fams)
        return k, n
    def fmt(kn, full):
        k, n = kn
        if n == 0:
            return "--"
        return f"{k}" if n == full else f"{k}/{n}"
    if a.rows:
        out = []
        for p in [q for q in NAMES if q in pols]:
            hp, hl = cnt(p, FAMS, "hidden", "paused"), cnt(p, FAMS, "hidden", "live")
            vp, vl = cnt(p, FAMS, "visible", "paused"), cnt(p, FAMS, "visible", "live")
            sp, sl = cnt(p, FAMS, "static", "paused"), cnt(p, FAMS, "static", "live")
            def r(kn):
                return kn[0] / kn[1] if kn[1] else float("nan")
            dm = (r(hp) - r(hl)) - (r(vp) - r(vl))
            L = summ["latency"].get(p, {}).get("median")
            FULL = 10 * len(FAMS)
            row = [NAMES[p], f"{L:.1f}" if L else "--", fmt(hp, FULL), fmt(hl, FULL), fmt(vp, FULL), fmt(vl, FULL), fmt(sp, FULL), fmt(sl, FULL),
                   "--" if math.isnan(dm) else f"{100 * dm:+.0f}",
                   fmt(cnt(p, FAMS, "hidden", "live", "look"), FULL), fmt(cnt(p, FAMS, "hidden", "live", "oracle"), FULL),
                   fmt(cnt(p, FAMS, "hidden", "live", "none", "latest"), FULL)]
            out.append(" & ".join(row) + r" \\")
        open(a.rows, "w").write("\n".join(out) + "\n")
    if a.famrows:
        out = []
        FN = {"mugs": "Anna's mug", "knife": "Knife put away", "heat": "Residual heat", "point": "Shown jar",
              "apple": "Covered dishes"}
        for p in [q for q in NAMES if q in pols]:
            for i, f in enumerate(FAMS):
                row = [NAMES[p] if i == 0 else "", FN[f]]
                for m in MEMS:
                    row += [fmt(cnt(p, [f], m, "paused"), 10), fmt(cnt(p, [f], m, "live"), 10)]
                row += [fmt(cnt(p, [f], "hidden", "live", "look"), 10), fmt(cnt(p, [f], "visible", "live", "look"), 10),
                        fmt(cnt(p, [f], "hidden", "live", "oracle"), 10), fmt(cnt(p, [f], "hidden", "live", "none", "latest"), 10)]
                out.append(" & ".join(row) + r" \\")
            out.append(r"\midrule")
        if out and out[-1] == r"\midrule":
            out.pop()
        open(a.famrows, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
