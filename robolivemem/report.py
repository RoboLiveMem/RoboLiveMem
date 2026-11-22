"""Tables from episode rows: per task, per tier, per agent (the paper's main-table quantities).

    python -m robolivemem.report results/*.jsonl --out results/report

For every agent and tier, on the hidden families (class E, the execution control, is reported but not aggregated):
    P      success on event seeds, paused clock
    L      success on event seeds, live clock
    none   success on null seeds, live clock (nothing changes)
    L - P  the live drop, paired over (task, seed), with a 95% bootstrap interval
    V, mem on the families with a visible twin, event and decoy seeds: twin success and the memory term V - L
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from .benchmark import TIERS
from .metrics import paired_bootstrap


def load(paths) -> list[dict]:
    rows = OrderedDict()
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("valid"):
                continue
            k = (r.get("agent"), r["tier"], r["task"], int(r["seed"]), r["condition"], r.get("clock"), bool(r.get("commit")))
            rows[k] = r          # the last valid run of an episode counts
    return list(rows.values())


def _rate(rs):
    return (100.0 * sum(bool(r["success"]) for r in rs) / len(rs), len(rs)) if rs else (None, 0)


def _eps(rows, variants):
    return [dict(instance=(r["task"], r["seed"]), condition=r["condition"], success=bool(r["success"]))
            for r in rows if r.get("variant") in variants]


def summarize(rows: list[dict]) -> dict:
    """{agent: {tier: {"tasks": {task: {...}}, "summary": {...}}}}"""
    out = {}
    for agent in sorted({r["agent"] for r in rows}):
        out[agent] = {}
        for tier in [t for t in TIERS if any(r["tier"] == t and r["agent"] == agent for r in rows)]:
            spec = TIERS[tier]["tasks"]
            R = [r for r in rows if r["agent"] == agent and r["tier"] == tier]
            tasks = {}
            for task in [t for t in spec if any(r["task"] == t for r in R)]:
                T = [r for r in R if r["task"] == task]
                ev = lambda c, v=("event",): [r for r in T if r["condition"] == c and r.get("variant") in v]  # noqa: E731
                p, n_p = _rate(ev("P"))
                l, n_l = _rate(ev("L"))
                none, n_n = _rate([r for r in T if r["condition"] == "L" and r.get("variant") == "null"])
                v, n_v = _rate(ev("V", ("event", "decoy")))
                lv, _ = _rate(ev("L", ("event", "decoy")))
                tasks[task] = dict(cls=spec[task]["class"], P=p, L=l, none=none, V=v, L_ed=lv, n_P=n_p, n_L=n_l,
                                   n_none=n_n, n_V=n_v, drop=None if p is None or l is None else l - p,
                                   mem=None if v is None or lv is None else v - lv)
            mem_tasks = [t for t in tasks if spec[t]["class"] != "E"]
            H = [r for r in R if r["task"] in mem_tasks]
            p, n_p = _rate([r for r in H if r["condition"] == "P" and r.get("variant") == "event"])
            l, n_l = _rate([r for r in H if r["condition"] == "L" and r.get("variant") == "event"])
            none, n_n = _rate([r for r in H if r["condition"] == "L" and r.get("variant") == "null"])
            drop = paired_bootstrap(_eps(H, ("event",)), "L", "P")
            twin = [r for r in H if spec[r["task"]].get("twin")]
            mem = paired_bootstrap(_eps(twin, ("event", "decoy")), "V", "L")
            lat = [r["mean_latency_s"] for r in R if r.get("condition") == "L" and r.get("mean_latency_s") is not None]
            src = sorted({r.get("latency_source") for r in R if r.get("condition") != "P" and r.get("latency_source")})
            out[agent][tier] = {"tasks": tasks, "summary": dict(
                P=p, L=l, none=none, n_P=n_p, n_L=n_l, n_none=n_n, drop=drop, mem=mem, episodes=len(R),
                mean_latency_s=round(sum(lat) / len(lat), 2) if lat else None, latency_source=",".join(src) or None)}
    return out


def _f(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def _ci(d):
    if not d or not d.get("n"):
        return "–"
    return f"{d['diff']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]"


def to_markdown(summary: dict) -> str:
    lines = ["# RoboLiveMem results", ""]
    lines += ["| Agent | Tier | P | L | none | L − P [95% CI] | memory V − L [95% CI] | episodes | latency (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for agent, tiers in summary.items():
        for tier, d in tiers.items():
            s = d["summary"]
            lines.append(f"| {agent} | {tier} | {_f(s['P'])} | {_f(s['L'])} | {_f(s['none'])} | {_ci(s['drop'])} | "
                         f"{_ci(s['mem'])} | {s['episodes']} | {_f(s['mean_latency_s'], 2)} ({s['latency_source'] or '–'}) |")
    for agent, tiers in summary.items():
        for tier, d in tiers.items():
            lines += ["", f"## {agent} · {TIERS[tier]['title']}", "",
                      "| Task | Class | P | L | L − P | none | V | L (twin seeds) | memory | n (P/L/none/V) |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for task, t in d["tasks"].items():
                name = TIERS[tier]["tasks"][task].get("name", task)
                lines.append(f"| {name} | {t['cls']} | {_f(t['P'])} | {_f(t['L'])} | {_f(t['drop'])} | {_f(t['none'])} | "
                             f"{_f(t['V'])} | {_f(t['L_ed'])} | {_f(t['mem'])} | {t['n_P']}/{t['n_L']}/{t['n_none']}/{t['n_V']} |")
    lines += ["", "P, L: success (%) on event seeds, paused and live clock. none: live success on null seeds. V: visible "
              "twin, live, event and decoy seeds; memory = V − L on the same seeds. Class E is an execution control and "
              "is not in the tier summary. latency: mean charged seconds per decision (declared: reported by the agent; "
              "wall: measured by the server)."]
    return "\n".join(lines)


def write_report(paths, stem, agents=None) -> str:
    rows = load(paths)
    if agents:
        rows = [r for r in rows if r["agent"] in agents]
    s = summarize(rows)
    md = to_markdown(s)
    stem = Path(stem)
    stem.with_suffix(".md").write_text(md + "\n")
    stem.with_suffix(".json").write_text(json.dumps(s, indent=1))
    return md


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rows", nargs="+", help="episode rows (jsonl) from robolivemem.eval")
    ap.add_argument("--out", default="report", help="output stem (.md and .json)")
    a = ap.parse_args(argv)
    print(write_report(a.rows, a.out))


if __name__ == "__main__":
    main()
