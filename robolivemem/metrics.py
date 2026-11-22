"""Scoring: success rates, the live drop and its decomposition (paper, Eq. 2-4), with paired bootstrap intervals.

Every episode is a dict with at least
    instance   any hashable id shared by the paired runs of one family and seed (e.g. "mug/7")
    condition  "P" (hidden, paused), "L" (hidden, live) or "V" (visible twin, live)
    success    bool
and optionally
    stale      bool, the decisive decision observed before the change was complete (sigma, Eq. 3)
    lead_s     float, lambda = t_o(decisive) - tau_e

    live drop   Delta      = S_P - S_L
                           = (S_P - S_V)  execution (latency + observation gap)
                           + (S_V - S_L)  memory    (Delta_mem)
"""
from __future__ import annotations

import random
from collections import defaultdict
from statistics import median
from typing import Iterable


def success_rate(eps: Iterable[dict]) -> float:
    eps = list(eps)
    return 100.0 * sum(bool(e["success"]) for e in eps) / len(eps) if eps else float("nan")


def _paired(eps: Iterable[dict]) -> dict:
    """instance -> {condition: success (0/1)}"""
    by = defaultdict(dict)
    for e in eps:
        by[e["instance"]][e["condition"]] = 1.0 if e["success"] else 0.0
    return by


def paired_bootstrap(eps: Iterable[dict], a: str, b: str, n: int = 2000, seed: int = 0, level: float = 0.95):
    """S_a - S_b over the instances that ran both conditions, with a paired bootstrap interval (in points)."""
    by = _paired(eps)
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    if not pairs:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    diff = 100.0 * sum(x - y for x, y in pairs) / len(pairs)
    rng = random.Random(seed)
    k = len(pairs)
    boot = sorted(100.0 * sum(x - y for x, y in (pairs[rng.randrange(k)] for _ in range(k))) / k for _ in range(n))
    lo = boot[int((1 - level) / 2 * n)]
    hi = boot[min(n - 1, int((1 + level) / 2 * n))]
    return {"diff": diff, "lo": lo, "hi": hi, "n": k}


def decompose(eps: Iterable[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """S_P, S_L, S_V and the three differences of Eq. 4, each with its 95% paired bootstrap interval."""
    eps = list(eps)
    out = {f"S_{c}": success_rate(e for e in eps if e["condition"] == c) for c in ("P", "L", "V")}
    out["drop"] = paired_bootstrap(eps, "P", "L", n_boot, seed)       # Delta
    out["execution"] = paired_bootstrap(eps, "P", "V", n_boot, seed)  # S_P - S_V
    out["memory"] = paired_bootstrap(eps, "V", "L", n_boot, seed)     # Delta_mem
    return out


def stale_split(eps: Iterable[dict]) -> dict:
    """Success of stale and fresh decisive decisions (paper, Fig. 5) on live hidden episodes."""
    live = [e for e in eps if e["condition"] == "L" and e.get("stale") is not None]
    st = [e for e in live if e["stale"]]
    fr = [e for e in live if not e["stale"]]
    leads = [e["lead_s"] for e in live if e.get("lead_s") is not None]
    return {"n": len(live), "stale_share": 100.0 * len(st) / len(live) if live else float("nan"),
            "S_stale": success_rate(st), "S_fresh": success_rate(fr), "n_stale": len(st), "n_fresh": len(fr),
            "median_lead_s": median(leads) if leads else float("nan")}


def fmt_diff(d: dict) -> str:
    return f"{d['diff']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]"


def report(eps: Iterable[dict], title: str = "") -> str:
    """A short plain-text report of Eq. 2-4 and the stale/fresh split."""
    eps = list(eps)
    d = decompose(eps)
    s = stale_split(eps)
    lines = [title] if title else []
    lines += [
        f"  paused  S_P = {d['S_P']:5.1f}%      live  S_L = {d['S_L']:5.1f}%      visible twin  S_V = {d['S_V']:5.1f}%",
        f"  live drop   S_P - S_L = {fmt_diff(d['drop'])}",
        f"   execution  S_P - S_V = {fmt_diff(d['execution'])}",
        f"   memory     S_V - S_L = {fmt_diff(d['memory'])}",
    ]
    if s["n"]:
        pct = lambda v, n: f"{v:.0f}% ({n})" if n else "- (0)"  # noqa: E731
        lines.append(f"  live decisive decisions: {s['stale_share']:.0f}% stale; success stale "
                     f"{pct(s['S_stale'], s['n_stale'])}, fresh {pct(s['S_fresh'], s['n_fresh'])}")
    return "\n".join(lines)
