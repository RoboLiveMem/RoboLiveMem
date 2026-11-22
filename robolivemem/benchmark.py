"""The benchmark: tasks, classes, visible twins and the episodes of the paper's protocol, per tier.

Classes: U update (a remembered state is moved, swapped or hidden), W write (a one-time event must be stored),
T temporal (how long, how many), E execution control (no memory needed; excluded from the memory aggregates).
Seeds: seed % 3 gives event, decoy, null (the tier reports the variant of every episode).
"""
from __future__ import annotations

TIERS = {
    "tabletop": {
        "title": "Tabletop (2-D, built in)",
        "tasks": {"mug": {"class": "U", "twin": True}},
        "seeds": list(range(12)),
    },
    "realistic": {
        "title": "Realistic (BEHAVIOR-1K)",
        "tasks": {
            "v3_mugs": {"class": "U", "twin": True, "scene": "house_v3", "name": "Grandpa's mug"},
            "v3_totes": {"class": "U", "twin": True, "scene": "hall_v3", "name": "Express tote"},
            "v3_keys": {"class": "U", "twin": True, "scene": "house_v3", "name": "Car keys"},
            "v3_bins": {"class": "U", "twin": True, "scene": "hall_v3", "name": "Gauge in bins"},
            "v3_sugar": {"class": "W", "twin": True, "scene": "house_v3", "name": "Mom's tea"},
            "v3_reagent": {"class": "W", "twin": True, "scene": "hall_v3", "name": "Catalyst dose"},
            "s01_put_away": {"class": "U", "twin": True, "scene": "house", "name": "Put away"},
            "s02_annas_mug": {"class": "U", "twin": False, "scene": "house", "name": "Anna's mug"},
            "s03_vacuum_ball": {"class": "U", "twin": False, "scene": "house", "name": "Vacuum and ball"},
            "s04_salted_soup": {"class": "W", "twin": False, "scene": "house", "name": "Salted soup"},
            "s05_rotary_reject": {"class": "U", "twin": False, "scene": "industrial", "name": "Rotary reject"},
            "s06_conveyor_lanes": {"class": "U", "twin": False, "scene": "industrial", "name": "Conveyor lanes"},
            "s07_press_lamp": {"class": "W", "twin": True, "scene": "industrial", "name": "Press lamp"},
            "s08_parcel_locker": {"class": "U", "twin": True, "scene": "industrial", "name": "Parcel locker"},
            "s09_kitting_partner": {"class": "W", "twin": False, "scene": "industrial", "name": "Kitting partner"},
            "s10_cloche_pass": {"class": "U", "twin": False, "scene": "restaurant", "name": "Cloche pass"},
        },
        "seeds": list(range(12)),
        # paper: event seeds paused and live, every seed live, twins live on event and decoy seeds
        "plan": {"P": ("event",), "L": ("event", "decoy", "null"), "V": ("event", "decoy")},
    },
    "semi": {
        "title": "Semi-realistic (MuJoCo kitchen)",
        "tasks": {
            "mugs": {"class": "U", "twin": True, "name": "Mug swap"},
            "apple": {"class": "U", "twin": True, "name": "Covered dishes"},
            "heatw": {"class": "W", "twin": True, "name": "Residual heat"},
            "point": {"class": "W", "twin": True, "name": "Shown jar"},
            "knife": {"class": "W", "twin": True, "name": "Knife put away"},
            "heatdur": {"class": "T", "twin": True, "name": "Heat duration"},
            "sugar2": {"class": "T", "twin": True, "name": "Sugar count"},
            "heat": {"class": "E", "twin": True, "name": "Plate switched on (execution control)"},
        },
        "seeds": list(range(36)),
    },
    "controlled": {
        "title": "Controlled (ManiSkill, RoboMME scenes)",
        "tasks": {
            "SwapStream": {"class": "U", "twin": True},
            "InterruptSwap": {"class": "U", "twin": True},
            "RepickStream": {"class": "U", "twin": True},
            "HighlightLate": {"class": "W", "twin": True},
            "RevealLate": {"class": "W", "twin": True},
        },
        "seeds": list(range(40)),
    },
}

KIND_OF_SEED = ("event", "decoy", "null")


def seed_variant(tier: str, seed: int) -> str | None:
    """The variant a seed is expected to give (the tier's own report is authoritative)."""
    return KIND_OF_SEED[seed % 3] if tier in ("tabletop", "realistic", "semi") else None


def jobs(tier: str, tasks=None, seeds=None, conditions=("P", "L", "V"), clock: str | None = None, commit=False):
    """The episodes of the paper's protocol for one tier (optionally restricted)."""
    spec = TIERS[tier]
    tasks = list(tasks or spec["tasks"])
    seeds = list(spec["seeds"] if seeds is None else seeds)
    plan = spec.get("plan")
    out = []
    for task in tasks:
        info = spec["tasks"][task]
        for seed in seeds:
            for c in conditions:
                if c == "V" and not info.get("twin"):
                    continue
                if plan and seed_variant(tier, seed) not in plan[c]:
                    continue
                out.append({"tier": tier, "task": task, "class": info["class"], "seed": seed, "condition": c,
                            "clock": "paused" if c == "P" else (clock or "live"), "commit": bool(commit)})
    return out
