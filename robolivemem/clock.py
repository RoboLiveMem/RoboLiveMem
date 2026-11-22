"""The three clocks of RoboLiveMem (paper, Eq. 1).

A decision that observes the world at t_o and thinks for `latency` seconds starts its command at

    t_c = t_o + kappa(latency),   kappa_paused = 0,   kappa_live = latency,   kappa_fixed(L) = L.

Paused is current practice (the world waits for the model). Live charges the measured time of the model call.
Fixed charges the same L seconds to every decision, to vary latency on its own.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    name: str                  # "paused" | "live" | "fixed"
    fixed_s: float = 0.0       # L for the fixed clock

    def kappa(self, latency_s: float) -> float:
        """World seconds that pass while the model thinks for `latency_s` wall-clock seconds."""
        if self.name == "paused":
            return 0.0
        if self.name == "live":
            return max(0.0, float(latency_s))
        if self.name == "fixed":
            return float(self.fixed_s)
        raise ValueError(self.name)

    @property
    def label(self) -> str:
        return f"fixed{self.fixed_s:g}" if self.name == "fixed" else self.name


PAUSED = Clock("paused")
LIVE = Clock("live")


def fixed(seconds: float) -> Clock:
    return Clock("fixed", float(seconds))


def parse(spec: str) -> Clock:
    """'paused' | 'live' | 'fixed:4' (or 'fixed4')."""
    s = spec.strip().lower()
    if s in ("paused", "p"):
        return PAUSED
    if s in ("live", "l"):
        return LIVE
    if s.startswith("fixed"):
        return fixed(float(s[5:].lstrip(":=") or 0))
    raise ValueError(f"unknown clock {spec!r}: use paused, live or fixed:<seconds>")
