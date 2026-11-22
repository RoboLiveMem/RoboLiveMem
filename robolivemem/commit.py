"""The commit check (paper, Eq. 3): before a command runs, look at the target once more.

    c_i is cancelled if  max_{t in (t_last, t_c]}  d( W_m(o_t), W_m(o_{t_last}) )  > theta

t_last is the time of the last frame the agent saw, W_m a window around the target mark m and d the mean absolute
pixel difference. A cancelled command uses no budget; the agent gets a neutral message and decides again on the new
frames. It needs no model access and no training.
"""
from __future__ import annotations

import numpy as np

MESSAGE = "not executed: the camera view around mark {mark} changed since your last observation"


def window(img: np.ndarray, center_xy, half: int) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    return img[max(0, y - half):min(h, y + half), max(0, x - half):min(w, x + half)].astype(np.float32)


def changed(seen: np.ndarray, frames_since, center_xy, half: int = 28, theta: float = 6.0) -> bool:
    """True if any frame recorded after `seen` differs from it around `center_xy` by more than theta (0-255 scale)."""
    ref = window(seen, center_xy, half)
    for f in frames_since:
        if float(np.abs(window(f, center_xy, half) - ref).mean()) > theta:
            return True
    return False
