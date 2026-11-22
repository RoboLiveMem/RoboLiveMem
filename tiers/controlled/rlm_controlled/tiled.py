"""Tiled view of a turn for API models, matching what the file-interface agents read.

The file interface (file_client.py) shows the frames since the last turn and the demonstration keyframes as tiles of
up to 15 frames each (256 px per frame, time stamp above each), plus the newest frame with and without marks as full
images. TiledClient applies the same transformation to every turn before it goes to an API model, and keeps at most
`max_images` images per request (the demonstration tiles are always kept; older images become "[image omitted]").
"""
from __future__ import annotations

import numpy as np

from rlm_controlled.file_client import _tile

PER_TILE = 15


def _is_img(p):
    return not isinstance(p, str)


def tile_parts(parts):
    """Rewrite one user turn: demo keyframes and between-turn frames become tiles; other parts are kept in order."""
    out, i, n = [], 0, len(parts)
    while i < n:
        p = parts[i]
        if isinstance(p, str) and p.startswith("Demonstration video"):
            j, imgs = i + 1, []
            while j < n and _is_img(parts[j]):
                imgs.append(parts[j])
                j += 1
            if imgs:
                out.append(p.rstrip(":") + f"; tiled {PER_TILE} per image, each labelled 'video k'):")
                for c in range(0, len(imgs), PER_TILE):
                    out.append(np.asarray(_tile(imgs[c:c + PER_TILE], [f"video {k + 1}" for k in range(c, min(len(imgs), c + PER_TILE))])))
            else:
                out.append(p)
            i = j
            continue
        if isinstance(p, str) and p.startswith("Frames since your last turn"):
            j, imgs, labs, omitted = i + 1, [], [], 0
            while j + 1 < n and isinstance(parts[j], str) and parts[j].startswith("t="):
                if _is_img(parts[j + 1]):
                    imgs.append(parts[j + 1])
                    labs.append(parts[j])
                else:
                    omitted += 1
                j += 2
            out.append(p.rstrip(":") + f"; tiled in time order, {PER_TILE} per image, each labelled with its time stamp):")
            if labs:
                out.append("time stamps: " + ", ".join(labs))
            for c in range(0, len(imgs), PER_TILE):
                out.append(np.asarray(_tile(imgs[c:c + PER_TILE], labs[c:c + PER_TILE])))
            if omitted:
                out.append(f"[{omitted} frame(s) omitted]")
            i = j
            continue
        out.append(p)
        i += 1
    return out


class TiledClient:
    def __init__(self, inner, max_images=24):
        self.inner = inner
        self.short = getattr(inner, "short", "api")
        self.max_images = max_images

    def __call__(self, system, turns):
        tt = [(role, tile_parts(ps) if role == "user" else ps) for role, ps in turns]
        # cap the images per request: keep the demonstration tiles, then the newest images
        demo_ids, other_ids = set(), []
        for ti, (role, ps) in enumerate(tt):
            in_demo = False
            for pi, p in enumerate(ps):
                if isinstance(p, str):
                    in_demo = p.startswith("Demonstration video")
                    continue
                if in_demo and ti == 0:
                    demo_ids.add((ti, pi))
                else:
                    other_ids.append((ti, pi))
        keep = set(other_ids[-self.max_images:]) | demo_ids
        tt = [(role, [p if isinstance(p, str) or (ti, pi) in keep else "[image omitted]" for pi, p in enumerate(ps)])
              for ti, (role, ps) in enumerate(tt)]
        text, meta = self.inner(system, tt)
        meta["n_images_sent"] = len(keep)
        return text, meta
