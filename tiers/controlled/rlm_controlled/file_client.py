"""File-based policy client: the harness writes each turn (text + images) to a directory and waits for a reply file.
Lets an interactive agent session (e.g. a Claude Code subagent) act as the policy. Use with a fixed-latency clock, since
the session's own wall-clock time is not an API latency.

Turn k: turn_{k}.txt (system prompt on the first turn, then this turn's text in order, with [image i] placeholders),
turn_{k}_img{i}.png (each image of the turn; frames of the turn are also tiled into turn_{k}_frames.png with time stamps),
reply_{k}.json is written by the policy. DONE is written when the episode ends.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


def _tile(images, labels, cols=5):
    from PIL import Image, ImageDraw
    n = len(images)
    rows = (n + cols - 1) // cols
    W = 256
    G = Image.new("RGB", (cols * W, rows * (W + 18)), (255, 255, 255))
    d = ImageDraw.Draw(G)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x, y = (i % cols) * W, (i // cols) * (W + 18)
        G.paste(Image.fromarray(np.asarray(im, dtype=np.uint8)).resize((W, W)), (x, y + 18))
        d.text((x + 4, y + 3), lab, fill=(0, 0, 0))
    return G


class FileClient:
    short = "file"

    def __init__(self, workdir, name="claude-subagent", timeout_s=900):
        self.dir = workdir
        self.short = name
        self.timeout_s = timeout_s
        self.k = 0
        os.makedirs(workdir, exist_ok=True)

    def __call__(self, system, turns):
        from PIL import Image
        role, parts = turns[-1]
        k = self.k
        lines, frames, flabels, last_label, img_i = [], [], [], None, 0
        if k == 0:
            lines += ["=== SYSTEM ===", system, "=== TURN 0 ==="]
        else:
            lines += [f"=== TURN {k} ==="]
        for p in parts:
            if isinstance(p, str):
                lines.append(p)
                last_label = p
                continue
            path = os.path.join(self.dir, f"turn_{k}_img{img_i}.png")
            Image.fromarray(np.asarray(p, dtype=np.uint8)).save(path)
            lines.append(f"[image {img_i}: {os.path.basename(path)}]")
            if last_label and last_label.startswith("t="):
                frames.append(p)
                flabels.append(last_label)
            img_i += 1
        if frames:
            names = []
            for c in range(0, len(frames), 15):   # at most 15 frames per tile image, so each stays legible
                nm = f"turn_{k}_frames.png" if c == 0 else f"turn_{k}_frames_{c // 15 + 1}.png"
                _tile(frames[c:c + 15], flabels[c:c + 15]).save(os.path.join(self.dir, nm))
                names.append(nm)
            lines.append(f"(the frames since your last turn are also tiled, in time order, in {', '.join(names)})")
        demo = [p for p in parts if not isinstance(p, str)]
        if k == 0 and any(isinstance(p, str) and p.startswith("Demonstration video") for p in parts):
            n_demo = next(int(p.split("(")[1].split()[0]) for p in parts if isinstance(p, str) and p.startswith("Demonstration video"))
            names = []
            for c in range(0, n_demo, 15):   # at most 15 keyframes per tile image, so each stays legible
                nm = "turn_0_video.png" if c == 0 else f"turn_0_video_{c // 15 + 1}.png"
                _tile(demo[c:min(n_demo, c + 15)], [f"video {i + 1}" for i in range(c, min(n_demo, c + 15))]).save(
                    os.path.join(self.dir, nm))
                names.append(nm)
            lines.append(f"(the demonstration keyframes are also tiled in {', '.join(names)})")
        lines.append(f"Write your reply JSON to reply_{k}.json")
        tmp = os.path.join(self.dir, f".turn_{k}.txt")
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, os.path.join(self.dir, f"turn_{k}.txt"))
        t0 = time.monotonic()
        rp = os.path.join(self.dir, f"reply_{k}.json")
        while not os.path.exists(rp):
            if time.monotonic() - t0 > self.timeout_s:
                self.k += 1
                return "", {"latency_s": round(time.monotonic() - t0, 2), "error": "reply timeout"}
            time.sleep(0.5)
        time.sleep(0.2)
        text = open(rp).read()
        self.k += 1
        return text, {"latency_s": round(time.monotonic() - t0, 2)}

    def done(self, result):
        with open(os.path.join(self.dir, "DONE"), "w") as f:
            json.dump({k: result.get(k) for k in ("status", "steps", "success")}, f)
