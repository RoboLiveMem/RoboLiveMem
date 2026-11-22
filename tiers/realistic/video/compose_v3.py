"""Three-panel preview video of one v3 scenario: the event, look-alike and no-change variants side by side, each panel the
spectator camera with the robot's head camera inset, aligned on world time.

    python compose_v3.py --title "Mom's tea" --kind DONE --out sugar.mp4 \
        --panel DIR_event "event: ..." --panel DIR_decoy "look-alike: ..." --panel DIR_null "no change: ..."

Each DIR holds main.mp4, head.mp4, frames.jsonl and info.json written by the bench server.
"""
import argparse, json, subprocess
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--title", required=True)
ap.add_argument("--kind", default="")
ap.add_argument("--sub", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--panel", nargs=2, action="append", metavar=("DIR", "CAPTION"), required=True)
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--speed", type=float, default=2.0, help="world seconds per video second")
ap.add_argument("--pw", type=int, default=640)
a = ap.parse_args()

INK = (35, 27, 29); LIVE = (120, 47, 61); FAIL = (51, 58, 155); OK = (74, 90, 45); PAPER = (238, 244, 247)


class Stream:
    """Sequential frame access (indices only grow): keeps one decoded frame in memory."""

    def __init__(self, path, size):
        self.cap = cv2.VideoCapture(str(path)); self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.size = size
        self.i = -1; self.cur = None

    def get(self, i):
        i = max(0, min(i, self.n - 1))
        while self.i < i:
            ok, f = self.cap.read()
            if not ok:
                self.n = self.i + 1; break
            self.i += 1
            if self.i == i:
                self.cur = cv2.resize(f, self.size, interpolation=cv2.INTER_AREA)
        if self.cur is None:
            self.cur = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        return self.cur.copy()


class Panel:
    def __init__(self, d, cap, pw):
        d = Path(d); self.cap = cap; self.pw = pw
        self.mt = [json.loads(l)["world_t"] for l in open(d / "frames.jsonl")]
        cap0 = cv2.VideoCapture(str(d / "main.mp4")); w0, h0 = cap0.get(cv2.CAP_PROP_FRAME_WIDTH), cap0.get(cv2.CAP_PROP_FRAME_HEIGHT); cap0.release()
        self.h = int(pw * h0 / w0); self.hs = int(pw * 0.34)
        self.main = Stream(d / "main.mp4", (pw, self.h)); self.head = Stream(d / "head.mp4", (self.hs, self.hs))
        self.mt = self.mt[:self.main.n]
        f = d / "info.json"
        self.info = json.loads((f if f.exists() else d / "result.json").read_text())
        self.agent = not f.exists()
        ev = {e["kind"]: e["world_t"] for e in self.info["events"]}
        self.t_pre = ev.get("prelude_done", 30.0)
        self.t_end = self.mt[-1] if self.mt else 0.0
        self.events = [(e["world_t"], e["kind"].replace("_", " ")) for e in self.info["events"] if e["kind"].startswith(("person", "machine"))]
        for a in self.info.get("actions", []) if self.agent else []:
            tgt = a.get("target") if a.get("skill") != "wait" else f"{a.get('seconds', '')} s"
            self.events.append((a["world_t"], f"agent: {a['skill']} {tgt if tgt is not None else ''}".strip()))
        self.events.sort()

    def frame(self, t, pw):
        i = int(np.searchsorted(self.mt, t, side="right") - 1)
        m = self.main.get(i); h = self.h
        j = int(round(min(t, self.t_end) / 0.2))
        if self.head.n:
            hs = self.hs; hd = self.head.get(j)
            cv2.rectangle(hd, (0, 0), (hs - 1, hs - 1), (255, 255, 255), 3)
            m[h - hs - 8:h - 8, pw - hs - 8:pw - 8] = hd
            cv2.putText(m, "robot camera", (pw - hs - 4, h - hs - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        bar = np.full((58, pw, 3), PAPER, np.uint8)
        cv2.putText(bar, self.cap, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1, cv2.LINE_AA)
        busy = t < self.t_pre
        st = f"world t = {t:5.1f} s   " + ("robot busy with its chore, camera recording" if busy else ("agent in control" if t < self.t_end else "episode over"))
        cv2.putText(bar, st, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, LIVE if not busy else (110, 110, 110), 1, cv2.LINE_AA)
        last = [k for (te, k) in self.events if te <= t]
        if last:
            cv2.putText(m, last[-1], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if t >= self.t_end - 1e-6:
            ok = self.info.get("success")
            who = "agent" if self.agent else "oracle"
            txt = (f"{who}: {'SUCCESS' if ok else 'FAILURE'}, " + self.info.get("reason", "")) if ok is not None else ""
            cv2.putText(m, txt[:60], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 230, 120) if ok else (80, 80, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, m])


panels = [Panel(d, c, a.pw) for d, c in a.panel]
T = max(p.t_end for p in panels)
n = int(T / a.speed * a.fps) + a.fps
head_h = 70
W = a.pw * len(panels) + 10 * (len(panels) - 1)
PH = panels[0].h + 58
tmp = Path("/tmp") / (Path(a.out).stem + ".raw.mp4")
vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, head_h + PH))
for k in range(n):
    t = min(T, k / a.fps * a.speed)
    row = []
    for i, p in enumerate(panels):
        if i:
            row.append(np.full((PH, 10, 3), 255, np.uint8))
        row.append(p.frame(t, a.pw))
    body = np.hstack(row)
    top = np.full((head_h, W, 3), (120, 47, 61), np.uint8)
    cv2.putText(top, a.title + (f"   [{a.kind}]" if a.kind else ""), (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(top, a.sub or f"same scene, three seeds: the change that matters, a look-alike, and nothing   ({a.speed:.0f}x speed)",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 220, 245), 1, cv2.LINE_AA)
    vw.write(np.vstack([top, body]))
vw.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
                a.out], check=True)
tmp.unlink()
print("wrote", a.out, round(T, 1), "s world")
