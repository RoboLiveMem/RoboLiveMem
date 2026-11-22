"""Side-by-side paused | live video of one RoboLiveMem-Bench scenario.

    python compose_bench.py <paused_episode_dir> <live_episode_dir> <out.mp4> <config.json>

Both episodes come from the same agent (same decisions and the same deliberation latencies). The video
follows the agent's clock: deliberation is shown 6x faster, robot motion 1.5x. While the agent thinks,
the paused panel is dimmed and marked TIME FROZEN; a thought bubble spins above the robot in both panels.
"""
import json, sys, subprocess, math
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

FD = "/usr/share/fonts/opentype/inter/"


def F(w, s):
    return ImageFont.truetype(FD + {"r": "Inter-Regular.otf", "m": "Inter-Medium.otf", "s": "Inter-SemiBold.otf", "b": "Inter-Bold.otf"}[w], s)


INK, MUTED, BG = (32, 35, 41), (107, 114, 128), (250, 250, 248)
PAUSED, LIVE, FAIL, OK, AMBER, ICE = (96, 104, 118), (61, 47, 120), (170, 52, 44), (40, 120, 84), (176, 128, 40), (120, 170, 220)
W, H, FPS = 1920, 1080, 30
PW, PH, MX, GAP, Y0 = 912, 513, 32, 32, 176
THINK_SPEED, ACT_SPEED = 6.0, 1.5


class Vid:
    """Random access by frame index (sequential reads with a small cache)."""

    def __init__(self, path):
        self.path = path; self.cap = cv2.VideoCapture(str(path)); self.i = -1; self.img = None

    def get(self, i):
        if i < self.i:
            self.cap.release(); self.cap = cv2.VideoCapture(str(self.path)); self.i = -1
        while self.i < i:
            ok, fr = self.cap.read()
            if not ok:
                break
            self.i += 1; self.img = fr
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)


def mmss(t):
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def pill(d, xy, text, fill, fg=(255, 255, 255), font=None):
    font = font or F("s", 21); x, y = xy
    tw = d.textlength(text, font=font)
    d.rounded_rectangle([x, y, x + tw + 26, y + 36], radius=18, fill=fill)
    d.text((x + 13, y + 6), text, font=font, fill=fg)
    return x + tw + 26


def bubble(d, cx, cy, phase, scale=1.0):
    """Thought bubble with three spinning dots and two trailing circles."""
    r = 34 * scale
    bx, by = cx + 30 * scale, cy - 70 * scale
    d.ellipse([cx + 2 * scale, cy - 18 * scale, cx + 12 * scale, cy - 8 * scale], fill=(255, 255, 255), outline=INK, width=2)
    d.ellipse([cx + 10 * scale, cy - 36 * scale, cx + 26 * scale, cy - 20 * scale], fill=(255, 255, 255), outline=INK, width=2)
    d.ellipse([bx - r * 1.3, by - r, bx + r * 1.3, by + r], fill=(255, 255, 255), outline=INK, width=3)
    for k in range(3):
        a = phase * 2 * math.pi + k * 2 * math.pi / 3
        x = bx + 18 * scale * math.cos(a); y = by + 12 * scale * math.sin(a)
        rr = 6 * scale
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=LIVE)
    d.text((bx - 6 * scale, by - 15 * scale), "?", font=F("b", int(26 * scale)), fill=INK)


def load(ep):
    ep = Path(ep)
    rows = [json.loads(l) for l in open(ep / "frames.jsonl")]
    res = json.load(open(ep / "result.json"))
    thinks = [(x["agent_t0"], x["agent_t1"]) for x in res["timeline"] if x["kind"] == "think" and x["latency"] > 0.5]
    return dict(rows=rows, res=res, thinks=thinks, main=Vid(ep / "main.mp4"), head=Vid(ep / "head.mp4"),
                at=np.array([r["agent_t"] for r in rows]), ev=res["events"])


def think_at(S, t):
    for a, b in S["thinks"]:
        if a <= t < b:
            return (a, b)
    return None


def main(p_dir, l_dir, out, cfg_path):
    cfg = json.load(open(cfg_path))
    S = {"paused": load(p_dir), "live": load(l_dir)}
    t_end = max(S[m]["res"]["agent_t"] for m in S) + 0.5
    ts, t = [], 0.0
    while t <= t_end:
        ts.append(t)
        thinking = think_at(S["paused"], t) or think_at(S["live"], t)
        t += (THINK_SPEED if thinking else ACT_SPEED) / FPS
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart",
                           str(out)], stdin=subprocess.PIPE)
    base = Image.new("RGB", (W, H), BG); d0 = ImageDraw.Draw(base)
    d0.text((MX, 22), cfg["title"], font=F("b", 38), fill=INK)
    d0.text((MX, 72), cfg["task"], font=F("r", 22), fill=MUTED)
    d0.text((MX, 102), cfg["change"], font=F("r", 22), fill=MUTED)
    for k, (m, col, head) in enumerate([("paused", PAUSED, "PAUSED CLOCK  ·  the world waits while the agent thinks"),
                                         ("live", LIVE, "LIVE CLOCK  ·  the world keeps running while the agent thinks")]):
        x0 = MX + k * (PW + GAP)
        d0.rectangle([x0, 140, x0 + PW, 144], fill=col)
        d0.text((x0, 148), head, font=F("s", 20), fill=col)
    d0.text((MX, H - 32), cfg.get("footer", "Same agent, same decisions and deliberation times in both panels. Thinking shown 6x faster, robot motion 1.5x."),
            font=F("r", 17), fill=MUTED)
    idx = {m: 0 for m in S}
    hold = 4 * FPS
    for n in range(len(ts) + hold):
        t = ts[min(n, len(ts) - 1)]; final = n >= len(ts)
        im = base.copy(); d = ImageDraw.Draw(im)
        for k, m in enumerate(["paused", "live"]):
            s = S[m]; rows = s["rows"]
            j = int(np.searchsorted(s["at"], t, side="right") - 1); j = max(0, min(j, len(rows) - 1))
            row = rows[j]
            frame = s["main"].get(row["i"])
            x0 = MX + k * (PW + GAP)
            pan = Image.fromarray(cv2.resize(frame, (PW, PH), interpolation=cv2.INTER_AREA))
            th_ = think_at(s, t)
            frozen = m == "paused" and th_ is not None
            if frozen:
                arr = np.asarray(pan).astype(np.float32)
                arr = arr * 0.45 + np.array(ICE, np.float32) * 0.12
                pan = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            im.paste(pan, (x0, Y0))
            # robot camera inset (world-time indexed, 5 Hz)
            hi = int(round(row["world_t"] * 5))
            try:
                hv = s["head"].get(hi)
                hv = cv2.resize(hv, (220, 220), interpolation=cv2.INTER_AREA)
                if frozen:
                    hv = (hv * 0.5).astype(np.uint8)
                ix, iy = x0 + 12, Y0 + PH - 232
                d.rectangle([ix - 3, iy - 3, ix + 222, iy + 222], fill=(255, 255, 255))
                im.paste(Image.fromarray(hv), (ix, iy))
                lab = "robot's camera"; d.rectangle([ix, iy, ix + d.textlength(lab, font=F("s", 15)) + 12, iy + 22], fill=(255, 255, 255))
                d.text((ix + 6, iy + 2), lab, font=F("s", 15), fill=INK)
            except Exception:
                pass
            if th_ is not None and row.get("head_px"):
                hx, hy = row["head_px"]
                bx_ = min(max(x0 + hx * PW / 1280, x0 + 40), x0 + PW - 160)
                by_ = min(max(Y0 + hy * PH / 720, Y0 + 115), Y0 + PH - 40)
                bubble(d, bx_, by_, (t * 0.35) % 1.0, 1.0)
            if frozen:
                bx = x0 + PW // 2
                d.rounded_rectangle([bx - 150, Y0 + 18, bx + 150, Y0 + 70], radius=26, fill=(235, 244, 252), outline=ICE, width=3)
                cx_, cy_ = bx - 112, Y0 + 44
                d.ellipse([cx_ - 15, cy_ - 15, cx_ + 15, cy_ + 15], outline=(40, 80, 130), width=3)
                d.line([cx_, cy_, cx_, cy_ - 10], fill=(40, 80, 130), width=3); d.line([cx_, cy_, cx_ + 8, cy_], fill=(40, 80, 130), width=3)
                d.text((bx - 88, Y0 + 26), "TIME FROZEN", font=F("b", 28), fill=(40, 80, 130))
            # clocks and status
            col = PAUSED if m == "paused" else LIVE
            d.rounded_rectangle([x0 + PW - 300, Y0 + 12, x0 + PW - 12, Y0 + 84], radius=10, fill=(255, 255, 255))
            d.text((x0 + PW - 288, Y0 + 18), f"agent clock  {mmss(t)}", font=F("m", 20), fill=INK)
            d.text((x0 + PW - 288, Y0 + 48), f"world clock  {mmss(row['world_t'] if not th_ or m == 'live' else row['world_t'])}",
                   font=F("s", 20), fill=col if m == "live" else PAUSED)
            yb = Y0 + PH + 14
            if final:
                ok = s["res"]["success"]
                pill(d, (x0, yb), ("SUCCESS" if ok else "FAILURE"), OK if ok else FAIL, font=F("b", 22))
                d.text((x0, yb + 46), s["res"]["reason"], font=F("m", 20), fill=INK)
            elif th_ is not None:
                pill(d, (x0, yb), f"THINKING  {t - th_[0]:4.0f} / {th_[1] - th_[0]:.0f} s", LIVE if m == "live" else PAUSED)
                if m == "paused":
                    d.text((x0, yb + 46), "Nothing moves until the agent acts.", font=F("r", 19), fill=MUTED)
                else:
                    d.text((x0, yb + 46), "The world keeps changing; the robot's camera keeps recording.", font=F("r", 19), fill=MUTED)
            else:
                lab = row.get("label") or ""
                if lab.startswith("interlock"):
                    pill(d, (x0, yb), "REFUSED  safety interlock: robot holds still", MUTED)
                elif lab:
                    pill(d, (x0, yb), "ACTING  " + lab, AMBER)
            # event caption: the most recent event of the last 8 s of agent time (one line, no overlaps)
            cap = None
            for e in s["ev"]:
                txt = cfg.get("events", {}).get(e["kind"])
                if e["kind"] == "refused" and not txt:
                    txt = "command " + str(e.get("skill")) + " " + str(e.get("target") or "") + " refused: another robot or a machine is working there"
                if txt and 0 <= t - e["agent_t"] < 8.0 and not final:
                    cap = txt
            if cap:
                d.text((x0, yb + 76), "• " + cap, font=F("s", 19), fill=LIVE if m == "live" else PAUSED)
        ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close(); ff.wait()
    print("wrote", out, len(ts) + hold, "frames")


if __name__ == "__main__":
    main(*sys.argv[1:5])
