"""Re-simulate a recorded model episode at video frame rate (the simulator is deterministic given the commands and the
measured latencies) and write an MP4 with the model's status and notes.
python replay_video.py 'results glob[,glob]' JOB_ID OUT.mp4 [LABEL]
Paused episodes: the model's thinking time is shown as frozen frames (the world waits); live: the world runs."""
import glob
import json
import os
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import livemem2 as LM  # noqa: E402
from run_lm2 import to_call  # noqa: E402

EVERY = 4                     # ticks per video frame (12.5 frames per second of world time; played at 25 fps = 2x)
NAMES = {"gpt-6-astra": "GPT-6 Astra", "opus-5.5": "Claude Opus 5.5", "opus-5.5-think": "Opus 5.5 (thinking)",
         "gpt-6-luna": "GPT-6 Luna", "gpt-6.1-sol": "GPT-6.1 Sol", "haiku-5.5": "Claude Haiku 5.5", "fable-5.1": "Claude Fable 5.1"}


def font(sz, bold=True):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def describe(cmd, m):
    t = (cmd or {}).get("tool")
    if t in ("pick", "place", "press") and m:
        lab = m["label"].replace(" (knob)", "")
        return {"pick": f"pick up the {lab} [{m['id']}]", "place": f"put it on the {lab} [{m['id']}]",
                "press": f"press the {lab} [{m['id']}]"}[t]
    if t == "wait":
        return f"wait {cmd.get('seconds', 1)} s"
    return str(t)


def main():
    pats, jid, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else ""
    row = None
    for p in pats:
        for f in glob.glob(p):
            for x in open(f):
                if jid in x:
                    r = json.loads(x)
                    if r.get("job_id") == jid:
                        row = r
    assert row is not None, jid
    fam, member, seed, clock = row["family"], row["member"], int(row["seed"]), row["clock"]
    sc, ep = LM.make(fam, member, seed, clock)
    ep.lm_video, ep.lm_video_every = [], EVERY
    model = NAMES.get(row["policy"], row["policy"])
    note = ""
    frames = ep.lm_video                      # (t, img, status, note)
    ep.lm_status = "start"

    def push_frozen(seconds, status):
        img = ep.world.render("front")
        for _ in range(max(int(round(seconds / (EVERY * LM.TICK))), 1)):
            frames.append((round(ep.world.time, 2), img, status, note))

    for rec in row["trace"]:
        marks = sc.marks(ep)
        cmd = rec.get("cmd")
        call, m = to_call(cmd, marks) if isinstance(cmd, dict) else (None, None)
        note = (cmd or {}).get("note", "") if isinstance(cmd, dict) else ""
        lat = float((rec.get("meta") or {}).get("latency_s") or 0.0)
        d = describe(cmd, m) if call else ("done" if isinstance(cmd, dict) and cmd.get("tool") == "done" else "invalid reply")
        n0 = len(frames)
        if clock == "paused":
            push_frozen(lat, f"PAUSED (world frozen) | model thinks {lat:.1f} s  ->  {d}")
        else:
            ep.lm_status = f"LIVE (world keeps running) | model thinks {lat:.1f} s  ->  {d}"
        ep.lm_cmd_status = f"acting: {d}"
        if call is None and not (isinstance(cmd, dict) and cmd.get("tool") == "done"):
            ep.execute({"tool": "observe", "args": {"images": False}}, think_s=rec.get("think_s", lat))   # a wasted decision
            ep._seen = None
            continue
        if call is None or call["tool"] == "done":
            break
        ep.execute(call, think_s=rec.get("think_s", lat))
        ep._seen = None
        ep.lm_cmd_status = None
        # attach the note to the frames captured during this decision
        for i in range(n0, len(frames)):
            fr = frames[i]
            if len(fr) == 3:
                frames[i] = (fr[0], fr[1], fr[2], note)
    ok = bool(sc.success(ep))
    wrong = (fam == "knife" and any(c != sc.cover() for c in sc.lifted)) or (fam == "apple" and "cover_b" in sc.lifted)
    verdict = "success" if ok and not wrong else ("failure: lifted the wrong cover first" if ok else "failure")
    ep.lm_status = f"result: {verdict}"
    for _ in range(30):
        ep.hold(EVERY * LM.TICK)
    for i in range(len(frames)):
        if len(frames[i]) == 3:
            frames[i] = (frames[i][0], frames[i][1], frames[i][2], note if i < len(frames) - 30 else "")
    import imageio.v2 as imageio
    w = imageio.get_writer(out, fps=25, codec="libx264", quality=7, macro_block_size=1)
    f1, f2, f3 = font(16), font(14, bold=False), font(14)
    for t, img, st, nt in frames:
        im = Image.fromarray(img).convert("RGB")
        W = im.width
        canvas = Image.new("RGB", (W, im.height + 30 + 30 + 64), (248, 248, 246))
        dr = ImageDraw.Draw(canvas)
        top = (30, 30, 30)
        if "a person is acting" in st:
            top = (185, 28, 28)
        elif st.startswith("PAUSED"):
            top = (75, 85, 99)
        elif st.startswith("result: success"):
            top = (22, 101, 52)
        elif st.startswith("result"):
            top = (153, 27, 27)
        dr.rectangle([0, 0, W, 30], fill=top)
        dr.text((8, 6), f"{model} | {clock.upper()} | t = {t:5.1f} s", fill=(255, 255, 255), font=f1)
        canvas.paste(im, (0, 30))
        y = 30 + im.height
        dr.rectangle([0, y, W, y + 30], fill=(235, 235, 232))
        dr.text((8, y + 6), st.replace(" | a person is acting", "   [a person is acting]")[:92], fill=(20, 20, 20), font=f3)
        if nt:
            for i, line in enumerate(textwrap.wrap("model's note: " + nt, 84)[:3]):
                dr.text((8, y + 34 + 19 * i), line, fill=(60, 60, 60), font=f2)
        w.append_data(np.asarray(canvas))
    w.close()
    info = {"out": out, "job_id": jid, "model": model, "family": fam, "member": member, "seed": seed, "clock": clock,
            "frames": len(frames), "world_s": round(ep.world.time, 1), "replayed": verdict,
            "recorded_success": row["success"], "label": label,
            "decisions": [{"latency_s": round(float((r.get("meta") or {}).get("latency_s") or 0.0), 2),
                           "cmd": r.get("cmd")} for r in row["trace"]]}
    with open(out.rsplit(".", 1)[0] + ".json", "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    print(json.dumps({k: info[k] for k in ("out", "frames", "replayed", "recorded_success", "label")}))


if __name__ == "__main__":
    main()
