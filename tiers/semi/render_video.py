"""Example videos of Part H: a scripted robot under a fixed latency L, with the person's change during its thinking.
python render_video.py FAMILY MEMBER SEED POLICY(naive|oracle) L OUT.mp4 ["caption"]
Front camera at 12.5 frames per second of world time, played at 25 fps (2x); a bar shows world time and what the robot is
doing (thinking, acting) and when a person acts."""
import os
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import livemem2 as LM  # noqa: E402
from run_lm2 import ScriptedPolicy, to_call  # noqa: E402


def font(sz, bold=True):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def describe(cmd, m):
    t = cmd.get("tool")
    if t in ("pick", "place", "press") and m:
        lab = m["label"].replace(" (knob)", "")
        return {"pick": f"pick up the {lab}", "place": f"put it on the {lab}", "press": f"press the {lab}"}[t]
    if t == "wait":
        return f"wait {cmd.get('seconds', 1)} s"
    return str(t)


def main():
    fam, member, seed, kind, L, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]), sys.argv[6]
    caption = sys.argv[7] if len(sys.argv) > 7 else ""
    sc, ep = LM.make(fam, member, seed, "fixed")
    ep.lm_video = []
    ep.lm_status = "start"
    pol = ScriptedPolicy(kind)
    mem = {}
    for k in range(12):
        marks = sc.marks(ep)
        ep.lm_obs_t = ep.world.time
        cmd, _, _ = pol.decide(None, {"sc": sc, "ep": ep, "marks": marks, "mem": mem})
        call, m = to_call(cmd, marks)
        d = describe(cmd, m) if call else "nothing"
        ep.lm_status = f"robot thinking ({L:.0f} s), will: {d}"
        ep.lm_cmd_status = f"robot acting: {d}"
        if call is None or call["tool"] == "done":
            ep.execute({"tool": "observe", "args": {"images": False}}, think_s=min(L, 1.0))
            break
        ep.execute(call, think_s=L)
        ep._seen = None
        ep.lm_cmd_status = None
    ok = sc.success(ep)
    wrong = (fam == "knife" and any(c != sc.cover() for c in sc.lifted)) or (fam == "apple" and "cover_b" in sc.lifted)
    ep.lm_status = "done: " + ("failure (lifted the wrong cover first)" if ok and wrong else ("success" if ok else "failure"))
    ep.lm_cmd_status = None
    for _ in range(25):
        ep.hold(0.08)
    import imageio.v2 as imageio
    w = imageio.get_writer(out, fps=25, codec="libx264", quality=7, macro_block_size=1)
    f1, f2 = font(17), font(15, bold=False)
    cap = textwrap.wrap(caption, 78)
    for t, img, st in ep.lm_video:
        im = Image.fromarray(img).convert("RGB")
        H = im.height + 30 + 22 * max(len(cap), 1) + 8
        canvas = Image.new("RGB", (im.width, H), (250, 250, 250))
        canvas.paste(im, (0, 30))
        dr = ImageDraw.Draw(canvas)
        person = "a person is acting" in st
        dr.rectangle([0, 0, im.width, 30], fill=(185, 28, 28) if person else (30, 30, 30))
        dr.text((8, 6), f"t = {t:5.1f} s   {st}", fill=(255, 255, 255), font=f1)
        for i, line in enumerate(cap):
            dr.text((8, im.height + 34 + 22 * i), line, fill=(20, 20, 20), font=f2)
        w.append_data(np.asarray(canvas))
    w.close()
    print(out, len(ep.lm_video), "frames;", ep.lm_status)


if __name__ == "__main__":
    main()
