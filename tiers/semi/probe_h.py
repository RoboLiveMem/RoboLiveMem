"""Ablation B, part H (2026-11-15): offline perception probe. For each selected live episode, the model gets the exact images
it had received up to its decisive command (every frame of decisions 1..k with its world time, and the current image of
decision k with its marks), with no time pressure, and one multiple-choice question about the decisive state; the answer
comes from the privileged state logged at decision k (never shown to the model).
usage: python probe_h.py OUT.jsonl TAG MODELS CONDS VARIANTS [THREADS]"""
import concurrent.futures as cf, glob, json, os, random, re, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attr_h import decisive
import livemem2_strict as LMS
import livemem2 as LM
from robolivemem.models import Chat as BedrockChat, parse_json_reply

MT = {"opus-5.5": 4096, "sonnet-5.5": 4096, "haiku-5.5": 4096, "gpt-6-astra-med": 16000, "gpt-6-luna": 20000}


def question(fam, priv, member="hidden"):
    """(question, options, truth letter) from the privileged text, or None."""
    p = priv or ""
    if fam == "mugs":
        if "robot's hand is Anna's" in p:
            return "Whose mug is the robot holding now?", ["Anna's", "Ben's"], "A"
        if "robot's hand is Ben's" in p:
            return "Whose mug is the robot holding now?", ["Anna's", "Ben's"], "B"
        m = re.search(r"Anna's mug is now the (LEFT|RIGHT)", p)
        return ("Which of the two mugs is Anna's now?", ["the LEFT one", "the RIGHT one"], "A" if m.group(1) == "LEFT" else "B") if m else None
    if fam == "knife":
        opts = ["under the LEFT grey cover", "under the RIGHT grey cover", "on the plate", "in the robot's hand", "on the counter, in view"]
        for i, o in enumerate(opts):
            if o in p:
                return "Where is the knife now?", opts, "ABCDE"[i]
        return None
    if fam == "apple":
        if "not under a cover" in p:
            return "Where is the apple now?", ["on the LEFT covered dish", "on the RIGHT covered dish", "not under a cover"], "C"
        m = re.search(r"on the (LEFT|RIGHT) covered dish", p)
        return ("Where is the apple now?", ["on the LEFT covered dish", "on the RIGHT covered dish", "not under a cover"],
                "A" if m.group(1) == "LEFT" else "B") if m else None
    if fam == "point":
        opts = ["the LEFT jar", "the MIDDLE jar", "the RIGHT jar", "she has not shown a jar yet"]
        if "not shown a jar" in p:
            return "Which of the three spice jars did the cook show?", opts, "D"
        m = re.search(r"showed the (LEFT|MIDDLE|RIGHT) jar", p)
        return ("Which of the three spice jars did the cook show?", opts, "ABC"[["LEFT", "MIDDLE", "RIGHT"].index(m.group(1))]) if m else None
    if fam in ("heat", "heatw", "heatdur"):
        opts = ["hot", "cold"]
        if "HAS been switched on" in p:
            return "Is the hot plate hot or cold now, according to the task's rule?", opts, "A"
        if "has NOT been switched on" in p:
            return "Is the hot plate hot or cold now, according to the task's rule?", opts, "B"
        return None
    if fam in ("sugar", "sugar2"):
        m = re.search(r"tea has (\d+) sugar", p)
        if not m:
            return None
        n = int(m.group(1))
        if member == "visible":      # the twin's cubes lie on the saucer beside the mug (each one counts for the tea)
            return ("How many sugar cubes has the cook given for Ben's tea so far (cubes on the saucer beside his mug)?",
                    ["0", "1", "2", "3 or more"], "ABCD"[min(n, 3)])
        return "How many sugar cubes are in Ben's tea now?", ["0", "1", "2", "3 or more"], "ABCD"[min(n, 3)]
    return None


def tile(frames, per_row=3, rows=2):
    """frames: [(t, img)] -> list of tiled images (per_row x rows), each frame labelled with its time."""
    out = []
    n = per_row * rows
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for c in range(0, len(frames), n):
        chunk = frames[c:c + n]
        h, w = chunk[0][1].shape[:2]
        W = Image.new("RGB", (w * per_row, (h + 20) * ((len(chunk) + per_row - 1) // per_row)), (255, 255, 255))
        d = ImageDraw.Draw(W)
        for k, (t, img) in enumerate(chunk):
            x, y = w * (k % per_row), (h + 20) * (k // per_row)
            d.text((x + 4, y + 2), f"t = {t:.1f} s", fill=(0, 0, 0), font=font)
            W.paste(Image.fromarray(img), (x, y + 20))
        out.append(np.asarray(W))
    return out


def load_obs(path):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    imgs = [z[f"shape{j}"][i] for j, i in meta["where"]]
    return meta["dec"], imgs


SYS = ("You are shown what a robot's camera recorded while it worked on a kitchen task, up to one of its decisions. There is "
       "no time pressure now: the scene is frozen. Answer the question about the scene at the time of the current image. "
       "Reply with ONE JSON object and nothing else: {\"answer\": \"<letter>\"}")


def probe(r, chat):
    fam = r["family"]
    tr = r["trace"]
    i = decisive(fam, tr)
    if i is None:
        return {"job_id": r["job_id"], "skip": "no decisive command"}
    k = tr[i]["k"]
    q = question(fam, tr[i].get("priv"), r["member"])
    if q is None:
        return {"job_id": r["job_id"], "skip": "no question", "priv": tr[i].get("priv")}
    dec, imgs = load_obs(r["obs_file"])
    fr = {}
    for kk in sorted(int(x) for x in dec):
        if kk > k:
            break
        for t, ix in dec[str(kk)]["frames"]:
            fr[round(float(t), 2)] = ix
    frames = [(t, imgs[ix]) for t, ix in sorted(fr.items())]
    dk = dec[str(k)]
    marks = [{"id": m[0], "px": m[3]} for m in dk["marks"]]
    cur = LM.draw_marks(imgs[dk["cur"]], marks)
    sc = LMS.FAMILIES[fam](r["member"], int(r["seed"]))
    goal = sc.goal_text()
    qq, opts, truth = q
    parts = [f"The robot's task: {goal}",
             f"Camera frames the robot received up to this decision ({len(frames)} frames, two per second of world time, "
             f"oldest first, tiled six per image, each labelled with its time):"] + tile(frames) + [
        f"Current camera image (world time {tr[i]['t']:.1f} s), with the numbered marks the robot saw:", cur,
        "Question: " + qq + "\n" + "\n".join(f"{'ABCDE'[j]}) {o}" for j, o in enumerate(opts)) + "\nReply with the JSON object."]
    for tries in range(4):
        text, meta = chat(SYS, [("user", parts)])
        if meta.get("error") and (meta.get("http_status") in (429, 500, 502, 503, 504, None) or "throttl" in str(meta.get("error")).lower()):
            time.sleep(3 * (tries + 1) + random.random() * 3)
            continue
        break
    rep = parse_json_reply(text) or {}
    ans = str(rep.get("answer", "")).strip().upper()[:1]
    return {"job_id": r["job_id"], "family": fam, "model": r["policy"], "member": r["member"], "kind": (r.get("strict") or {}).get("kind"),
            "k": k, "question": qq, "options": opts, "truth": truth, "answer": ans, "correct": ans == truth,
            "n_frames": len(frames), "latency_s": meta.get("latency_s"), "error": meta.get("error"), "reply": (text or "")[:300]}


def main():
    out, tags, models, conds, variants = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4].split(","), sys.argv[5].split(",")
    nt = int(sys.argv[6]) if len(sys.argv) > 6 else 12
    done = set()
    if os.path.exists(out):
        for l in open(out):
            done.add(json.loads(l)["job_id"])
    rows = []
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(os.environ.get("RESULTS_DIR", "results"), f"{tag}_s*.jsonl"))):
            for l in open(f):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                s = r.get("strict") or {}
                if (r.get("success") is not None and r.get("clock") == "live" and r["policy"] in models and r["member"] in conds
                        and s.get("kind") in variants and r.get("obs_file") and r["job_id"] not in done):
                    rows.append(r)
    print(len(rows), "episodes to probe", flush=True)
    chats = {m: BedrockChat(m, max_tokens=MT.get(m, 4096), timeout_s=300.0) for m in models}
    with cf.ThreadPoolExecutor(nt) as ex, open(out, "a") as fo:
        futs = {ex.submit(probe, r, chats[r["policy"]]): r for r in rows}
        for fu in cf.as_completed(futs):
            try:
                res = fu.result()
            except Exception as e:
                res = {"job_id": futs[fu]["job_id"], "skip": f"error {type(e).__name__}: {e}"[:300]}
            fo.write(json.dumps(res) + "\n")
            fo.flush()


if __name__ == "__main__":
    main()
