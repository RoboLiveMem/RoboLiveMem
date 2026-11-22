"""Part H decisive command and (a)/(b) attribution (2026-11-15). Classes: a = the change had started before the command
could start (obs + think time) but had not finished at the decision's observation (its newest frame); b = the change had
finished at or before the observation; c = the change started after the command started; n = no change in the episode;
x = no decisive command issued."""
import json, os, statistics

_D = None


def durations(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "durations.json")):
    """Change duration (s) per family|kind, measured with the scripted aware policy (median over seeds)."""
    global _D
    if _D is None:
        _D = {}
        if os.path.exists(path):
            for k, v in json.load(open(path)).items():
                fam, mem, kind = k.split("|")
                v = [x for x in v if x is not None]
                if v and mem == "hidden":
                    _D[f"{fam}|{kind}"] = statistics.median(v)
    return _D


def decisive(fam, tr):
    """Index into the trace of the decisive command, or None."""
    prev_held = None
    if fam == "mugs":
        c = [i for i, t in enumerate(tr) if (t.get("cmd") or {}).get("tool") == "place" and t.get("obj") == "tray"]
        return c[-1] if c else None
    for i, t in enumerate(tr):
        tool = (t.get("cmd") or {}).get("tool") if isinstance(t.get("cmd"), dict) else None
        obj = t.get("obj")
        if fam == "apple" and tool == "pick" and obj in ("cover_a", "cover_b"):
            return i
        if fam == "knife" and tool == "pick" and obj in ("cover_l", "cover_r", "knife"):
            return i
        if fam == "point" and tool == "pick" and str(obj).startswith("jar"):
            return i
        if fam in ("heat", "heatw", "heatdur") and tool == "place" and prev_held == "bowl":
            return i
        if fam in ("sugar", "sugar2") and ((tool == "pick" and obj == "tea") or (tool == "place" and prev_held == "jar")):
            return i
        if "held_after" in t:
            prev_held = t.get("held_after")
    return None


def classify(r):
    fam, s = r["family"], r.get("strict") or {}
    tr = r.get("trace") or []
    st = s.get("started")
    i = decisive(fam, tr)
    if st is None:
        return ("n" if i is not None else "x"), i, None
    if i is None:
        return "x", None, None
    D = durations().get(f"{fam}|{s.get('kind')}")
    if D is None:
        return "?", i, None
    end = st + D
    t = tr[i]
    t_obs, t_cmd = float(t["t"]), float(t["t"]) + float(t.get("think_s") or 0.0)
    cls = "b" if end <= t_obs + 1e-6 else ("a" if st < t_cmd else "c")
    return cls, i, round(end, 2)
