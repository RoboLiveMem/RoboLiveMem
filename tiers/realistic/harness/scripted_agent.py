#!/usr/bin/env python3
"""Scripted agents for videos and validation (HTTP, fixed deliberation latencies).

    python3 scripted_agent.py --url http://127.0.0.1:8701 --scenario s02_annas_mug --seed 3 --clock live \
        --policy stale --think 25 --think2 8 --out /out/bench/videos/s02_live

stale  : decides from its first observation only (never reads the stream), like a policy whose memory is
         written once and never updated while it deliberates.
oracle : privileged; waits until helpers and machines are idle, then acts on the true state (solvability check).
"""
import argparse, json, time, urllib.request

STALE = {
    "s01_put_away": [("pick", "4"), ("place", "5")],
    "s02_annas_mug": [("pick", "1"), ("place", "4")],
    "s03_vacuum_ball": [("pick", "1"), ("place", "4")],
    "s04_salted_soup": [("salt", "1"), ("done", None)],
    "s05_rotary_reject": [("pick", "4"), ("place", "7")],
    "s06_conveyor_lanes": [("pick", "1"), ("place", "5")],
    "s08_parcel_locker": [("pick", "7"), ("place", "8")],
    "s09_kitting_partner": [("kit", "1"), ("kit", "2"), ("kit", "3"), ("kit", "4")],
    "s10_cloche_pass": [("pick", "1"), ("place", "5")],
}


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def act(url, skill, target, latency, tries=40):
    """/act that, when the safety interlock refuses it, waits 3 s and repeats the same command (same stale plan)."""
    r = post(url, "/act", dict(skill=skill, target=target, latency=latency))
    for _ in range(tries):
        if r.get("episode_over") or not str(r.get("last_result", "")).startswith("refused"):
            break
        post(url, "/act", dict(skill="wait", seconds=3.0, latency=1.0))
        r = post(url, "/act", dict(skill=skill, target=target, latency=1.0))
    return r


def slot_mark(spec_marks, label_prefix):
    for k, v in spec_marks.items():
        if v.startswith(label_prefix):
            return k


def oracle_actions(sid, priv, marks):
    names = list(marks.keys())
    def m(slot_index):
        return str(slot_index)
    if sid == "s01_put_away":
        a = priv["apple_at"]
        if a == "table":
            return [("pick", "4"), ("place", "5")]
        i = int(a[3:]) + 1
        return [("open", str(i)), ("pick", str(i)), ("place", "5")]
    if sid == "s02_annas_mug":
        return [("pick", {"spot1": "1", "spot2": "2", "spot3": "3"}[priv["annas_mug_at"]]), ("place", "4")]
    if sid == "s03_vacuum_ball":
        a = priv["ball_at"]
        return [("pick", "1" if a == "rug" else str(int(a[4:]) + 2)), ("place", "4")]
    if sid == "s04_salted_soup":
        return [("done", None)] if priv["salt_portions"] >= 1 else [("salt", "1"), ("done", None)]
    if sid == "s05_rotary_reject":
        return [("pick", str(int(priv["defective_at"][4:]) + 1)), ("place", "7")]
    if sid == "s06_conveyor_lanes":
        a = priv["order_tote_at"]
        return [("pick", "1" if a == "pick" else str(int(a[4:]) + 2)), ("place", "5")]
    if sid == "s07_press_lamp":
        return [("unload", None)] if priv["part"] == "ready" else [("wait", 5)]
    if sid == "s08_parcel_locker":
        a = priv["parcel_at"]
        if a == "drop":
            return [("pick", "7"), ("place", "8")]
        k = int(a[6:]) + 1
        return [("open", str(k)), ("pick", str(k)), ("place", "8")]
    if sid == "s09_kitting_partner":
        return [("kit", str(i + 1)) for i, c in enumerate(priv["parts_per_tote"]) if c == 0] + [("done", None)]
    if sid == "s10_cloche_pass":
        return [("pick", str(int(priv["veg_at"][4:]) + 1)), ("place", "5")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--clock", default="live")
    ap.add_argument("--policy", default="stale"); ap.add_argument("--think", type=float, default=25.0)
    ap.add_argument("--think2", type=float, default=8.0); ap.add_argument("--out", default=None)
    ap.add_argument("--record", type=int, default=1)
    a = ap.parse_args()
    spec = post(a.url, "/reset", dict(scenario=a.scenario, seed=a.seed, clock=a.clock, out=a.out, record=bool(a.record),
                                      privileged=(a.policy == "oracle")))
    marks = spec["marks"]
    r = post(a.url, "/obs", {})
    lat = a.think
    if a.policy == "stale":
        if a.scenario == "s07_press_lamp":
            # waits in 10 s steps and only notices the lamp if it lights during its own wait (it never reads the
            # frames recorded while it was deliberating); gives up after 4 waits
            seen = False
            for i in range(4):
                r = post(a.url, "/act", dict(skill="wait", seconds=10, latency=lat)); lat = a.think2
                t1 = r["world_time"]
                # it looks only at the frames of its own wait (the last 10 s of world time)
                if _lamp_in_wait(a.url, t1 - 10.0, t1):
                    seen = True; break
            if seen:
                post(a.url, "/act", dict(skill="unload", latency=lat))
        else:
            for sk, tg in STALE[a.scenario]:
                r = act(a.url, sk, tg, lat); lat = a.think2
                if r.get("episode_over"):
                    break
    else:
        r = post(a.url, "/act", dict(skill="wait", seconds=1.0, latency=lat))
        for _ in range(40):
            if r.get("episode_over"):
                break
            plan = oracle_actions(a.scenario, r["privileged"], marks)
            busy = "helper" in r["privileged"] and any(w in str(r["privileged"].get("helper", "")) for w in ("helper ", "courier", "partner", "expo", "vacuum"))
            if plan[0][0] == "wait" or busy:
                r = post(a.url, "/act", dict(skill="wait", seconds=5.0, latency=1.0)); continue
            for sk, tg in plan:
                r = act(a.url, sk, tg, 2.0)
                if r.get("episode_over"):
                    break
            break
    if not r.get("episode_over"):
        post(a.url, "/act", dict(skill="done", latency=2.0))
    final = post(a.url, "/end", {})
    print(json.dumps(dict(scenario=a.scenario, seed=a.seed, clock=a.clock, policy=a.policy, success=final["success"], reason=final["reason"],
                          hidden=final["hidden"], world_t=final["world_t"], agent_t=final["agent_t"])))


def _lamp_in_wait(url, t0, t1):
    ev = post(url, "/events", {})["events"]
    on = [e["world_t"] for e in ev if e["kind"] == "lamp_on"]
    off = [e["world_t"] for e in ev if e["kind"] == "lamp_off"] + [1e9]
    return any(a_ <= t1 and b_ >= t0 for a_, b_ in zip(on, off))


if __name__ == "__main__":
    main()
