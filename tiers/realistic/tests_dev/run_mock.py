"""Exercise every scenario's logic with the mock simulator: stale plan and privileged oracle, paused and live."""
import sys, json, math, types, traceback
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
SRV = HERE.parent / "server"
sys.path.insert(0, str(HERE))
import mock_og

SPEC = {
    "anchors": {
        "s01": {"base": [0, 0, 0], "d": 0.40, "h": 0.76, "depth": 0.90},
        "s02": {"base": [10, 0, 0], "d": 0.40, "h": 0.76, "depth": 0.90},
        "s03": {"base": [20, 0, 0], "ball": [21.2, 0.3], "ball_base": [20.6, 0.3, 0], "vac": [21.0, -1.5, 1.57],
                "box": [20.5, -1.2, 0], "box_base": [20.5, -0.5, -1.57],
                "hides": [{"pos": [22.6, 0.9], "label": "under the sofa", "base": [21.9, 0.9, 0]},
                          {"pos": [22.4, -0.6], "label": "under the armchair", "base": [21.8, -0.6, 0]},
                          {"pos": [21.5, 1.8], "label": "under the TV stand", "base": [21.5, 1.2, 1.57]}],
                "cam": [[18.5, -1.0, 2.0], [21.2, 0.3, 0.2], 15.0]},
        "s04": {"base": [30, 0, 0], "d": 0.35, "h": 0.90, "pot": [0.62, 0.0]},
        "s05": [40, 0, 0], "s06": [50, 0, 0], "s07": [60, 0, 0], "s08": [70, 0, 0], "s09": [80, 0, 0],
        "s10": {"base": [90, 0, 0], "d": 0.40, "h": 0.95, "depth": 0.70},
    },
    "models": {k: "xxxxxx" for k in ["mug", "chopping_board", "tray", "stockpot", "salt_shaker", "pepper_mill", "apple", "fruit_bowl",
                                     "soccer_ball", "toy_box"]},
}


def load_spec():
    import json as _j
    sp = {"anchors": {}, "models": {}}
    for f in ["house.json", "industrial.json"]:
        d = _j.load(open(SRV / "scenes" / f))
        sp["anchors"].update(d["anchors"]); sp["models"].update(d["models"])
    sp["anchors"]["s10"] = SPEC["anchors"]["s10"]
    sp["models"].setdefault("pepper_grinder", "x")
    return sp


def build_ns(sids):
    global SPEC
    SPEC = load_spec()
    ns = dict(np=np, th=th, SCENE_SPEC=SPEC, sess=Path("/tmp"))
    for mod in ("core.py", "station.py", "props.py", "common.py"):
        exec(compile((SRV / mod).read_text(), mod, "exec"), ns)
    modns = []
    for sid in sids:
        m = dict(ns); exec(compile((SRV / "scenarios" / f"{sid}.py").read_text(), sid, "exec"), m); modns.append(m)
    objects = []
    for sid in sids:
        objects += ns["SCENARIOS"][sid].OBJECTS
    og, env, objs = mock_og.make_env(objects)
    ns["og"] = og; ns["env"] = env; ns["obj"] = lambda n: objs[n]

    def make_cam(name, w, h, eye, tgt, focal=None):
        s = mock_og.MockSensor(name, w, h)
        s.set_position_orientation(th.tensor(eye, dtype=th.float32), th.tensor(ns["look_at_quat"](eye, tgt), dtype=th.float32))
        return s
    ns["make_cam"] = make_cam

    class V:
        def __init__(self, *a, **k):
            self.n = 0

        def add(self, x):
            self.n += 1

        def close(self):
            pass
    ns["Video"] = V
    ns["rgb_of"] = lambda s: np.zeros((720, 720, 3), np.uint8)
    for m in modns:
        for k in ("og", "env", "obj", "make_cam", "Video", "rgb_of"):
            m[k] = ns[k]
    return ns


class Client:
    def __init__(self, ns, ep):
        self.ns, self.ep = ns, ep

    def obs(self, lat=0.0):
        self.ep.charge_latency(lat) if self.ep.started else None
        r = self.ep.obs(); self.ep.t_resp = 0.0
        return r

    def act(self, skill, target=None, lat=0.0, seconds=None):
        self.ep.charge_latency(lat)
        self.ep.act(skill, target, seconds)
        r = self.ep.obs(); self.ep.t_resp = 0.0
        return r


def stale_plan(sid):
    return {"s01_put_away": [("pick", "4"), ("place", "5")], "s02_annas_mug": [("pick", "1"), ("place", "4")],
            "s03_vacuum_ball": [("pick", "1"), ("place", "4")], "s04_salted_soup": [("salt", "1"), ("done", None)],
            "s05_rotary_reject": [("pick", "4"), ("place", "7")], "s06_conveyor_lanes": [("pick", "1"), ("place", "5")],
            "s07_press_lamp": [("wait", 10), ("wait", 10), ("wait", 10), ("wait", 10), ("unload", None)],
            "s08_parcel_locker": [("pick", "7"), ("place", "8")],
            "s09_kitting_partner": [("kit", "1"), ("kit", "2"), ("kit", "3"), ("kit", "4")],
            "s10_cloche_pass": [("pick", "1"), ("place", "5")]}[sid]


def oracle_plan(sid, sc, priv):
    """Privileged decisions after the world has settled."""
    if sid == "s01_put_away":
        a = priv["apple_at"]
        if a == "table":
            return [("pick", "4"), ("place", "5")]
        i = int(a[3:]) + 1
        return [("open", str(i)), ("pick", str(i)), ("place", "5")]
    if sid == "s02_annas_mug":
        k = list(sc.slots).index(priv["annas_mug_at"]) + 1
        return [("pick", str(k)), ("place", "4")]
    if sid == "s03_vacuum_ball":
        k = list(sc.slots).index(priv["ball_at"]) + 1
        return [("pick", str(k)), ("place", "4")]
    if sid == "s04_salted_soup":
        return [("done", None)] if priv["salt_portions"] >= 1 else [("salt", "1"), ("done", None)]
    if sid == "s05_rotary_reject":
        k = list(sc.slots).index(priv["defective_at"]) + 1
        return [("pick", str(k)), ("place", "7")]
    if sid == "s06_conveyor_lanes":
        k = list(sc.slots).index(priv["order_tote_at"]) + 1
        return [("pick", str(k)), ("place", "5")]
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
        k = list(sc.slots).index(priv["veg_at"]) + 1
        return [("pick", str(k)), ("place", "5")]


def run(ns, sid, seed, clock, policy, think=25.0):
    sc = ns["BUILT"][sid]
    ep = ns["Episode"](sc, seed, clock, Path(f"/tmp/mock/{sid}_{clock}_{seed}"), record=True)
    c = Client(ns, ep)
    r = c.obs()
    if policy == "stale":
        lat = think
        for sk, tg in stale_plan(sid):
            if ep.over:
                break
            if sk == "wait":
                c.act("wait", None, lat=lat, seconds=tg)
            else:
                c.act(sk, tg, lat=lat)
            lat = 8.0
    else:
        # oracle: thinks as long as the stale agent, then waits until helpers / machines are idle, then acts on privileged state
        c.act("wait", None, lat=think, seconds=1.0)
        for _ in range(12):
            busy = getattr(sc, "helper", None) is not None and sc.helper.gen is not None
            busy = busy or (sid == "s06_conveyor_lanes" and sc.mover.busy()) or (sid == "s05_rotary_reject" and (sc.schedule or sc.rot))
            if not busy:
                break
            c.act("wait", None, lat=0.0, seconds=5.0)
        for _ in range(30):
            if ep.over:
                break
            plan = oracle_plan(sid, sc, sc.privileged())
            sk, tg = plan[0]
            if sk == "wait":
                c.act("wait", None, lat=2.0, seconds=tg); continue
            for sk, tg in plan:
                if ep.over:
                    break
                c.act(sk, tg, lat=2.0)
            break
    if not ep.over:
        ep.act("done")
    f = ep.close()
    return f


def main():
    sids = sys.argv[1].split(",") if len(sys.argv) > 1 else ["s01_put_away", "s02_annas_mug", "s03_vacuum_ball", "s04_salted_soup",
                                                              "s05_rotary_reject", "s06_conveyor_lanes", "s07_press_lamp",
                                                              "s08_parcel_locker", "s09_kitting_partner", "s10_cloche_pass"]
    ns = build_ns(sids)
    built = {}
    for sid in sids:
        sc = ns["SCENARIOS"][sid](); sc.make_cams(); sc.build(); built[sid] = sc
    ns["BUILT"] = built
    seeds = [0, 1, 2, 3]
    table = {}
    for sid in sids:
        for policy in ["stale", "oracle"]:
            for clock in ["paused", "live"]:
                res = []
                for s in seeds:
                    try:
                        f = run(ns, sid, s, clock, policy)
                        res.append((f["success"], f["reason"][:60], f["hidden"]))
                    except Exception:
                        traceback.print_exc(); res.append((None, "ERROR", None))
                table[(sid, policy, clock)] = res
                print(f"{sid:22s} {policy:6s} {clock:6s} {sum(1 for r in res if r[0])}/{len(res)}  " +
                      " | ".join(f"{'Y' if r[0] else 'n'} {r[1]}" for r in res[:2]), flush=True)


if __name__ == "__main__":
    main()
