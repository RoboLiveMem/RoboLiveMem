"""Exercise the v3 scenarios with the mock simulator: stale, cautious and privileged-oracle policies, every variant."""
import sys, json, traceback
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
SRV = HERE.parent / "server"
sys.path.insert(0, str(HERE))
import mock_og


def build_ns(sids, scene):
    spec = json.load(open(SRV / "scenes" / scene))
    ns = dict(np=np, th=th, SCENE_SPEC=spec, sess=Path("/tmp"))
    for mod in ("core.py", "station.py", "props.py", "common.py", "v3.py"):
        exec(compile((SRV / mod).read_text(), mod, "exec"), ns)
    modns = []
    for sid in sids:
        m = dict(ns); exec(compile((SRV / "scenarios" / f"{sid}.py").read_text(), sid, "exec"), m); modns.append(m)
    objects = []
    for sid in sids:
        objects += ns["SCENARIOS"][sid].OBJECTS
    names = [o["name"] for o in objects]
    assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]
    og, env, objs = mock_og.make_env(objects, robot_names=("robot",))
    ns["og"] = og; ns["env"] = env; ns["obj"] = lambda n: objs[n]

    def make_cam(name, w, h, eye, tgt, focal=None):
        s = mock_og.MockSensor(name, w, h)
        s.set_position_orientation(th.tensor(eye, dtype=th.float32), th.tensor(ns["look_at_quat"](eye, tgt), dtype=th.float32))
        return s
    ns["make_cam"] = make_cam

    class V:
        def __init__(self, *a, **k):
            pass

        def add(self, x):
            pass

        def close(self):
            pass
    ns["Video"] = V
    ns["rgb_of"] = lambda s: np.zeros((720, 720, 3), np.uint8)
    for m in modns:
        for k in ("og", "env", "obj", "make_cam", "Video", "rgb_of"):
            m[k] = ns[k]
    return ns


def mark_of(sc, slot):
    for k in range(1, 12):
        if sc.slot_by_mark(str(k)) == slot:
            return str(k)
    raise KeyError(slot)


def plan(sc, policy):
    kind = type(sc).__mro__[1].__name__
    P = sc.privileged()
    if kind == "DoseOnce":
        if policy == "stale":
            return [("dose", "1"), ("done", None)]
        if policy == "cautious":
            return [("done", None)]
        return ([("dose", "1")] if P["count"][sc.RECEPTACLES[0][0]] == 0 else []) + [("done", None)]
    if kind == "TrackOne":
        dest = mark_of(sc, "dest")
        if policy == "stale":
            return [("pick", "1"), ("place", dest)]
        if policy == "cautious":
            return [("done", None)]
        return [("pick", mark_of(sc, P["target_at"])), ("place", dest)]
    if kind == "Stash":
        item, dest = mark_of(sc, "item"), mark_of(sc, "dest")
        if policy == "stale":
            return [("pick", item), ("place", dest)]
        if policy == "cautious":
            return [("open", "1"), ("pick", "1"), ("place", dest)]
        a = P["item_at"]
        if a == "item":
            return [("pick", item), ("place", dest)]
        k = mark_of(sc, a)
        return [("open", k), ("pick", k), ("place", dest)]
    if kind == "ZoneClear":
        if policy == "stale":
            return [("start", "1")]
        if policy == "cautious":
            return [("wait", 28.0), ("start", "1")]
        return [("wait", 2.0)] if P["occupied"] else [("start", "1")]


def run(ns, sid, seed, policy, visible, think=6.0):
    sc = ns["BUILT"][sid]
    ep = ns["Episode"](sc, seed, "live", Path(f"/tmp/mockv3/{sid}_{policy}_{seed}"), record=False, visible=visible)
    ep.obs(); ep.t_resp = 0.0
    lat = think
    for _ in range(80):
        if ep.over:
            break
        steps = plan(sc, policy)
        for sk, tg in steps:
            if ep.over:
                break
            ep.charge_latency(lat); lat = 2.0
            if sk == "wait":
                ep.act("wait", None, tg)
            else:
                ep.act(sk, tg)
            ep.obs(); ep.t_resp = 0.0
        if policy != "oracle" or not (steps and steps[0][0] == "wait"):
            break
    if not ep.over:
        ep.act("done")
    return ep.close()


def main():
    groups = {"house_v3.json": ["v3_sugar", "v3_mugs", "v3_keys"], "hall_v3.json": ["v3_totes", "v3_bins", "v3_reagent"]}
    for scene, sids in groups.items():
        ns = build_ns(sids, scene)
        built = {}
        for sid in sids:
            sc = ns["SCENARIOS"][sid](); sc.make_cams(); sc.build(); built[sid] = sc
        ns["BUILT"] = built
        for sid in sids:
            for visible in ((False, True) if __import__("os").environ.get("VIS", "1") == "1" else (False,)):
                for policy in tuple(__import__("os").environ.get("POL", "stale,cautious,oracle").split(",")):
                    res = []
                    for seed in range(int(__import__("os").environ.get("NSEED", "6"))):
                        try:
                            f = run(ns, sid, seed, policy, visible)
                            res.append(("Y" if f["success"] else "n", f["hidden"].get("variant"), f["reason"][:48],
                                        [e["kind"] for e in f["events"] if e["kind"].startswith("person")][:4],
                                        round(next((e["world_t"] for e in f["events"] if e["kind"] == "prelude_done"), -1), 1)))
                        except Exception:
                            traceback.print_exc(); res.append(("E", None, "ERROR", [], 0))
                    print(f"{sid:11s} vis={int(visible)} {policy:8s} " + " ".join(f"{r[1][:2] if r[1] else '??'}:{r[0]}" for r in res),
                          flush=True)
                    if policy == "oracle" and not visible:
                        for r in res[:3]:
                            print("     ", r)


if __name__ == "__main__":
    main()
