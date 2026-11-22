# dev job: scripted calibration of the ablation-A commit check (live clock). For each scenario x seed (hidden; visible
# event/decoy seeds where a twin exists):
#   stale+commit: plan fixed at the first observation, 3 s per decision; when a command comes back "not executed"
#                 (commit check), it looks again and re-plans from the true state (a perfect re-decision), 3 s per decision.
# Reports aborts, the largest changed-pixel fraction per check, and success. Null seeds measure false aborts.
import json, time
from pathlib import Path
CHK_DEFS_ONLY = True
exec(compile(open('/work/harness/main_check.py').read(), '/work/harness/main_check.py', 'exec'), globals())
CHK_DEFS_ONLY = False
cal_root = Path(globals().get("CAL_OUT") or (sess / "commit_cal")); cal_root.mkdir(parents=True, exist_ok=True)
cal_sids = globals().get("CAL_SIDS") or [s for s in BUILT if s != "v3_cell"]
cal_seeds = globals().get("CAL_SEEDS") or [0, 1, 2, 3, 4, 5]


def plan_true(sc):
    if sc.sid.startswith("v3_"):
        return v3_plan(sc, "oracle")
    return [(sk, None if slot is None else _mark_of(sc, slot)) for sk, slot in v2s_oracle(sc)]


def plan_stale(sc):
    if sc.sid.startswith("v3_"):
        return v3_plan(sc, "stale")
    return [(sk, None if slot is None else _mark_of(sc, slot)) for sk, slot in STALE[sc.sid]]


def cal_one(sid, seed, vis):
    sc = BUILT[sid]; t0 = time.time()
    og.sim.load_state(init_state, serialized=False)
    ep = Episode(sc, seed, "live", cal_root / f"{sid}{'_vis' if vis else ''}_s{seed}", record=False, visible=vis, commit=True)
    ep.obs(); ep.t_resp = time.time()
    plan = plan_stale(sc); i = 0; replans = 0
    while i < len(plan) and not ep.over:
        sk, tg = plan[i]
        ep.charge_latency(LAT); ep.act(sk, tg); ep.obs(); ep.t_resp = time.time()
        if ep.actions and ep.actions[-1].get("aborted") and replans < 5:
            replans += 1
            ep.charge_latency(LAT)
            while busy(sc) and not ep.over:
                ep.act("wait", None, 2.0); ep.obs(); ep.t_resp = time.time(); ep.charge_latency(LAT)
            plan = plan_true(sc); i = 0
            continue
        i += 1
    if not ep.over:
        ep.act("done")
    f = ep.close(); h = f["hidden"]
    return dict(sid=sid, seed=seed, variant=h.get("variant"), visible=vis, success=f["success"], reason=f["reason"],
                aborts=f.get("commit_aborts"), checks=f.get("commit_log"), secs=round(time.time() - t0, 1))


cal_rows = []
for sid in cal_sids:
    combos = [(s, False) for s in cal_seeds] + ([(s, True) for s in (0, 1, 3, 4)] if sid in HAS_VIS else [])
    for seed, vis in combos:
        try:
            cal_rows.append(cal_one(sid, seed, vis))
        except Exception:
            import traceback
            cal_rows.append(dict(sid=sid, seed=seed, visible=vis, error=traceback.format_exc()[-800:]))
        print(json.dumps(cal_rows[-1])[:300], flush=True)
        (cal_root / f"cal_{'+'.join(cal_sids)}.json").write_text(json.dumps(cal_rows, indent=1))
