# dev job: render the marked current_view of the first observation (start state handed to the agent) for every scenario
# on this server, hidden seeds 0/1/2 (event/decoy/null) and visible seeds 0/1 where a twin exists -> sess/marks_check/
import base64, json
from pathlib import Path
out = Path(sess / "marks_check"); out.mkdir(parents=True, exist_ok=True)
HAS_VIS = {"v3_sugar", "v3_mugs", "v3_keys", "v3_totes", "v3_bins", "v3_reagent", "s01_put_away", "s07_press_lamp", "s08_parcel_locker"}
for sid in [s for s in BUILT if s != "v3_cell"] if not globals().get("MR_SIDS") else MR_SIDS:
    sc = BUILT[sid]
    for seed, vis in [(0, False), (1, False), (2, False)] + ([(0, True), (1, True)] if sid in HAS_VIS else []):
        og.sim.load_state(init_state, serialized=False)
        ep = Episode(sc, seed, "paused", out / "ep", record=False, visible=vis)
        p = ep.obs()
        (out / f"{sid}_s{seed}{'_vis' if vis else ''}.jpg").write_bytes(base64.b64decode(p["current_view"]))
        (out / f"{sid}_marks.json").write_text(json.dumps(p["marks"]))
        ep.close()
        print(sid, seed, vis, "ok", flush=True)
