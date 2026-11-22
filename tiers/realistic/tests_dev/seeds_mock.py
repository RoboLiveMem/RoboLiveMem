import sys
sys.argv = ["x"]
import run_mock as rm
sids = ["s01_put_away", "s02_annas_mug", "s03_vacuum_ball", "s04_salted_soup", "s05_rotary_reject", "s06_conveyor_lanes", "s07_press_lamp", "s08_parcel_locker", "s09_kitting_partner", "s10_cloche_pass"]
ns = rm.build_ns(sids)
built = {}
for sid in sids:
    sc = ns["SCENARIOS"][sid](); sc.make_cams(); sc.build(); built[sid] = sc
ns["BUILT"] = built
for sid in sids:
    row = []
    for s in range(8):
        f = rm.run(ns, sid, s, "live", "stale")
        row.append(f"{s}:{'Y' if f['success'] else 'n'}")
    print(sid, " ".join(row), flush=True)
