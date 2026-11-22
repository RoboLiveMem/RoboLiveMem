"""Measure the change's start (t0) and end (first tick with sc.done) per family x seed kind, with the scripted aware policy
(fixed clock L=3). Analysis only; patches classes in this process."""
import os
import json, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_lm2 as R
import livemem2 as LM
import livemem2_strict as LMS
orig_tick = LM.LiveEpisode._tick
def _tick(self, row):
    orig_tick(self, row)
    sc = self.lm_sc
    if getattr(sc, "done", False) and getattr(sc, "_t_done", None) is None and getattr(sc, "t0", None) is not None:
        sc._t_done = self.world.time
LM.LiveEpisode._tick = _tick
orig_info = LMS.StrictMixin.strict_info
LMS.StrictMixin.strict_info = lambda self: {**orig_info(self), "t_done": getattr(self, "_t_done", None)}
out = collections.defaultdict(list)
for fam in sys.argv[1].split(","):
    for seed in range(int(sys.argv[2])):
        for member in ("hidden", "visible"):
            job = dict(job_id="m", family=fam, member=member, seed=seed, clock="fixed", L=3.0, policy="scripted:aware",
                       protocol="strict", person=True, max_frames=30)
            try:
                r = R.run_episode(job, R.StrictScriptedPolicy("aware"), None, 12, 30)
            except Exception as e:
                print(fam, seed, member, "ERR", e); continue
            s = r["strict"]
            d = None if (s["started"] is None or s.get("t_done") is None) else round(s["t_done"] - s["started"], 2)
            out[f"{fam}|{member}|{s['kind']}"].append(d)
            print(fam, member, seed, s["kind"], s["started"], s.get("t_done"), d, r["success"], flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
