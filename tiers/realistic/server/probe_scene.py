"""Load a scene and dump its object layout (no robots) to <out>/layout.json plus a top-down map.

    python probe_scene.py <scene_model> <out_dir> [room,room,...]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
scene, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rooms = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3].strip("'\"-") else None
with gm.unlocked():
    gm.HEADLESS = True; gm.ENABLE_OBJECT_STATES = False; gm.USE_GPU_DYNAMICS = False; gm.ENABLE_TRANSITION_RULES = False
t0 = time.time()
cfg = {"scene": {"type": "InteractiveTraversableScene", "scene_model": scene, **({"load_room_instances": rooms} if rooms else {})},
       "robots": [], "objects": []}
env = og.Environment(configs=cfg)
for _ in range(5):
    og.sim.step()
rows = []
for o in env.scene.objects:
    try:
        lo, hi = [np.asarray(v.tolist()) for v in o.aabb]
        room = None
        try:
            room = env.scene._seg_map.get_room_instance_by_point(((lo + hi) / 2)[:2])
        except Exception:
            pass
        rows.append(dict(name=o.name, category=getattr(o, "category", ""), lo=lo.round(3).tolist(), hi=hi.round(3).tolist(),
                         room=room, fixed=bool(getattr(o, "fixed_base", False)), n_joints=len(getattr(o, "joints", {}) or {})))
    except Exception as e:
        rows.append(dict(name=o.name, error=str(e)))
(out / "layout.json").write_text(json.dumps(dict(scene=scene, load_s=round(time.time() - t0, 1), objects=rows), indent=0))
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(22, 22))
    for r in rows:
        if "lo" not in r or r["category"] in ("floors", "ceilings", "lawn"):
            continue
        lo, hi = r["lo"], r["hi"]
        col = "k" if r["category"] == "walls" else ("tab:blue" if r["n_joints"] else "tab:orange")
        ax.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False, ec=col, lw=0.8 if col != "k" else 1.5))
        if r["category"] not in ("walls",):
            ax.text((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, f"{r['category'][:14]}\n{hi[2]:.2f}", fontsize=5, ha="center", va="center")
    ax.set_aspect("equal"); ax.autoscale(); ax.grid(True, lw=0.3)
    ax.set_xticks(np.arange(*np.floor(ax.get_xlim()), 1.0)); ax.set_yticks(np.arange(*np.floor(ax.get_ylim()), 1.0))
    fig.savefig(out / "map.png", dpi=110, bbox_inches="tight")
except Exception as e:
    print('no map', e)
print("DONE", len(rows), flush=True)
og.shutdown()
