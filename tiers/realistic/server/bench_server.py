"""RoboLiveMem-Bench scenario server (runs inside the BEHAVIOR / OmniGibson container).

    python bench_server.py --session /out/bench/<name> --scene /work/bench/scenes/<scene>.json \
                           --scenarios s01_pantry_bins,s02_annas_mug --port 8701

Loads one OmniGibson scene with the objects of every listed scenario, builds the scenarios once,
then serves the agent protocol over HTTP (single thread; all simulator calls stay on the main thread):

    POST /reset   {scenario, seed, clock: paused|live, out, record, privileged}  -> spec
    GET  /spec                                                              -> spec of the current episode
    POST /obs                                                               -> observation (+ stream since last delivery)
    POST /act     {skill, target, seconds}                                  -> result + observation
    POST /frames  {t_from, t_to}                                            -> contact sheet of recorded head-camera frames
    POST /end                                                               -> final result (also written to <out>/result.json)
    GET  /status

Between requests the server also executes development job files dropped into <session>/jobs/ (exec'd in
the server namespace), like og_server.py.
"""
import os, sys, json, time, argparse, traceback, io, contextlib
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--session", type=Path, required=True)
ap.add_argument("--scene", type=Path, required=True)
ap.add_argument("--scenarios", required=True)
ap.add_argument("--port", type=int, default=8701)
ap.add_argument("--code", type=Path, default=Path(__file__).resolve().parent)
a = ap.parse_args()
sess = a.session; (sess / "jobs").mkdir(parents=True, exist_ok=True)
status_f = sess / "server_status.json"


def st(**kw):
    kw["t"] = time.time(); status_f.write_text(json.dumps(kw, default=str))


st(state="importing")
# main_1115: optional cap on Kit / TBB / PhysX worker threads (8 servers share 32 cores; idle workers spin)
_kt = os.environ.get("RLM_KIT_THREADS")
if _kt:
    sys.argv += [f"--/plugins/carb.tasking.plugin/threadCount={_kt}", f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={_kt}",
                 f"--/persistent/physics/numThreads={_kt}"]
import omnigibson as og
from omnigibson.macros import gm
spec = json.loads(a.scene.read_text())
with gm.unlocked():
    gm.HEADLESS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = False
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_HQ_RENDERING = bool(spec.get("hq", False))
import torch as th

ns = dict(og=og, th=th, np=np, gm=gm, sess=sess, CODE=a.code, SCENE_SPEC=spec)
for mod in ("core.py", "station.py", "props.py", "common.py", "v3.py"):
    if (a.code / mod).exists():
        exec(compile((a.code / mod).read_text(), str(a.code / mod), "exec"), ns)
sids = [s for s in a.scenarios.split(",") if s]
MODNS = []
for sid in sids:
    p = a.code / "scenarios" / f"{sid}.py"
    mns = dict(ns)            # private module globals (scenario files reuse names like _A, _FR, _D)
    exec(compile(p.read_text(), str(p), "exec"), mns)
    MODNS.append(mns)
objects = list(spec.get("objects", []))
for sid in sids:
    objects += ns["SCENARIOS"][sid].OBJECTS
names = [o["name"] for o in objects]
assert len(names) == len(set(names)), "duplicate object names"

st(state="loading", scene=spec["scene"], n_objects=len(objects))
t0 = time.time()
cfg = {"env": {"action_frequency": 30, "physics_frequency": 120},
       "render": {"viewer_width": 1280, "viewer_height": 720},
       "scene": {"type": "InteractiveTraversableScene", "scene_model": spec["scene"],
                 **({"load_room_instances": spec["load_room_instances"]} if spec.get("load_room_instances") else {}),
                 **({"not_load_object_categories": spec["skip_categories"]} if spec.get("skip_categories") else {})},
       "robots": spec["robots"], "objects": objects}
env = og.Environment(configs=cfg)
ns["env"] = env
ns["obj"] = lambda name: env.scene.object_registry("name", name)
for mns in MODNS:
    mns["env"] = env; mns["obj"] = ns["obj"]
from omnigibson.object_states import ToggledOn
for o in env.scene.objects:
    if ToggledOn in o.states:
        try:
            o.states[ToggledOn].visual_marker.visible = False
        except Exception:
            pass
for _ in range(10):
    og.sim.step()
st(state="building", load_s=round(time.time() - t0, 1))
built = {sid: ns["SCENARIOS"][sid]() for sid in sids}
state0 = og.sim.dump_state(serialized=False)
og.sim.stop()
for sc in built.values():
    sc.make_cams()
og.sim.play()
og.sim.load_state(state0, serialized=False)
for _ in range(3):
    og.sim.step()
for sid, sc in built.items():
    st(state="building", scenario=sid)
    sc.build()
ns["BUILT"] = built
for _ in range(5):
    og.sim.step()
ns["init_state"] = og.sim.dump_state(serialized=False)
EP = {"ep": None}
ns["EP"] = EP
st(state="ready", load_s=round(time.time() - t0, 1), scenarios=sids, port=a.port)
print("READY", flush=True)


def spec_of(ep):
    sc = ep.sc
    clock_note = {
        "paused": "PAUSED clock: the world waits while you think. It only moves while the robot executes a skill or waits.",
        "live": "LIVE clock: the world keeps running while you think. Other robots, machines and people keep moving. "
                "Every response includes the robot camera frames recorded since your previous request, so you can see what "
                "happened while you were thinking.",
    }[ep.clock]
    # v2s: one neutral sentence for every scenario and seed (no refusal message exists any more)
    clock_note += (" Your robot never reaches into a place another robot, machine or person is handling right now: such a "
                   "command first waits until they are done (world time runs).")
    return dict(scenario=sc.sid, title=sc.title, setting=sc.setting, instruction=sc.instruction(), clock=ep.clock,
                clock_note=clock_note, skills=sc.skill_help(), marks=sc.legend(),
                manipulation_budget=sc.budget, deadline_world_seconds=sc.deadline,
                observation="current_view: the robot head camera now, with numbered marks on fixed places; "
                            "stream: a contact sheet of head-camera frames recorded since your previous request.")


def handle(path, body):
    ep = EP["ep"]
    if path == "/status":
        return dict(ok=True, scenarios=sids, episode=None if ep is None else dict(sid=ep.sc.sid, seed=ep.seed, clock=ep.clock,
                    world_t=round(ep.world_t, 2), over=ep.over))
    if path == "/reset":
        if ep is not None and not getattr(ep, "_closed", False):
            ep.close(); ep._closed = True
        og.sim.load_state(ns["init_state"], serialized=False)
        sc = built[body["scenario"]]
        out = Path(body.get("out") or (sess / "episodes" / f"{sc.sid}_{body.get('clock', 'paused')}_s{body.get('seed', 0)}"))
        ep = ns["Episode"](sc, body.get("seed", 0), body.get("clock", "paused"), out, record=bool(body.get("record", True)),
                           privileged=bool(body.get("privileged", False)), visible=bool(body.get("visible", False)),
                           hard=bool(body.get("hard", False)), commit=bool(body.get("commit", False)))
        EP["ep"] = ep
        return spec_of(ep)
    if ep is None:
        return dict(error="no episode; POST /reset first")
    if path in ("/hello", "/spec"):
        # the clock starts when the agent's tool connection comes up (MCP initialize -> /hello), or at its first request:
        # the agent's first deliberation is charged even if it acts without looking (else that command runs at world 0)
        if not ep.started:
            ep.started = True; ep.ev("clock_start", via=path[1:]); ep.responded()
        return spec_of(ep) if path == "/spec" else dict(ok=True, world_time=round(ep.world_t, 2))
    if path == "/events":
        return dict(events=ep.events)
    if path == "/end":
        if not getattr(ep, "_closed", False):
            ep.close(); ep._closed = True
        return ep.final
    lat = ep.charge_latency(body.get("latency"))
    if path == "/obs":
        r = ep.obs()
    elif path == "/act":
        ep.act(body.get("skill"), body.get("target"), body.get("seconds"))
        r = ep.obs()
    elif path == "/frames":
        t0_, t1_ = float(body.get("t_from", 0)), float(body.get("t_to", 1e9))
        sel = [f for f in ep.stream[:ep.delivered] if t0_ <= f[0] <= t1_]
        r = dict(world_time=round(ep.world_t, 2), n_frames=len(sel), **(ep.sheet(sel) if sel else {}))
    else:
        r = dict(error=f"unknown path {path}")
    r["charged_latency"] = round(lat, 2)
    ep.responded()
    return r


class H(BaseHTTPRequestHandler):
    def log_message(self, *a_):
        pass

    def _go(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        try:
            r = handle(self.path.split("?")[0], body); code = 200
        except Exception:
            r = dict(error=traceback.format_exc()); code = 500
        data = json.dumps(r, default=float).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    do_GET = _go
    do_POST = _go


srv = HTTPServer(("0.0.0.0", a.port), H)
srv.timeout = 0.5
while not (sess / "STOP").exists():
    srv.handle_request()
    jobs = sorted(p for p in (sess / "jobs").glob("*.py") if not p.with_suffix(".done").exists())
    for j in jobs[:1]:
        st(state="job", job=j.name)
        buf = io.StringIO(); t1 = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(j.read_text(), str(j), "exec"), ns)
            ok = True
        except BaseException:
            buf.write("\n" + traceback.format_exc()); ok = False
        j.with_suffix(".log").write_text(buf.getvalue() + f"\n[{'OK' if ok else 'FAIL'} {time.time() - t1:.1f}s]\n")
        j.with_suffix(".done").write_text("ok" if ok else "fail")
        st(state="ready", last=j.name, ok=ok, scenarios=sids, port=a.port)
st(state="stopped")
og.shutdown()
