#!/usr/bin/env python3
"""RoboLiveMem-Bench robot interface as an MCP server (stdio, JSON-RPC 2.0; Python standard library only).

The same tools serve every agent (Claude Code, Codex, API agents), so all models see identical observations:
  robot_spec, robot_observe, robot_act, robot_wait, robot_frames, robot_done
Images (current camera view, stream contact sheet) are returned as MCP image content.
Server URL: $RLM_URL (default http://127.0.0.1:8701). Every call is also appended to $RLM_LOG (jsonl, no images).
"""
import base64, json, os, sys, time, urllib.request

URL = os.environ.get("RLM_URL", "http://127.0.0.1:8701")
LOG = os.environ.get("RLM_LOG")
KEY = os.environ.get("RLM_KEY", "")
IMGDIR = os.environ.get("RLM_IMGDIR")   # main_1115: copies of the images returned to the agent (offline probe)
_NIMG = [0]
NOSTREAM = os.environ.get("RLM_NOSTREAM") == "1"   # ablation: the agent sees only the current view, never the recording

TOOLS = [
    {"name": "robot_spec", "description": "Task, skills, the numbered marks in the camera image, budget and clock.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "robot_observe", "description": "Look: the robot head camera now (numbered marks on fixed places) and a contact sheet of the "
                                             "camera frames recorded since your previous request.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "robot_act", "description": "Run one skill at a numbered mark, e.g. skill='pick', mark='2'. Returns the result, the camera "
                                         "view afterwards and the frames recorded since your previous request.",
     "inputSchema": {"type": "object", "properties": {"skill": {"type": "string"}, "mark": {"type": "string"}},
                     "required": ["skill"], "additionalProperties": False}},
    {"name": "robot_wait", "description": "Hold still and let the world run for some seconds (0.5 to 60). Free.",
     "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"], "additionalProperties": False}},
    {"name": "robot_frames", "description": "Contact sheet of recorded camera frames between two world times (seconds). Free.",
     "inputSchema": {"type": "object", "properties": {"t_from": {"type": "number"}, "t_to": {"type": "number"}},
                     "required": ["t_from", "t_to"], "additionalProperties": False}},
    {"name": "robot_done", "description": "You are finished: end the episode.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


if NOSTREAM:
    TOOLS = [t for t in TOOLS if t["name"] != "robot_frames"]
    for t in TOOLS:
        if t["name"] == "robot_observe":
            t["description"] = "Look: the robot head camera now (numbered marks on fixed places)."
        if t["name"] == "robot_act":
            t["description"] = "Run one skill at a numbered mark, e.g. skill='pick', mark='2'. Returns the result and the camera view afterwards."


def http(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json", "X-RLM-Key": KEY}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def img(b64):
    return {"type": "image", "data": b64, "mimeType": "image/jpeg"}


def obs_content(r):
    if "error" in r:
        return [{"type": "text", "text": "ERROR: " + str(r["error"])[-1500:]}], True
    lines = [f"world time: {r['world_time']:.1f} s   clock: {r['clock']}"]
    if r.get("last_result"):
        lines.append(f"last result: {r['last_result']}")
    lines.append(f"holding: {r['holding']}   manipulation actions left: {r['manipulation_actions_left']}")
    if r.get("episode_over"):
        lines.append(f"EPISODE OVER ({r.get('over_reason')})")
    s = {} if NOSTREAM else (r.get("stream") or {})
    content = []
    if s.get("n_frames"):
        lines.append(f"stream: {s['n_frames']} camera frames were recorded since your previous request (world {s['t_from']:.1f}-"
                     f"{s['t_to']:.1f} s). Image 1 below is a contact sheet of {len(s['sheet_times'])} of them (times printed on each "
                     f"tile). Use robot_frames to look at a window more closely.")
    elif not NOSTREAM:
        lines.append("stream: no new frames (the world did not move since your previous request).")
    lines.append(("Image 2" if s.get("n_frames") else "Image 1") + ": the camera view now, with the numbered marks.")
    content.append({"type": "text", "text": "\n".join(lines)})
    if s.get("n_frames"):
        content.append(img(s["sheet"]))
    content.append(img(r["current_view"]))
    return content, False


def call(name, args):
    if name == "robot_spec":
        r = http("/spec")
        if "error" in r:
            return [{"type": "text", "text": "ERROR: " + r["error"]}], True
        note = r["clock_note"]
        if NOSTREAM and " Every response includes" in note:
            note = note.split(" Every response includes")[0] + (" Safety interlock:" + note.split(" Safety interlock:")[1] if " Safety interlock:" in note else "")
        t = [f"SCENARIO: {r['title']} ({r['setting']})", "", "TASK:", r["instruction"], "", "CLOCK: " + note, "", "SKILLS:"]
        t += [f"  {k}: {v}" for k, v in r["skills"].items()]
        t += ["", "NUMBERED MARKS (fixed places in the camera image):"] + [f"  {k}: {v}" for k, v in r["marks"].items()]
        t += ["", f"manipulation actions allowed: {r['manipulation_budget']} (observe, frames and wait are free); world deadline "
                  f"{r['deadline_world_seconds']:.0f} s"]
        return [{"type": "text", "text": "\n".join(t)}], False
    if name == "robot_observe":
        return obs_content(http("/obs"))
    if name == "robot_act":
        body = {"skill": str(args.get("skill", ""))}
        if args.get("mark") not in (None, ""):
            body["target"] = str(args["mark"])
        return obs_content(http("/act", body))
    if name == "robot_wait":
        return obs_content(http("/act", {"skill": "wait", "seconds": float(args.get("seconds", 5))}))
    if name == "robot_frames" and NOSTREAM:
        return [{"type": "text", "text": "unknown tool robot_frames"}], True
    if name == "robot_frames":
        r = http("/frames", {"t_from": float(args["t_from"]), "t_to": float(args["t_to"])})
        if not r.get("n_frames"):
            return [{"type": "text", "text": "no recorded frames in that window"}], False
        return [{"type": "text", "text": f"{r['n_frames']} frames between {args['t_from']} and {args['t_to']} s; the sheet shows "
                                         f"{len(r['sheet_times'])} of them (times on the tiles)."}, img(r["sheet"])], False
    if name == "robot_done":
        r = http("/act", {"skill": "done"})
        return [{"type": "text", "text": str(r.get("last_result", r))}], False
    return [{"type": "text", "text": f"unknown tool {name}"}], True


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()


def save_imgs(tool, content):
    if not IMGDIR:
        return []
    names = []
    try:
        os.makedirs(IMGDIR, exist_ok=True)
        for c in content:
            if c.get("type") == "image":
                _NIMG[0] += 1
                fn = f"{_NIMG[0]:03d}_{tool}.jpg"
                with open(os.path.join(IMGDIR, fn), "wb") as f:
                    f.write(base64.b64decode(c["data"]))
                names.append(fn)
    except Exception:
        pass
    return names


def log(entry):
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue
        mid = m.get("id"); meth = m.get("method")
        if meth == "initialize":
            try:
                http("/hello")          # the live clock starts now: the agent is connected and has its task
            except Exception:
                pass
            pv = (m.get("params") or {}).get("protocolVersion", "2025-06-18")
            send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": pv, "capabilities": {"tools": {}},
                                                           "serverInfo": {"name": "rlm", "version": "1.0"}}})
        elif meth == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif meth == "tools/call":
            p = m.get("params") or {}
            t0 = time.time()
            try:
                content, err = call(p.get("name"), p.get("arguments") or {})
            except Exception as e:
                content, err = [{"type": "text", "text": f"ERROR: {e}"}], True
            log(dict(t=round(t0, 3), dt=round(time.time() - t0, 3), tool=p.get("name"), args=p.get("arguments"),
                     text=[c.get("text") for c in content if c["type"] == "text"], images=save_imgs(p.get("name"), content)))
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": content, "isError": err}})
        elif meth == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None and meth is not None:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {meth}"}})


if __name__ == "__main__":
    main()
