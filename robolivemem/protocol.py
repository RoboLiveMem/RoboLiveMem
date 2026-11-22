"""The RoboLiveMem environment protocol: what an agent receives and what it sends back (see PROTOCOL.md).

Every tier serves episodes over HTTP with the same four calls:

    POST /reset   {"task", "seed", "condition": "P"|"L"|"V", "clock": "paused"|"live"|"fixed:<s>", "commit": false}
                  -> first observation
    POST /act     {"episode", "command": {...}, "latency": <optional seconds>, "text": <optional raw reply>}
                  -> next observation (or the final result)
    POST /observe {"episode"}  -> the current observation again (does not touch the clock)
    POST /end     {"episode"}  -> stop the episode; returns the result
    GET  /info    -> tier, tasks, conditions, command vocabulary

An observation is

    {"episode": id, "step": k, "done": false, "world_time": t,
     "system": "<the tier's instructions: task setting, commands, reply format>",
     "content": [{"type": "text", "text": ...}, {"type": "image", "media_type": "image/png", "data": <base64>}, ...],
     "dialogue": [[role, content], ...]}   # optional: the context the paper's reference harness sends to its model

and the last one has "done": true and "result": {"success": ..., "variant": ..., ...}.

The clock: under the live clock the world advances by the time between the response that delivered an observation and
the agent's next /act, measured by the server (everything counts: model, harness, network). A client may instead
declare "latency" (for example the measured time of its model request, as the paper's reference agents do); the
result records which source was used. Under the paused clock nothing moves in between; under fixed:L every /act is
charged L seconds.
"""
from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONDITIONS = {"P": ("hidden", "paused"), "L": ("hidden", "live"), "V": ("visible", "live")}
VARIANTS = {"event": "event", "lookalike": "decoy", "decoy": "decoy", "null": "null", "none": "null"}


# ---------------------------------------------------------------------- content
def image_item(img, fmt: str = "png") -> dict:
    """numpy RGB image -> protocol image item."""
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    if fmt == "jpeg":
        im.convert("RGB").save(buf, format="JPEG", quality=90)
    else:
        im.save(buf, format="PNG")
    return {"type": "image", "media_type": f"image/{fmt}", "data": base64.b64encode(buf.getvalue()).decode()}


def to_content(parts, fmt: str = "png") -> list:
    """Model-message parts (str | numpy image | {"text"} | {"jpeg"} | {"png"}) -> protocol content items."""
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append({"type": "text", "text": p})
        elif isinstance(p, dict):
            if p.get("type") in ("text", "image"):
                out.append(p)
            elif "text" in p:
                out.append({"type": "text", "text": str(p["text"])})
            elif "jpeg" in p:
                out.append({"type": "image", "media_type": "image/jpeg", "data": p["jpeg"]})
            elif "png" in p:
                out.append({"type": "image", "media_type": "image/png", "data": p["png"]})
        else:
            out.append(image_item(p, fmt))
    return out


def decode_image(item: dict):
    """protocol image item -> numpy RGB image."""
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(item["data"]))).convert("RGB"))


def parse_clock(spec: str):
    """'paused' | 'live' | 'fixed:<s>' -> (name, seconds)."""
    s = (spec or "live").strip().lower()
    if s.startswith("fixed"):
        return "fixed", float(s[5:].lstrip(":=") or 0)
    if s not in ("paused", "live"):
        raise ValueError(f"unknown clock {spec!r}")
    return s, 0.0


# ---------------------------------------------------------------------- server side
class Aborted(Exception):
    """The agent ended the episode (/end), started a new one, or stopped answering."""


class Bridge:
    """Serves a tier's episode loop to remote agents.

    The tier's loop runs on the main thread (simulators stay on one thread): `runner(job, ask)` plays one episode and
    calls `ask(observation)` whenever the agent must decide; `ask` blocks until the agent posts its command and returns
    (command, text, latency_s, source).
    HTTP requests are handled on worker threads."""

    def __init__(self, runner, info: dict, decision_timeout_s: float = 1800.0):
        self.runner, self.info, self.timeout = runner, info, decision_timeout_s
        self.cv = threading.Condition()
        self.jobs: "queue.Queue[dict]" = queue.Queue()
        self.ep: dict | None = None

    # -- called by the tier loop (main thread)
    def _ask(self, ep: dict, obs: dict):
        with self.cv:
            ep["step"] += 1
            ep["obs"] = dict(obs, episode=ep["id"], step=ep["step"], done=False)
            ep["cmd"] = None
            self.cv.notify_all()
            t_end = time.time() + self.timeout
            while ep["cmd"] is None and not ep["abort"]:
                if not self.cv.wait(timeout=max(0.0, t_end - time.time())) and time.time() >= t_end:
                    ep["abort"] = "decision timeout"
            if ep["abort"]:
                raise Aborted(ep["abort"])
            c, ep["cmd"] = ep["cmd"], None
            ep["latency"].append(c[2])
            return c

    def serve_forever(self):
        while True:
            job = self.jobs.get()
            ep = job["_ep"]
            try:
                result = self.runner(job, lambda obs, _ep=ep: self._ask(_ep, obs))
            except Aborted as e:
                result = {"success": False, "status": "aborted", "error": str(e)}
            except Exception as e:  # noqa: BLE001
                result = {"success": False, "status": "crash", "error": f"{type(e).__name__}: {e}"[:400],
                          "traceback": traceback.format_exc()[-2000:]}
            lat = ep["latency"]
            result = dict(result, latency_source=ep["source"] or "none",
                          mean_latency_s=round(sum(lat) / len(lat), 3) if lat else None, decisions=len(lat))
            with self.cv:
                ep["done"], ep["result"] = True, result
                ep["obs"] = {"episode": ep["id"], "step": ep["step"] + 1, "done": True, "result": result}
                self.cv.notify_all()

    # -- called by HTTP handlers
    def _wait_obs(self, ep, after_step):
        t_end = time.time() + self.timeout
        while ep["step"] <= after_step and not ep["done"] and time.time() < t_end:
            self.cv.wait(timeout=1.0)
        return ep["obs"]

    def reset(self, body: dict):
        with self.cv:
            if self.ep is not None and not self.ep["done"]:
                self.ep["abort"] = "a new episode was started"
                self.cv.notify_all()
            ep = {"id": uuid.uuid4().hex[:12], "step": 0, "obs": None, "cmd": None, "abort": None, "done": False,
                  "result": None, "t_resp": None, "latency": [], "source": None}
            job = dict(body, _ep=ep)
            self.ep = ep
            self.jobs.put(job)
            obs = self._wait_obs(ep, 0)
            # the clock starts when the agent receives the observation: now, or at its first /observe when the
            # episode was started for an external harness ("defer_clock")
            ep["t_resp"] = None if body.get("defer_clock") else time.time()
            return obs

    def act(self, body: dict):
        with self.cv:
            ep = self.ep
            if ep is None or body.get("episode") not in (None, ep["id"]):
                return {"error": "unknown episode"}
            if ep["done"]:
                return ep["obs"]
            if ep["obs"] is None or ep["cmd"] is not None:
                return {"error": "no decision pending"}
            if body.get("latency") is not None:
                lat, src = max(0.0, float(body["latency"])), "declared"
            else:
                lat, src = max(0.0, time.time() - (ep["t_resp"] or time.time())), "wall"
            ep["source"] = src if ep["source"] in (None, src) else "mixed"
            step = ep["step"]
            ep["cmd"] = (body.get("command") or {}, body.get("text"), lat, src)
            self.cv.notify_all()
            obs = self._wait_obs(ep, step)
            ep["t_resp"] = time.time()
            return obs

    def observe(self, body: dict):
        with self.cv:
            ep = self.ep
            if ep is None:
                return {"error": "no episode; POST /reset first"}
            if ep["t_resp"] is None and ep["obs"] is not None:
                ep["t_resp"] = time.time()
            return ep["obs"] or {"error": "not ready"}

    def end(self, body: dict):
        with self.cv:
            ep = self.ep
            if ep is None:
                return {"error": "no episode"}
            if not ep["done"]:
                ep["abort"] = "ended by the agent"
                self.cv.notify_all()
                self._wait_obs(ep, 10 ** 9)
            return ep["obs"]

    # -- HTTP
    def handler(self):
        bridge = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                    path = self.path.split("?")[0].rstrip("/")
                    fn = {"/reset": bridge.reset, "/act": bridge.act, "/observe": bridge.observe, "/end": bridge.end,
                          "/info": lambda b: bridge.info}.get(path)
                    r, code = (fn(body), 200) if fn else ({"error": f"unknown path {path}"}, 404)
                except Exception as e:  # noqa: BLE001
                    r, code = {"error": f"{type(e).__name__}: {e}"}, 500
                data = json.dumps(r, default=_jsonable).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = _go
            do_POST = _go
        return H

    def run(self, host: str = "127.0.0.1", port: int = 8800):
        srv = ThreadingHTTPServer((host, port), self.handler())
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"RoboLiveMem {self.info.get('tier')} environment on http://{host}:{port}", flush=True)
        self.serve_forever()


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)
