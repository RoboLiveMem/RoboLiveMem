"""Client side of the environment protocol (robolivemem.protocol): one interface for every tier.

    from robolivemem.env import connect
    env = connect("semi", "http://127.0.0.1:8801")
    obs = env.reset({"task": "mugs", "seed": 0, "condition": "L"})
    while not obs["done"]:
        command = ...                        # your agent: obs["system"], obs["content"] -> one command (a JSON object)
        obs = env.act(command)               # optionally env.act(command, latency=<seconds>) to declare your latency
    print(obs["result"]["success"])

The realistic tier runs one BEHAVIOR-1K server per scene; connect("realistic", "house_v3=http://...,hall_v3=...")
maps scenes to servers and speaks that server's tool protocol (robot_observe, robot_act, ...).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from .benchmark import TIERS
from .protocol import CONDITIONS, VARIANTS, parse_clock

REPO = Path(__file__).resolve().parents[1]


def _post(url: str, path: str, body: dict | None = None, timeout: float = 3600.0) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Env:
    """A tier served by robolivemem.protocol.Bridge (tabletop, semi, controlled)."""

    def __init__(self, tier: str, url: str):
        self.tier, self.url, self.episode = tier, url.rstrip("/"), None

    def info(self) -> dict:
        return _post(self.url, "/info")

    def reset(self, job: dict) -> dict:
        obs = _post(self.url, "/reset", job)
        if "error" in obs:
            raise RuntimeError(obs["error"])
        self.episode = obs.get("episode")
        return obs

    def act(self, command: dict, latency: float | None = None, text: str | None = None) -> dict:
        body = {"episode": self.episode, "command": command}
        if latency is not None:
            body["latency"] = float(latency)
        if text is not None:
            body["text"] = text
        return _post(self.url, "/act", body)

    def observe(self) -> dict:
        return _post(self.url, "/observe", {"episode": self.episode})

    def end(self) -> dict:
        return _post(self.url, "/end", {"episode": self.episode})


class RealisticEnv:
    """The realistic tier: one BEHAVIOR-1K scenario server per scene (tiers/realistic/server/bench_server.py).

    Commands are tool calls: {"tool": "robot_observe"}, {"tool": "robot_act", "skill": "pick", "mark": "2"},
    {"tool": "robot_wait", "seconds": 3}, {"tool": "robot_frames", "t_from": 4, "t_to": 9}, {"tool": "robot_spec"},
    {"tool": "robot_done"}. The server charges the time between its last response and the next request."""

    tier = "realistic"

    def __init__(self, urls: str | dict):
        if isinstance(urls, str):
            urls = dict(kv.split("=", 1) for kv in urls.split(",") if kv)
        self.urls = {k: v.rstrip("/") for k, v in urls.items()}
        sys.path[:0] = [str(REPO / "tiers/realistic/client"), str(REPO / "tiers/realistic/harness")]
        import rlm_mcp  # noqa: E402
        import run_agent as RA  # noqa: E402
        from run_agent_api import PROTOCOL  # noqa: E402
        self.MCP, self.RA, self.PROTOCOL = rlm_mcp, RA, PROTOCOL
        self.url, self.job, self.fixed, self.over = None, None, None, False

    def scenes(self):
        return set(self.urls)

    def reset(self, job: dict) -> dict:
        scene = TIERS["realistic"]["tasks"][job["task"]]["scene"]
        if scene not in self.urls:
            raise RuntimeError(f"no server for scene {scene} (give {scene}=http://host:port)")
        self.url, self.job, self.over = self.urls[scene], job, False
        member, _ = CONDITIONS[job.get("condition", "L")]
        clock, self.fixed = parse_clock(job.get("clock", "live"))
        spec = _post(self.url, "/reset", dict(scenario=job["task"], seed=int(job["seed"]),
                                              clock="paused" if clock == "paused" else "live",
                                              visible=(member == "visible"), commit=bool(job.get("commit")),
                                              record=bool(job.get("record", False)), out=job.get("out")))
        if "error" in spec:
            raise RuntimeError(spec["error"])
        self.MCP.URL = self.url
        if not job.get("defer_clock"):         # an external harness starts the clock itself (MCP initialize)
            try:
                _post(self.url, "/hello")      # the clock starts when the agent connects
            except Exception:
                pass
        prompt = self.RA.prompt_for(spec, "api", "visible" if member == "visible" else "")
        return {"episode": f"{job['task']}_{job['seed']}", "step": 0, "done": False, "system": prompt + self.PROTOCOL,
                "content": [{"type": "text", "text": "Begin. Call one robot_* tool."}], "spec": spec}

    def act(self, command: dict, latency: float | None = None, text: str | None = None) -> dict:
        cmd = dict(command or {})
        name = cmd.pop("tool", None)
        if self.fixed:
            latency = self.fixed if name != "robot_spec" else 0.0
        orig = self.MCP.http
        if latency is not None:
            self.MCP.http = lambda path, b=None, _o=orig, _l=float(latency): _o(path, dict(b or {}, latency=_l))
        try:
            content, _ = self.MCP.call(name, cmd) if name else (
                [{"type": "text", "text": "No tool call found. Reply with one JSON tool call, e.g. {\"tool\": \"robot_observe\"}."}], True)
        except Exception as e:  # noqa: BLE001
            content = [{"type": "text", "text": f"ERROR: {e}"}]
        finally:
            self.MCP.http = orig
        content = [{"type": "image", "media_type": c.get("mimeType", "image/jpeg"), "data": c["data"]}
                   if c.get("type") == "image" else c for c in content]
        txt = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        if name == "robot_done" or "EPISODE OVER" in txt:
            return self.end()
        return {"episode": f"{self.job['task']}_{self.job['seed']}", "done": False, "content": content}

    def observe(self) -> dict:
        return {"done": self.over, "content": []}

    def end(self) -> dict:
        final = _post(self.url, "/end")
        self.over = True
        seed = int(self.job["seed"])
        result = {"success": bool(final.get("success")), "status": final.get("end"), "reason": final.get("reason"),
                  "variant": ("event", "decoy", "null")[seed % 3], "world_s": final.get("world_t"),
                  "manipulation_actions": final.get("manipulation_actions"),
                  "latencies": [round(x["latency"], 2) for x in final.get("timeline", []) if x.get("kind") == "think"]}
        lat = result["latencies"]
        result["mean_latency_s"] = round(sum(lat) / len(lat), 3) if lat else None
        result["decisions"] = len(lat)
        return {"done": True, "result": result}


def connect(tier: str, url: str):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; tiers: {', '.join(TIERS)}")
    return RealisticEnv(url) if tier == "realistic" else Env(tier, url)


def normalize_result(result: dict) -> dict:
    r = dict(result or {})
    if r.get("variant") is not None:
        r["variant"] = VARIANTS.get(str(r["variant"]), r["variant"])
    return r


def wait_ready(env, seconds: float = 600.0):
    """Block until a bridge server answers /info."""
    t0 = time.time()
    while True:
        try:
            return env.info()
        except Exception:
            if time.time() - t0 > seconds:
                raise
            time.sleep(2.0)
