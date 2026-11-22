"""End-to-end through the environment protocol: tabletop server <-> client, every kind of agent."""
import json
import socket
import threading
import time

import pytest

from robolivemem.agents import ChatAgent, ServerPolicy
from robolivemem.env import Env, wait_ready
from robolivemem.eval import run_job
from robolivemem.protocol import Bridge
from robolivemem.report import summarize
from robolivemem.tabletop import INFO, serve_runner


@pytest.fixture(scope="module")
def env():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    threading.Thread(target=Bridge(serve_runner, INFO).run, args=("127.0.0.1", port), daemon=True).start()
    e = Env("tabletop", f"http://127.0.0.1:{port}")
    wait_ready(e, 20)
    return e


def job(seed, cond, clock="live"):
    return {"tier": "tabletop", "task": "mug", "class": "U", "seed": seed, "condition": cond,
            "clock": "paused" if cond == "P" else clock, "commit": False}


def test_observation_shape(env):
    obs = env.reset({"task": "mug", "seed": 0, "condition": "L"})
    assert not obs["done"] and obs["system"] and obs["dialogue"]
    kinds = {c["type"] for c in obs["content"]}
    assert kinds == {"text", "image"}
    obs = env.act({"action": "wait", "seconds": 2})
    assert obs["world_time"] >= 2.0
    res = env.end()
    assert res["done"] and res["result"]["status"] == "aborted"


def test_wall_clock_is_charged(env):
    obs = env.reset({"task": "mug", "seed": 0, "condition": "L"})
    time.sleep(1.2)                                   # "thinking"
    obs = env.act({"action": "wait", "seconds": 0.5})
    assert obs["world_time"] >= 1.6                   # 1.2 s of thinking + 0.5 s of waiting
    env.end()


def test_declared_latency_and_reference_policies(env):
    rows = [run_job(env, ServerPolicy("privileged"), job(s, c)) for s in range(6) for c in "PLV"]
    assert all(r["success"] for r in rows)
    rows = [run_job(env, ServerPolicy("stale"), job(s, c)) for s in range(6) for c in "PL"]
    assert not any(r["success"] for r in rows if r["condition"] == "L" and r["variant"] == "event")
    agent = ChatAgent('mock:{"action": "wait", "seconds": 1}')
    row = run_job(env, agent, job(2, "L"), max_steps=3)
    assert row["latency_source"] == "declared" or row["status"] == "aborted"


def test_report_from_rows(env):
    rows = [dict(run_job(env, ServerPolicy("stale"), job(s, c)), agent="stale") for s in range(6) for c in "PLV"]
    s = summarize(rows)["stale"]["tabletop"]["summary"]
    assert s["P"] == 100.0 and s["L"] == 0.0 and s["none"] == 100.0
    assert s["drop"]["diff"] == -100.0
    json.dumps(s)
