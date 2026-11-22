"""Agents: anything that turns an observation into one command.

    class MyAgent(robolivemem.agents.Agent):
        def act(self, obs):              # obs["system"], obs["content"] (text and images), obs["world_time"], ...
            return {"tool": "wait", "seconds": 1}          # a command in the tier's vocabulary (see obs["system"])

An agent can also return (command, text, latency): `latency` (seconds) declares the time to charge instead of the
time the server measures; `text` is logged with the episode.

Built in:
    chat:<model spec>     the paper's reference harness around any chat model (robolivemem.models)
    server:<policy>       a reference policy that runs inside the environment server (privileged, stale, ...)
    path/to/file.py:Class your own agent class (constructed without arguments)
    cmd:<shell command>   an external harness (Codex, Claude Code, ...) that talks to the environment itself, for
                          example through `python -m robolivemem.mcp`; RLM_ENV_URL and RLM_TIER are set for it
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess

from .models import Chat, parse_json_reply


class Agent:
    name = "agent"

    def reset(self, obs: dict):
        """Called with the first observation of every episode."""

    def act(self, obs: dict):
        raise NotImplementedError

    def close(self):
        pass


def _limit_images(turns, keep: int):
    """Keep only the newest `keep` images of a dialogue (older images become a short note)."""
    n = 0
    out = []
    for role, content in reversed(turns):
        new = []
        for c in reversed(content):
            if c.get("type") == "image":
                n += 1
                if n > keep:
                    c = {"type": "text", "text": "(older image omitted)"}
            new.append(c)
        out.append((role, list(reversed(new))))
    return list(reversed(out))


class ChatAgent(Agent):
    """The paper's reference harness: send the observation to a chat model, read one JSON command from the reply.

    If the server provides obs["dialogue"] (semi and controlled tiers, tabletop), that context is sent exactly as the
    paper's runs sent it; otherwise the agent keeps the dialogue itself, with the newest `keep_images` images.
    By default it declares the measured time of its model request as its latency, as in the paper (`declare=False`
    lets the server measure the whole wall time instead)."""

    def __init__(self, spec: str, max_tokens: int = 4096, effort: str | None = None, keep_images: int = 12,
                 declare: bool = True, retries: int = 4):
        self.chat = Chat(spec, max_tokens=max_tokens, effort=effort)
        self.name, self.keep, self.declare, self.retries = spec, keep_images, declare, retries
        self.turns, self.system = [], ""

    def reset(self, obs):
        self.turns, self.system = [], obs.get("system", "")

    def act(self, obs):
        if obs.get("system"):
            self.system = obs["system"]
        if obs.get("dialogue") is not None:
            turns = [(r, c) for r, c in obs["dialogue"]]
        else:
            self.turns.append(("user", obs.get("content") or [{"type": "text", "text": "(no observation)"}]))
            turns = _limit_images(self.turns, self.keep)
        text, meta, lat = "", {}, 0.0
        for _ in range(self.retries):
            text, meta = self.chat(self.system, turns)
            if not meta.get("error"):
                break
        else:
            raise RuntimeError(f"model call failed: {meta.get('error')}")
        lat = float(meta.get("latency_s") or 0.0)   # only the answered request is charged
        if obs.get("dialogue") is None:
            self.turns.append(("assistant", [{"type": "text", "text": text or "{}"}]))
        cmd = parse_json_reply(text) or {}
        return cmd, text, (lat if self.declare else None)


class ServerPolicy(Agent):
    """Marker: the environment server runs one of its own reference policies (for validation)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.name = f"server:{policy}"


class CommandAgent(Agent):
    """An external harness started once per episode; it acts on the environment itself and exits when done."""

    def __init__(self, command: str, timeout_s: float = 6 * 3600):
        self.command, self.timeout = command, timeout_s
        self.name = "cmd:" + " ".join(command.split()[-2:])[-60:]

    def run_episode(self, tier: str, url: str):
        env = dict(os.environ, RLM_ENV_URL=url, RLM_TIER=tier, RLM_URL=url)
        return subprocess.run(shlex.split(self.command), env=env, timeout=self.timeout).returncode


def load_agent(spec: str, **kw) -> Agent:
    if spec.startswith("chat:"):
        return ChatAgent(spec[5:], **kw)
    if spec.startswith("server:"):
        return ServerPolicy(spec[7:])
    if spec.startswith("cmd:"):
        return CommandAgent(spec[4:])
    if ".py:" in spec:
        path, cls = spec.rsplit(":", 1)
        sp = importlib.util.spec_from_file_location("user_agent", path)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        agent = getattr(mod, cls)()
        agent.name = getattr(agent, "name", None) if getattr(agent, "name", "agent") != "agent" else cls
        return agent
    raise ValueError(f"unknown agent {spec!r}: use chat:<model>, server:<policy>, file.py:Class or cmd:<command>")
