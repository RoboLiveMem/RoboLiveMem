"""The environment protocol as MCP tools (stdio JSON-RPC), so tool-using harnesses (Claude Code, Codex, ...) can act.

    {"mcpServers": {"robot": {"command": "python", "args": ["-m", "robolivemem.mcp"]}}}

The server URL comes from $RLM_ENV_URL (set by `robolivemem.eval --agent cmd:...`). Tools:
    robot_instructions   the task, the commands and the current observation (starts the clock; free)
    robot_act            send one command (a JSON object, see the instructions); returns the next observation
    robot_end            give up and end the episode
For the realistic tier use its own tools instead (tiers/realistic/client/rlm_mcp.py: robot_observe, robot_act, ...).
"""
from __future__ import annotations

import json
import os
import sys

from .env import Env

TOOLS = [
    {"name": "robot_instructions", "description": "The task, the commands you can send and the current observation.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "robot_act", "description": "Send exactly one command, a JSON object in the format the instructions give. "
     "Returns the next observation, or the result when the episode is over.",
     "inputSchema": {"type": "object", "properties": {"command": {"type": "object"}}, "required": ["command"]}},
    {"name": "robot_end", "description": "End the episode now.", "inputSchema": {"type": "object", "properties": {}}},
]


def _content(obs: dict, with_system=False):
    if obs.get("done"):
        r = obs.get("result") or {}
        return [{"type": "text", "text": f"EPISODE OVER: {'success' if r.get('success') else 'failure'}."}]
    out = [{"type": "text", "text": obs["system"]}] if with_system and obs.get("system") else []
    for c in obs.get("content") or []:
        out.append({"type": "image", "data": c["data"], "mimeType": c.get("media_type", "image/png")}
                   if c.get("type") == "image" else c)
    return out


def main():
    env = Env(os.environ.get("RLM_TIER", "tabletop"), os.environ.get("RLM_ENV_URL", "http://127.0.0.1:8800"))
    send = lambda m: (sys.stdout.write(json.dumps(m) + "\n"), sys.stdout.flush())  # noqa: E731
    for line in sys.stdin:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        mid, meth = m.get("id"), m.get("method")
        if meth == "initialize":
            pv = (m.get("params") or {}).get("protocolVersion", "2025-06-18")
            send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": pv, "capabilities": {"tools": {}},
                                                           "serverInfo": {"name": "robolivemem", "version": "0.1"}}})
        elif meth == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif meth == "tools/call":
            p = m.get("params") or {}
            name, args = p.get("name"), p.get("arguments") or {}
            try:
                if name == "robot_instructions":
                    obs = env.observe()
                    env.episode = obs.get("episode")
                    content = _content(obs, with_system=True)
                elif name == "robot_act":
                    cmd = args.get("command")
                    if isinstance(cmd, str):
                        cmd = json.loads(cmd)
                    if env.episode is None:
                        env.episode = env.observe().get("episode")
                    content = _content(env.act(cmd or {}))
                elif name == "robot_end":
                    content = _content(env.end())
                else:
                    content = [{"type": "text", "text": f"unknown tool {name}"}]
                err = False
            except Exception as e:  # noqa: BLE001
                content, err = [{"type": "text", "text": f"ERROR: {e}"}], True
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": content, "isError": err}})
        elif meth == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None and meth is not None:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {meth}"}})


if __name__ == "__main__":
    main()
