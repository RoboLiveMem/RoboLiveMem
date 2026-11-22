"""An external harness: a separate program that talks to the environment itself (here over HTTP with the Python
client; Claude Code or Codex would use `python -m robolivemem.mcp` instead). robolivemem.eval starts it once per
episode with RLM_ENV_URL and RLM_TIER set, and reads the result from the server when it exits.

    python -m robolivemem.eval --tier tabletop --env http://127.0.0.1:8800 \
        --agent "cmd:python examples/external_harness.py" --out results/harness.jsonl
"""
import os
import re

from robolivemem.env import Env

env = Env(os.environ["RLM_TIER"], os.environ["RLM_ENV_URL"])
obs = env.observe()                       # starts the clock for this episode
env.episode = obs["episode"]
text = " ".join(c["text"] for c in obs["content"] if c["type"] == "text")
mark = int(re.search(r"mark (\d)", text).group(1))
obs = env.act({"action": "wait", "seconds": 6})       # let the person finish first
while not obs["done"]:
    obs = env.act({"action": "pick", "mark": mark})   # (a real harness would look at the frames here)
