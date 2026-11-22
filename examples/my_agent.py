"""A minimal custom agent for the tabletop tier (template for your own method).

    python -m robolivemem.serve --port 8800 &
    python -m robolivemem.eval --tier tabletop --env http://127.0.0.1:8800 --agent examples/my_agent.py:FirstLookAgent \
        --out results/first_look.jsonl

An agent gets the observation (obs["system"]: instructions and command format; obs["content"]: text and images) and
returns one command. This one reads the mark from the first instruction and picks it at once: it is fast, so under the
live clock it often acts before anything changes, but it never looks at the frames.
"""
import re

from robolivemem.agents import Agent


class FirstLookAgent(Agent):
    name = "first-look"

    def reset(self, obs):
        text = " ".join(c["text"] for c in obs["content"] if c["type"] == "text")
        m = re.search(r"mark (\d)", text)
        self.mark = int(m.group(1)) if m else 1

    def act(self, obs):
        return {"action": "pick", "mark": self.mark}
