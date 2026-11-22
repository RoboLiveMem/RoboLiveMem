"""The stale reference policy as a robolivemem agent (validation of the realistic tier, no model needed).

    python -m robolivemem.eval --tier realistic --env house=http://127.0.0.1:8701 --tasks s02_annas_mug \
        --agent tiers/realistic/harness/stale_agent.py:StaleAgent --seeds 0-2 --conditions P,L

It executes the plan that is right for the scene as first observed and never reads the recorded frames, declaring
THINK seconds of deliberation per decision (default 8). Expected: success on every paused seed; live, failure on event
seeds and success on decoy and null seeds (paper, Sec. 4 and Appendix B).
"""
import os
import sys

from robolivemem.agents import Agent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripted_agent import STALE  # noqa: E402


class StaleAgent(Agent):
    name = "stale"

    def __init__(self):
        self.think = float(os.environ.get("THINK", "8"))

    def reset(self, obs):
        self.plan = list(STALE[obs["spec"]["scenario"]])

    def act(self, obs):
        if not self.plan:
            return {"tool": "robot_done"}, None, 0.0
        skill, mark = self.plan.pop(0)
        if skill == "done":
            return {"tool": "robot_done"}, None, 0.0
        return {"tool": "robot_act", "skill": skill, "mark": mark}, None, self.think
