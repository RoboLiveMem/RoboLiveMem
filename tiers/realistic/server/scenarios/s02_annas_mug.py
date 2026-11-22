"""S02 Anna's mug (home dining room, multi-robot, identity + location among identical objects). v2s strict version.

Two identical mugs stand on the dining table; Anna's is the one at mark 1 when the episode starts. The task: put Anna's
mug on Anna's tray. The household helper robot (carrying a cutting board; never named to the agent) becomes due at world
U(0.3, 2.5) s and starts only while the agent's robot stands still. Variant = seed % 3:
  event : it moves Anna's mug (spot 1) to spot 3 and lays the board where the mug stood;
  decoy : it moves the other mug (spot 2) to spot 3 and lays the board at spot 2 (Anna's mug stays at spot 1);
  null  : it lays the board on the free spot 3 and leaves (no mug moves).
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["s02"]   # surface: [x, y, yaw] front-edge origin facing into the table; h; A, B: world poses
_FR = Frame(*_A["surface"])
_H = _A["h"]
_SPOTS = {"spot1": (0.25, 0.30), "spot2": (0.25, 0.0), "spot3": (0.25, -0.34), "tray": (0.30, 0.62)}
_M = SCENE_SPEC["models"]
_objs = [
    {"type": "DatasetObject", "name": "s02_mug_a", "category": "mug", "model": _M["mug"], "kinematic_only": True, "visual_only": True, "position": park(0)},
    {"type": "DatasetObject", "name": "s02_mug_b", "category": "mug", "model": _M["mug"], "kinematic_only": True, "visual_only": True, "position": park(1)},
    {"type": "DatasetObject", "name": "s02_board", "category": "chopping_board", "model": _M["chopping_board"], "kinematic_only": True, "visual_only": True, "position": park(2)},
    {"type": "DatasetObject", "name": "s02_tray", "category": "tray", "model": _M["tray"], "kinematic_only": True, "visual_only": True, "position": park(3)},
]


@register
class S02AnnasMug(StationScenario):
    sid = "s02_annas_mug"
    title = "Anna's mug"
    setting = "home dining room"
    ANCHOR_KEY = "s02"
    OBJECTS = _objs
    budget = 2
    skills = ("pick", "place", "wait", "done")
    A_WORLD = _A["A"]
    B_WORLD = _A["B"]
    B_START = (0, 0, 0)
    hand_z = _H + 0.15
    CAM = staticmethod(lambda fr: tuple(_A["cam"]))

    def __init__(self):
        super().__init__()
        self.fr = Frame(*_A["surface"])

    def build_scenario(self):
        fr = self.fr
        labels = {"spot1": "table spot 1", "spot2": "table spot 2", "spot3": "table spot 3", "tray": "Anna's tray"}
        for n in ["s02_mug_a", "s02_mug_b", "s02_board", "s02_tray"]:
            lo, hi = [npf(v) for v in obj(n).aabb]
            self.hz[n] = float(hi[2] - lo[2]) / 2 + 0.002
        for k, (df, dl) in _SPOTS.items():
            z = _H + (2 * self.hz["s02_tray"] - 0.004 if k == "tray" else 0.0)
            self.slots[k] = dict(pos=fr.p(df, dl, z), label=labels[k], grasp_z=self.hz["s02_mug_a"] * 1.2,
                                 arm="right", base=fr.base(-0.42, dl - 0.12), via=_A.get("via", []))

    def describe(self, name):
        return "a mug" if "mug" in name else "a cutting board"

    def instruction(self):
        return ("Anna's mug is the one at mark 1 right now (the two mugs look the same). Put Anna's mug on Anna's tray "
                "(mark 4).")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}
        self.place_robots()
        fr = self.fr
        self.pins.move(obj("s02_tray"), np.asarray(fr.p(*_SPOTS["tray"], _H + self.hz["s02_tray"])), fr.q())
        self.put("s02_mug_a", "spot1"); self.put("s02_mug_b", "spot2")
        self.occ["spot3"] = None; self.occ["tray"] = None
        # the helper carries the cutting board in its left hand
        self.B.apply()
        board = obj("s02_board")
        set_pose(board, npf(self.Br.get_eef_position("left")) + np.array([0, 0, -0.03]), fr.q(math.pi / 2))
        self.B.attach(board, "left", fixed_quat=fr.q(math.pi / 2))
        self.hidden = {}
        self.v2s_setup(ep, rng)
        self.move_which = {"event": "spot1", "decoy": "spot2", "null": None}[self.variant]
        self.hidden.update(moved={"spot1": "Anna's mug", "spot2": "the other mug", None: "no mug (board onto spot 3)"}[self.move_which])
        self.helper = Helper()

    def g_goto(self, bot, base, speed=0.5):
        yield from g_base(bot, [base[:2]], final_yaw=base[2], speed=speed)

    def helper_plan(self, ep):
        B, aB = self.B, self.armB
        src = self.move_which
        board = obj("s02_board")
        if src is None:
            # null: lays the cutting board on the free spot 3 and leaves; no mug moves
            s3 = self.slots["spot3"]
            yield from self.g_goto(B, s3["base"])
            p = np.asarray(s3["pos"]) + np.array([0, 0, self.hz["s02_board"]])
            yield from aB.g_place(s3["base"], "left", p + np.array([0, 0, 0.03]), obj=board, quat=self.fr.q(math.pi / 2), pins=self.pins)
            self.occ["spot3"] = "s02_board"
            ep.ev("helper_board_down", at="spot3")
            self.helper.status = "done: leaving"
            yield from self.g_goto(B, self.baseB0)
            return
        name = self.occ.get(src)
        if name is not None and not self.A_busy:
            o = obj(name); s0 = self.slots[src]; s3 = self.slots["spot3"]
            yield from self.g_goto(B, s0["base"])
            yield from aB.g_pick(s0["base"], "right", self.grasp_point(src), obj=o,
                                 on_grasp=lambda: (self.occ.__setitem__(src, None), self.pins.unpin(o)), keep_quat=self.fr.q())
            yield from self.g_goto(B, s3["base"], 0.3)
            yield from aB.g_place(s3["base"], "right", self.grasp_point("spot3"), obj=o, quat=self.fr.q(), pins=self.pins)
            self.occ["spot3"] = name
            ep.ev("helper_moved_mug", src=src)
            yield from self.g_goto(B, s0["base"], 0.3)
            p = np.asarray(s0["pos"]) + np.array([0, 0, self.hz["s02_board"]])
            yield from aB.g_place(s0["base"], "left", p + np.array([0, 0, 0.03]), obj=board, quat=self.fr.q(math.pi / 2), pins=self.pins)
            self.occ[src] = "s02_board"
            ep.ev("helper_board_down", at=src)
        else:
            self.helper.status = "table changed: helper leaves"
        self.helper.status = "done: leaving"
        yield from self.g_goto(B, self.baseB0)

    def world(self, ep):
        super().world(ep)
        if not self.helper.started and self.actor_due(ep):
            self.helper.start(self.helper_plan(ep), "helper making room for the cutting board"); self.actor_started(ep)
        self.helper.step()

    def privileged(self):
        return dict(annas_mug_at=self.where("s02_mug_a"), other_mug_at=self.where("s02_mug_b"), board_at=self.where("s02_board"),
                    helper=self.helper.status if hasattr(self, "helper") else "")

    def verify(self, final):
        if final["annas_mug_at"] == "tray":
            return True, "Anna's mug is on Anna's tray"
        if final["other_mug_at"] == "tray":
            return False, "the other mug is on Anna's tray"
        if final["board_at"] == "tray":
            return False, "the cutting board was put on Anna's tray"
        return False, f"Anna's mug is not on the tray (at {final['annas_mug_at']})"
