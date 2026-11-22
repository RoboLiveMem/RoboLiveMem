"""S01 Put-away (home kitchen, multi-robot, location inside a closed container). v2s strict live-memory version.

An apple lies on the kitchen counter; three identical closed storage bins stand behind it. The task: put the apple into
the fruit bowl on the island. The helper robot (never named to the agent) becomes due at world U(0.3, 2.5) s and starts
only while the agent's robot stands still. Variant = seed % 3:
  event : it carries the apple to bin k, lifts the lid, puts the apple in and closes the lid (apple hidden in bin k);
  decoy : it carries the apple to bin k, lifts the lid, puts the lid back and returns the apple to its counter spot;
  null  : it passes by the counter and leaves without touching anything.
visible control: in the event the lid is left off, so the apple shows in the open bin.
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["s01"]       # surface frame at the counter front edge; h; apple [df, dl]; bowl world [x, y, z, yaw]; A, B, cam
_FR = Frame(*_A["surface"])
_H = _A["h"]
_BINS = [(0.30, -0.36), (0.30, 0.0), (0.30, 0.36)]
_APPLE = tuple(_A["apple"])
_BW, _BH = 0.22, 0.17
_M = SCENE_SPEC["models"]
_objs = []
for i, (df, dl) in enumerate(_BINS):
    _objs += [box(f"s01_bin{i}", _FR, df, dl, _H + _BH / 2, _BW, _BW, _BH, [0.86, 0.84, 0.78, 1])]
    _objs += [box(f"s01_lid{i}", _FR, df, dl, _H + _BH + 0.012, _BW + 0.02, _BW + 0.02, 0.024, [0.55, 0.42, 0.30, 1])]
_objs += [{"type": "DatasetObject", "name": "s01_apple", "category": "apple", "model": _M["apple"], "kinematic_only": True, "visual_only": True, "position": park(20)},
          {"type": "DatasetObject", "name": "s01_bowl", "category": "bowl", "model": _M["fruit_bowl"], "kinematic_only": True, "visual_only": True, "position": park(21)}]


@register
class S01PutAway(StationScenario):
    sid = "s01_put_away"
    title = "Put-away"
    setting = "home kitchen"
    ANCHOR_KEY = "s01"
    OBJECTS = _objs
    budget = 3
    skills = ("open", "close", "pick", "place", "wait", "done")
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
        for i, (df, dl) in enumerate(_BINS):
            self.slots[f"bin{i}"] = dict(pos=fr.p(df, dl, _H + 0.012), label=f"storage bin {i + 1}", grasp_z=0.04, arm="right",
                                         base=fr.base(-0.42, dl - 0.10), mark_pos=fr.p(df, dl, _H + _BH + 0.03))
        for n in ["s01_apple", "s01_bowl"]:
            lo, hi = [npf(v) for v in obj(n).aabb]
            self.hz[n] = float(hi[2] - lo[2]) / 2 + 0.002
        self.slots["table"] = dict(pos=fr.p(*_APPLE, _H), label="the counter spot where the apple lies (just beyond this mark)",
                                   grasp_z=self.hz["s01_apple"], arm="right", base=fr.base(-0.42, _APPLE[1] - 0.10),
                                   mark_pos=fr.p(_APPLE[0] - 0.09, _APPLE[1], _H))   # main_1115: the mark hid the apple
        bx, by, bz, byaw = _A["bowl"]
        bb = _A["bowl_base"]
        self.slots["bowl"] = dict(pos=[bx, by, bz + 0.02], label="fruit bowl on the island", grasp_z=self.hz["s01_apple"] + 0.04,
                                  arm="right", stack=True, base=tuple(bb), jitter=[0.0, 0.0, 0.0])
        self.lids = {}
        for i, (df, dl) in enumerate(_BINS):
            self.lids[i] = LidBox(self, f"bin{i}", f"s01_lid{i}", fr.p(df, dl, _H + _BH + 0.012), fr.p(df - 0.24, dl, _H + 0.013))

    def describe(self, name):
        return "an apple" if "apple" in name else "a lid"

    def instruction(self):
        return ("Put the apple into the fruit bowl on the island (mark 5). Right now the apple lies on the counter (mark 4). "
                "Marks 1 to 3 are closed storage bins. "
                "Opening a bin, picking and placing each use one of your 3 manipulation actions.")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}; self.containers = {}
        self.place_robots()
        fr = self.fr
        for i, (df, dl) in enumerate(_BINS):
            self.pins.move(obj(f"s01_bin{i}"), fr.p(df, dl, _H + _BH / 2), fr.q())
            self.containers[f"bin{i}"] = {"open": False, "box": self.lids[i]}
            self.lids[i].cur_open = self.lids[i].open_pos.copy()
            self.lids[i].set(False)
            self.occ[f"bin{i}"] = None
        bx, by, bz, byaw = _A["bowl"]
        self.pins.move(obj("s01_bowl"), [bx, by, bz + self.hz["s01_bowl"]], [0, 0, math.sin(byaw / 2), math.cos(byaw / 2)])
        self.put("s01_apple", "table")
        self.hidden = {}
        self.v2s_setup(ep, rng)
        self.target = rng.randrange(3)
        self.hidden.update(bin=self.target + 1)
        self.helper = Helper()

    def g_container(self, bot, arm_, slot, open_, base):
        i = int(slot[3:]); df, dl = _BINS[i]
        if bot is self.A:
            yield from g_base(bot, [base[:2]], final_yaw=base[2], speed=0.4)
        yield from self.lids[i].g_move(bot, arm_, base, open_, arm="right", open_pos=self.fr.p(df - 0.24, dl, _H + 0.013))
        if bot is self.A:
            yield from g_base(bot, [self.baseA[:2]], final_yaw=self.baseA[2], speed=0.4)

    def helper_plan(self, ep):
        B, aB, k = self.B, self.armB, self.target
        st = self.slots["table"]; sb = self.slots[f"bin{k}"]
        if self.variant == "null":
            # passes by the counter on its own errand and leaves; nothing is touched
            yield from g_base(B, [st["base"][:2]], final_yaw=st["base"][2], speed=0.5)
            yield from g_still(2.0)
            self.helper.status = "passed by: leaving"
            yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5)
            return
        if self.occ.get("table") != "s01_apple":
            self.helper.status = "nothing to put away: leaving"
            yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5)
            return
        yield from g_base(B, [st["base"][:2]], final_yaw=st["base"][2], speed=0.5)
        o = obj("s01_apple")
        yield from aB.g_pick(st["base"], "right", self.grasp_point("table"), obj=o,
                             on_grasp=lambda: (self.occ.__setitem__("table", None), self.pins.unpin(o)))
        yield from g_base(B, [sb["base"][:2]], final_yaw=sb["base"][2], speed=0.4)
        df, dl = _BINS[k]
        yield from self.lids[k].g_move(B, aB, sb["base"], True, arm="left", open_pos=self.fr.p(df - 0.24, dl, _H + 0.013))
        if self.variant == "event":
            yield from aB.g_place(sb["base"], "right", self.grasp_point(f"bin{k}"), obj=o, pins=self.pins)
            self.occ[f"bin{k}"] = "s01_apple"
            if not self.visible:
                yield from self.lids[k].g_move(B, aB, sb["base"], False, arm="left")
            ep.ev("helper_put_away", bin=k + 1)
            self.helper.status = "put the apple away: leaving"
        else:
            # decoy: puts the lid back without storing anything and returns the apple to its spot on the counter
            yield from self.lids[k].g_move(B, aB, sb["base"], False, arm="left")
            yield from g_base(B, [st["base"][:2]], final_yaw=st["base"][2], speed=0.4)
            yield from aB.g_place(st["base"], "right", self.grasp_point("table"), obj=o, pins=self.pins)
            self.occ["table"] = "s01_apple"
            ep.ev("helper_decoy", bin=k + 1)
            self.helper.status = "returned the apple: leaving"
        yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5)

    def world(self, ep):
        super().world(ep)
        if not self.helper.started and self.actor_due(ep):
            self.helper.start(self.helper_plan(ep), "helper tidying the kitchen"); self.actor_started(ep)
        self.helper.step()

    def privileged(self):
        return dict(apple_at=self.where("s01_apple") or ("bowl" if "s01_apple" in self.occ.get("bowl_items", []) else None),
                    open_bins=[i + 1 for i in range(3) if self.containers.get(f"bin{i}", {}).get("open")],
                    helper=self.helper.status if hasattr(self, "helper") else "")

    def verify(self, final):
        if final["apple_at"] == "bowl":
            return True, "the apple is in the fruit bowl"
        return False, f"the apple is not in the bowl (at {final['apple_at']})"
