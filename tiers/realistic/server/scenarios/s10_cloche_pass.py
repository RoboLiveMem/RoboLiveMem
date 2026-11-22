"""S10 Cloche pass (restaurant, multi-robot, identity under identical covers). v2s strict version.

Three plates under identical steel cloches wait on the pass; the chef said the vegetarian plate is at mark 1. The task:
put the vegetarian plate on the tray for table 4. The expo robot on the kitchen side (never named to the agent) becomes
due at world U(0.3, 2.5) s and starts only while the agent's robot stands still. Variant = seed % 3:
  event : it regroups two covered plates and the vegetarian plate is one of them (1->4 then 2->1, or 1->4 then 3->1);
  decoy : it regroups two other plates (2->4 then 3->2); the vegetarian plate stays at mark 1;
  null  : it comes to the pass, stops for a moment and leaves.
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["s10"]       # dict(base, d, h, depth)
_FR = Frame(*_A["base"])

_D, _H, _DEP = _A["d"], _A["h"], _A["depth"]
_ROW = _D + _DEP / 2
_SPOTS = [(_ROW, 0.45), (_ROW, 0.15), (_ROW, -0.15), (_ROW, -0.45)]
_TRAY = (_D + 0.10, -0.80)
_DISH = {"veg": [0.35, 0.62, 0.22, 1], "steak": [0.45, 0.25, 0.15, 1], "fish": [0.93, 0.80, 0.62, 1]}
_objs = []
for i, (k, rgba) in enumerate(_DISH.items()):
    _objs += [cyl(f"s10_plate_{k}", _FR, 0, 0, -3.0 - 0.3 * i, 0.12, 0.015, WHITE),
              cyl(f"s10_food_{k}", _FR, 0, 0, -3.1 - 0.3 * i, 0.07, 0.03, rgba, visual=True),
              {"type": "PrimitiveObject", "name": f"s10_cloche_{k}", "primitive_type": "Sphere", "scale": [0.24, 0.24, 0.19],
               "rgba": [0.80, 0.81, 0.83, 1], "kinematic_only": True, "visual_only": True, "position": [0, 0, -3.2 - 0.3 * i]},
              sphere(f"s10_knob_{k}", _FR, 0, 0, -3.25 - 0.3 * i, 0.018, [0.6, 0.6, 0.62, 1], visual=True)]
_objs += [box("s10_tray", _FR, _TRAY[0], _TRAY[1], -2.0, 0.36, 0.30, 0.02, [0.22, 0.22, 0.24, 1])]
_PATTERNS = [((0, 3), (1, 0)), ((1, 3), (2, 1)), ((0, 3), (2, 0))]   # (from, to) moves


@register
class S10ClochePass(StationScenario):
    sid = "s10_cloche_pass"
    title = "Cloche pass"
    setting = "restaurant kitchen pass"
    ANCHOR_KEY = "s10"
    OBJECTS = _objs
    budget = 2
    skills = ("pick", "place", "wait", "done")
    B_START = (_D + _DEP + 0.45, 1.6, math.pi)
    hand_z = _H + 0.18
    CAM = staticmethod(lambda fr: tuple(_A["cam"]) if "cam" in _A else (fr.p(-0.9, -1.4, 2.0), fr.p(_ROW, -0.1, _H), 15.0))

    def build_scenario(self):
        fr = self.fr
        for i, (df, dl) in enumerate(_SPOTS):
            self.slots[f"pass{i}"] = dict(pos=fr.p(df, dl, _H), label=f"pass spot {i + 1}", grasp_z=0.035,
                                          arm="left" if dl > 0.0 else "right", mark_pos=fr.p(df, dl, _H + 0.13))
        self.slots["tray"] = dict(pos=fr.p(*_TRAY, _H + 0.02), label="tray for table 4", grasp_z=0.035, arm="right")
        self.baseB = fr.base(_D + _DEP + 0.40, 0.0, math.pi)

    def group(self, k):
        return [f"s10_plate_{k}", f"s10_food_{k}", f"s10_cloche_{k}", f"s10_knob_{k}"]

    def pose_group(self, k, p):
        p = np.asarray(p, float)
        offs = [0.0075, 0.03, 0.015, 0.11]
        for n, dz in zip(self.group(k), offs):
            self.pins.move(obj(n), p + np.array([0, 0, dz]), self.fr.q())
        # the cloche sphere is flattened by its scale; its centre sits at plate level so only the dome shows

    def put_dish(self, k, slot):
        self.occ[slot] = k
        self.pose_group(k, self.slots[slot]["pos"])

    def describe(self, name):
        return "a covered plate"

    def instruction(self):
        return ("Three plates wait under identical cloches on the pass (marks 1 to 4 are the pass spots). The chef said the "
                "VEGETARIAN plate is the one at mark 1 right now. Put the vegetarian plate on the tray for table 4 (mark 5).")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {f"pass{i}": None for i in range(4)}; self.occ["tray"] = None
        self.place_robots()
        fr = self.fr
        self.pins.move(obj("s10_tray"), fr.p(*_TRAY, _H + 0.01), fr.q())
        for k, slot in zip(_DISH, ["pass0", "pass1", "pass2"]):
            self.put_dish(k, slot)
        self.hidden = {}
        self.v2s_setup(ep, rng)
        self.pattern = {"event": rng.choice([0, 2]), "decoy": 1, "null": None}[self.variant]
        mv = _PATTERNS[self.pattern] if self.pattern is not None else []
        self.hidden.update(moves=[f"{a + 1}->{b + 1}" for a, b in mv])
        self.helper = Helper()

    # agent pick/place move whole covered plates
    def skill_pick(self, ep, slot, target):
        s = self.slots[slot]; base = self.baseA; arm = s["arm"]
        if self.A.holding():
            ep.result = "the hand is full; place the plate first"; return
        k = self.occ.get(slot)
        objs = [obj(n) for n in self.group(k)] if k else []

        def on_grasp():
            if k:
                self.occ[slot] = None
                for o in objs:
                    self.pins.unpin(o)
                    self.A.attach(o, arm, fixed_quat=self.fr.q())
        yield from self.armA.g_pick(base, arm, self.grasp_point(slot), obj=None, on_grasp=on_grasp)
        ep.result = f"picked up a covered plate at mark {target}" if k else f"nothing to grasp at mark {target}"

    def skill_place(self, ep, slot, target):
        s = self.slots[slot]; arm = s["arm"]
        held = self.A.holding()
        if not held:
            ep.result = "you are not holding anything"; return
        if self.occ.get(slot):
            ep.result = f"mark {target} is occupied"; return
        k = held[0].split("_")[-1]

        def release():
            for n in self.group(k):
                self.A.detach(obj(n))
            self.put_dish(k, slot)
        yield from self.armA.g_place(self.baseA, arm, self.grasp_point(slot), on_release=release)
        ep.result = f"put the covered plate at mark {target}"

    def helper_plan(self, ep):
        B, aB = self.B, self.armB
        yield from g_base(B, [self.baseB[:2]], final_yaw=self.baseB[2], speed=0.45)
        if self.pattern is None:
            yield from g_still(2.0)
        for a, b in (_PATTERNS[self.pattern] if self.pattern is not None else []):
            src, dst = f"pass{a}", f"pass{b}"
            k = self.occ.get(src)
            if k is None or self.occ.get(dst) is not None or self.A_busy:
                continue
            arm = "right" if self.slots[src]["pos"] is not None else "right"
            objs = [obj(n) for n in self.group(k)]

            def grab(k=k, src=src, objs=objs):
                self.occ[src] = None
                for o in objs:
                    self.pins.unpin(o); B.attach(o, "right", fixed_quat=self.fr.q())
            yield from aB.g_pick(self.baseB, "right", self.grasp_point(src), on_grasp=grab)
            def release(k=k, dst=dst):
                for n in self.group(k):
                    B.detach(obj(n))
                self.put_dish(k, dst)
            yield from aB.g_place(self.baseB, "right", self.grasp_point(dst), on_release=release)
            ep.ev("expo_moved", src=a + 1, dst=b + 1)
        self.helper.status = "regrouped: leaving"
        yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.45)

    def world(self, ep):
        super().world(ep)
        if not self.helper.started and self.actor_due(ep):
            self.helper.start(self.helper_plan(ep), "expo regrouping plates"); self.actor_started(ep)
        self.helper.step()

    def where_dish(self, k):
        for s, v in self.occ.items():
            if v == k:
                return s
        if f"s10_plate_{k}" in self.A.holding():
            return "agent_hand"
        if self.B is not None and f"s10_plate_{k}" in self.B.holding():
            return "helper_hand"
        return None

    def privileged(self):
        return dict(veg_at=self.where_dish("veg"), tray=self.occ.get("tray"), pass_=dict((k, v) for k, v in self.occ.items() if k.startswith("pass")))

    def verify(self, final):
        if final["veg_at"] == "tray":
            return True, "the vegetarian plate is on table 4's tray"
        if final["tray"]:
            return False, f"the {final['tray']} plate is on table 4's tray"
        return False, f"nothing delivered (vegetarian plate at {final['veg_at']})"
