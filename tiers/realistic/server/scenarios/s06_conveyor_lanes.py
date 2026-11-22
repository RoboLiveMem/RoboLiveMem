"""S06 Order tote on a sorting conveyor (warehouse, machine, location under machine motion). v2s strict version.

Three identical totes queue on a conveyor. At the first observation the order tote stands at the pick station under the
scanner. The task: put the order tote on the packing table. The conveyor (never named as an actor to the agent) becomes
due at world U(0.3, 2.5) s and starts only while the agent's robot stands still. Variant = seed % 3:
  event : the belt restarts and a right-angle sorter sends every tote into one of three spur lanes (random assignment);
          the pick station is empty and three identical totes wait at the spur ends;
  decoy : the scanner lamp goes off and the belt carries the order tote 0.3 m toward the sorter, then backs it up under
          the scanner again (re-scan) and stops; the order tote ends at the pick station;
  null  : the belt stays off.
"""
import math
import numpy as np

_FR = Frame(*SCENE_SPEC["anchors"]["s06"])
_BF = 1.02            # main belt centre line (forward); the robot stands back to see the belt, the sorter and the spurs
_ZB = 0.78            # belt top
_PICK = 0.42          # pick station (left offset)
_SPURS = [0.0, -0.42, -0.84]
_SPUR_END = 1.92      # forward position of a tote at the end of a spur
_TS = (0.34, 0.26, 0.20)
_objs = []
_objs += [box("s06_belt", _FR, _BF, 0.45, _ZB - 0.03, 0.46, 3.10, 0.06, [0.12, 0.12, 0.13, 1])]
_objs += [box("s06_rail_f", _FR, _BF + 0.25, 0.45, _ZB + 0.02, 0.03, 3.10, 0.08, [0.75, 0.75, 0.72, 1]),
          box("s06_rail_b", _FR, _BF - 0.25, 0.80, _ZB + 0.02, 0.03, 2.40, 0.08, [0.75, 0.75, 0.72, 1])]
_objs += [box("s06_frame", _FR, _BF, 0.45, (_ZB - 0.06) / 2, 0.40, 3.0, _ZB - 0.06, [0.30, 0.33, 0.36, 1], visual=True)]
for k, dl in enumerate(_SPURS):
    _objs += [box(f"s06_spur{k}", _FR, (_BF + 0.25 + _SPUR_END + 0.2) / 2, dl, _ZB - 0.03, _SPUR_END + 0.2 - _BF - 0.25, 0.34, 0.06, [0.12, 0.12, 0.13, 1])]
    _objs += [box(f"s06_spurframe{k}", _FR, (_BF + 0.25 + _SPUR_END + 0.2) / 2, dl, (_ZB - 0.06) / 2, _SPUR_END + 0.1 - _BF - 0.25, 0.28, _ZB - 0.06,
                  [0.30, 0.33, 0.36, 1], visual=True)]
    _objs += [box(f"s06_stop{k}", _FR, _SPUR_END + 0.21, dl, _ZB + 0.06, 0.03, 0.34, 0.12, YELLOW)]
# scanner arch over the pick station
_objs += [box("s06_arch_l", _FR, _BF + 0.33, _PICK, 1.05, 0.05, 0.05, 0.62, YELLOW),
          box("s06_arch_r", _FR, _BF - 0.33, _PICK, 1.05, 0.05, 0.05, 0.62, YELLOW, visual=True),
          box("s06_arch_t", _FR, _BF, _PICK, 1.38, 0.72, 0.07, 0.07, YELLOW),
          box("s06_scanner", _FR, _BF, _PICK, 1.31, 0.14, 0.10, 0.08, DARK)]
_objs += [sphere("s06_lamp_on", _FR, _BF + 0.12, _PICK, 1.44, 0.04, [0.2, 1.0, 0.3, 1])]
_objs += [sphere("s06_lamp_off", _FR, _BF + 0.12, _PICK, -3.0, 0.04, OFF)]
for i in range(3):
    _objs += [box(f"s06_tote{i}", _FR, _BF, _PICK + 0.5 * i, _ZB + _TS[2] / 2, _TS[0], _TS[1], _TS[2], BLUE),
              box(f"s06_tlid{i}", _FR, _BF, _PICK + 0.5 * i, -4.0, _TS[0] + 0.01, _TS[1] + 0.01, 0.02, [0.10, 0.24, 0.46, 1], visual=True)]
_objs += table("s06_pack", _FR, 0.30, 0.95, 0.50, 0.46, 0.74, rgba=WOOD)


@register
class S06ConveyorLanes(StationScenario):
    sid = "s06_conveyor_lanes"
    title = "Order tote"
    setting = "warehouse"
    ANCHOR_KEY = "s06"
    OBJECTS = _objs
    budget = 2
    skills = ("pick", "place", "wait", "done")
    hand_z = 1.0
    CAM = staticmethod(lambda fr: (fr.p(-1.1, -1.4, 2.2), fr.p(1.0, 0.0, 0.8), 14.0))

    def build_scenario(self):
        fr = self.fr
        self.slots["pick"] = dict(pos=fr.p(_BF, _PICK, _ZB), label="pick station under the scanner", grasp_z=_TS[2] + 0.02,
                                  arm="right", base=fr.base(_BF - 0.55, _PICK - 0.12))
        for k, dl in enumerate(_SPURS):
            self.slots[f"spur{k}"] = dict(pos=fr.p(_SPUR_END, dl, _ZB), label=f"end of spur lane {'ABC'[k]}", grasp_z=_TS[2] + 0.02,
                                          base=fr.base(_SPUR_END - 0.62, dl - 0.12), arm="right")
        self.slots["pack"] = dict(pos=fr.p(0.30, 0.95, 0.74), label="packing table", grasp_z=_TS[2] + 0.02, arm="right",
                                  base=fr.base(0.30, 0.95 - 0.55, math.pi / 2))
        for i in range(3):
            self.hz[f"s06_tote{i}"] = _TS[2] / 2

    def describe(self, name):
        return "a tote"

    def instruction(self):
        return ("The tote at the pick station under the scanner (mark 1) is order #17. Put the order #17 tote on the "
                "packing table (mark 5). All totes look the same.")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}
        self.place_robots()
        self.pin_home("s06_")
        self.order = "s06_tote0"
        self.put("s06_tote0", "pick")
        for i in (1, 2):
            self.pins.move(obj(f"s06_tote{i}"), self.fr.p(_BF, _PICK + 0.5 * i, _ZB + _TS[2] / 2), self.fr.q())
        self.queue = {"s06_tote1": _PICK + 0.5, "s06_tote2": _PICK + 1.0}
        self.hidden = {}
        self.v2s_setup(ep, rng)
        self.assign = rng.sample(range(3), 3)        # tote i -> spur assign[i]
        if self.variant == "event":
            self.hidden.update(order_spur="ABC"[self.assign[0]], assignment={f"tote{i}": "ABC"[k] for i, k in enumerate(self.assign)})
        self.mover = PathMover(); self.running = False; self.actor_on = False; self.lamp(True)

    def lamp(self, on):
        p = np.asarray(self.fr.p(_BF + 0.12, _PICK, 1.44))
        self.pins.move(obj("s06_lamp_on"), p if on else p + np.array([0, 0, -6]))
        self.pins.move(obj("s06_lamp_off"), p + np.array([0, 0, -6]) if on else p)

    def start_belt(self, ep):
        fr = self.fr; z = _ZB + _TS[2] / 2
        self.running = True; self.lamp(False)
        for i in range(3):
            name = f"s06_tote{i}"
            if name == self.order and self.occ.get("pick") != name:
                continue          # the agent already took the order tote
            dl0 = _PICK if name == self.order else self.queue[name]
            if name == self.order:
                self.occ["pick"] = None
            k = self.assign[i]
            pts = [fr.p(_BF, dl0, z), fr.p(_BF, _SPURS[k], z), fr.p(_SPUR_END, _SPURS[k], z)]
            self.pins.unpin(obj(name))
            self.mover.add(name, pts, 0.30, ep.world_t, quat=fr.q(), on_done=(lambda n=name, s=f"spur{k}": self.arrive(n, s)))
        ep.ev("belt_start")

    def start_jog(self, ep):
        """Decoy: the scanner lamp goes off and the belt starts, carrying the order tote downstream toward the sorter (as at
        the start of a sort), then the belt backs it up under the scanner again (re-scan) and stops; the lamp comes back on.
        The order tote ends where it started; the queued totes only creep forward."""
        fr = self.fr; z = _ZB + _TS[2] / 2
        self.running = True; self.lamp(False)
        o = self.order
        self.occ["pick"] = None
        self.pins.unpin(obj(o))
        p0 = fr.p(_BF, _PICK, z); p1 = fr.p(_BF, _PICK - 0.30, z)
        self.mover.add(o, [p0, p1, p0], 0.30, ep.world_t, quat=fr.q(), on_done=lambda: self._jog_back(o))
        for name in ("s06_tote1", "s06_tote2"):
            dl0 = self.queue[name]
            self.pins.unpin(obj(name))
            self.mover.add(name, [fr.p(_BF, dl0, z), fr.p(_BF, dl0 - 0.20, z), fr.p(_BF, dl0 - 0.20, z)], 0.30, ep.world_t, quat=fr.q())
        ep.ev("belt_jog")

    def _jog_back(self, o):
        self.occ["pick"] = o
        self.lamp(True)

    def arrive(self, name, slot):
        self.occ[slot] = name; self.pins.pin(obj(name))

    def workspace_busy(self):
        if self.running and any(not j["done"] for j in self.mover.jobs.values()):
            return "the conveyor is running"
        return None

    def world(self, ep):
        super().world(ep)
        if not self.actor_on and self.variant != "null" and self.actor_due(ep):
            self.actor_on = True; self.actor_started(ep)
            if self.variant == "event":
                self.start_belt(ep)
            else:
                self.start_jog(ep)
        if self.running:
            self.mover.step(self, ep.world_t)

    def privileged(self):
        return dict(order_tote_at=self.where(self.order), occ=dict(self.occ))

    def verify(self, final):
        if final["order_tote_at"] == "pack":
            return True, "the order tote is on the packing table"
        if final["occ"].get("pack"):
            return False, "a wrong tote is on the packing table"
        return False, f"order tote not delivered (at {final['order_tote_at']})"
