"""S08 Parcel locker (office mailroom, multi-robot courier, location inside a closed container). v2s strict version.

Dr. Lee's parcel waits on the drop-off shelf; six identical closed lockers stand in front of the robot. The task: bring
the parcel to the reception desk. The courier robot (never named to the agent) becomes due at world U(0.3, 2.5) s and
starts only while the agent's robot stands still. Variant = seed % 3:
  event : it carries the parcel to locker k, opens the door, puts the parcel in and closes the door;
  decoy : it stops at the shelf without taking the parcel, then opens locker k, looks in and closes it again;
  null  : it comes to the drop-off shelf, stops for a moment and leaves without the parcel.
visible control: in the event the courier leaves the locker door open.
"""
import math
import numpy as np

_FR = Frame(*SCENE_SPEC["anchors"]["s08"])
_F = 1.20                       # locker front face (forward distance)
_COLS = [0.38, 0.0, -0.38]      # column centres (left offsets)
_ROWS = [0.62, 1.02]            # compartment centre heights (lower, upper)
_W, _H, _DP = 0.36, 0.38, 0.42
_objs = []
_bank_rgba = [0.42, 0.45, 0.50, 1]
_objs += [box("s08_back", _FR, _F + _DP, 0.0, 0.81, 0.03, 1.20, 1.62, _bank_rgba)]
_objs += [box("s08_top", _FR, _F + _DP / 2, 0.0, 1.235, _DP, 1.20, 0.03, _bank_rgba),
          box("s08_plinth", _FR, _F + _DP / 2, 0.0, 0.21, _DP, 1.20, 0.42, _bank_rgba),
          box("s08_mid", _FR, _F + _DP / 2, 0.0, 0.82, _DP, 1.20, 0.025, _bank_rgba)]
for j, dl in enumerate([0.585, 0.19, -0.19, -0.585]):
    _objs += [box(f"s08_side{j}", _FR, _F + _DP / 2, dl, 0.83, _DP, 0.03, 0.82, _bank_rgba)]
for r, z in enumerate(_ROWS):
    for c, dl in enumerate(_COLS):
        k = r * 3 + c
        _objs += [box(f"s08_door{k}", _FR, _F - 0.012, dl, z, 0.02, _W - 0.02, _H - 0.02, [0.70, 0.73, 0.78, 1])]
_objs += [box("s08_parcel", _FR, 0.55, 0.78, 0.92, 0.18, 0.22, 0.13, CARD)]
_objs += table("s08_dropshelf", _FR, 0.55, 0.78, 0.40, 0.44, 0.85, rgba=WOOD)
_objs += table("s08_desk", _FR, 0.40, -0.85, 0.50, 0.70, 0.76, rgba=[0.85, 0.85, 0.82, 1])


@register
class S08ParcelLocker(StationScenario):
    sid = "s08_parcel_locker"
    title = "Parcel locker"
    setting = "office mailroom"
    ANCHOR_KEY = "s08"
    OBJECTS = _objs
    budget = 3
    skills = ("open", "close", "pick", "place", "wait", "done")
    B_START = (0.6, 2.4, -math.pi / 2)
    hand_z = 1.0
    # spectator: high behind the agent so the drop-off shelf (left), the locker bank and the desk (right) all show
    CAM = staticmethod(lambda fr: (fr.p(-1.45, 0.25, 3.05), fr.p(0.75, 0.0, 0.7), 13.0))

    def build_scenario(self):
        fr = self.fr
        for r, z in enumerate(_ROWS):
            for c, dl in enumerate(_COLS):
                k = r * 3 + c
                self.slots[f"locker{k}"] = dict(pos=fr.p(_F + 0.18, dl, z - _H / 2 + 0.02), label=f"locker {'lower' if r == 0 else 'upper'} row, column {c + 1}",
                                                base=fr.base(0.62, dl + 0.12), arm="right", grasp_z=0.065, mark_pos=fr.p(_F - 0.03, dl, z),
                                                pre=(np.asarray(fr.p(-0.22, 0, 0.04)) - np.asarray(fr.p(0, 0, 0))).tolist(),
                                                approach=[math.cos(fr.yaw), math.sin(fr.yaw), -0.15])
        self.slots["drop"] = dict(pos=fr.p(0.55, 0.78, 0.85), label="drop-off shelf", arm="left", grasp_z=0.065)
        self.slots["desk"] = dict(pos=fr.p(0.40, -0.85, 0.76), label="reception desk", arm="right", grasp_z=0.065)
        self.hz["s08_parcel"] = 0.065
        self.doors = {}
        for r, z in enumerate(_ROWS):
            for c, dl in enumerate(_COLS):
                k = r * 3 + c
                hinge = fr.p(_F - 0.012, dl + (_W - 0.02) / 2, z)
                ctr = np.asarray(fr.p(_F - 0.012, dl, z))
                self.doors[k] = Hinged(self, f"locker{k}", f"s08_door{k}", hinge, [0, 0, 1], ctr - np.asarray(hinge), -1.75)
        # helper poses: stands to the left of a column, right arm reaches the door / compartment
        self.B_at = {c: fr.base(0.66, dl + 0.42, 0.0) for c, dl in enumerate(_COLS)}
        self.B_drop = fr.base(0.55, 1.30, -math.pi / 2)

    def mark_points(self):
        return [m.get("mark_pos", m["pos"]) for m in self.marked()]

    def describe(self, name):
        return "a parcel"

    def instruction(self):
        return ("Dr. Lee's parcel (a cardboard box) is waiting for pickup in the mailroom; right now it is on the drop-off "
                "shelf (mark 7). Bring it to the reception desk (mark 8). Marks 1 to 6 are closed lockers. "
                "Opening a locker, picking and placing each use one of your 3 manipulation actions.")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}; self.containers = {}
        self.place_robots()
        self.pin_home("s08_")
        for k, d in self.doors.items():
            self.containers[f"locker{k}"] = {"open": False, "door": d}
            d.angle = 0.0; d.apply()
        self.put("s08_parcel", "drop")
        self.hidden = {}
        self.v2s_setup(ep, rng)
        self.target = rng.randrange(6)
        self.hidden.update(locker=self.target + 1)
        self.helper = Helper()

    def g_container(self, bot, arm_, slot, open_, base):
        k = int(slot.replace("locker", ""))
        if bot is self.A:
            yield from g_base(bot, [base[:2]], final_yaw=base[2], speed=0.35)
        yield from self.doors[k].g_swing(bot, arm_, base, open_, arm="right")
        if bot is self.A:
            yield from g_base(bot, [self.baseA[:2]], final_yaw=self.baseA[2], speed=0.35)

    def courier_plan(self, ep):
        B, aB, k = self.B, self.armB, self.target
        r, c = divmod(k, 3)
        yield from g_base(B, [self.B_drop[:2]], final_yaw=self.B_drop[2], speed=0.5)
        if self.variant == "null" or self.occ.get("drop") != "s08_parcel":
            yield from g_still(2.0)
            self.helper.status = "no parcel to store: leaving"
            yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5); return
        if self.variant == "decoy":
            # decoy: stops at the shelf without taking the parcel, then opens locker k, looks in, closes it and leaves
            yield from g_still(1.5)
            base = self.B_at[c]
            yield from g_base(B, [base[:2]], final_yaw=base[2], speed=0.45)
            yield from self.doors[k].g_swing(B, aB, base, True, arm="left")
            yield from g_still(1.0)
            yield from self.doors[k].g_swing(B, aB, base, False, arm="left")
            ep.ev("courier_decoy", locker=k + 1)
            self.helper.status = "checked a locker: leaving"
            yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5); return
        o = obj("s08_parcel")
        yield from aB.g_pick(self.B_drop, "right", self.grasp_point("drop"), obj=o, on_grasp=lambda: (self.occ.__setitem__("drop", None), self.pins.unpin(o)), keep_quat=self.fr.q())
        # carry with the right hand, open with the left
        base = self.B_at[c]
        yield from g_base(B, [base[:2]], final_yaw=base[2], speed=0.45)
        yield from self.doors[k].g_swing(B, aB, base, True, arm="left")
        if self.variant == "event":
            sl = self.slots[f"locker{k}"]
            yield from aB.g_place(base, "right", self.grasp_point(f"locker{k}"), obj=o, quat=self.fr.q(), pins=self.pins, pre=sl["pre"], approach=sl["approach"])
            self.occ[f"locker{k}"] = "s08_parcel"
            if not self.visible:
                yield from self.doors[k].g_swing(B, aB, base, False, arm="left")
            ep.ev("courier_stored", locker=k + 1)
            self.helper.status = "stored the parcel: leaving"
        else:
            # decoy: closes the door again without storing the parcel and puts the parcel back on the drop-off shelf
            yield from self.doors[k].g_swing(B, aB, base, False, arm="left")
            yield from g_base(B, [self.B_drop[:2]], final_yaw=self.B_drop[2], speed=0.45)
            yield from aB.g_place(self.B_drop, "right", self.grasp_point("drop"), obj=o, quat=self.fr.q(), pins=self.pins)
            self.occ["drop"] = "s08_parcel"
            ep.ev("courier_decoy", locker=k + 1)
            self.helper.status = "returned the parcel: leaving"
        yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5)

    def world(self, ep):
        super().world(ep)
        if not self.helper.started and self.actor_due(ep):
            self.helper.start(self.courier_plan(ep), "courier storing parcels"); self.actor_started(ep)
        self.helper.step()

    def privileged(self):
        return dict(parcel_at=self.where("s08_parcel"), open_lockers=[k + 1 for k, d in self.doors.items() if self.containers[f"locker{k}"]["open"]],
                    courier=self.helper.status)

    def verify(self, final):
        if final["parcel_at"] == "desk":
            return True, "the parcel is on the reception desk"
        return False, f"parcel not delivered (it is at {final['parcel_at']})"
