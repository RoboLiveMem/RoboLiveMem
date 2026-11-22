"""S07 Press lamp (factory cell, machine, transient event / missed write). v2s strict version.

A curing press holds a part behind its closed door. The press (never named as an actor to the agent) becomes due at
world U(0.3, 2.5) s and acts only while the agent's robot stands still. Variant = seed % 3:
  event : the cycle ends: the green READY lamp on top lights for 6 s and goes dark again; nothing else changes;
  decoy : the press vents: the door slides a few centimetres open and closes again; the lamp stays dark and the cycle
          goes on (look-alike activity that does not end the cycle);
  null  : nothing happens (the cycle does not end during the episode).
The part must be unloaded only after the cycle has ended and within 90 s of the end; if the cycle never ended, the press
must stay closed. The only evidence that the cycle ended is the lamp in the robot's camera stream.
visible control: the lamp stays lit after the cycle ends.
"""
import math
import numpy as np

_FR = Frame(*SCENE_SPEC["anchors"]["s07"])
_D = 1.20     # press front face distance (the robot watches from further back; it steps in to unload)
_objs = []
_objs += [box("s07_body", _FR, _D + 0.35, 0.0, 0.70, 0.70, 0.80, 1.40, [0.30, 0.42, 0.55, 1])]
_objs += [box("s07_base", _FR, _D + 0.35, 0.0, 0.04, 0.80, 0.90, 0.08, DARK)]
_objs += [box("s07_door", _FR, _D - 0.015, 0.0, 0.92, 0.03, 0.56, 0.42, [0.78, 0.80, 0.82, 1])]
_objs += [box("s07_window", _FR, _D - 0.035, 0.0, 0.95, 0.01, 0.30, 0.16, [0.08, 0.09, 0.1, 1], visual=True)]
_objs += [box("s07_handle", _FR, _D - 0.05, -0.20, 0.92, 0.03, 0.03, 0.20, DARK, visual=True)]
_objs += [box("s07_panel", _FR, _D - 0.02, 0.30, 1.22, 0.02, 0.14, 0.22, DARK, visual=True)]
_objs += [cyl("s07_lampbase", _FR, _D + 0.2, 0.25, 1.43, 0.05, 0.06, DARK)]
_objs += [sphere("s07_lamp_off", _FR, _D + 0.2, 0.25, 1.50, 0.06, [0.20, 0.26, 0.20, 1])]
_objs += [sphere("s07_lamp_on", _FR, _D + 0.2, 0.25, -3.0, 0.065, [0.2, 1.0, 0.3, 1])]
_objs += [box("s07_part", _FR, _D + 0.25, 0.0, 0.80, 0.16, 0.16, 0.06, [0.82, 0.55, 0.25, 1])]
_objs += table("s07_cart", _FR, _D - 0.40, 0.70, 0.40, 0.40, 0.72, rgba=[0.2, 0.45, 0.7, 1])
_objs += [box("s07_floorline", _FR, _D - 0.35, 0.0, 0.002, 0.04, 1.6, 0.004, YELLOW, visual=True)]


@register
class S07PressLamp(StationScenario):
    sid = "s07_press_lamp"
    title = "Press ready lamp"
    setting = "factory work cell"
    ANCHOR_KEY = "s07"
    OBJECTS = _objs
    budget = 1
    skills = ("unload", "wait", "done")
    extra_help = {"unload": "unload: open the press door, take the part out, put it on the blue cart, close the door"}
    hand_z = 1.0
    deadline = 400.0
    CAM = staticmethod(lambda fr: (fr.p(-1.0, -1.5, 2.1), fr.p(_D - 0.1, 0.1, 0.9), 15.0))

    def build_scenario(self):
        fr = self.fr
        self.slots["press"] = dict(pos=fr.p(_D - 0.05, -0.20, 0.92), label="press door handle", reach=False)
        # v2s: the mark sits just right of the lamp so that it never covers the lamp in the camera image
        self.slots["lamp"] = dict(pos=fr.p(_D + 0.2, 0.25, 1.50), label="READY lamp, just left of this mark (lights green for a few seconds when the cycle ends)",
                                  reach=False, mark_pos=fr.p(_D + 0.2, 0.08, 1.50))
        self.slots["cart"] = dict(pos=fr.p(_D - 0.40, 0.70, 0.72), label="blue cart", reach=False)
        self.baseW = fr.base(_D - 0.62, 0.0)
        bw = self.baseW
        self.q_handle, _ = self.armA.reach(bw, "right", fr.p(_D - 0.07, -0.20, 0.92), approach=[math.cos(fr.yaw), math.sin(fr.yaw), 0])
        self.q_open, _ = self.armA.reach(bw, "right", fr.p(_D - 0.30, -0.45, 0.92), approach=[math.cos(fr.yaw), math.sin(fr.yaw), 0])
        self.q_part, _ = self.armA.reach(bw, "right", fr.p(_D + 0.05, 0.0, 0.86), approach=[math.cos(fr.yaw), math.sin(fr.yaw), -0.3])

    def precompute(self):
        pass

    def describe(self, name):
        return "the cured part"

    def instruction(self):
        return ("The press is running a curing cycle on the part inside. When the cycle ends, the green READY lamp on top "
                "(mark 2) lights up for a few seconds and then goes dark again; nothing else about the press changes. "
                "If the cycle has ended, unload the part (skill `unload`) no later than 90 seconds after it ended. "
                "Opening the press while the cycle is still running ruins the part, so if the cycle has not ended, leave the "
                "press closed. You have one unload. Finish with `done`.")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}
        self.place_robots()
        self.pin_home("s07_")
        self.hidden = {}
        self.v2s_setup(ep, rng)
        self.t_done = None; self.actor_on = False; self.vent = None
        self.lamp = False; self.unloaded_at = None; self.part_state = "curing"
        self.set_lamp(False)

    def set_lamp(self, on):
        self.lamp = on
        p = np.asarray(self.fr.p(_D + 0.2, 0.25, 1.50))
        self.pins.move(obj("s07_lamp_on"), p if on else p + np.array([0, 0, -6.0]))
        self.pins.move(obj("s07_lamp_off"), p + np.array([0, 0, -6.0]) if on else p)

    def g_vent(self):
        """Decoy: the door slides a few centimetres open (like the start of an unload) and closes again."""
        door = obj("s07_door"); fr = self.fr
        p0 = np.asarray(fr.p(_D - 0.015, 0.0, 0.92)); p1 = np.asarray(fr.p(_D - 0.05, -0.12, 0.92))
        for _ in range(2):
            n = int(0.7 * SIM_HZ)
            for i in range(1, n + 1):
                self.pins.move(door, p0 + (p1 - p0) * ease(i / n)); yield
            yield from g_still(1.0)
            for i in range(1, n + 1):
                self.pins.move(door, p1 + (p0 - p1) * ease(i / n)); yield
            yield from g_still(0.6)
        self.pins.move(door, p0)

    def workspace_busy(self):
        return "the press door is moving" if self.vent is not None else None

    def world(self, ep):
        super().world(ep)
        t = ep.world_t
        if not self.actor_on and self.variant != "null" and self.actor_due(ep):
            self.actor_on = True; self.actor_started(ep)
            if self.variant == "event":
                self.t_done = t; self.hidden["t_cycle_end"] = round(t, 2)
            else:
                self.vent = self.g_vent(); ep.ev("press_vent")
        if self.vent is not None:
            try:
                next(self.vent)
            except StopIteration:
                self.vent = None
        on = self.t_done is not None and self.t_done <= t < self.t_done + (1e9 if self.visible else 6.0)
        if on != self.lamp:
            self.set_lamp(on); ep.ev("lamp_on" if on else "lamp_off")
        if self.t_done is None:
            return
        if self.part_state == "curing" and t >= self.t_done:
            self.part_state = "ready"
        if self.part_state == "ready" and t > self.t_done + 90.0 and self.unloaded_at is None:
            self.part_state = "over-cured"

    def skill_unload(self, ep, slot, target):
        fr = self.fr; A = self.A; arm = self.armA
        door = obj("s07_door"); part = obj("s07_part")
        bw = self.baseW
        yield from g_base(A, [bw[:2]], final_yaw=bw[2], speed=0.4)
        yield from arm.g_to(self.q_handle, 1.2, "right", closed=False)
        yield from arm.g_to(self.q_handle, 0.3, "right", closed=True)
        # door slides sideways to the robot's right while the hand pulls it
        p0 = np.asarray(fr.p(_D - 0.015, 0.0, 0.92)); p1 = np.asarray(fr.p(_D - 0.06, -0.56, 0.92))
        n = int(1.4 * SIM_HZ); q0 = A.q.clone()
        for i in range(1, n + 1):
            u = ease(i / n); self.pins.move(door, p0 + (p1 - p0) * u)
            A.q = q0 + (self.q_open - q0) * u; A.q[A.GRIP["right"]] = 0.0; yield
        t_open = ep.world_t
        if self.unloaded_at is None:
            self.unloaded_at = t_open
            if self.part_state == "curing":
                self.part_state = "ruined (opened during the cycle)"
            elif self.part_state == "ready":
                self.part_state = "unloaded"
        yield from arm.g_to(self.q_part, 1.2, "right", closed=False)
        yield from arm.g_to(self.q_part, 0.3, "right", closed=True)
        A.attach(part, "right", fixed_quat=fr.q()); self.pins.unpin(part)
        yield from arm.g_to(self.armA.neutral(bw), 1.0, "right", closed=True)
        q_c, _ = self.armA.reach(bw, "right", fr.p(_D - 0.40, 0.55, 0.86))
        yield from arm.g_to(q_c, 1.4, "right", closed=True)
        A.detach(part); self.pins.move(part, fr.p(_D - 0.40, 0.62, 0.75), fr.q())
        yield from arm.g_to(q_c, 0.3, "right", closed=False)
        yield from arm.g_to(self.q_open, 1.2, "right", closed=False)
        yield from arm.g_to(self.q_open, 0.3, "right", closed=True)
        q0 = A.q.clone()
        for i in range(1, n + 1):
            u = ease(i / n); self.pins.move(door, p1 + (p0 - p1) * u)
            A.q = q0 + (self.q_handle - q0) * u; A.q[A.GRIP["right"]] = 0.0; yield
        yield from arm.g_to(self.armA.neutral(bw), 1.0, "right", closed=False)
        yield from g_base(A, [self.baseA[:2]], final_yaw=self.baseA[2], speed=0.4)
        ep.result = "unloaded the part onto the cart"

    def privileged(self):
        return dict(part=self.part_state, lamp_on=self.lamp, unloaded_at=None if self.unloaded_at is None else round(self.unloaded_at, 2))

    def verify(self, final):
        s = final["part"]
        if s == "unloaded":
            return True, "unloaded after the cycle ended and within 90 s"
        if s == "curing":
            return True, "the cycle did not end and the press stayed closed"
        if s == "ready":
            return False, "the cycle ended but the part was not unloaded"
        if s.startswith("ruined"):
            return False, "opened the press during the cycle"
        if s == "over-cured":
            return False, "the part stayed in the press more than 90 s after the cycle ended"
        return False, f"part not unloaded (state: {s})"
