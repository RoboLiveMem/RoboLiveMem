"""S05 Defect reject on an indexing table (factory cell, machine, identity under machine motion). v2s strict version.

Six identical parts sit in the nests of an indexing table. At the first observation the vision station's lamp is red
over the part at the inspection nest (that part is defective). The task: put the defective part into the red reject bin
and nothing else. The table (never named as an actor to the agent) becomes due at world U(0.3, 2.5) s; every 60-degree
step starts only while the agent's robot stands still (one step every 4 s). Variant = seed % 3:
  event : 1 to 5 steps (the defective part ends in another nest; the lamp then shows green for a good part);
  decoy : 6 steps, a full revolution (the defective part ends where it started, under the red lamp);
  null  : the table stays still.
At action time all six parts look the same, so the robot has to count the steps it saw in its own camera stream.
"""
import math
import numpy as np

_FR = Frame(*SCENE_SPEC["anchors"]["s05"])
_C = (1.08, 0.0)          # table centre (forward, left) in the station frame (robot stands back to see the whole table)
_R = 0.26                 # nest radius
_ZT = 0.80                # disc top
_NEST_ANG = [math.pi + k * math.pi / 3 for k in range(6)]   # nest 0 faces the robot... inspection is at angle 0 (far side)
_INSPECT = 3              # nest index under the inspection lamp at angle 0 (far side)


def _nest(k, theta=0.0):
    a = _NEST_ANG[k] + theta
    return _C[0] + _R * math.cos(a), _C[1] + _R * math.sin(a)


_objs = []
_objs += [cyl("s05_pedestal", _FR, _C[0], _C[1], (_ZT - 0.05) / 2, 0.14, _ZT - 0.05, DARK)]
_objs += [cyl("s05_disc", _FR, _C[0], _C[1], _ZT - 0.025, 0.40, 0.05, STEEL)]
_objs += [cyl(f"s05_nest{k}", _FR, *_nest(k), _ZT + 0.003, 0.06, 0.008, DARK, visual=True) for k in range(6)]
_objs += [cyl(f"s05_part{k}", _FR, *_nest(k), _ZT + 0.04, 0.042, 0.07, [0.62, 0.64, 0.68, 1]) for k in range(6)]
# vision station behind the table: post, arm, camera, lamp
_objs += [box("s05_post", _FR, _C[0] + 0.55, 0.0, 0.70, 0.08, 0.08, 1.40, YELLOW)]
_objs += [box("s05_arm", _FR, _C[0] + 0.40, 0.0, 1.36, 0.36, 0.06, 0.06, YELLOW)]
_objs += [box("s05_camera", _FR, _C[0] + _R + 0.0, 0.0, 1.28, 0.10, 0.10, 0.10, DARK)]
_objs += [sphere("s05_lamp_red", _FR, _C[0] + _R, 0.0, 1.19, 0.035, RED)]
_objs += [sphere("s05_lamp_green", _FR, _C[0] + _R, 0.0, -3.0, 0.035, GREEN)]
_objs += [sphere("s05_lamp_off", _FR, _C[0] + _R, 0.0, -3.0, 0.035, OFF)]
# reject bin on a stand to the robot's right, good-parts tray to its left
_objs += table("s05_rstand", _FR, 0.80, -0.70, 0.36, 0.36, 0.62, rgba=DARK)
_objs += open_bin("s05_rbin", _FR, 0.80, -0.70, 0.62, 0.30, 0.30, 0.16, RED)
_objs += table("s05_fence", _FR, _C[0], 0.75, 1.2, 0.05, 1.1, rgba=YELLOW)


@register
class S05RotaryReject(StationScenario):
    sid = "s05_rotary_reject"
    title = "Reject the defective part"
    setting = "factory work cell"
    ANCHOR_KEY = "s05"
    OBJECTS = _objs
    budget = 2
    skills = ("pick", "place", "wait", "done")
    hand_z = 1.0
    CAM = staticmethod(lambda fr: (fr.p(-0.7, -1.8, 2.1), fr.p(0.95, 0.0, 0.8), 15.0))

    def build_scenario(self):
        fr = self.fr
        for k in range(6):
            df, dl = _nest(k)
            self.slots[f"nest{k}"] = dict(pos=fr.p(df, dl, _ZT), label=f"part nest {k + 1}", grasp_z=0.05,
                                          base=fr.base(df - 0.58, dl - 0.12), arm="right")
        self.slots["reject"] = dict(pos=fr.p(0.80, -0.70, 0.66), label="red reject bin", grasp_z=0.08, arm="right", stack=True,
                                    jitter=[0.02, -0.02, 0], base=fr.base(0.80 - 0.55, -0.70 + 0.10))
        for k in range(6):
            self.hz[f"s05_part{k}"] = 0.035
        # order the marks: nests 1..6 then the bin (dict order)

    def describe(self, name):
        return "a metal part"

    def instruction(self):
        return ("A vision station inspects the part in the nest under its lamp. When you start, its lamp is RED: the part "
                "under the red lamp is defective. Put that defective part into the red reject bin (mark 7). Do not reject "
                "good parts.")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}
        self.place_robots()
        self.theta = 0.0; self.steps_done = 0
        self.part_nest = {f"s05_part{k}": k for k in range(6)}
        self.defective = f"s05_part{_INSPECT}"
        self.hidden = {}
        self.v2s_setup(ep, rng)
        n_steps = {"event": rng.randint(1, 5), "decoy": 6, "null": 0}[self.variant]
        t1 = self.t_ev
        self.schedule = [t1 + 4.0 * i for i in range(n_steps)]
        self.hidden.update(defective=self.defective, n_steps=n_steps, step_times=[round(x, 2) for x in self.schedule])
        self.rot = None; self.lamp = "red"; self.rejected = []
        fr = self.fr
        self.pin_home("s05_")
        for k in range(6):
            self.put(f"s05_part{k}", f"nest{k}")
        self.set_lamp("red")

    def set_lamp(self, which):
        self.lamp = which
        p_on = np.asarray(self.fr.p(_C[0] + _R, 0.0, 1.19))
        for c in ("red", "green", "off"):
            self.pins.move(obj(f"s05_lamp_{c}"), p_on if c == which else p_on + np.array([0, 0, -6.0]))

    def workspace_busy(self):
        # the cell is interlocked for its whole indexing sequence (the table may pause between steps)
        running = self.rot is not None or (self.steps_done > 0 and bool(self.schedule))
        return "the indexing table is running its cycle" if running else None

    def world(self, ep):
        super().world(ep)
        t = ep.world_t
        if self.rot is None and self.schedule and t >= self.schedule[0] and not self.A_busy:
            self.schedule.pop(0)
            self.rot = dict(t0=t, th0=self.theta); self.set_lamp("off")
            if self.steps_done == 0:
                self.actor_started(ep)
            ep.ev("index_start")
        if self.rot is not None:
            u = min(1.0, (t - self.rot["t0"]) / 2.4)
            self.theta = self.rot["th0"] + (math.pi / 3) * ease(u)
            self.pose_table()
            if u >= 1.0:
                self.rot = None; self.steps_done += 1
                # relabel nests: each part advanced by one nest
                new = {}
                for s, o in list(self.occ.items()):
                    if s.startswith("nest"):
                        k = int(s[4:]); new[f"nest{(k + 1) % 6}"] = o
                for k in range(6):
                    self.occ[f"nest{k}"] = new.get(f"nest{k}")
                self.theta = 0.0; self.pose_table()
                at = self.occ.get(f"nest{_INSPECT}")
                self.set_lamp("red" if at == self.defective else ("green" if at else "off"))
                ep.ev("index_done", steps=self.steps_done)

    def pose_table(self):
        fr = self.fr
        for k in range(6):
            o = self.occ.get(f"nest{k}")
            if o is None:
                continue
            df, dl = _nest(k, self.theta)
            self.pins.move(obj(o), fr.p(df, dl, _ZT + 0.035), fr.q(self.theta))
        for k in range(6):
            df, dl = _nest(k, self.theta)
            self.pins.move(obj(f"s05_nest{k}"), fr.p(df, dl, _ZT + 0.003), fr.q(self.theta))

    def on_agent_place(self, ep, name, slot):
        if slot == "reject":
            self.rejected.append(name)

    def privileged(self):
        return dict(defective_at=self.where(self.defective), rejected=list(self.rejected), lamp=self.lamp,
                    steps_done=self.steps_done, occ={k: v for k, v in self.occ.items() if k.startswith("nest")})

    def verify(self, final):
        rej = final["rejected"]
        if self.defective in rej and len(rej) == 1:
            return True, "the defective part is in the reject bin and no good part was rejected"
        if self.defective in rej:
            return False, "defective part rejected together with good parts"
        if rej:
            return False, "a good part was rejected; the defective part is still on the table"
        return False, "nothing was rejected"
