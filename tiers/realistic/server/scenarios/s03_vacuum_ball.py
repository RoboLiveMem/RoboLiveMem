"""S03 Ball and robot vacuum (home living room, machine, location out of view). v2s strict version.

A soccer ball lies on the rug. The task: put the ball into the toy box. The robot vacuum (never named to the agent)
becomes due at world U(0.3, 2.5) s and starts only while the agent's robot stands still. Variant = seed % 3:
  event : its run crosses the ball and pushes it under sofa k, out of the robot's view;
  decoy : it drives up to the ball (same approach as the event), turns away without touching it and docks again;
  null  : it stays in its dock.
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["s03"]
_FR = Frame(*_A["base"])
_M = SCENE_SPEC["models"]
_objs = [
    {"type": "DatasetObject", "name": "s03_ball", "category": "soccer_ball", "model": _M["soccer_ball"], "kinematic_only": True, "visual_only": True, "position": park(30)},
    {"type": "DatasetObject", "name": "s03_box", "category": "toy_box", "model": _M["toy_box"], "kinematic_only": True, "visual_only": True, "position": park(31)},
    {"type": "PrimitiveObject", "name": "s03_vac", "primitive_type": "Cylinder", "scale": [0.34, 0.34, 0.08], "rgba": [0.12, 0.12, 0.13, 1],
     "kinematic_only": True, "visual_only": True, "position": park(32)},
    {"type": "PrimitiveObject", "name": "s03_vac_top", "primitive_type": "Cylinder", "scale": [0.23, 0.23, 0.012], "rgba": [0.32, 0.33, 0.36, 1],
     "kinematic_only": True, "visual_only": True, "position": park(33)},
    {"type": "PrimitiveObject", "name": "s03_vac_led", "primitive_type": "Sphere", "scale": [0.03, 0.03, 0.03], "rgba": [0.2, 0.9, 1.0, 1],
     "kinematic_only": True, "visual_only": True, "position": park(34)},
]


@register
class S03VacuumBall(StationScenario):
    sid = "s03_vacuum_ball"
    title = "Ball to the toy box"
    setting = "home living room"
    ANCHOR_KEY = "s03"
    OBJECTS = _objs
    budget = 2
    skills = ("pick", "place", "wait", "done")
    hand_z = 0.9
    CAM = staticmethod(lambda fr: tuple(_A["cam"]))

    def build_scenario(self):
        lo, hi = [npf(v) for v in obj("s03_ball").aabb]
        self.r_ball = float(hi[2] - lo[2]) / 2
        self.hz["s03_ball"] = self.r_ball
        bx, by = _A["ball"]
        self.slots["rug"] = dict(pos=[bx, by, 0.0], label="the ball's spot on the rug", base=tuple(_A["ball_base"]), arm="right",
                                 grasp_z=self.r_ball, approach=[0, 0, -1])
        for i, h in enumerate(_A["hides"]):
            self.slots[f"hide{i}"] = dict(pos=[h["pos"][0], h["pos"][1], 0.0], label=h["label"], base=tuple(h["base"]), arm="right",
                                          grasp_z=self.r_ball, approach=h.get("approach", [0, 0, -1]),
                                          mark_pos=h.get("mark", [h["pos"][0], h["pos"][1], 0.15]))
        b = _A["box"]
        lo, hi = [npf(v) for v in obj("s03_box").aabb]
        self.box_h = float(hi[2] - lo[2])
        self.slots["box"] = dict(pos=[b[0], b[1], self.box_h - 0.12], label="toy box", base=tuple(_A["box_base"]), arm="right",
                                 grasp_z=0.10, stack=True, mark_pos=[b[0], b[1], self.box_h + 0.05])

    def describe(self, name):
        return "a ball"

    def instruction(self):
        nb = 2 + len(_A["hides"])
        return (f"Put the soccer ball into the toy box (mark {nb}). Right now the ball lies on the rug (mark 1). "
                f"Marks 2 to {nb - 1} are places under the furniture that you can reach into.")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}
        self.place_robots()
        b = _A["box"]
        self.pins.move(obj("s03_box"), [b[0], b[1], self.box_h / 2 + 0.002], [0, 0, math.sin(b[2] / 2), math.cos(b[2] / 2)] if len(b) > 2 else None)
        self.put("s03_ball", "rug")
        self.hidden = {}
        self.v2s_setup(ep, rng)
        self.target = rng.randrange(len(_A["hides"]))
        self.hidden.update(hide=self.target, hide_label=_A["hides"][self.target]["label"])
        self.vac = dict(x=_A["vac"][0], y=_A["vac"][1], yaw=_A["vac"][2], state="docked")
        self.pose_vac()
        self.helper = Helper(); self.ball_spin = Rot.identity()

    def pose_vac(self):
        v = self.vac; q = [0, 0, math.sin(v["yaw"] / 2), math.cos(v["yaw"] / 2)]
        self.pins.move(obj("s03_vac"), [v["x"], v["y"], 0.045], q)
        self.pins.move(obj("s03_vac_top"), [v["x"], v["y"], 0.092], q)
        self.pins.move(obj("s03_vac_led"), [v["x"] + 0.12 * math.cos(v["yaw"]), v["y"] + 0.12 * math.sin(v["yaw"]), 0.095], q)

    def g_drive(self, pts, speed=0.30, push=False):
        v = self.vac
        for p in pts:
            dx, dy = p[0] - v["x"], p[1] - v["y"]; dist = math.hypot(dx, dy)
            if dist < 1e-3:
                continue
            yaw1 = math.atan2(dy, dx); d = (yaw1 - v["yaw"] + math.pi) % (2 * math.pi) - math.pi
            n = max(1, int(abs(d) / 1.2 * SIM_HZ)); y0 = v["yaw"]
            for i in range(1, n + 1):
                v["yaw"] = y0 + d * i / n; self.pose_vac(); yield
            n = max(1, int(dist / speed * SIM_HZ)); x0, y0_ = v["x"], v["y"]
            for i in range(1, n + 1):
                u = i / n; v["x"] = x0 + dx * u; v["y"] = y0_ + dy * u; self.pose_vac()
                if push:
                    self.push_ball()
                yield

    def push_ball(self):
        """Keep the ball just in front of the vacuum, rolling."""
        v = self.vac; o = obj("s03_ball")
        f = np.array([math.cos(v["yaw"]), math.sin(v["yaw"])])
        p_new = np.array([v["x"], v["y"]]) + f * (0.17 + self.r_ball + 0.01)
        p_old = self.pins.pos(o)[:2]
        step = p_new - p_old
        if np.linalg.norm(step) > 1e-6:
            axis = np.array([-step[1], step[0], 0.0]) / np.linalg.norm(step)
            self.ball_spin = Rot.from_rotvec(axis * np.linalg.norm(step) / self.r_ball) * self.ball_spin
        self.pins.move(o, [p_new[0], p_new[1], self.r_ball], self.ball_spin.as_quat())

    def vac_plan(self, ep):
        h = _A["hides"][self.target]
        bx, by = _A["ball"]
        hx, hy = h["pos"]
        d = np.array([hx - bx, hy - by]); d /= np.linalg.norm(d)
        behind = np.array([bx, by]) - d * (0.17 + self.r_ball + 0.35)
        contact = np.array([bx, by]) - d * (0.17 + self.r_ball + 0.01)
        via = [np.asarray(v, float) for v in _A.get("vac_via", {}).get(str(self.target), [])]
        yield from self.g_drive(via + [behind])
        if self.variant == "decoy":
            # same approach as the event, but it turns away without touching the ball and goes back to its dock
            yield from g_still(1.0)
            ep.ev("vacuum_decoy", near=h["label"])
            yield from self.g_drive(list(reversed(via)) + [np.asarray(_A["vac"][:2], float)], speed=0.3)
            return
        if self.occ.get("rug") == "s03_ball" and not self.A_busy:
            yield from self.g_drive([contact], speed=0.25)
            self.occ["rug"] = None
            end = np.array([hx, hy]) - d * (0.17 + self.r_ball + 0.01)
            yield from self.g_drive([end], speed=0.30, push=True)
            self.pins.pin(obj("s03_ball"))
            self.occ[f"hide{self.target}"] = "s03_ball"
            ep.ev("ball_pushed", where=h["label"])
            back = end - d * 0.5
            yield from self.g_drive([back], speed=0.3)
        for p in _A.get("vac_after", []):
            yield from self.g_drive([p], speed=0.3)

    def world(self, ep):
        super().world(ep)
        if not self.helper.started and self.variant != "null" and self.actor_due(ep):
            self.helper.start(self.vac_plan(ep), "vacuum cleaning"); self.actor_started(ep)
        self.helper.step()

    def privileged(self):
        return dict(ball_at=self.where("s03_ball") or ("box" if "s03_ball" in self.occ.get("box_items", []) else None))

    def verify(self, final):
        if final["ball_at"] == "box":
            return True, "the ball is in the toy box"
        return False, f"the ball is not in the toy box (at {final['ball_at']})"
