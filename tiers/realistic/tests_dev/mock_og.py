"""Minimal OmniGibson stand-in to exercise scenario logic without a GPU (dev only)."""
import math, types
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as Rot


class MockObj:
    def __init__(self, cfg):
        self.name = cfg["name"]; self.category = cfg.get("category", "object")
        self.p = np.asarray(cfg.get("position", [0, 0, 0]), float); self.q = np.asarray(cfg.get("orientation", [0, 0, 0, 1]), float)
        sc = cfg.get("scale")
        if sc is None:
            sc = [0.1, 0.1, 0.12]
        self.ext = np.asarray(sc, float) if len(np.atleast_1d(sc)) == 3 else np.array([sc, sc, sc], float)
        self.states = {}

    def get_position_orientation(self):
        return th.tensor(self.p, dtype=th.float32), th.tensor(self.q, dtype=th.float32)

    def set_position_orientation(self, position=None, orientation=None):
        if position is not None:
            self.p = np.asarray(position.tolist() if hasattr(position, "tolist") else position, float)
        if orientation is not None:
            self.q = np.asarray(orientation.tolist() if hasattr(orientation, "tolist") else orientation, float)

    def set_linear_velocity(self, v):
        pass

    def set_angular_velocity(self, v):
        pass

    @property
    def aabb(self):
        return th.tensor(self.p - self.ext / 2), th.tensor(self.p + self.ext / 2)


class MockJoint:
    lower_limit = -2.5; upper_limit = 2.5


class MockSensor:
    def __init__(self, name="cam", w=720, h=720):
        self.name = name; self.w, self.h = w, h; self.p = np.zeros(3); self.q = np.array([0, 0, 0, 1.0])
        self.focal_length = 17.0; self.horizontal_aperture = 20.955; self.enabled = True

    def get_obs(self):
        return [{"rgb": th.zeros((self.h, self.w, 4), dtype=th.uint8)}]

    def get_position_orientation(self):
        return th.tensor(self.p), th.tensor(self.q)

    def set_position_orientation(self, position=None, orientation=None):
        self.p = np.asarray(position.tolist(), float); self.q = np.asarray(orientation.tolist(), float)

    @property
    def intrinsic_matrix(self):
        fx = self.w * self.focal_length / self.horizontal_aperture
        return th.tensor([[fx, 0, self.w / 2], [0, fx, self.h / 2], [0, 0, 1]])


class MockRobot:
    """26 dofs: 6 virtual base, trunk 6..9, arms interleaved 10..21, grippers 22..25. FK: simple planar-ish arm."""

    def __init__(self, name):
        self.name = name; self.n = 26
        self.qq = th.zeros(self.n); self.p = np.zeros(3); self.q = np.array([0, 0, 0, 1.0])
        self.joints = {f"j{i}": MockJoint() for i in range(self.n)}
        self.arm_control_idx = {"left": th.tensor([10, 12, 14, 16, 18, 20]), "right": th.tensor([11, 13, 15, 17, 19, 21])}
        self.trunk_control_idx = th.tensor([6, 7, 8, 9]); self.gripper_control_idx = {"left": th.tensor([22, 23]), "right": th.tensor([24, 25])}
        self.sensors = {f"{name}:eyes:Camera:0": MockSensor(f"{name}_eyes"), f"{name}:left_eef_link:Camera:0": MockSensor()}

    def get_joint_positions(self):
        return self.qq.clone()

    def set_joint_positions(self, q, indices=None, drive=False):
        if indices is None:
            self.qq = q.clone()
        else:
            self.qq[indices] = q

    def set_joint_velocities(self, v):
        pass

    def get_position_orientation(self):
        return th.tensor(self.p), th.tensor(self.q)

    def set_position_orientation(self, position=None, orientation=None):
        self.p = np.asarray(position.tolist(), float); self.q = np.asarray(orientation.tolist(), float)

    def get_eef_position(self, arm):
        yaw = Rot.from_quat(self.q).as_euler("xyz")[2]
        ids = self.arm_control_idx[arm].tolist(); t = self.trunk_control_idx.tolist()
        a = self.qq[ids].numpy(); tr = self.qq[t].numpy()
        side = 0.22 if arm == "left" else -0.22
        reach = 0.35 + 0.45 * math.tanh(a[0]) + 0.2 * math.tanh(tr[0])
        lat = side + 0.4 * math.tanh(a[1])
        z = 1.0 + 0.5 * math.tanh(a[2]) + 0.2 * math.tanh(tr[1])
        f = np.array([math.cos(yaw), math.sin(yaw)]); l = np.array([-math.sin(yaw), math.cos(yaw)])
        xy = self.p[:2] + reach * f + lat * l
        return th.tensor([xy[0], xy[1], z], dtype=th.float32)

    def get_eef_orientation(self, arm):
        ids = self.arm_control_idx[arm].tolist(); a = self.qq[ids].numpy()
        return th.tensor(Rot.from_euler("xyz", [a[3], a[4], a[5]]).as_quat(), dtype=th.float32)


def make_env(objects, robot_names=("robot", "helper")):
    objs = {o["name"]: MockObj(o) for o in objects}
    robots = [MockRobot(n) for n in robot_names]

    class Reg:
        def __call__(self, key, name):
            return objs[name]
    scene = types.SimpleNamespace(object_registry=Reg(), objects=list(objs.values()))
    env = types.SimpleNamespace(scene=scene, robots=robots)
    sim = types.SimpleNamespace(step=lambda: None, render=lambda: None, _render_on_step=True,
                                viewer_camera=MockSensor("viewer", 1280, 720), dump_state=lambda serialized=False: {},
                                load_state=lambda s, serialized=False: None, stop=lambda: None, play=lambda: None)
    og = types.SimpleNamespace(sim=sim)
    return og, env, objs
