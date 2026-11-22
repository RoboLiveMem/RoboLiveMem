"""Primitive-built props (machines, totes, lockers) in a station frame.

A station frame is (x0, y0, yaw): the agent robot stands at (x0, y0) facing yaw. Points are given as
(forward, left, up) offsets. Every prop part is a kinematic PrimitiveObject so that scenario logic can
pose it every tick.
"""
import math
import numpy as np

GRAY = [0.55, 0.57, 0.6, 1]; DARK = [0.18, 0.19, 0.21, 1]; STEEL = [0.72, 0.74, 0.77, 1]; YELLOW = [0.93, 0.74, 0.12, 1]
BLUE = [0.16, 0.36, 0.66, 1]; ORANGE = [0.86, 0.45, 0.13, 1]; RED = [0.75, 0.12, 0.1, 1]; GREEN = [0.15, 0.62, 0.25, 1]
WOOD = [0.62, 0.46, 0.3, 1]; CARD = [0.72, 0.56, 0.36, 1]; WHITE = [0.92, 0.92, 0.9, 1]; OFF = [0.25, 0.25, 0.25, 1]


class Frame:
    def __init__(self, x0, y0, yaw):
        self.x0, self.y0, self.yaw = x0, y0, yaw
        self.f = np.array([math.cos(yaw), math.sin(yaw)]); self.l = np.array([-math.sin(yaw), math.cos(yaw)])

    def p(self, df, dl, z):
        xy = np.array([self.x0, self.y0]) + df * self.f + dl * self.l
        return [float(xy[0]), float(xy[1]), float(z)]

    def q(self, dyaw=0.0):
        a = self.yaw + dyaw
        return [0.0, 0.0, math.sin(a / 2), math.cos(a / 2)]

    def base(self, df=0.0, dl=0.0, dyaw=0.0):
        xy = np.array([self.x0, self.y0]) + df * self.f + dl * self.l
        return (float(xy[0]), float(xy[1]), self.yaw + dyaw)


def box(name, fr, df, dl, z, sf, sl, sz, rgba, dyaw=0.0, visual=True):
    """Axis-aligned (in the station frame) box centred at (df, dl, z) with sizes along forward/left/up. Props are
    visual-only by default: everything is posed kinematically, so collisions would only add depenetration jitter."""
    return {"type": "PrimitiveObject", "name": name, "primitive_type": "Cube", "scale": [sf, sl, sz], "rgba": rgba,
            "kinematic_only": True, "visual_only": visual, "position": fr.p(df, dl, z), "orientation": fr.q(dyaw)}


def cyl(name, fr, df, dl, z, r, h, rgba, visual=True):
    return {"type": "PrimitiveObject", "name": name, "primitive_type": "Cylinder", "scale": [2 * r, 2 * r, h], "rgba": rgba,
            "kinematic_only": True, "visual_only": visual, "position": fr.p(df, dl, z), "orientation": [0, 0, 0, 1]}


def sphere(name, fr, df, dl, z, r, rgba, visual=True):
    return {"type": "PrimitiveObject", "name": name, "primitive_type": "Sphere", "scale": [2 * r, 2 * r, 2 * r], "rgba": rgba,
            "kinematic_only": True, "visual_only": visual, "position": fr.p(df, dl, z), "orientation": [0, 0, 0, 1]}


def table(prefix, fr, df, dl, sf, sl, h, rgba=STEEL, leg=0.05, top=0.04, leg_rgba=DARK):
    parts = [box(f"{prefix}_top", fr, df, dl, h - top / 2, sf, sl, top, rgba)]
    for i, (a, b) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        parts.append(box(f"{prefix}_leg{i}", fr, df + a * (sf / 2 - leg), dl + b * (sl / 2 - leg), (h - top) / 2, leg, leg, h - top, leg_rgba, visual=True))
    return parts


def open_bin(prefix, fr, df, dl, z0, sf, sl, sz, rgba, wall=0.012):
    """Open-top bin: bottom + 4 walls (bottom at z0)."""
    return [box(f"{prefix}_b", fr, df, dl, z0 + wall / 2, sf, sl, wall, rgba),
            box(f"{prefix}_w0", fr, df + sf / 2 - wall / 2, dl, z0 + sz / 2, wall, sl, sz, rgba),
            box(f"{prefix}_w1", fr, df - sf / 2 + wall / 2, dl, z0 + sz / 2, wall, sl, sz, rgba),
            box(f"{prefix}_w2", fr, df, dl + sl / 2 - wall / 2, z0 + sz / 2, sf, wall, sz, rgba),
            box(f"{prefix}_w3", fr, df, dl - sl / 2 + wall / 2, z0 + sz / 2, sf, wall, sz, rgba)]


def park(i):
    """Out-of-the-way parking pose for objects not used in the current episode (under the floor)."""
    return [100.0 + 0.5 * (i % 40), 100.0 + 0.5 * (i // 40), -5.0]
