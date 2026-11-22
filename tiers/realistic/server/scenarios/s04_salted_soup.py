"""S04 Salted soup (home kitchen, multi-robot, invisible process state). v2s strict version.

A pot of soup stands on the stove; the salt shaker and the pepper mill stand on the counter next to it. The soup needs
exactly one portion of salt and nobody has salted it when the episode starts. The helper robot cooking beside the agent
(never named to the agent) becomes due at world U(0.3, 2.5) s and starts only while the agent's robot stands still.
Variant = seed % 3:
  event : it salts the soup (shakes the salt shaker over the pot);
  decoy : it peppers the soup (same gesture with the pepper mill; salt stays at 0);
  null  : it comes to the stove, stands for a moment and leaves.
Salt and pepper are invisible once in the soup, so at action time the pot looks the same in every variant.
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["s04"]          # surface frame at the counter front edge; h counter top; pot [df, dl], pot_z; A, B, cam
_FR = Frame(*_A["surface"])
_H = _A["h"]; _D = 0.0
_POT = tuple(_A["pot"]); _SALT = tuple(_A.get("salt", (0.22, 0.62))); _PEPPER = tuple(_A.get("pepper", (0.22, 0.78)))
_M = SCENE_SPEC["models"]
_objs = [
    {"type": "DatasetObject", "name": "s04_pot", "category": "stockpot", "model": _M["stockpot"], "kinematic_only": True, "visual_only": True, "position": park(10)},
    {"type": "DatasetObject", "name": "s04_salt", "category": "salt_shaker", "model": _M["salt_shaker"], "kinematic_only": True, "visual_only": True, "position": park(11)},
    {"type": "DatasetObject", "name": "s04_pepper", "category": "pepper_grinder", "model": _M["pepper_grinder"], "kinematic_only": True, "visual_only": True, "position": park(12)},
]
_objs += [sphere(f"s04_grain{k}", _FR, 0, 0, -4.0 - 0.1 * k, 0.006, [0.97, 0.97, 0.97, 1]) for k in range(6)]
_objs += [sphere(f"s04_pgrain{k}", _FR, 0, 0, -5.0 - 0.1 * k, 0.006, [0.15, 0.12, 0.1, 1]) for k in range(6)]


@register
class S04SaltedSoup(StationScenario):
    sid = "s04_salted_soup"
    title = "Salted soup"
    setting = "home kitchen"
    ANCHOR_KEY = "s04"
    OBJECTS = _objs
    budget = 1
    skills = ("salt", "wait", "done")
    extra_help = {"salt": "salt <mark>: shake one portion of salt from the salt shaker into the pot at the mark"}
    A_WORLD = _A["A"]
    B_WORLD = _A["B"]
    B_START = (0, 0, 0)
    hand_z = _H + 0.12
    CAM = staticmethod(lambda fr: tuple(_A["cam"]))

    def build_scenario(self):
        fr = self.fr
        for n in ["s04_pot", "s04_salt", "s04_pepper"]:
            lo, hi = [npf(v) for v in obj(n).aabb]
            self.hz[n] = float(hi[2] - lo[2]) / 2 + 0.002
        self.pot_z = _A.get("pot_z", _H)
        self.pot_top = self.pot_z + 2 * self.hz["s04_pot"]
        self.slots["pot"] = dict(pos=fr.p(*_POT, self.pot_z), label="the soup pot on the stove", reach=False,
                                 mark_pos=fr.p(*_POT, self.pot_top))
        self.slots["salt"] = dict(pos=fr.p(*_SALT, _H), label="salt shaker", reach=False, mark_pos=fr.p(*_SALT, _H + 2 * self.hz["s04_salt"]))
        self.slots["pepper"] = dict(pos=fr.p(*_PEPPER, _H), label="pepper mill", reach=False, mark_pos=fr.p(*_PEPPER, _H + 2 * self.hz["s04_pepper"]))
        self.baseB = fr.base(-0.40, (_POT[1] + _SALT[1]) / 2 - 0.05, 0.0)
        self.baseSalt = self.baseB
        over = np.asarray(fr.p(*_POT, self.pot_top + 0.16))
        self.shake_pts = [over, over + np.array([0, 0, 0.05]), over, over + np.array([0, 0, 0.05]), over]
        for n, (df, dl) in [("salt", _SALT), ("pepper", _PEPPER)]:
            self.armA.reach(self.baseSalt, "left", fr.p(df, dl, _H + self.hz[f"s04_{n}"] * 1.4), dz=0.0)

    def precompute(self):
        pass

    def describe(self, name):
        return "the salt shaker"

    def __init__(self):
        super().__init__()
        self.fr = Frame(*_A["surface"])

    def instruction(self):
        return ("The soup on the stove (mark 1) needs exactly ONE portion of salt. When you start, nobody has salted it yet. "
                "Use the skill `salt 1` to add one portion. Too much salt ruins it. "
                "When the soup has exactly one portion of salt, run `done`.")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}
        self.place_robots()
        fr = self.fr
        self.pins.move(obj("s04_pot"), np.asarray(fr.p(*_POT, self.pot_z + self.hz["s04_pot"])), fr.q())
        self.put("s04_salt", "salt"); self.put("s04_pepper", "pepper")
        for k in range(6):
            self.pins.pin(obj(f"s04_grain{k}")); self.pins.pin(obj(f"s04_pgrain{k}"))
        self.salt = 0; self.pepper = 0
        self.hidden = {}
        self.v2s_setup(ep, rng)
        self.helper_salts = self.variant == "event"
        self.hidden.update(helper_adds={"event": "salt", "decoy": "pepper", "null": "nothing"}[self.variant])
        self.helper = Helper(); self.agent_salted_before_helper = False

    def g_season(self, bot, arm_, base, which, arm, ep):
        """Pick the container, shake it over the pot (grains fall), put it back."""
        fr = self.fr; name = f"s04_{which}"; o = obj(name)
        src = np.asarray(self.slots[which]["pos"]) + np.array([0, 0, self.hz[name] * 1.4])
        yield from arm_.g_pick(base, arm, src, obj=o, on_grasp=lambda: self.pins.unpin(o), keep_quat=None, lift=0.12)
        q_prev = None
        grains = [f"s04_grain{k}" if which == "salt" else f"s04_pgrain{k}" for k in range(6)]
        for i, p in enumerate(self.shake_pts):
            q, _ = arm_.reach(base, arm, p, q_init=q_prev); q_prev = q
            yield from arm_.g_to(q, 0.45, arm, closed=True)
            if i in (1, 3):
                top = npf(bot.r.get_eef_position(arm))
                for k, g in enumerate(grains):
                    self.pins.move(obj(g), top + np.array([0.01 * (k % 3 - 1), 0.01 * (k // 3), -0.08 - 0.03 * k]))
        for g in grains:
            self.pins.move(obj(g), [0, 0, -6.0])
        if which == "salt":
            self.salt += 1
        else:
            self.pepper += 1
        yield from arm_.g_place(base, arm, src, obj=o, quat=fr.q(), pins=self.pins, lift=0.12, obj_offset=-self.hz[name] * 0.4)
        ep.ev("seasoned", who="agent" if bot is self.A else "helper", what=which)

    def helper_plan(self, ep):
        B, aB = self.B, self.armB
        yield from g_base(B, [self.baseB[:2]], final_yaw=self.baseB[2], speed=0.5)
        if self.variant == "null":
            yield from g_still(2.0)
            self.helper.status = "looked at the stove: leaving"
            yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5)
            return
        if self.salt > 0 and self.helper_salts:
            self.helper.status = "soup already salted: helper adds pepper instead"
            which = "pepper"
        else:
            which = "salt" if self.helper_salts else "pepper"
        yield from self.g_season(B, aB, self.baseB, which, "left", ep)
        self.helper.status = f"added {which}: leaving"
        yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5)

    def world(self, ep):
        super().world(ep)
        if not self.helper.started and self.actor_due(ep):
            self.helper.start(self.helper_plan(ep), "helper seasoning"); self.actor_started(ep)
        self.helper.step()

    def skill_salt(self, ep, slot, target):
        if slot != "pot":
            ep.result = f"mark {target} is not the pot"; return
        yield from g_base(self.A, [self.baseSalt[:2]], final_yaw=self.baseSalt[2], speed=0.4)
        yield from self.g_season(self.A, self.armA, self.baseSalt, "salt", "left", ep)
        yield from g_base(self.A, [self.baseA[:2]], final_yaw=self.baseA[2], speed=0.4)
        ep.result = "added one portion of salt to the soup"

    def privileged(self):
        return dict(salt_portions=self.salt, pepper_portions=self.pepper, helper=self.helper.status if hasattr(self, "helper") else "")

    def verify(self, final):
        s = final["salt_portions"]
        if s == 1:
            return True, "the soup has exactly one portion of salt"
        if s == 0:
            return False, "the soup was never salted"
        return False, f"over-salted: {s} portions"
