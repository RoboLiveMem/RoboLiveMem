"""S09 Kitting (warehouse, multi-robot, task progress of another robot). v2s strict version.

Four lidded kitting totes stand on a flow-rack shelf between the agent robot and a partner robot. Every tote must end
with exactly one part; all are empty at the start. The partner robot (never named to the agent) becomes due at world
U(0.3, 2.5) s and starts only while the agent's robot stands still (if the agent has already kitted, it leaves).
Variant = seed % 3:
  event : it kits a random subset of 1 to 3 totes (slides the lid open, drops a part in, closes the lid);
  decoy : it slides open and closes the lids of a random subset of 1 to 3 totes (inspection), dropping nothing;
  null  : it comes to the rack, stops for a moment and leaves.
Closed totes look the same whether they hold a part or not.
"""
import math
import numpy as np

_FR = Frame(*SCENE_SPEC["anchors"]["s09"])
_D = 0.72                 # tote centre distance from the agent robot
_ZS = 0.92                # shelf top
_DL = [0.45, 0.15, -0.15, -0.45]
_TW, _TD, _TH = 0.24, 0.30, 0.16
_objs = []
_objs += table("s09_rack", _FR, _D, 0.0, 0.46, 1.30, _ZS, rgba=[0.85, 0.55, 0.15, 1])
_objs += [box("s09_rail_l", _FR, _D, 0.66, 1.25, 0.46, 0.04, 0.66, [0.85, 0.55, 0.15, 1], visual=True),
          box("s09_rail_r", _FR, _D, -0.66, 1.25, 0.46, 0.04, 0.66, [0.85, 0.55, 0.15, 1], visual=True)]
for i, dl in enumerate(_DL):
    _objs += [box(f"s09_tote{i}", _FR, _D, dl, _ZS + _TH / 2, _TD, _TW, _TH, BLUE)]
    _objs += [box(f"s09_lid{i}", _FR, _D, dl, _ZS + _TH + 0.01, _TD + 0.01, _TW + 0.01, 0.02, [0.12, 0.27, 0.5, 1])]
# feeder bin for the agent (left) and a part tray for the partner
_objs += table("s09_feedstand", _FR, 0.28, 0.72, 0.36, 0.36, 0.70, rgba=DARK)
_objs += open_bin("s09_feeder", _FR, 0.28, 0.72, 0.70, 0.30, 0.30, 0.10, GRAY)
_objs += [box(f"s09_part{k}", _FR, 0.28 + 0.07 * ((k % 3) - 1), 0.72 + 0.07 * (k // 3 - 1), 0.735, 0.05, 0.05, 0.05,
              [0.9, 0.78, 0.2, 1]) for k in range(9)]
_objs += [box(f"s09_ppart{k}", _FR, 0.0, 3.0 + 0.1 * k, -3.0, 0.05, 0.05, 0.05, [0.9, 0.78, 0.2, 1]) for k in range(3)]


@register
class S09KittingPartner(StationScenario):
    sid = "s09_kitting_partner"
    title = "Kitting"
    setting = "warehouse kitting rack"
    ANCHOR_KEY = "s09"
    OBJECTS = _objs
    budget = 4
    skills = ("kit", "wait", "done")
    extra_help = {"kit": "kit <mark>: take a part from your feeder bin and drop it into the tote at the mark (opens and closes the lid)"}
    B_START = (1.60, 2.2, math.pi)
    hand_z = 1.0
    CAM = staticmethod(lambda fr: (fr.p(-0.6, 1.9, 2.0), fr.p(0.85, -0.1, 0.95), 15.0))

    def build_scenario(self):
        fr = self.fr
        for i, dl in enumerate(_DL):
            self.slots[f"tote{i}"] = dict(pos=fr.p(_D, dl, _ZS + _TH), label=f"kitting tote {i + 1}", reach=False)
        self.baseB = fr.base(1.42, 0.0, math.pi)
        # agent: right hand slides the lid, left hand drops the part
        self.qA_lid = {}; self.qA_lid2 = {}; self.qA_drop = {}
        for i, dl in enumerate(_DL):
            arm = "right" if dl < 0.3 else "left"
            self.qA_lid[i], _ = self.armA.reach(self.baseA, "right", fr.p(_D - 0.05, dl - 0.05, _ZS + _TH + 0.03))
            self.qA_lid2[i], _ = self.armA.reach(self.baseA, "right", fr.p(_D - 0.05, dl - 0.25, _ZS + _TH + 0.03), q_init=self.qA_lid[i])
            self.qA_drop[i], _ = self.armA.reach(self.baseA, "left", fr.p(_D, dl + 0.0, _ZS + _TH + 0.12))
        self.qA_feed, _ = self.armA.reach(self.baseA, "left", fr.p(0.28, 0.72, 0.76))
        self.qA_feed_up, _ = self.armA.reach(self.baseA, "left", fr.p(0.28, 0.72, 0.92))
        # partner: right hand slides the lid toward its right (agent's left), left hand drops a part
        self.qB_lid = {}; self.qB_lid2 = {}; self.qB_drop = {}
        for i, dl in enumerate(_DL):
            self.qB_lid[i], _ = self.armB.reach(self.baseB, "left", fr.p(_D + 0.05, dl - 0.05, _ZS + _TH + 0.03))
            self.qB_lid2[i], _ = self.armB.reach(self.baseB, "left", fr.p(_D + 0.05, dl - 0.25, _ZS + _TH + 0.03), q_init=self.qB_lid[i])
            self.qB_drop[i], _ = self.armB.reach(self.baseB, "right", fr.p(_D, dl, _ZS + _TH + 0.12))
        self.qB_carry = self.armB.neutral(self.baseB)

    def precompute(self):
        pass

    def describe(self, name):
        return "a part"

    def instruction(self):
        return ("Kit the four totes on the rack (marks 1 to 4): every tote must end up with exactly ONE part. A tote with two "
                "parts is a kitting error. When you start, all four totes are empty. "
                "The lids are closed, and you cannot see inside a closed tote. Use the skill `kit <mark>` for each tote "
                "that still needs a part, then `done`.")

    def reset(self, ep, rng):
        self.pins = Pins(); self.occ = {}
        self.place_robots()
        self.pin_home("s09_")
        self.count = [0, 0, 0, 0]; self.lid_open = [False] * 4
        self.free_parts = [f"s09_part{k}" for k in range(9)]
        self.partner_parts = [f"s09_ppart{k}" for k in range(3)]
        self.hidden = {}
        self.v2s_setup(ep, rng)
        k = rng.choice([1, 2, 2, 3])
        self.partner_set = sorted(rng.sample(range(4), k))
        self.hidden.update(partner_totes=[i + 1 for i in self.partner_set],
                           partner_kits=[i + 1 for i in self.partner_set] if self.variant == "event" else [])
        self.helper = Helper(); self.agent_touched = False
        # partner holds its own parts: put them in its right hand later (spawned from the pool, out of sight)
        self.B.place_base(*self.baseB0)

    def lid_pose(self, i, frac):
        return np.asarray(self.fr.p(_D, _DL[i] - 0.22 * frac, _ZS + _TH + 0.01 + 0.02 * min(1, frac * 4)))

    def g_lid(self, bot, q_from, q_to, i, open_, arm):
        n = int(1.0 * SIM_HZ); q0 = bot.q.clone()
        yield from g_joints(bot, q_from if open_ else q_to, 0.8)
        for s in range(1, n + 1):
            u = ease(s / n); f = u if open_ else 1 - u
            self.pins.move(obj(f"s09_lid{i}"), self.lid_pose(i, f))
            qa, qb = (q_from, q_to) if open_ else (q_to, q_from)
            bot.q = qa + (qb - qa) * u; bot.q[bot.GRIP[arm]] = 0.0; yield
        self.lid_open[i] = open_

    def g_drop(self, bot, q_drop, i, part, arm):
        yield from g_joints(bot, q_drop, 1.0)
        bot.detach(obj(part))
        p = np.asarray(self.fr.p(_D + 0.06 * (self.count[i] % 2) - 0.03, _DL[i], _ZS + 0.04 + 0.05 * self.count[i]))
        self.pins.move(obj(part), p)
        self.count[i] += 1
        q = bot.q.clone(); q[bot.GRIP[arm]] = 0.05
        yield from g_joints(bot, q, 0.3)

    def partner_plan(self, ep):
        B = self.B
        yield from g_base(B, [self.baseB[:2]], final_yaw=self.baseB[2], speed=0.5)
        if self.agent_touched or self.variant == "null":
            if self.variant == "null":
                yield from g_still(2.0)
            self.helper.status = "agent is working here: leaving" if self.agent_touched else "looked at the rack: leaving"
            yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5); return
        for i in self.partner_set:
            if self.variant == "event":
                part = self.partner_parts.pop()
                # a part the partner brought along appears in its right hand (hand lowered behind the rack)
                self.pins.unpin(obj(part))
                B.apply()
                set_pose(obj(part), npf(B.r.get_eef_position("right")) + np.array([0, 0, -0.03]))
                B.attach(obj(part), "right", fixed_quat=self.fr.q())
            yield from self.g_lid(B, self.qB_lid[i], self.qB_lid2[i], i, True, "left")
            if self.variant == "event":
                yield from self.g_drop(B, self.qB_drop[i], i, part, "right")
            else:
                # decoy: the partner only looks into the open tote (no hand goes in, nothing is dropped)
                yield from g_still(1.3)
            yield from g_joints(B, self.qB_carry, 0.6)
            yield from self.g_lid(B, self.qB_lid[i], self.qB_lid2[i], i, False, "left")
            yield from g_joints(B, self.qB_carry, 0.6)
            ep.ev("partner_kitted" if self.variant == "event" else "partner_inspected", tote=i + 1)
        self.helper.status = "done: leaving"
        yield from g_base(B, [self.baseB0[:2]], final_yaw=self.baseB0[2], speed=0.5)

    def world(self, ep):
        super().world(ep)
        if not self.helper.started and self.actor_due(ep):
            if self.agent_touched:
                self.helper.started = True; self.helper.status = "agent working: partner skipped"
                self.hidden["partner_skipped"] = True; ep.ev("partner_skipped")
            else:
                self.helper.start(self.partner_plan(ep), "partner kitting"); self.actor_started(ep)
        self.helper.step()

    def skill_kit(self, ep, slot, target):
        if slot is None or not slot.startswith("tote"):
            ep.result = f"'{target}' is not a tote mark"; return
        i = int(slot[4:]); A = self.A
        self.agent_touched = True
        part = self.free_parts.pop(0)
        o = obj(part)
        p = self.pins.pos(o)
        q_up, _ = self.armA.reach(self.baseA, "left", p + np.array([0, 0, 0.16]))
        q_dn, _ = self.armA.reach(self.baseA, "left", p + np.array([0, 0, 0.03]), q_init=q_up)
        yield from g_joints(A, q_up, 0.9)
        yield from g_joints(A, q_dn, 0.6)
        q = A.q.clone(); q[A.GRIP["left"]] = 0.0; yield from g_joints(A, q, 0.3)
        self.pins.unpin(o); A.attach(o, "left", fixed_quat=self.fr.q())
        yield from g_joints(A, q_up, 0.6)
        yield from self.g_lid(A, self.qA_lid[i], self.qA_lid2[i], i, True, "right")
        yield from self.g_drop(A, self.qA_drop[i], i, part, "left")
        yield from self.g_lid(A, self.qA_lid[i], self.qA_lid2[i], i, False, "right")
        yield from g_joints(A, self.qA0, 0.9)
        ep.result = f"dropped one part into tote {i + 1} and closed the lid"

    def privileged(self):
        return dict(parts_per_tote=list(self.count), partner=self.helper.status)

    def verify(self, final):
        c = final["parts_per_tote"]
        if all(x == 1 for x in c):
            return True, "every tote holds exactly one part"
        bad = [f"tote {i + 1}: {x}" for i, x in enumerate(c) if x != 1]
        return False, "wrong counts (" + ", ".join(bad) + ")"
