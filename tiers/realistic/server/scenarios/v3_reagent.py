"""V3 reagent (lab, DONE): flask A must receive exactly one dose of catalyst.

The robot labels sample tubes at the bench. Meanwhile the technician may dose flask A themself (event), dose flask B
next to it (decoy) or only stop at the bench (null). A dose leaves no visible trace in the flask.
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["v3_reagent"]
_FR = Frame(*_A["base"])
_H = 0.90
_objs = table("v3r_bench", _FR, 1.10, 0.0, 0.60, 1.50, _H, rgba=[0.88, 0.88, 0.86, 1]) + V3Scenario.cart_objects(_FR, "v3r_")
_objs += [box("v3r_shelf", _FR, 1.37, 0.0, _H + 0.22, 0.05, 1.40, 0.44, [0.75, 0.76, 0.78, 1])]
_objs += [ds("v3r_flaskA", "erlenmeyer_flask", "bpwjxr", 80), ds("v3r_flaskB", "erlenmeyer_flask", "bpwjxr", 81),
          ds("v3r_bottle", "reagent_bottle", "tnjpsf", 82),
          cyl("v3r_token", _FR, 0, 0, 0, 0.025, 0.012, TOKEN_RGBA)]
_objs[-1]["position"] = park(83)


@register
class V3Reagent(DoseOnce):
    sid = "v3_reagent"
    title = "Catalyst dose"
    setting = "chemistry lab bench"
    ANCHOR_KEY = "v3_reagent"
    PREFIX = "v3r_"
    TOKEN = "v3r_token"
    PEOPLE = (("v3r_tech", 0.96),)
    PEOPLE_K0 = 35
    OBJECTS = _objs + [person_cfg("v3r_tech", 35, 0.96)]
    h = _H
    hand_z = _H + 0.16
    chore = "labelling sample tubes at the bench"
    prelude_s = 30.0
    RECEPTACLES = (("v3r_flaskA", 1.00, 0.18, "flask A (batch 7)"), ("v3r_flaskB", 1.00, -0.16, "flask B (batch 8)"))
    SOURCE = ("v3r_bottle", 0.96, -0.44, "catalyst bottle")
    STAND = {"source": (1.76, -0.44), 0: (1.76, 0.18), 1: (1.76, -0.16)}
    ENTRY = (2.2, 2.4); EXIT = (2.2, 2.4)
    dose_word = "dose of catalyst"
    rec_word = "flask"
    extra_help = {"dose": "dose <mark>: add one dose of catalyst from the catalyst bottle to the flask at the numbered "
                          "place (one manipulation action)"}
    CAM = staticmethod(lambda fr: (fr.p(0.9, 2.7, 2.5), fr.p(1.2, 0.0, 0.95), 14.0))

    def task_text(self):
        return ("Flask A (mark 1) must receive exactly one dose of catalyst before it goes into the oven: no dose and two "
                "doses both ruin batch 7. Flask B (mark 2) is batch 8; leave it alone. A technician works at this bench "
                "too and sometimes doses flasks themself. The catalyst bottle is at mark 3. You have 2 manipulation actions; "
                "call done when flask A is right.")
