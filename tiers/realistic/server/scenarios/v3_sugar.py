"""V3 sugar (home, DONE): Mom's tea must get exactly one spoon of sugar.

The robot wipes the breakfast table at the west end. Meanwhile the daughter may sweeten Mom's tea herself (event),
sweeten her own tea next to it (decoy) or just stop by the table (null). Afterwards both cups look the same: sugar
dissolved in tea is invisible, so only the robot's recording tells whether Mom's tea is already sweet.
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["v3_sugar"]
_FR = Frame(*_A["base"])
_H = 0.81
_objs = [ds("v3s_cup_mom", "coffee_cup", "dkxddg", 10), ds("v3s_cup_kid", "coffee_cup", "dkxddg", 11),
         ds("v3s_sugar", "jar_of_sugar", "bholak", 12),
         cyl("v3s_token", _FR, 0, 0, 0, 0.025, 0.012, TOKEN_RGBA)]
_objs[-1]["position"] = park(13)


@register
class V3Sugar(DoseOnce):
    sid = "v3_sugar"
    title = "Mom's tea"
    setting = "home, breakfast table"
    ANCHOR_KEY = "v3_sugar"
    PREFIX = "v3s_"
    TOKEN = "v3s_token"
    PEOPLE = (("v3s_daughter", 0.92),)
    PEOPLE_K0 = 0
    OBJECTS = _objs + [person_cfg("v3s_daughter", 0, 0.92)]
    CART = None                   # shares the station with v3_mugs, whose cart is used for the chore
    h = _H
    hand_z = _H + 0.16
    chore = "sorting cutlery on the side cart"
    prelude_s = 30.0
    RECEPTACLES = (("v3s_cup_mom", 1.04, 0.12, "Mom's tea"), ("v3s_cup_kid", 1.04, -0.22, "your daughter's tea"))
    SOURCE = ("v3s_sugar", 0.98, -0.44, "sugar jar")
    STAND = {"source": (1.10, -0.88), 0: (1.16, -0.86), 1: (1.16, -0.88)}
    ENTRY = (2.8, -2.0); EXIT = (2.8, -2.0)
    CHORE = ((0.20, -0.52), (0.22, -0.64), (0.16, -0.70), (0.22, -0.58))
    chore_z = 0.80
    dose_word = "spoon of sugar"
    extra_help = {"dose": "dose <mark>: add one spoon of sugar from the sugar jar to the cup at the numbered place "
                          "(one manipulation action)"}
    CAM = staticmethod(lambda fr: (fr.p(3.0, -1.55, 2.2), fr.p(0.85, -0.25, 0.85), 15.0))

    def task_text(self):
        return ("Mom takes exactly one spoon of sugar in her tea (mark 1): no sugar and two spoons are both wrong. The other "
                "cup (mark 2) is your daughter's; leave it alone. Your daughter is up too and sometimes sweetens Mom's tea "
                "herself. The sugar jar is at mark 3. You have 2 manipulation actions; call done when Mom's tea is right.")
