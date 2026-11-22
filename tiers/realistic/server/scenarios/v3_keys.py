"""V3 keys (home, WHERE): bring Dad's car keys to the key tray.

The keys lie on the kitchen island next to three identical lidded baskets. While the robot wipes the island, the teenager
tidying up may drop the keys into one basket and close it (event), drop them into one basket and then move them to
another (decoy: only the last basket counts) or leave them where they are (null).
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["v3_keys"]
_FR = Frame(*_A["base"])
_H = 1.05


@register
class V3Keys(Stash):
    sid = "v3_keys"
    title = "Dad's car keys"
    setting = "home, kitchen island"
    ANCHOR_KEY = "v3_keys"
    PREFIX = "v3k_"
    PEOPLE = (("v3k_teen", 0.94),)
    PEOPLE_K0 = 10
    h = _H
    hand_z = _H + 0.16
    chore = "sorting cutlery on the side cart"
    prelude_s = 34.0
    ITEM = ("v3k_keys", 1.00, 0.50, "car keys spot on the island (just beyond this mark)")
    ITEM_MARK_D = (-0.12, 0.0)   # main_1115: the mark was drawn on the keys and hid them; now 12 cm nearer the robot   # 11-03 post-run fix: was (1.22, 0.42), its mark sat on basket 1's
    BOXES = (("v3k_box0", 1.02, 0.30, "basket 1"), ("v3k_box1", 1.02, 0.04, "basket 2"), ("v3k_box2", 1.02, -0.22, "basket 3"))
    BOX = (0.22, 0.22, 0.13)
    item_word = "car keys"
    DEST = (0.98, -0.46, "key tray")
    dest_z = 0.012
    STAND = {"item": (1.02, 0.88), 0: (1.06, 0.86), 1: (1.06, 0.86), 2: (1.08, 0.86)}
    ENTRY = (2.9, 2.2); EXIT = (2.9, 2.2)
    CAM = staticmethod(lambda fr: (fr.p(2.9, 1.75, 2.35), fr.p(0.85, 0.0, 1.0), 15.0))

    def task_text(self):
        return ("Dad is leaving for the airport and needs his car keys. When your robot started, the keys lay on the island "
                "at mark 4. Bring them to the key tray (mark 5). Your teenager is tidying the kitchen this morning; the "
                "three baskets (marks 1 to 3) have lids. Opening a basket, picking and placing each use one of your 3 "
                "manipulation actions.")


V3Keys.OBJECTS = (V3Keys.box_objects(_FR, _H, [0.62, 0.47, 0.30, 1], [0.55, 0.40, 0.25, 1])
                  + [ds("v3k_keys", "key_chain", "gqbwfv", 30),
                     box("v3k_tray", _FR, 0.98, -0.46, _H + 0.006, 0.18, 0.18, 0.012, [0.20, 0.20, 0.22, 1])]
                  + V3Scenario.cart_objects(_FR, "v3k_")
                  + [person_cfg("v3k_teen", 10, 0.94)])
