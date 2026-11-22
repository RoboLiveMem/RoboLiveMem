"""V3 bins (factory, WHERE): bring the inspection gauge to the calibration tray.

The gauge lies on the workbench next to three identical lidded parts bins. While the robot sorts fasteners, a technician
may put the gauge away into one bin (event), put it into one bin and then move it to another (decoy: the last one counts)
or leave it on the bench (null).
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["v3_bins"]
_FR = Frame(*_A["base"])
_H = 0.80


@register
class V3Bins(Stash):
    sid = "v3_bins"
    title = "Inspection gauge"
    setting = "factory workbench"
    ANCHOR_KEY = "v3_bins"
    PREFIX = "v3b_"
    PEOPLE = (("v3b_tech", 1.0),)
    PEOPLE_K0 = 30
    h = _H
    hand_z = _H + 0.16
    chore = "sorting fasteners at the workbench"
    prelude_s = 36.0
    ITEM = ("v3b_gauge", 0.98, 0.58, "gauge spot on the bench")
    BOXES = (("v3b_bin0", 1.06, 0.28, "bin 1"), ("v3b_bin1", 1.06, 0.00, "bin 2"), ("v3b_bin2", 1.06, -0.28, "bin 3"))
    BOX = (0.22, 0.22, 0.13)
    item_word = "inspection gauge"
    DEST = (0.98, -0.58, "calibration tray")
    dest_z = 0.012
    STAND = {"item": (1.82, 0.58), 0: (1.82, 0.28), 1: (1.82, 0.0), 2: (1.82, -0.28)}
    ENTRY = (2.2, -2.4); EXIT = (2.2, -2.4)
    CAM = staticmethod(lambda fr: (fr.p(-0.5, -2.0, 2.4), fr.p(1.2, 0.0, 0.8), 14.0))

    def task_text(self):
        return ("The inspection gauge must go to the calibration tray (mark 5) now. When your robot started, the gauge lay "
                "on the bench at mark 4. A technician works at this bench; the three parts bins (marks 1 to 3) have lids. "
                "Opening a bin, picking and placing each use one of your 3 manipulation actions.")


V3Bins.OBJECTS = (table("v3b_bench", _FR, 1.13, 0.0, 0.66, 1.70, _H, rgba=[0.45, 0.47, 0.50, 1]) + V3Scenario.cart_objects(_FR, "v3b_")
                  + V3Bins.box_objects(_FR, _H, [0.30, 0.40, 0.55, 1], [0.24, 0.32, 0.45, 1])
                  + [box("v3b_gauge", _FR, 0, 0, 0, 0.12, 0.08, 0.05, ORANGE),
                     box("v3b_tray", _FR, 0.98, -0.58, _H + 0.006, 0.20, 0.20, 0.012, [0.85, 0.85, 0.80, 1])]
                  + [person_cfg("v3b_tech", 30, 1.0)])
V3Bins.OBJECTS[-3]["position"] = park(70)
