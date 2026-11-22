"""V3 totes (warehouse, WHICH): the express-order tote goes onto the outbound conveyor.

Three identical blue totes stand on the packing table; the express order is the one at mark 1 when the robot starts
re-labelling. While the robot works, a picker consolidating totes may move the express tote to a free spot and put
another tote where it was (event), move only another tote (decoy) or just check the table (null).
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["v3_totes"]
_FR = Frame(*_A["base"])
_H = 0.80
_TOTE_RGBA = [0.14, 0.34, 0.66, 1]
_objs = table("v3t_table", _FR, 1.13, 0.0, 0.66, 1.70, _H, rgba=STEEL) + V3Scenario.cart_objects(_FR, "v3t_")
_objs += [box(f"v3t_tote{i}", _FR, 0, 0, 0, 0.24, 0.20, 0.14, _TOTE_RGBA) for i in range(3)]
for i in range(3):
    _objs[-3 + i]["position"] = park(60 + i)
_objs += [box("v3t_conv", _FR, 0.98, -0.62, _H + 0.006, 0.30, 0.34, 0.012, [0.15, 0.15, 0.16, 1]),
          box("v3t_conv_edge0", _FR, 0.98, -0.80, _H + 0.02, 0.30, 0.02, 0.04, YELLOW),
          box("v3t_conv_edge1", _FR, 0.98, -0.44, _H + 0.02, 0.30, 0.02, 0.04, YELLOW),
          sphere("v3t_marker", _FR, 0, 0, 0, 0.02, TOKEN_RGBA)]
_objs[-1]["position"] = park(63)


@register
class V3Totes(TrackOne):
    sid = "v3_totes"
    title = "Express tote"
    setting = "warehouse packing table"
    ANCHOR_KEY = "v3_totes"
    PREFIX = "v3t_"
    MARKER = "v3t_marker"
    PEOPLE = (("v3t_picker", 1.0),)
    PEOPLE_K0 = 25
    OBJECTS = _objs + [person_cfg("v3t_picker", 25, 1.0)]
    h = _H
    hand_z = _H + 0.16
    chore = "re-labelling cartons at the packing table"
    prelude_s = 30.0
    ITEMS = ("v3t_tote0", "v3t_tote1", "v3t_tote2")
    PLACES = ((0.98, 0.52, "table spot 1"), (0.98, 0.24, "table spot 2"), (0.98, -0.04, "table spot 3"),
              (1.24, 0.38, "table spot 4"), (1.24, 0.10, "table spot 5"))
    DEST = (0.98, -0.62, "outbound conveyor")
    dest_z = 0.012
    STAND = {i: (1.82, dl) for i, (df, dl, _) in enumerate(PLACES)}
    ENTRY = (2.2, 2.4); EXIT = (2.2, -2.6)
    item_word = "tote"
    target_word = "express tote"
    CAM = staticmethod(lambda fr: (fr.p(-0.5, 2.0, 2.4), fr.p(1.2, 0.0, 0.8), 14.0))

    def task_text(self):
        return ("When your robot started, the express-order tote stood at mark 1. The three totes look identical. Put the "
                "express tote on the outbound conveyor (mark 6) before the truck leaves. A picker is consolidating totes "
                "at this table. You have 2 manipulation actions (pick, place).")
