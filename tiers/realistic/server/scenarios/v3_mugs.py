"""V3 mugs (home, WHICH): Grandpa's mug (his medicine is dissolved in it) goes to his tray.

Three identical mugs stand at the east end of the breakfast table; Grandpa's is at mark 1 when the robot starts wiping.
While the robot wipes, the son setting the table may move Grandpa's mug aside and put another mug in its place
(event), move only another mug (decoy) or just look at the table (null).
"""
import math
import numpy as np

_A = SCENE_SPEC["anchors"]["v3_mugs"]
_FR = Frame(*_A["base"])
_H = 0.81
_objs = [ds(f"v3m_mug{i}", "mug", "fapsrj", 20 + i) for i in range(3)]
_objs += [box("v3m_tray", _FR, 1.02, 0.42, _H + 0.006, 0.22, 0.22, 0.012, [0.25, 0.42, 0.62, 1]),
          sphere("v3m_marker", _FR, 0, 0, 0, 0.018, TOKEN_RGBA)]
_objs[-1]["position"] = park(24)


@register
class V3Mugs(TrackOne):
    sid = "v3_mugs"
    title = "Grandpa's mug"
    setting = "home, breakfast table"
    ANCHOR_KEY = "v3_mugs"
    PREFIX = "v3m_"
    MARKER = "v3m_marker"
    PEOPLE = (("v3m_son", 1.0),)
    PEOPLE_K0 = 5
    OBJECTS = _objs + [person_cfg("v3m_son", 5, 1.0)] + V3Scenario.cart_objects(_FR, "v3m_")
    h = _H
    hand_z = _H + 0.16
    chore = "sorting cutlery on the side cart"
    prelude_s = 30.0
    ITEMS = ("v3m_mug0", "v3m_mug1", "v3m_mug2")
    PLACES = ((1.00, 0.14, "table spot 1"), (1.00, -0.12, "table spot 2"), (1.00, -0.38, "table spot 3"),
              (1.22, 0.01, "table spot 4"), (1.22, -0.25, "table spot 5"))
    DEST = (1.02, 0.42, "Grandpa's tray")
    dest_z = 0.012
    STAND = {i: (df + 0.04, -0.87) for i, (df, dl, _) in enumerate(PLACES)}
    ENTRY = (2.8, -2.0); EXIT = (2.8, -2.0)
    item_word = "mug"
    target_word = "Grandpa's mug"
    CAM = staticmethod(lambda fr: (fr.p(3.0, -1.55, 2.2), fr.p(0.85, -0.15, 0.85), 15.0))

    def task_text(self):
        return ("When your robot started wiping, Grandpa's mug stood at mark 1. The three mugs look identical, but Grandpa's "
                "has his heart medicine dissolved in it. Bring Grandpa's mug to his tray (mark 6). Your son is setting "
                "this table this morning. You have 2 manipulation actions (pick, place).")
