"""The real robot so far: chest (with its bay and shoulder actuators), pelvis, waist actuator, both legs (the right one the
left mirrored), at one pose (both legs the same, mirrored), as one STEP to render.

    python cad/tools/real_lower_view.py out.step [pitch knee ankle_pitch ankle_roll]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cadgen import build123d as bd
from real import pelvis as P, torso as TO
from real_leg_view import leg, AL, ACT

if __name__ == "__main__":
    pose = [float(v) for v in sys.argv[2:6]] if len(sys.argv) > 2 else []
    L = leg(*pose)
    kids = []
    for c in L.children:
        a = bd.Pos(0, P.HY, 0) * c
        a.color = c.color
        b = bd.mirror(a, bd.Plane.XZ)
        b.color = c.color
        kids += [a, b]
    pv, wa, wb = P.pelvis(), P.waist_actuator(), P.waist_bearing()
    pv.color, wa.color, wb.color = AL, ACT, bd.Color(0.55, 0.56, 0.58)
    bd.export_step(bd.Compound(children=kids + [pv, wa, wb] + TO.assembly()), sys.argv[1])
