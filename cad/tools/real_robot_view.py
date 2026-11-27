"""The whole real robot as drawn: legs (the right one the left mirrored), pelvis, waist, chest
with its bay, lidars, head pedestal and guard, both arms (the right one mirrored), at one pose.

    python cad/tools/real_robot_view.py out.step [stand|hang]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cadgen import build123d as bd
from real import pelvis as P, torso as TO, arm as A
from real_leg_view import leg, AL, ACT

POSES = {"stand": dict(flex=0, abd=10, rot=0, elb=-20),
         "hang": dict(flex=-170, abd=10, rot=0, elb=-20, wfl=0, finger=-30)}

if __name__ == "__main__":
    pose = POSES[sys.argv[2] if len(sys.argv) > 2 else "stand"]
    L = leg()
    kids = []
    for c in L.children:
        a = bd.Pos(0, P.HY, 0) * c
        a.color = c.color
        b = bd.mirror(a, bd.Plane.XZ)
        b.color = c.color
        kids += [a, b]
    for c in A.arm_assembly(**pose):
        b = bd.mirror(c, bd.Plane.XZ)
        b.color = c.color
        kids += [c, b]
    pv, wa, wb = P.pelvis(), P.waist_actuator(), P.waist_bearing()
    pv.color, wa.color, wb.color = AL, ACT, bd.Color(0.55, 0.56, 0.58)
    bd.export_step(bd.Compound(children=kids + [pv, wa, wb] + TO.assembly()), sys.argv[1])
