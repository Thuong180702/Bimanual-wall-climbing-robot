"""The real robot's left leg so far (hip, thigh, calf, foot) at one pose, as one STEP to render.

    python cad/tools/real_leg_view.py out.step [pitch knee ankle_pitch ankle_roll]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cadgen import build123d as bd
from real import hip as H, thigh as T, calf as K

AL, ACT, STEEL = bd.Color(0.78, 0.80, 0.83), bd.Color(0.36, 0.37, 0.40), bd.Color(0.55, 0.56, 0.58)


def leg(pitch=0.0, knee=0.0, ap=0.0, ar=0.0):
    kids = []
    def add(shape, color, loc=bd.Location()):
        s = loc * shape
        s.color = color
        kids.append(s)
    acts = H.actuators()
    for a in acts.values():
        add(a, ACT)
    q = H.roll_linkage(0.0)
    hr, hpins = H.roll_rod(q[1], q[2])
    for p in (H.yaw_link(), H.roll_yoke(), H.roll_crank(), H.roll_lever(), hr):
        add(p, AL)
    add(hpins, STEEL)
    th = T.about_y(pitch)                                     # thigh frame -> leg frame
    psi0 = T.knee_linkage(0.0)[0]
    psi, A, B = T.knee_linkage(knee)
    krod, kst = T.knee_rod(A, B)
    add(T.actuator(), ACT, th)
    for p in (T.thigh(), T.about_y(psi - psi0, T.ZK) * T.knee_crank(), krod):
        add(p, AL, th)
    add(kst, STEEL, th)
    cf = th * T.about_y(knee, T.ZKN) * bd.Pos(0, 0, T.ZKN)     # calf frame -> leg frame
    add(T.knee_lever(), AL, th * T.about_y(knee, T.ZKN))
    for a in K.actuators(vendor=True).values():
        add(a, ACT, cf)
    cr = K.ankle_cranks(ap, ar)
    add(K.calf(), AL, cf)
    for k, part, M in (("U", K.crank("U"), K.MU), ("L", K.crank("L"), K.ML)):
        add(bd.Pos(M[0], 0, M[1]) * bd.Rot(0, cr[k][0], 0) * bd.Pos(-M[0], 0, -M[1]) * part, AL, cf)
        add(K.rod(cr[k][1], cr[k][2]), STEEL, cf)
    add(bd.Pos(K.AK[0], 0, K.AK[1]) * bd.Rot(0, ap, 0) * bd.Pos(-K.AK[0], 0, -K.AK[1]) * K.cross(), STEEL, cf)
    add(K.foot_pose(ap, ar) * K.foot(), AL, cf)
    return bd.Compound(children=kids)


if __name__ == "__main__":
    pose = [float(v) for v in sys.argv[2:6]] if len(sys.argv) > 2 else []
    bd.export_step(leg(*pose), sys.argv[1])
