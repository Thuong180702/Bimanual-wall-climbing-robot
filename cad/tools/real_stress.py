"""Peak-load stress of the real robot's structural parts (voxel FEA, real_fea.py).

Loads (95 kg robot, running / landing on one leg; actuators at their peaks):
  ground reaction 3 x weight = 2.8 kN up, 0.84 kN fore-aft, 0.6 kN sideways;
  hip pitch 400, hip roll 462, knee 416, ankle 459 N.m; knee rod <= 8.7 kN (14-119 deg),
  ankle rods 4.6 kN each, roll rod 8 kN; hanging by both arms 2 g: 0.93 kN per shoulder
  with the shoulder-flexion reaction 330 N.m.
Each part: supports where it is bolted to the part before it, loads where the next one bolts
on (see CASES). Criterion (2026-11-26, the common one for legged robots): these loads are rare
peaks (all actuators at peak with 3 x weight at once), so the surface stress = element-centre
von Mises (99.5th percentile away from the loaded / fixed nodes) x 1.4 (voxel centres sit inside
the surface fibre) <= 0.52 x yield = 260 MPa; the running stride's loads (~0.6 of these) then
sit at <= 156 MPa, under 7075-T6's ~160 MPa fatigue strength at 5e8 cycles.

    python cad/tools/real_stress.py [part ...]
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_fea import Model, voxelize
from real.hip import AL7075

GRF = np.array([840.0, 600.0, 2800.0])
KNEE_ROD, ANKLE_ROD, ROLL_ROD = 8700.0, 4600.0, 8000.0   # ankle: 170 N.m on the 37 mm crank
FACTOR, ALLOW = 1.4, 260.0
PITCH = {"chest": 3.0}            # the chest is big: a coarser grid so the solver converges


def ring(c, axis, r0, r1, lo, hi):
    """nodes within radius r0..r1 of the axis line through c, axial coordinate lo..hi"""
    ax = "xyz".index(axis)
    def f(P):
        q = P - np.asarray(c, float)
        rad = np.sqrt((q ** 2).sum(1) - q[:, ax] ** 2)
        return (rad >= r0) & (rad <= r1) & (P[:, ax] >= lo) & (P[:, ax] <= hi)
    return f


def cases():
    from real import hip as H, thigh as T, calf as K, pelvis as P, torso as TO
    out = {}
    # thigh: fixed on the pitch output boss; the knee bearings take the calf (GRF + the rod's
    # reaction), the knee actuator's ring the rod force and the actuator's torque
    out["thigh"] = (T.thigh, ring((0, 0, 0), "y", 0, 35, T.H13 - 0.5, T.Y0 + 0.5),
                    [(ring((0, 0, T.ZKN), "y", 30, 45, T.Y0 - 5, T.Y0 + 11), (GRF + [0, 0, KNEE_ROD]) / 2, (0, 0, 0)),
                     (ring((0, 0, T.ZKN), "y", 30, 42, T.YI0 - 1, T.YI1 + 3), (GRF + [0, 0, KNEE_ROD]) / 2, (0, 0, 0)),
                     (ring((0, 0, T.ZK), "y", 70, 86, T.Y0 - 0.5, T.Y0 + 3), (0, 0, -KNEE_ROD), (0, 416e3, 0))])
    # calf: fixed on the knee shaft; the ankle bosses take the foot (GRF + both rods), each
    # actuator ring its rod's pull and its torque
    yo0, yi1 = K.YOU - 1, K.YOL + 1
    out["calf"] = (K.calf, ring((0, 0, 0), "y", 19, 23, -30, 45),
                   [(ring((0, 0, K.AK[1]), "y", 15, 25, -40, 40), GRF + [0, 0, 2 * ANKLE_ROD], (0, 0, 0)),
                    (ring((K.MU[0], 0, K.MU[1]), "y", 30, 42, yo0 - 0.5, yo0 + 2), (0, 0, -ANKLE_ROD), (0, 170e3, 0)),
                    (ring((K.ML[0], 0, K.ML[1]), "y", 30, 42, yi1 - 2, yi1 + 0.5), (0, 0, -ANKLE_ROD), (0, 170e3, 0))])
    # foot: fixed in the roll bearings; ground at the toe, both rods pulling the heel cups
    bu, bl = K.PINS["U"][3], K.PINS["L"][3]
    out["foot"] = (K.foot, ring((0, 0, K.AK[1]), "x", 11.5, 14, -33, 33),
                   [(lambda P: (P[:, 0] > 100) & (P[:, 2] < K.SOLE + 1), (0, 0, GRF[2]), (0, 0, 0)),
                    (ring((bu[0], bu[1], bu[2]), "z", 11, 15, bu[2] - 14, bu[2] + 2), (0, 0, -ANKLE_ROD), (0, 0, 0)),
                    (ring((bl[0], bl[1], bl[2]), "z", 11, 15, bl[2] - 14, bl[2] + 2), (0, 0, -ANKLE_ROD), (0, 0, 0))])
    # hip-yaw link: fixed on the yaw output; the roll bearings take the leg (GRF moved from the
    # hip centre: + the couple), the roll actuator's ring the rod pull and its torque
    xb = (H.XW - H.TW + H.BOSS_X) / 2
    M_b = np.cross([0 - xb, 0, 0], GRF)
    out["hip yaw link"] = (H.yaw_link, ring((0, 0, H.ZY), "z", 0, 30, H.ZY - 9.5, H.ZY + 0.5),
                           [(ring((0, 0, 0), "x", 27, 31, H.XW - H.TW, H.BOSS_X), GRF + [0, 0, ROLL_ROD], M_b),
                            (ring((0, H.S, H.ZR), "x", 52, 66, H.XW - 0.5, H.XW + 2), (0, 0, -ROLL_ROD), (330e3, 0, 0))])
    # roll yoke: fixed in the roll bearings (shaft); the pitch actuator's rear face (20 x M4 on
    # 155) brings the leg, the pitch actuator's torque and the roll torque
    out["roll yoke"] = (H.roll_yoke, ring((0, 0, 0), "x", 13, 15.5, -160, -115),
                        [(ring((0, 0, 0), "y", 68, 83, -H.H13 - 3, -H.H13 + 0.5), GRF, (462e3, -400e3, 0))])
    # pelvis: fixed in the waist bearing's seat; one leg's yaw actuator brings its stance
    out["pelvis"] = (P.pelvis, ring((0, 0, 0), "z", 58, 61, P.BR[2], P.BR[3]),
                     [(ring((0, P.HY, 0), "z", 44, 53, P.ZD0 - 0.5, P.ZD0 + 3), GRF, (462e3, 400e3, 90e3))])
    # chest: fixed at the waist (flange + bearing skirt); hanging, each shoulder pulls up with
    # its flexion actuator's reaction torque
    out["chest"] = (TO.chest, lambda Pn: (Pn[:, 2] < TO.ZB1 + 0.5) & (np.hypot(Pn[:, 0], Pn[:, 1]) < 84),
                    [(ring((0, TO.SY, TO.ZS), "y", 50, 66, TO.SY - 0.5, TO.SY + 3), (0, 0, 930), (0, 330e3, 0)),
                     (ring((0, -TO.SY, TO.ZS), "y", 50, 66, -TO.SY - 3, -TO.SY + 0.5), (0, 0, 930), (0, 330e3, 0)),
                     (lambda Pn: (Pn[:, 0] < -TO.XD + 1) & (np.abs(np.abs(Pn[:, 1]) - 70) < 6), (0, 0, -180), (0, 0, 0))])
    # shoulder bracket: fixed on the flexion output; the abduction actuator's front ring brings the
    # hang (0.93 kN), the abduction peak (400 N.m) and the flexion peak passed on to the arm (330 N.m)
    from real import arm as A
    out["shoulder bracket"] = (A.shoulder_bracket, ring((0, TO.SY, TO.ZS), "y", 0, 30.5, TO.SY - 0.5, TO.SY + 1),
                               [(ring((A.XO, A.ABD_Y, TO.ZS), "x", 70, 85, A.XO - 1.5, A.XO + 0.5), (300, 0, 930), (400e3, 330e3, 0))])
    # upper arm: fixed on the abduction output (its hub face); the rotation actuator's rear ring brings
    # the hang (0.93 kN) and, all at once, the abduction, flexion and rotation peaks (400, 330, 400 N.m)
    out["upper arm"] = (A.upper_arm, ring((A.XO, A.ABD_Y, TO.ZS), "x", 0, 34, A.XO - 1.5, A.XO + 0.5),
                        [(ring((0, A.ABD_Y, 0), "z", 70, 85, A.CAP0 - 0.5, A.CAP0 + 2), (300, 0, -930), (400e3, 330e3, 400e3))])
    # humerus: fixed on the rotation output; the elbow actuator's two rings (inboard front, outboard rear)
    # share the hang and the elbow, rotation and abduction peaks (400 each)
    yo = A.EY + A.X15_L
    out["humerus"] = (A.humerus, ring((0, A.ABD_Y, 0), "z", 0, 34, A.ROT_Z - 1.5, A.ROT_Z + 0.5),
                      [(ring((0, 0, A.ELB_Z), "y", 66, 85, yo - 0.5, yo + 1.5), (300, 0, -930), (400e3, 450e3, 400e3))])
    # forearm: fixed on the elbow output (its hub face); the roll actuator's front ring brings the hang
    # and the elbow peak (400, about y), the roll peak (150, about z) and a sideways 200 N.m
    out["forearm"] = (A.forearm, ring((0, 0, A.ELB_Z), "y", 0, 57, A.EY - 0.5, A.EY + 1.5),
                      [(ring((0, A.ABD_Y, 0), "z", 50, 66, A.FR_Z - 2, A.FR_Z + 0.5), (300, 0, -930), (200e3, 450e3, 150e3))])
    # wrist link: fixed on the roll output; the flexion AKE90's ring brings the hang and the wrist peaks (170 each)
    # with the roll's 150
    out["wrist link"] = (A.wrist_link, ring((0, A.ABD_Y, 0), "z", 0, 30, A.FR_Z - 1.5, A.FR_Z + 0.5),
                         [(ring((0, 0, A.WZ), "y", 30, 41, A.FX_Y + 0.5, A.FX_Y + 1.5), (300, 0, -930), (170e3, 170e3, 150e3))])
    # gimbal: fixed on the flexion output; the deviation AKE90's ring brings the hang and both wrist peaks
    out["gimbal"] = (A.gimbal, ring((0, 0, A.WZ), "y", 0, 25, A.FX_Y - 1.5, A.FX_Y + 0.5),
                     [(ring((0, A.ABD_Y, A.WZ), "x", 30, 41, A.GX0 - 0.5, A.GX0 + 1.5), (300, 0, -930), (170e3, 170e3, 100e3))])
    # hand: fixed on the deviation output; the finger's housing ring brings the hang (the hook's 0.93 kN) with
    # the deviation peak (170), the finger peak (72) and a twist of 100 N.m
    out["hand"] = (A.hand, ring((0, A.ABD_Y, A.WZ), "x", 0, 25, A.DV_O - 0.5, A.DV_O + 1.5),
                   [(ring((0, 0, A.FZ), "y", 30, 41, A.FO_Y - 6.5, A.FO_Y - 4.5), (300, 0, -930), (170e3, 72e3, 100e3))])
    # hook: fixed on the finger output; the rung pulls the throat (on the lip) with the whole hang
    out["hook"] = (A.hook, ring((0, 0, A.FZ), "y", 0, 26, A.FO_Y - 0.5, A.FO_Y + 1.5),
                   [(lambda P: (P[:, 2] > A.FZ - A.HOOK_B + A.HOOK_T - 0.5) & (P[:, 2] < A.FZ - A.HOOK_B + A.HOOK_T + 1.5)
                     & (P[:, 0] < -A.HOOK_T / 2 - 5), (0, 0, 930), (0, 0, 0))])
    return out


def check(name, case, pitch=None):
    pitch = pitch or PITCH.get(name, 2.0)
    make, fixed, loads = case
    part = make()
    t = time.time()
    m = Model(voxelize(part, pitch), pitch)
    fx = m.region(fixed)
    ld = [(m.region(f), F, M) for f, F, M in loads]
    if len(fx) == 0 or any(len(i) == 0 for i, _, _ in ld):
        return dict(name=name, error="empty region: fixed %d, loads %s" % (len(fx), [len(i) for i, _, _ in ld]))
    u = m.solve(fx, ld)
    r = m.report(u)
    r.update(name=name, mass=part.volume * AL7075, secs=time.time() - t, ok=r["stress"] * FACTOR <= ALLOW)
    return r


if __name__ == "__main__":
    C = cases()
    for n in (sys.argv[1:] or list(C)):
        r = check(n, C[n])
        if "error" in r:
            print("%-14s %s" % (n, r["error"]))
            continue
        print("%-14s %5.0f g  stress %4.0f x %.1f = %4.0f MPa (%s)  deflection %.2f mm  at %s  [%d elems, %.0f s%s]"
              % (n, r["mass"], r["stress"], FACTOR, r["stress"] * FACTOR, "OK" if r["ok"] else "OVER", r["defl"],
                 np.round(r["peak_at"]).astype(int).tolist(), r["elems"], r["secs"], "" if r["cg"] == 0 else ", cg %d" % r["cg"]))
