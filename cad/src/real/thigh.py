"""Thigh and knee drive of the real robot (left leg), option B: the hip-pitch actuator's
output turns the thigh; the knee actuator (A13720) sits high in the thigh and drives the
knee through a four-bar on the thigh's inboard side (KAIST-style).

Thigh frame = leg frame of hip.py at hip pitch 0 (mm): origin at the hip centre, x forward,
y outward, z up; the knee axis at z = -340.

Knee four-bar (searched, see the module note in docs): crank 80 mm on the knee actuator,
lever 85 mm on the calf, both pointing 200 deg (down, a little back) at knee 0, rod 164.7 mm.
Ratio knee/actuator 1.04-1.08 over knee 15-80 deg (running), >= 1.04 everywhere, 2.0 at
135 deg; rod force at the actuator's 400 N.m peak <= 8.7 kN over 14-119 deg, 15 kN at 0 and
18 kN at 135 -> M16 rod ends (static ~70 kN), 16 mm steel studs in single shear (the rod
runs in its own plane inboard of the crank, clear of both hubs).
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadgen import build123d as bd
from real.encos import body, SPEC
from real.hip import C, R, hull, _xz, AL7075, H13

ZK, ZKN = -184.0, -340.0                 # knee actuator axis (low enough that the crank, pointing up at
                                         # knee 135, clears the pitch actuator), knee axis
Y0 = H13 + 2.5                           # 44.75: spine inner face, 2.5 mm off the pitch housing
SPT = 8.0                                # spine thickness
YO = Y0 - SPEC["A13720"]["L"]            # -39.75: knee actuator output face (inboard)
YC0, YC1 = YO, YO - 12.0                 # crank / lever plane
YR = YC1 - 9.0                           # rod plane centre -60.75 (eye 14 wide)
YI0, YI1 = -37.0, -27.0                  # inboard knee cheek
R1, R2, ANG = 80.0, 85.0, 200.0          # crank, lever, their direction at knee 0 (deg from +z toward +x)


def u(a):
    return math.sin(math.radians(a)), math.cos(math.radians(a))


def knee_linkage(knee, step=1.0):
    """knee angle (deg, flexion +) -> crank angle (deg), crank pin A, lever pin B, all (x, z)"""
    A = (R1 * u(ANG)[0], ZK + R1 * u(ANG)[1])
    B0 = (R2 * u(ANG)[0], ZKN + R2 * u(ANG)[1])
    L = math.dist(A, B0)
    n = max(1, int(abs(knee) / step))
    for i in range(1, n + 1):
        t = knee * i / n
        B = (R2 * u(ANG + t)[0], ZKN + R2 * u(ANG + t)[1])
        dx, dz = B[0], B[1] - ZK
        d = math.hypot(dx, dz)
        a = (R1 ** 2 - L ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(max(R1 ** 2 - a ** 2, 0.0))
        ex, ez = dx / d, dz / d
        px, pz = a * ex, ZK + a * ez
        sols = [(px - h * ez, pz + h * ex), (px + h * ez, pz - h * ex)]
        A = min(sols, key=lambda s: math.dist(s, A))          # the branch it is already on
    B = (R2 * u(ANG + knee)[0], ZKN + R2 * u(ANG + knee)[1])
    return math.degrees(math.atan2(A[0], A[1] - ZK)), A, B


def actuator():
    return bd.Location(bd.Plane(origin=(0, YO, ZK), x_dir=(1, 0, 0), z_dir=(0, -1, 0))) * body("A13720")


def _ycyl(x, z, r, y0, y1):
    return bd.Pos(x, (y0 + y1) / 2, z) * bd.Rot(90, 0, 0) * bd.Cylinder(r, abs(y1 - y0))


def _ring(n, pcd, x, z, r, y0, y1, a0=0.0):
    cut = None
    for i in range(n):
        a = math.radians(a0 + 360.0 * i / n)
        c = _ycyl(x + pcd / 2 * math.cos(a), z + pcd / 2 * math.sin(a), r, y0, y1)
        cut = c if cut is None else cut + c
    return cut


def _output_holes(z, y0, y1):
    """A13720 output star: centre 29, 3 x M5 on 40, 3 x 6 mm dowels on 50"""
    return _ycyl(0, z, 14.5, y0, y1) + _ring(3, 40, 0, z, 2.75, y0, y1) + _ring(3, 50, 0, z, 3.0, y0, y1, 60)


def thigh():
    y1 = Y0 + SPT
    spine = _xz(hull(C(0, 0, 42), C(0, ZK, 86), C(0, ZKN, 44), R(80, 95, -285, -195)), Y0, SPT)
    spine = bd.fillet(spine.edges(), 2.0)
    # a single plate is floppy sideways (FEA: 34 mm under the 0.6 kN side load at the knee): a 12 mm
    # doubler outside the spine's upper half, and a front flange (an L section) past the knee actuator
    dsk = hull(C(0, 0, 38), C(0, ZK, 66), C(0, ZKN + 30, 36))                   # hip to knee: the side load's whole path
    doubler = _xz(dsk - hull(C(0, -48, 14), C(0, -110, 24)) - C(0, ZK, 50) - hull(C(0, -262, 12), C(0, -282, 12)),
                  Y0 + SPT - 0.5, 14.5)                                       # rims, open over the knee actuator (air)
    try:
        doubler = bd.fillet(doubler.edges(), 2.0)
    except Exception:                      # noqa: BLE001 -- sharp edges rather than no part
        pass
    # the flange only below z -195: higher, its front edge swings into the hip-yaw actuator at hip flexion
    # -115..-125; above that the doubler (20 mm) carries the side load
    fsk = bd.Polygon((-36, -290), (Y0 + SPT + 12, -290), (Y0 + SPT + 12, -195), (-36, -235), align=None)
    fsk = bd.fillet(fsk.vertices(), 6.0)
    flange = bd.extrude(bd.Plane(origin=(89, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * fsk, amount=6, dir=(1, 0, 0))   # tapers into the doubler
    boss = _ycyl(0, 0, 35, H13, Y0 + 1)                                   # on the pitch output's 70 mm pilot
    khous_o = _ycyl(0, ZKN, 44, Y0 - 4, y1)                               # outboard knee bearing housing
    # below the knee actuator: bridge, knee cap (front wall) and the inboard cheek round the calf
    # only in FRONT of the knee: at deep flexion everything behind it would fold into the calf
    # deep enough to carry the inboard knee bearing's half of the knee load (~5.8 kN) across to the
    # spine: 26 mm between the knee actuator (-268) and the calf's hub (-300); FEA in real_stress.py
    bridge = bd.Pos(45.5, (YI0 + Y0 + 1) / 2, -284) * bd.Box(99, Y0 + 1 - YI0, 26)
    cap = bd.Pos(47, (YI0 + Y0 + 1) / 2, -306) * bd.Box(6, Y0 + 1 - YI0, 68)
    cheek = _xz(hull(C(0, ZKN, 40), R(-4, 54, -297, -271)), YI0, YI1 - YI0)
    khous_i = _ycyl(0, ZKN, 40, YI0, YI1 + 2)
    part = spine + doubler + flange + boss + khous_o + bd.fillet(bridge.edges(), 2) + bd.fillet(cap.edges(), 2) + \
        bd.fillet(cheek.edges(), 2) + khous_i
    # holes: pitch output star, knee actuator's rear face 20 x M4 on 155 + centre, knee bearings (6908: 62 OD)
    part -= _output_holes(0, H13 - 1, y1 + 15)
    part -= _ring(20, 155, 0, ZK, 2.25, Y0 - 1, y1 + 1) + _ycyl(0, ZK, 14.5, Y0 - 1, y1 + 1)
    part -= _ycyl(0, ZKN, 31, Y0 - 5, y1 + 1) + _ycyl(0, ZKN, 31, YI0 - 1, YI1 + 3)
    # lightening windows in the spine: one between the pitch hub and the knee actuator's bolt ring,
    # four inside that ring (air over the actuator's rear cover)
    part -= _ycyl(0, ZK, 62, Y0 - 1, y1 + 1)          # open inside the knee actuator's bolt ring (r 70..86 stays)
    part -= _xz(hull(C(0, -48, 14), C(0, -100, 22)), Y0 - 1, SPT + 2)             # under the doubler's opening
    part -= _xz(hull(C(-50, -250, 9), C(-25, -262, 9)), Y0 - 1, SPT + 2)          # behind the knee actuator's ring
    part -= bd.Pos(92, 8, -262) * bd.Box(10, 46, 36)                              # flange: a window (it works in its rim)
    # the doubler milled from outside: 10 mm deep pockets between 7 mm rims and three ribs, a 4.5 mm
    # skin left on the spine (an I section sideways at ~half the doubler's mass)
    holes = hull(C(0, -48, 14), C(0, -110, 24)) + C(0, ZK, 50) + hull(C(0, -262, 12), C(0, -282, 12))
    pk = bd.offset(dsk, -7) - bd.offset(holes, 7)
    for zr in (-30.0, -140.0, -232.0):
        pk -= R(-120, 120, zr - 3.5, zr + 3.5)
    part -= _xz(pk, Y0 + SPT + 4, 12)
    for xw in (6.5, 31.0, 55.5, 80.0):                                            # bridge: webs across y, open between
        part -= bd.Pos(xw + 6, (YI1 + 2 + Y0 - 2) / 2, -284) * bd.Box(12.5, Y0 - YI1 - 6, 30)
    return part


def knee_crank():
    _, A, _ = knee_linkage(0.0)
    p = _xz(hull(C(0, ZK, 40), C(*A, 15)), YC1, YC0 - YC1)
    p = bd.fillet(p.edges(), 2.0)
    p -= _output_holes(ZK, YC1 - 1, YC0 + 1)
    p -= _ycyl(*A, 8.0, YC1 - 1, YC0 + 1)
    return p


def knee_lever():
    """on the calf's knee shaft (inboard end, clamped); drawn at knee 0"""
    _, _, B = knee_linkage(0.0)
    p = _xz(hull(C(0, ZKN, 28), C(*B, 15)), YC1, YC0 - YC1)
    p = bd.fillet(p.edges(), 2.0)
    p -= _ycyl(0, ZKN, 15, YC1 - 1, YC0 + 1)
    p -= _ycyl(*B, 8.0, YC1 - 1, YC0 + 1)
    return p


def knee_rod(A, B):
    eyes = _ycyl(*A, 18, YR - 7, YR + 7) + _ycyl(*B, 18, YR - 7, YR + 7)
    dx, dz = A[0] - B[0], A[1] - B[1]
    L = math.hypot(dx, dz)
    bar = bd.Pos((A[0] + B[0]) / 2, YR, (A[1] + B[1]) / 2) * bd.Rot(0, math.degrees(math.atan2(dx, dz)), 0) * bd.Box(22, 12, L - 30)
    rod = eyes + bar - _ycyl(*A, 8, YR - 8, YR + 8) - _ycyl(*B, 8, YR - 8, YR + 8)
    studs = _ycyl(*A, 8, YR - 8, YC0) + _ycyl(*B, 8, YR - 8, YC0) + _ycyl(*A, 13, YR - 11, YR - 8) + _ycyl(*B, 13, YR - 11, YR - 8)
    return rod, studs


def about_y(a, z0=0.0):
    """thigh / knee rotation: + = flexion of the knee (calf back), + pitch = thigh back"""
    return bd.Pos(0, 0, z0) * bd.Rot(0, a, 0) * bd.Pos(0, 0, -z0)


if __name__ == "__main__":
    from real import hip
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "STEP", "real")
    th, cr, lv = thigh(), knee_crank(), knee_lever()
    km = actuator()
    psi0, A0, B0 = knee_linkage(0.0)
    rod0, studs0 = knee_rod(A0, B0)
    for n, p in (("thigh_L", th), ("knee_crank_L", cr), ("knee_lever_L", lv), ("knee_rod_L", rod0)):
        print("%s: valid %s, %.0f g" % (n, p.is_valid, p.volume * AL7075))
    print("thigh vs knee actuator %.1f, crank vs it %.1f mm3" % ((th & km).volume, (cr & km).volume))
    hits = []
    # (a) the knee range, thigh fixed: calf-side lever + rod + crank against the thigh and each other
    for k in range(0, 136, 15):
        psi, A, B = knee_linkage(k)
        rod, st = knee_rod(A, B)
        mv = {"crank": about_y(psi - psi0, ZK) * cr, "lever": about_y(k, ZKN) * lv, "rod": rod, "studs": st}
        for a, b in (("crank", "thigh"), ("lever", "thigh"), ("rod", "thigh"), ("studs", "thigh"), ("rod", "lever"),
                     ("rod", "crank"), ("crank", "lever"), ("lever", "knee act"), ("rod", "knee act")):
            o = mv.get(b) or {"thigh": th, "knee act": km}[b]
            v = (mv[a] & o).volume
            if v > 0.5:
                hits.append(("knee %d" % k, a, b, round(v)))
    print("knee sweep:", hits or "clear")
    # (b) the hip-pitch range with the knee at 0 / 60 / 135: the thigh assembly against the hip
    acts = hip.actuators()
    fixed = {"yaw link": hip.yaw_link(), "roll yoke": hip.roll_yoke(), "pitch act": acts["pitch A13720"],
             "yaw act": acts["yaw A8112"], "roll act": acts["roll A10020"]}
    hits = []
    for k in (0, 60, 135):
        psi, A, B = knee_linkage(k)
        rod, st = knee_rod(A, B)
        asm = {"thigh": th, "knee act": km, "crank": about_y(psi - psi0, ZK) * cr, "rod": rod + st, "lever": about_y(k, ZKN) * lv}
        for pt in range(-125, 31, 10):
            for a, o in asm.items():
                for b, f in fixed.items():
                    if a == "thigh" and b == "pitch act":
                        continue
                    v = (about_y(pt) * o & f).volume
                    if v > 0.5:
                        hits.append((pt, k, a, b, round(v)))
    print("pitch sweep:", hits or "clear")
    for p, n in ((th, "thigh_L"), (cr, "knee_crank_L"), (lv, "knee_lever_L"), (rod0, "knee_rod_L")):
        p.label, p.color = n, bd.Color(0.78, 0.80, 0.83)
        bd.export_step(p, os.path.join(out, n + ".step"))
    studs0.label, studs0.color = "knee studs", bd.Color(0.55, 0.56, 0.58)
    km.label, km.color = "knee A13720", bd.Color(0.36, 0.37, 0.40)
    bd.export_step(bd.Compound(children=[th, cr, lv, rod0, studs0, km]), os.path.join(out, "thigh_L_assy.step"))
    hp = [hip.yaw_link(), hip.roll_yoke(), hip.roll_crank(), hip.roll_lever()]
    q0 = hip.roll_linkage(0.0)
    hr, hpins = hip.roll_rod(q0[1], q0[2])
    for p in hp + [hr]:
        p.color = bd.Color(0.78, 0.80, 0.83)
    hpins.color = bd.Color(0.55, 0.56, 0.58)
    for n, a in acts.items():
        a.label, a.color = n, bd.Color(0.36, 0.37, 0.40)
    bd.export_step(bd.Compound(children=[th, cr, lv, rod0, studs0, km] + hp + [hr, hpins] + list(acts.values())),
                   os.path.join(out, "upper_leg_L.step"))
