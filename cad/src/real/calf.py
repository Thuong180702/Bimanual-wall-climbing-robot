"""Calf, ankle drive and foot of the real robot (left leg).

Calf frame (mm): origin on the knee axis, x forward, y outward, z up, at knee 0 the same
axes as the thigh; the ankle (cross of the pitch and roll axes) at z = -340, the sole 60 mm
under it, the foot from 70 behind to 150 ahead of the ankle.

Two CubeMars AKE90-8 high on the calf drive the ankle through two rods (G1 / KAIST): the
upper one's crank outboard, the lower one's inboard; pitch = both rods together, roll =
against each other. Each crank (37 mm) stands square to its rod at neutral and the foot pins sit
26 mm under the ankle axis, so both see ~1.46 there (was 40 mm / 1.35). The actuators sit
35 mm ahead of the leg line: at knee 135 the thigh's knee actuator swings down behind the
calf and passes the upper one with 5 mm to spare.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadgen import build123d as bd
from cadgen import read_step
from real.hip import C, R, hull, _xz, AL7075

AK = (0.0, -340.0)                       # ankle centre (x, z)
SOLE = -400.0
MU, ML = (35.0, -130.0), (35.0, -242.0)  # upper / lower ankle actuator axes (x, z)
YOU, YOL = 30.0, -30.0                   # their output faces (upper outward, lower inward)
PL = 6.0                                 # calf side plates, on the actuators' 82 mm rings
RC = 37.0                                # crank (was 40: 1.35 -> 1.46 at neutral, the user's call 2026-11-26 for the
                                         # 101 kg robot; 22 / 1.46 = 15.1 rad/s against the 15 running needs)
YROD, YFOOT = 49.0, 50.0                 # rod-end planes at the cranks / the foot pins (+- y)
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "cubemars_ake90_8_kv35.step")


def ake90(origin, z_dir):
    s = bd.Pos(0, 0, -11.0) * (bd.Rot(-90, 0, 0) * read_step(VENDOR))      # canonical frame (purchased/ake90_8.py)
    return bd.Location(bd.Plane(origin=origin, x_dir=(1, 0, 0), z_dir=z_dir)) * s


def ake90_env(origin, z_dir):
    """clean solid envelope of the AKE90-8 (the vendor file is not a closed solid, so booleans on
    it are unreliable): output disc 53 (1 mm proud), ring 82 to -10.5, body 107.5 to -43.5,
    driver board 105.5 to -54.5 (estimate, motors.py)"""
    s = (bd.Pos(0, 0, -0.5) * bd.Cylinder(26.5, 1) + bd.Pos(0, 0, -5.75) * bd.Cylinder(41, 9.5) +
         bd.Pos(0, 0, -27) * bd.Cylinder(53.75, 33) + bd.Pos(0, 0, -49) * bd.Cylinder(52.75, 11))
    return bd.Location(bd.Plane(origin=origin, x_dir=(1, 0, 0), z_dir=z_dir)) * s


def actuators(vendor=False):
    f = ake90 if vendor else ake90_env
    return {"ankle U AKE90-8": f((MU[0], YOU, MU[1]), (0, 1, 0)),
            "ankle L AKE90-8": f((ML[0], YOL, ML[1]), (0, -1, 0))}


def _crank_perp(M, B):
    """crank pin on circle (M, RC) where the crank stands square to the rod to B (behind M)"""
    mx, mz = (M[0] + B[0]) / 2, (M[1] + B[1]) / 2
    rr = math.dist(M, B) / 2
    d = math.dist(M, (mx, mz))
    a = (RC ** 2 - rr ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(max(RC ** 2 - a ** 2, 0))
    ex, ez = (mx - M[0]) / d, (mz - M[1]) / d
    px, pz = M[0] + a * ex, M[1] + a * ez
    return min([(px - h * ez, pz + h * ex), (px + h * ez, pz - h * ex)], key=lambda q: q[0])


# foot pins 26 mm UNDER the ankle axis (pins above it swing into the calf plates at +-30 roll),
# placed (searched) where each rod gives 1.35 with its crank square to it at neutral
B_U0, B_L0 = (-62.0, -366.0), (-74.0, -366.0)
A_U0, A_L0 = _crank_perp(MU, B_U0), _crank_perp(ML, B_L0)
PINS = {"U": (MU, YROD, A_U0, (B_U0[0], YFOOT, B_U0[1])), "L": (ML, -YROD, A_L0, (B_L0[0], -YFOOT, B_L0[1]))}
LROD = {k: math.dist((a[0], y, a[1]), b) for k, (m, y, a, b) in PINS.items()}


def foot_pose(p, r):
    """ankle pitch p (deg, + toe down), roll r (deg, + sole turned out) -> the foot's placement"""
    return bd.Pos(AK[0], 0, AK[1]) * bd.Rot(0, p, 0) * bd.Rot(r, 0, 0) * bd.Pos(-AK[0], 0, -AK[1])


def _foot_point(P, p, r):
    v = (foot_pose(p, r) * bd.Location(P)).position
    return (v.X, v.Y, v.Z)


def ankle_cranks(p, r, steps=20):
    """crank angles (deg, about +y, 0 = neutral) and the pins, by continuation from neutral"""
    out = {}
    for k, (M, y, A0, B0) in PINS.items():
        A = A0
        for i in range(1, steps + 1):
            B = _foot_point(B0, p * i / steps, r * i / steps)
            rho = math.sqrt(LROD[k] ** 2 - (B[1] - y) ** 2)
            dx, dz = B[0] - M[0], B[2] - M[1]
            d = math.hypot(dx, dz)
            a = (RC ** 2 - rho ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(max(RC ** 2 - a ** 2, 0))
            ex, ez = dx / d, dz / d
            px, pz = M[0] + a * ex, M[1] + a * ez
            A = min([(px - h * ez, pz + h * ex), (px + h * ez, pz - h * ex)], key=lambda s: math.dist(s, A))
        ang = math.degrees(math.atan2(A[0] - M[0], A[1] - M[1]) - math.atan2(A0[0] - M[0], A0[1] - M[1]))
        out[k] = ((ang + 180) % 360 - 180, (A[0], y, A[1]), B)
    return out


def _ycyl(x, z, r, y0, y1):
    return bd.Pos(x, (y0 + y1) / 2, z) * bd.Rot(90, 0, 0) * bd.Cylinder(r, abs(y1 - y0))


def _ring(n, pcd, x, z, r, y0, y1, a0=0.0):
    cut = None
    for i in range(n):
        a = math.radians(a0 + 360.0 * i / n)
        c = _ycyl(x + pcd / 2 * math.cos(a), z + pcd / 2 * math.sin(a), r, y0, y1)
        cut = c if cut is None else cut + c
    return cut


def calf():
    yo0, yo1 = YOU - 1, YOU - 1 + PL                  # outboard plate on the upper actuator's ring
    yi0, yi1 = YOL + 1 - PL, YOL + 1                  # inboard plate on the lower one's
    hub = _ycyl(0, 0, 30, -24, 40)                    # between the thigh's knee bearings; 10 mm wall on the 40 shaft
                                                      # (was r 40: FEA 22 MPa, 2026-11-26 trim)
    # rear edges kept out of the thigh's knee actuator, which folds down behind the calf at knee 135
    # both plates reach back to x -26 over z -210..-300, where the rear spine joins them (above -200
    # the thigh's knee actuator passes at knee 135)
    out = _xz(hull(C(0, 0, 30), C(*MU, 54)) + hull(C(*MU, 54), C(20, -242, 26), C(-16, -210, 10), C(-12, -282, 10), C(2, -300, 18)) +
              hull(C(2, -300, 18), C(*AK, 24)), yo0, PL)
    # narrow toward the ankle: the sole's front rises into this corner at toe-up + inversion
    inn = _xz(hull(C(20, -170, 20), C(*ML, 57), C(-16, -210, 10), C(-12, -282, 10), C(2, -300, 18)) + hull(C(2, -300, 18), C(*AK, 24)), yi0, PL)
    web = bd.Pos(-23.5, 0, -245) * bd.Box(7, yo0 - yi1 + 4, 70)        # rear spine through both plates, under the knee actuator's reach
    # the knee hub sat on the outboard plate alone (FEA: 299 MPa, 24 mm): a neck block under the hub
    # across the calf's width, and the inboard plate brought up to it in FRONT of the knee (the thigh's
    # inboard cheek sweeps the sector behind / under the knee, but not the front-down one)
    neck = _xz(hull(C(0, 0, 30), C(10, -40, 20)), -24, 40 + 24)                    # within the knee hub's width near the knee
    neck += _xz(hull(C(12, -58, 14), C(18, -62, 13), C(-4, -64, 12)), yi0, yo1 - yi0)   # full width below it (r > 44: clear of the thigh's knee housings)
    inn_up = _xz(hull(C(20, -70, 16), C(28, -150, 22), C(20, -170, 20)), yi0, PL)
    part = bd.fillet(hub.edges(), 2) + bd.fillet(out.edges(), 2) + bd.fillet(inn.edges(), 2) + bd.fillet(web.edges(), 2) \
        + neck + bd.fillet(inn_up.edges(), 2)
    part += _ycyl(*AK, 24, yi0, yi1) + _ycyl(*AK, 24, yo0, yo1)                        # ankle pitch bearing bosses
    # holes: knee shaft 40, actuator outputs 56 + rings 8 x M4 on 73.5, ankle pitch bearings (6002: 32 OD)
    part -= _ycyl(0, 0, 20, -26, 42)
    part -= _ycyl(*MU, 28, yo0 - 1, yo1 + 1) + _ring(8, 73.5, *MU, 2.25, yo0 - 1, yo1 + 1, 22.5)
    part -= _ycyl(*ML, 28, yi0 - 1, yi1 + 1) + _ring(8, 73.5, *ML, 2.25, yi0 - 1, yi1 + 1, 22.5)
    part -= _ycyl(*AK, 16, yi0 - 1, yo1 + 1)
    part -= _xz(hull(C(4, -30, 8), C(12, -55, 6)), yi1 + 8, yo0 - yi1 - 16)        # the neck hollowed between its two walls
    # windows in the plates behind the actuators' rear covers
    part -= _xz(hull(C(-5, -45, 9), C(10, -60, 9)), yo0 - 1, PL + 2)
    # (2026-11-26, 260 MPa criterion) the outboard plate opened over the lower actuator's back
    part -= _xz(hull(C(28, -222, 15), C(48, -262, 15), C(10, -258, 10)), yo0 - 1, PL + 2)
    return part


def crank(side):
    """drawn at neutral; spigot through the plate on the 53 mm output (6 x M5 on 40), arm outside it"""
    M, y, A, _ = PINS[side]
    sg = 1 if side == "U" else -1
    yo = YOU if side == "U" else YOL
    y_sp = (yo, yo + sg * (PL + 1))
    y_arm = (yo + sg * (PL + 1), yo + sg * (PL + 9))
    sp = _ycyl(*M, 26, *sorted(y_sp))
    arm = _xz(hull(C(*M, 24), C(*A, 13)), min(y_arm), 8)
    p = sp + bd.fillet(arm.edges(), 2)
    p -= _ring(6, 40, *M, 2.75, min(y_sp) - 1, max(y_arm) + 1)
    p -= _ycyl(*A, 5, min(y_arm) - 1, max(y_arm) + 1)
    return p


def rod(A, B, shaft_only=False):
    """an M10 rod end on the crank (26 mm eye, 10 wide), a ball joint on the foot's stud (the foot
    turns up to 30 deg about the rod: more than a rod end's ~13 deg swivel), a 10 mm steel rod"""
    v = bd.Vector(*A) - bd.Vector(*B)
    L = v.length
    shaft = bd.Solid.make_cylinder(5, L - 26, bd.Plane(origin=bd.Vector(*B) + v.normalized() * 13, z_dir=v.normalized()))
    if shaft_only:
        return shaft
    return bd.Pos(*A) * bd.Rot(90, 0, 0) * bd.Cylinder(13, 10) + bd.Pos(*B) * bd.Sphere(12) + shaft


def cross():
    """the ankle cross: pitch trunnions (y) in the calf, roll trunnions (x) in the foot"""
    c = bd.Pos(AK[0], 0, AK[1]) * bd.Box(36, 36, 28)
    c += _ycyl(*AK, 8, -YOU - 1 + PL + 1 - PL, YOU - 1 + PL)                          # 16 mm pitch pins
    c += bd.Pos(AK[0], 0, AK[1]) * bd.Rot(0, 90, 0) * bd.Cylinder(8, 76)               # 16 mm roll pins
    return bd.fillet(c.edges().filter_by(bd.Axis.Y), 3) if False else c


def foot():
    """one body: a 14 mm sole (pocketed underneath, ribbed) in a rounded foot outline (~260 x 132), a raised heel that carries
    the two ball-joint cups, the roll-bearing saddle on top; under the ankle the top stays at
    -384 so the calf's ankle bosses (bottom -366) clear it"""
    bu, bl = PINS["U"][3], PINS["L"][3]
    outline = hull(C(bu[0], bu[1], 19), C(bl[0], bl[1], 19), C(-70, 0, 24), C(118, 36, 28), C(118, -36, 28), C(142, 0, 26))
    base = bd.extrude(bd.Plane.XY.offset(SOLE) * outline, amount=14)
    base = bd.fillet(base.edges().group_by(bd.Axis.Z)[-1], 6)                         # rounded top rim
    # underside pockets (a rubber sole goes over them): 4 mm skin, ribs along and across
    pocket = bd.extrude(bd.Plane.XY.offset(SOLE - 1) * bd.offset(outline, -7), amount=11)
    for rib in (bd.Pos(30, 0, SOLE + 4) * bd.Box(300, 8, 12),) + tuple(bd.Pos(x, 0, SOLE + 4) * bd.Box(8, 200, 12) for x in (-40, 20, 80)):
        pocket -= rib
    # an 8 mm frame under the ankle yoke (FEA: the toe load bends it here; its middle saw 22-25 MPa and was
    # pocketed 2026-11-26, the long and cross ribs run through it)
    pocket -= bd.Pos(20, 0, SOLE + 4) * (bd.Box(128, 60, 12) - bd.Box(112, 44, 12))   # its ends are the x -40 / 80 ribs
    base -= pocket
    # a keel on top from the yoke to the toe: the sole works as a T beam under the toe load
    keel = bd.extrude(bd.Plane(origin=(0, 8, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0)) *
                      bd.Polygon((20, SOLE + 10), (130, SOLE + 10), (40, SOLE + 34), (20, SOLE + 34), align=None), amount=16, dir=(0, -1, 0))
    heel = bd.extrude(bd.Plane.XY.offset(SOLE + 10) * hull(C(bu[0], bu[1], 15), C(bl[0], bl[1], 15), C(-36, 0, 24)), amount=16)
    heel = bd.fillet(heel.edges().group_by(bd.Axis.Z)[-1], 6)
    cups = None
    for B in (bu, bl):                                                                  # ball-joint cups, open on top
        c = bd.Pos(B[0], B[1], (SOLE + 20 + B[2]) / 2) * bd.Cylinder(16.5, B[2] - SOLE - 20)
        c = bd.fillet(c.edges().group_by(bd.Axis.Z)[-1], 3)
        cups = c if cups is None else cups + c
    saddle = _ankle_yoke()
    p = base + heel + cups + saddle + keel
    for B in (bu, bl):
        p -= bd.Pos(*B) * bd.Sphere(12.5)                                               # the ball seats
    p -= bd.Pos(AK[0], 0, AK[1]) * bd.Rot(0, 90, 0) * bd.Cylinder(12, 90)              # roll bearings (6001: 28 OD -> 24 seat shown)
    return p


def _ankle_yoke():
    """one U-shaped yoke for the roll bearings: its y-z section is a teardrop (20 mm round over
    a 26 mm neck), so rolling +-30 deg about x it stays clear of the calf's ankle bosses; the cradle
    between the cheeks leaves the cross its +-30 deg"""
    sk = hull(C(0, AK[1], 20), C(0, AK[1] - 32, 13)) + R(-13, 13, SOLE + 10, AK[1] - 32)   # narrow down to the sole: a flare there rolls up into the bosses
    blk = bd.extrude(bd.Plane(origin=(-32, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * sk, amount=64)
    cradle = bd.Pos(AK[0], 0, AK[1] + 20) * bd.Box(42, 60, 40) + bd.Pos(AK[0], 0, AK[1]) * bd.Rot(0, 90, 0) * bd.Cylinder(24, 42)
    y = blk - cradle
    try:
        return bd.fillet(y.edges(), 2.5)
    except Exception:                      # noqa: BLE001 -- keep the sharp yoke rather than fail
        return y


if __name__ == "__main__":
    from real import thigh as T
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "STEP", "real")
    cf, cU, cL, cx, ft = calf(), crank("U"), crank("L"), cross(), foot()
    acts = actuators()
    for n, p in (("calf_L", cf), ("ankle_crank_U_L", cU), ("ankle_crank_L_L", cL), ("ankle_cross_L", cx), ("foot_L", ft)):
        print("%s: valid %s, %.0f g" % (n, p.is_valid, p.volume * AL7075))
    for n, a in acts.items():
        print("  calf vs %s: %.1f mm3" % (n, (cf & a).volume))
    print("rods %.1f / %.1f mm; pins U %s %s, L %s %s" % (LROD["U"], LROD["L"], [round(v, 1) for v in A_U0], [round(v, 1) for v in B_U0],
          [round(v, 1) for v in A_L0], [round(v, 1) for v in B_L0]))
    # pitch ratio per rod and the crank swings over the ankle range
    for p_, r_ in ((0, 0), (40, 0), (-50, 0), (0, 30), (0, -30), (40, 30), (-50, -30)):
        cr = ankle_cranks(p_, r_)
        print("  ankle pitch %4d roll %4d: crank U %6.1f, L %6.1f deg" % (p_, r_, cr["U"][0], cr["L"][0]))
    d = ankle_cranks(0.5, 0)
    print("  pitch ratio at neutral: U %.2f  L %.2f" % (d["U"][0] / 0.5, d["L"][0] / 0.5))
    hits = []
    fixed = {"calf": cf, **acts}
    for p_ in range(-50, 41, 10):
        for r_ in range(-30, 31, 10):
            cr = ankle_cranks(p_, r_)
            mv = {"foot": foot_pose(p_, r_) * ft, "rod U": rod(cr["U"][1], cr["U"][2]), "rod L": rod(cr["L"][1], cr["L"][2]),
                  "crank U": bd.Pos(MU[0], 0, MU[1]) * bd.Rot(0, cr["U"][0], 0) * bd.Pos(-MU[0], 0, -MU[1]) * cU,
                  "crank L": bd.Pos(ML[0], 0, ML[1]) * bd.Rot(0, cr["L"][0], 0) * bd.Pos(-ML[0], 0, -ML[1]) * cL}
            for a, o in mv.items():
                for b, f in fixed.items():
                    if (a, b) in (("crank U", "ankle U AKE90-8"), ("crank L", "ankle L AKE90-8")):
                        continue
                    v = getattr(o & f, "volume", 0.0)
                    if v > 0.5:
                        hits.append((p_, r_, a, b, round(v)))
            sh = {"rod U": rod(cr["U"][1], cr["U"][2], True), "rod L": rod(cr["L"][1], cr["L"][2], True)}
            for a, b in (("rod U", "foot"), ("rod L", "foot"), ("rod U", "crank L"), ("rod L", "crank U")):
                v = getattr(sh[a] & mv[b], "volume", 0.0)   # the rod's shaft (its ends sit on the studs by design)
                if v > 0.5:
                    hits.append((p_, r_, a + " shaft", b, round(v)))
    print("ankle sweep:", hits[:20] or "clear", len(hits))
    # the knee range: the thigh (with its knee actuator and linkage) against the calf assembly
    th, kcr, klv = T.thigh(), T.knee_crank(), T.knee_lever()
    km = T.actuator()
    psi0 = T.knee_linkage(0.0)[0]
    hits = []
    for k in range(0, 136, 5):
        psi, A, B = T.knee_linkage(k)
        krod, kst = T.knee_rod(A, B)
        to_calf = bd.Rot(0, -k, 0) * bd.Pos(0, 0, 340)            # thigh frame -> calf frame at knee k
        thigh_side = {"thigh": th, "knee act": km, "knee crank": T.about_y(psi - psi0, T.ZK) * kcr, "knee rod": krod + kst}
        calf_side = {"calf": cf, **acts, "crank U": cU, "crank L": cL, "knee lever": bd.Pos(0, 0, 340) * T.knee_lever()}
        for a, o in thigh_side.items():
            for b, f in calf_side.items():
                if (a, b) == ("knee rod", "knee lever"):
                    continue
                v = ((to_calf * o) & f).volume
                if v > 0.5:
                    hits.append((k, a, b, round(v)))
    print("knee sweep:", hits[:20] or "clear", len(hits))
    for p, n in ((cf, "calf_L"), (cU, "ankle_crank_U_L"), (cL, "ankle_crank_L_L"), (cx, "ankle_cross_L"), (ft, "foot_L")):
        p.label, p.color = n, bd.Color(0.78, 0.80, 0.83)
        bd.export_step(p, os.path.join(out, n + ".step"))
    cr = ankle_cranks(0, 0)
    rods = rod(cr["U"][1], cr["U"][2]) + rod(cr["L"][1], cr["L"][2])
    rods.label, rods.color = "ankle rods", bd.Color(0.55, 0.56, 0.58)
    vend = actuators(vendor=True)
    for n, a in vend.items():
        a.label, a.color = n, bd.Color(0.36, 0.37, 0.40)
    bd.export_step(bd.Compound(children=[cf, cU, cL, cx, ft, rods] + list(vend.values())), os.path.join(out, "calf_L_assy.step"))
