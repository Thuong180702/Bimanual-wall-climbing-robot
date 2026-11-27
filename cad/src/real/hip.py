"""Hip module of the real robot (left leg; the right is its mirror), option B with 200 mm hip
centres (locked spec 3.1). Leg frame, millimetres: origin at the hip centre where the yaw,
roll and pitch axes cross, x forward, y outward (left), z up. cad/tools/real_legs_b.py is
the clearance model these numbers come from.

Parts so far: the hip-yaw link (7075-T6, CNC). It hangs from the A8112 yaw output, carries
the hip-roll actuator (A10020, axis x, output face to the rear) on a rear wall, and holds
the roll shaft's bearing pair (2 x 7006 angular contact, back to back) on the roll axis.
The roll yoke (7075-T6): a spoked plate on the hip-pitch actuator's rear (inboard) face, a
boom round the back of that actuator and the hollow roll shaft through the bearing pair;
the leg harness runs through the shaft (bore 18) and the pitch actuator's 29 mm centre.
The roll linkage behind the rear wall: a crank (40 mm) on the roll actuator's output, a rod
(140 mm, rod ends on M10 shoulder pins in double shear: crank and lever are forks) and a clamped lever (56 mm) on the roll shaft; ratio
1.40 at mid-range rising to 1.84 / 1.75 at the ends (roll_ratio), crank swing -49..49 deg.
"""
import math
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadgen import build123d as bd
from real.encos import body, SPEC

# ---- the frame (real_legs_b.P, mm)
ZY = 110.0            # yaw output face over the hip centre (clears the pitch actuator's roll sweep)
XW = -150.0           # rear wall front face = roll actuator output face
TW = 10.0             # rear wall thickness
S, ZR = 15.0, 140.0   # roll actuator axis: outboard shift, height
YW0, YW1 = 80.0, 92.0 # outboard side web (12 mm: it carries the roll bearing's couple to the hub)
ZB0, ZB1 = 58.0, 68.0 # bottom plate under the roll actuator
BORE, BOSS_R, BOSS_X = 55.0, 42.0, -115.0   # 7006 bearing seat, boss radius, boss front face
AL7075 = 2.81e-3      # g/mm3


def loc(origin, z_dir, x_dir=None):
    if x_dir is None:
        x_dir = (0, 0, 1) if abs(z_dir[2]) < 0.9 else (1, 0, 0)
    return bd.Location(bd.Plane(origin=origin, x_dir=x_dir, z_dir=z_dir))


def actuators():
    """the module's actuators placed in the leg frame (zero pose)"""
    return {
        "yaw A8112": loc((0, 0, ZY), (0, 0, -1)) * body("A8112"),
        "roll A10020": loc((XW, S, ZR), (-1, 0, 0)) * body("A10020"),
        "pitch A13720": loc((0, SPEC["A13720"]["L"] / 2, 0), (0, 1, 0)) * body("A13720"),
    }


def _yz(sk, x0, t):      # sketch in (y, z) at x = x0, extruded t toward -x
    return bd.extrude(bd.Plane(origin=(x0 - t, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * sk, amount=t)


def _xz(sk, y0, t):      # sketch in (x, z) at y = y0, extruded t toward +y
    return bd.extrude(bd.Plane(origin=(0, y0 + t, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0)) * sk, amount=t)


def _xy(sk, z0, t):      # sketch in (x, y) at z = z0, extruded t toward +z
    return bd.extrude(bd.Plane(origin=(0, 0, z0), x_dir=(1, 0, 0), z_dir=(0, 0, 1)) * sk, amount=t)


def C(u, v, r):
    return bd.Pos(u, v) * bd.Circle(r)


def R(u0, u1, v0, v1):
    return bd.Pos((u0 + u1) / 2, (v0 + v1) / 2) * bd.Rectangle(u1 - u0, v1 - v0)


def hull(*sks):
    return bd.make_hull(sum(sks[1:], sks[0]).edges())


def yaw_link():
    # 1. hub under the yaw output and the arm out to the side web (z 101..110)
    # deep wherever the pitch actuator's roll sweep (a 101 mm sphere round the hip centre) and the roll
    # actuator leave room: 9 mm under the hub, ~50 mm where it meets the web (FEA: the 9 mm plate saw 400 MPa)
    arm = _xy(hull(C(0, 0, 36), R(-64, -34, 60, YW1)), 60, ZY - 60)
    arm -= bd.Sphere(101) + bd.Pos(-110, S, ZR) * bd.Rot(0, 90, 0) * bd.Cylinder(65, 100)
    # 2. outboard side web. Chamfered (2026-11-26, the user's call): its top runs from the hub down to the
    # roll axis (was up to z 200 beside the roll actuator, where a hanging arm's elbow met it); the fins
    # above it are in open air, so the old window over them is gone
    web = _xz(hull(R(-64, -34, ZY - 30, ZY - 4), R(XW - TW + 3, XW + 12, ZB0, ZR)), YW0, YW1 - YW0)
    # 3. rear wall: motor ring + bearing boss outline, joined to the web; its outboard top follows the
    # motor ring (r 68) instead of a square corner up to z 206
    wall = _yz(hull(C(S, ZR, 68), C(0, 0, BOSS_R), R(60, YW1 - 3, ZB0, ZR)), XW, TW)
    # 4. bottom plate under the actuator and the keel down to the bearing boss
    bot = _xy(R(XW - TW + 3, -92, -22, YW1 - 3), ZB0, ZB1 - ZB0)
    keel = _xz(R(XW - TW + 3, BOSS_X - 3, 20, ZB1 - 3), -6, 12)
    boss = bd.Pos((XW + BOSS_X) / 2 - TW / 2, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(BOSS_R, BOSS_X - XW + TW)
    # the pieces overlap by 3 mm so no two rounded faces coincide (a coincident union is invalid)
    web = bd.fillet(web.edges(), 2.0)
    wall, bot, keel = (bd.fillet(p.edges(), 2.0) for p in (wall, bot, keel))   # machined look
    part = arm + web + wall + bot + keel + boss
    # lightening pocket in the rear wall's back face, between the motor's bolt ring and the bearing boss
    part -= _yz(hull(C(S - 10, 62, 16), C(S + 30, 62, 16)), XW - TW + 4, 5)
    # shelled (2026-11-26, FEA at the 260 MPa criterion): the deep arm hollowed from below inside 6 mm
    # walls, the web milled from outside to a 5 mm skin inside 8 mm rims, the bottom plate pocketed
    aol = hull(C(0, 0, 36), R(-64, -34, 60, YW1))
    part -= _xy(bd.offset(aol, -6), 59, ZY - 6 - 59) - bd.Sphere(107) - bd.Pos(-110, S, ZR) * bd.Rot(0, 90, 0) * bd.Cylinder(71, 110)
    wol = hull(R(-64, -34, ZY - 30, ZY - 4), R(XW - TW + 3, XW + 12, ZB0, ZR))
    wpk = bd.offset(wol, -8) - R(-200, 0, 95, 103)
    part -= _xz(wpk, YW0 + 5, YW1 - YW0)
    part -= _xy(R(XW - TW + 30, -100, 12, YW1 - 14), ZB0 - 1, 6)     # clear of the keel and the wall junction
    # holes: yaw output 6 x M4 on 50 (+ the pilot), roll output clearance, its 10 x M4 ring, bearing bore
    part -= bd.Pos(0, 0, ZY - 1) * bd.Cylinder(54 / 2 + 0.05, 2.1)
    for i in range(6):
        v = bd.Vector(25, 0, 0).rotate(bd.Axis.Z, 60 * i)
        part -= bd.Pos(v.X, v.Y, ZY - 5) * bd.Cylinder(2.25, 12)
        part -= bd.Pos(v.X, v.Y, ZY - 9) * bd.Cylinder(4.0, 8)      # counterbore from below (socket head)
    part -= bd.Pos(XW - TW / 2, S, ZR) * bd.Rot(0, 90, 0) * bd.Cylinder(66 / 2, TW + 2)
    for i in range(10):
        v = bd.Vector(0, 116.5 / 2, 0).rotate(bd.Axis.X, 36 * i)
        part -= bd.Pos(XW - TW / 2, S + v.Y, ZR + v.Z) * bd.Rot(0, 90, 0) * bd.Cylinder(2.25, TW + 2)
    part -= bd.Pos((XW + BOSS_X) / 2 - TW / 2, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(BORE / 2, BOSS_X - XW + TW + 2)
    return part


# ---- roll yoke
H13 = SPEC["A13720"]["L"] / 2        # 42.25: the pitch actuator's faces at y = +-H13
YD = 16.0                            # yoke plate thickness, on the rear face (y = -H13 - YD .. -H13): it carries the
                                     # roll + pitch moments from the bolt ring round to the boom (FEA)
RD = 82.0                            # plate radius: bolt circle 155 + head; corner 97 mm from the hip centre
XB1, XB0 = -86.0, BOSS_X + 2.0       # boom between the actuator's barrel (r 85) and the bearing boss
RBOOM = 54.0                         # boom section kept inside r 54 about the roll axis (the yaw link's
                                     # bottom plate sits at z 58 over it as the leg rolls)
SH_D, SH_B, SH_END = 30.0, 18.0, -182.0   # roll shaft: bearing seat, harness bore, rear end (lever seat)


def roll_yoke():
    y0, y1 = -H13 - YD, -H13
    plate = _xz(hull(C(0, 0, RD), R(-100, -40, -30, 30)), y0, YD)             # with a rear lobe out to the boom
    # six spokes: pockets between the hub (r 24) and the bolt ring (r 66)
    for i in range(6):
        a = 30 + 60 * i
        if 90 <= a <= 270:           # keep the web solid toward the boom (rear half), where the load goes
            continue
        c = bd.Vector(45, 0, 0).rotate(bd.Axis.Y, a)
        pocket = bd.Pos(c.X, (y0 + y1) / 2, c.Z) * bd.Rot(90, 0, 0) * bd.Cylinder(14, YD + 2)
        plate -= pocket
    top = bd.Plane(origin=(0, 0, -30), x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    # the boom wraps the back of the pitch actuator's barrel (r 85 + 2): full width from the plate to the
    # shaft instead of a neck beside the barrel (FEA: the neck saw 230-310 MPa)
    boom_sk = bd.Polygon((-50, y0 - 20), (XB0, y0 - 20), (XB0, 22), (-50, 22), align=None)
    boom = bd.extrude(top * boom_sk, amount=60, dir=(0, 0, 1)) & (bd.Pos((XB0 - 66) / 2, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(RBOOM, 66 - XB0 + 2))
    boom -= bd.Pos(0, 0, 0) * bd.Rot(90, 0, 0) * bd.Cylinder(87, 200)
    collar = bd.Pos(XB0 + 2.5, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(19, 7.5)      # presses the front inner race (x -114.5), sunk 5 into the boom
    # the shaft runs 20 mm INTO the boom: butted on its face alone it was a second, loose solid
    shaft = bd.Pos((XB0 + 20 + SH_END) / 2, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(SH_D / 2, XB0 + 20 - SH_END)
    plate = bd.fillet(plate.edges(), 1.5)
    part = plate + boom + collar + shaft
    # bolts: 20 x M4 on 155 through the plate (counterbored), the actuator's centre hole, the harness bore
    for i in range(20):
        v = bd.Vector(77.5, 0, 0).rotate(bd.Axis.Y, 18 * i)
        part -= bd.Pos(v.X, (y0 + y1) / 2, v.Z) * bd.Rot(90, 0, 0) * bd.Cylinder(2.25, YD + 2)
        part -= bd.Pos(v.X, y0 + 2, v.Z) * bd.Rot(90, 0, 0) * bd.Cylinder(3.75, 4.01)
    part -= bd.Pos(0, (y0 + y1) / 2, 0) * bd.Rot(90, 0, 0) * bd.Cylinder(15, YD + 2)
    part -= bd.Pos((XB1 + SH_END) / 2, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(SH_B / 2, XB1 - SH_END + 2)
    # the plate milled from its inboard face (2026-11-26): 9 mm pockets between six spokes, the hub and
    # the bolt ring, a 5 mm web left on the actuator (FEA at the 260 MPa criterion)
    pk = C(0, 0, 66) - C(0, 0, 26)
    for i in range(6):
        a = math.radians(60 * i)
        pk -= bd.Pos(45 * math.cos(a), 45 * math.sin(a)) * bd.Rot(0, 0, 60 * i) * bd.Rectangle(60, 12)
    part -= _xz(pk, y0 - 1, 12)
    return part


# ---- roll linkage (in the y-z plane behind the rear wall)
R1, R2, MID = 40.0, 56.0, 12.5        # crank, lever (mm), roll angle where both point +y (deg)
# the rod ends work in DOUBLE shear: crank and lever are forks, two 6 mm cheeks round the eye
# (a cantilevered M8 pin would see ~2 GPa bending at the ~10 kN rod force)
XA0, XA1 = XW - TW - 1, XW - TW - 7   # front cheeks: x -161 .. -167
XC0, XC1 = XW - TW - 17, XW - TW - 23 # rear cheeks:  x -177 .. -183
XR = XW - TW - 12                     # rod plane x -172, eye 9 thick between the cheeks
PIN = 10.0                            # M10 shoulder pins
O1 = (S, ZR)


def _lever_pin(q):
    a = math.radians(q - MID)
    return (R2 * math.cos(a), R2 * math.sin(a))


ROD = math.hypot(O1[0] + R1 - _lever_pin(MID)[0], O1[1] - _lever_pin(MID)[1])


def roll_linkage(q):
    """roll angle q (deg) -> crank angle (deg, about +x from +y), crank pin A, lever pin B (y, z)"""
    b = _lever_pin(q)
    dy, dz = b[0] - O1[0], b[1] - O1[1]
    d = math.hypot(dy, dz)
    a = (R1 ** 2 - ROD ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(R1 ** 2 - a ** 2)
    uy, uz = dy / d, dz / d
    py, pz = O1[0] + a * uy, O1[1] + a * uz
    A = max([(py - h * uz, pz + h * uy), (py + h * uz, pz - h * uy)], key=lambda t: t[0])
    return math.degrees(math.atan2(A[1] - O1[1], A[0] - O1[0])), A, b


def roll_ratio(q, dq=0.01):
    return (roll_linkage(q + dq)[0] - roll_linkage(q)[0]) / dq


def _bolt_ring(n, pcd, centre, x, length, d, angle0=0.0):
    cut = None
    for i in range(n):
        a = math.radians(angle0 + 360.0 * i / n)
        c = bd.Pos(x, centre[0] + pcd / 2 * math.cos(a), centre[1] + pcd / 2 * math.sin(a)) * bd.Rot(0, 90, 0) * bd.Cylinder(d / 2, length)
        cut = c if cut is None else cut + c
    return cut


def roll_crank():
    """at roll 0: bolted to the roll actuator's output (6 x M5 on 50) through the wall's 66 mm hole"""
    psi, A, _ = roll_linkage(0.0)
    hub = bd.Pos((XW + XA1) / 2, *O1) * bd.Rot(0, 90, 0) * bd.Cylinder(29, XW - XA1)
    neck = bd.Pos((XA1 + XC1) / 2, *O1) * bd.Rot(0, 90, 0) * bd.Cylinder(17, XA1 - XC1)   # joins the cheeks
    cheeks = [_yz(hull(C(*O1, 22), C(*A, 12)), x0, x0 - x1) for x0, x1 in ((XA0, XA1), (XC0, XC1))]
    part = bd.fillet(hub.edges(), 1.0) + neck + bd.fillet(cheeks[0].edges(), 1.5) + bd.fillet(cheeks[1].edges(), 1.5)
    part -= _bolt_ring(6, 50, O1, (XW + XA1) / 2, XW - XA1 + 2, 5.5)
    part -= bd.Pos((XA1 + XC1) / 2, *O1) * bd.Rot(0, 90, 0) * bd.Cylinder(10, XA1 - XC1 + 2)   # driver access
    part -= bd.Pos((XA0 + XC1) / 2, *A) * bd.Rot(0, 90, 0) * bd.Cylinder(PIN / 2, XA0 - XC1 + 2)
    return part


def roll_lever():
    """at roll 0: split clamp on the roll shaft's rear end (in the build: a spline), the arm to the rod"""
    _, _, B = roll_linkage(0.0)
    x0, x1 = XA0, XC1
    hub = bd.Pos((x0 + x1) / 2, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(24, x0 - x1)
    cheeks = [_yz(hull(C(0, 0, 20), C(*B, 12)), a, a - b) for a, b in ((XA0, XA1), (XC0, XC1))]
    part = bd.fillet(hub.edges(), 1.5) + bd.fillet(cheeks[0].edges(), 1.5) + bd.fillet(cheeks[1].edges(), 1.5)
    part -= bd.Pos((x0 + x1) / 2, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(SH_D / 2, x0 - x1 + 2)
    part -= bd.Pos((x0 + x1) / 2, -18, 0) * bd.Box(x0 - x1 + 2, 22, 1.5)       # clamp slit, opposite the arm
    for xc in ((XA0 + XA1) / 2 - 3, (XC0 + XC1) / 2 + 3):
        part -= bd.Pos(xc, -18, 0) * bd.Cylinder(2.75, 60)                     # 2 x M5 clamp screws across the slit
    part -= bd.Pos((XA0 + XC1) / 2, *B) * bd.Rot(0, 90, 0) * bd.Cylinder(PIN / 2, XA0 - XC1 + 2)
    return part


def roll_rod(A, B):
    """two rod-end eyes on the M10 pins and an 8 x 16 bar between them, in the plane x = XR"""
    eyes = [bd.Pos(XR, *P) * bd.Rot(0, 90, 0) * bd.Cylinder(11, 9) for P in (A, B)]
    dy, dz = A[0] - B[0], A[1] - B[1]
    L = math.hypot(dy, dz)
    body = bd.Pos(XR, (A[0] + B[0]) / 2, (A[1] + B[1]) / 2) * bd.Rot(-math.degrees(math.atan2(dy, dz)), 0, 0) * bd.Box(8, 16, L - 20)   # flat bar: fits the 10 mm fork gap
    part = eyes[0] + eyes[1] + body
    pins = [bd.Pos((XA0 + XC1) / 2, *P) * bd.Rot(0, 90, 0) * bd.Cylinder(PIN / 2, XA0 - XC1) +
            bd.Pos(XC1 - 2, *P) * bd.Rot(0, 90, 0) * bd.Cylinder(8, 4) for P in (A, B)]
    for P in (A, B):
        part -= bd.Pos(XR, *P) * bd.Rot(0, 90, 0) * bd.Cylinder(PIN / 2, 12)
    return part, pins[0] + pins[1]


def about_crank(a):
    return bd.Pos(0, *O1) * bd.Rot(a, 0, 0) * bd.Pos(0, -O1[0], -O1[1])


def roll_sweep(link, yoke, others, lo=-20, hi=45, step=5):
    """smallest gap proxy: overlap volume of the yoke (and what it carries) with the yaw side, over the roll range"""
    worst = []
    for a in range(lo, hi + 1, step):
        m = bd.Rot(a, 0, 0)          # +roll = leg out for the left leg = +x rotation
        for n, o in [("yaw link", link)] + others:
            for nm, y in yoke:
                v = (m * y & o).volume
                if v > 0.5:
                    worst.append((a, nm, n, round(v)))
    return worst


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "STEP", "real")
    acts = actuators()
    psi0, A0, B0 = roll_linkage(0.0)
    rod0, pins0 = roll_rod(A0, B0)
    parts = {"hip_yaw_link_L": yaw_link(), "roll_yoke_L": roll_yoke(), "roll_crank_L": roll_crank(),
             "roll_lever_L": roll_lever(), "roll_rod_L": rod0}
    for n, p in parts.items():
        print("%s: valid %s, %.0f g, bbox %s" % (n, p.is_valid, p.volume * AL7075,
              [round(v) for v in (*p.bounding_box().min, *p.bounding_box().max)]))
        for an, a in acts.items():
            v = (p & a).volume
            if v > 0.5:
                print("  OVERLAP with %s: %.1f mm3" % (an, v))
        p.label, p.color = n, bd.Color(0.78, 0.80, 0.83)
        bd.export_step(p, os.path.join(out, n + ".step"))
    print("roll sweep -20..45 deg, yoke + pitch actuator vs yaw link + yaw actuator:",
          roll_sweep(parts["hip_yaw_link_L"], [("yoke", parts["roll_yoke_L"]), ("pitch A13720", acts["pitch A13720"])],
                     [("yaw A8112", acts["yaw A8112"]), ("roll A10020", acts["roll A10020"])]) or "clear")
    print("roll ratio: %s" % ", ".join("%d deg %.2f" % (q, roll_ratio(q)) for q in (-20, 0, 12, 30, 45)))
    # the linkage over the roll range: the moving parts against everything else
    link, yoke, crank, lever = (parts[k] for k in ("hip_yaw_link_L", "roll_yoke_L", "roll_crank_L", "roll_lever_L"))
    hits = []
    for q in range(-20, 46, 5):
        psi, A, B = roll_linkage(q)
        rod, pins = roll_rod(A, B)
        mv = {"crank": about_crank(psi - psi0) * crank, "lever": bd.Rot(q, 0, 0) * lever, "rod": rod, "pins": pins,
              "yoke": bd.Rot(q, 0, 0) * yoke}
        fixed = {"yaw link": link, "roll A10020": acts["roll A10020"], "yaw A8112": acts["yaw A8112"]}
        pairs = [("crank", f) for f in fixed if f != "roll A10020"] + [("lever", f) for f in fixed] + \
                [("rod", f) for f in fixed] + [("crank", "lever"), ("crank", "rod"), ("lever", "rod"), ("rod", "yoke"), ("crank", "yoke")]
        for a, b in pairs:
            v = (mv[a] & (mv.get(b) or fixed.get(b))).volume
            if v > 0.5:
                hits.append((q, a, b, round(v)))
    print("linkage sweep -20..45:", hits or "clear")
    parts["roll_pins_L"] = pins0
    for (n, a) in acts.items():
        a.label, a.color = n, bd.Color(0.36, 0.37, 0.40)
    bd.export_step(bd.Compound(children=list(parts.values()) + list(acts.values())), os.path.join(out, "hip_module_L.step"))
