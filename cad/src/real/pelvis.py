"""Pelvis and waist of the real robot.

Pelvis frame (mm): origin midway between the hip centres, x forward, y left, z up; the hip
centres at y = +-100 (locked spec 3.1), the hip-yaw axes through them.

The pelvis is a 15 mm deck on the two hip-yaw actuators' top faces (A8112: 8 x M4 on 97),
each actuator in a short socket skirt, and carries the waist: an ENCOS A8112 (90 N.m; the
AK60-39 had the torque but its output bearing not the ~120-300 N.m tilt of a ~50 kg upper
body walking / landing) on a pedestal, inside a tube that holds the inner ring of a crossed
roller bearing (RB12016 class: 120 x 150 x 16) whose outer ring the chest carries, so the
tilt goes through the bearing and the actuator only turns the chest. The pedestal is a
40 mm column up to z 190: below that the hip-roll actuators pass within ~45 mm of the middle
when a hip turns 35 deg toe-out. Behind the
hips the deck is cut back to x = -15: the hip-roll actuators ride round the yaw axes there
(r 65..160 from each axis, z up to 202) and at 35 deg toe-out reach 24 mm from the middle.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadgen import build123d as bd
from cadgen import read_step
from real.hip import C, R, hull, AL7075, ZY

HY = 100.0                              # hip centres
ZD0, ZD1 = ZY + 53.0, ZY + 68.0          # deck: on the yaw actuators' top faces (z 163) .. 178
XF, XB = 36.0, -15.0                    # deck front edge, bridge rear edge
ZP1 = 196.0                             # waist pedestal top = the A8112's rear face
ZW = ZP1 + 53.0                         # waist output face (249)
BR = (60.0, 75.0, 228.0, 244.0)         # waist bearing: inner / outer radius, z bottom / top


def waist_actuator(vendor=False):
    """ENCOS A8112, output up, rear face on the pedestal"""
    from real.encos import body
    return bd.Pos(0, 0, ZW) * body("A8112")


def waist_bearing():
    r0, r1, z0, z1 = BR
    return bd.Pos(0, 0, (z0 + z1) / 2) * (bd.Cylinder(r1, z1 - z0) - bd.Cylinder(r0, z1 - z0))


def _ring(n, pcd, x, y, r, z0, z1, a0=0.0):
    cut = None
    for i in range(n):
        a = math.radians(a0 + 360.0 * i / n)
        c = bd.Pos(x + pcd / 2 * math.cos(a), y + pcd / 2 * math.sin(a), (z0 + z1) / 2) * bd.Cylinder(r, z1 - z0)
        cut = c if cut is None else cut + c
    return cut


def pelvis():
    outline = C(0, HY, 60) + C(0, -HY, 60) + R(XB, XF, -HY, HY)
    outline = bd.fillet(outline.vertices(), 10) if False else outline
    deck = bd.extrude(bd.Plane.XY.offset(ZD0) * outline, amount=ZD1 - ZD0)
    try:
        deck = bd.fillet(deck.edges().filter_by(bd.Axis.Z), 8)
    except Exception:                    # noqa: BLE001
        pass
    skirts = None
    for sy in (HY, -HY):                 # socket skirts round the yaw actuators' upper 23 mm
        s = bd.Pos(0, sy, (ZY + 30 + ZD0) / 2) * (bd.Cylinder(59, ZD0 - ZY - 30) - bd.Cylinder(54, ZD0 - ZY - 30))
        skirts = s if skirts is None else skirts + s
    column = bd.Pos(0, 0, (ZD1 - 1 + 190) / 2) * bd.Cylinder(40, 190 - ZD1 + 1)
    flange = bd.Pos(0, 0, (190 + ZP1) / 2) * bd.Cylinder(55, ZP1 - 190)
    tube = bd.Pos(0, 0, (ZP1 - 1 + BR[3]) / 2) * (bd.Cylinder(BR[0], BR[3] - ZP1 + 1) - bd.Cylinder(54.5, BR[3] - ZP1 + 1))
    shoulder = bd.Pos(0, 0, BR[2] - 2) * (bd.Cylinder(BR[0] + 4, 4) - bd.Cylinder(BR[0] - 1, 4))
    # two ribs along the deck, front and rear, make it a box beam between the hips (FEA: the bare 15 mm
    # deck saw 261 MPa under one leg's roll + pitch moments)
    ribs = [bd.Pos(xr, 0, (ZD1 - 1 + 192) / 2) * bd.Box(7, 2 * HY + 40, 192 - ZD1 + 1) for xr in (XB + 3.5, XF - 3.5)]
    part = deck + skirts + column + flange + tube + shoulder + ribs[0] + ribs[1]
    for sy in (HY, -HY):
        part -= _ring(8, 97, 0, sy, 2.25, ZD0 - 1, ZD1 + 1, 22.5)          # yaw actuator, 8 x M4 on 97
        part -= bd.Pos(0, sy, (ZD0 + ZD1) / 2) * bd.Cylinder(16, ZD1 - ZD0 + 2)   # its connectors / harness
    part -= _ring(8, 97, 0, 0, 2.25, 190 - 1, ZP1 + 1, 22.5)                       # waist A8112 rear face, 8 x M4 on 97
    part -= bd.Pos(0, 0, ZP1 - 3) * bd.Cylinder(30, 8)                              # its connectors
    # (2026-11-26 trim, FEA-guided) the bearing tube under the bearing seat: six windows 40 x 20 (r 6), the
    # rear one is the harness exit; the column hollowed to a 12 mm wall, the two ribs kept through it
    for i in range(6):
        t = math.radians(60 * i + 180)
        pl = bd.Plane(origin=(57 * math.cos(t), 57 * math.sin(t), 212), x_dir=(-math.sin(t), math.cos(t), 0), z_dir=(math.cos(t), math.sin(t), 0))
        part -= bd.extrude(pl * bd.RectangleRounded(40, 20, 6), amount=8, both=True)
    part -= bd.Pos(0, 0, (ZD1 + 190) / 2) * bd.Cylinder(28, 190 - ZD1) - ribs[0] - ribs[1]
    # underside pockets between the three actuators (ribs stay on the bolt circles and the edges)
    for sy in (52.0, -52.0):
        part -= bd.Pos(22, sy, ZD0 + 4.5) * bd.Box(30, 22, 11)
    # (2026-11-26, 260 MPa criterion) the deck between the ribs and round the sockets milled to 6 mm
    # from below; the bolt circles, the ribs and the pedestal stay full depth
    for sy in (HY, -HY):                       # over each yaw actuator, inside its bolt circle, from the top
        part -= bd.Pos(0, sy, ZD1 - 4.5) * bd.Cylinder(42, 9)
    for sy in (-62.0, 62.0):                   # the bridge between the ribs, from below
        part -= bd.Pos(10, sy, ZD0 + 4.5) * bd.Box(34, 40, 9)
    return part


def leg_parts(knee=0.0):
    """the left leg's parts by body, in its leg frame at zero pose"""
    from real import hip as H, thigh as T
    acts = H.actuators()
    q = H.roll_linkage(0.0)
    rr, rp = H.roll_rod(q[1], q[2])
    psi0, (psi, A, B) = T.knee_linkage(0.0)[0], T.knee_linkage(knee)
    krod, kst = T.knee_rod(A, B)
    return {"yaw": {"yaw link": H.yaw_link(), "roll act": acts["roll A10020"], "roll crank": H.roll_crank(), "roll rod": rr + rp},
            "roll": {"roll yoke": H.roll_yoke(), "pitch act": acts["pitch A13720"], "roll lever": H.roll_lever()},
            "pitch": {"thigh": T.thigh(), "knee act": T.actuator(), "knee crank": T.about_y(psi - psi0, T.ZK) * T.knee_crank(),
                      "knee rod": krod + kst}}


def place(body, yaw, roll, pitch):
    """left leg at (yaw + toe out, roll + out, pitch + thigh back) into the pelvis frame"""
    m = bd.Pos(0, HY, 0) * bd.Rot(0, 0, yaw)
    if body in ("roll", "pitch"):
        m = m * bd.Rot(roll, 0, 0)
    if body == "pitch":
        m = m * bd.Rot(0, pitch, 0)
    return m


if __name__ == "__main__":
    from real import hip as H
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "STEP", "real")
    pv, wa = pelvis(), waist_actuator()
    yaw_act = {s: bd.Pos(0, sy, 0) * H.actuators()["yaw A8112"] for s, sy in (("L", HY), ("R", -HY))}
    print("pelvis: valid %s, %.0f g" % (pv.is_valid, pv.volume * AL7075))
    for n, a in list(yaw_act.items()) + [("waist", wa)]:
        print("  pelvis vs %s actuator: %.1f mm3" % (n, (pv & a).volume))
    fixed = {"pelvis": pv, "waist act": wa, "waist bearing": waist_bearing()}
    print("  pelvis solids %d; pelvis vs bearing %.1f mm3" % (len(pv.solids()), (pv & waist_bearing()).volume))
    parts = leg_parts()
    bb = {k: {n: p.bounding_box() for n, p in v.items()} for k, v in parts.items()}
    hits = {}
    for yaw in (-45, -25, 0, 20, 35):
        for roll in (-20, 0, 20, 45):
            for pitch in (-125, -100, -70, -40, 0, 30):
                for body, ps in parts.items():
                    if body == "yaw" and (roll, pitch) != (0, 0):
                        continue
                    m = place(body, yaw, roll, pitch)
                    for n, p in ps.items():
                        q = m * p
                        for fn, f in fixed.items():
                            v = q & f
                            vol = getattr(v, "volume", 0.0)
                            if vol > 0.5:
                                k = (n, fn)
                                hits.setdefault(k, []).append((yaw, roll, pitch, round(vol)))
    print("leg vs pelvis sweep (yaw, roll, pitch, mm3):")
    for k, v in hits.items():
        print("  ", k, len(v), v[:6])
    if not hits:
        print("   clear")
    pv.label, pv.color = "pelvis", bd.Color(0.78, 0.80, 0.83)
    bd.export_step(pv, os.path.join(out, "pelvis.step"))
