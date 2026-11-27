"""Torso (chest) of the real robot, on the waist.

Pelvis frame (mm, pelvis.py): origin midway between the hip centres, x forward, y left, z up.
The waist A8112's output face at z 249; the shoulder-flexion axes (y) at z 436, their outputs
at y = +-89 (two ENCOS A10020 whose bodies fill the top of the chest, as v2's SH_Y).

The chest frame is one 7075 part: two side plates carrying the shoulder actuators by their
output-side rings (10 x M4 on 116.5), a bottom plate on the waist flange with the skirt that
holds the waist bearing's outer ring, a top plate over the shoulder actuators, a front plate
carrying two lidars, a full-height back plate that docks the battery. Inside, on the bottom
plate: Jetson Orin NX + carrier, the MTi-630 IMU on the centre line, the CAN board, the DC-DC.

Battery (user ruling 2026-11-26, was 13S6P 30 Ah 6.1 kg): a backpack, 13S4P 21700 high-drain
5 Ah cells (20 Ah, ~0.96 kWh, ~4.1 kg with case and BMS), 8 x 7 cells standing fore-aft, on the
back plate's rails, out in one move.

Lidars (cad/tools/lidar_airy_study.py): three RoboSense Airy (hemispherical 360 x 90 deg):
two on the chest front, the upper one tilted 20 deg up (the rung under the hands), the
lower 30 deg down (the footholds), their optical centres >= 10 mm proud of the front face
(sunk flush, the hand rung's view grazes the chest: 62 % -> 31 %); one on the head pedestal at
the back of the top plate, 20 deg up toward the back (the rungs above, over the raised arms).

Head pedestal (7075, head_mount): a flange bolted to the top plate (4 x M5), a hollow neck lofted
up to the tilted lidar disc (4 x M3 into the Airy's base); the lidar's cable runs down inside
the neck into the chest. The neck stays in front of the lidar's base plane: the Airy sees
nothing behind that plane, so the pedestal hides nothing.

Guard cage (cage): one bent 6061 tube (16 x 1.5) per side, as v2's cage -- in front of the chest
a hoop whose post stands 12 mm ahead of the front lidars' foremost point, from under the
lower one's bottom to over the upper one's top, so a wall or a rung meets the tube first; it
runs back over the top plate (saddle clamps), jogs in to y = +-50 (v2: the raised upper arms
pass the head there) and arches over the head lidar, landing on the top plate in front of
the backpack (the pack lifts 30 mm off its hooks and comes away backward, under the arch).
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadgen import build123d as bd
from real.encos import body
from real.hip import C, R, hull, _xz, AL7075
from real.pelvis import ZW, BR

ZS, SY = 436.0, 89.0            # shoulder-flexion axis height, output faces at +-SY
PT = 6.0                        # side plate thickness (y 89..95); FEA hanging case: ample
XD = 95.0                       # chest half depth (front / back plate outer faces)
ZB0, ZB1 = ZW + 3.0, ZW + 11.0  # bottom plate (252 .. 260)
ZT0 = ZS + 64.0                 # top plate underside (the shoulder actuators reach 498)
ZBOX = 400.0                    # the chest is a box up to here, then narrows to the shoulder rings
PACK = (86.0, 184.0, 163.0)     # backpack: 13S4P (52 of 8 x 7 x 21700 standing fore-aft) + case + BMS
PACK_M = 4.1
PACK_NAME = "battery 13S4P 20Ah (backpack)"
PACK_Z0 = ZB0 + 8.0
AIRY = (30.0, 63.0, 0.24)       # radius, height, kg
# lidar mounts: base centre (x, z), axis elevation (deg), azimuth 0 = forward / 180 = back
LIDARS = {"lidar front-up": ((107.0, 360.0), 20.0, 0.0),
          "lidar front-down": ((112.0, 285.0), -30.0, 0.0),
          "lidar back-up": ((-70.0, 600.0), 20.0, 180.0)}


HX = -60.0                      # head pedestal centre on the top plate (x); its flange x -95..-25
TUBE = (8.0, 1.5)               # guard tube radius, wall
# guard: one tube, left front foot -> left front hoop -> over the top plate -> head hoop -> right
# side, mirrored. Bends r 20. Posts at y +-88 (at +-95 the arm turned 12 deg inward met them): on what the lidars are for
# (lidar_airy_study targets) they cost the footholds 2 points and nothing else (start rung 62 %);
# at y +-75 the start rung lost 8. The head hoop stands at x -20, in front of the head lidar's
# base plane and behind the front ones': none of the three sees it.
GY = 88.0                       # the front hoops' plane: the arm (rotation actuator) turned 12 deg inward passes 5 mm outside
_HALF = [(103.0, GY, 300.0), (194.0, GY, 266.0), (194.0, GY, 436.0), (112.0, GY, 516.0),
         (40.0, GY, 516.0), (-20.0, 50.0, 516.0), (-20.0, 50.0, 690.0)]
CAGE = _HALF + [(x, -y, z) for x, y, z in _HALF[::-1]]


def _axis(el, az):
    e, a = math.radians(el), math.radians(az)
    return (math.cos(e) * math.cos(a), 0.0, math.sin(e)) if az in (0.0, 180.0) else None


def shoulder_actuators():
    return {s: bd.Location(bd.Plane(origin=(0, sg * SY, ZS), x_dir=(1, 0, 0), z_dir=(0, sg, 0))) * body("A10020")
            for s, sg in (("L", 1), ("R", -1))}


def lidars():
    """Airy bodies: base disc on the mount, the dome along the axis"""
    out = {}
    for n, ((x, z), el, az) in LIDARS.items():
        ax = _axis(el, az)
        out[n] = bd.Solid.make_cylinder(AIRY[0], AIRY[1], bd.Plane(origin=(x, 0, z), z_dir=ax))
    return out


def bay():
    """boards on the bottom plate and the backpack: name -> (centre, size, kg)"""
    z = ZB1
    return {"Jetson Orin NX + carrier": ((-30.0, -45.0, z + 20), (100.0, 80.0, 40.0), 0.35),
            "IMU MTi-630": ((0.0, 25.0, z + 12), (57.0, 42.0, 24.0), 0.06),
            "CAN board": ((50.0, -45.0, z + 8), (40.0, 60.0, 15.0), 0.10),
            "DC-DC 48 V": ((58.0, 50.0, z + 7), (58.4, 36.8, 12.7), 0.44),
            PACK_NAME: ((-XD - 4 - PACK[0] / 2, 0.0, PACK_Z0 + PACK[2] / 2), PACK, PACK_M)}


def _ring_y(n, pcd, x, z, r, y0, y1):
    cut = None
    for i in range(n):
        a = 2 * math.pi * i / n
        c = bd.Pos(x + pcd / 2 * math.cos(a), (y0 + y1) / 2, z + pcd / 2 * math.sin(a)) * bd.Rot(90, 0, 0) * bd.Cylinder(r, abs(y1 - y0))
        cut = c if cut is None else cut + c
    return cut


def _wedge(name):
    """the mount block for one front lidar: from the front plate to the tilted base disc"""
    (x, z), el, _ = LIDARS[name]
    e = math.radians(el)
    tx, tz = -math.sin(e), math.cos(e)                    # in-plane direction of the base disc (x-z)
    p1, p2 = (x + 32 * tx, z + 32 * tz), (x - 32 * tx, z - 32 * tz)
    pts = [(XD - 3, min(p1[1], p2[1]) - 4), p1 if p1[0] > p2[0] else p2, p2 if p1[0] > p2[0] else p1, (XD - 3, max(p1[1], p2[1]) + 4)]
    pts = sorted(pts, key=lambda q: math.atan2(q[1] - z, q[0] - (XD - 3 + x) / 2))
    sk = bd.Polygon(*pts, align=None)
    w = bd.extrude(bd.Plane(origin=(0, 33, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0)) * sk, amount=66)
    # hollow (2026-11-26 trim, FEA 52 MPa on the whole chest): 4 mm walls, open into the chest through
    # the front plate (the lidar's cable and its 4 x M3 on 44, driven from inside)
    zs = [q[1] for q in pts if q[0] == XD - 3]
    lo, hi = min(zs), max(zs)
    cav = hull(bd.offset(sk, -4), R(XD - 8, XD + 2, lo + 4, hi - 4))
    cav = bd.extrude(bd.Plane(origin=(0, 29, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0)) * cav, amount=58)
    ax = bd.Vector(math.cos(e), 0, math.sin(e))
    o, u, v = bd.Vector(x, 0, z), bd.Vector(tx, 0, tz), bd.Vector(0, 1, 0)
    for i in range(4):
        a = math.radians(90 * i + 45)
        cav += bd.Solid.make_cylinder(1.6, 12, bd.Plane(origin=o - 8 * ax + 22 * math.cos(a) * u + 22 * math.sin(a) * v, z_dir=ax))
    return w, cav


def chest():
    iy = SY + 3                                                           # cross members run 3 mm into the side plates
    pieces = []
    for sg in (1, -1):
        y0 = SY if sg > 0 else -SY - PT
        side = _xz(hull(C(0, ZS, 70), R(-XD, XD, ZB0, ZBOX)), y0, PT)
        side -= _xz(hull(C(-58, 300, 30), C(58, 300, 30)), y0 - 1, PT + 2)     # windows (air, cables, mass)
        # two upper windows under the shoulder ring: they follow the ring at r 72 (11 mm of metal outside
        # the M4 holes), a 28 mm web down the middle, 12 mm over the lower window, corners r 6
        for sx in (1, -1):
            win = R(sx * 14, sx * 86, 342, 390) - C(0, ZS, 72)
            side -= _xz(bd.fillet(win.vertices(), 6), y0 - 1, PT + 2)
        pieces.append(bd.fillet(side.edges(), 2))
    flange = bd.Pos(0, 0, (ZW + ZB1) / 2) * bd.Cylinder(36, ZB1 - ZW)
    skirt = bd.Pos(0, 0, (BR[2] + ZB1) / 2) * (bd.Cylinder(BR[1] + 8, ZB1 - BR[2]) - bd.Cylinder(BR[1], ZB1 - BR[2]))
    bottom = bd.Pos(0, 0, ZB0 + 3) * bd.Box(2 * XD, 2 * iy, 6)                    # 6 mm (was 8)
    top = bd.Pos(-12.5, 0, ZT0 + 5.5) * bd.Box(2 * XD - 25, 2 * iy, 5)          # 5 mm (was 8; FEA 52 MPa)
    front = bd.Pos(XD - 2, 0, (ZB0 + ZBOX) / 2) * bd.Box(4, 2 * iy, ZBOX - ZB0)
    back = bd.Pos(-XD + 2, 0, (ZB0 + ZT0 + 8) / 2) * bd.Box(4, 2 * iy, ZT0 + 8 - ZB0)
    rails = [bd.Pos(-XD - 1, sy, (PACK_Z0 + ZT0) / 2) * bd.Box(4, 10, ZT0 - PACK_Z0) for sy in (-70.0, 70.0)]
    part = pieces[0] + pieces[1] + flange + skirt + bd.fillet(bottom.edges().filter_by(bd.Axis.Z), 6) + top + front + back \
        + rails[0] + rails[1]
    wedges = [_wedge(n) for n in ("lidar front-up", "lidar front-down")]
    for w, _ in wedges:
        part += w
    for _, cav in wedges:
        part -= cav
    # holes: shoulder outputs 66 + their 10 x M4 rings; waist output 6 x M4 on 50 (+ 54 pilot)
    for sg in (1, -1):
        y0, y1 = (SY - 1, SY + PT + 1) if sg > 0 else (-SY - PT - 1, -SY + 1)
        part -= bd.Pos(0, (y0 + y1) / 2, ZS) * bd.Rot(90, 0, 0) * bd.Cylinder(33, PT + 2)
        part -= _ring_y(10, 116.5, 0, ZS, 2.25, y0, y1)
    for i in range(6):
        a = math.radians(60 * i)
        part -= bd.Pos(25 * math.cos(a), 25 * math.sin(a), (ZW + ZB1) / 2) * bd.Cylinder(2.25, ZB1 - ZW + 2)
    part -= bd.Pos(0, 0, ZW + 0.5) * bd.Cylinder(27.1, 1.2)
    # the pack's connector window in the back plate, vents in the front, harness passes in the bottom
    part -= bd.Pos(-XD + 3, 0, PACK_Z0 + 40) * bd.Box(10, 60, 30)
    part -= bd.Pos(XD - 3, 0, 320) * bd.Box(10, 120, 18)
    part -= bd.Pos(-60, 0, (ZB0 + ZB1) / 2) * bd.Box(24, 60, 12)
    part -= bd.Pos(HX, 0, ZT0 + 4) * bd.Cylinder(12, 10)               # the head lidar's cable into the chest
    for sx in (-28.0, 28.0):                                          # the head pedestal's 4 x M5
        for sy in (-24.0, 24.0):
            part -= bd.Pos(HX + sx, sy, ZT0 + 4) * bd.Cylinder(2.75, 10)
    # lightening: two windows in the back plate behind the pack, the top plate opened in front of the head
    for sy in (-42.0, 42.0):
        part -= bd.Pos(-XD + 3, sy, 385) * bd.Box(10, 56, 170)
    part -= bd.Pos(20, 0, ZT0 + 4) * bd.Box(90, 130, 10)
    for sy in (-56.0, 56.0):                       # the front plate beside the lidar blocks (clear of the guard's feet)
        part -= bd.Pos(XD - 2, sy, 330) * bd.Box(10, 32, 124)
    for sx in (-60.0, 60.0):                       # the bottom plate: four windows outside the waist skirt (FEA: 35 MPa)
        for sy in (-58.0, 58.0):
            part -= bd.Pos(sx, sy, (ZB0 + ZB1) / 2) * bd.Box(40, 40, 12)
    return part


def head_mount():
    (hx, hz), el, az = LIDARS["lidar back-up"]
    ax = bd.Vector(*_axis(el, az))
    o = bd.Vector(hx, 0, hz) - 8 * ax                                  # the disc's back face
    top = ZT0 + 8
    flange = bd.Pos(HX, 0, top) * bd.extrude(bd.RectangleRounded(70, 64, 10), amount=6)
    pl = bd.Plane(origin=o, x_dir=(0, 1, 0), z_dir=ax)
    neck = bd.loft([bd.Plane.XY.offset(top + 5) * bd.Pos(HX, 0) * bd.RectangleRounded(54, 50, 12), pl * bd.Circle(30)])
    disc = bd.Solid.make_cylinder(33, 8, bd.Plane(origin=o, z_dir=ax))
    part = flange + neck + disc
    # hollow: the cable channel from the lidar's base down into the chest
    core = bd.loft([bd.Plane.XY.offset(top - 1) * bd.Pos(HX, 0) * bd.RectangleRounded(40, 36, 8), (pl.offset(-3)) * bd.Circle(18)])
    part -= core + bd.Solid.make_cylinder(9, 12, bd.Plane(origin=o - 2 * ax, z_dir=ax))
    for sx in (-28.0, 28.0):
        for sy in (-24.0, 24.0):
            part -= bd.Pos(HX + sx, sy, top + 3) * bd.Cylinder(2.75, 8)
    for i in range(4):                                                 # 4 x M3 on 44 into the Airy's base
        a = math.radians(90 * i + 45)
        part -= bd.Solid.make_cylinder(1.6, 12, bd.Plane(origin=o - 2 * ax + pl.x_dir * 22 * math.cos(a) + pl.y_dir * 22 * math.sin(a), z_dir=ax))
    return part


def _tube(pts, r, wall=None, bend=20.0):
    """a bent tube along pts (wall None: the solid rod, to cut clamps with)"""
    path = bd.FilletPolyline(*pts, radius=bend)
    d = bd.Vector(*pts[1]) - bd.Vector(*pts[0])
    sec = bd.Circle(r) if wall is None else bd.Circle(r) - bd.Circle(r - wall)
    return bd.sweep(bd.Plane(origin=pts[0], z_dir=d) * sec, path=path, transition=bd.Transition.ROUND)


def cage():
    """the guard tube and its clamps (6061), as {name: solid}"""
    rod = _tube(CAGE, TUBE[0])
    top = ZT0 + 8
    blocks = None
    for sg in (1, -1):
        for b in (bd.Pos(XD + 4, sg * (GY - 3), 300) * bd.Box(8, 20, 26),            # foot on the front plate
                  bd.Pos(75, sg * GY, top + 11) * bd.Box(18, 22, 22),            # saddle on the top plate
                  bd.Pos(-20, sg * 50, top + 16) * bd.Box(26, 26, 32)):          # the head hoop's foot
            blocks = b if blocks is None else blocks + b
    return {"guard tube": _tube(CAGE, *TUBE), "guard clamps": blocks - rod}


def assembly(waist=0.0):
    """chest frame, shoulder actuators, bay, backpack and lidars, coloured, turned by the waist angle"""
    AL, ACT = bd.Color(0.78, 0.80, 0.83), bd.Color(0.36, 0.37, 0.40)
    COL = {"battery": bd.Color(0.20, 0.45, 0.75), "Jetson": bd.Color(0.15, 0.50, 0.30), "IMU": bd.Color(0.90, 0.55, 0.15),
           "CAN": bd.Color(0.15, 0.50, 0.30), "DC-DC": bd.Color(0.15, 0.50, 0.30)}
    out = []
    ch = chest(); ch.color = AL; out.append(ch)
    for a in shoulder_actuators().values():
        a.color = ACT; out.append(a)
    for n, (c, sz, _) in bay().items():
        b = bd.Pos(*c) * bd.Box(*sz)
        b.color = next(v for k, v in COL.items() if n.startswith(k)); b.label = n
        out.append(b)
    for n, l in lidars().items():
        l.color = bd.Color(0.12, 0.12, 0.14); l.label = n
        out.append(l)
    hm = head_mount(); hm.color = AL; hm.label = "head pedestal"; out.append(hm)
    for n, c in cage().items():
        c.color = bd.Color(0.20, 0.20, 0.22) if "tube" in n else bd.Color(0.55, 0.56, 0.58); c.label = n
        out.append(c)
    r = bd.Rot(0, 0, waist)
    res = []
    for o in out:
        q = r * o; q.color = o.color; res.append(q)
    return res


if __name__ == "__main__":
    from real import pelvis as P
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "STEP", "real")
    ch = chest()
    print("chest: valid %s, solids %d, %.0f g" % (ch.is_valid, len(ch.solids()), ch.volume * AL7075))
    things = {**{"shoulder " + k: v for k, v in shoulder_actuators().items()},
              **{n: bd.Pos(*c) * bd.Box(*s) for n, (c, s, _) in bay().items()}, **lidars(),
              "waist act": P.waist_actuator(), "waist bearing": P.waist_bearing()}
    for n, a in things.items():
        v = getattr(ch & a, "volume", 0.0)
        if v > 0.5:
            print("  OVERLAP chest / %s %.0f mm3" % (n, v))
    names = list(things)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = getattr(things[a] & things[b], "volume", 0.0)
            if v > 0.5:
                print("  OVERLAP %s / %s %.0f mm3" % (a, b, v))
    # the waist turning +-45 with everything on the chest, against the pelvis and both legs at zero pose
    from real.pelvis import leg_parts, place
    legs = [place(b, 0, 0, 0) * p for b, ps in leg_parts().items() for p in ps.values()]
    legs += [bd.mirror(q, bd.Plane.XZ) for q in legs]
    fixed = [P.pelvis()] + legs
    moving = [ch, head_mount()] + list(cage().values()) + list(lidars().values()) + [bd.Pos(*bay()[PACK_NAME][0]) * bd.Box(*PACK)]
    hits = []
    for yw in (-45, -20, 0, 20, 45):
        for mv in moving:
            c = bd.Rot(0, 0, yw) * mv
            for f in fixed:
                v = getattr(c & f, "volume", 0.0)
                if v > 0.5:
                    hits.append((yw, round(v)))
    print("waist sweep:", hits or "clear")
    m_frame = ch.volume * AL7075 / 1000
    m_bay = sum(m for _, _, m in bay().values())
    m_head = head_mount().volume * AL7075 / 1000
    m_cage = sum(v.volume for v in cage().values()) * 2.70e-3 / 1000
    print("torso: frame %.2f + head pedestal %.2f + guard %.2f + bay/pack %.2f + 3 Airy %.2f = %.2f kg (+ 2 shoulder actuators 4.36 kg)"
          % (m_frame, m_head, m_cage, m_bay, 3 * AIRY[2], m_frame + m_head + m_cage + m_bay + 3 * AIRY[2]))
    ch.label, ch.color = "chest", bd.Color(0.78, 0.80, 0.83)
    bd.export_step(ch, os.path.join(out, "chest.step"))
