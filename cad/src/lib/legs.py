"""The full robot's legs (spec.ROBOT == "full"), one connector per link, drawn by the
same rules as the arms: each plate exactly its motor's face with a hole for every
bolt, webs following both rims, nothing wrapped round a motor. The right leg is the
left one mirrored (build.mirrored). Importing this registers the builders.
"""
import math

from .build import (BUILDERS, CHILD_OF, FACE, PJOINT, _pt, _span_holes, clear_bolts, grow, hang,
                    mirrored, mount_plane, seat, side_corner, side_hang)
from cadgen import build123d as _bd

from .connect import _disc, cheek, face_od, face_plate, face_t, mate_bore_r, output_plate, plane_at
from .spec import (ANKLE_DRIVE, ANKLE_IN, ANKLE_RODS, CALF, CALF_T, CLEAR, HIP_Y, IFACE, MM, MOTOR_OF,
                   PLATE, SOLE, TORSO_LOW)
from . import spec as _spec

THIGH_T = 0.010     # the thigh's one plate, and its knee face plate


def hip_L():
    """hip pitch (RS04) output -> hip roll housing: the shoulder's L bracket
    (build.side_corner) one floor down, on the roll's FRONT ring now it is an AKE90-8
    (no rear flange: the driver board is there), as the full robot's shoulder on its X12.
    The RS04's output sits 5.5 mm inside the pelvis wall's bore, so its output plate is
    9 mm thick, through the bore, and the web starts 1.5 mm clear of the wall's outer
    face; the roll body points back."""
    # the pelvis wall's outer face, or the ring of an X12's flange standing past it
    k = _spec.HIP_KEY
    ring = TORSO_LOW[1] / 2 + face_t(k, "front") - PLATE
    return side_corner("hip_L", face=FACE["hip_L"], plate_t=max(PLATE, ring - HIP_Y + 0.0035),
                       web_from=ring - HIP_Y + CLEAR)


def hip_roll_L():
    """hip roll output -> hip yaw (AK60-39) housing: the upper arm's link one floor down,
    the yaw actuator hanging under the roll axis on its own axis, body up at the roll
    actuator, output down to the thigh. On an AKE90-8 (no rear flange) it is the full
    robot's upper arm (build.side_hang: down behind the yaw body, by its front ring)."""
    if not IFACE[MOTOR_OF["hip_L_roll"]]["rear_n"]:
        return side_hang("hip_roll_L")
    return hang("hip_roll_L", stand=0.009, plate_t=0.013)


# The knee's belt drive (spec.KNEE_BELT): a bearing boss at the knee in the thigh plate
# (two 6806 bearings, 30 x 42 x 7), a Ø30 hollow shaft through it clamped by both calf
# plates, the 45-tooth pulley on the calf's inboard plate, the 30-tooth one on the RS04.
BRG = (0.030, 0.042, 0.007)                 # 6806: bore, OD, width
BOSS_K = 0.028                              # the boss round them
KNEE_SHAFT = (0.010, 0.015)                 # the knee shaft, hollow: inside, outside
FLANGE = 0.0015                             # each pulley's flange, and its 2.5 mm over the belt
SLOT = 0.002                                # the motor's bolt slots run this far each way


def thigh_belt_L():
    """hip yaw (AK60-39) output -> the knee RS04 (FRONT ring) 112 mm up the thigh, and
    the knee's bearing boss below it: the same 10 mm plate down the leg's centre as the
    direct-drive thigh, carried on down past the motor to the knee. The motor's twelve
    M3 run in 2 mm slots along the thigh, so it slides to tension the belt; a lug over
    it holds the M5 jack screw that pushes it."""
    name = "thigh_L"
    pj = PJOINT[name]
    p_key = MOTOR_OF[pj[0]]
    dm = [d for d in ANKLE_DRIVE if d[2] == name][0]
    m_key, (_, _, zm) = dm[1], dm[3]
    zk = CHILD_OF[name][0][4][2]
    pl_p = mount_plane(pj, (0.0, 0.0, 0.0))
    pl_m = plane_at(dm[3], tuple(-v for v in dm[4]))
    a_od, b_od = IFACE[p_key]["out_od"] * MM, face_od(m_key, "front")
    top = pl_p * output_plate(p_key, a_od)
    side = pl_m * face_plate(m_key, "front", b_od, THIGH_T)
    bb = side.bounding_box()
    y0, y1 = bb.min.Y, bb.max.Y
    a = _pt(pl_p, IFACE[p_key]["out_z"] * MM + PLATE)
    web = cheek(math.sqrt((a_od / 2) ** 2 - max(abs(y0), abs(y1)) ** 2), a[2] + 0.004, b_od / 2, zm,
                y1, -(y1 - y0), lighten=_span_holes(zm + b_od / 2, a[2]))
    low = _xz_plate([(0.0, zm, b_od / 2), (0.0, zk, BOSS_K)], y0, y1)
    boss = _ycyl(BOSS_K, 0.0, zk, y0 - 0.005, y1 + 0.010)
    bore_m = pl_m * _disc(mate_bore_r(m_key, "front"), -0.1, 0.1)
    plate = web + low + boss
    plate -= bore_m
    plate -= _ycyl(BRG[1] / 2, 0.0, zk, -0.1, 0.1)
    # the motor's bolt holes as slots along the thigh, through the face plate and the web
    d = IFACE[m_key]
    for i in range(d["hou_n"]):
        t = 2 * math.pi * i / d["hou_n"]
        x, z = d["hou_bc"] / 2 * MM * math.cos(t), zm + d["hou_bc"] / 2 * MM * math.sin(t)
        slot = _bd.Pos(x, (y0 + y1) / 2, z) * _bd.Box(d["hou_thread"] * MM + 0.0004, 0.05, 2 * SLOT) + \
            _ycyl((d["hou_thread"] * MM + 0.0004) / 2, x, z - SLOT, -0.05, 0.05) + \
            _ycyl((d["hou_thread"] * MM + 0.0004) / 2, x, z + SLOT, -0.05, 0.05)
        plate -= slot
        side -= slot
    lug_z = zm + b_od / 2 + 0.006
    lug = _bd.Pos(0.0, y1 + 0.005, lug_z) * _bd.Box(0.016, 0.010, 0.010)
    lug -= _bd.Pos(0.0, y1 + 0.005, lug_z) * _bd.Cylinder(0.0021, 0.02)     # M5 tapped (4.2)
    plate = clear_bolts(plate, [(pl_p, p_key, "output")])
    plate = seat(plate, pl_p, p_key, "output", a_od)
    return {"plate_out": top, "plate_face": side, "plate": plate, "lug": lug}


def knee_drive_L():
    """The knee belt's parts that are not a link's structure, left leg, each in its own
    link's frame -> {"thigh": {...}, "calf": {...}}: on the thigh the small pulley (on
    the RS04's output), the belt and the two bearings; on the calf the big pulley (bolted
    to the inboard plate) and the shaft. Teeth are not drawn: the belt is its pitch line
    +- half its thickness, the pulleys their pitch circle with a flange each side."""
    kb = _spec.KNEE_BELT
    dm = [d for d in ANKLE_DRIVE if d[2] == "thigh_L"][0]
    zm = dm[3][2]
    zk = CHILD_OF["thigh_L"][0][4][2]
    by0, by1 = kb["y"]
    rs, rb, t = kb["small"] / 2, kb["big"] / 2, kb["t"]

    hub_r = mate_bore_r(dm[1], "front") - 0.001           # through the thigh's bore to the RS04

    def pulley(r, z, hub_to, hub):
        p = _ycyl(r - t / 2 - 0.0005, 0.0, z, by0, by1)    # under the belt's inside face
        for e in (by0 - FLANGE, by1):
            p += _ycyl(r + t / 2 + 0.0025, 0.0, z, e, e + FLANGE)
        if hub:
            p += _ycyl(hub, 0.0, z, by1 + FLANGE if hub_to > by1 else by0 - FLANGE, hub_to)
        return p

    # its hub is an output plate on the RS04's carrier (its M5s and dowels), 18 mm long
    # out through the thigh's bore to the belt
    pl_m = plane_at(dm[3], tuple(-v for v in dm[4]))
    small = pulley(rs, zm, 0.0, None) + pl_m * output_plate(dm[1], 2 * hub_r, abs(by1) + FLANGE)
    small -= _ycyl(0.008, 0.0, zm, -0.1, 0.1)
    outer = _bd.make_hull((_bd.Pos(0, zm) * _bd.Circle(rs + t / 2)).edges()
                          + (_bd.Pos(0, zk) * _bd.Circle(rb + t / 2)).edges())
    inner = _bd.make_hull((_bd.Pos(0, zm) * _bd.Circle(rs - t / 2)).edges()
                          + (_bd.Pos(0, zk) * _bd.Circle(rb - t / 2)).edges())
    band = _bd.extrude(_bd.Plane.XZ * (outer - inner), amount=by1 - by0)
    belt = _bd.Pos(0, by1 - band.bounding_box().max.Y, 0) * band
    brg = [_ycyl(BRG[1] / 2, 0.0, zk, y, y - BRG[2]) - _ycyl(BRG[0] / 2, 0.0, zk, -0.1, 0.1)
           for y in (0.0105, 0.0105 - BRG[2] - 0.001)]
    big = pulley(rb, 0.0, CALF_IN[1], 0.030) - _ycyl(KNEE_SHAFT[1], 0.0, 0.0, -0.1, 0.1)
    # pocketed both sides to a 5 mm web, as stock 8M pulleys this size are: solid it was 0.61 kg
    rim, ym = rb - t / 2 - 0.0005 - 0.005, (by0 + by1) / 2
    for a, b in ((by0 - FLANGE, ym - 0.0025), (ym + 0.0025, by1 + FLANGE)):
        big -= _ycyl(rim, 0.0, 0.0, a, b) - _ycyl(0.033, 0.0, 0.0, -0.1, 0.1)
    shaft = _ycyl(KNEE_SHAFT[1], 0.0, 0.0, CALF_IN[0], CALF_OUT[1]) - _ycyl(KNEE_SHAFT[0], 0.0, 0.0, -0.1, 0.1)
    return {"thigh": {"pulley_small": small, "belt": belt, "bearing_a": brg[0], "bearing_b": brg[1]},
            "calf": {"pulley_big": big, "shaft": shaft}}


def thigh_L():
    """hip yaw (AK60-39) output -> knee (RS04) housing, FRONT face: build.tee (the
    shoulder-rotation/elbow bracket) 196 mm long, with its cheek lying IN the knee's face
    plate. tee stands its cheek in the gap between a child's housing face and its output
    (7 mm on an AK80-64), but RS04's output is only 0.5 mm proud of its housing ring, so
    the thigh is one 10 mm plate down the leg's centre: Ø54 yaw flange at the top, the
    knee's Ø120 housing ring at the bottom, bored Ø83 for the calf's output plate. On the
    knee's 120 N.m it carries the calf's reaction in its own plane, and 400 N at the knee
    sideways bends it at ~20 MPa. Belt-driven knee (spec.KNEE_BELT): thigh_belt_L."""
    if MOTOR_OF["knee_L"] is None:
        return thigh_belt_L()
    name = "thigh_L"
    pj, cj = PJOINT[name], CHILD_OF[name][0]
    p_key, c_key = MOTOR_OF[pj[0]], MOTOR_OF[cj[0]]
    pl_p, pl_c = mount_plane(pj, (0.0, 0.0, 0.0)), mount_plane(cj, cj[4])
    a_od = IFACE[p_key]["out_od"] * MM
    b_od = face_od(c_key, "front")
    top = pl_p * output_plate(p_key, a_od)
    side = pl_c * face_plate(c_key, "front", b_od, THIGH_T)
    a = _pt(pl_p, IFACE[p_key]["out_z"] * MM + PLATE)      # underside of the top plate
    b = _pt(pl_c, IFACE[c_key]["hou_z"] * MM)              # the knee's housing face
    gy = grow(pl_c, "Y")
    y_far = max(abs(b[1]), abs(b[1] + gy * THIGH_T))
    z_top, z_knee = a[2] + 0.004, cj[4][2]
    web = cheek(math.sqrt((a_od / 2) ** 2 - y_far ** 2), z_top, b_od / 2, z_knee,
                b[1], gy * THIGH_T, bore=2 * mate_bore_r(c_key, "front"),
                lighten=_span_holes(z_knee + b_od / 2, a[2]))
    web = clear_bolts(web, [(pl_p, p_key, "output"), (pl_c, c_key, "front")])
    web = seat(seat(web, pl_p, p_key, "output", a_od), pl_c, c_key, "front", b_od, THIGH_T)
    return {"plate_out": top, "plate_face": side, "cheek": web}


# the calf box (spec, the ankle drive): plate faces in the calf frame, left leg
CALF_OUT = (0.014, 0.014 + CALF_T)          # outboard plate: clear of the thigh's knee plate + bolt heads
CALF_IN = (-0.0485 - CALF_T, -0.0485)       # inboard plate
SHIN_X = (0.031, 0.036)                     # the front web: 1.5 mm ahead of the ankle motors
PIN_R = 0.006                               # the ankle pitch pin (Ø12)
BOSS_R = 0.022                              # the plates' round ends round it


def _xz_plate(circles, y0, y1):
    """the tangent hull of circles [(x, z, r)] in the XZ plane, y0..y1 thick"""
    edges = []
    for x, z, r in circles:
        edges += (_bd.Pos(x, z) * _bd.Circle(r)).edges()
    p = _bd.extrude(_bd.Plane.XZ * _bd.make_hull(edges), amount=y1 - y0)   # along -Y
    return _bd.Pos(0, y0 - p.bounding_box().min.Y, 0) * p


def calf_L():
    """knee (RS04) output -> the two ankle motors (AK60-39, front faces) and the ankle
    pitch pin: a box of two 5 mm plates (spec: the ankle drive). The outboard plate
    takes the knee's output plate, which reaches out through the thigh's Ø83 bore, and
    motor A; the inboard one motor B; a shin web in front of the motors ties them, and
    both end in a boss round the ankle pin 320 mm down. The cranks and rods move with the
    motors and the foot: they are drawn by ankle_linkage_L."""
    name = "calf_L"
    pj = PJOINT[name]
    p_key = MOTOR_OF[pj[0]]
    belt = p_key is None                      # spec.KNEE_BELT: the shaft, not an RS04 output
    if belt:
        a_od = 2 * (KNEE_SHAFT[1] + 0.010)
        knee = None
    else:
        pl_p = mount_plane(pj, (0.0, 0.0, 0.0))
        a_od = IFACE[p_key]["out_od"] * MM
        knee = pl_p * output_plate(p_key, a_od, CALF_OUT[1])
    da, db = [d for d in ANKLE_DRIVE if d[2] == name]
    xa, za, xb, zb = da[3][0], da[3][2], db[3][0], db[3][2]
    ra = face_od(da[1], "front") / 2
    xp, yp, zp = 0.0, -ANKLE_IN, -CALF
    shin_edge = [(SHIN_X[1] - 0.01, z, 0.01) for z in (-0.075, -0.270)]   # the plates reach the shin
    # each plate covers both motors' outlines: closed sides, and the bolts of one motor's
    # face reached through the other plate
    both = [(xa, za, ra), (xb, zb, ra), (xp, zp, BOSS_R)] + shin_edge
    out = _xz_plate([(0.0, 0.0, a_od / 2)] + both, *CALF_OUT)
    inb = _xz_plate(([(0.0, 0.0, 0.035)] if belt else []) + both, *CALF_IN)   # the big pulley's seat
    shin = _bd.Pos((SHIN_X[0] + SHIN_X[1]) / 2, (CALF_IN[1] + CALF_OUT[0]) / 2, (-0.075 - 0.270) / 2) * \
        _bd.Box(SHIN_X[1] - SHIN_X[0], CALF_OUT[0] - CALF_IN[1] + 0.002, 0.270 - 0.075)
    parts = {"plate_out": knee, "side_out": out, "side_in": inb, "shin": shin}
    if belt:
        del parts["plate_out"]
        for k in ("side_out", "side_in"):                # the shaft's bore, clamped
            parts[k] -= _ycyl(KNEE_SHAFT[1], 0.0, 0.0, -0.1, 0.1)
    pin = _bd.Pos(xp, 0.0, zp) * _bd.Rot(90, 0, 0) * _bd.Cylinder(PIN_R, 0.2)
    faces = [] if belt else [(pl_p, p_key, "output")]
    for d, k in ((da, "side_out"), (db, "side_in")):
        pl = plane_at(d[3], tuple(-v for v in d[4]))
        parts["plate_" + d[0][-1]] = pl * face_plate(d[1], "front", 2 * ra, CALF_T)
        parts[k] -= pl * _disc(mate_bore_r(d[1], "front"), -0.1, 0.1)
        faces.append((pl, d[1], "front"))
    for k in ("side_out", "side_in"):
        parts[k] = clear_bolts(parts[k] - pin, faces)
    return parts


ROD_D, ROD_END_D = 0.008, 0.016           # Ø8 steel rods, M8 rod ends
CRANK_T, ARM_T = 0.008, 0.004             # a crank's hub on the output, and its arm on the hub's
                                          # outer 4 mm: clear of the face plate's bolt heads


def ankle_linkage_L():
    """The left ankle's moving drive at zero pose, in the calf frame: each motor's crank
    (its output plate drawn on out to the rod pin) and its rod with a rod end at each
    end. Not a link: the cranks turn with the motors and the rods float between crank and
    foot, so their mass is booked to the calf and the foot (massprops)."""
    parts = {}
    for d, (_n, u0, foot) in zip([d for d in ANKLE_DRIVE if d[2] == "calf_L"],
                                 [r for r in ANKLE_RODS if r[0].startswith("ankle_L")]):
        name, key, _h, xyz, ax, r = d
        pl = plane_at(xyz, tuple(-v for v in ax))
        hub = IFACE[key]["out_od"] / 2 * MM
        pin = tuple(xyz[i] + r * u0[i] for i in range(3))
        arm = _bd.make_hull((_bd.Circle(hub)).edges() + (_bd.Pos(r, 0) * _bd.Circle(0.011)).edges())
        # in the motor's frame: the plate grows along +z off the output face; turn the arm
        # so its pin lies along the crank's neutral direction
        crank = output_plate(key, 2 * hub, CRANK_T)
        z_top = IFACE[key]["out_z"] * MM + CRANK_T
        arm_s = _bd.Pos(0, 0, z_top - ARM_T) * _bd.extrude(arm, amount=ARM_T)
        loc_x = pl.x_dir
        # the arm drawn along the plane's own x, then turned onto u0 about the axis
        turn = math.degrees(math.atan2(loc_x.cross(_bd.Vector(*u0)).dot(pl.z_dir), loc_x.dot(_bd.Vector(*u0))))
        # placed piece by piece and then united, so the placement is in the solid itself:
        # a merely located part lost it when scaled for a STEP export
        parts["crank_" + name[-1]] = (pl * (_bd.Rot(0, 0, turn) * arm_s)) + (pl * crank)
        y_pin = foot[1]
        a = _bd.Vector(pin[0], y_pin, pin[2])
        b = _bd.Vector(*foot)
        L = (b - a).length
        rod = _bd.Plane(origin=a, z_dir=b - a) * _bd.Cylinder(ROD_D / 2, L, align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
        ends = _bd.Pos(*a) * _bd.Sphere(ROD_END_D / 2) + _bd.Pos(*b) * _bd.Sphere(ROD_END_D / 2)
        parts["rod_" + name[-1]] = rod + ends
    return parts


# The ankle's cross and the foot, in the ankle frame (origin: the ankle centre, between
# the calf plates). The cross turns on the calf's pitch pin and carries the roll shaft;
# the foot hangs from the shaft by two lugs, one ahead of the cross and one behind.
CROSS = (0.040, 0.0595, 0.030)            # x, y (1.5 mm clear of each calf plate), z
SHAFT_R = 0.006                           # the roll shaft (Ø12), fixed in the cross
BORE_R = 0.0062                           # bores the pins turn in
FOOT = (0.200, 0.130, 0.008)              # plate: long, wide (the rods' ends ride its edges), thick
HEEL = 0.080                              # ankle axis to heel: 40 % of the foot
# The sole's outline, a left foot (y + is the outside), in the ankle frame: a rounded
# heel, broad enough at the rods (x -0.060, y +-0.053, a 16 mm post under each ball),
# the outside edge curving out to the little toe, the inside edge near straight with a
# shallow arch, the big toe ahead on the inside -- a person's footprint, 200 long.
SOLE_OUTLINE = [(-0.080, 0.000), (-0.074, 0.042), (-0.058, 0.064), (-0.020, 0.064),
                (0.030, 0.066), (0.075, 0.066), (0.104, 0.050), (0.116, 0.022),
                (0.121, -0.010), (0.114, -0.042), (0.088, -0.064), (0.040, -0.066),
                (0.000, -0.060), (-0.040, -0.064), (-0.064, -0.062), (-0.077, -0.038)]
RUBBER = 0.005
LUG_T = 0.006


def _ycyl(r, x, z, y0, y1):
    lo, hi = sorted((y0, y1))
    return _bd.Pos(x, (lo + hi) / 2, z) * _bd.Rot(90, 0, 0) * _bd.Cylinder(r, hi - lo)


def _xcyl(r, y, z, x0, x1):
    return _bd.Pos((x0 + x1) / 2, y, z) * _bd.Rot(0, 90, 0) * _bd.Cylinder(r, x1 - x0)


def ankle_L():
    """The ankle's cross: a block turning on the calf's Ø12 pitch pin (along y) and the
    Ø12 roll shaft through it (along x), out to the foot's two lugs."""
    cx, cy, cz = CROSS
    block = _bd.Box(cx, cy, cz) - _ycyl(BORE_R, 0.0, 0.0, -cy, cy)
    shaft = _xcyl(SHAFT_R, 0.0, 0.0, -(cx / 2 + CLEAR + LUG_T + 0.004), cx / 2 + CLEAR + LUG_T + 0.004)
    return {"cross": block + shaft}


def foot_L():
    """The foot: an 8 mm plate on a 5 mm rubber sole, 200 x 130, the ankle axis 60 mm up
    and 80 mm from the heel; two lugs up to the roll shaft, ahead of and behind the cross;
    a post under each rod's lower ball, 60 mm behind the ankle axis."""
    L, W, t = FOOT
    z_sole = -SOLE
    # counter-clockwise seen from above, or the extruded solid comes out inside out
    outline = _bd.make_face(_bd.Spline(*SOLE_OUTLINE[::-1], periodic=True))
    plate = _bd.Pos(0, 0, z_sole + RUBBER) * _bd.extrude(outline, amount=t)
    sole = _bd.Pos(0, 0, z_sole) * _bd.extrude(outline, amount=RUBBER)
    top = z_sole + RUBBER + t
    parts = {"plate": plate, "sole": sole}
    cx = CROSS[0]
    for k, sx in (("front", 1), ("back", -1)):
        xa = sx * (cx / 2 + CLEAR)
        xb = xa + sx * LUG_T
        prof = _bd.make_hull((_bd.Pos(0.0, 0.0) * _bd.Circle(0.014)).edges()
                             + (_bd.Pos(0.0, top - 0.001) * _bd.Rectangle(0.040, 0.002)).edges())
        lug = _bd.extrude(_bd.Plane.YZ * prof, amount=abs(xb - xa))         # along +X
        lug = _bd.Pos(min(xa, xb), 0, 0) * lug
        parts["lug_" + k] = lug - _xcyl(BORE_R, 0.0, 0.0, -0.1, 0.1)
    for _n, _u, foot in [r for r in ANKLE_RODS if r[0].startswith("ankle_L")]:
        # the rod's lower ball: in the calf frame, the ankle centre at (0, -ANKLE_IN, -CALF)
        x, y, z = foot[0], foot[1] + ANKLE_IN, foot[2] + CALF
        h = z - ROD_END_D / 2 - top + 0.001          # 1 mm into the plate: one solid with it
        parts["post_" + _n[-1]] = _bd.Pos(x, y, top - 0.001 + h / 2) * _bd.Box(0.016, 0.016, h)
    return parts


BUILDERS.update(hip_L=hip_L, hip_R=mirrored("hip_R"),
                ankle_L=ankle_L, ankle_R=mirrored("ankle_R"),
                foot_L=foot_L, foot_R=mirrored("foot_R"),
                calf_L=calf_L, calf_R=mirrored("calf_R"),
                thigh_L=thigh_L, thigh_R=mirrored("thigh_R"),
                hip_roll_L=hip_roll_L, hip_roll_R=mirrored("hip_roll_R"))
