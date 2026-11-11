"""The connectors, one at a time. Each is built, checked and confirmed before
the next is drawn.

Every connector obeys the same three rules:
  1. it bolts to the PARENT actuator's output flange - a plate on that face,
     recessed over the spigot, one clearance hole per output bolt;
  2. it bolts to ONE housing face of the CHILD actuator - a plate on that face,
     bored for whatever stands proud, one clearance hole per mounting bolt;
  3. it is ONE connected solid, and it touches the actuators nowhere else.
"""
import math

from cadgen import build123d as _bd

from . import lidar
from .connect import (_disc, access_r, bolt_access, bolt_ring, cheek, face_od, face_plate,
                      hook_profile, mate_bore_r, output_plate,
                      plane_at)
from .spec import (BAY, BODY_DIR, CLEAR, FIXED, HOOK, IFACE, JOINTS, MOTOR_OF,
                   MOUNT_AT, MOUNT_ROT, PLATE, SHELL, TORSO_BC, TORSO_LOW,
                   TORSO_LOW_R, TORSO_N, HIPS, HIP_KEY, LOW_Z0, TORSO_TH, TORSO_UP, TRAY_Z, WAIST_DIA,
                   WALL)

MM = 0.001
MIN_ = (_bd.Align.MIN,) * 3
PJOINT = {j[2]: j for j in JOINTS}
CHILD_OF = {}
for _j in JOINTS:
    CHILD_OF.setdefault(_j[1], []).append(_j)


def clear_bolts(web, faces, reach=0.05):
    """Take every bolt hole on through the web, so the bolts can be reached."""
    for pl, key, face in faces:
        web -= pl * bolt_access(key, face, reach)
    return web


def seat(web, pl, key, face, od, t=PLATE):
    """Stop a web AT the plate it stands on instead of running through it.

    Every web used to overlap its plates - a cheek 3 mm deep into the disc it
    grows from, a leg with the whole face plate inside it. Where the two had
    the same rim that left two cylinders in one place, which the kernel does
    not merge: the part came out with a seam round every joint and looked
    like two blocks pushed together, because it was. Butted on the plate's
    face, the rims are one surface and the joint is a face, not a volume.
    """
    d = IFACE[key]
    z0 = {"output": d["out_z"] * MM, "rear": d["rear_z"] * MM - t,
          "front": d["hou_z"] * MM}[face]
    return web - pl * _disc(od / 2, z0, z0 + t)


def mount_plane(joint, origin):
    """That actuator's own frame, where it actually sits.

    MOUNT_AT can slide an actuator along its own axis, which is kinematically
    free, so the flange need not be on the joint origin (spec.MOUNT_AT).
    """
    a = MOUNT_AT.get(joint[0], 0.0)
    b = BODY_DIR[joint[0]]
    return plane_at(tuple(origin[k] + b[k] * a for k in range(3)), b)


def in_turned_frame(pl, a):
    """A plane given in a joint's frame, re-expressed in a child frame turned by
    `a` about z (spec.MOUNT_ROT). Origin AND both axes turn, so a bolt pattern
    drawn on it still lands on the actuator's real bolts."""
    c, s = math.cos(a), math.sin(a)

    def f(v):
        return (c * v.X + s * v.Y, -s * v.X + c * v.Y, v.Z)
    return _bd.Plane(origin=f(pl.origin), x_dir=f(pl.x_dir), z_dir=f(pl.z_dir))


# Which face a connector actually bolts to, where it is not the one
# mount_face_of() would pick. The shoulder is the only one so far: see
# shoulder_L().
FACE = {"shoulder_L": "rear", "shoulder_R": "rear"}


def grow(pl, axis):
    """Which way a plate on that face grows: along its own +Z, never into the
    actuator. Written out because the arm's actuators no longer all face the
    same way - see BODY_DIR in spec.py.
    """
    return 1.0 if getattr(pl.z_dir, axis) > 0 else -1.0


def mount_face_of(cj):
    """Which of the child actuator's two flanges this connector bolts to."""
    import math as _m
    L = _m.sqrt(sum(v * v for v in cj[4]))
    u = tuple(v / L for v in cj[4])
    c_along = abs(sum(a * b for a, b in zip(cj[3], u))) > 0.9
    toward_parent = sum(a * b for a, b in zip(BODY_DIR[cj[0]], u)) < 0
    return 'rear' if (c_along and toward_parent) else 'front'


def _lighten(part, bc, n, d, z0, z1):
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        part -= _bd.Pos(bc / 2 * math.cos(a), bc / 2 * math.sin(a), z0) * _bd.Cylinder(
            radius=d / 2, height=z1 - z0,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    return part


# --------------------------------------------------------------------------
# humerus_rot_L : shoulder-rotation output  ->  elbow housing, front face
#
#   The rotation actuator's axis runs DOWN the arm and its body points up
#   toward the shoulder, so its output flange faces down at the link origin.
#   The elbow actuator lies across the arm 60 mm below, body inboard, so the
#   face nearest this bracket is its front annulus at y = -8 mm.
#
#   The two plates are perpendicular and 55 mm apart. They cannot be joined
#   straight down the middle - that line runs through the elbow rotor - so the
#   bracket is a clevis: two ribs passing either side of it at x = +-47 mm,
#   which is clear of the 80 mm boss those ribs run past.
# --------------------------------------------------------------------------
def _pt(pl, z):
    """A point on an actuator's own axis, z into its frame, in LINK coordinates."""
    v = pl.origin + pl.z_dir * z
    return (v.X, v.Y, v.Z)


# --------------------------------------------------------------------------
# humerus_rot_L : shoulder-rotation output  ->  elbow housing, front face
#
#   Two mounting plates, each EXACTLY the diameter of the actuator face it
#   sits on - 72 mm on the rotation actuator's output flange, 97 mm on the
#   elbow's front annulus - joined by one cheek plate whose outline is tangent
#   to both rims. No material wider than the actuators, and no butt joints.
#
#   The cheek can run straight down the middle because the only thing between
#   the two faces is the elbow's 80 mm boss and 72 mm rotor, and the cheek is
#   bored 84 mm around them. Its own thickness spans the same 6 mm the face
#   plate does, so the two are one piece.
# --------------------------------------------------------------------------
#   Thickness is not free here, and web_t() is where that comes from.
def web_t(key):
    """How thick a web bolted to an actuator's FRONT face may be.

    Only the room between that face and the actuator's own output flange: the
    next link bolts to that flange and its web grows straight out of it, so a
    web thicker than the gap runs into it. Measured on the elbow, which gives
    7 mm - a 12 mm web there drove 853 mm3 into the forearm's web.
    """
    d = IFACE[key]
    return (d["out_z"] - d["hou_z"]) * MM


def tee(name):
    """A link whose parent actuator's output points DOWN it and whose child
    actuator lies ACROSS it - the shoulder-rotation/elbow pair, and the
    forearm-roll/wrist-flexion pair below it.

    A plate under the output flange, a plate on the child's front face, one
    cheek tangent to both rims. Everything comes off the two actuators.
    """
    pj, cj = PJOINT[name], CHILD_OF[name][0]
    p_key, c_key = MOTOR_OF[pj[0]], MOTOR_OF[cj[0]]
    pl_p, pl_c = mount_plane(pj, (0.0, 0.0, 0.0)), mount_plane(cj, cj[4])
    assert abs(pl_p.z_dir.Z) > 0.9 and abs(pl_c.z_dir.Y) > 0.9, \
        "tee(%s): wants the parent output along Z and the child's face along Y" % name

    a_od = IFACE[p_key]["out_od"] * MM               # the face, no more
    b_od = IFACE[c_key]["hou_od"] * MM               # ditto
    top = pl_p * output_plate(p_key, a_od)
    side = pl_c * face_plate(c_key, "front", b_od)

    a = _pt(pl_p, IFACE[p_key]["out_z"] * MM + PLATE)   # underside of the top plate
    b = _pt(pl_c, IFACE[c_key]["hou_z"] * MM)           # the child's mounting face
    # the cheek stops short of the output-flange plane: the actuator's
    # centering spigot lives in that last millimetre
    # The cheek meets the round top plate off its centre, on a chord, so it is
    # as wide as the plate is THERE - at the face of it furthest out - and not
    # as wide as the plate's diameter, which left both top corners standing
    # outside the rim.
    y_far = max(abs(b[1]), abs(b[1] + grow(pl_c, "Y") * web_t(c_key)))
    web = cheek(math.sqrt((a_od / 2) ** 2 - y_far ** 2), a[2] + 0.004,
                b_od / 2, cj[4][2],
                b[1], grow(pl_c, "Y") * web_t(c_key),
                bore=2 * mate_bore_r(c_key, "front"),
                lighten=[(0.0, (a[2] + cj[4][2]) / 2 + 0.008, 0.013)])
    web = clear_bolts(web, [(pl_p, p_key, "output"), (pl_c, c_key, "front")])
    web = seat(seat(web, pl_p, p_key, "output", a_od), pl_c, c_key, "front", b_od)
    return {"plate_out": top, "plate_face": side, "cheek": web}


def humerus_rot_L():
    """shoulder-rotation output -> elbow housing, front face. Two AK80-64."""
    return tee("humerus_rot_L")


def forearm_rot_L():
    """forearm-roll output -> wrist-flexion housing, front face. Two AK60-39.

    The same part as the elbow bracket one joint down and a size smaller: the
    faces are Ø54 and Ø78 instead of Ø72 and Ø97, and the AK60-39 leaves only
    5.5 mm between its front face and its output flange, so that is the web.
    """
    return tee("forearm_rot_L")


# --------------------------------------------------------------------------
# fore_arm_L : elbow output  ->  forearm-roll housing, rear face
#
#   Same two-plate-and-a-cheek part as the elbow bracket, turned inside out.
#   The elbow's output flange faces outboard, so its plate lies IN the plane of
#   the cheek and shows as a circle; the roll actuator hangs 96.5 mm below on
#   the arm axis with its rear flange up, so that plate shows edge-on. The
#   cheek is therefore the same tangent hull with the round end at the top.
#
#   Two things stand in its way and both are cleared, not dodged: the elbow's
#   Ø35.4 spigot, which the cheek starts just above (the plate's counterbore
#   takes that millimetre), and the roll actuator's Ø58 rear pilot boss, which
#   stands 18 mm up out of the face the plate bolts to.
# --------------------------------------------------------------------------
ARM_CHEEK_T = 0.012
LIGHT_GAP = 0.008           # metal left between lightening holes


def _span_holes(z_lo, z_hi, gap=LIGHT_GAP, pitch=0.050, rmin=0.008):
    """Lightening holes filling a clear run of web, as big as the run allows.

    A long web wants several; a short one wants one or none. Working it out
    from the run rather than typing a position in means the upper arm's 97 mm
    of clear web gets two holes without anyone deciding that by hand.
    """
    L = z_hi - z_lo
    n = int(round((L + gap) / pitch))
    if n < 1:
        return []
    r = (L - (n + 1) * gap) / (2.0 * n)
    if r < rmin:
        return []
    return [(0.0, z_hi - gap - r - i * (2 * r + gap), r) for i in range(n)]


def hang(name, t=ARM_CHEEK_T, stand=0.0, plate_t=PLATE):
    """A link whose parent actuator's output faces ACROSS it and whose child
    actuator hangs down the link on its own axis, body pointing back at the
    parent - the forearm, and the upper arm above it.

    The child is held by its REAR flange, which is the face that end of it
    offers, so the cheek's straight edge is that flange and its round end is
    the parent's output flange. Between them stands the child's rear pilot
    boss, which the cheek is bored for.

    The parent's output may face across the arm (Y: the elbow) or fore-and-aft
    (X: the pitch-first shoulder's roll actuator); the cheek lies against it
    either way.
    """
    pj, cj = PJOINT[name], CHILD_OF[name][0]
    p_key, c_key = MOTOR_OF[pj[0]], MOTOR_OF[cj[0]]
    dp, dc = IFACE[p_key], IFACE[c_key]
    pl_p, pl_c = mount_plane(pj, (0.0, 0.0, 0.0)), mount_plane(cj, cj[4])
    ax = "Y" if abs(pl_p.z_dir.Y) > 0.9 else "X"
    assert abs(getattr(pl_p.z_dir, ax)) > 0.9, "hang(%s): parent output not across" % name

    a_od = dp["out_od"] * MM                         # the parent's output face
    b_od = face_od(c_key, "rear")                    # the child's rear flange
    top = pl_p * output_plate(p_key, a_od, plate_t)
    side = pl_c * face_plate(c_key, "rear", b_od)

    # the cheek starts where the plate's spigot counterbore ends, so it is
    # clear of the spigot and still 3 mm deep into the plate
    w0 = _pt(pl_p, dp["out_z"] * MM + (stand if stand > 0.0 else
             dp["out_spigot_h"] * MM + CLEAR + 0.0005))["XY".index(ax)]
    z_rear = _pt(pl_c, dc["rear_z"] * MM)[2]
    z_pilot = _pt(pl_c, (dc["rear_z"] - dc["rear_pilot_h"]) * MM)[2]
    r_cable = access_r(p_key)
    # the pilot bore is the ring's own bore, so the two legs beside it come
    # down flush with the inside of the ring instead of a hair off it
    r_pilot = mate_bore_r(c_key, "rear")
    z_boss = (z_rear + z_pilot) / 2.0
    web = cheek(b_od / 2, z_rear, a_od / 2, 0.0, w0, grow(pl_p, ax) * t,
                lighten=[(0.0, 0.0, r_cable), (0.0, z_boss, r_pilot)]
                + _span_holes(z_boss + r_pilot, -r_cable), across=ax)
    # The cheek stands on the ring off its centre, on a chord, so a slab as
    # wide as the ring overhangs the rim at both corners (4.3 mm on the upper
    # arm). It is turned to the ring's own diameter instead.
    rz = dc["rear_z"] * MM
    web &= pl_c * _disc(b_od / 2, rz - 0.3, rz + 0.01)
    web = clear_bolts(web, [(pl_p, p_key, "output"), (pl_c, c_key, "rear")])
    web = seat(seat(web, pl_p, p_key, "output", a_od, plate_t),
               pl_c, c_key, "rear", b_od)
    return {"plate_out": top, "plate_face": side, "cheek": web}


def fore_arm_L():
    """elbow output -> forearm-roll housing, rear face. AK80-64 to AK60-39."""
    return hang("fore_arm_L")


def upper_arm_L():
    """shoulder-abduction (roll) output -> shoulder-rotation housing, rear face.

    The upper arm as the pitch-first humanoids hang theirs: off the roll
    actuator's output, with the rotation actuator on the arm's own axis below,
    its body pointing back up at the shoulder. The roll output faces aft (with
    the arm down), so the cheek lies across the arm in the YZ plane rather than
    fore-and-aft - otherwise the same part as the forearm, one size up and
    200 mm long: two AK80-64, a Ø72 round end, a Ø97 straight one, the
    rotation actuator's Ø71 pilot boss bored out of the web between them and
    two lightening holes in the 97 mm of clear web left.

    It carries the rotation actuator's reaction in TORSION - 120 N.m along the
    link at stall. A 12 x 97 plate takes that at 26 MPa and twists 0.7 deg over
    its length, so 12 mm holds here too.
    """
    return hang("upper_arm_L", stand=0.009, plate_t=0.013)


# --------------------------------------------------------------------------
# wrist_L : wrist-flexion output  ->  wrist-deviation housing, front face
#
#   The first corner in the arm. Flexion turns about Y, deviation about X, and
#   the link runs along Z, so the two mounting faces lie in PERPENDICULAR
#   planes and no single plate can sit flat on both. One leg per face then -
#   but the two legs share ONE SQUARE SPINE where they meet, 12 x 12 mm, the
#   flexion leg's width equal to the deviation leg's thickness and vice versa.
#
#   That is the whole trick. Sized any other way the legs stick out past each
#   other: the first version had a 26 mm-wide leg handing over to a leg that
#   flared 14 mm proud of it, which read as two square wings with a rib down
#   the middle - three pieces of metal that carried nothing. With one shared
#   section there is no step to see and nothing to trim.
#
#   Above the spine the flexion leg opens out into the Ø54 flange; below it
#   the deviation leg flares into the Ø78 face. Both are tangent, so the whole
#   part is one line from flange to flange, like OpenArm's wrist.
# --------------------------------------------------------------------------
#   Size of that spine is not free either. It carries the flexion actuator's
#   whole output: 16 x 16 mm gives W = 683 mm3, so the AK60-39's 72 N.m stall
#   puts 105 MPa in it against 276 for 6061-T6, and the 52 N.m the hang
#   actually asks for puts 76. At 12 mm it was 146 MPa at stall - the part
#   would have yielded before the motor gave up.
# The spine is a RECTANGLE, not a square, and the two numbers are not
# interchangeable. Across the arm there is open space on both wrist links, so
# that dimension is cheap. Fore-and-aft is where the NEXT link's own bracket
# has to pass, so every millimetre there costs the next link a millimetre of
# stand-off - and bending is what it buys, W going as the square of it.
#   16 x 14 is the smallest that keeps BOTH section moduli over 522 mm3,
#   which is what the AK60-39's 72 N.m stall needs at half of 6061-T6's
#   276 MPa: W = 597 about one axis and 523 about the other, so 121 and
#   138 MPa at a stall, 87 and 100 at the 52 N.m the hang actually asks for.
WRIST_X = 0.016             # spine, fore-and-aft
WRIST_Y = 0.014             # spine, across the arm
WAIST_R = 0.006             # the round where one leg turns into the other. 8 mm
                            # put 0.2 mm3 of wrist_L's into wrist_dev_L at -15 deg
NECK = 0.008                # how far each leg is drawn on into the other


def corner(name, face="front", stand=0.0, plate_t=PLATE, spine=None):
    """A link whose two joint axes are perpendicular to EACH OTHER and to the
    link - the two wrist links.

    The two mounting faces then lie in perpendicular planes and no single plate
    can sit flat on both, so there is one leg in the plane of each face. What
    makes it read as one part rather than three is that both legs share ONE
    SQUARE SPINE where they meet: each leg's thickness equals the other leg's
    width, and each leg's taper is centred on the other leg rather than on its
    own joint axis. Sized any other way they stand proud of each other and
    leave wings and a rib that carry nothing.

    Above the spine the parent leg opens into the parent's output flange; below
    it the child leg flares into the child's housing face. Both are tangent, so
    the part is one line from flange to flange.

    `face` says which of the child's two flanges to bolt. The near one leaves
    the bracket on the same side as the child's output, so the NEXT link has
    to stand off past it; the far one puts the bracket behind the actuator and
    leaves that side clear. Which is cheaper depends on the link - see
    shoulder_L, which takes the far one.
    """
    pj, cj = PJOINT[name], CHILD_OF[name][0]
    p_key, c_key = MOTOR_OF[pj[0]], MOTOR_OF[cj[0]]
    dp, dc = IFACE[p_key], IFACE[c_key]
    pl_p, pl_c = mount_plane(pj, (0.0, 0.0, 0.0)), mount_plane(cj, cj[4])

    # each leg is thick along the normal of its own mating face
    ax_p = "Y" if abs(pl_p.z_dir.Y) > 0.9 else "X"
    ax_c = "Y" if abs(pl_c.z_dir.Y) > 0.9 else "X"
    assert {ax_p, ax_c} == {"X", "Y"}, "corner(%s): the axes are not a corner" % name
    i_p, i_c = (1 if ax_p == "Y" else 0), (1 if ax_c == "Y" else 0)

    spine = spine or {"X": WRIST_X, "Y": WRIST_Y}
    t_p, t_c = spine[ax_p], spine[ax_c]
    # a plate on a rear flange grows back the other way along its own axis
    sgn = grow(pl_c, ax_c) * (-1.0 if face == "rear" else 1.0)
    sgp = grow(pl_p, ax_p)

    a_od = dp["out_od"] * MM
    b_od = face_od(c_key, face)
    top = pl_p * output_plate(p_key, a_od, plate_t)
    side = pl_c * face_plate(c_key, face, b_od)

    # the parent leg starts where its plate's spigot counterbore ends; the
    # child leg starts on the child's mounting face
    w_p = _pt(pl_p, (dp["out_z"] + dp["out_spigot_h"]) * MM + CLEAR + 0.0005)[i_p]
    w_p = w_p if stand <= 0.0 else _pt(pl_p, dp["out_z"] * MM + stand)[i_p]
    w_c = _pt(pl_c, dc["%s_z" % ("rear" if face == "rear" else "hou")] * MM)[i_c]
    # they hand over at the bottom of the parent actuator's own silhouette
    z_hand = -(dp["od"] / 2 * MM + 0.0005)

    # Each leg is drawn on into the other as a parallel neck, and the corner
    # where its taper turns into that neck is a radius. They used to stop 4 mm
    # apart in height, so the child's flare began beside the parent leg and
    # stood up its side as a spike; and where each taper ended on the other
    # leg there was a crease. The neck lies wholly inside the other leg, whose
    # section there is the same spine, so it adds nothing but the radius.
    #
    # One corner stays sharp: the child leg's, on the side of the parent
    # actuator. Flush on its flange the parent leg is 0.5 mm off that barrel.
    r_bar = 0.0 if stand <= 0.0 else WAIST_R
    lo_bar = (_pt(pl_p, -0.02)[i_p] < w_p)           # the barrel is on the low side
    c_p = w_c + sgn * t_c / 2
    web_p = cheek(t_c / 2, z_hand, a_od / 2, 0.0, w_p, sgp * t_p,
                  lighten=[(0.0, 0.0, max(dp["out_bore"] - 2.0, 4.0) / 2 * MM)],
                  across=ax_p, c=c_p, neck=(NECK, WAIST_R, WAIST_R))
    web_c = cheek(t_p / 2, z_hand, b_od / 2, cj[4][2], w_c, sgn * t_c,
                  across=ax_c, c=w_p + sgp * t_p / 2,
                  neck=(NECK, r_bar if lo_bar else WAIST_R,
                        WAIST_R if lo_bar else r_bar))
    # bored along the face plate's own axis and from its far side: drawn that
    # way round, the leg's bore and the plate's merge into one (check_seams.py)
    web_c -= pl_c * _bd.Rot(180, 0, 0) * _disc(mate_bore_r(c_key, face), -0.1, 0.1)

    webs = clear_bolts(web_p + web_c,
                       [(pl_p, p_key, "output"), (pl_c, c_key, face)])
    webs = seat(seat(webs, pl_p, p_key, "output", a_od, plate_t),
                pl_c, c_key, face, b_od)
    return {"plate_out": top, "plate_face": side, "web": webs}


def wrist_L():
    """wrist-flexion output -> wrist-deviation housing, front face.

    Flexion turns about Y, deviation about X, the link runs along Z.
    """
    return corner("wrist_L")


def side_corner(name, face="rear", plate_t=PLATE, web_min=0.016):
    """A link whose child actuator sits BESIDE the parent's output, at the same
    height and on a perpendicular axis - the pitch-first shoulder (spec.py).

    Seen from above it is an L. One leg is a web lying on the parent's output
    plate, as thick as the gap to the child actuator's flank allows, running
    fore-and-aft past that flank; the other turns across the child's flange.
    Each leg's outline is the tangent hull of the two faces it joins, so the
    part follows both rims; nothing goes round the actuator, and the child's
    output side is left clear for the next link.
    """
    pj, cj = PJOINT[name], CHILD_OF[name][0]
    p_key, c_key = MOTOR_OF[pj[0]], MOTOR_OF[cj[0]]
    dp, dc = IFACE[p_key], IFACE[c_key]
    pl_p, pl_c = mount_plane(pj, (0.0, 0.0, 0.0)), mount_plane(cj, cj[4])
    assert abs(pl_p.z_dir.Y) > 0.9 and abs(pl_c.z_dir.X) > 0.9, \
        "side_corner(%s): wants the parent output along Y, the child axis along X" % name
    sg = grow(pl_p, "Y")                             # the side the child is on
    a_od = dp["out_od"] * MM
    b_od = face_od(c_key, face)
    top = pl_p * output_plate(p_key, a_od, plate_t)
    side = pl_c * face_plate(c_key, face, b_od)

    # the web: from where the output plate's spigot recess ends to the child's flank
    y0 = _pt(pl_p, (dp["out_z"] + dp["out_spigot_h"]) * MM + CLEAR + 0.0005)[1]
    y1 = cj[4][1] - sg * (dc["od"] / 2 * MM + CLEAR)
    t = abs(y1 - y0)
    assert t >= web_min, "side_corner(%s): %.1f mm of web, want %.1f" % (
        name, t * 1000, web_min * 1000)
    sb = side.bounding_box()
    xs0, xs1, zc = sb.min.X, sb.max.X, cj[4][2]
    foot = _bd.Pos((xs0 + xs1) / 2, zc) * _bd.Rectangle(xs1 - xs0, b_od)
    hull_xz = _bd.make_hull(_bd.Circle(a_od / 2).edges() + foot.edges())
    web = _bd.extrude(_bd.Plane.XZ * hull_xz, amount=t)      # Plane.XZ extrudes along -Y
    web = _bd.Pos(0, min(y0, y1) - web.bounding_box().min.Y, 0) * web

    # the leg: across the child's flange, from the web's end to the whole face
    end = _bd.Pos((y0 + y1) / 2, zc) * _bd.Rectangle(t, b_od)
    disc = _bd.Pos(cj[4][1], zc) * _bd.Circle(b_od / 2)
    hull_yz = _bd.make_hull(end.edges() + disc.edges())
    # extruded back from its far face: drawn the other way round, its two flat
    # sides do not merge with the web's and the corner keeps a seam
    leg = _bd.Pos(xs1, 0, 0) * _bd.extrude(_bd.Plane.YZ * hull_yz, amount=xs0 - xs1)
    leg -= _bd.Pos(xs0 - 0.01, cj[4][1], zc) * _bd.Rot(0, 90, 0) * _bd.Cylinder(
        radius=mate_bore_r(c_key, face), height=xs1 - xs0 + 0.02,
        align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))

    faces = [(pl_p, p_key, "output"), (pl_c, c_key, face)]

    def fit(part):
        return seat(seat(clear_bolts(part, faces), pl_p, p_key, "output", a_od, plate_t),
                    pl_c, c_key, face, b_od)
    # the leg is butted on the web's end, not run through it
    return {"plate_out": top, "plate_face": side,
            "web": fit(web), "leg": fit(leg) - web}


def shoulder_L():
    """shoulder-flexion (pitch) output -> shoulder-abduction (roll) housing, REAR face.

    The shoulder the way ten of the twelve humanoids measured for spec.py build
    it: the pitch actuator on the torso, the roll actuator beside its output at
    the same height, its axis fore-and-aft. Two AK80-64.

    The web is 17.5 mm, the whole gap between the Ø72 output plate and the roll
    actuator's flank: the roll actuator's 120 N.m reaches the pitch output
    through it, as torsion about the roll axis - on a 97 x 17.5 section that is
    about 14 MPa against 6061-T6's ~160 in shear.

    It takes the roll actuator's FAR flange. The near one is on the output
    side, where upper_arm_L bolts on and swings; this leaves that whole side to
    the upper arm, and the bracket sits on the inboard flank and behind.
    """
    return side_corner("shoulder_L", face="rear", plate_t=0.010)


def wrist_dev_L():
    """wrist-deviation output -> finger housing, front face.

    The same corner one joint down, with the axes swapped: deviation about X
    now feeds the finger about Y. The parent is the same AK60-39, so the spine
    is the same 16 mm.

    The finger actuator is the small one, and two of its numbers had to be
    handled rather than copied. Its front face is Ø53 with M3 at Ø48, which
    leaves the bolt head hanging 0.25 mm over the rim - face_od() takes the
    plate out to the Ø55 barrel so the heads are carried. And three Ø4 dowel
    pins stand 2 mm proud of its OUTPUT flange on a Ø24 circle; they sit
    inside the Ø40 bore this web already needs for the rotor, so they clear.

    It also has to stand off. wrist_L's own web is 16 mm thick on the far side
    of this actuator's mounting face, which puts it 10 mm past the output
    flange this link bolts to; measured, the two webs overlapped by 2181 mm3
    at every deviation angle. So this link's plate is 16 mm thick and its leg
    starts 12 mm out, clear of wrist_L by 2 mm through the whole sweep.
    """
    return corner("wrist_dev_L", stand=0.012, plate_t=0.016)


# --------------------------------------------------------------------------
# hook_L : finger output  ->  the hooked hand itself
#
#   The last link, and the only one with no second actuator to bolt to. It is
#   still the same corner: the finger's output flange faces sideways (+Y) and
#   the palm has to face the wall (normal X), so one leg lies in each plane
#   and they share a spine. The difference is that the far end is a hook, not
#   a flange, so the child leg keeps going and turns into the blade.
#
#   The palm, the lip and their 65 mm width are the numbers the RL work was
#   built on - urdf/climbing_robot.urdf carries them as two
#   collision boxes - so they are copied, not redesigned. What is drawn here
#   is how they attach.
#
#   v1's hook did not attach: it put the palm's box flat against the plate's
#   outer face, and to keep it off the finger actuator it slid the whole hook
#   36.5 mm sideways, which is why its mesh and its collision boxes disagreed
#   about where the hand was. The palm is back on the axis here, and it is the
#   TOP of the palm that is cut away instead - which costs nothing, because
#   the actuator only reaches 27.5 mm down and the rung never touches there.
# --------------------------------------------------------------------------
HOOK_SPINE = 0.014          # the palm's neck, across the hand
HOOK_WIDE = -0.0775         # where the shank has spread to the full blade
HOOK_FILLET = 0.020


def hook_L():
    pj = PJOINT["hook_L"]                            # finger_L
    p_key = MOTOR_OF[pj[0]]
    dp = IFACE[p_key]
    # the hook is bolted on turned over (spec.MOUNT_ROT): its throat stays on
    # its own +X, and its plate follows the flange round to the other side
    pl_p = in_turned_frame(mount_plane(pj, (0.0, 0.0, 0.0)), MOUNT_ROT["hook_L"])

    a_od = dp["out_od"] * MM                         # 36 mm - the finger's face

    # The hook has to stand off, for the same reason wrist_dev_L did. The link
    # above ends in a 14 mm ring around this actuator's front face, and the
    # hook's web has to cross exactly that band of radius to get from the Ø36
    # flange out to the palm: measured, 1734 mm3 of overlap at every finger
    # angle. So the flange plate is deep enough to carry the web clear of it.
    sg0 = grow(pl_p, "Y")
    y_ring = _pt(pl_p, dp["hou_z"] * MM)[1] + sg0 * WRIST_Y
    y_lo = y_ring + sg0 * 0.002
    t_top = abs(y_lo - _pt(pl_p, dp["out_z"] * MM)[1]) + 0.004
    top = pl_p * output_plate(p_key, a_od, t_top)
    z_neck = -(dp["od"] / 2 * MM + 0.0025)           # clear of the actuator
    t = HOOK["thk"]

    sgp = grow(pl_p, "Y")
    # web and palm hand over like a wrist's two legs: each is drawn on into the
    # other as a neck and rounded where it leaves it. The palm's flank on the
    # side of the link above stays sharp - that link's ring is 2 mm away.
    web = cheek(t / 2, z_neck, a_od / 2, 0.0, y_lo, sgp * HOOK_SPINE,
                lighten=[(0.0, 0.0, max(dp["out_bore"] - 2.0, 4.0) / 2 * MM)],
                across="Y", c=0.0, neck=(NECK, WAIST_R, WAIST_R))
    palm = _bd.extrude(_bd.Plane.YZ * hook_profile(
        y_lo + sgp * HOOK_SPINE / 2, HOOK_SPINE, z_neck, HOOK["width"] / 2,
        HOOK_WIDE, -HOOK["back"], HOOK_FILLET, stub=NECK,
        r_neck=(0.0, WAIST_R) if sgp > 0 else (WAIST_R, 0.0)), amount=t)
    bb = palm.bounding_box()
    palm = _bd.Pos(-t / 2 - bb.min.X, 0, 0) * palm
    lip = _bd.Pos(t / 2 + HOOK["lip"] / 2, 0.0, -HOOK["back"] + t / 2) * \
        _bd.Box(HOOK["lip"], HOOK["width"], t)

    web = seat(clear_bolts(web, [(pl_p, p_key, "output")]),
               pl_p, p_key, "output", a_od, t_top)
    return {"plate_out": top, "web": web, "palm": palm + lip}


# --------------------------------------------------------------------------
# The three torso links. These are not two-flange connectors - a torso is a
# frame - so they are panels and decks rather than cheeks. What stays the same
# is the rule the arm was drawn to: every actuator is held by a plate lying on
# a face it really has, with a clearance hole for every one of its bolts.
#
# There is exactly one mating interface in the whole robot that no actuator
# provides: waist_link to torso_upper, which the URDF calls a fixed joint.
# spec.TORSO_BC names it - Ø88, 8 x M4 - and both sides are drawn from it.
# --------------------------------------------------------------------------
CMIN = (_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN)
FLOOR = 0.005               # the lower torso's floor, under the battery
DECK = 0.006
CHEST_BAND = 0.018          # the metal left round a window in the chest
CABLE_SLOT = (0.044, 0.016)  # the harness slots through the chest's deck, long along x...
CABLE_Y = 0.0635             # ...between the waist flange (50 mm) and the walls (77 mm)


def _prism(face, z0, h):
    return _bd.Pos(0.0, 0.0, z0) * _bd.extrude(_bd.Plane.XY * face, amount=h)


def _low_section(inset=0.0):
    d, w = TORSO_LOW[0], TORSO_LOW[1]
    return (_bd.Rectangle(d - 2 * inset, w - 2 * inset)
            & _bd.Circle(TORSO_LOW_R - inset))


def _windows(d, w, h, zs):
    """Lightening windows. A torso panel is a shear web, not a pressure vessel."""
    cut = None
    for zc in zs:
        for c in (_bd.Pos(0, 0, h * zc) * _bd.Box(d * 3, w * 0.44, h * 0.15),
                  _bd.Pos(0, 0, h * zc) * _bd.Box(d * 0.44, w * 3, h * 0.15)):
            cut = c if cut is None else cut + c
    return cut


def base_link():
    """The pelvis: the waist actuator nose-up in its top, and under it the two
    side walls that each hip's first actuator (spec.HIPS, a pitch AK80-64) will
    bolt its front face to from inside, output out through the wall -- the
    chest's shoulder mounting, a floor lower. No hip actuator is fitted yet.

    The section is the 140 x 146 box cut back to TORSO_LOW_R: the waist turns
    the whole upper body over this pelvis, stationary, and a hanging forearm
    sweeps a circle 88 mm round the waist axis down here. The front and the back
    are a 2.5 mm shell with lightening windows; the floor closes it under the hip
    actuators. The waist actuator is held by its REAR flange on a deck, its
    output finishing flush with the rim.
    """
    cj = CHILD_OF["base_link"][0]                    # waist_yaw
    c_key = MOTOR_OF[cj[0]]
    dc = IFACE[c_key]
    pl_c = mount_plane(cj, cj[4])
    d, w, h = TORSO_LOW
    z0 = LOW_Z0
    sec = _prism(_low_section(), z0, h)

    shell = sec - _prism(_low_section(SHELL), z0 + FLOOR, h)
    for sy in (1, -1):                                       # its sides are the hip walls
        shell -= _bd.Pos(0, sy * w / 2, 0) * _bd.Box(2 * d, 2 * PLATE, 4 * h)
    for zc in (0.30, 0.62):                                  # windows front and back
        shell -= _bd.Pos(0, 0, z0 + h * zc) * _bd.Box(d * 3, w * 0.44, h * 0.15)
    parts = {"shell": shell}

    floor = _prism(_low_section(), z0, FLOOR)
    floor -= bolt_ring(0.110, 6, 0.030, z0 - 0.001, z0 + FLOOR + 0.001)
    parts["floor"] = floor

    for hj in HIPS:                                          # the legs' sockets
        pl_h = mount_plane(hj, hj[4])
        b_od = face_od(HIP_KEY, "front")
        ya = _pt(pl_h, IFACE[HIP_KEY]["hou_z"] * MM)[1]
        yb = _pt(pl_h, IFACE[HIP_KEY]["hou_z"] * MM + PLATE)[1]
        assert abs(max(abs(ya), abs(yb)) - w / 2) < 1e-6, \
            "TORSO_LOW's width is not the hip walls' outer faces"
        zc = hj[4][2]
        z_lo, z_hi = zc + b_od / 2 + CHEST_BAND, z0 + h - CHEST_BAND
        prof = (_bd.Pos(0, z0 + h / 2) * _bd.Rectangle(d, h)
                - _bd.Pos(0, zc) * _bd.Circle(b_od / 2))
        if z_hi - z_lo > 0.02:
            prof -= _bd.Pos(0, (z_lo + z_hi) / 2) * _bd.RectangleRounded(
                d - 2 * CHEST_BAND, z_hi - z_lo, 0.008)
        wall = _bd.extrude(_bd.Plane.XZ * prof, amount=PLATE)   # along -Y
        wall = _bd.Pos(0, min(ya, yb) - wall.bounding_box().min.Y, 0) * wall
        parts["hip_" + hj[0].split("_")[1]] = (wall + pl_h * face_plate(HIP_KEY, "front", b_od)) & sec

    z_rear = _pt(pl_c, dc["rear_z"] * MM)[2]
    deck = (pl_c * face_plate(c_key, "rear", 0.40)) & _prism(
        _low_section(), z_rear - PLATE - 0.005, PLATE + 0.005)
    deck -= bolt_ring(0.150, 8, 0.024, z_rear - PLATE - 0.01, z_rear + 0.01)
    parts["deck"] = deck
    return parts


def waist_link():
    """Between the waist actuator's output and the upper torso frame.

    40 mm of link, and the only job is to take a Ø54 flange out to the Ø88
    bolt circle the frame sits on. A cone does that in one surface: it leaves
    the output flange at its own diameter, which is the rule everywhere else
    in this robot, and finishes at the frame's without a step anywhere.
    """
    pj = PJOINT["waist_link"]
    p_key = MOTOR_OF[pj[0]]
    dp = IFACE[p_key]
    pl_p = mount_plane(pj, (0.0, 0.0, 0.0))
    a_od = dp["out_od"] * MM
    top = pl_p * output_plate(p_key, a_od)

    z0 = _pt(pl_p, dp["out_z"] * MM + PLATE)[2]      # the plate's far face
    zt = FIXED[0][3][2]                              # where the frame bolts on
    r_f = WAIST_DIA / 2
    hc = zt - PLATE - z0
    cone = (_bd.Pos(0, 0, z0) * _bd.Cone(bottom_radius=a_od / 2, top_radius=0.046,
                                         height=hc, align=CMIN)
            - _bd.Pos(0, 0, z0 - 0.001) * _bd.Cone(
                bottom_radius=a_od / 2 - WALL, top_radius=0.042,
                height=hc + 0.002, align=CMIN))
    flange = _bd.Pos(0, 0, zt - PLATE) * (
        _bd.Cylinder(radius=r_f, height=PLATE, align=CMIN)
        - _bd.Cylinder(radius=0.038, height=3 * PLATE, align=CMIN))
    flange -= bolt_ring(TORSO_BC, TORSO_N, TORSO_TH + 0.0004, zt - 2 * PLATE,
                        zt + PLATE)
    return {"plate_out": top, "cone": cone, "flange": flange}


def _blk(x0, x1, y0, y1, z0, z1):
    (x0, x1), (y0, y1), (z0, z1) = sorted((x0, x1)), sorted((y0, y1)), sorted((z0, z1))
    return _bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * _bd.Box(x1 - x0, y1 - y0, z1 - z0)


def _grid(x0, x1, y0, y1, z0, z1, axis="z", pitch=0.010, d=0.0034):
    """M3 clearance holes on a 10 mm grid through a plate, for boards bolted on standoffs"""
    cut = None
    for i in range(int(round((x1 - x0) / pitch)) + 1):
        for j in range(int(round((y1 - y0) / pitch)) + 1):
            u, v = x0 + i * pitch, y0 + j * pitch
            if axis == "z":
                c = _bd.Pos(u, v, (z0 + z1) / 2) * _bd.Cylinder(d / 2, z1 - z0)
            else:                                # through x: u is y, v is z
                c = _bd.Pos((z0 + z1) / 2, u, v) * _bd.Cylinder(d / 2, z1 - z0, rotation=(0, 90, 0))
            cut = c if cut is None else cut + c
    return cut


def _bay_frame():
    """What holds spec.BAY's boards and battery in the chest: a drawer for the battery
    on the deck -- two L rails it slides in on from the back, under the back panel's
    bottom band, and a stop at the front with a window for its connector -- four posts
    carrying the electronics tray over it, and on the tray the bulkhead the Jetson
    stands on between the two sunk lidars. The tray and the bulkhead are drilled on an
    M3 grid; the tray is notched over the deck's two harness slots."""
    (bx, by, bz), (bl, bw, bh), _ = BAY["battery"]
    x0, x1, z0, z1 = bx - bl / 2, bx + bl / 2, bz - bh / 2, bz + bh / 2
    p = {}
    for sy, k in ((1, "l"), (-1, "r")):
        ledge = _blk(x0, x1 + 0.002, sy * (bw / 2 - 0.006), sy * (bw / 2 + 0.0035), DECK, z0)
        guide = _blk(x0, x1 + 0.002, sy * (bw / 2 + 0.0005), sy * (bw / 2 + 0.0035), z0, z0 + 0.012)
        p["batt_rail_" + k] = ledge + guide
    stop = _blk(x1 + 0.0005, x1 + 0.0065, -bw / 2 + 0.008, bw / 2 - 0.008, DECK, z0 + 0.025)
    p["batt_stop"] = stop - _blk(x1, x1 + 0.007, -0.020, 0.020, z0 + 0.004, z0 + 0.020)
    t = WALL
    for sx in (1, -1):
        for sy in (1, -1):
            p["post_%s%s" % ("f" if sx > 0 else "b", "l" if sy > 0 else "r")] = _blk(sx * 0.055, sx * 0.061, sy * 0.0565, sy * 0.0625,
                                               DECK, TRAY_Z)
    tray = _blk(-0.062, 0.062, -0.0625, 0.0625, TRAY_Z, TRAY_Z + t)
    for sy in (1, -1):                                   # over the harness slots
        tray -= _blk(-0.026, 0.026, sy * 0.0535, sy * 0.07, TRAY_Z - 0.001, TRAY_Z + t + 0.001)
    tray -= _grid(-0.055, 0.055, -0.045, 0.045, TRAY_Z - 0.001, TRAY_Z + t + 0.001)
    p["tray"] = tray
    zt = TRAY_Z + t
    (jx, _, jz), (jl, jw, jh), _ = BAY["jetson"]
    xb = jx - jl / 2                                     # the bulkhead's front face
    bulk = _blk(xb - t, xb, -0.055, 0.055, zt, jz + jh / 2 + 0.005)
    bulk -= _grid(-0.045, 0.045, zt + 0.015, zt + 0.085, xb - t - 0.001, xb + 0.001, axis="x")
    foot = _blk(xb - t, xb + 0.013, -0.055, 0.055, zt, zt + t)
    p["bulkhead"] = bulk + foot
    return p


def torso_upper():
    """The chest: a box round the two shoulder PITCH actuators, carried the way
    the pitch-first humanoids carry theirs (Berkeley Humanoid Lite, Robotis
    OP3, the Unitree torsos): each actuator's body inside the chest, its FRONT
    face bolted to the chest's side wall, its output flange showing through it.

    So the side walls ARE the two mounting plates - 6 mm on the Ø97 front
    annulus, eight M3 heads outside. No thicker: shoulder_L's web starts 3 mm
    past the output flange and turns over those heads. The deck takes the box
    down to the waist adapter's Ø88 circle; a front panel, a back panel and a
    lid close it. Open, each wall would be a plate bending on its own under a
    pitch actuator's reaction; closed, that reaction goes round the box as
    shear, and the panels can be 3 mm webs with windows in them.
    """
    d, w, h = TORSO_UP
    parts = {}
    for cj in CHILD_OF["torso_upper"]:
        key = MOTOR_OF[cj[0]]
        pl_c = mount_plane(cj, cj[4])
        b_od = face_od(key, "front")
        ya = _pt(pl_c, IFACE[key]["hou_z"] * MM)[1]
        yb = _pt(pl_c, IFACE[key]["hou_z"] * MM + PLATE)[1]
        assert abs(max(abs(ya), abs(yb)) - w / 2) < 1e-6, \
            "TORSO_UP's width is not the side walls' outer faces"
        zc = cj[4][2]
        z_lo, z_hi = DECK + CHEST_BAND, zc - b_od / 2 - CHEST_BAND
        prof = (_bd.Pos(0, h / 2) * _bd.Rectangle(d, h)
                - _bd.Pos(0, zc) * _bd.Circle(b_od / 2)
                - _bd.Pos(0, (z_lo + z_hi) / 2) * _bd.RectangleRounded(
                    d - 2 * CHEST_BAND, z_hi - z_lo, 0.010))
        wall = _bd.extrude(_bd.Plane.XZ * prof, amount=PLATE)   # along -Y
        wall = _bd.Pos(0, min(ya, yb) - wall.bounding_box().min.Y, 0) * wall
        parts["wall_" + cj[0].split("_")[1]] = wall + pl_c * face_plate(key, "front", b_od)

    for sg in (1.0, -1.0):
        panel = _bd.extrude(_bd.Plane.YZ * (
            _bd.Pos(0, h / 2) * _bd.Rectangle(w, h)
            - _bd.Pos(0, h / 2) * _bd.RectangleRounded(
                w - 2 * CHEST_BAND, h - 2 * CHEST_BAND, 0.012)), amount=WALL)
        # Plane.YZ extrudes along +X, from 0 to WALL
        parts["front" if sg > 0 else "back"] = _bd.Pos(
            d / 2 - WALL if sg > 0 else -d / 2, 0, 0) * panel

    parts["lid"] = _bd.Pos(0, 0, h - WALL) * _bd.extrude(_bd.Plane.XY * (
        _bd.Rectangle(d, w)
        - _bd.RectangleRounded(d - 2 * CHEST_BAND, w - 2 * CHEST_BAND, 0.012)),
        amount=WALL)
    # Neither actuator is hollow (an 8 mm rod down an AK60-39's axis meets 5.3 cm3 of
    # it), so nothing passes THROUGH the waist: the harness from the pelvis -- legs in
    # through the window over each hip, the waist actuator's own -- leaves the pelvis's
    # open top beside the waist actuator, hangs a loop in the 40 mm under this deck for
    # the waist's +-45 deg, and comes up through these two slots, clear of the waist
    # flange (R50) and of the side walls.
    slots = [_bd.Pos(0, sy * CABLE_Y, 0) * _bd.SlotOverall(*CABLE_SLOT) for sy in (1, -1)]
    deck = _bd.extrude(_bd.Plane.XY * (_bd.Rectangle(d, w) - _bd.Circle(0.038) - slots[0] - slots[1]),
                       amount=DECK)
    parts["deck"] = deck - bolt_ring(TORSO_BC, TORSO_N, TORSO_TH + 0.0004,
                                     -0.005, DECK + 0.005)
    parts.update(_bay_frame())
    mounts, cuts = lidar.torso_upper_mounts()
    for k, c in cuts.items():
        parts[k] -= c
    parts.update(mounts)
    return parts


def mirrored(name):
    """The right arm is the left one reflected in the sagittal plane.

    Not re-derived: a left and a right bracket ARE mirror images, machined as
    two parts off one drawing, and deriving them twice is how the two halves
    drift apart. The actuators are placed by the same mirrored joint list, so
    what this has to reflect is only the metal between them.
    """
    def build():
        return {k: _bd.mirror(v, about=_bd.Plane.XZ)
                for k, v in BUILDERS[name[:-2] + "_L"]().items()}
    return build


BUILDERS = {"base_link": base_link,
            "waist_link": waist_link,
            "torso_upper": torso_upper,
            "shoulder_L": shoulder_L,
            "humerus_rot_L": humerus_rot_L,
            "upper_arm_L": upper_arm_L,
            "fore_arm_L": fore_arm_L,
            "forearm_rot_L": forearm_rot_L,
            "wrist_L": wrist_L,
            "wrist_dev_L": wrist_dev_L,
            "hook_L": hook_L}
for _n in ("shoulder", "upper_arm", "humerus_rot", "fore_arm",
           "forearm_rot", "wrist", "wrist_dev", "hook"):
    BUILDERS[_n + "_R"] = mirrored(_n + "_R")


def fuse(parts):
    """A link's parts as the one solid they are machined as.

    The union alone leaves a seam wherever two parts share a surface that is
    the same only to within arithmetic - a hull's tangent comes out 3e-7 rad
    off the plane it is tangent to - so the faces are merged again here, at
    1 um and 1e-4 rad, which no intended feature is finer than.
    """
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    live = [p for p in parts.values() if p is not None]
    out = live[0]
    for p in live[1:]:
        out = out + p
    u = ShapeUpgrade_UnifySameDomain(out.wrapped, True, True, False)
    u.SetLinearTolerance(1e-6)
    u.SetAngularTolerance(1e-4)
    u.Build()
    return _bd.Part(u.Shape())
