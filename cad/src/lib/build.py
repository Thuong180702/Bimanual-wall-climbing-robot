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

from .connect import (bolt_access, bolt_ring, cheek, face_od, face_plate,
                      hook_profile, mate_bore_r, output_plate,
                      plane_at, yoke_profile)
from .spec import (BODY_DIR, CLEAR, FIXED, HOOK, IFACE, JOINTS, MOTOR_OF,
                   MOUNT_AT, MOUNT_ROT, PLATE, SHELL, TORSO_BC, TORSO_LOW,
                   TORSO_LOW_R, TORSO_N, TORSO_TH, TORSO_UP, WAIST_DIA,
                   WALL)

MM = 0.001
PJOINT = {j[2]: j for j in JOINTS}
CHILD_OF = {}
for _j in JOINTS:
    CHILD_OF.setdefault(_j[1], []).append(_j)


def clear_bolts(web, faces, reach=0.05):
    """Take every bolt hole on through the web, so the bolts can be reached."""
    for pl, key, face in faces:
        web -= pl * bolt_access(key, face, reach)
    return web


def mount_plane(joint, origin):
    """That actuator's own frame, where it actually sits.

    MOUNT_AT slides an actuator along its own axis, which is kinematically
    free - the shoulder abduction one is slid 30 mm back so the raised arm's
    rotation actuator sweeps past it - so the flange is not always on the
    joint origin.
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
    web = cheek(a_od / 2, a[2] + 0.004, b_od / 2, cj[4][2],
                b[1], grow(pl_c, "Y") * web_t(c_key),
                bore=2 * mate_bore_r(c_key, "front"),
                lighten=[(0.0, (a[2] + cj[4][2]) / 2 + 0.008, 0.013)])
    web = clear_bolts(web, [(pl_p, p_key, "output"), (pl_c, c_key, "front")])
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
PILOT_CLEAR = 0.002         # radial clearance around the rear pilot boss; any
                            # more and the two legs beside it drop under 8 mm
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
    """
    pj, cj = PJOINT[name], CHILD_OF[name][0]
    p_key, c_key = MOTOR_OF[pj[0]], MOTOR_OF[cj[0]]
    dp, dc = IFACE[p_key], IFACE[c_key]
    pl_p, pl_c = mount_plane(pj, (0.0, 0.0, 0.0)), mount_plane(cj, cj[4])

    a_od = dp["out_od"] * MM                         # the parent's output face
    b_od = face_od(c_key, "rear")                    # the child's rear flange
    top = pl_p * output_plate(p_key, a_od, plate_t)
    side = pl_c * face_plate(c_key, "rear", b_od)

    # the cheek starts where the plate's spigot counterbore ends, so it is
    # clear of the spigot and still 3 mm deep into the plate
    y0 = _pt(pl_p, dp["out_z"] * MM + (stand if stand > 0.0 else
             dp["out_spigot_h"] * MM + CLEAR + 0.0005))[1]
    z_rear = _pt(pl_c, dc["rear_z"] * MM)[2]
    z_pilot = _pt(pl_c, (dc["rear_z"] - dc["rear_pilot_h"]) * MM)[2]
    r_cable = max(dp["out_bore"] - 2.0, 4.0) / 2 * MM
    r_pilot = dc["rear_pilot"] / 2 * MM + PILOT_CLEAR
    z_boss = (z_rear + z_pilot) / 2.0
    web = cheek(b_od / 2, z_rear, a_od / 2, 0.0, y0, grow(pl_p, "Y") * t,
                lighten=[(0.0, 0.0, r_cable), (0.0, z_boss, r_pilot)]
                + _span_holes(z_boss + r_pilot, -r_cable))
    web = clear_bolts(web, [(pl_p, p_key, "output"), (pl_c, c_key, "rear")])
    return {"plate_out": top, "plate_face": side, "cheek": web}


def fore_arm_L():
    """elbow output -> forearm-roll housing, rear face. AK80-64 to AK60-39."""
    return hang("fore_arm_L")


def upper_arm_L():
    """shoulder-flexion output -> shoulder-rotation housing, rear face.

    The same part as the forearm, one size up and 200 mm long instead of 145:
    two AK80-64, a Ø72 round end and a Ø97 straight one, with the rotation
    actuator's Ø71 pilot boss bored out of the web between them. The clear run
    of web left over is 97 mm, so it takes two lightening holes rather than
    the forearm's one.

    It is also the web that carries the shoulder-rotation actuator's reaction
    in TORSION - 120 N.m along the link at stall. A 12 x 97 plate takes that
    at 26 MPa and twists 0.7 deg over its length, so 12 mm holds here too.

    It stands 9 mm off its flange rather than the usual 2. Folded right back -
    shoulder flexion -170, which the climbing envelope does use - the outer
    rim of shoulder_L's own web sweeps through where this web would otherwise
    start: 585 mm3, measured, from -165 deg on. Six millimetres of stand-off
    buys the last 7 degrees of the range back for 77 g.
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
    z_bar = -(dp["od"] / 2 + 1.0) * MM              # below which its barrel ends

    c_p = w_c + sgn * t_c / 2
    web_p = cheek(t_c / 2, z_hand, a_od / 2, 0.0, w_p, sgp * t_p,
                  lighten=[(0.0, 0.0, max(dp["out_bore"] - 2.0, 4.0) / 2 * MM)],
                  across=ax_p, c=c_p)
    web_c = cheek(t_p / 2, z_hand + 0.004, b_od / 2, cj[4][2], w_c, sgn * t_c,
                  bore=2 * mate_bore_r(c_key, face), across=ax_c,
                  c=w_p + sgp * t_p / 2)
    # the parent's barrel fills everything inboard of its own flange, so above
    # z_bar the child leg's inboard edge is a straight line flush with that
    # flange instead of the tangent hull
    cut = _bd.Box(0.12, 0.12, 0.2, align=(_bd.Align.MIN,) * 3)
    o = w_p - 0.12 if sgp > 0 else w_p        # the barrel side of that flange
    if sgp < 0:
        cut = _bd.Pos(0, 0, 0) * cut
    web_c -= (_bd.Pos(-0.06, o, z_bar) if ax_p == "Y"
              else _bd.Pos(o, -0.06, z_bar)) * cut

    webs = clear_bolts(web_p + web_c,
                       [(pl_p, p_key, "output"), (pl_c, c_key, face)])
    return {"plate_out": top, "plate_face": side, "web": webs}


def wrist_L():
    """wrist-flexion output -> wrist-deviation housing, front face.

    Flexion turns about Y, deviation about X, the link runs along Z.
    """
    return corner("wrist_L")


def shoulder_L():
    """shoulder-abduction output -> shoulder-flexion housing, REAR face.

    A corner like the wrist, but with both actuators the big AK80-64 and the
    joints 104 mm apart. Two things differ.

    It bolts the flexion actuator's FAR flange, not its near one. Both of its
    faces are available; the near one is on the same side as its output, which
    is where upper_arm_L has to grow, and a 16 mm leg there would have cost
    that link an 11 mm stand-off and a 15 mm-thick Ø72 plate to carry it -
    about 130 g. The far flange puts this bracket on the inboard side instead,
    31 mm clear of the torso, and leaves the whole outboard side free.

    And the spine is 18 x 16, not the wrist's 16 x 14. The AK80-64 stalls at
    120 N.m about both of these axes: W = 768 mm3 across and 864 along, so
    156 and 139 MPa against 276 for 6061-T6, and 87 and 95 MPa at the 67 and
    82 N.m the hang measurement actually asks for.

    It also stands 6 mm off its own flange. The torso's shoulder yoke has to
    reach this same actuator from behind, and its band and this bracket's leg
    were sharing 2 mm of the same millimetres: 1036 mm3, measured. Six here
    and two off the yoke's thickness leaves 3 mm of air between them.
    """
    return corner("shoulder_L", face="rear", stand=0.006, plate_t=0.010,
                  spine={"X": 0.018, "Y": 0.016})


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
#   The palm, the lip and their 65 mm width are the numbers the controller and
#   the RL work were built on - urdf/climbing_robot.urdf carries them as two
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
    top = pl_p * output_plate(
        p_key, a_od,
        abs(y_lo - _pt(pl_p, dp["out_z"] * MM)[1]) + 0.004)
    z_neck = -(dp["od"] / 2 * MM + 0.0025)           # clear of the actuator
    t = HOOK["thk"]

    sgp = grow(pl_p, "Y")
    web = cheek(t / 2, z_neck, a_od / 2, 0.0, y_lo, sgp * HOOK_SPINE,
                lighten=[(0.0, 0.0, max(dp["out_bore"] - 2.0, 4.0) / 2 * MM)],
                across="Y", c=0.0)
    palm = _bd.extrude(_bd.Plane.YZ * hook_profile(
        y_lo + sgp * HOOK_SPINE / 2, HOOK_SPINE, z_neck, HOOK["width"] / 2,
        HOOK_WIDE, -HOOK["back"], HOOK_FILLET), amount=t)
    bb = palm.bounding_box()
    palm = _bd.Pos(-t / 2 - bb.min.X, 0, 0) * palm
    lip = _bd.Pos(t / 2 + HOOK["lip"] / 2, 0.0, -HOOK["back"] + t / 2) * \
        _bd.Box(HOOK["lip"], HOOK["width"], t)

    web = clear_bolts(web, [(pl_p, p_key, "output")])
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
MIN_ = (_bd.Align.MIN,) * 3
CMIN = (_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN)
FLOOR = 0.005               # the lower torso's floor, under the battery
TORSO_WEB = 0.010           # the shoulder wishbone
YOKE_W = 0.030              # ...and how wide its bands are. Wider and
                            # the band's outboard edge reaches into the
                            # ring shoulder_L carries on the flexion
                            # actuator's rear flange - measured, 51 mm3.
TORSO_FLANK = 0.005         # the upper torso's side panels
DECK = 0.006


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
    """The lower torso: a stadium-section shell carrying the waist actuator on
    a deck, with the battery and the electronics below it.

    The section is the 140 x 180 box cut back to R90 because the waist turns
    the whole upper body over this shell - stationary - and the elbow actuator
    sweeps a 93 mm circle about the waist axis, so nothing here may reach past
    90. The waist actuator is held by its REAR flange and sits nose-up in the
    top of the shell, its output finishing flush with the rim.
    """
    cj = CHILD_OF["base_link"][0]                    # waist_yaw
    c_key = MOTOR_OF[cj[0]]
    dc = IFACE[c_key]
    pl_c = mount_plane(cj, cj[4])
    d, w, h = TORSO_LOW

    wall = _prism(_low_section(), 0.0, h) - _prism(_low_section(SHELL), FLOOR, h)
    wall -= _windows(d, w, h, (0.30, 0.60))
    floor = _prism(_low_section(), 0.0, FLOOR)
    floor -= bolt_ring(0.110, 6, 0.030, -0.001, FLOOR + 0.001)

    z_rear = _pt(pl_c, dc["rear_z"] * MM)[2]
    deck = (pl_c * face_plate(c_key, "rear", 0.40)) & _prism(
        _low_section(), z_rear - PLATE - 0.005, PLATE + 0.005)
    deck -= bolt_ring(0.150, 8, 0.024, z_rear - PLATE - 0.01, z_rear + 0.01)
    return {"wall": wall, "floor": floor, "deck": deck}


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


def torso_upper():
    """The upper frame: a shoulder wishbone, two side panels and a deck.

    The two abduction actuators sit 250 mm apart on a lateral pair of Ø97
    faces, both normal to X, so ONE transverse plate can lie flat on both -
    a wishbone whose two arms are the same tangent hull the rest of the robot
    is drawn with, meeting over the waist. The side panels give it the
    fore-and-aft stiffness a 12 mm plate on its own has not got, and the deck
    takes it down to the waist adapter's Ø88 circle.
    """
    d, w, h = TORSO_UP
    parts = {}
    for cj in CHILD_OF["torso_upper"]:
        key = MOTOR_OF[cj[0]]
        pl_c = mount_plane(cj, cj[4])
        b_od = face_od(key, "front")
        parts["plate:" + cj[0]] = pl_c * face_plate(key, "front", b_od)
        sg = 1.0 if cj[4][1] > 0 else -1.0
        x_lo = _pt(pl_c, IFACE[key]["hou_z"] * MM)[0]
        prof = yoke_profile(-sg * YOKE_W, 0.0, 0.002,
                            cj[4][1], cj[4][2], b_od / 2,
                            mate_bore_r(key, "front"), -sg)
        arm = _bd.extrude(_bd.Plane.YZ * prof, amount=TORSO_WEB)
        bb = arm.bounding_box()
        arm = _bd.Pos(x_lo - bb.min.X, 0, 0) * arm
        arm = clear_bolts(arm, [(pl_c, key, "front")])
        # Two parts per arm - the ring round the actuator face and the strip
        # down to the deck - because the collision engine hulls each part, and
        # hulled whole the arm fills the triangle between strip and ring: the
        # space the raised upper arm's root swings through when the shoulders
        # flare (8 mm deep, on a third of the poses in the climbing envelope).
        ring = _bd.Location(_bd.Plane(origin=(x_lo - 0.01, cj[4][1], cj[4][2]),
                                      z_dir=(1, 0, 0))) * _bd.Cylinder(
            radius=b_od / 2 + 0.0005, height=TORSO_WEB + 0.02,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
        side = cj[0].split("_")[1]
        parts["wishbone_%s_ring" % side] = arm & ring
        parts["wishbone_%s_strip" % side] = arm - ring

    # the deck spans the frame, not just the bolt circle, so the side panels
    # land on it too and the yoke is not carrying them
    deck = _bd.extrude(_bd.Plane.XY * (
        _bd.Rectangle(d, w) - _bd.Circle(0.038)
        - _bd.Pos(0.052, 0.052) * _bd.Circle(0.022)
        - _bd.Pos(-0.052, 0.052) * _bd.Circle(0.022)
        - _bd.Pos(0.052, -0.052) * _bd.Circle(0.022)
        - _bd.Pos(-0.052, -0.052) * _bd.Circle(0.022)), amount=DECK)
    deck -= bolt_ring(TORSO_BC, TORSO_N, TORSO_TH + 0.0004, -0.005, DECK + 0.005)
    parts["deck"] = deck

    flank = None
    for sg in (1.0, -1.0):
        y0 = sg * (w / 2 - TORSO_FLANK) if sg > 0 else sg * w / 2
        p = _bd.Pos(0, y0, 0) * _bd.Box(d, TORSO_FLANK, cj[4][2],
                                        align=(_bd.Align.CENTER, _bd.Align.MIN,
                                               _bd.Align.MIN))
        flank = p if flank is None else flank + p
    flank -= _windows(d, w, cj[4][2], (0.26, 0.50, 0.74))
    parts["flanks"] = flank
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
    live = [p for p in parts.values() if p is not None]
    out = live[0]
    for p in live[1:]:
        out = out + p
    return out
