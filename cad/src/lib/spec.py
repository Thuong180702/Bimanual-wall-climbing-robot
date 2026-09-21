"""The v2 robot: which actuator sits at which joint, and where the joints are.

Frame (URDF): +X forward (toward the wall), +Y left, +Z up. Metres.
Joint sign convention, unchanged from v1: about Y, NEGATIVE = flexion.

WHAT MOVED FROM v1, and why. Every offset below that differs from v1 differs
because two real motor bodies could not occupy the same space. Two motors on
perpendicular axes clear each other only if the distance between their axis
LINES is at least r1 + r2 + 3 mm, or if their finite bodies miss along the
link. Each change is checked by check_clearance() and by a CAD interference
run on the built assembly.

  torso_upper -> shoulder_abduction   z 0.165 -> 0.269   (+104 mm)
  shoulder    -> shoulder_flexion     z 0     -> -0.104  (-104 mm)
      the two together leave the shoulder FLEXION axis at world z = 0.405,
      exactly where v1 had it: the arm kinematics below the shoulder are
      untouched, the torso got taller instead. Two AK80-64 (r 49 each) on
      perpendicular axes need 101 mm between those axes; 104 given.
  wrist       -> wrist_deviation      z -0.020 -> -0.084 (-64 mm)
      two AK60-39 (r 39.5 each) need 82 mm; 84 given.
  wrist_dev   -> finger               z -0.020 -> -0.072 (-52 mm)
      AK60-39 (r 39.5) and AK45-36 (r 27.5) need 70 mm; 72 given.
  torso_up width 0.240 -> 0.145, height 0.165 -> 0.315
      the abduction actuator is Ø98 on a lateral axis at y = 0.125, so it
      reaches in to y = 0.076 and the frame has to end before that; and the
      frame runs 46 mm past the shoulder axis so the shoulder housings sit in
      the torso's silhouette instead of perched on its lid.

  Net effect: the hand sits 95 mm further from the wrist than in v1, and the
  robot is 95 mm taller. Everything between the shoulder and the wrist is
  dimensionally identical to v1.

THE CROSS-AXIS ACTUATORS NOW ALTERNATE. All four of them - shoulder flexion,
elbow, wrist flexion, finger - used to have their bodies inboard, so every one
of them hung its own mass to the same side and none of it cancelled: measured
in tools/balance.py, the arm's centre of mass sat 13.5 mm inboard of the axis
it hangs on. The elbow and the finger are turned round, so the four now go
in, out, in, out down the arm. It is what OpenArm does with the joint origins
themselves - its lateral offsets read -30.1, +30.1, +31.5, -31.5, +37.5, -37.5.

TWO LIMITS ALSO MOVED, both measured in tools/check_fit.py:

  shoulder_flexion  -180 -> -170 deg.  Past -173 the shoulder ROTATION
      actuator grazes the ABDUCTION actuator. v1's own README already said not
      to go there - at -180 the torso hangs flat against the wall and rams the
      rung below - so this costs nothing that was being used.
  elbow             -120 -> -110 deg.  -118 was where the two ACTUATOR bodies
      touch; -112.3 is where the two BRACKETS do - at full fold the forearm
      web sweeps into the Ø72 plate on the shoulder-rotation flange, 434 mm^3
      of it, measured in tools/check_pair.py. -110 leaves ~2.8 mm of gap. The
      climbing envelope only ever asks for elbow -90..0, so this costs nothing
      that is used; the upper arm is 60 mm from the rotation flange to the
      elbow axis and two Ø98 actuators that close do not fold further.

And shoulder_abduction's actuator slides 30 mm back along its own axis
(MOUNT_AT) rather than sitting on the joint origin, so the raised arm's
rotation actuator sweeps past it. Sliding an actuator along its own axis is
kinematically free. 40 mm was a guess; tools/check_swing.py sweeps that pair
over BOTH joints between them and finds the clash starts below 27 mm, so 30
is the number with a margin - and it brings 1.7 kg of shoulder actuator
10 mm back toward the middle of the torso.
"""
import math

from .motors import MOTORS

MM = 0.001

# ---------------------------------------------------------------- actuators
# joint suffix -> catalogue key. Sized in tools/size.py against the measured
# hang-equilibrium torque; margins are re-checked on the finished model.
PICK = {
    "waist_yaw":          "AK60-39",
    "shoulder_abduction": "AK80-64",
    "shoulder_flexion":   "AK80-64",
    "shoulder_rotation":  "AK80-64",
    "elbow":              "AK80-64",
    "forearm_roll":       "AK60-39",
    "wrist_flexion":      "AK60-39",
    "wrist_deviation":    "AK60-39",
    "finger":             "AK45-36",
}

# Mounting interfaces, measured off the vendor STEP (tools/holes.py), in mm.
# out_bc / hou_bc are bolt-circle DIAMETERS; hou_z is the housing face the
# parent link bolts to, in the canonical motor frame (+Z = axis, z=0 = output).
IFACE = {
    # out_*  the OUTPUT flange the child bolts to: the face's z, its outside
    #        diameter, the centering spigot standing proud of it, the through
    #        bore for cables, and the bolt circle.
    # hou_*  the FRONT housing face the parent bolts to: z, the annulus it
    #        offers (id..od), any boss rising out of it, and its bolt circle.
    # rear_* the REAR flange, at the other end of the barrel: same again, plus
    #        the pilot boss that centres a plate on it. Bolting BOTH ends is
    #        what makes the bracket a housing instead of a collar - the barrel
    #        between them carries the load, and the actuator is held, not
    #        merely avoided.
    # barrel the body the parent's cup has to clear.
    # Every number read off the vendor STEP by tools/profile.py; nothing here
    # is from a drawing or a guess.
    "AK80-64": dict(
        od=98.0, length=61.9,
        out_z=-1.0, out_od=72.0, out_bore=21.0, out_spigot=35.4, out_spigot_h=1.0,
        out_bc=28.0, out_n=6, out_thread=4.0,
        hou_z=-8.0, hou_id=79.8, hou_od=97.0, hou_boss=80.0, hou_boss_h=3.0,
        hou_bc=89.0, hou_n=8, hou_thread=3.0,
        rear_z=-50.7, rear_id=71.5, rear_od=97.0, rear_pilot=71.0, rear_pilot_h=11.2,
        rear_bc=85.0, rear_n=8, rear_thread=3.0,
        pin_bc=0.0, pin_n=0, pin_d=0.0, pin_h=0.0,
        barrel=98.0),
    "AK60-39": dict(
        od=79.0, length=67.0,
        out_z=-0.5, out_od=54.0, out_bore=20.1, out_spigot=40.6, out_spigot_h=0.5,
        out_bc=30.0, out_n=6, out_thread=4.0,
        hou_z=-6.0, hou_id=59.4, hou_od=78.0, hou_boss=59.4, hou_boss_h=0.0,
        hou_bc=71.0, hou_n=8, hou_thread=3.0,
        rear_z=-48.5, rear_id=59.3, rear_od=78.0, rear_pilot=58.0, rear_pilot_h=18.2,
        rear_bc=68.0, rear_n=8, rear_thread=3.0,
        pin_bc=0.0, pin_n=0, pin_d=0.0, pin_h=0.0,
        barrel=79.0),
    "AK45-36": dict(
        od=55.0, length=54.0,
        out_z=-2.0, out_od=36.0, out_bore=20.8, out_spigot=0.0, out_spigot_h=0.0,
        out_bc=27.0, out_n=6, out_thread=2.5,
        # three dowel pins stand 2 mm proud of the output face on a 24 mm
        # circle - this actuator has no spigot, they are what locates the
        # child plate, and a plate without holes for them will not sit down.
        pin_bc=24.0, pin_n=3, pin_d=4.0, pin_h=2.0,
        hou_z=-3.0, hou_id=42.6, hou_od=53.0, hou_boss=42.6, hou_boss_h=0.0,
        hou_bc=48.0, hou_n=6, hou_thread=3.0,
        rear_z=-54.0, rear_id=18.0, rear_od=54.4, rear_pilot=0.0, rear_pilot_h=0.0,
        rear_bc=48.0, rear_n=6, rear_thread=3.0,
        barrel=55.0),
}

# --------------------------------------------------------------- kinematics
# (name, parent, child, axis, xyz from the parent frame, lower, upper)
# body_dir: unit vector, in the PARENT frame, from the joint origin toward the
#           motor BODY. The child structure occupies the other side.
D = math.radians


def _arm(s, sg):
    """One arm. sg = +1 left, -1 right; the limits mirror about the sagittal plane."""
    return [
        ("shoulder_%s_abduction" % s, "torso_upper", "shoulder_%s" % s,
         (1, 0, 0), (0.0, sg * 0.125, 0.269),
         (D(0), D(180)) if sg > 0 else (D(-180), D(0)), (-1, 0, 0)),
        ("shoulder_%s_flexion" % s, "shoulder_%s" % s, "upper_arm_%s" % s,
         (0, 1, 0), (0.0, sg * 0.030, -0.104), (D(-170), D(60)), (0, -sg, 0)),
        ("shoulder_%s_rotation" % s, "upper_arm_%s" % s, "humerus_rot_%s" % s,
         (0, 0, 1), (0.0, 0.0, -0.200),
         (D(-70), D(90)) if sg > 0 else (D(-90), D(70)), (0, 0, 1)),
        ("elbow_%s" % s, "humerus_rot_%s" % s, "fore_arm_%s" % s,
         (0, 1, 0), (0.0, 0.0, -0.060), (D(-110), D(0)), (0, sg, 0)),
        ("forearm_%s_roll" % s, "fore_arm_%s" % s, "forearm_rot_%s" % s,
         (0, 0, 1), (0.0, 0.0, -0.145), (D(-85), D(85)), (0, 0, 1)),
        ("wrist_%s_flexion" % s, "forearm_rot_%s" % s, "wrist_%s" % s,
         (0, 1, 0), (0.0, 0.0, -0.060), (D(-25), D(25)), (0, -sg, 0)),
        ("wrist_%s_deviation" % s, "wrist_%s" % s, "wrist_dev_%s" % s,
         (1, 0, 0), (0.0, 0.0, -0.084),
         (D(-30), D(20)) if sg > 0 else (D(-20), D(30)), (-1, 0, 0)),
        ("finger_%s" % s, "wrist_dev_%s" % s, "hook_%s" % s,
         (0, 1, 0), (0.0, 0.0, -0.072), (D(-90), D(20)), (0, sg, 0)),
    ]


JOINTS = ([("waist_yaw", "base_link", "waist_link", (0, 0, 1),
            (0.0, 0.0, 0.200), (D(-45), D(45)), (0, 0, -1))]
          + _arm("L", 1) + _arm("R", -1))

FIXED = [("torso_fixed", "waist_link", "torso_upper", (0.0, 0.0, 0.040))]

# ------------------------------------------------------------------ structure
TORSO_LOW = (0.140, 0.180, 0.200)     # d, w, h; w 0.210 -> 0.180 to clear the elbow motors
# ...and the four vertical corners are rounded away, because the waist turns the
# arms over a STATIONARY lower torso: with the arms down, the elbow motor's
# inboard face sweeps a circle of radius 93.1 mm about the waist axis, so every
# point of the lower torso has to stay inside R = 90 mm. A 140 x 180 box corner
# reaches 114 mm. The section is therefore that box intersected with a 180 mm
# circle - a stadium, flat front and back, round left and right.
TORSO_LOW_R = 0.090
TORSO_UP = (0.150, 0.145, 0.315)      # w 0.240 -> 0.165, h 0.165 -> 0.260: see below
WAIST_DIA, WAIST_H = 0.100, 0.040
HOOK = dict(back=0.110, lip=0.040, thk=0.012, width=0.065)
# The one mating interface this design INVENTS rather than measures: the
# bolted joint between the waist adapter and the upper torso frame. Every
# other joint lands on a face an actuator already has.
TORSO_BC, TORSO_N, TORSO_TH = 0.088, 8, 0.004   # Ø88 circle, 8 x M4
PLATE = 0.006                          # 6 mm 6061 mounting plates
FIT = 0.0005                           # radial slip fit around an actuator barrel
CLEAR = 0.0015                         # the keep-out every bracket is cut back by
WALL = 0.003                           # 3 mm webs and tube walls: at 120 N.m a
                                       # 70x40x3 box section sees 15 MPa, 6061-T6
                                       # yields at 276, so the wall is set by
                                       # handling and local buckling, not stress.
SHELL = 0.0025                         # 2.5 mm torso panels
AL = 2700.0                            # 6061-T6, kg/m^3

# Non-actuator, non-structural masses carried in the lower torso.
PAYLOAD = [("battery_48V_6Ah", 1.60, (0.0, 0.0, 0.070)),
           ("compute_and_can", 0.55, (0.0, 0.0, 0.140)),
           ("dcdc_and_wiring", 0.35, (0.0, 0.0, 0.040))]

# Where along its own axis an actuator's OUTPUT FLANGE sits, measured from the
# joint origin in the body_dir direction, metres. The joint axis is a line, so
# sliding the actuator along it changes nothing kinematically - it is free
# clearance. shoulder_abduction uses it: with a 98 mm rotation actuator on the
# raised arm, the abduction actuator has to sit back out of its sweep.
MOUNT_AT = {}
for _s in "LR":
    MOUNT_AT["shoulder_%s_abduction" % _s] = 0.030

# The hooks are bolted on TURNED OVER: a hook's frame is its finger joint's
# frame turned 180 deg about the forearm axis (z). v1 learnt this the hard way
# (commit dc2f07b): with the arm raised overhead to hang, an unturned hook's
# throat faces AWAY from the wall, and turning it back through the arm pins
# the shoulder rotation at its stop, which aims the elbow hinge at the wall -
# the elbows folded 44 mm inboard on every pull-up. Everything read in
# hook-LOCAL coordinates (the throat, the pads, the finger angle) is unchanged
# by it. Radians about the child link's own z; every other link is 0.
MOUNT_ROT = {"hook_L": math.pi, "hook_R": math.pi}

PARENT = {j[2]: j[1] for j in JOINTS}
PJOINT = {j[2]: j for j in JOINTS} | {f[2]: f[1] for f in FIXED}
AXIS = {j[0]: j[3] for j in JOINTS}
MOTOR_OF = {}
for _j in JOINTS:
    _k = _j[0].replace("_L_", "_").replace("_R_", "_")
    _k = _k[:-2] if _k.endswith(("_L", "_R")) else _k
    MOTOR_OF[_j[0]] = PICK[_k]


def fk(q=None):
    """World placement of every link frame at joint angles q (default all zero).

    Returns {link: (origin xyz, 3x3 rotation)} with the base at the origin.
    Every link frame has identity rotation at q=0 except the hooks, which are
    turned by MOUNT_ROT; a joint's axis is in its CHILD's frame, as in MJCF.
    """
    q = q or {}
    out = {"base_link": ((0.0, 0.0, 0.0), _I)}
    todo = list(JOINTS) + [(f[0], f[1], f[2], None, f[3], None, None) for f in FIXED]
    while todo:
        rest = []
        for j in todo:
            _, par, ch, ax, off, _, _ = j
            if par not in out:
                rest.append(j)
                continue
            po, pR = out[par]
            o = tuple(p + sum(pR[i][k] * off[k] for k in range(3))
                      for i, p in enumerate(po))
            R = pR if ax is None else _mul(
                _mul(pR, _rot((0, 0, 1), MOUNT_ROT.get(ch, 0.0))), _rot(ax, q.get(j[0], 0.0)))
            out[ch] = (o, R)
        if len(rest) == len(todo):
            raise RuntimeError("cycle in the joint list")
        todo = rest
    return out


_I = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _rot(ax, a):
    c, s = math.cos(a), math.sin(a)
    x, y, z = ax
    return ((c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s),
            (y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s),
            (z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)))


def motor_placements(q=None):
    """Every actuator as (joint, motor key, world FLANGE, world axis, world body_dir).

    The body runs from the flange along body_dir for the motor's length. The
    flange sits MOUNT_AT[joint] along body_dir from the joint origin (0 for
    almost every joint), so for most of them the flange IS the joint origin.
    """
    F = fk(q)
    out = []
    for name, par, ch, ax, off, _lim, _bd in JOINTS:
        po, pR = F[par]
        o = F[ch][0]
        cR = F[ch][1]                    # the axis is in the child's frame
        wax = tuple(sum(cR[i][k] * ax[k] for k in range(3)) for i in range(3))
        bd = BODY_DIR[name]
        wbd = tuple(sum(pR[i][k] * bd[k] for k in range(3)) for i in range(3))
        a = MOUNT_AT.get(name, 0.0)
        flange = tuple(o[k] + wbd[k] * a for k in range(3))
        out.append((name, MOTOR_OF[name], flange, wax, wbd))
    return out


BODY_DIR = {j[0]: j[6] for j in JOINTS}
