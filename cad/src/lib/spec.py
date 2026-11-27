"""The v2 robot: which actuator sits at which joint, and where the joints are.

Frame (URDF): +X forward (toward the wall), +Y left, +Z up. Metres.
Joint sign convention, unchanged from v1: about Y, NEGATIVE = flexion.

WHAT MOVED FROM v1, and why. Every offset below that differs from v1 differs
because two real motor bodies could not occupy the same space. Two motors on
perpendicular axes clear each other only if the distance between their axis
LINES is at least r1 + r2 + 3 mm, or if their finite bodies miss along the
link. Each change is checked by check_clearance() and by a CAD interference
run on the built assembly.

  The shoulder: see THE SHOULDER IS PITCH-FIRST below.
  wrist       -> wrist_deviation      z -0.020 -> -0.084 (-64 mm)
      two AK60-39 (r 39.5 each) need 82 mm; 84 given.
  wrist_dev   -> finger               z -0.020 -> -0.072 (-52 mm)
      AK60-39 (r 39.5) and AK45-36 (r 27.5) need 70 mm; 72 given.

  Net effect: the hand sits 116 mm further from the wrist than in v1.
  Everything between the shoulder and the wrist is dimensionally identical
  to v1.

THE SHOULDER IS PITCH-FIRST, like the humanoids it was checked against.
v2 first stacked it the other way - abduction on the torso, flexion 104 mm
BELOW it - and with the arm raised to hang, the arm grew out of a pivot under
the abduction motor and folded back up past it. On the true CAD meshes that
cost: flaring the raised arm swung its root 105 mm inward and drove it into
the torso from -38 deg, and the -170 flexion stop pressed it on the torso.
Twelve published humanoids, measured off their own URDF/MJCF (Unitree H1,
H1-2, G1; Booster T1; Fourier N1; PNDbotics Adam Lite; Berkeley Humanoid Lite;
ToddlerBot; Robotis OP3; iCub; Apptronik Apollo; PAL TALOS): ten put shoulder
PITCH on the torso, then ROLL offset SIDEWAYS by 27-78 mm at almost the same
height (|dz| <= 15 mm), then YAW down the upper arm. None drops the second
shoulder axis below the first.

  torso_upper -> shoulder_flexion    y 0.085, z 0.165: a lateral axis, the
      actuator body inboard in the torso, at v1's shoulder height
  shoulder    -> shoulder_abduction  y +0.070, same height: the roll actuator
      sits outboard of the pitch output. 70 = its 49 mm radius + 1.5 of air +
      a 16 mm web + the 2 mm the output plate's spigot recess takes + 1.5.
      The web is what carries 120 N.m of roll torque into the pitch output
      (16 mm is the spine the old shoulder bracket needed for the same load);
      65 left it 4.5 mm. The survey spans 27-78; Fourier N1 uses 78.
  => the arm's root is at y 0.155, z 0.165, exactly v1's, so the hands, the
     reach and every lesson's goals are v1's; and a flare now turns the raised
     arm about a point on its own root instead of swinging the root inward.

THE CROSS-AXIS ACTUATORS NOW ALTERNATE. All four of them - shoulder flexion,
elbow, wrist flexion, finger - used to have their bodies inboard, so every one
of them hung its own mass to the same side and none of it cancelled: measured
in tools/balance.py, the arm's centre of mass sat 13.5 mm inboard of the axis
it hangs on. The elbow and the finger are turned round, so the four now go
in, out, in, out down the arm. It is what OpenArm does with the joint origins
themselves - its lateral offsets read -30.1, +30.1, +31.5, -31.5, +37.5, -37.5.

TWO LIMITS ALSO MOVED, both measured in tools/check_fit.py:

  shoulder_flexion  -180 -> -170 deg.  v1's own README said not to go there -
      at -180 the torso hangs flat against the wall and rams the rung below.
      (On the abduction-first shoulder the rotation actuator also grazed the
      abduction one past -173; pitch-first, the true CAD is clear to -180.)
  elbow             -120 -> -110 deg.  -118 was where the two ACTUATOR bodies
      touch; -112.3 is where the two BRACKETS do - at full fold the forearm
      web sweeps into the Ø72 plate on the shoulder-rotation flange, 434 mm^3
      of it, measured in tools/check_pair.py. -110 leaves ~2.8 mm of gap. The
      climbing envelope only ever asks for elbow -90..0, so this costs nothing
      that is used; the upper arm is 60 mm from the rotation flange to the
      elbow axis and two Ø98 actuators that close do not fold further.

MOUNT_AT can slide an actuator along its own axis, which is kinematically
free. The abduction-first shoulder needed it (30 mm, so the raised arm's
rotation actuator swept past the abduction actuator); pitch-first does not.
"""
import math
import os

from .motors import MOTORS

MM = 0.001

# Which robot: "v2", the arms-only robot the paper trains, or "full", the same robot
# with legs and the arm actuators that hang its extra mass
# (docs/superpowers/specs/2026-11-24-legs-design.md). Set by the environment, so every
# tool builds either one unchanged; "full" writes its model under OUT_DIR.
ROBOT = os.environ.get("ROBOT", "v2")
assert ROBOT in ("v2", "full"), ROBOT
OUT_DIR = "urdf" if ROBOT == "v2" else "urdf/full"

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
    # RobStride 04, read off robstride_04.step the same way (vendor z + 25.5 = here):
    # the output is the planetary carrier, 9 x M5 and three 6 mm dowels standing
    # 2.7 mm proud on a 36 mm circle; the housing face is the front ring, 12 x M3 on
    # 115.2 (an inner ring of 12 x M3 on 70 is the gearbox's own); the rear cover
    # carries 3 x M3 on 68.
    "RS04": dict(
        od=120.0, length=55.7,
        out_z=0.0, out_od=49.6, out_bore=30.2, out_spigot=0.0, out_spigot_h=0.0,
        out_bc=36.0, out_n=9, out_thread=5.0,
        pin_bc=36.0, pin_n=3, pin_d=6.0, pin_h=2.7,
        hou_z=-0.5, hou_id=83.2, hou_od=120.0, hou_boss=0.0, hou_boss_h=0.0,
        hou_bc=115.2, hou_n=12, hou_thread=3.0,
        rear_z=-52.7, rear_id=27.7, rear_od=92.0, rear_pilot=0.0, rear_pilot_h=0.0,
        rear_bc=68.0, rear_n=3, rear_thread=3.0,
        barrel=120.0),
    # MyActuator RMD-X12-P20-320, off MyActuator's 2D drawing (the STEP has no holes): the
    # output is the Ø85 flange, 16 x M4 on 72, 3 mm proud of the Ø109 front cover, which
    # stands 12.5 mm out of the housing face: a Ø124 shoulder with 12 x M5 on 116.5 -- the
    # only face it mounts by (its back is a vented cover on three M3 screws, not a flange)
    "X12-320": dict(
        od=124.0, length=85.0,
        out_z=0.0, out_od=85.0, out_bore=12.6, out_spigot=0.0, out_spigot_h=0.0,
        out_bc=72.0, out_n=16, out_thread=4.0,
        # the Ø109 cover (0/-0.1) is a PILOT: a mounting flange slides over it at a slip fit
        # and stands 0.5 mm past its top, so the M5 heads sit above it (they reach 4 mm
        # inside the bolt circle and the cover is 3.75 inside it) -- a 13 mm flange
        hou_z=-15.5, hou_id=109.0, hou_od=124.0, hou_boss=109.0, hou_boss_h=12.5, hou_pilot=True,
        hou_bc=116.5, hou_n=12, hou_thread=5.0,
        rear_z=-85.0, rear_id=0.0, rear_od=124.0, rear_pilot=0.0, rear_pilot_h=0.0,
        rear_bc=0.0, rear_n=0, rear_thread=3.0,
        pin_bc=0.0, pin_n=0, pin_d=0.0, pin_h=0.0,
        barrel=124.0),
    # MyActuator RMD-X15-P20-450, off its 2D drawing: the output is the Ø115 flange, 16 x M5
    # on 100 and two Ø5 dowel holes 100 apart, 3 mm proud of the Ø144 cover (a 0/-0.04
    # pilot) standing 10 mm out of the Ø166 shoulder, 12 x M5 on 154; the rear face, 69 mm
    # back, has 12 x M5 on 157 round a Ø124 connector hub standing 21 mm out of it
    "X15-450": dict(
        od=166.0, length=90.0,
        out_z=0.0, out_od=115.0, out_bore=21.5, out_spigot=0.0, out_spigot_h=0.0,
        out_bc=100.0, out_n=16, out_thread=5.0,
        pin_bc=100.0, pin_n=2, pin_d=5.0, pin_h=0.0,
        hou_z=-13.0, hou_id=144.0, hou_od=166.0, hou_boss=144.0, hou_boss_h=10.0, hou_pilot=True,
        hou_bc=154.0, hou_n=12, hou_thread=5.0,
        rear_z=-69.0, rear_id=124.0, rear_od=166.0, rear_pilot=124.0, rear_pilot_h=21.0,
        rear_bc=157.0, rear_n=12, rear_thread=5.0,
        barrel=166.0),
    # CubeMars AKE90-8 KV35, off cubemars_ake90_8_kv35.step (vendor axis on Y, output at
    # y = -11: here it is turned and lowered 11 mm) and its drawing: the output is the
    # Ø53 (-0.02) disc, 6 x M5 and three Ø4 dowel holes on Ø40, 1 mm proud of the
    # housing face, the top of a Ø82 ring with 8 x M4 on 73.5 standing 8 mm out of the
    # Ø107.5 body; the back is a spoked cover over the encoder magnet, where the driver
    # board goes -- `length` takes 11 mm for that board (estimate), no rear flange
    "AKE90-8": dict(
        od=107.5, length=54.5,
        out_z=0.0, out_od=53.0, out_bore=22.0, out_spigot=0.0, out_spigot_h=0.0,
        out_bc=40.0, out_n=6, out_thread=5.0,
        pin_bc=40.0, pin_n=3, pin_d=4.0, pin_h=0.0,
        hou_z=-1.0, hou_id=67.0, hou_od=82.0, hou_boss=0.0, hou_boss_h=0.0,
        hou_bc=73.5, hou_n=8, hou_thread=4.0,
        rear_z=-42.5, rear_id=75.0, rear_od=105.5, rear_pilot=0.0, rear_pilot_h=0.0,
        rear_bc=0.0, rear_n=0, rear_thread=3.0,
        barrel=107.5),
}

# --------------------------------------------------------------- kinematics
# (name, parent, child, axis, xyz from the parent frame, lower, upper)
# body_dir: unit vector, in the PARENT frame, from the joint origin toward the
#           motor BODY. The child structure occupies the other side.
D = math.radians


# The waist turns where a person's does, about navel height: 0.10 m above the lower
# torso's floor (where hip joints will go), 0.305 m under the shoulders -- for the
# 1.40 m person this arm is (upper arm 0.186 H, forearm 0.146 H), 0.07 H and 0.218 H.
# It sat 0.20 m up until 2026-11-11, 0.205 m under the shoulders. The axis is
# vertical, so lowering it moves nothing on the arms: DROP comes off the lower torso
# and goes on the chest, and everything on the chest keeps its place in the world.
DROP = 0.100
SH_Y, SH_Z = 0.085, 0.165 + DROP   # shoulder pitch axis in the torso frame
SH_ROLL = 0.070                 # pitch output -> roll axis, sideways
# each arm joint's origin along the arm from the one before (the comments in _arm and at
# the top of this file say what sets each); the full robot's bigger actuators re-space
# them by the same rules (ROBOT == "full" below)
ARM_OFF = dict(roll=SH_ROLL, rot=0.200, elbow=0.060, froll=0.145, wflex=0.060, wdev=0.084,
               finger=0.072)


def _arm(s, sg):
    """One arm. sg = +1 left, -1 right; the limits mirror about the sagittal plane."""
    return [
        # pitch first, on the torso: its body inboard, its output facing out
        ("shoulder_%s_flexion" % s, "torso_upper", "shoulder_%s" % s,
         (0, 1, 0), (0.0, sg * SH_Y, SH_Z), (D(-170), D(60)), (0, -sg, 0)),
        # then roll, outboard at the same height. Its body points forward with
        # the arm down, so with the arm raised to hang it points AWAY from the
        # wall. The flare of a raised arm is POSITIVE abduction on the left.
        # Inward it stops at -12: the upper arm's cheek meets this shoulder
        # bracket's web from -15 (tools/check_pair.py). The humanoids surveyed
        # stop their roll inward at -7 to -25 (N1 -10, BHL -15, H1 -19.5).
        ("shoulder_%s_abduction" % s, "shoulder_%s" % s, "upper_arm_%s" % s,
         (1, 0, 0), (0.0, sg * ARM_OFF["roll"], 0.0),
         (D(-12), D(180)) if sg > 0 else (D(-180), D(12)), (1, 0, 0)),
        ("shoulder_%s_rotation" % s, "upper_arm_%s" % s, "humerus_rot_%s" % s,
         (0, 0, 1), (0.0, 0.0, -ARM_OFF["rot"]),
         (D(-70), D(90)) if sg > 0 else (D(-90), D(70)), (0, 0, 1)),
        ("elbow_%s" % s, "humerus_rot_%s" % s, "fore_arm_%s" % s,
         (0, 1, 0), (0.0, 0.0, -ARM_OFF["elbow"]), (D(-110), D(0)), (0, sg, 0)),
        ("forearm_%s_roll" % s, "fore_arm_%s" % s, "forearm_rot_%s" % s,
         (0, 0, 1), (0.0, 0.0, -ARM_OFF["froll"]), (D(-85), D(85)), (0, 0, 1)),
        ("wrist_%s_flexion" % s, "forearm_rot_%s" % s, "wrist_%s" % s,
         (0, 1, 0), (0.0, 0.0, -ARM_OFF["wflex"]), (D(-25), D(25)), (0, -sg, 0)),
        ("wrist_%s_deviation" % s, "wrist_%s" % s, "wrist_dev_%s" % s,
         (1, 0, 0), (0.0, 0.0, -ARM_OFF["wdev"]),
         (D(-30), D(20)) if sg > 0 else (D(-20), D(30)), (-1, 0, 0)),
        ("finger_%s" % s, "wrist_dev_%s" % s, "hook_%s" % s,
         (0, 1, 0), (0.0, 0.0, -ARM_OFF["finger"]), (D(-90), D(20)), (0, sg, 0)),
    ]


JOINTS = ([("waist_yaw", "base_link", "waist_link", (0, 0, 1),
            (0.0, 0.0, 0.200 - DROP), (D(-45), D(45)), (0, 0, -1))]
          + _arm("L", 1) + _arm("R", -1))

FIXED = [("torso_fixed", "waist_link", "torso_upper", (0.0, 0.0, 0.040))]

# The pelvis (base_link) is drawn ready for legs: each hip's first actuator, a PITCH
# AK80-64 as on Unitree G1 (pitch -> roll -> yaw), bolts its front face to a pelvis
# side wall from inside, its output showing through -- the way the shoulders sit in
# the chest. Only the mounting is drawn: no hip actuator, no leg, no mass, no joint in
# the model yet. HIP_Y puts the walls' outer faces at y = +-71 mm, so the wall plate's
# edge stays 86 mm from the waist axis (TORSO_LOW_R) and a hanging forearm (88 mm)
# clears the pelvis as the waist turns; the hips are then 146 mm apart (a 1.40 m
# person's ~150). HIP_Z puts each body's top 1.5 mm under the waist actuator's pilot.
HIP_KEY = "AK80-64"
HIP_Y, HIP_Z = 0.073, -0.020
HIPS = [("hip_%s_pitch" % s, "base_link", None, (0, 1, 0), (0.0, sg * HIP_Y, HIP_Z),
         (D(-150), D(60)), (0, -sg, 0)) for s, sg in (("L", 1), ("R", -1))]
LOW_Z0 = -0.076                       # the pelvis floor's underside, under the hips

# ------------------------------------------------------------------ legs (ROBOT == "full")
# Two 6-DOF legs, G1's order: hip pitch (on the pelvis) -> roll -> yaw -> knee -> ankle
# pitch -> ankle roll (docs/superpowers/specs/2026-11-24-legs-design.md). A 1.40 m
# person's leg scaled toward G1: hip pitch axis to knee THIGH, knee to ankle CALF, ankle
# axis to sole SOLE. Hip pitch and knee are RobStride 04 (fast enough to walk), hip roll
# CubeMars AKE90-8 (an AK80-64 held 0.90x walking: 134 N.m on one foot at 1.5 x the
# weight; the user's choice 2026-11-25), hip yaw AK60-39; the ankle has no motor at its joints: two AK60-39 high on the
# calf (ANKLE_DRIVE) turn cranks of CRANK_R that push two rods to the foot, as on G1.
THIGH, CALF, SOLE = 0.320, 0.320, 0.060
# pitch output -> roll axis, sideways (the X12 hips, running, 2026-11-25): the output
# stands 2.5 mm past its flange's ring, the web starts 2 mm past the output, 16.5 mm of
# web, 1.5 of air and the roll X12's 62 radius
HIP_ROLL = 0.002 + 0.0165 + 0.0015 + 0.062
# roll axis -> yaw output, down: the roll X12's radius 62 + a 6 mm plate + 1.5 of air +
# the yaw AK80-64's 61.9 mm body standing on it
HIP_YAW_Z = 0.062 + 0.0075 + 0.0619
# The ankle drive (G1's parallel ankle). The calf is a box of two 5 mm plates: the
# outboard one at y 0.014..0.019 (clear of the thigh's 10 mm knee plate and the 3 mm
# heads of its knee bolts), the inboard one 72.5 mm in. Motor A (high) bolts its FRONT face to the outboard plate's inside, body
# in, output out through it; motor B (low) the mirror image on the inboard plate. Each
# turns a crank of CRANK_R on the outside of its plate, pointing back, and a rod drops
# from it to the foot FOOT_LEVER behind the ankle axis, one each side of the roll axis
# (46 mm out): together they pitch the foot, against each other they roll it. The ankle
# sits ANKLE_IN inboard of the knee so the two rods straddle it.
CRANK_R = 0.050
FOOT_LEVER = 0.060
ANKLE_IN = 0.017
CALF_T = 0.005


def _leg(s, sg):
    """One leg, in the arm's format. sg = +1 left, -1 right; limits mirror about the
    sagittal plane. About y, NEGATIVE hip pitch lifts the thigh forward; POSITIVE knee
    bends it (the foot goes back). Hip roll is positive outward on the left."""
    return [
        ("hip_%s_pitch" % s, "base_link", "hip_%s" % s, (0, 1, 0), (0.0, sg * HIP_Y, HIP_Z),
         (D(-150), D(60)), (0, -sg, 0)),
        # the roll actuator outboard at the same height, its body pointing back, clear
        # of a thigh lifting forward. Inward it stops at -20: the hip_roll cheek meets the
        # hip bracket from -23 (check_pair); walking asks ~10-15 of adduction
        ("hip_%s_roll" % s, "hip_%s" % s, "hip_roll_%s" % s, (1, 0, 0), (0.0, sg * HIP_ROLL, 0.0),
         (D(-20), D(90)) if sg > 0 else (D(-90), D(20)), (-1, 0, 0)),
        ("hip_%s_yaw" % s, "hip_roll_%s" % s, "thigh_%s" % s, (0, 0, 1), (0.0, 0.0, -HIP_YAW_Z),
         (D(-45), D(45)), (0, 0, 1)),
        # the knee stops at 110: from 115 the high ankle motor meets the thigh's plate
        # (a walk asks ~65, stairs ~100)
        ("knee_%s" % s, "thigh_%s" % s, "calf_%s" % s, (0, 1, 0), (0.0, 0.0, -(THIGH - HIP_YAW_Z)),
         (D(0), D(110)), (0, -sg, 0)),
        ("ankle_%s_pitch" % s, "calf_%s" % s, "ankle_%s" % s, (0, 1, 0), (0.0, -sg * ANKLE_IN, -CALF),
         (D(-45), D(30)), None),
        ("ankle_%s_roll" % s, "ankle_%s" % s, "foot_%s" % s, (1, 0, 0), (0.0, 0.0, 0.0),
         (D(-20), D(20)), None),
    ]


ANKLE_DRIVE = []    # (name, motor key, host link, output flange xyz in it, output axis, crank m)
# With legs, an arm hanging straight down meets the hip roll and yaw motors: the hips
# (0.38 m across the roll motors) are wider than the arm line (+-0.155). From 15 deg of
# shoulder abduction the forearms clear them, also with the waist hard over and the
# thighs swinging +-30 (check_fit); a hanging arm keeps ARMS_DOWN_ABD out (the user's
# choice 2026-11-24, over narrower hips). Climbing raises the arms and never meets it.
ARMS_DOWN_ABD = D(18)
# Where the legs fold while the arms climb (cad/tools/stow.py): the thighs 30 deg back,
# knees and ankles straight. At every lesson's hang it clears the walls and rungs by 49
# mm and hides no rung cell from the lidars; legs straight down meet lesson 4's rows
# below. (The recorded v2 climbs bring the pelvis's underside to the wall, where the
# hips' own motors are: a full robot climbs those differently -- 2026-11-24, CAD round.)
STOW = {"hip_L_pitch": D(30), "hip_R_pitch": D(30)} if ROBOT == "full" else {}
if ROBOT == "full":
    # Running (the user's choice, 2026-11-25): the hip pitch is an X12-320 (320 N.m; a jog
    # asks ~145 at this mass). Two of them face each other in the pelvis, 69.5 mm of body
    # behind each housing face, so the walls go out until 1.5 mm stands between the two
    # bodies (the RS04s fitted the v2 walls at +-71): housing faces at +-70.25, the 6 mm
    # walls on them, the flange's ring over the Ø109 pilot 7 mm proud of the wall
    # (connect.face_t), the output 2.5 mm past that.
    HIP_KEY = "X12-320"
    _hk = IFACE[HIP_KEY]
    _hou = 0.00075 + (_hk["length"] + _hk["hou_z"]) * MM     # housing face, off the centre
    HIP_W = _hou + 0.006                                     # the walls' outer faces (PLATE)
    HIP_Y = _hou - _hk["hou_z"] * MM                         # the output
    HIP_Z = HIP_Z - (_hk["od"] - IFACE["AK80-64"]["od"]) / 2 * MM   # its top stays put
    LOW_Z0 = HIP_Z - _hk["od"] / 2 * MM - 0.007                     # 7 mm under it, as before
    # the arms that hang the robot with its legs: an X12-320 at shoulder pitch, roll and
    # rotation and at the elbow, an AK80-64 at forearm roll and at both wrist joints (the
    # gimbal, below), an AK60-39 at the finger; all on 48 V
    PICK.update(shoulder_flexion="X12-320", shoulder_abduction="X12-320",
                shoulder_rotation="X12-320", elbow="X12-320", wrist_flexion="AK80-64",
                forearm_roll="AK80-64", wrist_deviation="AK80-64", finger="AK60-39")
    # re-spaced by v2's own rules for the bigger bodies (radii X12 62, AK80 49, AK60 39.5):
    #   SH_Y      the two 85 mm shoulder bodies inboard of their outputs, 2.5 mm apart
    #   roll      the roll's radius + 1.5 air + a 16 mm web + 2.5 (the output plate's bolts)
    #   elbow     the elbow's radius (X15 83) + v2's 11 mm to the rotation output
    #   froll     the elbow's radius + the AK80-64 forearm roll's 61.9 mm body + 6 mm
    #   wflex     the X12's radius + v2's 20.5 mm to the forearm roll output
    #   wdev      X12 + AKE90-8 radii + 3 mm (v2: two AK60-39, 82 needed, 84 given)
    #   finger    AKE90-8 + AK60-39 radii + 3, + v2's 2 mm
    # 2026-11-25 (the user's go-ahead): the arm was 39 % longer than a 1.40 m person's,
    # the hand most of all (0.327 m wrist to hook tip, a person's 0.151) -- three actuators
    # in a row. The wrist is now a gimbal: the flexion and deviation axes MEET (wdev 0),
    # each actuator's body to one side of that point (MOUNT_AT below), so nothing stands
    # down the hand; the elbow then needs 230 N.m, not 308, and an X12-320 holds it on
    # 48 V (the X15-450 and its 72 V branch go); the wrists need ~90 N.m: AK80-64.
    #   elbow     X12 radius 62 + v2's 11
    #   froll     the elbow's radius + the AK80-64 forearm roll's 61.9 mm body + 6 mm
    #   wflex     forearm roll output -> wrist centre: the deviation actuator, swung
    #             +-25 deg about the flexion axis, passes 11 mm under the forearm's plate
    #   finger    AK80-64 + AK60-39 radii + 3, + v2's 2 mm
    SH_Y = 0.0875
    ARM_OFF.update(roll=0.082, elbow=0.073, froll=0.130, wflex=0.110, wdev=0.0, finger=0.0935)
    JOINTS = JOINTS[:1] + _arm("L", 1) + _arm("R", -1) + _leg("L", 1) + _leg("R", -1)
    # the gimbal's flexion actuator stands 60 mm off the wrist centre: pointing inboard,
    # as v2's does, it met the hip yaw actuators with the arms down and the legs apart
    # (check_fit): it points outboard
    JOINTS = [j[:6] + ((0, 1 if "_L_" in j[0] else -1, 0),) if j[0].startswith("wrist_") and
              j[0].endswith("_flexion") else j for j in JOINTS]
    # the X12 elbow sits 21 mm nearer the shoulder than the X15 did: folded past -108 deg
    # the forearm roll's body meets the shoulder rotation's (check_fit): -107
    JOINTS = [j[:5] + ((D(-107), j[5][1]),) + j[6:] if j[0] in ("elbow_L", "elbow_R") else j
              for j in JOINTS]
    # and its deviation actuator's body stood toward the wall with the arms raised: on a
    # rung 60 mm off the wall (rl lessons) its rim met the wall and levered the hook off
    # the rung (urdf/_gen_keyframe.py fell): it points away from the wall
    JOINTS = [j[:6] + ((1, 0, 0),) if j[0].startswith("wrist_") and j[0].endswith("_deviation")
              else j for j in JOINTS]
    HIPS = []                       # the hip pitch is a joint now
    # running (2026-11-25): X12-320 hip pitch and roll, AK80-64 hip yaw
    PICK.update(hip_pitch="X12-320", hip_roll="X12-320", hip_yaw="AK80-64", knee=None,
                ankle_pitch=None, ankle_roll=None)
    # A over B (79 mm bodies, 90 apart) just under the knee, axes on the leg line so the
    # cranks reach FOOT_LEVER behind it (A's top 5.5 mm under the knee's RS04);
    # each output face 1 mm past its plate's outer face (AK60-39: front face 6 mm behind
    # the output), the plates inner faces at y 0.014 and -0.0485 (each 67 mm body ends
    # 1.5 mm short of the other plate)
    _out_a, _out_b = 0.014 + 0.006, -0.0485 - 0.006
    ANKLE_DRIVE = [("ankle_%s_a" % _s, "AK60-39", "calf_%s" % _s, (CRANK_R - FOOT_LEVER, _sg * _out_a, -0.105), (0, _sg, 0), CRANK_R)
                   for _s, _sg in (("L", 1), ("R", -1))] + \
                  [("ankle_%s_b" % _s, "AK60-39", "calf_%s" % _s, (CRANK_R - FOOT_LEVER, _sg * _out_b, -0.195), (0, -_sg, 0), CRANK_R)
                   for _s, _sg in (("L", 1), ("R", -1))]
    # (motor, crank direction at zero, the rod's foot end at zero), in the calf frame: the
    # crank points back, the rod hangs plumb from its pin (16 mm outside the output face:
    # an 8 mm crank hub whose arm runs on its outer 4 mm, over the plate's bolt heads, and
    # half a rod end) to the foot, 30 mm under the ankle axis
    ANKLE_RODS = [("ankle_%s_%s" % (_s, _k), (-1.0, 0.0, 0.0),
                   (-FOOT_LEVER, _sg * _y, -CALF - 0.030))
                  for _s, _sg in (("L", 1), ("R", -1))
                  for _k, _y in (("a", _out_a + 0.016), ("b", _out_b - 0.016))]
    # each linkage-driven ankle joint, at zero: its two motors and how many radians of
    # crank a radian of the joint asks of each (both rods FOOT_LEVER behind the pitch
    # axis; each its own distance off the roll axis). The joint's peak torque is
    # sum(peak * gear), its speed the motor's / max(gear), its reflected inertia
    # sum(gear^2 * the motor's) -- at zero; check_full.ankle has the whole range.
    ANKLE_JOINT = {}
    for _s, _sg in (("L", 1), ("R", -1)):
        _rods = [r for r in ANKLE_RODS if r[0].startswith("ankle_%s_" % _s)]
        ANKLE_JOINT["ankle_%s_pitch" % _s] = dict(motor="AK60-39", gear=[FOOT_LEVER / CRANK_R] * 2)
        ANKLE_JOINT["ankle_%s_roll" % _s] = dict(
            motor="AK60-39", gear=[abs(r[2][1] + _sg * ANKLE_IN) / CRANK_R for r in _rods])
    # The knee through a toothed belt (the user's choice, 2026-11-25): on one foot at a
    # 90 deg knee it needs 99 N.m, climbing a step 129, where a RobStride 04 at the joint
    # held 1.21x / 0.93x. The RS04 moves up the thigh, KNEE_BELT["c"] over the knee axis,
    # its body outboard and its output inboard through the thigh plate; a 30-tooth pulley
    # on it drives a 45-tooth one fixed to the calf's inboard plate by an 8 mm pitch
    # (8MGT) carbon belt of 66 teeth (528 mm), 21 mm wide, in the 39 mm between the thigh
    # plate and the calf's inboard plate: 1.5:1, 180 N.m peak and 133 rpm no-load at the
    # knee (HECTOR's knee belt is 1.55:1). Its pull at that peak is 3.1 kN on the big
    # pulley -- the width is to be confirmed against Gates' rating tables. The motor's
    # bolts sit in slots: the belt is tensioned by sliding the motor up the thigh.
    _pd = lambda n: n * 0.008 / math.pi                       # pitch diameter
    KNEE_BELT = dict(small=_pd(30), big=_pd(45), teeth=66, pitch=0.008, width=0.021, t=0.0056,
                     y=(-0.0405, -0.0195))                    # the belt's faces, left leg
    _L, _D, _d = 66 * 0.008, _pd(45), _pd(30)
    _c = 0.11
    for _ in range(30):                                       # centre distance for that belt
        _c = (_L - math.pi * (_D + _d) / 2 - (_D - _d) ** 2 / (4 * _c)) / 2
    KNEE_BELT["c"] = _c
    _zk = -(THIGH - HIP_YAW_Z)                                # the knee axis, thigh frame
    ANKLE_DRIVE += [("knee_%s_m" % _s, "RS04", "thigh_%s" % _s, (0.0, 0.0, _zk + _c), (0, -_sg, 0),
                     KNEE_BELT["small"] / 2) for _s, _sg in (("L", 1), ("R", -1))]
    for _s in "LR":
        ANKLE_JOINT["knee_%s" % _s] = dict(motor="RS04", gear=[KNEE_BELT["big"] / KNEE_BELT["small"]])
else:
    ANKLE_RODS = []
    ANKLE_JOINT = {}

# ------------------------------------------------------------------ structure
TORSO_LOW = (0.140, 2 * HIP_W if ROBOT == "full" else 0.142, 0.200 - DROP - LOW_Z0)   # w: the hip walls' outer faces, +-71  # d, w, h from LOW_Z0; w 0.210 -> 0.180 to clear the elbow motors
# ...and the four vertical corners are rounded away, because the waist turns the
# arms over a STATIONARY lower torso: with the arms down, the elbow motor's
# inboard face sweeps a circle of radius 93.1 mm about the waist axis, so every
# point of the lower torso has to stay inside R = 90 mm. A 140 x 180 box corner
# reaches 114 mm. The section is therefore that box intersected with a 180 mm
# circle - a stadium, flat front and back, round left and right.
TORSO_LOW_R = 0.0865    # 0.090 until the pelvis reached down past the forearms (2026-11-11)
if ROBOT == "full":
    # the RS04 hips are 120 mm across: the walls must reach x +-61.5 at y +-71 to cover
    # their housing rings, so the round sides go out to 94 mm. A hanging forearm (88 mm)
    # then meets the pelvis's corners when the waist turns far with the arms straight
    # down: the waist is limited in that posture (measured by check_fit, Task 9), not
    # in its joint range, which climbing uses whole.
    TORSO_LOW_R = 0.094
    # the X12 hips (2026-11-25): their Ø124 housing faces at y +-70.25, their 12 x M5 on
    # Ø116.5 -- the walls reach x +-62 at y +-76.25
    TORSO_LOW_R = math.hypot(0.062, HIP_W) + 0.0005
# The chest is a box round the two shoulder pitch actuators (build.torso_upper):
# as deep as the lower torso; as wide as its side walls, which are 6 mm plates
# on the actuators' front faces (y = SH_Y - 8 mm .. - 2 mm); as tall as the
# actuators' tops plus 1.5 mm of air and a 3 mm lid.
_SH = IFACE[PICK["shoulder_flexion"]]   # the shoulder pitch actuator: walls on its front face
TORSO_UP = (0.140, 2 * (SH_Y + _SH["hou_z"] * MM + 0.006), SH_Z + _SH["od"] / 2 * MM + 0.0015 + 0.003)
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

# What the chest carries besides its actuators and lidars, each as (centre, size, mass)
# in torso_upper's frame (deck top z = 6 mm): drawn as boxes in STEP/robot.step,
# checked for room against the structure, actuators and lidars (tools/check_bay.py),
# and carried in the model as point masses (PAYLOAD).
#   battery  48 V 6 Ah 13S2P drawer: slides in from the back on two rails, under the
#            back panel's bottom band, onto a stop at the front
#   jetson   Jetson Orin NX on a Seeed reComputer J401 carrier (100 x 80 mm PCB, DC 9-19 V,
#            10-40 W) with its heatsink and fan, standing on the bulkhead between the two
#            sunk lidars. ponytail: J401's hole pattern was not found in its public docs,
#            so the bulkhead carries an M3 grid on 10 mm; drill its own four once Seeed's
#            mechanical drawing is in hand
#   imu      Xsens MTi-630 (31.5 x 28 x 13 mm, 8.9 g, 3 x M2: Xsens MTi 600-series HW
#            integration manual, fig. 13), on the tray behind the bulkhead
#   can      the CAN interface for the actuators' buses, on the tray beside the IMU
#   dcdc     a quarter-brick 48 V -> 19 V converter (58.4 x 36.8 x 12.7 mm) for the Jetson,
#            and the wiring, on the tray in front of the bulkhead
PAYLOAD_LINK = "torso_upper"
BAY = {"battery": ((-0.007, 0.0, 0.055), (0.120, 0.104, 0.070), 1.60),
       "jetson": ((0.0005, 0.0, 0.1525), (0.055, 0.100, 0.085), 0.35),
       "imu": ((-0.046, 0.0, 0.1035), (0.028, 0.0315, 0.013), 0.01),
       "can": ((-0.046, 0.040, 0.1045), (0.030, 0.040, 0.015), 0.10),
       "dcdc": ((0.043, 0.0, 0.10335), (0.0368, 0.0584, 0.0127), 0.44)}
PAYLOAD = [(k, m, c) for k, (c, _, m) in BAY.items()]
TRAY_Z = 0.094            # the electronics tray's underside, over the battery's 90 mm top

# ------------------------------------------------------------------ lidars
# Four Livox Mid-360 (livoxtech.com/mid-360/specs; the STEP is Livox's own):
# 65 x 65 x 60 mm, 265 g, 10 Hz, 200 000 points/s, range precision 3 cm (1 sigma)
# at 0.2 m, 2 cm at 10 m. The field of view is 360 deg ABOUT ITS OWN AXIS by -7..52
# deg of elevation, so there is a 38 deg blind cone along the axis, and nothing
# inside 0.1 m is seen. It mounts by its bottom face only: 4 x M3, 5 mm deep, on
# 48 x 36 mm, with a dowel hole and a slot on the centre line (measured off the
# STEP). Livox asks for >= 10 mm of air round it and for a >= 3 mm metal plate
# with >= 10 000 mm^2 of it in the air, which is why every pad below sits on a
# plate that size rather than on a bare bracket.
MID360 = dict(mass=0.265, size=(0.065, 0.065, 0.0601), fov=(-7.0, 52.0), blind=0.10,
              rate_hz=10.0, range_sig=(0.03, 0.02), holes=0.024, holes_y=0.018,
              dowel=(0.0, 0.016), slot=(0.0, -0.023), window=(0.033, 0.0601), window_r=0.0255,
              base=(0.0648, 0.0648, 0.033),   # the body under the window (+ the M12 plug to x 40.3)
              origin_z=0.047)          # optical origin above the mounting face, estimated
                                       # from the window (Livox gives only its IMU's
                                       # offset from it, 44 mm)
LIDAR_PAD = (0.072, 0.006)             # the pad each one bolts to: side, thickness
_C45 = math.sqrt(0.5)
# Sensor frames in their link's frame: (link, the mounting face's centre, the
# sensor's axis, toward its connector). Chosen by ray-casting the rung plane and
# the room from the secured hang (2026-09-30), against a first draft with 40 deg
# wedges on the head and both backs (73 % of the reach box seen, 79 % of the room)
# and a flat head with two sunk backs (50 %, 87 %): this one sees 75 % and 92 %, and
# 83 % and 90 % inside the tube cage below (solid rails in its place: 67 %).
# A lidar standing flat on the head sees only a band round itself - the wall is
# 6-11 cm in front and the rungs above fall in the 38 deg blind cone - so the head
# carries two, on a roof, one 45 deg toward the wall and one 45 deg away. Nothing
# stands proud of the chest or the back: the chest is 11-113 mm from the wall
# while climbing, so the chest lidar and the back one are sunk into the empty bay
# under the shoulder actuators, back to back, only their windows outside, and
# a cage of tube stands proud of all four (lidar.cage).
LIDARS = {
    "lidar_head_front": ("torso_upper", (0.030, 0.0, 0.252 + DROP), (_C45, 0.0, _C45), (_C45, 0.0, -_C45)),
    "lidar_head_back": ("torso_upper", (-0.030, 0.0, 0.252 + DROP), (-_C45, 0.0, _C45), (-_C45, 0.0, -_C45)),
    "lidar_chest": ("torso_upper", (0.040, 0.0, 0.068 + DROP), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "lidar_back": ("torso_upper", (-0.040, 0.0, 0.068 + DROP), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
}
# A cage of bent 6061 tube guards the four: one bar per corner of the torso, and a
# ring over the head joining their tops. Each bar leaves the front (back) panel's
# bottom band, turns up, runs `standoff` proud of the panel (its outer surface 10 mm
# ahead of the most either lidar stands out), jogs inward from y to head_y between
# the two heights of `jog`, and rises to the ring. Inward because the raised upper
# arms pass the head: at +-71 mm above the lid a guard touched them in 1130 of 3000
# poses near the hang, at +-50 in none. One bar, not a chest guard and a head guard:
# two would each turn back into the panel at the lid and leave a dip there that a
# rung could settle into, against the head lidar's lower edge. A strut ties each bar
# to the panel's top band at z strut.
GUARD = dict(tube_d=0.016, tube_t=0.002, standoff=0.040, bend_r=0.020, y=0.071, head_y=0.050,
             foot_z=0.011 + DROP, jog=(0.190 + DROP, 0.230 + DROP), strut_z=0.2095 + DROP,
             ring_z=0.345 + DROP)
HEAD_RISE = 0.0                        # a riser under the head's roof (full only)
if ROBOT == "full":
    # the X12 shoulders are 26 mm wider than the AK80-64s: the chest is 13 mm taller, and
    # the head's lidars and the cage's upper half go up with its lid. The bigger arm
    # actuators also hang the whole body 111 mm lower under the rung (the head lidars at
    # every lesson's hang: z 0.338 against v2's 0.449), and from there the head saw 8-20
    # cells of the start rung where v2 sees 24-30: the roof stands on a 111 mm riser
    # (the user's choice, 2026-11-25), the cage's ring with it
    HEAD_RISE = 0.111
    _DZ = (IFACE["X12-320"]["od"] - IFACE["AK80-64"]["od"]) / 2 * MM
    for _k in ("lidar_head_front", "lidar_head_back"):
        _l, _o, _z, _x = LIDARS[_k]
        LIDARS[_k] = (_l, (_o[0], _o[1], _o[2] + _DZ + HEAD_RISE), _z, _x)
    GUARD.update(jog=(GUARD["jog"][0] + _DZ, GUARD["jog"][1] + _DZ),
                 strut_z=GUARD["strut_z"] + _DZ, ring_z=GUARD["ring_z"] + _DZ + HEAD_RISE)
    # The riser is a room of its own (122 x 148 inside, windows on all four faces): the
    # full robot's extra electronics go in it, on its floor (the lid), clear of its lips.
    #   leg_can  the legs' CAN interface and their harness to the pelvis: the design's
    #            0.5 kg allowance. (Everything runs on the 48 V battery: the X15 elbows
    #            and their 72 V boost converter went with the gimbal wrist.) Alone in
    #            the room it sits on the centre line: 45 mm off, it leaned the hang 0.4 mm
    #            to one side and the left wrist deviation took 0.7 N.m more (1.29x)
    _h = TORSO_UP[2]
    BAY.update(leg_can=((0.0, 0.0, _h + 0.010 + 0.0075), (0.040, 0.030, 0.015), 0.50))
    PAYLOAD = [(k, m, c) for k, (c, _, m) in BAY.items()]

# Where along its own axis an actuator's OUTPUT FLANGE sits, measured from the
# joint origin in the body_dir direction, metres. The joint axis is a line, so
# sliding the actuator along it changes nothing kinematically - it is free
# clearance. The abduction-first shoulder slid its abduction actuator 30 mm
# back out of the raised arm's sweep; nothing needs it pitch-first.
MOUNT_AT = {}
WRIST_GIMBAL = ROBOT == "full"
if WRIST_GIMBAL:
    # the gimbal wrist's two actuators each stand off the wrist centre along their own
    # axis, flange toward it: flexion 60 mm (its output plate then clears the deviation
    # actuator's Ø98 barrel by 5 mm), deviation 54 mm (its barrel clears the flexion one's)
    for _s in "LR":
        MOUNT_AT["wrist_%s_flexion" % _s] = 0.092
        MOUNT_AT["wrist_%s_deviation" % _s] = 0.054

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
        if MOTOR_OF[name] is None:       # driven through a linkage (ANKLE_DRIVE)
            continue
        po, pR = F[par]
        o = F[ch][0]
        cR = F[ch][1]                    # the axis is in the child's frame
        wax = tuple(sum(cR[i][k] * ax[k] for k in range(3)) for i in range(3))
        bd = BODY_DIR[name]
        wbd = tuple(sum(pR[i][k] * bd[k] for k in range(3)) for i in range(3))
        a = MOUNT_AT.get(name, 0.0)
        flange = tuple(o[k] + wbd[k] * a for k in range(3))
        out.append((name, MOTOR_OF[name], flange, wax, wbd))
    for name, key, host, xyz, ax, _r in ANKLE_DRIVE:   # the full robot's ankle motors
        ho, hR = F[host]
        rot = lambda v: tuple(sum(hR[i][k] * v[k] for k in range(3)) for i in range(3))
        out.append((name, key, tuple(ho[i] + rot(xyz)[i] for i in range(3)), rot(ax),
                    tuple(-v for v in rot(ax))))
    po, pR = F["base_link"]              # the hips' actuators, where legs will bolt on
    rot = lambda v: tuple(sum(pR[i][k] * v[k] for k in range(3)) for i in range(3))
    for name, _par, _ch, ax, off, _lim, bd in HIPS:
        out.append((name, HIP_KEY, tuple(po[i] + rot(off)[i] for i in range(3)), rot(ax), rot(bd)))
    return out


BODY_DIR = {j[0]: j[6] for j in JOINTS + HIPS}
