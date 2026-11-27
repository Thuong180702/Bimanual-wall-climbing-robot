"""The whole robot at its zero pose, in MILLIMETRES (CAD default): 17 vendor actuators,
four Livox Mid-360 lidars and the aluminium between them.

Every actuator solid is the manufacturer's own STEP (step.parts, checksum
verified); everything else is drawn here. Placement comes from the same
forward kinematics that writes the URDF, so the CAD and the simulation model
cannot drift apart.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cadgen import build123d as bd
from cadgen import report, step, track

from lib.build import BUILDERS, fuse
from lib.spec import BAY, BODY_DIR, IFACE, LIDARS, MOTOR_OF, ROBOT, fk, motor_placements
from purchased.ak45_36 import ak45_36
from purchased.ake90_8 import ake90_8
from purchased.ak60_39 import ak60_39
from purchased.ak80_64 import ak80_64
from purchased.mid360 import mid360
from purchased.rs04 import rs04
from purchased.x12_320 import x12_320
from purchased.x15_450 import x15_450

MM = 0.001
MOTOR_MODEL = {"AK80-64": ak80_64, "AK60-39": ak60_39, "AK45-36": ak45_36, "RS04": rs04, "X12-320": x12_320, "X15-450": x15_450,
               "AKE90-8": ake90_8}


def _loc(origin, z_dir, x_hint=(1, 0, 0)):
    z = bd.Vector(*z_dir)
    x = bd.Vector(*x_hint)
    if abs(x.dot(z)) > 0.9:
        x = bd.Vector(0, 0, 1)
    x = x - z * (x.dot(z) / z.dot(z))
    return bd.Location(bd.Plane(origin=origin, x_dir=(x.X, x.Y, x.Z), z_dir=z_dir))


def assemble():
    F = fk()
    kids = []

    report("actuators")
    built = {k: MOTOR_MODEL[k]() for k in MOTOR_MODEL}
    for name, key, o, ax, bd_ in track(motor_placements(),
                                       label=lambda r: r[0]):
        if ROBOT != "full" and name.startswith("hip_"):   # spec.HIPS: the pelvis is ready, none fitted
            continue
        # The vendor part is drawn in mm in the joint frame with its body on
        # NEGATIVE z and the output face at z = 0, so its +Z has to point the
        # way the output faces - the opposite of body_dir.
        out_dir = tuple(-v for v in bd_)
        s = _loc(tuple(v * 1000 for v in o), out_dir) * built[key]
        s.label = "%s  %s" % (name, key)
        kids.append(s)

    report("lidars")
    sensor = mid360()
    for name, (link, o, z, x) in LIDARS.items():
        lo, R = F[link]
        link_pl = bd.Plane(origin=tuple(v * 1000 for v in lo), x_dir=(R[0][0], R[1][0], R[2][0]),
                           z_dir=(R[0][2], R[1][2], R[2][2]))
        s = bd.Location(link_pl) * bd.Location(bd.Plane(
            origin=tuple(v * 1000 for v in o), x_dir=x, z_dir=z)) * sensor
        s.label = "%s  Livox Mid-360" % name
        kids.append(s)

    report("chest bay")                      # spec.BAY: what the chest carries, as boxes
    lo, R = F["torso_upper"]
    link_pl = bd.Plane(origin=tuple(v * 1000 for v in lo), x_dir=(R[0][0], R[1][0], R[2][0]),
                       z_dir=(R[0][2], R[1][2], R[2][2]))
    for name, (c, sz, _m) in BAY.items():
        s = bd.Location(link_pl) * bd.Pos(*(v * 1000 for v in c)) * bd.Box(*(v * 1000 for v in sz))
        s.label = "bay: " + name
        kids.append(s)

    report("structure")
    for link, build in track(list(BUILDERS.items()), label=lambda r: r[0]):
        o, R = F[link]
        pl = bd.Plane(origin=tuple(v * 1000 for v in o), x_dir=(R[0][0], R[1][0], R[2][0]),
                      z_dir=(R[0][2], R[1][2], R[2][2]))
        p = bd.Location(pl) * fuse(build()).scale(1000.0)
        p.label = link
        kids.append(p)

    if ROBOT == "full":                      # the ankles' cranks and rods, on each calf
        from lib import legs
        report("ankle linkage")
        for side in "LR":
            o, R = F["calf_" + side]
            pl = bd.Plane(origin=tuple(v * 1000 for v in o), x_dir=(R[0][0], R[1][0], R[2][0]),
                          z_dir=(R[0][2], R[1][2], R[2][2]))
            for k, v in legs.ankle_linkage_L().items():
                v = v if side == "L" else bd.mirror(v, about=bd.Plane.XZ)
                p = bd.Location(pl) * v.scale(1000.0)
                p.label = k.replace("_L", "_" + side)
                kids.append(p)
            for link, kd in legs.knee_drive_L().items():         # the knee belt
                o, R = F["%s_%s" % (link, side)]
                lp = bd.Plane(origin=tuple(v * 1000 for v in o), x_dir=(R[0][0], R[1][0], R[2][0]),
                              z_dir=(R[0][2], R[1][2], R[2][2]))
                for k, v in kd.items():
                    v = v if side == "L" else bd.mirror(v, about=bd.Plane.XZ)
                    p = bd.Location(lp) * v.scale(1000.0)
                    p.label = "knee_%s %s" % (side, k)
                    kids.append(p)

    a = bd.Compound(children=kids)
    a.label = "bimanual wall climber, full (legs)" if ROBOT == "full" else "bimanual wall climber v2"
    return a


@step(out="../STEP/robot.step")
def robot():
    return assemble()


if __name__ == "__main__":
    if ROBOT == "full":
        # cadgen's runner does not pass ROBOT on: the full robot is exported here
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "STEP", "robot_full.step")
        bd.export_step(assemble(), out)
        print("wrote", os.path.abspath(out))
    else:
        robot()
