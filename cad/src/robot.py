"""The whole robot at its zero pose, in MILLIMETRES (CAD default): 17 vendor actuators and the aluminium
between them.

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

from lib.links import BUILDERS
from lib.spec import BODY_DIR, IFACE, MOTOR_OF, fk, motor_placements
from purchased.ak45_36 import ak45_36
from purchased.ak60_39 import ak60_39
from purchased.ak80_64 import ak80_64

MM = 0.001
MOTOR_MODEL = {"AK80-64": ak80_64, "AK60-39": ak60_39, "AK45-36": ak45_36}


def _loc(origin, z_dir, x_hint=(1, 0, 0)):
    z = bd.Vector(*z_dir)
    x = bd.Vector(*x_hint)
    if abs(x.dot(z)) > 0.9:
        x = bd.Vector(0, 0, 1)
    x = x - z * (x.dot(z) / z.dot(z))
    return bd.Location(bd.Plane(origin=origin, x_dir=(x.X, x.Y, x.Z), z_dir=z_dir))


@step(out="../STEP/robot.step")
def robot():
    F = fk()
    kids = []

    report("actuators")
    built = {k: MOTOR_MODEL[k]() for k in MOTOR_MODEL}
    for name, key, o, ax, bd_ in track(motor_placements(),
                                       label=lambda r: r[0]):
        # The vendor part is drawn in mm in the joint frame with its body on
        # NEGATIVE z and the output face at z = 0, so its +Z has to point the
        # way the output faces - the opposite of body_dir.
        out_dir = tuple(-v for v in bd_)
        s = _loc(tuple(v * 1000 for v in o), out_dir) * built[key]
        s.label = "%s  %s" % (name, key)
        kids.append(s)

    report("structure")
    for link, build in track(list(BUILDERS.items()), label=lambda r: r[0]):
        o, R = F[link]
        pl = bd.Plane(origin=tuple(v * 1000 for v in o), x_dir=(R[0][0], R[1][0], R[2][0]),
                      z_dir=(R[0][2], R[1][2], R[2][2]))
        p = bd.Location(pl) * build().scale(1000.0)
        p.label = link
        kids.append(p)

    a = bd.Compound(children=kids)
    a.label = "bimanual wall climber v2"
    return a


if __name__ == "__main__":
    robot()
