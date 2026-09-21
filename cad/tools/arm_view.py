"""Every connector drawn so far, bolted to its actuators, in one STEP.

The left arm below the shoulder-rotation flange. Run it after adding a
connector to see the chain, not just the part.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from lib.build import BUILDERS, CHILD_OF, PJOINT, PJOINT as _PJ, fuse, mount_plane
from lib.connect import plane_at
from lib.spec import BODY_DIR, MOTOR_OF

STEPS = {"AK80-64": "ak80_64", "AK60-39": "ak60_39", "AK45-36": "ak45_36"}
CHAIN = ["shoulder_L", "upper_arm_L", "humerus_rot_L", "fore_arm_L", "forearm_rot_L",
         "wrist_L", "wrist_dev_L", "hook_L"]


def motor(j, org):
    m = mount_plane(j, org)
    return bd.Location(bd.Plane(origin=tuple(v * 1000 for v in m.origin),
                                x_dir=m.x_dir, z_dir=m.z_dir)) * bd.import_step(
        os.path.join(HERE, "..", "STEP", "purchased",
                     STEPS[MOTOR_OF[j[0]]] + ".step"))


def main():
    kids, at, mass = [], (0.0, 0.0, 0.0), 0.0
    kids.append(motor(PJOINT[CHAIN[0]], (0.0, 0.0, 0.0)))
    for name in CHAIN:
        if name not in BUILDERS:
            break
        link = fuse(BUILDERS[name]())
        mass += link.volume * 2700
        loc = bd.Location(tuple(v * 1000 for v in at))
        kids.append(loc * link.scale(1000.0))
        cj = CHILD_OF.get(name, [None])[0]
        if cj is None:
            break
        kids.append(loc * motor(cj, cj[4]))
        at = tuple(at[k] + cj[4][k] for k in range(3))
    out = os.path.join(HERE, "..", "tmp", "arm.step")
    bd.export_step(bd.Compound(children=kids), out)
    print("%d connectors, %.3f kg of 6061 -> tmp/arm.step" % (len(CHAIN), mass))


if __name__ == "__main__":
    main()
