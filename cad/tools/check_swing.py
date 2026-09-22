"""Does a link hit an actuator it is NOT bolted to?

check_one asks about a link and its own two actuators; check_pair asks about
two neighbours. Neither sees an actuator on the TORSO being swept by the upper
arm and its rotation actuator, two joints down - the case that set MOUNT_AT on
the old abduction-first shoulder. This sweeps that pair over both joints
between them.
"""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from lib.build import BUILDERS, CHILD_OF, fuse, mount_plane
from lib.connect import keepout
from lib.spec import JOINTS, MOTOR_OF

J = {j[0]: j for j in JOINTS}
N = 9


def solid_at(joint, origin):
    m = mount_plane(joint, origin)
    return bd.Location(bd.Plane(origin=m.origin, x_dir=m.x_dir, z_dir=m.z_dir)) \
        * keepout(MOTOR_OF[joint[0]])


def main():
    """the actuator on the torso vs the upper arm and the rotation actuator,
    swept over both shoulder joints between them, in chain order"""
    j1 = [j for j in JOINTS if j[1] == "torso_upper" and "_L_" in j[0]][0]
    j2 = [j for j in JOINTS if j[1] == j1[2]][0]
    rot = [j for j in JOINTS if j[1] == j2[2]][0]
    fixed = solid_at(j1, j1[4])                      # in torso_upper's frame
    arm = fuse(BUILDERS[j2[2]]()) + solid_at(rot, rot[4])
    worst, at = 0.0, None
    for i in range(N):
        q1 = math.degrees(j1[5][0] + (j1[5][1] - j1[5][0]) * i / (N - 1.0))
        L1 = bd.Location(j1[4], tuple(k * q1 for k in j1[3]))
        for k in range(N):
            q2 = math.degrees(j2[5][0] + (j2[5][1] - j2[5][0]) * k / (N - 1.0))
            L2 = bd.Location(j2[4], tuple(v * q2 for v in j2[3]))
            v = (fixed & (L1 * L2 * arm)).volume * 1e9
            if v > worst:
                worst, at = v, (q1, q2)
    ok = worst <= 1.0
    print("  %-4s %s actuator vs %s + rotation actuator"
          % ("PASS" if ok else "FAIL", j1[0], j2[2]))
    print("       worst %9.1f mm3%s" % (
        worst, "" if at is None else
        " at %s %.0f, %s %.0f deg" % (j1[0], at[0], j2[0], at[1])))
    return not ok


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
