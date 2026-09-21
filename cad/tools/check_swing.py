"""Does a link hit an actuator it is NOT bolted to?

check_one asks about a link and its own two actuators; check_pair asks about
two neighbours. Neither sees the case that set MOUNT_AT: the shoulder ROTATION
actuator, two joints down, sweeping into the ABDUCTION actuator when the arm
comes up. This sweeps that pair over both joints between them.
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
    abd, flx, rot = J["shoulder_L_abduction"], J["shoulder_L_flexion"], \
        J["shoulder_L_rotation"]
    fixed = solid_at(abd, abd[4])                    # in torso_upper's frame
    arm = fuse(BUILDERS["upper_arm_L"]()) + solid_at(rot, rot[4])
    worst, at = 0.0, None
    for i in range(N):
        qa = math.degrees(abd[5][0] + (abd[5][1] - abd[5][0]) * i / (N - 1.0))
        La = bd.Location(abd[4], tuple(k * qa for k in abd[3]))
        for k in range(N):
            qf = math.degrees(flx[5][0] + (flx[5][1] - flx[5][0]) * k / (N - 1.0))
            Lf = bd.Location(flx[4], tuple(v * qf for v in flx[3]))
            v = (fixed & (La * Lf * arm)).volume * 1e9
            if v > worst:
                worst, at = v, (qa, qf)
    ok = worst <= 1.0
    print("  %-4s abduction actuator vs upper arm + rotation actuator"
          % ("PASS" if ok else "FAIL"))
    print("       worst %9.1f mm3%s" % (
        worst, "" if at is None else
        " at abduction %.0f, flexion %.0f deg" % at))
    return not ok


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
