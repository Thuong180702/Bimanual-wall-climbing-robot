"""Three questions about one connector, asked of the real actuator geometry.

  1. is it ONE connected solid?
  2. does it CLEAR both actuators?  (no volume inside either envelope)
  3. does it BOLT ON?               (how much of each mating face it presses)

A connector that merely dodges the actuators passes 2 and fails 3. That was
the fault in every earlier version of these parts.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from lib.build import BUILDERS, CHILD_OF, PJOINT, fuse, mount_plane
from lib.connect import keepout, witness
from lib.spec import MOTOR_OF

MIN_COVER = 0.55


def main(name):
    parts = BUILDERS[name]()
    link = fuse(parts)
    bad = 0

    n = len(link.solids())
    ok = n == 1
    bad += not ok
    print("  %-4s one connected solid                      %d solid(s), %.1f cm3, %.3f kg"
          % ("PASS" if ok else "FAIL", n, link.volume * 1e6, link.volume * 2700))

    from lib.build import FACE, mount_face_of
    # the root link has no parent joint, and a link whose parent joint is the
    # fixed one has no parent ACTUATOR to bolt to
    jobs = ([(PJOINT[name][0], "output", mount_plane(PJOINT[name], (0, 0, 0)))]
            if name in PJOINT and not isinstance(PJOINT[name], str) else [])
    for cj in CHILD_OF.get(name, []):
        jobs.append((cj[0], FACE.get(name, mount_face_of(cj)),
                     mount_plane(cj, cj[4])))

    for jn, face, pl in jobs:
        key = MOTOR_OF[jn]
        v = (link & (pl * keepout(key))).volume * 1e9
        ok = v <= 1.0
        bad += not ok
        print("  %-4s clears %-22s %-7s %9.1f mm3"
              % ("PASS" if ok else "FAIL", jn, "", v))
        w = pl * witness(key, face)
        frac = (link & w).volume / w.volume
        ok = frac >= MIN_COVER
        bad += not ok
        print("  %-4s bolts to %-20s %-7s %5.0f %% of the face"
              % ("PASS" if ok else "FAIL", jn, face, 100 * frac))
    print("\n  %s" % ("all checks pass" if not bad else "%d FAILED" % bad))
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
