"""Do two finished connectors clash as their joint moves?

check_one.py asks whether ONE connector is sound against ITS two actuators.
It cannot see the link next door. Two brackets that each pass on their own
still have to swing past each other, and that is a different question - it is
what sent the upper arm's web from 12 mm to 7 mm, and the elbow limit from
-118 to -110 deg. Every pair of neighbouring connectors that both exist gets
swept here.
"""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from lib.build import BUILDERS, fuse
from lib.spec import JOINTS, MOUNT_ROT

N = 13


def sweep(j):
    name, par, ch, ax, off, lim, _ = j
    a, c = fuse(BUILDERS[par]()), fuse(BUILDERS[ch]())
    worst, at = 0.0, 0.0
    for i in range(N):
        q = math.degrees(lim[0] + (lim[1] - lim[0]) * i / (N - 1.0))
        # the child frame: its offset, its fixed mounting turn (the hooks), then the joint
        place = (bd.Location(off) * bd.Rot(0, 0, math.degrees(MOUNT_ROT.get(ch, 0.0)))
                 * bd.Location((0, 0, 0), tuple(k * q for k in ax)))
        v = (a & (place * c)).volume * 1e9
        if v > worst:
            worst, at = v, q
    ok = worst <= 1.0
    print("  %-4s %-14s %-14s  %.0f..%.0f deg   worst %8.1f mm3 at %.0f deg"
          % ("PASS" if ok else "FAIL", par, ch, math.degrees(lim[0]),
             math.degrees(lim[1]), worst, at))
    return not ok


if __name__ == "__main__":
    bad = sum(sweep(j) for j in JOINTS
              if j[1] in BUILDERS and j[2] in BUILDERS)
    print("\n  %s" % ("all pairs clear" if not bad else "%d FAILED" % bad))
    sys.exit(1 if bad else 0)
