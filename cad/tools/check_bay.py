"""Is there room for what the chest carries (spec.BAY)?

Each box against the chest's own metal, the shoulder actuators, the four lidars
and every other box; and the battery's way out: its drawer swept 150 mm back
through the back panel must meet nothing.
"""
import itertools, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from lib.build import BUILDERS, CHILD_OF, fuse, mount_plane
from lib.connect import keepout
from lib.spec import BAY, LIDARS, MID360, MOTOR_OF


def box(c, s):
    return bd.Pos(*c) * bd.Box(*s)


def lidar_box(o, z, x):
    a, b, h = MID360["size"]
    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
    c = tuple(o[i] + z[i] * h / 2 for i in range(3))
    return bd.Location(bd.Plane(origin=c, x_dir=x, z_dir=z)) * bd.Box(a, b, h)


def main():
    chest = fuse(BUILDERS["torso_upper"]())
    other = {}
    for j in CHILD_OF["torso_upper"]:
        m = mount_plane(j, j[4])
        other[j[0]] = bd.Location(bd.Plane(origin=m.origin, x_dir=m.x_dir, z_dir=m.z_dir)) * keepout(MOTOR_OF[j[0]])
    for n, (host, o, z, x) in LIDARS.items():
        if host == "torso_upper":
            other[n] = lidar_box(o, z, x)
    boxes = {k: box(c, s) for k, (c, s, _) in BAY.items()}
    bad = 0
    for k, b in boxes.items():
        for n, s in [("chest metal", chest)] + list(other.items()):
            v = (b & s).volume * 1e9
            if v > 1e-3:
                print("  FAIL %-8s meets %-20s %8.1f mm3" % (k, n, v)); bad += 1
    for (a, ba), (b, bb) in itertools.combinations(boxes.items(), 2):
        v = (ba & bb).volume * 1e9
        if v > 1e-3:
            print("  FAIL %-8s meets %-20s %8.1f mm3" % (a, b, v)); bad += 1
    (c, s, _) = BAY["battery"]
    way = box((c[0] - 0.075, c[1], c[2]), (s[0] + 0.150, s[1], s[2]))
    v = (way & chest).volume * 1e9
    print("  %s battery drawer pulled 150 mm out the back meets %.1f mm3 of the chest"
          % ("PASS" if v < 1e-3 else "FAIL", v)); bad += v >= 1e-3
    for k, b in boxes.items():
        gap = min(b.distance_to(chest), *[b.distance_to(s) for s in other.values()])
        print("  %-8s nearest neighbour %.1f mm" % (k, gap * 1e3))
    print("\n  all checks pass" if not bad else "\n  %d failure(s)" % bad)
    return bad


if __name__ == "__main__":
    sys.exit(main())
