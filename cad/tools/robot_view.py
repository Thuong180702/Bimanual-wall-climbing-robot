"""The whole robot as drawn so far: every connector, bolted to its actuators.

Walks the joint tree from base_link, so it picks up both arms and shows what
the parts actually look like together.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from lib.build import BUILDERS, CHILD_OF, fuse, mount_plane
from lib.spec import FIXED, MOTOR_OF

STEPS = {"AK80-64": "ak80_64", "AK60-39": "ak60_39", "AK45-36": "ak45_36"}
_CACHE = {}


def vendor(key):
    if key not in _CACHE:
        _CACHE[key] = bd.import_step(os.path.join(
            HERE, "..", "STEP", "purchased", STEPS[key] + ".step"))
    return _CACHE[key]


def walk(link, at, kids, mirror=False):
    if link in BUILDERS:
        part = fuse(BUILDERS[link]())
        kids.append(bd.Location(tuple(v * 1000 for v in at)) * part.scale(1000.0))
        print("   %-14s %7.1f cm3  %5.0f g" % (link, part.volume * 1e6,
                                               part.volume * 2.7e6))
    for cj in CHILD_OF.get(link, []):
        m = mount_plane(cj, cj[4])
        mo = (m.origin.X, m.origin.Y, m.origin.Z)
        o = tuple(at[k] + mo[k] for k in range(3))
        kids.append(bd.Location(bd.Plane(origin=tuple(v * 1000 for v in o),
                                         x_dir=m.x_dir, z_dir=m.z_dir))
                    * vendor(MOTOR_OF[cj[0]]))
        walk(cj[2], tuple(at[k] + cj[4][k] for k in range(3)), kids)
    for f in FIXED:
        if f[1] == link:
            walk(f[2], tuple(at[k] + f[3][k] for k in range(3)), kids)


if __name__ == "__main__":
    kids = []
    walk("base_link", (0.0, 0.0, 0.0), kids)
    out = os.path.join(HERE, "..", "tmp", "robot.step")
    bd.export_step(bd.Compound(children=kids), out)
    print("wrote tmp/robot.step")
