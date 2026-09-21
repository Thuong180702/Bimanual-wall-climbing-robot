"""Render one connector on its own and bolted between its two actuators."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from lib.build import BUILDERS, CHILD_OF, PJOINT, fuse, mount_plane
from lib.connect import plane_at
from lib.spec import BODY_DIR, MOTOR_OF

MM = 0.001
STEPS = {"AK80-64": "ak80_64", "AK60-39": "ak60_39", "AK45-36": "ak45_36"}


def vendor(key):
    return bd.import_step(os.path.join(HERE, "..", "STEP", "purchased",
                                       STEPS[key] + ".step"))


def main(name):
    parts = BUILDERS[name]()
    link = fuse(parts)
    print("%s: %d component(s) -> %d solid(s), %.1f cm3, %.3f kg 6061"
          % (name, len(parts), len(link.solids()), link.volume * 1e6,
             link.volume * 2700))
    b = link.bounding_box()
    print("   bbox %.0f x %.0f x %.0f mm"
          % ((b.max.X - b.min.X) * 1000, (b.max.Y - b.min.Y) * 1000,
             (b.max.Z - b.min.Z) * 1000))
    out = os.path.join(HERE, "..", "tmp")
    os.makedirs(out, exist_ok=True)
    bd.export_step(link.scale(1000.0), os.path.join(out, "%s.step" % name))

    kids = [link.scale(1000.0)]
    if name in PJOINT and not isinstance(PJOINT[name], str):
        pj = PJOINT[name]
        m = mount_plane(pj, (0.0, 0.0, 0.0))
        kids.append(bd.Location(bd.Plane(origin=tuple(v * 1000 for v in m.origin),
                                         x_dir=m.x_dir, z_dir=m.z_dir))
                    * vendor(MOTOR_OF[pj[0]]))
    for cj in CHILD_OF.get(name, []):
        m = mount_plane(cj, cj[4])
        kids.append(bd.Location(bd.Plane(origin=tuple(v * 1000 for v in m.origin),
                                         x_dir=m.x_dir, z_dir=m.z_dir))
                    * vendor(MOTOR_OF[cj[0]]))
    bd.export_step(bd.Compound(children=kids),
                   os.path.join(out, "%s_assy.step" % name))
    print("   wrote tmp/%s.step and tmp/%s_assy.step" % (name, name))


if __name__ == "__main__":
    main(sys.argv[1])
