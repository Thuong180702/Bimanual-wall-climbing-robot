"""Where the mass actually sits, per link and for the whole robot.

Every bracket cantilevers off ONE face, and every actuator's body hangs to one
side of its own joint, so each link has a lateral offset. Whether those
offsets cancel down the chain or pile up is the question this answers.
"""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from lib.build import BUILDERS, CHILD_OF, fuse, mount_plane
from lib.motors import MOTORS
from lib.spec import AL, FIXED, MOTOR_OF, PAYLOAD

AX = "XYZ"


def com_of(shape):
    c = shape.center()
    return (c.X, c.Y, c.Z)


def link_mass(name):
    """(mass, com) of the link's own structure plus the actuators it holds."""
    items = []
    if name in BUILDERS:
        p = fuse(BUILDERS[name]())
        items.append((p.volume * AL, com_of(p)))
    for cj in CHILD_OF.get(name, []):
        key = MOTOR_OF[cj[0]]
        m = mount_plane(cj, cj[4])
        z = -MOTORS[key]["len_mm"] / 2000.0          # middle of the barrel
        o = m.origin + m.z_dir * z
        items.append((MOTORS[key]["mass_kg"], (o.X, o.Y, o.Z)))
    if name == "base_link":
        items += [(m, p) for _, m, p in PAYLOAD]
    M = sum(m for m, _ in items)
    if M <= 0.0:                     # a link whose connector is not drawn yet
        return 0.0, (0.0, 0.0, 0.0)
    c = tuple(sum(m * p[k] for m, p in items) / M for k in range(3))
    return M, c


def walk(link, at, out):
    M, c = link_mass(link)
    if M > 0.0:
        out.append((link, M, tuple(at[k] + c[k] for k in range(3))))
    for cj in CHILD_OF.get(link, []):
        walk(cj[2], tuple(at[k] + cj[4][k] for k in range(3)), out)
    for f in FIXED:
        if f[1] == link:
            walk(f[2], tuple(at[k] + f[3][k] for k in range(3)), out)


if __name__ == "__main__":
    rows = []
    walk("base_link", (0.0, 0.0, 0.0), rows)
    print("  %-14s %7s  %s" % ("link", "kg", "com in the WORLD frame, mm"))
    for n, m, c in rows:
        print("  %-14s %7.3f  x %7.1f  y %7.1f  z %7.1f"
              % (n, m, c[0] * 1e3, c[1] * 1e3, c[2] * 1e3))
    def com(rs, flip=False):
        M = sum(m for _, m, _ in rs)
        sy = -1.0 if flip else 1.0
        return M, [sum(m * (c[k] * (sy if k == 1 else 1.0))
                       for _, m, c in rs) / M for k in range(3)]

    torso = [r for r in rows if not (r[0].endswith("_L") or r[0].endswith("_R"))]
    arm = [r for r in rows if r[0].endswith("_L")]
    for nm, rs, fl in (("torso", torso, False), ("left arm", arm, False)):
        M, C = com(rs, fl)
        print("\n  %-16s %6.3f kg   com x %6.1f  y %6.1f  z %6.1f mm"
              % (nm, M, C[0] * 1e3, C[1] * 1e3, C[2] * 1e3))
    # the right arm is the left one mirrored; the loose actuators listed for it
    # above are dropped so nothing is counted twice
    Mt, Ct = com(torso)
    Ma, Ca = com(arm)
    Mr, Cr = com(arm, flip=True)
    M = Mt + Ma + Mr
    C = [(Mt * Ct[k] + Ma * Ca[k] + Mr * Cr[k]) / M for k in range(3)]
    print("  %-16s %6.3f kg   com x %6.1f  y %6.1f  z %6.1f mm"
          % ("whole robot", M, C[0] * 1e3, C[1] * 1e3, C[2] * 1e3))
    print("\n  the arm hangs on y = 155 below the shoulder; its own com is at"
          " y = %.1f, so it is %.1f mm off its own axis." % (Ca[1] * 1e3,
                                                             Ca[1] * 1e3 - 155.0))
