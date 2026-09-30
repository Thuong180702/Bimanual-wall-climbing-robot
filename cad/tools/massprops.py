"""Mass, centre of mass and inertia for every link - from the solids, not a guess.

A link's mass is its own aluminium plus every actuator whose HOUSING is bolted
to it (the rotor is a small fraction of an actuator and turns with the child, so
the whole catalogue mass is booked to the parent; the error is bounded by the
rotor fraction and is called out in the README).

The actuator itself is modelled for inertia as a uniform cylinder of its
catalogue mass with its real OD and length - the vendor STEP has no materials,
so its volume cannot give a mass, and a uniform cylinder is the honest stand-in.
"""
import math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cadgen import build123d as _bd  # noqa: F401  (kernel first)
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from lib.build import BUILDERS, CHILD_OF, fuse, mount_plane
from lib.spec import AL, BODY_DIR, IFACE, JOINTS, LIDARS, MID360, MOTOR_OF, MOTORS, PAYLOAD

MM = 0.001


def solid_props(shape, density):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, g)
    v = g.Mass()
    c = g.CentreOfMass()
    com = (c.X(), c.Y(), c.Z())
    M = g.MatrixOfInertia()
    # OCC returns the matrix of inertia ABOUT THE CENTRE OF MASS (verified
    # against an offset box), so no parallel-axis correction belongs here.
    I0 = [[M.Value(i + 1, j + 1) * density for j in range(3)] for i in range(3)]
    return v * density, com, I0


def shift(I, d, m):
    """Parallel axis: add m * (|d|^2 delta_ij - d_i d_j). Negative m removes it."""
    d2 = sum(x * x for x in d)
    return [[I[i][j] + m * ((d2 if i == j else 0.0) - d[i] * d[j])
             for j in range(3)] for i in range(3)]


def cyl_inertia(m, r, L, axis):
    """Uniform solid cylinder, mass m, about its own centre, in link axes."""
    a, b = m * (3 * r * r + L * L) / 12.0, m * r * r / 2.0
    k = [i for i, v in enumerate(axis) if abs(v) > 0.9][0]
    I = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        I[i][i] = b if i == k else a
    return I


def motors_on(link):
    """(key, centre, axis) for every actuator whose housing this link carries."""
    out = []
    for cj in CHILD_OF.get(link, []):
        key = MOTOR_OF[cj[0]]
        d, bd = IFACE[key], BODY_DIR[cj[0]]
        o = mount_plane(cj, cj[4]).origin
        c = tuple(v + b * d["length"] * MM / 2
                  for v, b in zip((o.X, o.Y, o.Z), bd))
        out.append((cj[0], key, c, cj[3]))
    return out


def link_props(link):
    parts = []
    m, com, I = solid_props(fuse(BUILDERS[link]()), AL)
    parts.append(("structure", m, com, I))
    for jname, key, c, axis in motors_on(link):
        d = IFACE[key]
        mm_ = MOTORS[key]["mass_kg"]
        parts.append((key + " @" + jname, mm_, c,
                      cyl_inertia(mm_, d["od"] / 2 * MM, d["length"] * MM, axis)))
    for nm, (host, o, z, x) in LIDARS.items():
        if host != link:
            continue
        # a Livox Mid-360: its catalogue mass as a uniform block of its envelope
        mp, (a, b, h) = MID360["mass"], MID360["size"]
        y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
        R = [x, y, z]                                  # rows: the sensor's axes in the link
        Id = [mp * (b * b + h * h) / 12, mp * (a * a + h * h) / 12, mp * (a * a + b * b) / 12]
        I = [[sum(R[k][i] * Id[k] * R[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        parts.append(("Mid-360 " + nm, mp, tuple(o[i] + z[i] * h / 2 for i in range(3)), I))
    if link == "base_link":
        for nm, mp, c in PAYLOAD:
            r = 0.05
            parts.append((nm, mp, c, [[mp * r * r * 2 / 5 if i == j else 0.0
                                       for j in range(3)] for i in range(3)]))
    M = sum(p[1] for p in parts)
    C = tuple(sum(p[1] * p[2][k] for p in parts) / M for k in range(3))
    I = [[0.0] * 3 for _ in range(3)]
    for _, mp, c, Ip in parts:
        d = tuple(c[k] - C[k] for k in range(3))
        S = shift(Ip, d, mp)
        for i in range(3):
            for j in range(3):
                I[i][j] += S[i][j]
    return M, C, I, parts


if __name__ == "__main__":
    tot = 0.0
    print("%-16s %8s %8s   %-26s %s" % ("link", "kg", "struct", "com (m)", "diag(I) kg m^2"))
    for link in BUILDERS:
        M, C, I, parts = link_props(link)
        tot += M
        print("%-16s %8.4f %8.4f   (%6.3f %6.3f %6.3f)   %.4g %.4g %.4g"
              % (link, M, parts[0][1], C[0], C[1], C[2], I[0][0], I[1][1], I[2][2]))
        if "-v" in sys.argv:
            for n, mp, c, _ in parts[1:]:
                print("      + %-24s %6.3f kg at (%6.3f %6.3f %6.3f)" % (n, mp, *c))
    print("%-16s %8.4f" % ("TOTAL", tot))
