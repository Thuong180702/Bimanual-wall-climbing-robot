"""ENCOS / CubeMars actuator bodies for the real robot, drawn from the datasheet drawings
(ENCOS joint-module datasheet V3.10; docs/superpowers/specs/2026-11-25-real-robot-cad-plan.md).
No vendor STEP exists for ENCOS, so these are envelopes with the mounting interfaces:
housing rings, output pilots and bolt circles, finned barrels.

Canonical actuator frame (as cad/src/purchased): +Z the axis, z = 0 the OUTPUT face, the
body at z < 0. Millimetres.
"""
from cadgen import build123d as bd

# key: dia, length, front ring (n, M, pcd), output (pilot dia, n, M, pcd), rear (n, M, pcd),
#      finned barrel length, centre hole dia
SPEC = {
    "A13720": dict(d=170, L=84.5, front=(12, 4, 155), out=(70, 3, 5, 40), rear=(20, 4, 155), fin=50, hole=29),
    "A10020": dict(d=124, L=85.0, front=(10, 4, 116.5), out=(62, 6, 5, 50), rear=None, fin=66, hole=8),
    # EC-A10020-P1-12 (forearm roll): the P2-24's diameter and interfaces, 60 mm long (cad/tools/real_layout.py;
    # an estimate until checked against the V3.10 drawing)
    "A10020s": dict(d=124, L=60.0, front=(10, 4, 116.5), out=(62, 6, 5, 50), rear=None, fin=40, hole=8),
    "A8112":  dict(d=105, L=53.0, front=(8, 4, 97), out=(54, 6, 4, 50), rear=(8, 4, 97), fin=0, hole=0),
}


def _holes(n, m, pcd, z, depth):
    return [bd.Pos(pcd / 2 * bd.Vector(1, 0, 0).rotate(bd.Axis.Z, 360 * i / n).X,
                   pcd / 2 * bd.Vector(1, 0, 0).rotate(bd.Axis.Z, 360 * i / n).Y, z - depth / 2)
            * bd.Cylinder(m / 2, depth) for i in range(n)]


def body(key):
    s = SPEC[key]
    d, L = s["d"], s["L"]
    b = bd.Pos(0, 0, -L / 2 - 1.5) * bd.Cylinder(d / 2, L - 3)            # housing
    b += bd.Pos(0, 0, -1.5) * bd.Cylinder(d / 2 - 4, 3)                       # front ring step
    b += bd.Pos(0, 0, -L + 1.5) * bd.Cylinder(d / 2 - 6, 3)                   # rear cover
    b += bd.Pos(0, 0, -1.0) * bd.Cylinder(s["out"][0] / 2, 2.0)               # output pilot, flush at z = 0
    if s["fin"]:                                                              # cooling fins: grooves round the barrel
        z0, n = -(L - s["fin"]) / 2 - 3, int(s["fin"] / 5)
        for i in range(n):
            g = bd.Pos(0, 0, z0 - 5 * i - 1.25) * (bd.Cylinder(d / 2 + 1, 2.5) - bd.Cylinder(d / 2 - 3, 2.5))
            b -= g
    for h in _holes(*s["front"], z=-3, depth=6) + _holes(*s["out"][1:], z=0, depth=6):
        b -= h
    if s["rear"]:
        for h in _holes(*s["rear"], z=-L + 6, depth=6):
            b -= h
    if s["hole"]:
        b -= bd.Pos(0, 0, -L / 2) * bd.Cylinder(s["hole"] / 2, L + 2)
    return b
