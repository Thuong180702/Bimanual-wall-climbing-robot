"""The axial profile of a vendor actuator: which face is at which z, and how big.

This is what a bracket has to be designed against - the output flange face and
its spigot, the housing face the parent bolts to, and the barrel the parent's
cup has to clear.
"""
import collections, sys
from cadgen import build123d as bd
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_SurfaceType

s = bd.import_step(sys.argv[1])
print(sys.argv[1].split("/")[-1])
print("  -- planar faces normal to Z (the mating candidates) --")
print("  %8s %9s %9s %9s" % ("z", "area", "od", "id"))
g = collections.defaultdict(lambda: [0.0, 0.0, 1e9])
for f in s.faces():
    ad = BRepAdaptor_Surface(f.wrapped)
    if ad.GetType() != GeomAbs_SurfaceType.GeomAbs_Plane:
        continue
    d = ad.Plane().Axis().Direction()
    if abs(d.Z()) < 0.99:
        continue
    b = f.bounding_box()
    z = round((b.min.Z + b.max.Z) / 2, 1)
    od = max(abs(b.min.X), abs(b.max.X), abs(b.min.Y), abs(b.max.Y)) * 2
    e = g[z]
    e[0] += f.area
    e[1] = max(e[1], od)
for z, (a, od, _) in sorted(g.items(), reverse=True):
    if a < 150:
        continue
    import math
    idd = math.sqrt(max(0.0, (od / 2) ** 2 - a / math.pi)) * 2   # if it is an annulus
    print("  %8.1f %9.0f %9.1f %9.1f" % (z, a, od, idd))
print("  -- coaxial cylinders over 15 mm diameter --")
c = collections.defaultdict(lambda: [1e9, -1e9])
for f in s.faces():
    ad = BRepAdaptor_Surface(f.wrapped)
    if ad.GetType() != GeomAbs_SurfaceType.GeomAbs_Cylinder:
        continue
    cy = ad.Cylinder()
    d = cy.Axis().Direction()
    loc = cy.Axis().Location()
    if abs(d.Z()) < 0.99 or abs(loc.X) > 0.5 or abs(loc.Y) > 0.5:
        continue
    if cy.Radius() < 7.5:
        continue
    b = f.bounding_box()
    e = c[round(cy.Radius() * 2, 1)]
    e[0] = min(e[0], b.min.Z)
    e[1] = max(e[1], b.max.Z)
for od, (lo, hi) in sorted(c.items(), reverse=True):
    print("  d=%-7.1f z %7.1f .. %7.1f" % (od, lo, hi))
