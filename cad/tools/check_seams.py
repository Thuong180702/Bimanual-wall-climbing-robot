"""Is each connector drawn as ONE part, or as blocks pushed together?

check_one.py asks whether a connector is one solid. A solid can still be two
blocks: where a web and its plate share a rim, or a leg and its neck share a
face, the kernel keeps both faces unless they are the same surface drawn the
same way round, and the part comes out with a seam across what should be one
skin. Every edge whose two faces are the SAME plane or the SAME cylinder is
such a seam. There should be none.

    python tools/check_seams.py            every link
    python tools/check_seams.py wrist_L    one, with where each seam is
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from lib.build import BUILDERS, fuse

TOL = 1e-7                  # metres


def seams(solid):
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(solid.wrapped, TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, m.Extent() + 1):
        fl = [bd.Face(f) for f in m.FindFromIndex(i)]
        if len(fl) != 2 or fl[0].wrapped.IsSame(fl[1].wrapped) \
                or fl[0].geom_type != fl[1].geom_type:
            continue
        a, b = (BRepAdaptor_Surface(f.wrapped) for f in fl)
        if fl[0].geom_type == bd.GeomType.PLANE:
            pa, pb = a.Plane(), b.Plane()
            same = (pa.Axis().Direction().IsParallel(pb.Axis().Direction(), 1e-6)
                    and pa.Distance(pb.Location()) < TOL)
        elif fl[0].geom_type == bd.GeomType.CYLINDER:
            ca, cb = a.Cylinder(), b.Cylinder()
            same = (abs(ca.Radius() - cb.Radius()) < TOL
                    and ca.Axis().IsCoaxial(cb.Axis(), 1e-6, TOL))
        else:
            continue
        if same:
            out.append((bd.Edge(m.FindKey(i)), fl[0].geom_type.name))
    return out


def main(names):
    bad = 0
    for n in names:
        found = seams(fuse(BUILDERS[n]()))
        bad += bool(found)
        print("  %-4s %-14s %d seam(s)" % ("PASS" if not found else "FAIL", n, len(found)))
        if len(names) == 1:
            for e, kind in found:
                b = e.bounding_box()
                print("       %-8s %5.1f mm   x %6.1f..%6.1f  y %6.1f..%6.1f  z %6.1f..%6.1f"
                      % (kind, e.length * 1e3, b.min.X * 1e3, b.max.X * 1e3, b.min.Y * 1e3,
                         b.max.Y * 1e3, b.min.Z * 1e3, b.max.Z * 1e3))
    print("\n  %s" % ("no seams" if not bad else "%d link(s) with seams" % bad))
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:] or [n for n in BUILDERS if not n.endswith("_R")]) else 0)
