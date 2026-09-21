"""Two questions about every bracket, asked of the real actuator geometry.

  1. Does it CLEAR the actuator?  No bracket may intersect the actuator's
     envelope (a solid of revolution proven to contain 100 % of the vendor
     solids - a raw boolean against the 51-solid vendor compound returns
     nonsense, including "intersections" whose box falls outside both inputs).

  2. Does it actually BOLT ON?  For each mating face, how much of that face
     does the bracket cover? A bracket that merely dodges the actuator passes
     question 1 and fails this one. That is the failure this file exists for:
     the first version of these brackets was a hoop floating around the motor
     with nothing pressed against any face of it.
"""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd

from lib.links import (BUILDERS, CHILD_OF, PARTS, PJOINT, flange_of,
                       mount_face)
from lib.shapes import _plane, _tube, keepout
from lib.spec import BODY_DIR, IFACE, MOTOR_OF, MOUNT_AT, fk

MM = 0.001
SKIN = 0.2 * MM          # how thin a witness slab is
MIN_COVER = 0.55         # bolt holes and windows take a share of every face


def _loc(origin, body_dir):
    return bd.Location(_plane(origin, tuple(-v for v in body_dir)))


def faces_of(key):
    """(name, z, inner d, outer d, which side the bracket is on) per mating face."""
    d = IFACE[key]
    # the output face is only usable OUTSIDE the centering spigot - a plate
    # has to be recessed over that - and outside the central cable bore.
    out_id = max(d["out_spigot"] + 4.0, d["out_bore"])
    # only the two faces a link actually bolts to. The rear flange is left
    # free: these actuators are mounted on one side, like every open-source
    # arm built on them.
    return [("front", d["hou_z"], d["hou_id"], d["hou_od"], +1),
            ("rear", d["rear_z"], d["rear_id"], d["rear_od"], -1),
            ("output", d["out_z"], out_id, d["out_od"], +1)]


def witness(key, name):
    for nm, z, idd, od, side in faces_of(key):
        if nm != name:
            continue
        z0 = z * MM if side > 0 else z * MM - SKIN
        return _tube(od / 2 * MM, idd / 2 * MM, z0, z0 + SKIN)
    raise KeyError(name)


def main():
    print("1. clearance - no bracket may touch an actuator")
    F = fk()
    world, mots = {}, {}
    for link in BUILDERS:
        o, R = F[link]
        pl = bd.Plane(origin=o, x_dir=(R[0][0], R[1][0], R[2][0]),
                      z_dir=(R[0][2], R[1][2], R[2][2]))
        # component-wise, not the fused link: a boolean against a fused,
        # subtracted solid returned an "intersection" bigger than either
        # operand. A union cannot overlap what none of its parts overlaps.
        src = PARTS[link[:-2] + "_L"]() if link.endswith("_R") else PARTS[link]()
        comps = {}
        for nm, q in src.items():
            if q is None:
                continue
            comps[nm] = bd.Location(pl) * (bd.mirror(q, about=bd.Plane.XZ)
                                           if link.endswith("_R") else q)
        world[link] = comps
        for cj in CHILD_OF.get(link, []):
            mots[cj[0]] = (bd.Location(pl) * _loc(flange_of(cj), BODY_DIR[cj[0]])
                           * keepout(MOTOR_OF[cj[0]]), MOTOR_OF[cj[0]])
    boxes = {k: v[0].bounding_box() for k, v in mots.items()}
    bad = 0
    for link in sorted(BUILDERS):
        for cn, comp in world[link].items():
            lb = comp.bounding_box()
            for jn, (m, key) in mots.items():
                mb = boxes[jn]
                if (lb.min.X > mb.max.X or lb.max.X < mb.min.X
                        or lb.min.Y > mb.max.Y or lb.max.Y < mb.min.Y
                        or lb.min.Z > mb.max.Z or lb.max.Z < mb.min.Z):
                    continue
                v = (comp & m).volume * 1e9
                if v > 1.0:
                    print("   FAIL %-16s %-14s cuts %-22s %9.1f mm3"
                          % (link, cn, jn, v))
                    bad += 1
    print("   %s" % ("clear" if not bad else "%d interpenetrations" % bad))

    print("\n2. contact - every mating face must actually be covered")
    for link in sorted(BUILDERS):
        part = BUILDERS[link]()
        checks = []
        pj = PJOINT.get(link)
        if pj is not None:
            # the parent actuator may be slid along its own axis by MOUNT_AT,
            # so its output face is not necessarily at the link origin
            hub_o = tuple(v * MOUNT_AT.get(pj[0], 0.0) for v in BODY_DIR[pj[0]])
            checks.append((pj[0], MOTOR_OF[pj[0]], "output",
                           _loc(hub_o, BODY_DIR[pj[0]])))
        for cj in CHILD_OF.get(link, []):
            L = math.sqrt(sum(v * v for v in cj[4]))
            u = tuple(v / L for v in cj[4])
            # exactly ONE face per actuator, the same one the bracket chose
            checks.append((cj[0], MOTOR_OF[cj[0]], mount_face(cj, u),
                           _loc(flange_of(cj), BODY_DIR[cj[0]])))
        for jn, key, nm, loc in checks:
            w = loc * witness(key, nm)
            frac = (part & w).volume / w.volume
            ok = frac >= MIN_COVER
            if not ok:
                bad += 1
            print("   %-4s %-16s %-22s %-7s %5.0f %% of the face"
                  % ("PASS" if ok else "FAIL", link, jn, nm, 100 * frac))
    print("\n%s" % ("all brackets clear the actuators and bolt to them"
                    if not bad else "%d problem(s)" % bad))
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
