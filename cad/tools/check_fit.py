"""Do the real actuator bodies fit? Exact, on solids, not on axis arithmetic.

Each motor is a cylinder of its catalogue OD and length, placed at its joint by
the forward kinematics, running from the output-flange plane along body_dir.
Every pair that is not the same joint gets a boolean intersection; anything with
volume is a clash. The torso and lower-torso envelopes are in the set too.
"""
import itertools, math, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from build123d import Cylinder, Box, Plane, Pos, Rot, Vector, Axis, Align
from lib.spec import (JOINTS, IFACE, MOTOR_OF, TORSO_LOW, TORSO_LOW_R, TORSO_UP,
                      WAIST_DIA, WAIST_H, motor_placements, fk)

MM = 0.001


def _cyl(origin, axis, length_m, r_m):
    """Cylinder starting at `origin`, running `length_m` along `axis`."""
    pl = Plane(origin=origin, z_dir=axis)
    return pl * Cylinder(radius=r_m, height=length_m,
                         align=(Align.CENTER, Align.CENTER, Align.MIN))


def envelopes(q=None):
    out = {}
    for name, key, o, ax, bd in motor_placements(q):
        d = IFACE[key]
        out[name] = _cyl(o, bd, d["length"] * MM, d["od"] / 2 * MM)
    F = fk(q)
    for name, link, (d, w, h) in (("#torso_low", "base_link", TORSO_LOW),
                                  ("#torso_up", "torso_upper", TORSO_UP)):
        o, R = F[link]
        pl = Plane(origin=o, x_dir=(R[0][0], R[1][0], R[2][0]),
                   z_dir=(R[0][2], R[1][2], R[2][2]))
        b = Pos(0, 0, h / 2) * Box(d, w, h)
        if name == "#torso_low":
            b &= Pos(0, 0, h / 2) * Cylinder(radius=TORSO_LOW_R, height=h)
        out[name] = pl * b
    o, R = F["waist_link"]
    out["#waist"] = Plane(origin=o, z_dir=(R[0][2], R[1][2], R[2][2])) * (
        Pos(0, 0, WAIST_H / 2) * Cylinder(radius=WAIST_DIA / 2, height=WAIST_H))
    return out


ADJACENT = set()
_byparent = {}
for _j in JOINTS:
    _byparent.setdefault(_j[1], []).append(_j[0])
for _j in JOINTS:
    for _k in _byparent.get(_j[2], []):
        ADJACENT.add(frozenset((_j[0], _k)))
# a motor and the shell it is bolted into are meant to touch
for _j in JOINTS:
    if _j[1] in ("base_link",):
        ADJACENT.add(frozenset((_j[0], "#torso_low")))
    if _j[1] == "torso_upper":
        ADJACENT.add(frozenset((_j[0], "#torso_up")))
ADJACENT.add(frozenset(("#torso_low", "#waist")))
ADJACENT.add(frozenset(("#torso_up", "#waist")))
ADJACENT.add(frozenset(("waist_yaw", "#waist")))


LIMIT = {j[0]: j[5] for j in JOINTS}


def clamp(q):
    """A pose is only meaningful inside the joint box; project it there."""
    return {k: min(max(v, LIMIT[k][0]), LIMIT[k][1]) for k, v in (q or {}).items()}


def report(q=None, label="q=0", adjacent_too=True, quiet=False):
    q = clamp(q)
    env = envelopes(q)
    bad = []
    for a, b in itertools.combinations(sorted(env), 2):
        if not adjacent_too and frozenset((a, b)) in ADJACENT:
            continue
        try:
            v = (env[a] & env[b]).volume
        except Exception:
            v = 0.0
        if v > 1e-9:
            bad.append((a, b, v * 1e9, frozenset((a, b)) in ADJACENT))
    if not quiet:
        print("--- %s ---" % label)
        if not bad:
            print("  clear: no pair of bodies intersects")
        for a, b, v, adj in sorted(bad, key=lambda r: -r[2]):
            print("  %-24s %-24s %10.0f mm^3%s"
                  % (a, b, v, "   (bolted pair, expected)" if adj else "   <-- CLASH"))
    return [r for r in bad if not r[3]]


def sweep(n=2000, seed=0):
    """Random poses in the joint box: how often does the machine hit itself?"""
    import random
    rng = random.Random(seed)
    lim = {j[0]: j[5] for j in JOINTS}
    hits, pairs = 0, {}
    for _ in range(n):
        q = {k: rng.uniform(*v) for k, v in lim.items()}
        bad = report(q, quiet=True)
        if bad:
            hits += 1
            for a, b, v, _ in bad:
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
    return hits, n, pairs


if __name__ == "__main__":
    import math as _m
    d = _m.radians
    HANG = {"shoulder_L_flexion": d(-160.1), "shoulder_R_flexion": d(-160.1),
            "forearm_L_roll": d(-5), "forearm_R_roll": d(5)}
    poses = [
        ("all joints at zero (arms straight down)", {}),
        ("scene.xml keyframe hang_on_rung", HANG),
        ("hang + reach: right arm up, left locked", dict(HANG, **{
            "shoulder_R_flexion": d(-180), "elbow_R": d(0),
            "shoulder_L_flexion": d(-140), "elbow_L": d(-60)})),
        ("arms abducted 90 deg", {"shoulder_L_abduction": d(90),
                                  "shoulder_R_abduction": d(-90)}),
        ("elbows fully flexed", {"elbow_L": d(-120), "elbow_R": d(-120)}),
        ("hang + both arms flared 40 deg", dict(HANG, **{
            "shoulder_L_abduction": d(40), "shoulder_R_abduction": d(-40)})),
        ("hang + flared 40, elbows bent 90 (the climber's pull)", dict(HANG, **{
            "shoulder_L_abduction": d(40), "shoulder_R_abduction": d(-40),
            "elbow_L": d(-90), "elbow_R": d(-90)})),
        ("arms at the -170 flexion stop", {"shoulder_L_flexion": d(-170),
                                           "shoulder_R_flexion": d(-170)}),
        ("waist hard over, arms down", {"waist_yaw": d(45)}),
        ("waist hard over while hanging", dict(HANG, waist_yaw=d(45))),
        ("README reach pose: abd 180, flex -177, elbow -114",
         {"shoulder_L_abduction": d(180), "shoulder_L_flexion": d(-177),
          "elbow_L": d(-114)}),
    ]
    n = 0
    for lab, q in poses:
        n += len(report(q, lab))
    print("\n%d clash(es) across %d named poses" % (n, len(poses)))
    if "--sweep" in sys.argv:
        h, N, pairs = sweep(int(sys.argv[sys.argv.index("--sweep") + 1]))
        print("\nrandom sweep: %d / %d poses self-collide (%.1f%%)"
              % (h, N, 100.0 * h / N))
        for (a, b), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]:
            print("  %-24s %-24s %5d" % (a, b, c))
