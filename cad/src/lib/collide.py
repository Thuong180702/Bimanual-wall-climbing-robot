"""What each link presents to the collision engine.

Two kinds of geom, and the split is deliberate.

Every ACTUATOR is a cylinder of its real diameter and length. That is exact,
and it is most of the swept volume of the arm.

Every bracket PART - a plate, a web, a shell - is its own mesh, which MuJoCo
takes the convex hull of. For a tapered plate the hull is the plate; for a
ring it is a disc, which is conservative and costs nothing because the only
thing inside a ring is the actuator it is bolted to, and MuJoCo does not test
a body against its own parent.

One hull per PART, not per link. Fused into one mesh, the upper torso's
wishbone, flanks and shoulder plates were one hull - a solid block out to the
shoulder plates - and the raised upper arm's hull sank 10-22 mm into it at
every flexion past -120 deg, 18-52 mm from any real metal: the hang keyframe
pressed the arms into the torso with 1500 N and held four shoulder actuators
at 88 % of peak torque doing it.

The HOOKS are the exception: their whole job is the throat between the palm
and the lip, and a convex hull fills it in. They keep the two explicit boxes
the RL work was built on, verbatim.

So is the lidar CAGE: a bent tube's hull is a slab, and the ring's a lid over the
head. Each tube is capsules along its centre line (lidar.guard_segments), exact
but at the bends. The four Mid-360s are a box and a cylinder each, as their STEP
measures. What holds them inside the torso - the hats under the chest and the back
lidar - meets nothing, and the head's roof (plate, pads, gussets) is one hull.
"""
import functools, re

from . import lidar
from .build import BUILDERS, CHILD_OF, mount_plane
from .spec import BODY_DIR, GUARD, HOOK, IFACE, LIDARS, MID360, MOTOR_OF

MM = 0.001


def link_collision(link):
    """[(kind, size, pos, z_dir, x_dir)] in the link frame, metres."""
    return [g for _, g in named_collision(link)]


@functools.lru_cache(maxsize=None)
def named_collision(link):
    """link_collision with a name for the geoms other tools find by it (None for
    the rest): '<lidar>_body' / '_window', and each guard tube's own"""
    out = []
    for cj in CHILD_OF.get(link, []):
        key = MOTOR_OF[cj[0]]
        d = IFACE[key]
        pl = mount_plane(cj, cj[4])
        L = d["length"] * MM
        c = pl.origin + pl.z_dir * (d["out_z"] * MM - L / 2.0)
        out.append(("cylinder", (d["od"] / 2 * MM, L), (c.X, c.Y, c.Z),
                    tuple(-v for v in BODY_DIR[cj[0]]), None))
    if link.startswith("hook"):
        t, b, l, w = HOOK["thk"], HOOK["back"], HOOK["lip"], HOOK["width"]
        out.append(("box", (t, w, b), (0.0, 0.0, -b / 2), (0, 0, 1), (1, 0, 0)))
        out.append(("box", (l, w, t), (t / 2 + l / 2, 0.0, -b + t / 2),
                    (0, 0, 1), (1, 0, 0)))
        key = MOTOR_OF[__import__("lib.build", fromlist=["PJOINT"]).PJOINT[link][0]]
        od = IFACE[key]["out_od"] * MM
        out.append(("cylinder", (od / 2, 0.030), (0.0, 0.0, 0.0),
                    (0, 1, 0) if link.endswith("_L") else (0, -1, 0), None))
    else:
        for name, _ in col_parts(link):
            out.append(("mesh", (name,), (0.0, 0.0, 0.0), (0, 0, 1), (1, 0, 0)))
    out = [(None, g) for g in out]
    for name, (host, o, z, x) in LIDARS.items():
        if host != link:
            continue
        (bx, by, bz), (w0, w1), r = MID360["base"], MID360["window"], MID360["window_r"]
        at = lambda h: tuple(o[i] + z[i] * h for i in range(3))
        out.append((name + "_body", ("box", (bx, by, bz), at(bz / 2), z, x)))
        out.append((name + "_window", ("cylinder", (r, w1 - w0), at((w0 + w1) / 2), z, None)))
    if link == "torso_upper":
        for name, p0, p1 in lidar.guard_segments():
            out.append((name, ("capsule", (GUARD["tube_d"] / 2,), (p0, p1), None, None)))
    return out


@functools.lru_cache(maxsize=None)
def col_parts(link):
    """[(mesh name, solid in metres)], one per CAD part of a non-hook link - but the
    lidar mounts: the cage is capsules (link_collision), the hats inside the torso
    nothing, and the head's roof one part"""
    parts, roof = [], []
    for k, v in BUILDERS[link]().items():
        if v is None or k.startswith("guard_") or k.startswith(("lidar_chest", "lidar_back")):
            continue
        if k.startswith("lidar_head"):
            roof.append(v)
            continue
        parts.append(("%s_col_%s" % (link, re.sub(r"\W+", "_", k)), v))
    if roof:
        parts.append(("%s_col_lidar_head_roof" % link, sum(roof[1:], roof[0])))
    return parts
