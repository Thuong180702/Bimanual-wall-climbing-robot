"""The four Livox Mid-360 on the torso, as the wall map sees through them.

The map the policy reads (`climb_task.noisy_map`) holds a rung only once a lidar has
seen it, and only as far along as it has been seen: a lidar cannot tell a rung's end
from where the robot's own arm, a guard tube, or another rung hides the rest of it.
Each frame, every rung's K cells are ray-cast from the four optical origins; a cell
counts as seen when it is inside one lidar's field of view and range and nothing lies
on the way. The map keeps what it has seen (a SLAM map accumulates) and spans, rung
by rung, from the first seen cell to the last -- a line fitted through the points.

The datasheet (livoxtech.com/mid-360/specs), and what each number does here:

    field of view     360 deg about its own axis x -7..+52 deg of elevation: a
                      direction is seen 38..97 deg off the axis (FOV_COS), so
                      straight along it is a 38 deg blind cone
    range             0.1 m blind zone .. 40 m at 10 % reflectivity (70 m at 80 %)
    frame rate        10 Hz: a frame lands every 2 or 3 control steps
                      (climb_task.scan_due), with the SLAM fix it carries
    range precision   2 cm at 10 m, 3 cm at 0.2 m (1 sigma), angular 0.15 deg:
                      what a fitted rung is off by, climb_task.MAP_*
    point rate        200 000 /s. ponytail: a cell in view counts as seen in the
                      frame it is in view; the non-repetitive pattern needs a few
                      frames to fill a 30 mm rung at 1 m (40-line equivalent, 1.5 deg
                      apart), so that is up to ~0.3 s early. A per-cell detection
                      draw would model it; nothing the policy does is that fast.

Where the lidars sit and what blocks them is the robot description's, generated from
the CAD (urdf/_gen_scene.py): a site per lidar at its optical origin, z its axis,
and an `occ_*` site per collision geom of the robot -- a motor's cylinder, a hook's
box, a guard tube's capsule, a bracket's tightest box. The other rungs block too, as
capsules. The walls do not: every rung stands in front of its own wall, and the
rooms are convex. A ray that starts inside a shape -- a lidar's own window -- is not
blocked by it.

Arithmetic only, `xp` = numpy or jax.numpy, so `rl.climb_task` and `rl.mjx_task`
run the same lines.
"""
import numpy as np, mujoco

NAMES = ('lidar_head_front', 'lidar_head_back', 'lidar_chest', 'lidar_back')
FOV_COS = (float(np.cos(np.radians(90.0 + 7.0))), float(np.cos(np.radians(90.0 - 52.0))))
RANGE = (0.10, 40.0)     # m
SCAN_HZ = 10             # frames per second
K = 32                   # cells per rung: 1.1 cm on the shortest (0.35 m), 15 on the
                         # longest (4.8 m, family 5) -- a seen end is off by half one
BIG = 1e9
_EPS = 1e-12


class Rig:
    """what the model says about the lidars and what blocks them, as arrays"""

    def __init__(self, m):
        self.lidar = np.array([m.site(n).id for n in NAMES])
        occ = [i for i in range(m.nsite) if m.site(i).name.startswith('occ_')]
        typ = m.site_type[occ]
        known = {int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                 int(mujoco.mjtGeom.mjGEOM_BOX)}
        assert set(typ.tolist()) <= known, set(typ.tolist()) - known
        pick = lambda t: np.array([s for s, ty in zip(occ, typ) if ty == int(t)], int)
        self.cap = pick(mujoco.mjtGeom.mjGEOM_CAPSULE)
        self.cyl = pick(mujoco.mjtGeom.mjGEOM_CYLINDER)
        self.box = pick(mujoco.mjtGeom.mjGEOM_BOX)
        self.cap_size = m.site_size[self.cap][:, :2].copy()     # radius, half length
        self.cyl_size = m.site_size[self.cyl][:, :2].copy()
        self.box_size = m.site_size[self.box].copy()            # half sizes


def _cyl(xp, o, u, c, ax, h, r):
    """(t_in, t_out) of the ray o + t u in a finite cylinder: centre c, unit axis ax,
    half length h, radius r (t_in > t_out: it misses)"""
    w = o - c
    ua, wa = (u * ax).sum(-1), (w * ax).sum(-1)
    up, wp = u - ua[..., None] * ax, w - wa[..., None] * ax
    A, B, C = (up * up).sum(-1), (up * wp).sum(-1), (wp * wp).sum(-1) - r * r
    disc = B * B - A * C
    sq = xp.sqrt(xp.maximum(disc, 0.0))
    ok = A > _EPS
    As = xp.where(ok, A, 1.0)
    t0 = xp.where(ok, xp.where(disc >= 0.0, (-B - sq) / As, BIG), xp.where(C <= 0.0, -BIG, BIG))
    t1 = xp.where(ok, xp.where(disc >= 0.0, (-B + sq) / As, -BIG), xp.where(C <= 0.0, BIG, -BIG))
    par = xp.abs(ua) <= _EPS                  # along the end faces: inside the slab or not
    us = xp.where(par, 1.0, ua)
    z0, z1 = (-h - wa) / us, (h - wa) / us
    inside = xp.abs(wa) <= h
    lo = xp.where(par, xp.where(inside, -BIG, BIG), xp.minimum(z0, z1))
    hi = xp.where(par, xp.where(inside, BIG, -BIG), xp.maximum(z0, z1))
    return xp.maximum(t0, lo), xp.minimum(t1, hi)


def _ball(xp, o, u, c, r):
    w = o - c
    B, C = (u * w).sum(-1), (w * w).sum(-1) - r * r
    disc = B * B - C
    sq = xp.sqrt(xp.maximum(disc, 0.0))
    return xp.where(disc >= 0.0, -B - sq, BIG), xp.where(disc >= 0.0, -B + sq, -BIG)


def _capsule(xp, o, u, c, ax, h, r):
    """the core cylinder and the two balls: a capsule is convex, so the ray is inside
    it over one interval, from the first entry to the last exit of the three"""
    ins, outs = zip(_cyl(xp, o, u, c, ax, h, r), _ball(xp, o, u, c - h[..., None] * ax, r),
                    _ball(xp, o, u, c + h[..., None] * ax, r))
    hit = [a <= b for a, b in zip(ins, outs)]
    t_in = xp.minimum(xp.minimum(xp.where(hit[0], ins[0], BIG), xp.where(hit[1], ins[1], BIG)),
                      xp.where(hit[2], ins[2], BIG))
    t_out = xp.maximum(xp.maximum(xp.where(hit[0], outs[0], -BIG), xp.where(hit[1], outs[1], -BIG)),
                       xp.where(hit[2], outs[2], -BIG))
    return t_in, t_out


def _box(xp, o, u, c, R, b):
    """slabs, in the box's frame (R: its columns, the box's axes in the world)"""
    p = ((o - c)[..., None] * R).sum(-2)
    v = (u[..., None] * R).sum(-2)
    par = xp.abs(v) <= _EPS
    vs = xp.where(par, 1.0, v)
    t0, t1 = (-b - p) / vs, (b - p) / vs
    inside = xp.abs(p) <= b
    lo = xp.where(par, xp.where(inside, -BIG, BIG), xp.minimum(t0, t1))
    hi = xp.where(par, xp.where(inside, BIG, -BIG), xp.maximum(t0, t1))
    return lo.max(-1), hi.min(-1)


def cells(xp, c, a, h):
    """(N, K, 3): the centres of each rung's K cells along its axis"""
    s = h[:, None] * ((2.0 * xp.arange(K) + 1.0) / K - 1.0)
    return c[:, None] + s[..., None] * a[:, None]


def visible(xp, rig, site_xpos, site_xmat, c, a, h, r, valid):
    """(N, K) bool: the cells of the rungs (centre c, unit axis a, half length h, radius
    r, `valid` the slots in use) that at least one lidar sees from where the robot is
    now. `site_xpos`, `site_xmat`: the model's, from MjData (C) or mjx.Data."""
    X = site_xmat.reshape(-1, 3, 3)
    O, A = site_xpos[rig.lidar], X[rig.lidar][:, :, 2]              # (4, 3)
    P = cells(xp, c, a, h)                                            # (N, K, 3)
    D = P[None] - O[:, None, None]                                    # (4, N, K, 3)
    L = xp.sqrt((D * D).sum(-1))
    U = D / L[..., None]
    cos = (U * A[:, None, None]).sum(-1)
    see = (cos >= FOV_COS[0]) & (cos <= FOV_COS[1]) & (L >= RANGE[0]) & (L <= RANGE[1])
    reach = L - r[None, :, None]                   # where the ray meets the rung itself
    o, u = O[:, None, None, None], U[..., None, :]                   # (4,1,1,1,3), (4,N,K,1,3)
    blocked = xp.zeros(L.shape, bool)

    def block(t_in, t_out, mask=None):
        hit = (t_in > 0.0) & (t_in <= t_out) & (t_in < reach[..., None])
        return (hit if mask is None else hit & mask).any(-1)

    if len(rig.cap):
        ax = X[rig.cap][:, :, 2]
        blocked = blocked | block(*_capsule(xp, o, u, site_xpos[rig.cap], ax,
                                            rig.cap_size[:, 1], rig.cap_size[:, 0]))
    if len(rig.cyl):
        ax = X[rig.cyl][:, :, 2]
        blocked = blocked | block(*_cyl(xp, o, u, site_xpos[rig.cyl], ax,
                                        rig.cyl_size[:, 1], rig.cyl_size[:, 0]))
    if len(rig.box):
        blocked = blocked | block(*_box(xp, o, u, site_xpos[rig.box], X[rig.box], rig.box_size))
    # the other rungs, as capsules; a slot not in use is nowhere
    other = valid[None, :] & ~xp.eye(len(h), dtype=bool)             # (N target, N occluder)
    blocked = blocked | block(*_capsule(xp, o, u, c, a, h, r), other[None, :, None, :])
    return ((see & ~blocked).any(0)) & valid[:, None]


def extent(xp, seen, h):
    """(lo, hi), each (N,): the seen span of each rung along its axis, from the first
    seen cell's near edge to the last one's far edge; (0, 0) for a rung never seen"""
    k = seen.shape[-1]
    cell = 2.0 * h / k
    first = xp.argmax(seen, -1)
    last = k - 1 - xp.argmax(seen[..., ::-1], -1)
    any_ = seen.any(-1)
    return (xp.where(any_, -h + first * cell, 0.0), xp.where(any_, -h + (last + 1) * cell, 0.0))
