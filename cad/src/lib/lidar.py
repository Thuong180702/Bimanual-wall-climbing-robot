"""The lidar mounts: a pad for each Mid-360, the roof on the head that tilts two of
them, the hats that sink the chest and back ones into the torso, and the tube cage
that stands proud of all four.

Everything is in torso_upper's frame, in metres, and follows spec.LIDARS: the
sensor frames are the truth and the metal is derived from them, with an assert
wherever the two could disagree.

Every sensor bolts onto a pad by its bottom face (4 x M3 on 48 x 36, two dowels),
and every pad stands on a plate at least 3 mm thick with more than 10 000 mm^2 of
it in the air - Livox's condition for the sensor to shed its 6.5 W - which is
also what spreads a knock into the torso instead of into four M3 threads.
"""
from cadgen import build123d as _bd

from .spec import GUARD, LIDAR_PAD, LIDARS, MID360, PLATE, TORSO_UP, WALL

M3, M4, M4_TAP = 0.0034, 0.0045, 0.0033
GUSSET = 0.005
CABLE = 0.016                     # hole for the M12 lead, under each head lidar's plug:
                                  # a right-angle plug, turned down into the chest between
                                  # the front (back) panel and the shoulder actuators
BRIM_Z = (0.035, 0.065, 0.095)   # M4 from outside the panel into each hat brim, at y = GUARD['y']
FOOT = (0.020, 0.018, 0.005)      # a guard's foot on its panel: across, up, thick; 2 x M4, 14 apart


def _v(p):
    return _bd.Vector(*p)


def sensor_plane(name):
    """the sensor frame: z = its axis, x = toward its connector, origin on its mounting face"""
    _, o, z, x = LIDARS[name]
    return _bd.Plane(origin=o, x_dir=x, z_dir=z)


def _cyl(pl, x, y, d, z0, z1):
    return pl * _bd.Pos(x, y, (z0 + z1) / 2) * _bd.Cylinder(d / 2, z1 - z0)


def _box(x0, x1, y0, y1, z0, z1):
    (x0, x1), (y0, y1), (z0, z1) = sorted((x0, x1)), sorted((y0, y1)), sorted((z0, z1))
    return _bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * _bd.Box(x1 - x0, y1 - y0, z1 - z0)


def _x_bolt(x0, x1, y, z, dia):
    return _bd.Pos((x0 + x1) / 2, y, z) * _bd.Cylinder(dia / 2, abs(x1 - x0), rotation=(0, 90, 0))


def sensor_holes(name, z0, z1):
    """the four M3 clearance holes and the two dowel holes, between z0 and z1 of the sensor frame"""
    pl = sensor_plane(name)
    cut = None
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = _cyl(pl, sx * MID360["holes"], sy * MID360["holes_y"], M3, z0, z1)
            cut = c if cut is None else cut + c
    for p in (MID360["dowel"], MID360["slot"]):
        cut += _cyl(pl, p[0], p[1], 0.003, z0, z1)
    return cut


def pad(name):
    P, t = LIDAR_PAD
    return (sensor_plane(name) * _bd.Pos(0, 0, -t / 2) * _bd.Box(P, P, t)
            - sensor_holes(name, -2 * t, t))


def tilted(name, host_o, n):
    """A lidar tilted off a flat host face: its pad, and a gusset under each end
    of the pad's hinge. -> (parts, the base plate's thickness the hinge asks for,
    the hole its lead goes down through)"""
    P, t = LIDAR_PAD
    _, S, a, x = LIDARS[name]
    S, a, x, n, ho = _v(S), _v(a), _v(x), _v(n), _v(host_o)
    u = (n - a * n.dot(a)).normalized()             # up the pad, away from the host
    hinge = S - a * t - u * (P / 2)                  # the pad's low edge, on the plate
    tb = n.dot(hinge - ho)
    assert 0.003 <= tb <= 0.008, "%s: a %.1f mm base plate" % (name, tb * 1000)
    w = (u - n * u.dot(n)).normalized()
    e = n.cross(w)
    hi = hinge + u * P
    tri = [(w.dot(p - ho), n.dot(p - ho)) for p in (hinge, hi, hi - n * (n.dot(hi - ho) - tb))]
    side = _bd.Plane(origin=ho, x_dir=w, z_dir=w.cross(n))     # local y = n
    gus = _bd.extrude(side * _bd.Polygon(*tri, align=None), amount=GUSSET / 2, both=True)
    e_s = e.dot(S - ho)
    parts = {name + "_pad": pad(name)}
    for k, s in (("a", -1), ("b", 1)):
        parts["%s_gusset_%s" % (name, k)] = _bd.Pos(e * (e_s + s * (P - GUSSET) / 2)) * gus
    plug = S + x * 0.040                              # under the M12 plug, off the pad's low edge
    plug = plug - n * n.dot(plug - ho)
    cable = _bd.Pos(plug) * _bd.Cylinder(CABLE / 2, 0.03)
    return parts, tb, cable


def head_roof():
    """The two head lidars back to back on a roof: a base plate on the lid (the
    heat plate, and what closes the lid's window), a pad for each 45 deg slope,
    and a gable at each end of the ridge. -> (parts, cut from the lid)"""
    d, _, h = TORSO_UP
    ho, n = (0.0, 0.0, h), (0.0, 0.0, 1.0)
    parts, cut, tbs = {}, None, []
    for name in ("lidar_head_front", "lidar_head_back"):
        p, tb, cable = tilted(name, ho, n)
        parts.update(p)
        tbs.append(tb)
        cut = cable if cut is None else cut + cable
    assert abs(tbs[0] - tbs[1]) < 1e-6, tbs
    tb = tbs[0]
    for sx in (-1, 1):                               # M3 through the lid's band, clear of the tabs
        for sy in (-0.074, 0.0, 0.074):
            cut += _bd.Pos(sx * 0.058, sy, h + (tb - WALL) / 2) * _bd.Cylinder(M3 / 2, tb + WALL + 0.002)
    parts["lidar_head_plate"] = _box(-0.064, 0.064, -0.077, 0.077, h, h + tb) - cut
    return parts, cut


def sunk(name):
    """A lidar sunk into the bay under the shoulder actuators, looking out through
    the front or the back panel: a hat whose crown is its pad and whose brims are
    screwed to that panel's side bands from outside. -> (parts, cut from the panel,
    which also takes the guard cage's feet)"""
    d, _, _ = TORSO_UP
    _, S, a, _ = LIDARS[name]
    s = a[0]
    assert abs(s) == 1.0 and a[1] == a[2] == 0.0, name
    xb, x_in, z0, z1 = S[0], s * (d / 2 - WALL), 0.020, 0.110
    parts = {name + "_crown": _box(xb - s * PLATE, xb, -0.065, 0.065, z0, z1)
             - sensor_holes(name, -2 * PLATE, 0.001)}
    cut = None
    for sy, k in ((1, "l"), (-1, "r")):
        for z in BRIM_Z:
            c = _x_bolt(x_in - s * (PLATE + 0.001), s * (d / 2 + 0.001), sy * GUARD["y"], z, M4)
            cut = c if cut is None else cut + c
        for y, z in _feet(sy):
            for dy in (-0.007, 0.007):
                cut += _x_bolt(x_in - s * 0.001, s * (d / 2 + 0.001), y + dy, z, M4)
        parts["%s_leg_%s" % (name, k)] = _box(xb - s * PLATE, x_in - s * PLATE,
                                              sy * 0.059, sy * 0.065, z0, z1)
        parts["%s_brim_%s" % (name, k)] = _box(x_in - s * PLATE, x_in, sy * 0.059,
                                               sy * (GUARD["y"] + 0.006), z0, z1)
    for k in [k for k in parts if "_brim_" in k]:
        parts[k] -= cut
    return parts, cut


def _strut_y():
    """where a bar's jog passes the strut's height"""
    (z0, z1), y0, y1 = GUARD["jog"], GUARD["y"], GUARD["head_y"]
    return y0 + (y1 - y0) * (GUARD["strut_z"] - z0) / (z1 - z0)


def _feet(sy):
    """(y, z) of the two feet of the bar on side sy (+1 left)"""
    return ((sy * GUARD["y"], GUARD["foot_z"]), (sy * _strut_y(), GUARD["strut_z"]))


def _bar(sx=1, sy=1):
    """the centre line of the bar at corner (sx, sy) of the torso (+1: front, left):
    its foot on the panel, out, up past the sunk lidar, in past the shoulders, up to
    the ring"""
    d, _, _ = TORSO_UP
    x0, xb = d / 2, d / 2 + GUARD["standoff"] - GUARD["tube_d"] / 2
    (zj0, zj1), y, yh, fz = GUARD["jog"], GUARD["y"], GUARD["head_y"], GUARD["foot_z"]
    pts = [(x0, y, fz), (xb, y, fz), (xb, y, zj0), (xb, yh, zj1), (xb, yh, GUARD["ring_z"])]
    return [(sx * x, sy * y_, z) for x, y_, z in pts]


def _ring_nodes():
    xb, yh = _bar()[-1][0], GUARD["head_y"]
    return [(sx * xb, sy * yh, GUARD["ring_z"]) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def guard_segments():
    """The cage as straight tubes of radius GUARD['tube_d'] / 2 -> [(name, p0, p1)] in
    torso_upper's frame, for the simulation: each bar from its foot bend up (the run
    back to the panel lies behind the bend's cap), and the ring. A bend is its corner
    here, so the foot bend stands R(sqrt 2 - 1) = 8 mm proud of the drawn 90 deg one
    along its diagonal, and the jog's 28 deg bends 0.6 mm."""
    out = []
    for sx, fb in ((1, "front"), (-1, "back")):
        for sy, lr in ((1, "l"), (-1, "r")):
            pts = _bar(sx, sy)
            for i in range(1, len(pts) - 1):
                out.append(("guard_%s_%s_%d" % (fb, lr, i), pts[i], pts[i + 1]))
    nodes = _ring_nodes()
    for i, (a, b) in enumerate(zip(nodes, nodes[1:] + nodes[:1])):
        out.append(("guard_ring_%d" % i, a, b))
    return out


def _tube(pts):
    """a 6061 tube bent along the polyline pts, GUARD['bend_r'] at every bend"""
    r, t = GUARD["tube_d"] / 2, GUARD["tube_t"]
    path = _bd.FilletPolyline(*pts, radius=GUARD["bend_r"])
    pl = _bd.Plane(origin=path @ 0, z_dir=path % 0)
    return _bd.sweep(pl * (_bd.Circle(r) - _bd.Circle(r - t)), path=path, transition=_bd.Transition.ROUND)


def _straight(p0, p1):
    r, t = GUARD["tube_d"] / 2, GUARD["tube_t"]
    v = _v(p1) - _v(p0)
    pl = _bd.Plane(origin=p0, z_dir=v)
    return _bd.extrude(pl * (_bd.Circle(r) - _bd.Circle(r - t)), amount=v.length)


def cage():
    """The guard cage: a bent tube up each corner of the torso and a ring over the
    head joining their tops, all 6061 like the rest of the frame. Each bar leaves its
    panel's bottom band on a foot, turns up, runs GUARD['standoff'] proud of the panel
    past the sunk lidar, jogs in past the shoulders and rises to the ring; a strut ties
    it to the panel's top band on the way. A rung the body climbs past meets a bend,
    rides up it onto the bar and on up to the ring, never an edge and never a lidar."""
    r = GUARD["tube_d"] / 2
    pts = _bar()
    x0, xb = pts[0][0], pts[1][0]                     # the panel, the bars' centre line
    fa, fz, ft = FOOT
    bar = _tube(pts)
    (fy0, fz0), (fy1, fz1) = _feet(1)
    for fy, fzz in ((fy0, fz0), (fy1, fz1)):
        foot = _box(x0, x0 + ft, fy - fa / 2, fy + fa / 2, fzz - fz / 2, fzz + fz / 2)
        for dy in (-0.007, 0.007):
            foot -= _x_bolt(x0 - 0.001, x0 + ft + 0.001, fy + dy, fzz, M4)
        bar += foot
    bar += _bd.Pos(x0 + (xb - x0) / 2, fy1, fz1) * _bd.Cylinder(0.005, xb - x0, rotation=(0, 90, 0))
    out = {}
    for sy, k in ((1, "l"), (-1, "r")):
        left = bar if sy > 0 else _bd.mirror(bar, about=_bd.Plane.XZ)
        out["guard_front_" + k] = left
        out["guard_back_" + k] = _bd.mirror(left, about=_bd.Plane.YZ)
    ring = None
    nodes = _ring_nodes()
    for a, b in zip(nodes, nodes[1:] + nodes[:1]):
        seg = _straight(a, b)
        ring = seg if ring is None else ring + seg
    for n in nodes:
        ring += _bd.Pos(*n) * _bd.Sphere(r + 0.001)
    out["guard_ring"] = ring
    return out


def torso_upper_mounts():
    """-> (parts, {torso_upper part: what to cut from it})"""
    parts, cuts = {}, {}
    p, cuts["lid"] = head_roof()
    parts.update(p)
    for name, panel in (("lidar_chest", "front"), ("lidar_back", "back")):
        p, cuts[panel] = sunk(name)
        parts.update(p)
    parts.update(cage())
    return parts, cuts
