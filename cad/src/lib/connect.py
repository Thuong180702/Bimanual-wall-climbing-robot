"""Connectors: the parts that bolt one actuator to the next.

Everything here is drawn against the actuator's REAL mating geometry, which
tools/profile.py read off the vendor STEP files and lib.spec.IFACE records.

A connector is made of exactly three kinds of feature:

  output_plate   a plate that sits flat on an actuator's OUTPUT flange, with a
                 recess for its centering spigot / locating pins and a
                 clearance hole for every one of its output bolts
  face_plate     a plate that sits flat on ONE housing face of the next
                 actuator, with a bore for whatever stands proud of that face
                 and a clearance hole for every one of its mounting bolts
  rib            a straight bar of metal joining the two plates

and it must come out as ONE connected solid that touches both actuators only
on those two faces. tools/check_mount.py asserts all three.

Nothing wraps an actuator. Each one is held by the bolts on a single face and
cantilevers off it, which is how every open-source arm built on this class of
actuator does it (OpenArm's hardware notes: motors "supported on one side").
"""
import math

from cadgen import build123d as _bd

from .spec import CLEAR, IFACE, PLATE, WALL

MM = 0.001


def plane_at(origin, body_dir, x_hint=None):
    """The actuator's own frame placed in a link: +Z is the way its output faces."""
    z = tuple(-v for v in body_dir)
    if x_hint is None:
        x_hint = (1, 0, 0) if abs(z[2]) > 0.9 else (0, 0, 1)
    zv, xv = _bd.Vector(*z), _bd.Vector(*x_hint)
    if abs(xv.dot(zv)) > 0.9:
        xv = _bd.Vector(0, 0, 1) if abs(zv.Z) < 0.9 else _bd.Vector(1, 0, 0)
    xv = xv - zv * (xv.dot(zv) / zv.dot(zv))
    return _bd.Plane(origin=origin, x_dir=(xv.X, xv.Y, xv.Z), z_dir=z)


def _disc(r, z0, z1):
    return _bd.Pos(0, 0, z0) * _bd.Cylinder(
        radius=r, height=z1 - z0,
        align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))


def _holes(bc, n, d, z0, z1):
    out = None
    for i in range(n):
        a = 2 * math.pi * i / n
        c = _bd.Pos(bc / 2 * math.cos(a), bc / 2 * math.sin(a), z0) * _bd.Cylinder(
            radius=d / 2, height=z1 - z0,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
        out = c if out is None else out + c
    return out


def _screws(bc, n, d, z_seat, up=True):
    """Socket head cap screws seated on a face, so the joint reads as bolted."""
    out = None
    h = d * 0.65
    for i in range(n):
        a = 2 * math.pi * i / n
        z0 = z_seat if up else z_seat - h
        c = _bd.Pos(bc / 2 * math.cos(a), bc / 2 * math.sin(a), z0) * _bd.Cylinder(
            radius=d * 0.9, height=h,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
        out = c if out is None else out + c
    return out


def output_plate(key, od, t=PLATE):
    """Plate bolted to the actuator's OUTPUT flange, in that actuator's frame.

    The flange face is at out_z and its spigot stands proud of it toward +Z, so
    the plate lies on the face and grows in +Z with a recess over the spigot.
    """
    d = IFACE[key]
    z0 = d["out_z"] * MM
    p = _disc(od / 2, z0, z0 + t)
    if d["out_spigot"] > 0:
        p -= _disc(d["out_spigot"] / 2 * MM + CLEAR + 1.0 * MM,
                   z0 - t, z0 + d["out_spigot_h"] * MM + CLEAR + 0.5 * MM)
    if d.get("pin_n"):
        p -= _holes(d["pin_bc"] * MM, d["pin_n"], (d["pin_d"] + 0.3) * MM,
                    z0 - t, z0 + t * 2)
    p -= _disc(max(d["out_bore"] - 2.0, 4.0) / 2 * MM, z0 - t, z0 + t * 2)
    p -= _holes(d["out_bc"] * MM, d["out_n"], (d["out_thread"] + 0.4) * MM,
                z0 - t, z0 + t * 2)
    p += _screws(d["out_bc"] * MM, d["out_n"], d["out_thread"] * MM, z0 + t)
    return p


def mate_bore_r(key, face="front"):
    """Radius a plate or web has to keep clear at one of an actuator's faces.

    Behind the REAR flange stands the pilot boss that centres a plate on it.

    Where a boss stands out of that face, stand off the boss. Where none does,
    the face is a plain annulus and the only thing to miss is the rotor turning
    just inside it - so the plate may overhang the annulus a little and pick up
    the metal its own bolts need. On the AK60-39 that is the difference between
    1.6 mm of aluminium beside each M3 and 4.8 mm.
    """
    d = IFACE[key]
    if face == "rear":
        return max(d["rear_pilot"], d["rear_id"]) / 2 * MM + CLEAR + 0.5 * MM
    if d["hou_boss_h"] > 0:
        return d["hou_boss"] / 2 * MM + CLEAR + 0.5 * MM
    return d["out_od"] / 2 * MM + 0.002


def face_od(key, face):
    """How wide a plate on that face should be.

    The face itself, unless the actuator's bolt circle sits so close to its rim
    that a socket head would hang over the edge - then as wide as the heads
    need, and never wider than the actuator's own body. The AK45-36 is the one
    that needs it: M3 at Ø48 on a Ø53 face leaves 0.8 mm of metal outside the
    clearance hole and the head overhangs by 0.25 mm. Ø55 is its barrel, so
    the plate still finishes flush with the actuator.
    """
    d = IFACE[key]
    if face == "rear":
        od, bc, th = d["rear_od"], d["rear_bc"], d["rear_thread"]
    else:
        od, bc, th = d["hou_od"], d["hou_bc"], d["hou_thread"]
    return min(d["od"], max(od, bc + 2 * (0.9 * th + 0.8))) * MM


def face_plate(key, face, od, t=PLATE):
    """Plate bolted to ONE housing face of the actuator, in that actuator's frame.

    'front' is the annulus beside the output; 'rear' is the flange at the other
    end of the barrel. The plate lies on the face and grows away from the body.
    """
    d = IFACE[key]
    if face == "rear":
        bc, n, th = d["rear_bc"], d["rear_n"], d["rear_thread"]
        br = mate_bore_r(key, "rear")
        z0, z1 = d["rear_z"] * MM - t, d["rear_z"] * MM
        seat, up = z0, False
    else:
        bc, n, th = d["hou_bc"], d["hou_n"], d["hou_thread"]
        br = mate_bore_r(key, "front")
        z0, z1 = d["hou_z"] * MM, d["hou_z"] * MM + t
        seat, up = z1, True
    p = _disc(od / 2, z0, z1)
    p -= _disc(br, z0 - t, z1 + t)
    p -= _holes(bc * MM, n, (th + 0.4) * MM, z0 - t, z1 + t)
    p += _screws(bc * MM, n, th * MM, seat, up)
    return p


def bolt_access(key, face, reach):
    """The bolt holes continued on through whatever structure covers them.

    A plate gets a clearance hole for every bolt, but the web fused on top of
    that plate then sits over the same holes and there is no way to get a
    driver in. Real brackets carry the hole straight through. `reach` is how
    far past the face to take it, in the actuator's own frame.
    """
    d = IFACE[key]
    if face == "output":
        bc, n, th, z0 = d["out_bc"], d["out_n"], d["out_thread"], d["out_z"] * MM
    elif face == "rear":
        bc, n, th, z0 = d["rear_bc"], d["rear_n"], d["rear_thread"], d["rear_z"] * MM
        return _holes(bc * MM, n, (th + 0.4) * MM, z0 - reach, z0)
    else:
        bc, n, th, z0 = d["hou_bc"], d["hou_n"], d["hou_thread"], d["hou_z"] * MM
    return _holes(bc * MM, n, (th + 0.4) * MM, z0, z0 + reach)


def bolt_ring(bc, n, d, z0, z1):
    """A ring of clearance holes, for the one joint that is not an actuator's."""
    return _holes(bc, n, d, z0, z1)


def rib(p0, p1, w, h, x_dir):
    """A straight bar from p0 to p1, w across x_dir and h across the other axis."""
    a, b = _bd.Vector(*p0), _bd.Vector(*p1)
    v = b - a
    L = v.length
    if L < 1e-9:
        return None
    pl = plane_at(((a.X + b.X) / 2, (a.Y + b.Y) / 2, (a.Z + b.Z) / 2),
                  (-v.X / L, -v.Y / L, -v.Z / L), x_dir)
    return pl * _bd.Box(w, h, L)


_ENV = None


def envelope_bands(key):
    """The actuator's real radial envelope, read off the vendor mesh by
    tools/gen_envelope.py and clipped exact at its mating faces."""
    global _ENV
    if _ENV is None:
        import json
        import os
        _ENV = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "_envelope.json")))
    return _ENV[key]["bands"]


def keepout(key):
    """Solid of revolution over that envelope, plus any locating pins. No part
    of a connector may be inside it; the mating faces are excluded, because
    connectors are meant to sit on them."""
    d = IFACE[key]
    part = None
    if d.get("pin_n"):
        oz = d["out_z"] * MM
        part = _holes(d["pin_bc"] * MM, d["pin_n"], d["pin_d"] * MM,
                      oz, oz + d["pin_h"] * MM)
    for lo, hi, r in envelope_bands(key):
        if r <= 0.0 or hi - lo < 1e-9:
            continue
        c = _disc(r * MM, lo * MM, hi * MM)
        part = c if part is None else part + c
    return part


def witness(key, face):
    """A 0.2 mm slab sitting on a mating face, on the connector's side of it.
    How much of it the connector fills is how much of that face it presses on."""
    d = IFACE[key]
    skin = 0.2 * MM
    if face == "output":
        z = d["out_z"] * MM
        idd = max(d["out_spigot"] + 4.0, d["out_bore"])
        return _disc(d["out_od"] / 2 * MM, z, z + skin) - \
            _disc(idd / 2 * MM, z - skin, z + 2 * skin)
    if face == "rear":
        z = d["rear_z"] * MM
        return _disc(d["rear_od"] / 2 * MM, z - skin, z) - \
            _disc(max(d["rear_pilot"], d["rear_id"]) / 2 * MM, z - 2 * skin, z + skin)
    z = d["hou_z"] * MM
    return _disc(d["hou_od"] / 2 * MM, z, z + skin) - \
        _disc(max(d["hou_boss"], d["hou_id"]) / 2 * MM, z - skin, z + 2 * skin)


def _bez(p0, p1, p2, p3, t):
    a = (1 - t) ** 3
    b = 3 * (1 - t) ** 2 * t
    c = 3 * (1 - t) * t ** 2
    d = t ** 3
    return tuple(a * p0[k] + b * p1[k] + c * p2[k] + d * p3[k] for k in range(3))


def _bez_tan(p0, p1, p2, p3, t):
    a = -3 * (1 - t) ** 2
    b = 3 * (1 - t) ** 2 - 6 * (1 - t) * t
    c = 6 * (1 - t) * t - 3 * t ** 2
    d = 3 * t ** 2
    v = [a * p0[k] + b * p1[k] + c * p2[k] + d * p3[k] for k in range(3)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def horn(a_ctr, a_dir, a_r, b_ctr, b_dir, b_r, wall, bulge=0.45, n=9):
    """A lofted shell joining the rim of one mounting face to the rim of the other.

    This is the shape the open-source arms use: the body follows the outline of
    both faces and flows between them, instead of two straight bars butted on.
    The path is a cubic that leaves each face along its own normal, so the
    transition is tangent at both ends and bulges away from the actuator in
    between - which is also what keeps it clear of the rotor.
    """
    d = math.dist(a_ctr, b_ctr) * bulge
    p1 = tuple(a_ctr[k] + a_dir[k] * d for k in range(3))
    p2 = tuple(b_ctr[k] - b_dir[k] * d for k in range(3))

    def shell(scale):
        secs = []
        for i in range(n):
            t = i / (n - 1.0)
            c = _bez(a_ctr, p1, p2, b_ctr, t)
            z = _bez_tan(a_ctr, p1, p2, b_ctr, t)
            r = (a_r + (b_r - a_r) * t) * scale
            pl = _bd.Plane(origin=c, z_dir=z)
            secs.append(_bd.Face(_bd.Circle(r).wire()).located(pl.location))
        return _bd.loft(secs)

    outer = shell(1.0)
    inner_scale_a = max(0.18, (a_r - wall) / a_r)
    inner = None
    secs = []
    for i in range(n):
        t = i / (n - 1.0)
        c = _bez(a_ctr, p1, p2, b_ctr, t)
        z = _bez_tan(a_ctr, p1, p2, b_ctr, t)
        r0 = a_r * inner_scale_a
        r = (r0 + ((b_r - wall) - r0) * t)
        pl = _bd.Plane(origin=c, z_dir=z)
        secs.append(_bd.Face(_bd.Circle(r).wire()).located(pl.location))
    inner = _bd.loft(secs)
    return outer - inner


def _tangent(pu, pv, r, cu, cv, side):
    """Tangent point from (pu, pv) to the circle of radius r centred (0, cv).

    Two exist; `side` picks the one further out in +u or -u, which is the one
    the outline wraps with.
    """
    du, dv = pu - cu, pv - cv
    d = max(math.hypot(du, dv), r * 1.02)
    al = math.acos(min(1.0, r / d))
    uu, uv = du / d, dv / d
    ca, sa = math.cos(al), math.sin(al)
    cand = [(r * (uu * ca + uv * sa), r * (-uu * sa + uv * ca)),
            (r * (uu * ca - uv * sa), r * (uu * sa + uv * ca))]
    t = max(cand) if side > 0 else min(cand)
    return (cu + t[0], cv + t[1])


def cheek_profile(half_w, z_top, r_b, z_b, c=0.0, u_b=0.0):
    """2D outline of a cheek plate: the straight edge on the mounting face that
    shows edge-on, two TANGENT lines across to the face that shows as a circle,
    and that face's rim.

    Works either way round - the circle may be above the straight edge or below
    it - because which end is which depends only on how the two actuators sit.

    `c` offsets the straight edge sideways and `u_b` the rim. A leg of an angle bracket has to
    hand over to the other leg, which is not on the joint axis, so its taper
    leans that way: symmetric, it leaves two square corners of metal either
    side of the hand-over doing nothing.

    Built as ONE closed wire. Unioning a polygon with a circle looked the same
    but left two faces, and extruding those pushed them in opposite directions -
    the plate came out as two pieces that never touched.
    """
    lo, hi = c - half_w, c + half_w
    t1 = _tangent(hi, z_top, r_b, u_b, z_b, +1)
    t2 = _tangent(lo, z_top, r_b, u_b, z_b, -1)
    # the arc takes the long way round, through the point on the rim furthest
    # from the straight edge
    far = (u_b, z_b - math.copysign(r_b, z_top - z_b))
    outline = (_bd.Polyline((lo, z_top), (hi, z_top), t1)
               + _bd.ThreePointArc(t1, far, t2)
               + _bd.Polyline(t2, (lo, z_top)))
    return _bd.make_face(outline)


def hook_profile(neck_c, neck_w, z_neck, half_w, z_wide, z_bot, r, bow=0.18):
    """The hooked hand's outline, in its own plane.

    A neck the width of the web it hands over to, two flanks spreading to the
    blade, and the blade down to the lip. The actuator can only be reached
    from one side, so the neck is off to one side and the two flanks are not
    the same length - straight, the long one reads as a flat triangular wing.
    They are arcs instead, hollowed by `bow` of their own spread, which is
    what the throat of a hook looks like and takes metal out rather than
    adding it.
    """
    lo, hi = neck_c - neck_w / 2.0, neck_c + neck_w / 2.0
    zm = (z_neck + z_wide) / 2.0
    ml = ((lo - half_w) / 2.0 + bow * abs(lo + half_w), zm)
    mr = ((hi + half_w) / 2.0 - bow * abs(half_w - hi), zm)
    outline = (_bd.Polyline((lo, z_neck), (hi, z_neck))
               + _bd.ThreePointArc((hi, z_neck), mr, (half_w, z_wide))
               + _bd.Polyline((half_w, z_wide), (half_w, z_bot),
                              (-half_w, z_bot), (-half_w, z_wide))
               + _bd.ThreePointArc((-half_w, z_wide), ml, (lo, z_neck)))
    face = _bd.make_face(outline)
    sh = [v for v in face.vertices() if abs(v.Y - z_wide) < 1e-9]
    return _bd.fillet(sh, r) if sh else face


def cheek(half_w, z_top, r_b, z_b, w0, t, bore=None, lighten=None,
          across="Y", c=0.0, u_b=0.0):
    """The cheek plate as a solid: profile extruded across, bored for the rotor.

    `across` is the axis the plate is thick along - "Y" lays it in the XZ plane,
    "X" in the YZ plane. A wrist whose two axes are perpendicular to each other
    needs one of each, because no single plate lies flat on both faces.

    `w0` is the mating face the plate starts on and `t` is how thick it is,
    SIGNED: a plate on a front face grows along that axis, a plate on a rear
    flange grows back the other way, and both start on their own face.
    """
    plane = _bd.Plane.XZ if across == "Y" else _bd.Plane.YZ
    rot = _bd.Rot(-90, 0, 0) if across == "Y" else _bd.Rot(0, 90, 0)
    tt = abs(t)
    lo = w0 if t > 0 else w0 - tt
    part = _bd.extrude(plane * cheek_profile(half_w, z_top, r_b, z_b, c, u_b),
                       amount=tt)
    bb = part.bounding_box()
    d = lo - (bb.min.Y if across == "Y" else bb.min.X)
    part = (_bd.Pos(0, d, 0) if across == "Y" else _bd.Pos(d, 0, 0)) * part
    holes = ([(u_b, z_b, bore / 2)] if bore else []) + list(lighten or [])
    for cu, cv, cr in holes:
        pos = (_bd.Pos(cu, lo - tt, cv) if across == "Y"
               else _bd.Pos(lo - tt, cu, cv))
        part -= pos * rot * _bd.Cylinder(
            radius=cr, height=3 * tt,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    return part
