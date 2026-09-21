"""The parts of this robot that are NOT bought: the aluminium between actuators.

Every actuator is held the way the vendor built it to be held:

  the PARENT link bolts a FLANGE RING onto the actuator's housing face, on its
  real bolt circle, and a SHROUD hangs off that ring over the barrel, so the
  actuator sits inside the link's own silhouette instead of beside it;

  the CHILD link bolts a HUB DISC onto the output flange face, on its real bolt
  circle, with a counterbore over the centering spigot.

Both are cut back by `keepout()` afterwards, which is the actuator's real
stepped profile grown by CLEAR. That is a guarantee, not a substitute for
designing it right: tools/verify.py asserts the intersection is zero.

All dimensions here come from lib.spec.IFACE, which tools/profile.py read off
the vendor STEP files.
"""
import math

from cadgen import build123d as _bd

from .spec import CLEAR, FIT, IFACE, PLATE, WALL

MM = 0.001


def _plane(origin, z_dir, x_hint=None):
    if x_hint is None:
        z = _bd.Vector(*z_dir)
        x_hint = (1, 0, 0) if abs(z.Z) > 0.9 else (0, 0, 1)
    return _bd.Plane(origin=origin, z_dir=z_dir, x_dir=_ortho(z_dir, x_hint))


def _ortho(z, x):
    z, x = _bd.Vector(*z), _bd.Vector(*x)
    if abs(x.dot(z)) > 0.9 * x.length * z.length:
        x = _bd.Vector(0, 0, 1) if abs(z.Z) < 0.9 * z.length else _bd.Vector(1, 0, 0)
    v = x - z * (x.dot(z) / z.dot(z))
    return (v.X, v.Y, v.Z)


def _tube(r_out, r_in, z0, z1):
    """Annular tube between two z stations, in the local frame."""
    h = z1 - z0
    c = _bd.Pos(0, 0, z0) * _bd.Cylinder(
        radius=r_out, height=h, align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    if r_in > 0:
        c -= _bd.Pos(0, 0, z0 - h) * _bd.Cylinder(
            radius=r_in, height=3 * h,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    return c


def _bolt_holes(bc, n, d, z0, z1, phase=0.0):
    out = None
    for i in range(n):
        a = 2 * math.pi * i / n + phase
        c = _bd.Pos(bc / 2 * math.cos(a), bc / 2 * math.sin(a), z0) * _bd.Cylinder(
            radius=d / 2, height=z1 - z0,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
        out = c if out is None else out + c
    return out


def _cap_screws(bc, n, d, z_head, head_h, phase=0.0):
    """Socket head cap screws sitting on a face - so the joint reads as bolted."""
    out = None
    for i in range(n):
        a = 2 * math.pi * i / n + phase
        c = _bd.Pos(bc / 2 * math.cos(a), bc / 2 * math.sin(a), z_head) * _bd.Cylinder(
            radius=d * 0.9, height=head_h,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
        out = c if out is None else out + c
    return out


_ENV = None


def envelope_bands(key):
    """The actuator's real radial envelope, read off the vendor mesh by
    tools/gen_envelope.py and clipped exact at its two mating faces."""
    global _ENV
    if _ENV is None:
        import json
        import os
        _ENV = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "_envelope.json")))
    return _ENV[key]["bands"]


def keepout(key, grow=0.0):
    """Solid of revolution over that envelope, plus any locating pins as
    discrete cylinders. Nothing of any bracket may be inside it; the mating
    faces themselves are excluded, because brackets are meant to sit on them."""
    d = IFACE[key]
    part = None
    if d.get("pin_n"):
        oz = d["out_z"] * MM
        part = _bolt_holes(d["pin_bc"] * MM, d["pin_n"], d["pin_d"] * MM + 2 * grow,
                           oz, oz + d["pin_h"] * MM + grow)
    for lo, hi, r in envelope_bands(key):
        if r <= 0.0 or hi - lo < 1e-9:
            continue
        c = _tube(r * MM + grow, 0.0, lo * MM - (grow if lo < -1 else 0.0),
                  hi * MM + (grow if hi > -1 else 0.0))
        part = c if part is None else part + c
    return part


def rear_plate_h(key):
    """Thickness of the rear mounting plate. One plate thickness - the pilot
    boss is allowed to stand proud of it and the next link's end has a
    clearance bore for it, which is lighter than burying the boss in 19 mm of
    aluminium (that cost 245 g a joint, 2 kg over the robot)."""
    return PLATE


def housing_od(key):
    return IFACE[key]["barrel"] * MM + 2 * (CLEAR + 0.5 * MM + WALL)


def pad_face(key, face):
    """(z of the face, bore diameter, bolt circle, count, thread, outward sign)."""
    d = IFACE[key]
    if face == "rear":
        return (d["rear_z"], max(d["rear_pilot"], d["rear_id"]), d["rear_bc"],
                d["rear_n"], d["rear_thread"], d["rear_od"], -1)
    return (d["hou_z"], max(d["hou_boss"], d["hou_id"]), d["hou_bc"],
            d["hou_n"], d["hou_thread"], d["hou_od"], +1)


def pad_thickness():
    return PLATE


def motor_pad(key, w, h, face="front"):
    """PARENT side: ONE flat wall of the link, bolted to ONE face of the actuator.

    This is how the open-source arms built on these actuators do it - OpenArm's
    own hardware notes say the motors are "supported on one side". The wall is
    a plate carrying the actuator's real bolt circle with a bore for whatever
    stands proud of the face; the actuator then cantilevers off it. No shroud,
    no second flange: wrapping a barrel around each actuator added 2.7 kg to
    this robot and held it no better than the eight M3 screws already do.

    The outline is the LINK's own section unioned with a boss the size of the
    mating face, so the bolt ring is fully backed - it is a wall of the link,
    not a ring around the motor.
    """
    z, bore_d, bc, n, thread, face_od, sgn = pad_face(key, face)
    z0 = z * MM if sgn > 0 else z * MM - PLATE
    plate = _bd.Pos(0, 0, z0 + PLATE / 2) * _stadium(w, h, PLATE)
    plate += _bd.Pos(0, 0, z0) * _bd.Cylinder(
        radius=face_od / 2 * MM, height=PLATE,
        align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    plate -= _bd.Pos(0, 0, z0 - PLATE) * _bd.Cylinder(
        radius=bore_d / 2 * MM + CLEAR + 0.5 * MM, height=3 * PLATE,
        align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    plate -= _bolt_holes(bc * MM, n, (thread + 0.4) * MM, z0 - PLATE, z0 + 2 * PLATE)
    plate += _cap_screws(bc * MM, n, thread * MM,
                         z0 + PLATE if sgn > 0 else z0 - thread * 0.65 * MM,
                         thread * 0.65 * MM)
    return plate


def output_hub(key):
    """CHILD side: the disc that bolts to the output flange, spigot counterbored."""
    d = IFACE[key]
    oz = d["out_z"] * MM
    r_o = d["out_od"] / 2 * MM
    top = oz + PLATE
    disc = _tube(r_o, max(d["out_bore"] - 2.0, 4.0) / 2 * MM, oz, top)
    if d["out_spigot"] > 0:
        cb_r = d["out_spigot"] / 2 * MM + CLEAR + 2.0 * MM
        cb_h = d["out_spigot_h"] * MM + CLEAR + 0.5 * MM
        disc -= _tube(cb_r, 0.0, oz - PLATE, oz + cb_h)
    disc -= _bolt_holes(d["out_bc"] * MM, d["out_n"], (d["out_thread"] + 0.4) * MM,
                        oz - PLATE, top + PLATE)
    if d.get("pin_n"):
        disc -= _bolt_holes(d["pin_bc"] * MM, d["pin_n"], (d["pin_d"] + 0.2) * MM,
                            oz - PLATE, top + PLATE)
    disc += _cap_screws(d["out_bc"] * MM, d["out_n"], d["out_thread"] * MM,
                        top, d["out_thread"] * 0.65 * MM)
    return disc


def _stadium(w, h, L):
    """A true stadium prism: a flat web with semicircular edges. An arm segment
    is a machined limb, not a crate.

    Extruded from one closed profile rather than unioned from a box and two
    cylinders: the three-primitive version left tangent seams that made a
    later fusion produce an inverted solid of negative volume - on one arm
    only, which is exactly how a silent modelling bug looks.
    """
    swap = h >= w
    if swap:
        w, h = h, w
    sec = _bd.extrude(_bd.Plane.XY * _bd.SlotOverall(w, h), amount=L / 2,
                      both=True)
    return _bd.Rot(0, 0, 90) * sec if swap else sec


def stadium_tube(a, b, w, h, x_dir, hollow=True):
    """A rounded-rectangular arm section from a to b - box intersected with a
    cylinder, so the arm reads as a machined limb rather than a crate."""
    a, b = _bd.Vector(*a), _bd.Vector(*b)
    v = b - a
    L = v.length
    if L < 1e-6:
        return None
    pl = _plane(((a.X + b.X) / 2, (a.Y + b.Y) / 2, (a.Z + b.Z) / 2),
                (v.X / L, v.Y / L, v.Z / L), x_dir)
    sec = _stadium(w, h, L)
    if hollow and L > 0.030 and min(w, h) > 4 * WALL:
        sec -= _stadium(w - 2 * WALL, h - 2 * WALL, L - 2 * WALL)
    return pl * sec


def hook_body(width, back, lip, thk):
    """The hooked hand: a vertical palm plate and a lip that closes the throat."""
    palm = _bd.Pos(0, 0, -back / 2) * _bd.Box(thk, width, back)
    finger = _bd.Pos(thk / 2 + lip / 2, -0.0, -back + thk / 2) * \
        _bd.Box(lip, width, thk)
    return palm + finger
