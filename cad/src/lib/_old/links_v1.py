"""Every link of the robot, as a solid in its own URDF frame.

One pattern builds the whole arm, because every link does the same job:

    output_hub  bolted to the PROXIMAL actuator's output flange
        |
    a rounded-rectangular section, offset just far enough to clear that
    actuator, running to
        |
    motor_housing  a flange ring bolted to the DISTAL actuator's housing face,
                   with a shroud that puts the actuator inside this link

The section has to step aside because an actuator's output flange and its body
are at the SAME end: the child bolts to the middle of a face whose outside is
already occupied by the parent's flange ring, so the child's structure can only
grow away from it. That is why a real arm zigzags around its joints instead of
running straight through them. The step is `out_z + PLATE + h/2` - as small as
the parts allow - and it does not accumulate: every joint axis is still on the
arm's centre line.

Finally every link is cut by `keepout()` for each actuator it touches. That is
a guarantee, and tools/verify.py asserts it holds.
"""
import math

from cadgen import build123d as _bd

from .shapes import (hook_body, housing_od, keepout, motor_pad, pad_face,
                     output_hub, stadium_tube, _plane)
from .spec import (AXIS, BODY_DIR, CLEAR, FIXED, HOOK, IFACE, JOINTS, MOTOR_OF,
                   MOUNT_AT, AL, PLATE, SHELL, TORSO_LOW, TORSO_LOW_R, TORSO_UP,
                   WAIST_DIA, WAIST_H, WALL)

MM = 0.001

CHILD_OF = {}
for _j in JOINTS:
    CHILD_OF.setdefault(_j[1], []).append(_j)
PJOINT = {_j[2]: _j for _j in JOINTS}


def flange_of(joint, origin=None):
    """Where that joint's actuator output face sits, in the PARENT link frame."""
    o = origin if origin is not None else joint[4]
    a = MOUNT_AT.get(joint[0], 0.0)
    bd = BODY_DIR[joint[0]]
    return tuple(o[k] + bd[k] * a for k in range(3))


def _at(origin, body_dir, x_hint=None):
    """Plane whose +Z is the actuator's own +Z - the direction its output faces."""
    return _plane(origin, tuple(-v for v in body_dir), x_hint)


def _section(link, pj, cj):
    """(across, along-the-parent-axis) of this link's section, metres."""
    w = housing_od(MOTOR_OF[cj[0]])
    h = max(0.42 * w, 0.030)
    return w, h


def _step(pj, h):
    """How far off the parent joint axis the section has to sit, and which way."""
    d = IFACE[MOTOR_OF[pj[0]]]
    return -(d["out_z"] * MM + PLATE + h / 2)     # along +body_dir; negative = out


def mount_face(cj, u):
    """Which of the actuator's two flanges the PARENT bolts to.

    One face per actuator - but which one depends on where the parent comes
    from. On a roll joint whose body points back up the link, the face nearest
    the parent is the REAR flange; everywhere else it is the front.
    """
    c_along = abs(sum(a * b for a, b in zip(cj[3], u))) > 0.9
    toward_parent = sum(a * b for a, b in zip(BODY_DIR[cj[0]], u)) < 0
    return "rear" if (c_along and toward_parent) else "front"


def sections(link):
    """[(a, b, w, h, x_hint)] - this link's structural sections, link frame.

    arm_link() builds the solids from this and link_collision() reports boxes
    from the same list, so the simulation envelope cannot drift from the CAD.
    """
    pj = PJOINT[link]
    p_axis, p_body, p_key = pj[3], BODY_DIR[pj[0]], MOTOR_OF[pj[0]]
    hub_o = tuple(v * MOUNT_AT.get(pj[0], 0.0) for v in p_body)
    out = []
    for cj in CHILD_OF.get(link, []):
        w, h = _section(link, pj, cj)
        off = cj[4]
        L = math.sqrt(sum(v * v for v in off))
        u = tuple(v / L for v in off)
        along = abs(sum(a * b for a, b in zip(p_axis, u))) > 0.9
        if along:
            shift = tuple(-v * (IFACE[p_key]["out_z"] * MM + PLATE) for v in p_body)
            x_hint = None
        else:
            shift = tuple(hub_o[k] + p_body[k] * _step(pj, h) for k in range(3))
            x_hint = (p_axis[1] * u[2] - p_axis[2] * u[1],
                      p_axis[2] * u[0] - p_axis[0] * u[2],
                      p_axis[0] * u[1] - p_axis[1] * u[0])
        # Stop the section ON the child's housing, not at the joint centre.
        # Running it to the centre drives it straight through the actuator -
        # which the old keep-out subtraction then quietly carved back out,
        # leaving a section that dodged the actuator instead of landing on it.
        c_key = MOTOR_OF[cj[0]]
        r_h = housing_od(c_key) / 2
        # which way the CHILD's axis runs decides where its housing presents a
        # face to land on: head-on (coaxial) it is the housing's far end plate,
        # side-on it is the housing's outer wall.
        c_along = abs(sum(a * b for a, b in zip(cj[3], u))) > 0.9
        face = mount_face(cj, u)
        if c_along:
            # head-on: stop on the far side of the pad it bolts to
            fz, _, _, _, _, _, sgn = pad_face(c_key, face)
            back = abs(fz) * MM + (PLATE if sgn < 0 else 0.0)
        else:
            back = r_h - WALL
        base = tuple(off[k] + (shift[k] - hub_o[k] if not along else 0.0)
                     for k in range(3))
        # a coaxial child's pilot boss stands proud of its rear plate; the
        # section's end gets a clearance bore for it, the way a real link's
        # end plate does
        bore = None
        if False:
            bd_ = BODY_DIR[cj[0]]
            r = IFACE[c_key]["rear_pilot"] / 2 * MM + CLEAR + 0.5 * MM
            zr = abs(IFACE[c_key]["rear_z"]) * MM
            c0 = tuple(cj[4][k] + bd_[k] * zr for k in range(3))
            bore = (c0, bd_, r, IFACE[c_key]["length"] * MM - zr + 0.01)
        back = _trim(shift, base, u, w, h, x_hint, bore, cj, back)
        b = tuple(base[k] - u[k] * back for k in range(3))
        out.append((shift, b, w, h, x_hint, bore))
    return out


def section_solid(a, b, w, h, x_hint, bore):
    sec = stadium_tube(a, b, w, h, x_hint)
    if sec is not None and bore is not None:
        c0, ax, r, L = bore
        sec = sec - (_bd.Location(_plane(c0, ax)) * _bd.Cylinder(
            radius=r, height=L,
            align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN)))
    return sec


_TRIM = {}


def _trim(a, base, u, w, h, x_hint, bore, cj, back):
    """How far the section really has to stop short, measured not estimated.

    The analytic guess puts the section's AXIS clear of the actuator; its
    corners can still be inside. So the guess is the starting point and the
    section is pulled back 2 mm at a time until a boolean against the
    actuator's envelope comes out empty, then one more for luck.
    """
    key = (cj[0], round(w, 4), round(h, 4), round(back, 4))
    if key in _TRIM:
        return _TRIM[key]
    ko = _bd.Location(_at(flange_of(cj), BODY_DIR[cj[0]])) * keepout(MOTOR_OF[cj[0]])
    d = back
    for _ in range(24):
        b = tuple(base[k] - u[k] * d for k in range(3))
        sec = section_solid(a, b, w, h, x_hint, bore)
        if sec is None or (sec & ko).volume <= 1e-12:
            break
        d += 0.002
    _TRIM[key] = d + 0.001
    return _TRIM[key]


def arm_parts(link):
    """The link's components, before they are fused.

    Kept separate on purpose. A boolean against a fused, subtracted solid is
    not trustworthy here - one returned an "intersection" bigger than either
    operand, and silently dropped a housing wall on one arm - so
    tools/check_mount.py tests these components one at a time, where every
    operand is a simple solid. A union cannot overlap what none of its parts
    overlaps.

    There is deliberately NO keep-out subtraction any more. The brackets are
    designed onto the actuator's real faces; cutting them back afterwards by a
    solid whose faces were coincident with theirs was both the boolean hazard
    above and a way to hide a mistake instead of failing on it.
    """
    pj = PJOINT[link]
    p_body, p_key = BODY_DIR[pj[0]], MOTOR_OF[pj[0]]
    hub_o = tuple(v * MOUNT_AT.get(pj[0], 0.0) for v in p_body)
    out = {"hub": _bd.Location(_at(hub_o, p_body)) * output_hub(p_key)}
    secs = sections(link)
    for i, cj in enumerate(CHILD_OF.get(link, [])):
        a, b, w, h, _x, _bore = secs[i]
        L = math.sqrt(sum(v * v for v in cj[4]))
        u = tuple(v / L for v in cj[4])
        out["pad:" + cj[0]] = _bd.Location(
            _at(flange_of(cj), BODY_DIR[cj[0]])) * motor_pad(
                MOTOR_OF[cj[0]], w, h, mount_face(cj, u))
    for i, d in enumerate(sections(link)):
        beam = section_solid(*d)
        if beam is not None:
            out["section%d" % i] = beam
    return out


def _fuse(parts):
    """Union the components, and refuse to return a broken solid.

    A fusion that fails here does not raise - it returns an inverted or
    partial solid, and it did: one arm's shoulder came out at -389 cm3 while
    the mirror image of it was fine. So the result is checked (positive, and
    at least as big as its largest part) and retried with a fuzzy boolean
    tolerance when the check fails.
    """
    live = [p for p in parts.values() if p is not None]
    big = max(p.volume for p in live)
    for tol in (None, 1e-6, 1e-5, 1e-4):
        part = live[0]
        for p in live[1:]:
            part = (part + p) if tol is None else part.fuse(p, tol=tol).clean()
        # only the FINISHED union can be judged: a running total is legitimately
        # smaller than the largest part until that part has been added.
        if part.volume >= big and part.volume > 0:
            return part
    raise RuntimeError(
        "fusion produced an invalid solid even with a fuzzy tolerance: "
        "%.2f cm3 from parts of %s cm3"
        % (part.volume * 1e6,
           ", ".join("%.1f" % (p.volume * 1e6) for p in live)))


def arm_link(link):
    return _fuse(arm_parts(link))


def hook_parts(link):
    pj = PJOINT[link]
    p_body, p_key = BODY_DIR[pj[0]], MOTOR_OF[pj[0]]
    loc = _bd.Location(_at((0.0, 0.0, 0.0), p_body))
    # the palm plate steps clear of the finger actuator, exactly as an arm
    # section does - it bolts to the output hub and grows away from the body
    step = -(IFACE[p_key]["out_z"] * MM + PLATE + HOOK["width"] / 2)
    off = tuple(v * step for v in p_body)
    return {"hub": loc * output_hub(p_key),
            "hook": _bd.Pos(*off) * hook_body(HOOK["width"], HOOK["back"],
                                              HOOK["lip"], HOOK["thk"])}


def hook_link(link):
    return _fuse(hook_parts(link))


def _windows(d, w, h, front=False):
    """Lightening windows. A torso panel is a shear web, not a pressure vessel."""
    cut = None
    for zc in (0.22, 0.50, 0.78):
        for sy in ((-1, 1, 0) if front else (-1, 1)):
            if sy == 0:
                c = _bd.Pos(0, 0, h * zc) * _bd.Box(d * 3, w * 0.52, h * 0.19)
            else:
                c = _bd.Pos(0, sy * w / 2, h * zc) * _bd.Box(d * 0.58, w * 3, h * 0.19)
            cut = c if cut is None else cut + c
    return cut


def base_parts():
    """Lower torso: a stadium-section shell carrying the waist actuator."""
    d, w, h = TORSO_LOW
    sec = _bd.Pos(0, 0, h / 2) * (_bd.Box(d, w, h)
                                  & _bd.Cylinder(radius=TORSO_LOW_R, height=h))
    inner = _bd.Pos(0, 0, h / 2) * (
        _bd.Box(d - 2 * SHELL, w - 2 * SHELL, h - 2 * PLATE)
        & _bd.Cylinder(radius=TORSO_LOW_R - SHELL, height=h - 2 * PLATE))
    wj = PJOINT["waist_link"]
    key = MOTOR_OF[wj[0]]
    bore = _bd.Pos(0, 0, wj[4][2] - IFACE[key]["length"] * MM - 0.02) * _bd.Cylinder(
        radius=housing_od(key) / 2, height=0.12,
        align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    shell = sec - inner - _windows(d, w, h, front=True) - bore
    loc = _bd.Location(_at(flange_of(wj), BODY_DIR[wj[0]]))
    L = math.sqrt(sum(v * v for v in wj[4]))
    u = tuple(v / L for v in wj[4])
    return {"shell": shell,
            "pad:waist_yaw": loc * motor_pad(key, TORSO_LOW[0], TORSO_LOW[1],
                                             mount_face(wj, u))}


def waist_parts():
    """Between the waist output and the upper torso floor."""
    wj = PJOINT["waist_link"]
    loc = _bd.Location(_at((0.0, 0.0, 0.0), BODY_DIR[wj[0]]))
    part = loc * output_hub(MOTOR_OF[wj[0]])
    z = FIXED[0][3][2]
    r = WAIST_DIA / 2
    tube = _bd.Cylinder(radius=r, height=z,
                        align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    tube -= _bd.Cylinder(radius=r - WALL, height=z,
                         align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    cap = _bd.Pos(0, 0, z - PLATE) * _bd.Cylinder(
        radius=r, height=PLATE, align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    cap -= _bd.Pos(0, 0, z - PLATE) * _bd.Cylinder(
        radius=r - 3 * WALL, height=3 * PLATE,
        align=(_bd.Align.CENTER, _bd.Align.CENTER, _bd.Align.MIN))
    return {"hub": part, "tube": tube, "cap": cap}


def torso_parts():
    """Upper torso frame, with a shoulder housing each side.

    The abduction actuators sit in housings that are part of this link, so they
    read as shoulders rather than as motors bolted to the outside of a box.
    """
    d, w, h = TORSO_UP
    shell = _bd.Pos(0, 0, h / 2) * _bd.Box(d, w, h)
    shell -= _bd.Pos(0, 0, h / 2) * _bd.Box(d - 2 * SHELL, w - 2 * SHELL,
                                            h - 2 * PLATE)
    shell -= _windows(d, w, h, front=True)
    out = {"shell": shell}
    for cj in CHILD_OF["torso_upper"]:
        key, c_body = MOTOR_OF[cj[0]], BODY_DIR[cj[0]]
        f = flange_of(cj)
        loc = _bd.Location(_at(f, c_body))
        out["pad:" + cj[0]] = loc * motor_pad(key, housing_od(key),
                                             housing_od(key) * 0.62)
        # a shoulder yoke, not a stalk: it starts inside the frame wall, is as
        # deep as the housing, and reaches back far enough to pick the actuator
        # up where MOUNT_AT has slid it.
        # a shoulder yoke, not a stalk: one hollow section running from inside
        # the frame wall out to where MOUNT_AT has slid the actuator. Solid, it
        # weighed 0.9 kg a side, which is more than the actuator it carries.
        od = housing_od(key)
        sg = 1 if cj[4][1] > 0 else -1
        y0 = sg * (w / 2 - 2 * SHELL)
        r_h = od / 2
        y_end = cj[4][1] - (1 if sg > 0 else -1) * (r_h - WALL)
        out["yoke:" + cj[0]] = stadium_tube(
            (cj[4][0], y0, cj[4][2]), (f[0], y_end, cj[4][2]),
            od, od * 0.50, x_dir=(0, 0, 1))
    return out


PARTS = {"base_link": base_parts, "waist_link": waist_parts,
         "torso_upper": torso_parts}
for _s in "LR":
    PARTS["hook_%s" % _s] = (lambda s=_s: hook_parts("hook_%s" % s))
    for _n in ("shoulder", "upper_arm", "humerus_rot", "fore_arm",
               "forearm_rot", "wrist", "wrist_dev"):
        PARTS["%s_%s" % (_n, _s)] = (
            lambda n=_n, s=_s: arm_parts("%s_%s" % (n, s)))

def _mirror(part):
    """A right-hand link is the LEFT one reflected in the sagittal plane.

    Not rebuilt from mirrored inputs: every joint offset and body direction on
    this robot is already a reflection of its opposite number, so the two
    brackets are mirror parts in the machine shop sense - and rebuilding the
    right one from scratch made OCC fuse two identically-shaped solids into an
    inverted -389 cm3 result on one side only. Reflecting the finished left
    part cannot do that, and guarantees the pair really is symmetric.
    """
    return _bd.mirror(part, about=_bd.Plane.XZ)


def _build(name):
    if name.endswith("_R"):
        return _mirror(_fuse(PARTS[name[:-2] + "_L"]()))
    return _fuse(PARTS[name]())


BUILDERS = {k: (lambda n=k: _build(n)) for k in PARTS}
LINKS = list(BUILDERS)


# ------------------------------------------------------------------ collision
# MuJoCo takes the CONVEX HULL of a mesh, which would fill the hook's throat and
# destroy the grip, so collision is primitives. Each actuator plus the shroud
# around it is one cylinder; each structural section is one box; the lower torso
# is the one mesh, because it is convex already.

def link_collision(link):
    """[(kind, size, pos, z_dir, x_dir)] in the link frame, metres."""
    out = []
    for cj in CHILD_OF.get(link, []):
        key = MOTOR_OF[cj[0]]
        d, bd_ = IFACE[key], BODY_DIR[cj[0]]
        f = flange_of(cj)
        L = d["length"] * MM + WALL
        c = tuple(o + b * (L / 2 + d["out_z"] * MM) for o, b in zip(f, bd_))
        out.append(("cylinder", (housing_od(key) / 2, L), c, bd_, None))
    if link in PJOINT:
        for a, b, w, h, x_hint, _bore in sections(link):
            ln = math.sqrt(sum((y - x) ** 2 for x, y in zip(a, b)))
            if ln < 1e-6:
                continue
            ctr = tuple((x + y) / 2 for x, y in zip(a, b))
            zdir = tuple((y - x) / ln for x, y in zip(a, b))
            out.append(("box", (w, h, ln), ctr, zdir, x_hint))
    if link.startswith("hook"):
        t, b_, l_, w_ = HOOK["thk"], HOOK["back"], HOOK["lip"], HOOK["width"]
        out.append(("box", (t, w_, b_), (0, 0, -b_ / 2), (0, 0, 1), (1, 0, 0)))
        out.append(("box", (l_, w_, t), (t / 2 + l_ / 2, 0, -b_ + t / 2),
                    (0, 0, 1), (1, 0, 0)))
    if link == "torso_upper":
        d, w, h = TORSO_UP
        out.append(("box", (d, w, h), (0, 0, h / 2), (0, 0, 1), (1, 0, 0)))
        for cj in CHILD_OF["torso_upper"]:
            key = MOTOR_OF[cj[0]]
            od = housing_od(key)
            sg = 1 if cj[4][1] > 0 else -1
            y0 = sg * (w / 2 - 2 * SHELL)
            f = flange_of(cj)
            ln = math.sqrt((f[0] - cj[4][0]) ** 2 + (cj[4][1] - y0) ** 2)
            out.append(("box", (od, od * 0.50, ln),
                        ((f[0] + cj[4][0]) / 2, (cj[4][1] + y0) / 2, cj[4][2]),
                        ((f[0] - cj[4][0]) / ln, (cj[4][1] - y0) / ln, 0.0),
                        (0, 0, 1)))
    if link == "base_link":
        out.append(("mesh", ("base_link_col",), (0, 0, 0), (0, 0, 1), (1, 0, 0)))
    if link == "waist_link":
        out.append(("cylinder", (WAIST_DIA / 2, FIXED[0][3][2]),
                    (0, 0, FIXED[0][3][2] / 2), (0, 0, 1), None))
    return out
