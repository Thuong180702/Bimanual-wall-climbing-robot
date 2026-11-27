"""Layout (packaging) study of the locked real robot: every actuator as its true-size body
at its joint (docs/superpowers/specs/2026-11-25-full-robot-locked-spec.md), simple links
and the torso, standing with the arms 12 deg out. Prints the overall sizes and any
actuator bodies that overlap; writes a STEP to render.

    python cad/tools/real_layout.py out.step
"""
import math, os, sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cadgen import build123d as bd                       # noqa: E402

MM = 1000.0
# actuator: (diameter, length) m -- ENCOS datasheet V3.10 drawings, CubeMars sheets
ACT = {"A13720": (0.170, 0.0845), "A10020": (0.124, 0.085), "A10020s": (0.124, 0.060),
       "A8112": (0.105, 0.053), "AKE90": (0.1075, 0.0545), "AK60": (0.079, 0.067)}

# ---------------------------------------------------------------- the locked kinematics
SOLE, CALF, THIGH = 0.06, 0.34, 0.34
Z_HIP = SOLE + CALF + THIGH                     # hip centre over the floor, legs straight
HIP_Y = 0.085                                   # hip centres +-85 mm
Z_SH = Z_HIP + 0.436                            # shoulder axis (the full robot's torso)
# the waist actuator sits on the pelvis above the two hip-yaw actuators (at a person's
# navel, 305 mm under the shoulders as on v2, it would sit between them)
Z_WAIST = Z_HIP + 0.090 + ACT["A8112"][1] + 0.005 + ACT["AK60"][1]
SH_Y = 0.089                                    # shoulder-flexion output: two A10020 bodies in the chest
ABD_Y = SH_Y + 0.002 + 0.0165 + 0.0015 + ACT["A13720"][0] / 2   # v2's rule: web + air + radius
ROT, ELB, FROLL, WRIST, FING, HOOK = 0.200, 0.070, 0.135, 0.110, 0.100, 0.110
ARM_ABD = math.radians(12)
GIM = ACT["AKE90"][0] / 2 + 0.003 + ACT["AKE90"][1] / 2                      # arms hang 12 deg out (clear of the hips)


def cyl(key, c, axis, name):
    """an actuator body centred at c along axis ('x', 'y', 'z')"""
    d, L = ACT[key]
    s = bd.Cylinder(d / 2 * MM, L * MM)
    if axis == "x":
        s = bd.Rot(0, 90, 0) * s
    elif axis == "y":
        s = bd.Rot(90, 0, 0) * s
    return name, key, bd.Pos(*(v * MM for v in c)) * s


def box(c, size, name):
    return name, None, bd.Pos(*(v * MM for v in c)) * bd.Box(*(v * MM for v in size))


def leg(sg):
    y = sg * HIP_Y
    yo = sg * (HIP_Y + 0.015 + ACT["A13720"][1] / 2)         # pitch / knee motors, outboard
    zk = Z_HIP - THIGH
    s = "L" if sg > 0 else "R"
    return [
        cyl("A8112", (0, y, Z_HIP + 0.090 + 0.0265), "z", "hip_yaw_" + s),
        cyl("A10020", (-0.1425, y, Z_HIP), "x", "hip_roll_" + s),           # belt 1.25:1 to the roll axis
        cyl("A13720", (0, yo, Z_HIP), "y", "hip_pitch_" + s),
        cyl("A13720", (0, yo, Z_HIP - 0.180), "y", "knee_m_" + s),          # high on the thigh, four-bar
        box((0, y, Z_HIP - THIGH / 2), (0.09, 0.03, THIGH), "thigh_" + s),
        box((0, y, zk - CALF / 2), (0.07, 0.06, CALF), "calf_" + s),
        cyl("AKE90", (-0.03, y, zk - 0.090), "y", "ankle_a_" + s),
        cyl("AKE90", (-0.03, y, zk - 0.205), "y", "ankle_b_" + s),
        box((0.04, y, 0.015), (0.22, 0.09, 0.03), "foot_" + s),
    ]


def arm(sg):
    s = "L" if sg > 0 else "R"
    out = [cyl("A10020", (0, sg * (SH_Y - 0.0425), Z_SH), "y", "sh_flex_" + s),
           cyl("A13720", (0.045, sg * ABD_Y, Z_SH), "x", "sh_abd_" + s)]
    # below the abduction axis the arm hangs ARM_ABD out: place along that line
    def at(dz, dy=0.0, dx=0.0):
        a = ARM_ABD
        return (dx, sg * (ABD_Y + dz * math.sin(a) + dy * math.cos(a)), Z_SH - dz * math.cos(a) + dy * math.sin(a) * 0)
    out += [
        cyl("A13720", at(ROT - ACT["A13720"][1] / 2), "z", "sh_rot_" + s),
        cyl("A10020", at(ROT + ELB), "y", "elbow_" + s),                   # belt 1.5:1, drawn coaxial here
        cyl("A10020s", at(ROT + ELB + FROLL - ACT["A10020s"][1] / 2 - 0.01), "z", "froll_" + s),
        # the gimbal: each body stands off the crossing by the other's radius + 3 mm
        cyl("AKE90", at(ROT + ELB + FROLL + WRIST, dy=GIM), "y", "wrist_flex_" + s),
        cyl("AKE90", at(ROT + ELB + FROLL + WRIST, dx=GIM), "x", "wrist_dev_" + s),
        cyl("AK60", at(ROT + ELB + FROLL + WRIST + FING), "y", "finger_" + s),
        box(at(ROT + ELB + FROLL + WRIST + FING + HOOK / 2), (0.04, 0.06, HOOK), "hook_" + s),
    ]
    return out


def torso():
    return [
        box((0, 0, Z_HIP + 0.115), (0.16, 2 * HIP_Y + 0.11, 0.07), "pelvis"),
        cyl("AK60", (0, 0, Z_WAIST - 0.034), "z", "waist"),
        box((0, 0, (Z_WAIST + 0.04 + Z_SH + 0.075) / 2), (0.15, 2 * (SH_Y + 0.006), Z_SH + 0.075 - Z_WAIST - 0.04), "chest"),
        box((0, 0, Z_SH + 0.075 + 0.111 / 2), (0.128, 0.154, 0.111), "riser"),
        box((0, 0, Z_SH + 0.075 + 0.111 + 0.08), (0.16, 0.16, 0.16), "head_lidars"),
    ]


def parts():
    return leg(1) + leg(-1) + arm(1) + arm(-1) + torso()


if __name__ == "__main__":
    P = parts()
    allb = bd.Compound(children=[p[2] for p in P]).bounding_box()
    print("overall: height %.0f mm, width %.0f mm, depth %.0f mm" % (allb.max.Z, allb.size.Y, allb.size.X))
    def width_at(z0, z1, only=None):
        ys = [b.bounding_box() for n, _, b in P if (only is None or n.startswith(only)) and
              not (b.bounding_box().max.Z < z0 * MM or b.bounding_box().min.Z > z1 * MM)]
        return max(v.max.Y for v in ys) - min(v.min.Y for v in ys)
    print("hip width (motors) %.0f mm; shoulder width %.0f mm; waist at %.0f mm, %.0f under the shoulders"
          % (width_at(Z_HIP - 0.09, Z_HIP + 0.09, only=("hip", "thigh", "knee")), width_at(Z_SH - 0.09, Z_SH + 0.09),
             Z_WAIST * MM, (Z_SH - Z_WAIST) * MM))
    acts = [p for p in P if p[1]]
    bad = []
    for (n1, _, s1), (n2, _, s2) in combinations(acts, 2):
        v = (s1 & s2).volume
        if v > 1.0:
            bad.append((n1, n2, v / 1000.0))
    print("actuator overlaps at this pose:", bad or "none")
    if len(sys.argv) > 1:
        bd.export_step(bd.Compound(children=[p[2] for p in P]), sys.argv[1])
