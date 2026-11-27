"""Pick the lightest off-the-shelf 48 V actuator for every joint of the full robot, for
fast running + jumping legs and arms that hang the whole robot, and re-solve the
robot's mass until the choice stops changing (bigger actuators -> heavier robot ->
bigger needs). A sizing pass to lock the actuator table before CAD; the task-level
check (motions + inverse dynamics on urdf/full) confirms it afterwards.

    python cad/tools/size_actuators.py

Needs are N.m per kg of robot. Arms: the hang box/envelope needs measured on urdf/full
(check_full.arms_static, 63.85 kg model, 2026-11-25) divided by that mass, x1.3.
Legs: Schache et al. 2011 (Med Sci Sports Exerc, Table 1) at 5.02 m/s -- hip extension
2.95, knee extension 3.52, ankle plantarflexion 3.55, hip abduction 2.42 N.m/kg; hip roll
taken from this robot's own stance model instead (wider hips: 224 N.m at 2.5 x weight on
63.85 kg = 3.5), hip yaw 0.56 -> 0.7; x1.25 for jumping and landing. Continuous load:
legs ~0.4 of the peak need over a running stride, arms the two-hand hang (measured).
"""
# name: (peak N.m, rated N.m, max output speed rad/s at 48 V, mass kg, source)
CAT = {
    "AK60-39":       (72, 24, 10.3, 0.75, "CubeMars"),
    "AK80-64":       (120, 48, 7.85, 0.85, "CubeMars"),
    "RS04":          (120, 40, 20.9, 1.42, "RobStride"),
    "AKE90-8":       (170, 55, 22.0, 1.45, "CubeMars"),
    "X12-320":       (320, 85, 13.1, 2.37, "MyActuator"),
    "EC-A8112-18":   (90, 30, 16.4, 0.86, "ENCOS datasheet V3.10"),
    "EC-A10020-12":  (150, 50, 14.7, 1.362, "ENCOS datasheet V3.10"),
    "EC-A10020-24":  (330, 100, 12.9, 2.18, "ENCOS datasheet V3.10"),
    "EC-A13715-12.67": (320, 110, 14.1, 2.657, "ENCOS datasheet V3.10"),
    "EC-A13720-11.4":  (400, 120, 14.6, 3.109, "ENCOS datasheet V3.10"),
    # a 72 V motor on the 48 V bus: torque from current (69.2 A peak), no-load speed 48 / 29.9 V/krpm / 20.25
    "X15-450@48V":   (450, 145, 8.3, 3.50, "MyActuator RMD-X15-P20-450 2D drawing"),
}
STAGE = (1.0, 1.25, 1.5)          # belt/linkage ratios a joint may add
STAGE_KG = 0.6                    # what a belt stage weighs (pulleys, belt, bearings)
BELT_COST = 0.5                   # a belt is a part to build and tension: chosen only when it
                                  # saves more than this (kg-equivalent) over a direct drive
# joints driven through the leg's own linkage (KAIST's knee four-bar, the ankle rods): any
# ratio up to the limit, from the link lengths, no extra parts
LINKAGE = {"knee": 1.2, "ankle": 1.6}

# joint: (count, peak need N.m/kg, margin, continuous need N.m/kg, speed need rad/s, actuators per joint)
JOINTS = {
    # arms (each x2): hang box needs / 63.85 kg, speeds 1.5 x the reference climbs' p99.9
    "shoulder_flexion":   (2, 183.8 / 63.85, 1.3, 27.7 / 63.85, 1.5 * 4.72, 1),
    "shoulder_abduction": (2, 221.7 / 63.85, 1.3, 1.8 / 63.85, 1.5 * 1.66, 1),
    "shoulder_rotation":  (2, 199.3 / 63.85, 1.3, 0.7 / 63.85, 1.5 * 1.58, 1),
    "elbow":              (2, 233.1 / 63.85, 1.3, 18.4 / 63.85, 1.5 * 3.47, 1),
    "forearm_roll":       (2, 64.7 / 63.85, 1.3, 0.4 / 63.85, 1.5 * 2.12, 1),
    "wrist_flexion":      (2, 92.2 / 63.85, 1.3, 10.4 / 63.85, 1.5 * 2.13, 1),
    "wrist_deviation":    (2, 92.8 / 63.85, 1.3, 1.0 / 63.85, 1.5 * 2.13, 1),
    "finger":             (2, 35.5 / 63.85, 1.3, 7.9 / 63.85, 1.5 * 2.20, 1),
    # legs (each x2): running at 5 m/s + jumping
    "hip_pitch":          (2, 2.95, 1.25, 0.4 * 2.95, 12.0, 1),
    "hip_roll":           (2, 3.50, 1.25, 0.4 * 3.50, 6.0, 1),
    "hip_yaw":            (2, 0.70, 1.25, 0.4 * 0.70, 6.0, 1),
    "knee":               (2, 3.52, 1.25, 0.4 * 3.52, 12.0, 1),
    # the ankle: two actuators through rods, pitch from both (85 % of the ideal over range)
    "ankle":              (2, 3.55 / 0.85, 1.25, 0.4 * 3.55 / 0.85, 15.0, 2),
}
WAIST = 0.75                       # AK60-39, kept (waist check passes at 8.5x)
S = 0.25                           # brackets/links per kg of limb actuator
# the rest (torso, battery, lidars, cage, electronics, feet...): 63.85 kg minus the
# actuators of that model (39.39 kg) minus its brackets (0.25 x 38.64)
FIXED = 63.85 - 39.39 - S * 38.64


def pick(j, M, need, margin, cont, speed, n):
    """lightest (actuator, ratio) whose n units meet peak, continuous and speed at mass M"""
    best = None
    for k, (pk, rt, w, kg, _) in CAT.items():
        if j in LINKAGE:              # the smallest ratio that meets the torque, if fast enough
            g = max(1.0, margin * need * M / (n * pk), cont * M / (n * rt))
            gs = [g] if g <= LINKAGE[j] else []
        else:
            gs = STAGE
        for g in gs:
            if (n * pk * g >= margin * need * M - 1e-9 and n * rt * g >= cont * M - 1e-9
                    and w / g >= speed):
                belt = g > 1 and j not in LINKAGE
                mass = n * kg + (STAGE_KG if belt else 0.0)
                cost = mass + (BELT_COST if belt else 0.0)
                if best is None or cost < best[3]:
                    best = (k, g, mass, cost)
                break                 # a higher ratio of the same actuator only costs speed
    return best


def solve():
    M = 63.85
    for it in range(50):
        sel = {j: pick(j, M, need, mg, c, w, n) for j, (cnt, need, mg, c, w, n) in JOINTS.items()}
        if any(v is None for v in sel.values()):
            return M, sel, it
        act = sum(JOINTS[j][0] * v[2] for j, v in sel.items())
        M_new = FIXED + WAIST + act * (1 + S)
        if abs(M_new - M) < 1e-3:
            return M_new, sel, it
        M = M_new
    return M, sel, it


if __name__ == "__main__":
    M, sel, it = solve()
    print("converged in %d passes: %.1f kg (fixed %.1f kg)" % (it, M, FIXED))
    for j, (cnt, need, mg, c, w, n) in JOINTS.items():
        v = sel[j]
        if v is None:
            print("  %-19s NOTHING in the catalogue meets it (need %.0f N.m)" % (j, mg * need * M))
            continue
        k, g, mass, _ = v
        pk, rt, sp, kg, src = CAT[k]
        print("  %-19s %s%s%-16s %s  peak %4.0f/%4.0f (%.2fx)  cont %4.0f/%4.0f  speed %4.1f/%4.1f rad/s  %.2f kg"
              % (j, "%d x " % n if n > 1 else "", "", k, ("link %.2f:1" if j in LINKAGE else "belt %.2f:1") % g if g > 1 else "direct     ",
                 n * pk * g, need * M, n * pk * g / (need * M), n * rt * g, c * M, sp / g, w, mass))
    # the check: the loop's own fixed point and the margins it claims
    for j, (cnt, need, mg, c, w, n) in JOINTS.items():
        k, g, mass, _ = sel[j]
        assert n * CAT[k][0] * g >= mg * need * M - 1e-6 and CAT[k][2] / g >= w
