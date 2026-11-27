"""Arms of the real robot (left drawn; the right is its mirror about y = 0).

Chest frame (mm, torso.py): x forward, y left, z up; the shoulder-flexion axis is y at
(x 0, z 436), its A10020 output face at y = SY (89), inside the chest side plate's 66 mm
bore (plate y 89..95).

v2's chain (spec.py _arm): flexion (y) -> abduction (x) -> rotation (z) -> elbow (y) ->
forearm roll (z) -> wrist gimbal -> finger + hook. The abduction axis crosses the flexion
axis at (0, ABD_Y, 436); its A13720 points forward from its output face at x = XO (with the
arm raised to hang it points away from the wall), the upper arm on that face behind it.

1. shoulder bracket (7075): a boss through the chest bore onto the flexion output
   (6 x M5 on 50), a hub cup between the chest and the abduction barrel (its 12 mm floor
   on the output, its 6 mm wall out to y 139: the abduction axis at y 225, 30 mm wider than
   the barrel alone needs, so the arm turns 12 deg inward past the chest and the guard,
   v2's range; the M5 heads sit on the floor inside the cup, fitted before the abduction
   actuator), and the front
   flange that carries the abduction A13720 by its front ring (12 x M4 on 155); the
   upper arm's hub passes through the flange's bore onto the abduction output.
2. upper arm (7075): a hub through the bracket's bore onto the abduction output (3 x M5 on
   40), a back plate 2 mm behind the bracket's flange, two side gussets and a cap on the
   rotation A13720's rear face (20 x M4 on 155); the rotation actuator stands under the
   abduction axis, its output facing down at ROT_Z (200 mm under the abduction axis, v2's
   ARM_OFF rot), its 29 mm hollow centre and the cap's centre hole for the arm's harness.
3. humerus (7075): under the rotation output (3 x M5 on 40), an L (the user's call): a top plate
   and an outboard plate on the elbow actuator's rear face, a gusset behind. The elbow is driven
   DIRECTLY by a MyActuator RMD-X15-P20-450 (user's call 2026-11-26, was an ENCOS A13720: 450 N.m
   peak / 145 rated, 3.5 kg, a 72 V motor run at 48 V: ~8.3 rad/s no-load against the 5.2 the
   climbs need). Belts and crank-rod-levers fail at this torque (spec 3.2). Elbow axis 97 mm
   under the rotation output; its body outboard, the output facing in at the arm's line y = ABD_Y.
4. forearm (7075): a C-cradle round the forearm-roll A10020-P1-12, which mounts only by the
   front ring at its output (10 x M4 on 116.5): the hub on the elbow output (6 of its 12 x M5
   on 100), a back plate over the roll actuator, two side arms either side
   of its body (x +-64..72) and a ring flange under it with the bore the wrist's hub passes
   through. The roll output faces down 156 mm under the elbow axis (v2 130: the elbow's
   radius 85 + 3 air + 8 cap room + the 60 mm body).
5. wrist (gimbal: flexion y and deviation x crossing at the wrist centre W, 110 mm under the
   roll output). Two CubeMars AKE90-8, each held by its 82 mm front ring (8 x M4 on 73.5),
   its output (6 x M5 on 40) reached through a 56 mm bore, as at the shoulder and elbow.
   The deviation actuator's body sits CENTRED on W, in the room under the forearm, its
   output facing forward at x 27 (the user's call, 2026-11-26: v2's forward-pointing one
   stuck out 112 mm); the flexion one stands outboard of it (output y 302):
   - wrist link (7075) on the roll output: a top plate under the forearm's ring and a plate
     down the outboard side on the flexion actuator's ring;
   - gimbal (7075) on the flexion output: an L, its side plate between the deviation body and
     the wrist link, its flange on the deviation ring, the hand's hub through its bore.
6. hand (7075): on the deviation output (6 x M5 on 40) through the gimbal's bore, a plate down
   to the finger AK60-39 (axis y, 97 mm under W: v2's 93.5 + 3.5 to clear the deviation body), held by its 78 mm housing ring
   (8 x M3 on 71). The finger's body points INBOARD (v2: outboard; here, 64 mm off W, the
   flexion actuator is nearer than v2's 92 and an outboard finger met it at +20 deviation).
   The hook (7075) on the finger output: v2's size (back 110, lip 40, 12 thick, 65 wide),
   bolted on turned over, its lip toward -x with the arm down (toward the wall when hanging).
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadgen import build123d as bd
from real.encos import body
from real.hip import C, R, hull, _xz, _yz, loc, AL7075
from real.torso import ZS, SY, PT

HUB_Y0, HUB_F, HUB_Y1 = SY + PT + 2.0, 109.0, 139.0   # hub cup: floor y 97..109 (2 mm off the chest side plate), rim 139
XO, FT = 0.0, 8.0                         # abduction output face (x); front flange x -8..0
ABD_Y = HUB_Y1 + 1.0 + 85.0               # abduction axis: 1 mm air between the hub and the A13720 barrel
HUB_R = 48.0
FL_R = 85.0                               # flange radius = the actuator's: flush with its barrel


def _ycyl(r, y0, y1, x=0.0, z=ZS):
    return bd.Pos(x, (y0 + y1) / 2, z) * bd.Rot(90, 0, 0) * bd.Cylinder(r, y1 - y0)


def _xcyl(r, x0, x1, y=ABD_Y, z=ZS):
    return bd.Pos((x0 + x1) / 2, y, z) * bd.Rot(0, 90, 0) * bd.Cylinder(r, x1 - x0)


def abduction_actuator():
    return loc((XO, ABD_Y, ZS), (-1, 0, 0)) * body("A13720")


def shoulder_bracket():
    boss = _ycyl(30.5, SY, SY + 2.5) + _ycyl(32.0, SY + 2.0, HUB_Y0 + 1.0)   # on the output only (r < 31), 2 mm off its housing
    hub = bd.fillet(_xz(C(0, ZS, HUB_R), HUB_Y0, HUB_Y1 - HUB_Y0).edges(), 2.0)
    hub -= _xz(C(0, ZS, HUB_R - 6), HUB_F, HUB_Y1 - HUB_F + 1)
    # the flange runs across the cup as a diaphragm (from its floor), clear of the M5 heads beside it
    fl = bd.fillet(_yz(hull(C(ABD_Y, ZS, FL_R), R(HUB_F - 3, HUB_Y1, ZS - HUB_R, ZS + HUB_R)), XO, FT).edges(), 2.0)
    part = boss + hub + fl
    # flange: the bore the upper arm's hub passes through, six windows between it and the bolt ring
    part -= _xcyl(38.0, XO - FT - 1, XO + 1)
    for i in range(6):
        a = math.radians(60 * i + 30)
        part -= _yz(hull(C(ABD_Y + 52 * math.cos(a - 0.30), ZS + 52 * math.sin(a - 0.30), 9),
                         C(ABD_Y + 52 * math.cos(a + 0.30), ZS + 52 * math.sin(a + 0.30), 9)), XO + 1, FT + 2)
    # 12 x M4 on 155 into the actuator's front ring, heads sunk in the back face (the upper arm swings 2 mm behind it)
    for i in range(12):
        a = math.radians(30 * i + 15)
        y, z = ABD_Y + 77.5 * math.cos(a), ZS + 77.5 * math.sin(a)
        part -= _xcyl(2.25, XO - FT - 1, XO + 1, y, z) + _xcyl(3.75, XO - FT - 1, XO - FT + 4.2, y, z)
    # (2026-11-26 trim) the cup's wall front and back, away from the flange: two windows 22 x 60 (r 6),
    # 4 mm off the floor and the rim (FEA: 6-12 MPa there)
    win = bd.Pos((HUB_F + HUB_Y1) / 2, ZS) * bd.RectangleRounded(HUB_Y1 - HUB_F - 8, 60, 6)
    part -= _yz(win, 55.0, 35.0) + _yz(win, -20.0, 35.0)
    # 6 x M5 on 50 into the flexion output, heads on the cup's floor
    for i in range(6):
        a = math.radians(60 * i)
        x, z = 25 * math.cos(a), ZS + 25 * math.sin(a)
        part -= _ycyl(2.75, SY - 1, HUB_F + 1, x, z)
    return part


ROT_Z = ZS - 200.0                        # rotation output face (down); the actuator's rear face at ROT_Z + 84.5
CAP0, CAP_T = ROT_Z + 84.5, 10.0          # cap on the rear face: z 320.5 .. 330.5 (the abduction barrel 20 mm over it)
BP0, BP_T = XO - FT - 2.0, 12.0           # back plate x -22 .. -10, 2 mm behind the bracket's flange
G0, G_T = 56.0, 8.0                       # side gussets at y = ABD_Y +- 56..64


def rotation_actuator():
    return loc((0.0, ABD_Y, ROT_Z), (0, 0, -1)) * body("A13720")


def upper_arm():
    hub = _xcyl(34.0, BP0 - 1, XO)                                        # on the output only (pilot 70), inside the bracket's 76 bore
    # round the abduction output r 40 (was 50; FEA 2026-11-26: the rim over the axis saw < 10 MPa): 15 mm
    # of plate outside the 3 x M5 counterbores
    back = _yz(hull(C(ABD_Y, ZS, 40), R(ABD_Y - 80, ABD_Y + 80, CAP0, CAP0 + 30)), BP0, BP_T)
    cap = bd.Pos(0, ABD_Y, CAP0 + CAP_T / 2) * bd.Cylinder(85, CAP_T)
    # gussets: back plate -> cap, kept outside r 88 round the abduction axis where they pass under its barrel
    gus = bd.Polygon((BP0 - BP_T + 3, CAP0 + 3), (40, CAP0 + 3), (40, CAP0 + CAP_T + 4), (BP0, ZS - 86),
                     (BP0 - BP_T + 3, ZS - 36), align=None)
    gs = [_xz(gus, ABD_Y + G0 if sg > 0 else ABD_Y - G0 - G_T, G_T) for sg in (1, -1)]
    back, cap = bd.fillet(back.edges(), 2.0), bd.fillet(cap.edges(), 1.5)
    gs = [bd.fillet(g.edges(), 1.5) for g in gs]
    part = hub + back + cap + gs[0] + gs[1]
    # cap: harness hole over the actuator's hollow centre, four windows between the gussets
    part -= bd.Pos(0, ABD_Y, CAP0 + CAP_T / 2) * bd.Cylinder(16, CAP_T + 2)
    for a in (40, -40, 140, -140):
        c = (52 * math.cos(math.radians(a)), ABD_Y + 52 * math.sin(math.radians(a)))
        part -= bd.Pos(c[0], c[1], CAP0 + CAP_T / 2) * bd.Cylinder(14, CAP_T + 2)
    for i in range(20):                                                   # 20 x M4 on 155 into the rear ring
        a = math.radians(18 * i + 9)
        part -= bd.Pos(77.5 * math.cos(a), ABD_Y + 77.5 * math.sin(a), CAP0 + CAP_T / 2) * bd.Cylinder(2.25, CAP_T + 2)
    # back plate: two windows above the gussets' tops
    part -= _yz(hull(C(ABD_Y - 30, ZS - 72, 10), C(ABD_Y + 30, ZS - 72, 10)), BP0 + 1, BP_T + 2)
    for i in range(3):                                                    # 3 x M5 on 40 into the abduction output, heads behind
        a = math.radians(120 * i + 90)
        y, z = ABD_Y + 20 * math.cos(a), ZS + 20 * math.sin(a)
        part -= _xcyl(2.75, BP0 - BP_T - 1, XO + 1, y, z) + _xcyl(4.75, BP0 - BP_T - 1, BP0 - BP_T + 5.5, y, z)
    return part


ELB_Z = ROT_Z - 97.0                      # elbow axis (y) height at the zero pose
EY = ABD_Y                                # elbow output face (faces -y); body y 225 .. 294, its connector boss to 315
TP0, TP_T = ROT_Z - 11.0, 10.0            # top plate z 225 .. 235, 1 mm under the rotation actuator's face
YP_T = 8.0                                # yoke plates: inboard y 217..225, outboard y 309.5..317.5


X15_L = 69.0                              # RMD-X15-P20-450 barrel length (output face .. rear face)


def x15():
    """MyActuator RMD-X15-P20-450, from its 2D drawing (2026-04-22), output face at z = 0, body z < 0:
    output flange 115 proud 3 (12 x M5 on 100), housing front 144 to 13 (16 x M5 on 154), barrel 166
    to 69 (rear face: 12 x M5 on 157), the connector boss 124 to 90, a 21.5 bore through"""
    b = bd.Pos(0, 0, -1.5) * bd.Cylinder(57.5, 3) + bd.Pos(0, 0, -8) * bd.Cylinder(72, 10)
    b += bd.Pos(0, 0, -(13 + X15_L) / 2) * bd.Cylinder(83, X15_L - 13) + bd.Pos(0, 0, -(X15_L + 90) / 2) * bd.Cylinder(62, 90 - X15_L)
    for i in range(9):                                                    # fins round the barrel
        b -= bd.Pos(0, 0, -18 - 5 * i) * (bd.Cylinder(84, 2.5) - bd.Cylinder(80, 2.5))
    for n, pcd, z in ((12, 100, 0), (16, 154, -3)):
        for i in range(n):
            a = 2 * math.pi * (i + 0.5) / n
            b -= bd.Pos(pcd / 2 * math.cos(a), pcd / 2 * math.sin(a), z - 4.5) * bd.Cylinder(2.5, 9)
    for i in range(12):
        a = 2 * math.pi * (i + 0.5) / 12
        b -= bd.Pos(78.5 * math.cos(a), 78.5 * math.sin(a), -X15_L + 4.5) * bd.Cylinder(2.5, 9)
    return b - bd.Pos(0, 0, -45) * bd.Cylinder(10.75, 92)


def elbow_actuator():
    return loc((0.0, EY, ELB_Z), (0, -1, 0)) * x15()


def humerus():
    yo = EY + X15_L
    hub = bd.Pos(0, ABD_Y, ROT_Z - 1.0) * bd.Cylinder(34, 2.0)                     # on the rotation output only (r < 35)
    # an L: no inboard plate (the user's call, 2026-11-26) -- the elbow X15 hangs on its rear face from a
    # 12 mm outboard plate (its connector boss through the plate's bore); a gusset behind ties the corner
    ot = 12.0
    # the front edge reaches x 75 outboard of the forearm (y > 262; inboard it stays at 55, the forearm's
    # side arms pass there at full flexion): FEA put the 450 N.m's peak on that front corner
    top = bd.Pos(0, 0, TP0) * bd.extrude(hull(C(0, ABD_Y, 46), R(-94, 55, EY - YP_T, yo + ot)) + R(50, 75, 262, yo + ot), amount=TP_T)
    side = hull(C(0, ELB_Z, 88), R(-94, 75, TP0 + 3, TP0 + TP_T))                # r 88: 6.75 mm outside the M5 holes
    outb = _xz(side, yo, ot)
    gus = _yz(bd.Polygon((EY + 5, TP0 + 3), (yo + 3, ELB_Z + 10), (yo + 3, TP0 + 3), align=None), -86, 8)   # CCW: extrudes x -94..-86
    # a lip outside the plate along its top (FEA: the 450 N.m bent the top/outboard corner, 293 MPa): the corner
    # becomes a box; inside it the X15's barrel leaves no room, outside nothing passes under z 236
    lip = _xz(R(-94, 75, TP0 - 14, TP0 + TP_T), yo + ot - 1, 13)
    top, outb, gus, lip = (bd.fillet(p.edges(), 2.0) for p in (top, outb, gus, lip))
    part = hub + top + outb + gus + lip
    # outboard plate: the bore for the X15's connector boss (124) and the harness, its rear face 12 x M5 on 157
    part -= _ycyl(63.5, yo - 1, yo + ot + 1, 0, ELB_Z)
    for i in range(12):
        a = 2 * math.pi * (i + 0.5) / 12
        x, z = 78.5 * math.cos(a), ELB_Z + 78.5 * math.sin(a)
        part -= _ycyl(2.75, yo - 1, yo + ot + 1, x, z)
    for i in range(3):                                                     # 3 x M5 on 40 into the rotation output
        a = math.radians(120 * i + 90)
        x, y = 20 * math.cos(a), ABD_Y + 20 * math.sin(a)
        part -= bd.Pos(x, y, TP0 + 5) * bd.Cylinder(2.75, 14) + bd.Pos(x, y, TP0 + 2) * bd.Cylinder(4.75, 5.5)
    # top plate: windows either side of the hub
    for sx in (-1, 1):
        part -= bd.Pos(sx * 36, EY + 50, TP0 + TP_T / 2) * bd.extrude(bd.RectangleRounded(22, 60, 8), amount=TP_T + 2, both=True)
    return part


FR_Z = ELB_Z - 156.0                      # forearm roll output face (down); body z FR_Z .. FR_Z + 60
FL_T = 8.0                                # cradle plates
BK1 = EY - YP_T - 2.0                     # back plate y 203 .. 215, 2 mm inboard of the humerus' inboard plate
ARM_X, ARM_Y = (64.0, 69.0), (BK1 - 12.0, ABD_Y + 30.0)   # side arms: x +-64..69, y 203 .. 255 (at 72 the front one met the humerus at full flexion)


def roll_actuator():
    return loc((0.0, ABD_Y, FR_Z), (0, 0, -1)) * body("A10020s")


def forearm():
    top = FR_Z + 60.0 + 2.0                                                # back plate bottom: 2 mm over the body
    hub = _ycyl(57.0, BK1 - 1, EY, 0, ELB_Z) - _ycyl(42.0, BK1, EY + 1, 0, ELB_Z)   # a ring on the X15's 115 output flange, round its bolts
    back = _xz(hull(C(0, ELB_Z, 57), R(-ARM_X[1], ARM_X[1], top, top + 12)), BK1 - 12, 12)   # r 57 (60 grazed the humerus at -107)
    arm_sk = R(ARM_Y[0], ARM_Y[1], FR_Z - FL_T + 3, top + 12)
    arms = [_yz(arm_sk, ARM_X[1], ARM_X[1] - ARM_X[0]), _yz(arm_sk, -ARM_X[0], ARM_X[1] - ARM_X[0])]
    ring = bd.Pos(0, 0, FR_Z - FL_T) * bd.extrude(hull(C(0, ABD_Y, 66), R(-ARM_X[1], ARM_X[1], ARM_Y[0], ARM_Y[1])), amount=FL_T)
    # a top plate over the body ties the arms along their length (FEA: on the back plate alone 314 MPa);
    # its top 88 mm under the elbow axis, clear of the elbow and the humerus at every flexion
    deck = bd.Pos(0, 0, top) * bd.extrude(R(-ARM_X[1], ARM_X[1], ARM_Y[0], ARM_Y[1]), amount=ELB_Z - 88.0 - top)
    back, ring, deck = (bd.fillet(p.edges(), 2.0) for p in (back, ring, deck))
    arms = [bd.fillet(a.edges(), 2.0) for a in arms]
    part = hub + back + arms[0] + arms[1] + ring + deck
    part -= bd.Pos(0, ABD_Y, FR_Z - FL_T / 2) * bd.Cylinder(36, FL_T + 2)  # the wrist hub's bore
    for i in range(10):                                                    # 10 x M4 on 116.5, heads underneath
        a = math.radians(36 * i + 18)
        x, y = 58.25 * math.cos(a), ABD_Y + 58.25 * math.sin(a)
        part -= bd.Pos(x, y, FR_Z - FL_T / 2) * bd.Cylinder(2.25, FL_T + 2) + bd.Pos(x, y, FR_Z - FL_T + 2) * bd.Cylinder(3.75, 4.2)
    for i in range(6):                                                     # 6 of the X15 output's 12 x M5 on 100, heads inboard
        a = 2 * math.pi * (2 * i + 0.5) / 12
        x, z = 50 * math.cos(a), ELB_Z + 50 * math.sin(a)
        part -= _ycyl(2.75, BK1 - 13, EY + 1, x, z) + _ycyl(4.75, BK1 - 13, BK1 - 7.5, x, z)
    part -= _ycyl(10.75, BK1 - 13, EY + 1, 0, ELB_Z)                       # the harness through the X15's bore
    for sx in (ARM_X[1], -ARM_X[0]):                                       # side arm windows
        part -= _yz(hull(C(ARM_Y[0] + 18, FR_Z + 8, 8), C(ARM_Y[1] - 14, FR_Z + 8, 8), C(ARM_Y[0] + 18, top - 4, 8),
                         C(ARM_Y[1] - 14, top - 4, 8)), sx + 1, ARM_X[1] - ARM_X[0] + 2)
    part -= _xz(hull(C(-30, ELB_Z - 74, 9), C(30, ELB_Z - 74, 9)), BK1 - 13, 13)   # back plate window, under the hub ring
    return part.fix()                                                      # (the deck's union leaves a bad face otherwise)


def about_elbow(q):
    """the elbow at q degrees (- = flexed, forearm forward and up)"""
    return bd.Pos(0, 0, ELB_Z) * bd.Rot(0, q, 0) * bd.Pos(0, 0, -ELB_Z)


WZ = FR_Z - 110.0                         # wrist centre W = (0, ABD_Y, WZ)
WT = 8.0                                  # wrist plates
DV_O = 27.25                              # deviation AKE90 output face (faces +x): its body CENTRED on W, x -27.25 .. 27.25,
DV_R = 53.75                              # in the room under the forearm (the user's call, 2026-11-26: nothing sticks out forward)
GX0, GX1 = DV_O - 1.0, DV_O - 1.0 + WT    # gimbal flange on the deviation ring (ring face 1 mm behind the output): x 26.25 .. 34.25
GY0, GY_T = ABD_Y + DV_R + 2.0, 12.0      # gimbal side plate y 280.75 .. 292.75, outboard of the deviation body
FX_Y = GY0 + GY_T + 9.25                  # flexion AKE90 output face (faces -y) at y 302: body y 302 .. 356.5


def wrist_actuators():
    from real.calf import ake90_env
    e = ake90_env((0, 0, 0), (0, 0, 1))
    return {"wrist flexion": loc((0.0, FX_Y, WZ), (0, -1, 0)) * e, "wrist deviation": loc((DV_O, ABD_Y, WZ), (1, 0, 0)) * e}


def _ring8(axis, c, y0, y1):
    """8 x M4 on 73.5 through a plate normal to axis ('x' or 'y') at centre c, thickness y0..y1"""
    cut = None
    for i in range(8):
        a = math.radians(45 * i + 22.5)
        u, v = 36.75 * math.cos(a), 36.75 * math.sin(a)
        h = _ycyl(2.25, y0, y1, c[0] + u, c[2] + v) if axis == "y" else _xcyl(2.25, y0, y1, c[1] + u, c[2] + v)
        cut = h if cut is None else cut + h
    return cut


def wrist_link():
    top0 = FR_Z - FL_T - 2.0 - WT                                          # z -35 .. -27, 2 mm under the forearm's ring
    yr = FX_Y + 1.0                                                        # the flexion actuator's ring face
    hub = bd.Pos(0, ABD_Y, FR_Z - 5) * bd.Cylinder(30, 10)                 # on the roll output (pilot 62) through the ring's bore
    top = bd.Pos(0, 0, top0) * bd.extrude(hull(C(0, ABD_Y, 40), R(-45, 45, yr - 11, yr)), amount=WT)
    down = _xz(hull(C(0, WZ, 47), R(-45, 45, top0, top0 + WT)), yr - WT, WT)
    top, down = bd.fillet(top.edges(), 2.0), bd.fillet(down.edges(), 2.0)
    part = hub + top + down
    part -= _ycyl(28.0, yr - WT - 1, yr + 1, 0, WZ) + _ring8("y", (0, 0, WZ), yr - WT - 1, yr + 1)
    for i in range(6):                                                     # 6 x M5 on 50 into the roll output
        a = math.radians(60 * i + 30)
        x, y = 25 * math.cos(a), ABD_Y + 25 * math.sin(a)
        part -= bd.Pos(x, y, top0 + 5) * bd.Cylinder(2.75, 30) + bd.Pos(x, y, top0 + 1.5) * bd.Cylinder(4.75, 4)
    return part


def gimbal():
    """an L: the side plate on the flexion output (outboard of the deviation body), the flange on the deviation ring"""
    hub = _ycyl(25.0, GY0 + GY_T - 1, FX_Y, 0, WZ)                          # through the wrist link's bore onto the flexion output
    side = _xz(hull(C(0, WZ, 36), R(GX0, GX1, WZ - 45, WZ + 45)), GY0, GY_T)
    flange = _yz(hull(C(ABD_Y, WZ, 45), R(ABD_Y + 20, GY0 + 3, WZ - 45, WZ + 45)), GX1, WT)
    side, flange = bd.fillet(side.edges(), 2.0), bd.fillet(flange.edges(), 2.0)
    part = hub + side + flange
    part -= _xcyl(28.0, GX0 - 1, GX1 + 1, ABD_Y, WZ) + _ring8("x", (0, ABD_Y, WZ), GX0 - 1, GX1 + 1)
    for i in range(6):                                                     # 6 x M5 on 40 into the flexion output, heads sunk
        a = math.radians(60 * i)                                           # in the side plate (the deviation body is 2 mm off)
        x, z = 20 * math.cos(a), WZ + 20 * math.sin(a)
        part -= _ycyl(2.75, GY0 - 1, FX_Y + 1, x, z) + _ycyl(4.75, GY0 - 1, GY0 + 5.5, x, z)
    return part


FZ = WZ - 97.0                            # finger axis (y): v2's 93.5 + 3.5 so the finger body clears the deviation body
FO_Y = ABD_Y - 6.0                        # finger output face (faces +y); housing ring face at FO_Y - 6, body to FO_Y - 67
HOOK_B, HOOK_LIP, HOOK_T, HOOK_W = 110.0, 40.0, 12.0, 65.0


def finger_actuator():
    """AK60-39 envelope: output disk 54 proud 6 of the housing ring, body 79"""
    return _ycyl(27.0, FO_Y - 6, FO_Y, 0, FZ) + _ycyl(39.5, FO_Y - 67, FO_Y - 6, 0, FZ)


def hand():
    hub = _xcyl(25.0, DV_O, DV_O + 10, ABD_Y, WZ)                        # on the deviation output through the gimbal's bore
    x1 = DV_O + 18.0                                                      # plate x 36.25 .. 45.25, 2 mm off the gimbal
    # a sleeve round the finger's body (r 40.5..47, back to y 175) carries it with the ring plate on its housing
    # ring; the side plate comes down over the sleeve (FEA: a 6 mm ring plate alone on the side plate, 575 MPa)
    plate = _yz(hull(C(ABD_Y, WZ, 30), R(ABD_Y - 25, ABD_Y + 20, FZ + 20, FZ + 52)), x1, 9)
    sleeve = _ycyl(47.0, FO_Y - 44, FO_Y - 3, 0, FZ) - bd.Pos(0, FO_Y - 24, FZ + 60) * bd.Box(58, 50, 60)   # open on top under the deviation body
    ringp = _xz(hull(C(0, FZ, 40.5), R(29, x1, FZ + 20, FZ + 36)) + R(35, x1, FZ + 30, FZ + 52), FO_Y - 6, 6)   # under r 56 of W where it meets the deviation body
    plate, ringp = bd.fillet(plate.edges(), 2.0), bd.fillet(ringp.edges(), 1.5)
    part = hub + plate + ringp + sleeve
    part -= _ycyl(40.5, FO_Y - 70, FO_Y - 6, 0, FZ)                       # the finger body's room
    part -= _ycyl(28.0, FO_Y - 7, FO_Y + 1, 0, FZ)
    for i in range(8):                                                    # 8 x M3 on 71
        a = math.radians(45 * i + 22.5)
        part -= _ycyl(1.7, FO_Y - 7, FO_Y + 1, 35.5 * math.cos(a), FZ + 35.5 * math.sin(a))
    for i in range(6):                                                    # 6 x M5 on 40 into the deviation output, heads outside
        a = math.radians(60 * i)
        y, z = ABD_Y + 20 * math.cos(a), WZ + 20 * math.sin(a)
        part -= _xcyl(2.75, DV_O - 1, x1 + 1, y, z) + _xcyl(4.75, x1 - 5.5, x1 + 1, y, z)
    return part


def hook():
    """the hook on the finger output: web, palm (back 110 under the finger axis) and lip toward -x"""
    y0 = FO_Y + 2.0                                                       # web y 221 .. 233, 2 mm off the hand's ring plate
    boss = _ycyl(26.0, FO_Y, y0 + 1, 0, FZ)
    web = _xz(hull(C(0, FZ, 30), R(-HOOK_T / 2, HOOK_T / 2 - 4, FZ - 60, FZ - 46)), y0, 12)   # neck flush with the 8 mm palm   # narrow neck: it swings past the hand at -90
    z_top = FZ - 49.0                                                     # palm starts under the hand's sleeve (r 47)
    # palm and lip 8 mm (were 12; FEA 45 MPa, 2026-11-26 trim), thinned from outside: the throat (palm's -x
    # face, the lip's top) stays where it was, so the rung sees the same hook
    palm = _yz(R(ABD_Y - HOOK_W / 2, ABD_Y + HOOK_W / 2, FZ - HOOK_B + 4, z_top), HOOK_T / 2 - 4, HOOK_T - 4)
    lip = bd.Pos(-HOOK_T / 2 - HOOK_LIP / 2 + 3, ABD_Y, FZ - HOOK_B + HOOK_T / 2 + 2) * bd.Box(HOOK_LIP + 6, HOOK_W, HOOK_T - 4)
    web, palm, lip = (bd.fillet(p.edges(), 2.0) for p in (web, palm, lip))
    part = boss + web + palm + lip
    for i in range(6):                                                    # 6 x M4 on 30 into the finger output
        a = math.radians(60 * i)
        part -= _ycyl(2.25, FO_Y - 1, y0 + 13, 15 * math.cos(a), FZ + 15 * math.sin(a))
    return part


def about_deviation(a):
    return bd.Pos(0, ABD_Y, WZ) * bd.Rot(a, 0, 0) * bd.Pos(0, -ABD_Y, -WZ)


def about_finger(a):
    return bd.Pos(0, 0, FZ) * bd.Rot(0, a, 0) * bd.Pos(0, 0, -FZ)


def about_wrist_flexion(a):
    return bd.Pos(0, 0, WZ) * bd.Rot(0, a, 0) * bd.Pos(0, 0, -WZ)


def about_roll(a):
    return bd.Pos(0, ABD_Y, 0) * bd.Rot(0, 0, a) * bd.Pos(0, -ABD_Y, 0)


def about_rotation(a):
    """the shoulder rotation joint at a degrees (about z through the arm's line)"""
    return bd.Pos(0, ABD_Y, 0) * bd.Rot(0, 0, a) * bd.Pos(0, -ABD_Y, 0)


def about_abduction(a):
    """the abduction joint at a degrees (+ = left arm out / up)"""
    return bd.Pos(0, ABD_Y, ZS) * bd.Rot(a, 0, 0) * bd.Pos(0, -ABD_Y, -ZS)


def about_flexion(a):
    """the flexion joint at a degrees (- = arm forward and up)"""
    return bd.Pos(0, 0, ZS) * bd.Rot(0, a, 0) * bd.Pos(0, 0, -ZS)


def arm_assembly(flex=0.0, abd=0.0, rot=0.0, elb=0.0, roll=0.0, wfl=0.0, wdev=0.0, finger=0.0):
    """the left arm's parts (structure + actuators, coloured) at the given joint angles (deg), chest frame"""
    AL, ACT = bd.Color(0.78, 0.80, 0.83), bd.Color(0.36, 0.37, 0.40)
    T1 = about_flexion(flex)
    T2 = T1 * about_abduction(abd)
    T3 = T2 * about_rotation(rot)
    T4 = T3 * about_elbow(elb)
    T5 = T4 * about_roll(roll)
    T6 = T5 * about_wrist_flexion(wfl)
    T7 = T6 * about_deviation(wdev)
    T8 = T7 * about_finger(finger)
    W = wrist_actuators()
    out = []
    for T, p, c in ((T1, shoulder_bracket(), AL), (T1, abduction_actuator(), ACT), (T2, upper_arm(), AL),
                    (T2, rotation_actuator(), ACT), (T3, humerus(), AL), (T3, elbow_actuator(), ACT), (T4, forearm(), AL),
                    (T4, roll_actuator(), ACT), (T5, wrist_link(), AL), (T5, W["wrist flexion"], ACT), (T6, gimbal(), AL),
                    (T6, W["wrist deviation"], ACT), (T7, hand(), AL), (T7, finger_actuator(), ACT), (T8, hook(), AL)):
        q = T * p
        q.color = c
        out.append(q)
    return out


STRUCTURE = ("shoulder_bracket", "upper_arm", "humerus", "forearm", "wrist_link", "gimbal", "hand", "hook")


def _bb_hit(a, b):
    p, q = a.bounding_box(), b.bounding_box()
    return not (p.max.X < q.min.X or q.max.X < p.min.X or p.max.Y < q.min.Y or q.max.Y < p.min.Y or
                p.max.Z < q.min.Z or q.max.Z < p.min.Z)


def hits(moving, fixed, tol=0.5):
    out = []
    for mn, m in moving.items():
        for fn, f in fixed.items():
            if _bb_hit(m, f):
                v = getattr(m & f, "volume", 0.0)
                if v > tol:
                    out.append((mn, fn, round(v)))
    return out


if __name__ == "__main__":
    from real import torso as TO
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "STEP", "real")
    br, ab, ua, ro = shoulder_bracket(), abduction_actuator(), upper_arm(), rotation_actuator()
    fx = TO.shoulder_actuators()["L"]
    for n, p in (("shoulder bracket", br), ("upper arm", ua)):
        print("%s: valid %s, solids %d, %.0f g" % (n, p.is_valid, len(p.solids()), p.volume * AL7075))
    print("  static:", hits({"bracket": br, "upper arm": ua}, {"abduction act": ab, "flexion act": fx, "rotation act": ro})
          + hits({"upper arm": ua}, {"bracket": br}) or "clear")
    ch = TO.chest()
    body_parts = {"chest": ch, "head": TO.head_mount(), **TO.cage(), **TO.lidars(),
                  **{n: bd.Pos(*c) * bd.Box(*s) for n, (c, s, _) in TO.bay().items()}}
    # flexion -170..60 with the bracket and the abduction actuator, against the chest, its bay, lidars, head and guard
    h = []
    for a in range(-170, 61, 10):
        h += [(a,) + t for t in hits({"bracket": about_flexion(a) * br, "abduction act": about_flexion(a) * ab}, body_parts)]
    print("flexion sweep (shoulder):", h or "clear")
    # abduction -12..180 at flexion 0: the upper arm + rotation actuator against the shoulder
    h = []
    for a in list(range(-12, 181, 6)) + [180]:
        mv = {"upper arm": about_abduction(a) * ua, "rotation act": about_abduction(a) * ro}
        h += [(a,) + t for t in hits(mv, {"bracket": br, "abduction act": ab})]
    print("abduction sweep vs shoulder:", h or "clear")
    # the most inward abduction clear of the body, per flexion angle
    lim = {}
    for f in range(-170, 61, 20):
        for a in range(10, -31, -2):
            mv = {k: about_flexion(f) * about_abduction(a) * v for k, v in (("upper arm", ua), ("rotation act", ro))}
            if hits(mv, body_parts):
                lim[f] = a + 2
                break
        else:
            lim[f] = "<= -30"
    print("inward abduction limit by flexion (deg):", lim)
    hu, el = humerus(), elbow_actuator()
    print("humerus: valid %s, solids %d, %.0f g" % (hu.is_valid, len(hu.solids()), hu.volume * AL7075))
    h = []
    for a in range(-70, 91, 10):
        h += [(a,) + t for t in hits({"humerus": about_rotation(a) * hu, "elbow act": about_rotation(a) * el},
                                     {"upper arm": ua, "rotation act": ro, "bracket": br, "abduction act": ab})]
    print("rotation sweep vs upper arm / shoulder:", h or "clear")
    for k, v in (("shoulder_bracket_L", br), ("upper_arm_L", ua), ("humerus_L", hu)):
        v.label, v.color = k, bd.Color(0.78, 0.80, 0.83)
        bd.export_step(v, os.path.join(out, k + ".step"))
    for a in (ab, ro, el):
        a.color = bd.Color(0.36, 0.37, 0.40)
    kids = [br, ab, ua, ro, hu, el]
    kids += [bd.mirror(k, bd.Plane.XZ) for k in kids]
    for k, c in zip(kids[6:], kids[:6]):
        k.color = c.color
    bd.export_step(bd.Compound(children=kids + TO.assembly()), os.path.join(out, "torso_shoulders.step"))
