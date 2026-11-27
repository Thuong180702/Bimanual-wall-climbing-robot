"""Can the running robot's leg actuators do the tasks? A parametric dynamics model (no
CAD: the locked kinematics, each actuator's catalogue mass where it sits, structure as an
allowance) and, per task, a prescribed whole-body motion -> floating-base inverse dynamics
-> each actuator's (speed, torque) over the motion against its 48 V envelope, its rated
torque (RMS over a continuous task) and the joint ranges.

    python cad/tools/run_check.py            # every task, the table
    python cad/tools/run_check.py run jump   # some

The design is the actuator set chosen by cad/tools/size_actuators.py (Encos-based, the
user's choice 2026-11-25) on a KAIST-style leg: hip yaw -> roll -> pitch with the three
axes crossing at the hip centre (H1, MIT Humanoid, Cassie), the knee actuator high on the
thigh driving the knee through a four-bar (KAIST), both ankle actuators high on the calf
driving the foot through two rods (KAIST, G1). Motions are templates, not optimised gaits:
running is a spring-mass stance (ground force from the foot to the COM, half-sine
vertical) with a level pelvis and the foot rolling heel -> flat -> toe; jumps, stairs,
the box and sit-to-stand are minimum-jerk pelvis paths with the feet placed. They bound
what a controller will ask, they do not replace it.
"""
import math, os, sys
import numpy as np
import mujoco

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
G = 9.81

# ---------------------------------------------------------------- actuators (48 V)
# peak, rated N.m, mass kg, envelope [(rad/s, N.m)] -- what the actuator can give at a
# speed. ENCOS: datasheet V3.10 ("缓增负载" tables: the speed it held as the load rose at
# 48 V, and the stall peak); the knee of the A13720's line is extrapolated to its 400 N.m.
# AKE90-8: no published curve -- ASSUMED flat to 60 % of its 22 rad/s no-load, then a line
# to zero (to be replaced by CubeMars' curve).
ACT = {
    "EC-A13720-11.4": (400, 120, 3.109, [(0, 400), (4.6, 400), (9.4, 227), (10.0, 207), (11.1, 172),
                                         (12.0, 133), (13.0, 80), (13.6, 8), (13.65, 0)]),
    "EC-A10020-24":   (330, 100, 2.180, [(0, 330), (7.5, 330), (12.9, 112), (13.1, 0)]),
    "EC-A8112-18":    (90, 30, 0.860, [(0, 90), (13.0, 90), (16.9, 0)]),
    "AKE90-8":        (170, 55, 1.450, [(0, 170), (13.2, 170), (22.0, 0)]),
}


# the bus voltage the envelopes are read at: a motor's speed at a torque scales with the
# voltage (back-EMF), its torque does not (current-limited). The tables are at 48 V; a 13S
# pack runs 54.6 V full down to ~42 V empty, so 42 is the honest design point at "48 V".
VBUS = float(os.environ.get("VBUS", 42.0))


def avail(key, w):
    e = ACT[key][3]
    s = VBUS / 48.0
    return float(np.interp(abs(w), [p[0] * s for p in e], [p[1] for p in e], right=0.0))


# ---------------------------------------------------------------- the design (left leg)
D = dict(
    mass=87.9,              # size_actuators.py, Encos-based set, converged
    hip_y=0.085,            # hip centres +-85 mm (a 1.40-1.60 m person's 80-90)
    thigh=0.34, calf=0.34,  # hip centre -> knee -> ankle axis
    sole=0.06,              # ankle axis over the sole
    toe=0.15, heel=0.07,    # sole ahead of / behind the ankle axis
    knee_ratio=1.03,        # four-bar, joint torque / motor torque (sizing)
    ankle_ratio=1.35,       # rod lever / crank, pitch (sizing)
    ankle_rod_y=0.046,      # each rod's foot end off the roll axis
    ankle_eff=0.85,         # the rods' worst fraction of the ideal over the range
    bracket=0.25,           # structure per kg of actuator
)
# joint: (actuator, range deg (left leg))
JOINT = {
    "hip_yaw":     ("EC-A8112-18",    (-45, 45)),
    "hip_roll":    ("EC-A10020-24",   (-20, 45)),     # + = abduction (left); belt 1.25:1
    "hip_pitch":   ("EC-A13720-11.4", (-125, 30)),    # - = flexion (thigh forward)
    "knee":        ("EC-A13720-11.4", (0, 135)),
    "ankle_pitch": ("AKE90-8",        (-50, 40)),     # + = toe down; two motors, rods
    "ankle_roll":  ("AKE90-8",        (-30, 30)),
}
ROLL_BELT = 1.25


def model_xml():
    """the robot as MJCF: base_link (the whole upper body) and two legs"""
    b = 1 + D["bracket"]
    a = {k: ACT[v[0]][2] for k, v in JOINT.items()}
    leg = dict(hip=a["hip_roll"] * b + 0.6,            # roll motor + its belt stage, behind the hip
               hip_roll=a["hip_pitch"] * b,             # pitch motor, outboard
               thigh=a["knee"] * b + 0.5,               # knee motor high + tube + four-bar rod
               calf=2 * a["ankle_pitch"] * b + 0.6,     # two ankle motors high + tube + rods
               ankle=0.1, foot=1.0)
    upper = D["mass"] - 2 * sum(leg.values())
    T, C, s = D["thigh"], D["calf"], D["sole"]

    def legx(side, sg):
        y = sg * D["hip_y"]
        lo, hi = JOINT["hip_roll"][1]
        rr = "%g %g" % ((lo, hi) if sg > 0 else (-hi, -lo))
        return f"""
    <body name="hip_{side}" pos="0 {y} 0">
      <joint name="hip_{side}_yaw" axis="0 0 1" range="{JOINT['hip_yaw'][1][0]} {JOINT['hip_yaw'][1][1]}"/>
      <geom type="cylinder" fromto="-0.06 0 0 -0.145 0 0" size="0.062" mass="{leg['hip']:.4f}"/>
      <body name="hip_roll_{side}">
        <joint name="hip_{side}_roll" axis="1 0 0" range="{rr}"/>
        <geom type="cylinder" fromto="0 {sg*0.045} 0 0 {sg*0.13} 0" size="0.085" mass="{leg['hip_roll']:.4f}"/>
        <body name="thigh_{side}">
          <joint name="hip_{side}_pitch" axis="0 1 0" range="{JOINT['hip_pitch'][1][0]} {JOINT['hip_pitch'][1][1]}"/>
          <geom type="cylinder" fromto="0 {-sg*0.0} -0.07 0 {-sg*0.0845} -0.07" size="0.085" mass="{ACT[JOINT['knee'][0]][2]*b:.4f}"/>
          <geom type="capsule" fromto="0 0 -0.1 0 0 {-T+0.03}" size="0.03" mass="0.5"/>
          <body name="calf_{side}" pos="0 0 {-T}">
            <joint name="knee_{side}" axis="0 1 0" range="{JOINT['knee'][1][0]} {JOINT['knee'][1][1]}"/>
            <geom type="cylinder" fromto="-0.02 {sg*0.03} -0.07 -0.02 {sg*0.085} -0.07" size="0.054" mass="{a['ankle_pitch']*b:.4f}"/>
            <geom type="cylinder" fromto="-0.02 {-sg*0.03} -0.07 -0.02 {-sg*0.085} -0.07" size="0.054" mass="{a['ankle_pitch']*b:.4f}"/>
            <geom type="capsule" fromto="0 0 -0.12 0 0 {-C+0.03}" size="0.025" mass="0.6"/>
            <body name="ankle_{side}" pos="0 0 {-C}">
              <joint name="ankle_{side}_pitch" axis="0 1 0" range="{JOINT['ankle_pitch'][1][0]} {JOINT['ankle_pitch'][1][1]}"/>
              <geom type="sphere" size="0.02" mass="0.1"/>
              <body name="foot_{side}">
                <joint name="ankle_{side}_roll" axis="1 0 0" range="{JOINT['ankle_roll'][1][0]} {JOINT['ankle_roll'][1][1]}"/>
                <geom type="box" pos="{(D['toe']-D['heel'])/2} 0 {-s+0.015}" size="{(D['toe']+D['heel'])/2} 0.045 0.015" mass="1.0"/>
                <site name="sole_{side}" pos="0 0 {-s}"/>
                <site name="toe_{side}" pos="{D['toe']} 0 {-s}"/>
                <site name="heel_{side}" pos="{-D['heel']} 0 {-s}"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""
    return f"""<mujoco model="run_check">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option gravity="0 0 -{G}" timestep="0.002"/>
  <default><geom contype="0" conaffinity="0"/></default>   <!-- masses only: the feet's ground force is solved, not simulated -->
  <worldbody>
    <body name="base_link" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" pos="0 0 0.33" size="0.15 0.17 0.40" mass="{upper:.4f}"/>
      {legx("L", 1)}
      {legx("R", -1)}
    </body>
  </worldbody>
</mujoco>"""


M = mujoco.MjModel.from_xml_string(model_xml())
DAT = mujoco.MjData(M)
LEGJ = {s: ["hip_%s_yaw" % s, "hip_%s_roll" % s, "hip_%s_pitch" % s, "knee_%s" % s,
            "ankle_%s_pitch" % s, "ankle_%s_roll" % s] for s in "LR"}
QA = {n: M.joint(n).qposadr[0] for s in "LR" for n in LEGJ[s]}
DA = {n: M.joint(n).dofadr[0] for s in "LR" for n in LEGJ[s]}


def _quat_y(a):
    return np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])


def ik(q, base, pitch_base, feet, loose=()):
    """set the base (pos, pitch) and solve each leg so its sole site sits at feet[s] =
    (pos, foot pitch about y): position and orientation, damped Gauss-Newton"""
    q[0:3] = base
    q[3:7] = _quat_y(pitch_base)
    for s, (p, fpitch) in feet.items():
        sid = M.site("sole_" + s).id
        cols = [DA[n] for n in LEGJ[s]]
        if abs(q[QA["knee_" + s]]) < 0.2:          # straight (or the first call): seed the
            q[QA["knee_" + s]] = 0.8                # knee bent backwards, the human branch
            q[QA["hip_%s_pitch" % s]] = -0.4
            q[QA["ankle_%s_pitch" % s]] = -0.4
        Rt = np.array([[math.cos(fpitch), 0, math.sin(fpitch)], [0, 1, 0],
                       [-math.sin(fpitch), 0, math.cos(fpitch)]])
        for it in range(200):
            DAT.qpos[:] = q
            mujoco.mj_kinematics(M, DAT)
            mujoco.mj_comPos(M, DAT)
            R = DAT.site_xmat[sid].reshape(3, 3)
            e = np.concatenate([p - DAT.site_xpos[sid], 0.5 * (np.cross(R[:, 0], Rt[:, 0])
                                + np.cross(R[:, 1], Rt[:, 1]) + np.cross(R[:, 2], Rt[:, 2]))])
            if np.abs(e).max() < 1e-11:
                break
            jp, jr = np.zeros((3, M.nv)), np.zeros((3, M.nv))
            mujoco.mj_jacSite(M, DAT, jp, jr, sid)
            J = np.vstack([jp, jr])[:, cols]
            dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ e)
            for n, d in zip(LEGJ[s], np.clip(dq, -0.3, 0.3)):
                q[QA[n]] = (q[QA[n]] + d + math.pi) % (2 * math.pi) - math.pi
        assert s in loose or np.abs(e).max() < 1e-3, ("IK", s, np.abs(e).max())   # a swing foot
        # may be out of reach: it carries nothing, the leg goes as near as it can
    return q


FC = 15.0     # Hz: motion data is low-passed at 6-15 Hz before inverse dynamics


def inverse(qs, dt, stance):
    """qs (T x nq) sampled at dt, stance[t] = feet on the ground -> joint torques (T x nv),
    joint speeds, and the residual wrench the feet could not carry (flight)"""
    T = len(qs)
    v = np.zeros((T, M.nv))
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        mujoco.mj_differentiatePos(M, v[t], (b - a) * dt, qs[a], qs[b])
    from scipy.signal import butter, filtfilt
    bb, aa = butter(2, FC / (0.5 / dt))
    if T > 20:
        v = filtfilt(bb, aa, v, axis=0)
    acc = np.gradient(v, dt, axis=0)
    tau, res = np.zeros((T, M.nv)), np.zeros(T)
    cop, fric = np.zeros(T), np.zeros(T)     # centre of pressure off the sole (m); |tangential|/normal
    for t in range(T):
        DAT.qpos[:], DAT.qvel[:], DAT.qacc[:] = qs[t], v[t], acc[t]
        mujoco.mj_inverse(M, DAT)
        f = DAT.qfrc_inverse.copy()
        if stance[t]:
            A = []
            for s in stance[t]:
                jp, jr = np.zeros((3, M.nv)), np.zeros((3, M.nv))
                mujoco.mj_jacSite(M, DAT, jp, jr, M.site("sole_" + s).id)
                A.append(np.vstack([jp, jr]).T)
            A = np.hstack(A)
            F = np.linalg.pinv(A[:6]) @ f[:6]
            f = f - A @ F
            for i, s_ in enumerate(stance[t]):
                R = DAT.site_xmat[M.site("sole_" + s_).id].reshape(3, 3)
                fl, ml = R.T @ F[6 * i:6 * i + 3], R.T @ F[6 * i + 3:6 * i + 6]
                if fl[2] <= 1e-6:
                    cop[t] = max(cop[t], 1.0); continue          # the ground would have to pull
                if fl[2] < 0.05 * D["mass"] * G:
                    continue                                       # unloading: no centre of pressure to speak of
                cx, cy = -ml[1] / fl[2], ml[0] / fl[2]
                th = math.asin(max(-1.0, min(1.0, -R[2, 0])))     # foot pitch: on its toe / heel edge
                lo, hi = (D["toe"], D["toe"]) if th > 0.02 else (-D["heel"], -D["heel"]) if th < -0.02 else (-D["heel"], D["toe"])
                off = max(lo - cx, cx - hi, abs(cy) - 0.045, 0.0)
                cop[t] = max(cop[t], off)
                fric[t] = max(fric[t], math.hypot(fl[0], fl[1]) / fl[2])
        res[t] = np.abs(f[:6]).max()
        tau[t] = f
    return tau, v, res, cop, fric


# ---------------------------------------------------------------- task motions
def minjerk(a, b, s):
    s = np.clip(s, 0, 1)
    return a + (b - a) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


BASE = M.body("base_link").id


def com_of(q):
    DAT.qpos[:] = q
    mujoco.mj_kinematics(M, DAT)
    mujoco.mj_comPos(M, DAT)
    return DAT.subtree_com[BASE].copy()


def place(q, com, pitch, feet, loose=()):
    """IK with the base moved until the whole robot's centre of mass is at com (x, z)"""
    base = np.array([com[0], 0.0, com[1] - 0.2]) if q is None else q[0:3].copy()
    q = M.qpos0.copy() if q is None else q
    for _ in range(30):
        q = ik(q, base, pitch, feet, loose)
        c = com_of(q)
        d = np.array([com[0] - c[0], 0.0, com[1] - c[2]])
        base += d
        if np.abs(d).max() < 1e-10:
            break
    return ik(q, base, pitch, feet, loose)


def foot_at(x_ankle, y, th=0.0, pivot="toe", ground=0.0):
    """the sole site of a foot pitched th (toe down +) resting on its toe (or heel) on
    the ground at `ground`, the ankle's flat-foot x being x_ankle"""
    if th == 0.0:
        return (np.array([x_ankle, y, ground]), 0.0)
    if pivot == "toe":
        px = x_ankle + D["toe"]
        return (np.array([px - D["toe"] * math.cos(th), y, ground + D["toe"] * math.sin(th)]), th)
    px = x_ankle - D["heel"]
    return (np.array([px + D["heel"] * math.cos(th), y, ground - D["heel"] * math.sin(th)]), th)


MID = (D["toe"] - D["heel"]) / 2          # the sole's middle, ahead of the ankle
STAND = None


def stand_com():
    """the centre-of-mass height standing, knees 10 deg"""
    global STAND
    if STAND is None:
        y = D["hip_y"]
        q = ik(M.qpos0.copy(), np.array([0, 0, 0.735]), 0.0,
               {"L": (np.array([0, y, 0]), 0.0), "R": (np.array([0, -y, 0]), 0.0)})
        STAND = float(com_of(q)[2])
    return STAND


def _build(frames):
    """frames: [(com (x, z), pelvis pitch, feet {side: pose}, stance [sides])] -> qs, stance"""
    qs, st, q = [], [], None
    for _ in range(3):                           # settle the first frame before the motion
        q = place(q, frames[0][0], frames[0][1], frames[0][2], [k for k in frames[0][2] if k not in frames[0][3]])
    for com, pitch, feet, sup in frames:
        q = place(q, com, pitch, feet, [k for k in feet if k not in sup])
        qs.append(q.copy()); st.append(sup)
    return np.array(qs), st


def task_sit_stand(chair=0.42, T=1.2, dt=0.002):
    """seat-off to standing: the centre of mass already over the feet (the trunk leant 40
    deg forward), rising to standing over the middle of the soles, feet flat"""
    y, zs = D["hip_y"], stand_com()
    z0 = chair + 0.27                              # seated hips + the upper body's centre over them
    fr = []
    for t in np.arange(0, T, dt):
        s = t / T
        fr.append(((MID - 0.02, minjerk(z0, zs, s)), math.radians(minjerk(40.0, 0.0, s)),
                   {"L": foot_at(0, y), "R": foot_at(0, -y)}, ["L", "R"]))
    qs, st = _build(fr)
    return qs, st, dt, "sit-to-stand, %.0f cm chair, %.1f s" % (chair * 100, T)


def task_step(rise, T, lean, dt=0.002, down=False, depth=0.30):
    """one leg on a step `rise` up carries the robot from the moment the trail foot leaves
    the floor (centre of mass over the lead foot, at standing height plus 30 % of the rise,
    the trunk leant) to standing on the step; down = the same motion reversed"""
    y, zs = D["hip_y"], stand_com()
    z0, z1 = zs + 0.3 * rise - 0.03, zs + rise
    fr = []
    for t in np.arange(0, T, dt):
        s = t / T
        zc = minjerk(z0, z1, s)
        # the trail foot swings up beside the body (it carries nothing; kept within reach)
        trail = np.array([depth + minjerk(-0.15, 0.0, s), -y, max(0.0, zc - zs + 0.02) + 0.10 * math.sin(math.pi * s)])
        fr.append(((depth + MID - 0.02, zc), math.radians(minjerk(lean, 0.0, s)),
                   {"L": foot_at(depth, y, ground=rise), "R": (trail, 0.0)}, ["L"]))
    if down:
        fr = fr[::-1]
    qs, st = _build(fr)
    return qs, st, dt, "%s a %.0f cm step on one leg, %.1f s" % ("down" if down else "up", rise * 100, T)


def task_jump(h=0.40, depth=0.20, lean=30.0, toe=35.0, dt=0.002, to_dz=-0.02):
    """countermovement jump of h: the centre of mass over the soles' middle drops `depth`
    leaning in, pushes up (acceleration ramped in over 60 ms) onto the toes, flies, lands
    on the toes and absorbs the same depth"""
    y, zs = D["hip_y"], stand_com()
    v0 = math.sqrt(2 * G * h)
    z_lo, z_to = zs - depth, zs + to_dz                    # squat; take-off (knees still bent, heels up)
    a = v0 ** 2 / (2 * (z_to - z_lo))
    pitch = lambda z: math.radians(lean) * float(np.clip((zs - z) / depth, 0, 1))
    feet = lambda th, g=0.0: {"L": foot_at(0, y, th, ground=g), "R": foot_at(0, -y, th, ground=g)}
    fr = []
    for t in np.arange(0, 0.5, dt):
        z = minjerk(zs, z_lo, t / 0.5)
        fr.append(((MID, z), pitch(z), feet(0.0), ["L", "R"]))
    z, vz, t = z_lo, 0.0, 0.0
    while z < z_to:
        vz += a * min(1.0, t / 0.06) * dt; z += vz * dt; t += dt
        th = math.radians(toe) * float(np.clip((z - (z_to - 0.06)) / 0.06, 0, 1)) ** 2   # heels up over the last 60 mm
        fr.append(((MID, z), pitch(z), feet(th), ["L", "R"]))
    t_air = 2 * vz / G
    th_land = math.radians(20)
    for t in np.arange(dt, t_air, dt):
        zc = z_to + vz * t - 0.5 * G * t * t
        th = math.radians(toe) + (th_land - math.radians(toe)) * t / t_air
        fr.append(((MID, zc), pitch(z_to), feet(th, zc - z_to), []))
    vz, z, t = -vz, z_to, 0.0
    dec = vz ** 2 / (2 * (z_to - z_lo))
    while vz < 0:
        vz += dec * min(1.0, t / 0.05) * dt; z += vz * dt; t += dt
        th = th_land * max(0.0, 1 - t / 0.06)
        fr.append(((MID, z), pitch(z), feet(th), ["L", "R"]))
    qs, st = _build(fr)
    return qs, st, dt, "jump %.2f m: take-off %.1f m/s, squat %.2f m" % (h, v0, depth)


def task_run(v=3.5, f=3.2, tc=0.17, swing_h=0.25, steps=6, dt=0.002, crouch=0.12):
    """running at v, one step = one stance + one flight: the centre of mass on a
    spring-mass path (half-sine vertical force, the ground force pointing from the centre
    of pressure to the centre of mass), the stance foot rolling heel -> flat -> toe under
    it, the swing foot on a minimum-jerk path, a level pelvis"""
    T, m = 1.0 / f, D["mass"]
    n = int(round(T / dt))
    Fpk = m * G * T * math.pi / (2 * tc)
    zc0 = stand_com() - crouch                   # centre of mass at mid-stance (knees bent)
    # integrate the stance from mid-stance (x = 0 over the sole's middle) forward
    zs_, xs_, z, vz, x, vx = [], [], zc0, 0.0, 0.0, v
    for i in range(n):
        tt = tc / 2 + i * dt
        Fz = Fpk * math.sin(math.pi * tt / tc) if tt < tc else 0.0
        Fx = Fz * x / z
        zs_.append(z); xs_.append(x)
        vz += (Fz / m - G) * dt; z += vz * dt
        vx += (Fx / m) * dt; x += vx * dt
    zs_, xs_ = np.array(zs_), np.array(xs_)
    h = n // 2                                   # mid-flight: mirror the second half
    zstep = np.concatenate([zs_[:h], zs_[:n - h][::-1]])
    xstep = np.concatenate([xs_[:h], 2 * xs_[h - 1] - xs_[:n - h][::-1]])
    L = 2 * xs_[h - 1]                           # step length
    y = 0.8 * D["hip_y"]
    def stance_pose(tau, xm, yy):                # tau in [0, tc] since touch-down
        s = tau / tc
        if s < 0.25:
            return foot_at(xm - MID, yy, -math.radians(10) * (1 - s / 0.25), pivot="heel")
        if s < 0.5:
            return foot_at(xm - MID, yy)
        return foot_at(xm - MID, yy, math.radians(25) * ((s - 0.5) / 0.5) ** 2, pivot="toe")
    com_x = lambda i: (i // n) * L + xstep[i % n]
    com_z = lambda i: zstep[i % n]
    fr = []
    for i in range(n * steps):
        k, j = divmod(i, n)
        t = j * dt
        com = (com_x(i), com_z(i))
        feet, sup = {}, []
        for side, sg, par in (("L", 1, 0), ("R", -1, 1)):
            mine = (k % 2) == par                # this step's stance is this leg's
            yy = sg * y
            if mine and t <= tc / 2:
                feet[side] = stance_pose(tc / 2 + t, k * L, yy); sup.append(side)
            elif (not mine) and t >= T - tc / 2:
                feet[side] = stance_pose(t - (T - tc / 2), (k + 1) * L, yy); sup.append(side)
            else:
                if mine:                         # lifted off this step, lands two steps on
                    since, x0, x1 = t - tc / 2, k * L, (k + 2) * L
                else:
                    since, x0, x1 = t + T - tc / 2, (k - 1) * L, (k + 1) * L
                tsw = 2 * T - tc
                sw = since / tsw
                # fore-aft: a cubic Hermite relative to the centre of mass, its end slopes the
                # body's speed backwards, so the foot leaves and lands at zero ground speed and
                # swings back under the body before landing (leg retraction); height: a lift
                # that is zero at both ends
                p0, th0 = stance_pose(tc, x0, yy)
                p1, th1 = stance_pose(0.0, x1, yy)
                i0, i1 = i - int(round(since / dt)), i + int(round((tsw - since) / dt))
                r0, r1 = p0[0] - com_x(i0), p1[0] - com_x(i1)
                mt = -(com_x(i0 + 1) - com_x(i0)) / dt * tsw
                s1 = sw
                h00, h10 = 1 - 3 * s1 ** 2 + 2 * s1 ** 3, s1 - 2 * s1 ** 2 + s1 ** 3
                h01, h11 = 3 * s1 ** 2 - 2 * s1 ** 3, s1 ** 3 - s1 ** 2
                xr = h00 * r0 + h10 * mt + h01 * r1 + h11 * mt
                lift = swing_h * math.sin(math.pi * sw) ** 1.5
                feet[side] = (np.array([com_x(i) + xr, yy, minjerk(p0[2], p1[2], sw) + lift]),
                              float(minjerk(th0, th1, sw)))
        fr.append((com, 0.0, feet, sup))
    qs, st = _build(fr)
    return qs, st, dt, "run %.1f m/s: %.1f steps/s, contact %.0f ms, step %.2f m" % (v, f, tc * 1000, L)


TASKS = {
    "sit_stand": lambda: task_sit_stand(),
    "stairs_up": lambda: task_step(0.18, 0.6, 15.0),
    "stairs_down": lambda: task_step(0.18, 0.6, 15.0, down=True),
    "box": lambda: task_step(0.40, 1.0, 30.0),
    "jump": lambda: task_jump(0.25, toe=30.0),
    "run": lambda: task_run(3.5, f=3.6, tc=0.14),
    "run_fast": lambda: task_run(4.5, f=3.9, tc=0.12),
}
CONTINUOUS = {"run", "run_fast"}


# ---------------------------------------------------------------- actuator loads
def actuator_loads(tau, v):
    """joint torques/speeds -> each actuator's (torque, speed) time series, left leg"""
    s = "L"
    out = {}
    for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee"):
        n = "hip_%s_%s" % (s, j[4:]) if j.startswith("hip") else "knee_" + s
        g = ROLL_BELT if j == "hip_roll" else D["knee_ratio"] if j == "knee" else 1.0
        out[j] = (tau[:, DA[n]] / g, v[:, DA[n]] * g)
    tp, tr = tau[:, DA["ankle_L_pitch"]], tau[:, DA["ankle_L_roll"]]
    wp, wr = v[:, DA["ankle_L_pitch"]], v[:, DA["ankle_L_roll"]]
    gp = D["ankle_ratio"]
    gr = D["ankle_rod_y"] / (0.06 / gp)              # rod offset / crank (crank = lever / gp)
    e = D["ankle_eff"]
    out["ankle_a"] = ((tp / (2 * gp) + tr / (2 * gr)) / e, gp * wp + gr * wr)
    out["ankle_b"] = ((tp / (2 * gp) - tr / (2 * gr)) / e, gp * wp - gr * wr)
    return out


KEY = {"hip_yaw": JOINT["hip_yaw"][0], "hip_roll": JOINT["hip_roll"][0], "hip_pitch": JOINT["hip_pitch"][0],
       "knee": JOINT["knee"][0], "ankle_a": "AKE90-8", "ankle_b": "AKE90-8"}


def check(name, verbose=True):
    qs, stance, dt, label = TASKS[name]()
    tau, v, res, cop, fric = inverse(qs, dt, stance)
    E = int(round(0.02 / dt))                    # 20 ms at each end: differentiation edges
    qs, stance, tau, v, res, cop, fric = (x[E:-E] for x in (qs, stance, tau, v, res, cop, fric))
    # +-30 ms round each touch-down / lift-off: the impact instant, which the legs' own
    # compliance and the actuators' back-drive take; reported apart, not held to the envelope
    W = int(round(0.03 / dt))
    imp = np.zeros(len(stance), bool)
    for i in range(1, len(stance)):
        if set(stance[i]) != set(stance[i - 1]):
            imp[max(0, i - W):i + W] = True
    sup = np.array([len(s) > 0 for s in stance]) & ~imp
    worst = res[sup].max() if sup.any() else 0.0
    rows, ok = [], True
    for a, (t, w) in actuator_loads(tau, v).items():
        k = KEY[a]
        av = np.array([avail(k, x) for x in w])
        use = np.where(av > 1e-6, np.abs(t) / np.maximum(av, 1e-6), np.inf)   # fraction of the envelope
        imp_peak = float(np.abs(t[imp]).max()) if imp.any() else 0.0
        t, w, use = t[~imp], w[~imp], use[~imp]
        peak_t, peak_w, worst_u = np.abs(t).max(), np.abs(w).max(), use.max()
        rms = math.sqrt(float(np.mean(t ** 2)))
        rated = ACT[k][1] * (ROLL_BELT if a == "hip_roll" else 1)
        th_ok = (name not in CONTINUOUS) or rms <= ACT[k][1]
        good = worst_u <= 1.0 and th_ok
        ok &= good
        rows.append((a, k, peak_t, peak_w, worst_u, rms, ACT[k][1], good, imp_peak))
    # ranges
    rng = []
    for n in LEGJ["L"]:
        q = np.degrees(qs[:, QA[n]])
        lo, hi = M.jnt_range[M.joint(n).id]
        rng.append((n, q.min(), q.max(), math.degrees(lo), math.degrees(hi)))
        ok &= q.min() >= math.degrees(lo) - 0.5 and q.max() <= math.degrees(hi) + 0.5
    if verbose:
        print("== %s: %s   (%s)" % (name, "PASS" if ok else "SHORT", label))
        print("   feet carry it all: residual %.1f N; centre of pressure off the sole by up to %.0f mm "
              "(contact edge when the foot rolls); friction needed %.2f"
              % (worst, 1000 * cop[sup].max() if sup.any() else 0.0, fric[sup].max() if sup.any() else 0.0))
        for a, k, pt, pw, u, rms, rt, good, ip in rows:
            print("   %-10s %-15s peak %6.1f N.m  speed %5.1f rad/s  envelope use %s  RMS %5.1f / rated %3.0f%s"
                  % (a, k, pt, pw, "OVERSPEED" if not np.isfinite(u) else "%4.0f %%" % (100 * u), rms, rt,
                     "" if good else "   <-- SHORT") + ("   (impact %.0f)" % ip if ip else ""))
        for n, a, b, lo, hi in rng:
            flag = "" if a >= lo - 0.5 and b <= hi + 0.5 else "   <-- OUT OF RANGE"
            print("   %-15s %7.1f .. %6.1f deg   (range %4.0f .. %4.0f)%s" % (n, a, b, lo, hi, flag))
    return ok


if __name__ == "__main__":
    print("model: %.2f kg, legs %.2f kg each; envelopes read at %.1f V"
          % (sum(M.body_mass), sum(M.body_mass[M.body("hip_L").id:M.body("foot_L").id + 1]), VBUS))
    assert abs(sum(M.body_mass) - D["mass"]) < 1e-3
    names = sys.argv[1:] or list(TASKS)
    res = {n: check(n) for n in names}
    print("\n" + "  ".join("%s %s" % (n, "PASS" if r else "SHORT") for n, r in res.items()))
