"""Physical proof for the running robot's legs: the sagittal-plane model of run_check (same
masses, trunk free to pitch) simulated in MuJoCo with real foot contacts, every actuator's
torque clamped each step to its envelope at its present speed (run_check.ACT at
run_check.VBUS: 42 V, an empty 13S pack), and the motion found by search (CEM). If the
search finds the motion, these actuators can do it; the best it finds bounds what they can.

    python cad/tools/gait_sim.py jump        # the highest squat jump
"""
import math, os, sys, time
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_check as R                                    # noqa: E402
import gait_opt as GO                                    # noqa: E402

DT = 0.001
ROOT_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def sim_xml():
    """the planar robot over a floor; only the feet touch it"""
    s = GO.planar_xml()
    s = s.replace('timestep="0.002"', 'timestep="%g" integrator="implicitfast"' % DT)
    s = s.replace('<worldbody>', '<worldbody>\n    <geom name="floor" type="plane" size="20 2 0.1" pos="0 0 0" '
                  'contype="1" conaffinity="1" friction="0.8 0.02 0.001"/>')
    out = []
    for line in s.splitlines():
        if 'type="box"' in line and 'mass="1.0"' in line:          # the feet
            line = line.replace("<geom ", '<geom contype="1" conaffinity="1" condim="3" friction="0.8 0.02 0.001" ')
        out.append(line)
    s = "\n".join(out)
    acts = "".join('<motor name="m_%s" joint="%s"/>' % (j, j) for j in
                   ("hip_L_pitch", "knee_L", "ankle_L_pitch", "hip_R_pitch", "knee_R", "ankle_R_pitch"))
    return s.replace("</worldbody>", "</worldbody>\n  <actuator>%s</actuator>" % acts)


SM = mujoco.MjModel.from_xml_string(sim_xml())
PITCH = SM.joint("bp").qposadr[0]
SJ = ["hip_L_pitch", "knee_L", "ankle_L_pitch", "hip_R_pitch", "knee_R", "ankle_R_pitch"]
SQ = np.array([SM.joint(j).qposadr[0] for j in SJ])
SV = np.array([SM.joint(j).dofadr[0] for j in SJ])
KEYS = [R.JOINT["hip_pitch"][0], R.JOINT["knee"][0], "AKE90-8"] * 2
RATIO = [1.0, R.D["knee_ratio"], R.D["ankle_ratio"]] * 2
NMOT = [1, 1, 2] * 2                                     # the ankle: two motors through the rods
EFF = [1.0, 1.0, R.D["ankle_eff"]] * 2


ENV_SCALE = float(os.environ.get("ENV_SCALE", 1.0))   # >1: a what-if, actuators stronger than catalogued


def joint_limit(i, w):
    """the most joint torque actuator i can give at joint speed w"""
    return ENV_SCALE * NMOT[i] * RATIO[i] * EFF[i] * R.avail(KEYS[i], w * RATIO[i] / ENV_SCALE ** 0)


def clamp(tau, w):
    return np.array([np.clip(tau[i], -joint_limit(i, w[i]), joint_limit(i, w[i])) for i in range(6)])


def stand(d, z_drop=0.0, knee=math.radians(10)):
    """both feet flat on the floor, knees bent, base pitch 0"""
    mujoco.mj_resetData(SM, d)
    q = np.zeros(6)
    q[[1, 4]] = knee
    q[[0, 3]] = -knee / 2
    q[[2, 5]] = -knee / 2
    d.qpos[SQ] = q
    mujoco.mj_kinematics(SM, d)
    zmin = min(d.site_xpos[SM.site("%s_%s" % (p, s)).id][2] for p in ("heel", "toe") for s in "LR")
    d.qpos[SM.joint("bz").qposadr[0]] -= zmin
    mujoco.mj_forward(SM, d)


# ---------------------------------------------------------------- jump
def jump_run(p, d=None, info=None):
    """a squat jump from standing: p = [squat knee, squat hip, squat ankle, t_down, t_push,
    end hip, end knee, end ankle, kp scale, push feed-forward (fraction of the envelope),
    trunk feedback]; joints PD-tracked plus a feed-forward extending torque in the push,
    the sum clamped to the envelope at the joint's speed. Score: the flight rise its
    take-off speed gives (v^2 / 2g) -- only for a clean take-off: straight up (|vx| < 0.3
    m/s), the trunk within 15 deg and turning < 2 rad/s, no leg joint pressed against its
    end stop in the push (limit force < 30 N.m), else a penalty"""
    d = d or mujoco.MjData(SM)
    stand(d)
    q0 = d.qpos[SQ].copy()
    sq = np.array([p[1], p[0], p[2]] * 2)
    en = np.array([p[5], p[6], p[7]] * 2)
    td, tp = p[3], p[4]
    kp = np.array([1500.0, 2000.0, 600.0] * 2) * p[8]
    kd = 2 * np.sqrt(kp) * 0.6
    sign = np.array([-1.0, -1.0, 1.0] * 2)
    mujoco.mj_forward(SM, d)
    t, stop, air = 0.0, 0.0, 0
    while t < td + tp + 0.3:
        if t < td:
            s_ = t / td; tgt = q0 + (sq - q0) * (10 * s_ ** 3 - 15 * s_ ** 4 + 6 * s_ ** 5); ff = 0.0
        else:
            s_ = min(1.0, (t - td) / tp); tgt = sq + (en - sq) * (10 * s_ ** 3 - 15 * s_ ** 4 + 6 * s_ ** 5); ff = p[9]
        q, w = d.qpos[SQ], d.qvel[SV]
        tgt = tgt.copy()
        tgt[[0, 3]] -= p[10] * d.qpos[PITCH]
        tau = kp * (tgt - q) - kd * w
        if ff:
            tau = tau + sign * ff * np.array([joint_limit(i, w[i]) for i in range(6)])
        d.ctrl[:] = clamp(tau, w)
        mujoco.mj_step(SM, d)
        t += DT
        stop = max(stop, float(np.abs(d.qfrc_constraint[SV]).max()) if d.ncon == 0 or t > td else 0.0)
        air = air + 1 if d.ncon == 0 else 0
        if t > td and air >= 5:                          # take-off: 5 ms off the ground, going up
            mujoco.mj_subtreeVel(SM, d)
            v = d.subtree_linvel[0]
            if v[2] < 0.2:
                continue
            ok = abs(v[0]) < 0.3 and abs(d.qpos[PITCH]) < math.radians(15) and abs(d.qvel[SM.joint("bp").dofadr[0]]) < 2.0
            # a joint's end stop pushes back only through the limit's own constraint rows;
            # contacts fill qfrc_constraint too, so use the limit rows alone
            lim = sum(abs(d.efc_force[i]) for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
            if info is not None:
                info.update(v=v.copy(), pitch=d.qpos[PITCH], t=t, limit=lim)
            rise = v[2] ** 2 / (2 * G_) if v[2] > 0 else 0.0
            # graded: how far past each cleanliness limit (so a search can climb back)
            pen = (1.0 * max(0.0, abs(v[0]) - 0.3) + 2.0 * max(0.0, abs(d.qpos[PITCH]) - math.radians(15))
                   + 0.3 * max(0.0, abs(d.qvel[SM.joint("bp").dofadr[0]]) - 2.0) + 0.002 * max(0.0, LIMIT_HIT[0] - 30.0))
            if info is not None:
                info.update(pen=pen, clean=pen == 0.0)
            return rise - pen
        LIMIT_HIT[0] = max(LIMIT_HIT[0], sum(abs(d.efc_force[i]) for i in range(d.nefc)
                                             if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)) if t > td else 0.0
        if abs(d.qpos[PITCH]) > math.radians(45):
            return -1.0
    return -1.0


LIMIT_HIT = [0.0]
G_ = 9.81


# ---------------------------------------------------------------- running
L1, L2 = R.D["thigh"], R.D["calf"]
FOOT_G = {sd: [g for g in range(SM.ngeom) if SM.geom_bodyid[g] == SM.body("foot_" + sd).id and SM.geom_contype[g]]
          for sd in "LR"}
BX, BZ = SM.joint("bx").qposadr[0], SM.joint("bz").qposadr[0]
VX, VP = SM.joint("bx").dofadr[0], SM.joint("bp").dofadr[0]


def in_contact(d, side):
    gs = FOOT_G[side]
    return any((d.contact[i].geom1 in gs) or (d.contact[i].geom2 in gs) for i in range(d.ncon))


def place_angle(x, knee):
    """the thigh's world angle phi (0 down, + back) that puts the ankle x ahead of the hip
    with this knee angle: x = -L1 sin(phi) - L2 sin(phi + knee)"""
    phi = 0.0
    for _ in range(20):
        f = -L1 * math.sin(phi) - L2 * math.sin(phi + knee) - x
        df = -L1 * math.cos(phi) - L2 * math.cos(phi + knee)
        phi -= f / df
    return phi


# p: 0 Ts (expected stance s), 1 kv (foot placement per m/s of speed error), 2 x_off,
#    3 stance knee target, 4 stance knee kp scale, 5 knee thrust (fraction of envelope),
#    6 ankle push (fraction), 7 trunk lean target, 8 trunk kp, 9 swing knee fold,
#    10 swing fold fraction of the expected swing, 11 touch-down knee
def run_ctrl(p, v_des, T=4.0, d=None, info=None):
    d = d or mujoco.MjData(SM)
    mujoco.mj_resetData(SM, d)
    # start in flight at speed: left leg reaching to land, right leg folded behind
    pitch0 = p[7]
    d.qpos[PITCH] = pitch0
    phiL = place_angle(v_des * p[0] / 2 + p[2], p[11])
    d.qpos[SQ[[0, 1, 2]]] = [phiL - pitch0, p[11], -(phiL + p[11])]
    d.qpos[SQ[[3, 4, 5]]] = [0.3 - pitch0, p[9], -(0.3 + p[9])]
    mujoco.mj_kinematics(SM, d)
    zmin = min(d.site_xpos[SM.site("%s_%s" % (q_, s_)).id][2] for q_ in ("heel", "toe") for s_ in "LR")
    d.qpos[BZ] -= zmin - 0.03
    d.qvel[VX] = v_des
    mujoco.mj_forward(SM, d)
    kp = np.array([1500.0, 2000.0, 400.0] * 2)
    kd = 2 * np.sqrt(kp) * 0.6
    phase = {"L": "swing", "R": "swing"}
    t_lo = {"L": -1.0, "R": 0.0}
    t_td = {"L": -1.0, "R": -1.0}
    swing_T = 2 * (1 / 3.6) - p[0]
    ts_meas = p[0]
    t, xs, loads, fell = 0.0, [], [], False
    tau = np.zeros(6)
    while t < T:
        pitch, dpitch = d.qpos[PITCH], d.qvel[VP]
        vx = d.qvel[VX]
        q, w = d.qpos[SQ], d.qvel[SV]
        tgt = q.copy() * 0
        ff = np.zeros(6)
        for k, sd in enumerate("LR"):
            i = 3 * k
            c = in_contact(d, sd)
            if phase[sd] == "swing" and c and t - t_lo[sd] > 0.05:
                phase[sd], t_td[sd] = "stance", t
            elif phase[sd] == "stance" and not c and t - t_td[sd] > 0.04:
                phase[sd], t_lo[sd] = "swing", t
                ts_meas = 0.5 * ts_meas + 0.5 * (t - t_td[sd])
            if phase[sd] == "stance":
                s_ = (t - t_td[sd]) / max(p[0], 1e-3)
                tgt[i + 1] = p[3]
                kp_i = kp.copy(); kp_i[i + 1] *= p[4]
                # the stance hip turns the trunk to its lean; knee spring; ankle keeps the foot
                ff[i] = -(p[8] * (pitch - p[7]) + 2 * math.sqrt(p[8]) * dpitch) * -1
                phi = pitch + q[i]
                tgt[i] = q[i]
                tgt[i + 2] = -(phi + q[i + 1])
                if s_ > 0.45:
                    ff[i + 1] = -p[5] * joint_limit(i + 1, w[i + 1])
                    ff[i + 2] = p[6] * joint_limit(i + 2, w[i + 2])
            else:
                s_ = (t - t_lo[sd]) / max(swing_T, 1e-3)
                if s_ < p[10]:
                    knee = p[9]
                    xh = -0.10
                else:
                    knee = p[11]
                    xh = vx * ts_meas / 2 + p[1] * (vx - v_des) + p[2]
                phi = place_angle(xh, knee)
                tgt[i:i + 3] = [phi - pitch, knee, -(phi + knee)]
        err = tgt - q
        tq = kp * err - kd * w
        for k, sd in enumerate("LR"):
            i = 3 * k
            if phase[sd] == "stance":
                tq[i] = 0.0 - 0.1 * kd[i] * w[i]           # the stance hip: trunk only
                tq[i + 1] = kp[i + 1] * p[4] * err[i + 1] - kd[i + 1] * w[i + 1]
        tq = tq + ff
        tq = clamp(tq, w)
        d.ctrl[:] = tq
        mujoco.mj_step(SM, d)
        t += DT
        if info is not None:
            loads.append((tq.copy(), w.copy(), q.copy()))
        xs.append((t, d.qpos[BX], vx))
        hip_z = d.xpos[SM.body("base_link").id][2]
        if abs(pitch) > math.radians(40) or hip_z < 0.45:
            fell = True
            break
    xs = np.array(xs)
    tail = xs[xs[:, 0] > max(0.0, xs[-1, 0] - 2.0)]
    v_mean = (tail[-1, 1] - tail[0, 1]) / max(tail[-1, 0] - tail[0, 0], 1e-3) if len(tail) > 10 else 0.0
    score = xs[-1, 0] - (0.0 if not fell else 1.0) - 2.0 * abs(v_mean - v_des) * (xs[-1, 0] / T)
    if info is not None:
        info.update(t=xs[-1, 0], fell=fell, v=v_mean, loads=loads)
    return score


def cem(f, mu, sd, lo, hi, n=48, elite=8, iters=25, seed=0, label=""):
    rng = np.random.default_rng(seed)
    best, bx = -np.inf, mu.copy()
    for it in range(iters):
        X = np.clip(rng.normal(mu, sd, size=(n, len(mu))), lo, hi)
        F = np.array([f(x) for x in X])
        idx = np.argsort(-F)[:elite]
        mu, sd = X[idx].mean(0), np.maximum(X[idx].std(0), 0.02 * (hi - lo))
        if F[idx[0]] > best:
            best, bx = F[idx[0]], X[idx[0]].copy()
        print("  %s iter %2d: best %.3f, mean elite %.3f" % (label, it, best, F[idx].mean()), flush=True)
    return best, bx


RUN_LO = np.array([0.08, 0.0, -0.10, math.radians(15), 0.2, 0.0, 0.0, math.radians(-5), 50.0, math.radians(60), 0.3, math.radians(5)])
RUN_HI = np.array([0.20, 0.3, 0.15, math.radians(60), 2.0, 1.0, 1.0, math.radians(20), 1500.0, math.radians(130), 0.8, math.radians(40)])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "jump"
    if what == "run":
        v = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
        mu = np.array([0.14, 0.1, 0.0, math.radians(35), 1.0, 0.4, 0.4, math.radians(5), 500.0, math.radians(100), 0.5, math.radians(20)])
        sd = (RUN_HI - RUN_LO) / 4
        t0 = time.time()
        best, x = cem(lambda p: run_ctrl(p, v), mu, sd, RUN_LO, RUN_HI, n=48, elite=8, iters=60, label="run %.1f" % v)
        info = {}; run_ctrl(x, v, T=6.0, info=info)
        print("[envelope x%.0f] " % ENV_SCALE + "running %.1f m/s at %.0f V: best score %.2f; 6 s replay: ran %.2f s, fell %s, mean speed %.2f m/s (%.0f s)"
              % (v, R.VBUS, best, info["t"], info["fell"], info["v"], time.time() - t0))
        print("  parameters:", np.round(x, 3).tolist())
        np.save(os.path.join(ROOT_, "urdf", "run_ctrl_%.1f.npy" % v), x)
    if what == "jump":
        r = math.radians
        lo = np.array([r(60), r(-110), r(-45), 0.35, 0.12, r(-25), r(5), r(5), 0.3, 0.0, 0.0])
        hi = np.array([r(130), r(-30), r(-5), 0.8, 0.45, r(5), r(25), r(35), 2.0, 1.0, 1.5])
        mu = np.array([r(95), r(-65), r(-30), 0.5, 0.25, r(-5), r(5), r(25), 1.0, 0.6, 1.0])
        sd = (hi - lo) / 4
        t0 = time.time()
        best, x = cem(jump_run, mu, sd, lo, hi, label="jump", iters=40)
        info = {}; jump_run(x, info=info)
        print("  clean take-off: %s; vx %.2f, vz %.2f m/s, trunk %.0f deg, end-stop force in the push %.0f N.m"
              % (info.get("clean"), info["v"][0], info["v"][2], math.degrees(info["pitch"]), LIMIT_HIT[0]))
        print("highest squat jump at %.0f V: the centre of mass rises %.3f m (%.0f s)" % (R.VBUS, best, time.time() - t0))
        print("  parameters:", np.round(x, 3).tolist())
