"""Optimised running stride and jump push-off for the running robot, to settle whether its
leg actuators can do them (cad/tools/run_check.py's templates could not: a level pelvis and
fixed paths ask the stance foot for moments no sole can give).

Direct collocation on the sagittal-plane version of run_check's model (same masses; the
trunk free to pitch, the arms lumped in it): joint positions at N nodes are the decision
variables, velocities and accelerations their finite differences, and each node's inverse
dynamics with the ground forces at each foot's heel and toe point gives the joint
torques. Constraints: the trunk's three rows carry no torque (the ground forces do it),
the forces push (fz >= 0) inside a friction cone (mu 0.8), stance points stay put, the
other foot and a lifted heel stay above the ground, the landing foot arrives at zero
speed, joint ranges, and every actuator's |torque| <= u x its envelope at its speed
(run_check.ACT at run_check.VBUS). Objective: u, the worst fraction of an envelope used --
u <= 1 means the motion is possible with these actuators; u > 1 says how far short.

    python cad/tools/gait_opt.py run 3.5
    python cad/tools/gait_opt.py jump 0.30
"""
import math, os, sys, time
import numpy as np
import mujoco
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_check as R                                    # noqa: E402

MU = 0.8
WT = R.D["mass"] * R.G          # force variables are in robot weights
FOOT = R.D["toe"] + R.D["heel"]


def planar_xml():
    """run_check's robot in the sagittal plane: the base slides in x, z and pitches; each
    leg keeps hip pitch, knee and ankle pitch (yaw, roll and ankle roll locked)"""
    s = R.model_xml()
    s = s.replace('<freejoint name="root"/>',
                  '<joint name="bx" type="slide" axis="1 0 0"/><joint name="bz" type="slide" axis="0 0 1"/>'
                  '<joint name="bp" axis="0 1 0"/>')
    out = []
    for line in s.splitlines():
        if ('<joint name="hip_' in line and ('_yaw"' in line or '_roll"' in line)) or '_roll" axis="1 0 0"' in line:
            continue
        out.append(line)
    return "\n".join(out)


PM = mujoco.MjModel.from_xml_string(planar_xml())
PD = mujoco.MjData(PM)
assert PM.nq == 9, PM.nq
J = ["bx", "bz", "bp", "hip_R_pitch", "knee_R", "ankle_R_pitch", "hip_L_pitch", "knee_L", "ankle_L_pitch"]
assert [PM.joint(i).name for i in range(PM.njnt)] == ["bx", "bz", "bp", "hip_L_pitch", "knee_L", "ankle_L_pitch",
                                                      "hip_R_pitch", "knee_R", "ankle_R_pitch"]
# the model's own order is L then R; ours (stance leg first) is a permutation
PERM = [0, 1, 2, 6, 7, 8, 3, 4, 5]               # ours -> model qpos index
Z0 = 1.0                                          # base body sits at z = 1 + bz
SITES = {(s, p): PM.site("%s_%s" % (p, s)).id for s in "LR" for p in ("heel", "toe")}
LO = np.radians([R.JOINT["hip_pitch"][1][0], R.JOINT["knee"][1][0], R.JOINT["ankle_pitch"][1][0]])
HI = np.radians([R.JOINT["hip_pitch"][1][1], R.JOINT["knee"][1][1], R.JOINT["ankle_pitch"][1][1]])
G_ANK = R.D["ankle_ratio"]


def to_model(q):
    m = np.zeros(9)
    m[PERM] = q
    m[1] -= Z0
    return m


def fd(Q, h_prev, h_next, Qm, Qp):
    """non-uniform central differences: velocities and accelerations at each node"""
    h1, h2 = h_prev[:, None], h_next[:, None]
    V = (-h2 / (h1 * (h1 + h2))) * Qm + ((h2 - h1) / (h1 * h2)) * Q + (h1 / (h2 * (h1 + h2))) * Qp
    A = 2 * (Qm / (h1 * (h1 + h2)) - Q / (h1 * h2) + Qp / (h2 * (h1 + h2)))
    return V, A


def dynamics(Q, V, A, forces):
    """per node: the 9 generalised forces left after the ground forces (base rows must be
    zero), and the points' positions; forces[k] = {(side, point): (fx, fz)}"""
    T = np.zeros((len(Q), 9))
    P = np.zeros((len(Q), 4, 2))                  # R heel, R toe, L heel, L toe: x, z
    jp = np.zeros((3, PM.nv))
    for k in range(len(Q)):
        PD.qpos[:] = to_model(Q[k])
        PD.qvel[:] = 0; PD.qvel[PERM] = V[k]
        PD.qacc[:] = 0; PD.qacc[PERM] = A[k]
        mujoco.mj_inverse(PM, PD)
        f = PD.qfrc_inverse.copy()
        for (side, pt), (fx, fz) in forces[k].items():
            mujoco.mj_jacSite(PM, PD, jp, None, SITES[(side, pt)])
            f -= (jp[0] * fx + jp[2] * fz) * WT
        T[k] = f[PERM]
        for i, key in enumerate((("R", "heel"), ("R", "toe"), ("L", "heel"), ("L", "toe"))):
            P[k, i] = PD.site_xpos[SITES[key]][[0, 2]]
    return T, P


def motor_loads(T, V, leg):
    """(torque, speed) per actuator of leg 0 (R, the stance leg first) or 1: hip, knee, each ankle motor"""
    i = 3 + 3 * leg
    kr = R.D["knee_ratio"]
    e = R.D["ankle_eff"]
    return [("hip", R.JOINT["hip_pitch"][0], T[:, i], V[:, i]),
            ("knee", R.JOINT["knee"][0], T[:, i + 1] / kr, V[:, i + 1] * kr),
            ("ankle", "AKE90-8", T[:, i + 2] / (2 * G_ANK * e), V[:, i + 2] * G_ANK)]


def avail_x(key, w):
    """run_check.avail, its last segment carried on past top speed (negative there), so an
    optimiser asking too much speed sees which way to go"""
    e = R.ACT[key][3]
    sc = R.VBUS / 48.0
    ws, ts = [p[0] * sc for p in e], [p[1] for p in e]
    w = abs(w)
    if w <= ws[-1]:
        return float(np.interp(w, ws, ts))
    return ts[-1] + (ts[-1] - ts[-2]) / (ws[-1] - ws[-2]) * (w - ws[-1])


def envelope_rows(T, V, u, legs=(0, 1)):
    """u * avail(speed) - |torque| >= 0, both signs, every node and actuator"""
    out = []
    for leg in legs:
        for _, key, t, w in motor_loads(T, V, leg):
            av = np.array([avail_x(key, x) for x in w])
            out += [u * av - t, u * av + t]
    return np.concatenate(out)


# ---------------------------------------------------------------- running
class Run:
    """one running step: the right leg's stance (flat, then on its toe) and a flight; the
    next step is this one mirrored (left <-> right) and moved on by the step length"""

    def __init__(self, v, n_flat=8, n_toe=5, n_fly=9):
        self.v, self.nf, self.nt, self.nfl = v, n_flat, n_toe, n_fly
        self.N = n_flat + n_toe + n_fly
        self.ns = n_flat + n_toe
        nfrc = n_flat * 4 + n_toe * 2
        # x = [Q (N*9), forces, dt_stance, dt_flight, heel x of the stance foot, u]
        self.iF = self.N * 9
        self.iD = self.iF + nfrc
        self.n = self.iD + 4

    def unpack(self, x):
        N = self.N
        Q = x[:N * 9].reshape(N, 9)
        f = x[self.iF:self.iD]
        dts, dtf, hx, u = x[self.iD:self.iD + 4]
        forces, j = [], 0
        for k in range(N):
            d = {}
            if k < self.nf:
                d[("R", "heel")] = (f[j], f[j + 1]); d[("R", "toe")] = (f[j + 2], f[j + 3]); j += 4
            elif k < self.ns:
                d[("R", "toe")] = (f[j], f[j + 1]); j += 2
            forces.append(d)
        return Q, forces, dts, dtf, hx, u

    def steps(self, dts, dtf):
        h = np.array([dts] * (self.ns - 1) + [dtf] * (self.nfl + 1))   # spacing to the next node
        T = h.sum()
        return h, T

    def mirror(self, q, dx):
        m = q.copy()
        m[3:6], m[6:9] = q[6:9], q[3:6]
        m[0] += dx
        return m

    def kin(self, x):
        Q, forces, dts, dtf, hx, u = self.unpack(x)
        h, T = self.steps(dts, dtf)
        L = self.v * T
        Qm = np.vstack([self.mirror(Q[-1], -L), Q[:-1]])
        Qp = np.vstack([Q[1:], self.mirror(Q[0], L)])
        hp = np.concatenate([[h[-1]], h[:-1]])
        V, A = fd(Q, hp, h, Qm, Qp)
        return Q, V, A, forces, h, T, L, hx, u

    def cons(self, x):
        Q, V, A, forces, h, T, L, hx, u = self.kin(x)
        Tq, P = dynamics(Q, V, A, forces)
        eq = [Tq[:, :3].ravel() / (R.D["mass"] * R.G)]
        ineq = []
        for k in range(self.N):
            if k < self.nf:                                     # flat: heel on its spot, toe down
                eq.append([P[k, 0, 0] - hx, P[k, 0, 1], P[k, 1, 1]])
            elif k < self.ns:                                   # on the toe, the heel up
                eq.append([P[k, 1, 0] - hx - FOOT, P[k, 1, 1]])
                ineq.append([P[k, 0, 1]])
            else:                                               # flight: both feet clear
                ineq.append([P[k, 0, 1], P[k, 1, 1]])
            if k < self.N - 1:
                ineq.append([P[k, 2, 1] - 0.01, P[k, 3, 1] - 0.01])   # the swing foot clear
        # the swing foot lands at the last node at zero speed, on the next foothold
        eq.append([P[-1, 2, 0] - (hx + L), P[-1, 2, 1], P[-1, 3, 1]])
        for d in forces:
            for fx, fz in d.values():
                ineq.append([fz, MU * fz - fx, MU * fz + fx])
        ineq.append(envelope_rows(Tq, V, u) / 100.0)
        return np.concatenate([np.ravel(e) for e in eq]), np.concatenate([np.ravel(i) for i in ineq])

    def bounds(self):
        b = []
        for k in range(self.N):
            b += [(None, None), (-0.6, 0.2), (-0.6, 0.6)]
            for leg in range(2):
                b += list(zip(LO, HI))
        b += [(-3.0, 3.0)] * (self.iD - self.iF)
        b += [(0.008, 0.03), (0.008, 0.04), (-1.0, 1.0), (0.05, 5.0)]
        b[0] = (0.0, 0.0)                                       # x starts at 0
        return b

    def guess(self):
        """from run_check's template: one step of its stride, the right leg's stance"""
        f = 3.6 if self.v < 4 else 3.9
        tc = 0.14 if self.v < 4 else 0.12
        qs, st, dt, _ = R.task_run(self.v, f=f, tc=tc, crouch=0.06, steps=4)
        n = int(round(1 / f / dt))
        # the step whose stance is R: in task_run, step k's stance leg is L for even k
        k0 = 1 * n
        t_td = k0 + n - int(round(tc / 2 / dt))                 # R touches down
        t_lo = t_td + int(round(tc / dt))
        t_end = t_td + n
        idx = list(np.linspace(t_td, t_lo, self.ns).round().astype(int)) + \
            list(np.linspace(t_lo, t_end, self.nfl + 2)[1:-1].round().astype(int))
        x = np.zeros(self.n)
        for j, i in enumerate(idx):
            q = qs[i]
            pitch = 2 * math.atan2(q[5], q[3])
            row = [q[0], q[2] - Z0, pitch,
                   q[R.QA["hip_R_pitch"]], q[R.QA["knee_R"]], q[R.QA["ankle_R_pitch"]],
                   q[R.QA["hip_L_pitch"]], q[R.QA["knee_L"]], q[R.QA["ankle_L_pitch"]]]
            x[j * 9:(j + 1) * 9] = row
        x[:self.N * 9] -= np.tile([x[0], 0, 0, 0, 0, 0, 0, 0, 0], self.N)
        x[self.iD:self.iD + 4] = [tc / (self.ns - 1), (1 / f - tc) / (self.nfl + 1), 0.0, 3.0]
        seed_forces(self, x)
        Q = x[:self.N * 9].reshape(self.N, 9)
        PD.qpos[:] = to_model(Q[0]); mujoco.mj_kinematics(PM, PD)
        x[self.iD + 2] = PD.site_xpos[SITES[("R", "heel")]][0]
        return np.clip(x, [b[0] if b[0] is not None else -np.inf for b in self.bounds()],
                       [b[1] if b[1] is not None else np.inf for b in self.bounds()])


class Jump:
    """a squat jump's push-off on both legs together, from rest to take-off at the speed that
    flies the centre of mass up h: feet flat, then on the toes; the last node leaves the
    ground (no force), the centre of mass going straight up, the trunk barely turning"""

    def __init__(self, h, n_flat=12, n_toe=6):
        self.h, self.nf, self.nt = h, n_flat, n_toe
        self.N = n_flat + n_toe + 1
        self.iF = self.N * 6
        self.iD = self.iF + n_flat * 4 + n_toe * 2
        self.n = self.iD + 3                       # dt, heel x, u
        self.v0 = math.sqrt(2 * R.G * h)

    def full(self, q6):
        return np.array([q6[0], q6[1], q6[2], q6[3], q6[4], q6[5], q6[3], q6[4], q6[5]])

    def kin(self, x):
        N = self.N
        Q6 = x[:N * 6].reshape(N, 6)
        Q = np.array([self.full(q) for q in Q6])
        dt, hx, u = x[self.iD:self.iD + 3]
        Qm = np.vstack([Q[:1], Q[:-1]])           # at rest before the first node
        Qp = np.vstack([Q[1:], 2 * Q[-1] - Q[-2]])
        h = np.full(N, dt)
        V, A = fd(Q, h, h, Qm, Qp)
        # the take-off node: one-sided, second order
        V[-1] = (3 * Q[-1] - 4 * Q[-2] + Q[-3]) / (2 * dt)
        A[-1] = (Q[-1] - 2 * Q[-2] + Q[-3]) / dt ** 2
        f = x[self.iF:self.iD]
        forces, j = [], 0
        for k in range(N):
            d = {}
            if k < self.nf:
                for sd in "LR":
                    d[(sd, "heel")] = (f[j], f[j + 1]); d[(sd, "toe")] = (f[j + 2], f[j + 3])
                j += 4
            elif k < self.nf + self.nt:
                for sd in "LR":
                    d[(sd, "toe")] = (f[j], f[j + 1])
                j += 2
            forces.append(d)
        return Q, V, A, forces, dt, hx, u

    def com(self, q):
        PD.qpos[:] = to_model(q)
        mujoco.mj_kinematics(PM, PD); mujoco.mj_comPos(PM, PD)
        return PD.subtree_com[0].copy()

    def cons(self, x):
        Q, V, A, forces, dt, hx, u = self.kin(x)
        Tq, P = dynamics(Q, V, A, forces)
        eq = [Tq[:, :3].ravel() / (R.D["mass"] * R.G)]
        ineq = []
        for k in range(self.N):
            if k < self.nf:
                eq.append([P[k, 0, 0] - hx, P[k, 0, 1], P[k, 1, 1]])
            else:
                eq.append([P[k, 1, 0] - hx - FOOT, P[k, 1, 1]])
                ineq.append([P[k, 0, 1]])
        c1, c2, c3 = self.com(Q[-1]), self.com(Q[-2]), self.com(Q[-3])
        vc = (3 * c1 - 4 * c2 + c3) / (2 * dt)
        eq.append([(vc[2] - self.v0) / self.v0, vc[0] / self.v0])
        ineq.append([1.0 - abs(V[-1, 2])])                     # the trunk turns < 1 rad/s at take-off
        for d in forces:
            for fx, fz in d.values():
                ineq.append([fz, MU * fz - fx, MU * fz + fx])
        ineq.append(envelope_rows(Tq, V, u, legs=(0,)) / 100.0)
        return np.concatenate([np.ravel(e) for e in eq]), np.concatenate([np.ravel(i) for i in ineq])

    def bounds(self):
        b = []
        for k in range(self.N):
            b += [(-0.3, 0.3), (-0.6, 0.2), (-0.2, 0.9)] + list(zip(LO, HI))
        b += [(-3.0, 3.0)] * (self.iD - self.iF)
        b += [(0.008, 0.04), (-0.3, 0.3), (0.05, 5.0)]
        return b

    def guess(self):
        """a squat at rest, extending to the toes, the forces sharing the weight"""
        x = np.zeros(self.n)
        for k in range(self.N):
            s = k / (self.N - 1)
            knee = math.radians(100 - 85 * s)
            hip = -math.radians(70 - 60 * s)
            ank = -math.radians(40 - 30 * s) + (math.radians(30) * (s - 0.65) / 0.35 if s > 0.65 else 0)
            z = -0.40 + 0.30 * s
            x[k * 6:(k + 1) * 6] = [0.0, z, math.radians(30 * (1 - s)), hip, knee, ank]
        x[self.iD:self.iD + 3] = [0.018, -R.D["heel"], 2.0]
        seed_forces(self, x)
        return x


def seed_forces(prob, x):
    """each stance node's force variables: the least-squares ground forces that make its
    base rows balance, the vertical ones kept pushing"""
    Q, V, A, forces, *_ = prob.kin(x)
    jp = np.zeros((3, PM.nv))
    j = prob.iF
    for k in range(prob.N):
        keys = list(forces[k])
        if not keys:
            continue
        PD.qpos[:] = to_model(Q[k]); PD.qvel[:] = 0; PD.qvel[PERM] = V[k]
        PD.qacc[:] = 0; PD.qacc[PERM] = A[k]
        mujoco.mj_inverse(PM, PD)
        cols = []
        for key in keys:
            mujoco.mj_jacSite(PM, PD, jp, None, SITES[key])
            cols += [jp[0][PERM][:3] * WT, jp[2][PERM][:3] * WT]
        Acol = np.array(cols).T
        sol = np.linalg.lstsq(Acol, PD.qfrc_inverse[PERM][:3], rcond=None)[0]
        sol[1::2] = np.maximum(sol[1::2], 0.05)
        # the variables are laid out per node as the problem's unpack expects (one foot's
        # points for Run; one foot's points, shared by both feet, for Jump)
        n = len(sol) if isinstance(prob, Run) else len(sol) // 2
        x[j:j + n] = sol[:n] * (1 if isinstance(prob, Run) else 2)
        j += n


def solve(prob, x0, iters=3000, label=""):
    """two stages with scipy's interior-point trust-constr: (1) any consistent motion
    (dynamics, contacts) with the envelope loose (u fixed at 3), smallest torques;
    (2) from there, the smallest u"""
    from scipy.optimize import NonlinearConstraint, Bounds
    cache = {}
    def both(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear(); cache[key] = prob.cons(x)
        return cache[key]
    bl = np.array([b[0] if b[0] is not None else -np.inf for b in prob.bounds()])
    bu = np.array([b[1] if b[1] is not None else np.inf for b in prob.bounds()])
    t0 = time.time()
    neq = both(x0)[0].size
    cons = [NonlinearConstraint(lambda x: both(x)[0], 0.0, 0.0),
            NonlinearConstraint(lambda x: both(x)[1], 0.0, np.inf)]
    x = np.clip(x0, bl, bu)
    # stage 1: u pinned at 3, torques small
    bl1, bu1 = bl.copy(), bu.copy(); bl1[-1] = bu1[-1] = 3.0; x[-1] = 3.0
    def torque_cost(x):
        Q, V, A, forces = prob.kin(x)[:4]
        Tq, _ = dynamics(Q, V, A, forces)
        return float(np.mean((Tq[:, 3:] / 300.0) ** 2))
    r1 = minimize(torque_cost, x, method="trust-constr", bounds=Bounds(bl1, bu1), constraints=cons,
                  options=dict(maxiter=iters, verbose=0, gtol=1e-6, xtol=1e-9))
    eq, iq = prob.cons(r1.x)
    print("%s stage 1 (consistent motion): %s, %d iterations, %.0f s, |eq| %.1e, min ineq %.1e"
          % (label, r1.status, r1.nit, time.time() - t0, np.abs(eq).max(), iq.min()), flush=True)
    r = minimize(lambda x: x[-1], r1.x, method="trust-constr", bounds=Bounds(bl, bu), constraints=cons,
                 options=dict(maxiter=iters, verbose=0, gtol=1e-6, xtol=1e-9))
    eq, iq = prob.cons(r.x)
    print("%s stage 2: status %s, %d iterations, %.0f s; worst envelope use u = %.2f; |eq| %.1e, min ineq %.1e"
          % (label, r.status, r.nit, time.time() - t0, r.x[-1], np.abs(eq).max(), iq.min()), flush=True)
    return r


def report_run(prob, x):
    Q, V, A, forces, h, T, L, hx, u = prob.kin(x)
    Tq, P = dynamics(Q, V, A, forces)
    tc = h[:prob.ns - 1].sum()
    print("   step %.3f s (contact %.0f ms, flight %.0f ms), step length %.2f m, trunk pitch %.0f..%.0f deg"
          % (T, 1000 * tc, 1000 * (T - tc), L, *np.degrees([Q[:, 2].min(), Q[:, 2].max()])))
    for leg in (0, 1):
        for name, key, t, w in motor_loads(Tq, V, leg):
            av = np.array([R.avail(key, s) for s in w])
            use = np.abs(t) / np.maximum(av, 1e-6)
            i = int(np.argmax(use))
            print("   %s %-5s %-15s peak %6.1f N.m, %5.1f rad/s; worst %3.0f %% of the envelope (%.0f N.m at %.1f rad/s)"
                  % ("stance" if leg == 0 else "swing ", name, key, np.abs(t).max(), np.abs(w).max(),
                     100 * use[i], abs(t[i]), abs(w[i])))
    for leg in (0, 1):
        print("   %s angles: hip %4.0f..%4.0f  knee %4.0f..%4.0f  ankle %4.0f..%4.0f deg"
              % (("stance", "swing ")[leg], *np.degrees([Q[:, 3 + 3 * leg].min(), Q[:, 3 + 3 * leg].max(),
                                                         Q[:, 4 + 3 * leg].min(), Q[:, 4 + 3 * leg].max(),
                                                         Q[:, 5 + 3 * leg].min(), Q[:, 5 + 3 * leg].max()])))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what == "run":
        v = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
        p = Run(v)
        x0 = p.guess()
        print("running %.1f m/s: %d variables, envelopes at %.0f V" % (v, p.n, R.VBUS), flush=True)
        r = solve(p, x0, label="run %.1f" % v)
        report_run(p, r.x)
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "urdf", "run_%.1f.npy" % v), r.x)
    elif what == "jump":
        for h in [float(a) for a in sys.argv[2:]] or [0.2, 0.3, 0.4]:
            p = Jump(h)
            r = solve(p, p.guess(), label="jump %.2f m" % h)
            Q, V, A, forces, dt, hx, u = p.kin(r.x)
            Tq, P = dynamics(Q, V, A, forces)
            print("   push %.0f ms; squat: hip %.0f knee %.0f ankle %.0f deg at the start"
                  % (1000 * dt * (p.N - 1), *np.degrees(Q[0, 3:6])))
            for name, key, t, w in motor_loads(Tq, V, 0):
                av = np.array([R.avail(key, s_) for s_ in w]); use = np.abs(t) / np.maximum(av, 1e-6); i = int(np.argmax(use))
                print("   %-5s %-15s peak %6.1f N.m, %5.1f rad/s; worst %3.0f %% (%.0f N.m at %.1f rad/s)"
                      % (name, key, np.abs(t).max(), np.abs(w).max(), 100 * use[i], abs(t[i]), abs(w[i])))
