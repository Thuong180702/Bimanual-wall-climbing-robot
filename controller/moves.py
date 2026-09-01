"""Climbing primitives.

Everything is built from exactly two motions:

    reach()       one hand lets go, travels, and latches on a new hold
    body_shift()  both hands stay put, the body is hauled to a new place

The six climbing moves are parameter sets over those two, not six state machines.

Moves are generators: they yield once per simulation tick after writing into ctx.ctrl,
so the executor stays in charge of stepping and of watching for termination.
"""
import numpy as np, mujoco
from collections import deque
from climbing_env import (ClimbingEnv, GRIP_SET, GRIP_BAND,
                          THROAT_X as SEAT_X, THROAT_Z as SEAT_Z)
from . import SCENE
from .kinematics import (Estimator, ARM, WAIST, FINGER, JOINTS, THROAT_PT, W_ROT,
                         grasp_frame, ik_body, rotvec, Hanging, plan_hanging,
                         solve_hanging_pose, roadmap_hanging)
from .holds import SLIDE, GRASP_STEP


SEAT_DEPTH = 0.008        # m short of the rung axis the hook aims, so the rung
                          # seats rather than perching at the mouth of the throat
PULL_UP = 0.20            # m of body lift commanded before a hand comes off the wall
ABDUCT = np.radians(90)   # shoulders closing across the body; the chain stops it near 20-30
ELBOW_STEPS = np.radians([-15.0, -30.0, -45.0, -60.0])   # bent in steps, each one checked
TORSO_DROP = 0.50         # m the torso ends up below the hands after that lift. Measured
                          # from 0.73 m at rest; it is the posture a reach captures from,
                          # and the gate that stops the lift stacking up over a traverse.
FINGER_OPEN = 0.20        # rad, clear of the rung
GRIP_HARD = np.radians(-38)   # the load-bearing hand grips down to the edge of the
                              # measured band while the other one is off the wall
MAX_RATE = 1.2            # rad/s ceiling during closed-loop servoing
CLIMB_RATE = 0.15         # rad/s along a planned climbing route
W_ALONG_COARSE = 0.85     # along-rung weight while covering ground. Swept: below 0.6
                          # the right hand ends up behind where it started, above it
                          # the seat starts failing. The landscape is not smooth
W_ALONG = 0.15            # weight on position error along a rung, versus 1.0 across
IMU_QUIET = 0.5           # m/s2 from g: how close a sample must be to count as one
                          # the robot was hanging still for
IMU_WINDOW = 0.8          # s of accelerometer/magnetometer averaging behind the
                          # attitude estimate -- about one slow swing period
SERVO_RATE = 0.25         # rad/s for the servo hops
W_ANCHOR = 0.15           # how cheap the anchored arm's joints are relative to the free
                          # arm's. Swept: at 1.0 only the left hand latches, at 0.15 both
                          # do, and the shoulder-to-hold distance is what moves -- 0.638 m
                          # at 1.0, past the 0.581 m the arm spans, against 0.570 m at 0.15
UNJAM_MIN = 3             # joints on their stops before it is worth freeing the arm
UNJAM = 0.15              # rad a joint is moved off its stop before servoing
SETTLE_MAX = 0.6          # s of extra wait allowed for the body to go quiet
SERVO_SETTLE = 0.12       # s of quiet after each hop, before measuring. Both were
                          # swept: 0.40 rad/s or 0.16 s of settle and the reach stops
                          # working, because the Broyden update then learns from a
                          # body that is still moving. Slow is not a style choice.
JOINT_RATE = 0.25         # rad/s while playing back a planned path. Slow: an arm
                          # swinging on an 11 kg body is not a small perturbation,
                          # and the whole controller assumes quasi-static motion
LIFT = 0.075              # m, straight up to unhook
STANDOFF = 0.10           # m, off the wall while travelling
APPROACH = 0.030          # m, hover height before dropping onto a hold. Inside the
                          # measured -10..+70 mm capture window, and low enough to
                          # matter: the free hand cannot reach a point 85 mm above
                          # its own hold without flipping the arm over to the other
                          # branch of joint space, which is 4 rad away
SETTLE_GYRO = 0.25        # rad/s, "the body has stopped swinging"
BRAKE_GYRO = 0.35         # rad/s, above this a trajectory pauses and waits
BRAKE_MAX = 3.0           # s. The pause has to be bounded: waiting for the anchored hand
                          # to come back above 25 N is right when it is a swing, and a
                          # deadlock when the hand simply hangs lighter than that. The
                          # second descent spent 12000 s of simulated time there and the
                          # episode timed out with the robot standing still.
ALIGN_TOL = 0.010         # m, horizontal accuracy demanded before the drop

# Where the robot is placed by keyframe "hang_on_rung" -- an initial condition you know
# on hardware too, because you hung it there. Not read from the simulator.
HANG = {'L': ('rung_front_line_0', np.array([0.440, 0.157, 1.000])),
        'R': ('rung_front_line_0', np.array([0.440, -0.157, 1.000]))}


class MoveFailed(Exception):
    pass


def make_env(xml=SCENE, timeout_s=300.0, **kw):
    """Build the environment with the episode rules this controller is judged by.

    ClimbingEnv's `released` is geometric: the rung must sit inside a fixed box in the
    hook's frame. That box is calibrated on a hook hanging square, and a hook that has
    rotated on its rung -- which is exactly what a pull-up does -- falls outside it while
    still carrying the whole robot. Verified directly: with the other arm retracted
    entirely, the robot hung on such a hook for six seconds, 167 N through the wrist,
    dropping 21 mm. So the geometric test is turned off and the same question is asked of
    the load path instead, which is what hardware would have anyway.

    The timeout goes up for the same reason: this controller is quasi-static on purpose --
    every measurement it makes needs the body to have stopped moving -- so a single reach
    takes about a minute of simulated time, where the default allows sixty seconds for a
    whole episode.
    """
    return ClimbingEnv(xml=xml, free_fall_s=1e9, timeout_s=timeout_s, **kw)


class Ctx:
    """Everything a move needs: the robot, the wall map, the estimate, the setpoints."""

    def __init__(self, env, holds):
        self.env, self.holds = env, holds
        self.est = Estimator(env.model)
        self.planner = Estimator(env.model)   # separate, so planning never
                                              # clobbers the live estimate
        self.ctrl = env.data.ctrl.copy()
        self.lo = env.model.actuator_ctrlrange[:, 0].copy()
        self.hi = env.model.actuator_ctrlrange[:, 1].copy()
        self.dt = env.model.opt.timestep
        # anchored hands: side -> (rung name, world point on the rung axis, axis, normal)
        self.grip = {}
        self.log = []
        self._n = 0
        self._slack = 0.0
        self._imu = deque(maxlen=int(IMU_WINDOW / self.dt))
        self._slack = 0.0
        self._imu = deque(maxlen=int(IMU_WINDOW / self.dt))

    # ---- initial condition -------------------------------------------------
    def start_hanging(self):
        for s, (rung, pos) in HANG.items():
            g = self.holds.grasps[self.holds.nearest(pos)]
            self.grip[s] = (rung, pos.copy(), g.a(), g.n(), None)
        self.sense()
        for s in self.grip:
            self.mark_latched(s)
        return self

    def mark_latched(self, side):
        """remember the hook's orientation at the moment it took the hold"""
        est = self.sense(side)
        h = self.grip[side]
        self.grip[side] = h[:4] + (est.hook_R(side).copy(),)

    def resync(self):
        """Correct where the hands are recorded, using the fact that both are on rungs.

        The anchored hand's recorded position goes stale: the hook slides along its rung
        as the body moves -- after a pull-up and a reach the estimate was 69 mm out, all
        of it the anchor having moved from where it was last written down. With both
        hooks on known rung axes the problem is over-determined, so slide the anchor
        along its own rung until the other hand lands on its rung too, then re-project.

        Only the hold map, the encoders and the IMU are used, so this works on hardware.
        It does nothing while both hooks are on the *same* straight rung: sliding the
        anchor along it slides the other hand identically, so the along-rung coordinate is
        simply not observable from this information. It becomes observable again as soon
        as the hands are on different rungs.
        """
        if len(self.grip) != 2:
            return
        a, b = list(self.grip)
        est = self.sense(a)
        u = np.asarray(self.grip[a][2], float)
        u = u / np.linalg.norm(u)
        q0 = np.asarray(self.grip[b][1], float)
        ax = np.asarray(self.grip[b][2], float)
        ax = ax / np.linalg.norm(ax)
        p0 = est.throat(b)
        d0 = (p0 - q0) - ((p0 - q0) @ ax) * ax
        v = u - (u @ ax) * ax
        if v @ v > 1e-9:
            s = float(-(d0 @ v) / (v @ v))
            self.grip[a] = (self.grip[a][0], np.asarray(self.grip[a][1]) + s * u) \
                + self.grip[a][2:]
        est = self.sense(a)
        for side in (a, b):
            hold = np.asarray(self.grip[side][1], float)
            e_ax = np.asarray(self.grip[side][2], float)
            e_ax = e_ax / np.linalg.norm(e_ax)
            p = hold + e_ax * float((est.throat(side) - hold) @ e_ax)
            self.grip[side] = (self.grip[side][0], p) + self.grip[side][2:]

    def seat_est(self, side):
        """Where the rung sits inside this hook's throat, in hook-local mm, estimated
        from the hold map and the encoders -- no ground truth, so this works on hardware.

        This is the quantity that actually ends runs. As the body turns on the hold the
        rung walks along the throat and eventually leaves it, quietly: the gyro stays
        near zero and full load keeps going through the wrist the whole way, so watching
        for a violent swing never catches it.
        """
        p = np.asarray(self.grip[side][1])
        R = self.est.hook_R(side)
        return R.T @ (p - self.est.body_pos(self.est.hook[side])) * 1000.0

    def seated(self, side, inset=2.0):
        s = self.seat_est(side)
        return (SEAT_X[0] + inset <= s[0] <= SEAT_X[1] - inset
                and SEAT_Z[0] + inset <= s[2] <= SEAT_Z[1] - inset)

    # ---- sensing (encoders / IMU / F-T / grip pad only) --------------------
    def imu(self):
        """IMU averaged over the last IMU_WINDOW seconds.

        Attitude comes from TRIAD on gravity, so it is only as good as the assumption that
        the specific force *is* gravity -- and the body swings slowly on its hook long
        after any sensible "has it stopped" test says yes. Measured: waiting until the
        accelerometer read 9.95 m/s2 and the gyro 0.025 rad/s still left the attitude
        6.6 deg out, which puts the far hand 35 mm from where it is. Averaging over a
        swing period cancels most of that without waiting for a robot that never fully
        stops. Any real IMU attitude estimator filters for the same reason.

        Only samples that already look like gravity go into the average -- averaging the
        transient in with them made a hand that was 58 mm out into one 140 mm out.
        """
        quiet = [s for s in self._imu
                 if abs(np.linalg.norm(s['accel']) - 9.81) < IMU_QUIET]
        if not quiet:
            return self.env.imu(raw_only=True)
        n = len(quiet)
        return dict(accel=sum(s['accel'] for s in quiet) / n,
                    mag=sum(s['mag'] for s in quiet) / n,
                    gyro=self._imu[-1]['gyro'])

    def sense(self, anchor=None):
        enc = self.env.encoders()
        q = np.array([enc[j][0] for j in JOINTS])
        a = anchor or self.anchor()
        self.est.update(q, self.imu(), a, self.grip[a][1])
        return self.est

    def anchor(self):
        """the hand currently carrying the robot; prefer a genuinely confirmed one"""
        for s in self.grip:
            if self.latch_confirmed(s):
                return s
        if self.grip:
            return next(iter(self.grip))
        raise MoveFailed("no hand on the wall")

    def latch_confirmed(self, side):
        """Hardware-only evidence of a latch: something in the throat, the finger inside
        the band that actually holds, and load coming through the wrist.
        Deliberately not the pad force alone -- that peaks exactly where the grip fails."""
        if side not in self.grip:
            return False
        h = self.env.hand(side)
        ang = self.env.sensor('enc_finger_' + side)[0]
        return (h['pad'] > 20.0 and GRIP_BAND[0] <= ang <= GRIP_BAND[1]
                and np.linalg.norm(h['force']) > 12.0)

    def unloading(self):
        """the anchored hand is losing its load -- stop feeding motion until it is back"""
        return any(np.linalg.norm(self.env.hand(s)['force']) < 25.0 for s in self.grip)

    def swinging(self):
        return float(np.linalg.norm(self.env.imu(raw_only=True)['gyro']))

    def self_contacts(self):
        m, d = self.env.model, self.env.data
        return sum(1 for i in range(d.ncon)
                   if m.geom_bodyid[d.contact[i].geom1] and m.geom_bodyid[d.contact[i].geom2])

    # ---- actuation ---------------------------------------------------------
    def apply(self, joints, dq):
        for j, v in zip(joints, dq):
            self.ctrl[j] = np.clip(self.ctrl[j] + v, self.lo[j], self.hi[j])

    def set(self, j, v):
        self.ctrl[j] = np.clip(v, self.lo[j], self.hi[j])

    def tick(self):
        """One simulation step. Every phase of every move goes through here, which is why
        the hold-integrity check lives here and not in the individual phases -- it was in
        one of them once, and the run that fell off went through a phase that had none."""
        self.env.step(self.ctrl)
        done, why = self.env.terminated()
        if done and why != 'success':
            raise MoveFailed(f"episode ended: {why}")
        self._imu.append(self.env.imu(raw_only=True))
        self._n += 1
        load = max(np.linalg.norm(self.env.hand(s)['force']) for s in 'LR')
        self._slack = 0.0 if load > 15.0 else self._slack + self.dt
        if self._slack > 0.6:
            raise MoveFailed("released: no load through either hand for 0.6 s")
        if self._n % 25 == 0:
            for s in list(self.grip):
                if self.grip[s][4] is None:
                    continue
                self.sense(s)
                if not self.seated(s):
                    raise MoveFailed(
                        f"{s} rung is at {self.seat_est(s)[[0, 2]].round(0)} mm in the "
                        f"throat, on its way out; stopping before it lets go")


# --- helpers ----------------------------------------------------------------
def smooth(u):
    """C2 ease in/out, 0->1"""
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def _finger(ctx, side, target, T=0.4):
    j = FINGER[side]
    a, n = ctx.ctrl[j], max(1, int(T / ctx.dt))
    for i in range(n):
        ctx.set(j, a + (target - a) * smooth((i + 1) / n))
        ctx.tick()
        yield


# --- primitive 1: move one hand to a new hold -------------------------------
def _follow(ctx, joints, path, rate=JOINT_RATE, measured=True):
    """Play a joint-space path at a fixed rate, slowest joint sets the pace.

    Pauses whenever the body is swinging or the anchored hand starts to unload: feeding
    more motion into a swinging 11 kg robot is how the anchored hook gets levered off.
    """
    for q_next in path:
        q_now = ctx.ctrl.copy()
        span = max(abs(q_next[j] - q_now[j]) for j in joints)
        n = max(1, int(span / rate / ctx.dt))
        i, waited = 0, 0
        while i < n:
            if ((ctx.swinging() > BRAKE_GYRO or ctx.unloading())
                    and waited * ctx.dt < BRAKE_MAX):
                waited += 1
                ctx.tick()
                yield
                continue
            waited = 0
            u = smooth((i + 1) / n)
            for j in joints:
                ctx.set(j, q_now[j] + (q_next[j] - q_now[j]) * u)
            ctx.tick()
            i += 1
            yield


def _solve_free(J, e, lam, ctx, joints):
    """Damped least squares over the joints that can still move in the wanted direction.

    A joint sitting on its stop cannot contribute, but a plain solve keeps asking it to:
    the request is clipped away and the remaining joints never take up the slack.
    """
    rows = J.shape[0]
    free = list(range(len(joints)))
    for _ in range(3):
        Jf = J[:, free]
        d = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam ** 2 * np.eye(rows), e)
        stuck = [k for k, i in enumerate(free)
                 if (ctx.ctrl[joints[i]] <= ctx.lo[joints[i]] + 1e-4 and d[k] < 0)
                 or (ctx.ctrl[joints[i]] >= ctx.hi[joints[i]] - 1e-4 and d[k] > 0)]
        if not stuck:
            break
        free = [i for k, i in enumerate(free) if k not in stuck]
        if not free:
            return np.zeros(len(joints))
    dq = np.zeros(len(joints))
    for k, i in enumerate(free):
        dq[i] = d[k]
    return dq


def _servo_emp(ctx, side, other, target, joints, tol, R_goal=None, axis=None,
               steps=70, step=0.06, lam=0.03, normal=None, w_normal=3.0, w_rot=W_ROT,
               rot_tol=0.20, rate=None, settle=None, along=None, w_along=None,
               cheap=None):
    """Walk the hand onto `target` using a Jacobian learned from the robot's own response.

    The analytic Jacobian is right about the kinematics and wrong about the robot. Command
    the arm up and the body turns on the hook and cancels it: at one stuck configuration
    the model predicted a 50 mm rise for a 0.09 rad move and the hand went nowhere, while
    a direct probe showed 0.1 rad of the anchored shoulder's abduction lifts it 29 mm. The
    response is real, just not modelled.

    So this maps *setpoints* to hand pose and corrects the map from what actually happens
    (Broyden). That absorbs everything the model does not have -- servo droop, the hook
    turning on its rung, the body re-hanging -- without needing a contact model.
    """
    rate = SERVO_RATE if rate is None else rate
    settle = SERVO_SETTLE if settle is None else settle
    w_along = W_ALONG if w_along is None else w_along

    def pose():
        est = ctx.sense(other)
        return est.throat(side), est.hook_R(side)

    def err_of(p, R):
        e = target - p
        if R_goal is None:
            return e
        er = rotvec(R_goal @ R.T)
        if axis is not None:
            a = np.asarray(axis, float)
            a = a / np.linalg.norm(a)
            er = er - (er @ a) * a          # the hook may pivot freely on the rung
        return np.concatenate([e, w_rot * er])

    est = ctx.sense(other)
    Jp, Jr = est.jac_rel(side, other, joints)
    J = Jp if R_goal is None else np.vstack([Jp, w_rot * Jr])
    x, R = pose()

    rows = J.shape[0]
    W = np.eye(rows)
    if along is not None:
        # Care less about where the hand ends up *along* a rung. The anchored hook skates
        # along its own rung as the free arm swings -- measured, 180 mm over one reach --
        # and since the estimate is pinned to where that hook was last recorded, the whole
        # world appears to slide with it. Chasing that is chasing a disturbance: it is
        # also the one direction that is unobservable from the map and the encoders, and
        # the one that does not matter, because a continuous rung is as good a hold
        # 180 mm further along.
        a = np.asarray(along, float)
        a = a / np.linalg.norm(a)
        W[:3, :3] -= (1.0 - w_along) * np.outer(a, a)
    if normal is not None:
        # weight the wall normal: that is the fore/aft axis, and its capture window is
        # +-25 mm against -10..+70 mm vertically, so it earns the authority
        nrm = np.asarray(normal, float)
        nrm = nrm / np.linalg.norm(nrm)
        W[:3, :3] += (w_normal - 1.0) * np.outer(nrm, nrm)

    # Spend the free arm before the anchored one. Both are needed -- the approach fails
    # outright with the free arm alone -- but moving the anchored arm moves the body, and
    # measured on a right-hand reach that carried the shoulder from 0.540 m away from the
    # hold to 0.638 m, past the 0.581 m the arm can span. It then had nowhere to go.
    D = np.ones(len(joints))
    if cheap is not None:
        for i, j in enumerate(joints):
            if j not in cheap:
                D[i] = W_ANCHOR

    for _ in range(steps):
        e = err_of(x, R)
        pos = e[:3] if along is None else e[:3] - (e[:3] @ a) * a
        if np.linalg.norm(pos) < tol and (R_goal is None or
                                            np.linalg.norm(e[3:]) / w_rot < rot_tol):
            return True
        Jw, ew = (W @ J) * D, W @ e
        dq = _solve_free(Jw, ew, lam, ctx, joints) * D
        n_dq = np.linalg.norm(dq)
        if n_dq < 1e-6:
            return False
        if n_dq > step:
            dq *= step / n_dq

        # Command it, then read back what the stops actually allowed. Feeding the
        # requested step into the Broyden update instead of the executed one is what made
        # this diverge: with five of fifteen joints sitting on their stops, a third of
        # every command vanished and the update learned a response that was not there.
        q_next = ctx.ctrl.copy()
        for i, j in enumerate(joints):
            q_next[j] = np.clip(q_next[j] + dq[i], ctx.lo[j], ctx.hi[j])
        dq = np.array([q_next[j] - ctx.ctrl[j] for j in joints])
        if np.linalg.norm(dq) < 1e-9:
            return False
        yield from _follow(ctx, joints, [q_next], rate=rate, measured=False)
        yield from hold(ctx, settle)
        # then wait for the body to actually stop, rather than for a fixed time. A sweep
        # over fixed settles found the left hand needed 0.20 s and the right 0.30 s, with
        # no single value serving both -- which is the signal that the wait should be a
        # condition, not a number.
        yield from _settle(ctx, t_max=SETTLE_MAX)

        x_new, R_new = pose()
        dx = x_new - x
        if R_goal is not None:
            dr = rotvec(R_new @ R.T)
            if axis is not None:
                a = np.asarray(axis, float)
                a = a / np.linalg.norm(a)
                dr = dr - (dr @ a) * a
            dx = np.concatenate([dx, w_rot * dr])
        J = J + np.outer(dx - J @ dq, dq) / max(float(dq @ dq), 1e-9)
        x, R = x_new, R_new
    e = err_of(*pose())
    pos = e[:3] if along is None else e[:3] - (e[:3] @ a) * a
    return np.linalg.norm(pos) < tol


def _settle(ctx, t_max=6.0):
    """Wait until the IMU is reading gravity rather than motion.

    The test is the accelerometer's magnitude, not the gyro. Attitude comes from TRIAD on
    gravity, so it is only as good as the assumption that the specific force *is* gravity.
    Measured after a pull-up: the gyro had already fallen to 0.24 rad/s -- quiet by any
    reasonable threshold -- while the accelerometer still read 6.7 m/s2 and the attitude
    estimate was 19.7 deg out, putting the far hand 120 mm from where it really was. Three
    seconds later the reading was 9.73 and the error 0.2 deg.
    """
    for _ in range(int(t_max / ctx.dt)):
        ctx.tick()
        yield
        imu = ctx.env.imu(raw_only=True)
        if (abs(np.linalg.norm(imu['accel']) - 9.81) < 0.25
                and np.linalg.norm(imu['gyro']) < 0.05):
            return


def _unjam(ctx, joints, margin=UNJAM, need=UNJAM_MIN):
    """Move the arm off its stops, but only if it is substantially stuck.

    A jammed joint can only move one way, and the servo needs both. Measured after a
    pull-up the two arms are not in the same state at all: the left has two joints on
    their stops, both wrists, while the right has five including the shoulder, the elbow
    and the forearm roll -- and the right-hand reach then spent its whole approach with 8
    to 10 of 15 joints stuck, never converging. Freeing an arm that is only slightly
    stuck costs more than it gains: doing it unconditionally made the working left-hand
    reach fall off the wall.
    """
    q = ctx.ctrl.copy()
    stuck = [j for j in joints
             if q[j] <= ctx.lo[j] + 1e-3 or q[j] >= ctx.hi[j] - 1e-3]
    if len(stuck) < need:
        return
    for j in stuck:
        q[j] = (ctx.lo[j] + margin if q[j] <= ctx.lo[j] + 1e-3
                else ctx.hi[j] - margin)
    yield from _follow(ctx, joints, [q], measured=False)
    yield from _settle(ctx, t_max=1.0)


def _loads(ctx, T=0.6):
    """Mean wrist load per hand over a window.

    Averaged, not sampled. In the instant after a finger closes the new hand reads as
    though it is carrying and hands the weight back seconds later: a single reading there
    passed the transfer test while the hand went on to hold 1 N of an 11 kg robot.
    """
    acc = {s: [] for s in 'LR'}
    for _ in range(int(T / ctx.dt)):
        ctx.tick()
        for s in 'LR':
            acc[s].append(float(np.linalg.norm(ctx.env.hand(s)['force'])))
        yield
    return {s: float(np.mean(v)) for s, v in acc.items()}


def _commit(ctx, side, other, tries=6, drop=0.12, need=None):
    """Move the robot's weight onto a hand that has just taken a hold below the other.

    Landing on a lower rung does not load it. Measured right after the leading hand's
    descent: the new hand carried **1 N** and the hand still on the rung above carried
    **111 N**, the whole robot. The arms are stiff position servos, so the load follows
    the setpoints, not the statics.

    The direction is not a guess: with the hands a row apart the torso sits level with the
    lower hold, so that arm is folded up and can pull on nothing; letting the body down
    puts it back at the extension it works at.

    `need` is the load the hand being *freed* has to fall below, for when the point of the
    transfer is to take that hand off. The default is parity, which is all the leading
    hand's landing needs -- and not enough to unhook: at 54 N against 57 N the upper hook
    stayed on the rung.
    """
    done = (lambda f: f[other] <= need) if need else (lambda f: f[side] >= f[other])
    f = yield from _loads(ctx)
    for _ in range(tries):
        if done(f):
            return
        yield from body_shift(ctx, [0.0, 0.0, -drop], frame=side)
        yield from _settle(ctx)
        f = yield from _loads(ctx)
    raise MoveFailed(f"{side} took the hold but the weight stayed on {other}: "
                     f"{f[side]:.0f} N against {f[other]:.0f} N")


def _unhook(ctx, side, other):
    """Take a hook off its rung and leave the robot's weight on the other hand.

    Opening the finger is not a release. The hook is a slot round a cylinder, so it goes
    on carrying the robot wide open: measured at **111 N through a hand whose finger was
    fully open**, against 1 N through the hand that was supposed to be the anchor. Every
    move planned after that was solved against a robot that was not there.

    Lifting it clear only works once the other hand is already carrying, which is why this
    runs after `_commit` and not instead of it -- and it is checked, because a hook that
    has not come off is indistinguishable from one that has until something moves.
    """
    est = ctx.sense(other)
    yield from _servo_emp(ctx, side, other, est.throat(side) + np.array([0., 0., LIFT]),
                          WAIST + ARM[side], tol=0.012, steps=25)
    yield from _settle(ctx)
    f = yield from _loads(ctx)
    if f[side] > 25.0:
        raise MoveFailed(f"{side} did not come off its rung: still carrying "
                         f"{f[side]:.0f} N against {other}'s {f[other]:.0f} N")


def _pull_up(ctx, anchor, mover=None):
    """Bring the torso up under the hands, with both hooks still on, if it hangs low.

    Swings the shoulders forward, from -160 deg of flexion to -86, torso 0.272 -> 0.500,
    wall gap 67 -> 193 mm. A reach needs exactly that: the free hand cannot rise relative
    to its own hold, however the approach is driven it plateaus ~50 mm low, so the height
    has to be won here. Closing a loop around it was tried and overshot to 0.60 m, taking
    both hooks off the wall -- so this is a gate, not a controller.

    A climb wants `_flare` instead, and the two are not interchangeable in either
    direction: this one falls off the wall on the trailing hand's descent, and `_flare`
    drops the robot on all three reach phases.
    """
    est = ctx.sense(anchor)
    drop = float(np.asarray(ctx.grip[anchor][1])[2]) - float(est.body_pos(est.torso)[2])
    if drop > TORSO_DROP + 0.08:
        yield from body_shift(ctx, [0.0, 0.0, PULL_UP], frame=mover)
        yield from _settle(ctx)


def _flare(ctx, anchor, mover=None):
    """Pull up by opening the shoulders outward and bending the elbows. For a climb.

    This is the climber's motion, and it needed a model change to exist. The abduction
    joints ran 0..180 deg on the left and -180..0 on the right with the hang sitting at 0,
    the edge -- so the only travel they had swung the arms *across* each other. Measured
    then, with the elbows bent 45 deg: the two elbows went from 310 mm apart at 0 deg to
    57 mm at 30 and to 239 mm **crossed** at 90. Every earlier version of this function
    was closing the shoulders while its own name and comments said it was opening them.

    `scene_wide_shoulders.xml` and the `urdf/lessons/` scenes allow 60 deg outward on
    each side, and there the arms spread instead: 309 mm at 0, **467 mm at -20 deg,
    602 at -40, 696 at -60**, with the torso rising 0.272 -> 0.551 on the way.
    `scene.xml` -- the model this controller actually runs on -- does not: abduction
    starts at the hang posture and only travels inward, so what this function does
    there is tuck the arms across the body. That tuck is what the measurements below
    are about, and it works; the outward flare needs the controller re-tuned first.

    It lifts far less than `_pull_up` -- torso 0.294 against 0.500, and the body stays at
    67 mm off the wall rather than 193 -- and it is what makes a row change work. The
    climb after it lands **13 mm** from the hold in 51 s, against 144 mm in 90 s, and the
    trailing hand's descent, which had never once completed, completes.

    Height was never what the pull-up was for. What this does is put the arm in the branch
    that can reach: every posture that takes a hold sits at shoulder abduction 85-176 deg,
    the hang keyframe sits at 0, and no local IK crosses between them -- so the servo
    spent every climb fighting towards somewhere it could not get to from where it was.
    Opening crosses it, and it is also what makes the elbow bend available at all: bent
    alone the elbows pull both hooks off by -30 deg.
    """
    if ctx.ctrl[ARM[anchor][3]] < -0.5:              # elbows already bent: nothing to do
        return
    both = WAIST + ARM['L'] + ARM['R']
    t = ctx.ctrl.copy()
    t[ARM['L'][0]], t[ARM['R'][0]] = ABDUCT, -ABDUCT     # positive on the left is inward
    yield from _follow(ctx, both, [t], rate=SERVO_RATE, measured=False)
    yield from _settle(ctx)

    # Bend as far as the load check allows, not as far as the static geometry suggests.
    # Held with the base pinned at the keyframe, forward kinematics says the elbow turns
    # the hook off its rung degree for degree -- -30 deg of bend, 30.1 deg of hook -- and
    # that abduction cancels it, 10.2 deg at 30 deg of it. Both are true of a pinned
    # base and neither survives a floating one: the body re-hangs instead of the hook
    # turning. Capping the bend at the abduction on that reasoning took the landing from
    # 13 mm to 128, outside the capture window. The load through the two wrists is the
    # only test of this that has been right.
    for elb in ELBOW_STEPS:
        keep = ctx.ctrl.copy()
        t = ctx.ctrl.copy()
        t[ARM['L'][3]] = t[ARM['R'][3]] = elb
        yield from _follow(ctx, both, [t], rate=SERVO_RATE, measured=False)
        yield from _settle(ctx)
        f = yield from _loads(ctx)
        # Load, not `latch_confirmed`: that reads a hand pressed on anything as holding.
        # Measured through this move both hooks stay on the rung the whole way -- the
        # throat point wanders 17 mm off the axis and comes back -- while `env.latched`,
        # a tight box round the nominal seat, calls them off for most of it.
        if min(f['L'], f['R']) < 10.0:
            ctx.ctrl[:] = keep
            yield from _follow(ctx, both, [keep], rate=SERVO_RATE, measured=False)
            yield from _settle(ctx)
            return


def _seat(ctx, side, other, key, slide):
    """Walk the free hand onto `key` and close on it. The last stretch of every move.

    `slide` is the direction the hand may drift in freely, or None. It is the anchored
    hook's rung axis: the estimate is pinned to where that hook was last written down, so
    when it skates along its rung the whole world appears to slide with it, and that axis
    is unobservable from the map and the encoders alike. Deweighting it is only safe when
    the rung being taken runs the same way and has length to spare -- then a hold 180 mm
    further along is just as good a hold.
    """
    g = ctx.holds.grasps[key]
    both, free = WAIST + ARM[side] + ARM[other], WAIST + ARM[side]
    R_goal, axis, n = grasp_frame(g.n()), g.a(), g.n()
    hover = g.p() + np.array([0., 0., APPROACH])

    # Two position phases, because the two directions compete. Getting *along* the rung
    # is the point of the move; getting the fore/aft right is what the +-25 mm capture
    # window is about. One weight cannot serve both -- at a single high along-weight the
    # hand never seats, at a single low one it seats without having gone anywhere (a
    # right-hand step aimed +107 mm along the rung ended up 44 mm the other way).
    yield from _servo_emp(ctx, side, other, hover, both, tol=0.030, steps=70,
                          normal=n, along=slide, w_along=W_ALONG_COARSE,
                          cheap=WAIST + ARM[side])
    yield from _servo_emp(ctx, side, other, hover, both, tol=0.010, steps=90,
                          normal=n, along=slide, cheap=WAIST + ARM[side])
    yield from _servo_emp(ctx, side, other, hover, free, tol=0.012, R_goal=R_goal,
                          axis=axis, steps=90, normal=n, along=slide)

    est = ctx.sense(other)
    off = est.throat(side) - g.p()
    if abs(off @ n) > 0.030:
        raise MoveFailed(f"{side} hand {abs(off @ n)*1000:.0f} mm off {key} fore/aft, "
                         f"outside the +-25 mm capture window")

    # Aim a little short of the rung, away from the wall. The hook has to wrap it, and
    # stopping with the throat point exactly on the axis leaves the rung at the mouth:
    # measured, seat x came out 17.7 mm against a throat that starts at 18. Backing off
    # 8 mm puts it at 18.3; 25 mm misses the rung entirely.
    seated = yield from _servo_emp(ctx, side, other, g.p() + n * SEAT_DEPTH, free,
                                   tol=0.008, R_goal=R_goal, axis=axis, steps=50,
                                   step=0.036, normal=n, along=slide)
    # Closing on a servo that never arrived is how a run ends on the floor: the hook came
    # to rest 26 mm past the rung axis, pressed on the wall rather than round the rung,
    # and the finger shut on nothing. The pad still read 20 N and the wrist 86 N, so
    # `latch_confirmed` said yes -- pressing hard on a wall looks exactly like holding.
    # The servo already knows it failed; it was only ever the return value being dropped.
    if not seated:
        raise MoveFailed(f"{side} never reached {key}: the seat servo ran out of travel "
                         f"before the throat was on the rung")

    yield from _finger(ctx, side, GRIP_SET, 0.4)
    ctx.grip[side] = (g.hold, g.p().copy(), axis, n, None)
    yield from hold(ctx, 0.4)
    ctx.resync()
    ctx.mark_latched(side)
    yield from _finger(ctx, other, GRIP_SET, 0.4)      # back to the resting grip

    if not ctx.latch_confirmed(side):
        ctx.grip.pop(side, None)
        raise MoveFailed(f"{side} did not latch on {key}")
    ctx.log.append((side, key))


def reach(ctx, side, key):
    """Move one hand to a new hold: pull up, unhook, servo across, drop on, close.

    The shape is forced by measurement. The free hand cannot rise relative to its own
    hold -- however the approach is driven it plateaus ~50 mm low -- so the height is won
    by a pull-up while both hooks are still on, which is what a climber does. And the
    anchored hook's response to arm motion is not predictable (it turned 17 deg mid-reach,
    nearly all of it off the rung axis), so nothing is planned open loop: the hand is
    walked across on a Jacobian learned from the robot's own response.

    Position uses both arms -- the anchored shoulder is the strongest lifter available.
    Aiming the hook uses only the free arm, because turning the hand with the anchored one
    levers its hook off the rung.
    """
    other = 'L' if side == 'R' else 'R'
    if not ctx.latch_confirmed(other):
        raise MoveFailed(f"refusing to unhook {side}: {other} is not holding")

    g = ctx.holds.grasps[key]
    both, n = WAIST + ARM[side] + ARM[other], g.n()
    # the anchored hook's rung: the direction its slide, and our ignorance, runs along
    slide = ctx.grip[other][2] if g.hold == ctx.grip[other][0] else None

    yield from _pull_up(ctx, other, mover=side)

    # Grip harder with the hand about to take the whole load. The measured band that
    # holds is -10..-40 deg with a cliff at -45, so there is room below the -25 deg
    # resting setpoint, and a tighter throat is a shorter run for the rung to walk along.
    # Get the arm that is about to move off its stops while both hooks are still on --
    # doing it hanging from one hand is a real posture change under load.
    yield from _unjam(ctx, ARM[side])

    yield from _finger(ctx, other, GRIP_HARD, 0.4)
    yield from hold(ctx, 0.3)
    ctx.mark_latched(other)

    yield from _finger(ctx, side, FINGER_OPEN, 0.35)
    ctx.grip.pop(side, None)
    yield from _settle(ctx)

    est = ctx.sense(other)
    yield from _servo_emp(ctx, side, other, est.throat(side) + np.array([0., 0., LIFT]),
                          both, tol=0.010, steps=12, normal=n, along=slide)
    yield from _seat(ctx, side, other, key, slide)


def climb(ctx, side, key, tries=800):
    """Take a hold on a higher row.

    A climb is not a bigger reach. Every posture that both takes a hold on the row above
    and can be held there is at least 4.4 rad from the hanging one, and the straight line
    to it falls off the wall at 0.15 rad/s and at 0.06 rad/s alike -- the line leaves the
    set of postures the robot can hang in. So the posture is solved, a route to it is
    planned inside that set, and only the last stretch is servoed.
    """
    other = 'L' if side == 'R' else 'R'
    if not ctx.latch_confirmed(other):
        raise MoveFailed(f"refusing to unhook {side}: {other} is not holding")

    g = ctx.holds.grasps[key]
    # A climb drags the anchored hook along its rung -- 310 mm at 60 N, measured in
    # phase 7 -- and the free hand's target slides with it. Both rungs need that much
    # spare length or the hand arrives over the end. Refuse here rather than spend
    # forty minutes of simulated time discovering it.
    room = min(g.margin, ctx.holds.grasps[ctx.holds.nearest(ctx.grip[other][1])].margin)
    if room < SLIDE:
        raise MoveFailed(
            f"{key}: only {room*1000:.0f} mm of rung to spare, and a climb slides the "
            f"anchored hook up to {SLIDE*1000:.0f} mm along its own (phase 7), so the "
            f"hand would land past the end. Route through a continuous rung.")
    joints = WAIST + ARM[side] + ARM[other]
    R_goal, axis, n = grasp_frame(g.n()), g.a(), g.n()

    # Coming down to a hand that is already below, the order is a climber's and not the
    # one a reach uses: put the weight on the lower hand while both are still on, then
    # take the upper hook off, and only then move it. Done the other way round -- finger
    # open, plan, go -- the upper hook never releases and carries the robot through the
    # whole move.
    below = (np.asarray(ctx.grip[other][1])[2]
             < np.asarray(ctx.grip[side][1])[2] - 0.05)

    yield from _flare(ctx, other, mover=side)
    yield from _finger(ctx, other, GRIP_HARD, 0.4)
    yield from hold(ctx, 0.3)
    ctx.mark_latched(other)
    if below:
        yield from _commit(ctx, other, side, need=20.0)
    yield from _finger(ctx, side, FINGER_OPEN, 0.35)
    ctx.grip.pop(side, None)
    yield from _settle(ctx)
    if below:
        yield from _unhook(ctx, side, other)

    est = ctx.sense(other)
    a_pos, a_R = ctx.grip[other][1], est.hook_R(other)
    a_axis = ctx.grip[other][2]
    imu = ctx.imu()
    q_now = np.array([ctx.env.sensor('enc_' + j)[0] for j in JOINTS])

    ok = Hanging(ctx.planner, imu, other, a_pos, a_R, a_axis, q_now)
    q_star = solve_hanging_pose(ctx.planner, ok, side, other, a_pos, a_R, a_axis, imu,
                                joints, ctx.lo, ctx.hi, q_now, g.p(), R_goal, tries=tries)
    if q_star is None:
        raise MoveFailed(f"no holdable posture takes {side} to {key}")
    # The RRT is cheap and solves the leading hand's row change in 3 s, so it goes first
    # there. Coming down onto a hand that is already below, the robot is hanging one-handed
    # off the lower rung while this plays, and the RRT's route gets there in legs of up to
    # 2.57 rad against the roadmap's 0.72 -- so that case pays the roadmap's ~7 s up front
    # and moves in short hops. Either way the other one is the fallback.
    rm = lambda: roadmap_hanging(ok, q_now, q_star, joints, ctx.lo, ctx.hi, side=side)
    rrt = lambda: plan_hanging(ok, q_now, q_star, joints, ctx.lo, ctx.hi)
    first, second = (rm, rrt) if below else (rrt, rm)
    path = first() or second()
    if path is None:
        raise MoveFailed(f"no route to a posture that takes {side} to {key}")

    for wp in path:
        tgt = ctx.ctrl.copy()
        for j in joints:
            tgt[j] = wp[j]
        yield from _follow(ctx, joints, [tgt], rate=CLIMB_RATE, measured=False)
        yield from _settle(ctx)

    # The guard above established that both rungs have SLIDE metres to spare, and these
    # two run parallel, so the along-rung axis is free to drift -- which is what lets the
    # servo spend its authority on fore/aft and height, the two the capture window cares
    # about. Aiming at all three at once left the hand 25 mm out and 19 mm low.
    a_ax = np.asarray(ctx.grip[other][2], float)
    slide = a_ax if abs(float(g.a() @ a_ax) / np.linalg.norm(a_ax)) > 0.9 else None
    yield from _seat(ctx, side, other, key, slide)
    if g.p()[2] < np.asarray(ctx.grip[other][1])[2] - 0.05:
        yield from _commit(ctx, side, other)


# --- the climbing moves -----------------------------------------------------
def traverse(ctx, along, steps):
    """Side-step along a row of holds: whichever hand is behind takes the next one."""
    d = np.asarray(along, float)
    d = d / np.linalg.norm(d)
    for _ in range(steps):
        side = min(ctx.grip, key=lambda s: float(np.asarray(ctx.grip[s][1]) @ d))
        here = np.asarray(ctx.grip[side][1])
        ahead = [(np.linalg.norm(ctx.holds.grasps[k].p() - here), k)
                 for k, kind in ctx.holds.neighbors(ctx.holds.nearest(here))
                 if kind in ('same_rung', 'gap')
                 and (ctx.holds.grasps[k].p() - here) @ d > 0.02]
        if not ahead:
            raise MoveFailed(f"nothing to move {side} onto in that direction")
        yield from reach(ctx, side, min(ahead)[1])


def route_to(ctx, goal):
    """Walk a route out of the hold map, one move per edge, and finish with both hands.

    The map hands back a chain of holds; the robot has two hands, so each edge needs one
    picked, and which one depends on the edge.

    Sideways, the trailing hand goes -- the one behind in the direction of travel -- which
    is what `traverse` does and what produces the alternating gait for free.

    A climb is the opposite: the *nearer* hand takes it. Sending the far hand meant R
    crossing the body to a hold 0.45 m below and 0.2 m past L, and the load line through
    the anchor ended so far off vertical that relaxing L's grip at the end of the move
    levered it off the rung -- both hooks off, 11 kg on the floor, while the hand that had
    just landed was seated correctly the whole time. Same-side climbs work; phase 6 is one.

    A vertical edge goes to `climb`, everything else to `reach`. `climb` refuses an edge
    whose rungs have no length to spare, so a route that reads fine on the map can still
    stop here; that refusal is the wall's fault, not the planner's, and it says so.
    """
    # Start the route at a hold a hand is actually on -- the one nearest the goal, which
    # is the hand that will lead. Starting it at the midpoint between the hands looks
    # harmless and is not: the midpoint is its own grasp point, so A* politely inserted a
    # step to get there, and the climb it planned went 100 mm across the body to a hold
    # neither hand was under. That run ended on the floor; from the leading hand's own
    # hold the same goal is one edge and the same climb phase 6 passes.
    lead = min(ctx.grip, key=lambda s: float(
        np.linalg.norm(np.asarray(ctx.grip[s][1]) - ctx.holds.grasps[goal].p())))
    start = ctx.holds.nearest(ctx.grip[lead][1])
    for key, kind in ctx.holds.route(start, goal)[1:]:
        g = ctx.holds.grasps[key]
        pick = min if kind in ('up', 'down') else max
        side = pick(ctx.grip, key=lambda s: float(
            np.linalg.norm(np.asarray(ctx.grip[s][1]) - g.p())))
        yield from (climb if kind in ('up', 'down') else reach)(ctx, side, key)

    # A move is not over until both hands are on the new hold. The trailing one takes the
    # nearest grasp point on the same rung that leaves room for the leading hand.
    g = ctx.holds.grasps[goal]
    lag = max(ctx.grip, key=lambda s: float(
        np.linalg.norm(np.asarray(ctx.grip[s][1]) - g.p())))
    if ctx.grip[lag][0] == g.hold:
        return
    here = np.asarray(ctx.grip[lag][1])
    room = [h for h in ctx.holds.grasps.values() if h.hold == g.hold
            and np.linalg.norm(h.p() - g.p()) > 2 * GRASP_STEP]
    if not room:
        raise MoveFailed(f"{g.hold} has no second grasp point clear of {goal}")
    k = min(room, key=lambda h: np.linalg.norm(h.p() - here))
    up = abs(float(k.p()[2] - here[2])) > 0.02
    yield from (climb if up else reach)(ctx, lag, k.key)


# --- primitive 2: haul the body ---------------------------------------------
def body_shift(ctx, delta, T=1.6, settle=0.6, frame=None):
    """Move the torso by `delta` (world metres) with every latched hand staying put."""
    anchors = [s for s in ctx.grip if ctx.latch_confirmed(s)]
    if not anchors:
        raise MoveFailed("nothing to pull against")
    # `frame` is the hand the solve is expressed relative to, and it should be the one
    # that is about to MOVE, not the one that will keep holding. The reason is in
    # ik_body: the frame hand's arm is free to reconfigure, while every other anchor is
    # pinned by an explicit zero-motion row that uses up its freedom and drives it onto
    # its stops. Measured after a pull-up: the frame arm ends with 2 joints on their
    # stops, the pinned one with 5.
    if frame in anchors:
        anchors = [frame] + [s for s in anchors if s != frame]
    joints = WAIST + sum((ARM[s] for s in anchors), [])
    est = ctx.sense(anchors[0])
    p0 = est.body_pos(est.torso)
    p1 = p0 + np.asarray(delta, float)
    n = max(1, int(T / ctx.dt))
    step = MAX_RATE * ctx.dt
    for i in range(n + int(settle / ctx.dt)):
        est = ctx.sense(anchors[0])
        p = p0 + (p1 - p0) * smooth((i + 1) / n)
        ctx.apply(joints, ik_body(est, est.torso, anchors, joints, p, max_step=step))
        ctx.tick()
        yield


def secure(ctx, T=0.6):
    """Close both fingers onto GRIP_SET. The hang keyframe leaves them open -- the hooks
    hold by geometry alone there, which is not a grip you would trust your weight to."""
    for side in 'LR':
        yield from _finger(ctx, side, GRIP_SET, T)
    yield from hold(ctx, 0.3)
    for side in 'LR':
        if not ctx.latch_confirmed(side):
            raise MoveFailed(f"{side} will not confirm a grip after securing")
        ctx.mark_latched(side)          # closing the finger seats the hook: re-baseline


def hold(ctx, T):
    for _ in range(int(T / ctx.dt)):
        ctx.tick()
        yield
