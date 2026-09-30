"""RL task over the lesson scenes: Gymnasium-shaped API, no gym dependency.

    reset()      -> obs, info
    step(action) -> obs, reward, terminated, truncated, info

Observation is HARDWARE-LEGAL only -- encoders, IMU (raw), F/T, grip pads, own
setpoints, plus the goal expressed relative to the estimated hand poses -- and every
sensor in it reads as badly as a real one: white noise and a per-episode offset
(SEN_SIG, SEN_BIAS), a SLAM that drifts, comes with each 10 Hz lidar frame, lands
SLAM_LAG late and is believed along a rung only where a wall faces that way
(`along_seen`: FAST-LIO2 cannot see along a flat wall), and a wall map that holds only
what the four Mid-360 have seen past the
robot's own body (rl.lidar), off by MAP_* rung by rung. `ClimbTask(noise=False)` turns
the noise off; the frame rate, the lag and what the lidars can see stay. The pose
estimate is the same anchor-rooted Estimator the classical controller uses: the
world is known through the hold the robot last latched, exactly what the wall map
plus encoders give on hardware. Ground truth (data.xpos, env.latched) is used only
in the REWARD and the success test -- a training-time oracle that never ships.

Action: 17 deltas in [-1, 1] on the position-servo setpoints, ACT_SCALE rad per
control step (25 Hz), clipped to the ctrlrange. The policy owns every joint,
fingers included.

    python -m rl.climb_task   # smoke check: every lesson hangs stable, API runs
"""
from fractions import Fraction
import numpy as np, mujoco
from controller.kinematics import Estimator, JOINTS, THROAT_PT, triad, rotvec
from .lessons import LESSONS, xml_path, rung_segments, spec_of, canon_axis
from climbing_env import ClimbingEnv, GRIP_BAND, GRIP_SET, GRIP_CLIFF
from . import lidar

FRAME_SKIP = 20          # sim steps per control step: 0.002 s x 20 = 25 Hz
CONTROL_DT = FRAME_SKIP * 0.002   # s; scene.xml's timestep, asserted in ClimbTask
ACT_SCALE = 0.06         # rad of setpoint travel per control step (1.5 rad/s max)
LATCH_TOL = 0.15         # m from the goal point that still counts, given a latch
RUNG_TOL = 0.06          # m: estimated throat this close to a rung axis = that rung
SLAM_TAU = 1.0           # s: how slowly the anchor's map point follows the SLAM position
                         # along its rung. 0.3 s was better on replay (free hand p90 69 ->
                         # 48 mm on lesson 1.2) and, alone, stopped a fresh policy from
                         # learning lesson -1 (bisect A: 0 % for 1M steps; C at 1.0 s:
                         # passed at 0.3M). Mechanism unknown; see `_sense` and the README

# --- what the real sensors add to the truth. None of it reaches the reward, which
# reads the truth. ponytail: datasheet-order guesses, not one measured on this robot or
# its lidar stack -- calibrate each on the hardware before trusting a sim-to-real number.
SLAM_SIG = 0.02          # m, one sigma per axis: how far the lidar SLAM's base position is off
SLAM_DRIFT_TAU = 5.0     # s it stays off by about the same: a SLAM drifts, it does not
                         # jitter, so SLAM_TAU's filter cannot average its error away
SCAN = Fraction(round(CONTROL_DT * 1e6) * lidar.SCAN_HZ, 10 ** 6)   # lidar frames per control
                         # step: 2/5, a Mid-360 frame every 2 or 3 senses (`scan_due`)
SLAM_LAG = 2             # control steps a fix is old when it lands (80 ms): the four frames
                         # merged and registered on the onboard computer, 30-60 ms for
                         # FAST-LIO2-class odometry on an embedded CPU, plus the driver, rounded
                         # up. It is held against the estimate of the moment it describes
SLAM_RHO = float(np.exp(-1.0 / (lidar.SCAN_HZ * SLAM_DRIFT_TAU)))   # frame-to-frame correlation
SLAM_GAIN = 1.0 / (lidar.SCAN_HZ * SLAM_TAU)   # how far each fix pulls the anchor
assert 1 <= SLAM_LAG <= SCAN.denominator // SCAN.numerator, \
    "a fix must be held against an estimate that already has the one before it"
FACING = 0.5             # |cos| between a wall's normal and a rung's axis for that wall to
                         # tell the SLAM where along the rung it is (`along_seen`)
MAP_SIG = 0.01           # m: a rung's axis in the lidar map, off across the axis. Not the
                         # points' own noise -- 2-3 cm a point (Mid-360, 1 sigma) fitted over
                         # the hundreds a rung gets in a second is 1-2 mm -- but the frames'
                         # registration into one map, and the four lidars' extrinsics
MAP_ROT_SIG = 0.0175     # rad (1 deg): its direction -- 5 mm at each end of a 0.35 m rung
MAP_END_SIG = 0.01       # m: each END along the axis: the angular precision (0.15 deg, 3 mm
                         # at 1 m) and the points' spacing along it. Where an end is HIDDEN
                         # is rl.lidar's business: the map ends where the seen part does
SEN_SHAPE = dict(q=(17,), dq=(17,), tau=(17,), acc=(3,), gyro=(3,), mag=(3,),
                 f=(2, 3), t=(2, 3), pad=(2,))
SEN_SIG = dict(q=0.002, dq=0.05, tau=0.3, acc=0.05, gyro=0.005, mag=0.005,   # white, a reading
               f=0.5, t=0.02, pad=1.0)   # rad, rad/s, N.m, m/s^2, rad/s, field (0.5), N, N.m, N
SEN_BIAS = dict(q=0.003, acc=0.1, gyro=0.005, mag=0.02, f=2.0, t=0.1)   # one per episode:
                         # encoder zero, IMU offsets, the motors' hard iron, F/T drift
STEP_NOISE = dict(slam=(3,), w=(3,), sen=SEN_SHAPE)   # standard normals drawn every sense

AHRS_TAU = 2.0           # s: how slowly the torso attitude leans on the accelerometer.
                         # The bare triad (accel + mag, no state) is what the classical
                         # controller uses, and while the body swings it is wrong by
                         # 26.5 deg rms -- the accelerometer reads the swing as tilt --
                         # which put the free hand 91-130 mm from where it was. The
                         # gyro integrated across the control step and corrected
                         # toward the triad at this time constant reads 3.6 / 2.1 deg
                         # rms on lessons 1 / 2, and the free hand 16 / 12 mm. Measured
                         # over tau: 0.5 s 24 mm, 2 s 16, 5 s 18, 20 s 37, gyro alone
                         # 83 -- the accelerometer still has to pin the drift.
SEAT_IN = 0.0325         # m: half the hook's width along the rung. A hook centred on the
                         # geometric end of a rung hangs half off it; the last full seat
                         # is this far in. The map the policy sees, and the potential it
                         # is paid on, describe SEATS, not geometry -- the user's point on
                         # the win9 video: it reached for the end itself and got half a hold
SEAT_TOL = 0.005         # m a hold may sit past the last full seat and still COUNT (`_holding`):
                         # the whole hook on the rung, give or take a rounding. gpu1 took
                         # lesson 2's next segment at its very end (throat p50 at the rung's
                         # end, half the hook off it) and was paid in full for it -- 7/30 on
                         # the full robot, 17/30 on the simplified one it trained on
REACH_TOL = 0.35        # m: a hold placed by arm reach (see `_reach_point`) may sit
                         # this far from the IMU-based guess and still be believed.
                         # The guess is only asked which side of the anchor the hand
                         # is on, and mid-swing it is 148 mm off on average, 1.4 m at
                         # worst -- the candidates it chooses between are 2x the
                         # hand separation apart, so a loose gate is the right one.
N_NEAR = 3               # nearest rungs reported per hand. Without these the policy
                         # cannot see the wall at all: it knows where the goal is and
                         # nothing about what lies between, so lesson 2 (gaps) and 6
                         # (staggered) are guesswork and one shared policy can only
                         # memorise each fixed layout. Three is what a hand chooses
                         # between; the map supplies them, so this is hardware-legal.
FAR = np.array([9.0, 9.0, 9.0])   # padding for a scene with fewer than N_NEAR rungs


def scan_due(k):
    """does a lidar frame land at sense k (0: the reset's)? Integer arithmetic, so
    rl.mjx_task calls it on jax arrays"""
    return (k * SCAN.numerator) // SCAN.denominator != ((k - 1) * SCAN.numerator) // SCAN.denominator


def along_seen(a, normals):
    """Can the lidar SLAM tell where the robot is ALONG a rung of axis `a`? Only if a wall
    faces that way. A flat wall and the horizontal rungs on it constrain every direction
    across the rungs and none along them: measured with FAST-LIO2 on simulated Mid-360s
    (rl.slam_test), a single wall drifts 0.2-2.5 m rms along the rung in 10 s, and a
    corner holds 5 mm. So the estimator takes the SLAM's word along a rung only when a
    wall's normal is within 60 deg of it (FACING), and otherwise its own kinematics'.
    Arithmetic only (rl.mjx_task)."""
    return any(abs(n @ a) > FACING for n in normals)


def wall_normals(m):
    """the unit normal of every wall of a model (geoms named wall_*): a box's thin axis,
    a plane's own z"""
    out = []
    for g in range(m.ngeom):
        if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('wall_'):
            continue
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        k = 2 if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE else int(np.argmin(m.geom_size[g]))
        out.append(R.reshape(3, 3)[:, k].copy())
    return out


def episode_noise(n_rung):
    """the shapes of the standard normals drawn once an episode"""
    return dict(map=(n_rung, 8), slam=(3,), bias={k: SEN_SHAPE[k] for k in SEN_BIAS})


def sensed(true, z, bias):
    """what the sensors read: the truth, SEN_SIG x `z` of white noise and SEN_BIAS x
    `bias` of this episode's offset. Arithmetic only, so rl.mjx_task calls it on jax
    arrays."""
    return {k: v + SEN_SIG[k] * z[k] + (SEN_BIAS[k] * bias[k] if k in SEN_BIAS else 0.0)
            for k, v in true.items()}


def sensed_rates(w_sum, z, bias):
    """the gyro summed over a control step's FRAME_SKIP readings, each off by the bias
    and by its own white noise"""
    return w_sum + FRAME_SKIP * SEN_BIAS['gyro'] * bias + FRAME_SKIP ** 0.5 * SEN_SIG['gyro'] * z


def slam_drift(e, z):
    """the SLAM's position error one frame later: SLAM_SIG in size, SLAM_DRIFT_TAU long"""
    return SLAM_RHO * e + SLAM_SIG * (1.0 - SLAM_RHO ** 2) ** 0.5 * z


def noisy_map(segs, goals, z, ext=None):
    """the wall map as the lidar fitted it: of each rung [(centre, unit axis, half
    length)] the span it has seen, `ext` = (lo, hi) along its axis from its centre
    (rl.lidar.extent; None: all of it), moved across its axis by MAP_SIG, turned about
    the rung's centre by MAP_ROT_SIG, and each end moved along it by MAP_END_SIG; a
    goal moves with the rung it sits on, seen or not -- it is given, not seen. `z`:
    (n, 8) standard normals, a row per rung -> (segs, goals)"""
    out = []
    for i, ((c, a, h), zi) in enumerate(zip(segs, z)):
        R = _rotm(MAP_ROT_SIG * zi[3:6])
        n = MAP_SIG * zi[:3]
        lo, hi = MAP_END_SIG * zi[6], MAP_END_SIG * zi[7]
        a2 = canon_axis(R @ a)
        u0, u1 = (-h, h) if ext is None else (ext[0][i], ext[1][i])
        rigid = c + (n - (n @ a) * a) + 0.5 * (lo + hi) * a2
        out.append((R, rigid, rigid + 0.5 * (u0 + u1) * (R @ a), a2,
                    max(0.5 * (u1 - u0) + 0.5 * (hi - lo), 0.0)))
    g2 = []
    for g in goals:
        i = int(np.argmin([np.linalg.norm(g - (c + np.clip((g - c) @ a, -h, h) * a)) for c, a, h in segs]))
        g2.append(out[i][1] + out[i][0] @ (g - segs[i][0]))
    return [o[2:] for o in out], g2


def _rung_points(segs):
    """(points, rung index of each): every rung's seats sampled ROUTE_STEP apart"""
    P, R = [], []
    for i, (c, a, h) in enumerate(segs):
        hs = max(h - SEAT_IN, 0.0)
        for s in np.linspace(-hs, hs, max(1, int(np.ceil(2 * hs / ROUTE_STEP))) + 1):
            P.append(np.asarray(c, float) + s * np.asarray(a, float))
            R.append(i)
    return np.array(P), np.array(R)


def _reach(d):
    """a move d (..., 3) against the reach: 1 at its edge, (across / HOP, up / HOP_UP)'s
    length"""
    return np.sqrt((d[..., 0] ** 2 + d[..., 1] ** 2) / HOP ** 2 + d[..., 2] ** 2 / HOP_UP ** 2)


def _rung_graph(P, R, hop=None):
    """edge lengths between the points (0 = none): neighbours along a rung, and any two
    points of different rungs a move apart that fits the reach (`_reach`), or at most
    `hop` apart every way when it is given"""
    d = P[:, None] - P[None]
    D = np.linalg.norm(d, axis=-1)
    k = np.arange(len(P))
    fits = _reach(d) <= 1.0 if hop is None else D <= hop
    return np.where(R[:, None] == R[None], np.where(np.abs(k[:, None] - k[None]) == 1, D, 0.0),
                    np.where(fits, D, 0.0))


def route_table(segs, goals):
    """(points, G): every rung's seats sampled ROUTE_STEP apart, and each point's route
    distance to each goal [L, R] -- Dijkstra over moves along a rung and hops of at most
    HOP between rungs, each goal joining its own rung's points in a straight line. The
    live cost (`ClimbTask._cost`) is measured along these routes: the straight line to a
    goal ran across family 5's break, which no hand can cross, and made the way over the
    top uphill. `segs`: [(centre, unit axis, half length)]; rl.mjx_task builds the same
    tables from the same numbers."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    P, R = _rung_points(segs)
    n = len(P)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = _rung_graph(P, R)
    goals = [np.asarray(g, float) for g in goals]
    own = [int(np.argmin([np.linalg.norm(g - (np.asarray(c) + np.clip((g - c) @ a, -h, h) * np.asarray(a)))
                          for c, a, h in segs])) for g in goals]
    G = []
    for g, r in zip(goals, own):
        A[n, :n] = np.where(R == r, np.linalg.norm(P - g, axis=1), 0.0)
        A[:n, n] = A[n, :n]
        G.append(dijkstra(csr_matrix(A), indices=n)[:n])
    # ...and the goals themselves, which sit between the samples: without them the cost
    # at a goal read twice the way to its nearest sample, 25 mm on lesson 1
    ext = [np.concatenate([Gi, [min((np.linalg.norm(g - P[R == r], axis=1) + Gi[R == r]).min(), 0.0 if j == i else np.inf)
                                for j, (g, r) in enumerate(zip(goals, own))]])
           for i, Gi in enumerate(G)]
    return np.concatenate([P, goals]), np.array(ext)


def _hill(t, i, hop):
    """the largest rise of hand i's live cost above its running minimum, walking its
    shortest route from where it starts to its goal, through the rungs with hops of at
    most `hop` (straight lines through the air): what the reward charges a climber for
    going the right way before it pays"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    P, R = _rung_points(list(t.segs.values()))
    src = int(np.argmin(np.linalg.norm(P - t._throat_true('LR'[i]), axis=1)))
    dst = int(np.argmin(np.linalg.norm(P - t.goals[i], axis=1)))
    _, pred = dijkstra(csr_matrix(_rung_graph(P, R, hop)), indices=src, return_predecessors=True)
    path = [dst]
    while path[-1] != src:
        path.append(pred[path[-1]])
    Q = P[path[::-1]]
    c = np.array([t._cost(q0 + u * (q1 - q0), i) for q0, q1 in zip(Q, Q[1:])
                  for u in np.linspace(0, 1, max(2, int(np.linalg.norm(q1 - q0) / 0.01)), endpoint=False)])
    return float(np.max(c - np.minimum.accumulate(c)))


def _rotm(v):
    """rotation vector -> matrix (Rodrigues)"""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    a = v / th
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * K @ K


# A hand is on its hold when its hook is in CONTACT with a rung and bearing force --
# see `_contact`. Two distance thresholds used to do this job and both were wrong in
# a way that was measured: 80 mm for "unambiguously off" and 20 mm for "on". A seated
# hook reads 2 mm hanging still but 31 mm under a 50 N side pull, and a hand merely
# hovering reads 42 mm, so the two states overlap and no threshold separates them.
# Contact force does not overlap at all: 72-229 N holding, exactly 0.0 N free.
# ponytail: first-guess reward weights, sweep them before trusting a run
K_BANK = 5.0             # per metre of progress BANKED by taking a new hold
K_REACH = 2.0            # per metre of live approach -- shaping only, see below
AIR = 0.5                # the live cost's extra charge per metre a hand hangs below EVERY
                         # rung (`_cost`): what keeps a dive under a far rung a loss. Charged
                         # for any air it outran what a hook lifted straight up at lesson 5's
                         # break gains (a 21 mm hill); for air below the NEAREST rung it
                         # jumped where a climbing hook's nearest rung became the one above
                         # (100-120 mm on lessons 3, 5, 6)
ROUTE_STEP = 0.04        # m between the points each rung is sampled at for `route_table`
HOP = 1.20               # m: the longest move ACROSS between two rungs a route may take
                         # -- family 9's widest gap (1.10), swung; family 5's break (1.80
                         # and up) is not one, so its route goes round, over the top --
HOP_UP = 0.65            # ...and up or down: the static reach (0.56 m up, measured), which
                         # a swing does not stretch. A move fits when (across / HOP)^2 +
                         # (up / HOP_UP)^2 <= 1. With 1.2 m every way, a route took off from
                         # lesson 5's break 1.1 m ahead and 0.45 up, and a hook lifted
                         # straight up there climbed a 36 mm hill against that fiction
K_OVER = 2.0             # extra metres charged a metre of air a hand is outside that reach
                         # of a rung point: no straight line across a break a route goes round
K_SEP = 0.05             # per step, per metre the hands are closer than MIN_SEP
MIN_SEP = 0.12           # m the hooks must keep along the rung. The hooks are 65 mm
                         # wide, so anything under that is two hands claiming the
                         # same span of rung -- the grab then fails on self-collision
                         # and the robot drops, which is what it was doing. 120 mm
                         # is not a guess: the one policy that traversed cleanly
                         # never came closer than 119 mm.
HOLD_LOAD = 20.0         # N of hook-on-rung CONTACT force before a hand counts as
                         # holding. Measured either side of it: 72-229 N while
                         # holding, 0.0 N free -- so the exact value does not matter.
DANGLE_GRACE = 6.0       # s of TOTAL off-time per hand that is free -- one reach, and
                         # one swing: at 3 s lesson 9's swing (the free hand off the
                         # whole 6.6 s it pumps and reaches) was charged 0.85 for it
DANGLE_MAX_FRAC = 0.5    # and past that, a hand may be off at most this fraction of
                         # the episode so far. Being off half the time is what
                         # alternating looks like; 79 % is one hand doing everything.
K_DANGLE = 0.01          # per step, per hand over its share
STALE_AFTER = 2.5        # s a hand may stay on its hold without ever letting go.
                         # Was 8.0, longer than the episodes it was supposed to
                         # police: the winning lesson-1 episode runs 5.8 s, so the
                         # hand that never released was charged exactly 0.00.
K_STALE = 0.001          # per step, per hand past that. Was 0.01, and with two hands
                         # a 60 s hang then cost 28.6 -- three times a fall. The
                         # 15M-step win5 policy lived in that landscape: it never
                         # fell, hung from one hook and fluttered the other to dodge
                         # its share, and collapsed its action noise to 0.13 there.
                         # Nudges must never add up to more than the fall they are
                         # meant to prevent; the incentive test below holds that.
K_ACT = 2e-4             # action magnitude penalty -- smoothness
K_VEL = 1e-4             # joint velocity penalty -- smoothness

# --- doing it the way a climber would: cheaply, and without fighting yourself -----
# Measured on the 10.4M policy against a robot just hanging still (action = 0):
#
#                        torque      both hands' load
#   hanging still       29.6 N.m         107 N          <- the cost of existing
#   lesson 1 (4 s)      55.5 N.m         131 N
#   lesson 2 (60 s)    145.1 N.m         202 N          <- on a 109 N robot
#
# 202 N of grip on a 109 N robot means the two arms are pulling against each other:
# ~95 N of pure internal force that lifts nothing. That is the opposite of shifting
# your weight onto one hand before you lift the other, which is what the gait is
# supposed to be.
#
# Both terms price the EXCESS over hanging still, never the absolute. Penalising the
# absolute would charge the robot for existing, and the cheapest way to stop paying
# would be to end the episode -- exactly the wrong lesson.
#
# Deliberately NOT rewarded: the number of times the load majority swaps. The 10.4M
# policy already swaps it 5.8 times a second on lesson 1 and 9.2 on lesson 2 -- that
# is chatter, not weight shifting, and paying per swap would buy more of it.
#
# The two baselines -- the torque of a still hang, and body weight -- are READ OFF
# THE MODEL (`ClimbTask.tau_idle`, `.weight`), not typed in. They were typed in, as
# 30 N.m and 109 N, and when the robot became the real 21.9 kg one (still hang
# ~61 N.m, 215 N) nothing moved them: hanging still was charged 0.021/step of effort
# and 0.0053 of "fighting", a 60 s idle cost -58 against -10 for a fall, and the
# incentive test below failed. The numbers in the table above are that 11.1 kg robot.
K_EFFORT = 5e-5          # per step, per N.m of torque above the idle hang
K_FIGHT = 5e-5           # per step, per N of internal force (`_internal`: what the
                         # wrists' forces cancel) above the idle hang's
K_TIME = 0.003           # per step. A whole 1500-step episode costs 4.5 against
                         # SUCCESS_R's 10, so finishing early is worth real points
                         # and giving up early still loses (FAIL_R is -10).
MIN_GRABS = 1            # real holds each hand must take before success counts. A
                         # grab is the `took` event the anchor already uses: the hand
                         # lost contact with the rung entirely and then came back
                         # bearing force with the finger inside the grip band. One is deliberately the floor -- the
                         # point is to make zero impossible, not to script a gait.
K_FLIGHT = 0.15          # per control step during which, for even one physics
                         # substep, no hook bore HOLD_LOAD: a jump. (The finger is
                         # not asked here -- that is a grab-quality question, and
                         # asking it made 52 % of an alternating episode a jump.) The 21M policy's
                         # entire lesson-1 gait was jumps -- the right hook let go
                         # and re-caught within 1-5 control steps while the left hung
                         # in the air (off 96 % of the time, 0.0 load swaps per
                         # episode), and each brief tap of the left hand then banked
                         # the distance the jump had covered. A jump also blinds the
                         # robot: the pose estimate is rooted in the hand that holds,
                         # and with none holding the position along the rung is
                         # unobservable -- measured 0.6-2.8 m of drift, which is why
                         # a broken rung was never once reached. The charge is small;
                         # the teeth are that a jump BANKS NOTHING (see step()).
K_SLIDE = 10.0           # per metre a LOADED hook (contact > HOLD_LOAD) moves along
                         # its rung. Sliding is allowed -- the rung is smooth and it
                         # is meant to stay that way -- but a loaded hook dragged
                         # along the bar is friction work, and it is also invisible:
                         # the pose estimate is rooted in the hand that holds, so
                         # every millimetre it slides is a millimetre the map is
                         # wrong by. Measured on the 20.8M policy: the anchor slid
                         # 100 mm in 6 steps with both hooks seated. This is NOT
                         # what makes dragging unprofitable -- the relative credit
                         # below is, see the block above `class ClimbTask`. It was
                         # tried at 20 and that stalled learning: the same charge
                         # lands on the 30-120 mm the holding hook settles by during
                         # every honest reach, and the 3M-step local run at 20 never
                         # left the first lesson (30 %, std 0.42) where 10 had it at
                         # 75 % by 400k. It only has to beat the live shaping term.
assert K_SLIDE > K_BANK and K_SLIDE > 2 * K_REACH, (
    "a loaded hook moving along its rung is credited nothing either way, so the only "
    "thing that stops 'reach free, drag back loaded' from being a pump is this charge")
K_GRAB = 1.0             # once per hand per episode, for its first supported hold.
                         # Seating a hook from the air is the one skill the reach
                         # credit cannot teach by itself: the 600k policy carried
                         # the free hand to within 40 mm of its goal and hovered
                         # there for the whole episode -- 0.2 grabs per episode,
                         # paying the dangle charge rather than come down, because
                         # the grab that would have banked its credit was a rare
                         # event with a small payoff. Once per hand, so flutter
                         # cannot farm it: a partial payment of the success
                         # condition (MIN_GRABS), not a goal of its own.
SUCCESS_R = 10.0
FAIL_R = -10.0
GRIP_STOP = np.radians(-37.0)   # finger setpoint limit: 3 deg inside GRIP_BAND's closed edge

# Progress is CREDITED to a hand only while it is free and the other hand holds,
# and only for the part of its approach that is its own -- the holder's motion is
# subtracted -- then BANKED when it takes its next supported hold. Retreats always
# debit. Three earlier shapes failed, and each is worth writing down.
#
# Paying for live distance however it closed paid for the phase-7 slide: the first
# run crossed 0.75 m with the left hook never once opening its finger.
#
# Paying live distance only to a hand that was OFF a hold looked like the fix and
# was a reward PUMP. Measured over one episode of the policy it produced: net
# distance closed 0.000 m, progress paid +27.28, and 99 release-reseat cycles on the
# right hand. Swing the hand out (counted free, moving toward the goal, PAID), swing
# it back (counted seated, moving away, NOT CHARGED) -- the asymmetry is the bug and
# no choice of threshold fixes it, because any boundary can be oscillated across.
#
# K_SEP is the anatomy the reward was missing. The distance term is a SUM over the
# two hands, so it is indifferent to which hand does the work: one hand travelling
# twice as far scores exactly the same as the two alternating. Every policy took that
# deal -- on lesson 2 at 7.6M steps the left hand moved 11 mm all episode while the
# right crossed the body and did everything. It is not a kinematic necessity: driven
# on a robot that really hangs, the leading shoulder abducted 30 deg OUTWARD moves
# that hand +126 mm toward the goal and stays latched, which is the climber's flare.
#
# It took two goes to price. Measuring in the TORSO frame is useless -- the body
# spins on its hooks, torso yaw swept -90..+13 deg in one episode, so the body-frame
# test disagreed with the real hand order on 54 % of steps. It is measured along the
# RUNG axis instead, which does not spin, with the sign fixed at reset.
#
# And pricing only the CROSSED case was still not enough: max(0, -separation) is zero
# at zero separation, so the policy parked both hands on the same point instead --
# mean separation 1 mm, 99 % of steps inside 100 mm. Two 65 mm hooks cannot share a
# span of rung, so the second grab failed on contact and the robot fell. Hence a
# floor, MIN_SEP, rather than a sign test.
#
# The third shape was an ANCHOR: bank how far the hold travelled, in the world,
# between one supported grab and the next. It telescopes and it cannot be pumped by
# oscillation -- and it was beaten a third way. The 2M local run dragged the loaded
# right hook along the rung 864 mm per lesson-1 episode while the free left hand
# tapped its way along: every tap re-seated the left hand where it already was
# RELATIVE to the right, and banked the distance the drag had carried the body.
# One drag advanced both anchors, 14/m earned against 10/m of slide charge. Raising
# the charge to 20 stalled learning instead, because the same charge lands on the
# few centimetres the holding hook settles by during every honest reach.
#
# So progress is measured the way the sensors measure it: relative to the hand that
# holds. Per step a light hand (under HOLD_LOAD) whose partner is loaded is credited
# its approach MINUS the partner's own displacement toward that goal, both ways, so
# the partner settling back and forth cancels out; a loaded hook is credited nothing
# either way -- its motion along the rung is a slide and K_SLIDE's business; and a
# hand nobody is carrying (a jump) is credited nothing forward and debited in full
# back. The first draft debited every retreat and subtracted only the partner's
# forward motion, both one-sided to be safe, and both bled on ordinary physics: a
# known-good one-hand step came out at -220 mm banked per episode, -7.5 m of debits
# against +6.3 m of credits over ten of them, because a holding hook settles by
# millimetres every step. One-sided rules charge for noise. A drag, a tap and a jump
# earn exactly nothing here; the one loop that would pay -- reach free, get dragged
# back loaded -- costs K_SLIDE > K_BANK, and the no-pump test holds the books to
# that. A small live-distance term stays as shaping, charged symmetrically so it
# telescopes.
#
# The dangle penalty is the other half, and its first version was WRONG in an
# instructive way: charged flat, every step a hand was off, it taxed the release
# that climbing is made of. Measured over one episode it came to -15.25 against
# +5.57 of progress, so the cheapest policy was to keep both hooks on and shove --
# the flat penalty actively pushed towards the exploit it was meant to close.
#
# What deserves a penalty is not letting go, it is a hand PARKED off the wall. So
# each hand gets DANGLE_GRACE seconds free -- long enough for a reach. The first
# version then charged every further step of a CONTINUOUS stretch off the wall, and
# a policy walked straight through it: measured on the 10.4M run, the right hand was
# off the rung for 79 % of the episode and paid 0.13 in total, because it made 88
# short hops and only one of them outlasted the grace. The grace reset on every
# touch. So the budget is now cumulative over the episode and capped as a FRACTION
# of it -- flutter cannot dodge a share.
#
# K_STALE is the mirror image, and it exists because of the single most consistent
# finding across every run: the LEFT HAND HAS NEVER ONCE LET GO. Not in any policy,
# at any step count. The two hands deadlock -- the right is airborne 79-84 % of the
# time, so there is never a safe moment for the left to release, and the roles
# freeze. K_DANGLE says do not stay off; K_STALE says do not stay on. Together the
# only cheap gait left is alternating.
#
# Both hooks off needs no term of its own: the episode already ends with FAIL_R
# after 0.6 s of it.


class ClimbTask:
    def __init__(self, lesson=1, visual=True, xml=None, mirror=False, noise=True):
        """`xml`: another scene of the same lesson -- `rl.mjx_scene.c_path`'s simplified
        robot, for the cross-checks against MJX. `mirror`: the lesson toward the
        robot's right (rl.lessons.mirrored). `noise=False`: sensors, SLAM and map exact
        (the SLAM's rate and lag stay)"""
        self.lesson = lesson
        self.noise = noise
        self.rng = np.random.default_rng()
        self.spec = spec_of(lesson, mirror)
        self.env = ClimbingEnv(xml=xml or xml_path(lesson, mirror), success_z=1e9,
                               timeout_s=self.spec['timeout'],
                               held_fn=self._attached, visual=visual)
        self.est = Estimator(self.env.model)
        self.segs = rung_segments(self.env.model)
        self.goals = [np.asarray(g, float) for g in self.spec['goals']]
        self._pts, self._route_g = route_table(list(self.segs.values()), self.goals)
        self.lo = self.env.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.env.model.actuator_ctrlrange[:, 1].copy()
        # The finger joint runs to -90 deg, but the README measures the hook levering
        # itself off the rung past GRIP_CLIFF (-45): pull-off drops from >1200 N to
        # 72 N. The 20.8M policy drove the right finger to -83 and the hook walked
        # out of the throat while carrying 100 N. Nothing above the cliff is useful,
        # so it is a mechanical stop, not a reward term -- 45 deg of pure
        # self-destruct removed from the search space. Opening is untouched: a hand
        # still has the full range up to +20 to let go.
        # ...and not at the cliff itself but at the band's edge. With the stop at
        # GRIP_CLIFF (-45) a policy that squeezes as hard as it can sits at -40..-44
        # under load -- physically a fine hold (98-179 N of contact) that neither
        # the reward nor the estimator recognised, because both ask for the band.
        # Measured on the 10.8M policy: 90 % of loaded steps on lesson 1.5 and 34 %
        # on lesson 1 were "not gripping" for that reason alone, the estimator's
        # anchor stayed on a hand that was moving, and the far ledge was seen
        # 240-500 mm from where it was. Squeeze as hard as you like: it is in band.
        # ...and 3 deg INSIDE the edge, not on it. A stop at exactly -40 had no
        # margin: a hook whose setpoint sits on the stop reads 0.5-2.5 deg past it
        # whenever the rung is not pressing the finger open -- a hook that has handed
        # its weight to the other one, a landing, a swing. On the collapsed win16 and
        # win16 2 policies the right hook's real holds (93 N of contact and 0.69 mm of
        # slide per step, against 97 N and 0.91 mm in band) read -41 to -42.5 on
        # 71-78 % of loaded steps and counted as not holding, in the reward and in the
        # estimator's anchor alike. With the stop at -37 the same policies' loaded
        # holds count 100 %, and -37 is still deep in the >1200 N band.
        for j in ('finger_L', 'finger_R'):
            self.lo[JOINTS.index(j)] = GRIP_STOP
        self.dt = self.env.model.opt.timestep
        assert abs(FRAME_SKIP * self.dt - CONTROL_DT) < 1e-12, (self.dt, CONTROL_DT)
        m = self.env.model
        self._rung_name = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                           for g in self.env.rungs}
        self._hook_geoms = {}
        paired = set(m.pair_geom1) | set(m.pair_geom2)   # rl.mjx_scene collides by <pair>
        for sd in 'LR':
            ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm % sd)
                   for nm in ('hook_%s', 'finger_%s')}
            self._hook_geoms[sd] = {g for g in range(m.ngeom) if m.geom_bodyid[g] in ids
                                    and (m.geom_contype[g] or g in paired)}
        self._flight = 0
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')
        self._gyro_adr = m.sensor_adr[sid]
        self._w_sum = np.zeros(3)          # body rates summed over the substeps
        self._Rf = None                    # filtered world<-torso attitude
        # the two baselines effort and fight are priced above (see K_EFFORT)
        self.weight = float(m.body_subtreemass[self.env.base] * -m.opt.gravity[2])
        self.tau_idle = None               # N.m, set by the first reset: the secured hang
        self.fight_idle = None             # N, its internal force (`_internal`), likewise
        self._ft = {s: self.env.model.site('s_ft_' + s).id for s in 'LR'}
        # what the lidars look at: the rungs as they are (rl.lidar), and the walls
        self.rig = lidar.Rig(m)
        self.walls_n = wall_normals(m)
        v = list(self.segs.values())
        self._rungs = (np.array([c for c, _, _ in v]), np.array([a for _, a, _ in v]),
                       np.array([h for _, _, h in v]),
                       np.array([m.geom_size[m.geom(nm).id][0] for nm in self.segs]),
                       np.ones(len(v), bool))
        self.obs_dim = len(self.reset()[0])

    # ---- hardware-legal sensing -------------------------------------------
    def _in_band(self, side):
        """the finger in GRIP_BAND, as its encoder reads"""
        return bool(GRIP_BAND[0] <= self.sen['q'][JOINTS.index('finger_' + side)] <= GRIP_BAND[1])

    def _confirmed(self, side):
        """same latch evidence the classical controller trusts: pad + finger + load"""
        i = 'LR'.index(side)
        return bool(self.sen['pad'][i] > 20.0 and self._in_band(side)
                    and np.linalg.norm(self.sen['f'][i]) > 12.0)

    def _touching(self, side):
        """is the hook still on its rung, loaded or not? Pad contact and the finger
        in the band. `_confirmed` asks for HOLD_LOAD-scale force, which a hook that
        has handed its weight to the other one no longer has -- but it has not gone
        anywhere, and its hold is still the map's best anchor."""
        return bool(self.sen['pad']['LR'.index(side)] > 2.0 and self._in_band(side))

    def _normals(self, shapes):
        """standard normals in `shapes` (a dict, nested), zeros for noise=False"""
        return {k: self._normals(v) if isinstance(v, dict)
                else self.rng.standard_normal(v) if self.noise else np.zeros(v)
                for k, v in shapes.items()}

    def _hung(self, side):
        """(rung name, point on its map axis) the keyframe hangs this hook on -- an
        initial condition you know on hardware too, because you hung it there"""
        nm, gap = self._seat(side)
        assert gap < RUNG_TOL, "keyframe hooks are not on a rung"
        c, a, h = self.map[nm]
        return nm, c + np.clip((self._throat_true(side) - c) @ a, -h, h) * a

    def _look(self):
        """a lidar frame: what the four see from here joins the map, which spans each
        rung from the first seen cell to the last (rl.lidar)"""
        d = self.env.data
        self.seen |= lidar.visible(np, self.rig, d.site_xpos, d.site_xmat, *self._rungs)
        m, self.map_goals = noisy_map(list(self.segs.values()), self.goals, self.ep_z['map'],
                                      lidar.extent(np, self.seen, self._rungs[2]))
        self.map = dict(zip(self.segs, m))
        self.map_seen = {nm for nm, sn in zip(self.segs, self.seen) if sn.any()}

    def _nearest_rung(self, p):
        """(rung name, point on its axis) if p is within RUNG_TOL of a rung the map has,
        else None"""
        best = None
        for nm, (c, a, h) in self.map.items():
            if nm not in self.map_seen:
                continue
            q = c + np.clip((p - c) @ a, -h, h) * a
            d = float(np.linalg.norm(p - q))
            if best is None or d < best[0]:
                best = (d, nm, q)
        return (best[1], best[2]) if best and best[0] < RUNG_TOL else None

    def _sense(self):
        """read sensors, update the estimate, and keep the hold/anchor books.

        `self.true` is what the sensors would read if they were perfect -- the reward's
        -- and `self.sen` what they do read, `sensed`: everything below and the
        observation see only that."""
        if scan_due(self._k):
            self._look()
        enc = self.env.encoders()
        imu = self.env.imu(raw_only=True)
        hand = [self.env.hand(s) for s in 'LR']
        self.true = dict(q=np.array([enc[j][0] for j in JOINTS]),
                         dq=np.array([enc[j][1] for j in JOINTS]),
                         tau=np.array([enc[j][2] for j in JOINTS]),
                         acc=imu['accel'], gyro=imu['gyro'], mag=imu['mag'],
                         f=np.stack([h['force'] for h in hand]),
                         t=np.stack([h['torque'] for h in hand]),
                         pad=np.array([h['pad'] for h in hand]))
        z = self.z = self._normals(STEP_NOISE)
        bias = self.ep_z['bias']
        self.sen = sensed(self.true, z['sen'], bias)
        self.q, self.dq, self.tau = self.sen['q'], self.sen['dq'], self.sen['tau']
        self.imu_s = dict(accel=self.sen['acc'], gyro=self.sen['gyro'], mag=self.sen['mag'])
        if self._k == 0:                  # at rest the triad is right, as far as it reads
            self._Rf = triad(self.imu_s['accel'], self.imu_s['mag'])
        # Attitude: the gyro carries it across the step, the IMU triad pins the
        # drift. The triad alone reads the body's own swing as tilt -- 26.5 deg rms,
        # the free hand 91-130 mm off; filtered it reads 1.1 deg on a still hang and
        # 4 deg p50 / 6 p90 / 15 max during the climb, where every hand-over is an
        # impact (35-55 mm at the hand on a 0.5 m lever). The SLAM's attitude fixed
        # that on replay -- 0.2 deg, the hand 13/40 mm instead of 36/80 -- and a
        # trained policy ran fine under it, but a policy trained FROM SCRATCH with it
        # never learnt lesson -1 (0 % for 1-4M steps, six runs, two independent
        # changes each sufficient: this and SLAM_TAU 0.3), against 0.3-0.4M with
        # this filter. Not the observation's scale (a variance floor in VecNormalize
        # did not help), mechanism unknown -- see the README. So this is what the
        # policy trains under, and the SLAM's attitude is not modelled.
        dt = FRAME_SKIP * self.dt
        self._Rf = self._Rf @ _rotm(sensed_rates(self._w_sum, z['w'], bias['gyro']) * (dt / FRAME_SKIP))
        self._w_sum[:] = 0.0
        R_tri = triad(self.imu_s['accel'], self.imu_s['mag'])
        self._Rf = self._Rf @ _rotm((dt / AHRS_TAU) * rotvec(self._Rf.T @ R_tri))
        self.est.update(self.q, self.imu_s, self.anchor, self.hold[self.anchor][1],
                        R_torso=self._Rf)
        # The loaded hook creeps along its rung while it holds -- 85-210 mm per
        # episode on the mac16 policy, 100 mm by the first hand-over -- and nothing
        # on the arm can see it: two hooks on a smooth rung have no world reference,
        # so the anchor's map point went stale by exactly that and at the gap the far
        # segment read 60-210 mm further than it was (0/40 for 3M steps). The lidar
        # SLAM on the real robot sees the body move against the wall. Its position,
        # low-passed over SLAM_TAU, says where along the rung the anchor has crept
        # to. Along the rung only: correcting all three axes was tried and made the
        # free hand WORSE (p50 36 -> 46 mm, p90 80 -> 94 on the mac20 policy) -- a
        # translation at the base cannot stand in for the attitude's error at the
        # hands, and off the axis that is most of the discrepancy.
        #
        # A fix comes with every lidar frame and describes the body SLAM_LAG steps
        # ago, so it is held against the estimate of that moment (`_buf`): against
        # the current one, a body swinging along the rung would read as creep.
        base = self.env.data.xpos[self.env.base].copy()
        if self._k == 0:
            self._buf = [(base, self.est.body_pos(self.est.base))] * SLAM_LAG
        dp = self._slam_fix()
        if dp is not None:
            nm, pt = self.hold[self.anchor]
            ax = self.map[nm][1]
            self.hold[self.anchor] = (nm, pt + SLAM_GAIN * float(dp @ ax) * ax)
            self.est.update(self.q, self.imu_s, self.anchor, self.hold[self.anchor][1],
                            R_torso=self._Rf)
        # The anchor's hold is otherwise left alone while it holds: re-projecting a
        # hand's own estimate onto its own rung learns nothing. The OTHER hand's hold,
        # while both are seated, comes from arm reach -- exact along the rung,
        # which the IMU-rooted estimate is not (see `_reach_point`).
        other = 'R' if self.anchor == 'L' else 'L'
        cA, cB = self._confirmed(self.anchor), self._confirmed(other)
        # The anchor's hold is trusted for one step after it stops being confirmed:
        # a hook that has only just unloaded has not gone anywhere in 40 ms. Without
        # that, a quick hand-over -- the old hand under HOLD_LOAD before the new one
        # is over it -- switched to the new hand's STALE hold from its previous
        # tenure. Measured on the 3M policy: 86 of 129 switches had no overlap and
        # baked in 62 mm each, 5.3 m over four episodes; the 43 with overlap, 1 mm.
        both = cB and (cA or self._anchor_fresh or self._touching(self.anchor))
        if both:
            pA = self.hold[self.anchor][1]
            delta = self.est.throat(other) - self.est.throat(self.anchor)
            # which side: at the moment of the grab, the filtered estimate; after
            # that, wherever this hand was a step ago -- the branch never flips
            # under a hand that has not moved
            guess = self.hold[other][1] if self._both_conf else self.est.throat(other)
            hit = (self._reach_point(pA, float(np.linalg.norm(delta)), guess)
                   or self._nearest_rung(self.est.throat(other)))
            if hit:
                self.hold[other] = hit
        # anchor: keep the current one while it is confirmed, else the other hand
        if not cA and cB:
            self.anchor = other
            self.est.update(self.q, self.imu_s, self.anchor,
                            self.hold[self.anchor][1], R_torso=self._Rf)
            cA = True
        self._both_conf = both
        self._anchor_fresh = cA
        self._buf = self._buf[1:] + [(base, self.est.body_pos(self.est.base))]
        self._k += 1

    def _slam_fix(self):
        """the base's position as the SLAM last reported it, minus the estimate of the
        moment that report describes, when one lands this sense; else None. Here the
        SLAM is its error model (`slam_drift`); rl.slam_test puts FAST-LIO in its place."""
        if not scan_due(self._k):
            return None
        self._slam_err = slam_drift(self._slam_err, self.z['slam'])
        if not along_seen(self.map[self.hold[self.anchor][0]][1], self.walls_n):
            return None                      # a flat wall: it cannot say (`along_seen`)
        then, est_then = self._buf[0]
        return then + self._slam_err - est_then

    def _reach_point(self, anchor_pt, L, guess):
        """(rung name, point) for a seated hand exactly L from the anchor's hold.

        Both throats sit on rung axes the map knows, and L -- the distance between
        them -- is exact from the encoders whatever the attitude: relative poses
        inside the scratch tree do not depend on where the tree is put. So the new
        hold is a sphere of radius L about the anchor's hold cut by a rung's axis:
        two candidates per rung, and `guess` (the filtered estimate) is asked only
        which one.

        Why this exists: rooting everything in the IMU attitude put the free hand
        148 mm off on average (up to 1.4 m) while the body swung, and with an
        exact attitude the very same arithmetic was off by 0 mm. Every hold taken
        through that error was baked into the map frame for good -- measured
        0.25-1.3 m of drift along the rung over one alternating episode, so a
        broken rung's gap was never where the policy saw it.
        """
        best = None
        for nm, (c, a, h) in self.map.items():
            if nm not in self.map_seen:
                continue
            w = c - np.asarray(anchor_pt)
            b = float(w @ a)
            disc = b * b - (float(w @ w) - L * L)
            if disc < 0.0:
                continue
            for s_ in (-b - np.sqrt(disc), -b + np.sqrt(disc)):
                if abs(s_) > h + 0.03:          # off the end of this segment
                    continue
                q = c + float(np.clip(s_, -h, h)) * a
                err = float(np.linalg.norm(q - guess))
                if best is None or err < best[0]:
                    best = (err, nm, q)
        return (best[1], best[2]) if best and best[0] < REACH_TOL else None

    def _near_rungs(self, p):
        """the N_NEAR rungs closest to p: offset to each one's axis, and its direction;
        and for the nearest one, how far p is from each of its two ends along it.

        Every point along a rung is a seat -- the rungs are smooth, so where a hand
        lands along one is free, which is also why it can slide there. But a rung
        ENDS, and a policy that cannot see where was measured reaching for the far
        ledge with its holding hand still half a metre from the edge of its own:
        127-160 mm short at the static reach limit, in three runs, never once
        across. The two end distances are signed along the rung's axis (which is
        also in the observation), negative once past that end.

        Estimated, not ground truth: `p` is the estimator's throat and the rung
        geometry is the lidar's wall map, as far off as one (`noisy_map`) and holding
        only what the lidars have seen. A rung not seen yet is not there.
        """
        cand = []
        for nm, (c, a, h) in self.map.items():
            h = max(h - SEAT_IN, 0.0)              # the span a hook can actually sit on
            s = float((p - c) @ a)
            q = c + np.clip(s, -h, h) * a
            cand.append((float(np.linalg.norm(p - q)) if nm in self.map_seen else np.inf,
                         q - p, a, s, h))
        cand.sort(key=lambda v: v[0])              # stable: none seen, the first rung's ends
        out = []
        for i in range(N_NEAR):
            if i < len(cand) and np.isfinite(cand[i][0]):
                out += [cand[i][1], cand[i][2]]
            else:
                out += [FAR, np.zeros(3)]
        _, _, _, s, h = cand[0]
        out.append(np.array([h - s, h + s]))
        return np.concatenate(out)

    def _obs(self):
        s = self.sen
        goal = np.concatenate([self.map_goals[0] - self.est.throat('L'),
                               self.map_goals[1] - self.est.throat('R')])
        near = np.concatenate([self._near_rungs(self.est.throat(s)) for s in 'LR'])
        # where the OTHER hand is, from each hand's point of view. It is derivable
        # from the encoders, but only by doing forward kinematics, which is a lot to
        # ask of a 64x64 MLP -- and without it the policy kept sending one hand to
        # the span of rung the other was already holding, so the grab failed on
        # contact and the robot dropped.
        other = np.concatenate([self.est.throat('R') - self.est.throat('L'),
                                self.est.throat('L') - self.est.throat('R')])
        conf = [float(self._confirmed(s)) for s in 'LR']
        # float32, not the float64 everything above is computed in: Metal has no
        # float64 at all, so an f64 observation makes `--device mps` fail outright.
        # It is also the SB3/gym convention and halves the rollout buffer.
        return np.concatenate([
            s['q'], s['dq'], s['tau'], self.ctrl, s['acc'], s['gyro'], s['mag'],
            s['f'][0], s['t'][0], s['pad'][:1], s['f'][1], s['t'][1], s['pad'][1:],
            goal, near, other, conf]).astype(np.float32)

    # ---- training-time oracle (reward / success only) ----------------------
    def _throat_true(self, side):
        b = self.env.hook[side]
        return self.env.data.xpos[b] + self.env.data.xmat[b].reshape(3, 3) @ THROAT_PT

    def _route(self, p):
        """(how far p is off the nearest rung, the point on it) -- the wall map"""
        best = None
        for c, a, h in self.segs.values():
            hs = max(h - SEAT_IN, 0.0)
            q = c + np.clip(float((p - c) @ a), -hs, hs) * a
            d = float(np.linalg.norm(p - q))
            if best is None or d < best[0]:
                best = (d, q)
        return best

    def _cost(self, p, i):
        """metres to go for hand i (0 = L) at p, LIVE: along the route through the rungs
        (`route_table`) from the rung point that is cheapest to fly to -- the air to it,
        and K_OVER a metre more of it outside the reach (`_reach`) -- plus AIR x how
        far p hangs below every rung. Progress is measured along the routes a climber
        can take, not through the air.

        The plain distance |p - goal| paid for leaving the rung. On lesson 1.5 the
        left hook walked off the end of its segment, and with the goal 1.4 m along
        the wall the cheapest way to get "closer" was to hang 0.4 m below the far
        segment: +0.14 m of straight-line progress, +0.3 of reach, and a state no
        lesson had taught it to leave -- win9 (5.3M steps, lesson 1.5 at 0 %) and
        mac13 both: 20/20 and 7/8 episodes hanging there for 58 s.

        Two measures after it failed differently, 2026-09-26. Through the NEAREST rung
        (the detour off it plus that point's straight line to the goal) the first half
        of every reach for another rung was uphill: a hook lifted toward the row above
        paid its lift until the upper rung became the nearer -- K_REACH x 0.15 m for
        trying, on family 3's easiest walls -- and gpu1 (102.9M steps) never lifted a
        hook past 28 mm there, 0 % for 90M steps. Through the BEST rung's nearest point
        the hill went, the dive came back (a straight line through the air to the
        goal's own segment, 49 mm shorter from 0.47 m down than from the edge), AIR
        sent it away again -- and the climb over lesson 5's break stayed uphill, 0.37 m
        for a hook lifted straight up, because the straight line ran across a break
        no hand crosses; lessons 4, 6 and 7 kept 0.06-0.09 m hills where the nearest
        point of the next rung was not on the way. The barrier test now walks every
        lesson's shortest route with each hand and asserts no hill.

        On a rung this is the route distance itself (its own point costs no air, and
        no other within a hop can be cheaper); off it, flying to a point of a rung the
        route uses only gets cheaper as the hand closes in, faster than AIR charges the
        air; the dive costs 158 mm against hovering at the edge. A function of where
        the hand is, so it telescopes and cannot be pumped.

        Live only. The detour must stay out of the bank: the bank credits a free
        hand and a loaded hook nothing, and a loaded hook's throat is pulled 30-80 mm
        off its rung by the load (the README's 31 mm under a 50 N side pull). With
        the detour in the bank that uncharged distortion was refunded on every
        lift-and-reseat: the first check run (mac14, 3M) fluttered the right hand
        698 times per episode, banked 56 m and scored +241 on a body that never
        moved. `_hand_dists`, which the bank uses, is measured at the rung point,
        which a loaded hook can move only by sliding -- and sliding is charged.
        """
        d = self._pts - p
        air = np.linalg.norm(d, axis=1)
        out = air * np.maximum(1.0 - 1.0 / np.maximum(_reach(d), 1e-9), 0.0)   # outside the reach
        via = float(np.min(air + K_OVER * out + self._route_g[i]))
        below = min(max(0.0, float(c[2] + np.clip((p - c) @ a, -max(h - SEAT_IN, 0.0), max(h - SEAT_IN, 0.0)) * a[2] - p[2]))
                    for c, a, h in self.segs.values())
        return via + AIR * below

    def _hand_dists(self):
        """the bank's potential: each hand's rung point to its goal"""
        return np.array([float(np.linalg.norm(self._route(self._throat_true(s))[1] - self.goals[i]))
                         for i, s in enumerate('LR')])

    def _live(self):
        """the live potential: each hand's way to its goal along the routes (`_cost`)"""
        return np.array([self._cost(self._throat_true(s), i) for i, s in enumerate('LR')])

    def _contact(self, side):
        """(N of normal force between this hook and a rung, names of rungs touched)

        The only unambiguous answer to "is this hand on its hold", and the one that
        should have been used from the start. Every proxy tried before it failed in a
        way that was measured on this model:

          `env.latched`   a box round the nominal seat. Reads False on BOTH hands for
                          32 % of an episode of the 20.8M policy, 89 % of those with a
                          hook carrying more than 20 N.
          seat gap        a radial distance. A hook that is genuinely seated reads
                          2 mm hanging still but 31 mm under a 50 N side pull, and a
                          hand merely hovering reads 42 mm -- the two states overlap,
                          so no threshold separates them.
          wrist F/T       a hand pressed flat on a wall reads 86 N with the finger
                          shut on nothing.

        Contact force has no overlap at all: measured 72-229 N while holding, and
        exactly 0.0 N free or fallen, across static hangs, one-handed hangs, a 50 N
        side pull and a 200 N downward pull.
        """
        m, d = self.env.model, self.env.data
        tot, hit, f6 = 0.0, set(), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self._hook_geoms[side] and b in self._rung_name:
                    mujoco.mj_contactForce(m, d, i, f6)
                    tot += abs(f6[0])
                    hit.add(self._rung_name[b])
        return tot, hit

    def _seat(self, side):
        """(rung name, gap) for the rung this hook is nearest to"""
        p = self._throat_true(side)
        return min(((nm, float(np.linalg.norm(
            p - (c + np.clip((p - c) @ a, -h, h) * a))))
            for nm, (c, a, h) in self.segs.items()), key=lambda v: v[1])

    def _sep_error(self, con):
        """metres of hand separation missing along the rung BOTH hooks are on (`con`:
        each hand's `_contact`), 0 unless they share one.

        Pricing only the crossed case was not enough, and the failure is instructive:
        `max(0, -separation)` is zero at zero separation, so the policy moved the
        hands off each other's far side and onto exactly the same point instead --
        measured, mean separation 1 mm and 99 % of steps inside 100 mm. Two 65 mm
        hooks cannot share a span of rung, so the second hand's grab failed on
        contact and the robot fell. That is the drop the videos show.

        The torso frame was tried first and is not a usable reference: the body spins
        on its hooks -- measured, the torso yawed -90..+13 deg over one episode -- so
        a body-frame test disagreed with the real hand order on 54 % of steps and
        charged 0.93 for an episode that was crossed 99 % of the time. The rung axis
        does not spin, and the sign (`_order`) says "each hand on its own side".

        Only two hooks on ONE rung can claim the same span of it. Measured along the
        start rung whatever the hooks were on, until 2026-09-26, it charged what could
        never collide: a climb with the hands on two rows, 0.7-2.1 a climb (family 3,
        CEM-found, replayed), lesson 9's swing with the free hand in the air passing
        the holding one, 1.56 over 6.6 s -- and the corner's side rung, where both
        goals sit at one y, read MIN_SEP short to the end.
        """
        shared = con[0][1] & con[1][1]
        if not shared:
            return 0.0
        a = self.segs[next(n for n in self.segs if n in shared)][1]
        sep, o = float((self._throat_true('L') - self._throat_true('R')) @ a), self._order(a)
        return max(0.0, MIN_SEP - (o * sep if o else abs(sep)))

    def _internal(self):
        """N of force the two wrists spend pulling against each other: the part of their
        (world-frame) forces that cancels, (|F_L| + |F_R| - |F_L + F_R|) / 2.

        It was the two forces' SUM over body weight until 2026-09-26, which also billed
        whatever accelerates the body: lesson 9's swing loads the holding wrist to 634 N,
        three times body weight, at the bottom of each arc, with the free hand at 4 N --
        charged 0.38 as fighting over 6.6 s -- and every pull-up of a climb likewise
        (0.5-2.0 a climb). What moves the body is the forces' resultant; only what
        cancels lifts nothing."""
        F = [self.env.data.site_xmat[self._ft[s]].reshape(3, 3) @ self.env.hand(s)['force'] for s in 'LR']
        return 0.5 * float(np.linalg.norm(F[0]) + np.linalg.norm(F[1]) - np.linalg.norm(F[0] + F[1]))

    def _order(self, a):
        """which way round the hands belong along a rung of axis a: +1 when the left
        one's side is +a. The goals say it where they lie apart along it; the start
        order carried over where they do not (the front rung, on the corner lesson);
        0 -- either way round -- on a rung across both"""
        g = float((self.goals[0] - self.goals[1]) @ a)
        if abs(g) > 0.05:
            return float(np.sign(g))
        c = float(self.cross_axis @ a)
        return self.cross_sign * float(np.sign(c)) if abs(c) > 0.5 else 0.0

    def _attached(self):
        """is ANY hook still on the wall? -- the `released` timer's question.

        The reward and the terminator have to agree. They did not: the reward called
        a hand held at a seat-gap threshold, while the episode was ended by
        `env.latched`, a box round the nominal seat. Measured over one episode of the
        20.8M policy the box read False on BOTH hands for 32 % of the steps, and 89 %
        of those had a hook carrying more than 20 N -- so every one-handed phase was
        0.6 s from a spurious `released` and a -10. That is the phase a climbing gait
        is made of, and it is why 20.8M steps bought a 2.5 % success rate.

        Union with `latched`, never a replacement, so this can only ever keep an
        episode alive longer than the old rule did.
        """
        con = [self._contact(s)[0] for s in 'LR']
        self._w_sum += self.env.data.sensordata[self._gyro_adr:self._gyro_adr + 3]
        # Sampled every physics substep, because a jump can start and end inside
        # one control step: the 21M policy let go and re-caught in 1-5 steps.
        if not any(c > HOLD_LOAD for c in con):
            self._flight += 1
        return any(c > 0.0 for c in con)

    def _gripping(self, side):
        """is the finger in the band that actually holds?

        `GRIP_BAND` is -40..-10 deg, and the model README measures why: inside it the
        pull-off force is >1200 N, at 0 deg it is 764 N, and past -45 deg the rung
        walks out of the throat and it collapses to 72 N. So there are TWO ways out
        of a grip, and the 10M policy used both -- the left hand opened to +20 deg
        and dangled, while the right ratcheted along the rung by cycling -36 -> -46
        -> -37. Nothing in the reward had ever asked a finger to close, because the
        simulator lets an open hook carry the whole robot (the README says so
        outright), so closing cost action and bought nothing.
        """
        return GRIP_BAND[0] <= self.env.sensor('enc_finger_' + side)[0] <= GRIP_BAND[1]

    def _holding(self):
        """per hand: seated, carrying, AND actually gripping.

        Load alone is not a grip: the README records a hand pressed on a wall reading
        >20 N through the pad and 86 N through the wrist with the finger shut on
        nothing.

        And seated means on a SEAT: the throat at most SEAT_TOL past the last full seat
        of a rung it touches. A hook half off a rung's end carries the robot, in the
        simulator, often enough to be learnt: gpu1 landed lesson 2's leading hook
        there (throat p50 at the rung's end) and was banked, graded and paid for it --
        and there the simplified robot it trained on reads its hook 160 N (p50) heavier
        than the full one at the same states, so what it learnt did not transfer.
        """
        out = []
        for s in 'LR':
            load, on = self._contact(s)
            p = self._throat_true(s)
            seat = any(abs(float((p - c) @ a)) <= h - SEAT_IN + SEAT_TOL
                       for c, a, h in (self.segs[r] for r in on))
            out.append(load > HOLD_LOAD and self._gripping(s) and seat)
        return np.array(out)

    def _dist(self):
        return float(self._hand_dists().sum())

    def _off_band_under_side_load(self, push=0.55, steps=150):
        """fingers squeezed onto their stop, the torso pushed `push` x body weight
        along the rung (60 N on the 109 N robot it was measured on) so one hook hands
        its weight to the other: how many steps does a hook still on the rung read
        outside GRIP_BAND? The stop's margin is what keeps this at 0."""
        push *= self.weight
        m, d = self.env.model, self.env.data
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_upper')
        shut = np.zeros(17)
        shut[[JOINTS.index('finger_L'), JOINTS.index('finger_R')]] = -1.0
        out = 0
        for sign in (1.0, -1.0):
            self.reset()
            d.xfrc_applied[tb] = [0.0, sign * push, 0.0, 0.0, 0.0, 0.0]
            try:
                for _ in range(steps):
                    self.step(shut)
                    out += sum(not self._gripping(sd) for sd in 'LR'
                               if self.env.hand(sd)['pad'] > 2.0)
            finally:
                d.xfrc_applied[tb] = 0.0
        return out

    def _success(self):
        """both hooks seated on a goal rung, near their goal points.

        Same seat test as the reward, on purpose: when this used `env.latched` and
        the reward used something else, the dense signal and the thing being scored
        disagreed, which is how the first runs came to be graded on a box the model
        README already says reads False on a hook carrying 167 N.

        And each hand must have taken MIN_GRABS real holds along the way. That turns
        success from a state into a state plus a method, which is normally the wrong
        move -- but the whole task is reachable without it: the 10M policy traversed
        1 m with the right hook never further than 24 mm from the rung, ratcheting
        along it, and the left hand hanging in the air carrying under 10 N for 89 %
        of the episode. Alternating grips is not one way of doing this task, it IS
        the task.
        """
        if (self.grabs < MIN_GRABS).any():
            return False
        held = self._holding()
        for i, s in enumerate('LR'):
            if not held[i] or self._contact(s)[1].isdisjoint(self.spec['goal_rungs']):
                return False
            if np.linalg.norm(self._throat_true(s) - self.goals[i]) > LATCH_TOL:
                return False
        return True

    # ---- episode -----------------------------------------------------------
    def reset(self, seed=None):
        self.env.reset()
        self.ctrl = self.env.data.ctrl.copy()
        # secure: the keyframe leaves the fingers open, close them like the
        # classical controller's secure() before handing the robot to the policy
        fi = [JOINTS.index('finger_L'), JOINTS.index('finger_R')]
        n = int(0.4 / self.dt)
        for i in range(n):
            for j in fi:
                self.ctrl[j] = GRIP_SET * (i + 1) / n
            self.env.step(self.ctrl)
        # this episode's map noise, sensor offsets and SLAM error, and the map: what
        # the lidars see from the hang (the robot has hung there a while)
        self.ep_z = self._normals(episode_noise(len(self.segs)))
        self.seen = np.zeros((len(self.segs), lidar.K), bool)
        self._look()
        self._slam_err = SLAM_SIG * self.ep_z['slam']
        self._k = 0                          # senses this episode (`_sense`)
        self.hold = {s: self._hung(s) for s in 'LR'}
        self.anchor = 'L'
        self._both_conf = False
        self._anchor_fresh = True
        self._w_sum[:] = 0.0
        self._sense()
        if self.tau_idle is None:            # deterministic: keyframe + the same secure()
            self.tau_idle = float(np.abs(self.true['tau']).sum())
            self.fight_idle = self._internal()
        self.prev_d, self.prev_live = self._hand_dists(), self._live()
        self.credit = np.zeros(2)            # relative approach earned since the last hold
        self.held = self._holding()
        self.prev_thr = np.array([self._throat_true(s) for s in 'LR'])
        self.prev_loaded = np.array([self._contact(s)[0] > HOLD_LOAD for s in 'LR'])
        self.let_go = np.zeros(2, bool)      # has this hand been clear since anchoring
        self.grabs = np.zeros(2, int)        # real holds taken, per hand, this episode
        # the line the hands travel along, and which way round they start on it
        self.cross_axis = self.segs[self._seat('L')[0]][1]
        sep = self._throat_true('L') - self._throat_true('R')
        self.cross_sign = float(np.sign(sep @ self.cross_axis)) or 1.0
        self.slide_ref = [None, None]     # each hook's loaded run: (rung, where along it it began)
        self.slide_net = np.zeros(2)      # ...and how far from there it is now
        self.off_t = np.zeros(2)          # TOTAL control steps each hand has been off
        self.off_run = np.zeros(2)        # consecutive control steps each hand has been off
        self.held_t = np.zeros(2)         # steps since each hand was last fully clear
        self.n_steps = 0
        self._flight = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.ctrl = np.clip(self.ctrl + ACT_SCALE * a, self.lo, self.hi)
        self._flight = 0
        for _ in range(FRAME_SKIP):
            self.env.step(self.ctrl)
        self._sense()
        con = [self._contact(s) for s in 'LR']
        touch = np.array([c[0] for c in con])
        d, live = self._hand_dists(), self._live()
        free, held = touch <= 0.0, self._holding()
        # a loaded hook dragged along its rung: metres, either direction, truth
        thr = np.array([self._throat_true(s) for s in 'LR'])
        dp = thr - self.prev_thr
        # -- NET, over each loaded run: charged as the hook gets further from where the
        # run began and refunded as it comes back. Charged step by step, both ways,
        # it billed jitter: a loaded hook's throat rocks 0.1-0.3 mm a step, every step,
        # and 16-62 % of the slide charged on four CEM-found climbs was that (0.8-2.4 a
        # climb). A drag is still charged in full -- its net is its length -- so
        # K_SLIDE > K_BANK still closes 'reach free, get dragged back loaded'.
        loaded, slide = touch > HOLD_LOAD, 0.0
        for k, (_, on) in enumerate(con):
            if loaded[k] and self.prev_loaded[k] and on:
                rung = next(iter(on))
                ax = self.segs[rung][1]
                if self.slide_ref[k] is None or self.slide_ref[k][0] != rung:
                    self.slide_ref[k], self.slide_net[k] = (rung, float(self.prev_thr[k] @ ax)), 0.0
                net = abs(float(thr[k] @ ax) - self.slide_ref[k][1])
                slide += net - self.slide_net[k]
                self.slide_net[k] = net
            else:
                self.slide_ref[k], self.slide_net[k] = None, 0.0
        self.prev_thr, self.prev_loaded, was_loaded = thr, loaded, self.prev_loaded
        self.n_steps += 1
        self.off_t += free                      # cumulative: never reset by a touch
        self.off_run = np.where(free, self.off_run + 1, 0)   # consecutive steps clear
        self.held_t = np.where(free, 0.0, self.held_t + 1)
        grace = DANGLE_GRACE / (FRAME_SKIP * self.dt)
        parked = float(np.sum(free & (self.off_t > grace)
                              & (self.off_t > DANGLE_MAX_FRAC * self.n_steps)))
        # ...but only while the other hand is loaded, when letting go is possible at
        # all: a hand that holds while its partner is in the air has no choice, and
        # was charged for it through every reach, climb and swing (0.1-0.3 each)
        stale = float(np.sum((self.held_t > STALE_AFTER / (FRAME_SKIP * self.dt)) & loaded[::-1]))
        # A hand re-anchors only after it has genuinely let go and taken hold again.
        # Without the let_go gate, a weight shift that drops the wrist load counts as
        # taking a new hold, and a hand that never leaves the rung banks progress --
        # which is what the 5.1M policy was doing on both hands.
        flight = self._flight > 0
        if not flight:
            self.let_go |= free             # letting go while the other hand holds
        # ...and while the OTHER hand still carries the weight: three points of
        # contact. A hold taken with nothing else loaded was a jump's landing or a
        # hand-over with a gap in it, and the gap is where the map goes wrong: on
        # the 3M policy 86 of 129 anchor switches had no step with both hands
        # confirmed and baked in 62 mm each. What is paid is what can be seen.
        took = held & ~self.held & self.let_go & loaded[::-1]
        self.let_go &= ~took
        self.grabs += took
        first = took & (self.grabs == 1)        # each hand's first real hold this episode
        # Progress, RELATIVE to the hand that holds (see the block above the class).
        # A light hand whose partner is loaded is credited its approach minus the
        # partner's own motion toward that goal, both ways; a loaded hook is
        # credited nothing either way; a hand nobody carries (a jump) is credited
        # nothing forward and debited in full back, and since let_go is not granted
        # during a jump it lands as no cycle. Banked at the next supported hold.
        #
        # "Loaded" here means loaded at BOTH ends of the step -- the same test the
        # slide charge uses, on purpose: every motion of a hand's rung point is then
        # either a charged slide or a two-way potential difference, and no cycle
        # between the two pays (back as a slide costs K_SLIDE, forward as credit
        # earns K_BANK, K_SLIDE > K_BANK). Classifying by ONE end of the step always
        # leaves a mirror pump: by the end, a one-step lift (forward on the lift,
        # credited; back on the touch-down, ignored) paid 578 holds and 20 m per
        # episode; by the start, the mac14 rock (back on the lift, ignored; forward
        # on the touch-down, credited) paid 698 holds and 23 m -- the hook rocking
        # on the rung at 12.5 Hz, its throat 31 mm along per step. Both are in the
        # jump/drag test. And the partner's motion is subtracted only while the
        # partner is loaded at both ends -- a hook that loads and unloads on
        # alternate steps could otherwise creep backwards and credit the other
        # hand the difference.
        approach = self.prev_d - d
        was = was_loaded
        both = was & loaded
        credit = np.zeros(2)
        for k in (0, 1):
            o = 1 - k
            if both[k]:
                continue
            if both[o]:
                u = self.goals[k] - thr[k]
                u /= max(float(np.linalg.norm(u)), 1e-9)
                credit[k] = approach[k] - float(dp[o] @ u)
            else:
                credit[k] = min(0.0, approach[k])
        self.credit += credit
        bank = float(self.credit[took].sum())
        self.credit[took] = 0.0
        sep_err = self._sep_error(con)
        effort = max(0.0, float(np.abs(self.true['tau']).sum()) - self.tau_idle)
        fight = max(0.0, self._internal() - self.fight_idle)
        r = (K_BANK * bank + K_GRAB * float(first.sum())
             + K_REACH * float((self.prev_live - live).sum())
             - K_DANGLE * parked - K_STALE * stale - K_SEP * sep_err
             - K_EFFORT * effort - K_FIGHT * fight - K_TIME - K_FLIGHT * flight
             - K_SLIDE * slide
             - K_ACT * float(a @ a) - K_VEL * float(self.true['dq'] @ self.true['dq']))
        self.held, self.prev_d, self.prev_live = held, d, live
        info = {'dist': float(d.sum()), 'live': float(live.sum()), 'lesson': self.lesson,
                'steps': self.n_steps,
                'off': float(free.sum()), 'parked': parked, 'stale': stale,
                'touch': touch.copy(), 'flight': float(flight), 'bank': bank,
                'loaded_start': was.copy(),
                'first': float(first.sum()),
                'credit': credit.copy(), 'slide': slide,
                'effort': effort, 'fight': fight,
                'banked': float(self.credit.sum()), 'd': d.copy(), 'sep_err': sep_err,
                'grabs': self.grabs.copy()}
        obs = self._obs()
        terminated, truncated = False, False
        if self._success():
            r += SUCCESS_R
            terminated, info['reason'] = True, 'success'
        else:
            done, why = self.env.terminated()
            # An observation that is not finite is a blowup as well: the estimator's
            # attitude is carried from step to step, so one degenerate IMU triad
            # (a zero accelerometer in free fall) would feed NaN to the learner for
            # the rest of the episode, and one NaN in VecNormalize's statistics ends
            # the whole run. The GPU path's thousands of envs make rare events common.
            if not np.isfinite(obs).all():
                done, why = True, 'blowup'
            if done:
                info['reason'] = why
                if why == 'timeout':
                    truncated = True
                else:
                    # a blowup is paid FAIL_R alone: the books of an exploded state
                    # (a teleport's distance, a 1e10 rad/s joint) are not a reward
                    r = FAIL_R if why == 'blowup' else r + FAIL_R
                    terminated = True
        return np.where(np.isfinite(obs), obs, np.float32(0.0)), r, terminated, truncated, info


def main():
    for k in sorted(LESSONS):
        t = ClimbTask(k)
        o, _ = t.reset()
        assert np.isfinite(o).all()
        info = {}
        for _ in range(50):                       # 2 s of hanging still
            o, r, te, tr, info = t.step(np.zeros(17))
            assert np.isfinite(o).all()
            assert not te and not tr, (k, info)
        assert t._confirmed('L') and t._confirmed('R'), k
        print(f"lesson {k} {t.spec['name']:>20}: obs {t.obs_dim}, hang stable, "
              f"dist {info['dist']:.2f} m")
    t, rng = ClimbTask(1), np.random.default_rng(0)
    t.reset()
    ends = []
    for _ in range(300):
        o, r, te, tr, info = t.step(rng.uniform(-1, 1, 17))
        assert np.isfinite(o).all() and np.isfinite(r)
        if te or tr:
            ends.append(info.get('reason'))
            t.reset()
    print(f"random rollout: 300 steps, episode ends: {ends or 'none'}")

    # A NaN estimate -- what a zero accelerometer's triad leaves in the attitude it
    # carries -- ends the episode as a blowup at FAIL_R with a finite observation,
    # never hands NaN to the learner (rl.mjx_check holds the GPU path to the same)
    t.reset()
    t._Rf = np.full((3, 3), np.nan)
    o, r, te, tr, info = t.step(np.zeros(17))
    assert info.get('reason') == 'blowup' and r == FAIL_R and te and np.isfinite(o).all(), \
        (info.get('reason'), r, te)
    print("blowup test: a NaN estimate ends the episode at FAIL_R, observation finite")

    # Falling must never be the cheap way out. A full episode of hanging still is the
    # most a policy can pay for doing nothing; if that exceeds FAIL_R, the gradient
    # at the start of training points at the floor. It did: the per-step nudges
    # summed to -33 over 60 s against a -10 fall.
    t = ClimbTask(1)
    t.reset()
    idle = 0.0
    while True:
        _, r, te, tr, info = t.step(np.zeros(17))
        idle += r
        if te or tr:
            break
    print(f"incentive test: hanging still through the whole of lesson 1 costs "
          f"{idle:+.2f}; a fall costs {FAIL_R:+.0f} plus the steps before it")
    assert info.get('reason') == 'timeout' and idle > FAIL_R, (
        f"doing nothing ({idle:+.2f}) costs more than falling ({FAIL_R:+.0f})")

    # The reward change, tested against the exploit it exists for. This is phase 7's
    # own experiment: shove the robot along the rung with both hooks on and the hook
    # skates, because along its axis it is held by friction alone. The distance to
    # the goals falls the whole way; the reward must not follow it.
    # Two guards against the exploit that solved lesson 1 without climbing, one
    # physical and one in the reward. The grip pad cuts the slide to ~16 mm under
    # 40 N (`rl.lessons` asserts that directly) but deliberately does not abolish
    # it -- a climber slips too. So the reward has to hold the line as well: over
    # every step where both hooks are seated, whatever distance the body does
    # manage to close must pay exactly nothing.
    # Side pulls here are a fraction of body weight -- 25 N on the 109 N robot these
    # tests were written on. The same 25 N drags the 215 N robot's hooks 15 mm, too
    # little to prove anything, because what holds a hook along its rung is friction,
    # and friction scales with the weight it carries.
    t = ClimbTask(1)
    t.reset()
    shove = 0.23 * t.weight
    base = mujoco.mj_name2id(t.env.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    t.env.data.xfrc_applied[base] = [0, shove, 0, 0, 0, 0]
    moved_while_held, held, n, slid, paid = 0.0, 0, 0, 0.0, 0.0
    prev = t._dist()
    for _ in range(150):
        _, r, te, tr, info = t.step(np.zeros(17))
        slid += info['slide']
        paid += info['bank']
        now = t._dist()
        if info['off'] == 0:                    # both hooks seated this step
            moved_while_held += abs(prev - now)
            held += 1
        prev = now
        n += 1
        if te or tr:
            break
    t.env.data.xfrc_applied[base] = np.zeros(6)
    banked = paid + max(0.0, info['banked'])
    print(f"slide test: {shove:.0f} N along the rung, both hooks seated for {held}/{n} steps, "
          f"body moved {moved_while_held*1000:.0f} mm over those -- "
          f"credited {banked*1000:+.1f} mm, loaded hooks dragged {slid*1000:.0f} mm "
          f"and charged {K_SLIDE*slid:.2f}")
    assert slid > 0.02 and 0.5 < slid / max(moved_while_held, 1e-9) < 2.0, (
        f"the hooks were dragged {moved_while_held*1000:.0f} mm under load but "
        f"K_SLIDE saw {slid*1000:.0f} mm")
    assert held > 100, (f"only {held}/{n} steps with both hooks seated: {shove:.0f} N should "
                        f"not take a hook off the rung, so something else is")
    assert paid == 0.0 and abs(info['banked']) <= max(0.01, 0.2 * moved_while_held), (
        f"sliding with both hooks on banked {paid*1000:+.1f} mm and left "
        f"{info['banked']*1000:+.1f} mm of credit pending against {moved_while_held*1000:.0f} mm "
        f"of body motion: the shove is being paid for again. The lighter hook may be "
        f"credited what it moved RELATIVE to the heavier one, which reverses with the "
        f"shove; the body's motion itself must not be")

    # The dangle penalty must charge a hand PARKED off the wall and leave a hand
    # that is mid-reach alone -- the flat version of this taxed every release and
    # pushed the policy back towards keeping both hooks on and shoving.
    t = ClimbTask(1)
    t.reset()
    grace = int(DANGLE_GRACE / (FRAME_SKIP * t.dt))
    # hold the left hand off by faking its contact force -- step() derives free,
    # held and let_go from that one measurement, so this is the quantity to intercept
    real_con = t._contact
    t._contact = lambda s: (0.0, set()) if s == 'L' else real_con(s)
    charged = []
    for _ in range(grace + 30):
        _, r, te, tr, info = t.step(np.zeros(17))
        charged.append(info['parked'])
        if te or tr:
            break
    assert sum(charged[:grace]) == 0, (
        f"charged {sum(charged[:grace])} steps inside the {grace}-step grace: a "
        f"normal reach is being taxed again")
    assert charged[-1] == 1.0, (
        f"a hand parked for {len(charged)} steps is still free; the grace never "
        f"runs out")
    print(f"dangle test: {grace} steps ({DANGLE_GRACE:.0f} s) of reach are free, "
          f"a hand parked past that is charged")

    # The evasion the 10.4M policy found: 88 short hops kept every stretch under the
    # grace, so a hand off 79 % of the episode paid 0.13 in total. A cumulative
    # budget has to charge that.
    t = ClimbTask(1)
    t.reset()
    real_con = t._contact
    k = {'i': 0}
    def flutter(sd):
        if sd != 'L':
            return real_con(sd)
        # 13 steps off, 4 steps on -- 76 % off, no stretch anywhere near the grace
        return (0.0, set()) if (k['i'] % 17) < 13 else real_con(sd)
    t._contact = flutter
    charged, longest, run = 0, 0, 0
    for _ in range(600):
        k['i'] += 1
        _, _, te, tr, info = t.step(np.zeros(17))
        charged += info['parked'] > 0
        run = run + 1 if info['off'] else 0
        longest = max(longest, run)
        if te or tr:
            break
    print(f"flutter test: a hand off 76% of the episode in hops of at most "
          f"{longest} steps (grace {grace}) is charged on {charged} steps")
    assert longest < grace, "the test is not fluttering -- a stretch outlasted the grace"
    assert charged > 100, (f"only {charged} steps charged: short hops still dodge the "
                           f"dangle budget, which is what the 10.4M run exploited")

    # And its mirror: a hand that never lets go must start costing. Across every run
    # measured, the left hand has never once released.
    t = ClimbTask(1)
    t.reset()
    stale_at, n = None, int(STALE_AFTER / (FRAME_SKIP * t.dt))
    for i in range(n + 60):
        _, _, te, tr, info = t.step(np.zeros(17))
        if info['stale'] > 0 and stale_at is None:
            stale_at = i + 1
        if te or tr:
            break
    print(f"stale test: hanging still, a hand that never releases is charged from "
          f"step {stale_at} (threshold {n} = {STALE_AFTER:.0f} s)")

    # Effort and internal force must both read ~zero for a robot that is simply
    # hanging. If existing costs, the cheapest policy is to stop existing.
    t = ClimbTask(1)
    t.reset()
    eff, fig = [], []
    for _ in range(200):
        _, _, te, tr, info = t.step(np.zeros(17))
        eff.append(info['effort']); fig.append(info['fight'])
        if te or tr:
            break
    eff, fig = np.mean(eff), np.mean(fig)
    print(f"idle test: hanging still costs {K_EFFORT*eff:.4f}/step in effort and "
          f"{K_FIGHT*fig:.4f}/step in internal force")

    # An open hand must not count as holding. The 10.4M policy held the robot with
    # the finger at +20 deg for most of an episode and banked progress doing it,
    # because nothing checked the finger -- and there are two ways out of the band,
    # opening AND over-closing past the -45 deg cliff.
    t = ClimbTask(1)
    t.reset()
    fi = JOINTS.index('finger_L')
    assert t._holding()[0], "the hang keyframe should start out holding"
    t.reset()
    act = np.zeros(17)
    act[fi] = +1.0
    for _ in range(120):
        _, _, te, tr, _ = t.step(act)
        ang = np.degrees(t.env.sensor('enc_finger_L')[0])
        if not (np.degrees(GRIP_BAND[0]) <= ang <= np.degrees(GRIP_BAND[1])) or te or tr:
            break
    print(f"grip test (mở toang): ngón {ang:+.0f} deg, tiếp xúc {t._contact('L')[0]:.0f} N "
          f"-> holding = {t._holding()[0]}")
    assert not t._holding()[0], f"a hand opened to {ang:+.0f} deg still counts as holding"

    # ...nor a hook hanging off the end of its rung (SEAT_TOL): the start row cut short
    # 20 mm inside the left hook's last full seat, then 4 mm past it -- loaded and
    # gripping both times, holding only the second
    import os
    from .lessons import OUT, _ry, scene_xml
    yL = float(t._throat_true('L')[1])
    fp = os.path.join(OUT, 'check_seat.xml')
    seat = {}
    for cut, want in ((-0.020, False), (0.004, True)):
        spec = dict(LESSONS[1], rungs=[_ry('seat_check', -0.90, yL + SEAT_IN + cut, 1.00)])
        open(fp, 'w').write(scene_xml(spec, 'check_seat'))
        ts = ClimbTask(1, visual=False, xml=fp)
        ts.reset()
        load, ok = ts._contact('L')[0], ts._holding()[0]
        assert load > HOLD_LOAD and ts._gripping('L'), f"the seat check's hook is not loaded ({load:.0f} N)"
        assert ok == want, f"a hook {cut * 1e3:+.0f} mm from its last full seat: holding {ok}"
        seat[cut] = (load, ok)
    print(f"seat test: a hook 20 mm past its last full seat, {seat[-0.020][0]:.0f} N on the rung, "
          f"is not holding; 4 mm inside it, {seat[0.004][0]:.0f} N, is")
    # squeezing as hard as possible must STAY in the band, loaded and swinging: the
    # stop is at the band's edge, not at the cliff. Random actions on every other
    # joint swing the body; only the fingers are held shut.
    t.reset()
    rng = np.random.default_rng(7)
    inband, n_loaded = 0, 0
    for k in range(200):
        act = rng.uniform(-0.6, 0.6, 17)
        act[fi] = -1.0
        act[JOINTS.index('finger_R')] = -1.0
        _, _, te, tr, _ = t.step(act)
        for sd in 'LR':
            if t._contact(sd)[0] > HOLD_LOAD:
                n_loaded += 1
                inband += t._gripping(sd)
        if te or tr:
            break
    print(f"grip test (siết hết cỡ): {inband}/{n_loaded} loaded steps in band while swinging")
    assert n_loaded > 50 and inband >= 0.97 * n_loaded, (
        f"squeezing shut left the band on {n_loaded - inband} of {n_loaded} loaded "
        f"steps: the finger stop is outside the band")
    off = t._off_band_under_side_load()
    print(f"grip test (tay nhả tải): fingers on the {np.degrees(GRIP_STOP):.0f} deg stop, "
          f"{0.55 * t.weight:.0f} N along the rung both ways -- {off} step(s) out of band on a hook still on the rung")
    assert off == 0, (f"a hook on its stop read outside GRIP_BAND on {off} steps once the other "
                      f"hook took the weight: the stop has no margin inside the band")

    # And success must need real grabs, not just arriving. Geometry is stubbed to
    # the goal so only the grab count and the finger are left to decide.
    t = ClimbTask(1)
    t.reset()
    t._contact = lambda s: (HOLD_LOAD * 5, {t.spec['goal_rungs'][0]})
    t._throat_true = lambda s: t.goals[0] if s == 'L' else t.goals[1]
    t.grabs = np.zeros(2, int)
    assert not t._success(), "arriving with zero grabs counts as success"
    t.grabs = np.array([MIN_GRABS, 0])
    assert not t._success(), "one hand having grabbed is enough for success"
    t.grabs = np.array([MIN_GRABS, MIN_GRABS])
    assert t._success(), "seated, gripping and both hands grabbed should succeed"
    print(f"grab test: success needs {MIN_GRABS} real hold(s) from EACH hand; "
          f"0 and one-sided are both refused")
    assert K_EFFORT * eff < 0.002, (
        f"a still hang is charged {K_EFFORT*eff:.4f}/step of effort -- existing must "
        f"be free, or ending the episode becomes the cheapest policy")
    assert K_FIGHT * fig < 0.002, (
        f"a still hang is charged {K_FIGHT*fig:.4f}/step of internal force; the two "
        f"hooks together carry about body weight and that must read as no fighting")
    assert stale_at is not None, "a hand held for the whole episode is never charged"
    assert stale_at > n - 2, f"charged at step {stale_at}, before the {n}-step threshold"

    # A hand whose wrist load merely dips -- a weight shift -- must not count as
    # having taken a new hold. On the 5.1M policy every anchor re-take came from
    # exactly that, with the hook never leaving the rung.
    t = ClimbTask(1)
    t.reset()
    paid, flick = 0.0, [True]
    t._holding = lambda: np.array([flick[0], True])   # left hold flickers on load
    for i in range(80):
        flick[0] = (i // 5) % 2 == 0
        _, _, te, tr, info = t.step(np.zeros(17))
        paid += abs(info['bank'])
        if te or tr:
            break
    moved = paid
    print(f"let-go test: a hold flickering {8} times without the hook leaving the "
          f"rung banked {moved*1000:.1f} mm")
    assert moved < 1e-9, (
        f"the anchor moved {moved*1000:.1f} mm on load flicker alone: a hand that "
        f"never let go is banking progress again")

    # Crossing must read zero in the normal hang and grow once the hands swap sides --
    # on the rung both hooks are on. With no rung shared nothing can collide: the
    # hands on two rows (a climb) or one in the air (a swing) are charged nothing.
    t = ClimbTask(1)
    t.reset()
    con = [t._contact(s) for s in 'LR']
    at_hang = t._sep_error(con)
    sep = t.cross_sign * float(
        (t._throat_true('L') - t._throat_true('R')) @ t.cross_axis)
    real = t._throat_true
    # inverted: the hands swap sides. together: both hands on the same point --
    # the case that only pricing "crossed" missed, and that dropped the robot.
    t._throat_true = lambda s: real('R') if s == 'L' else real('L')
    inverted = t._sep_error(con)
    t._throat_true = lambda s: real('R')
    together = t._sep_error(con)
    t._throat_true = real
    apart = t._sep_error([con[0], (con[1][0], {'rung_other'})])   # on two different rungs
    flying = t._sep_error([(0.0, set()), con[1]])                  # the left one in the air
    t._throat_true = lambda s: real('R') if s == 'L' else real('L')
    flying_crossed = t._sep_error([(0.0, set()), con[1]])
    t._throat_true = real
    print(f"separation test: hands {sep*1000:+.0f} mm apart -> penalty "
          f"{at_hang*1000:.0f} mm | swapped -> {inverted*1000:.0f} mm | "
          f"on the same point -> {together*1000:.0f} mm | on two rungs, or one hand in the air "
          f"past the other -> {max(apart, flying, flying_crossed)*1000:.0f} mm")
    assert sep > MIN_SEP, "the hang keyframe starts inside the separation floor"
    assert at_hang == 0.0, f"the hang posture is charged {at_hang*1000:.0f} mm"
    assert abs(together - MIN_SEP) < 1e-9, (
        f"two hands on the same point are charged {together*1000:.0f} mm, expected "
        f"{MIN_SEP*1000:.0f} -- this is the case that made the second grab fail")
    assert inverted > together, "swapping the hands must cost more than merging them"
    assert apart == flying == flying_crossed == 0.0, "hands that share no rung were charged for their separation"

    # The attitude filter must beat the bare triad while the body moves, and must
    # not drift while it hangs still. Random actions swing the body hard.
    t = ClimbTask(1, noise=False)             # the filter, not the IMU's offsets
    t.reset()
    torso = mujoco.mj_name2id(t.env.model, mujoco.mjtObj.mjOBJ_BODY, 'torso_upper')
    rng = np.random.default_rng(5)
    e_f, e_t = [], []
    for k in range(200):
        _, _, te, tr, _ = t.step(rng.uniform(-0.7, 0.7, 17) if k < 150 else np.zeros(17))
        R_true = t.env.data.xmat[torso].reshape(3, 3)
        e_f.append(np.linalg.norm(rotvec(t._Rf.T @ R_true)))
        e_t.append(np.linalg.norm(rotvec(triad(t.imu_s['accel'], t.imu_s['mag']).T @ R_true)))
        if te or tr:
            break
    rf, rt = np.degrees(np.sqrt(np.mean(np.square(e_f)))), np.degrees(np.sqrt(np.mean(np.square(e_t))))
    print(f"ahrs test: over {len(e_f)} steps of random swinging the torso attitude is "
          f"{rf:.1f} deg rms off filtered against {rt:.1f} from the triad alone")
    assert rf < 0.5 * rt, f"the filter ({rf:.1f} deg) is not clearly better than the triad ({rt:.1f})"
    assert np.degrees(e_f[-1]) < 3.0, f"still hanging, the filter reads {np.degrees(e_f[-1]):.1f} deg off"

    # Odometry by reach: the point on a rung exactly L from the anchor's hold, with
    # the IMU guess deciding only which side. Exact where the guess is 148 mm off.
    t = ClimbTask(2, noise=False)            # the map exact: the geometry is the test
    t.reset()
    t.seen[:] = True                         # ...all of it: from the hang the lidars miss
    t._look()                                # seg1's near 2 cm, behind the left upper arm
    c, a, h = t.segs['rung_start']
    pA = c - 0.10 * a                          # the anchor, on the start rung
    T = c + 0.15 * a                           # the truth: 0.25 m further along it
    bad = T + np.array([0.10, 0.08, -0.09])    # a guess 157 mm off, wrong in x and z
    nm, q = t._reach_point(pA, 0.25, bad)
    assert nm == 'rung_start' and np.linalg.norm(q - T) < 1e-9, (nm, q - T)
    c1, a1, h1 = t.segs['rung_seg1']
    T1 = c1 - (h1 - 0.02) * a1                 # 20 mm in from the near end of seg1
    L1 = float(np.linalg.norm(T1 - pA))
    nm, q = t._reach_point(pA, L1, T1 + np.array([0.12, -0.06, 0.05]))
    assert nm == 'rung_seg1' and np.linalg.norm(q - T1) < 1e-9, (nm, q - T1)
    nm2, q2 = t._reach_point(pA, 0.25, T - 0.5 * a)
    assert np.linalg.norm(q2 - T) > 0.4, "the guess must be able to choose the other side"
    assert t._reach_point(pA, 0.25, T + 0.6 * a) is None, "a guess 0.6 m off was believed"
    print(f"reach test: a hold 0.25 m along the rung is placed to "
          f"{np.linalg.norm(q - T1)*1e3:.4f} mm from a guess 157 mm off; across the "
          f"gap onto seg1 too; a guess on the far side picks the far side")

    # A rung is a line, not an arrow: the observation may not depend on which end a
    # scene names first. MuJoCo points a fromto geom's axis at its first point, and
    # before rl.lessons.canon_axis every benchmark rung reached the policy reversed
    # against the generated walls it trains on (gpu1's pilot, full robot, its own
    # training walls: 3/24 -> 21/24, as in MJX). Lesson 7: a front and a side rung.
    import os, re
    from .lessons import OUT
    src = open(xml_path(7)).read()
    rev = re.sub(r'(<geom name="rung_[^"]*"[^>]*fromto=")([^"]*)"',
                 lambda g: g.group(1) + ' '.join(g.group(2).split()[3:] + g.group(2).split()[:3]) + '"', src)
    p = os.path.join(OUT, 'check_fromto_reversed.xml')
    open(p, 'w').write(rev)
    obs = []
    for xml in (None, p):
        t = ClimbTask(7, xml=xml)
        t.rng = np.random.default_rng(0)
        obs.append(t.reset()[0])
    dev = float(np.abs(obs[0] - obs[1]).max())
    assert rev != src and dev < 1e-6, f"the same rungs written end for end change the observation by {dev:.2e}"
    print(f"axis test: lesson 7 with every rung written end for end observes the same ({dev:.1e})")

    # SLAM keeps the anchor honest -- where it can. The loaded hook creeps along the rung
    # under a side load and the arm cannot see it: hang, then pull the body along the
    # rung for 150 steps (the slide test's scenario and force, the hooks drag 25-50 mm).
    # In the corner (lesson 7) a wall faces along the rung and the SLAM says where the
    # body is along it, and the anchor follows the creep; on a flat wall (lesson 1) it
    # cannot say (`along_seen`), and the anchor stays where it was, stale by the creep.
    # With the sensors exact, so this is the mechanism; the sensor test has the noise.
    z = np.zeros(17)

    def pull(k):
        t = ClimbTask(k, noise=False)
        t.reset()
        ax = t.segs['rung_start'][1]
        stale = lambda: float((t._throat_true(t.anchor) - t.hold[t.anchor][1]) @ ax)
        for _ in range(50):
            t.step(z)
        wander = []
        for _ in range(100):
            t.step(z)
            wander.append(abs(stale()))
        p0 = t._throat_true(t.anchor).copy()
        for _ in range(150):
            t.env.data.xfrc_applied[t.env.base, :3] = 0.23 * t.weight * ax
            t.step(z)
        t.env.data.xfrc_applied[t.env.base, :3] = 0.0
        return t, ax, stale, max(wander), float((t._throat_true(t.anchor) - p0) @ ax)

    t, ax, stale, wander, crept = pull(1)
    flat = stale()
    assert abs(crept) > 0.015, f"the pull crept the hook only {crept*1e3:.0f} mm; the test proves nothing"
    assert abs(flat - crept) < 0.4 * abs(crept), \
        f"on a flat wall the anchor moved with the creep ({flat*1e3:+.0f} of {crept*1e3:+.0f} mm): who told it?"
    t, ax, stale, wander, crept = pull(7)
    # (the wander is the filtered attitude's ~1 deg on the lever from the anchor to
    # the base. That lever was 0.5 m; the real-actuator robot's arm is 116 mm longer
    # and its torso 104 mm taller, and with white SLAM noise on top the wander read
    # 15-24 mm over seeds 0-9 -- where the old 20 mm bound sat for the old robot's ~12)
    assert wander < 0.030, f"hanging still, the anchor wandered {wander*1e3:.0f} mm"
    assert abs(crept) > 0.015, f"the pull crept the hook only {crept*1e3:.0f} mm; the test proves nothing"
    assert abs(stale()) < 0.020, f"after {crept*1e3:+.0f} mm of creep the anchor is {stale()*1e3:+.0f} mm stale"
    print(f"slam test: in the corner, hanging still the anchor wanders {wander*1e3:.1f} mm on an exact "
          f"SLAM {SLAM_LAG * CONTROL_DT * 1e3:.0f} ms late, and a {0.23 * t.weight:.0f} N pull creeps the "
          f"loaded hook {crept*1e3:+.0f} mm along the rung and the anchor follows to within "
          f"{stale()*1e3:+.0f} mm; on a flat wall the SLAM cannot see along the rung and the anchor "
          f"is left {flat*1e3:+.0f} mm stale")

    # The sensors read what a real robot's would, and only the robot reads them. Off,
    # the readings are the truth to the bit and the map is the part of each rung the
    # lidars have seen, on its axis (the goals, carried along with their rungs, to
    # rounding); the SLAM's error is SLAM_SIG in size and as correlated frame to frame
    # as SLAM_DRIFT_TAU says; on, hanging still, the anchor goes wherever that drift
    # takes it along the rung -- no filter averages a drift away, which is the point of
    # modelling one.
    assert all(np.array_equal(t.sen[k], v) for k, v in t.true.items())
    for n, (c, a, h) in t.segs.items():
        mc, ma, mh = t.map[n]
        assert (np.array_equal(ma, a) and np.linalg.norm(np.cross(mc - c, a)) < 1e-12
                and abs((mc - c) @ a) + mh <= h + 1e-12 and n in t.map_seen), n
    assert np.allclose(t.map_goals, t.goals, rtol=0, atol=1e-12)
    rng = np.random.default_rng(0)
    e = [SLAM_SIG * rng.standard_normal(3)]
    for _ in range(20000):
        e.append(slam_drift(e[-1], rng.standard_normal(3)))
    e = np.array(e)
    rho = np.mean([np.corrcoef(e[:-1, i], e[1:, i])[0, 1] for i in range(3)])
    assert abs(e.std() / SLAM_SIG - 1.0) < 0.1 and abs(rho - SLAM_RHO) < 0.02, (e.std(), rho)
    t = ClimbTask(7)                           # a corner: the SLAM sees along the rung
    t.rng = np.random.default_rng(0)
    t.reset()
    stale = lambda: float((t._throat_true(t.anchor) - t.hold[t.anchor][1]) @ ax)
    off, slam = [], []
    for _ in range(300):                       # 12 s
        t.step(z)
        off.append(stale())
        slam.append(float(t._slam_err @ ax))
    # (the anchor reads off by minus the drift, plus the lever of the attitude's error,
    # which the IMU's offsets move: over seeds 0-2 the correlation is -0.45..-0.68)
    follow = float(np.corrcoef(off, slam)[0, 1])
    assert follow < -0.3 and max(map(abs, off)) < 0.1, (follow, max(map(abs, off)))
    print(f"sensor test: exact when off; SLAM error {e.std()*1e3:.1f} mm, fix-to-fix correlation {rho:.3f} "
          f"({SLAM_RHO:.3f}); on, hanging still 12 s, the anchor is up to {max(map(abs, off))*1e3:.0f} mm off "
          f"along the rung, following the drift (correlation {follow:+.2f})")

    # The map holds what the four Mid-360 can see (rl.lidar). Straight along a lidar's
    # axis is its 38 deg blind cone and nearer than 0.1 m its blind zone, whatever is
    # there; from the hang every lesson's start rung is in the map; and a map only
    # grows -- a rung seen once stays, and one hidden behind the arms joins it once
    # the arms move.
    t = ClimbTask(6, noise=False)
    t.reset()
    d, X = t.env.data, t.env.data.site_xmat.reshape(-1, 3, 3)
    import copy
    for s_ in t.rig.lidar:
        one = copy.copy(t.rig)
        one.lidar = np.array([s_])                 # this lidar alone
        o, A = d.site_xpos[s_], X[s_][:, 2]
        side = np.cross(A, [0.0, 0.0, 1.0] if abs(A[2]) < 0.9 else [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        for p_ in (o + 3.0 * A, o + 0.07 * (A + side) / np.linalg.norm(A + side)):
            v = lidar.visible(np, one, d.site_xpos, d.site_xmat, p_[None], side[None],
                              np.array([0.01]), np.array([0.015]), np.array([True]))
            assert not v.any(), (t.env.model.site(s_).name, p_)
    start = {k: ClimbTask(k, noise=False) for k in sorted(LESSONS)}
    for k, tk in start.items():
        tk.reset()
        assert tk._seat('L')[0] in tk.map_seen, f"lesson {k}: the start rung is not in the map"
    first, rng = t.seen.copy(), np.random.default_rng(1)
    for _ in range(150):
        before = t.seen.copy()
        _, _, te, tr, _ = t.step(rng.uniform(-0.6, 0.6, 17))
        assert (t.seen >= before).all(), "the map forgot a cell"
        if te or tr:
            break
    cov = lambda sn: ", ".join(f"{nm[5:]} {100 * v.mean():.0f} %" for nm, v in zip(t.segs, sn))
    assert t.seen.sum() > first.sum(), "150 steps of flailing showed the lidars nothing new"
    print(f"lidar test: blind along each axis and inside 0.1 m; every lesson starts with its "
          f"start rung mapped; lesson 6 from the hang: {cov(first)} -- after 6 s of flailing "
          f"{cov(t.seen)}")

    # Progress is measured at the rung. The win9 dive: the left hook off the end of
    # its segment, level with it (y 0.39), then hanging 0.47 m below the far segment
    # at y 0.65 -- nearer the goal in a straight line, further along the wall.
    t = ClimbTask(1.5)
    t.reset()
    g = t.goals[0]
    c0, a0, h0 = t.segs['rung_start']
    a0 = a0 if a0 @ (g - c0) > 0 else -a0        # along the rung, toward the goal
    edge = c0 + h0 * a0
    hover = edge + 0.04 * a0
    hang = edge + 0.30 * a0 - np.array([0.134, 0.0, 0.467])
    assert np.linalg.norm(hang - g) < np.linalg.norm(hover - g), "the dive is no longer nearer in 3D"
    # (through the best rung alone the dive would GAIN 49 mm on the hover, by a straight
    # line to the goal's own segment; AIR's charge on the 0.41 m more of air makes it a
    # 158 mm loss)
    assert t._cost(hang, 0) > t._cost(hover, 0) + 0.05, (t._cost(hang, 0), t._cost(hover, 0))
    on = c0 - 0.10 * a0
    assert abs((t._cost(on, 0) - t._cost(on + 0.30 * a0, 0)) - 0.30) < 1e-9, "progress along the rung must be paid in full"
    assert t._cost(hover, 0) < t._cost(edge, 0) + 0.05, "hovering just past the edge is charged the air, nothing more"
    pulled = on + np.array([0.0, 0.0, -0.03])           # a loaded hook's throat, off its line
    assert abs(np.linalg.norm(t._route(pulled)[1] - g) - np.linalg.norm(t._route(on)[1] - g)) < 1e-12, \
        "the bank must not see a loaded hook pulled off its line (mac14: 698 holds and 56 m banked per episode when it did)"
    # (the live term charges it instead -- AIR a metre at least: in a row of segments the
    # straight line on to the goal's own one barely lengthens for a 30 mm drop)
    assert t._cost(pulled, 0) - t._cost(on, 0) >= AIR * 0.03 - 1e-9, "the live term does not see the detour"
    # ...and a reach for the next rung is progress from its first millimetre. Through
    # the NEAREST rung, a hook lifted off the start row toward the row above paid for
    # the first half of the way: gpu1 never lifted one past 28 mm on family 3, 0 % for
    # 90M steps.
    t3 = ClimbTask(3)
    t3.reset()
    g3 = t3.goals[0]
    zs = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30)
    lift = [t3._cost(np.array([g3[0], g3[1], 1.00 + z]), 0) for z in zs]
    assert all(b < a - (1 - AIR) * (z1 - z0) + 1e-9 for a, b, z0, z1 in zip(lift, lift[1:], zs, zs[1:])), \
        f"lifting toward the row above is not paid: {np.round(lift, 3)}"
    print(f"route test: the win9 dive is {(np.linalg.norm(hover - g) - np.linalg.norm(hang - g))*1e3:+.0f} mm "
          f"of straight-line progress and {(t._cost(hover, 0) - t._cost(hang, 0))*1e3:+.0f} mm along the routes; "
          f"0.30 m along the rung pays 0.30; a loaded hook pulled 30 mm off its line moves the bank 0; "
          f"a hook lifted 0.30 m toward the rows above is paid {lift[0] - lift[-1]:.2f}")

    # ...and no hill anywhere a climber has to go: every lesson, each hand, along its
    # shortest route through the rungs at the robot's own reach (0.65 m; 1.0 across
    # lesson 9's gap, which is swung), the live cost never rises above what it has
    # already come down to. Through the nearest rung it rose 0.15 m on family 3's
    # first walls; through the best rung's nearest point, 0.06-0.09 m on lessons 4, 6
    # and 7, and 0.37 m for a hook lifted straight up at lesson 5's break.
    hills = []
    for k in range(1, 10):
        tk = ClimbTask(k, visual=False)
        tk.reset()
        hills += [(_hill(tk, i, 1.0 if k == 9 else 0.65), k, 'LR'[i]) for i in (0, 1)]
    h, k, sd = max(hills)
    assert h < 0.01, f"lesson {k}, hand {sd}: the live cost climbs {h*1e3:.0f} mm on the way to its goal"
    t5 = ClimbTask(5, visual=False)
    t5.reset()
    c5, a5, h5 = t5.segs['rung_start']
    brk = c5 + (h5 - SEAT_IN) * (a5 if a5 @ (t5.goals[0] - c5) > 0 else -a5)   # the last seat before the break
    up = [t5._cost(brk + np.array([0.0, 0.0, z]), 0) for z in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)]
    assert all(b < a for a, b in zip(up, up[1:])), f"a hook lifted at lesson 5's break is not paid: {np.round(up, 3)}"
    print(f"barrier test: along each lesson's shortest route, both hands, the live cost never climbs "
          f"(worst {h*1e3:.1f} mm, lesson {k} {sd}); at lesson 5's break a hook lifted 0.4 m toward the row "
          f"over it is paid {up[0] - up[-1]:.2f}")

    # Progress must be paid for a supported reach, and for nothing else. Three ways
    # policies moved the body without one, each of which banked under an earlier
    # reward: a JUMP (no hand holding, for even one physics substep) -- the 21M
    # policy's whole gait; a TAP right after one; and a DRAG -- the 2M local run's
    # inchworm, the loaded right hook pulled along the rung while the free left hand
    # tapped its way along. All three must bank 0; the reach banks in full.
    t = ClimbTask(1)
    t.reset()
    real_con, real_d, real_thr = t._contact, t._hand_dists, t._throat_true
    on = {s_: real_con(s_) for s_ in 'LR'}
    st = {'L': on['L'], 'R': on['R'], 'd': real_d().copy(),
          'thr': {s_: real_thr(s_).copy() for s_ in 'LR'}}
    t._contact = lambda s_: st[s_]
    t._hand_dists = lambda: st['d'].copy()
    t._throat_true = lambda s_: st['thr'][s_].copy()
    z = np.zeros(17)
    uL = t.goals[0] - st['thr']['L']
    uL /= np.linalg.norm(uL)
    # a supported reach: left off while the right holds, 0.3 m toward the goal.
    # The lift-off step itself is worth nothing (loaded at its start), so the hand
    # lifts first and then moves.
    st['L'] = (0.0, set())
    t.step(z)
    for _ in range(3):
        st['d'][0] -= 0.1
        st['thr']['L'] += 0.1 * uL
        _, _, _, _, info = t.step(z)
        assert info['flight'] == 0.0, "a one-handed hang is being called a jump"
    st['L'] = on['L']
    _, _, _, _, info = t.step(z)
    assert list(info['grabs']) == [1, 0] and abs(info['bank'] - 0.3) < 1e-9, (
        f"a supported 0.3 m reach banked {info['bank']*1000:.0f} mm and counted "
        f"{info['grabs']} grabs")
    # a jump: both hands off, 0.3 m toward the goal, then both land
    charged, paid = 0.0, 0.0
    st['L'] = st['R'] = (0.0, set())
    for _ in range(3):
        st['d'] -= 0.1
        for s_ in 'LR':
            st['thr'][s_] += 0.1 * uL
        _, _, _, _, info = t.step(z)
        charged += info['flight']
        paid += info['bank']
        assert (info['credit'] <= 1e-12).all(), f"a jump was credited {info['credit']}"
    st['L'], st['R'] = on['L'], on['R']
    _, _, _, _, info = t.step(z)
    paid += info['bank']
    assert charged == 3.0, f"3 jump steps charged {charged:.0f}"
    assert list(info['grabs']) == [1, 0], f"landing from a jump counted a grab: {info['grabs']}"
    assert abs(paid) < 1e-9 and info['banked'] <= 1e-9, (
        f"a 0.6 m jump banked {paid*1000:.0f} mm, {info['banked']*1000:.0f} pending")
    # the tap: the left hand lets go and touches the rung again right after the
    # jump -- a hold, worth nothing
    st['L'] = (0.0, set())
    t.step(z)
    st['L'] = on['L']
    _, _, _, _, info = t.step(z)
    assert list(info['grabs']) == [2, 0], "a supported let-go after a jump must count again"
    assert abs(info['bank']) < 1e-9, f"a tap after a jump banked {info['bank']*1000:.0f} mm"
    # the drag: the right hook stays loaded and is pulled 0.2 m along the rung,
    # carrying the body and the free left hand with it; then the left hand taps
    st['L'] = (0.0, set())
    slid = 0.0
    for _ in range(4):
        st['d'] -= 0.05
        for s_ in 'LR':
            st['thr'][s_] += 0.05 * uL
        _, _, _, _, info = t.step(z)
        slid += info['slide']
        assert abs(info['credit'][0]) < 1e-9 and abs(info['credit'][1]) < 1e-9, (
            f"a drag was credited {info['credit']*1000} mm")
    st['L'] = on['L']
    _, _, _, _, info = t.step(z)
    assert list(info['grabs']) == [3, 0] and abs(info['bank']) < 1e-9, (
        f"the tap after a drag banked {info['bank']*1000:.0f} mm")
    assert slid > 0.15, f"a 0.2 m drag of a loaded hook registered {slid*1000:.0f} mm of slide"
    # the one-step lift, both ways round: the right hand lifts for one step, moving
    # 30 mm toward its goal, and sets down the next, moving 30 mm back -- a pump
    # when the credit was classified by the load at the END of the step (+30 mm per
    # cycle, 578 holds and 20 m per episode). And its mirror, the mac14 rock: back
    # on the lift, forward on the touch-down -- a pump when classified by the
    # START (+31 mm per cycle, 698 holds and 23 m per episode). A step is a slide's
    # business only with load at both ends; every other step is a potential
    # difference both ways, so both cycles net exactly nothing.
    for fwd_on_lift in (True, False):
        st['L'] = on['L']
        t.step(z)
        paid = 0.0; cr = 0.0
        for j in range(10):
            up = j % 2 == 0
            st['R'] = (0.0, set()) if up else on['R']
            dm = 0.03 if (up == fwd_on_lift) else -0.03
            st['thr']['R'] += dm * uL
            st['d'][1] -= dm
            _, _, _, _, info = t.step(z)
            paid += info['bank']; cr += info['credit'][1]
        assert paid <= 1e-9 and cr <= 1e-9, (
            f"the one-step lift ({'forward' if fwd_on_lift else 'back'} on the lift) banked "
            f"{paid*1000:+.0f} mm and was credited {cr*1000:+.0f} mm")
        st['R'] = on['R']
        t.step(z)
        t.credit[:] = 0.0
    # the flutter drag: the holder loads on alternate steps only while creeping
    # backwards; the free hand must not be credited the relative gain, because
    # the partner's motion counts only while it is loaded at both ends of a step
    st['R'] = (0.0, set())
    st['L'] = (5.0, {'rung_start'})               # light at the start of the first creep step
    t.step(z)
    cr = 0.0
    for j in range(12):
        st['L'] = on['L'] if j % 2 == 0 else (5.0, {'rung_start'})   # loaded / light, on the rung
        st['thr']['L'] -= 0.02 * uL                                   # creeping back
        st['d'][0] += 0.02
        _, _, _, _, info = t.step(z)
        cr += info['credit'][1]
    assert cr <= 1e-9, f"a holder creeping back on alternate loads credited the free hand {cr*1000:+.0f} mm"
    st['R'] = on['R']
    t.step(z)
    t.credit[:] = 0.0                             # the creep's debit is not the next case's

    # ordinary physics must not bleed: the holder settling back and forth under a
    # stationary free hand, and two loaded hooks settling together, both net 0.
    # The one-sided draft of this rule charged a known-good one-hand step -220 mm
    # per episode for exactly these.
    st['L'] = (0.0, set())
    net = 0.0
    for j in range(6):
        sgn = 1.0 if j % 2 == 0 else -1.0
        st['thr']['R'] += sgn * 0.02 * uL
        st['d'][1] -= sgn * 0.02
        _, _, _, _, info = t.step(z)
        net += info['credit'][0]
    st['L'] = on['L']
    _, _, _, _, info = t.step(z)
    assert abs(net) < 1e-9 and abs(info['bank']) < 1e-9, (
        f"a holder settling +-20 mm under a stationary free hand credited it {net*1000:+.1f} mm")
    net = 0.0
    for j in range(6):
        sgn = 1.0 if j % 2 == 0 else -1.0
        for s_ in 'LR':
            st['thr'][s_] += sgn * 0.02 * uL
        st['d'] -= sgn * 0.02
        _, _, _, _, info = t.step(z)
        net += float(info['credit'].sum())
    assert abs(net) < 1e-9, f"two loaded hooks settling +-20 mm were credited {net*1000:+.1f} mm"
    print("jump/drag test: a supported 0.3 m reach banks 300 mm; a 0.6 m jump banks 0, "
          "counts no grab and is charged; the tap after it banks 0; a 0.2 m drag of "
          f"the loaded hook credits 0, banks 0 on the next tap and is charged {K_SLIDE*slid:.1f}; "
          "a holder settling +-20 mm credits the free hand 0, two loaded hooks settling credit 0; "
          "a one-step lift nets nothing either way round; a holder creeping on alternate loads credits nothing")

    # No pump, proved rather than hoped. The live term is a potential and must
    # telescope to the endpoints exactly; the bank pays credit that never exceeds
    # the world approach, so what it pays plus what is still pending can never be
    # more than the distance the trajectory actually closed. The first reward
    # failed exactly here: it paid +27.28 over an episode that closed 0.000 m.
    # Random actions, because a pump shows up as the total drifting away.
    t = ClimbTask(1)
    t.reset()
    rng = np.random.default_rng(3)
    d0, acc, paid, forgiven, slid = t._dist(), 0.0, 0.0, 0.0, 0.0
    live0 = float(t._live().sum())
    prev_dh = t._hand_dists().copy()
    for _ in range(400):
        a = rng.uniform(-1, 1, 17)
        _, r, te, tr, info = t.step(a)
        # strip every per-step cost; what is left must be the two potential terms.
        # Each cost added to the reward has to be added here too -- the assert below
        # is what notices when one is forgotten.
        acc += (r + K_DANGLE * info['parked'] + K_STALE * info['stale']
                + K_SEP * info['sep_err'] + K_EFFORT * info['effort']
                + K_FIGHT * info['fight'] + K_TIME + K_FLIGHT * info['flight']
                + K_SLIDE * info['slide']
                + K_ACT * float(a @ a) + K_VEL * float(t.true['dq'] @ t.true['dq']))
        acc -= K_GRAB * info['first']    # a bounded bonus, two per episode, not a potential
        paid += info['bank']
        slid += info['slide']
        # a loaded hook retreating is credited nothing, so that much of the world's
        # retreat is forgiven by the credit -- and charged by K_SLIDE instead
        forgiven += float(np.maximum(0.0, info['d'] - prev_dh)[info['loaded_start']].sum())
        prev_dh = info['d'].copy()
        if te or tr:
            # the terminal bonus is not a potential and is not meant to be
            acc -= SUCCESS_R if info.get('reason') == 'success' else FAIL_R
            break
    closed = d0 - t._dist()
    expect = K_BANK * paid + K_REACH * (live0 - float(t._live().sum()))
    print(f"no-pump test: {acc:+.4f} accumulated against {expect:+.4f} from the bank "
          f"paid plus the endpoints; bank paid {paid*1000:+.0f} mm + pending "
          f"{info['banked']*1000:+.0f} mm against {closed*1000:+.0f} mm actually closed "
          f"+ {forgiven*1000:.0f} mm of loaded retreat forgiven (charged {K_SLIDE*slid:.2f} "
          f"by K_SLIDE against {K_BANK*forgiven:.2f} it could have bought)")
    assert abs(acc - expect) < 1e-6, (
        f"reward accumulated {acc:+.4f} but the books say {expect:+.4f}: a term is "
        f"missing from the strip list above")
    assert paid + info['banked'] <= closed + forgiven + 1e-9, (
        f"credited {(paid + info['banked'])*1000:.0f} mm over a trajectory that closed "
        f"{closed*1000:.0f} mm: the credit can be pumped")
    # (what a loaded hook is forgiven on the way back it is charged for on the way
    # forward: a loaded hook approaching is credited nothing either)
    print("OK")


if __name__ == '__main__':
    main()
