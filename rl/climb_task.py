"""RL task over the lesson scenes: Gymnasium-shaped API, no gym dependency.

    reset()      -> obs, info
    step(action) -> obs, reward, terminated, truncated, info

Observation is HARDWARE-LEGAL only -- encoders, IMU (raw), F/T, grip pads, own
setpoints, plus the goal expressed relative to the estimated hand poses. The pose
estimate is the same anchor-rooted Estimator the classical controller uses: the
world is known through the hold the robot last latched, exactly what the wall map
plus encoders give on hardware. Ground truth (data.xpos, env.latched) is used only
in the REWARD and the success test -- a training-time oracle that never ships.

Action: 17 deltas in [-1, 1] on the position-servo setpoints, ACT_SCALE rad per
control step (25 Hz), clipped to the ctrlrange. The policy owns every joint,
fingers included.

    python -m rl.climb_task   # smoke check: every lesson hangs stable, API runs
"""
import numpy as np, mujoco
from controller.kinematics import Estimator, JOINTS, THROAT_PT, triad, rotvec
from .lessons import LESSONS, xml_path, rung_segments
from climbing_env import ClimbingEnv, GRIP_BAND, GRIP_SET, GRIP_CLIFF

FRAME_SKIP = 20          # sim steps per control step: 0.002 s x 20 = 25 Hz
ACT_SCALE = 0.06         # rad of setpoint travel per control step (1.5 rad/s max)
LATCH_TOL = 0.15         # m from the goal point that still counts, given a latch
RUNG_TOL = 0.06          # m: estimated throat this close to a rung axis = that rung
SLAM_TAU = 0.3           # s: how slowly the anchor's map point follows the SLAM position
                         # along its rung. Swept on the mac20 policy's replays: at 1 s the
                         # free hand's error was p90 69 mm on lesson 1.2 and 85 on lesson 1
                         # (the anchor's creep and rocking outrun the filter); at 0.3 s,
                         # 48 and 41, for ~5 mm more noise at the median (`_sense`)
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
REACH_TOL = 0.35         # m: a hold placed by arm reach (see `_reach_point`) may sit
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
DANGLE_GRACE = 3.0       # s of TOTAL off-time per hand that is free -- one reach
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
TAU_IDLE = 30.0          # N.m of joint torque a still hang already costs
WEIGHT = 109.0           # N, the robot
K_EFFORT = 5e-5          # per step, per N.m of torque above the idle hang
K_FIGHT = 5e-5           # per step, per N of grip above body weight
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
    def __init__(self, lesson=1):
        self.lesson = lesson
        self.spec = LESSONS[lesson]
        self.env = ClimbingEnv(xml=xml_path(lesson), success_z=1e9,
                               timeout_s=self.spec['timeout'],
                               held_fn=self._attached)
        self.est = Estimator(self.env.model)
        self.segs = rung_segments(self.env.model)
        self.goals = [np.asarray(g, float) for g in self.spec['goals']]
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
        for j in ('finger_L', 'finger_R'):
            self.lo[JOINTS.index(j)] = GRIP_BAND[0]
        self.dt = self.env.model.opt.timestep
        m = self.env.model
        self._rung_name = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                           for g in self.env.rungs}
        self._hook_geoms = {}
        for sd in 'LR':
            ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm % sd)
                   for nm in ('hook_%s', 'finger_%s')}
            self._hook_geoms[sd] = {g for g in range(m.ngeom)
                                    if m.geom_bodyid[g] in ids and m.geom_contype[g]}
        self._flight = 0
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')
        self._gyro_adr = m.sensor_adr[sid]
        self._w_sum = np.zeros(3)          # body rates summed over the substeps
        self._Rf = None                    # filtered world<-torso attitude
        self.obs_dim = len(self.reset()[0])

    # ---- hardware-legal sensing -------------------------------------------
    def _confirmed(self, side):
        """same latch evidence the classical controller trusts: pad + finger + load"""
        h = self.env.hand(side)
        ang = self.env.sensor('enc_finger_' + side)[0]
        return bool(h['pad'] > 20.0 and GRIP_BAND[0] <= ang <= GRIP_BAND[1]
                    and np.linalg.norm(h['force']) > 12.0)

    def _touching(self, side):
        """is the hook still on its rung, loaded or not? Pad contact and the finger
        in the band. `_confirmed` asks for HOLD_LOAD-scale force, which a hook that
        has handed its weight to the other one no longer has -- but it has not gone
        anywhere, and its hold is still the map's best anchor."""
        return bool(self.env.hand(side)['pad'] > 2.0 and self._gripping(side))

    def _nearest_rung(self, p):
        """(rung name, point on its axis) if p is within RUNG_TOL of one, else None"""
        best = None
        for nm, (c, a, h) in self.segs.items():
            q = c + np.clip((p - c) @ a, -h, h) * a
            d = float(np.linalg.norm(p - q))
            if best is None or d < best[0]:
                best = (d, nm, q)
        return (best[1], best[2]) if best[0] < RUNG_TOL else None

    def _sense(self):
        """read sensors, update the estimate, and keep the hold/anchor books"""
        enc = self.env.encoders()
        self.q = np.array([enc[j][0] for j in JOINTS])
        self.dq = np.array([enc[j][1] for j in JOINTS])
        self.tau = np.array([enc[j][2] for j in JOINTS])
        self.imu_s = self.env.imu(raw_only=True)
        # Attitude: the gyro carries it across the step, the SLAM attitude pins the
        # drift. The IMU triad did that first: alone it reads the body's own swing
        # as tilt (26.5 deg rms; the free hand 91-130 mm off) and filtered it was
        # 1.1 deg on a still hang -- but 4 deg p50 / 6 deg p90 / 15 max during the
        # climb itself, where every hand-over is an impact, and on a 0.5 m lever
        # that was 35-55 mm at the hand, the floor under everything else (mac20).
        # The lidar SLAM's attitude does not care about accelerations.
        dt = FRAME_SKIP * self.dt
        self._Rf = self._Rf @ _rotm(self._w_sum * (dt / FRAME_SKIP))
        self._w_sum[:] = 0.0
        p_slam, R_slam_base = self.env.slam()
        # the SLAM reports the base; the filter tracks torso_upper, a waist away.
        # Their relative rotation depends on the joint angles only, so read it off
        # the scratch model posed with this step's encoders
        self.est.update(self.q, self.imu_s, self.anchor, self.hold[self.anchor][1],
                        R_torso=self._Rf)
        d_ = self.est.d
        R_slam = R_slam_base @ (d_.xmat[self.est.base].reshape(3, 3).T
                                @ d_.xmat[self.est.torso].reshape(3, 3))
        self._Rf = self._Rf @ _rotm((dt / AHRS_TAU) * rotvec(self._Rf.T @ R_slam))
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
        dp = p_slam - self.est.body_pos(self.est.base)
        nm, pt = self.hold[self.anchor]
        ax = self.segs[nm][1]
        self.hold[self.anchor] = (nm, pt + (dt / SLAM_TAU) * float(dp @ ax) * ax)
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
        for nm, (c, a, h) in self.segs.items():
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
        geometry is the wall map, which vision/SLAM supplies on hardware.
        """
        cand = []
        for c, a, h in self.segs.values():
            h = max(h - SEAT_IN, 0.0)              # the span a hook can actually sit on
            s = float((p - c) @ a)
            q = c + np.clip(s, -h, h) * a
            cand.append((float(np.linalg.norm(p - q)), q - p, a, s, h))
        cand.sort(key=lambda v: v[0])
        out = []
        for i in range(N_NEAR):
            if i < len(cand):
                out += [cand[i][1], cand[i][2]]
            else:
                out += [FAR, np.zeros(3)]
        _, _, _, s, h = cand[0]
        out.append(np.array([h - s, h + s]))
        return np.concatenate(out)

    def _obs(self):
        hl, hr = self.env.hand('L'), self.env.hand('R')
        goal = np.concatenate([self.goals[0] - self.est.throat('L'),
                               self.goals[1] - self.est.throat('R')])
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
            self.q, self.dq, self.tau, self.ctrl,
            self.imu_s['accel'], self.imu_s['gyro'], self.imu_s['mag'],
            hl['force'], hl['torque'], [hl['pad']],
            hr['force'], hr['torque'], [hr['pad']],
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

    def _cost(self, p, goal):
        """metres to go for a hand at p, LIVE: to the nearest point on any rung, then
        from that point to the goal. Progress is measured AT THE RUNG, not at the hand.

        The plain distance |p - goal| paid for leaving the rung. On lesson 1.5 the
        left hook walked off the end of its segment, and with the goal 1.4 m along
        the wall the cheapest way to get "closer" was to hang 0.4 m below the far
        segment: +0.14 m of straight-line progress, +0.3 of reach, and a state no
        lesson had taught it to leave -- win9 (5.3M steps, lesson 1.5 at 0 %) and
        mac13 both: 20/20 and 7/8 episodes hanging there for 58 s. Measured at the
        rung the same drop is a 0.15 m LOSS. Along a rung the two agree exactly; off
        a rung this one charges the detour and refunds it on the way back -- still
        a function of where the hand is, so it telescopes and cannot be pumped.

        Live only. The detour must stay out of the bank: the bank credits a free
        hand and a loaded hook nothing, and a loaded hook's throat is pulled 30-80 mm
        off its rung by the load (the README's 31 mm under a 50 N side pull). With
        the detour in the bank that uncharged distortion was refunded on every
        lift-and-reseat: the first check run (mac14, 3M) fluttered the right hand
        698 times per episode, banked 56 m and scored +241 on a body that never
        moved. `_hand_dists`, which the bank uses, is measured at the rung point,
        which a loaded hook can move only by sliding -- and sliding is charged.
        """
        off, q = self._route(p)
        return off + float(np.linalg.norm(q - goal))

    def _hand_dists(self):
        """the bank's potential: each hand's rung point to its goal"""
        return np.array([float(np.linalg.norm(self._route(self._throat_true(s))[1] - self.goals[i]))
                         for i, s in enumerate('LR')])

    def _live(self):
        """the live potential: the same plus the detour off the rung"""
        return np.array([self._cost(self._throat_true(s), self.goals[i])
                         for i, s in enumerate('LR')])

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

    def _sep_error(self):
        """metres of hand separation missing ALONG the rung they travel on.

        Pricing only the crossed case was not enough, and the failure is instructive:
        `max(0, -separation)` is zero at zero separation, so the policy moved the
        hands off each other's far side and onto exactly the same point instead --
        measured, mean separation 1 mm and 99 % of steps inside 100 mm. Two 65 mm
        hooks cannot share a span of rung, so the second hand's grab failed on
        contact and the robot fell. That is the drop the videos show.

        The torso frame was tried first and is not a usable reference: the body spins
        on its hooks -- measured, the torso yawed -90..+13 deg over one episode -- so
        a body-frame test disagreed with the real hand order on 54 % of steps and
        charged 0.93 for an episode that was crossed 99 % of the time.

        The rung axis does not spin. `cross_axis` and `cross_sign` are fixed at reset
        from the hand order the robot starts in, which is the order its shoulders put
        them in, so the sign means "the left hand is still on its own side".
        """
        sep = self._throat_true('L') - self._throat_true('R')
        return max(0.0, MIN_SEP - self.cross_sign * float(sep @ self.cross_axis))

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
        """
        return np.array([self._contact(s)[0] > HOLD_LOAD and self._gripping(s)
                         for s in 'LR'])

    def _dist(self):
        return float(self._hand_dists().sum())

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
        # initial holds: the keyframe hangs both hooks on the start rung -- an
        # initial condition you know on hardware too, because you hung it there
        self.hold = {s: self._nearest_rung(self._throat_true(s)) for s in 'LR'}
        assert all(self.hold.values()), "keyframe hooks are not on a rung"
        self.anchor = 'L'
        self._both_conf = False
        self._anchor_fresh = True
        imu = self.env.imu(raw_only=True)
        self._Rf = triad(imu['accel'], imu['mag'])   # at rest the triad is right
        self._w_sum[:] = 0.0
        self._sense()
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
        loaded, slide = touch > HOLD_LOAD, 0.0
        for k, (_, on) in enumerate(con):
            if loaded[k] and self.prev_loaded[k] and on:
                ax = self.segs[next(iter(on))][1]
                slide += abs(float(dp[k] @ ax))
        self.prev_thr, self.prev_loaded, was_loaded = thr, loaded, self.prev_loaded
        self.n_steps += 1
        self.off_t += free                      # cumulative: never reset by a touch
        self.off_run = np.where(free, self.off_run + 1, 0)   # consecutive steps clear
        self.held_t = np.where(free, 0.0, self.held_t + 1)
        grace = DANGLE_GRACE / (FRAME_SKIP * self.dt)
        parked = float(np.sum(free & (self.off_t > grace)
                              & (self.off_t > DANGLE_MAX_FRAC * self.n_steps)))
        stale = float(np.sum(self.held_t > STALE_AFTER / (FRAME_SKIP * self.dt)))
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
        sep_err = self._sep_error()
        effort = max(0.0, float(np.abs(self.tau).sum()) - TAU_IDLE)
        grip = sum(float(np.linalg.norm(self.env.hand(s)['force'])) for s in 'LR')
        fight = max(0.0, grip - WEIGHT)
        r = (K_BANK * bank + K_GRAB * float(first.sum())
             + K_REACH * float((self.prev_live - live).sum())
             - K_DANGLE * parked - K_STALE * stale - K_SEP * sep_err
             - K_EFFORT * effort - K_FIGHT * fight - K_TIME - K_FLIGHT * flight
             - K_SLIDE * slide
             - K_ACT * float(a @ a) - K_VEL * float(self.dq @ self.dq))
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
        terminated, truncated = False, False
        if self._success():
            r += SUCCESS_R
            terminated, info['reason'] = True, 'success'
        else:
            done, why = self.env.terminated()
            if done:
                info['reason'] = why
                if why == 'timeout':
                    truncated = True
                else:
                    r += FAIL_R
                    terminated = True
        return self._obs(), r, terminated, truncated, info


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
    t = ClimbTask(1)
    t.reset()
    base = mujoco.mj_name2id(t.env.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    t.env.data.xfrc_applied[base] = [0, 25, 0, 0, 0, 0]
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
    print(f"slide test: 25 N along the rung, both hooks seated for {held}/{n} steps, "
          f"body moved {moved_while_held*1000:.0f} mm over those -- "
          f"credited {banked*1000:+.1f} mm, loaded hooks dragged {slid*1000:.0f} mm "
          f"and charged {K_SLIDE*slid:.2f}")
    assert slid > 0.02 and 0.5 < slid / max(moved_while_held, 1e-9) < 2.0, (
        f"the hooks were dragged {moved_while_held*1000:.0f} mm under load but "
        f"K_SLIDE saw {slid*1000:.0f} mm")
    assert held > 100, (f"only {held}/{n} steps with both hooks seated: 25 N should "
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
        f"steps: the finger stop is not at the band's edge")

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

    # Crossing must read zero in the normal hang and grow once the hands swap sides.
    t = ClimbTask(1)
    t.reset()
    at_hang = t._sep_error()
    sep = t.cross_sign * float(
        (t._throat_true('L') - t._throat_true('R')) @ t.cross_axis)
    real = t._throat_true
    # inverted: the hands swap sides. together: both hands on the same point --
    # the case that only pricing "crossed" missed, and that dropped the robot.
    t._throat_true = lambda s: real('R') if s == 'L' else real('L')
    inverted = t._sep_error()
    t._throat_true = lambda s: real('R')
    together = t._sep_error()
    t._throat_true = real
    print(f"separation test: hands {sep*1000:+.0f} mm apart -> penalty "
          f"{at_hang*1000:.0f} mm | swapped -> {inverted*1000:.0f} mm | "
          f"on the same point -> {together*1000:.0f} mm")
    assert sep > MIN_SEP, "the hang keyframe starts inside the separation floor"
    assert at_hang == 0.0, f"the hang posture is charged {at_hang*1000:.0f} mm"
    assert abs(together - MIN_SEP) < 1e-9, (
        f"two hands on the same point are charged {together*1000:.0f} mm, expected "
        f"{MIN_SEP*1000:.0f} -- this is the case that made the second grab fail")
    assert inverted > together, "swapping the hands must cost more than merging them"

    # The attitude filter must beat the bare triad while the body moves, and must
    # not drift while it hangs still. Random actions swing the body hard.
    t = ClimbTask(1)
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
    t = ClimbTask(2)
    t.reset()
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

    # SLAM keeps the anchor honest. The loaded hook creeps along the rung under a
    # side load and the arm cannot see it: hang, then pull the body 25 N along the
    # rung for 150 steps (the slide test's scenario, the hooks drag 25-50 mm).
    from urdf.climbing_env import SLAM_SIG
    t = ClimbTask(1)
    t.reset()
    z = np.zeros(17)
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
        t.env.data.xfrc_applied[t.env.base, :3] = 25.0 * ax
        t.step(z)
    t.env.data.xfrc_applied[t.env.base, :3] = 0.0
    crept = float((t._throat_true(t.anchor) - p0) @ ax)
    # (the wander is mostly the filtered attitude's ~1 deg on the 0.5 m lever from
    # the anchor to the base, ~9 mm, plus ~3 mm of averaged SLAM noise)
    assert max(wander) < 0.020, f"hanging still, the anchor wandered {max(wander)*1e3:.0f} mm"
    assert abs(crept) > 0.015, f"the pull crept the hook only {crept*1e3:.0f} mm; the test proves nothing"
    assert abs(stale()) < 0.020, f"after {crept*1e3:+.0f} mm of creep the anchor is {stale()*1e3:+.0f} mm stale"
    print(f"slam test: hanging still the anchor wanders {max(wander)*1e3:.1f} mm under {SLAM_SIG*1e3:.0f} mm "
          f"of SLAM noise; a 25 N pull creeps the loaded hook {crept*1e3:+.0f} mm along the rung and the "
          f"anchor follows to within {stale()*1e3:+.0f} mm")

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
    # (the hover is itself 72 mm from the last SEAT of the segment, SEAT_IN, so the
    # dive's loss against it is ~80 mm, not the 146 against the geometric end)
    assert t._cost(hang, g) > t._cost(hover, g) + 0.05, (t._cost(hang, g), t._cost(hover, g))
    on = c0 - 0.10 * a0
    assert abs((t._cost(on, g) - t._cost(on + 0.30 * a0, g)) - 0.30) < 1e-9, "progress along the rung must be paid in full"
    assert t._cost(hover, g) < t._cost(edge, g) + 0.05, "hovering just past the edge is charged the air, nothing more"
    pulled = on + np.array([0.0, 0.0, -0.03])           # a loaded hook's throat, off its line
    assert abs(np.linalg.norm(t._route(pulled)[1] - g) - np.linalg.norm(t._route(on)[1] - g)) < 1e-12, \
        "the bank must not see a loaded hook pulled off its line (mac14: 698 holds and 56 m banked per episode when it did)"
    assert abs(t._cost(pulled, g) - t._cost(on, g) - 0.03) < 1e-9, "the live term sees the whole detour"
    print(f"route test: the win9 dive is {(np.linalg.norm(hover - g) - np.linalg.norm(hang - g))*1e3:+.0f} mm "
          f"of straight-line progress and {(t._cost(hover, g) - t._cost(hang, g))*1e3:+.0f} mm at the rung; "
          f"0.30 m along the rung pays 0.30; a loaded hook pulled 30 mm off its line moves the bank 0")

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
                + K_ACT * float(a @ a) + K_VEL * float(t.dq @ t.dq))
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
