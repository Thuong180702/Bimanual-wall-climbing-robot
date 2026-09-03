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
from controller.kinematics import Estimator, JOINTS, THROAT_PT
from .lessons import LESSONS, xml_path, rung_segments
from climbing_env import ClimbingEnv, GRIP_BAND, GRIP_SET

FRAME_SKIP = 20          # sim steps per control step: 0.002 s x 20 = 25 Hz
ACT_SCALE = 0.06         # rad of setpoint travel per control step (1.5 rad/s max)
LATCH_TOL = 0.15         # m from the goal point that still counts, given a latch
RUNG_TOL = 0.06          # m: estimated throat this close to a rung axis = that rung
N_NEAR = 3               # nearest rungs reported per hand. Without these the policy
                         # cannot see the wall at all: it knows where the goal is and
                         # nothing about what lies between, so lesson 2 (gaps) and 6
                         # (staggered) are guesswork and one shared policy can only
                         # memorise each fixed layout. Three is what a hand chooses
                         # between; the map supplies them, so this is hardware-legal.
FAR = np.array([9.0, 9.0, 9.0])   # padding for a scene with fewer than N_NEAR rungs
CLEAR_OF = 0.08          # m of gap at which a hook is unambiguously OFF the rung.
                         # A hand must get this far clear before it may re-anchor:
                         # measured on the 5.1M policy, every single anchor re-take
                         # came from the wrist load dipping (min 1.0 N against a
                         # 20 N HOLD_LOAD) while the hook never left the rung -- so
                         # the reward was paying for weight shifts, and a hand that
                         # never let go still banked progress. The throat is ~50 mm
                         # deep, and a seated hook measures 2-8 mm, so 80 mm cannot
                         # be reached without genuinely letting go.
ON_RUNG = 0.06           # m of throat-to-rung-axis gap below which a hook is on its
                         # hold. Measured: 2-8 mm while
                         # seated, 68 mm and up once genuinely clear, so the two
                         # states are far apart and any threshold in between works.
                         # `env.latched` cannot do this job -- it is a tight box
                         # round the nominal seat, the model README records it
                         # reading False on a hook carrying 167 N, and measured
                         # through a real episode it called a hand FREE on 65 % of
                         # the steps where the hook had never left the rung. Gating
                         # the reward on that paid for sliding all over again.
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
HOLD_LOAD = 20.0         # N through the wrist before a seated hook counts as held
DANGLE_GRACE = 3.0       # s of TOTAL off-time per hand that is free -- one reach
DANGLE_MAX_FRAC = 0.5    # and past that, a hand may be off at most this fraction of
                         # the episode so far. Being off half the time is what
                         # alternating looks like; 79 % is one hand doing everything.
K_DANGLE = 0.01          # per step, per hand over its share
STALE_AFTER = 8.0        # s a hand may stay on its hold without ever letting go
K_STALE = 0.01           # per step, per hand past that
K_ACT = 2e-4             # action magnitude penalty -- smoothness
K_VEL = 1e-4             # joint velocity penalty -- smoothness
SUCCESS_R = 10.0
FAIL_R = -10.0

# Progress is BANKED, not live: each hand carries an anchor -- where it last took a
# hold -- and the reward is the reduction in the distance from that anchor to the
# goal. Two earlier shapes both failed, and the second failed hard enough to be
# worth writing down.
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
# An anchor cannot be pumped: it only moves when a hand actually takes a hold, and
# it moves in both directions, so the term telescopes to K_BANK x (how far the holds
# actually travelled). Sliding does not move it either, because nothing was taken.
# A small live-distance term stays as shaping, charged symmetrically so it also
# telescopes -- over a full swing it nets zero, and the grip pad keeps what sliding
# can win inside it down to about 16 mm.
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
                               timeout_s=self.spec['timeout'])
        self.est = Estimator(self.env.model)
        self.segs = rung_segments(self.env.model)
        self.goals = [np.asarray(g, float) for g in self.spec['goals']]
        self.lo = self.env.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.env.model.actuator_ctrlrange[:, 1].copy()
        self.dt = self.env.model.opt.timestep
        self.obs_dim = len(self.reset()[0])

    # ---- hardware-legal sensing -------------------------------------------
    def _confirmed(self, side):
        """same latch evidence the classical controller trusts: pad + finger + load"""
        h = self.env.hand(side)
        ang = self.env.sensor('enc_finger_' + side)[0]
        return bool(h['pad'] > 20.0 and GRIP_BAND[0] <= ang <= GRIP_BAND[1]
                    and np.linalg.norm(h['force']) > 12.0)

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
        self.est.update(self.q, self.imu_s, self.anchor, self.hold[self.anchor][1])
        for s in 'LR':
            if self._confirmed(s):
                hit = self._nearest_rung(self.est.throat(s))
                if hit:
                    self.hold[s] = hit
        # anchor: keep the current one while it is confirmed, else the other hand
        other = 'R' if self.anchor == 'L' else 'L'
        if not self._confirmed(self.anchor) and self._confirmed(other):
            self.anchor = other
            self.est.update(self.q, self.imu_s, self.anchor,
                            self.hold[self.anchor][1])

    def _near_rungs(self, p):
        """the N_NEAR rungs closest to p: offset to each one's axis, and its direction.

        Every point along a rung is a seat -- the rungs are smooth, so where a hand
        lands along one is free, which is also why it can slide there.

        Estimated, not ground truth: `p` is the estimator's throat and the rung
        geometry is the wall map, which vision/SLAM supplies on hardware.
        """
        cand = []
        for c, a, h in self.segs.values():
            q = c + np.clip((p - c) @ a, -h, h) * a
            cand.append((float(np.linalg.norm(p - q)), q - p, a))
        cand.sort(key=lambda v: v[0])
        out = []
        for i in range(N_NEAR):
            if i < len(cand):
                out += [cand[i][1], cand[i][2]]
            else:
                out += [FAR, np.zeros(3)]
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

    def _hand_dists(self):
        return np.array([float(np.linalg.norm(self._throat_true(s) - self.goals[i]))
                         for i, s in enumerate('LR')])

    def _seat(self, side):
        """(rung name, gap) for the rung this hook is nearest to"""
        p = self._throat_true(side)
        return min(((nm, float(np.linalg.norm(
            p - (c + np.clip((p - c) @ a, -h, h) * a))))
            for nm, (c, a, h) in self.segs.items()), key=lambda v: v[1])

    def _free(self):
        """per hand: True when it is not seated on a rung"""
        return np.array([self._seat(s)[1] >= ON_RUNG for s in 'LR'])

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

    def _holding(self):
        """per hand: seated AND carrying. Load is what separates a hook that has
        taken a hold from one hovering next to the rung -- and hovering next to it
        is exactly what the pump learned to do."""
        return np.array([self._seat(s)[1] < ON_RUNG
                         and np.linalg.norm(self.env.hand(s)['force']) > HOLD_LOAD
                         for s in 'LR'])

    def _dist(self):
        return float(self._hand_dists().sum())

    def _success(self):
        """both hooks seated on a goal rung, near their goal points.

        Same seat test as the reward, on purpose: when this used `env.latched` and
        the reward used something else, the dense signal and the thing being scored
        disagreed, which is how the first runs came to be graded on a box the model
        README already says reads False on a hook carrying 167 N.
        """
        for i, s in enumerate('LR'):
            nm, gap = self._seat(s)
            if gap >= ON_RUNG or nm not in self.spec['goal_rungs']:
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
        self._sense()
        self.prev_d = self._hand_dists()
        self.anchor_d = self.prev_d.copy()   # goal distance of each hand's last hold
        self.held = self._holding()
        self.let_go = np.zeros(2, bool)      # has this hand been clear since anchoring
        # the line the hands travel along, and which way round they start on it
        self.cross_axis = self.segs[self._seat('L')[0]][1]
        sep = self._throat_true('L') - self._throat_true('R')
        self.cross_sign = float(np.sign(sep @ self.cross_axis)) or 1.0
        self.off_t = np.zeros(2)          # TOTAL control steps each hand has been off
        self.held_t = np.zeros(2)         # steps since each hand was last fully clear
        self.n_steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.ctrl = np.clip(self.ctrl + ACT_SCALE * a, self.lo, self.hi)
        for _ in range(FRAME_SKIP):
            self.env.step(self.ctrl)
        self._sense()
        gaps = np.array([self._seat(s)[1] for s in 'LR'])
        d, free, held = self._hand_dists(), gaps >= ON_RUNG, self._holding()
        self.n_steps += 1
        self.off_t += free                      # cumulative: never reset by a touch
        self.held_t = np.where(gaps > CLEAR_OF, 0.0, self.held_t + 1)
        grace = DANGLE_GRACE / (FRAME_SKIP * self.dt)
        parked = float(np.sum(free & (self.off_t > grace)
                              & (self.off_t > DANGLE_MAX_FRAC * self.n_steps)))
        stale = float(np.sum(self.held_t > STALE_AFTER / (FRAME_SKIP * self.dt)))
        # A hand re-anchors only after it has genuinely let go and taken hold again.
        # Without the let_go gate, a weight shift that drops the wrist load counts as
        # taking a new hold, and a hand that never leaves the rung banks progress --
        # which is what the 5.1M policy was doing on both hands.
        self.let_go |= gaps > CLEAR_OF
        took = held & ~self.held & self.let_go
        self.let_go &= ~took
        new_anchor = np.where(took, d, self.anchor_d)
        sep_err = self._sep_error()
        r = (K_BANK * float((self.anchor_d - new_anchor).sum())
             + K_REACH * float((self.prev_d - d).sum())
             - K_DANGLE * parked - K_STALE * stale - K_SEP * sep_err
             - K_ACT * float(a @ a) - K_VEL * float(self.dq @ self.dq))
        self.anchor_d, self.held, self.prev_d = new_anchor, held, d
        info = {'dist': float(d.sum()), 'lesson': self.lesson,
                'off': float(free.sum()), 'parked': parked, 'stale': stale,
                'banked': float(self.anchor_d.sum()), 'sep_err': sep_err}
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
    moved_while_held, held, n = 0.0, 0, 0
    prev, bank0 = t._dist(), None
    for _ in range(150):
        _, r, te, tr, info = t.step(np.zeros(17))
        if bank0 is None:
            bank0 = info['banked']
        now = t._dist()
        if info['off'] == 0:                    # both hooks seated this step
            moved_while_held += abs(prev - now)
            held += 1
        prev = now
        n += 1
        if te or tr:
            break
    t.env.data.xfrc_applied[base] = np.zeros(6)
    banked = bank0 - info['banked']
    print(f"slide test: 25 N along the rung, both hooks seated for {held}/{n} steps, "
          f"body moved {moved_while_held*1000:.0f} mm over those -- "
          f"banked {banked*1000:+.1f} mm")
    assert held > 100, (f"only {held}/{n} steps with both hooks seated: 25 N should "
                        f"not take a hook off the rung, so something else is")
    assert abs(banked) < 1e-6, (
        f"sliding banked {banked*1000:+.1f} mm: an anchor moved without a hold "
        f"being taken, so the slide is being paid for again")

    # The dangle penalty must charge a hand PARKED off the wall and leave a hand
    # that is mid-reach alone -- the flat version of this taxed every release and
    # pushed the policy back towards keeping both hooks on and shoving.
    t = ClimbTask(1)
    t.reset()
    grace = int(DANGLE_GRACE / (FRAME_SKIP * t.dt))
    # hold the left hand off by faking its seat gap -- step() derives free, held and
    # let_go from that one measurement, so this is the quantity to intercept
    real_seat = t._seat
    t._seat = lambda s: (real_seat(s)[0], 9.0) if s == 'L' else real_seat(s)
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
    real_seat = t._seat
    k = {'i': 0}
    def flutter(sd):
        if sd != 'L':
            return real_seat(sd)
        # 13 steps off, 4 steps on -- 76 % off, no stretch anywhere near the grace
        return (real_seat(sd)[0], 9.0 if (k['i'] % 17) < 13 else real_seat(sd)[1])
    t._seat = flutter
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
    assert stale_at is not None, "a hand held for the whole episode is never charged"
    assert stale_at > n - 2, f"charged at step {stale_at}, before the {n}-step threshold"

    # A hand whose wrist load merely dips -- a weight shift -- must not count as
    # having taken a new hold. On the 5.1M policy every anchor re-take came from
    # exactly that, with the hook never leaving the rung.
    t = ClimbTask(1)
    t.reset()
    bank0, flick = t.anchor_d.sum(), [True]
    t._holding = lambda: np.array([flick[0], True])   # left hold flickers on load
    for i in range(80):
        flick[0] = (i // 5) % 2 == 0
        _, _, te, tr, info = t.step(np.zeros(17))
        if te or tr:
            break
    moved = abs(float(t.anchor_d.sum() - bank0))
    print(f"let-go test: a hold flickering {8} times without the hook leaving the "
          f"rung moved the anchor {moved*1000:.1f} mm")
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

    # No pump, proved rather than hoped. Both reward terms are differences of a
    # stored quantity, so over any trajectory whatsoever they must telescope to the
    # endpoint difference -- swing a hand out and back a hundred times and the sum
    # is unchanged. The previous reward failed exactly here: it paid +27.28 over an
    # episode that closed 0.000 m. Random actions, because a pump would show up as
    # the accumulated total drifting away from the endpoints.
    t = ClimbTask(1)
    t.reset()
    rng = np.random.default_rng(3)
    d0, bank0, acc = t._dist(), t.anchor_d.sum(), 0.0
    for _ in range(400):
        a = rng.uniform(-1, 1, 17)
        _, r, te, tr, info = t.step(a)
        # strip the penalties; what is left must be the two potential terms
        acc += (r + K_DANGLE * info['parked'] + K_SEP * info["sep_err"]
                + K_ACT * float(a @ a) + K_VEL * float(t.dq @ t.dq))
        if te or tr:
            # the terminal bonus is not a potential and is not meant to be
            acc -= SUCCESS_R if info.get('reason') == 'success' else FAIL_R
            break
    expect = K_BANK * (bank0 - info['banked']) + K_REACH * (d0 - t._dist())
    print(f"telescoping test: {acc:+.4f} accumulated over the trajectory against "
          f"{expect:+.4f} from the endpoints alone")
    assert abs(acc - expect) < 1e-6, (
        f"reward accumulated {acc:+.4f} but the endpoints say {expect:+.4f}: the "
        f"terms are not potentials any more and the trajectory can be pumped")
    print("OK")


if __name__ == '__main__':
    main()
