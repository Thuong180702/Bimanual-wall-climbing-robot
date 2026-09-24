"""Train ONE PPO policy across the lessons, with a curriculum and resume.

    python -m rl.train_ppo --name climber                 # every lesson, curriculum on
    python -m rl.train_ppo --name climber --resume        # continue where it stopped
    python -m rl.train_ppo --name t1 --lessons 1,2        # a subset
    python -m rl.train_ppo --name t1 --no-curriculum      # all lessons from step 0

Ctrl-C saves and exits cleanly; `--resume` picks the run back up, timestep counter,
optimizer state, observation statistics and curriculum progress included.

A run looks after itself; nobody has to watch it for the two things that used to
need a person. The next lesson unlocks only once EVERY lesson in the pool is being
passed (`UNLOCK_AT`), so a new one is not added on top of half-learnt ones. And if
the pool forgets -- the best checkpoint's lessons, scored now, a whole lesson below
that best for a million steps (`GUARD_DROP`, `GUARD_STEPS`) -- it goes back to the
best checkpoint by itself and carries on at half the learning rate. Everything it
prints is also appended to out/<name>/train.log.

The update size looks after itself too. Each action dimension's std is held in
[STD_MIN, 1], and the learning rate moves to keep every update's KL near the run's
KL target (AdaptiveLR). Without them the std of the dimensions a policy uses
precisely shrank without limit, and since a Gaussian's KL goes as 1/std^2 every
update grew with it until the policy was being rewritten wholesale each time --
the drift that took apart every long run since win11. The target starts at
KL_START and the guard halves it on each rollback, so a run takes the biggest
step that has not yet cost it a lesson rather than a small one forever.

## Where the simulator runs

    python -m rl.train_ppo --name g1                    # picks by itself (rl.hardware)
    python -m rl.train_ppo --name g1 --backend mjx      # force either one
    python -m rl.train_ppo --name g1 --backend cpu

With an NVIDIA GPU that JAX can use and `mujoco-mjx` installed, the physics and the
whole task run on the GPU (`rl.mjx_task`, MJX_ENVS envs in one batch) and the
policy trains there too. Otherwise it is the CPU path below, unchanged: C MuJoCo in
one process per core. The curriculum, the guard, the checkpoints and the policy are
the same code on both, and a checkpoint from either resumes on the other; the GPU
simulates `rl.mjx_scene`'s simplified robot, which `python -m rl.mjx_check`
compares against this one.

## Why one policy and not nine

Three shapes were on the table. This is the middle one, and the reasons are
structural rather than a preference:

*Nine specialists + a dispatcher* trains easily and debugs beautifully, but the nine
share almost everything that is hard -- hanging without falling, transferring weight
onto a hand, seating a hook in a +-25 mm window -- and each would have to learn it
from scratch off its own reward. It also leaves the seams untrained: the posture a
traverse ends in is not in the climb specialist's training distribution.

*A meta-policy that calls nine sub-policies* is hierarchical RL, and it would be
replacing something that already works. `controller/holds.py` picks the next hold
with A* over the wall map, deterministically, today. Learning that choice is a much
harder problem than the one it solves.

*One policy over all lessons* shares the common skill and needs no dispatcher,
because the goal is already in the observation -- it is goal-conditioned, not
lesson-conditioned, and at deploy time the planner supplies the goal exactly as the
lesson spec does here. The failure mode to avoid is catastrophic forgetting, which
is what training the lessons strictly one after another would cause: fine-tune on
lesson 9 and lesson 1 is gone. So lessons are MIXED, not sequential -- each env
picks from the active pool on every reset -- and the curriculum only controls which
lessons are IN the pool. Nothing ever leaves it.
"""
import argparse, hashlib, json, os, pickle, signal, sys, time
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor, VecNormalize
except ImportError as e:
    raise SystemExit(f"missing training deps ({e}); pip install gymnasium stable-baselines3")

from controller import SCENE
from . import default_envs, hardware
from .climb_task import ClimbTask
from .lessons import LESSONS, lesson_key

WINDOW = 40          # episodes per lesson the success rate is measured over
UNLOCK_AT = 0.70     # success rate EVERY lesson in the pool needs before the next
                     # one unlocks. It was 0.60 on the newest lesson alone, and win17
                     # unlocked lesson 2 with 1.5 and 1.85 at ~60 %: lesson 2 then
                     # took 30 % of the steps at 2 %, and the pool forgot -- lesson -2
                     # went 20/20 -> 13/20 in replay. Consolidate, then add.
UNLOCK_MIN = 20      # episodes needed before that rate is believed
MIN_SHARE = 0.10     # the floor on a mastered lesson's need, so it is not forgotten
WARM_RATE = 0.20     # the rate at which a new lesson counts as started, for the log
                     # A freshly unlocked lesson is, at first, a stream of failures,
                     # and drawn as often as the rest it is half the pool's gradient
                     # when there are two lessons. The 10.8M win8 run spent 8M steps
                     # on a lesson at 0 % and took the one before it from 95 % down
                     # to 32 %, and its action std from 0.67 to a floor of 0.04.
GUARD_DROP = 1.0     # the best checkpoint's own lessons falling this far below its
GUARD_STEPS = 1_000_000  # score -- a whole lesson's worth -- for this long sends the
                     # run back to that checkpoint. Every long run from a peak has
                     # drifted off it over millions of steps (win11, win15 2, win16,
                     # win16 2, win16 3, win17) and each one needed a person to
                     # notice and resume from best. A million steps, because the
                     # first exposure to a new lesson dips and recovers inside 600k
                     # on its own (the win16 A/B), and that should not trip it.
LR_DECAY = 0.5       # each rollback halves the learning rate: the untested suspect
LR_MIN = 1e-5        # for the drift is the update size, 3e-4 for millions of steps
                     # with no KL limit, so the retry takes smaller steps
STD_MIN = 0.05       # floor on every action dimension's std. win17's logs: with no
                     # floor, six dims (the right shoulder's three, its forearm roll,
                     # its finger, the left wrist deviation) shrank from ~0.09 to
                     # 0.02, entropy fell in a straight line (+2.4 -> -7.7 nats over
                     # 7M steps) and never levelled, and since a Gaussian's KL goes as
                     # 1/std^2 every update grew with it: approx_kl 0.12 -> 1.1, clip
                     # fraction 0.5 -> 0.8. At 0.05 the noise on a setpoint is
                     # 0.003 rad a step; the best policy passed the same with it
                     # (lesson 1.5 14/20 floored against 13/20 as trained).
KL_START = 0.1       # the learning rate is moved to keep each update's approx_kl
KL_MIN = 0.01        # within [target/2, 2 target] -- rsl_rl's rule, the PPO behind
LR_MAX = 3e-4        # the ETH legged-robot work -- never above 3e-4, the rate the
                     # early curriculum was validated at. win17 ran at KL 0.2-1.1,
                     # 10-50x a usual PPO update, and drifted exactly as every long
                     # run before it had.
                     #
                     # The TARGET is the run's own, and the guard owns it: each
                     # rollback halves it, down to KL_MIN. Small updates are what
                     # stops the forgetting and they are also what makes a lesson
                     # take millions of steps, so the run should take the largest
                     # step that has not cost it anything yet rather than a small one
                     # forever. It starts at 0.1 because win17 spent 2.5M steps at
                     # KL 0.12-0.3 improving and forgetting nothing, and drifted at
                     # 0.6-1.1. Halving it is what the guard's LR_DECAY used to do on
                     # its own: with an adaptive rate, a halved rate is pulled back up
                     # within a few updates, and only the target sticks.
MJX_ENVS = 4096      # envs in one GPU batch, 33 KB of state each (130 MB). ponytail:
                     # the usual MJX size, unmeasured on a GPU here -- `python -m
                     # rl.mjx_task --envs N` on the card decides
# PPO on the GPU batch. The CPU numbers below (256 steps x 12 envs, 512 per
# minibatch, 10 epochs) would make one update a million samples; this is the
# massively-parallel shape instead (rsl_rl's: short rollouts, 4 minibatches, 5
# epochs), with every other setting -- gamma, lambda, entropy, the adaptive rate and
# its KL target -- the CPU path's.
MJX_PPO = dict(n_steps=32, minibatches=4, n_epochs=5)


class Tee:
    """a stream that writes to the console and to the run's log at once"""

    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)
        return len(s)

    def flush(self):
        for f in self.files:
            f.flush()

    def __getattr__(self, name):          # isatty, encoding, fileno: the console's
        return getattr(self.files[0], name)


def robot_stamp():
    """which robot a checkpoint learned: a hash of urdf/scene.xml, line endings
    normalised so a Windows checkout of the same file stamps the same"""
    return hashlib.sha256(open(SCENE, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:16]


def check_robot(state, allow):
    """refuse to resume a policy onto a robot it never trained on.

    The observation and action are the same shape on the 11.1 kg robot and the
    21.9 kg one, so such a resume loads without a word and trains a policy -- and
    VecNormalize statistics -- learnt on a different machine. win13 trained 7.6M
    steps on a robot that had already been replaced; this is the same mistake one
    level up. Checkpoints from before the stamp have none, and count as different.
    """
    have = state.get('robot')
    if have != robot_stamp() and not allow:
        raise SystemExit(
            f"this checkpoint was trained on a different robot (stamp {have}, the "
            f"model is now {robot_stamp()}): urdf/scene.xml has changed since. Start a "
            f"new run with a new --name, or add --new-robot to fine-tune it anyway.")


class MultiLesson(gym.Env):
    """Every lesson pre-loaded; each reset draws one from the active pool.

    All eighteen models cost ~330 MB and 0.4 s to build in one process, so holding
    them all beats reloading a scene on every reset -- WITHOUT their visual meshes
    (`visual=False`, the default here), which training never draws. With them it
    was 1.5 GB per env on the real-actuator robot; only `rl.render` asks for them.
    """

    def __init__(self, lessons, seed=0, visual=False):
        self.tasks = {k: ClimbTask(k, visual=visual) for k in lessons}
        self.active = list(lessons)
        self.rng = np.random.default_rng(seed)
        self.cur = self.tasks[self.active[0]]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                (self.cur.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (17,), np.float32)

    def set_active(self, active, weights=None):
        """called from the training process by the curriculum callback"""
        self.active = list(active)
        self.weights = None if weights is None else np.asarray(weights, float) / np.sum(weights)
        return self.active

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        i = int(self.rng.choice(len(self.active), p=getattr(self, 'weights', None)))
        self.cur = self.tasks[self.active[i]]
        return self.cur.reset(seed)

    def step(self, a):
        return self.cur.step(a)


class MjxVecEnv(VecEnv):
    """The GPU path's envs: one `rl.mjx_task.Batch`, dressed as the VecEnv the CPU
    path's SubprocVecEnv of MultiLessons is -- the same infos on an episode's end
    (reason, lesson, steps, dist, the terminal observation, TimeLimit.truncated) and
    the same `set_active`, so everything downstream is unchanged."""

    def __init__(self, lessons, n, seed=0):
        from . import mjx_task          # JAX only on this path
        self.mt = mjx_task
        self.batch = mjx_task.Batch(n, seed)
        dim = np.asarray(self.batch.reset()).shape[1]
        super().__init__(n, gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32),
                         gym.spaces.Box(-1.0, 1.0, (17,), np.float32))
        self.set_active(lessons)

    def set_active(self, active, weights=None):
        self.batch.set_probs(dict(zip(active, weights if weights is not None
                                      else [1.0] * len(active))))
        return list(active)

    def reset(self):
        return np.asarray(self.batch.reset())

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, end, r, term, trunc, info = self.batch.step(self._actions)
        obs, term, trunc = np.asarray(obs), np.asarray(term), np.asarray(trunc)
        done = term | trunc
        infos = [{} for _ in range(self.num_envs)]
        if done.any():
            end = np.asarray(end)
            why, lesson, steps, dist = (np.asarray(info[k]) for k in ('reason', 'lesson', 'steps', 'dist'))
            for i in np.nonzero(done)[0]:
                infos[i] = {'reason': self.mt.REASONS[why[i]], 'lesson': self.mt.LESSON_KEYS[lesson[i]],
                            'steps': int(steps[i]), 'dist': float(dist[i]),
                            'terminal_observation': end[i],
                            'TimeLimit.truncated': bool(trunc[i] and not term[i])}
        return obs, np.asarray(r, np.float32), done, infos

    def env_method(self, method_name, *args, indices=None, **kwargs):
        out = getattr(self, method_name)(*args, **kwargs)
        return [out for _ in self._get_indices(indices)]

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def seed(self, seed=None):
        if seed is not None:
            import jax
            self.batch.key = jax.random.PRNGKey(seed)
        return [seed] * self.num_envs

    def close(self):
        pass


class Curriculum(BaseCallback):
    """Unlock the next lesson once every lesson in the pool is being passed, and keep
    the old ones in the pool so the policy does not forget them."""

    def __init__(self, pool, active=None, history=None, enabled=True):
        super().__init__()
        self.pool = list(pool)
        self.active = list(active or ([self.pool[0]] if enabled else self.pool))
        self.hist = {lesson_key(k): list(v) for k, v in (history or {}).items()}
        self.lens = {}                   # episode lengths per lesson, for the draw weights
        self.enabled = enabled

    def rate(self, lesson):
        h = self.hist.get(lesson, [])[-WINDOW:]
        return (len(h), sum(h) / len(h) if h else 0.0)

    def warming(self):
        """is the newest lesson still failing? -- for the log line only; the draw
        weights come from every lesson's rate (`weights`). Once it has passed
        WARM_RATE it counts as started for good, so the line is printed once."""
        if not self.enabled or len(self.active) < 2:
            return False
        newest = self.active[-1]
        passed = getattr(self, '_passed', set())
        if newest in passed:
            return False
        n, r = self.rate(newest)
        if n >= UNLOCK_MIN and r >= WARM_RATE:
            passed.add(newest)
            self._passed = passed
            return False
        return True

    def weights(self):
        """draw probabilities over the active pool: each lesson's share of the STEPS
        is how much it still fails, and the probability is that divided by its mean
        episode length.

        Both fixed rules failed, in opposite directions. Drawing by EPISODE gave the
        newest lesson ~96 % of the steps -- it times out at 1490 while a passed lesson
        ends in 20 -- and a lesson stuck for millions of steps took lesson -1 down to
        0-2 % (mac14, mac16, mac17). Fixed step shares fixed that and then starved the
        frontier: at 25 % lesson 1 got a quarter of the samples win9 gave it and win10
        was still at 0 % at 1.4M; at 60 % the curriculum ran -2 to lesson 2 in 2M
        steps (mac26, by far the best run) but lesson -1 fell to 0 % at 0.6M and never
        came back, because nothing pointed samples at a lesson that was not the newest.
        Proportional to what is still failing does both: a lesson at 0 % gets the
        samples whether it is the frontier or the one behind that regressed, and a
        mastered lesson keeps MIN_SHARE, which for a 20-step lesson is thousands of
        episodes. Measured over 2.5-4.5M steps each: this held every lesson at
        85-100 % including the one that had collapsed (mac27), where the fixed 60 %
        reached lesson 2 in 2M but left lesson -1 at 0 % from 0.6M on (mac26).
        Squaring the need concentrates harder on the worst lesson -- 66 % of the
        steps instead of 49 % for a stuck lesson 1.5 -- but with several lessons
        half-learnt it starves the rest, and the early curriculum stalled at lesson
        1 for 2M (mac28). Linear is the only one of the three with no failure mode
        measured; the frontier is slower for it, which more steps fix, while a
        forgotten lesson does not come back on its own.
        """
        need, lens = [], []
        for lesson in self.active:
            n, r = self.rate(lesson)
            need.append(max(1.0 - r, MIN_SHARE) if n else 1.0)
            v = [x for x in self.lens.get(lesson, [])[-WINDOW:] if x > 0]
            lens.append(sum(v) / len(v) if v else None)
        known = [x for x in lens if x is not None]
        fill = sum(known) / len(known) if known else 1.0
        w = [nd / (ln if ln is not None else fill) for nd, ln in zip(need, lens)]
        return [x / sum(w) for x in w]

    def state(self):
        return {'active': self.active,
                'history': {str(k): v[-WINDOW:] for k, v in self.hist.items()}}

    def restore(self, state):
        """back to a saved active set and history: the curriculum half of a rollback"""
        self.active = [k for k in map(lesson_key, state['active']) if k in self.pool] \
            or self.active
        self.hist = {lesson_key(k): list(v) for k, v in state['history'].items()}
        self._warm = self.warming()
        self.training_env.env_method('set_active', self.active, self.weights())

    def _on_step(self):
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done and 'reason' in info:
                self.hist.setdefault(info['lesson'], []).append(
                    info['reason'] == 'success')
                if info.get('steps'):
                    self.lens.setdefault(info['lesson'], []).append(info['steps'])
        if not self.enabled:
            return True
        warm = self.warming()
        if warm != getattr(self, '_warm', None):
            # the newest lesson just started being passed (or a resume): re-weight
            self._warm = warm
            self.training_env.env_method('set_active', self.active, self.weights())
            if not warm and len(self.active) > 1:
                print(f"\n[curriculum] lesson {self.active[-1]} is being passed\n")
        elif any(done for done in self.locals['dones']):
            # the lengths moved: keep the step shares honest (cheap; a list of floats)
            self.training_env.env_method('set_active', self.active, self.weights())
        locked = [k for k in self.pool if k not in self.active]
        if not locked:
            return True
        # every lesson in the pool, not only the newest: see UNLOCK_AT
        rates = [self.rate(k) for k in self.active]
        n, r = rates[-1]
        if all(m >= UNLOCK_MIN and q >= UNLOCK_AT for m, q in rates):
            # the next lesson is the first one in the pool that is not yet active,
            # NOT `pool[len(active)]`: inserting a lesson into the middle of the pool
            # on a resume (1.85 between 1.7 and 2, say) leaves the restored active set
            # out of step with pool positions, and the index form then re-appends a
            # lesson that is already active while the new one is never trained at all
            self.active.append(locked[0])
            self.hist.setdefault(self.active[-1], [])
            self._warm = self.warming()
            self.training_env.env_method('set_active', self.active, self.weights())
            print(f"\n[curriculum] every lesson at {UNLOCK_AT:.0%}+, the newest "
                  f"({self.active[-2]}) at {r:.0%} over {n} "
                  f"episodes -> unlocking lesson {self.active[-1]} "
                  f"({LESSONS[self.active[-1]]['name']}); pool now {self.active}, "
                  f"the pool's steps go where the failures are\n")
        return True


class BoundExploration(BaseCallback):
    """Keep the action noise from running away: log_std may never exceed its own
    starting value.

    `win11` (15.3M steps) peaked at 8.0M -- lesson 1.35 at 92 %, lesson 1.5 at 52 %,
    the best policy this project has had -- and then took itself apart: by 15.3M
    lesson 1.5 and 2 were at 0 %, -1 at 28 %, and the DETERMINISTIC policy failed
    too (0/10 on lesson 1.5), so it was the mean that went, not only the sampling.
    The signature is the action std: 0.48 at the peak, 1.33 at the end, 4.24 on one
    joint. It is a feedback loop -- once most of the pool is failing, every action
    looks equally bad, the advantage signal flattens, `ent_coef` is the only term
    left with an opinion and it pays for noise; the noisier policy fails more.
    SB3 starts log_std at 0 (std 1.0) and every good run here came DOWN from it
    (0.3-0.6), so nothing legitimate needs more. This does not touch the swings
    themselves; it stops them from becoming a spiral.
    """

    def _on_rollout_start(self):
        # ...and never below STD_MIN, the other way the noise ends a run (see there)
        self.model.policy.log_std.data.clamp_(min=float(np.log(STD_MIN)), max=0.0)
        return True

    def _on_step(self):
        return True


class AdaptiveLR(BaseCallback):
    """Move the learning rate so each update changes the policy by about the run's
    KL target: x1/1.5 when the last update's approx_kl passed twice it, x1.5 when it
    was under half, within [LR_MIN, LR_MAX]. The update that just finished has
    recorded its KL by the time the next rollout starts, and nothing has written it
    out yet. The target belongs to the checkpoint callback, which lowers it whenever
    the guard has to undo a drift."""

    def __init__(self, checkpoint):
        super().__init__()
        self.ck = checkpoint

    def _on_rollout_start(self):
        kl = self.model.logger.name_to_value.get('train/approx_kl')
        if kl is None:
            return
        lr, target = self.model.lr_schedule(1.0), self.ck.kl_target
        new = (max(lr / 1.5, LR_MIN) if kl > 2 * target else
               min(lr * 1.5, LR_MAX) if kl < target / 2 else lr)
        if new != lr:
            self.model.learning_rate = new
            self.model._setup_lr_schedule()

    def _on_step(self):
        return True


class Checkpoint(BaseCallback):
    """Model + observation statistics + curriculum state, written together.

    All three or none: a model restored against stale VecNormalize statistics sees a
    different observation scale than it was trained on, which reads as the policy
    having silently degraded.
    """

    def __init__(self, run_dir, curriculum, every=100_000, best=None, best_active=None,
                 rollbacks=0, kl_target=None, backend='cpu'):
        super().__init__()
        self.dir, self.cur, self.every, self.backend = run_dir, curriculum, every, backend
        self.last, self.best = 0, best if best is not None else -1.0
        self.kl_target = kl_target or KL_START
        # the lessons the best checkpoint was scored over; `held` scores them now
        self.best_active = [lesson_key(k) for k in (best_active or [])]
        self.low_since, self.due, self.rollbacks = None, False, rollbacks

    def score(self):
        """SUM of the success rates over the active lessons. Not the mean.

        The mean was a bug with a clean signature. When lesson 2 unlocked at 0 % it
        halved the score, so on the 10.4M run the stored best froze at step 308 512
        with 57 % while the policy went on to reach 85 % on lesson 1 -- and
        best_model.zip held the weaker of the two. The comment here used to claim the
        mean kept the number comparable across an unlock; it did the opposite.

        A sum has the three properties that matter: unlocking a lesson adds 0, so it
        can never lower the score; getting better at any lesson raises it; and
        forgetting an old one lowers it, which a "score the newest lesson" rule would
        have missed.
        """
        rates = [self.cur.rate(k) for k in self.cur.active]
        if any(n < UNLOCK_MIN for n, _ in rates):
            return None
        return sum(r for _, r in rates)

    def held(self):
        """The best checkpoint's own lessons, scored now. Forgetting lowers it; a
        lesson unlocked since the best cannot, so a new lesson's first failures are
        never read as the pool coming apart."""
        rates = [self.cur.rate(k) for k in self.best_active]
        if not rates or any(n < UNLOCK_MIN for n, _ in rates):
            return None
        return sum(r for _, r in rates)

    def guard(self):
        """at each checkpoint: has the pool been a lesson below its best for long
        enough to go back? The rollback itself waits for the next rollout to start
        (`_on_rollout_start`), so no update ever mixes the two policies' data."""
        h = self.held()
        if h is None or h >= self.best - GUARD_DROP:
            self.low_since = None
            return ""
        if self.low_since is None:
            self.low_since = self.model.num_timesteps
        low = self.model.num_timesteps - self.low_since
        if low >= GUARD_STEPS:
            self.due = True
        return f" | forgetting: {h:.2f} vs best {self.best:.2f} for {low:,} steps"

    def _on_rollout_start(self):
        if self.due:
            self.rollback()

    def rollback(self):
        """Back to the best checkpoint -- policy and optimizer, observation
        statistics and curriculum, the same three `save` writes together -- and on
        from there at half the learning rate. What a person did by hand after every
        drift: stop, `--resume --from-best`, try a smaller step."""
        m, p = self.model, os.path.join(self.dir, 'best_')
        h = self.held()
        m.set_parameters(p + 'model.zip', exact_match=True, device=m.device)
        with open(p + 'vecnorm.pkl', 'rb') as f:
            stats = pickle.load(f)
        env = m.get_vec_normalize_env()
        env.obs_rms, env.ret_rms = stats.obs_rms, stats.ret_rms
        self.cur.restore(json.load(open(p + 'state.json')))
        was, was_kl = m.lr_schedule(1.0), self.kl_target
        m.learning_rate = max(was * LR_DECAY, LR_MIN)
        m._setup_lr_schedule()              # lr_schedule is built from learning_rate
        # ...and the target the rate is adapted to, which is what actually sticks
        self.kl_target = max(self.kl_target * LR_DECAY, KL_MIN)
        self.rollbacks += 1
        self.low_since, self.due = None, False
        print(f"\n[guard] {h:.2f} against the best {self.best:.2f} for "
              f"{GUARD_STEPS:,}+ steps: back to the best checkpoint (pool "
              f"{self.cur.active}), learning rate {was:g} -> {m.learning_rate:g}, "
              f"KL target {was_kl:g} -> {self.kl_target:g}, "
              f"rollback {self.rollbacks}\n")
        self.save()

    def save(self, tag=''):
        os.makedirs(self.dir, exist_ok=True)
        p = os.path.join(self.dir, tag or '')
        self.model.save(p + 'model')
        self.model.get_vec_normalize_env().save(p + 'vecnorm.pkl')
        state = {'timesteps': int(self.model.num_timesteps), 'robot': robot_stamp(),
                 'backend': self.backend, 'best': self.best, 'best_active': self.best_active,
                 'lr': self.model.lr_schedule(1.0), 'kl_target': self.kl_target,
                 'rollbacks': self.rollbacks,
                 **self.cur.state()}
        json.dump(state, open(p + 'state.json', 'w'), indent=1)
        return p

    def _on_step(self):
        # Keep the best policy, not just the newest. The 2.6M checkpoint that scored
        # 12/12 was overwritten by a later, worse one and had to be re-run from a
        # backup copy; a run that peaks and then drifts should not lose its peak.
        s = self.score()
        if s is not None and s > self.best:
            self.best, self.best_active, self.low_since = s, list(self.cur.active), None
            self.save('best_')
            print(f"[best] score {s:.2f} over {len(self.cur.active)} lesson(s) "
                  f"at {self.model.num_timesteps:,} steps")
        if self.model.num_timesteps - self.last >= self.every:
            self.last = self.model.num_timesteps
            self.save()
            rates = " ".join(f"L{k} {self.cur.rate(k)[1]:.0%}" for k in self.cur.active)
            print(f"[checkpoint] {self.model.num_timesteps:,} steps | {rates} | "
                  f"best {self.best:.2f} | lr {self.model.lr_schedule(1.0):g} "
                  f"kl<={self.kl_target:g}"
                  f"{self.guard()} | out/{os.path.basename(self.dir)}")
        return True


def self_check():
    """Exercise the curriculum's unlock rule and its round-trip through state.json.

        python -m rl.train_ppo --self-check
    """
    class FakeEnv:
        def __init__(self):
            self.calls = []

        def env_method(self, name, *a):
            self.calls.append((name, a))

    class FakeModel:                 # BaseCallback.training_env reads model.get_env()
        def __init__(self, env):
            self.env = env

        def get_env(self):
            return self.env

    def feed(c, lesson, reason, n=1, steps=0):
        for _ in range(n):
            c.locals = {'infos': [{'reason': reason, 'lesson': lesson, 'steps': steps}],
                        'dones': [True]}
            c._on_step()

    env = FakeEnv()
    c = Curriculum([1, 2, 3])
    c.model = FakeModel(env)
    assert c.active == [1], "curriculum must start with one lesson"

    feed(c, 1, 'success', UNLOCK_MIN - 1)
    assert c.active == [1], f"unlocked on {UNLOCK_MIN - 1} episodes, below UNLOCK_MIN"
    feed(c, 1, 'success')
    assert c.active == [1, 2], f"did not unlock at {UNLOCK_MIN} successes: {c.active}"
    assert env.calls[-1][0] == 'set_active' and env.calls[-1][1][0] == [1, 2], env.calls[-1]
    # the point of mixing: unlocking must never drop a lesson from the pool
    assert 1 in c.active, "lesson 1 left the pool -- that is the forgetting bug"

    feed(c, 2, 'fell', WINDOW)
    assert c.active == [1, 2], "unlocked lesson 3 while lesson 2 was failing"
    # ...and while it fails it gets the samples: the steps go where the failures are
    w = c.weights()
    assert w[1] > 5 * w[0], f"a lesson at 0 % must outweigh one at 100 %: {w}"
    feed(c, 2, 'success', WINDOW)
    assert c.active == [1, 2, 3], "never unlocked lesson 3 despite a clean window"
    assert np.allclose(c.weights()[:2], c.weights()[1]) and c.weights()[2] > c.weights()[0], (
        "two mastered lessons must share equally and the failing new one get more: "
        f"{c.weights()}")
    # The share is of the STEPS, not the episodes: a failing lesson whose episodes
    # run 1490 steps against 200 must not swallow the pool by being slow (drawing
    # by episode gave it 96 % of the samples and cost lesson -1), and must not be
    # starved either (a fixed 25 % of the steps cost win10 lesson 1).
    feed(c, 1, 'success', WINDOW, steps=200)
    feed(c, 2, 'success', WINDOW, steps=200)
    feed(c, 3, 'fell', WINDOW, steps=1490)
    w = c.weights()
    steps = [w[0] * 200, w[1] * 200, w[2] * 1490]
    share = steps[2] / sum(steps)
    assert 0.5 < share < 0.95, f"the failing lesson's share of the steps is {share:.0%}: {w}"
    assert min(steps[0], steps[1]) / sum(steps) > 0.05, (
        f"a mastered lesson must keep its share of the steps: {[x/sum(steps) for x in steps]}")
    # ...and a lesson that regressed gets them too, even when it is not the newest:
    # lesson -1 sat at 0 % from 0.6M to 2M in mac26 because only the frontier was fed
    feed(c, 1, 'fell', WINDOW, steps=1490)
    w = c.weights()
    assert w[0] * 1490 > 0.4 * (w[0] * 1490 + w[1] * 200 + w[2] * 1490), (
        f"a regressed lesson behind the frontier must get the samples: {w}")
    feed(c, 1, 'success', WINDOW, steps=200)
    feed(c, 3, 'success', WINDOW, steps=400)
    assert c.active == [1, 2, 3], "unlocked past the end of the pool"
    w = c.weights()
    assert abs(w[0] * 200 - w[2] * 400) < 1e-9, f"a mastered pool must share the steps evenly: {w}"

    n, r = c.rate(2)
    assert n == WINDOW and r == 1.0, (n, r)
    restored = Curriculum([1, 2, 3], **json.loads(json.dumps(c.state())))
    assert restored.active == c.active, (restored.active, c.active)
    assert restored.rate(2) == c.rate(2), "history did not survive state.json"

    # a resume onto another robot is refused unless asked for
    check_robot({'robot': robot_stamp()}, False)
    for stale in ({}, {'robot': 'not-this-robot'}):
        try:
            check_robot(stale, False)
            raise AssertionError(f"resumed {stale} onto a different robot")
        except SystemExit:
            pass
        check_robot(stale, True)

    off = Curriculum([1, 2, 3], enabled=False)
    assert off.active == [1, 2, 3], "--no-curriculum must start with the whole pool"

    # the gate: a lesson slipping behind the frontier holds the next one back
    g = Curriculum([1, 2, 3])
    g.model = FakeModel(FakeEnv())
    feed(g, 1, 'success', WINDOW)
    feed(g, 1, 'fell', WINDOW // 2)                 # lesson 1 slips to 50 %
    feed(g, 2, 'success', WINDOW)
    assert g.active == [1, 2], f"unlocked while lesson 1 was being forgotten: {g.active}"
    feed(g, 1, 'success', WINDOW)
    assert g.active == [1, 2, 3], f"never unlocked once lesson 1 recovered: {g.active}"

    guard_check(FakeModel, FakeEnv, feed)

    # The best-checkpoint score must never fall just because a lesson unlocked. Under
    # the old mean it did: 85% on lesson 1 alone scored 0.85, and unlocking lesson 2
    # at 0% dropped it to 0.42, so the 10.4M run kept a 308k-step policy as "best".
    class Cp:
        active = None
        def rate(self, k):
            return (WINDOW, {1: 0.85, 2: 0.0, 3: 0.0}[k])
    cp, sc = Cp(), Checkpoint.score
    cp.active = [1]
    before = sc(type('X', (), {'cur': cp})())
    cp.active = [1, 2]
    after = sc(type('X', (), {'cur': cp})())
    print(f"score test: lesson 1 at 85% scores {before:.2f}; unlocking lesson 2 at 0% "
          f"scores {after:.2f}")
    assert after >= before, (f"unlocking a lesson dropped the score {before:.2f} -> "
                             f"{after:.2f}; best_model.zip will freeze at the old one")
    print(f"curriculum: unlocks at {UNLOCK_AT:.0%} over {UNLOCK_MIN}+ episodes on EVERY "
          f"lesson in the pool, keeps them all, and spends its STEPS where the failures "
          f"are -- frontier or regressed alike, {MIN_SHARE:.0%} floor for a mastered one. OK")


def guard_check(FakeModel, FakeEnv, feed):
    """A real (tiny) PPO: make a best, let the pool forget, and see the run put
    itself back -- policy, optimizer, statistics, curriculum -- at half the rate."""
    import tempfile
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv

    class Toy(gym.Env):
        observation_space = gym.spaces.Box(-5.0, 5.0, (3,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros(3, np.float32), {}

        def step(self, a):
            return np.random.randn(3).astype(np.float32), 0.0, False, False, {}

    out = sys.stdout
    with tempfile.TemporaryDirectory() as d:
        venv = VecNormalize(DummyVecEnv([Toy]), norm_reward=False)
        m = PPO('MlpPolicy', venv, n_steps=16, batch_size=16, verbose=0)
        cur = Curriculum([1, 2])
        cur.model = FakeModel(FakeEnv())
        feed(cur, 1, 'success', WINDOW)               # pool [1, 2]
        feed(cur, 2, 'success', UNLOCK_MIN)
        ck = Checkpoint(d, cur, every=1)
        ck.init_callback(m)
        sys.stdout = open(os.devnull, 'w')            # its [best]/[checkpoint] lines
        try:
            ck._on_step()                             # a best, over [1, 2]
            assert ck.best == 2.0 and ck.best_active == [1, 2], (ck.best, ck.best_active)
            best_w = {k: v.clone() for k, v in m.policy.state_dict().items()}
            best_mean = venv.obs_rms.mean.copy()
            with torch.no_grad():                     # the drift: weights, statistics...
                for v in m.policy.parameters():
                    v.add_(0.1)
            venv.obs_rms.mean += 1.0
            feed(cur, 1, 'fell', WINDOW)              # ...and lesson 1 forgotten
            feed(cur, 2, 'fell', 5)
            m.num_timesteps = 10
            ck._on_step()
            assert not ck.due and ck.low_since == 10, "tripped on the first low checkpoint"
            m.num_timesteps = 10 + GUARD_STEPS - 1
            ck._on_step()
            assert not ck.due, "tripped before GUARD_STEPS"
            m.num_timesteps = 10 + GUARD_STEPS
            ck._on_step()
            assert ck.due, f"a lesson forgotten for {GUARD_STEPS:,} steps did not trip it"
            ck._on_rollout_start()
        finally:
            sys.stdout.close()
            sys.stdout = out
        assert all(torch.equal(v, best_w[k]) for k, v in m.policy.state_dict().items()), \
            "rollback did not restore the best policy"
        assert np.allclose(venv.obs_rms.mean, best_mean), \
            "rollback did not restore the observation statistics"
        assert cur.rate(1)[1] == 1.0 and cur.active == [1, 2], \
            "rollback did not restore the curriculum"
        assert abs(m.lr_schedule(1.0) - 3e-4 * LR_DECAY) < 1e-12, m.lr_schedule(1.0)
        st = json.load(open(os.path.join(d, 'state.json')))
        assert st['rollbacks'] == 1 and abs(st['lr'] - 3e-4 * LR_DECAY) < 1e-12, st
        assert not ck.due and ck.low_since is None
        sys.stdout = open(os.devnull, 'w')
        try:
            for _ in range(12):
                ck.rollback()
        finally:
            sys.stdout.close()
            sys.stdout = out
        assert abs(m.lr_schedule(1.0) - LR_MIN) < 1e-15, "the learning rate went below LR_MIN"
        ck.kl_target, ck.rollbacks = KL_START, 0      # the target's own test is below

        # the noise floor and ceiling, on the same model
        be = BoundExploration()
        be.init_callback(m)
        with torch.no_grad():
            m.policy.log_std.copy_(torch.tensor([-6.0, 1.0]))
        be._on_rollout_start()
        sd = m.policy.log_std.exp().detach().numpy()
        assert abs(sd[0] - STD_MIN) < 1e-6 and abs(sd[1] - 1.0) < 1e-6, \
            f"std not held in [STD_MIN, 1]: {sd}"
        # the learning rate follows the KL of the update before, within its bounds
        from stable_baselines3.common.logger import configure
        m.set_logger(configure(None, []))
        al = AdaptiveLR(ck)
        al.init_callback(m)

        def after(kl, lr):
            m.learning_rate = lr
            m._setup_lr_schedule()
            m.logger.name_to_value['train/approx_kl'] = kl
            al._on_rollout_start()
            return m.lr_schedule(1.0)
        kt = ck.kl_target
        assert abs(after(10 * kt, 1.5e-4) - 1e-4) < 1e-12, "a big update did not slow it"
        assert abs(after(0.1 * kt, 1e-4) - 1.5e-4) < 1e-12, "a small one did not speed it"
        assert after(kt, 1e-4) == 1e-4, "an update on target moved the rate"
        assert after(0.1 * kt, LR_MAX) == LR_MAX, "the rate went past LR_MAX"
        assert after(10 * kt, LR_MIN) == LR_MIN, "the rate went under LR_MIN"
        # the guard owns that target: every rollback halves it, down to KL_MIN
        assert abs(ck.kl_target - KL_START) < 1e-12, ck.kl_target
        sys.stdout = open(os.devnull, 'w')
        try:
            ck.rollback()
            first = ck.kl_target
            for _ in range(12):
                ck.rollback()
        finally:
            sys.stdout.close()
            sys.stdout = out
        assert abs(first - KL_START * LR_DECAY) < 1e-12, first
        assert abs(ck.kl_target - KL_MIN) < 1e-12, "the KL target went below KL_MIN"
    print(f"guard test: the best's lessons a whole lesson down for {GUARD_STEPS:,} steps "
          f"sends the run back to its best checkpoint (policy, statistics, curriculum) at "
          f"x{LR_DECAY:g} the learning rate, never below {LR_MIN:g}. OK")
    print(f"update-size test: action std held in [{STD_MIN:g}, 1]; the learning rate "
          f"follows each update's KL toward the run's target, within [{LR_MIN:g}, "
          f"{LR_MAX:g}]; the target starts at {KL_START:g} and each rollback halves "
          f"it to {KL_MIN:g}. OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='climber', help='run directory under out/')
    # A pool that starts with a negative lesson must be written --lessons=-2,-1,...
    # With a space, argparse before Python 3.14 reads "-2,-1,..." as an unknown
    # option and stops with "expected one argument"; the Mac venv is 3.14 and took
    # it, the Windows one did not.
    ap.add_argument('--lessons', default=','.join(str(k) for k in sorted(LESSONS)),
                    help='comma-separated; if it starts with a minus, use --lessons=-2,...')
    ap.add_argument('--steps', type=int, default=2_000_000,
                    help='steps to run; with --resume, ADDITIONAL steps')
    # Default to the machine's core count rather than a number tuned here -- on a
    # 64-core Linux box, 12 would leave most of it idle. Measured end-to-end fps on
    # lesson 1 on the 12-core M4 Pro: 6->3413, 8->3950, 12->4330, 20->4977,
    # 32->5259. It keeps creeping up past the core count because the envs sit idle
    # during each policy update, so on a big machine try more than this default.
    # Each env holds every lesson scene, ~330 MB without visual meshes (see
    # MultiLesson), and each worker process imports torch as well: budget RAM at
    # --envs x 0.6 GB.
    ap.add_argument('--envs', type=int, default=None,
                    help=f'default: the usable cores on the cpu backend, {MJX_ENVS} on mjx')
    ap.add_argument('--device', default=None,
                    choices=['cpu', 'cuda', 'mps', 'auto'],
                    help='policy device. Default cpu on the cpu backend -- an MlpPolicy '
                         'is usually too small to beat it, run `python -m rl.bench` '
                         'before deciding -- and cuda on mjx, whose batches are not')
    ap.add_argument('--backend', default='auto', choices=['auto', 'cpu', 'mjx'],
                    help='where the simulator runs; auto = mjx when rl.hardware finds '
                         'a GPU JAX can use, else cpu')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--from-best', dest='from_best', action='store_true',
                    help='with --resume, continue from best_model.zip instead of the '
                         'latest checkpoint -- what to do after a run has degraded')
    ap.add_argument('--new-robot', dest='new_robot', action='store_true',
                    help='with --resume, accept a checkpoint trained on a different '
                         'robot model (fine-tuning across a model change)')
    ap.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                    help='put every lesson in the pool from step 0')
    # 3e-3 was too high: after 4M steps the action std had grown to 1.52, above the
    # 1.0 it starts at, so the entropy bonus was beating the policy gradient and the
    # mean action stayed useless -- deterministic eval fell 12/12 while sampling from
    # the same policy succeeded 8/12. Lower it and the distribution can tighten.
    # Left unset it means "whatever this run already uses": a fresh run gets 1e-3,
    # a resume keeps what its checkpoint was trained with. Passing it on a resume
    # used to do NOTHING -- `PPO.load` restores the saved value and only the fresh
    # branch below read the flag -- so `win15 3` was asked for 3e-4 and trained all
    # 10M steps at 1e-3 without a word. A silently ignored knob is worse than no
    # knob; the resume now prints what it will actually use.
    ap.add_argument('--ent-coef', type=float, default=None)
    # Same "unset means keep what this run uses" rule as --ent-coef, and the same
    # reason: PPO.load restores both, so a resume that wants a smaller step has to
    # say so and be told what it got. Fine-tuning a peak is not the regime the 3e-4
    # default was chosen in -- that was the early curriculum (mac21-mac24).
    ap.add_argument('--lr', type=float, default=None,
                    help='learning rate; on a resume, overrides the checkpoint')
    ap.add_argument('--target-kl', dest='target_kl', type=float, default=None,
                    help='stop an update early once approx_kl passes 1.5x this')
    # The one knob for how much the policy can HOLD. SB3's default is 64x64 for both
    # nets, which is what every run here has used; twelve lessons through it is a
    # standing suspect for the interference the guard keeps catching, and rsl_rl's
    # locomotion policies are 512-256-128. It cannot be changed on a resume (the
    # checkpoint has the old shape) and it is nearly free in wall clock -- the
    # simulator is the bottleneck, and cutting 90 % of the gradient work moved
    # throughput by 0 %. mac22 tried 256x256 early in the curriculum and lesson -2
    # stalled for 1.5M steps, before the adaptive rate existed to size the steps for
    # it; it is worth re-running as an arm against a 64x64 one, not as a default.
    ap.add_argument('--net', default=None,
                    help='hidden sizes for both nets, e.g. 256,256 (default 64,64); '
                         'fresh runs only')
    ap.add_argument('--ckpt-every', type=int, default=None,
                    help='steps between checkpoints; default 100k on cpu, 1M on mjx')
    ap.add_argument('--self-check', action='store_true',
                    help='test the curriculum logic, train nothing')
    a = ap.parse_args()
    if a.self_check:
        return self_check()

    pool = [lesson_key(x) for x in a.lessons.split(',')]
    if a.net and a.resume:
        raise SystemExit("--net cannot change on a resume: the checkpoint holds the "
                         "old shape. Start a new run with a new --name.")
    run = os.path.join('out', a.name)
    # the console, kept: approx_kl, clip_fraction and explained_variance per update
    # are the evidence a checkpoint cannot reconstruct, and every rollback and unlock
    # is printed here. Appended, so a resumed run keeps one log.
    os.makedirs(run, exist_ok=True)
    log = open(os.path.join(run, 'train.log'), 'a', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")
    print(hardware.report())
    backend = hardware.backend()[0] if a.backend == 'auto' else a.backend
    mjx = backend == 'mjx'
    envs = a.envs or (MJX_ENVS if mjx else default_envs())
    if a.device is None:
        import torch
        a.device = 'cuda' if mjx and torch.cuda.is_available() else 'cpu'
    # the update's shape follows the batch; everything else is the run's own
    shape = (dict(n_steps=MJX_PPO['n_steps'], n_epochs=MJX_PPO['n_epochs'],
                  batch_size=envs * MJX_PPO['n_steps'] // MJX_PPO['minibatches'])
             if mjx else dict(n_steps=256, n_epochs=10, batch_size=512))
    print(f"[backend] {backend}: {envs} envs, policy on {a.device}, "
          f"{envs * shape['n_steps']:,} steps per update")
    state = {}
    ck = 'best_' if a.from_best else ''
    if a.resume:
        f = os.path.join(run, ck + 'state.json')
        if not os.path.exists(f):
            raise SystemExit(f"nothing to resume: {f} does not exist")
        state = json.load(open(f))
        check_robot(state, a.new_robot)
        was = state.get('backend', 'cpu')
        if was != backend:
            print(f"[backend] this run trained on {was} until now and carries on on "
                  f"{backend}: the same task, and mjx's simplified robot is checked "
                  f"against the full one by rl.mjx_check")

    base = VecMonitor(MjxVecEnv(pool, envs) if mjx else SubprocVecEnv(
        [(lambda i=i: MultiLesson(pool, seed=i)) for i in range(envs)]))
    if a.resume:
        venv = VecNormalize.load(os.path.join(run, ck + 'vecnorm.pkl'), base)
        venv.training = True
        # custom_objects: a checkpoint carries its backend's rollout shape, and 256
        # steps x 4096 envs would be a million-sample update
        model = PPO.load(os.path.join(run, ck + 'model'), env=venv, device=a.device,
                         custom_objects=shape)
        if a.ent_coef is not None:
            model.ent_coef = a.ent_coef
        if a.lr is not None:
            model.learning_rate = a.lr
            model._setup_lr_schedule()      # lr_schedule is built from learning_rate
        if a.target_kl is not None:
            model.target_kl = a.target_kl
        print(f"resumed {run} at {model.num_timesteps} steps "
              f"ent_coef {model.ent_coef:g} lr {model.lr_schedule(1.0):g} "
              f"target_kl {model.target_kl} "
              f"{'from its BEST checkpoint ' if a.from_best else ''}"
              f"pool {state.get('active')}")
    else:
        venv = VecNormalize(base, norm_reward=False)
        # SB3's defaults, and every attempt to improve on them cost the early
        # curriculum. With seven lessons in the pool the policy swings between
        # them (lesson 1 went 92 % -> 5 % in 2M steps three times in mac20, the
        # estimator right throughout), so: 8192 steps per update left lesson -1 at
        # 0 % for 4M steps (mac21); a 256x256 net left lesson -2 at 0 % for 1.5M
        # (mac22); a linear learning-rate decay plus a VecNormalize variance floor
        # left -1 at 0 % for 1.1M (mac24) -- against 0.14M here. The swings are
        # real and unsolved; `best_` keeps the peak. Do not change these three
        # (the cpu `shape` above, the net) without a 1M run that reaches lesson 1:
        # that is the cheap canary.
        net = [int(x) for x in a.net.split(',')] if a.net else None
        model = PPO('MlpPolicy', venv, device=a.device, verbose=1, **shape,
                    gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4 if a.lr is None else a.lr,
                    target_kl=a.target_kl,
                    policy_kwargs=None if net is None else dict(net_arch=net),
                    ent_coef=1e-3 if a.ent_coef is None else a.ent_coef)
        if net:
            print(f"policy and value nets {net} (SB3's default is [64, 64])")

    # A resume may narrow the pool -- dropping a mastered lesson is the point of
    # `--lessons` on a resume -- so the restored active set has to be filtered to it
    # rather than carried over wholesale.
    active = [k for k in (state.get('active') or []) if k in pool] or None
    # A narrowed resume changes what the score MEANS. `score` sums the success
    # rates over the active lessons, so a best of 6.35 earned over eight of them
    # can never be beaten by three, and `best_model.zip` is then frozen for the
    # whole run: a check that resumed win15's best on lessons 1.2-2 took lesson 1.5
    # from 10 % to 22 % and then swung back to 2 %, and the 22 % policy was never
    # written. Dropping a lesson resets the bar.
    best = state.get('best')
    if active is not None and active != (state.get('active') or []):
        print(f"[curriculum] pool narrowed to {active}; the best score starts over")
        best = None
    cur = Curriculum(pool, active, state.get('history'), a.curriculum)
    venv.env_method('set_active', cur.active, cur.weights())
    # which lessons the best was scored over: in the state since the guard exists,
    # and before that it is best_state.json's own active set
    best_active = state.get('best_active')
    bf = os.path.join(run, 'best_state.json')
    if best is not None and not best_active and os.path.exists(bf):
        best_active = json.load(open(bf))['active']
    ckpt = Checkpoint(run, cur, a.ckpt_every or (1_000_000 if mjx else 100_000), best,
                      best_active if best is not None else None,
                      state.get('rollbacks', 0), state.get('kl_target'), backend)

    # Ctrl-C during a rollout would otherwise lose everything since the last
    # checkpoint; SIGTERM likewise, so a killed run is resumable too.
    try:
        signal.signal(signal.SIGTERM,
                      lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    except (ValueError, AttributeError, OSError):
        pass          # Windows accepts only a short list of signals; Ctrl-C still works
    try:
        # no progress_bar: it pulls in tqdm + rich, and verbose=1 already prints the
        # rollout table with elapsed time and total_timesteps every update
        model.learn(total_timesteps=a.steps,
                    callback=[cur, ckpt, AdaptiveLR(ckpt), BoundExploration()],
                    reset_num_timesteps=not a.resume)
    except KeyboardInterrupt:
        print("\ninterrupted -- saving")
    except (BrokenPipeError, EOFError, ConnectionResetError) as e:
        # A worker process ended. On Windows that is what Ctrl-C usually looks
        # like from here -- the console signal reaches the workers first and the
        # parent reads a closed pipe instead of KeyboardInterrupt -- and it is
        # also what an out-of-memory kill looks like. Either way the run is over
        # and the last checkpoint window must not go with it: win17 lost the
        # 55k steps since its last one this way, at 41.7M.
        print(f"\na worker process ended ({type(e).__name__}) -- saving")
    ckpt.save()
    for k in pool:
        n, r = cur.rate(k)
        if n:
            print(f"  lesson {k} {LESSONS[k]['name']:>20}: {r:.0%} over last {n}")
    print(f"resume with: python -m rl.train_ppo --name {a.name} --resume")


if __name__ == '__main__':
    main()
