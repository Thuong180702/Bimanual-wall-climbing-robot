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
that best for 325 policy updates (`GUARD_DROP`, `GUARD_UPDATES`) -- it goes back to
the best checkpoint by itself and carries on at half the learning rate. Everything
it prints is also appended to out/<name>/train.log.

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
learning something a search already does: which hold next is A* over the wall map,
deterministic (the classical controller did it; git history). Learning that choice
is a much harder problem than the one it solves.

*One policy over all lessons* shares the common skill and needs no dispatcher,
because the goal is already in the observation -- it is goal-conditioned, not
lesson-conditioned, and at deploy time the planner supplies the goal exactly as the
lesson spec does here. The failure mode to avoid is catastrophic forgetting, which
is what training the lessons strictly one after another would cause: fine-tune on
lesson 9 and lesson 1 is gone. So lessons are MIXED, not sequential -- each env
picks from the active pool on every reset -- and the curriculum only controls which
lessons are IN the pool. Nothing ever leaves it.
"""
import argparse, collections, faulthandler, hashlib, heapq, json, os, pickle, signal, sys, time
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor, VecNormalize
except ImportError as e:
    raise SystemExit(f"missing training deps ({e}); pip install gymnasium stable-baselines3")

from . import SCENE
from . import default_envs, hardware, rsi
from .climb_task import CONTROL_DT, ClimbTask
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
GUARD_UPDATES = 325  # score -- a whole lesson's worth -- for this many policy updates
                     # sends the run back to that checkpoint. Every long run from a
                     # peak has drifted off it over millions of steps (win11, win15 2,
                     # win16, win16 2, win16 3, win17) and each one needed a person to
                     # notice and resume from best. It was a million steps, because
                     # the first exposure to a new lesson dips and recovers inside
                     # 600k on its own (the win16 A/B), and that should not trip it.
                     # Counted in UPDATES now, because the policy only changes at an
                     # update, so that is the clock a dip and its recovery run on: 325
                     # is that million on the CPU path's 12 envs x 256 steps. An MJX
                     # update is 4096 x 32 = 131k steps, and a million steps there is
                     # eight updates -- every new lesson's first dip would roll back.
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
WALL_WINDOW = 200    # WINDOW and UNLOCK_MIN on the generated walls (Walls), and the probes
                     # an ADR vote counts. 40 was sized for 12 CPU envs. 4096 envs end
                     # hundreds of a family's episodes between two updates and the gate
                     # is asked after every one, so a 40-episode window is a sample
                     # drawn again and again: a family truly at 60 % passes it at 70 %
                     # 13 % of the time and one at 65 % 31 % -- the 70 % gate was a
                     # 55-65 % one. Over 200: 0.2 % and 8 %. It costs no time: a vote
                     # waits a cohort's lag (6.2M steps on MJX) for its probes anyway.
STUCK_STEPS = 50_000_000   # a family in the pool this long that has never once passed
                     # stops holding the next one back (Walls). gpu1's family 3 sat at
                     # 0 % for 90M steps with 4-9 locked behind it; families that could
                     # be learnt passed the gate 6-12M steps after they entered (1 and
                     # 2, the first moments the cohort lag let them be counted). A family
                     # that HAS passed blocks again whenever it slips: that is
                     # consolidation, not a jam. The stuck one stays in the pool and
                     # keeps the steps its failures draw (`weights`).
BANK = 1024          # generated walls drawn at the start of every rollout (rl.walls); each
                     # env takes one at every reset, so a 4096-env batch meets ~1000 new
                     # walls an update and never the same bank twice
MJX_ENVS = 4096      # envs in one GPU batch, 33 KB of state each (130 MB). ponytail:
                     # the usual MJX size, unmeasured on a GPU here -- `python -m
                     # rl.mjx_task --envs N` on the card decides
# PPO on the GPU batch. The CPU numbers below (256 steps x 12 envs, 512 per
# minibatch, 10 epochs) would make one update a million samples, so the rollout is
# the massively-parallel one instead (rsl_rl's 24-32 steps an env). What an update
# does with it is the CPU path's: 10 epochs, and minibatches (8, a power of two so
# 4096 x 32 splits evenly) that make its Adam steps 80 against the CPU path's 60, at
# the same learning rate, its same cap and the same KL target. rsl_rl's own 4 x 5 =
# 20 steps were tried first and moved the policy 20x less an update than every CPU
# run from scratch had (approx_kl 0.0007-0.005 against 0.012-0.017), with the rate
# already at its 3e-4 cap -- lesson -2 would have cost hundreds of millions of steps.
MJX_PPO = dict(n_steps=32, minibatches=8, n_epochs=10)
# ...and its nets: 256x256 for both, a fresh run's default on MJX (--net overrides, a
# resume keeps its checkpoint's). Nine families in SB3's 64x64 is the standing suspect
# for the forgetting the guard keeps catching, rsl_rl's locomotion policies on such
# batches are 512-256-128, and on the GPU the net's cost is lost in the physics. The CPU
# path keeps 64x64: the 2026-10-04 canary on its 3k-sample updates had 256x256 hold what
# it learnt better (-2 100 %, -1 82 % against 85 %, 48 %) and reach lesson 0 later
# (0 % at 1.5M against 88 %). ponytail: unmeasured on a GPU -- NET=64,64 is the other arm
MJX_NET = [256, 256]


# out/<name>/metrics.jsonl: what a run did, one JSON object a line, for audits and
# plots (`python -m rl.plot`). Appended across resumes; None outside a run.
METRICS = None


def record(kind, **fields):
    """one line of metrics.jsonl: a checkpoint, an unlock, a best, a rollback, a
    session's start or end"""
    if METRICS is not None:
        METRICS.write(json.dumps(dict(kind=kind, t=round(time.time(), 1), **fields)) + '\n')


class JsonLines(KVWriter):
    """SB3's per-update table as out/<name>/progress.jsonl, one line an update,
    appended across resumes (SB3's own csv and json writers start the file afresh
    on every start, and a resumed run would lose its history)"""

    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8', buffering=1)

    def write(self, key_values, key_excluded, step=0):
        row = {k: (v.item() if hasattr(v, 'item') else v) for k, v in key_values.items()
               if isinstance(v, (int, float, np.integer, np.floating))}
        self.f.write(json.dumps(dict(row, t=round(time.time(), 1))) + '\n')

    def close(self):
        self.f.close()


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


def _replace(src, dst):
    """os.replace, retried for a moment: on Windows it fails while another process
    (rl.status, reading state.json) has the target open"""
    for _ in range(50):
        try:
            return os.replace(src, dst)
        except PermissionError:
            time.sleep(0.1)
    return os.replace(src, dst)


def save_vecnorm(v, path):
    """VecNormalize.save without its per-env buffers (old_obs, returns: 4.7 MB at 4096
    envs, and VecNormalize.load rebuilds them for whatever env it is given)"""
    kept = {k: v.__dict__[k] for k in ('old_obs', 'old_reward', 'returns') if k in v.__dict__}
    v.__dict__.update({k: np.zeros(0) for k in kept})
    try:
        v.save(path)
    finally:
        v.__dict__.update(kept)


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

    def __init__(self, lessons, seed=0, visual=False, mirror=False):
        self.tasks = {k: ClimbTask(k, visual=visual, mirror=mirror) for k in lessons}
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

    def __init__(self, lessons, n, seed=0, bank=0):
        from . import mjx_task          # JAX only on this path
        self.mt = mjx_task
        self.bank = bank                # > 0: procedural walls (Walls), drawn from set_bank
        # fit: collide only what the walls in play need (rl.mjx_task.Batch._fit)
        self.batch = mjx_task.Batch(n, seed, bank=bank, fit=True)
        # substeps where the solver's contact cap was full, of all simulated; the
        # checkpoint line prints the share and zeroes it (rl.mjx_scene.MAX_CONTACTS)
        self.capped = self.substeps = 0
        dim = np.asarray(self.batch.reset()).shape[1]
        super().__init__(n, gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32),
                         gym.spaces.Box(-1.0, 1.0, (17,), np.float32))
        self.set_active(lessons)

    def set_active(self, active, weights=None):
        if not self.bank:               # on walls the pool's weights are in the bank itself
            self.batch.set_probs(dict(zip(active, weights if weights is not None
                                          else [1.0] * len(active))))
        return list(active)

    def set_bank(self, layouts):
        self.batch.set_bank(layouts)

    def reset(self):
        return np.asarray(self.batch.reset())

    def step_async(self, actions):
        if not np.isfinite(actions).all():
            # the policy has already gone; stop before MJX is fed NaN and before a
            # checkpoint can save it -- a restart resumes from the last whole one
            raise FloatingPointError("the policy's actions are not finite")
        self._actions = actions

    def step_wait(self):
        obs, end, r, term, trunc, info = self.batch.step(self._actions)
        obs, term, trunc = np.asarray(obs), np.asarray(term), np.asarray(trunc)
        done = term | trunc
        self.capped += int(info['capped'].sum())
        self.substeps += self.num_envs * self.mt.ct.FRAME_SKIP
        infos = [{} for _ in range(self.num_envs)]
        if done.any():
            end = np.asarray(end)
            why, lesson, steps, dist, fam, probe, pval = (
                np.asarray(info[k]) for k in ('reason', 'lesson', 'steps', 'dist', 'family',
                                              'probe', 'probe_val'))
            for i in np.nonzero(done)[0]:
                # a generated wall reports its family (= its benchmark lesson's key)
                infos[i] = {'reason': self.mt.REASONS[why[i]],
                            'lesson': int(fam[i]) if fam[i] >= 0 else self.mt.LESSON_KEYS[lesson[i]],
                            'probe': int(probe[i]), 'probe_val': float(pval[i]),
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


def episode_steps(timeout):
    """the most control steps an episode can last: it ends at the first past `timeout` s.
    1 % over, because MJX keeps time in float32 and it runs slow: 120 s ends at step
    3002, not 3001 (order_check replays it)"""
    return int(np.ceil(1.01 * timeout / CONTROL_DT)) + 1


class Curriculum(BaseCallback):
    """Unlock the next lesson once every lesson in the pool is being passed, and keep
    the old ones in the pool so the policy does not forget them.

    The books count episodes in the order they STARTED, each once every episode begun
    before it has ended (`lag`: the longest one can last). Counted as they end, the
    first in are the short ones, and a batch of envs makes that the whole sample: with
    256 envs the first 20 family-1 walls to end were ones a re-grip in place solves,
    20/20 at 5k steps, and family 2 unlocked on it -- over whole cohorts family 1 was
    at 20 % at 131k. Which of the episodes begun in one span end first says nothing
    about any of them, so a rate over all of them is the policy's however long each
    ran. The unlock, the draw weights, the score, the guard and ADR all read it.
    """
    window, unlock_min = WINDOW, UNLOCK_MIN
    stuck = None                         # STUCK_STEPS (Walls): see `stalled`
    needs = {}                           # lesson -> the lessons it builds on (Walls: rl.walls.NEEDS)

    def __init__(self, pool, active=None, history=None, enabled=True, lag=None,
                 cleared=None, entered=None):
        super().__init__()
        self.pool = list(pool)
        self.active = list(active or ([self.pool[0]] if enabled else self.pool))
        self.hist = {lesson_key(k): list(v) for k, v in (history or {}).items()}
        self._books(cleared, entered)
        self.lens = {}                   # episode lengths per lesson, for the draw weights
        self.ends = collections.Counter()   # (lesson, end reason) since the last checkpoint
        self.enabled = enabled
        # ended episodes waiting for every one begun before them: (start, env, n, episode)
        self.pending, self.n, self.clock, self.since = [], 0, 0, -np.inf   # since: a rollback
        self.lag = (episode_steps(max(LESSONS[k]['timeout'] for k in self.pool))
                    if lag is None else lag)

    def _books(self, cleared, entered):
        """which lessons have passed the gate at least once, and the step each entered
        the pool at. A checkpoint from before these were kept: every lesson but the
        newest had passed (the gate let the next one in), and the clock starts now"""
        self.cleared = ({lesson_key(k) for k in cleared} if cleared is not None
                        else set(self.active[:-1]))
        self.entered = {lesson_key(k): v for k, v in (entered or {}).items()}

    def rate(self, lesson):
        h = self.hist.get(lesson, [])[-self.window:]
        return (len(h), sum(h) / len(h) if h else 0.0)

    def stalled(self, lesson):
        """in the pool `stuck` steps and never once passed. It stays in the pool, with the
        steps its failures draw, and its upgrades (`needs`) stay locked until it passes;
        it only stops holding back lessons that do not build on it (Walls; STUCK_STEPS)"""
        if self.stuck is None or lesson in self.cleared:
            return False
        return self.num_timesteps - self.entered.setdefault(lesson, self.num_timesteps) >= self.stuck

    def guarded(self):
        """the lessons the guard holds a best checkpoint to: all of the pool"""
        return list(self.active)

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
        if n >= self.unlock_min and r >= WARM_RATE:
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
            v = [x for x in self.lens.get(lesson, [])[-self.window:] if x > 0]
            lens.append(sum(v) / len(v) if v else None)
        known = [x for x in lens if x is not None]
        fill = sum(known) / len(known) if known else 1.0
        w = [nd / (ln if ln is not None else fill) for nd, ln in zip(need, lens)]
        return [x / sum(w) for x in w]

    def score(self):
        """SUM of the success rates over the active lessons (Checkpoint.score says why)"""
        rates = [self.rate(k) for k in self.active]
        if any(n < self.unlock_min for n, _ in rates):
            return None
        return sum(r for _, r in rates)

    def held(self, lessons):
        """the same sum over `lessons` -- the best checkpoint's -- scored now"""
        rates = [self.rate(k) for k in lessons]
        if not rates or any(n < self.unlock_min for n, _ in rates):
            return None
        return sum(r for _, r in rates)

    def state(self):
        return {'active': self.active,
                'history': {str(k): v[-self.window:] for k, v in self.hist.items()},
                'cleared': sorted(self.cleared), 'entered': {str(k): v for k, v in self.entered.items()}}

    def restore(self, state):
        """back to a saved active set and history: the curriculum half of a rollback"""
        self.active = [k for k in map(lesson_key, state['active']) if k in self.pool] \
            or self.active
        self.hist = {lesson_key(k): list(v) for k, v in state['history'].items()}
        self._books(state.get('cleared'), state.get('entered'))
        self.pending.clear()
        self.since = self.clock          # episodes the dropped policy began do not count
        self._warm = self.warming()
        self.training_env.env_method('set_active', self.active, self.weights())

    def _push(self, books, lesson, x):
        """append, keeping only the tail `rate` and `weights` read: a GPU batch ends
        millions of episodes a day, and these lists used to keep every one"""
        h = books.setdefault(lesson, [])
        h.append(x)
        if len(h) > 2 * self.window:
            del h[:-self.window]

    def _count(self, ep):
        """one episode into the books `rate` and `weights` read, in start order"""
        self._push(self.hist, ep['lesson'], ep['reason'] == 'success')
        if ep['steps']:
            self._push(self.lens, ep['lesson'], ep['steps'])

    def _on_step(self):
        self.clock += 1
        for i, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            if done and 'reason' in info:
                self.ends[info['lesson'], info['reason']] += 1
                steps = info.get('steps') or 0
                if self.clock - steps >= self.since:
                    self.n += 1
                    heapq.heappush(self.pending, (self.clock - steps, i, self.n, {
                        k: info.get(k) for k in ('lesson', 'reason', 'steps', 'probe', 'probe_val')}))
        while self.pending and self.pending[0][0] <= self.clock - self.lag:
            self._count(heapq.heappop(self.pending)[3])
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
        # every lesson in the pool, not only the newest: see UNLOCK_AT
        rates = {k: self.rate(k) for k in self.active}
        ok = {k: m >= self.unlock_min and q >= UNLOCK_AT for k, (m, q) in rates.items()}
        self.cleared |= {k for k, v in ok.items() if v}
        locked = [k for k in self.pool if k not in self.active]
        if not locked:
            return True
        # the next lesson is the first one in the pool that is not yet active and whose
        # needs have all passed -- NOT `pool[len(active)]`: inserting a lesson into the
        # middle of the pool on a resume (1.85 between 1.7 and 2, say) leaves the restored
        # active set out of step with pool positions, and the index form then re-appends
        # a lesson that is already active while the new one is never trained at all
        nxt = next((k for k in locked if set(self.needs.get(k, ())) <= self.cleared), None)
        stalled = [k for k in self.active if not ok[k] and self.stalled(k)]
        if nxt is not None and all(ok[k] or k in stalled for k in self.active):
            self.active.append(nxt)
            self.hist.setdefault(self.active[-1], [])
            self.entered[self.active[-1]] = int(self.num_timesteps)
            self._warm = self.warming()
            self.training_env.env_method('set_active', self.active, self.weights())
            record('unlock', timesteps=int(self.num_timesteps), lesson=self.active[-1],
                   pool=self.active, stalled=stalled,
                   rates={str(k): self.rate(k) for k in self.active[:-1]})
            n, r = rates[self.active[-2]]
            why = (f"{stalled} not passed in {self.stuck:,} steps: still trained, its upgrades "
                   f"still locked; the rest at {UNLOCK_AT:.0%}+, and {nxt} needs only "
                   f"{list(self.needs.get(nxt, ()))}" if stalled else
                   f"every lesson at {UNLOCK_AT:.0%}+, the newest ({self.active[-2]}) at {r:.0%} "
                   f"over {n} episodes")
            print(f"\n[curriculum] {why} -> unlocking lesson {self.active[-1]} "
                  f"({LESSONS[self.active[-1]]['name']}); pool now {self.active}, "
                  f"the pool's steps go where the failures are\n")
        return True


class Walls(Curriculum):
    """The procedural-wall curriculum (rl.walls): what the GPU path trains on.

    Nine families of walls, one per benchmark lesson 1..9 and fixed before any training,
    enter in the benchmark's order by the parent class's own rule -- every family in
    the pool passed at UNLOCK_AT over its last WALL_WINDOW walls, and each family only
    once the families it is made of (rl.walls.NEEDS) have passed -- and share the steps
    by it (`weights`: where the failures are). A family that has not passed once in
    STUCK_STEPS stays in the pool and keeps its upgrades locked, and stops holding back
    the families that do not build on it. Inside a family
    every parameter's top moves by automatic domain randomization: half of the walls
    put one parameter exactly at its top (rl.walls.draw), and each WALL_WINDOW of those
    passed at UNLOCK_AT or better moves that top a step toward its last value, at
    WARM_RATE or worse a step back. A probe counts only at the top it was drawn at, so
    walls still out when a top moves do not vote on the new one. The score -- the best
    checkpoint -- is how far the tops have come, 0..1 a family; the guard adds what the
    families the best had passed still hold of the gate (`held`).

    enabled=False is the no-curriculum ablation: every family, every top at its last
    value from the first step -- plain domain randomization over the same walls.
    """
    window = unlock_min = WALL_WINDOW
    stuck = STUCK_STEPS

    def __init__(self, active=None, history=None, walls=None, enabled=True, seed=0, lag=None,
                 cleared=None, entered=None):
        from . import walls as W
        self.W = W
        self.needs = W.NEEDS
        super().__init__(list(W.FAMILIES), active, history, enabled,
                         episode_steps(W.TIMEOUT) if lag is None else lag, cleared, entered)
        self._load(walls)
        self.rng = np.random.default_rng(seed)
        self.checker = None

    def _load(self, walls):
        """a checkpoint's tops and probes, each family's only if its ranges are the ones
        they were grown over (rl.walls.spec; LEGACY's for a checkpoint that predates the
        record). A family whose ranges changed since starts over: first tops, no probes,
        no books -- gpu1's family 3 kept 0.30 m, its old first top, and the new 0.10 would
        have been skipped on the resume"""
        w = walls or {}
        was = w.get('spec') or {str(f): s for f, s in self.W.LEGACY.items()}
        moved = {f for f in self.W.FAMILIES if was.get(str(f), self.W.spec(f)) != self.W.spec(f)}
        self.tops = self.W.first_tops() if self.enabled else self.W.last_tops()
        for f, t in (w.get('tops') or {}).items():
            if int(f) not in moved:
                self.tops[int(f)].update(t)
        self.probes = {k: list(v) for k, v in (w.get('probes') or {}).items()
                       if int(k.split(':')[0]) not in moved}
        for f in sorted(moved):
            self.hist.pop(f, None)
            if w.get('tops'):
                print(f"[adr] family {f}'s ranges changed since this checkpoint: it starts over "
                      f"at {self.tops[f]}")

    def progress(self):
        """how far every top has come, first -> last, 0..1 a family, summed"""
        return float(sum(np.mean([(self.tops[f][n] - a) / (b - a) for n, _, a, b, _ in ps])
                         for f, ps in self.W.FAMILIES.items()))

    def score(self):
        """how far the tops have come, then -- among equals -- the pool's success on its
        walls: 0.01 for a whole pool passed, under the smallest step any top takes
        (0.038), so it only orders checkpoints ADR has taken equally far. Without it the
        best froze the moment every top reached its last, where a run spends most of
        its steps, and step 3 of gpu_run.sh scored that first policy to get there"""
        s = super().score()
        return self.progress() + 0.01 * (s or 0.0) / len(self.W.FAMILIES)

    def guarded(self):
        """the families the guard holds a best to: those that have passed the gate. A
        frontier family is still being learnt, and its rate says nothing about forgetting"""
        return [f for f in self.active if f in self.cleared]

    def held(self, lessons):
        """ADR's progress, plus how much of the gate each of `lessons` (the best's passed
        families) still holds: 1 at UNLOCK_AT or over, 0 at 0 %. Progress alone cannot see
        a family forgotten inside ADR's dead band -- probes passed 20-70 % move no top --
        so a family that fell from 95 % to 30 % and stayed there never brought the run
        back. Rates over UNLOCK_AT are capped: ADR raising a top lowers them by design.
        One family forgotten outright is a whole family's worth (GUARD_DROP), as losing
        its range is"""
        out = self.progress()
        for f in lessons:
            n, r = self.rate(f)
            out += min(r, UNLOCK_AT) / UNLOCK_AT if n >= self.unlock_min else 1.0
        return out

    def draw_bank(self, env):
        """a fresh bank of BANK walls into `env`: families by the pool's weights, their
        parameters under the tops"""
        if self.checker is None:
            self.checker = self.W.Checker()
        bank = self.W.draw(self.checker, dict(zip(self.active, self.weights())), self.tops,
                           self.rng, BANK)
        if self.enabled:                 # some of family 3's start partway up a climb (rl.rsi)
            bank = rsi.mix(bank, self.rng, self.tops[3]['spacing'])
        env.env_method('set_bank', bank)

    def _on_training_start(self):
        """every env onto a wall of the bank before the first step. SB3's learn() has
        just reset them on the lessons' rows -- all nine lessons and their mirrors --
        and those episodes would run for up to their timeouts, off the curriculum, and
        hold the batch's collision pairs at their widest (rl.mjx_task.Batch._fit)"""
        self.draw_bank(self.training_env)
        self.model._last_obs = self.training_env.reset()
        self.model._last_episode_starts = np.ones(self.training_env.num_envs, dtype=bool)

    def _on_rollout_start(self):
        self.draw_bank(self.training_env)

    def _count(self, ep):
        f, p = ep['lesson'], ep.get('probe')
        if p is not None and p >= 0 and self.enabled:     # the ablation's tops never move
            name = self.W.FAMILIES[f][p][0]
            if abs(ep['probe_val'] - self.tops[f][name]) <= 1e-6:   # not a top moved since
                h = self.probes.setdefault(f'{f}:{name}', [])
                h.append(ep['reason'] == 'success')
                if len(h) >= self.window:
                    rate = sum(h) / len(h)
                    h.clear()
                    self._move(f, name, rate)
        super()._count(ep)

    def _move(self, f, name, rate):
        _, _, first, last, step = next(s for s in self.W.FAMILIES[f] if s[0] == name)
        t = self.tops[f][name]
        new = (min(t + step, last) if rate >= UNLOCK_AT else
               max(t - step, first) if rate <= WARM_RATE else t)
        new = int(new) if isinstance(first, int) else round(float(new), 6)
        if new != t:
            self.tops[f][name] = new
            record('adr', timesteps=int(self.num_timesteps), family=f, param=name,
                   top=[t, new], rate=rate, progress=self.progress())
            print(f"[adr] family {f} {name}: top {t:g} -> {new:g} "
                  f"(walls at the top passed {rate:.0%}); progress {self.progress():.2f} of 9")

    def state(self):
        return dict(super().state(), walls=dict(
            tops={str(f): t for f, t in self.tops.items()}, probes=self.probes,
            spec={str(f): self.W.spec(f) for f in self.W.FAMILIES}))

    def restore(self, state):
        super().restore(state)
        self._load(state.get('walls'))


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
                 rollbacks=0, kl_target=None, backend='cpu', walls='lessons', snap_every=0,
                 best_held=None):
        super().__init__()
        self.dir, self.cur, self.every, self.backend = run_dir, curriculum, every, backend
        self.walls, self.snap_every, self.snapped = walls, snap_every, None
        self.last, self.best = 0, best if best is not None else -1.0
        self.kl_target = kl_target or KL_START
        # the lessons the best checkpoint was scored over; `held` scores them now
        self.best_active = [lesson_key(k) for k in (best_active or [])]
        # what `held` read at the best: the guard's bar. On the lesson ladder it is the
        # best's score itself; on generated walls the score is ADR's progress and the
        # guard reads the passed families' success as well (Walls.held)
        self.best_held = best_held if best_held is not None else self.best
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
        return self.cur.score()

    def held(self):
        """The best checkpoint's own lessons, scored now. Forgetting lowers it; a
        lesson unlocked since the best cannot, so a new lesson's first failures are
        never read as the pool coming apart."""
        return self.cur.held(self.best_active)

    def guard(self):
        """at each checkpoint: has the pool been a lesson below its best for long
        enough to go back? The rollback itself waits for the next rollout to start
        (`_on_rollout_start`), so no update ever mixes the two policies' data."""
        h = self.held()
        if h is None or h > self.best_held - GUARD_DROP:
            self.low_since = None
            return ""
        if self.low_since is None:
            self.low_since = self.model.num_timesteps
        low = self.model.num_timesteps - self.low_since
        if low >= self.patience():
            self.due = True
        return f" | forgetting: {h:.2f} vs best {self.best_held:.2f} for {low:,} steps"

    def patience(self):
        """GUARD_UPDATES in steps, at this run's rollout size"""
        return GUARD_UPDATES * self.model.n_steps * self.model.n_envs

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
        if isinstance(self.cur, Walls):     # this rollout's walls were drawn at the dropped tops
            self.cur.draw_bank(self.training_env)
        was, was_kl = m.lr_schedule(1.0), self.kl_target
        m.learning_rate = max(was * LR_DECAY, LR_MIN)
        m._setup_lr_schedule()              # lr_schedule is built from learning_rate
        # ...and the target the rate is adapted to, which is what actually sticks
        self.kl_target = max(self.kl_target * LR_DECAY, KL_MIN)
        self.rollbacks += 1
        self.low_since, self.due = None, False
        record('rollback', timesteps=int(m.num_timesteps), held=h, best=self.best_held,
               pool=self.cur.active, lr=[was, m.learning_rate], kl_target=[was_kl, self.kl_target],
               rollbacks=self.rollbacks)
        print(f"\n[guard] {h:.2f} against the best {self.best_held:.2f} for "
              f"{GUARD_UPDATES}+ updates: back to the best checkpoint (pool "
              f"{self.cur.active}), learning rate {was:g} -> {m.learning_rate:g}, "
              f"KL target {was_kl:g} -> {self.kl_target:g}, "
              f"rollback {self.rollbacks}\n")
        self.save()

    def save(self, tag=''):
        p = os.path.join(self.dir, tag or '')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # each file is written aside and renamed over the old one: a run killed in
        # the middle of a save (out of memory, a rented box stopped) must still have
        # a whole checkpoint to resume from, not a torn zip
        self.model.save(p + 'model.tmp.zip')
        save_vecnorm(self.model.get_vec_normalize_env(), p + 'vecnorm.tmp.pkl')
        state = {'timesteps': int(self.model.num_timesteps), 'robot': robot_stamp(),
                 'backend': self.backend, 'walls_mode': self.walls,
                 'axes': 'canonical',     # rung axes as rl.lessons.canon_axis turns them
                 'best': self.best, 'best_active': self.best_active, 'best_held': self.best_held,
                 'lr': self.model.lr_schedule(1.0), 'kl_target': self.kl_target,
                 'rollbacks': self.rollbacks, 'window': self.cur.window,
                 **self.cur.state()}
        with open(p + 'state.tmp.json', 'w') as f:
            json.dump(state, f, indent=1)
        for f in ('model.zip', 'vecnorm.pkl', 'state.json'):
            _replace(p + f.replace('.', '.tmp.'), p + f)
        return p

    def _on_rollout_end(self):
        # Keep the best policy, not just the newest. The 2.6M checkpoint that scored
        # 12/12 was overwritten by a later, worse one and had to be re-run from a
        # backup copy; a run that peaks and then drifts should not lose its peak.
        # Judged once a rollout, on the policy that collected it: it only changes at
        # an update, and 4096 envs move the score on almost every step.
        s = self.score()
        if s is not None and s > self.best:
            self.best, self.best_active, self.low_since = s, self.cur.guarded(), None
            self.best_held = self.held()
            self.save('best_')
            record('best', timesteps=int(self.model.num_timesteps), score=s,
                   pool=self.cur.active)
            print(f"[best] score {s:.4g} over {len(self.cur.active)} lesson(s) "
                  f"at {self.model.num_timesteps:,} steps")

    def _on_step(self):
        # A snapshot every snap_every steps, kept: the benchmark scored on each after the
        # run (python -m rl.curve) is the learning curve, and it is scored on the full
        # robot, which training never sees -- nothing is chosen by it
        n = self.model.num_timesteps
        if self.snap_every and n // self.snap_every != self.snapped:
            if self.snapped is not None:
                self.save(os.path.join('snaps', f'{n:012d}_'))
            self.snapped = n // self.snap_every
        if self.model.num_timesteps - self.last >= self.every:
            self.last = self.model.num_timesteps
            self.save()
            # '-': nothing counted yet (a cohort is counted once all of it has ended)
            rates = " ".join(f"L{k} " + (f"{r:.0%}" if n else "-")
                             for k, (n, r) in ((k, self.cur.rate(k)) for k in self.cur.active))
            # how the episodes since the last line ended: a 'blowup' count that is
            # not ~0 is the simulator, not the policy, and says so here first
            why, by_lesson = collections.Counter(), collections.defaultdict(dict)
            for (k, w), n in self.cur.ends.items():
                why[w] += n
                by_lesson[str(k)][w] = n
            self.cur.ends.clear()
            ends = ", ".join(f"{k} {n}" for k, n in sorted(why.items())) or "none"
            base, cap = self.training_env.unwrapped, None
            if isinstance(base, MjxVecEnv) and base.substeps:
                # not ~0 is MJX dropping live contacts: raise rl.mjx_scene.MAX_CONTACTS
                cap = base.capped / base.substeps
                ends += f" | contact cap full {cap:.3%} of substeps"
                base.capped = base.substeps = 0
            guard = self.guard()
            record('checkpoint', timesteps=int(self.model.num_timesteps),
                   updates=int(self.model.num_timesteps // (self.model.n_steps * self.model.n_envs)),
                   pool=self.cur.active,
                   rate={str(k): self.cur.rate(k) for k in self.cur.active},
                   length={str(k): float(np.mean(v[-self.cur.window:])) for k, v in self.cur.lens.items() if v},
                   ends=by_lesson, best=self.best, held=self.held(), best_held=self.best_held,
                   lr=self.model.lr_schedule(1.0), kl_target=self.kl_target,
                   rollbacks=self.rollbacks, contact_cap=cap,
                   std=float(self.model.policy.log_std.detach().exp().mean()),
                   tops=getattr(self.cur, 'tops', None))
            print(f"[checkpoint] {self.model.num_timesteps:,} steps | {rates} | "
                  f"best {self.best:.2f} | lr {self.model.lr_schedule(1.0):g} "
                  f"kl<={self.kl_target:g}"
                  f"{guard} | ends: {ends} | out/{os.path.basename(self.dir)}")
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
    c = Curriculum([1, 2, 3], lag=0)     # the rules; order_check tests the order
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
    # the books keep only their tail, and the rate is the same for it
    feed(c, 2, 'success', 5 * WINDOW, steps=100)
    assert len(c.hist[2]) <= 2 * WINDOW and len(c.lens[2]) <= 2 * WINDOW, \
        (len(c.hist[2]), len(c.lens[2]))
    assert c.rate(2) == (WINDOW, 1.0) and c.ends[2, 'success'] > 0, (c.rate(2), c.ends)

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
    g = Curriculum([1, 2, 3], lag=0)
    g.model = FakeModel(FakeEnv())
    feed(g, 1, 'success', WINDOW)
    feed(g, 1, 'fell', WINDOW // 2)                 # lesson 1 slips to 50 %
    feed(g, 2, 'success', WINDOW)
    assert g.active == [1, 2], f"unlocked while lesson 1 was being forgotten: {g.active}"
    feed(g, 1, 'success', WINDOW)
    assert g.active == [1, 2, 3], f"never unlocked once lesson 1 recovered: {g.active}"

    order_check(FakeModel, FakeEnv)
    guard_check(FakeModel, FakeEnv, feed)
    walls_check(FakeModel, FakeEnv)

    # The best-checkpoint score must never fall just because a lesson unlocked. Under
    # the old mean it did: 85% on lesson 1 alone scored 0.85, and unlocking lesson 2
    # at 0% dropped it to 0.42, so the 10.4M run kept a 308k-step policy as "best".
    class Cp:
        unlock_min = UNLOCK_MIN
        active = None
        def rate(self, k):
            return (WINDOW, {1: 0.85, 2: 0.0, 3: 0.0}[k])
    cp, sc = Cp(), Curriculum.score
    cp.active = [1]
    before = sc(cp)
    cp.active = [1, 2]
    after = sc(cp)
    print(f"score test: lesson 1 at 85% scores {before:.2f}; unlocking lesson 2 at 0% "
          f"scores {after:.2f}")
    assert after >= before, (f"unlocking a lesson dropped the score {before:.2f} -> "
                             f"{after:.2f}; best_model.zip will freeze at the old one")
    print(f"curriculum: unlocks at {UNLOCK_AT:.0%} over {UNLOCK_MIN}+ episodes on EVERY "
          f"lesson in the pool, keeps them all, and spends its STEPS where the failures "
          f"are -- frontier or regressed alike, {MIN_SHARE:.0%} floor for a mastered one. OK")


def order_check(FakeModel, FakeEnv):
    """A batch starts together; half its walls pass at step 10, half fall at 900. The
    fast half alone is 100 %, and counted as they ended it unlocked the next lesson;
    counted by start, nothing is in until the whole cohort is, and then it is 50 %."""
    c = Curriculum([1, 2], lag=1000)
    c.model = FakeModel(FakeEnv())
    n = 2 * UNLOCK_MIN
    for t in range(1, 1001):
        ends = ({i: 'success' for i in range(0, n, 2)} if t == 10 else
                {i: 'fell' for i in range(1, n, 2)} if t == 900 else {})
        c.locals = {'infos': [{'reason': ends[i], 'lesson': 1, 'steps': t} if i in ends else {}
                              for i in range(n)], 'dones': [i in ends for i in range(n)]}
        c._on_step()
        if t == 10:
            assert c.rate(1) == (0, 0.0) and c.active == [1], "counted a cohort's fast ends first"
    assert c.rate(1) == (n, 0.5) and c.active == [1], (c.rate(1), c.active)
    c.restore(c.state())                              # a rollback: what was begun before
    c.locals = {'infos': [{'reason': 'success', 'lesson': 1, 'steps': 5}], 'dones': [True]}
    c._on_step()                                      # it, by the dropped policy, is out
    assert not c.pending, "counted an episode begun before a rollback"
    from .climb_task import FRAME_SKIP
    from .walls import TIMEOUT
    for T in (TIMEOUT, max(v['timeout'] for v in LESSONS.values())):
        t, n = np.float32(0.0), 0                     # MJX's float32 clock, a substep at a time
        while t <= T:
            for _ in range(FRAME_SKIP):
                t = np.float32(t + np.float32(CONTROL_DT / FRAME_SKIP))
            n += 1
        assert episode_steps(T) >= n, f"a {T} s episode can last {n} steps, lag {episode_steps(T)}"
    k = max(LESSONS, key=lambda k: LESSONS[k]['timeout'])       # the longest lesson
    assert Walls().lag == episode_steps(TIMEOUT) and Curriculum([k]).lag == episode_steps(LESSONS[k]['timeout'])
    print(f"order test: a cohort's fast ends wait for its slow ones ({UNLOCK_MIN} passes "
          f"at step 10 did not unlock; the cohort is 50 %); a rollback drops what the old "
          f"policy began. OK")


def walls_check(FakeModel, FakeEnv):
    """ADR on the generated walls: a top moves only on a full WALL_WINDOW of probes drawn at
    it, up on a pass, down on a fail, never past its ends; the families enter in order;
    and the tops survive state.json"""
    from .walls import FAMILIES
    WW = WALL_WINDOW
    w = Walls(lag=0)
    w.model = FakeModel(FakeEnv())
    assert w.active == [1] and w.tops[1]['shift'] == FAMILIES[1][0][2], (w.active, w.tops[1])

    def probe(f, name, ok, n, val=None):
        i = [s[0] for s in FAMILIES[f]].index(name)
        v = w.tops[f][name] if val is None else val
        for _ in range(n):
            w.locals = {'infos': [dict(reason='success' if ok else 'fell', lesson=f, probe=i,
                                       probe_val=v, steps=100)], 'dones': [True]}
            w._on_step()
    top0, p0 = w.tops[1]['shift'], w.progress()
    probe(1, 'shift', True, WW - 1)
    assert w.tops[1]['shift'] == top0, "moved before a full window of probes"
    probe(1, 'shift', True, 1)
    assert abs(w.tops[1]['shift'] - (top0 + 0.1)) < 1e-9 and w.progress() > p0, w.tops[1]
    probe(1, 'shift', False, WW, val=0.123)          # drawn at a top long gone: no vote
    assert abs(w.tops[1]['shift'] - (top0 + 0.1)) < 1e-9, "counted probes of an old top"
    probe(1, 'shift', False, WW)
    assert abs(w.tops[1]['shift'] - top0) < 1e-9, "a failed window did not move the top back"
    probe(1, 'shift', False, WW)
    assert abs(w.tops[1]['shift'] - top0) < 1e-9, "went below the first top"
    for _ in range(40):
        probe(1, 'shift', True, WW)
    assert w.tops[1]['shift'] == FAMILIES[1][0][3], "went past the last top"
    assert w.active[:2] == [1, 2], f"family 2 never entered after family 1 was passed: {w.active}"
    st = json.loads(json.dumps(w.state()))
    r = Walls(st['active'], st['history'], st['walls'])
    assert r.tops == w.tops and r.active == w.active, "tops did not survive state.json"
    # ...but only against the ranges they were grown over: a family whose ranges changed
    # starts over -- tops, probes, books -- and a checkpoint from before the ranges were
    # recorded (gpu1 to 102.9M) is read against LEGACY's
    from .walls import LEGACY, first_tops
    moved = json.loads(json.dumps(st))
    moved['walls']['spec']['1'] = 'another range'
    r = Walls(moved['active'], moved['history'], moved['walls'])
    assert r.tops[1] == first_tops()[1] and 1 not in r.hist and r.tops[2] == w.tops[2] \
        and not any(k.startswith('1:') for k in r.probes), "a family whose ranges changed kept its tops"
    old = json.loads(json.dumps(st))
    del old['walls']['spec']
    old['walls']['tops']['3'] = {'rows': 1, 'spacing': 0.30}
    old['history']['3'] = [False] * WW
    r = Walls(old['active'], old['history'], old['walls'])
    assert r.tops[3] == first_tops()[3] and 3 not in r.hist and r.tops[1] == w.tops[1] \
        and all(r.tops[f] == w.tops[f] for f in FAMILIES if f not in LEGACY), "a legacy checkpoint's tops"
    r.model = FakeModel(FakeEnv())
    r.restore(old)                                       # a rollback onto a legacy best, too
    assert r.tops[3] == first_tops()[3] and 3 not in r.hist and r.tops[1] == w.tops[1], "a legacy rollback"
    step = min(st / (b - a) / len(ps) for ps in FAMILIES.values() for _, _, a, b, st in ps)
    assert 0.01 < step, f"a pool's success (0.01) can outrank a step of a top ({step:.3f})"
    off = Walls(enabled=False, lag=0)
    off.model = FakeModel(FakeEnv())
    assert off.active == list(FAMILIES) and off.tops[9][FAMILIES[9][0][0]] == FAMILIES[9][0][3]
    assert off.score() == off.progress() == len(FAMILIES), "a tie-break with nothing counted"
    for f in FAMILIES:                                   # every family passed: + 0.01
        off.locals = {'infos': [dict(reason='success', lesson=f, probe=-1, probe_val=0.0,
                                     steps=100)] * WW, 'dones': [True] * WW}
        off._on_step()
    assert abs(off.score() - (len(FAMILIES) + 0.01)) < 1e-9, off.score()
    tops = json.dumps(off.tops, sort_keys=True)
    off.locals = {'infos': [dict(reason='fell', lesson=1, probe=0, probe_val=off.tops[1]['shift'],
                                 steps=100)] * WW, 'dones': [True] * WW}
    off._on_step()
    assert json.dumps(off.tops, sort_keys=True) == tops, "the no-curriculum ablation moved a top"

    # The gate over WALL_WINDOW, asked after every episode as it is on the GPU: a family
    # truly at 60 % never lets the next one in (over 40 it did, 13 % a window); at 80 % it does
    g = Walls(lag=0)
    g.model = FakeModel(FakeEnv())
    rng = np.random.default_rng(0)

    def ep(f, ok, n=1):
        for _ in range(n):
            g.locals = {'infos': [dict(reason='success' if (ok() if callable(ok) else ok) else 'fell',
                                       lesson=f, probe=-1, probe_val=0.0, steps=100)], 'dones': [True]}
            g._on_step()
    ep(1, lambda: rng.random() < 0.60, 20 * WW)
    assert g.active == [1] and not g.cleared, f"a family at 60 % unlocked the next one: {g.active}"
    ep(1, lambda: rng.random() < 0.80, 3 * WW)
    assert g.active == [1, 2] and g.cleared == {1}, (g.active, g.cleared)
    # gpu1's jam, replayed: family 3 (climbing up) never passes. It is never dropped and
    # nothing that climbs up (5 over a break, 6 tiers) opens before it passes; after
    # STUCK_STEPS it stops holding back what does not need it -- 4 down, 7 the corner, 8
    # the wall behind, 9 the swing -- one at a time, each consolidated before the next
    ep(2, True, WW)
    assert g.active == [1, 2, 3], g.active
    ep(3, False, WW)
    g.num_timesteps = STUCK_STEPS - 1
    ep(3, False)
    assert g.active == [1, 2, 3], "let a family past a stuck one before STUCK_STEPS"
    g.num_timesteps = STUCK_STEPS
    ep(3, False)
    assert g.active == [1, 2, 3, 4] and 3 not in g.cleared, f"a stuck family blocks what does not need it: {g.active}"
    ep(1, False, WW)                                     # family 1, passed once, forgotten:
    ep(4, True, WW)                                      # consolidation holds the pool
    assert g.active == [1, 2, 3, 4], "unlocked while a family that had passed was slipping"
    ep(1, True, WW)
    assert g.active == [1, 2, 3, 4, 7], f"5 and 6 climb up and 3 has not passed: {g.active}"
    for f, nxt in ((7, 8), (8, 9)):
        ep(f, True, WW)
        assert g.active[-1] == nxt, (f, g.active)
    ep(9, True, WW)
    assert g.active == [1, 2, 3, 4, 7, 8, 9] and 3 in g.active, f"opened an upgrade of a stuck family: {g.active}"
    ep(3, True, WW)                                      # the stuck one passes after all:
    assert g.active[-1] == 5, g.active                   # its upgrades open, in order
    ep(5, True, WW)
    assert g.active[-1] == 6, g.active
    ep(3, False, WW)                                     # ...and once passed, it holds the pool
    ep(6, True, WW)                                      # like any other when it slips
    g.num_timesteps = 3 * STUCK_STEPS
    ep(6, True)
    assert 3 in g.cleared and not g.stalled(3), "a family that has passed was treated as stuck"
    # rl.rsi's episodes, started partway up a climb, are a book of their own: a window of
    # them all passed moves neither family 3's rate nor its tops
    rate3, tops3 = g.rate(3), dict(g.tops[3])
    g.locals = {'infos': [dict(reason='success', lesson=3 + rsi.BOOK, probe=-1, probe_val=0.0,
                               steps=100)] * WW, 'dones': [True] * WW}
    g._on_step()
    assert g.rate(3) == rate3 and g.tops[3] == tops3 and g.rate(3 + rsi.BOOK) == (WW, 1.0), \
        "episodes started partway up a climb counted toward family 3's gate or its ADR"
    if rsi.load() is not None:           # and the bank mixes them into family 3's walls only
        lays = [dict(family=f) for f in (1, 3) * 50]
        mixed = rsi.mix(lays, np.random.default_rng(0), top=0.50, share=1.0)
        assert all(m['family'] == (3 + rsi.BOOK if l['family'] == 3 else l['family'])
                   for l, m in zip(lays, mixed)) and min(m.get('d0_row', 99) for m in mixed) >= len(LESSONS)
        assert rsi.mix(lays, np.random.default_rng(0), top=0.10, share=1.0) == lays, \
            "a start was mixed into walls lower than its own climb"
    st = json.loads(json.dumps(g.state()))
    r = Walls(st['active'], st['history'], st['walls'], cleared=st['cleared'], entered=st['entered'])
    assert r.cleared == g.cleared and r.entered == g.entered, "the gate's books did not survive state.json"
    assert r.hist.get(3 + rsi.BOOK) == g.hist[3 + rsi.BOOK], "the starts' book did not survive state.json"
    # the guard reads the passed families' success as well as ADR's progress, capped at the
    # gate: at 80 % a family holds all of it, forgotten outright none
    h = Walls(lag=0)
    h.model = FakeModel(FakeEnv())
    h.hist[1] = [True] * WW
    full = h.held([1])
    h.hist[1] = [True] * int(0.8 * WW) + [False] * int(0.2 * WW)
    assert abs(h.held([1]) - full) < 1e-9, "a family above the gate was read as forgetting"
    h.hist[1] = [False] * WW
    assert abs(full - h.held([1]) - 1.0) < 1e-9 and full - h.progress() == 1.0, (full, h.held([1]))
    print(f"walls test: ADR moves a top by its step on a full window of {WW} probes at it "
          f"({UNLOCK_AT:.0%} up, {WARM_RATE:.0%} down), within its ends; families enter in "
          f"order; tops survive state.json, a family whose ranges changed starting over (a legacy "
          f"checkpoint read against rl.walls.LEGACY); the ablation starts every family at its last tops; the gate over {WW} walls holds a family at 60 %; a family stuck {STUCK_STEPS:,} steps stays in the pool, keeps its upgrades locked (rl.walls.NEEDS) and stops blocking the rest; the guard reads the passed families' success. OK")


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
        cur = Curriculum([1, 2], lag=0)
        cur.model = FakeModel(FakeEnv())
        feed(cur, 1, 'success', WINDOW)               # pool [1, 2]
        feed(cur, 2, 'success', UNLOCK_MIN)
        ck = Checkpoint(d, cur, every=1)
        ck.init_callback(m)
        sys.stdout = open(os.devnull, 'w')            # its [best]/[checkpoint] lines
        try:
            ck._on_rollout_end()                      # a best, over [1, 2]
            assert ck.best == ck.best_held == 2.0 and ck.best_active == [1, 2], (ck.best, ck.best_active)
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
            # the patience is in updates: GUARD_UPDATES of this toy's 16-step rollout,
            # and the same count of an MJX batch's 131k-step one
            P = ck.patience()
            assert P == GUARD_UPDATES * 16, P
            m.num_timesteps = 10 + P - 1
            ck._on_step()
            assert not ck.due, "tripped before GUARD_UPDATES"
            m.num_timesteps = 10 + P
            ck._on_step()
            assert ck.due, f"a lesson forgotten for {GUARD_UPDATES} updates did not trip it"
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

        # On generated walls the score is ADR's progress, and a passed family forgotten
        # inside ADR's dead band moves no top: the guard reads its success as well, and
        # one family forgotten outright, every top where it was, trips it
        wc = Walls(lag=0)
        wc.model = FakeModel(FakeEnv())
        ep = lambda f, ok, n: [setattr(wc, 'locals', {'infos': [dict(reason='success' if ok else 'fell',
                               lesson=f, probe=-1, probe_val=0.0, steps=100)], 'dones': [True]})
                               or wc._on_step() for _ in range(n)]
        ep(1, True, WALL_WINDOW)                      # family 1 passed, family 2 in
        ck2 = Checkpoint(os.path.join(d, 'w'), wc, every=1)
        ck2.init_callback(m)
        sys.stdout = open(os.devnull, 'w')
        try:
            ck2._on_rollout_end()                     # the best: progress 0, family 1 held
            assert ck2.best_active == [1] and abs(ck2.best_held - 1.0) < 1e-9, (ck2.best_active, ck2.best_held)
            ep(1, False, WALL_WINDOW)
            tops = json.dumps(wc.tops, sort_keys=True)
            m.num_timesteps = 100
            ck2._on_step()
            m.num_timesteps = 100 + ck2.patience()
            ck2._on_step()
        finally:
            sys.stdout.close()
            sys.stdout = out
        assert json.dumps(wc.tops, sort_keys=True) == tops and ck2.due, \
            "a passed family forgotten with ADR's tops untouched did not trip the guard"

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
    print(f"guard test: the best's lessons a whole lesson down for {GUARD_UPDATES} updates "
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
    # What an episode's wall is. `adr`, the GPU path's default: generated walls
    # (rl.walls), nine families entering in the benchmark's order, each parameter's
    # range grown by ADR; the benchmark lessons 1..9 are for scoring only. `random`: the
    # same walls at their last ranges from step 0, all families -- the no-curriculum
    # ablation. `lessons`: the fixed lesson scenes and their unlock ladder (--lessons,
    # --no-curriculum), the only one the CPU path can run.
    ap.add_argument('--walls', default=None, choices=['adr', 'random', 'lessons'],
                    help='default: adr on mjx, lessons on cpu')
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
                    help='hidden sizes for both nets, e.g. 256,256 (default: 256,256 on '
                         'mjx, 64,64 on cpu); fresh runs only')
    ap.add_argument('--snap-every', type=int, default=None,
                    help='steps between kept snapshots for the learning curve (rl.curve); '
                         'default 50M on mjx, 2M on cpu; 0 = none')
    ap.add_argument('--ckpt-every', type=int, default=None,
                    help='steps between checkpoints; default 100k on cpu, 1M on mjx')
    ap.add_argument('--seed', type=int, default=None,
                    help="a fresh run's policy init and its envs' draws; a paper reports "
                         "several (default: an unseeded init, env seeds 0..n-1)")
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
    # a crash below Python (XLA, CUDA, MuJoCo) still leaves its stack in the log
    faulthandler.enable(file=log, all_threads=True)
    global METRICS
    METRICS = open(os.path.join(run, 'metrics.jsonl'), 'a', encoding='utf-8', buffering=1)
    print(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")
    print(hardware.report())
    backend = hardware.backend()[0] if a.backend == 'auto' else a.backend
    mjx = backend == 'mjx'
    a.walls = a.walls or ('adr' if mjx else 'lessons')
    gen = a.walls != 'lessons'
    if gen and not mjx:
        raise SystemExit("generated walls (--walls adr/random) run on the MJX backend only")
    if gen:
        from .walls import FAMILIES
        pool = list(FAMILIES)
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
        if state.get('walls_mode', 'lessons') != a.walls:
            raise SystemExit(f"this run trained on --walls {state.get('walls_mode', 'lessons')}; "
                             f"it cannot carry on on --walls {a.walls}")
        was = state.get('backend', 'cpu')
        if was != backend:
            print(f"[backend] this run trained on {was} until now and carries on on "
                  f"{backend}: the same task, and mjx's simplified robot is checked "
                  f"against the full one by rl.mjx_check")

    # a resumed MJX batch draws from a key of its own, not the start's sequence again
    seed = a.seed or 0
    base = VecMonitor(MjxVecEnv(pool, envs, seed + state.get('timesteps', 0),
                                bank=BANK if gen else 0) if mjx else
                      SubprocVecEnv([(lambda i=i: MultiLesson(pool, seed=1000 * seed + i))
                                     for i in range(envs)]))
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
        # PPO.load re-seeds torch, numpy and the envs (the MJX batch's key) with the
        # bare --seed, so every restart -- the pilot's hand-over, a crash -- replayed
        # the start's draws; it draws on from the seed its batch was built with
        if a.seed is not None:
            model.set_random_seed(a.seed + model.num_timesteps)
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
        net = [int(x) for x in a.net.split(',')] if a.net else (MJX_NET if mjx else None)
        model = PPO('MlpPolicy', venv, device=a.device, verbose=1, **shape,
                    gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4 if a.lr is None else a.lr,
                    target_kl=a.target_kl,
                    policy_kwargs=None if net is None else dict(net_arch=net),
                    ent_coef=1e-3 if a.ent_coef is None else a.ent_coef, seed=a.seed)
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
    cur = (Walls(active, state.get('history'), state.get('walls'), a.walls == 'adr',
                 seed + state.get('timesteps', 0), cleared=state.get('cleared'),
                 entered=state.get('entered'))
           if gen else Curriculum(pool, active, state.get('history'), a.curriculum,
                                  cleared=state.get('cleared'), entered=state.get('entered')))
    venv.env_method('set_active', cur.active, cur.weights())
    if gen:
        cur.draw_bank(venv)              # the first episodes' walls, before learn's reset
    # which lessons the best was scored over: in the state since the guard exists,
    # and before that it is best_state.json's own active set
    best_active = state.get('best_active')
    bf = os.path.join(run, 'best_state.json')
    if best is not None and not best_active and os.path.exists(bf):
        best_active = json.load(open(bf))['active']
    ckpt = Checkpoint(run, cur, a.ckpt_every or (1_000_000 if mjx else 100_000), best,
                      best_active if best is not None else None,
                      state.get('rollbacks', 0), state.get('kl_target'), backend, a.walls,
                      a.snap_every if a.snap_every is not None else
                      (50_000_000 if mjx else 2_000_000),
                      best_held=state.get('best_held') if best is not None else None)

    # Ctrl-C during a rollout would otherwise lose everything since the last
    # checkpoint; SIGTERM likewise, so a killed run is resumable too.
    try:
        signal.signal(signal.SIGTERM,
                      lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    except (ValueError, AttributeError, OSError):
        pass          # Windows accepts only a short list of signals; Ctrl-C still works
    # the rollout table every update, on the console and train.log as before, and as
    # one line of progress.jsonl (rl.plot reads it)
    model.set_logger(Logger(None, [HumanOutputFormat(sys.stdout),
                                   JsonLines(os.path.join(run, 'progress.jsonl'))]))
    record('start', argv=sys.argv[1:], backend=backend, envs=envs, device=str(model.device),
           seed=a.seed, timesteps=int(model.num_timesteps), pool=cur.active,
           n_steps=model.n_steps, batch_size=model.batch_size, n_epochs=model.n_epochs,
           gamma=model.gamma, gae_lambda=model.gae_lambda, ent_coef=model.ent_coef,
           clip_range=float(model.clip_range(1.0)), lr=model.lr_schedule(1.0),
           kl_target=ckpt.kl_target, net_arch=str(model.policy.net_arch), robot=robot_stamp(),
           walls=a.walls, bank=BANK if gen else 0,
           tops=getattr(cur, 'tops', None))
    how = 'finished'
    try:
        # no progress_bar: it pulls in tqdm + rich, and verbose=1 already prints the
        # rollout table with elapsed time and total_timesteps every update
        model.learn(total_timesteps=a.steps,
                    callback=[cur, ckpt, AdaptiveLR(ckpt), BoundExploration()],
                    reset_num_timesteps=not a.resume)
    except KeyboardInterrupt:
        how = 'interrupted'
        print("\ninterrupted -- saving")
    except (BrokenPipeError, EOFError, ConnectionResetError) as e:
        how = f'worker ended ({type(e).__name__})'
        # A worker process ended. On Windows that is what Ctrl-C usually looks
        # like from here -- the console signal reaches the workers first and the
        # parent reads a closed pipe instead of KeyboardInterrupt -- and it is
        # also what an out-of-memory kill looks like. Either way the run is over
        # and the last checkpoint window must not go with it: win17 lost the
        # 55k steps since its last one this way, at 41.7M.
        print(f"\na worker process ended ({type(e).__name__}) -- saving")
    except BaseException as e:
        # not saved: whatever broke may be in the weights (the last checkpoint is whole)
        record('exit', how='crashed', error=repr(e)[:500], timesteps=int(model.num_timesteps))
        raise
    ckpt.save()
    record('exit', how=how, timesteps=int(model.num_timesteps))
    for k in pool:
        n, r = cur.rate(k)
        if n:
            print(f"  lesson {k} {LESSONS[k]['name']:>20}: {r:.0%} over last {n}")
    print(f"resume with: python -m rl.train_ppo --name {a.name} --resume")


if __name__ == '__main__':
    main()
