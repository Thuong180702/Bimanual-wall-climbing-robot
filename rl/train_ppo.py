"""Train ONE PPO policy across the lessons, with a curriculum and resume.

    python -m rl.train_ppo --name climber                 # all 13, curriculum on
    python -m rl.train_ppo --name climber --resume        # continue where it stopped
    python -m rl.train_ppo --name t1 --lessons 1,2        # a subset
    python -m rl.train_ppo --name t1 --no-curriculum      # all lessons from step 0

Ctrl-C saves and exits cleanly; `--resume` picks the run back up, timestep counter,
optimizer state, observation statistics and curriculum progress included.

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
import argparse, json, os, signal
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
except ImportError as e:
    raise SystemExit(f"missing training deps ({e}); pip install gymnasium stable-baselines3")

from . import default_envs
from .climb_task import ClimbTask
from .lessons import LESSONS, lesson_key

WINDOW = 40          # episodes per lesson the success rate is measured over
UNLOCK_AT = 0.60     # success rate on the newest lesson that unlocks the next one
UNLOCK_MIN = 20      # episodes needed before that rate is believed
MIN_SHARE = 0.10     # the floor on a mastered lesson's need, so it is not forgotten
WARM_RATE = 0.20     # the rate at which a new lesson counts as started, for the log
                     # A freshly unlocked lesson is, at first, a stream of failures,
                     # and drawn as often as the rest it is half the pool's gradient
                     # when there are two lessons. The 10.8M win8 run spent 8M steps
                     # on a lesson at 0 % and took the one before it from 95 % down
                     # to 32 %, and its action std from 0.67 to a floor of 0.04.




class MultiLesson(gym.Env):
    """Every lesson pre-loaded; each reset draws one from the active pool.

    All nine models cost 66 MB and 0.09 s to build in one process, so holding them
    all beats reloading a scene on every reset.
    """

    def __init__(self, lessons, seed=0):
        self.tasks = {k: ClimbTask(k) for k in lessons}
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


class Curriculum(BaseCallback):
    """Unlock the next lesson once the newest one is being passed, and keep the old
    ones in the pool so the policy does not forget them."""

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
        if self.active[-1] == self.pool[-1]:
            return True
        n, r = self.rate(self.active[-1])
        if n >= UNLOCK_MIN and r >= UNLOCK_AT:
            self.active.append(self.pool[len(self.active)])
            self.hist.setdefault(self.active[-1], [])
            self._warm = self.warming()
            self.training_env.env_method('set_active', self.active, self.weights())
            print(f"\n[curriculum] lesson {self.active[-2]} at {r:.0%} over {n} "
                  f"episodes -> unlocking lesson {self.active[-1]} "
                  f"({LESSONS[self.active[-1]]['name']}); pool now {self.active}, "
                  f"the pool's steps go where the failures are\n")
        return True


class Checkpoint(BaseCallback):
    """Model + observation statistics + curriculum state, written together.

    All three or none: a model restored against stale VecNormalize statistics sees a
    different observation scale than it was trained on, which reads as the policy
    having silently degraded.
    """

    def __init__(self, run_dir, curriculum, every=100_000, best=None):
        super().__init__()
        self.dir, self.cur, self.every = run_dir, curriculum, every
        self.last, self.best = 0, best if best is not None else -1.0

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

    def save(self, tag=''):
        os.makedirs(self.dir, exist_ok=True)
        p = os.path.join(self.dir, tag or '')
        self.model.save(p + 'model')
        self.model.get_vec_normalize_env().save(p + 'vecnorm.pkl')
        state = {'timesteps': int(self.model.num_timesteps),
                 'best': self.best, **self.cur.state()}
        json.dump(state, open(p + 'state.json', 'w'), indent=1)
        return p

    def _on_step(self):
        # Keep the best policy, not just the newest. The 2.6M checkpoint that scored
        # 12/12 was overwritten by a later, worse one and had to be re-run from a
        # backup copy; a run that peaks and then drifts should not lose its peak.
        s = self.score()
        if s is not None and s > self.best:
            self.best = s
            self.save('best_')
            print(f"[best] score {s:.2f} over {len(self.cur.active)} lesson(s) "
                  f"at {self.model.num_timesteps:,} steps")
        if self.model.num_timesteps - self.last >= self.every:
            self.last = self.model.num_timesteps
            self.save()
            rates = " ".join(f"L{k} {self.cur.rate(k)[1]:.0%}" for k in self.cur.active)
            print(f"[checkpoint] {self.model.num_timesteps:,} steps | {rates} | "
                  f"best {self.best:.2f} | out/{os.path.basename(self.dir)}")
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

    off = Curriculum([1, 2, 3], enabled=False)
    assert off.active == [1, 2, 3], "--no-curriculum must start with the whole pool"

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
    print(f"curriculum: unlocks at {UNLOCK_AT:.0%} over {UNLOCK_MIN}+ episodes, keeps "
          f"every lesson in the pool, and spends its STEPS where the failures are -- "
          f"frontier or regressed alike, {MIN_SHARE:.0%} floor for a mastered one. OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='climber', help='run directory under out/')
    ap.add_argument('--lessons', default=','.join(str(k) for k in sorted(LESSONS)))
    ap.add_argument('--steps', type=int, default=2_000_000,
                    help='steps to run; with --resume, ADDITIONAL steps')
    # Default to the machine's core count rather than a number tuned here -- on a
    # 64-core Linux box, 12 would leave most of it idle. Measured end-to-end fps on
    # lesson 1 on the 12-core M4 Pro: 6->3413, 8->3950, 12->4330, 20->4977,
    # 32->5259. It keeps creeping up past the core count because the envs sit idle
    # during each policy update, so on a big machine try more than this default.
    # Each env holds all 13 scenes, ~94 MB, so budget RAM at --envs x 94 MB.
    ap.add_argument('--envs', type=int, default=default_envs())
    ap.add_argument('--device', default='cpu',
                    choices=['cpu', 'cuda', 'mps', 'auto'],
                    help='policy device. cpu is the default because an MlpPolicy is '
                         'usually too small to beat it; auto = cuda if present. '
                         'Run `python -m rl.bench` on the machine before deciding')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                    help='put every lesson in the pool from step 0')
    # 3e-3 was too high: after 4M steps the action std had grown to 1.52, above the
    # 1.0 it starts at, so the entropy bonus was beating the policy gradient and the
    # mean action stayed useless -- deterministic eval fell 12/12 while sampling from
    # the same policy succeeded 8/12. Lower it and the distribution can tighten.
    ap.add_argument('--ent-coef', type=float, default=1e-3)
    ap.add_argument('--ckpt-every', type=int, default=100_000)
    ap.add_argument('--self-check', action='store_true',
                    help='test the curriculum logic, train nothing')
    a = ap.parse_args()
    if a.self_check:
        return self_check()

    pool = [lesson_key(x) for x in a.lessons.split(',')]
    run = os.path.join('out', a.name)
    state = {}
    if a.resume:
        f = os.path.join(run, 'state.json')
        if not os.path.exists(f):
            raise SystemExit(f"nothing to resume: {f} does not exist")
        state = json.load(open(f))

    base = VecMonitor(SubprocVecEnv(
        [(lambda i=i: MultiLesson(pool, seed=i)) for i in range(a.envs)]))
    if a.resume:
        venv = VecNormalize.load(os.path.join(run, 'vecnorm.pkl'), base)
        venv.training = True
        model = PPO.load(os.path.join(run, 'model'), env=venv, device=a.device)
        print(f"resumed {run} at {model.num_timesteps} steps, "
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
        # without a 1M run that reaches lesson 1: that is the cheap canary.
        model = PPO('MlpPolicy', venv, device=a.device, verbose=1,
                    n_steps=256, batch_size=512, gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4, ent_coef=a.ent_coef)

    # A resume may narrow the pool -- dropping a mastered lesson is the point of
    # `--lessons` on a resume -- so the restored active set has to be filtered to it
    # rather than carried over wholesale.
    active = [k for k in (state.get('active') or []) if k in pool] or None
    cur = Curriculum(pool, active, state.get('history'), a.curriculum)
    venv.env_method('set_active', cur.active, cur.weights())
    ckpt = Checkpoint(run, cur, a.ckpt_every, state.get('best'))

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
        model.learn(total_timesteps=a.steps, callback=[cur, ckpt],
                    reset_num_timesteps=not a.resume)
    except KeyboardInterrupt:
        print("\ninterrupted -- saving")
    ckpt.save()
    for k in pool:
        n, r = cur.rate(k)
        if n:
            print(f"  lesson {k} {LESSONS[k]['name']:>20}: {r:.0%} over last {n}")
    print(f"resume with: python -m rl.train_ppo --name {a.name} --resume")


if __name__ == '__main__':
    main()
