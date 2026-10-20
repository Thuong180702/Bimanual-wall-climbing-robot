"""Train ONE goal-conditioned PPO policy over every family of walls, on MJX.

    python -m rl.train_ppo --name gpu1                  # a fresh run
    python -m rl.train_ppo --name gpu1 --resume         # carry on where it stopped
    python -m rl.train_ppo --name abl --walls random    # the no-curriculum ablation
    python -m rl.train_ppo --self-check                 # the curriculum, guard and cloning, tested

Ctrl-C or SIGTERM saves and exits; `--resume` picks the run back up -- timestep counter,
optimizer, observation statistics and curriculum included.

## What a run does by itself

  * The CURRICULUM (`Curriculum`). The walls are rl.walls' nine families, one per benchmark
    lesson. A family enters the pool once every family in it passes its gate (UNLOCK_AT
    over its last WINDOW walls from the hang) and the families it is made of have passed
    (rl.walls.NEEDS); nothing ever leaves. Inside a family ADR raises each parameter's
    level once its candidate's probes pass the gate (rl.walls). The pool's steps go where
    its failures are (`weights`). A family that has not passed yet and has not progressed in
    STUCK_STEPS -- its rate from the hang, or the share of its reference starts it finishes,
    no better than its best -- stops holding back the families that do not build on it.
  * REFERENCE CLIMBS (rl.rsi). A share of the walls of each family that has climbs begin
    partway up one, toward the frontier of what the policy finishes; where it does not,
    its mean is pulled toward the climb's own action (`Clone`).
  * The UPDATE SIZE (`UpdateSize`). Each action dimension's std is held in [STD_MIN, STD_MAX],
    and the learning rate moves to keep every update's KL near the run's target (rsl_rl's
    rule). Without them the std of the dimensions a policy uses precisely shrank without
    limit, and since a Gaussian's KL goes as 1/std^2 every update grew with it until the
    policy was rewritten wholesale each time.
  * The GUARD (`Checkpoint`). The best checkpoint is kept, scored by how far ADR has come.
    If the families the best had passed fall a whole family below what they held there
    for GUARD_UPDATES updates, the run goes back to it -- policy, optimizer, statistics and
    curriculum -- and halves its KL target.

## Why one policy and not nine

Nine specialists would each learn from scratch what all of them share -- hanging without
falling, transferring weight onto a hand, seating a hook in a +-25 mm window -- and leave
the seams between them untrained; a meta-policy over nine sub-policies would learn what a
search already does (which hold next is A* over the wall map). One policy over all of
them shares the common skill and needs no dispatcher, because the goal is in the
observation: it is goal-conditioned, not lesson-conditioned, and at deploy time a planner
supplies the goal as the walls do here. Its failure mode is forgetting, so the families are
MIXED, every reset drawing from the pool, never trained one after another.

## What it prints

One line every LINE_EVERY steps -- steps, wall-clock, speed, each lesson's success from the
hang, ADR's progress and the best -- and a line for each event: a lesson entering, a level
raised, a rollback. Everything else goes to files in out/<name>/ (rl.plot draws them):
train.log (the console), progress.jsonl (PPO's numbers, one line an update) and
metrics.jsonl (one line a checkpoint, and one an event).
"""
import argparse, collections, faulthandler, hashlib, heapq, json, os, pickle, signal, sys, time
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import KVWriter, Logger
    from stable_baselines3.common.vec_env import VecEnv, VecMonitor, VecNormalize
except ImportError as e:
    raise SystemExit(f"missing training deps ({e}); pip install -r requirements-gpu.txt")

from . import SCENE, hardware, rsi, walls as W
from .climb_task import CONTROL_DT
from .lessons import LESSONS

UNLOCK_AT = 0.70     # a family's gate: this success rate over its last WINDOW walls from the hang,
                     # and an ADR candidate's over WINDOW probes at it. Every family in the pool
                     # needs it before the next enters: consolidate, then add
WINDOW = 200         # walls a rate is read over. 4096 envs end hundreds of a family's episodes
                     # between two updates and the gate is asked after every one: over 40, a
                     # family truly at 60 % reads 70 % 13 % of the time; over 200, 0.2 %
MIN_SHARE = 0.10     # the floor on a passed family's need, so it is not forgotten
WARM_RATE = 0.20     # a stuck family under this, with no climbs to learn from, draws MIN_SHARE
STUCK_STEPS = 50_000_000   # a family not yet passed that has not progressed in this many steps stops
                     # holding the next one back; it stays in the pool, its upgrades locked. Counted
                     # from its entry, gpu12 let 7, 8 and 9 in while 6 was still learning, and then
                     # 6, 8 and 9 shared ~70 % of the steps at ~25 % each (gpu11's one stuck family
                     # had 60-90 %): a family still progressing keeps its steps
PROGRESS_RATE = 0.05 # progress: the rate from the hang this much over its best...
PROGRESS_STARTS = 0.02    # ...or this share more of the family's reference starts passed
PROBE_MIN = 20       # probes at a candidate before its rate counts in the family's share of the steps
GUARD_DROP = 1.0     # the guard: the best's passed families a whole family below what they held...
GUARD_UPDATES = 325  # ...for this many policy updates (a new family's first dip recovers sooner)
LR_DECAY = 0.5       # each rollback halves the learning rate and the KL target
LR_MIN, LR_MAX = 1e-5, 3e-4
KL_START, KL_MIN = 0.1, 0.01   # the KL target: where it starts, and the least a rollback leaves it
STD_MIN = 0.1        # floor on every action dimension's std: 0.006 rad of setpoint noise a step. At 0.05
                     # gpu12's shoulders sat on it from 200M, and a policy that places its hands with them
                     # was locked: moving their mean 0.5 is 10 sigma, a KL no update is let take. Pulled
                     # toward gpu11's actions (it climbs families 6 and 8), its mean stayed 30 sigma off
                     # for 20 updates at 0.05 and came 18 -> 13 at 0.1, the KL as small; families 1-5
                     # and 7 from the hang held at 0.1 (C, 35 walls each)
STD_MAX = 0.3        # ...and its ceiling. At 1 (SB3's start) gpu12's wrists and elbows drifted up to it
                     # from 200M on (0.4 -> 1.0) while the shoulders sat at the floor, and every catch
                     # paid: at its 315M checkpoint lesson 8's release, from the starts it finishes
                     # deterministically 40/40, went 15/40 sampled, 25/40 under 0.5, 35/40 under 0.3;
                     # from the hang under 0.3, lessons 3 and 4 went 23 -> 29 and 22 -> 31 of 35
BANK = 1024          # walls drawn at the start of every rollout; each env takes one at every reset
ENVS = 4096          # envs in one GPU batch (gpu_run.sh halves it until it fits the card)
PPO_SHAPE = dict(n_steps=32, minibatches=8, n_epochs=10)   # rsl_rl's rollout, 80 Adam steps an update
NET = [256, 256]     # policy and value nets; nine families through SB3's 64x64 kept forgetting
LINE_EVERY = 5_000_000     # steps between the console's status lines

CLONE_RATIO = 0.3    # the cloning gradient's norm, this times PPO's own on the actor...
CLONE_OFF = 2.5      # ...and times min(1, CLONE_OFF / off), `off` the minibatch's mean |mu - a| / sigma:
                     # far off it pulls gently (a clone 12.8 sigma off moved one update's KL to 4.8)
CLONE_BATCH = 512    # reference rows a PPO minibatch step clones
CLONE_DRAWS = 16     # sensor-noise draws observed at each reference row
CLONE_CHUNK = 1024   # resets at a time while observing them

METRICS = None       # out/<name>/metrics.jsonl while a run is on


def record(kind, **fields):
    """one line of metrics.jsonl: a checkpoint or an event"""
    if METRICS is not None:
        METRICS.write(json.dumps(dict(kind=kind, t=round(time.time(), 1), **fields)) + '\n')


def say(steps, msg):
    """an event on the console (a run's: the self-check's fakes print none)"""
    if METRICS is not None:
        print(f"{steps / 1e6:>8.1f}M  {msg}", flush=True)


class JsonLines(KVWriter):
    """SB3's per-update numbers as out/<name>/progress.jsonl, one line an update, appended
    across resumes"""

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
    """which robot a checkpoint learned: a hash of urdf/scene.xml, line endings normalised"""
    return hashlib.sha256(open(SCENE, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:16]


def check_robot(state):
    """refuse to resume a policy onto a robot it never trained on: the observation and action
    are the same shape on another robot, so such a resume loads without a word"""
    if state.get('robot') != robot_stamp():
        raise SystemExit(f"this checkpoint was trained on another robot (stamp {state.get('robot')}, "
                         f"urdf/scene.xml is now {robot_stamp()}): start a new run with a new --name")


def save_vecnorm(v, path):
    """VecNormalize.save without its per-env buffers (4.7 MB at 4096 envs, rebuilt on load)"""
    kept = {k: v.__dict__[k] for k in ('old_obs', 'old_reward', 'returns') if k in v.__dict__}
    v.__dict__.update({k: np.zeros(0) for k in kept})
    try:
        v.save(path)
    finally:
        v.__dict__.update(kept)


def episode_steps(timeout):
    """the most control steps an episode can last: it ends at the first past `timeout` s.
    1 % over, because MJX keeps time in float32 and it runs slow: 120 s ends at step 3002"""
    return int(np.ceil(1.01 * timeout / CONTROL_DT)) + 1


class MjxVecEnv(VecEnv):
    """one `rl.mjx_task.Batch` as SB3's VecEnv, its walls from a bank (`set_bank`). An
    episode's end reports its reason, family, probe, start, steps and distance"""

    def __init__(self, n, seed=0, impl='jax'):
        from . import mjx_task
        self.mt = mjx_task
        # fit: collide only what the walls in play need; impl: MJX's JAX or Warp backend
        self.batch = mjx_task.Batch(n, seed, bank=BANK, fit=True, impl=impl)
        self.capped = self.substeps = 0  # substeps with the solver's contact cap full, of all
        dim = np.asarray(self.batch.reset()).shape[1]
        super().__init__(n, gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32),
                         gym.spaces.Box(-1.0, 1.0, (17,), np.float32))

    def set_bank(self, layouts):
        self.batch.set_bank(layouts)

    def reset(self):
        return np.asarray(self.batch.reset())

    def step_async(self, actions):
        if not np.isfinite(actions).all():
            # the policy has already gone; stop before MJX is fed NaN and before a checkpoint
            # can save it -- a restart resumes from the last whole one
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
            why, steps, dist, fam, probe, pval, row = (
                np.asarray(info[k]) for k in ('reason', 'steps', 'dist', 'family', 'probe', 'probe_val', 'row'))
            for i in np.nonzero(done)[0]:
                infos[i] = {'reason': self.mt.REASONS[why[i]], 'lesson': int(fam[i]),
                            'probe': int(probe[i]), 'probe_val': float(pval[i]),
                            'start': int(row[i]) - len(self.mt.LESSON_KEYS),   # rl.rsi's row, < 0 off them
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
    """Which families are in the pool, their ADR levels, and every wall of the next rollout.

    The books count episodes in the order they STARTED, each once every episode begun
    before it can have ended (`lag`: the longest one can last). Counted as they end, the
    first in are the short ones, and a batch of envs makes that the whole sample: with 256
    envs the first 20 family-1 walls to end were ones a re-grip in place solves, 20/20, and
    the next family entered on it at 20 % over whole cohorts. Three kinds of book:

      gate     each family's walls from the hang, every parameter at or under its level:
               the gate, the step shares and the guard read it
      probes   each (family, parameter)'s walls at its candidate: ADR's vote
      starts   each rl.rsi start's episodes: its climb's frontier

    enabled=False is the no-curriculum ablation: every family, every level at its last, from
    the first step, and no reference climbs -- plain domain randomization over the same walls.
    """

    def __init__(self, state=None, enabled=True, seed=0):
        super().__init__()
        self.enabled = enabled
        self.ref = rsi.load()
        self.taught = set() if self.ref is None else {int(f) for f in self.ref['climb_family']}
        self.lag = episode_steps(W.TIMEOUT)
        self.lag_start = {f: episode_steps(rsi.timeout(self.ref, f)) for f in self.taught}
        self.rng = np.random.default_rng(seed)
        self.checker = None
        self.pending, self.n, self.clock = [], 0, 0
        self.ends = collections.Counter()        # (family, end reason) since the last checkpoint
        self.load(state or {})

    # ---- books ------------------------------------------------------------------
    def load(self, state):
        """the curriculum of a state.json (a resume, a rollback). A family whose ranges
        changed since starts over at its first levels, its books empty"""
        c = state.get('curriculum') or {}
        moved = {f for f in W.FAMILIES if c.get('spec', {}).get(str(f), W.spec(f)) != W.spec(f)}
        keep = lambda k: int(str(k).split(':')[0]) % rsi.BOOK not in moved
        self.active = [f for f in c.get('active', []) if f in W.FAMILIES] or \
            ([W.ORDER[0]] if self.enabled else list(W.FAMILIES))
        self.levels = W.first_levels() if self.enabled else W.last_levels()
        for f, lv in c.get('levels', {}).items():
            if keep(f):
                self.levels[int(f)].update(lv)
        self.gate = {int(f): list(v) for f, v in c.get('gate', {}).items() if keep(f)}
        self.lens = {int(f): list(v) for f, v in c.get('lens', {}).items() if keep(f)}
        self.probes = {k: list(v) for k, v in c.get('probes', {}).items() if keep(k)}
        same = self.ref is not None and c.get('rsi') == rsi.signature(self.ref)
        self.starts = {int(j): list(v) for j, v in c.get('starts', {}).items()} if same else {}
        self.cleared = {f for f in c.get('cleared', []) if keep(f)}
        self.entered = {int(f): v for f, v in c.get('entered', {}).items()}
        self.moved = {int(f): v for f, v in c.get('moved', {}).items()}        # its last progress
        self.best_prog = {int(f): list(v) for f, v in c.get('best_prog', {}).items()}
        self.pending.clear()
        self.since = self.clock          # episodes a dropped policy began do not count

    def state(self):
        tail = lambda b, n: {str(k): v[-n:] for k, v in b.items()}
        return {'curriculum': dict(
            active=self.active, levels={str(f): v for f, v in self.levels.items()},
            gate=tail(self.gate, WINDOW), lens=tail(self.lens, WINDOW), probes=self.probes,
            starts=tail(self.starts, rsi.START_WINDOW), cleared=sorted(self.cleared),
            entered={str(f): v for f, v in self.entered.items()},
            moved={str(f): v for f, v in self.moved.items()},
            best_prog={str(f): v for f, v in self.best_prog.items()},
            spec={str(f): W.spec(f) for f in W.FAMILIES},
            rsi=None if self.ref is None else rsi.signature(self.ref))}

    @staticmethod
    def _push(books, key, x, keep):
        """append, keeping only the tail a rate reads"""
        h = books.setdefault(key, [])
        h.append(x)
        if len(h) > 2 * keep:
            del h[:-keep]

    def rate(self, f):
        """(episodes, success) over family f's last WINDOW walls from the hang"""
        h = self.gate.get(f, [])[-WINDOW:]
        return len(h), (sum(h) / len(h) if h else 0.0)

    def _on_step(self):
        self.clock += 1
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if not done or 'reason' not in info:
                continue
            f = info['lesson']
            self.ends[f, info['reason']] += 1
            t0 = self.clock - (info.get('steps') or 0)
            if t0 < self.since:
                continue
            lag = self.lag_start.get(f - rsi.BOOK, self.lag) if f >= rsi.BOOK else self.lag
            self.n += 1
            heapq.heappush(self.pending, (t0 + lag, self.n, {k: info.get(k) for k in (
                'lesson', 'reason', 'steps', 'probe', 'probe_val', 'start')}))
        while self.pending and self.pending[0][0] <= self.clock:
            self._count(heapq.heappop(self.pending)[2])
        if self.enabled:
            self._unlock()
        return True

    def _count(self, ep):
        f, ok = ep['lesson'], ep['reason'] == 'success'
        if ep['steps']:
            self._push(self.lens, f, ep['steps'], WINDOW)
        if f >= rsi.BOOK:                                  # from one of rl.rsi's starts
            if ep.get('start', -1) >= 0:
                self._push(self.starts, int(self.ref['start'][ep['start']]), ok, rsi.START_WINDOW)
        elif ep.get('probe') is not None and ep['probe'] >= 0:     # at a candidate
            if self.enabled:
                self._vote(f, W.FAMILIES[f][ep['probe']][0], ep['probe_val'], ok)
        else:
            self._push(self.gate, f, ok, WINDOW)

    def _vote(self, f, name, val, ok):
        """a probe at family f's candidate for `name`: a window of them passed raises its level"""
        cand = W.candidates(f, self.levels[f]).get(name)
        if cand is None or abs(val - cand) > 1e-6:         # drawn at a candidate passed since
            return
        h = self.probes.setdefault(f'{f}:{name}', [])
        h.append(ok)
        if len(h) < WINDOW:
            return
        r = sum(h) / len(h)
        h.clear()
        if r < UNLOCK_AT:
            return
        was = self.levels[f][name]
        self.levels[f][name] = cand
        for k in self.probes:      # drawn with the old level: the family's other votes start over
            if k.split(':')[0] == str(f):
                self.probes[k] = []
        record('adr', timesteps=int(self.num_timesteps), family=f, param=name, level=[was, cand],
               rate=r, progress=W.progress(self.levels))
        say(self.num_timesteps, f"lesson {f} {name} {was:g} -> {cand:g}")

    # ---- the pool ---------------------------------------------------------------
    def progress(self, f):
        """(its rate from the hang, the share of its reference starts it finishes)"""
        n, r = self.rate(f)
        if self.ref is None or f not in self.taught:
            return r, 0.0
        ok = rsi.allowed(self.ref, {f: self.levels[f]})
        ok = ok[self.ref['climb_family'][self.ref['start_climb'][ok]] == f]
        return r, float(np.mean([rsi.passed(self.starts.get(int(j), [])) for j in ok])) if len(ok) else 0.0

    def track(self):
        """each family not yet passed: when it last bettered its best progress"""
        for f in self.active:
            if f in self.cleared:
                continue
            r, s = self.progress(f)
            best = self.best_prog.setdefault(f, [r, s])
            if r >= best[0] + PROGRESS_RATE or s >= best[1] + PROGRESS_STARTS:
                self.moved[f] = int(self.num_timesteps)
            self.best_prog[f] = [max(best[0], r), max(best[1], s)]

    def stalled(self, f):
        """not passed yet, and no progress in STUCK_STEPS"""
        since = self.moved.get(f, self.entered.setdefault(f, self.num_timesteps))
        return f not in self.cleared and self.num_timesteps - since >= STUCK_STEPS

    def _unlock(self):
        rates = {f: self.rate(f) for f in self.active}
        ok = {f: n >= WINDOW and r >= UNLOCK_AT for f, (n, r) in rates.items()}
        self.cleared |= {f for f, v in ok.items() if v}
        nxt = next((f for f in W.ORDER if f not in self.active and set(W.NEEDS[f]) <= self.cleared), None)
        if nxt is None or not all(ok[f] or self.stalled(f) for f in self.active):
            return
        self.active.append(nxt)
        self.entered[nxt] = int(self.num_timesteps)
        record('unlock', timesteps=int(self.num_timesteps), lesson=nxt, pool=self.active,
               stalled=[f for f in self.active if not ok.get(f, True)],
               rates={str(f): rates[f] for f in rates})
        say(self.num_timesteps, f"lesson {nxt} ({LESSONS[nxt]['name']}) enters training")

    def share(self, f):
        """of family f's walls begun at one of its rl.rsi starts"""
        if not self.enabled or f not in self.taught:
            return 0.0
        if f in self.cleared:
            return rsi.SHARE[1]
        return rsi.SHARE_SLOW if f in rsi.SLOW else rsi.SHARE[0]

    def length(self, f):
        """mean steps of a wall of family f as the bank draws it: its share begun at a start
        runs as long as those, the rest as from the hang. None before any"""
        mean = lambda k: (lambda v: sum(v) / len(v) if v else None)(self.lens.get(k, [])[-WINDOW:])
        own, part, s = mean(f), mean(f + rsi.BOOK), self.share(f)
        return own if own is None or part is None else (1.0 - s) * own + s * part

    def fails(self, f):
        """how much family f fails over the walls it is drawn: its rate from the hang under its
        levels, and PROBE_P of its walls at a candidate (each read once its window has
        PROBE_MIN). By the levels alone a family past its gate drew ~4 % of gpu12's steps
        however its candidates did: family 3's two rows sat at 11 %, family 2's 0.20 m gaps at
        36 %, family 5's second step at 25 %, and ADR raised nothing from 264M to 537M"""
        fail = 1.0 - self.rate(f)[1]
        rs = [sum(h) / len(h) for k in W.candidates(f, self.levels[f])
              if len(h := self.probes.get(f'{f}:{k}', [])) >= PROBE_MIN]
        if self.enabled and rs:
            fail = (1.0 - W.PROBE_P) * fail + W.PROBE_P * (1.0 - float(np.mean(rs)))
        return fail

    def weights(self):
        """draw probabilities over the pool: each family's share of the STEPS is how much it
        still fails (`fails`; MIN_SHARE at least), the probability that divided by its
        episodes' mean length. Drawn by episode, a failing family timing out at 1500 steps
        against 20 took 96 % of the steps; at fixed step shares a family behind the frontier
        that regressed got none. A stuck family with no climbs to learn from draws MIN_SHARE"""
        need, lens = [], []
        for f in self.active:
            n, r = self.rate(f)
            capped = self.stalled(f) and r < WARM_RATE and f not in self.taught
            need.append(MIN_SHARE if capped else max(self.fails(f), MIN_SHARE) if n else 1.0)
            lens.append(self.length(f))
        known = [x for x in lens if x]
        fill = sum(known) / len(known) if known else 1.0
        w = [nd / (ln or fill) for nd, ln in zip(need, lens)]
        return [x / sum(w) for x in w]

    def levels_in(self):
        """{family: its levels} over the pool: what rl.rsi offers climbs from"""
        return {f: self.levels[f] for f in self.active}

    def score(self):
        """how far the levels have come -- 0..1 a family -- then, among equals, the pool's
        success from the hang: 0.01 at most, under the smallest step a level takes"""
        return W.progress(self.levels) + 0.01 * float(np.mean([self.rate(f)[1] for f in self.active]))

    def guarded(self):
        """the families the guard holds a best to: those that have passed the gate"""
        return [f for f in self.active if f in self.cleared]

    def held(self, families):
        """ADR's progress, plus how much of the gate each of `families` still holds: 1 at
        UNLOCK_AT or over, 0 at 0 %. Progress alone never falls, so it cannot see a family
        forgotten: one forgotten outright is a whole family's worth (GUARD_DROP)"""
        out = W.progress(self.levels)
        for f in families:
            n, r = self.rate(f)
            out += min(r, UNLOCK_AT) / UNLOCK_AT if n >= WINDOW else 1.0
        return out

    # ---- the walls --------------------------------------------------------------
    def draw_bank(self, env):
        """a fresh bank of BANK walls into `env`: families by the pool's weights, under their
        levels with their probes, and each taught family's share begun at its starts"""
        if self.checker is None:
            self.checker = W.Checker()
        bank = W.draw(self.checker, dict(zip(self.active, self.weights())), self.levels, self.rng,
                      BANK, W.PROBE_P if self.enabled else 0.0)
        if self.enabled and self.ref is not None:
            bank = rsi.mix(bank, self.rng, self.levels_in(), {f: self.share(f) for f in self.active},
                           self.starts, self.ref)
        env.env_method('set_bank', bank)

    def _on_training_start(self):
        """every env onto a wall of the bank before the first step: SB3's learn() has just
        reset them on the lessons' rows"""
        self.draw_bank(self.training_env)
        self.model._last_obs = self.training_env.reset()
        self.model._last_episode_starts = np.ones(self.training_env.num_envs, dtype=bool)

    def _on_rollout_start(self):
        if self.enabled:
            self.track()
        self.draw_bank(self.training_env)


class UpdateSize(BaseCallback):
    """At every rollout's start: each action dimension's std into [STD_MIN, STD_MAX] -- SB3
    starts at 1; a run whose std ran away (to 4.2 on one joint) took itself apart, and one held
    under 1 still let the joints it uses least drift up to it, where they spoil every catch --
    and the learning rate x1/1.5 when the last update's approx_kl passed twice the run's target,
    x1.5 when it was under half, within [LR_MIN, LR_MAX]. The target is the Checkpoint's: each
    rollback halves it"""

    def __init__(self, ck):
        super().__init__()
        self.ck = ck

    def _on_rollout_start(self):
        self.model.policy.log_std.data.clamp_(min=float(np.log(STD_MIN)), max=float(np.log(STD_MAX)))
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


class Clone(BaseCallback):
    """Behaviour cloning of rl.rsi's climbs where the policy does not finish them: every PPO
    optimizer step also pulls the policy's MEAN action at the rows of the starts not passed
    in each climb's frontier window (rl.rsi.cloned) toward the climb's own action there,
    observed as an episode begun at that row observes it. Its gradient is added to PPO's
    after SB3 clips it, scaled to CLONE_RATIO of that one's norm on the actor (less while far
    off: CLONE_OFF), so PPO keeps most of every step whatever the two losses' scales.
    Demonstration-augmented policy gradient (Rajeswaran et al. 2018, DAPG), only where the
    demonstration is better (Nair et al. 2018's Q-filter). Off in the ablation."""

    def __init__(self, cur):
        super().__init__()
        self.cur = cur
        self.ref = cur.ref
        self.act = None if self.ref is None else self.ref['action'].astype(np.float32)
        self.obs = None          # raw observations [row, CLONE_DRAWS, dim] of the climbs...
        self.seen = set()        # ...observed so far, each when first wanted
        self.batch = None
        self.rows = np.zeros(0, int)
        self.rng = np.random.default_rng(0)
        self.hooked = None
        self.log = []

    def wanted(self):
        if self.ref is None or not self.cur.enabled:
            return np.zeros(0, int)
        return rsi.cloned(self.cur.starts, self.ref, self.cur.levels_in())

    def observe(self, climb):
        """(its rows, [rows, CLONE_DRAWS, dim]): each row of climb `climb` observed as an episode
        begun there observes it (its own sensor noise each draw), on JAX MJX -- a reset needs
        no physics, and this batch never steps"""
        import jax
        import jax.numpy as jp
        from . import mjx_task as mt
        if self.batch is None:
            self.batch = mt.Batch(CLONE_CHUNK, bank=CLONE_CHUNK, impl='jax')
        b, L = self.batch, len(mt.LESSON_KEYS)
        rows = np.nonzero(self.ref['climb'] == climb)[0]
        js = np.repeat(rows, CLONE_DRAWS)
        key, out = jax.random.PRNGKey(int(rows[0])), []
        for i in range(0, len(js), CLONE_CHUNK):
            part = js[i:i + CLONE_CHUNK]
            pad = np.r_[part, np.full(CLONE_CHUNK - len(part), part[-1])]
            b.set_bank([dict(rsi.start_wall(self.ref, climb, self.rng), d0_row=L + int(j)) for j in pad])
            key, k = jax.random.split(key)
            _, o = b._reset(b.task.tables, jp.asarray(L + np.arange(CLONE_CHUNK)), jax.random.split(k, CLONE_CHUNK))
            out.append(np.asarray(o)[:len(part)])
        return rows, np.concatenate(out).reshape(len(rows), CLONE_DRAWS, -1)

    def _on_rollout_start(self):
        if self.log:                     # the update just done, into its progress.jsonl line
            off, scale = np.mean(self.log, axis=0)
            self.logger.record('clone/rows', len(self.rows))
            self.logger.record('clone/off_sigma', off)
            self.logger.record('clone/scale', scale)
            self.log.clear()
        rows = self.wanted()
        for climb in set(self.ref['climb'][rows].tolist()) - self.seen if len(rows) else ():
            j, o = self.observe(climb)
            if self.obs is None:
                self.obs = np.zeros((len(self.ref['start']), CLONE_DRAWS, o.shape[-1]), np.float32)
            self.obs[j] = o
            self.seen.add(climb)
        self.rows = rows
        opt = self.model.policy.optimizer
        if self.hooked is not opt:       # a resume builds a new one; a rollback keeps it
            opt.register_step_pre_hook(lambda *_: self.add())
            self.hooked = opt
        return True

    def _on_step(self):
        return True

    def add(self):
        """into the actor's gradients, before the optimizer steps on them"""
        if not len(self.rows):
            return
        import torch
        pol = self.model.policy
        ps = [p for m in (pol.pi_features_extractor, pol.mlp_extractor.policy_net, pol.action_net)
              for p in m.parameters() if p.grad is not None]
        j = self.rng.choice(self.rows, CLONE_BATCH)
        obs, act = self.obs[j, self.rng.integers(CLONE_DRAWS, size=CLONE_BATCH)], self.act[j]
        x = torch.as_tensor(self.model.get_env().normalize_obs(obs), dtype=torch.float32, device=pol.device)
        y = torch.as_tensor(act, device=pol.device)
        with torch.enable_grad():
            mu = pol.get_distribution(x).distribution.mean
            g = torch.autograd.grad(((mu - y) ** 2).sum(1).mean(), ps)
        with torch.no_grad():
            own = torch.sqrt(sum((p.grad ** 2).sum() for p in ps))
            norm = torch.sqrt(sum((d ** 2).sum() for d in g))
            if norm < 1e-12:
                return
            off = float((((mu - y) / pol.log_std.exp()) ** 2).sum(1).sqrt().mean())
            s = float(CLONE_RATIO * min(1.0, CLONE_OFF / max(off, 1e-12)) * own / norm)
            for p, d in zip(ps, g):
                p.grad.add_(d, alpha=s)
            self.log.append((off, s))


class Checkpoint(BaseCallback):
    """Model + observation statistics + curriculum, written together (a model against stale
    statistics sees another observation scale); the best of them kept; the guard; the
    snapshots rl.curve scores; and the console's status line."""

    def __init__(self, run_dir, cur, every, state=None, snap_every=0, walls='adr'):
        super().__init__()
        st = state or {}
        self.dir, self.cur, self.every, self.walls = run_dir, cur, every, walls
        self.snap_every, self.snapped = snap_every, None
        self.best = st.get('best', -1.0)
        self.best_active = st.get('best_active', [])     # the families the best had passed...
        self.best_held = st.get('best_held', self.best)  # ...and what they held there (`held`)
        self.kl_target = st.get('kl_target', KL_START)
        self.rollbacks = st.get('rollbacks', 0)
        self.low_since, self.due, self.last, self.line_at = None, False, 0, None
        self.clock = (time.time(), 0)                     # (wall-clock, steps) at the last line
        self.t0 = time.time()

    def held(self):
        return self.cur.held(self.best_active)

    def patience(self):
        """GUARD_UPDATES in steps, at this run's rollout size"""
        return GUARD_UPDATES * self.model.n_steps * self.model.n_envs

    def guard(self):
        """at each checkpoint: has the pool been a family below its best for long enough to go
        back? The rollback waits for the next rollout to start, so no update mixes two policies"""
        h = self.held()
        if h > self.best_held - GUARD_DROP:
            self.low_since = None
            return
        if self.low_since is None:
            self.low_since = self.model.num_timesteps
        if self.model.num_timesteps - self.low_since >= self.patience():
            self.due = True

    def _on_rollout_start(self):
        if self.due:
            self.rollback()

    def rollback(self):
        """back to the best checkpoint -- policy and optimizer, statistics and curriculum --
        at half the learning rate and half the KL target"""
        m, p = self.model, os.path.join(self.dir, 'best_')
        h = self.held()
        m.set_parameters(p + 'model.zip', exact_match=True, device=m.device)
        with open(p + 'vecnorm.pkl', 'rb') as f:
            stats = pickle.load(f)
        env = m.get_vec_normalize_env()
        env.obs_rms, env.ret_rms = stats.obs_rms, stats.ret_rms
        self.cur.load(json.load(open(p + 'state.json')))
        self.cur.draw_bank(self.training_env)     # this rollout's walls were the dropped curriculum's
        was, was_kl = m.lr_schedule(1.0), self.kl_target
        m.learning_rate = max(was * LR_DECAY, LR_MIN)
        m._setup_lr_schedule()
        self.kl_target = max(self.kl_target * LR_DECAY, KL_MIN)
        self.rollbacks += 1
        self.low_since, self.due = None, False
        record('rollback', timesteps=int(m.num_timesteps), held=h, best=self.best_held,
               pool=self.cur.active, lr=[was, m.learning_rate], kl_target=[was_kl, self.kl_target],
               rollbacks=self.rollbacks)
        say(m.num_timesteps, f"forgetting ({h:.2f} against the best's {self.best_held:.2f} for "
                             f"{GUARD_UPDATES} updates): back to the best, lr {was:g} -> {m.learning_rate:g}")
        self.save()

    def save(self, tag=''):
        p = os.path.join(self.dir, tag)
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        # each file written aside and renamed over the old one: a run killed in the middle
        # of a save must still have a whole checkpoint to resume from
        self.model.save(p + 'model.tmp.zip')
        save_vecnorm(self.model.get_vec_normalize_env(), p + 'vecnorm.tmp.pkl')
        state = {'timesteps': int(self.model.num_timesteps), 'robot': robot_stamp(),
                 'walls_mode': self.walls, 'best': self.best, 'best_active': self.best_active,
                 'best_held': self.best_held, 'lr': self.model.lr_schedule(1.0),
                 'kl_target': self.kl_target, 'rollbacks': self.rollbacks, **self.cur.state()}
        with open(p + 'state.tmp.json', 'w') as f:
            json.dump(state, f, indent=1)
        for f in ('model.zip', 'vecnorm.pkl', 'state.json'):
            os.replace(p + f.replace('.', '.tmp.'), p + f)
        return p

    def _on_rollout_end(self):
        # the best is judged once a rollout, on the policy that collected it
        s = self.cur.score()
        if s > self.best:
            self.best, self.best_active, self.low_since = s, self.cur.guarded(), None
            self.best_held = self.held()
            self.save('best_')
            record('best', timesteps=int(self.model.num_timesteps), score=s, pool=self.cur.active)

    def line(self):
        """the status line: steps, time, speed, each lesson's success from the hang ('-' not
        in the pool yet, '..' nothing counted yet), ADR's progress, the best"""
        n, now = self.model.num_timesteps, time.time()
        sps = (n - self.clock[1]) / max(now - self.clock[0], 1e-9)
        self.clock = (now, n)
        h, m = divmod(int(now - self.t0) // 60, 60)
        rates = []
        for f in W.FAMILIES:
            k, r = self.cur.rate(f)
            rates.append(f"{f}:{'  -' if f not in self.cur.active else ' ..' if not k else f'{100 * r:3.0f}'}")
        return (f"{n / 1e6:>8.1f}M  {h:d}h{m:02d}  {sps / 1e3:5.1f}k/s | " + " ".join(rates)
                + f" | ADR {W.progress(self.cur.levels):.2f}/9  best {max(self.best, 0.0):.2f}")

    def _on_step(self):
        n = self.model.num_timesteps
        if self.snap_every and n // self.snap_every != self.snapped:
            if self.snapped is not None:     # kept: rl.curve scores each on the full robot
                self.save(os.path.join('snaps', f'{n:012d}_'))
            self.snapped = n // self.snap_every
        if n - self.last >= self.every:
            self.last = n
            self.save()
            self.guard()
            base = self.training_env.unwrapped
            cap = base.capped / base.substeps if getattr(base, 'substeps', 0) else None
            if cap is not None:
                base.capped = base.substeps = 0
            ends = collections.defaultdict(dict)
            for (f, why), k in self.cur.ends.items():
                ends[str(f)][why] = k
            self.cur.ends.clear()
            ref = self.cur.ref
            record('checkpoint', timesteps=int(n), pool=self.cur.active,
                   rate={str(f): self.cur.rate(f) for f in self.cur.active},
                   length={str(f): self.cur.length(f) for f in self.cur.active},
                   ends=ends, best=self.best, held=self.held(), best_held=self.best_held,
                   lr=self.model.lr_schedule(1.0), kl_target=self.kl_target, rollbacks=self.rollbacks,
                   contact_cap=cap, std=float(self.model.policy.log_std.detach().exp().mean()),
                   levels=self.cur.levels, progress=W.progress(self.cur.levels),
                   frontier=None if ref is None else rsi.where(self.cur.starts, ref, self.cur.levels_in()),
                   low_since=self.low_since)
        if self.line_at is None or n - self.line_at >= LINE_EVERY:
            if self.line_at is not None:
                print(self.line(), flush=True)
            self.line_at = n
        return True


def build(a, state):
    """(model, VecNormalize env) for a fresh run or a resume"""
    import torch
    envs = a.envs or ENVS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shape = dict(n_steps=PPO_SHAPE['n_steps'], n_epochs=PPO_SHAPE['n_epochs'],
                 batch_size=envs * PPO_SHAPE['n_steps'] // PPO_SHAPE['minibatches'])
    seed = a.seed or 0
    # a resumed batch draws from a key of its own, not the start's sequence again
    base = VecMonitor(MjxVecEnv(envs, seed + state.get('timesteps', 0), impl=a.physics))
    run = os.path.join('out', a.name)
    ck = 'best_' if a.from_best else ''
    if state:
        venv = VecNormalize.load(os.path.join(run, ck + 'vecnorm.pkl'), base)
        venv.training = True
        model = PPO.load(os.path.join(run, ck + 'model'), env=venv, device=device, custom_objects=shape)
        # PPO.load re-seeds torch, numpy and the batch with the bare seed: draw on from it
        model.set_random_seed(seed + model.num_timesteps)
    else:
        venv = VecNormalize(base, norm_reward=False)
        net = [int(x) for x in a.net.split(',')] if a.net else NET
        model = PPO('MlpPolicy', venv, device=device, verbose=0, **shape, gamma=0.995, gae_lambda=0.95,
                    learning_rate=LR_MAX, ent_coef=1e-3, policy_kwargs=dict(net_arch=net), seed=a.seed)
    return model, venv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='climber', help='run directory under out/')
    ap.add_argument('--steps', type=int, default=2_000_000_000, help='steps to run; on a resume, more')
    ap.add_argument('--envs', type=int, default=None, help=f'envs in the GPU batch (default {ENVS})')
    ap.add_argument('--physics', default='jax', choices=['jax', 'warp'],
                    help="MJX's JAX backend or its Warp one (gpu_run.sh passes what rl.mjx_check warp found)")
    ap.add_argument('--walls', default='adr', choices=['adr', 'random'],
                    help='adr: the curriculum; random: every family at its last levels from step 0 (the ablation)')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--from-best', dest='from_best', action='store_true',
                    help='with --resume, carry on from the best checkpoint instead of the latest')
    ap.add_argument('--net', default=None, help=f'hidden sizes of both nets, a fresh run only (default {NET})')
    ap.add_argument('--ckpt-every', type=int, default=1_000_000, help='steps between checkpoints')
    ap.add_argument('--snap-every', type=int, default=50_000_000,
                    help='steps between kept snapshots for the learning curve (rl.curve); 0 = none')
    ap.add_argument('--seed', type=int, default=None,
                    help="a fresh run's policy init and its batch's draws; a paper reports several")
    ap.add_argument('--self-check', action='store_true', help='test the curriculum logic, train nothing')
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if a.net and a.resume:
        raise SystemExit("--net cannot change on a resume: the checkpoint holds the old shape")

    run = os.path.join('out', a.name)
    os.makedirs(run, exist_ok=True)
    log = open(os.path.join(run, 'train.log'), 'a', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    faulthandler.enable(file=log, all_threads=True)     # a crash below Python leaves its stack
    global METRICS
    METRICS = open(os.path.join(run, 'metrics.jsonl'), 'a', encoding='utf-8', buffering=1)
    print(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")
    print(hardware.report())

    state = {}
    if a.resume:
        f = os.path.join(run, ('best_' if a.from_best else '') + 'state.json')
        if not os.path.exists(f):
            raise SystemExit(f"nothing to resume: {f} does not exist")
        state = json.load(open(f))
        check_robot(state)
        if state.get('walls_mode') != a.walls:
            raise SystemExit(f"this run trained on --walls {state.get('walls_mode')}, not {a.walls}")
    model, venv = build(a, state)
    cur = Curriculum(state, enabled=a.walls == 'adr', seed=(a.seed or 0) + state.get('timesteps', 0))
    cur.draw_bank(venv)                  # the first episodes' walls, before learn's reset
    ckpt = Checkpoint(run, cur, a.ckpt_every, state, a.snap_every, a.walls)
    print(f"{'resumed' if state else 'new run'} out/{a.name} at {model.num_timesteps:,} steps: "
          f"{venv.num_envs} envs, {venv.num_envs * model.n_steps:,} steps an update, policy on "
          f"{model.device}, {a.physics} physics, walls {a.walls}; lessons in training {cur.active}")
    try:                                 # SIGTERM saves too, so a killed run is resumable
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    except (ValueError, AttributeError, OSError):
        pass
    model.set_logger(Logger(None, [JsonLines(os.path.join(run, 'progress.jsonl'))]))
    record('start', argv=sys.argv[1:], physics=a.physics, envs=venv.num_envs, device=str(model.device),
           seed=a.seed, timesteps=int(model.num_timesteps), pool=cur.active, n_steps=model.n_steps,
           batch_size=model.batch_size, n_epochs=model.n_epochs, gamma=model.gamma,
           gae_lambda=model.gae_lambda, ent_coef=model.ent_coef, clip_range=float(model.clip_range(1.0)),
           lr=model.lr_schedule(1.0), kl_target=ckpt.kl_target, net_arch=str(model.policy.net_arch),
           robot=robot_stamp(), walls=a.walls, bank=BANK, levels=cur.levels)
    how = 'finished'
    try:
        model.learn(total_timesteps=a.steps, callback=[cur, ckpt, UpdateSize(ckpt), Clone(cur)],
                    reset_num_timesteps=not a.resume)
    except KeyboardInterrupt:
        how = 'interrupted'
    except BaseException as e:
        # not saved: whatever broke may be in the weights (the last checkpoint is whole)
        record('exit', how='crashed', error=repr(e)[:500], timesteps=int(model.num_timesteps))
        raise
    ckpt.save()
    record('exit', how=how, timesteps=int(model.num_timesteps))
    print(ckpt.line())
    print(f"{how} at {model.num_timesteps:,} steps; resume with: python -m rl.train_ppo --name {a.name} --resume")


# --- self-check ------------------------------------------------------------------
class _FakeEnv:
    def __init__(self):
        self.banks = []

    def env_method(self, name, *a):
        self.banks.append(a)


class _FakeModel:
    def __init__(self):
        self.env, self.num_timesteps = _FakeEnv(), 0

    def get_env(self):
        return self.env


def _feed(c, f, ok, n, steps=0, probe=-1, val=0.0, start=-1):
    for _ in range(n):
        c.locals = {'infos': [dict(reason='success' if ok else 'fell', lesson=f, probe=probe,
                                   probe_val=val, steps=steps, start=start)], 'dones': [True]}
        c._on_step()


def _curriculum(**kw):
    c = Curriculum(**kw)
    c.model = _FakeModel()
    c.lag, c.lag_start = 0, {f: 0 for f in c.lag_start}
    c.since = -np.inf                    # its clock is one step an episode: long ones begin before 0
    return c


def self_check():
    """the curriculum, ADR, the guard, the update size and the cloning, without a simulator"""
    # the books count a cohort in the order it began: half pass at step 10, half fall at 900
    c = _curriculum()
    c.lag = 1000
    n = WINDOW
    for t in range(1, 1001):
        ends = ({i: 'success' for i in range(0, n, 2)} if t == 10 else
                {i: 'fell' for i in range(1, n, 2)} if t == 900 else {})
        c.locals = {'infos': [dict(reason=ends[i], lesson=1, steps=t, probe=-1) if i in ends else {}
                              for i in range(n)], 'dones': [i in ends for i in range(n)]}
        c._on_step()
        if t == 10:
            assert c.rate(1) == (0, 0.0), "counted a cohort's fast ends first"
    assert c.rate(1) == (n, 0.5) and c.active == [1], (c.rate(1), c.active)
    c.load(c.state())                                # a rollback: what the old policy began is out
    _feed(c, 1, True, 1, steps=5)
    assert not c.pending, "counted an episode begun before a rollback"
    from .climb_task import FRAME_SKIP
    t, k = np.float32(0.0), 0                       # MJX's float32 clock, a substep at a time
    while t <= W.TIMEOUT:
        for _ in range(FRAME_SKIP):
            t = np.float32(t + np.float32(CONTROL_DT / FRAME_SKIP))
        k += 1
    assert episode_steps(W.TIMEOUT) >= k, "the lag is shorter than an episode can last"
    print("order test: a cohort's fast ends wait for its slow ones; a rollback drops what the old policy began. OK")

    # the gate and the order: family 2 after 1, and only while every family in the pool passes
    c = _curriculum()
    _feed(c, 1, True, WINDOW - 1)
    assert c.active == [1], "entered before a full window"
    _feed(c, 1, True, 1)
    assert c.active == [1, 2], c.active
    _feed(c, 2, False, WINDOW)
    w = c.weights()
    assert w[1] > 5 * w[0], f"a family at 0 % must outweigh one at 100 %: {w}"
    _feed(c, 2, True, WINDOW)
    assert c.active == [1, 2, 3], c.active
    _feed(c, 1, False, WINDOW // 2)                 # family 1 slips to 50 %: 4 waits
    _feed(c, 3, True, WINDOW)
    assert c.active == [1, 2, 3], "entered while family 1 was being forgotten"
    _feed(c, 1, True, WINDOW)
    assert c.active == [1, 2, 3, 4], c.active
    # the steps go where the failures are: a failing family's 1500-step episodes neither
    # swallow the pool nor starve
    _feed(c, 4, False, WINDOW, steps=1500)
    for f in (1, 2, 3):
        _feed(c, f, True, WINDOW, steps=200)
    w = c.weights()
    steps = [x * c.length(f) for x, f in zip(w, c.active)]
    assert 0.5 < steps[3] / sum(steps) < 0.95 and min(steps[:3]) / sum(steps) > 0.03, steps
    # a passed family failing at its candidate draws more steps than one passing at it
    c.probes['2:width'] = [False] * PROBE_MIN
    c.probes['3:spacing'] = [True] * PROBE_MIN
    assert c.fails(2) > c.fails(3) + 0.9 * W.PROBE_P and abs(c.fails(3) - (1 - W.PROBE_P) * (1 - c.rate(3)[1])) < 1e-9
    c.probes.clear()
    # a family still progressing keeps the next one out past STUCK_STEPS; one that stops, not
    c.track()                                       # family 4 at 0 %: its best
    c.num_timesteps = c.entered[4] + STUCK_STEPS - 1
    _feed(c, 4, True, WINDOW // 5, steps=1500)      # 20 % now: progress
    c.track()
    c.num_timesteps = c.entered[4] + STUCK_STEPS
    _feed(c, 1, True, 1)
    assert c.active[-1] == 4, f"a family progressing was taken for stuck: {c.active}"
    c.track()                                       # no better than its best since
    c.num_timesteps = c.moved[4] + STUCK_STEPS
    _feed(c, 1, True, 1)
    assert c.active[-1] == 5, f"a family stuck for STUCK_STEPS still blocks: {c.active}"
    c2 = _curriculum(state=c.state())               # its progress clock survives state.json
    assert c2.moved == c.moved and c2.best_prog == c.best_prog
    print("curriculum test: families enter in order, each once the whole pool passes its gate; the steps "
          "go where the failures are; a family stops blocking only once it stops progressing. OK")

    # ADR: a level rises on a full window passed at its candidate, never past its last; a probe
    # drawn at a candidate since passed does not vote; a rise clears the family's other votes
    c = _curriculum()
    i = [s[0] for s in W.FAMILIES[2]].index('width')
    lv0 = dict(c.levels[2])
    cand = W.candidates(2, lv0)
    _feed(c, 2, True, WINDOW - 1, probe=i, val=cand['width'])
    assert c.levels[2] == lv0, "moved before a full window"
    _feed(c, 2, True, 1, probe=0, val=cand['gaps'])  # the other parameter's window starts...
    _feed(c, 2, True, 1, probe=i, val=cand['width'])
    assert c.levels[2]['width'] == cand['width'] and c.probes['2:gaps'] == [], (c.levels[2], c.probes)
    _feed(c, 2, True, WINDOW, probe=i, val=cand['width'])          # ...and an old candidate's do not count
    assert c.levels[2]['width'] == cand['width'], "counted probes at a candidate already passed"
    _feed(c, 2, False, WINDOW, probe=i, val=W.candidates(2, c.levels[2])['width'])
    assert c.levels[2]['width'] == cand['width'], "a failed window moved the level"
    for _ in range(40):
        _feed(c, 2, True, WINDOW, probe=i, val=W.candidates(2, c.levels[2]).get('width', -1))
    assert c.levels[2]['width'] == W.FAMILIES[2][1][3], "never reached, or went past, its last"
    assert 2 not in c.gate, "a probe counted toward the gate"
    step = min(s / (b - a) / len(ps) for ps in W.FAMILIES.values() for _, _, a, b, s in ps)
    assert 0.01 < step, "a pool's success can outrank a step of a level"
    st = json.loads(json.dumps(c.state()))
    r = Curriculum(st)
    assert r.levels == c.levels and r.active == c.active and r.gate == c.gate, "a state.json round trip"
    st['curriculum']['spec']['2'] = 'other ranges'
    assert Curriculum(st).levels[2] == W.first_levels()[2], "a family whose ranges changed kept its levels"
    off = Curriculum(enabled=False)
    assert off.active == list(W.FAMILIES) and off.levels == W.last_levels()
    print("adr test: a level rises one step on a window passed at its candidate, to its last and no "
          "further; stale probes do not vote; the levels survive state.json. OK")

    if c.ref is not None:                            # the reference climbs' books, by start
        f = int(c.ref['climb_family'][0])
        c.levels[f] = W.first_levels()[f]
        _feed(c, f + rsi.BOOK, True, 3, start=0)
        j = int(c.ref['start'][0])
        assert c.starts.get(j) == [True] * 3 and f not in c.gate, c.starts
        assert Curriculum(json.loads(json.dumps(c.state()))).starts == c.starts, "the start books' round trip"
    guard_check()
    clone_check()
    check_robot({'robot': robot_stamp()})
    try:
        check_robot({'robot': 'another'})
        raise AssertionError("resumed onto another robot")
    except SystemExit:
        pass
    print("self-check: OK")


def _toy_ppo(n_steps=16, batch=16, obs=None):
    from stable_baselines3.common.vec_env import DummyVecEnv

    class Toy(gym.Env):
        observation_space = gym.spaces.Box(-5.0, 5.0, (3,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def reset(self, seed=None, options=None):
            return np.random.randn(3).astype(np.float32), {}

        def step(self, a):
            return np.random.randn(3).astype(np.float32), float(np.random.randn()), False, False, {}

    venv = VecNormalize(DummyVecEnv([Toy]), norm_reward=False)
    return PPO('MlpPolicy', venv, n_steps=n_steps, batch_size=batch, verbose=0, seed=0), venv


def guard_check():
    """a real (tiny) PPO: make a best, let a passed family be forgotten, and see the run put
    itself back -- policy, statistics, curriculum -- at half the rate and half the target"""
    import tempfile
    import torch
    m, venv = _toy_ppo()
    with tempfile.TemporaryDirectory() as d:
        cur = _curriculum()
        cur.draw_bank = lambda env: None              # the toy env takes no walls
        _feed(cur, 1, True, WINDOW)                   # family 1 passed, 2 in
        ck = Checkpoint(d, cur, every=1)
        ck.init_callback(m)
        ck._on_rollout_end()
        assert ck.best_active == [1] and abs(ck.best_held - (W.progress(cur.levels) + 1.0)) < 1e-9
        best_w = {k: v.clone() for k, v in m.policy.state_dict().items()}
        best_mean = venv.obs_rms.mean.copy()
        with torch.no_grad():                         # the drift: weights, statistics...
            for v in m.policy.parameters():
                v.add_(0.1)
        venv.obs_rms.mean += 1.0
        _feed(cur, 1, False, WINDOW)                  # ...and family 1 forgotten
        m.num_timesteps = 10
        ck.guard()
        assert not ck.due and ck.low_since == 10, "tripped on the first low checkpoint"
        P = ck.patience()
        assert P == GUARD_UPDATES * 16, P
        m.num_timesteps = 10 + P - 1
        ck.guard()
        assert not ck.due, "tripped before GUARD_UPDATES"
        m.num_timesteps = 10 + P
        ck.guard()
        assert ck.due, "a passed family forgotten for GUARD_UPDATES did not trip the guard"
        out = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            ck._on_rollout_start()
        finally:
            sys.stdout.close()
            sys.stdout = out
        assert all(torch.equal(v, best_w[k]) for k, v in m.policy.state_dict().items()), "the policy"
        assert np.allclose(venv.obs_rms.mean, best_mean), "the observation statistics"
        assert cur.rate(1)[1] == 1.0 and cur.active == [1, 2], "the curriculum"
        assert abs(m.lr_schedule(1.0) - LR_MAX * LR_DECAY) < 1e-12 and ck.kl_target == KL_START * LR_DECAY
        sys.stdout = open(os.devnull, 'w')
        try:
            for _ in range(12):
                ck.rollback()
        finally:
            sys.stdout.close()
            sys.stdout = out
        assert abs(m.lr_schedule(1.0) - LR_MIN) < 1e-15 and abs(ck.kl_target - KL_MIN) < 1e-12
        # the update size: std held in [STD_MIN, STD_MAX], the rate following the KL toward the target
        from stable_baselines3.common.logger import configure
        m.set_logger(configure(None, []))
        us = UpdateSize(ck)
        us.init_callback(m)
        with torch.no_grad():
            m.policy.log_std.copy_(torch.tensor([-6.0, 1.0]))

        def after(kl, lr):
            m.learning_rate = lr
            m._setup_lr_schedule()
            m.logger.name_to_value['train/approx_kl'] = kl
            us._on_rollout_start()
            return m.lr_schedule(1.0)
        kt = ck.kl_target
        assert abs(after(10 * kt, 1.5e-4) - 1e-4) < 1e-12 and abs(after(0.1 * kt, 1e-4) - 1.5e-4) < 1e-12
        assert after(kt, 1e-4) == 1e-4 and after(0.1 * kt, LR_MAX) == LR_MAX and after(10 * kt, LR_MIN) == LR_MIN
        sd = m.policy.log_std.exp().detach().numpy()
        assert abs(sd[0] - STD_MIN) < 1e-6 and abs(sd[1] - STD_MAX) < 1e-6, sd
    print(f"guard test: a passed family forgotten for {GUARD_UPDATES} updates sends the run back to its best "
          f"(policy, statistics, curriculum) at x{LR_DECAY:g} the rate; std in [{STD_MIN:g}, {STD_MAX:g}]; the rate "
          f"follows the KL within [{LR_MIN:g}, {LR_MAX:g}]. OK")


def clone_check():
    """on a real (tiny) PPO the cloning pulls the mean to the reference with CLONE_RATIO of PPO's
    own gradient (less far off), follows a rollback and a resume, and is off with nothing to clone"""
    import tempfile
    import torch

    class Cur:
        enabled, ref = True, None
        starts, levels_in = {}, staticmethod(lambda: {})

    m, venv = _toy_ppo(64, 16)
    c = Clone(Cur())
    c.ref = dict(climb=np.zeros(8, int), start=np.arange(8) // 2)
    c.act = np.random.default_rng(1).uniform(-0.8, 0.8, (8, 2)).astype(np.float32)
    g = np.random.default_rng(2)            # a row's observations: its state, plus sensor noise
    c.obs = (g.normal(size=(8, 1, 3)) + 0.1 * g.normal(size=(8, CLONE_DRAWS, 3))).astype(np.float32)
    c.seen, rows = {0}, np.arange(4, 8)
    c.wanted = lambda: rows
    c.init_callback(m)
    calls = []
    c.add, add = (lambda: (calls.append(1), add())[1]), c.add

    def off():
        x = torch.as_tensor(venv.normalize_obs(c.obs[rows].reshape(-1, 3)), dtype=torch.float32)
        with torch.no_grad():
            mu = m.policy.get_distribution(x).distribution.mean.numpy()
        return float(np.abs(mu - np.repeat(c.act[rows], CLONE_DRAWS, axis=0)).mean())
    before = off()
    m.learn(64 * 40, callback=[c])
    after = off()
    assert after < 0.5 * before, f"cloning did not pull the mean to the reference: {before:.3f} -> {after:.3f}"
    pol = m.policy
    ps = [p for mod in (pol.mlp_extractor.policy_net, pol.action_net) for p in mod.parameters()]
    for p in pol.parameters():
        p.grad = None
    for p in ps:
        p.grad = torch.full_like(p, 1e-4)
    own = float(torch.sqrt(sum((p.grad ** 2).sum() for p in ps)))
    was = [p.grad.clone() for p in ps]
    add()
    added = float(torch.sqrt(sum(((p.grad - w) ** 2).sum() for p, w in zip(ps, was))))
    assert abs(added - CLONE_RATIO * own) <= 1e-4 * own, f"cloning added {added:g} to PPO's {own:g}"
    near, c.act = c.act, c.act + 20.0                   # the reference far off: it pulls gently
    for p, w in zip(ps, was):
        p.grad = w.clone()
    add()
    far = float(torch.sqrt(sum(((p.grad - w) ** 2).sum() for p, w in zip(ps, was))))
    off_far, c.act = c.log[-1][0], near
    assert off_far > CLONE_OFF and abs(far - CLONE_RATIO * CLONE_OFF / off_far * own) <= 1e-4 * own
    with tempfile.TemporaryDirectory() as d:        # a rollback keeps the optimizer, a resume does not
        m.save(os.path.join(d, 'm'))
        m.set_parameters(os.path.join(d, 'm.zip'), exact_match=True)
        k = len(calls)
        m.learn(64, callback=[c], reset_num_timesteps=False)
        assert len(calls) > k, "the cloning stopped at a rollback"
        m2 = PPO.load(os.path.join(d, 'm'), env=venv)
        c.init_callback(m2)
        k = len(calls)
        m2.learn(64, callback=[c], reset_num_timesteps=False)
        once = len(calls) - k
        m2.learn(64, callback=[c], reset_num_timesteps=False)
        assert once > 0 and len(calls) - k == 2 * once, "the cloning did not follow a resume, or hooked twice"
    rows = np.zeros(0, int)
    c._on_rollout_start()
    assert len(c.rows) == 0, "cloning on with nothing to clone"
    print(f"clone test: {CLONE_RATIO:g}x PPO's gradient (less past {CLONE_OFF:g} sigma off), |mu - a| "
          f"{before:.2f} -> {after:.2f} on a toy; it follows a rollback and a resume. OK")


if __name__ == '__main__':
    main()
