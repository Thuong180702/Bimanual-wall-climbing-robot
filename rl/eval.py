"""Score a saved checkpoint on the benchmark -- the full robot in C MuJoCo -- against a
do-nothing control.

    python -m rl.eval --name gpu15                 # every lesson in its pool
    python -m rl.eval --name gpu15 --lessons 1 --episodes 20
    python -m rl.eval --name gpu15 --lessons 1 --mirror     # the lesson toward the right
    python -m rl.eval --name gpu15 --from-best --walls 200  # 200 unseen walls a family (held-out)

A benchmark lesson is ONE wall: its episodes differ only through the sensor noise, and its
parameters lie inside the family's training range. `--walls N` scores what the paper calls
the held-out set instead: N walls a family drawn as training draws them (rl.walls, up to
the checkpoint's levels or `--levels last`), from seeds no training run draws from, one
episode each, so the episodes are independent walls and the interval is over walls.

The control matters more than the score. "mean final distance 2.1 m" says nothing on
its own -- the robot starts hanging, and hanging still is already a legal episode. So
every lesson is also run with zero actions, and the policy's number is only evidence
if it beats that.
"""
import argparse, json, os
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    raise SystemExit(f"missing training deps ({e}); pip install -r requirements.txt")

from .climb_task import ClimbTask
from .lessons import LESSONS, lesson_key


WALL_SEED = 2 ** 40      # + family: past every seed + timesteps a training run draws walls from


class Lesson(gym.Env):
    """one benchmark lesson on the full robot, as a gym env (without its visual meshes
    unless asked: only rl.render draws them); or, with `xml` and `spec`, a generated wall
    starting from lesson `lesson`'s hang"""

    def __init__(self, lesson, mirror=False, visual=False, xml=None, spec=None):
        self.task = ClimbTask(lesson, visual=visual, mirror=mirror, xml=xml, spec=spec)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.task.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (17,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.task.reset(seed)

    def step(self, a):
        return self.task.step(a)


def run(venv, episodes, policy=None):
    """episodes of one lesson; returns (successes, final distances, end reasons)"""
    obs = venv.reset()
    ok, dists, why = 0, [], []
    while len(why) < episodes:
        if policy is None:
            act = np.zeros((1, 17), np.float32)
        else:
            act, _ = policy.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(act)
        for info, done in zip(infos, dones):
            if done:
                why.append(info.get('reason', 'timeout'))
                dists.append(info['dist'])
                ok += info.get('reason') == 'success'
    return ok, dists, why


def wilson(k, n, z=1.96):
    """95 % Wilson score interval for k successes in n episodes -- what a success
    rate over a few dozen episodes is honestly worth"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def held_out(run_dir, ck, state, families, n, levels, robot, device, jsonl):
    """`n` unseen walls of each family, one deterministic episode each (`--walls`)"""
    from . import walls as W
    from .mjx_scene import c_scene
    if state.get('families'):
        W.keep(state['families'])
    lv = W.last_levels() if levels == 'last' else \
        {int(f): v for f, v in state['curriculum']['levels'].items()}
    model = PPO.load(os.path.join(run_dir, f'{ck}model'), device=device)
    checker = W.Checker()
    print(f"{'family':>8} {'success':>12} {'95% CI':>12}   ends   ({n} walls, {robot} robot, {levels} levels)")
    for f in families:
        rng = np.random.default_rng(WALL_SEED + f)
        ok, why = 0, []
        for lay in W.draw(checker, {f: 1.0}, lv, rng, n, probe_p=0.0):
            path, spec = c_scene(lay, robot)
            try:
                venv = DummyVecEnv([lambda: Lesson(lay['hang'], xml=path, spec=spec)])
                venv = VecNormalize.load(os.path.join(run_dir, f'{ck}vecnorm.pkl'), venv)
                venv.training, venv.norm_reward = False, False
                k, _, w = run(venv, 1, model)
                venv.close()
            finally:
                os.remove(path)
            ok, why = ok + k, why + w
        tally = {r: why.count(r) for r in sorted(set(why))}
        lo, hi = wilson(ok, n)
        print(f"{f:>8} {ok:>6}/{n:<5} {lo:>5.0%}-{hi:<5.0%}  {tally}", flush=True)
        if jsonl:
            with open(jsonl, 'a') as fh:
                fh.write(json.dumps(dict(mode='walls', family=f, robot=robot, levels=levels,
                                         level=lv[f], run=os.path.basename(run_dir),
                                         checkpoint=ck + 'model', timesteps=state['timesteps'],
                                         success=ok, episodes=n, ci95=(lo, hi), reasons=tally)) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='climber')
    ap.add_argument('--lessons', default=None, help='default: the unlocked pool')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--from-best', dest='from_best', action='store_true',
                    help='score best_model.zip instead of the latest checkpoint')
    ap.add_argument('--ckpt', default=None,
                    help='score this checkpoint prefix instead, e.g. snaps/000050000000_ '
                         '(rl.curve does, for every snapshot)')
    ap.add_argument('--jsonl', default=None,
                    help='also append one JSON line per lesson here (rl.plot reads it)')
    ap.add_argument('--mirror', action='store_true',
                    help='each lesson mirrored, toward the robot\'s right (rl.lessons.mirrored)')
    ap.add_argument('--walls', type=int, default=0,
                    help='score N unseen generated walls per family instead of the benchmark lessons')
    ap.add_argument('--levels', default='reached', choices=['reached', 'last'],
                    help="--walls: draw up to the checkpoint's ADR levels, or every family's last")
    ap.add_argument('--robot', default='full', choices=['full', 'simp'],
                    help='--walls: the full robot, or the simplified one training uses (both in C)')
    a = ap.parse_args()

    run_dir = os.path.join('out', a.name)
    ck = a.ckpt if a.ckpt is not None else ('best_' if a.from_best else '')
    state = json.load(open(os.path.join(run_dir, f'{ck}state.json')))
    pool = state['curriculum']['active'] if 'curriculum' in state else state['active']
    lessons = [lesson_key(x) for x in a.lessons.split(',')] if a.lessons else pool
    print(f"{run_dir}: {state['timesteps']} steps, pool {pool}\n")
    if a.walls:
        fams = [int(x) for x in a.lessons.split(',')] if a.lessons else pool
        return held_out(run_dir, ck, state, fams, a.walls, a.levels, a.robot, a.device, a.jsonl)
    print(f"{'lesson':>24} {'policy':>18} {'do nothing':>18}")

    for k in lessons:
        row = []
        for policy_on in (True, False):
            venv = DummyVecEnv([lambda: Lesson(k, mirror=a.mirror)])
            venv = VecNormalize.load(os.path.join(run_dir, f'{ck}vecnorm.pkl'), venv)
            venv.training = False          # freeze the statistics; an eval that keeps
            venv.norm_reward = False       # updating them is scoring a moving target
            model = (PPO.load(os.path.join(run_dir, f'{ck}model'), env=venv,
                              device=a.device) if policy_on else None)
            # doing nothing is one episode: the reset is deterministic, and the SLAM
            # noise that makes the policy's episodes differ never reaches a zero action
            ok, dists, why = run(venv, a.episodes if policy_on else 1, model)
            row.append(f"{ok}/{len(why)} ok {np.mean(dists):.2f} m")
            if policy_on:
                tally = {r: why.count(r) for r in sorted(set(why))}
                res = dict(success=ok, episodes=len(why), ci95=wilson(ok, len(why)),
                           reasons=tally, dist=float(np.mean(dists)))
            else:
                res.update(control_success=ok, control_dist=float(np.mean(dists)))
            venv.close()
        lo, hi = res['ci95']
        print(f"{k:>2}{'m' if a.mirror else ' '} {LESSONS[k]['name']:>20} {row[0]:>18} "
              f"{row[1]:>18}   {tally}   95% CI {lo:.0%}-{hi:.0%}")
        if a.jsonl:
            with open(a.jsonl, 'a') as f:
                f.write(json.dumps(dict(lesson=k, name=LESSONS[k]['name'], mirror=a.mirror,
                                        run=a.name, checkpoint=ck + 'model',
                                        timesteps=state['timesteps'], **res)) + '\n')

    print("\n'm' is mean distance from both goals at the episode end -- lower is "
          "better; a lesson marked 'm' ran mirrored, toward the robot's right.\nThe "
          "policy has learned something only where it beats the control.")


if __name__ == '__main__':
    main()
