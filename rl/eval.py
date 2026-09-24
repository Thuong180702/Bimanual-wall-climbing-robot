"""Score a saved checkpoint, against a do-nothing control.

    python -m rl.eval --name gpu15                 # every lesson it trained on
    python -m rl.eval --name gpu15 --lessons 1 --episodes 20

The control matters more than the score. "mean final distance 2.1 m" says nothing on
its own -- the robot starts hanging, and hanging still is already a legal episode. So
every lesson is also run with zero actions, and the policy's number is only evidence
if it beats that.
"""
import argparse, json, os
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    raise SystemExit(f"missing training deps ({e}); pip install -r requirements.txt")

from .train_ppo import MultiLesson
from .lessons import LESSONS, lesson_key


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
    a = ap.parse_args()

    run_dir = os.path.join('out', a.name)
    ck = a.ckpt if a.ckpt is not None else ('best_' if a.from_best else '')
    state = json.load(open(os.path.join(run_dir, f'{ck}state.json')))
    lessons = ([lesson_key(x) for x in a.lessons.split(',')] if a.lessons
               else state['active'])
    print(f"{run_dir}: {state['timesteps']} steps, pool {state['active']}\n")
    print(f"{'lesson':>24} {'policy':>18} {'do nothing':>18}")

    for k in lessons:
        row = []
        for policy_on in (True, False):
            venv = DummyVecEnv([lambda: MultiLesson([k])])
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
        print(f"{k:>2} {LESSONS[k]['name']:>21} {row[0]:>18} {row[1]:>18}   {tally}"
              f"   95% CI {lo:.0%}-{hi:.0%}")
        if a.jsonl:
            with open(a.jsonl, 'a') as f:
                f.write(json.dumps(dict(lesson=k, name=LESSONS[k]['name'], run=a.name,
                                        checkpoint=ck + 'model', timesteps=state['timesteps'],
                                        **res)) + '\n')

    print("\n'm' is mean distance from both goals at the episode end -- lower is "
          "better.\nThe policy has learned something only where it beats the control.")


if __name__ == '__main__':
    main()
