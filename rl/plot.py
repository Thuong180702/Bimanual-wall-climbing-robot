"""The figures of a run, from the logs it keeps -- for reading a run and for the paper.

    python -m rl.plot --name gpu1          # -> out/gpu1/plots/*.png

Reads out/<name>/progress.jsonl (PPO, one line an update), metrics.jsonl (one line a
checkpoint or event) and eval.jsonl (the full-robot score), all written by
rl.train_ppo / rl.eval / rl/gpu_run.sh, and draws:

    success.png    each lesson's success rate over the run, unlocks and rollbacks marked
    training.png   episode return, approx_kl against its target, learning rate, action std
    ends.png       how the episodes ended, as shares of each checkpoint's episodes
    adr.png        on generated walls: each family's progress through its ranges (rl.walls)
    eval.png       the full robot's success per lesson with its 95 % interval, and the
                   do-nothing control
    curve.png      the benchmark on the full robot at every kept snapshot (rl.curve)

Needs matplotlib (not in requirements-gpu.txt: plot at home, after croc).
"""
import argparse, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mark(ax, events):
    for e in events:
        c = 'tab:green' if e['kind'] == 'unlock' else 'tab:red'
        ax.axvline(e['timesteps'] / 1e6, color=c, ls=':', lw=0.8)
        if e['kind'] == 'unlock':
            ax.text(e['timesteps'] / 1e6, 1.02, str(e['lesson']), color=c, fontsize=7,
                    ha='center', transform=ax.get_xaxis_transform())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    run = os.path.join('out', a.name)
    out = os.path.join(run, 'plots')
    os.makedirs(out, exist_ok=True)
    prog, met, ev, cv = (rows(os.path.join(run, f)) for f in
                         ('progress.jsonl', 'metrics.jsonl', 'eval.jsonl', 'curve.jsonl'))
    cks = [m for m in met if m['kind'] == 'checkpoint']
    events = [m for m in met if m['kind'] in ('unlock', 'rollback')]

    if cks:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        lessons = sorted({k for c in cks for k in c['rate']}, key=float)
        for k in lessons:
            pts = [(c['timesteps'] / 1e6, c['rate'][k][1]) for c in cks
                   if c['rate'].get(k, [0])[0]]         # [0]: nothing counted yet
            if pts:
                ax.plot(*zip(*pts), label=f"{'family' if cks[-1].get('tops') else 'lesson'} {k}", lw=1.2)
        mark(ax, events)
        ax.set(xlabel='env steps (M)', ylabel='success rate (last 40 episodes)', ylim=(0, 1.05))
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'success.png'), dpi=150)

        fig, ax = plt.subplots(figsize=(9, 4))
        reasons = sorted({w for c in cks for d in c['ends'].values() for w in d})
        x = [c['timesteps'] / 1e6 for c in cks]
        tot = [max(1, sum(n for d in c['ends'].values() for n in d.values())) for c in cks]
        share = [[sum(d.get(w, 0) for d in c['ends'].values()) / t for c, t in zip(cks, tot)]
                 for w in reasons]
        ax.stackplot(x, *share, labels=reasons)
        ax.set(xlabel='env steps (M)', ylabel='share of episodes', ylim=(0, 1))
        ax.legend(fontsize=7, loc='upper left')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'ends.png'), dpi=150)

    if cks and cks[-1].get('tops'):
        from .walls import FAMILIES
        fig, ax = plt.subplots(figsize=(9, 4))
        for f, ps in FAMILIES.items():
            pts = [(c['timesteps'] / 1e6,
                    sum((c['tops'][str(f)][n] - a0) / (b0 - a0) for n, _, a0, b0, _ in ps) / len(ps))
                   for c in cks if c.get('tops')]
            ax.plot(*zip(*pts), label=f'family {f}', lw=1.2)
        mark(ax, events)
        ax.set(xlabel='env steps (M)', ylabel='ADR progress (first -> last range)', ylim=(0, 1.05))
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'adr.png'), dpi=150)

    if prog:
        x = [p.get('time/total_timesteps', 0) / 1e6 for p in prog]
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        for ax, key, label in ((axs[0, 0], 'rollout/ep_rew_mean', 'episode return'),
                               (axs[0, 1], 'train/approx_kl', 'approx KL per update'),
                               (axs[1, 0], 'train/learning_rate', 'learning rate'),
                               (axs[1, 1], 'train/std', 'action std (mean)')):
            pts = [(xi, p[key]) for xi, p in zip(x, prog) if key in p]
            if pts:
                ax.plot(*zip(*pts), lw=0.8)
            ax.set(ylabel=label)
            mark(ax, events)
        if cks:
            axs[0, 1].plot([c['timesteps'] / 1e6 for c in cks], [c['kl_target'] for c in cks],
                           'k--', lw=0.8, label='KL target')
            axs[0, 1].legend(fontsize=7)
        axs[0, 1].set_yscale('log')
        for ax in axs[1]:
            ax.set(xlabel='env steps (M)')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'training.png'), dpi=150)

    if ev:
        ev = sorted(ev, key=lambda e: float(e['lesson']))
        fig, ax = plt.subplots(figsize=(10, 4))
        xs = range(len(ev))
        p = [e['success'] / e['episodes'] for e in ev]
        err = [[pi - e['ci95'][0] for pi, e in zip(p, ev)], [e['ci95'][1] - pi for pi, e in zip(p, ev)]]
        ax.bar(xs, p, yerr=err, capsize=3, label=f"policy ({ev[0]['episodes']} episodes, 95 % CI)")
        ax.plot(xs, [e.get('control_success', 0) for e in ev], 'kx', label='do nothing')
        ax.set_xticks(list(xs), [f"{e['lesson']}\n{e['name']}" for e in ev], fontsize=6, rotation=60)
        ax.set(ylabel='success on the full robot', ylim=(0, 1.05))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'eval.png'), dpi=150)
    if cv:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        by = {}
        for r in cv:
            by.setdefault(str(r['lesson']), []).append((r['timesteps'] / 1e6, r['success'] / r['episodes']))
        for k in sorted(by, key=float):
            ax.plot(*zip(*sorted(by[k])), label=f'lesson {k}', lw=1.0, marker='.')
        steps = sorted({t for v in by.values() for t, _ in v})
        mean = [sum(p for t2, p in sum(by.values(), []) if t2 == t) / sum(1 for t2, _ in sum(by.values(), []) if t2 == t)
                for t in steps]
        ax.plot(steps, mean, 'k-', lw=2.0, label='mean of the benchmark')
        ax.set(xlabel='env steps (M)', ylabel='success on the full robot', ylim=(0, 1.05))
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'curve.png'), dpi=150)
    print(f"plots in {out}: {sorted(os.listdir(out))}")


if __name__ == '__main__':
    main()
