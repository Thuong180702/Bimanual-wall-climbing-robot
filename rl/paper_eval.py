"""Every evaluation the paper reports, over its runs, into out/<run>/paper.jsonl -- resumable:
a job done is recorded in out/<run>/paper_done.txt and skipped the next time.

    python -m rl.paper_eval --list                   # what would run, over the runs found in out/
    python -m rl.paper_eval --jobs 8                 # run it (a rented CPU box: one job a core)
    python -m rl.paper_eval --only p_s0 --jobs 1     # one run's jobs
    python -m rl.paper_eval --quick --only gpu14_now # a smoke test: 2 episodes / walls a job

Runs are found by name (paper/REVISION_PLAN.md section 8, rl/gpu_run.sh <name>):

    p_s{seed}          the method, 1.5B steps, seeds 0..4      every job below
    a_{name}_s{seed}   an ablation (train_ppo --ablate name), 0.75B, seeds 0..2; a_none_s{seed}
                       the method under the same budget        benchmark, held out at the last levels
    b{f}_s{seed}       family f's specialist                   its own lesson and family
    w_{tag}_s{seed}    a reward-weight sensitivity run         benchmark, held out at the last levels
    x_{rule}           an exploit run (--ablate rule, short)   nothing here: paper/figsrc/tables.py
                                                               reads its metrics.jsonl

and the do-nothing control and the left/right symmetry once (no run). paper/figsrc/tables.py
turns the jsonl into the paper's tables. Each job is one rl.eval (or rl.mjx_check policy)
process: its CPU is one core (OMP_NUM_THREADS=1), so --jobs is how many cores it takes.
"""
import argparse, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

OUT = 'out'
FAM = range(1, 8)
# one step past each family's last level, its hardest parameter (rl.walls.FAMILIES)
BEYOND = {1: 'shift=1.50', 2: 'width=0.32', 3: 'spacing=0.50', 4: 'spacing=0.50', 5: 'drop=0.50',
          6: 'rise=0.55', 7: 'side=1.15'}
# robustness: inside the training randomization (climb_task.DR_*) and past it (dagger)
CONDS = ['noise=2', 'noise=3', 'mass=1.1', 'mass=1.2', 'friction=0.7', 'friction=1.3', 'friction=0.5',
         'torque=0.8', 'torque=0.7', 'lag=20', 'lag=40', 'kp=0.7', 'kp=1.3']
OPEN_CONDS = ['', 'mass=1.1', 'friction=0.7', 'torque=0.8', 'lag=20']


def jobs_of(run, quick=False):
    """[(key, argv)] of one run, by its role"""
    E = lambda n: str(2 if quick else n)
    ev = ['-m', 'rl.eval', '--name', run, '--from-best', '--jsonl', os.path.join(OUT, run, 'paper.jsonl')]
    # every lesson and family named: left to its default rl.eval scores the checkpoint's pool, and an
    # ablation that never let a family in would have no number for it -- its score is what it does there
    ALL = ['--lessons', ','.join(map(str, FAM))]
    bench = [(f'bench{m}', ev + ['--episodes', E(100)] + ([m] if m else [])) for m in ('', '--mirror')]
    last = [('heldout_last', ev + ['--walls', E(200), '--levels', 'last'])]
    m = re.fullmatch(r'b(\d)_s\d+', run)
    if m:                                    # a specialist: its own lesson and family
        f = m.group(1)
        return [(k, a + ['--lessons', f]) for k, a in bench + last]
    if run.startswith(('a_', 'w_')):
        return [(k, a + ALL) for k, a in bench + last]
    if not run.startswith('p_s'):
        return []
    out = [(k, a + ALL) for k, a in bench + last + [
        ('heldout_reached', ev + ['--walls', E(200), '--levels', 'reached']),
        ('bench_simp', ev + ['--episodes', E(100), '--robot', 'simp']),
        ('torque', ev + ['--episodes', E(20), '--torque']),
        ('torque_m', ev + ['--episodes', E(20), '--torque', '--mirror']),
    ]]
    out += [('sim2sim_mjx', ['-m', 'rl.mjx_check', 'policy', '--run', os.path.join(OUT, run), '--episodes', E(50)])]
    out += [(f'beyond{f}', ev + ['--walls', E(50), '--levels', 'last', '--lessons', str(f), '--set', BEYOND[f]])
            for f in FAM]
    out += [(f'distract_{d}', ev + ['--walls', E(100), '--levels', 'last', '--distract', d] + ALL)
            for d in ('above', 'below', 'long')]
    out += [(f'robust_{c}{m}', ev + ['--episodes', E(50), '--cond', c] + ([m] if m else []) + ALL)
            for c in CONDS for m in ('', '--mirror')]
    out += [(f'openloop_{c or "nominal"}', ev + ['--open-loop', '--episodes', E(20), '--lessons', '1,2,3,4,5,6,7']
             + (['--cond', c] if c else []))
            for c in OPEN_CONDS]
    return out


def once(quick=False):
    """the jobs with no run: the climbs open loop on both robots, the symmetry"""
    j = os.path.join(OUT, '_paper', 'paper.jsonl')
    ev = ['-m', 'rl.eval', '--name', '', '--jsonl', j]
    return [(f'openloop_{r}', ev + ['--open-loop', '--robot', r]) for r in ('full', 'simp')] + [
        ('symmetry', ['-m', 'rl.eval', '--symmetry'])]


def runs():
    pat = re.compile(r'(p_s\d+|a_\w+_s\d+|b\d_s\d+|w_\w+_s\d+)$')
    return sorted(r for r in os.listdir(OUT) if pat.match(r) and os.path.exists(os.path.join(OUT, r, 'best_model.zip')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1, help='processes at once (one core each)')
    ap.add_argument('--only', default=None, help='comma-separated run names (any name: its role by prefix)')
    ap.add_argument('--list', action='store_true', help='print the jobs not yet done, run nothing')
    ap.add_argument('--quick', action='store_true', help='2 episodes / walls a job, into paper_quick.*')
    a = ap.parse_args()
    names = a.only.split(',') if a.only else runs() + ['_paper']
    todo = []
    for r in names:
        js = once(a.quick) if r == '_paper' else jobs_of(r, a.quick)
        if r != '_paper' and not js and a.only:   # any other name, e.g. gpu14_now: as a method run
            js = [(k, [x.replace('p_s0', r) for x in v]) for k, v in jobs_of('p_s0', a.quick)]
        tag = 'paper_quick' if a.quick else 'paper'
        js = [(k, [x.replace('paper.jsonl', tag + '.jsonl') for x in v]) for k, v in js]
        os.makedirs(os.path.join(OUT, r), exist_ok=True)
        done_f = os.path.join(OUT, r, tag + '_done.txt')
        done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
        todo += [(r, k, v, done_f) for k, v in js if k not in done]
    if a.list or not todo:
        for r, k, v, _ in todo:
            print(f"{r:>16} {k:<22} python {' '.join(v)}")
        print(f"{len(todo)} jobs to do")
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               JAX_PLATFORMS=os.environ.get('JAX_PLATFORMS', 'cpu'),
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')

    def one(job):
        r, k, v, done_f = job
        log = os.path.join(OUT, r, f"{os.path.basename(done_f)[:-9]}_{k}.log")
        with open(log, 'w') as fh:
            p = subprocess.run([sys.executable, *v], stdout=fh, stderr=subprocess.STDOUT, env=env)
        # mjx_check policy asserts the three robots agree: its rows are the result either way
        ok = p.returncode == 0 or k == 'sim2sim_mjx' and 'total' in open(log).read()
        if ok:
            with open(done_f, 'a') as fh:
                fh.write(k + '\n')
        print(f"{'done' if ok else 'FAILED':>6} {r} {k}" + ('' if ok else f" (see {log})"), flush=True)
        return ok

    with ThreadPoolExecutor(a.jobs) as pool:
        ok = list(pool.map(one, todo))
    print(f"{sum(ok)}/{len(ok)} jobs done")
    sys.exit(0 if all(ok) else 1)


if __name__ == '__main__':
    main()
