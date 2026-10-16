"""The learning curve: every kept snapshot of a run scored on the benchmark, full robot.

    python -m rl.curve --name gpu1                     # -> out/gpu1/curve.jsonl
    python -m rl.curve --name gpu1 --episodes 20 --jobs 12
    python -m rl.plot --name gpu1                      # curve.png from it

rl.train_ppo keeps a snapshot every --snap-every steps (out/<name>/snaps/). This runs
rl.eval on each one, on lessons 1..9 in C MuJoCo on the full robot -- walls training never
draws as such -- one process per (snapshot, lesson), so a Mac's cores or the rented box's
do the whole curve at once. Appends, and skips what curve.jsonl already
has, so it can be run again as a run goes on.
"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--lessons', default='1,2,3,4,5,6,7,8,9')
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    a = ap.parse_args()
    run = os.path.join('out', a.name)
    out = os.path.join(run, 'curve.jsonl')
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r['checkpoint'], str(r['lesson']), bool(r.get('mirror'))))
    snaps = sorted(os.path.basename(p)[:-len('model.zip')]
                   for p in glob.glob(os.path.join(run, 'snaps', '*_model.zip')))
    # both ways round: each lesson as drawn (toward the robot's left) and mirrored
    todo = [(f'snaps/{s}', k, mir) for s in snaps for k in a.lessons.split(',')
            for mir in (False, True) if (f'snaps/{s}model', k, mir) not in done]
    print(f"{len(snaps)} snapshots, {len(todo)} (snapshot, lesson, direction) jobs to score, "
          f"{a.episodes} episodes each, {a.jobs} at a time")

    def one(job):
        ck, k, mir = job
        r = subprocess.run([sys.executable, '-m', 'rl.eval', '--name', a.name, '--ckpt', ck,
                            f'--lessons={k}', '--episodes', str(a.episodes), '--jsonl', out]
                           + (['--mirror'] if mir else []), capture_output=True, text=True)
        return job, r.returncode, r.stderr[-300:]

    with ThreadPoolExecutor(a.jobs) as ex:
        for (ck, k, mir), rc, err in ex.map(one, todo):
            print(f"  {ck} lesson {k}{' mirrored' if mir else ''}: "
                  f"{'ok' if rc == 0 else 'FAILED ' + err}", flush=True)
    print(f"-> {out}")


if __name__ == '__main__':
    main()
