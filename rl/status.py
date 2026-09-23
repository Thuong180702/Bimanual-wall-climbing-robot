"""What a training run is doing, read from its checkpoint. Safe to run while it trains.

    python -m rl.status --name win1
    python -m rl.status --name win1 --watch      # refresh every 30 s

Deliberately imports nothing heavy -- no torch, no mujoco -- so it starts instantly
and can be run from a second terminal without competing for cores with the run it is
reporting on.
"""
import argparse, json, os, time
from datetime import datetime

WINDOW = 40          # must match train_ppo.WINDOW; duplicated to keep this import-free


def read(name):
    d = os.path.join('out', name)
    f = os.path.join(d, 'state.json')
    if not os.path.exists(f):
        raise SystemExit(f"no checkpoint yet at {f}\n"
                         f"the first one lands at --ckpt-every steps (default 100000)")
    st = json.load(open(f))
    st['_age'] = time.time() - os.path.getmtime(f)
    st['_best_exists'] = os.path.exists(os.path.join(d, 'best_model.zip'))
    return st


def show(name, names):
    st = read(name)
    from .lessons import lesson_key
    hist = {lesson_key(k): v for k, v in st['history'].items()}
    print(f"{'run':>12}: out/{name}")
    print(f"{'steps':>12}: {st['timesteps']:,}")
    print(f"{'pool':>12}: {st['active']}")
    b = st.get('best')
    # a sum of per-lesson success rates, so 1.00 can be one lesson passed or two
    # half-passed; the per-lesson table below is what to read for detail
    print(f"{'best':>12}: " + (f"{b:.2f} of {len(st['active'])}"
                               if isinstance(b, (int, float)) and b >= 0
                               else "not scored yet")
          + ("  (best_model.zip saved)" if st['_best_exists'] else ""))
    if 'lr' in st:
        n = st.get('rollbacks', 0)
        kl = st.get('kl_target')
        print(f"{'lr':>12}: {st['lr']:g}"
              + (f", KL target {kl:g}" if kl else "")
              + (f"   (went back to the best {n} time(s))" if n else ""))
    print(f"{'updated':>12}: {st['_age']:.0f} s ago"
          + ("   <-- STALE, is the run still alive?" if st['_age'] > 600 else ""))
    print(f"\n  {'lesson':>22} {'success':>9} {'episodes':>9}")
    for k in st['active']:
        h = hist.get(k, [])[-WINDOW:]
        rate = f"{sum(h)/len(h):.0%}" if h else "-"
        print(f"  {names.get(k, k):>22} {rate:>9} {len(h):>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='climber')
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--every', type=float, default=30.0)
    a = ap.parse_args()
    # lesson names come from the spec table, which needs no simulator to read
    from .lessons import LESSONS
    names = {k: v['name'] for k, v in LESSONS.items()}
    while True:
        print(f"\n=== {datetime.now():%H:%M:%S} ===")
        try:
            show(a.name, names)
        except SystemExit as e:
            print(e)
        if not a.watch:
            return
        time.sleep(a.every)


if __name__ == '__main__':
    main()
