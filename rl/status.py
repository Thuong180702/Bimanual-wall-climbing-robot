"""What a training run is doing, read from its checkpoint. Safe to run while it trains.

    python -m rl.status --name win1
    python -m rl.status --name win1 --watch      # refresh every 30 s

Deliberately imports nothing heavy -- no torch, no mujoco -- so it starts instantly
and can be run from a second terminal without competing for cores with the run it is
reporting on.
"""
import argparse, json, os, time
from datetime import datetime

WINDOW = 40          # train_ppo.WINDOW, for a state.json that does not say its own window;
                     # duplicated to keep this import-free


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
    tops = (st.get('walls') or {}).get('tops')
    # a sum of per-lesson success rates, so 1.00 can be one lesson passed or two
    # half-passed; the per-lesson table below is what to read for detail. On generated
    # walls it is ADR's progress instead, 0..1 a family over all of them
    print(f"{'best':>12}: " + ((f"{b:.2f} of {len(tops)} (ADR progress)" if tops else
                                f"{b:.2f} of {len(st['active'])}")
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
    if tops:                               # a generated-wall run: ADR's ranges so far
        print(f"{'walls':>12}: {st.get('walls_mode')}, ranges' tops (rl.walls):")
        for f, t in sorted(tops.items(), key=lambda kv: int(kv[0])):
            print(f"{'':>14}family {f}: " + ", ".join(f"{k} {v:g}" for k, v in t.items()))
    print(f"\n  {'family' if tops else 'lesson':>22} {'success':>9} {'episodes':>9}")
    for k in st['active']:
        h = hist.get(k, [])[-st.get('window', WINDOW):]
        rate = f"{sum(h)/len(h):.0%}" if h else "-"
        print(f"  {f'{k} {names.get(k, k)}' if tops else names.get(k, k):>22} {rate:>9} {len(h):>9}")
    for k in sorted(x for x in hist if isinstance(x, int) and x >= 100):   # rl.rsi's book
        h = hist[k][-st.get('window', WINDOW):]
        print(f"  {f'{k - 100} started partway up':>22} {sum(h) / len(h):>9.0%} {len(h):>9}"
              "   (rl.rsi: not the gate's)")
    starts = (st.get('walls') or {}).get('starts')
    if starts is not None and tops:        # where along each climb its starts are learnt from
        from . import rsi
        ref = rsi.load()
        if ref is not None and st['walls'].get('rsi') != rsi.signature(ref):
            print(f"  {'starts frontier':>22}   - (its books are by another rl.rsi fixture's starts; "
                  "a resume starts them over)")
            return
        fr = rsi.where({int(j): v for j, v in starts.items()},
                       {int(f): t for f, t in tops.items() if int(f) in rsi.TOP and int(f) in st['active']})
        print(f"  {'starts frontier':>22}   " + ", ".join(
            f"{s} climb {'all passed' if v is None else f'from {v:g} s'}" for s, v in fr.items()))


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
