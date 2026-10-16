"""What a training run is doing, read from its checkpoint. Safe to run while it trains.

    python -m rl.status --name gpu1
    python -m rl.status --name gpu1 --watch      # refresh every 30 s

Imports nothing heavy -- no torch, no mujoco, no JAX -- so it starts instantly and does not
compete for cores with the run it is reporting on.
"""
import argparse, json, os, time
from datetime import datetime

from .lessons import LESSONS             # numpy and a table: no simulator


def show(name):
    f = os.path.join('out', name, 'state.json')
    if not os.path.exists(f):
        raise SystemExit(f"no checkpoint yet at {f}")
    st = json.load(open(f))
    age = time.time() - os.path.getmtime(f)
    c = st['curriculum']
    print(f"{'run':>10}: out/{name}, walls {st.get('walls_mode')}")
    print(f"{'steps':>10}: {st['timesteps']:,}" + (f"   (checkpoint {age:.0f} s old: is the run alive?)"
                                                    if age > 600 else ""))
    print(f"{'best':>10}: {st['best']:.2f} of 9 (ADR progress)" if st.get('best', -1) >= 0 else
          f"{'best':>10}: not scored yet")
    print(f"{'lr':>10}: {st['lr']:g}, KL target {st['kl_target']:g}"
          + (f", {st['rollbacks']} rollback(s)" if st.get('rollbacks') else ""))
    print(f"\n  {'lesson':>18} {'success':>8} {'walls':>6}   levels")
    for k, nm in ((k, v['name']) for k, v in LESSONS.items()):
        h = c['gate'].get(str(k), [])
        rate = f"{sum(h) / len(h):.0%}" if h else "-"
        on = k in c['active']
        lv = ", ".join(f"{p} {v:g}" for p, v in c['levels'][str(k)].items())
        print(f"  {k} {nm:>16} {rate if on else 'locked':>8} {len(h) if on else '':>6}   {lv if on else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='climber')
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--every', type=float, default=30.0)
    a = ap.parse_args()
    while True:
        print(f"\n=== {datetime.now():%H:%M:%S} ===")
        try:
            show(a.name)
        except SystemExit as e:
            print(e)
        if not a.watch:
            return
        time.sleep(a.every)


if __name__ == '__main__':
    main()
