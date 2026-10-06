"""Reference-state initialisation for family 3 (rl.walls): a share of its walls start partway
up a climb that is known to work, so the policy meets the end of a climb before it can find
the start of one.

    python -m rl.rsi            # rebuild rl/fixtures/rsi_family3.npz from the climbs, and check it

Why (2026-10-06): gpu2 resumed onto family 3's rows standing out (35M steps, 7792 episodes,
39 % of its steps) and never once touched the row above. Not the resumed policy: on the
same walls in MJX a fresh SB3 policy, uniform random actions and gpu2's own noise all
touched it in 0 of 32 episodes, their throats getting no nearer than 191-195 mm of a goal
283 mm off. A climb is 15-20 s of swing, reach out and up, seat, follow -- a chain nothing
random strings together -- and with no success there is nothing for PPO to follow.

The climbs are joint-setpoint sequences found by CEM (out/climb_demo/plan_chain.py) on the
robot MJX trains -- the simplified one, `XML=` that robot's scene -- each on one wall:
the start row and one row SPACING up and `walls.OUT` out, full length, goals straight
above the hang. Replayed here through ClimbTask.step on that robot, every EVERY-th control
step's state becomes a start, up to a second before the climb succeeds: the joints and
their speeds, the setpoints, and the holds each hand has really taken by then (success
asks each hand for one). Family 3's rows are the same full-length rungs whatever its
goals and mirroring, so a start fits every family-3 wall of its spacing.

Those episodes are booked as family 3 + 100 (rl.train_ppo.Walls): they never count toward
family 3's gate or its ADR probes, which stay what the policy does from the hang.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, 'fixtures', 'rsi_family3.npz')
# spacing -> the found climb's setpoints (ctrl0, then one row per control step)
CLIMBS = {s: os.path.join(HERE, 'fixtures', f'climb_f3_s{round(s * 100)}.npz')
          for s in (0.20, 0.25, 0.30)}
SHARE = 0.25         # of the bank's family-3 walls, while family 3 trains
                     # ponytail: a fixed share; fade it as family 3's own rate rises if
                     # the policy comes to lean on the starts
EVERY = 5            # control steps between two starts along a climb
MARGIN = 25          # control steps (1 s) before the climb's success: the last start
BOOK = 100           # its episodes report family + BOOK


def wall(spacing):
    """the wall a climb was found on, as a rl.walls layout: the start row and one row
    `spacing` up and OUT out, both full length, goals straight above the hang"""
    from .walls import _row, _front, X, Y0, Y1, Z0, HY, OUT, TIMEOUT
    z, x = Z0 + spacing, X - OUT
    rungs = [_row(Y0, Y1, Z0), _row(Y0, Y1, z, x=x)]
    return dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
                walls=[_front()], goals=np.array([[x, HY, z], [x, -HY, z]]), goal_rungs=[1],
                timeout=TIMEOUT, family=3, probe=None, probe_val=0.0)


def load():
    """{spacing, qpos, qvel, ctrl, grabs}, one row a start; None before `python -m rl.rsi`"""
    if not os.path.exists(FILE):
        return None
    z = np.load(FILE)
    return {k: z[k] for k in z.files}


def mix(layouts, rng, top, share=SHARE):
    """`layouts` (rl.walls.draw's) with `share` of the family-3 ones started partway up a
    climb: a start whose spacing is at most family 3's `top`, on a family-3 wall of that
    spacing (its own goals and mirroring), booked as family 3 + BOOK and probing nothing"""
    ref = load()
    if ref is None:
        return layouts
    from . import walls as W
    from .lessons import LESSONS
    L = len(LESSONS)                     # rl.mjx_task.LESSON_KEYS: the starts follow them
    ok = np.nonzero(ref['spacing'] <= top + 1e-9)[0]
    out = []
    for lay in layouts:
        if lay.get('family') == 3 and len(ok) and rng.random() < share:
            j = int(rng.choice(ok))
            lay = dict(W.sample(3, dict(rows=1, spacing=float(ref['spacing'][j])), rng, 'spacing'),
                       family=3 + BOOK, probe=None, probe_val=0.0, d0_row=L + j)
        out.append(lay)
    return out


def _replay(spacing, path):
    """the climb through ClimbTask.step on the simplified robot -> (states, success step)"""
    import tempfile
    from . import lessons
    from .lessons import _ry
    from .climb_task import ClimbTask, ACT_SCALE
    from .mjx_scene import lesson_layout, build
    lay = wall(spacing)
    (_, c1, _, _, _) = lay['rungs'][1]
    saved = lessons.LESSONS[3]
    lessons.LESSONS[3] = dict(saved, rungs=[_ry('start', -0.90, 1.50, 1.00),
                                            _ry('up1', -0.90, 1.50, float(c1[2]), x=float(c1[0]))],
                              goal_rungs=['rung_up1'], goals=lay['goals'].tolist(), timeout=90.0)
    try:
        ll = lesson_layout(3)
        ll['lesson'] = 3
        with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False) as f:
            f.write(build(ll).to_xml())
        t = ClimbTask(3, visual=False, xml=f.name)
        t.rng = np.random.default_rng(0)
        t.reset()
    finally:
        lessons.LESSONS[3] = saved
        os.remove(f.name)
    z = np.load(path)
    assert np.allclose(t.ctrl, z['ctrl0'], atol=1e-6), "the climb does not start from this task's hang"
    d, out, won = t.env.data, [], None
    for k, c in enumerate(z['ctrl']):
        out.append(dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=t.ctrl.copy(),
                        grabs=t.grabs.copy()))
        _, _, te, tr, info = t.step(np.clip((c - t.ctrl) / ACT_SCALE, -1, 1))
        if te or tr:
            won = k + 1 if info.get('reason') == 'success' else None
            assert won, f"the {spacing} m climb ended '{info.get('reason')}' on replay"
            break
    assert won, f"the {spacing} m climb did not succeed on replay"
    return out, won


def build_fixture():
    rows = []
    for s, path in sorted(CLIMBS.items()):
        states, won = _replay(s, path)
        keep = states[:won - MARGIN:EVERY]
        rows += [dict(st, spacing=s) for st in keep]
        print(f"rsi: the {s:.2f} m climb succeeds at step {won} ({won * 0.04:.1f} s); "
              f"{len(keep)} starts, every {EVERY} steps to {MARGIN} before it; "
              f"holds taken by the last: {keep[-1]['grabs']}")
    np.savez_compressed(FILE, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"rsi: {len(rows)} starts -> {FILE}")


if __name__ == '__main__':
    build_fixture()
    z = load()
    assert z['qpos'].shape[0] == z['spacing'].shape[0] and np.isfinite(z['qpos']).all()
    assert z['grabs'].max() >= 1, "no start has a hold taken: the starts never reach the row above"
    print("rsi: OK")
