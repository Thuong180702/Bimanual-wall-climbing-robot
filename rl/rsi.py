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

The climbs are joint-setpoint sequences found by CEM (out/climb_demo/plan_chain.py, HILL=2:
PLAN=f3x20 seed 42; f3x25w seed 43 and f3x30w seed 47 warm-started from it, INIT=) on the
robot MJX trains -- the simplified one, `XML=` that robot's scene -- each on one wall:
the start row and one row SPACING up and `walls.OUT` out, full length, goals straight
above the hang. Replayed here through ClimbTask.step on that robot, every EVERY-th control
step's state becomes a start, up to a second before the climb succeeds: the joints and
their speeds, the setpoints, and the holds each hand has really taken by then (success
asks each hand for one). Family 3's rows are the same full-length rungs whatever its
goals and mirroring, so a start fits every family-3 wall of its spacing.

None may climb more than HILL_MAX of live potential on the way (the barrier test, run on the
motion itself, `build_fixture`): the first climbs, found without HILL, swung the free arm
across the body up 0.40-2.44 m of it, and gpu3 (2026-10-06, 60M steps) learnt every start
after the swing (38/40) and none on it (0/80) -- going back to the start rung paid more.

Those episodes are booked as family 3 + 100 (rl.train_ppo.Walls): they never count toward
family 3's gate or its ADR probes, which stay what the policy does from the hang.

Which start a wall gets is drawn toward each climb's FRONTIER -- the latest start along it
the policy does not yet finish from (`frontier`, from Walls' books per start) -- and not
evenly: gpu4 (2026-10-07, 64M steps) finished every 0.20 m start from 5.0 s on, the hook
already over the row above, and none before, at 50M as at 64M; the start at 4.8 s, the
hook 30 mm under that row and rising, had seen ~128 episodes in the whole run (~2 per 1M
steps) and the policy pulled it straight into the rung's underside every time. Three
control steps of the climb from there and it finished 8/8; the climb from there with the
policy's own noise on every joint, 4/8. A start one step back from what is learnt is
learnable; drawn 1 time in 66 it is never learnt. FOCUS of the starts now go to the BACK + 1
starts ending at the frontier, the rest evenly, so what is learnt is still visited.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, 'fixtures', 'rsi_family3.npz')
# spacing -> the found climb's setpoints (ctrl0, then one row per control step)
CLIMBS = {s: os.path.join(HERE, 'fixtures', f'climb_f3_s{round(s * 100)}.npz')
          for s in (0.20, 0.25, 0.30)}
SHARE = 0.25         # of the bank's family-3 walls once family 3 has passed its gate...
SHARE_HI = 0.75      # ...and until then. From the hang family 3 teaches nothing until the
                     # starts' frontier reaches it: gpu4 gave it 44 % of all steps for 0 %,
                     # the starts 6 %. A quarter stays from the hang, for its gate and ADR.
TIMEOUT = 30.0       # s an episode from a start may run: twice the longest rest of a climb
                     # (14.2 s, 0.20 m from the hang). A start not finished by then is not
                     # going to be; gpu4's failed starts idled out the family's 60 s
FOCUS = 0.75         # of the starts drawn: at each climb's frontier...
BACK = 2             # ...the frontier start and the BACK before it, evenly
START_WINDOW = 20    # a start's book: its last episodes...
START_MIN = 10       # ...at least this many before it can count as passed...
PASS = 0.7           # ...at this rate (train_ppo.UNLOCK_AT, the families' gate)
EVERY = 5            # control steps between two starts along a climb
MARGIN = 25          # control steps (1 s) before the climb's success: the last start
BOOK = 100           # its episodes report family + BOOK
HILL_MAX = 0.05      # m the live potential (both hands, what K_REACH pays) may rise above
                     # its lowest so far along a climb. The first climbs swung the free arm
                     # across the body up a 0.40-2.44 m hill of it before the reach paid;
                     # gpu3 (60M steps) never learnt that stretch from these starts -- going
                     # back to the start rung paid more -- and stayed at 0 % from the hang.


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


def passed(h):
    """a start's book (its episodes' successes, oldest first): passed?"""
    h = h[-START_WINDOW:]
    return len(h) >= START_MIN and sum(h) >= PASS * len(h)


def frontier(books, spacing, ok=None):
    """{spacing: the latest start along that climb (its row in the fixture) not passed in
    `books` ({row: [success, ...]}), None once every start has}, over the starts `ok`.
    The latest, not the earliest: one that slips behind the frontier is drawn first again"""
    ok = np.arange(len(spacing)) if ok is None else np.asarray(ok)
    out = {}
    for s in np.unique(spacing[ok]):
        bad = [int(j) for j in ok[spacing[ok] == s] if not passed(books.get(int(j), []))]
        out[float(s)] = max(bad) if bad else None
    return out


def where(books, top):
    """{spacing: s into its climb of `frontier`'s start, None once every start has passed},
    over the climbs family 3's `top` lets in -- for the log and rl.status"""
    ref = load()
    if ref is None:
        return {}
    sp = ref['spacing']
    ok = np.nonzero(sp <= top + 1e-9)[0]
    return {s: None if f is None else round((f - int(np.argmax(sp == s))) * EVERY * 0.04, 2)
            for s, f in frontier(books, sp, ok).items()}     # 0.04 s: climb_task.CONTROL_DT


def weights(books, spacing, ok):
    """draw probabilities over the starts `ok`: FOCUS shared by the climbs' frontiers, each
    over its BACK + 1 starts ending there, and the rest evenly over every start"""
    ok = np.asarray(ok)
    w = np.full(len(ok), (1.0 - FOCUS) / len(ok))
    fr = [f for f in frontier(books, spacing, ok).values() if f is not None]
    for f in fr:
        win = (spacing[ok] == spacing[f]) & (ok <= f) & (ok >= f - BACK)
        w[win] += FOCUS / len(fr) / win.sum()
    return w / w.sum()


def mix(layouts, rng, top, share=SHARE, books=None):
    """`layouts` (rl.walls.draw's) with `share` of the family-3 ones started partway up a
    climb: a start whose spacing is at most family 3's `top`, drawn toward the frontiers
    in `books` (`weights`), on a family-3 wall of that spacing (its own goals and
    mirroring) cut to TIMEOUT, booked as family 3 + BOOK and probing nothing"""
    ref = load()
    if ref is None:
        return layouts
    from . import walls as W
    from .lessons import LESSONS
    L = len(LESSONS)                     # rl.mjx_task.LESSON_KEYS: the starts follow them
    ok = np.nonzero(ref['spacing'] <= top + 1e-9)[0]
    p = weights(books or {}, ref['spacing'], ok) if len(ok) else None
    out = []
    for lay in layouts:
        if lay.get('family') == 3 and len(ok) and rng.random() < share:
            j = int(rng.choice(ok, p=p))
            lay = dict(W.sample(3, dict(rows=1, spacing=float(ref['spacing'][j])), rng, 'spacing'),
                       family=3 + BOOK, probe=None, probe_val=0.0, d0_row=L + j, timeout=TIMEOUT)
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
    cmin, hill = float(t._live().sum()), 0.0
    for k, c in enumerate(z['ctrl']):
        out.append(dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=t.ctrl.copy(),
                        grabs=t.grabs.copy()))
        _, _, te, tr, info = t.step(np.clip((c - t.ctrl) / ACT_SCALE, -1, 1))
        hill = max(hill, info['live'] - cmin)
        cmin = min(cmin, info['live'])
        if te or tr:
            won = k + 1 if info.get('reason') == 'success' else None
            assert won, f"the {spacing} m climb ended '{info.get('reason')}' on replay"
            break
    assert won, f"the {spacing} m climb did not succeed on replay"
    return out, won, hill


def build_fixture():
    rows = []
    for s, path in sorted(CLIMBS.items()):
        states, won, hill = _replay(s, path)
        keep = states[:won - MARGIN:EVERY]
        rows += [dict(st, spacing=s) for st in keep]
        print(f"rsi: the {s:.2f} m climb succeeds at step {won} ({won * 0.04:.1f} s); "
              f"{len(keep)} starts, every {EVERY} steps to {MARGIN} before it; "
              f"holds taken by the last: {keep[-1]['grabs']}; live potential's hill {hill * 1000:.0f} mm")
        assert hill <= HILL_MAX, (f"the {s:.2f} m climb climbs {hill:.2f} m of live potential before "
                                  f"it pays (HILL_MAX {HILL_MAX}): a policy is paid to go back from its starts")
    np.savez_compressed(FILE, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"rsi: {len(rows)} starts -> {FILE}")


if __name__ == '__main__':
    build_fixture()
    z = load()
    assert z['qpos'].shape[0] == z['spacing'].shape[0] and np.isfinite(z['qpos']).all()
    assert z['grabs'].max() >= 1, "no start has a hold taken: the starts never reach the row above"
    print("rsi: OK")
