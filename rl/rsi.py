"""Reference-state initialisation for family 3 (rl.walls): a share of its walls start partway
up a climb that is known to work, so the policy meets the end of a climb before it can find
the start of one.

    python -m rl.rsi                    # rebuild rl/fixtures/rsi_family3.npz from the climbs, and check it
    python -m rl.rsi --teach out/RUN    # record the TAUGHT climbs with RUN's policy first

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
above the hang. Replayed here through ClimbTask.step on that robot, every control step's
state becomes a row, up to a second before the climb succeeds, and every EVERY of them a
start, booked as one and begun from any of its rows (`mix`): the joints and
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

A start booked fewer than START_MIN times is drawn as one at the frontier. Drawn evenly, cold
books (a resume, a climb ADR lets in) put the frontier at the climb's last start, and it
walked down through every start the policy already finishes from, BACK + 1 of them per
TIMEOUT lag: for gpu4, 41 of them, 13-25M steps simulated before it reached 4.8 s; 6-9M now.

Where the books say the policy does not finish, rl.train_ppo.Clone also clones the climb's own
action at every row (`action`, `unpassed`): gpu6 (2026-10-10, 61M steps) finished every row
from 4.88 s and none before, 0/64 at 4.84 s, the catch there 7-8 sigma of its noise away.

Two rows (`TAUGHT`, `teach`): gpu7 (2026-10-10, 173M steps) climbed one row 24/24 and two
0/24, stopped on the middle row by the start row in its view (see `teach`). Those climbs are
the policy's own, recorded with the rows below both its hands out of its observation, and
join the found ones as climbs of their own: starts, books and frontier per climb, and
`mix` draws each on a family-3 wall of its rows and spacing, offered by spacing alone (`allowed`).
What Clone clones on them is the teacher's mean action at each state, not the step it took.

A start's episodes begin from any of its EVERY control steps, not its first only: gpu4's
policy finished 0/192 from 4.8 s, the first step of a start, and every one from 5.0 s, the
next start's -- a cliff with nothing between to learn from, however often it is drawn.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, 'fixtures', 'rsi_family3.npz')
# spacing -> the found climb's setpoints (ctrl0, then one row per control step)
CLIMBS = {s: os.path.join(HERE, 'fixtures', f'climb_f3_s{round(s * 100)}.npz')
          for s in (0.20, 0.25, 0.30)}
# (spacing, rows) -> a 2-row climb from the hang a trained policy made (`teach`): its states and
# actions, recorded in MJX, the policy's own climb up the first row and a reference for the second
TAUGHT = {(s, 2): os.path.join(HERE, 'fixtures', f'climb_f3x2_s{round(s * 100)}.npz')
          for s in (0.20, 0.25)}
SHARE = 0.25         # of the bank's family-3 walls once family 3 has passed its gate and every
SHARE_HI = 0.75      # start its tops let in has, and until then. From the hang family 3 teaches
                     # nothing until the starts' frontier reaches it: gpu4 gave it 44 % of all
                     # steps for 0 %, the starts 6 %. A quarter stays from the hang, for its
                     # gate and ADR. (gpu7 passed the gate on one row with two rows at 0 %.)
TIMEOUT = 30.0       # s an episode from a start may run: twice the longest rest of a climb
                     # (14.2 s, 0.20 m from the hang). A start not finished by then is not
                     # going to be; gpu4's failed starts idled out the family's 60 s
FOCUS = 0.75         # of the starts drawn: at each climb's frontier...
BACK = 2             # ...the frontier start and the BACK before it, evenly
START_WINDOW = 20    # a start's book: its last episodes...
START_MIN = 10       # ...at least this many before it can count as passed...
PASS = 0.7           # ...at this rate (train_ppo.UNLOCK_AT, the families' gate)
EVERY = 5            # control steps a start spans along a climb (rows: one a control step)
MARGIN = 25          # control steps (1 s) before the climb's success: the last start
BOOK = 100           # its episodes report family + BOOK
SEEN = 0.08          # m: a row is a start's only if each hook is seen at most this far from
                     # where it is when an episode begins there (`_seen`; rl.mjx_check task holds
                     # every start under 0.1 on noise draws of its own, 2 cm of them above this).
                     # The found climbs see 72 mm at most; the taught ones, fast, reach for a
                     # rung with one hook loaded off its seat or none (0.25 m x2: 130-180 mm
                     # off for 0.5 s), where a climb from the hang carries what it knew and a
                     # start knows nothing -- gpu5 learnt to finish on such a view (2026-10-09)
HILL_MAX = 0.05      # m the live potential (both hands, what K_REACH pays) may rise above
                     # its lowest so far along a climb. The first climbs swung the free arm
                     # across the body up a 0.40-2.44 m hill of it before the reach paid;
                     # gpu3 (60M steps) never learnt that stretch from these starts -- going
                     # back to the start rung paid more -- and stayed at 0 % from the hang.


def wall(spacing, rows=1):
    """the wall a climb was found on, as a rl.walls layout: the start row and `rows` rows
    `spacing` up and OUT out each, all full length, goals on the top one straight above the
    hang -- rl.walls' family 3 with d = 0"""
    from .walls import _row, _front, X, Y0, Y1, Z0, HY, OUT, TIMEOUT
    z, x = Z0 + rows * spacing, X - rows * OUT
    rungs = [_row(Y0, Y1, Z0 + k * spacing, x=X - k * OUT) for k in range(rows + 1)]
    return dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
                walls=[_front()], goals=np.array([[x, HY, z], [x, -HY, z]]), goal_rungs=[rows],
                timeout=TIMEOUT, family=3, probe=None, probe_val=0.0)


def teach(run, spacings=tuple(s for s, _ in TAUGHT), tries=64, seed=0):
    """record each TAUGHT climb, rl.walls' family 3 with 2 rows from the hang, as a trained
    policy (`run`) does it when it is shown no rung below the row both its hands hold.

    Why (gpu7, 2026-10-10, 173M steps): it climbed one row 24/24 and two 0/24, every time both
    hooks on the middle row to the timeout. From a hang there, the start row in its near-rung
    observation alone stopped it -- 0/16 as it is, 8-11/16 with that row out of the
    observation (still there in the physics), and 10/16 -> 0/16 with one added to the
    observation of a wall that has none; reordering the rungs did not help. In every climb it
    had finished, nothing was below. Shown the climb it makes without that row as a reference
    (rl.train_ppo.Clone), it is taught to make it with the row in view.
    What is cloned at each of its states is the teacher's MEAN action there (`target`), not the
    step it took: a sampled climb's steps are 4 sigma of noise each, and cloned they took gpu7's
    family 1 from 13/16 to 5/16 offline (out/gpu7_diag/bc_probe2b.log).
    -> {spacing: the attempts' successes}; the fastest success of each saved to TAUGHT, the
    mean's own if it has one"""
    import pickle
    import jax
    import jax.numpy as jp
    from stable_baselines3 import PPO
    from . import mjx_task as MT

    class Teacher(MT.MjxClimb):
        def _obs(self, T, st, sen, m=None):
            m = self._map(st)
            under = jp.arange(m.rung_valid.shape[0]) < st.hold_idx.min()    # rows below both hands
            return super()._obs(T, st, sen, m._replace(rung_valid=m.rung_valid & ~under))

    model = PPO.load(os.path.join(run, 'model.zip'), device='cpu')
    with open(os.path.join(run, 'vecnorm.pkl'), 'rb') as f:
        vn = pickle.load(f)
    L, out = len(MT.LESSON_KEYS), {}
    for s in spacings:
        b = MT.Batch(tries, bank=tries, task=Teacher(tries))
        b.set_bank([dict(wall(s, 2), family=3, probe=None, probe_val=0.0)] * tries)
        b.state, o = b._reset(b.task.tables, jp.asarray(L + np.arange(tries)),
                              jax.random.split(jax.random.PRNGKey(seed), tries))
        o, det = np.asarray(o), np.arange(tries) % 2 == 0          # half the mean, half sampled
        rec, won, done = {k: [] for k in ('qpos', 'qvel', 'ctrl', 'grabs', 'action', 'target')}, {}, np.zeros(tries, bool)
        for n in range(int(1.01 * wall(s, 2)['timeout'] / 0.04)):
            st = b.state
            mean = model.predict(vn.normalize_obs(o), deterministic=True)[0]
            a = np.where(det[:, None], mean, model.predict(vn.normalize_obs(o), deterministic=False)[0])
            for k, v in (('qpos', st.d.qpos), ('qvel', st.d.qvel), ('ctrl', st.ctrl), ('grabs', st.grabs),
                         ('action', a), ('target', mean)):
                rec[k].append(np.asarray(v))
            o, _, _, te, tr, info = b.step(a)
            o, end = np.asarray(o), np.asarray(te | tr) & ~done
            for i in np.nonzero(end)[0]:
                if MT.REASONS[int(info['reason'][i])] == 'success':
                    won[int(i)] = n + 1
            done |= end
            if done.all():
                break
        out[s] = won
        print(f"rsi.teach: {s:.2f} m, 2 rows: {len(won)}/{tries} succeed "
              f"({sum(i % 2 == 0 for i in won)}/{tries // 2} the mean), steps {sorted(won.values())}")
        if won:                          # the fastest of the mean's, else of any: a sampled climb's
            mine = {i: v for i, v in won.items() if det[i]} or won       # steps carry its noise
            i = min(mine, key=mine.get)
            np.savez_compressed(TAUGHT[(s, 2)], won=won[i], run=run,
                                **{k: np.stack(v)[:won[i], i] for k, v in rec.items()})
            print(f"rsi.teach: -> {TAUGHT[(s, 2)]}, {won[i]} steps ({won[i] * 0.04:.1f} s)")
    return out


def load():
    """{spacing, rows, qpos, qvel, ctrl, grabs, action, start, climb, startable}, one row a
    control step, `start` the one it is booked as, `action` the climb's own step from it and
    `startable` whether an episode may begin there (SEEN); start_climb, one
    a start; climb_spacing and climb_rows, one a climb. None before
    `python -m rl.rsi`"""
    if not os.path.exists(FILE):
        return None
    z = np.load(FILE)
    return {k: z[k] for k in z.files}


def label(ref, c):
    """climb `c`'s name in the log: '0.2 m', or '0.2 m x2' over two rows"""
    s, r = float(ref['climb_spacing'][c]), int(ref['climb_rows'][c])
    return f"{s:g} m" + (f" x{r}" if r > 1 else '')


def allowed(ref, top):
    """the starts of the climbs family 3's tops (`top`: {'spacing': .., ...}) let in, each with a
    row an episode may begin from (`startable`). By spacing only: a climb over more rows is
    offered as soon as its spacing is, so the policy meets ADR's next row ready -- gpu7 met it
    at 28M with nothing to learn it from, and once ADR may step a count back (rl.walls.draw), a
    top of one row would take the references away in the middle of learning them"""
    c, some = ref['start_climb'], np.bincount(ref['start'], weights=ref['startable']) > 0
    return np.nonzero((ref['climb_spacing'][c] <= top['spacing'] + 1e-9) & some)[0]


def unpassed(books, ref, top):
    """the startable fixture rows of every start not yet passed in `books`, over the climbs
    family 3's tops let in: where the policy does not finish the climb (rl.train_ppo.Clone)"""
    bad = [int(j) for j in allowed(ref, top) if not passed(books.get(int(j), []))]
    return np.nonzero(np.isin(ref['start'], bad) & ref['startable'])[0]


def passed(h):
    """a start's book (its episodes' successes, oldest first): passed?"""
    h = h[-START_WINDOW:]
    return len(h) >= START_MIN and sum(h) >= PASS * len(h)


def frontier(books, climb, ok=None):
    """{climb: the latest start along it (its index) not passed in `books` ({start: [success,
    ...]}), None once every start has}, over the starts `ok`; `climb`: each start's.
    The latest, not the earliest: one that slips behind the frontier is drawn first again"""
    ok = np.arange(len(climb)) if ok is None else np.asarray(ok)
    out = {}
    for c in np.unique(climb[ok]):
        bad = [int(j) for j in ok[climb[ok] == c] if not passed(books.get(int(j), []))]
        out[int(c)] = max(bad) if bad else None
    return out


def where(books, top):
    """{climb's label: s into it of `frontier`'s start, None once every start has passed},
    over the climbs family 3's tops let in -- for the log and rl.status"""
    ref = load()
    if ref is None:
        return {}
    sc = ref['start_climb']
    return {label(ref, c): None if f is None else round((f - int(np.argmax(sc == c))) * EVERY * 0.04, 2)
            for c, f in frontier(books, sc, allowed(ref, top)).items()}   # 0.04 s: climb_task.CONTROL_DT


def weights(books, climb, ok):
    """draw probabilities over the starts `ok`: FOCUS evenly over the starts to learn from
    -- each climb's BACK + 1 ending at its frontier, and every start booked fewer than
    START_MIN times -- and the rest evenly over every start"""
    ok = np.asarray(ok)
    hot = np.array([len(books.get(int(j), [])) < START_MIN for j in ok])
    for f in frontier(books, climb, ok).values():
        if f is not None:
            hot |= (climb[ok] == climb[f]) & (ok <= f) & (ok >= f - BACK)
    w = np.full(len(ok), (1.0 - FOCUS) / len(ok))
    w[hot] += FOCUS / max(hot.sum(), 1)
    return w / w.sum()


def mix(layouts, rng, top, share=SHARE, books=None):
    """`layouts` (rl.walls.draw's) with `share` of the family-3 ones started partway up a
    climb: a start of a climb family 3's spacing (`top`) lets in, drawn toward the frontiers in
    `books` (`weights`), from any of its rows, on a family-3 wall of that climb's rows and
    spacing (its own goals and mirroring) cut to TIMEOUT, booked as family 3 + BOOK and
    probing nothing"""
    ref = load()
    if ref is None:
        return layouts
    from . import walls as W
    from .lessons import LESSONS
    L = len(LESSONS)                     # rl.mjx_task.LESSON_KEYS: the rows follow them
    sc, ok = ref['start_climb'], allowed(ref, top)
    p = weights(books or {}, sc, ok) if len(ok) else None
    out = []
    for lay in layouts:
        if lay.get('family') == 3 and len(ok) and rng.random() < share:
            j = int(rng.choice(ok, p=p))
            row = int(rng.choice(np.nonzero((ref['start'] == j) & ref['startable'])[0]))
            c = sc[j]
            lay = dict(W.sample(3, dict(rows=int(ref['climb_rows'][c]), spacing=float(ref['climb_spacing'][c])),
                                rng, ('rows', 'spacing')),
                       family=3 + BOOK, probe=None, probe_val=0.0, d0_row=L + row, timeout=TIMEOUT)
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
        a = np.clip((c - t.ctrl) / ACT_SCALE, -1, 1)
        out.append(dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=t.ctrl.copy(),
                        grabs=t.grabs.copy(), action=a))
        _, _, te, tr, info = t.step(a)
        hill = max(hill, info['live'] - cmin)
        cmin = min(cmin, info['live'])
        if te or tr:
            won = k + 1 if info.get('reason') == 'success' else None
            assert won, f"the {spacing} m climb ended '{info.get('reason')}' on replay"
            break
    assert won, f"the {spacing} m climb did not succeed on replay"
    return out, won, hill


def _seen(z):
    """each row's farther hook from its estimate at an episode's first sense there, on its
    climb's wall (rl.mjx_task's start: the estimator from the hooks' holds and the torso)"""
    import jax
    import jax.numpy as jp
    from . import mjx_task as MT
    n, L, out = 256, len(MT.LESSON_KEYS), []
    b = MT.Batch(n, bank=n)
    thr = jax.jit(jax.vmap(b.task._throat_true))
    walls = [wall(float(s), int(r)) for s, r in zip(z['climb_spacing'], z['climb_rows'])]
    js = np.arange(len(z['climb']))
    for i in range(0, len(js), n):
        part = js[i:i + n]
        pad = np.r_[part, np.full(n - len(part), part[-1])]
        b.set_bank([dict(walls[z['climb'][j]], family=3 + BOOK, d0_row=L + int(j)) for j in pad])
        st, _ = b._reset(b.task.tables, jp.asarray(L + np.arange(n)), jax.random.split(jax.random.PRNGKey(i), n))
        out.append(np.linalg.norm(np.asarray(st.est_thr) - np.asarray(thr(st.d)), axis=2).max(1)[:len(part)])
    return np.concatenate(out)


def build_fixture():
    rows, j0, climbs = [], 0, []
    for s, path in sorted(CLIMBS.items()):
        states, won, hill = _replay(s, path)
        keep = states[:won - MARGIN]
        rows += [dict(st, spacing=s, rows=1, start=j0 + k // EVERY, climb=len(climbs)) for k, st in enumerate(keep)]
        climbs.append((s, 1))
        j0 = rows[-1]['start'] + 1
        print(f"rsi: the {s:.2f} m climb succeeds at step {won} ({won * 0.04:.1f} s); "
              f"{len(keep)} rows to {MARGIN} steps before it, a start every {EVERY}; "
              f"holds taken by the last: {keep[-1]['grabs']}; live potential's hill {hill * 1000:.0f} mm")
        assert hill <= HILL_MAX, (f"the {s:.2f} m climb climbs {hill:.2f} m of live potential before "
                                  f"it pays (HILL_MAX {HILL_MAX}): a policy is paid to go back from its starts")
    # after the found ones, so their starts keep their numbers (a run's books are by start).
    # Recorded in MJX by the policy itself, no replay; no hill test: the policy's own climb
    for (s, r), path in sorted(TAUGHT.items()):
        if not os.path.exists(path):
            print(f"rsi: no {path} (python -m rl.rsi --teach RUN): the {s:.2f} m climb over {r} rows left out")
            continue
        z = np.load(path)
        won = int(z['won'])
        rows += [dict(qpos=z['qpos'][k], qvel=z['qvel'][k], ctrl=z['ctrl'][k], grabs=z['grabs'][k],
                      action=z['target'][k], spacing=s, rows=r, start=j0 + k // EVERY, climb=len(climbs))
                 for k in range(won - MARGIN)]          # `action`: the teacher's mean there (`teach`)
        climbs.append((s, r))
        j0 = rows[-1]['start'] + 1
        print(f"rsi: the {s:.2f} m climb over {r} rows ({z['run']}) succeeds at step {won} ({won * 0.04:.1f} s); "
              f"{won - MARGIN} rows; holds taken by the last: {z['grabs'][won - MARGIN - 1]}")
    z = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    first = np.unique(z['start'], return_index=True)[1]
    z['start_climb'] = z['climb'][first]
    z['climb_spacing'], z['climb_rows'] = np.array([c[0] for c in climbs]), np.array([c[1] for c in climbs])
    err = _seen(z)
    z['startable'] = err < SEEN
    for c in range(len(climbs)):
        e = err[z['climb'] == c]
        print(f"rsi: the {label(z, c)} climb: hooks seen at most {e.max() * 1000:.0f} mm off at a start; "
              f"{int((e >= SEEN).sum())} of its {len(e)} rows past SEEN, no episode begins there")
    np.savez_compressed(FILE, **z)
    print(f"rsi: {len(rows)} rows, {j0} starts, {len(climbs)} climbs -> {FILE}")


if __name__ == '__main__':
    import sys
    if '--teach' in sys.argv:            # python -m rl.rsi --teach out/<run>: TAUGHT first
        teach(sys.argv[sys.argv.index('--teach') + 1])
    build_fixture()
    from .climb_task import ACT_SCALE
    z = load()
    assert z['qpos'].shape[0] == z['spacing'].shape[0] == z['start'].shape[0] == z['action'].shape[0] \
        and np.isfinite(z['qpos']).all()
    assert (np.diff(z['start']) >= 0).all() and np.bincount(z['start']).max() == EVERY
    assert (z['start_climb'][z['start']] == z['climb']).all(), "a start spans two climbs"
    assert z['grabs'].max() >= 1, "no start has a hold taken: the starts never reach the row above"
    assert (np.bincount(z['start'], weights=z['startable']) > 0).mean() > 0.9, "most starts have no row to begin from"
    for c in range(len(CLIMBS)):            # a found climb's action is the step to its next row
        j = np.nonzero(z['climb'] == c)[0]
        err = np.abs(z['ctrl'][j[1:]] - z['ctrl'][j[:-1]] - ACT_SCALE * z['action'][j[:-1]]).max()
        assert err < 1e-9, f"climb {label(z, c)}: its actions are not its rows' steps ({err:g})"
    assert np.isfinite(z['action']).all() and np.abs(z['action']).max() <= 1.0, "an action outside [-1, 1]"
    print("rsi: OK")
