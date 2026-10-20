"""Reference climbs: episodes begun partway up a climb known to work, and the climb's own
action cloned where the policy does not yet finish it.

    python -m rl.rsi            # rebuild rl/fixtures/rsi.npz from the climbs (REFS), and check it

Some families are a long chain of moves -- swing, reach out and up, seat, follow -- that
PPO's noise never strings together from the hang, and with no success there is nothing to
follow: families 3, 6, 8 and 9 each sat at 0 % for 50-90M steps from the hang alone, and
fresh policies and uniform random actions never touched the row above either. So a share
of each such family's walls (`share`) begin partway up a reference climb, and the policy
meets the end of a climb before it has to find its start: reference state initialisation
(Peng et al. 2018, DeepMimic) run as a reverse curriculum (Florensa et al. 2017).

The climbs (REFS) are FOUND -- joint setpoints from CEM (rl.plan) on the simplified robot
MJX trains, replayed here through ClimbTask.step -- or TAUGHT -- a trained policy's own
climb, recorded in MJX (rl.plan record), for the moves a search did not find or found only
one way round. A climb's states up to MARGIN steps before its success are its rows, and
every EVERY rows are one START.

Which start a wall gets is drawn toward each climb's FRONTIER: the latest start along it
the policy does not yet finish from (`frontier`). FOCUS of the draws go to the frontier
and the `back` starts either side of it, the rest evenly over every start, so what is
learnt is still visited. A start one step back from what is learnt is learnable; one drawn
1 time in 66 was never learnt.

At the starts in that window the policy has not passed, its mean action is also pulled
toward the climb's own (rl.train_ppo.Clone; DAPG, Rajeswaran et al. 2018): a catch can
lie 7-8 sigma of the policy's noise away, out of reach however long it runs.

Episodes from a start report family + BOOK: they never count toward the family's gate or
its ADR probes, which stay what the policy does from the hang.

Every climb is checked as the fixture is built:
  * it climbs at most HILL_MAX of live potential (what K_REACH pays) on the way: a policy
    does not learn a stretch of a climb that going back would pay more for. A swing past
    the static reach is its own hill, a pendulum of 3.5-4.4 m (HILL_SWING);
  * a row is `startable` only where each hook is seen at most SEEN from where it is as an
    episode begins there, whatever the sensor noise (the estimator starts from the holds and
    the torso).
"""
import hashlib, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, 'fixtures', 'rsi.npz')

# (family, height, side, rows, file in rl/fixtures). height: the value of the family's TOP
# parameter the climb is on; side 1 the wall as rl.walls draws it, -1 mirrored, 0 either (a
# row of full length: any wall of the family with those rows and that height fits).
REFS = [
    (3, 0.20, 0, 1, 'climb_f3_s20'), (3, 0.25, 0, 1, 'climb_f3_s25'), (3, 0.30, 0, 1, 'climb_f3_s30'),
    (3, 0.20, 0, 2, 'climb_f3x2_s20'), (3, 0.25, 0, 2, 'climb_f3x2_s25'),    # taught, the rows
    (3, 0.35, 0, 2, 'climb_f3x2_s35'), (3, 0.40, 0, 2, 'climb_f3x2_s40'),    # below hidden first
    (4, 0.20, 0, 1, 'climb_f4_r20'), (4, 0.30, 0, 1, 'climb_f4_r30'), (4, 0.40, 0, 1, 'climb_f4_r40'),
    (4, 0.40, 0, 2, 'climb_f4x2_r40'),
    (5, 0.30, 1, 1, 'climb_f5_d30'), (5, 0.30, -1, 1, 'climb_f5m_d30'),      # taught
    (6, 0.30, 1, 1, 'climb_f6_r30'), (6, 0.30, -1, 1, 'climb_f6m_r30'),
    (6, 0.40, 1, 1, 'climb_f6_r40'), (6, 0.40, -1, 1, 'climb_f6m_r40'),
    (7, 0.60, 1, 1, 'climb_f7_s60'), (7, 0.60, -1, 1, 'climb_f7m_s60'),      # taught
    *[(8, b, 1, 1, f'climb_f8_b{round(b * 100)}') for b in (0.35, 0.50, 0.55, 0.60, 0.70)],
    *[(9, g, s, 1, f'climb_f9{"m" if s < 0 else ""}_g{round(g * 100)}')
      for g in (0.55, 0.60, 0.70, 0.80, 0.90) for s in (1, -1)],
]
TOP = {3: 'spacing', 4: 'spacing', 5: 'drop', 6: 'rise', 7: 'side', 8: 'behind', 9: 'gap'}

SHARE = (0.75, 0.25)  # of a family's walls begun at a start: until its gate is first passed --
                      # from the hang it teaches nothing until the frontier reaches it -- and
                      # after, when three in four go back to the hang: given harder climbs and
                      # fewer hangs, families forgot the walls they had passed
SLOW = (6, 8, 9)      # families whose hang runs to the 60 s timeout until they are learnt...
SHARE_SLOW = 0.95     # ...begin this share at a start until then. At 0.75 a quarter of their walls
                      # idle out 60 s from the hang: 63 % of gpu10's family-6 steps, and over half of
                      # gpu12's family-8 steps (75 hangs a 1M steps, ~1500 steps each), where the
                      # mean then learnt its own stalled reach. gpu11, which learnt 6 and 8, ran 0.95
FOCUS = 0.75          # of the starts drawn: in each climb's frontier window
BACK = 2              # starts each side of the frontier, at the least: `back`
START_WINDOW = 20     # a start's book: its last episodes...
START_MIN = 10        # ...at least this many before it can count as passed...
PASS = 0.7            # ...at this rate (rl.train_ppo.UNLOCK_AT, the families' gate)
SLIP = 0.5            # ...and holds the frontier only under this: a start learnt at 0.85 reads
                      # under PASS over 20 episodes 7 % of the time, and with 100+ starts past a
                      # climb's cliff one always did -- the frontier sat on it, not at the cliff
FAIL = 0.3            # ...and before those, the starts under this hold it: gpu13's lesson 6 -y had
                      # 127 starts past its cliff at 0.91 on average, one under SLIP at a read 35 %
                      # of the time (under FAIL 0.3 %), and its frontier sat off the cliff at 39
                      # of 77 checkpoints from 230M to 307M
EVERY = 5             # control steps a start spans along a climb
MARGIN = 5            # control steps (0.2 s) before the climb's success: its last start
BOOK = 100            # an episode from a start reports family + BOOK
STALL = (9,)          # families whose episodes from a start end 'stalled' (truncated, as a
STALL_S = 20.0        # timeout) once the live potential has made no new low by STALL_D in
STALL_D = 0.05        # STALL_S s: gpu13's missed lesson-9 throws hung one-handed to the 60 s
                      # timeout, 6.6k of 20.6k episodes from its starts at 475-531M, ~72 % of
                      # their steps. Along the climbs themselves the longest such stretch is
                      # 7.6 s (f9 0.55-0.6), 13.1 s (f9 0.7 -y), measured on their rows in C
SEEN = 0.08           # m: the farther hook's estimate off where it is, at most, as an episode
                      # begins at a row, on every one of SEEN_DRAWS sensor-noise draws (`_seen`):
SEEN_DRAWS = 16       # the error's tail is long: on one draw, or the worst of 4, rows of f8 0.70
                      # read 116-125 mm on the next
HILL_MAX = 0.05       # m the live potential may rise above its lowest so far along a climb
HILL_SWING = 5.0      # ...and family 9's swings past STATIC9: the pendulum is the hill (+1.6 m
STATIC9 = 0.65        # and +2.5 m at lesson 9's), paid back within ~6 s, the climb ends +16.4


def ref_wall(f, h, side=1, rows=1):
    """the wall a climb of family f at height h was found or taught on, as an rl.walls layout:
    the family's wall with its draws at the middle of their ranges -- goals straight above or
    below the hang on families 3, 4 and 8 -- mirrored for side -1"""
    from .walls import _front, _row, mirror, FACE, HY, OUT, R, SEAT, TIMEOUT, X, Y0, Y1, Z0
    walls, hang = [_front()], 1
    if f in (3, 4):
        up = 1 if f == 3 else -1
        rr = [_row(Y0, Y1, Z0 + up * k * h, x=X - up * k * OUT) for k in range(rows + 1)]
        x, z = X - up * rows * OUT, Z0 + up * rows * h
        goals, goal_rungs = [[x, HY, z], [x, -HY, z]], [rows]
        if f == 4:
            walls, hang = [_front(x + FACE)], 4
    elif f in (5, 9):                     # one step `h` down, 0.10 m on (5); a gap `h` (9)
        e = HY + 0.21
        y0, z = (e + 0.10, Z0 - h) if f == 5 else (e + h, Z0)
        yR = y0 + SEAT + 0.03
        yL = yR + 0.31
        rr = [_row(-0.90 if f == 5 else -0.52, e, Z0), _row(y0, yL + SEAT + 0.08, z)]
        goals, goal_rungs = [[X, yL, z], [X, yR, z]], [1]
    elif f == 6:
        z = Z0 + h
        rr = [_row(-0.90, 0.37, Z0), _row(0.47, 0.87, z), _row(1.12, 1.47, z)]
        goals, goal_rungs = [[X, 1.295, z], [X, 0.67, z]], [1, 2]
    elif f == 7:
        ys = -h + 0.06
        rr = [_row(-(h - 0.10), 1.2, Z0), (0.125, np.array([0.125, ys, Z0]), np.array([1.0, 0.0, 0.0]), 0.305)]
        walls.append(('wall_side', np.array([0.125, -h, 1.0]), np.array([0.0, 1.0, 0.0])))
        goals, goal_rungs = [[0.35, ys, Z0], [0.04, ys, Z0]], [1]
    elif f == 8:
        xb = X - h
        rr = [_row(Y0, Y1, Z0), _row(Y0, Y1, Z0, x=xb)]
        walls.append(('wall_back', np.array([xb - FACE, 0.30, 1.0]), np.array([1.0, 0.0, 0.0])))
        goals, goal_rungs = [[xb, HY, Z0], [xb, -HY, Z0]], [1]
    lay = dict(rungs=[(f'rung_g{i}', c, a, hh, R) for i, (_, c, a, hh) in enumerate(rr)], walls=walls,
               goals=np.array(goals, float), goal_rungs=goal_rungs, timeout=TIMEOUT, family=f,
               hang=hang, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def load():
    """the fixture: one entry a row (qpos, qvel, ctrl, grabs, action -- the climb's own step
    from it --, start, climb, startable), a start (start_climb) and a climb (climb_family,
    climb_height, climb_side, climb_rows, climb_found, climb_length in s). None before
    `python -m rl.rsi`"""
    if not os.path.exists(FILE):
        return None
    z = np.load(FILE)
    return {k: z[k] for k in z.files}


def signature(ref):
    """the fixture's starts and their states, hashed: a run's books are by start number, and
    hold only for the fixture they were kept on"""
    return hashlib.sha1(b''.join(np.ascontiguousarray(ref[k]).tobytes()
                                 for k in ('start', 'climb', 'qpos'))).hexdigest()[:16]


def label(ref, c):
    """climb c's name: 'f3 0.2', 'f3 0.2 x2' over two rows, 'f6 0.3 -y' mirrored"""
    r, s = int(ref['climb_rows'][c]), int(ref['climb_side'][c])
    return (f"f{int(ref['climb_family'][c])} {float(ref['climb_height'][c]):g}"
            + (f" x{r}" if r > 1 else '') + (' -y' if s < 0 else ''))


def climb_wall(ref, c):
    """the wall climb c was found or taught on"""
    return ref_wall(int(ref['climb_family'][c]), float(ref['climb_height'][c]),
                    int(ref['climb_side'][c]) or 1, int(ref['climb_rows'][c]))


def timeout(ref, f):
    """s an episode from one of family f's starts may run: twice its longest climb and 10 s
    more, at most the walls' own -- a start not finished by then is not going to be"""
    from .walls import TIMEOUT
    n = ref['climb_family'] == f
    return min(TIMEOUT, 2.0 * float(ref['climb_length'][n].max()) + 10.0) if n.any() else TIMEOUT


def back(ref, c):
    """starts each side of climb c's frontier in its window: a start a second of every five
    of the climb. A book is read once every episode begun before it can have ended, a
    timeout later, so the frontier moves at most back + 1 starts a timeout: a 37 s climb's
    185 starts at 2 a lag took 0.2 s of climb per 2.6-5.4M steps"""
    return max(BACK, int(round(float(ref['climb_length'][c]) / 5.0)))


def start_wall(ref, c, rng):
    """a wall an episode from one of climb c's starts runs on: family 3's or 4's any wall of its
    rows and height (they are full length: any goals and mirroring fit), the others their own"""
    from . import walls as W
    f, h, r = int(ref['climb_family'][c]), float(ref['climb_height'][c]), int(ref['climb_rows'][c])
    lay = (W.sample(f, {'rows': r, 'spacing': h}, rng, {'rows': r, 'spacing': h}) if f in (3, 4)
           else climb_wall(ref, c))
    return dict(lay, family=f + BOOK, probe=None, probe_val=0.0, timeout=timeout(ref, f))


def allowed(ref, levels):
    """the starts of the climbs `levels` ({family: its levels}, the families in the pool) let
    in, each with a row an episode may begin from: a family's climbs no higher than the
    candidate of its TOP parameter (rl.walls.candidates; its level once that is the last), so
    the policy meets its next level with the climbs that teach it"""
    from .walls import candidates
    lim = {f: candidates(f, lv).get(TOP[f], lv[TOP[f]]) for f, lv in levels.items() if f in TOP}
    fam, c = ref['climb_family'], ref['start_climb']
    top = np.array([lim.get(int(f), -np.inf) for f in fam])[c]
    some = np.bincount(ref['start'], weights=ref['startable'], minlength=len(c)) > 0
    return np.nonzero((ref['climb_height'][c] <= top + 1e-9) & some)[0]


def passed(h):
    """a start's book (its episodes' successes, oldest first): passed?"""
    h = h[-START_WINDOW:]
    return len(h) >= START_MIN and sum(h) >= PASS * len(h)


def slipped(h, under=SLIP):
    """a start's book: holding the frontier -- booked fewer than START_MIN times, or under SLIP.
    Between SLIP and PASS a start is not passed, and does not hold the frontier"""
    h = h[-START_WINDOW:]
    return len(h) < START_MIN or sum(h) < under * len(h)


def frontier(books, ref, ok):
    """{climb: the latest of its starts `ok` under FAIL in `books` ({start: [success, ...]}), or
    with none under FAIL the latest slipped; None once none is}. The latest, not the earliest:
    one that slips behind it is drawn first again"""
    ok, sc = np.asarray(ok), ref['start_climb']
    out = {}
    for c in np.unique(sc[ok]):
        hs = [(int(j), books.get(int(j), [])) for j in ok[sc[ok] == c]]
        bad = [j for j, h in hs if slipped(h, FAIL)] or [j for j, h in hs if slipped(h)]
        out[int(c)] = max(bad) if bad else None
    return out


def _window(books, ref, ok):
    """the starts `ok` in their climb's frontier window: from `back` before the frontier to
    `back` after it, those after it not yet passed"""
    ok, sc = np.asarray(ok), ref['start_climb']
    learnt = np.array([passed(books.get(int(j), [])) for j in ok], bool)
    hot = np.zeros(len(ok), bool)
    for c, f in frontier(books, ref, ok).items():
        if f is not None:
            b = back(ref, c)
            hot |= (sc[ok] == c) & (ok >= f - b) & ((ok <= f) | ((ok <= f + b) & ~learnt))
    return hot, learnt


def weights(books, ref, ok):
    """draw probabilities over the starts `ok`: FOCUS evenly over each climb's frontier window
    and every start booked fewer than START_MIN times, the rest evenly over every start"""
    hot, _ = _window(books, ref, ok)
    hot |= np.array([len(books.get(int(j), [])) < START_MIN for j in ok], bool)
    w = np.full(len(ok), (1.0 - FOCUS) / len(ok))
    w[hot] += FOCUS / max(hot.sum(), 1)
    return w / w.sum()


def cloned(books, ref, levels):
    """the startable rows of the starts in each climb's frontier window not yet passed: where
    the policy learns its climbs now (rl.train_ppo.Clone). Every start not passed was too many:
    a family just let in has all of them -- 2786 rows, 2.8 sigma off the policy's mean -- and
    in 12M steps cloning them took four families down 20-90 points to a rollback"""
    ok = allowed(ref, levels)
    hot, learnt = _window(books, ref, ok)
    return np.nonzero(np.isin(ref['start'], ok[hot & ~learnt]) & ref['startable'])[0]


def where(books, ref, levels):
    """{climb's label: s into it of its frontier start, None once every start has passed}"""
    sc, dt = ref['start_climb'], EVERY * 0.04                  # 0.04 s: climb_task.CONTROL_DT
    return {label(ref, c): None if f is None else round((f - int(np.argmax(sc == c))) * dt, 2)
            for c, f in frontier(books, ref, allowed(ref, levels)).items()}


def window(books, ref, levels):
    """{climb's label: [[s into it, successes, episodes] of each start in its frontier window,
    from its book]}: whether the starts at a frontier are being learnt or not, between two
    snapshots (`where` says only which one is the latest failing)"""
    ok, sc, dt = allowed(ref, levels), ref['start_climb'], EVERY * 0.04
    hot, _ = _window(books, ref, ok)
    out = {}
    for j in ok[hot]:
        c, h = int(sc[j]), books.get(int(j), [])[-START_WINDOW:]
        out.setdefault(label(ref, c), []).append(
            [round((int(j) - int(np.argmax(sc == c))) * dt, 2), int(sum(h)), len(h)])
    return out


def mix(layouts, rng, levels, share, books, ref):
    """`layouts` (rl.walls.draw's) with share[f] of each family f's walls begun at a start of one
    of its climbs `levels` let in, drawn toward the frontiers in `books` (`weights`), from any of
    its rows, on a wall it fits (`start_wall`)"""
    from .lessons import LESSONS
    L = len(LESSONS)                     # rl.mjx_task.LESSON_KEYS: the starts' rows follow them
    sc, ok = ref['start_climb'], allowed(ref, levels)
    fam = ref['climb_family'][sc[ok]]
    ok = {int(f): ok[fam == f] for f in np.unique(fam)}
    p = {f: weights(books, ref, o) for f, o in ok.items()}
    out = []
    for lay in layouts:
        f = lay.get('family')
        if f in ok and rng.random() < share.get(f, 0.0):
            j = int(rng.choice(ok[f], p=p[f]))
            row = int(rng.choice(np.nonzero((ref['start'] == j) & ref['startable'])[0]))
            lay = dict(start_wall(ref, sc[j], rng), d0_row=L + row)
        out.append(lay)
    return out


# --- building the fixture ----------------------------------------------------------
def replay(lay, path):
    """a found climb (its setpoints) through ClimbTask.step on the simplified robot, on its
    wall -> (states, success step, live potential's hill)"""
    from .climb_task import ClimbTask, ACT_SCALE
    from .mjx_scene import c_scene
    xml, spec = c_scene(dict(lay, timeout=90.0))
    try:
        t = ClimbTask(spec=spec, xml=xml, visual=False)
        t.rng = np.random.default_rng(0)
        t.reset()
    finally:
        os.remove(xml)
    z = np.load(path)
    assert np.allclose(t.ctrl, z['ctrl0'], atol=1e-6), f"{path} does not start from this wall's hang"
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
            assert won, f"{path} ended '{info.get('reason')}' on replay"
            break
    assert won, f"{path} did not succeed on replay"
    return out, won, hill


def _seen(z):
    """each row's farther hook from its estimate at an episode's first sense there, on its
    climb's wall, the worst of SEEN_DRAWS noise draws (rl.mjx_task's start: the estimator from
    the hooks' holds and the torso)"""
    import jax
    import jax.numpy as jp
    from . import mjx_task as MT
    n, L, out = 256, len(MT.LESSON_KEYS), []
    b = MT.Batch(n, bank=n)
    thr = jax.jit(jax.vmap(b.task._throat_true))
    walls = [dict(climb_wall(z, c), family=int(z['climb_family'][c]) + BOOK)
             for c in range(len(z['climb_family']))]
    js = np.arange(len(z['climb']))
    for i in range(0, len(js), n):
        part = js[i:i + n]
        pad = np.r_[part, np.full(n - len(part), part[-1])]
        b.set_bank([dict(walls[z['climb'][j]], d0_row=L + int(j)) for j in pad])
        err = 0.0
        for k in range(SEEN_DRAWS):
            key = jax.random.PRNGKey(i + 1_000_003 * k)
            st, _ = b._reset(b.task.tables, jp.asarray(L + np.arange(n)), jax.random.split(key, n))
            err = np.maximum(err, np.linalg.norm(np.asarray(st.est_thr) - np.asarray(thr(st.d)), axis=2).max(1))
        out.append(err[:len(part)])
    return np.concatenate(out)


def build():
    from .climb_task import CONTROL_DT
    rows, climbs, j0 = [], [], 0
    for f, h, side, nrows, name in REFS:
        path = os.path.join(HERE, 'fixtures', name + '.npz')
        z = np.load(path)
        found = 'ctrl0' in z.files
        if found:
            states, won, hill = replay(ref_wall(f, h, side or 1, nrows), path)
            limit = HILL_SWING if f == 9 and h > STATIC9 else HILL_MAX
            assert hill <= limit, (f"{name} climbs {hill:.2f} m of live potential before it pays "
                                   f"(limit {limit}): a policy is paid to go back from its starts")
            how = f"found, live potential's hill {hill * 1000:.0f} mm"
        else:                           # recorded in MJX by the policy: its mean action the reference
            won = int(z['won'])
            states = [dict(qpos=z['qpos'][k], qvel=z['qvel'][k], ctrl=z['ctrl'][k], grabs=z['grabs'][k],
                           action=z['target'][k]) for k in range(won)]
            how = f"taught ({z['run']})"
        keep = states[:won - MARGIN]
        rows += [dict(st, start=j0 + k // EVERY, climb=len(climbs)) for k, st in enumerate(keep)]
        climbs.append((f, h, side, nrows, found, won * CONTROL_DT))
        j0 = rows[-1]['start'] + 1
        print(f"rsi: {name}: succeeds at {won * CONTROL_DT:.1f} s, {len(keep)} rows, {how}")
    z = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    z['start_climb'] = z['climb'][np.unique(z['start'], return_index=True)[1]]
    for i, k in enumerate(('family', 'height', 'side', 'rows', 'found', 'length')):
        z['climb_' + k] = np.array([c[i] for c in climbs])
    # on disk first: _seen's batch (rl.mjx_task's tables) begins its episodes from the FILE's rows
    z['startable'] = np.ones(len(rows), bool)
    np.savez_compressed(FILE, **z)
    err = _seen(z)
    z['startable'] = err < SEEN
    for c in range(len(climbs)):
        e = err[z['climb'] == c]
        print(f"rsi: {label(z, c)}: hooks seen at most {e.max() * 1000:.0f} mm off at a start; "
              f"{int((e >= SEEN).sum())} of its {len(e)} rows past SEEN")
    np.savez_compressed(FILE, **z)
    print(f"rsi: {len(rows)} rows, {j0} starts, {len(climbs)} climbs -> {FILE}")
    return z


def check(z):
    from .climb_task import ACT_SCALE
    assert z['qpos'].shape[0] == z['start'].shape[0] == z['action'].shape[0] and np.isfinite(z['qpos']).all()
    assert (np.diff(z['start']) >= 0).all() and np.bincount(z['start']).max() == EVERY
    assert (z['start_climb'][z['start']] == z['climb']).all(), "a start spans two climbs"
    assert z['grabs'].max() >= 1, "no start has a hold taken"
    assert (np.bincount(z['start'], weights=z['startable']) > 0).mean() > 0.9, "most starts have no row to begin from"
    for c in np.nonzero(z['climb_found'])[0]:                     # a found climb's action is
        j = np.nonzero(z['climb'] == c)[0]                        # the step to its next row
        err = np.abs(z['ctrl'][j[1:]] - z['ctrl'][j[:-1]] - ACT_SCALE * z['action'][j[:-1]]).max()
        assert err < 1e-9, f"climb {label(z, c)}: its actions are not its rows' steps ({err:g})"
    assert np.isfinite(z['action']).all() and np.abs(z['action']).max() <= 1.0, "an action outside [-1, 1]"
    # the frontier logic on a toy book: every start passed but one, which holds the window
    ok = np.nonzero(z['start_climb'] == 0)[0]
    books = {int(j): [True] * START_WINDOW for j in ok}
    books[int(ok[len(ok) // 2])] = [False] * START_WINDOW
    assert frontier(books, z, ok) == {0: int(ok[len(ok) // 2])}
    w = weights(books, z, ok)
    assert w[len(ok) // 2] > 10 * w[0], "the frontier start is not drawn first"
    books[int(ok[-3])] = [True, False] * (START_WINDOW // 2)       # a learnt start's bad read
    books[int(ok[-3])][:3] = [False] * 3                          # (0.4) past a cliff at 0 %
    assert frontier(books, z, ok) == {0: int(ok[len(ok) // 2])}, "noise past the cliff holds the frontier"
    books[int(ok[len(ok) // 2])] = [True] * START_WINDOW           # with the cliff passed, it does
    assert frontier(books, z, ok) == {0: int(ok[-3])}
    from .walls import first_levels                               # what a checkpoint records there
    rec = window(books, z, {3: first_levels()[3]})[label(z, 0)]
    t = round((int(ok[-3]) - int(ok[0])) * EVERY * 0.04, 2)
    assert [t, sum(books[int(ok[-3])]), START_WINDOW] in rec and all(n <= START_WINDOW for *_, n in rec), rec
    print("rsi: OK")


if __name__ == '__main__':
    check(build())
