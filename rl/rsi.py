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

Family 6 too (`STEPS6`, 2026-10-11): gpu8 (162M steps) passed families 1-4 and stayed at
0 % on 5 for 50M steps and on 6 for 43M, the same in both: the policy goes to the start
row's end and hangs there, its hook never more than 2-22 cm up (the rung above 30-50 cm).
Not the reward (lifting there is downhill all the way, 0.0 mm of hill) nor the map (the
lidar has every rung from the first step): family 3's rows stand OUT from the wall, and the
diagonal step up onto a rung in the wall's own plane, past the end of the one below, was
never taught -- family 3 in gpu1-gpu6 again. A family-6 climb is found on one wall
(`wall6`) and fits only that one, each way round (`side`), so an episode from its start is
on that wall, mirrored for the climb found the other way, booked as family 6 + BOOK.

Family 4 (`TAUGHT4`, 2026-10-13): gpu9 (ea972f3, 179M steps) sat at 45 % on family 4 from 51M
on, ADR never off its first tops, because it only stepped down toward -y: on 24 walls every
one toward +y (the benchmark's own way) timed out, no hook ever more than 6 cm down, and
10/11 toward -y succeeded in 3-7 s. gpu7 sat the same way for 90M steps; gpu8 got out, and
at 162M stepped down both ways (+y 12/13, -y 9/11). Which side a policy learns first is
chance; that it learns the other is not given. Each way round, gpu8's own step down on one
wall (`wall4`), recorded as the two-row climbs are (`teach4`), booked as family 4 + BOOK.

BACK6 (2026-10-13): a start's episode is booked only once every episode begun before it can
have ended, its family's timeout later -- family 6's 60 s, 6.2M steps at 4096 envs -- so the
frontier moves at most BACK + 1 starts per lag. gpu9's family-6 frontier, at BACK 2, moved
0.2 s of climb per 2.6M steps (-y) and per 5.4M (+y), the -y at ~80 % of that cap, and its
policy at 179M finished from 29.0 s 70 % and 30.0 s 90 % (-y; the books: 30 %, 25 %) and
from 30.0 s 77-80 % (+y; books 15 %), where the books still put the frontiers at 30.4 s and
33.0 s. Family 6's climbs are ~37 s, 187 starts each; family 3's and 4's, 4-15 s and 30 s
(3.1M steps) a lag, keep BACK: spread three times wider their hardest start, the one the
frontier waits at, would be drawn a third as often (gpu4's: 1 time in 66 was never learnt).
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
# (rise, side) -> family 6's found climb onto one tier `rise` up (`wall6`), side 1 toward +y
# (PLAN=f6), -1 toward -y (PLAN=f6m): setpoints as CLIMBS'
STEPS6 = {(r, side): os.path.join(HERE, 'fixtures', f'climb_f6{"m" if side < 0 else ""}_r{round(r * 100)}.npz')
          for r in (0.30, 0.40) for side in (1, -1)}
# (drop, side) -> family 4's step down `drop` (`wall4`), side 1 toward +y, a trained policy's own
# (`teach4`): recorded as TAUGHT's
TAUGHT4 = {(d, side): os.path.join(HERE, 'fixtures', f'climb_f4{"m" if side < 0 else ""}_d{round(d * 100)}.npz')
           for d in (0.30,) for side in (1, -1)}
GAP4 = 0.10          # m on past the start row's end that `wall4`'s step starts: family 4's first top
TOP = {3: 'spacing', 4: 'drop', 6: 'rise'}   # the top a family's climbs are offered by (`allowed`): their height
SHARE = 0.25         # of the bank's family-3 walls once family 3 has passed its gate and every
SHARE_HI = 0.75      # start its tops let in has, and until then. From the hang family 3 teaches
                     # nothing until the starts' frontier reaches it: gpu4 gave it 44 % of all
                     # steps for 0 %, the starts 6 %. A quarter stays from the hang, for its
                     # gate and ADR. (gpu7 passed the gate on one row with two rows at 0 %.)
TIMEOUT = 30.0       # s an episode from a start may run: twice the longest rest of a climb
                     # (14.2 s, 0.20 m from the hang). A start not finished by then is not
                     # going to be; gpu4's failed starts idled out the family's 60 s.
TIMEOUT6 = 60.0      # ...family 6's (`timeout`): its found climbs take ~40 s from the hang
FOCUS = 0.75         # of the starts drawn: at each climb's frontier...
BACK = 2             # ...the frontier start and the BACK before it, evenly: BACK + 1 a lag
BACK6 = 8            # ...family 6's (above)
START_WINDOW = 20    # a start's book: its last episodes...
START_MIN = 10       # ...at least this many before it can count as passed...
PASS = 0.7           # ...at this rate (train_ppo.UNLOCK_AT, the families' gate)
EVERY = 5            # control steps a start spans along a climb (rows: one a control step)
MARGIN = 25          # control steps (1 s) before the climb's success: the last start
MARGIN6 = 5          # ...family 6's (0.2 s): its last hand's catch and seat are its last second.
                     # gpu9 (122M) finished the -y climb from 34.2 s, its catch inside the rows,
                     # and the +y one 45 % from its last start: L in the air 0.42 m from its goal
                     # there, the catch ~0.6 s on -- no start, nothing cloned. Its mean reached
                     # t1b on time from 33-35 s, slipped off and went back to t1a, 0/9.
                     # Family 4's too: a step down is ~5-8 s, its second hand's seat its last second
BOOK = 100          # its episodes report family + BOOK
SEEN = 0.08          # m: a row is a start's only if each hook is seen at most this far from
                     # where it is when an episode begins there (`_seen`; rl.mjx_check task holds
                     # every start under 0.1 on noise draws of its own, 2 cm of them above this).
                     # The found climbs see 78 mm at most (family 6's 78, since a start has
                     # its hooks' rungs end to end: MjxClimb.start); the taught ones, fast, reach for a
                     # rung with one hook loaded off its seat or none (0.25 m x2: up to 187 mm
                     # off for 0.2 s), where a climb from the hang carries what it knew and a
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


def wall6(rise, side=1):
    """the wall family 6's climbs were found on (out/climb_demo/plan_chain.py, PLAN=f6): the
    start row ending 0.37 m out, one tier `rise` up of two short rungs, the first 0.47-0.87 m,
    the second 1.12-1.47 m, goals on their middles (R the first's, L the second's) -- rl.walls'
    family 6, one tier, scatter 0 -- mirrored (rl.walls.mirror) for `side` -1"""
    from .walls import _row, _front, mirror, X, Z0
    z = Z0 + rise
    rungs = [_row(-0.90, 0.37, Z0), _row(0.47, 0.87, z), _row(1.12, 1.47, z)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front()], goals=np.array([[X, 1.295, z], [X, 0.67, z]]), goal_rungs=[1, 2],
               timeout=TIMEOUT6, family=6, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def wall4(drop, side=1):
    """the wall family 4's step down is taught on (`teach4`): rl.walls' family 4 with one step
    `drop` down, GAP4 on past the start row's end, that end 0.21 m past the left hook and the
    goals at the middle of rl.walls' draws -- mirrored (rl.walls.mirror) for `side` -1"""
    from .walls import _row, _front, mirror, X, Z0, HY, SEAT, TIMEOUT
    y = HY + 0.21
    y0, z = y + GAP4, Z0 - drop
    yR = y0 + SEAT + 0.03
    yL = yR + 0.31
    rungs = [_row(-0.90, y, Z0), _row(y0, yL + SEAT + 0.08, z)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front()], goals=np.array([[X, yL, z], [X, yR, z]]), goal_rungs=[1],
               timeout=TIMEOUT, family=4, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def family(ref):
    """each climb's family (a fixture from before family 6's: all 3)"""
    return ref['climb_family'] if 'climb_family' in ref else np.full(len(ref['climb_spacing']), 3)


def timeout(f):
    """s an episode from one of family f's starts may run"""
    return TIMEOUT6 if f == 6 else TIMEOUT


def climb_wall(ref, c):
    """the wall climb `c` was found or taught on"""
    s = float(ref['climb_spacing'][c])
    if family(ref)[c] == 6:
        return wall6(s, int(ref['climb_side'][c]))
    if family(ref)[c] == 4:
        return wall4(s, int(ref['climb_side'][c]))
    return wall(s, int(ref['climb_rows'][c]))


def start_wall(ref, c, rng):
    """a wall an episode from one of climb c's starts runs on: family 3's any wall of its rows and
    spacing (its rows are full length: any goals and mirroring fit), family 6's its own"""
    f = int(family(ref)[c])
    if f == 3:
        from . import walls as W
        lay = W.sample(3, dict(rows=int(ref['climb_rows'][c]), spacing=float(ref['climb_spacing'][c])),
                       rng, ('rows', 'spacing'))
    else:
        lay = climb_wall(ref, c)
    return dict(lay, family=f + BOOK, probe=None, probe_val=0.0, timeout=timeout(f))


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
    import jax.numpy as jp
    from . import mjx_task as MT

    class Teacher(MT.MjxClimb):
        def _obs(self, T, st, sen, m=None):
            m = self._map(st)
            under = jp.arange(m.rung_valid.shape[0]) < st.hold_idx.min()    # rows below both hands
            return super()._obs(T, st, sen, m._replace(rung_valid=m.rung_valid & ~under))

    return {s: _record(run, Teacher(tries), dict(wall(s, 2), family=3, probe=None, probe_val=0.0),
                       tries, seed, TAUGHT[(s, 2)], f"{s:.2f} m, 2 rows") for s in spacings}


def teach4(run, tries=32, seed=0):
    """record each TAUGHT4 step down, rl.walls' family 4 from the hang on `wall4`, as a trained
    policy (`run`, one that steps down both ways: gpu8's) does it -> {(drop, side): successes}"""
    from . import mjx_task as MT
    return {k: _record(run, MT.MjxClimb(tries), wall4(*k), tries, seed, path,
                       f"family 4's {k[0]:.2f} m step down{' the other way' if k[1] < 0 else ''}")
            for k, path in TAUGHT4.items()}


def _record(run, task, lay, tries, seed, path, what):
    """`tries` episodes of `run`'s policy on the wall `lay` from the hang on `task`, half its mean
    and half sampled, every state recorded -> {try: steps to its success}; the fastest of the
    mean's (else of any: a sampled climb's steps carry its noise) saved to `path`"""
    import pickle
    import jax
    import jax.numpy as jp
    from stable_baselines3 import PPO
    from . import mjx_task as MT
    model = PPO.load(os.path.join(run, 'model.zip'), device='cpu')
    with open(os.path.join(run, 'vecnorm.pkl'), 'rb') as f:
        vn = pickle.load(f)
    L = len(MT.LESSON_KEYS)
    b = MT.Batch(tries, bank=tries, task=task)
    b.set_bank([lay] * tries)
    b.state, o = b._reset(b.task.tables, jp.asarray(L + np.arange(tries)),
                          jax.random.split(jax.random.PRNGKey(seed), tries))
    o, det = np.asarray(o), np.arange(tries) % 2 == 0          # half the mean, half sampled
    rec, won, done = {k: [] for k in ('qpos', 'qvel', 'ctrl', 'grabs', 'action', 'target')}, {}, np.zeros(tries, bool)
    for n in range(int(1.01 * lay['timeout'] / 0.04)):
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
    print(f"rsi.teach: {what}: {len(won)}/{tries} succeed "
          f"({sum(i % 2 == 0 for i in won)}/{tries // 2} the mean), steps {sorted(won.values())}")
    if won:
        mine = {i: v for i, v in won.items() if det[i]} or won
        i = min(mine, key=mine.get)
        np.savez_compressed(path, won=won[i], run=run, **{k: np.stack(v)[:won[i], i] for k, v in rec.items()})
        print(f"rsi.teach: -> {path}, {won[i]} steps ({won[i] * 0.04:.1f} s)")
    return won


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


def signature(ref):
    """the fixture's starts and their states, hashed: a run's books are by start number, and
    hold only for the fixture they were kept on (train_ppo.Walls drops them otherwise)"""
    import hashlib
    return hashlib.sha1(b''.join(np.ascontiguousarray(ref[k]).tobytes()
                                 for k in ('start', 'climb', 'qpos'))).hexdigest()[:16]


def label(ref, c):
    """climb `c`'s name in the log: '0.2 m', or '0.2 m x2' over two rows; family 6's 'f6 0.3 m',
    'f6 0.3 m -y' the other way round (family 4's 'f4 ...' the same)"""
    s, r = float(ref['climb_spacing'][c]), int(ref['climb_rows'][c])
    if family(ref)[c] in (4, 6):
        return f"f{family(ref)[c]} {s:g} m" + (' -y' if ref['climb_side'][c] < 0 else '')
    return f"{s:g} m" + (f" x{r}" if r > 1 else '')


def allowed(ref, tops):
    """the starts of the climbs the tops `tops` ({family: its tops}, the families in the pool)
    let in, each with a row an episode may begin from (`startable`): a family's climbs no higher
    than its top (TOP). By height only: a family-3 climb over more rows is offered as soon as
    its spacing is, so the policy meets ADR's next row ready -- gpu7 met it at 28M with nothing
    to learn it from, and once ADR may step a count back (rl.walls.draw), a top of one row
    would take the references away in the middle of learning them"""
    c, some = ref['start_climb'], np.bincount(ref['start'], weights=ref['startable']) > 0
    lim = np.array([tops[f][TOP[f]] if f in tops else -np.inf for f in family(ref)])[c]
    return np.nonzero((ref['climb_spacing'][c] <= lim + 1e-9) & some)[0]


def unpassed(books, ref, tops):
    """the startable fixture rows of every start not yet passed in `books`, over the climbs
    the tops `tops` let in: where the policy does not finish the climb (rl.train_ppo.Clone)"""
    bad = [int(j) for j in allowed(ref, tops) if not passed(books.get(int(j), []))]
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


def where(books, tops):
    """{climb's label: s into it of `frontier`'s start, None once every start has passed},
    over the climbs the tops `tops` let in -- for the log and rl.status"""
    ref = load()
    if ref is None:
        return {}
    sc = ref['start_climb']
    return {label(ref, c): None if f is None else round((f - int(np.argmax(sc == c))) * EVERY * 0.04, 2)
            for c, f in frontier(books, sc, allowed(ref, tops)).items()}   # 0.04 s: climb_task.CONTROL_DT


def weights(books, climb, ok, back=BACK):
    """draw probabilities over the starts `ok`: FOCUS evenly over the starts to learn from
    -- each climb's `back` + 1 ending at its frontier, and every start booked fewer than
    START_MIN times -- and the rest evenly over every start"""
    ok = np.asarray(ok)
    hot = np.array([len(books.get(int(j), [])) < START_MIN for j in ok])
    for f in frontier(books, climb, ok).values():
        if f is not None:
            hot |= (climb[ok] == climb[f]) & (ok <= f) & (ok >= f - back)
    w = np.full(len(ok), (1.0 - FOCUS) / len(ok))
    w[hot] += FOCUS / max(hot.sum(), 1)
    return w / w.sum()


def mix(layouts, rng, tops, share=SHARE, books=None):
    """`layouts` (rl.walls.draw's) with `share` ({family: share}, or one for all) of each
    family's walls started partway up one of its climbs: a start of a climb the tops `tops`
    let in, drawn toward the frontiers in `books` (`weights`) among the family's own, from any
    of its rows, on a wall it fits (`start_wall`) cut to `timeout`, booked as family + BOOK
    and probing nothing"""
    ref = load()
    if ref is None:
        return layouts
    from .lessons import LESSONS
    L = len(LESSONS)                     # rl.mjx_task.LESSON_KEYS: the rows follow them
    sc, ok = ref['start_climb'], allowed(ref, tops)
    fam = family(ref)[sc[ok]]
    ok = {int(f): ok[fam == f] for f in np.unique(fam)}
    p = {f: weights(books or {}, sc, o, BACK6 if f == 6 else BACK) for f, o in ok.items()}
    out = []
    for lay in layouts:
        f = lay.get('family')
        if f in ok and rng.random() < (share.get(f, 0.0) if isinstance(share, dict) else share):
            j = int(rng.choice(ok[f], p=p[f]))
            row = int(rng.choice(np.nonzero((ref['start'] == j) & ref['startable'])[0]))
            lay = dict(start_wall(ref, sc[j], rng), d0_row=L + row)
        out.append(lay)
    return out


def _replay(spacing, path):
    """the family-3 climb through ClimbTask.step on the simplified robot -> (states, success step)"""
    from .lessons import _ry
    lay = wall(spacing)
    (_, c1, _, _, _) = lay['rungs'][1]
    return _replay_on(3, [_ry('start', -0.90, 1.50, 1.00), _ry('up1', -0.90, 1.50, float(c1[2]), x=float(c1[0]))],
                      ['rung_up1'], lay['goals'], path, f"the {spacing} m climb")


def _replay6(rise, side, path):
    """family 6's climb (`wall6`) through ClimbTask.step on the simplified robot, on lesson 6
    with its rungs as plan_chain.py's PLAN=f6 / f6m names them"""
    from .lessons import _ry
    lay = wall6(rise, side)
    rungs = [_ry(n, float(c[1] - h), float(c[1] + h), float(c[2]), x=float(c[0]))
             for n, (_, c, _, h, _) in zip(('start', 't1a', 't1b'), lay['rungs'])]
    return _replay_on(6, rungs, ['rung_t1a', 'rung_t1b'], lay['goals'], path,
                      f"family 6's {rise} m climb{' the other way' if side < 0 else ''}")


def _replay_on(lesson, rungs, goal_rungs, goals, path, what):
    """a found climb (`path`: its setpoints) through ClimbTask.step on the simplified robot, on
    lesson `lesson` with those rungs and goals -> (states, success step, live potential's hill)"""
    import tempfile
    from . import lessons
    from .climb_task import ClimbTask, ACT_SCALE
    from .mjx_scene import lesson_layout, build
    saved = lessons.LESSONS[lesson]
    lessons.LESSONS[lesson] = dict(saved, rungs=rungs, goal_rungs=goal_rungs,
                                   goals=np.asarray(goals).tolist(), timeout=90.0)
    try:
        ll = lesson_layout(lesson)
        ll['lesson'] = lesson
        with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False) as f:
            f.write(build(ll).to_xml())
        t = ClimbTask(lesson, visual=False, xml=f.name)
        t.rng = np.random.default_rng(0)
        t.reset()
    finally:
        lessons.LESSONS[lesson] = saved
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
            assert won, f"{what} ended '{info.get('reason')}' on replay"
            break
    assert won, f"{what} did not succeed on replay"
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
    walls = [climb_wall(z, c) for c in range(len(z['climb_spacing']))]
    fam = family(z)
    js = np.arange(len(z['climb']))
    for i in range(0, len(js), n):
        part = js[i:i + n]
        pad = np.r_[part, np.full(n - len(part), part[-1])]
        b.set_bank([dict(walls[z['climb'][j]], family=int(fam[z['climb'][j]]) + BOOK, d0_row=L + int(j)) for j in pad])
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
    # family 6's after both, the same way: family 3's starts keep their numbers
    for (rise, side), path in sorted(STEPS6.items()):
        if not os.path.exists(path):
            print(f"rsi: no {path}: family 6's {rise:.2f} m climb (side {side}) left out")
            continue
        states, won, hill = _replay6(rise, side, path)
        keep = states[:won - MARGIN6]
        rows += [dict(st, spacing=rise, rows=1, start=j0 + k // EVERY, climb=len(climbs)) for k, st in enumerate(keep)]
        climbs.append((rise, 1, 6, side))
        j0 = rows[-1]['start'] + 1
        print(f"rsi: family 6's {rise:.2f} m climb (side {side:+d}) succeeds at step {won} ({won * 0.04:.1f} s); "
              f"{len(keep)} rows; holds taken by the last: {keep[-1]['grabs']}; live potential's hill {hill * 1000:.0f} mm")
        assert hill <= HILL_MAX, (f"family 6's {rise:.2f} m climb climbs {hill:.2f} m of live potential before "
                                  f"it pays (HILL_MAX {HILL_MAX})")
    # family 4's after those, as TAUGHT's: a policy's own, its mean action the reference
    for (drop, side), path in sorted(TAUGHT4.items()):
        if not os.path.exists(path):
            print(f"rsi: no {path} (python -m rl.rsi --teach4 RUN): family 4's {drop:.2f} m step (side {side}) left out")
            continue
        z = np.load(path)
        won = int(z['won'])
        rows += [dict(qpos=z['qpos'][k], qvel=z['qvel'][k], ctrl=z['ctrl'][k], grabs=z['grabs'][k],
                      action=z['target'][k], spacing=drop, rows=1, start=j0 + k // EVERY, climb=len(climbs))
                 for k in range(won - MARGIN6)]
        climbs.append((drop, 1, 4, side))
        j0 = rows[-1]['start'] + 1
        print(f"rsi: family 4's {drop:.2f} m step (side {side:+d}, {z['run']}) succeeds at step {won} "
              f"({won * 0.04:.1f} s); {won - MARGIN6} rows; holds taken by the last: {z['grabs'][won - MARGIN6 - 1]}")
    climbs = [c if len(c) == 4 else c + (3, 0) for c in climbs]     # (height, rows, family, side)
    z = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    first = np.unique(z['start'], return_index=True)[1]
    z['start_climb'] = z['climb'][first]
    z['climb_spacing'], z['climb_rows'] = np.array([c[0] for c in climbs]), np.array([c[1] for c in climbs])
    z['climb_family'], z['climb_side'] = np.array([c[2] for c in climbs]), np.array([c[3] for c in climbs])
    # on disk first: _seen's batch (rl.mjx_task's tables) begins its episodes from the FILE's rows.
    # Measured before it was written, a row new to the fixture was measured at a stale file's row
    z['startable'] = np.ones(len(rows), bool)
    np.savez_compressed(FILE, **z)
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
    if '--teach4' in sys.argv:           # python -m rl.rsi --teach4 out/<run>: TAUGHT4 first
        teach4(sys.argv[sys.argv.index('--teach4') + 1])
    build_fixture()
    from .climb_task import ACT_SCALE
    z = load()
    assert z['qpos'].shape[0] == z['spacing'].shape[0] == z['start'].shape[0] == z['action'].shape[0] \
        and np.isfinite(z['qpos']).all()
    assert (np.diff(z['start']) >= 0).all() and np.bincount(z['start']).max() == EVERY
    assert (z['start_climb'][z['start']] == z['climb']).all(), "a start spans two climbs"
    assert z['grabs'].max() >= 1, "no start has a hold taken: the starts never reach the row above"
    assert (np.bincount(z['start'], weights=z['startable']) > 0).mean() > 0.9, "most starts have no row to begin from"
    for c in [*range(len(CLIMBS)), *np.nonzero(family(z) == 6)[0]]:   # a found climb's action is
        j = np.nonzero(z['climb'] == c)[0]                              # the step to its next row
        err = np.abs(z['ctrl'][j[1:]] - z['ctrl'][j[:-1]] - ACT_SCALE * z['action'][j[:-1]]).max()
        assert err < 1e-9, f"climb {label(z, c)}: its actions are not its rows' steps ({err:g})"
    assert np.isfinite(z['action']).all() and np.abs(z['action']).max() <= 1.0, "an action outside [-1, 1]"
    print("rsi: OK")
