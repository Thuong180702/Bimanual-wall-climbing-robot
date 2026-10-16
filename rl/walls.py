"""Procedurally generated climbing walls, and the ranges ADR grows them over.

    python -m rl.walls          # every family at its first and its last levels: usable walls,
                                # and each benchmark lesson inside its family's last range

The benchmark -- lessons 1..9 of rl.lessons -- is what a policy is scored on, on the full
robot. It is not what it trains on. Training draws walls from nine families, one per
lesson and fixed before any training: each is that lesson's structure with a few physical
parameters free (FAMILIES). Every parameter has a range [lo, last] and a LEVEL, which
starts at the easy end (`first`) and rises one step at a time by automatic domain
randomization (OpenAI et al. 2019, "Solving Rubik's Cube with a Robot Hand"; legged_gym's
terrain curriculum, Rudin et al. 2022):

  * a family's walls draw every parameter uniformly between lo and its level;
  * PROBE_P of them put one parameter exactly at its CANDIDATE, a step above its level;
  * when a window of probes at a candidate is passed at rl.train_ppo.UNLOCK_AT, the
    candidate becomes the level (rl.train_ppo.Curriculum).

So training never draws past what has been passed, and a new level reaches the walls only
once it is learnt -- a candidate is a fifth of a family's walls however hard it is.

Every wall is a layout in rl.mjx_scene.lesson_layout's format -- rungs (name, centre, axis,
half length, radius), walls (name, point on the inner face, normal), the two goals [L, R],
the goal rungs, and `hang`, the lesson whose start hang it begins from -- so the MJX slots
take it as they take a lesson. Half of them are mirrored (y -> -y, the goals swapped
between hands): the robot is symmetric, the benchmark only goes one way. An episode lasts
at most TIMEOUT: lesson 1's, at which hanging still costs less than falling (climb_task's
incentive test).
"""
import numpy as np, mujoco
from .climb_task import SEAT_IN

X, Z0, R = 0.44, 1.00, 0.015     # front rungs' axis x, the start row, rung radius
HY = 0.157                       # |y| of each hook at the start (the keyframe)
Y0, Y1 = -0.90, 1.50             # a full row, as the benchmark draws it
MIN_W = 0.25                     # m two goals on one rung keep apart (hooks 65 mm, MIN_SEP 0.12)
SEAT = SEAT_IN + 0.005           # a goal's closest to a rung end
TIMEOUT = 60.0
CLEAR = 0.02                     # m every rung but the start one keeps from the robot at the start
OUT = 0.20                       # m each row above stands further from the wall (family 3), each
                                 # row below nearer it (family 4), than the row it is climbed from
FACE = 0.06                      # m from the nearest rung's axis to its wall's face
PROBE_P = 0.25                   # of a family's walls put one parameter at its candidate

# family -> ((parameter, sampled from, first level, last level, step), ...). Integers are
# counts. The last levels are the benchmark's value or a margin past it, and the robot's
# measured reach (rl.lessons) bounds the rest.
FAMILIES = {
    1: (('shift', 0.0, 0.30, 1.40, 0.10),),                        # lesson 1: 1.10
    2: (('gaps', 1, 1, 2, 1), ('width', 0.04, 0.04, 0.30, 0.02)),  # lesson 2: 2 x 0.25
    # rows above, each OUT further from the wall: a row is at most 0.45 up -- reaching out
    # costs the free hook height (0.52 m up 0.15-0.25 m out); CEM found 0.20-0.40, not 0.45.
    # 0.05-0.10 out, the base swung onto the lowest of two rows
    3: (('rows', 1, 1, 2, 1), ('spacing', 0.20, 0.20, 0.45, 0.05)),   # lesson 3: 2 x 0.40
    4: (('rows', 1, 1, 2, 1), ('spacing', 0.20, 0.20, 0.45, 0.05)),   # lesson 4: 2 x 0.40, down
    # a staircase step stays inside the static reach down (0.65 m at 0.15 m aside)
    5: (('steps', 1, 1, 3, 1), ('drop', 0.25, 0.30, 0.45, 0.05),   # lesson 5: 3 steps,
        ('gap', 0.05, 0.10, 0.30, 0.05)),                          #   <= 0.40 down, <= 0.20 on
    6: (('rise', 0.30, 0.30, 0.50, 0.05), ('scatter', 0.0, 0.02, 0.08, 0.02)),   # lesson 6: 0.46
    7: (('side', 0.60, 0.60, 1.10, 0.05),),                        # lesson 7: 1.00
    8: (('behind', 0.35, 0.35, 0.70, 0.05),),                      # lesson 8: 0.63
    # the swing gap runs from the 0.58 m a still hang crosses to lesson 9's 0.90: the one
    # motion found across 0.90 needed all the reach there is (the catch 1.10 m from the
    # other hook) and no search from further back reached past 1.07 m
    9: (('gap', 0.30, 0.55, 0.90, 0.05),),
}

# family -> the families whose skills it is made of. A family enters the pool only once
# each of these has passed (train_ppo.Curriculum), so an upgrade never comes before what it
# upgrades, and a family stuck at 0 % holds back its upgrades and nothing else. With
# nothing stuck the families enter in their own order (`ORDER`).
NEEDS = {1: (),
         2: (1,),        # gaps in the row: a traverse that changes rungs
         3: (1,),        # rows above: a traverse's re-grip, reached upward
         4: (1,),        # rows below: the same, reached downward
         5: (2,),        # down a staircase: each step another rung across a gap
         6: (2, 3),      # up a tier of short rungs: up, and across the gap between them
         7: (1,),        # the corner: from the front row onto the side rung, level
         8: (1,),        # the wall behind: onto the back rung, level
         9: (2,)}        # a gap past the static reach: crossed by swinging


def _order(needs):
    """the order families enter with nothing stuck: each time the lowest-numbered one whose
    needs are all in (train_ppo.Curriculum's own rule)"""
    out = []
    while len(out) < len(needs):
        out.append(next(f for f in sorted(needs) if f not in out and set(needs[f]) <= set(out)))
    return out


assert set(NEEDS) == set(FAMILIES) and all(p in NEEDS for ps in NEEDS.values() for p in ps)
ORDER = _order(NEEDS)                # raises StopIteration on a cycle: no family could ever enter
assert ORDER == sorted(FAMILIES), ORDER

# where each benchmark lesson sits in its family: every value must lie inside the family's
# last range, so the benchmark is a wall training can draw
BENCHMARK = {1: dict(shift=1.10), 2: dict(gaps=2, width=0.25), 3: dict(rows=2, spacing=0.40),
             4: dict(rows=2, spacing=0.40), 5: dict(steps=3, drop=0.40, gap=0.20),
             6: dict(rise=0.46, scatter=0.04), 7: dict(side=1.00), 8: dict(behind=0.63),
             9: dict(gap=0.90)}


def spec(f):
    """family f's ranges as a checkpoint records them: its levels mean something only
    against the ranges they were grown over"""
    return repr(FAMILIES[f])


def first_levels():
    return {f: {p[0]: p[2] for p in ps} for f, ps in FAMILIES.items()}


def last_levels():
    return {f: {p[0]: p[3] for p in ps} for f, ps in FAMILIES.items()}


def candidates(f, level):
    """{parameter: the step above its level} of family f's parameters not yet at their last"""
    out = {}
    for name, lo, _, last, step in FAMILIES[f]:
        if level[name] < last:
            v = min(level[name] + step, last)
            out[name] = int(v) if isinstance(lo, int) else round(float(v), 6)
    return out


def progress(levels):
    """how far every level has come, first -> last, 0..1 a family, summed over the families"""
    return float(sum(np.mean([(levels[f][n] - a) / (b - a) for n, _, a, b, _ in ps])
                     for f, ps in FAMILIES.items()))


def _row(y0, y1, z, x=X):
    """a rung along y from y0 to y1"""
    return ((y0 + y1) / 2, np.array([x, (y0 + y1) / 2, z]), np.array([0.0, 1.0, 0.0]),
            (y1 - y0) / 2)


def _front(face=X + FACE):
    return ('wall_front', np.array([face, 0.30, 1.00]), np.array([-1.0, 0.0, 0.0]))


def _goals_from(y_lo, rng, z=Z0, x=X):
    """both goals on a rung that starts at y_lo: R just past the start, L ~0.3 m on
    -> (goals [L, R], where the rung ends)"""
    yR = y_lo + SEAT + rng.uniform(0.005, 0.06)
    yL = yR + rng.uniform(0.28, 0.34)
    return [np.array([x, yL, z]), np.array([x, yR, z])], yL + SEAT + rng.uniform(0.02, 0.15)


def sample(family, level, rng, exact=None):
    """one wall of `family`: every parameter uniform from its lo to its `level`, but those in
    `exact` ({parameter: value}) at that value -- a probe, or a reference climb's wall
    (rl.rsi) -> layout dict (rl.mjx_scene.lesson_layout's)"""
    exact = exact or {}

    def draw(name, lo, *_):
        if name in exact:
            return exact[name]
        t = level[name]
        return int(rng.integers(lo, t + 1)) if isinstance(lo, int) else float(rng.uniform(lo, t))

    p = {s[0]: draw(*s) for s in FAMILIES[family]}
    rungs, walls, hang = [], [_front()], 1
    d = rng.uniform(-0.15, 0.15)                   # the goals off centre, on a full row
    if family == 1:                                # along the start row
        dL, dR = rng.uniform(0, p['shift'], 2)
        if 'shift' in exact:                       # one hand all the way
            dL, dR = (p['shift'], dR) if rng.random() < 0.5 else (dL, p['shift'])
        dL = max(dL, dR - (2 * HY - MIN_W))        # the left hand stays left
        yL, yR = HY + dL, -HY + dR
        rungs = [_row(rng.uniform(-0.9, -0.6), yL + SEAT + rng.uniform(0.03, 1.2), Z0)]
        goals, goal_rungs = [np.array([X, yL, Z0]), np.array([X, yR, Z0])], [0]
    elif family == 2:                              # gaps in the start row
        y = HY + rng.uniform(0.12, 0.30)
        rungs = [_row(rng.uniform(-0.6, -0.45), y, Z0)]
        for i in range(p['gaps']):
            y0 = y + (p['width'] if 'width' in exact else rng.uniform(0.04, p['width']))
            if i < p['gaps'] - 1:
                y = y0 + rng.uniform(0.35, 0.60)
                rungs.append(_row(y0, y, Z0))
        goals, end = _goals_from(y0, rng)
        rungs.append(_row(y0, end, Z0))
        goal_rungs = [len(rungs) - 1]
    elif family in (3, 4):                         # rows above, each further out (3), or
        m, dz = p['rows'], p['spacing']            # below, each further in (4), to the last
        up = 1 if family == 3 else -1
        rungs = [_row(Y0, Y1, Z0 + up * k * dz, x=X - up * k * OUT) for k in range(m + 1)]
        x, z = X - up * m * OUT, Z0 + up * m * dz
        goals, goal_rungs = [np.array([x, HY + d, z]), np.array([x, -HY + d, z])], [m]
        if family == 4:                            # the wall behind the lowest row; nothing
            walls, hang = [_front(x + FACE)], 4    # to lean on, so lesson 4's free hang
    elif family == 5:                              # down a staircase: each step lower, further on
        y = HY + rng.uniform(0.12, 0.30)           # the start row ends past the left hook
        rungs, z = [_row(-0.9, y, Z0)], Z0
        for i in range(p['steps']):
            y0 = y + (p['gap'] if 'gap' in exact else rng.uniform(0.05, p['gap']))
            z -= p['drop'] if 'drop' in exact else rng.uniform(0.25, p['drop'])
            if i < p['steps'] - 1:
                y = y0 + rng.uniform(0.40, 0.60)   # room for both hooks on every step
                rungs.append(_row(y0, y, z))
        goals, end = _goals_from(y0, rng, z=z)
        rungs.append(_row(y0, end, z))
        goal_rungs = [len(rungs) - 1]
    elif family == 6:                              # a tier of two short rungs past the start
        y = HY + rng.uniform(0.12, 0.30)           # row's end, each at its own height
        rungs, z = [_row(-0.9, y, Z0)], Z0 + p['rise']
        ln, gp = rng.uniform(0.30, 0.45, 2), rng.uniform(0.20, 0.30, 1)
        y += rng.uniform(0.05, 0.20)
        for j in range(2):
            rungs.append(_row(y, y + ln[j], z + rng.uniform(-p['scatter'], p['scatter'])))
            y += ln[j] + (gp[j] if j < 1 else 0.0)
        (cR, zR), (cL, zL) = [(r[1][1], r[1][2]) for r in rungs[-2:]]
        goals = [np.array([X, cL + rng.uniform(-0.03, 0.03), zL]),
                 np.array([X, cR + rng.uniform(-0.03, 0.03), zR])]
        goal_rungs = [1, 2]
    elif family == 7:                              # the side wall, in the corner
        s = p['side']
        ys = -s + 0.06
        rungs = [_row(-(s - 0.10), rng.uniform(0.9, 1.5), Z0),
                 (0.125, np.array([0.125, ys, Z0]), np.array([1.0, 0.0, 0.0]), 0.305)]
        walls.append(('wall_side', np.array([0.125, -s, 1.0]), np.array([0.0, 1.0, 0.0])))
        goals = [np.array([0.35 + rng.uniform(-0.03, 0.03), ys, Z0]),
                 np.array([0.04 + rng.uniform(-0.03, 0.03), ys, Z0])]
        goal_rungs = [1]
    elif family == 8:                              # the wall behind the back
        xb = X - p['behind']
        rungs = [_row(Y0, Y1, Z0), _row(Y0, Y1, Z0, x=xb)]
        walls.append(('wall_back', np.array([xb - FACE, 0.30, 1.0]), np.array([1.0, 0.0, 0.0])))
        goals, goal_rungs = [np.array([xb, HY + d, Z0]), np.array([xb, -HY + d, Z0])], [1]
    elif family == 9:                              # one gap past the static reach: swing
        e = HY + rng.uniform(0.12, 0.30)
        y0 = e + p['gap']
        goals, end = _goals_from(y0, rng)
        rungs = [_row(rng.uniform(-0.6, -0.45), e, Z0), _row(y0, end, Z0)]
        goal_rungs = [1]
    rungs = [(f'rung_g{i}', c, a, h, R) for i, (_, c, a, h) in enumerate(rungs)]
    lay = dict(rungs=rungs, walls=walls, goals=np.asarray(goals, float), goal_rungs=goal_rungs,
               timeout=TIMEOUT, family=family, hang=hang, probe=None, probe_val=0.0)
    return mirror(lay) if rng.random() < 0.5 else lay


def mirror(lay):
    """y -> -y, and each hand takes the other's goal so the left one stays on the left"""
    flip = np.array([1.0, -1.0, 1.0])
    rungs = [(n, c * flip, a, h, r) for n, c, a, h, r in lay['rungs']]
    walls = [(n, c * flip, v * flip) for n, c, v in lay['walls']]
    return dict(lay, rungs=rungs, walls=walls, goals=(lay['goals'] * flip)[::-1].copy())


class Checker:
    """Is a wall usable? Its goals on seats of their rungs, and every rung but the start
    one -- and the extra wall -- at least CLEAR from each robot part it can touch, with
    the robot in the secured start hang it begins from (`hang`): nothing may be inside or
    against the robot there. And the same hang moved to the goals is CLEAR of every rung
    but the goal ones: a row straight below where the robot ends is in its arms or body.
    Checked in C on the robot MJX trains (rl.mjx_scene), through its own collision pairs."""

    def __init__(self):
        from .mjx_scene import N_RUNG, model
        self.m = model()
        self.d = mujoco.MjData(self.m)
        self.hangs = {}                            # lesson -> (its secured hang, the hooks' midpoint)
        self.rungs = [self.m.geom(f'slot_rung{i}').id for i in range(N_RUNG)]
        self.wall, self.front = self.m.geom('slot_wall').id, self.m.geom('wall_front').id
        # C MuJoCo skips a geom's own rotation when it was compiled at its body's (the
        # parked slots were), so a quat set here would be ignored; MJX never skips it
        self.m.geom_sameframe[self.rungs + [self.wall, self.front]] = 0
        pairs = {}
        for a, b in zip(self.m.pair_geom1, self.m.pair_geom2):
            pairs.setdefault(a, []).append(b)
            pairs.setdefault(b, []).append(a)
        self.pairs = pairs
        self.n_rung = N_RUNG

    def hang(self, k):
        """lesson k's secured hang on the simplified robot, and its hooks' midpoint"""
        if k not in self.hangs:
            from .climb_task import ClimbTask
            from .mjx_scene import c_path
            t = ClimbTask(k, visual=False, xml=c_path(k))
            t.reset()
            self.hangs[k] = (t.env.data.qpos.copy(),
                             np.mean([t._route(t._throat_true(s))[1] for s in 'LR'], axis=0))
        return self.hangs[k]

    def ok(self, lay):
        from .mjx_scene import slots
        rungs = lay['rungs']
        if len(rungs) > self.n_rung or len(lay['walls']) > 2:
            return False

        def seated(g, r):
            _, c, a, h, _ = rungs[r]
            s = (g - c) @ a
            return abs(s) <= h - SEAT and np.linalg.norm(g - c - s * a) < 1e-6
        # each goal on the axis of a goal rung, a hook's half-width in from its ends
        if not all(any(seated(g, r) for r in lay['goal_rungs']) for g in lay['goals']):
            return False
        sl = slots(lay)
        m = self.m
        for i, g in enumerate(self.rungs):
            m.geom_pos[g], m.geom_quat[g], m.geom_size[g] = (sl['rung_pos'][i], sl['rung_quat'][i],
                                                            sl['rung_size'][i])
            m.geom_rbound[g] = sl['rung_size'][i][0] + sl['rung_size'][i][1]
        m.geom_pos[self.wall], m.geom_quat[self.wall] = sl['wall_pos'], sl['wall_quat']
        m.geom_pos[self.front], m.geom_quat[self.front] = sl['front_pos'], sl['front_quat']
        ft = np.zeros(6)
        clear = lambda gs: not any(mujoco.mj_geomDistance(m, self.d, g, o, CLEAR, ft) < CLEAR
                                   for g in gs for o in self.pairs.get(g, []))
        q0, mid0 = self.hang(lay.get('hang', 1))
        self.d.qpos[:] = q0
        mujoco.mj_kinematics(m, self.d)
        if not clear(self.rungs[1:len(rungs)] + ([self.wall] if len(lay['walls']) > 1 else [])):
            return False
        self.d.qpos[:3] += np.mean(lay['goals'], axis=0) - mid0
        mujoco.mj_kinematics(m, self.d)
        return clear([g for i, g in enumerate(self.rungs[:len(rungs)]) if i not in lay['goal_rungs']])


def draw(checker, weights, levels, rng, n, probe_p=PROBE_P):
    """n usable walls: a family by `weights` ({family: weight}), every parameter under its
    level, and with probability probe_p one of them -- of those not yet at their last --
    exactly at its candidate (`candidates`), the layout saying which (`probe`, `probe_val`)"""
    fams = list(weights)
    w = np.asarray([weights[f] for f in fams], float)
    out = []
    while len(out) < n:
        f = fams[rng.choice(len(fams), p=w / w.sum())]
        cand = candidates(f, levels[f])
        probe = list(cand)[rng.integers(len(cand))] if cand and rng.random() < probe_p else None
        exact = {probe: cand[probe]} if probe else None
        for _ in range(50):
            lay = sample(f, levels[f], rng, exact)
            if checker.ok(lay):
                out.append(dict(lay, probe=probe, probe_val=cand[probe] if probe else 0.0))
                break
        else:
            raise RuntimeError(f"family {f} at {levels[f]} (probe {exact}): no usable wall in 50 draws")
    return out


def main():
    import time
    from .mjx_scene import lesson_layout
    rng = np.random.default_rng(0)
    ck = Checker()
    for label, levels in (('first', first_levels()), ('last', last_levels())):
        for f in FAMILIES:
            # every parameter under its level, and each one at its last value on its own
            tries = [None] + [{s[0]: s[3]} for s in FAMILIES[f]]
            n_ok = sum(ck.ok(sample(f, levels[f], rng, ex)) for _ in range(40) for ex in tries)
            tot = 40 * len(tries)
            print(f"family {f} at its {label} levels {levels[f]}: {n_ok}/{tot} walls usable")
            assert n_ok >= 0.5 * tot, f"family {f}: most walls at the {label} levels are unusable"
    for k in FAMILIES:
        lay = lesson_layout(k)
        assert ck.ok(lay), f"benchmark lesson {k} fails the wall check"
        assert ck.ok(mirror(lay)), f"benchmark lesson {k}, mirrored, fails the wall check"
        for name, lo, _, last, _ in FAMILIES[k]:
            v = BENCHMARK[k][name]
            assert lo <= v <= last, f"lesson {k}: {name} {v} outside its family's [{lo}, {last}]"
    print("benchmark lessons 1..9, both ways round: usable walls, each inside its family's last range")
    t0 = time.perf_counter()
    walls = draw(ck, {f: 1.0 for f in FAMILIES}, first_levels(), rng, 256)
    probed = sum(w['probe'] is not None for w in walls)
    print(f"draw: 256 usable walls in {time.perf_counter() - t0:.2f} s, {probed} probes; "
          f"families {sorted({w['family'] for w in walls})}")
    assert 0.15 < probed / 256 < 0.35, probed


if __name__ == '__main__':
    main()
