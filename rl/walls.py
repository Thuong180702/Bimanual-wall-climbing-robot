"""Procedurally generated climbing walls, and the ranges ADR grows them over.

    python -m rl.walls          # every family at its first and its last ranges: valid,
                                # and each benchmark lesson inside its family's last range

The benchmark -- lessons 1..9 of rl.lessons -- is what a policy is scored on, on the full
robot. It is not what it trains on. Training draws walls from nine families, one per
benchmark lesson and fixed before any training: each is that lesson's structure with a
few physical parameters free (FAMILIES). Every parameter has a range whose top starts at
the easy end and moves by a fixed rule (automatic domain randomization, OpenAI et al.
2019 "Solving Rubik's Cube with a Robot Hand"; legged_gym's terrain curriculum, Rudin et
al. 2022): half of a family's walls put one parameter exactly at its top, and when 40
of those in a row are passed 70 % of the time the top moves up a step (down one at
20 % or less). Nothing is added by hand when training is stuck, and the benchmark
lesson sits inside its family's last range (`main` asserts it).

Every wall is a lesson layout in `rl.mjx_scene.lesson_layout`'s format -- rungs
(name, centre, axis, half length, radius), walls (name, point on the inner face,
normal), the two goals [L, R] and the goal rungs -- so the MJX slots take it as they
take a lesson. Half of them are mirrored (y -> -y, the goals swapped between hands):
the robot is symmetric, the benchmark only goes one way. A generated episode lasts at
most TIMEOUT: lesson 1's, at which hanging still costs less than falling
(climb_task's incentive test).
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
OUT = 0.20                       # m each family-3 row stands further from the wall than the one below

# family -> ((parameter, sampled from, first top, last top, step), ...). Integers are
# counts. The last tops are the benchmark's value or a margin past it, and the
# robot's measured reach (rl.lessons) bounds the rest: a staircase step stays inside
# the static reach down (0.65 m at 0.15 m aside); the break of climb_over starts past
# the 1.55 m the robot can bridge with momentum, so it can only be climbed round; and
# the swing gap runs from the 0.58 m a still hang crosses to lesson 9's own 0.90 m.
FAMILIES = {
    1: (('shift', 0.0, 0.30, 1.40, 0.10),),                        # lesson 1: 1.10
    2: (('gaps', 1, 1, 2, 1), ('width', 0.04, 0.04, 0.30, 0.02)),  # lesson 2: 2 x 0.25
    # lesson 3: 2 x 0.40, each row OUT 0.20 m further from the wall than the one below.
    # Rows stacked straight (as until 2026-10-06) are in the robot: its arms hang 59-107
    # mm off the rung plane, so hanging one row up, a row 0.10-0.15 m below is 15-23 mm
    # inside the wrists, 0.20-0.50 m is 6-25 mm off the forearms and upper arms, and two
    # rows of 0.45 put the base into the lowest one (the robot fell off the top row) --
    # every CEM climb of a stacked row pressed an arm on the row below at 100-415 N, and
    # gpu2 butted the row above from below for 130M steps. Standing out, the rows below
    # are behind the robot: after 3 s hanging from the top one, 92 mm (2 x 0.45, the
    # base) to 164 mm clear over this whole range; out 0.05-0.10 the base still swings
    # onto the lowest of two rows. Reaching OUT costs height: the free hook rises 0.56 m
    # in the rung plane, 0.52 m 0.15-0.25 m out (urdf/_reach.py's hang), so a row is at
    # most 0.45 up (0.50 + the 45 mm over it would be past the reach; CEM: 0.20 and 0.40
    # found, 0.45 not in 3 seeds). It started at 0.30 up, gpu1's 0 % for 90M steps (a reward
    # barrier since fixed); 0.20 at the user's call.
    3: (('rows', 1, 1, 2, 1), ('spacing', 0.20, 0.20, 0.45, 0.05)),
    4: (('steps', 1, 1, 3, 1), ('drop', 0.25, 0.30, 0.45, 0.05),   # lesson 4: 3 steps,
        ('gap', 0.05, 0.10, 0.30, 0.05)),                          #   <= 0.40 down, <= 0.20 on
    5: (('break', 1.80, 1.80, 2.50, 0.10),),                       # lesson 5: 2.20
    6: (('tiers', 1, 1, 2, 1), ('rise', 0.30, 0.30, 0.50, 0.05),   # lesson 6: 2, ~0.46,
        ('scatter', 0.0, 0.02, 0.08, 0.02)),                       #   +-0.06
    7: (('side', 0.60, 0.60, 1.10, 0.05),),                        # lesson 7: 1.00
    8: (('behind', 0.35, 0.35, 0.70, 0.05),),                      # lesson 8: 0.63
    # lesson 9: 0.90. It ran to 1.10 m, inside the 1.47 m kinematics allow a swing; the one
    # motion found across 0.90 (CEM, 2026-09-26) needed every bit of the reach there is --
    # the catch 1.10 m from the other hook, 2.5 mm inside the far rung's seat -- and no
    # search from further back reached past 1.07 m. Past 0.90 no wall is known passable.
    9: (('gap', 0.30, 0.55, 0.90, 0.05),),
}


# family -> the families whose skills it is made of. A family enters the pool only once
# each of these has passed (train_ppo.Walls), so an upgrade never comes before what it
# upgrades, and a family stuck at 0 % holds back its upgrades and nothing else -- gpu1's
# family 3 kept 4-9 locked for 90M steps, and only 5 and 6 climb up. Fixed with the
# families, before any training; numbered order is a valid order, so with nothing stuck
# the families enter 1, 2, ..., 9 exactly as before.
NEEDS = {1: (),
         2: (1,),        # gaps in the row: a traverse that changes rungs
         3: (1,),        # rows above: a traverse's re-grip, reached upward
         4: (2,),        # down a staircase: each step is another rung across a gap
         5: (3, 4),      # over a break: up a row, along it, down again
         6: (2, 3),      # tiers of short rungs: up, and across the gaps between them
         7: (1,),        # the corner: from the front row onto the side rung, level
         8: (1,),        # the wall behind: onto the back rung, level
         9: (2,)}        # a gap past the static reach: crossed by swinging
assert set(NEEDS) == set(FAMILIES) and all(p < f for f, ps in NEEDS.items() for p in ps), \
    "every family's needs come before it in the numbered order"


# where each benchmark lesson sits in its family (rl.lessons draws them): every value
# must lie inside the family's last range, so the benchmark is a wall training can draw
BENCHMARK = {1: dict(shift=1.10), 2: dict(gaps=2, width=0.25), 3: dict(rows=2, spacing=0.40),
             4: dict(steps=3, drop=0.40, gap=0.20), 5: dict(**{'break': 2.20}),
             6: dict(tiers=2, rise=0.46, scatter=0.06), 7: dict(side=1.00),
             8: dict(behind=0.63), 9: dict(gap=0.90)}


def spec(f):
    """family f's ranges as a checkpoint records them: its tops mean something only
    against the ranges they were grown over (train_ppo.Walls starts a family over when
    they changed)"""
    return repr(FAMILIES[f])


# the ranges of checkpoints from before they were recorded with their tops (gpu1 up to
# 102.9M steps): where they differ from today's
LEGACY = {3: repr((('rows', 1, 1, 2, 1), ('spacing', 0.25, 0.30, 0.50, 0.05))),
          9: repr((('gap', 0.30, 0.55, 1.10, 0.05),))}


def first_tops():
    return {f: {p[0]: p[2] for p in ps} for f, ps in FAMILIES.items()}


def last_tops():
    return {f: {p[0]: p[3] for p in ps} for f, ps in FAMILIES.items()}


def _row(y0, y1, z, x=X):
    """a rung along y from y0 to y1"""
    return ((y0 + y1) / 2, np.array([x, (y0 + y1) / 2, z]), np.array([0.0, 1.0, 0.0]),
            (y1 - y0) / 2)


def _front():
    return ('wall_front', np.array([0.50, 0.30, 1.00]), np.array([-1.0, 0.0, 0.0]))


def _draw(rng, spec, top, probe):
    """one parameter: its top when it is the one probed, else uniform below it"""
    name, lo, _, _, _ = spec
    t = top[name]
    if probe == name:
        return t
    return int(rng.integers(lo, t + 1)) if isinstance(lo, int) else float(rng.uniform(lo, t))


def _goals_from(y_lo, rng, z=Z0, x=X):
    """both goals on a rung that starts at y_lo: R just past the start, L ~0.3 m on
    -> (goals [L, R], where the rung ends)"""
    yR = y_lo + SEAT + rng.uniform(0.005, 0.06)
    yL = yR + rng.uniform(0.28, 0.34)
    return [np.array([x, yL, z]), np.array([x, yR, z])], yL + SEAT + rng.uniform(0.02, 0.15)


def sample(family, top, rng, probe=None):
    """one wall of `family` with its tops `top` ({parameter: top}); `probe` names the
    parameter put exactly at its top -> layout dict (rl.mjx_scene.lesson_layout's)"""
    p = {s[0]: _draw(rng, s, top, probe) for s in FAMILIES[family]}
    rungs, walls, goal_rungs = [], [_front()], None
    d = rng.uniform(-0.15, 0.15)                   # the goals off centre, on a full row
    if family == 1:                                # along the start row
        dL, dR = rng.uniform(0, p['shift'], 2)
        if probe == 'shift':
            dL, dR = (p['shift'], dR) if rng.random() < 0.5 else (dL, p['shift'])
        dL = max(dL, dR - (2 * HY - MIN_W))        # the left hand stays left
        yL, yR = HY + dL, -HY + dR
        rungs = [_row(rng.uniform(-0.9, -0.6), yL + SEAT + rng.uniform(0.03, 1.2), Z0)]
        goals, goal_rungs = [np.array([X, yL, Z0]), np.array([X, yR, Z0])], [0]
    elif family == 2:                              # gaps in the start row
        y = HY + rng.uniform(0.12, 0.30)
        rungs = [_row(rng.uniform(-0.6, -0.45), y, Z0)]
        for i in range(p['gaps']):
            y0 = y + (p['width'] if probe == 'width' else rng.uniform(0.04, p['width']))
            if i < p['gaps'] - 1:
                y = y0 + rng.uniform(0.35, 0.60)
                rungs.append(_row(y0, y, Z0))
        goals, end = _goals_from(y0, rng)
        rungs.append(_row(y0, end, Z0))
        goal_rungs = [len(rungs) - 1]
    elif family == 3:                              # rows above, each further out, to the top one
        m, dz = p['rows'], p['spacing']
        rungs = [_row(Y0, Y1, Z0 + k * dz, x=X - k * OUT) for k in range(m + 1)]
        x, z = X - m * OUT, Z0 + m * dz
        goals, goal_rungs = [np.array([x, HY + d, z]), np.array([x, -HY + d, z])], [m]
    elif family == 4:                              # down a staircase: each step lower, further on
        y = HY + rng.uniform(0.12, 0.30)           # the start row ends past the left hook
        rungs, z = [_row(-0.9, y, Z0)], Z0
        for i in range(p['steps']):
            y0 = y + (p['gap'] if probe == 'gap' else rng.uniform(0.05, p['gap']))
            z -= p['drop'] if probe == 'drop' else rng.uniform(0.25, p['drop'])
            if i < p['steps'] - 1:
                y = y0 + rng.uniform(0.40, 0.60)   # room for both hooks on every step
                rungs.append(_row(y0, y, z))
        goals, end = _goals_from(y0, rng, z=z)
        rungs.append(_row(y0, end, z))
        goal_rungs = [len(rungs) - 1]
    elif family == 5:                              # a break in the start row, a row above it
        e = HY + rng.uniform(0.12, 0.30)           # alone: up and down are diagonal steps (6, 4)
        f0 = e + p['break']
        goals, end = _goals_from(f0, rng)
        rungs = [_row(-0.9, e, Z0), _row(e + rng.uniform(0.05, 0.20), f0 - rng.uniform(0.05, 0.20),
                                         Z0 + rng.uniform(0.35, 0.50)),
                 _row(f0, end, Z0)]
        goal_rungs = [2]
    elif family == 9:                              # one gap past the static reach: swing
        e = HY + rng.uniform(0.12, 0.30)
        y0 = e + p['gap']
        goals, end = _goals_from(y0, rng)
        rungs = [_row(rng.uniform(-0.6, -0.45), e, Z0), _row(y0, end, Z0)]
        goal_rungs = [1]
    elif family == 6:                              # tiers of short rungs, each tier past the
        y = HY + rng.uniform(0.12, 0.30)           # end of the one below: climbed diagonally
        rungs, z = [_row(-0.9, y, Z0)], Z0
        for t in range(p['tiers']):
            z += p['rise']
            c = 2 if t == p['tiers'] - 1 else int(rng.integers(2, 4))
            ln, gp = rng.uniform(0.30, 0.45, c), rng.uniform(0.20, 0.30, c - 1)
            y += rng.uniform(0.05, 0.20)
            for j in range(c):
                rungs.append(_row(y, y + ln[j], z + rng.uniform(-p['scatter'], p['scatter'])))
                y += ln[j] + (gp[j] if j < c - 1 else 0.0)
        (cR, zR), (cL, zL) = [(r[1][1], r[1][2]) for r in rungs[-2:]]
        goals = [np.array([X, cL + rng.uniform(-0.03, 0.03), zL]),
                 np.array([X, cR + rng.uniform(-0.03, 0.03), zR])]
        goal_rungs = [len(rungs) - 2, len(rungs) - 1]
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
        walls.append(('wall_back', np.array([xb - 0.06, 0.30, 1.0]), np.array([1.0, 0.0, 0.0])))
        goals, goal_rungs = [np.array([xb, HY + d, Z0]), np.array([xb, -HY + d, Z0])], [1]
    rungs = [(f'rung_g{i}', c, a, h, R) for i, (_, c, a, h) in enumerate(rungs)]
    lay = dict(rungs=rungs, walls=walls, goals=np.asarray(goals, float),
               goal_rungs=goal_rungs, timeout=TIMEOUT, family=family, probe=probe,
               probe_val=float(p[probe]) if probe else 0.0)
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
    the robot in its secured start hang: `rl.mjx_task` starts every generated wall from
    that one hang, so nothing may be inside or against the robot there. And the same hang
    moved to the goals is CLEAR of every rung but the goal ones: a row straight below
    where the robot ends is in its arms or body (family 3 and lessons 3 and 6 until
    2026-10-06). A static test: hanging, the body swings toward the wall, and two rows
    standing out 0.05-0.10 m put the base on the lowest one -- OUT is set from 3 s
    hangs. Checked in C on the robot MJX trains (rl.mjx_scene), through its own
    collision pairs."""

    def __init__(self):
        from .climb_task import ClimbTask
        from .mjx_scene import N_RUNG, c_path, model
        t = ClimbTask(1, visual=False, xml=c_path(1))
        t.reset()
        self.m = model()
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = t.env.data.qpos
        self.q0 = t.env.data.qpos.copy()
        self.mid0 = np.mean([t._route(t._throat_true(s))[1] for s in 'LR'], axis=0)
        self.rungs = [self.m.geom(f'slot_rung{i}').id for i in range(N_RUNG)]
        self.wall = self.m.geom('slot_wall').id
        # C MuJoCo skips a geom's own rotation when it was compiled at its body's (the
        # parked slots were), so a quat set here would be ignored; MJX never skips it
        self.m.geom_sameframe[self.rungs + [self.wall]] = 0
        pairs = {}
        for a, b in zip(self.m.pair_geom1, self.m.pair_geom2):
            pairs.setdefault(a, []).append(b)
            pairs.setdefault(b, []).append(a)
        self.pairs = pairs
        self.n_rung = N_RUNG

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
        ft = np.zeros(6)
        clear = lambda gs: not any(mujoco.mj_geomDistance(m, self.d, g, o, CLEAR, ft) < CLEAR
                                   for g in gs for o in self.pairs.get(g, []))
        self.d.qpos[:] = self.q0
        mujoco.mj_kinematics(m, self.d)
        if not clear(self.rungs[1:len(rungs)] + ([self.wall] if len(lay['walls']) > 1 else [])):
            return False
        self.d.qpos[:3] += np.mean(lay['goals'], axis=0) - self.mid0
        mujoco.mj_kinematics(m, self.d)
        return clear([g for i, g in enumerate(self.rungs[:len(rungs)]) if i not in lay['goal_rungs']])


def draw(checker, weights, tops, rng, n, probe_p=0.5):
    """n usable walls: a family by `weights` ({family: weight}), and with probability
    probe_p one of its parameters not yet at its last top put exactly at its current top"""
    fams = list(weights)
    w = np.asarray([weights[f] for f in fams], float)
    out = []
    while len(out) < n:
        f = fams[rng.choice(len(fams), p=w / w.sum())]
        free = [s[0] for s in FAMILIES[f] if tops[f][s[0]] < s[3]]
        probe = free[rng.integers(len(free))] if free and rng.random() < probe_p else None
        for _ in range(50):
            lay = sample(f, tops[f], rng, probe)
            if checker.ok(lay):
                out.append(lay)
                break
        else:
            raise RuntimeError(f"family {f} at {tops[f]} (probe {probe}): no usable wall in 50 draws")
    return out


def main():
    from .lessons import lesson_key
    from .mjx_scene import lesson_layout
    rng = np.random.default_rng(0)
    ck = Checker()
    for label, tops in (('first', first_tops()), ('last', last_tops())):
        for f in FAMILIES:
            n_ok = sum(ck.ok(sample(f, tops[f], rng, pr)) for _ in range(40)
                       for pr in [None] + [s[0] for s in FAMILIES[f]])
            tot = 40 * (1 + len(FAMILIES[f]))
            print(f"family {f} at its {label} tops {tops[f]}: {n_ok}/{tot} walls usable")
            assert n_ok >= 0.5 * tot, f"family {f}: most walls at the {label} tops are unusable"
    # each benchmark lesson is usable, and inside its family's last range
    for k in range(1, 10):
        lay = lesson_layout(lesson_key(k))
        assert ck.ok(lay), f"benchmark lesson {k} fails the wall check"
        assert ck.ok(mirror(lay)), f"benchmark lesson {k}, mirrored, fails the wall check"
        for name, lo, _, last, _ in FAMILIES[k]:
            v = BENCHMARK[k][name]
            assert lo <= v <= last, f"lesson {k}: {name} {v} outside its family's [{lo}, {last}]"
    print("benchmark lessons 1..9, both ways round: usable walls, each inside its family's last range")
    t0 = __import__('time').perf_counter()
    walls = draw(ck, {f: 1.0 for f in FAMILIES}, last_tops(), rng, 256)
    dt = __import__('time').perf_counter() - t0
    print(f"draw: 256 usable walls in {dt:.2f} s; families {sorted({w['family'] for w in walls})}")


if __name__ == '__main__':
    main()
