"""The reference climbs of rl.rsi: found by search, or recorded from a trained policy.

    python -m rl.plan search f4r20 --seed 0 --iters 80 --workers 6   # -> out/plan/f4r20_s0.pkl
    python -m rl.plan fixture out/plan/f4r20_s0.pkl climb_f4_r20     # -> rl/fixtures/climb_f4_r20.npz
    python -m rl.plan mirror climb_f9_g60 climb_f9m_g60              # a found climb, y -> -y
    python -m rl.plan record out/gpu8 5 0.30 1 climb_f5_d30          # a policy's own climb (taught)

SEARCH: a motion found phase by phase on its wall (rl.rsi.ref_wall) on the simplified robot
MJX trains, in C MuJoCo, by the cross-entropy method over joint-setpoint keyframes moved at
most ACT_SCALE a control step -- a policy's own speed limit. One hand moves a phase: 'over'
lifts it and puts its throat PRE above a band of a rung, its finger opening; 'seat' brings
it down onto that rung, finger shut, holding; 'slide' carries a loaded hook along its rung
while the other hand is free (a swing's). The other hand keeps its hold throughout, and the
left hook stays at least SEP ahead of the right along the wall. Each phase ends with a hold,
so a posture that only passes through its target does not count. Options:

    --hill W    weight on how far the live potential (what K_REACH pays) climbs above its
                lowest so far, past HILL_TOL: without it every family-3 climb began with a
                swing up a 0.4-2.4 m hill of it, which a policy never learns from its starts
    --turn W    on a rung behind the start row a hook holds only turned round, its tip toward
                the back wall: weight on its x axis past -TURN_X (family 8)
    --init P    warm start: each phase searched near that chain's phase (a climb raised a
                row, or carried to the next gap)
    --start P   carry on from that chain's end (a climb found in parts)
    --ik i:F    phase i's keyframes on the way to the pose in F (a 6-number base, then the
                joints in rl.estimator.JOINTS' order), found by least squares (`ik`)

FIXTURE: a chain as rl.rsi reads a found climb, its first setpoints and one row of them a
control step. MIRROR: a found climb y -> -y -- the simplified robot is its own mirror image
(forward kinematics exact), so it is the climb on the mirrored wall; a swing does not
survive it (mirrored, 0.90 fell), and each way round of those was searched. RECORD: a trained
policy's climb on its wall from the hang, recorded in MJX, the fastest of its deterministic
episodes kept; what rl.rsi clones on it is the policy's mean action at each state, not the
sampled step. With --hide-below, the rungs below both hands are out of its observation (gpu7
climbed one row 24/24 and two 0/24, the start row in its view; without it, 8-11/16).

Where each fixture in rl/fixtures came from (seeds, warm starts) is listed in PROVENANCE.
"""
import argparse, os, pickle, sys, time
import numpy as np, mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'out', 'plan')
PRE, SUB = 0.045, 20          # m the throat goes over a rung before seating; substeps a control step
HILL_TOL = 0.01
TURN_X = 0.7
OPEN, SHUT = 0.349, np.radians(-25)
S = 0.0375                    # rl.walls.SEAT: a goal's closest to a rung end
HY = 0.157

# phase: (hand, rung slot, y band lo, hi, mode, T s, keyframes, hold s[, keep the order])
def _step(rung, L=(0.10, 0.25), R=(-0.25, -0.05), over=(5.0, 5), last=(5.0, 5)):
    """both hands onto a full row straight above, below or behind the hang, left first"""
    return [('L', rung, *L, 'over', *over, 0.5), ('L', rung, *L, 'seat', 2.0, 2, 0.8),
            ('R', rung, *R, 'over', *last, 0.5), ('R', rung, *R, 'seat', 2.0, 2, 0.8)]


def _f6():
    """onto one tier of two short rungs (rl.rsi.ref_wall(6)), a climber's order: L up onto the
    tier, R on to the start row's end, R up behind L, L across the gap"""
    return [('L', 'rung_g0', 0.27, 0.3375, 'over', 2.0, 2, 0.4), ('L', 'rung_g0', 0.27, 0.3375, 'seat', 1.5, 2, 0.6),
            ('R', 'rung_g0', 0.08, 0.15, 'over', 2.0, 2, 0.4), ('R', 'rung_g0', 0.08, 0.15, 'seat', 1.5, 2, 0.6),
            ('L', 'rung_g1', 0.74, 0.8375, 'over', 6.0, 6, 0.5), ('L', 'rung_g1', 0.74, 0.8375, 'seat', 2.0, 2, 0.8),
            ('R', 'rung_g0', 0.22, 0.3375, 'over', 2.5, 3, 0.4), ('R', 'rung_g0', 0.22, 0.3375, 'seat', 1.5, 2, 0.6),
            ('R', 'rung_g1', 0.53, 0.59, 'over', 5.0, 5, 0.5), ('R', 'rung_g1', 0.53, 0.59, 'seat', 2.0, 2, 0.8),
            ('L', 'rung_g2', 1.16, 1.43, 'over', 5.0, 5, 0.5), ('L', 'rung_g2', 1.16, 1.43, 'seat', 2.0, 2, 0.8)]


def _f9(gap):
    """across one gap statically (rl.rsi.ref_wall(9)): L to the row's end, R behind it, L across,
    R to the row's end, L on along the far rung to make room, R across behind it, L to its goal"""
    e = HY + 0.21
    y0 = e + gap
    yL = y0 + S + 0.03 + 0.31
    return [('L', 'rung_g0', e - 0.10, e - S, 'over', 2.0, 2, 0.4), ('L', 'rung_g0', e - 0.10, e - S, 'seat', 1.5, 2, 0.6),
            ('R', 'rung_g0', e - 0.25, e - S - 0.12, 'over', 2.0, 2, 0.4),
            ('R', 'rung_g0', e - 0.25, e - S - 0.12, 'seat', 1.5, 2, 0.6),
            ('L', 'rung_g1', y0 + S, y0 + 0.13, 'over', 6.0, 6, 0.5), ('L', 'rung_g1', y0 + S, y0 + 0.13, 'seat', 2.0, 2, 0.8),
            ('R', 'rung_g0', e - 0.10, e - S, 'over', 2.5, 3, 0.4), ('R', 'rung_g0', e - 0.10, e - S, 'seat', 1.5, 2, 0.6),
            ('L', 'rung_g1', y0 + S + 0.13, y0 + 0.22, 'over', 4.0, 4, 0.5),
            ('L', 'rung_g1', y0 + S + 0.13, y0 + 0.22, 'seat', 2.0, 2, 0.8),
            ('R', 'rung_g1', y0 + S, y0 + 0.08, 'over', 5.0, 5, 0.5), ('R', 'rung_g1', y0 + S, y0 + 0.08, 'seat', 2.0, 2, 0.8),
            ('L', 'rung_g1', yL - 0.06, yL + 0.04, 'over', 3.0, 3, 0.4), ('L', 'rung_g1', yL - 0.06, yL + 0.04, 'seat', 1.5, 2, 0.6)]


def _f9s(gap):
    """across a gap past the static reach by lesson 9's swing: L from the row's end swung across,
    R let go and the loaded L carried on along the far rung by the swing, R across behind it, L on"""
    e = HY + 0.21
    y0 = e + gap
    yL = y0 + S + 0.03 + 0.31
    return [('L', 'rung_g0', e - 0.077, e - S, 'over', 2.0, 2, 0.4), ('L', 'rung_g0', e - 0.077, e - S, 'seat', 1.5, 2, 0.6),
            ('R', 'rung_g0', e - 0.19, e - 0.15, 'over', 2.0, 2, 0.4), ('R', 'rung_g0', e - 0.19, e - 0.15, 'seat', 1.5, 2, 0.6),
            ('L', 'rung_g1', y0 + S, y0 + 0.45, 'over', 8.0, 8, 0.5), ('L', 'rung_g1', y0 + S, y0 + 0.45, 'seat', 2.0, 2, 0.8),
            ('L', 'rung_g1', y0 + 0.18, y0 + 0.37, 'slide', 5.0, 5, 0.5),
            ('R', 'rung_g1', y0 + S, y0 + 0.21, 'over', 5.0, 5, 0.5), ('R', 'rung_g1', y0 + S, y0 + 0.21, 'seat', 2.0, 2, 0.8),
            ('L', 'rung_g1', yL - 0.10, yL + 0.07, 'over', 2.0, 2, 0.4), ('L', 'rung_g1', yL - 0.10, yL + 0.07, 'seat', 1.5, 2, 0.8)]


def mirrored(phases):
    """phases the other way round: the hands swapped, each band turned over"""
    return [('R' if p[0] == 'L' else 'L', p[1], -p[3], -p[2]) + tuple(p[4:]) for p in phases]


# name -> (family, height, side, rows, phases): the wall is rl.rsi.ref_wall(family, height, side, rows)
PLANS = {}
for h in (0.20, 0.25, 0.30):
    PLANS[f'f3r{round(h * 100)}'] = (3, h, 1, 1, _step('rung_g1'))
for h in (0.20, 0.30, 0.40):                     # down: the hook comes over the lower row from above
    PLANS[f'f4r{round(h * 100)}'] = (4, h, 1, 1, _step('rung_g1'))
PLANS['f4x2r40'] = (4, 0.40, 1, 2, _step('rung_g1') + _step('rung_g2'))
for h in (0.30, 0.40):
    PLANS[f'f6r{round(h * 100)}'] = (6, h, 1, 1, _f6())
    PLANS[f'f6mr{round(h * 100)}'] = (6, h, -1, 1, mirrored(_f6()))
for b in (0.35, 0.50, 0.55, 0.60, 0.70):
    PLANS[f'f8b{round(b * 100)}'] = (8, b, 1, 1, _step('rung_g1', L=(0.08, 0.25), over=(6.0, 6), last=(6.0, 6)))
for g in (0.55, 0.60):
    PLANS[f'f9g{round(g * 100)}'] = (9, g, 1, 1, _f9(g))
    PLANS[f'f9mg{round(g * 100)}'] = (9, g, -1, 1, mirrored(_f9(g)))
for g in (0.70, 0.80, 0.90):
    PLANS[f'f9sg{round(g * 100)}'] = (9, g, 1, 1, _f9s(g))
    PLANS[f'f9msg{round(g * 100)}'] = (9, g, -1, 1, mirrored(_f9s(g)))

# rl/fixtures/<file>.npz <- how it was made (searches on the robot as of 2026-10; their chains in
# out/climb_demo/ on the machine that ran them)
PROVENANCE = {
    'climb_f3_s20': "search f3r20 --hill 2, seed 42", 'climb_f3_s25': "search f3r25 --hill 2 --init f3_s20, seed 43",
    'climb_f3_s30': "search f3r30 --hill 2 --init f3_s20, seed 47",
    'climb_f3x2_s20': "record gpu7 (173M) --hide-below", 'climb_f3x2_s25': "record gpu7 (173M) --hide-below",
    'climb_f3x2_s35': "record gpu11 (100M)", 'climb_f3x2_s40': "record gpu11 (100M)",
    'climb_f5_d30': "record gpu8 (162M)", 'climb_f5m_d30': "record gpu8 (162M)",
    'climb_f6_r30': "search f6r30", 'climb_f6m_r30': "search f6mr30, then its last two phases again (--start)",
    'climb_f6_r40': "search f6r40 --init f6r30", 'climb_f6m_r40': "search f6mr40 --init f6mr30",
    'climb_f7_s60': "record gpu10 (215M)", 'climb_f7m_s60': "record gpu10 (215M)",
    'climb_f8_b35': "search f8b35 --turn 2 --ik 4:<least squares, the left hook over the back rung turned round>",
    'climb_f8_b60': "search f8b60 --turn 2 --hill 2 --ik (as 0.35)",
    'climb_f8_b50': "search f8b50 --turn 2 --hill 2 --init f8b60", 'climb_f8_b55': "search f8b55 --turn 2 --hill 2 --init f8b60",
    'climb_f8_b70': "search f8b70 --turn 2 --hill 2 --init f8b60",
    'climb_f9_g55': "search f9g55", 'climb_f9m_g55': "search f9mg55",
    'climb_f9_g60': "search f9g60", 'climb_f9m_g60': "mirror climb_f9_g60",
    **{f'climb_f9{m}_g{g}': f"search f9{m}sg{g} --init lesson 9's swing on the full robot (the other way: "
                            f"its mirror), no hill weight" for g in (70, 80, 90) for m in ('', 'm')},
    'climb_f4_r20': "search f4r20 --hill 2, seed 0",
    'climb_f4_r30': "search f4r30 --hill 2 --init f4r20, seed 1",
    'climb_f4_r40': "search f4r40 --hill 2 --init f4r30, seed 2",
    'climb_f4x2_r40': "search f4x2r40 --hill 2 --init f4r40, seed 3",
}


# --- the search -------------------------------------------------------------------
G = {}


def task(plan):
    """a ClimbTask on the plan's wall, the simplified robot, at its secured hang"""
    from .climb_task import ClimbTask
    from .mjx_scene import c_scene
    from .rsi import ref_wall
    f, h, side, rows, _ = PLANS[plan]
    xml, spec = c_scene(dict(ref_wall(f, h, side, rows), timeout=90.0))
    try:
        t = ClimbTask(spec=spec, xml=xml, visual=False)
    finally:
        os.remove(xml)
    t.reset()
    return t


def _init(plan, opts):
    from .estimator import JOINTS
    t = task(plan)
    G.update(t=t, m=t.env.model, d=t.env.data, phases=PLANS[plan][4], opts=opts,
             FL=JOINTS.index('finger_L'), FR=JOINTS.index('finger_R'),
             MOV=[i for i, n in enumerate(JOINTS) if not n.startswith('finger')])


def _state():
    m, d = G['m'], G['d']
    st = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
    return st


def _band(y, lo, hi):
    return max(0.0, lo - y, y - hi)


def _objective(i):
    hand, rung, lo, hi, mode = G['phases'][i][:5]
    order = G['phases'][i][8] if len(G['phases'][i]) > 8 else True
    t, d, o = G['t'], G['d'], G['opts']
    P = {s: t._throat_true(s) for s in 'LR'}
    p, z = P[hand], t.segs[rung][0][2]
    v = float(np.linalg.norm(d.qvel[:3]))
    f = _band(p[1], lo, hi) + (_band(P['L'][1] - P['R'][1], o['sep'], 9.0) if order else 0.0)
    anchor = 'R' if hand == 'L' else 'L'          # the holding hook keeps its place
    drift = 2.0 * max(0.0, float(np.linalg.norm(P[anchor] - G['anchor0'])) - 0.02)
    xr = t.segs[rung][0][0]                       # the rung's own distance from the wall
    if o['turn'] and xr < 0.3:                    # a rung behind the start row: turned round
        f += o['turn'] * max(0.0, float(d.xmat[t.env.hook[hand]].reshape(3, 3)[0, 0]) + TURN_X)
    if mode == 'over':
        return float(np.hypot(p[0] - xr, p[2] - (z + PRE))) + f + drift + 0.1 * v
    ok = t._holding()['LR'.index(hand)] and rung in t._contact(hand)[1]
    return float(np.hypot(p[0] - xr, p[2] - z)) + f + (0.0 if mode == 'slide' else drift) + \
        (0.0 if ok else 0.3) + 0.2 * v


def rollout(theta, job, record=False):
    from .climb_task import ACT_SCALE
    i, start, ctrl0, cmin = job
    hand, _, _, _, mode, T, K, hold = G['phases'][i][:8]
    anchor = hand if mode == 'slide' else ('R' if hand == 'L' else 'L')
    t, m, d, MOV = G['t'], G['m'], G['d'], G['MOV']
    mujoco.mj_setState(m, d, start, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, d)
    G['anchor0'] = t._throat_true(anchor).copy()
    ctrl = np.array(ctrl0, float)
    d.ctrl[:] = ctrl
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    keys = np.clip(np.asarray(theta[:K * 15]).reshape(K, 15), lo[MOV], hi[MOV])
    tf = float(np.clip(theta[K * 15], 0.0, 1.0)) * T
    pts = np.vstack([ctrl[MOV].copy(), keys])
    times = np.linspace(0, T, K + 1)
    dt = SUB * m.opt.timestep
    n, nh = int(round(T / dt)), int(round(hold / dt))
    fm, fa = (G['FL'], G['FR']) if hand == 'L' else (G['FR'], G['FL'])
    if mode == 'slide':                        # the free hand is the other one: it opens
        fm, fa = fa, fm
    off, log, hill = 0, [], 0.0
    cmin = min(cmin, float(t._live().sum()))
    for k in range(n + nh):
        tk = (k + 1) * dt
        tgt = np.array([np.interp(min(tk, T), times, pts[:, j]) for j in range(15)])
        ctrl[MOV] += np.clip(tgt - ctrl[MOV], -ACT_SCALE, ACT_SCALE)
        want = (OPEN if tk >= tf else SHUT) if mode in ('over', 'slide') else (SHUT if tk >= tf else ctrl[fm])
        ctrl[fm] += np.clip(want - ctrl[fm], -ACT_SCALE, ACT_SCALE)
        ctrl[fa] += np.clip(SHUT - ctrl[fa], -ACT_SCALE, ACT_SCALE)
        ctrl = np.clip(ctrl, lo, hi)
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d, SUB)
        if record:
            log.append(ctrl.copy())
        if not np.all(np.isfinite(d.qpos)):
            return 50.0, log, ctrl
        off = off + 1 if t._contact(anchor)[0] < 5.0 else 0
        if off * dt > 0.25:                    # the holding hand let go: a fall
            return 20.0 + 10.0 * (n + nh - k) / (n + nh), log, ctrl
        if G['opts']['hill'] or record:
            c = float(t._live().sum())
            hill, cmin = max(hill, c - cmin), min(cmin, c)
    G['hill'], G['cmin'] = hill, cmin
    return _objective(i) + G['opts']['hill'] * max(0.0, hill - HILL_TOL), log, ctrl


def _eval(args):
    theta, job = args
    return rollout(theta, job)[0]


def cem(job, mean, sigma, iters, pool, pop=120, elite=12, seed=0):
    rng = np.random.default_rng(seed)
    best = (1e9, mean.copy())
    t0 = time.time()
    for it in range(iters):
        X = mean + sigma * rng.standard_normal((pop, len(mean)))
        X[0] = best[1]
        f = np.array(pool.map(_eval, [(x, job) for x in X]))
        idx = np.argsort(f)[:elite]
        if f[idx[0]] < best[0]:
            best = (float(f[idx[0]]), X[idx[0]].copy())
        mean = 0.3 * mean + 0.7 * X[idx].mean(0)
        sigma = 0.3 * sigma + 0.7 * X[idx].std(0) + 0.01
        if it % 10 == 0 or it == iters - 1:
            print(f"  phase {job[0]} it {it:3d}: best {best[0]:.3f}  elite {f[idx].mean():.3f}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        if best[0] < 0.004 and it >= 15:
            break
    return best


def search(plan, seed, iters, workers, opts):
    from multiprocessing import Pool
    _init(plan, opts)
    phases = PLANS[plan][4]
    start, ctrl = _state(), G['d'].ctrl.copy()
    chain = dict(plan=plan, start=start.copy(), ctrl0=ctrl.copy(), phases=[])
    if opts['start']:                          # carry on from another chain's end
        prev = pickle.load(open(opts['start'], 'rb'))
        m, d = G['m'], G['d']
        for ph in prev['phases']:
            for c in ph['ctrl_log']:
                d.ctrl[:] = c
                mujoco.mj_step(m, d, SUB)
        chain['phases'] = list(prev['phases'])
        start, ctrl = _state(), d.ctrl.copy()
    init = pickle.load(open(opts['init'], 'rb')) if opts['init'] else None
    ik = dict(kv.split(':') for kv in opts['ik'].split(',')) if opts['ik'] else {}
    cmin = float(G['t']._live().sum())
    MOV = G['MOV']
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{plan}_s{seed}.pkl")
    with Pool(workers, initializer=_init, initargs=(plan, opts)) as pool:
        for i, ph in enumerate(phases):
            hand, rung, lo, hi, mode, T, K, hold = ph[:8]
            sig = 0.6 if mode == 'over' else 0.25
            mean = np.concatenate([np.tile(ctrl[MOV], K), [0.3]])
            sigma = np.concatenate([np.full(K * 15, sig), [0.25]])
            if str(i) in ik:                   # on the way to a pose found by least squares
                q = np.load(ik[str(i)])[6:][MOV]
                lo_, hi_ = G['m'].actuator_ctrlrange[MOV, 0], G['m'].actuator_ctrlrange[MOV, 1]
                keys = [ctrl[MOV] + (q - ctrl[MOV]) * (k + 1) / K for k in range(K)]
                mean = np.concatenate([np.clip(np.concatenate(keys), np.tile(lo_, K), np.tile(hi_, K)), [0.6]])
                sigma = np.concatenate([np.full(K * 15, 0.3), [0.15]])
            if init and i < len(init['phases']) and len(init['phases'][i]['theta']) == len(mean):
                mean = np.array(init['phases'][i]['theta'], float)          # searched near that phase
                sigma = np.concatenate([np.full(K * 15, 0.5 * sig), [0.15]])
            job = (i, start, ctrl, cmin)
            f, theta = cem(job, mean, sigma, iters, pool, seed=seed + i)
            f2, log, ctrl = rollout(theta, job, record=True)
            cmin = G['cmin']
            start = _state()
            t = G['t']
            print(f"phase {i} {hand} {mode} {rung} [{lo:.3f}, {hi:.3f}]: objective {f2:.3f}; holding "
                  f"{t._holding()}; live hill {G['hill'] * 1000:.0f} mm", flush=True)
            chain['phases'].append(dict(phase=i, spec=ph, theta=theta, objective=f2, ctrl_log=log, hill=G['hill']))
            pickle.dump(chain, open(path, 'wb'))
            if f2 > 0.05:
                print(f"phase {i} not solved (objective {f2:.3f}); stopping here")
                return None
    print(f"-> {path}")
    return path


# --- fixtures ---------------------------------------------------------------------
def fixture(chain_path, name):
    chain = pickle.load(open(chain_path, 'rb'))
    ctrl = np.concatenate([np.asarray(p['ctrl_log']) for p in chain['phases']])
    out = os.path.join(HERE, 'fixtures', name + '.npz')
    np.savez_compressed(out, ctrl0=np.asarray(chain['ctrl0']), ctrl=ctrl, plan=chain['plan'],
                        hill=max(p['hill'] for p in chain['phases']))
    print(f"{chain_path}: {len(chain['phases'])} phases, {len(ctrl)} steps ({len(ctrl) * 0.04:.1f} s) -> {out}")


def mirror_joints(rows):
    """joint rows (rl.estimator.JOINTS' order) y -> -y: each arm's joints the other's, those
    turning about an axis that mirrors onto its own reverse negated"""
    from .estimator import JOINTS
    neg = ('waist_yaw', 'abduction', 'rotation', 'roll', 'deviation')
    rows = np.asarray(rows, float)
    out = np.empty_like(rows)
    for i, n in enumerate(JOINTS):
        src = n.replace('_L', '_X').replace('_R', '_L').replace('_X', '_R')
        out[..., i] = rows[..., JOINTS.index(src)] * (-1.0 if any(k in n for k in neg) else 1.0)
    return out


def mirror(src, dst):
    z = np.load(os.path.join(HERE, 'fixtures', src + '.npz'))
    np.savez_compressed(os.path.join(HERE, 'fixtures', dst + '.npz'), ctrl0=mirror_joints(z['ctrl0']),
                        ctrl=mirror_joints(z['ctrl']), plan=str(z['plan']) + ' mirrored', hill=z['hill'])


def record(run, f, h, side, name, rows=1, tries=32, seed=0, hide_below=False):
    """`tries` episodes of `run`'s policy on rl.rsi.ref_wall(f, h, side, rows) from the hang in MJX,
    half its mean and half sampled, every state recorded; the fastest of the mean's successes
    (else of any) saved as rl/fixtures/<name>.npz"""
    import jax
    import jax.numpy as jp
    from stable_baselines3 import PPO
    from . import mjx_task as MT
    from .rsi import ref_wall

    class HideBelow(MT.MjxClimb):
        def _obs(self, T, st, sen, m=None):
            m = self._map(st)
            under = jp.arange(m.rung_valid.shape[0]) < st.hold_idx.min()    # rows below both hands
            return super()._obs(T, st, sen, m._replace(rung_valid=m.rung_valid & ~under))

    model = PPO.load(os.path.join(run, 'model.zip'), device='cpu')
    with open(os.path.join(run, 'vecnorm.pkl'), 'rb') as fh:
        vn = pickle.load(fh)
    lay = dict(ref_wall(f, h, side, rows), probe=None, probe_val=0.0)
    L = len(MT.LESSON_KEYS)
    b = MT.Batch(tries, bank=tries, task=(HideBelow if hide_below else MT.MjxClimb)(tries))
    b.set_bank([lay] * tries)
    b.state, o = b._reset(b.task.tables, jp.asarray(L + np.arange(tries)),
                          jax.random.split(jax.random.PRNGKey(seed), tries))
    o, det = np.asarray(o), np.arange(tries) % 2 == 0
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
    print(f"record: {len(won)}/{tries} succeed ({sum(i % 2 == 0 for i in won)}/{tries // 2} the mean)")
    if won:
        mine = {i: v for i, v in won.items() if det[i]} or won
        i = min(mine, key=mine.get)
        out = os.path.join(HERE, 'fixtures', name + '.npz')
        np.savez_compressed(out, won=won[i], run=run, **{k: np.stack(v)[:won[i], i] for k, v in rec.items()})
        print(f"record: -> {out}, {won[i]} steps ({won[i] * 0.04:.1f} s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search')
    s.add_argument('plan', choices=sorted(PLANS))
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--iters', type=int, default=80)
    s.add_argument('--workers', type=int, default=max(1, int(0.5 * (os.cpu_count() or 2))))
    s.add_argument('--sep', type=float, default=0.15, help='m the left hook keeps ahead (the task asks 0.12)')
    s.add_argument('--hill', type=float, default=0.0)
    s.add_argument('--turn', type=float, default=0.0)
    s.add_argument('--init', default=None)
    s.add_argument('--start', default=None)
    s.add_argument('--ik', default=None)
    s = sub.add_parser('fixture')
    s.add_argument('chain')
    s.add_argument('name')
    s = sub.add_parser('mirror')
    s.add_argument('src')
    s.add_argument('dst')
    s = sub.add_parser('record')
    s.add_argument('run')
    s.add_argument('family', type=int)
    s.add_argument('height', type=float)
    s.add_argument('side', type=int)
    s.add_argument('name')
    s.add_argument('--rows', type=int, default=1)
    s.add_argument('--hide-below', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'search':
        ok = search(a.plan, a.seed, a.iters, a.workers,
                    dict(sep=a.sep, hill=a.hill, turn=a.turn, init=a.init, start=a.start, ik=a.ik))
        sys.exit(0 if ok else 1)
    if a.cmd == 'fixture':
        fixture(a.chain, a.name)
    elif a.cmd == 'mirror':
        mirror(a.src, a.dst)
    else:
        record(a.run, a.family, a.height, a.side, a.name, rows=a.rows, hide_below=a.hide_below)


if __name__ == '__main__':
    main()
