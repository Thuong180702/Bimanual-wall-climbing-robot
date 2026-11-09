"""Score a saved checkpoint on the benchmark -- the full robot in C MuJoCo -- against a
do-nothing control.

    python -m rl.eval --name gpu15                 # every lesson in its pool
    python -m rl.eval --name gpu15 --lessons 1 --episodes 20
    python -m rl.eval --name gpu15 --lessons 1 --mirror     # the lesson toward the right
    python -m rl.eval --name gpu15 --from-best --walls 200  # 200 unseen walls a family (held-out)
    python -m rl.eval --name gpu15 --from-best --walls 50 --lessons 3 --set spacing=0.50   # past training
    python -m rl.eval --name gpu15 --from-best --episodes 100 --cond torque=0.8,lag=20     # robustness
    python -m rl.eval --name gpu15 --from-best --episodes 20 --torque   # joint torques against ratings
    python -m rl.eval --symmetry                                        # the robot's left vs its right
    python -m rl.eval --name gpu15 --from-best --walls 100 --distract below   # rungs outside the move
    python -m rl.eval --name gpu15 --from-best --open-loop --episodes 20       # the reference climbs
                                                                        # played open loop, vs the policy

A benchmark lesson is ONE wall: its episodes differ only through the sensor noise, and its
parameters lie inside the family's training range. `--walls N` scores what the paper calls
the held-out set instead: N walls a family drawn as training draws them (rl.walls, up to
the checkpoint's levels or `--levels last`), from seeds no training run draws from, one
episode each, so the episodes are independent walls and the interval is over walls.

The control matters more than the score. "mean final distance 2.1 m" says nothing on
its own -- the robot starts hanging, and hanging still is already a legal episode. So
every lesson is also run with zero actions, and the policy's number is only evidence
if it beats that.
"""
import argparse, json, os
import numpy as np
import mujoco

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    raise SystemExit(f"missing training deps ({e}); pip install -r requirements.txt")

from . import climb_task as ct
from .climb_task import ClimbTask
from .lessons import LESSONS, lesson_key


WALL_SEED = 2 ** 40      # + family: past every seed + timesteps a training run draws walls from
RATED = {120.0: 48.0, 72.0: 24.0, 24.0: 8.0}   # N.m, peak -> continuous rating: AK80-64, AK60-39,
                                               # AK45-36 (CubeMars datasheets, urdf/README.md)
# --cond: what each condition scales. noise: every sensor, SLAM and map sigma (climb_task's
# module constants, so for the whole process); the rest the episode's own model
CONDS = ('noise', 'mass', 'torque', 'friction', 'lag', 'kp')   # kp: every servo's stiffness x
DISTRACT = ('above', 'below', 'long')   # --distract: rungs that take no part in the move
_NOISE = dict(SEN_SIG=dict(ct.SEN_SIG), SEN_BIAS=dict(ct.SEN_BIAS), SLAM_SIG=ct.SLAM_SIG,
              MAP_SIG=ct.MAP_SIG, MAP_ROT_SIG=ct.MAP_ROT_SIG, MAP_END_SIG=ct.MAP_END_SIG)


def conditions(s):
    """'torque=0.8,lag=20' -> {'torque': 0.8, 'lag': 20.0}: noise x, mass x (and inertia),
    torque limits x, sliding friction x, setpoint lag in ms"""
    out = {}
    for part in filter(None, (s or '').split(',')):
        k, _, v = part.partition('=')
        if k not in CONDS:
            raise SystemExit(f"--cond {k}: not one of {', '.join(CONDS)}")
        out[k] = float(v)
    return out


def scale_noise(x):
    """every noise magnitude of climb_task x `x`, from their values at import"""
    ct.SEN_SIG = {k: v * x for k, v in _NOISE['SEN_SIG'].items()}
    ct.SEN_BIAS = {k: v * x for k, v in _NOISE['SEN_BIAS'].items()}
    for k in ('SLAM_SIG', 'MAP_SIG', 'MAP_ROT_SIG', 'MAP_END_SIG'):
        setattr(ct, k, _NOISE[k] * x)


def perturb(task, cond):
    """the episode's model as `cond` says (`conditions`; noise is `scale_noise`'s)"""
    m = task.env.model
    task.peak0 = m.actuator_forcerange[:, 1].copy()   # the datasheet peaks, before --cond torque
    if 'mass' in cond:
        m.body_mass[:] *= cond['mass']
        m.body_inertia[:] *= cond['mass']
    if 'torque' in cond:
        m.actuator_forcerange[:] *= cond['torque']
        m.jnt_actfrcrange[:] *= cond['torque']
    if 'friction' in cond:
        m.geom_friction[:, 0] *= cond['friction']   # the full robot's contacts mix these...
        m.pair_friction[:, :2] *= cond['friction']  # ...the simplified robot's are explicit pairs
    if 'kp' in cond:                     # position servos: force = kp (ctrl - q) - kv qdot
        assert (m.actuator_biasprm[:, 1] == -m.actuator_gainprm[:, 0]).all(), "not position servos"
        m.actuator_gainprm[:, 0] *= cond['kp']
        m.actuator_biasprm[:, 1] *= cond['kp']
    if 'lag' in cond:
        task.lag = int(round(cond['lag'] / 1000 / m.opt.timestep))
        if task.lag > ct.FRAME_SKIP:     # ClimbTask holds the last setpoint only, one control step
            raise SystemExit(f"--cond lag {cond['lag']:g} ms: at most {ct.FRAME_SKIP * m.opt.timestep * 1000:g} ms")


class Lesson(gym.Env):
    """one benchmark lesson on the full robot, as a gym env (without its visual meshes
    unless asked: only rl.render draws them); or, with `xml` and `spec`, a generated wall
    starting from lesson `lesson`'s hang"""

    def __init__(self, lesson, mirror=False, visual=False, xml=None, spec=None, cond=None, torque=False, seed=None):
        self.task = ClimbTask(lesson, visual=visual, mirror=mirror, xml=xml, spec=spec)
        if seed is not None:                 # the sensor noise reproducible, and the same for every run compared
            self.task.rng = np.random.default_rng(seed)
        perturb(self.task, cond or {})
        self.tau = [] if torque else None    # each episode's actuator torques, a row a control step
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.task.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (17,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.tau is not None:
            self.tau.append([])
        return self.task.reset(seed)

    def step(self, a):
        out = self.task.step(a)
        if self.tau is not None:
            self.tau[-1].append(self.task.env.data.actuator_force.copy())
        return out


def torque_report(task, episodes):
    """{actuator: its peak and rms torque over `episodes` (lists of rows a control step), the
    share of steps over its continuous rating and the longest stretch over it (s)}"""
    m = task.env.model
    eps = [np.abs(np.asarray(e)) for e in episodes if len(e)]
    out = {}
    for i in range(m.nu):
        peak = float(getattr(task, 'peak0', m.actuator_forcerange[:, 1])[i])
        rated = RATED[peak]
        x = np.concatenate([e[:, i] for e in eps])
        longest = 0
        for e in eps:                        # the longest run of steps over the rating
            run_ = 0
            for over in e[:, i] > rated:
                run_ = run_ + 1 if over else 0
                longest = max(longest, run_)
        out[m.actuator(i).name[4:]] = dict(peak=peak, rated=rated, max=float(x.max()),
                                          rms=float(np.sqrt((x ** 2).mean())),
                                          over=float((x > rated).mean()), longest_s=longest * ct.CONTROL_DT)
    return out


def symmetry():
    """the robot's left side against its right, mirrored in y: each _L body's mass, principal
    inertia, centre of mass and frame against its _R one's, and its geoms' sizes and places --
    the full robot (the benchmark's) and the simplified one (training's)"""
    from .lessons import xml_path
    from .mjx_scene import c_path
    flip = np.array([1.0, -1.0, 1.0])
    for label, path in (('full', xml_path(1)), ('simplified', c_path(1))):
        m = mujoco.MjModel.from_xml_path(path)
        # the geoms that collide: by contype/conaffinity (the full robot) or by an explicit pair
        # (the simplified one); the full robot's visual meshes collide with nothing
        paired = set(m.pair_geom1) | set(m.pair_geom2)
        hits = lambda g: g in paired or m.geom_contype[g] or m.geom_conaffinity[g]
        print(f"{label} robot: body, |dmass| g, |dinertia| %, |dcom| mm, |dframe| mm, colliding geoms "
              f"L/R, worst geom place / size mm")
        for b in range(m.nbody):
            name = m.body(b).name
            if not name.endswith('_L'):
                continue
            r = m.body(name[:-2] + '_R').id
            di = np.abs(np.sort(m.body_inertia[b]) - np.sort(m.body_inertia[r])).max() / m.body_inertia[r].max()
            gb, gr = [[g for g in np.nonzero(m.geom_bodyid == x)[0] if hits(g)] for x in (b, r)]
            place = size = 0.0
            for g in gb:                     # each left geom against the right one of its type nearest its mirror
                same = [h for h in gr if m.geom_type[h] == m.geom_type[g]]
                if not same:
                    place = size = np.inf
                    continue
                h = min(same, key=lambda h: np.linalg.norm(m.geom_pos[g] * flip - m.geom_pos[h]))
                place = max(place, np.linalg.norm(m.geom_pos[g] * flip - m.geom_pos[h]))
                size = max(size, np.abs(np.sort(m.geom_size[g]) - np.sort(m.geom_size[h])).max())
            print(f"  {name[:-2]:>14} {1e3 * abs(m.body_mass[b] - m.body_mass[r]):8.1f} {100 * di:8.2f} "
                  f"{1e3 * np.linalg.norm(m.body_ipos[b] * flip - m.body_ipos[r]):8.2f} "
                  f"{1e3 * np.linalg.norm(m.body_pos[b] * flip - m.body_pos[r]):8.2f} "
                  f"{len(gb):>4}/{len(gr):<4} {1e3 * place:8.2f} / {1e3 * size:.2f}")
        names = [m.actuator(i).name for i in range(m.nu)]
        bad = [n for n in names if '_L' in n and not np.allclose(
            m.actuator_forcerange[names.index(n)], m.actuator_forcerange[names.index(n.replace('_L', '_R'))])]
        print(f"  actuator torque limits differing left/right: {bad or 'none'}")


def run(venv, episodes, policy=None):
    """episodes of one lesson; returns (successes, final distances, end reasons)"""
    obs = venv.reset()
    ok, dists, why = 0, [], []
    while len(why) < episodes:
        if policy is None:
            act = np.zeros((1, 17), np.float32)
        else:
            act, _ = policy.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(act)
        for info, done in zip(infos, dones):
            if done:
                why.append(info.get('reason', 'timeout'))
                dists.append(info['dist'])
                ok += info.get('reason') == 'success'
    return ok, dists, why


def wilson(k, n, z=1.96):
    """95 % Wilson score interval for k successes in n episodes -- what a success
    rate over a few dozen episodes is honestly worth"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def _seg_dist(p, c, a, h):
    """distance from points p (n, 3) to the segment centre c, axis a, half length h"""
    s = np.clip((p - c) @ a, -h, h)
    return np.linalg.norm(p - c - s[:, None] * a, axis=1)


def distract(lay, mode, rng, L=0.5, keep=0.25):
    """`lay` with rungs that take no part in its move (`--distract`), or None where they do not
    fit: 'above' a full row 0.45-0.65 m over the highest row and OUT further out (as a family-3
    row), 'below' one 0.90-1.30 m under the lowest (under the hanging robot), as far out as it; 'long' every rung end
    carried on by L where that stays `keep` from every other rung and FACE from every wall.
    rl.walls.Checker then keeps the robot's start and goal hangs clear of them, as for any wall"""
    from .walls import FACE, OUT, R
    rungs = list(lay['rungs'])
    along = [r for r in rungs if abs(r[2][1]) > 0.99]           # rows along y
    if mode == 'long':
        out = []
        for i, (nm, c, a, h, r) in enumerate(rungs):
            lo, hi = -h, h
            for sgn in (-1, 1):
                ext = c + a * (sgn * h + sgn * np.linspace(0.0, L, 11)[1:, None])
                near = min([_seg_dist(ext, o[1], o[2], o[3]).min() for j, o in enumerate(rungs) if j != i]
                           + [np.abs((ext - w[1]) @ w[2]).min() + keep - FACE + 1e-6 for w in lay['walls']] + [np.inf])
                if near >= keep:
                    lo, hi = (lo - L, hi) if sgn < 0 else (lo, hi + L)
            out.append((nm, c + a * (lo + hi) / 2, a, (hi - lo) / 2, r))
        return dict(lay, rungs=out) if any(o[3] > r[3] for o, r in zip(out, rungs)) else None
    if not along:
        return None
    top = max(along, key=lambda r: r[1][2]) if mode == 'above' else min(along, key=lambda r: r[1][2])
    y0 = min(r[1][1] - r[3] for r in along)
    y1 = max(r[1][1] + r[3] for r in along)
    dz = rng.uniform(0.45, 0.65) if mode == 'above' else -rng.uniform(0.90, 1.30)
    x = top[1][0] - (OUT if mode == 'above' else 0.0)
    c = np.array([x, (y0 + y1) / 2, top[1][2] + dz])
    return dict(lay, rungs=rungs + [(f'rung_g{len(rungs)}', c, np.array([0.0, 1.0, 0.0]), (y1 - y0) / 2, R)])


def held_out(run_dir, ck, state, families, n, levels, robot, device, jsonl, exact=None, cond=None,
             distract_mode=None):
    """`n` unseen walls of each family, one deterministic episode each (`--walls`); `exact`
    ({parameter: value}, --set): those parameters at that value, past training's range too;
    `distract_mode` (--distract): each wall with rungs that take no part in the move"""
    from . import walls as W
    from .mjx_scene import c_scene
    if state.get('families'):
        W.keep(state['families'])
    lv = W.last_levels() if levels == 'last' else \
        {int(f): v for f, v in state['curriculum']['levels'].items()}
    model = PPO.load(os.path.join(run_dir, f'{ck}model'), device=device)
    checker = W.Checker()
    print(f"{'family':>8} {'success':>12} {'95% CI':>12}   ends   ({n} walls, {robot} robot, {levels} levels)")
    for f in families:
        rng = np.random.default_rng(WALL_SEED + f)
        ok, why = 0, []
        if exact or distract_mode:       # the walls one by one: each must be usable as changed
            ints = {p[0]: isinstance(p[1], int) for p in W.FAMILIES[f]}
            ex = {k: int(v) if ints[k] else v for k, v in (exact or {}).items()}
            walls = []
            for _ in range(50 * n):
                lay = W.sample(f, lv[f], rng, ex)
                if distract_mode and checker.ok(lay):    # usable before, so the change is what differs
                    lay = distract(lay, distract_mode, rng)
                if lay is not None and checker.ok(lay):
                    walls.append(lay)
                if len(walls) == n:
                    break
            if len(walls) < n:
                raise SystemExit(f"family {f} at {ex} {distract_mode or ''}: {len(walls)} usable walls "
                                 f"in {50 * n} draws")
        else:
            walls = W.draw(checker, {f: 1.0}, lv, rng, n, probe_p=0.0)
        for i, lay in enumerate(walls):
            path, spec = c_scene(lay, robot)
            try:
                venv = DummyVecEnv([lambda: Lesson(lay['hang'], xml=path, spec=spec, cond=cond,
                                                   seed=WALL_SEED + 1000 * f + i)])
                venv = VecNormalize.load(os.path.join(run_dir, f'{ck}vecnorm.pkl'), venv)
                venv.training, venv.norm_reward = False, False
                k, _, w = run(venv, 1, model)
                venv.close()
            finally:
                os.remove(path)
            ok, why = ok + k, why + w
        tally = {r: why.count(r) for r in sorted(set(why))}
        lo, hi = wilson(ok, n)
        print(f"{f:>8} {ok:>6}/{n:<5} {lo:>5.0%}-{hi:<5.0%}  {tally}", flush=True)
        if jsonl:
            with open(jsonl, 'a') as fh:
                fh.write(json.dumps(dict(mode='walls', family=f, robot=robot, levels=levels,
                                         level=lv[f], set=exact or {}, cond=cond or {},
                                         distract=distract_mode,
                                         run=os.path.basename(run_dir),
                                         checkpoint=ck + 'model', timesteps=state['timesteps'],
                                         success=ok, episodes=n, ci95=(lo, hi), reasons=tally)) + '\n')


def open_loop(run_dir, ck, families, episodes, robot, device, jsonl, cond=None):
    """the reference climbs themselves as a controller (`--open-loop`): each climb of rl.rsi.REFS
    in `families` played open loop on its own wall from the hang -- a searched climb's setpoints,
    a recorded one's actions, then the last setpoints held --, against the policy's `episodes`
    on the same wall when `run_dir` is given. The climbs are open loop, so one episode is all of
    them; the sensor noise reaches only the policy"""
    from . import rsi
    from .mjx_scene import c_scene
    model = PPO.load(os.path.join(run_dir, f'{ck}model'), device=device) if run_dir else None
    print(f"{'climb':>14} {'open loop':>10} {'policy':>12}   ({robot} robot, {cond or 'nominal'})")
    for f, h, side, rows, name in rsi.REFS:
        if f not in families:
            continue
        z = np.load(os.path.join(rsi.HERE, 'fixtures', name + '.npz'))
        found = 'ctrl0' in z.files
        lay = dict(rsi.ref_wall(f, h, side or 1, rows), timeout=90.0)   # as rsi.replay: room to finish
        path, spec = c_scene(lay, robot)
        try:
            env = Lesson(lay['hang'], xml=path, spec=spec, cond=cond)
            env.reset(seed=0)
            t, why = env.task, None
            plan = z['ctrl'] if found else z['target'][:int(z['won'])]
            for k in range(int(lay['timeout'] / ct.CONTROL_DT) + 1):
                if k < len(plan):
                    a = np.clip((plan[k] - t.ctrl) / ct.ACT_SCALE, -1, 1) if found else plan[k]
                else:
                    a = np.zeros(17)                     # the last setpoints held
                _, _, te, tr, info = env.step(a)
                if te or tr:
                    why = info.get('reason', 'timeout')
                    break
            res = dict(mode='openloop', climb=name, family=f, height=h, side=side, rows=rows,
                       found=found, robot=robot, cond=cond or {}, openloop=why)
            row = f"{name:>14} {why:>10}"
            if model is not None:
                venv = DummyVecEnv([lambda: Lesson(lay['hang'], xml=path, spec=spec, cond=cond, seed=WALL_SEED)])
                venv = VecNormalize.load(os.path.join(run_dir, f'{ck}vecnorm.pkl'), venv)
                venv.training, venv.norm_reward = False, False
                k, _, w = run(venv, episodes, model)
                venv.close()
                res.update(run=os.path.basename(run_dir), checkpoint=ck + 'model', success=k,
                           episodes=episodes, ci95=wilson(k, episodes),
                           reasons={r: w.count(r) for r in sorted(set(w))})
                row += f" {k:>5}/{episodes:<6}"
            print(row, flush=True)
        finally:
            os.remove(path)
        if jsonl:
            with open(jsonl, 'a') as fh:
                fh.write(json.dumps(res) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='climber')
    ap.add_argument('--lessons', default=None, help='default: the unlocked pool')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--from-best', dest='from_best', action='store_true',
                    help='score best_model.zip instead of the latest checkpoint')
    ap.add_argument('--ckpt', default=None,
                    help='score this checkpoint prefix instead, e.g. snaps/000050000000_ '
                         '(rl.curve does, for every snapshot)')
    ap.add_argument('--jsonl', default=None,
                    help='also append one JSON line per lesson here (rl.plot reads it)')
    ap.add_argument('--mirror', action='store_true',
                    help='each lesson mirrored, toward the robot\'s right (rl.lessons.mirrored)')
    ap.add_argument('--walls', type=int, default=0,
                    help='score N unseen generated walls per family instead of the benchmark lessons')
    ap.add_argument('--levels', default='reached', choices=['reached', 'last'],
                    help="--walls: draw up to the checkpoint's ADR levels, or every family's last")
    ap.add_argument('--robot', default='full', choices=['full', 'simp'],
                    help='the full robot, or the simplified one training uses (both in C): --walls, '
                         '--open-loop, and the benchmark lessons (toward the left only)')
    ap.add_argument('--set', default=None,
                    help="--walls: 'param=value,...' fixed on every wall of the families in --lessons, "
                         "past training's last level too")
    ap.add_argument('--cond', default=None,
                    help="'noise=2,mass=1.1,torque=0.8,friction=0.7,lag=20': the robustness conditions "
                         "(scales; lag in ms)")
    ap.add_argument('--torque', action='store_true',
                    help="also report each joint's torque against its continuous rating")
    ap.add_argument('--distract', default=None, choices=DISTRACT,
                    help="--walls: each wall with rungs that take no part in the move (`distract`)")
    ap.add_argument('--open-loop', dest='open_loop', action='store_true',
                    help="play rl.rsi's reference climbs open loop on their walls, against the policy "
                         "(--name; '' for the climbs alone) on the same walls")
    ap.add_argument('--symmetry', action='store_true', help="compare the robot's left and right sides, and exit")
    a = ap.parse_args()
    if a.symmetry:
        return symmetry()
    cond = conditions(a.cond)
    scale_noise(cond.get('noise', 1.0))

    if a.open_loop and not a.name:
        return open_loop(None, '', [int(x) for x in (a.lessons or '1,2,3,4,5,6,7').split(',')],
                         a.episodes, a.robot, a.device, a.jsonl, cond)
    run_dir = os.path.join('out', a.name)
    ck = a.ckpt if a.ckpt is not None else ('best_' if a.from_best else '')
    state = json.load(open(os.path.join(run_dir, f'{ck}state.json')))
    pool = state['curriculum']['active'] if 'curriculum' in state else state['active']
    lessons = [lesson_key(x) for x in a.lessons.split(',')] if a.lessons else pool
    print(f"{run_dir}: {state['timesteps']} steps, pool {pool}\n")
    if a.open_loop:
        return open_loop(run_dir, ck, [int(x) for x in a.lessons.split(',')] if a.lessons else pool,
                         a.episodes, a.robot, a.device, a.jsonl, cond)
    if a.walls:
        fams = [int(x) for x in a.lessons.split(',')] if a.lessons else pool
        exact = {k: float(v) for k, _, v in (p.partition('=') for p in a.set.split(','))} if a.set else None
        return held_out(run_dir, ck, state, fams, a.walls, a.levels, a.robot, a.device, a.jsonl, exact, cond,
                        a.distract)
    if a.robot == 'simp' and a.mirror:
        raise SystemExit("--robot simp: the simplified robot's lesson scenes go toward the left only")
    from .mjx_scene import c_path
    xml_of = (lambda k: c_path(k)) if a.robot == 'simp' else (lambda k: None)
    if cond:
        print(f"conditions: {cond}")
    print(f"{'lesson':>24} {'policy':>18} {'do nothing':>18}")

    for k in lessons:
        row = []
        for policy_on in (True, False):
            venv = DummyVecEnv([lambda: Lesson(k, mirror=a.mirror, cond=cond, torque=a.torque and policy_on,
                                               xml=xml_of(k), seed=WALL_SEED + k)])
            venv = VecNormalize.load(os.path.join(run_dir, f'{ck}vecnorm.pkl'), venv)
            venv.training = False          # freeze the statistics; an eval that keeps
            venv.norm_reward = False       # updating them is scoring a moving target
            model = (PPO.load(os.path.join(run_dir, f'{ck}model'), env=venv,
                              device=a.device) if policy_on else None)
            # doing nothing is one episode: the reset is deterministic, and the SLAM
            # noise that makes the policy's episodes differ never reaches a zero action
            ok, dists, why = run(venv, a.episodes if policy_on else 1, model)
            row.append(f"{ok}/{len(why)} ok {np.mean(dists):.2f} m")
            if policy_on:
                tally = {r: why.count(r) for r in sorted(set(why))}
                res = dict(success=ok, episodes=len(why), ci95=wilson(ok, len(why)),
                           reasons=tally, dist=float(np.mean(dists)), cond=cond)
                if a.torque:
                    env = venv.venv.envs[0]
                    res['torque'] = torque_report(env.task, env.tau)
            else:
                res.update(control_success=ok, control_dist=float(np.mean(dists)))
            venv.close()
        lo, hi = res['ci95']
        print(f"{k:>2}{'m' if a.mirror else ' '} {LESSONS[k]['name']:>20} {row[0]:>18} "
              f"{row[1]:>18}   {tally}   95% CI {lo:.0%}-{hi:.0%}")
        for j, t in (res.get('torque') or {}).items():
            if t['over'] > 0:
                print(f"{'':>8}{j:>22}: max {t['max']:5.1f} rms {t['rms']:5.1f} of {t['rated']:g} rated "
                      f"({t['peak']:g} peak) N.m, over the rating {t['over']:.0%} of steps, "
                      f"{t['longest_s']:.1f} s at most in a row")
        if a.jsonl:
            with open(a.jsonl, 'a') as f:
                f.write(json.dumps(dict(lesson=k, name=LESSONS[k]['name'], mirror=a.mirror, robot=a.robot,
                                        run=a.name, checkpoint=ck + 'model',
                                        timesteps=state['timesteps'], **res)) + '\n')

    print("\n'm' is mean distance from both goals at the episode end -- lower is "
          "better; a lesson marked 'm' ran mirrored, toward the robot's right.\nThe "
          "policy has learned something only where it beats the control.")


if __name__ == '__main__':
    main()
