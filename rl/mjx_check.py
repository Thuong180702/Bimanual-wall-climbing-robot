"""Cross-checks: is the GPU path the same task, on the same robot, as the CPU path?

    python -m rl.mjx_check              # task, physics, model, policy -- a process each
    python -m rl.mjx_check task         # the JAX task against ClimbTask, state by state
    python -m rl.mjx_check physics      # MJX against C MuJoCo, same simplified robot
    python -m rl.mjx_check model        # the simplified robot against the full one
    python -m rl.mjx_check policy       # a trained policy: full robot, simplified in C, MJX
    python -m rl.mjx_check contacts     # which parts touch under a trained policy
    python -m rl.mjx_check policy --run out/g1   # another run's best checkpoint

Four things could each make a GPU-trained policy worthless on the CPU path, and
each gets its own check, because a failure in one is invisible to the others:

  task      the port. ClimbTask runs on C MuJoCo; at every control step its state
            is handed to the JAX task (`mjx.put_data`) with the same substep
            books and the same SLAM noise, and every observation, reward, book and
            termination must come out equal. float64, so equal means ~1e-9.
  physics   the engine. The same simplified robot in C MuJoCo and in MJX (float32,
            a few solver iterations): the secured hang, a still hang, a side pull
            along the rung, one control step from C's own state along the policy's
            gait (what caught MJX's capsule-box test inventing contacts), and how
            many contacts are really live at once against the MAX_CONTACTS the
            solver is given.
  model     the simplification. The full robot and the simplified one in C, at the
            same states along the policy's gait: what touches, how hard, and one
            control step from there. (What the first simplified robot failed.)
  policy    the robot. A trained policy (default: win17's best, learnt on the full
            robot) replayed on the full robot, the simplified one in C, and MJX.
            The model simplification and the engine are only good enough if it
            climbs as well on all three. After a GPU run, `--run out/<name>` asks
            the question the other way round: does what MJX taught hold on the
            full robot?
"""
import os, pickle, subprocess, sys, time
import numpy as np

# The policy the checks replay (--run to change): win17's best, 38.06M steps on the
# full robot, lessons -2..2. Frozen in the repo, so every machine -- the GPU box
# included -- replays the same one; out/win17 moves whenever that run is resumed.
RUN = os.path.join(os.path.dirname(__file__), 'fixtures', 'win17')


def _task(k, xml=None, seed=0):
    """a C ClimbTask whose SLAM noise comes from `seed`. Every check here is a function
    of the code: unseeded, a policy's gait -- and what the model check counts along
    it -- came out different on every run (lesson 1's wrist on the rung: 2 % one run,
    9 % the next)."""
    from .climb_task import ClimbTask
    t = ClimbTask(k, visual=False, xml=xml)
    t.env._slam_rng = np.random.default_rng(seed)
    return t


# where rl.climb_task._obs puts each hand's _near_rungs block (20 each): after q, dq, tau
# and ctrl (4 x 17), the IMU (9), each hand's F/T and pad (2 x 7) and the goals (6)
NEAR = (4 * 17 + 9 + 2 * 7 + 6, 4 * 17 + 9 + 2 * 7 + 6 + 20)


def _fromto_axes(o):
    """the observation as a checkpoint trained on the lesson scenes before
    rl.lessons.canon_axis saw it: every rung's axis the way `fromto` pointed it, toward
    the rung's lower end -- the canonical one negated -- and the distances to its two
    ends the other way round. win17, the fixture, is one."""
    o = np.array(o, np.float32)
    for n0 in NEAR:
        for j in (3, 9, 15):                     # the three nearest rungs' axes
            o[..., n0 + j:n0 + j + 3] *= -1
        o[..., [n0 + 18, n0 + 19]] = o[..., [n0 + 19, n0 + 18]]
    return o


def _policy(run=None, best=True):
    """(predict(obs batch) -> actions, pool) for a checkpoint, or None if absent"""
    run = run or RUN
    ck = 'best_' if best else ''
    p = os.path.join(run, ck + 'model.zip')
    if not os.path.exists(p):
        return None
    import json
    from stable_baselines3 import PPO
    model = PPO.load(p, device='cpu')
    with open(os.path.join(run, ck + 'vecnorm.pkl'), 'rb') as f:
        vn = pickle.load(f)
    state = json.load(open(os.path.join(run, ck + 'state.json')))
    # generated walls (rl.walls) were canonical from the start; lesson scenes were not
    # until the checkpoint says so
    old = 'axes' not in state and state.get('walls_mode', 'lessons') == 'lessons'
    see = _fromto_axes if old else (lambda o: np.asarray(o, np.float32))
    from .lessons import lesson_key
    return (lambda o: model.predict(vn.normalize_obs(see(o)), deterministic=True)[0]), \
        [lesson_key(k) for k in state['active']]


# --- task ---------------------------------------------------------------------------
def check_task(lessons=(1, 1.5, 7), steps=260):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jp
    from mujoco import mjx
    from .climb_task import ClimbTask
    from .mjx_scene import c_path
    from .mjx_task import MjxClimb, LESSON_KEYS, REASONS

    print(f"task: the JAX task on {jax.devices()[0].device_kind}, float64 -- compiling first")
    task = MjxClimb()
    start = jax.jit(task.start)
    update = jax.jit(task.update)
    pol = _policy()
    pol = pol and pol[0]
    worst = {}

    def close(name, a, b, tol=1e-6):
        a, b = np.asarray(a, float), np.asarray(b, float)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst[name] = max(worst.get(name, 0.0), err)
        return err <= tol * max(1.0, float(np.max(np.abs(b))) if b.size else 1.0)

    for k in lessons:
        li = LESSON_KEYS.index(k)
        t = _task(k, c_path(k))
        rec = {}
        slam, sense = t.env.slam, t._sense

        def slam_rec():
            p, R = slam()
            rec['noise'] = p - t.env.data.xpos[t.env.base]
            return p, R

        def sense_rec():
            rec['w'] = t._w_sum.copy()
            return sense()

        t.env.slam, t._sense = slam_rec, sense_rec
        mc = t.env.model
        T = None

        def begin():
            nonlocal T
            o, _ = t.reset()
            T = task.tables._replace(tau_idle=task.tables.tau_idle.at[li].set(t.tau_idle),
                                     fight_idle=task.tables.fight_idle.at[li].set(t.fight_idle))
            st, oj = start(T, li, mjx.put_data(mc, t.env.data), jp.asarray(t.ctrl),
                           jp.asarray(t.env._no_grip_t), jax.random.PRNGKey(0),
                           jp.asarray(rec['noise']))
            assert close('obs', oj, o, 1e-5), f"lesson {k}: reset observation differs"
            return st, o

        st, o = begin()
        rng = np.random.default_rng(1)
        n, ends, grabs, banks, switches = 0, [], 0, 0, 0
        for i in range(steps):
            # hang, swing with the fingers shut, let the left hand go and re-close it,
            # then the policy if there is one, else thrash -- every book gets written
            if pol and i >= 120:
                a = pol(o[None])[0]
            elif i < 20:
                a = np.zeros(17)
            elif i < 70:
                a = rng.uniform(-0.5, 0.5, 17)
                a[[8, 16]] = -1.0
            elif i < 120:
                a = rng.uniform(-0.3, 0.3, 17)
                a[8] = 1.0 if i < 95 else -1.0
                a[16] = -1.0
            else:
                a = rng.uniform(-1, 1, 17)
            anchor = t.anchor
            o, r, te, tr, info = t.step(a)
            st = st._replace(d=mjx.put_data(mc, t.env.data), ctrl=jp.asarray(t.ctrl),
                             no_grip_t=jp.asarray(t.env._no_grip_t))
            st, oj, rj, tej, trj, ij = update(T, st, jp.asarray(a), t._flight,
                                             jp.asarray(rec['w']), jp.asarray(rec['noise']))
            n += 1
            why = REASONS[int(ij['reason'])]
            ok = [close('obs', oj, o, 1e-5), close('reward', rj, r),
                  close('hold points', np.asarray(st.hold_pt), [t.hold[s][1] for s in 'LR']),
                  close('attitude', np.asarray(st.Rf), t._Rf),
                  close('bank', ij['bank'], info['bank']), close('slide', ij['slide'], info['slide']),
                  close('credit', ij['credit'], info['credit']),
                  close('pending', ij['banked'], info['banked']),
                  close('touch', ij['touch'], info['touch'], 1e-5),
                  close('effort', ij['effort'], info['effort'], 1e-5),
                  close('fight', ij['fight'], info['fight'], 1e-5),
                  int(st.anchor) == 'LR'.index(t.anchor),
                  list(np.asarray(ij['grabs'])) == list(info['grabs']),
                  float(ij['flight']) == info['flight'], float(ij['parked']) == info['parked'],
                  float(ij['stale']) == info['stale'], bool(tej) == te, bool(trj) == tr,
                  why == info.get('reason', '')]
            assert all(ok), (f"lesson {k} step {i}: JAX and ClimbTask disagree "
                             f"{[j for j, x in enumerate(ok) if not x]}; worst so far {worst}")
            grabs += int(info['first'])
            banks += info['bank'] != 0.0
            switches += t.anchor != anchor
            if te or tr:
                ends.append(info['reason'])
                st, o = begin()
        print(f"task: lesson {k}: {n} steps equal (ends {ends or 'none'}, {banks} banked "
              f"steps, {switches} anchor switches, first grabs {grabs})")
    print("task: largest differences " + ", ".join(f"{a} {b:.1e}" for a, b in worst.items()))
    print("task: OK -- the JAX task is ClimbTask's, observation by observation")


# --- physics ------------------------------------------------------------------------
def check_physics(lessons=(1, 2, 7, 8)):
    import jax, jax.numpy as jp
    from mujoco import mjx
    from . import climb_task as ct
    from .climb_task import ClimbTask
    from .mjx_scene import c_path, MAX_CONTACTS, ITERATIONS
    from .mjx_task import MjxClimb, LESSON_KEYS

    print(f"physics: MJX on {jax.devices()[0].device_kind}, one robot at a time -- compiling first")
    task = MjxClimb()
    step = jax.jit(task.step)
    reset = jax.jit(task.reset)
    z = jp.zeros(17)
    for k in lessons:
        li = LESSON_KEYS.index(k)
        t = _task(k, c_path(k))
        t.reset()
        st, _ = reset(task.tables, li, jax.random.PRNGKey(0))
        dq = float(np.abs(np.asarray(st.d.qpos) - t.env.data.qpos).max())
        c0 = [t._contact(s)[0] for s in 'LR']
        j0 = np.asarray(task._contact(st.d)[0])
        base_c, base_j = t.env.data.xpos[t.env.base].copy(), np.asarray(st.d.xpos[task.base])
        for _ in range(50):                                   # 2 s of hanging still
            t.step(np.zeros(17))
            st, *_ = step(task.tables, st, z)
        cs = [t._contact(s)[0] for s in 'LR']
        js = np.asarray(task._contact(st.d)[0])
        drift_c = np.linalg.norm(t.env.data.xpos[t.env.base] - base_c) * 1e3
        drift_j = np.linalg.norm(np.asarray(st.d.xpos[task.base]) - base_j) * 1e3
        print(f"physics: lesson {k:>4}: secured hang {dq*1e3:.2f} mrad/mm from C; hook loads "
              f"C {c0[0]:.0f}/{c0[1]:.0f} MJX {j0[0]:.0f}/{j0[1]:.0f} N, after 2 s C "
              f"{cs[0]:.0f}/{cs[1]:.0f} MJX {js[0]:.0f}/{js[1]:.0f} N; base drift C "
              f"{drift_c:.1f} MJX {drift_j:.1f} mm")
        assert dq < 0.02, f"lesson {k}: MJX's secured hang is {dq:.3f} from C's"
        assert (js > 50).all() and np.allclose(js, cs, rtol=0.25), \
            f"lesson {k}: MJX hangs on {js} N against C's {cs}"
        assert drift_j < 5.0, f"lesson {k}: MJX drifted {drift_j:.1f} mm hanging still"

    # the side pull climb_task's slide test uses: 0.23 x body weight along the rung
    k, li = 1, LESSON_KEYS.index(1)
    t = _task(k, c_path(k))
    t.reset()
    st, _ = reset(task.tables, li, jax.random.PRNGKey(0))
    ax = np.array([0.0, 1.0, 0.0])
    f = np.zeros(6)
    f[:3] = 0.23 * t.weight * ax
    p0c = t._throat_true('R').copy()
    p0j = np.asarray(task._throat_true(st.d))[1]
    t.env.data.xfrc_applied[t.env.base] = f
    st = st._replace(d=st.d.replace(xfrc_applied=st.d.xfrc_applied.at[task.base].set(jp.asarray(f))))
    for _ in range(150):
        t.step(np.zeros(17))
        st, *_ = step(task.tables, st, z)
    mc_ = (t._throat_true('R') - p0c) @ ax * 1e3
    mj_ = (np.asarray(task._throat_true(st.d))[1] - p0j) @ ax * 1e3
    print(f"physics: {0.23 * t.weight:.0f} N along the rung for 6 s slides the loaded hook "
          f"C {mc_:+.1f} mm, MJX {mj_:+.1f} mm")
    assert abs(mj_) < 60 and abs(mj_ - mc_) < max(10.0, 0.5 * abs(mc_)), \
        "the hook slides differently in MJX: the pad or the solver is off"

    # one control step from the SAME state: C's state under the policy's gait, handed
    # to MJX, both stepped 20 substeps, compared. This is what showed MJX's
    # capsule-box test inventing 50-75 mm contacts (rl.mjx_scene): p90 joint error
    # 0.16, contact forces 5000 N off. A trajectory comparison cannot see it -- two
    # simulations of a swinging robot part within a second whatever is wrong.
    pol = _policy()
    k, li = 1, LESSON_KEYS.index(1)
    t = _task(k, c_path(k))
    m_env = task._env_model(task.wall(task.tables, li))

    @jax.jit
    def one(d, ctrl):
        def sub(d, _):
            return mjx.step(m_env, d.replace(ctrl=ctrl)), None
        return jax.lax.scan(sub, d, None, length=20)[0]

    rng = np.random.default_rng(0)
    o, _ = t.reset()
    eq, ef = [], []
    for i in range(300):
        a = pol[0](o[None])[0] if pol else rng.uniform(-0.5, 0.5, 17)
        d0 = mjx.put_data(t.env.model, t.env.data)
        ctrl = np.clip(t.ctrl + ct.ACT_SCALE * np.clip(a, -1, 1), task.lo, task.hi)
        o, r, te, tr, _ = t.step(a)
        dj = one(d0, jp.asarray(ctrl, jp.float32))
        eq.append(np.abs(np.asarray(dj.qpos) - t.env.data.qpos).max())
        ef.append(np.abs(np.asarray(task._contact(dj)[0]) - [t._contact(s)[0] for s in 'LR']).max())
        if te or tr:
            o, _ = t.reset()
    eq, ef = np.array(eq), np.array(ef)
    print(f"physics: one control step from C's own state, {len(eq)} steps of "
          f"{'the policy' if pol else 'random actions'}: joints off by p50 {np.median(eq):.1e} "
          f"p90 {np.percentile(eq, 90):.1e} max {eq.max():.1e}; hook loads p50 "
          f"{np.median(ef):.2f} p90 {np.percentile(ef, 90):.2f} max {ef.max():.0f} N")
    assert np.percentile(eq, 90) < 1e-3 and np.percentile(ef, 90) < 5.0, \
        "MJX and C part in one control step from the same state: a collision or the solver differs"

    # contacts live at once, measured in C where nothing caps them
    seen = []
    for k in (lessons if pol is None else pol[1]):
        t = _task(k, c_path(k))
        o, _ = t.reset()
        rng = np.random.default_rng(0)
        for i in range(400):
            a = pol[0](o[None])[0] if pol else rng.uniform(-1, 1, 17)
            o, r, te, tr, _ = t.step(a)
            seen.append(t.env.data.ncon)
            if te or tr:
                o, _ = t.reset()
    seen = np.array(seen)
    print(f"physics: live contacts over {len(seen)} steps ({'policy' if pol else 'random'}): "
          f"p50 {np.percentile(seen, 50):.0f}, p99 {np.percentile(seen, 99):.0f}, max "
          f"{seen.max()} -- MJX's solver takes the deepest {MAX_CONTACTS}, "
          f"{ITERATIONS} Newton iterations")
    assert seen.max() <= MAX_CONTACTS, "more contacts at once than MJX keeps: raise MAX_CONTACTS"

    # A blown-up env is an episode's failure, never a NaN handed to the learner: one
    # NaN reward poisons PPO's advantages, one NaN observation VecNormalize's
    # statistics, and either ends the run. Physics NaN in env 0, a NaN attitude
    # estimate (a zero accelerometer's triad) in env 1, env 2 healthy.
    from .mjx_task import Batch, REASONS
    b = Batch(3)
    b.reset()
    d = b.state.d
    b.state = b.state._replace(d=d.replace(qvel=d.qvel.at[0, 0].set(jp.nan)),
                               Rf=b.state.Rf.at[1].set(jp.nan))
    z3 = np.zeros((3, 17), np.float32)
    obs, end, r, term, trunc, info = b.step(z3)
    why = [REASONS[int(x)] for x in np.asarray(info['reason'])]
    assert why == ['blowup', 'blowup', ''] and np.allclose(np.asarray(r)[:2], ct.FAIL_R), (why, r)
    for _ in range(3):
        assert all(np.isfinite(np.asarray(x)).all() for x in (obs, end, r)), "a NaN got out"
        obs, end, r, term, trunc, info = b.step(z3)
    print("physics: a NaN state or estimate ends as 'blowup' at FAIL_R, finite, and resets")
    print("physics: OK")


# --- model --------------------------------------------------------------------------
def check_model(lessons=(1, 1.85, 2), episodes=3, below=(0.35, 0.45, 0.55)):
    """The simplified robot against the FULL one, at the same states. The full robot
    runs; at every control step its state is copied into the simplified robot (both in
    C MuJoCo) and compared two ways: which body pairs touch there and how hard, and
    where one control step from that state takes each. `physics` cannot see any of
    this -- it runs the simplified robot on both engines.

    Two sources of states. win17's gait on `lessons`, the hooks' work. And the robot
    hanging over a full row `below` metres under its hooks, moving its arms at random:
    the posture of every climb up, down or over, which rests a row against the arms,
    and which no single-row gait reaches -- the first simplified robot's arms went
    through rungs and this check, on lessons 1..2 alone, could not tell.

    This is the check the first simplified robot lacked, and it failed it: the box
    round the body's chamfered hull stood 9 mm proud of it into the wall, the motor
    boxes 9 mm into each other when the hands met, and hook plates drawn as two 6 mm
    rods lost the load at the plate edges and the rung ends. One control step then
    moved the joints 1.1e-2 rad and the body 8 mm from the full robot's (lesson 1), and
    win17 timed out on the wide gap where the full robot crossed it."""
    import collections, mujoco
    from . import climb_task as ct
    from .lessons import LESSONS, OUT, _ry, scene_xml
    from .mjx_scene import ARM, build, c_path, lesson_layout

    pol = _policy()

    def pairs(t):
        m, d = t.env.model, t.env.data
        name = lambda g: ('rung' if (m.geom(g).name or '').startswith('rung_') else
                          'wall' if m.geom_bodyid[g] == 0 else m.body(m.geom_bodyid[g]).name)
        out, f6 = collections.defaultdict(float), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f6)
            out[tuple(sorted((name(c.geom1), name(c.geom2))))] += abs(f6[0])
        return out

    ft = np.zeros(6)

    def gap(t, geoms, rung):
        return min(mujoco.mj_geomDistance(t.env.model, t.env.data, g, rung, 0.05, ft) for g in geoms)

    def compare(label, F, S, act, steps=None, arms=None):
        """`arms`: a rung's name. Its contacts with the arm are judged by surface, not by
        touch/no-touch: a row grazed by arms moving at random flips on a millimetre, far
        below the contact's own softness (a loaded hook sinks 1-3 mm), so the stand-ins'
        surface is measured against the real links' wherever either is within 30 mm"""
        rng = np.random.default_rng(0)
        step, touch, n = [], collections.defaultdict(lambda: np.zeros(5)), 0
        err = collections.defaultdict(list)
        if arms:
            mf, msm = F.env.model, S.env.model
            links = {f'{b}_{s}': ([g for g in range(mf.ngeom) if mf.geom_bodyid[g] == mf.body(f'{b}_{s}').id
                                   and mf.geom_contype[g]],
                                  [g for g in range(msm.ngeom) if (msm.geom(g).name or '').startswith(f'{b}_{s}:arm')])
                     for b in ARM for s in 'LR'}
            rf, rs = mf.geom(arms).id, msm.geom(arms).id
        for _ in range(episodes):
            o, _ = F.reset()
            a = np.zeros(17)
            for i in range(steps or 10 ** 9):
                a = act(o, a, rng)
                for f in ('qpos', 'qvel', 'act', 'qacc_warmstart', 'ctrl'):
                    getattr(S.env.data, f)[:] = getattr(F.env.data, f)
                S.env.data.time = F.env.data.time
                mujoco.mj_forward(S.env.model, S.env.data)
                mujoco.mj_forward(F.env.model, F.env.data)
                pf, ps = pairs(F), pairs(S)
                for key in set(pf) | set(ps):
                    touch[key] += [key in pf and key in ps, key in pf and key not in ps,
                                   key in ps and key not in pf, pf.get(key, 0.0), ps.get(key, 0.0)]
                n += 1
                if arms:
                    for link, (gf, gs) in links.items():
                        df, ds = gap(F, gf, rf), gap(S, gs, rs)
                        if min(df, ds) < 0.03:
                            err[link.rsplit('_', 1)[0]].append((ds - df) * 1e3)
                ctrl = np.clip(F.ctrl + ct.ACT_SCALE * np.clip(a, -1, 1), F.lo, F.hi)
                o, r, te, tr, info = F.step(a)
                for _ in range(ct.FRAME_SKIP):
                    S.env.step(ctrl)
                step.append((np.abs(S.env.data.qpos[7:] - F.env.data.qpos[7:]).max(),
                             np.linalg.norm(S.env.data.qpos[:3] - F.env.data.qpos[:3]) * 1e3,
                             max(abs(F._contact(s)[0] - S._contact(s)[0]) for s in 'LR')))
                if te or tr:
                    break
        q, x, f = np.array(step).T
        print(f"model: {label}, {n} states: one control step from them, joints p50 "
              f"{np.median(q):.1e} p90 {np.percentile(q, 90):.1e} rad, body p50 "
              f"{np.median(x):.2f} p90 {np.percentile(x, 90):.2f} mm, hook loads p50 "
              f"{np.median(f):.1f} p90 {np.percentile(f, 90):.1f} N")
        for key, (b, fo, so, ff, fs) in sorted(touch.items(), key=lambda kv: -kv[1][:3].sum()):
            if b + fo + so >= 0.01 * n:
                arm = arms and 'rung' in key and any(k.rsplit('_', 1)[0] in ARM for k in key)
                print(f"         {key[0] + ' - ' + key[1]:>31}: both {b / n:5.1%}, only full "
                      f"{fo / n:5.1%}, only simplified {so / n:5.1%}; mean N {ff / n:6.1f} "
                      f"full, {fs / n:6.1f} simplified" + ('  (judged by surface, below)' if arm else ''))
                assert arm or (fo <= 0.08 * n and so <= 0.08 * n), (
                    f"{label}: {key} touches on one robot and not the other in "
                    f"{max(fo, so) / n:.0%} of the same states")
        for link, v in sorted(err.items()):
            v = np.array(v)
            p10, p50, p90 = np.percentile(v, [10, 50, 90])
            print(f"         {link:>14} stand-in surface vs the real link near the rung: "
                  f"p10 {p10:+5.1f} p50 {p50:+5.1f} p90 {p90:+5.1f} mm ({len(v)} states)")
            # unbiased to 5 mm, and no stand-in off by more than a rung and a half in its
            # worst tenth -- an arm with no stand-in misses by the whole arm. 5, not the
            # 3 first guessed: the upper arm's motor is a disc, 98 mm across and 62 thick,
            # and no shape MJX collides exactly with a rung fits one closer (a capsule,
            # a ball and rings of rods were each measured: 4.4 mm is the best of them)
            assert abs(p50) <= 5.0 and -25.0 <= p10 and p90 <= 25.0, (
                f"{label}: the {link} stand-in is {p50:+.1f} mm (p50), {p10:+.1f}..{p90:+.1f} "
                f"(p10..p90) off the real link near the rung")
        assert np.median(q) < 5e-3 and np.median(x) < 0.5, (
            f"{label}: one control step from the same state takes the simplified robot "
            f"{np.median(q):.1e} rad / {np.median(x):.2f} mm from the full one (p50)")

    gait = lambda o, a, rng: pol[0](o[None])[0] if pol else rng.uniform(-0.5, 0.5, 17)
    for k in lessons:
        compare(f"lesson {k:>4}, the full robot's gait", _task(k), _task(k, c_path(k)), gait)

    # a row under the hang: lesson 1 with one more full row dz lower, both robots
    wander = lambda o, a, rng: np.clip(0.8 * a + 0.6 * rng.normal(0, 1, 17), -1, 1)
    os.makedirs(OUT, exist_ok=True)
    for dz in below:
        spec = dict(LESSONS[1], title=f'lesson 1 with a row {dz} m under the hang',
                    rungs=LESSONS[1]['rungs'] + [_ry('below', -0.90, 1.50, 1.0 - dz)])
        fp = os.path.join(OUT, f'check_below{dz:.2f}.xml')
        open(fp, 'w').write(scene_xml(spec, f'check_below{dz:.2f}'))
        lay = lesson_layout(1)
        lay['rungs'] = lay['rungs'] + [('rung_below', np.array([0.44, 0.3, 1.0 - dz]),
                                        np.array([0.0, 1.0, 0.0]), 1.2, 0.015)]
        lay['lesson'] = 1
        sp = os.path.join(OUT, f'check_below{dz:.2f}_mjx.xml')
        open(sp, 'w').write(build(lay).to_xml())
        compare(f"a row {dz} m under the hang, the arms moving at random",
                _task(1, fp), _task(1, sp), wander, steps=250, arms='rung_below')
    print("model: OK -- the simplified robot touches what the full one touches, where it "
          "does, and moves as it moves")


# --- policy -------------------------------------------------------------------------
def check_policy(lessons=None, episodes=8, run=None):
    from .climb_task import ClimbTask
    from .mjx_scene import c_path
    from .lessons import LESSONS
    from .mjx_task import Batch, REASONS

    run = run or RUN
    pol = _policy(run)
    if pol is None:
        print(f"policy: no checkpoint at {run} -- skipped")
        return
    predict, pool = pol
    lessons = lessons or pool
    batch = Batch(episodes)

    def c_run(k, xml):
        t = _task(k, xml)
        ok, lens = 0, []
        for _ in range(episodes):
            o, _ = t.reset()
            while True:
                o, r, te, tr, info = t.step(predict(o[None])[0])
                if te or tr:
                    ok += info.get('reason') == 'success'
                    lens.append(info['steps'])
                    break
        return ok, np.mean(lens)

    def j_run(k):
        batch.set_probs({k: 1.0})
        o = np.asarray(batch.reset())
        ok, lens, done = 0, [], np.zeros(episodes, bool)
        while not done.all():
            o, _, _, term, trunc, info = batch.step(predict(o))
            o = np.asarray(o)
            end = np.asarray(term | trunc) & ~done
            for i in np.nonzero(end)[0]:
                ok += REASONS[int(info['reason'][i])] == 'success'
                lens.append(int(info['steps'][i]))
            done |= end
        return ok, np.mean(lens)

    print(f"policy: {run} (best), {episodes} deterministic episodes a lesson\n"
          f"  {'lesson':>28} {'full robot, C':>16} {'simplified, C':>16} {'simplified, MJX':>16}")
    tot = np.zeros(3)
    for k in lessons:
        t0 = time.perf_counter()
        res = [c_run(k, None), c_run(k, c_path(k)), j_run(k)]
        tot += [r[0] for r in res]
        print(f"  {k:>6} {LESSONS[k]['name']:>21} "
              + " ".join(f"{ok:>3}/{episodes} {ln:6.0f} st" for ok, ln in res)
              + f"   ({time.perf_counter() - t0:.0f} s)", flush=True)
    n = episodes * len(lessons)
    print(f"  {'total':>28} " + " ".join(f"{int(x):>4}/{n:<11}" for x in tot))
    # both ways round: a policy learnt on MJX that only climbs there is the failure a
    # GPU run could hide, and one learnt on the full robot that fails on MJX is the
    # simplification changing the task
    assert tot.max() - tot.min() <= 0.2 * tot.max() + 2, (
        "the same policy climbs clearly better on one of the three: the simplified "
        "robot or MJX is not the full robot's task")
    print("policy: OK -- the full robot, the simplified one in C and MJX climb alike")


# --- contacts -----------------------------------------------------------------------
def check_contacts(lessons=None, episodes=2):
    import collections, mujoco
    from .climb_task import ClimbTask
    from .mjx_scene import c_path
    pol = _policy()
    if pol is None:
        print("contacts: no checkpoint -- skipped")
        return
    predict, pool = pol
    for label, xml in (('full robot', None), ('simplified', c_path)):
        stats, n, f6 = collections.defaultdict(lambda: [0, 0.0]), 0, np.zeros(6)
        for k in lessons or pool:
            t = _task(k, xml(k) if xml else None)
            m, d = t.env.model, t.env.data
            name = lambda g: (m.geom(g).name.split('_')[0] if m.geom_bodyid[g] == 0
                              else m.body(m.geom_bodyid[g]).name)
            for _ in range(episodes):
                o, _ = t.reset()
                while True:
                    o, r, te, tr, info = t.step(predict(o[None])[0])
                    seen = {}
                    for i in range(d.ncon):
                        c = d.contact[i]
                        mujoco.mj_contactForce(m, d, i, f6)
                        key = tuple(sorted((name(c.geom1), name(c.geom2))))
                        seen[key] = seen.get(key, 0.0) + abs(f6[0])
                    for key, f in seen.items():
                        stats[key][0] += 1
                        stats[key][1] += f
                    n += 1
                    if te or tr:
                        break
        print(f"contacts, {label} ({n} control steps):")
        for key, (c, f) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            if c / n >= 0.001:
                print(f"  {key[0]:>16} - {key[1]:<16} {c / n * 100:6.2f} % of steps, mean {f / c:6.1f} N")


CHECKS = dict(task=check_task, physics=check_physics, model=check_model,
              policy=check_policy, contacts=check_contacts)

if __name__ == '__main__':
    which = sys.argv[1:]
    if '--run' in which:
        i = which.index('--run')
        RUN = which[i + 1]
        del which[i:i + 2]
    which = which or ['task', 'physics', 'model', 'policy']
    if len(which) == 1:
        CHECKS[which[0]]()
    else:
        # one process each: the task check runs JAX in float64, the others as training does
        for w in which:
            r = subprocess.run([sys.executable, '-m', 'rl.mjx_check', w, '--run', RUN])
            if r.returncode:
                raise SystemExit(f"rl.mjx_check {w} FAILED")
