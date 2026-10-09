"""Cross-checks: is the GPU path the same task, on the same robot, as the CPU path?

    python -m rl.mjx_check              # task, physics, model, policy -- a process each
    python -m rl.mjx_check task         # the JAX task against ClimbTask, state by state
    python -m rl.mjx_check physics      # MJX against C MuJoCo, same simplified robot
    python -m rl.mjx_check model        # the simplified robot against the full one
    python -m rl.mjx_check policy       # a trained policy: full robot, simplified in C, MJX
    python -m rl.mjx_check contacts     # which parts touch under a trained policy
    python -m rl.mjx_check policy --run out/g1   # another run's best checkpoint
    python -m rl.mjx_check warp --out f               # is MJX's Warp backend right? ('warp' / 'jax')
    python -m rl.mjx_check warp --speed --envs 4096 --out f   # ...and faster at 4096 envs

Four things could each make a GPU-trained policy worthless on the CPU path, and
each gets its own check, because a failure in one is invisible to the others:

  task      the port. ClimbTask runs on C MuJoCo; at every control step its state
            is handed to the JAX task (`mjx.put_data`) with the same substep
            books and the same noise draws (sensors, SLAM, map), and every observation, reward, book and
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
  warp      the backend. MJX's Warp implementation, which only tests the pairs its
            broadphase finds near, against C by physics' own measures, and a
            policy's episodes on it against the JAX backend's; then both timed. Not
            in the default set: gpu_run.sh runs it to choose what training steps on.
"""
import os, pickle, subprocess, sys, time
import numpy as np

# The policy the checks replay (--run to change): win17's best, 38.06M steps on the
# full robot, lessons -2..2. Frozen in the repo, so every machine -- the GPU box
# included -- replays the same one; out/win17 moves whenever that run is resumed.
RUN = os.path.join(os.path.dirname(__file__), 'fixtures', 'win17')
# ...and one learnt on what training runs now, the generated walls (rl.walls): gpu4's
# best, 61.3M steps, families 1 and 2 at 85-90 %, family 3's rl.rsi starts at ~60 %
# (its Adam moments and last rollout dropped: the policy alone, the same actions)
WALLS_RUN = os.path.join(os.path.dirname(__file__), 'fixtures', 'gpu4')
# m an rl.rsi start may see a hook from where it is: the 0.20 m climb from the hang sees
# its free hook 28-44 mm off around the catch, 60 at most (out/gpu5_diag/ahrs_path.py);
# its starts, 72 at most since 2026-10-09, up to 0.3 m before
RSI_SEEN = 0.1


def _task(k, xml=None, seed=0):
    """a C ClimbTask whose noise comes from `seed`. Every check here is a function
    of the code: unseeded, a policy's gait -- and what the model check counts along
    it -- came out different on every run (lesson 1's wrist on the rung: 2 % one run,
    9 % the next)."""
    from .climb_task import ClimbTask
    t = ClimbTask(k, visual=False, xml=xml)
    t.rng = np.random.default_rng(seed)
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
    from .mjx_task import MjxClimb, LESSON_KEYS, REASONS, N_RUNG

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
        sense = t._sense

        def sense_rec():
            rec['w'] = t._w_sum.copy()
            return sense()

        t._sense = sense_rec
        mc = t.env.model
        T = None

        def begin():
            nonlocal T
            o, _ = t.reset()
            T = task.tables._replace(tau_idle=task.tables.tau_idle.at[li].set(t.tau_idle),
                                     fight_idle=task.tables.fight_idle.at[li].set(t.fight_idle))
            ep = dict(t.ep_z, map=np.pad(t.ep_z['map'], ((0, N_RUNG - len(t.segs)), (0, 0))))
            st, oj = start(T, li, mjx.put_data(mc, t.env.data), jp.asarray(t.ctrl),
                           jp.asarray(t.env._no_grip_t), jax.random.PRNGKey(0), ep, t.z)
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
                                             jp.asarray(rec['w']), t.z)
            n += 1
            why = REASONS[int(ij['reason'])]
            ok = [close('obs', oj, o, 1e-5), close('reward', rj, r),
                  close('hold points', np.asarray(st.hold_pt), [t.hold[s][1] for s in 'LR']),
                  np.array_equal(np.asarray(st.seen)[:len(t.segs)], t.seen),
                  close('attitude', np.asarray(st.Rf), t._Rf),
                  close('bank', ij['bank'], info['bank']), close('slide', ij['slide'], info['slide']),
                  close('credit', ij['credit'], info['credit']),
                  close('pending', ij['banked'], info['banked']),
                  close('touch', ij['touch'], info['touch'], 1e-5),
                  close('effort', ij['effort'], info['effort'], 1e-5),
                  close('fight', ij['fight'], info['fight'], 1e-5),
                  int(st.anchor) == 'LR'.index(t.anchor),
                  list(np.asarray(ij['grabs'])) == list(info['grabs']), float(ij['first']) == info['first'],
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
    # rl.rsi's starts, which ClimbTask has none of: the estimator begins where a climb's
    # has got to, each hook seen about as near where it is as along the climb from the
    # hang -- not where one accelerometer reading mid-swing puts it: 0.1-0.3 m
    # around the 0.20 m climb's catch until 2026-10-09, and gpu5 learnt to finish on that
    from . import rsi
    ref = rsi.load()
    if ref is not None:
        reset, L, W0, far = jax.jit(task.reset), len(LESSON_KEYS), task.tables.walls, 0.0
        for j in range(0, len(ref['spacing']), 7):
            new = task.rows([dict(rsi.wall(float(ref['spacing'][j])), family=3 + rsi.BOOK, d0_row=L + j)])
            T = task.tables._replace(walls=W0._replace(**{f: getattr(W0, f).at[:1].set(jp.asarray(v))
                                                          for f, v in new.items()}))
            st, _ = reset(T, 0, jax.random.PRNGKey(j))
            err = float(np.linalg.norm(np.asarray(st.est_thr) - np.asarray(task._throat_true(st.d)), axis=1).max())
            far = max(far, err)
            assert err < RSI_SEEN, (f"rl.rsi's row {j} (the {ref['spacing'][j]:.2f} m climb): "
                                    f"a hook seen {err * 1000:.0f} mm from where it is as the episode starts")
        print(f"task: rl.rsi's starts: each hook seen at most {far * 1000:.0f} mm from where it is")
    print("task: OK -- the JAX task is ClimbTask's, observation by observation")


# --- physics ------------------------------------------------------------------------
def check_physics(lessons=(1, 2, 7, 8)):
    import jax, jax.numpy as jp
    from mujoco import mjx
    from . import climb_task as ct
    from .climb_task import ClimbTask
    from .mjx_scene import c_path, MAX_CONTACTS, ITERATIONS, N_RUNG
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

    # A batch that collides only the rung slots and walls in play (Batch's `fit`, what
    # training runs) must step as one that collides every pair: families 1..3's walls
    # at their last tops, 20 steps of random actions from the same keys. On the CPU,
    # whatever card is here: the pairs dropped are a property of the model, and only
    # there are two programs' sums done in the same order. On an RTX 3090 the two
    # compiled steps round differently and 20 steps of thrashing grew that to 9.6e-2
    # (2026-10-05), while every physical check above passed
    from . import walls as W
    lays = W.draw(W.Checker(), {1: 1.0, 2: 1.0, 3: 1.0}, W.last_tops(), np.random.default_rng(0), 32)
    with jax.default_device(jax.devices('cpu')[0]):
        shared, runs = MjxClimb(32), []
        for fit in (False, True):
            b = Batch(16, seed=3, task=shared, fit=fit)
            b.set_bank(lays)
            seq = [np.asarray(b.reset())]
            rng = np.random.default_rng(1)
            for _ in range(20):
                obs, end, r, term, trunc, info = b.step(rng.uniform(-1, 1, (16, 17)).astype(np.float32))
                seq += [np.asarray(obs), np.asarray(r)]
            runs.append((shared.pairs, seq))
    gap = max(float(np.abs(x - y).max() / max(1.0, np.abs(y).max())) for x, y in zip(runs[0][1], runs[1][1]))
    print(f"physics: a batch fitted to families 1..3 collides {runs[1][0][0]} rung slots"
          f"{' and the extra wall' if runs[1][0][1] else ', not the extra wall'}, and over 20 steps "
          f"differs from one colliding every pair by {gap:.1e} (relative)")
    assert runs[0][0] == (N_RUNG, True) and runs[1][0] == (3, False) and gap <= 1e-5, \
        "a batch colliding only the pairs in play is not the batch colliding every pair"
    print("physics: OK")


# --- model --------------------------------------------------------------------------
def check_model(lessons=(1, 1.85, 2), episodes=3, below=(0.35, 0.45, 0.55)):
    """The simplified robot against the FULL one, at the same states. The full robot
    runs; at every control step its state is copied into the simplified robot (both in
    C MuJoCo) and compared three ways: which body pairs touch there and how hard, the
    force on a hook at a rung's end (where the two robots' rungs differ in shape), and
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

    ends = []           # (near an end, full N, simplified N, full pad, simplified pad) a hook
    surface = collections.defaultdict(list)   # stand-in minus real link, mm, every row pooled

    def end_loads(F, S):
        """each loaded hook's rung force and pad on both robots, and whether its outer
        edge is within 10 mm of the rung's end or past it"""
        for sd in 'LR':
            load, on = F._contact(sd)
            if load <= ct.HOLD_LOAD or not on:
                continue
            c, a, h = F.segs[next(iter(on))]
            near = abs(float((F._throat_true(sd) - c) @ a)) > h - ct.SEAT_IN - 0.010
            ends.append((near, load, S._contact(sd)[0], F.env.hand(sd)['pad'], S.env.hand(sd)['pad']))

    def compare(label, F, S, act, steps=None, arms=None, hooks=False):
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
                if hooks:
                    end_loads(F, S)
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
            surface[link] += list(v)
            p10, p50, p90 = np.percentile(v, [10, 50, 90])
            print(f"         {link:>14} stand-in surface vs the real link near the rung: "
                  f"p10 {p10:+5.1f} p50 {p50:+5.1f} p90 {p90:+5.1f} mm ({len(v)} states)")
        assert np.median(q) < 5e-3 and np.median(x) < 0.5, (
            f"{label}: one control step from the same state takes the simplified robot "
            f"{np.median(q):.1e} rad / {np.median(x):.2f} mm from the full one (p50)")

    gait = lambda o, a, rng: pol[0](o[None])[0] if pol else rng.uniform(-0.5, 0.5, 17)
    for k in lessons:
        compare(f"lesson {k:>4}, the full robot's gait", _task(k), _task(k, c_path(k)), gait, hooks=True)

    # A hook near a rung's END, where the full robot's flat-ended cylinder and this
    # robot's capsule differ (rl.mjx_scene: RUNGS). The touch pairs above cannot see it
    # -- the hook is on the rung in both -- but the force can, and the policy reads it
    # through the pad and the wrist: gpu1's gait on lesson 2 read 1398 N here against
    # the full robot's 199 (mean), p90 2855 N apart.
    E = np.array(ends, float)
    for name, m in (('near a rung end', E[:, 0] == 1), ('further in', E[:, 0] == 0)):
        dl, dp = np.abs(E[m, 1] - E[m, 2]), np.abs(E[m, 3] - E[m, 4])
        print(f"model: hooks {name} ({m.sum()} states): |full - simplified| rung force p50 "
              f"{np.median(dl):.0f} p90 {np.percentile(dl, 90):.0f} N, pad p50 {np.median(dp):.0f} "
              f"p90 {np.percentile(dp, 90):.0f} N; mean force {E[m, 1].mean():.0f} full, "
              f"{E[m, 2].mean():.0f} simplified")
    # a known gap, not a pass: on win17's gait p90 689 N apart and 3.3x the full robot's
    # mean. Guarded so a stand-in that makes it worse is seen; the fix is in
    # rl.mjx_scene's RUNGS note, and SEAT_TOL keeps the reward away from it meanwhile
    m = E[:, 0] == 1
    ratio = E[m, 2].mean() / E[m, 1].mean()
    p90 = np.percentile(np.abs(E[m, 1] - E[m, 2]), 90)
    assert m.sum() >= 50 and p90 <= 1000.0 and ratio <= 4.5, (
        f"a hook near a rung end bears {ratio:.1f}x the full robot's force on the simplified "
        f"one, p90 {p90:.0f} N apart ({m.sum()} states): worse than the known 3.3x, 689 N")

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
    # Each stand-in judged on every row's states at once: its shape does not depend on
    # which row it meets, and one row alone can bring a link near it in a handful of
    # states, all one posture -- the heavier lidar robot brought the forearm roll's motor
    # to the 0.55 m row in 9 (-8.3 mm p50) where the other two rows put it at +1.0 in 13
    # and +0.1 in 22. Unbiased to 5 mm, and no stand-in off by more than a rung and a
    # half in its worst tenth -- an arm with no stand-in misses by the whole arm. 5, not
    # the 3 first guessed: the upper arm's motor is a disc, 98 mm across and 62 thick,
    # and no shape MJX collides exactly with a rung fits one closer (a capsule, a ball
    # and rings of rods were each measured: 4.4 mm is the best of them)
    for link, v in sorted(surface.items()):
        v = np.array(v)
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        print(f"model: {link:>14} stand-in against the real link, every row: p10 {p10:+5.1f} "
              f"p50 {p50:+5.1f} p90 {p90:+5.1f} mm ({len(v)} states)")
        if len(v) < 20:
            print(f"       ({link}: too few states near a row to judge)")
            continue
        assert abs(p50) <= 5.0 and -25.0 <= p10 and p90 <= 25.0, (
            f"the {link} stand-in is {p50:+.1f} mm (p50), {p10:+.1f}..{p90:+.1f} (p10..p90) "
            f"off the real link near a rung, over {len(v)} states")
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


# --- warp -------------------------------------------------------------------------
def check_warp(envs=None, out=None, speed=False):
    """MJX's Warp backend against C MuJoCo by check_physics' own measures and a policy's
    episodes against the JAX backend's; with `envs`, on a CUDA GPU, both timed at
    `envs` envs. -> 'warp' only if it is right (and, timed, faster), else 'jax', written
    to `out` with the reason printed: gpu_run.sh trains on what it says, and on 'jax'
    if this never finishes. `speed`: the timing alone, Warp already found right -- the
    preflight checks it beside the other checks, then times it with the card to itself"""
    def decide(impl, why):
        print(f"warp: training physics -> {impl}: {why}", flush=True)
        if out:
            with open(out, 'w') as f:
                f.write(impl + '\n')
        return impl

    if not speed:
        try:
            import warp  # noqa: F401 -- MJX's Warp backend needs warp-lang (requirements-gpu.txt)
            _warp_right()
        except AssertionError as e:
            return decide('jax', f"Warp is not C MuJoCo's physics here: {e}")
        except Exception as e:                # an import, a kernel, a shape: not usable
            return decide('jax', f"Warp did not run: {type(e).__name__}: {e}")
        if not envs:
            return decide('warp', "Warp is right; not timed yet (--speed)")
    import jax
    if jax.devices()[0].platform != 'gpu':
        return decide('jax', "Warp is right, but there is no CUDA GPU here to be faster on")
    sps = {impl: _warp_speed(impl, envs or 4096) for impl in ('jax', 'warp')}
    ratio = sps['warp'] / sps['jax']
    why = f"{sps['warp']:,.0f} env-steps/s against JAX's {sps['jax']:,.0f} ({ratio:.2f}x)"
    return decide('warp', why) if ratio >= WARP_GAIN else decide('jax', why + f", under {WARP_GAIN}x")


WARP_GAIN = 1.1          # how much faster Warp must be to be worth a second physics


def _warp_right(lessons=(1, 2, 7, 8), episodes=24):
    """check_physics' hang, side pull and one-step-from-C's-state on Warp, the same
    thresholds; then a policy's episodes on the walls training runs, on Warp and on
    JAX, check_policy's margin"""
    import jax, jax.numpy as jp
    from mujoco import mjx
    from .mjx_scene import c_path
    from .mjx_task import MjxClimb, Batch, REASONS

    print(f"warp: MJX's Warp backend on {jax.devices()[0].device_kind}, one robot at a time "
          f"-- compiling first", flush=True)
    task = MjxClimb(episodes, impl='warp')      # bank rows: the closed loop's walls
    b = Batch(1, task=task)
    step = jax.jit(task.step_batch)
    throats = jax.jit(jax.vmap(task._throat_true))
    z = np.zeros((1, 17), np.float32)
    over = 0

    def start(k):
        b.set_probs({k: 1.0})
        b.reset()
        return b.state

    for k in lessons:
        t = _task(k, c_path(k))
        t.reset()
        st = start(k)
        base0 = np.asarray(st.d.xpos[0, task.base])
        for _ in range(50):                                   # 2 s of hanging still
            t.step(np.zeros(17))
            st, *_, info = step(task.tables, st, z)
            over += int(np.asarray(info['capped']).sum())
        cs, js = [t._contact(s)[0] for s in 'LR'], np.asarray(info['touch'])[0]
        drift = np.linalg.norm(np.asarray(st.d.xpos[0, task.base]) - base0) * 1e3
        print(f"warp: lesson {k:>4}: after 2 s hanging, hook loads C {cs[0]:.0f}/{cs[1]:.0f} "
              f"Warp {js[0]:.0f}/{js[1]:.0f} N; base drift {drift:.1f} mm", flush=True)
        assert (js > 50).all() and np.allclose(js, cs, rtol=0.25), \
            f"lesson {k}: Warp hangs on {js} N against C's {cs}"
        assert drift < 5.0, f"lesson {k}: Warp drifted {drift:.1f} mm hanging still"

    # the side pull: 0.23 x body weight along the rung for 6 s
    t = _task(1, c_path(1))
    t.reset()
    st = start(1)
    ax = np.array([0.0, 1.0, 0.0])
    f = np.zeros(6)
    f[:3] = 0.23 * t.weight * ax
    p0c, p0j = t._throat_true('R').copy(), np.asarray(throats(st.d))[0, 1]
    t.env.data.xfrc_applied[t.env.base] = f
    st = st._replace(d=st.d.replace(xfrc_applied=st.d.xfrc_applied.at[0, task.base].set(jp.asarray(f))))
    for _ in range(150):
        t.step(np.zeros(17))
        st, *_ = step(task.tables, st, z)
    mc_ = (t._throat_true('R') - p0c) @ ax * 1e3
    mj_ = (np.asarray(throats(st.d))[0, 1] - p0j) @ ax * 1e3
    print(f"warp: {0.23 * t.weight:.0f} N along the rung for 6 s slides the loaded hook "
          f"C {mc_:+.1f} mm, Warp {mj_:+.1f} mm", flush=True)
    assert abs(mj_) < 60 and abs(mj_ - mc_) < max(10.0, 0.5 * abs(mc_)), \
        "the hook slides differently on Warp: the pad or the solver is off"

    # one control step from C's own state along the policy's gait (check_physics')
    pol = _policy()
    rng = np.random.default_rng(0)
    t = _task(1, c_path(1))
    o, _ = t.reset()
    st0 = start(1)
    eq, ef = [], []
    for i in range(300):
        a = pol[0](o[None])[0] if pol else rng.uniform(-0.5, 0.5, 17)
        dc = mjx.put_data(t.env.model, t.env.data)
        st = st0._replace(d=st0.d.replace(**{f: jp.asarray(getattr(dc, f))[None] for f in task.public}),
                          ctrl=jp.asarray(t.ctrl, jp.float32)[None])
        o, r, te, tr, _ = t.step(a)
        st1, *_, info = step(task.tables, st, jp.asarray(a, jp.float32)[None])
        over += int(np.asarray(info['capped']).sum())
        eq.append(np.abs(np.asarray(st1.d.qpos)[0] - t.env.data.qpos).max())
        ef.append(np.abs(np.asarray(info['touch'])[0] - [t._contact(s)[0] for s in 'LR']).max())
        if te or tr:
            o, _ = t.reset()
    eq, ef = np.array(eq), np.array(ef)
    print(f"warp: one control step from C's own state, {len(eq)} steps of "
          f"{'the policy' if pol else 'random actions'}: joints off by p50 {np.median(eq):.1e} "
          f"p90 {np.percentile(eq, 90):.1e} max {eq.max():.1e}; hook loads p50 "
          f"{np.median(ef):.2f} p90 {np.percentile(ef, 90):.2f} max {ef.max():.0f} N", flush=True)
    assert np.percentile(eq, 90) < 1e-3 and np.percentile(ef, 90) < 5.0, \
        "Warp and C part in one control step from the same state: a collision or the solver differs"
    assert over == 0, f"Warp gave up contacts or constraint rows in {over} substeps: raise WARP_CONTACTS/WARP_EFC"

    # a blown-up env ends as 'blowup', finite, and leaves the others alone: on Warp the
    # batch shares one contact buffer (the closed loop's batch, one compile fewer)
    bw = Batch(episodes, task=task)
    bw.set_probs({1: 1.0})
    bw.reset()
    d = bw.state.d
    bw.state = bw.state._replace(d=d.replace(qvel=d.qvel.at[0, 0].set(jp.nan)),
                                 Rf=bw.state.Rf.at[1].set(jp.nan))
    zb = np.zeros((episodes, 17), np.float32)
    obs, end, r, term, trunc, info = bw.step(zb)
    why = [REASONS[int(x)] for x in np.asarray(info['reason'])]
    assert why[:2] == ['blowup', 'blowup'] and not any(why[2:]), f"a NaN on Warp ended as {why}"
    for _ in range(3):
        assert all(np.isfinite(np.asarray(x)).all() for x in (obs, end, r)), "a NaN got out on Warp"
        obs, end, r, term, trunc, info = bw.step(zb)
    print("warp: a NaN state or estimate ends as 'blowup', finite, and resets", flush=True)

    # closed loop, on what training runs: gpu4's policy on families 1 and 2 at its tops
    # and on family 3's rl.rsi starts (its climbs from the hang never finish), each env's
    # first episode, the same draws on both. win17's old lessons were never trained on
    pol = _policy(WALLS_RUN)
    if pol is None:
        return
    import json
    from . import walls as W, rsi
    tops = W.first_tops()
    for f, t in json.load(open(os.path.join(WALLS_RUN, 'best_state.json')))['walls']['tops'].items():
        tops[int(f)].update(t)
    lays = rsi.mix(W.draw(W.Checker(), {1: 1.0, 2: 1.0, 3: 1.0}, tops, np.random.default_rng(0), episodes),
                   np.random.default_rng(1), tops[3]['spacing'], share=1.0)
    runs = {}
    for impl in ('jax', 'warp'):
        bb = bw if impl == 'warp' else Batch(episodes, task=MjxClimb(episodes))
        bb.set_bank(lays)
        bb.key = jax.random.PRNGKey(0)   # the same walls and noise: bw has stepped already
        o, done, ok = np.asarray(bb.reset()), np.zeros(episodes, bool), {}
        while not done.all():
            o, _, _, term, trunc, info = bb.step(pol[0](o))
            o = np.asarray(o)
            end = np.asarray(term | trunc) & ~done
            for i in np.nonzero(end)[0]:
                f, won = int(info['family'][i]), REASONS[int(info['reason'][i])] == 'success'
                ok[f] = [ok.get(f, [0, 0])[0] + won, ok.get(f, [0, 0])[1] + 1]
            done |= end
        runs[impl] = ok
    say = lambda ok: ", ".join(f"family {f if f < rsi.BOOK else f'{f - rsi.BOOK} from its starts'} "
                               f"{w}/{n}" for f, (w, n) in sorted(ok.items()))
    print(f"warp: gpu4's policy, {episodes} episodes on the walls training runs: JAX {say(runs['jax'])}; "
          f"Warp {say(runs['warp'])}", flush=True)
    runs = {k: sum(w for w, _ in v.values()) for k, v in runs.items()}
    assert abs(runs['warp'] - runs['jax']) <= 0.2 * max(runs.values()) + 2, \
        "the same policy climbs clearly better on one backend"


def _warp_speed(impl, envs, steps=20):
    """env-steps/s of a batch fitted to families 1..3's walls at their last tops, as
    training runs them, random actions, after its compile"""
    import jax
    from . import walls as W
    from .mjx_task import Batch
    from .train_ppo import BANK
    b = Batch(envs, impl=impl, bank=BANK, fit=True)
    b.set_bank(W.draw(W.Checker(), {1: 1.0, 2: 1.0, 3: 1.0}, W.last_tops(),
                      np.random.default_rng(0), BANK))
    b.reset()
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, (envs, 17)).astype(np.float32)
    jax.block_until_ready(b.step(a))                         # compile
    t0 = time.perf_counter()
    for _ in range(steps):
        out = b.step(a)
    jax.block_until_ready(out)
    sps = envs * steps / (time.perf_counter() - t0)
    print(f"warp: {envs} envs on {impl}: {sps:,.0f} env-steps/s", flush=True)
    return sps


CHECKS = dict(task=check_task, physics=check_physics, model=check_model,
              policy=check_policy, contacts=check_contacts, warp=check_warp)

if __name__ == '__main__':
    which = sys.argv[1:]
    if '--run' in which:
        i = which.index('--run')
        RUN = which[i + 1]
        del which[i:i + 2]
    opts = {}
    for flag, kind in (('--envs', int), ('--out', str)):       # check_warp's
        if flag in which:
            i = which.index(flag)
            opts[flag[2:]] = kind(which[i + 1])
            del which[i:i + 2]
    if '--speed' in which:
        which.remove('--speed')
        opts['speed'] = True
    which = which or ['task', 'physics', 'model', 'policy']
    if len(which) == 1:
        CHECKS[which[0]](**opts)
    else:
        # one process each: the task check runs JAX in float64, the others as training does
        for w in which:
            r = subprocess.run([sys.executable, '-m', 'rl.mjx_check', w, '--run', RUN])
            if r.returncode:
                raise SystemExit(f"rl.mjx_check {w} FAILED")
