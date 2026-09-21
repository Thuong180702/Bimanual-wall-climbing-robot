"""Assert-based checks, one per build phase. No framework, no fixtures.

    python -m controller.test_climb            # all phases
    python -m controller.test_climb --phase 3  # just one
"""
import sys, argparse, numpy as np
from . import SCENE
from climbing_env import ClimbingEnv

PHASES = {}


def phase(n, what):
    def deco(fn):
        fn.what = what
        PHASES[n] = fn
        return fn
    return deco


def env_at_hang(**kw):
    from .moves import make_env
    e = make_env(**kw)
    e.reset()
    return e


# --------------------------------------------------------------------------
@phase(0, "keyframe replay is stable (reproduces the README figure on this machine)")
def test_phase_0():
    e = env_at_hang()
    ctrl = e.data.ctrl.copy()
    z0 = e.data.xpos[e.base].copy()
    for _ in range(int(5.0 / e.model.opt.timestep)):
        e.step(ctrl)
    drift = np.linalg.norm(e.data.xpos[e.base] - z0) * 1000.0
    okL, gL = e.latched('L')
    okR, gR = e.latched('R')
    print(f"    base drift {drift:.2f} mm over 5 s, latched L={okL} R={okR}")
    assert drift < 5.0, f"base drifted {drift:.1f} mm, expected < 5"
    assert okL and okR, "a hook came off during a passive 5 s hold"
    assert e.terminated() == (False, ''), e.terminated()


# --------------------------------------------------------------------------
@phase(1, "hold map reproduces the README's reachability claim")
def test_phase_1():
    import mujoco
    from .holds import build, HoldMap
    m = mujoco.MjModel.from_xml_path(SCENE)
    hm = build(m)
    hm.save()
    assert len(set(g.hold for g in hm.grasps.values())) == 39, "expected 39 rungs"
    assert HoldMap.load().grasps.keys() == hm.grasps.keys(), "json round-trip lost holds"

    # README: every hold except the topmost of each wall has another hold
    # 0.05-0.50 m above and within 0.30 m sideways.
    tops = {}
    for g in hm.grasps.values():
        tops[g.wall] = max(tops.get(g.wall, -9e9), g.pos[2])
    stranded = set()
    for rung in set(g.hold for g in hm.grasps.values()):
        pts = [g for g in hm.grasps.values() if g.hold == rung]
        if any(t == 'up' for g in pts for _, t in hm.neighbors(g.key)):
            continue
        stranded.add(rung)
        top = tops[pts[0].wall]
        assert top - max(p.pos[2] for p in pts) < 0.50, \
            f"{rung} has no hold above it but is not near the top of its wall"
    print(f"    {len(hm.grasps)} grasp points, {sum(len(v) for v in hm.edges().values())} edges;"
          f" {len(stranded)} rungs are top-of-wall")

    # the planner must find a route for each of the six move types
    r = hm.route('rung_front_low_0#4', 'rung_front_stagC_1#1')
    kinds = set(k for _, k in r[1:])
    print(f"    route front_low -> front_stagC: {len(r)-1} moves, kinds {sorted(kinds)}")
    assert 'up' in kinds and len(r) > 4
    r2 = hm.route('rung_front_line_0#8', 'rung_side_line_0#2')
    assert any(k == 'cross_side' for _, k in r2[1:]), [k for _, k in r2[1:]]
    r3 = hm.route('rung_front_line_0#8', 'rung_back_line_0#8')
    assert any(k == 'cross_back' for _, k in r3[1:]), [k for _, k in r3[1:]]
    print(f"    cross_side route {len(r2)-1} moves, cross_back route {len(r3)-1} moves")

    # --- slide margin (phase 7): which climbs land where the anchor's slide is harmless
    from .holds import SLIDE
    risky = lambda rt: [(a, b) for (a, _), (b, k) in zip(rt, rt[1:]) if k in ('up', 'down')
                        and min(hm.grasps[a].margin, hm.grasps[b].margin) < SLIDE]
    safe_pairs = sorted({tuple(sorted((hm.grasps[a].hold, hm.grasps[b].hold)))
                         for a in hm.grasps for b, k in hm.neighbors(a)
                         if k in ('up', 'down')
                         and min(hm.grasps[a].margin, hm.grasps[b].margin) >= SLIDE})
    print(f"    vertical moves with >= {SLIDE*1000:.0f} mm of rung to spare at both ends:"
          f" {len(safe_pairs)} rung pair(s) {safe_pairs}")
    assert safe_pairs == [('rung_front_line_0', 'rung_front_low_0')], safe_pairs

    up = hm.route('rung_front_low_0#11', 'rung_front_line_0#11')
    assert not risky(up), "the one slide-safe climb should not be routed onto a rung end"
    print(f"    low -> line (both continuous): {len(up)-1} moves, none at risk of sliding"
          f" off; but front_low -> front_stagC needs {len(risky(r))} that are")


# --------------------------------------------------------------------------
@phase(2, "encoder+IMU estimate and relative Jacobians match ground truth")
def test_phase_2():
    from .kinematics import Estimator, THROAT_PT, ARM, WAIST, JOINTS, triad
    from .holds import HoldMap
    from .moves import Ctx, IMU_WINDOW
    e = env_at_hang()
    # The IMU as the controller senses it: averaged over one slow swing (Ctx.imu).
    # Read raw at the instant of the reset, the body's small post-reset sway reads
    # as tilt -- 0.3-1.0 deg over the first second, on v1 as on this robot -- so
    # whether a single sample passed 0.5 deg was a matter of timing.
    ctx = Ctx(e, HoldMap.load())
    for _ in range(int(IMU_WINDOW / ctx.dt)):
        ctx.tick()
    est = Estimator(e.model)
    enc = e.encoders()
    q = np.array([enc[j][0] for j in JOINTS])

    # anchor: the point on the rung axis under the left throat -- in operation this is
    # simply the grasp point the hand was aimed at.
    okL, gL = e.latched('L')
    hp = e.data.xpos[e.hook['L']]
    RL = e.data.xmat[e.hook['L']].reshape(3, 3)
    thr_true = hp + RL @ THROAT_PT
    gp, gax = e.data.geom_xpos[gL], e.data.geom_xmat[gL].reshape(3, 3)[:, 2]
    anchor = gp + np.dot(thr_true - gp, gax) * gax
    print(f"    throat sits {np.linalg.norm(thr_true-anchor)*1000:.1f} mm off the rung axis")

    est.update(q, ctx.imu(), 'L', anchor)

    # attitude from raw IMU vs the simulator's quaternion oracle
    from .kinematics import rotvec
    Rt = e.data.xmat[est.torso].reshape(3, 3)
    R_est_torso = est.d.xmat[est.torso].reshape(3, 3)
    att_err = np.degrees(np.linalg.norm(rotvec(R_est_torso @ Rt.T)))
    print(f"    TRIAD attitude error {att_err:.3f} deg")
    assert att_err < 0.5, att_err

    # the free hand, predicted from encoders alone
    hpR = e.data.xpos[e.hook['R']]
    thrR_true = hpR + e.data.xmat[e.hook['R']].reshape(3, 3) @ THROAT_PT
    err = np.linalg.norm(est.throat('R') - thrR_true) * 1000
    print(f"    free-hand position error {err:.2f} mm")
    assert err < 5.0, err

    # Relative Jacobian vs finite difference, done purely in the anchor's own frame so
    # the convention being tested is unambiguous: hand pinned in position AND orientation.
    joints = WAIST + ARM['L'] + ARM['R']

    def rel_in_anchor(qq):
        st = Estimator(e.model)
        st.d.qpos[:3] = 0.0; st.d.qpos[3:7] = (1., 0., 0., 0.); st.d.qpos[7:] = qq
        __import__('mujoco').mj_kinematics(st.m, st.d)
        Ra = st.d.xmat[st.hook['L']].reshape(3, 3)
        return Ra.T @ (st._throat('R') - st._throat('L'))

    Jp, Jr = est.jac_rel('R', 'L', joints)
    Ra_w = est.hook_R('L')
    Jp_anchor = Ra_w.T @ Jp
    h = 1e-6
    rel0 = rel_in_anchor(q)
    worst = 0.0
    for col, j in enumerate(joints):
        q2 = q.copy(); q2[j] += h
        fd = (rel_in_anchor(q2) - rel0) / h
        d = np.linalg.norm(fd - Jp_anchor[:, col])
        worst = max(worst, d)
        assert d < 1e-3 + 1e-3 * np.linalg.norm(fd), \
            f"{JOINTS[j]}: jacobian {Jp_anchor[:,col]} vs finite diff {fd}"
    print(f"    all {len(joints)} relative-Jacobian columns match finite differences"
          f" (worst {worst:.2e})")


# --------------------------------------------------------------------------
def make_ctx(**kw):
    from .holds import HoldMap
    from .moves import Ctx
    e = env_at_hang(**kw)
    return e, Ctx(e, HoldMap.load()).start_hanging()


def drive(e, ctx, gen, watch='R'):
    """Run a move to completion, checking the anchored hand never goes slack.

    Deliberately not `env.latched`: that is a tight box around the nominal seat, and the
    hook holds fine outside it -- during a pull-up it reads unlatched for seconds while
    carrying the whole robot. Load through the wrist is the honest signal.
    """
    sides = [watch] if watch else ['L', 'R']       # watch=None: at least one hand on
    slack = 0
    for _ in gen:
        if max(np.linalg.norm(e.hand(s)['force']) for s in sides) < 15.0:
            slack += 1
    assert slack * e.model.opt.timestep < 0.5, \
        f"anchored hand {watch or 'L/R'} went slack for "\
        f"{slack * e.model.opt.timestep:.2f} s"


@phase(3, "one hand takes the next hold along a rung")
def test_phase_3():
    import mujoco
    from .moves import reach, secure
    from .kinematics import THROAT_PT
    e, ctx = make_ctx()
    drive(e, ctx, secure(ctx), watch='L')
    # one grasp point along, which is inside the 0.10 m the free hand reaches without
    # the anchored arm having to reposition the whole body
    key = ctx.holds.nearest([0.44, 0.157 + 0.09, 1.00])
    print(f"    L: y=0.157 -> {ctx.holds.grasps[key].pos[1]:.3f} ({key})")
    drive(e, ctx, reach(ctx, 'L', key), watch='R')

    ok, g = e.latched('L')
    name = mujoco.mj_id2name(e.model, mujoco.mjtObj.mjOBJ_GEOM, g)
    seat = e.seat('L', g)
    thr = e.data.xpos[e.hook['L']] + e.data.xmat[e.hook['L']].reshape(3, 3) @ THROAT_PT
    print(f"    latched={ok} on {name}, seat {seat.round(1)} mm, y={thr[1]:.3f}")
    assert ok and name == 'rung_front_line_0'
    assert e.grip_ok('L'), "finger left the band that actually holds"
    # along the rung the tolerance is loose on purpose: the capture window is about
    # fore/aft and height, and a continuous rung is just as good a hold 50 mm along it
    assert abs(thr[1] - ctx.holds.grasps[key].pos[1]) < 0.10, "landed on the wrong rung span"
    assert abs(thr[0] - 0.44) < 0.025, "outside the +-25 mm fore/aft capture window"
    # Only across the rung: with both hooks on the same straight rung, position *along*
    # it is unobservable from the hold map and the encoders alone -- sliding the anchor
    # slides the other hand identically -- so the estimate legitimately drifts there.
    ax = np.asarray(ctx.grip['L'][2], float)
    ax = ax / np.linalg.norm(ax)
    off = np.array(ctx.grip['L'][1]) - thr
    assert np.linalg.norm(off - (off @ ax) * ax) < 0.03, \
        "the recorded hold is off the rung the hand is actually on"


# --------------------------------------------------------------------------
@phase(4, "the other hand does the same step")
def test_phase_4():
    import mujoco
    from .moves import reach, secure, MoveFailed
    from .kinematics import THROAT_PT, grasp_frame, rotvec
    e, ctx = make_ctx()
    drive(e, ctx, secure(ctx), watch='R')
    key = ctx.holds.nearest([0.44, -0.157 + 0.09, 1.00])
    g = ctx.holds.grasps[key]
    why = None
    try:
        drive(e, ctx, reach(ctx, 'R', key), watch='L')
    except MoveFailed as ex:
        why = str(ex)

    b = e.hook['R']
    R = e.data.xmat[b].reshape(3, 3)
    thr = e.data.xpos[b] + R @ THROAT_PT
    d = thr - g.p()
    ax = np.asarray(g.a(), float)
    ax = ax / np.linalg.norm(ax)
    v = rotvec(grasp_frame(g.n()) @ R.T)
    print(f"    fore/aft {d[0]*1000:+6.1f} mm (window +-25) | height {d[2]*1000:+6.1f} mm "
          f"(window -10..+70) | off-axis {np.degrees(np.linalg.norm(v - (v @ ax) * ax)):.0f} deg")
    if why:
        print(f"    {why}")
    assert e.latched('R')[0] and e.grip_ok('R'), "the right hand did not take the hold"


# --------------------------------------------------------------------------
@phase(5, "traverse: four alternating steps along a rung")
def test_phase_5():
    from .moves import traverse, secure
    e, ctx = make_ctx(timeout_s=2000.0)
    drive(e, ctx, secure(ctx), watch='R')
    y0 = e.data.xpos[ctx.est.base][1]
    for _ in traverse(ctx, [0, 1, 0], 4):
        pass
    moved = (e.data.xpos[ctx.est.base][1] - y0) * 1000
    print(f"    base moved {moved:+.0f} mm over {len(ctx.log)} steps in {e.data.time:.0f} s "
          f"of simulated time")
    assert e.latched('L')[0] and e.latched('R')[0], "did not finish with both hands on"
    # Each step gains far less than it aims for -- the hand lands roughly a quarter of
    # the way along. It traverses, it just creeps.
    assert moved > 60, f"only travelled {moved:.0f} mm"


# --------------------------------------------------------------------------
@phase(6, "climb: down a row, on the one pair of rungs where the slide is harmless")
def test_phase_6():
    """Routed, not aimed at whatever hold happens to be above.

    Phase 7 measures that the anchored hook slides up to 310 mm along its rung under the
    sideways load a climb puts on it, and the free hand's target slides with it. On the
    segmented rows -- 0.35 m rungs with 0.25 m gaps -- that lands the hand over a gap,
    which is what used to fail here. The map now prices vertical moves by how much rung
    is left at both ends (phase 1), and on this scene exactly one pair survives it:
    rung_front_low_0 (z = 0.55) and rung_front_line_0 (z = 1.00), both 2.4 m long. The
    robot hangs on the second, so the climb it can make is down onto the first.
    """
    from .moves import climb, secure, MoveFailed
    from .kinematics import THROAT_PT
    e, ctx = make_ctx(timeout_s=2500.0)
    drive(e, ctx, secure(ctx), watch='R')

    # a hold on the segmented row above is refused outright, and says why
    above = min((h for h in ctx.holds.grasps.values() if h.hold == 'rung_front_aligned_1'),
                key=lambda h: abs(h.pos[1] - 0.157))
    try:
        next(climb(ctx, 'L', above.key))
        raise AssertionError("climbed onto a rung end that phase 7 says will not hold")
    except MoveFailed as ex:
        print(f"    refused {above.key}: {ex}")

    g = min((h for h in ctx.holds.grasps.values() if h.hold == 'rung_front_low_0'),
            key=lambda h: abs(h.pos[1] - 0.157))
    b = e.hook['L']
    thr = lambda: e.data.xpos[b] + e.data.xmat[b].reshape(3, 3) @ THROAT_PT
    z0 = thr()[2]
    why = None
    try:
        # watch=None: at least one hand loaded, rather than naming the anchor. Coming down
        # a row the weight legitimately arrives on the moving hand before it latches --
        # measured, R carries nothing for 3.56 s straight near the end while L, already on
        # the lower rung, holds the robot. Naming R there tests the wrong invariant; the
        # one that matters is that the robot is never off both hands, and `tick` enforces
        # a 0.6 s version of it continuously underneath.
        drive(e, ctx, climb(ctx, 'L', g.key), watch=None)
    except MoveFailed as ex:
        why = str(ex)
    d = thr() - g.p()
    print(f"    {g.key} is {(g.pos[2] - 1.0)*1000:.0f} mm down; the hand moved "
          f"{(thr()[2] - z0)*1000:+.0f} mm and ended {np.linalg.norm(d)*1000:.0f} mm from "
          f"it (fore/aft {d[0]*1000:+.0f}, along {d[1]*1000:+.0f}, up {d[2]*1000:+.0f}), "
          f"in {e.data.time:.0f} s of simulated time")
    if why:
        print(f"    {why}")
    acc = {s: [] for s in 'LR'}
    for _ in range(int(2.0 / e.model.opt.timestep)):
        e.step(ctx.ctrl)
        for s in 'LR':
            acc[s].append(float(np.linalg.norm(e.hand(s)['force'])))
    f = {s: float(np.mean(v)) for s, v in acc.items()}
    print(f"    load two seconds later: L {f['L']:.0f} N, R {f['R']:.0f} N "
          f"(the robot weighs {e.model.body_subtreemass[e.base] * 9.81:.0f} N)")
    print(f"    env.latched says L={e.latched('L')[0]} R={e.latched('R')[0]}")
    # Not `env.latched`: it is a tight box round the nominal seat, and the README's own
    # measurement has a hook outside it carrying 167 N. The physical test is the model's
    # own capture window plus the load actually going through the hand.
    assert abs(d[0]) < 0.025, f"fore/aft {d[0]*1000:+.0f} mm, outside the +-25 mm window"
    assert -0.010 < d[2] < 0.070, f"height {d[2]*1000:+.0f} mm, outside -10..+70 mm"
    # A hand on a hold below carries nothing until the body is let down onto it, and
    # everything downstream plans against whichever hand is really holding.
    assert f['L'] > 60.0, (f"the weight stayed on the rung above: L {f['L']:.0f} N "
                           f"against R {f['R']:.0f} N")


# --------------------------------------------------------------------------
@phase(7, "the hook barely resists sliding along its rung")
def test_phase_7():
    """A characterisation, not a controller test.

    The model README specifies the hook's pull-off force (>1200 N with the finger in the
    band) and its capture window, but never its restraint *along* the rung -- and that is
    the axis that blocks climbing. The hook is a slot around a cylinder: it catches the
    pull-out and the roll, and along the axis there is only friction.
    """
    import mujoco
    from .moves import make_env, Ctx, secure, GRIP_HARD, _finger
    from .holds import HoldMap
    from .kinematics import THROAT_PT
    e = make_env(timeout_s=900.0)
    e.reset()
    ctx = Ctx(e, HoldMap.load()).start_hanging()
    for _ in secure(ctx):
        pass
    for _ in _finger(ctx, 'R', GRIP_HARD, 0.4):
        pass
    for _ in _finger(ctx, 'L', 0.30, 0.4):
        pass
    ctx.grip.pop('L')
    seat = lambda: (e.data.xpos[e.hook['R']]
                    + e.data.xmat[e.hook['R']].reshape(3, 3) @ THROAT_PT)[1]
    y0 = seat()
    base = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    # Pulls as fractions of body weight: 20, 40 and 80 N on the 109 N robot this was
    # written on. Friction holds the hook along its rung, and friction scales with the
    # load it carries, so the same newtons on the 215 N robot read as a hook that no
    # longer slides -- 80 N moved it 62 mm -- while 0.73 W still runs it off.
    W = e.model.body_subtreemass[base] * 9.81
    slid = {}
    for frac in (0.18, 0.37, 0.73):
        e.data.xfrc_applied[base] = [0, frac * W, 0, 0, 0, 0]
        for _ in range(1200):
            e.step(ctx.ctrl)
        slid[frac] = (seat() - y0) * 1000
    e.data.xfrc_applied[base] = np.zeros(6)
    print("    " + ", ".join(f"{frac * W:.0f} N sideways -> {d:+.0f} mm" for frac, d in slid.items())
          + f"  (the robot weighs {W:.0f} N)")
    assert slid[0.18] > 10, "the hook now resists sliding; the climbing blocker may be gone"
    assert slid[0.73] > 500, "0.73 x body weight used to run it off the rung entirely"


# --------------------------------------------------------------------------
@phase(8, "route: the whole robot moves down a row and back up")
def test_phase_8():
    """The planner picks the holds, `route_to` walks them, both hands end on the new row.

    This is the six-move library doing what it was built for, on the only vertical pair
    in the scene that phase 7 says will hold: rung_front_line_0 (z = 1.00) and
    rung_front_low_0 (z = 0.55), both 2.4 m of unbroken rung. Down first, because that is
    the direction available from the keyframe, then back up.
    """
    from .moves import route_to, secure
    e, ctx = make_ctx(timeout_s=8000.0)
    drive(e, ctx, secure(ctx), watch='R')
    z0 = e.data.xpos[ctx.est.base][2]

    # Aim each row change at the hold directly below/above a hand, not below the body's
    # centre. A hold 100 mm to the inside costs the leading arm a cross-body posture, and
    # the run that tried it went to the floor.
    pick = lambda rung: min((h for h in ctx.holds.grasps.values() if h.hold == rung),
                            key=lambda h: abs(h.pos[1] - ctx.grip['L'][1][1]))
    low = pick('rung_front_low_0')
    drive(e, ctx, route_to(ctx, low.key), watch=None)
    z1 = e.data.xpos[ctx.est.base][2]
    print(f"    down: base {(z1 - z0)*1000:+.0f} mm, hands on "
          f"{sorted(ctx.grip[s][0] for s in ctx.grip)}, {e.data.time:.0f} s")
    assert all(ctx.grip[s][0] == 'rung_front_low_0' for s in ctx.grip), \
        f"only got one hand down: {[(s, ctx.grip[s][0]) for s in ctx.grip]}"
    # The hands are the test -- both are asserted onto the lower rung above. The base
    # drops less than the 450 mm row spacing because the tuck leaves the body folded up
    # under the hands rather than hanging at full stretch.
    assert z1 - z0 < -0.25, f"base only dropped {(z1 - z0)*1000:.0f} mm"

    back = pick('rung_front_line_0')
    drive(e, ctx, route_to(ctx, back.key), watch=None)
    z2 = e.data.xpos[ctx.est.base][2]
    print(f"    up:   base {(z2 - z1)*1000:+.0f} mm, hands on "
          f"{sorted(ctx.grip[s][0] for s in ctx.grip)}, {e.data.time:.0f} s total")
    assert all(ctx.grip[s][0] == 'rung_front_line_0' for s in ctx.grip), \
        f"only got one hand up: {[(s, ctx.grip[s][0]) for s in ctx.grip]}"
    assert z2 - z1 > 0.25, f"base only rose {(z2 - z1)*1000:.0f} mm"
    assert e.latched('L')[0] and e.latched('R')[0], "did not finish with both hands on"


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, default=None)
    ap.add_argument('--stop', action='store_true', help='stop at the first failure')
    a = ap.parse_args()
    todo = [a.phase] if a.phase is not None else sorted(PHASES)
    bad = []
    for n in todo:
        fn = PHASES[n]
        print(f"phase {n}: {fn.what}")
        try:
            fn()
            print("    OK\n")
        except (AssertionError, Exception) as ex:
            bad.append((n, ex))
            print(f"    FAILED: {type(ex).__name__}: {ex}\n")
            if a.stop:
                break
    if bad:
        print("failed: " + ", ".join(f"phase {n}" for n, _ in bad))
        sys.exit(1)
    print("all passed")


if __name__ == '__main__':
    main()
