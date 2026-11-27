"""The full robot (legs, upgraded arms): every acceptance check of
docs/superpowers/specs/2026-11-24-legs-design.md, as asserts.

    python cad/tools/check_full.py variant kinematics ...   # the named checks, in order

The full robot is spec.ROBOT == "full" (environment ROBOT=full); the default, "v2",
is the robot the paper trains, and must stay exactly as it is.
"""
import json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
V2_TEXT = ["urdf/climbing_robot.urdf", "urdf/climbing_robot_floating.urdf",
           "urdf/_actuators.json", "urdf/scene.xml"]


def _spec(robot, expr):
    """evaluate `expr` with lib.spec imported under ROBOT=robot (None: unset)"""
    env = dict(os.environ)
    env.pop("ROBOT", None)
    if robot:
        env["ROBOT"] = robot
    code = ("import sys, json; sys.path.insert(0, %r); from lib import spec; "
            "from lib.motors import MOTORS; print(json.dumps(%s))" % (os.path.join(ROOT, "cad", "src"), expr))
    out = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _run(cmd, robot=None):
    env = dict(os.environ)
    env.pop("ROBOT", None)
    if robot:
        env["ROBOT"] = robot
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, (cmd, out.stdout[-1500:], out.stderr[-1500:])
    return out.stdout


def check_variant():
    """v2 by default, untouched; ROBOT=full switches; RobStride 04 catalogued"""
    assert _spec(None, "spec.ROBOT") == "v2"
    assert _spec(None, "len(spec.JOINTS)") == 17
    assert _spec("full", "spec.ROBOT") == "full"
    assert _spec("full", "spec.OUT_DIR") == "urdf/full"
    assert _spec(None, "spec.OUT_DIR") == "urdf"
    assert _spec(None, "['RS04' in MOTORS, 'RS04' in spec.IFACE]") == [True, True]
    for cmd in (["cad/tools/gen_urdf.py"], ["urdf/_gen_scene.py"], ["urdf/_gen_keyframe.py", "--write"]):
        _run([PY] + cmd)
    # the CAD kernel's mass properties wobble in the 9th-12th digit run to run, and
    # STL meshes re-tessellate: same model when every word matches and every number
    # is within 1e-9 (relative) of the committed one; then the committed files go back
    for f in V2_TEXT:
        old = subprocess.run(["git", "show", "HEAD:" + f], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
        bad = same_numbers(old, open(os.path.join(ROOT, f)).read())
        assert bad is None, "regenerating v2 changed %s: %s" % (f, bad)
    subprocess.run(["git", "checkout", "--", "urdf/meshes"] + V2_TEXT, cwd=ROOT, check=True)
    print("variant: v2 by default and regenerates to the committed model (1e-9); "
          "ROBOT=full switches; RS04 catalogued")


NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def same_numbers(a, b, rel=1e-9, abs_=1e-9):
    """None if texts a and b differ only by numbers within rel/abs_, else the first difference"""
    wa, wb = NUM.split(a), NUM.split(b)
    if wa != wb:
        i = next(k for k, (x, y) in enumerate(zip(wa, wb)) if x != y) if len(wa) == len(wb) else min(len(wa), len(wb))
        return "text differs near %r" % (wa[i][:80] if i < len(wa) else "<end>")
    for x, y in zip(NUM.findall(a), NUM.findall(b)):
        fx, fy = float(x), float(y)
        if abs(fx - fy) > max(abs_, rel * max(abs(fx), abs(fy))):
            return "%s -> %s" % (x, y)
    return None


def check_kinematics():
    """the legs' joints and lengths (full), and v2's kinematics untouched"""
    expr = ("dict(n=len(spec.JOINTS), fk={k: [list(o), [list(r) for r in R]] for k, (o, R) in spec.fk().items()},"
            " mp=[[m[0], m[1]] + [list(v) for v in m[2:]] for m in spec.motor_placements()], sole=spec.SOLE)")
    f = _spec("full", expr)
    assert f["n"] == 29, f["n"]
    o = {k: v[0] for k, v in f["fk"].items()}
    for s in "LR":
        hip, knee, ankle, foot = (o["hip_" + s], o["calf_" + s], o["ankle_" + s], o["foot_" + s])
        assert abs(hip[2] - knee[2] - 0.320) < 0.001, (s, hip, knee)
        assert abs(knee[2] - ankle[2] - 0.320) < 0.001, (s, knee, ankle)
        assert abs(hip[2] - (foot[2] - f["sole"]) - 0.700) < 0.002, (s, hip, foot)
    for k, v in o.items():
        if k.endswith("_R"):
            l = o[k[:-2] + "_L"]
            assert max(abs(v[0] - l[0]), abs(v[1] + l[1]), abs(v[2] - l[2])) < 1e-9, (k, v, l)
    assert len(f["mp"]) == 17 + 2 * (4 + 2), len(f["mp"])
    base = json.load(open(os.path.join(HERE, "fixtures", "v2_kinematics.json")))
    v2 = _spec(None, expr)
    for k, (p, R) in base["fk"].items():
        q, S = v2["fk"][k]
        assert max(abs(a - b) for a, b in zip(p + sum(R, []), q + sum(S, []))) < 1e-12, k
    assert [[m[0], m[1]] for m in v2["mp"]] == [[m[0], m[1]] for m in base["mp"]]
    print("kinematics: 29 joints, thigh 0.320, calf 0.320, sole 0.700 under the hip axis, "
          "right = left mirrored, 29 actuators; v2 unchanged")


def _rot(axis, a):
    import numpy as np
    c, s_ = math.cos(a), math.sin(a)
    x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s_, x * z * (1 - c) + y * s_],
                     [y * x * (1 - c) + z * s_, c + y * y * (1 - c), y * z * (1 - c) - x * s_],
                     [z * x * (1 - c) - y * s_, z * y * (1 - c) + x * s_, c + z * z * (1 - c)]])


def ankle_linkage(drive, rods, ankle, pitch, roll):
    """crank angles (rad, about each motor's output axis from its neutral) that keep both
    rods at their length with the foot at (pitch, roll), and the crank and rod vectors;
    None where a rod cannot reach. drive: spec.ANKLE_DRIVE rows of one leg; rods:
    spec.ANKLE_RODS rows of the same leg (crank neutral unit, foot point at zero pose,
    both in the calf frame); ankle: the ankle centre in the calf frame."""
    import numpy as np
    A = np.array(ankle)
    R = _rot((0, 1, 0), pitch) @ _rot((1, 0, 0), roll)
    out = []
    for (name, _k, _h, flange, axis, r), (_n, u0, foot0) in zip(drive, rods):
        M, ax, u0 = np.array(flange), np.array(axis, float), np.array(u0, float)
        tip0 = M + r * u0
        L = np.linalg.norm(np.array(foot0) - tip0)
        P = A + R @ (np.array(foot0) - A)
        f = lambda th: np.linalg.norm(P - (M + r * (_rot(ax, th) @ u0))) - L
        ths = np.radians(np.arange(-150.0, 150.01, 0.5))
        fs = [f(t) for t in ths]
        roots = []
        for t0, t1, f0, f1 in zip(ths, ths[1:], fs, fs[1:]):
            if f0 == 0.0 or f0 * f1 < 0:
                lo, hi = t0, t1
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
                roots.append((lo + hi) / 2)
        if not roots:
            return None
        th = min(roots, key=abs)          # the branch the linkage is assembled on
        tip = M + r * (_rot(ax, th) @ u0)
        out.append((th, tip - M, P - tip))
    return out


def check_ankle(need_pitch=50.6, need_roll=19.0):
    """the two calf motors, through cranks and rods, hold the ankle's needs (x 1.3) over
    the whole ankle range, with no crank and rod within 10 deg of a line"""
    import numpy as np
    g = _spec("full", "dict(drive=[d for d in spec.ANKLE_DRIVE if d[2] == 'calf_L'], "
                      "rods=[r for r in spec.ANKLE_RODS if r[0].startswith('ankle_L')], "
                      "ankle=[j[4] for j in spec.JOINTS if j[0] == 'ankle_L_pitch'][0], "
                      "lim=[[j[5] for j in spec.JOINTS if j[0] == 'ankle_L_' + k][0] for k in ('pitch', 'roll')], "
                      "peak=MOTORS['AK60-39']['peak_Nm'])")
    (p0, p1), (r0, r1) = g["lim"]
    worst = dict(pitch=1e9, roll=1e9, sing=180.0)
    for p in np.arange(math.degrees(p0), math.degrees(p1) + 1e-9, 5.0):
        for r in np.arange(math.degrees(r0), math.degrees(r1) + 1e-9, 5.0):
            q = (math.radians(p), math.radians(r))
            s0 = ankle_linkage(g["drive"], g["rods"], g["ankle"], *q)
            assert s0 is not None, "a rod cannot reach at pitch %.0f roll %.0f" % (p, r)
            for _, c, rod in s0:
                a = math.degrees(math.acos(abs(c @ rod) / np.linalg.norm(c) / np.linalg.norm(rod)))
                worst["sing"] = min(worst["sing"], a)
            h = 1e-4                         # J: crank angles per ankle angle
            J = np.zeros((2, 2))
            for k in range(2):
                dq = list(q)
                dq[k] += h
                s1 = ankle_linkage(g["drive"], g["rods"], g["ankle"], *dq)
                J[:, k] = [(b[0] - a_[0]) / h for a_, b in zip(s0, s1)]
            # ankle torque = J^T motor torque; the most each pure need allows with |tau| <= peak
            for k, key in ((0, "pitch"), (1, "roll")):
                e = np.zeros(2)
                e[k] = 1.0
                tau = np.linalg.solve(J.T, e)          # motor torques per N.m at the ankle
                worst[key] = min(worst[key], g["peak"] / np.abs(tau).max())
    print("ankle: over its range the rods hold %.0f N.m in pitch (need %.1f) and %.0f in roll "
          "(need %.1f); crank and rod at least %.0f deg off a line"
          % (worst["pitch"], need_pitch, worst["roll"], need_roll, worst["sing"]))
    assert worst["pitch"] >= 1.3 * need_pitch and worst["roll"] >= 1.3 * need_roll, worst
    assert worst["sing"] >= 10.0, worst


def check_model():
    """the full robot's URDF, scene and hang keyframe, generated under urdf/full/: it
    loads, has 29 hinges and 29 actuators, weighs what the CAD weighs, and hangs still"""
    import mujoco
    out = {}
    for cmd in (["cad/tools/gen_urdf.py"], ["urdf/_gen_scene.py"], ["urdf/_gen_keyframe.py", "--write"]):
        out[cmd[0]] = _run([PY] + cmd, robot="full")
    xml = os.path.join(ROOT, "urdf", "full", "scene.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    assert m.nq == 7 + 29 and m.nu == 29, (m.nq, m.nu)
    cad = float(re.search(r"total mass ([0-9.]+) kg", out["cad/tools/gen_urdf.py"]).group(1))
    sim = float(m.body_subtreemass[m.body("base_link").id])
    assert abs(cad - sim) < 1e-4, (cad, sim)
    drift = float(re.search(r"drift over 5 s: ([0-9.]+) mm", out["urdf/_gen_keyframe.py"]).group(1))
    assert drift < 1.0, drift
    check_variant()
    print("model: urdf/full loads, 29 hinges, 29 actuators, %.3f kg as the CAD, hangs %.3f mm in 5 s"
          % (sim, drift))


def check_stow():
    """spec.STOW at every lesson's hang, simulated on the full robot (stow.settle): it stays
    hooked (the drift is reported: off the wall, lesson 4 still swings), no leg touches a
    wall or rung, and hip pitch and knee hold it at or
    under the RS04's rated 40 N.m. Reported, not asserted: the lidar cells per rung against
    v2's (the bigger arm actuators hide part of the held rung: the user kept the 111 mm head
    riser, 2026-11-25), and the reference climbs, whose v2 motions put the pelvis's underside
    on the wall whatever the stow (they are searched again when the full robot enters the
    simulation; this round is CAD only)."""
    import mujoco, numpy as np
    sys.path.insert(0, HERE)
    os.environ["ROBOT"] = "full"
    import stow as S
    from rl.lessons import LESSONS
    st = _spec("full", "spec.STOW")
    stow = {**S.stow_of(0.0, 0.0), **st}
    bad = []
    for k in sorted(LESSONS):
        r = S.settle(k, stow)
        print("  lesson %d: %s, %.2f mm / 2 s, legs touching %s, lidar cells %s (v2 %s)"
              % (k, "hooked" if r["hooked"] else "FELL", r["drift"] * 1000, r["legs"] or "nothing",
                 r["seen"], r["v2"]))
        if not r["hooked"] or r["legs"]:       # lesson 4 hangs clear of its wall: a pendulum
            bad.append(k)
    W = [w for nm, w in S.worlds(every=20) if nm.startswith("climb")]
    climb = min(w.clearance(stow, cap=0.2) for w in W)
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "urdf", "full", "scene.xml"))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("hang_on_rung").id)
    for n, v in stow.items():
        d.qpos[m.joint(n).qposadr[0]] = v
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_inverse(m, d)
    gear = _spec("full", "{k: v['gear'][0] for k, v in spec.ANKLE_JOINT.items() if k.startswith('knee_')}")
    hold = {n: abs(float(d.qfrc_inverse[m.joint(n).dofadr[0]])) / gear.get(n, 1.0) for n in stow
            if n.startswith("knee_") or (n.startswith("hip_") and n.endswith("_pitch"))}   # at the motor
    print("stow %s: holding %s N.m (RS04 rated 40); reference climbs, legs vs walls and rungs: "
          "%.1f mm (reported)" % ({k: round(math.degrees(v)) for k, v in st.items()},
                                   {k: round(v, 1) for k, v in hold.items()}, climb * 1000))
    assert not bad, bad
    assert max(hold.values()) <= 40.0, hold


# ---------------------------------------------------------------- legs, standing
# A foot on the ground is a wrench on it: the floating base has no motor, so its six rows
# of the inverse dynamics (qacc = 0) fix the stance feet's ground wrench, and the joints
# carry the rest: tau = bias - J^T F. The stance leg is posed by its knee with hip and ankle
# pitch splitting it (torso upright, sole flat), then leaned in pitch and roll (hip against
# ankle) until the centre of pressure sits where the case puts it on the sole.
TOE, HEEL_X, EDGE, MID = 0.111, -0.070, 0.056, 0.0205   # CoP targets ahead of / beside the ankle (m)
CASES = [  # name, stance sides, knee deg, gravity x, CoP (x, y) on the sole (y + outward)
    ("squat, both feet, knee 90", "LR", 90, 1.0, (MID, 0.0)),
    ("one foot, knee 30", "L", 30, 1.0, (MID, 0.0)),
    ("walking, one foot, 1.5 x weight, knee 20", "L", 20, 1.5, (MID, 0.0)),
    ("one foot, COM over the toe", "L", 20, 1.0, (TOE, 0.0)),
    ("one foot, COM over the heel", "L", 20, 1.0, (HEEL_X, 0.0)),
    ("one foot, COM over the outer edge", "L", 20, 1.0, (MID, EDGE)),
    ("one foot, COM over the inner edge", "L", 20, 1.0, (MID, -EDGE))]


def _stand(m, d, stance, knee, cop, sg):
    """pose, solve, return (joint torques, CoP error m, foot tilt rad)"""
    import mujoco, numpy as np
    from scipy.optimize import fsolve
    down = _spec("full", "spec.ARMS_DOWN_ABD")
    def J(n):
        return m.joint(n).qposadr[0], m.joint(n).dofadr[0]
    def pose(lean):
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[J("shoulder_L_abduction")[0]], d.qpos[J("shoulder_R_abduction")[0]] = down, -down
        for s in "LR":
            k = math.radians(knee if s in stance else 40.0)
            dp, dr = (lean if s in stance else (0.0, 0.0))
            sy = 1.0 if s == "L" else -1.0
            d.qpos[J("knee_" + s)[0]] = k
            d.qpos[J("hip_%s_pitch" % s)[0]] = sg[0] * (k / 2 + dp)
            d.qpos[J("ankle_%s_pitch" % s)[0]] = sg[1] * (k / 2 - dp)
            d.qpos[J("hip_%s_roll" % s)[0]] = sg[2] * sy * dr
            d.qpos[J("ankle_%s_roll" % s)[0]] = sg[3] * sy * dr
        mujoco.mj_forward(m, d)
    def solve():
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        mujoco.mj_inverse(m, d)
        feet, A = [], []
        for s in stance:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            p = d.xanchor[m.joint("ankle_%s_pitch" % s).id].copy()
            mujoco.mj_jac(m, d, jp, jr, p, m.body("foot_" + s).id)
            feet.append(p)
            A.append(np.vstack([jp, jr]))                 # F = (force, moment) at the ankle
        A = np.hstack([a.T for a in A])                   # nv x 6n
        F = np.linalg.pinv(A[:6]) @ d.qfrc_inverse[:6]
        tau = d.qfrc_inverse - A @ F
        cops = []
        for i, (s, p) in enumerate(zip(stance, feet)):
            f, n = F[6 * i:6 * i + 3], F[6 * i + 3:6 * i + 6]
            h = SOLE                                       # ankle axis over the ground
            dx = (-n[1] - h * f[0]) / f[2]                 # moment about the ground point is 0
            dy = (n[0] - h * f[1]) / f[2]
            sy = 1.0 if s == "L" else -1.0
            cops.append((dx - cop[0], sy * dy - cop[1]))
        return tau, np.array(cops).ravel()
    SOLE = _spec("full", "spec.SOLE")
    def err(lean):
        pose(lean)
        e = solve()[1]
        return e[:2] if len(stance) == 1 else [e[0], lean[1]]    # both feet: no lean sideways
    lean = fsolve(err, [0.0, 0.0], xtol=1e-10)
    # the residual of what was solved: on both feet each sole's sideways CoP is not, the
    # COM sits 0.4 mm left (spec.BAY leg_can) and the two soles share that unevenly
    e = np.abs(err(lean)).max()
    tau, _ = solve()
    R = d.xmat[m.body("foot_" + stance[0]).id].reshape(3, 3)
    return {m.joint(j).name: tau[m.jnt_dofadr[j]] for j in range(1, m.njnt)}, \
        float(e), float(math.acos(min(1.0, R[2, 2]))), lean


def leg_needs(verbose=False, cases=None):
    """joint (left side name) -> worst |N.m| over `cases` (CASES), standing on urdf/full"""
    import mujoco, numpy as np
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "urdf", "full", "scene.xml"))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    g0 = m.opt.gravity.copy()
    import itertools
    sg = None                     # hip/ankle pitch and roll signs that keep the sole flat
    for c in itertools.product((1.0, -1.0), repeat=4):
        _, e, tilt, lean = _stand(m, d, "L", 40, (MID, 0.0), c)
        if e < 1e-4 and tilt < 0.035 and abs(lean[1]) > 1e-3:
            sg = c
            break
    assert sg, "no sign of hip/ankle keeps the sole flat"
    need = {}
    for name, stance, knee, gx, cop in (cases or CASES):
        m.opt.gravity[:] = g0 * gx
        tau, e, tilt, lean = _stand(m, d, stance, knee, cop, sg)
        assert e < 1e-4 and tilt < 0.035, (name, e, tilt)   # hip roll before the pitches: under 2 deg
        if verbose:
            print("  %-40s lean %+5.1f / %+5.1f deg  hip %5.1f  knee %5.1f  ankle p %5.1f r %5.1f N.m"
                  % (name, *np.degrees(lean), abs(tau["hip_L_pitch"]), abs(tau["knee_L"]),
                     abs(tau["ankle_L_pitch"]), abs(tau["ankle_L_roll"])))
        for j, t in tau.items():
            if j.startswith(("hip_", "knee_", "ankle_")):
                k = j.replace("_R", "_L")
                need[k] = max(need.get(k, 0.0), abs(t))
    m.opt.gravity[:] = g0
    return need


def check_legs():
    """every leg actuator >= 1.3x its worst need standing (both feet, one foot, the COM
    over the toe, heel and edges) and >= 1.0x walking (one foot carrying 1.5 x the weight:
    the hip roll's AKE90-8 holds 1.18x there at 69.2 kg, the user's choice over an X12-320,
    2026-11-25); the ankle rods hold the measured ankle needs (check_ankle)"""
    walk = [c for c in CASES if c[0].startswith("walking")]
    stand = leg_needs(verbose=True, cases=[c for c in CASES if c not in walk])
    moving = leg_needs(cases=walk)
    # a joint driven through a stage (the belt knee, spec.ANKLE_JOINT): its motor's peak x the ratio
    peak = _spec("full", "{j[0]: (MOTORS[spec.MOTOR_OF[j[0]]]['peak_Nm'] if spec.MOTOR_OF[j[0]] else "
                         "sum(MOTORS[spec.ANKLE_JOINT[j[0]]['motor']]['peak_Nm'] * g for g in spec.ANKLE_JOINT[j[0]]['gear'])) "
                         "for j in spec.JOINTS if j[0].startswith(('hip_L', 'knee_L'))}")
    short = []
    for j in sorted(peak):
        rs, rw = peak[j] / max(stand[j], 1e-9), peak[j] / max(moving[j], 1e-9)
        print("  %-16s standing %6.1f N.m %5.2fx   walking %6.1f N.m %5.2fx   peak %4.0f%s" % (
            j, stand[j], rs, moving[j], rw, peak[j], "  <-- short" if rs < 1.3 or rw < 1.0 else ""))
        if rs < 1.3 or rw < 1.0:
            short.append(j)
    check_ankle(max(stand["ankle_L_pitch"], moving["ankle_L_pitch"]),
                max(stand["ankle_L_roll"], moving["ankle_L_roll"]))
    assert not short, short
    print("legs: every leg actuator >= 1.3x standing and >= 1.0x walking")


C0R_AK60 = 2520.0      # N, AK60-39 V3.0 basic static load rating (CubeMars datasheet; Cr 2000)


def _waist_load(m, d, part, gx=1.0):
    """the waist bearing's static load: `part` ('upper': chest and arms on the pelvis, or
    'lower': pelvis and legs hanging from the chest) under gravity x gx -> (axial N, radial
    N, tilting N.m), in the waist axis' frame"""
    import mujoco, numpy as np
    j = m.joint("waist_yaw").id
    b = m.jnt_bodyid[j]
    up_m, up_c = m.body_subtreemass[b], d.subtree_com[b]
    tot_m, tot_c = m.body_subtreemass[1], d.subtree_com[1]
    if part == "upper":
        mass, com = up_m, up_c
    else:
        mass = tot_m - up_m
        com = (tot_m * tot_c - up_m * up_c) / mass
    F = mass * m.opt.gravity * gx
    M = np.cross(com - d.xanchor[j], F)
    ax = d.xaxis[j]
    fa = abs(F @ ax)
    return fa, float(np.linalg.norm(F - fa * np.sign(F @ ax) * ax)), float(np.linalg.norm(M - (M @ ax) * ax))


def check_waist():
    """the waist's AK60-39: its bearing's static equivalent load (0.6 Fr + 0.5 Fa, not less
    than Fr) <= C0r / 1.3 hanging (legs stowed, the pelvis and legs under it) and standing
    and walking (the chest and arms on it, the walking case at 1.5 x weight); its yaw
    torque turning the chest and arms at 10 rad/s^2 <= 72 / 1.3. The tilting moment is
    reported: CubeMars publishes no moment rating."""
    import mujoco, numpy as np
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "urdf", "full", "scene.xml"))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    stow = _spec("full", "spec.STOW")
    mujoco.mj_resetDataKeyframe(m, d, m.key("hang_on_rung").id)
    for n, v in stow.items():
        d.qpos[m.joint(n).qposadr[0]] = v
    mujoco.mj_forward(m, d)
    rows = [("hanging, legs stowed", _waist_load(m, d, "lower"))]
    g0 = m.opt.gravity.copy()
    for name, stance, knee, gx, cop in CASES:
        if name.startswith(("squat", "walking", "one foot, knee")):
            _stand(m, d, stance, knee, cop, (-1, -1, 1, -1))
            rows.append((name, _waist_load(m, d, "upper", gx)))
    worst = 0.0
    for name, (fa, fr, mt) in rows:
        p0 = max(0.6 * fr + 0.5 * fa, fr)
        worst = max(worst, p0)
        print("  %-42s axial %6.1f N  radial %5.1f N  P0 %6.1f N (C0r/P0 %5.1fx)  tilting %5.1f N.m"
              % (name, fa, fr, p0, C0R_AK60 / p0, mt))
    # yaw: the chest and arms spun up at 10 rad/s^2, nothing else moving, no gravity
    mujoco.mj_resetDataKeyframe(m, d, m.key("hang_on_rung").id)
    m.opt.gravity[:] = 0.0
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    dof = m.joint("waist_yaw").dofadr[0]
    d.qacc[dof] = 10.0
    mujoco.mj_inverse(m, d)
    yaw = abs(float(d.qfrc_inverse[dof]))
    m.opt.gravity[:] = g0
    print("waist: bearing worst P0 %.0f N, C0r %.0f N = %.1fx; yaw at 10 rad/s^2 %.1f N.m of 72 (%.1fx)"
          % (worst, C0R_AK60, C0R_AK60 / worst, yaw, 72.0 / yaw))
    assert C0R_AK60 / worst >= 1.3, worst
    assert 72.0 / yaw >= 1.3, yaw


def _climbs():
    """every reference climb as (name, its wall's layout, v2 states [(qpos, qvel)]): the 27
    kept in rl.rsi, and rsi.DROPPED's fixtures replayed open loop (their motion still says
    what a swing asks of a joint)"""
    sys.path.insert(0, ROOT)
    from rl import rsi
    z = rsi.load()
    out = []
    for c in range(len(z["climb_family"])):
        k = z["climb"] == c
        out.append(("climb %d (family %d)" % (c, z["climb_family"][c]), rsi.climb_wall(z, c),
                    list(zip(z["qpos"][k], z["qvel"][k]))))
    for f, h, side, rows, name in [(8, 0.60, 1, 1, "climb_f8_b60")] + [
            (9, g, sd, 1, "climb_f9%s_g%d" % ("m" if sd < 0 else "", round(g * 100)))
            for g in (0.55, 0.60, 0.70, 0.80, 0.90) for sd in (1, -1)]:
        if (f, h, side) not in rsi.DROPPED:
            continue
        path = os.path.join(ROOT, "rl", "fixtures", name + ".npz")
        lay = rsi.ref_wall(f, h, side, rows)
        states, why = _replay_until(lay, path)
        out.append(("%s (dropped: %s)" % (name, why), lay, states))
    return out


def _replay_until(lay, path):
    """rsi.replay without its asserts: a dropped climb's setpoints through ClimbTask.step on
    its wall, its states up to wherever it ends today -> ([(qpos, qvel)], how it ended)"""
    import numpy as np
    from rl.climb_task import ClimbTask, ACT_SCALE
    from rl.mjx_scene import c_scene
    xml, spec = c_scene(dict(lay, timeout=90.0))
    try:
        t = ClimbTask(spec=spec, xml=xml, visual=False)
        t.rng = np.random.default_rng(0)
        t.reset()
    finally:
        os.remove(xml)
    d, out, why = t.env.data, [], "ran out"
    for c in np.load(path)["ctrl"]:
        out.append((d.qpos.copy(), d.qvel.copy()))
        *_, te, tr, info = t.step(np.clip((c - t.ctrl) / ACT_SCALE, -1, 1))
        if te or tr:
            why = info.get("reason")
            break
    return out, why


def check_arms_dynamic():
    """every arm joint's speed over the reference climbs (v2's recorded states, rsi.DROPPED's
    replayed): its 99.9th percentile under the actuator's no-load speed (the maxima, a few
    contact impacts that backdrive a joint, are reported). Torques are not judged here: v2's
    own servos sit at their peak through much of these climbs (CEM found them within those
    limits: shoulder flexion at 120 N.m 99 % of steps, 3.5x its static hold), so their motion
    says what v2's actuators allowed, not what the full robot needs; and replayed on the full
    robot neither a hook wrench nor v2's contact forces x the mass ratio holds it up (base
    residual ~1.2 kN). The full robot's dynamic torques come with its own climbs, searched
    again within its actuators' limits when it enters the simulation; its static hold (box
    and climbing envelope) is check_arms_static."""
    import numpy as np
    sys.path.insert(0, ROOT)
    import mujoco
    lim = _spec("full", "{j[0]: MOTORS[spec.MOTOR_OF[j[0]]]['noload_rpm'] * 3.14159265 / 30 "
                        "for j in spec.JOINTS if spec.MOTOR_OF.get(j[0]) "
                        "and not j[0].startswith(('hip_', 'knee_', 'ankle_', 'waist'))}")
    v2 = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "urdf", "scene.xml"))
    w = np.abs(np.vstack([np.array([v for _, v in st]) for _, _, st in _climbs()]))
    short = []
    for j, wmax in sorted(lim.items()):
        x = w[:, v2.joint(j).dofadr[0]]
        p = np.percentile(x, 99.9)
        print("  %-22s p99.9 %5.2f rad/s, max %5.2f (%d steps over), no-load %5.2f: %.1fx"
              % (j, p, x.max(), (x > wmax).sum(), wmax, wmax / p))
        if p > wmax:
            short.append(j)
    assert not short, short
    print("arms dynamic: every arm joint's speed (p99.9) under its no-load speed, %d states" % len(w))


def _stowed_scene(path):
    """urdf/full/scene.xml with every leg joint pinned at spec.STOW (0 where unnamed)"""
    stow = _spec("full", "spec.STOW")
    s = open(os.path.join(ROOT, "urdf", "full", "scene.xml")).read()
    def pin(m):
        v = stow.get(m.group(2), 0.0)
        return '%srange="%.6f %.6f"' % (m.group(1), v - 1e-4, v + 1e-4)
    s = re.sub(r'(<joint name="((?:hip|knee|ankle)_[^"]+)"[^>]*?)range="[^"]+"', pin, s)
    open(path, "w").write(s)


def arm_static():
    """joint -> (need box, need climbing envelope, capacity) for the full robot's arm
    joints, hanging, the legs at the stow (cad/tools/torque.py, 25000 poses each)"""
    tmp = os.path.join(ROOT, "urdf", "full", "_stowed.xml")
    _stowed_scene(tmp)
    out = {}
    try:
        for tag, extra in (("box", []), ("env", ["--envelope"])):
            dst = os.path.join(ROOT, "urdf", "full", "t_%s.json" % tag)
            _run([PY, "cad/tools/torque.py", "25000", tmp, dst] + extra)
            out[tag] = json.load(open(dst))["joints"]
    finally:
        os.remove(tmp)
    acts = json.load(open(os.path.join(ROOT, "urdf", "full", "_actuators.json")))["actuators"]
    return {j: (out["box"][j]["need"], out["env"][j]["need"], acts[j]["peak_Nm"]) for j in out["box"]
            if not j.startswith(("hip_", "knee_", "ankle_", "waist"))}


def check_arms_static():
    """every arm joint holds 1.3x its need, over the joint box and the climbing envelope,
    hanging the whole robot with its legs folded"""
    res = arm_static()
    # the user's ruling, 2026-11-25: the left wrist deviation (AK80-64) at 1.29x over the
    # box (0.5 N.m short of 1.3x, 1.30x on the climbing envelope) is accepted
    floor = dict(wrist_L_deviation=1.29)
    short = []
    for j, (nb, ne, pk) in sorted(res.items()):
        lo = floor.get(j, 1.3)
        print("  %-22s need %6.1f box %6.1f env  peak %5.0f  %4.2fx / %4.2fx%s" % (
            j, nb, ne, pk, pk / max(nb, 1e-9), pk / max(ne, 1e-9),
            "  <-- under %.2f" % lo if min(pk / max(nb, 1e-9), pk / max(ne, 1e-9)) < lo
            else "  (accepted >= %.2f)" % lo if lo < 1.3 else ""))
        if min(pk / max(nb, 1e-9), pk / max(ne, 1e-9)) < lo:
            short.append(j)
    assert not short, short
    print("arms static: every arm joint >= 1.3x peak, box and climbing envelope (wrist_L_deviation 1.29x, accepted)")


def check_all():
    """every check above, in order (the model regenerated first), then the report: the
    whole robot's mass, the walking runtime on the 48 V 6 Ah battery (288 Wh) at an
    estimated walking power (the leg actuators' rated power x 0.3), and the chest lidar's
    view of the ground at the feet, standing (reported, no threshold)"""
    for name in ("variant", "kinematics", "model", "ankle", "legs", "waist", "stow",
                 "arms_static", "arms_dynamic"):
        print("== %s" % name, flush=True)
        globals()["check_" + name]()
    check_report()


def check_report():
    import mujoco, numpy as np
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "urdf", "full", "scene.xml"))
    mass = m.body_subtreemass[m.body("base_link").id]
    legs = _spec("full", "sum(MOTORS[spec.MOTOR_OF[j[0]]]['rated_Nm'] * MOTORS[spec.MOTOR_OF[j[0]]]['rated_rpm']"
                         " * 3.14159265 / 30 for j in spec.JOINTS if spec.MOTOR_OF.get(j[0]) and "
                         "j[0].startswith(('hip_', 'knee_'))) + sum(MOTORS[k]['rated_Nm'] * MOTORS[k]['rated_rpm']"
                         " * 3.14159265 / 30 for _, k, *_ in spec.ANKLE_DRIVE)")
    walk = 0.3 * legs
    # the chest lidar standing: rays over its field of view, the share hitting the floor
    # within 0.6 m of the base's projection (legs straight, arms down)
    d = mujoco.MjData(m)
    d.qpos[3] = 1.0
    down = _spec("full", "spec.ARMS_DOWN_ABD")
    d.qpos[m.joint("shoulder_L_abduction").qposadr[0]], d.qpos[m.joint("shoulder_R_abduction").qposadr[0]] = down, -down
    mujoco.mj_kinematics(m, d)
    sole = min(d.geom_xpos[g][2] - 0.01 for g in range(m.ngeom)
               if m.body(m.geom_bodyid[g]).name.startswith("foot_"))
    site = m.site("lidar_chest").id
    o, R = d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3)
    robot = m.body("base_link").id
    hit = total = 0
    # the chest's own mesh and hulls are left out: the lidar sits sunk in it, behind a window
    # its panel's convex hull closes; what can block it standing is the arms, pelvis and legs
    chest = m.body("torso_upper").id
    for el in np.radians(np.arange(-7.0, 52.1, 1.0)):
        for az in np.radians(np.arange(0.0, 360.0, 2.0)):
            u = R @ np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            if u[2] >= -1e-6:
                continue
            total += 1
            t_floor = (sole - o[2]) / u[2]
            gid = np.zeros(1, np.int32)
            t_body = mujoco.mj_ray(m, d, o + u * 0.02, u, None, 0, chest, gid)   # arms, pelvis, legs
            if t_body >= 0 and t_body + 0.02 < t_floor:
                continue                                  # the body (thighs, arms) is in the way
            p = o + u * t_floor
            if np.hypot(*(p[:2] - d.xpos[robot][:2])) < 0.6:
                hit += 1
    print("report: %.2f kg; walking ~%.0f W (0.3 x the leg actuators' %.0f W rated) -> ~%.1f h on 288 Wh "
          "(the arms, lidars and Jetson not counted); the chest lidar standing sees the floor within "
          "0.6 m of the feet on %d of its %d downward rays"
          % (mass, walk, legs, 288.0 / walk, hit, total))


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()["check_" + name]()
