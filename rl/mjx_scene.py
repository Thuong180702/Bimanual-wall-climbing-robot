"""The robot as MJX can simulate it, and the lessons as data instead of scene files.

    python -m rl.mjx_scene      # build it, write urdf/lessons/mjx_*.xml, check every lesson

The CPU path simulates `urdf/scene.xml` as it is: every CAD part's convex hull
collides with everything, and each lesson is its own scene file. MJX can run
neither. Its JAX backend has no box-cylinder or mesh-cylinder collision (hook
against rung), pays for every pair on every env and substep whether they are near
or not, has no `jointlimitfrc` sensor, and compiles ONE model for every env of a
batch -- so a batch cannot mix lesson scenes.

So each kind of contact gets the shape MJX computes EXACTLY for it, and where no
exact pair exists the approximation is chosen by one measurement: copy the full
robot's state into this one along a trained gait, and compare what touches and
where one control step takes each (`python -m rl.mjx_check model`).

  * WALLS are planes, and what meets them is the robot's own collision geometry,
    every body's, as scene.xml draws it: MJX's plane against a mesh (its vertices)
    or a cylinder is exact and cheap. (The first version used the box round each
    part, and the body's box stood 9 mm proud of its chamfered hull into the wall
    whenever the body yawed: at the same states it touched the wall in 63 % of them
    at 200 N, the full robot in 9 % at 36 N.) ponytail: a plane does not end -- the
    front wall does, at y = -4 and 4, z = -2.5 and 3.5 (rl.lessons); no lesson or
    family reaches past it, mirrored or not.
  * RUNGS are capsules, their straight part the real rung's length and the caps
    past its ends, so every seat on a rung holds exactly as on the cylinder and the
    error is only under a hook hanging off the end. (Caps inside the length left
    the last 15 mm of every rung unable to hold a hook edge, and the gap lessons
    land hooks exactly there.) That error is large: the real rungs are flat-ended
    cylinders, and at the same states a hook near an end bore 1398 N on this robot
    against 199 on the full one (mean, gpu1's gait on lesson 2; p90 2855 N apart),
    the pad and wrist the policy reads with it (`rl.mjx_check model` reports it).
    Pulling the caps 10 mm in matched the force (151 against 199) and broke MJX
    against C instead: C MuJoCo gives two capsules lying parallel two contacts, at
    the ends of their overlap, and MJX one -- at the inner end of a rod that runs
    past the straight part -- and each hook plate has a rod along the rung at each
    end. One control step apart went 2.8e-6 -> 8.4e-5 rad (p50) near rung ends, and
    gpu1 on lesson 2 from 22 and 20 of 40 (C, MJX) to 13 and 28. So the caps stay;
    what a hold half off an end is worth is `climb_task.SEAT_TOL`'s business. The way
    to match both: those parallel rods as two balls each, at their ends, where C's
    two contacts fall -- more pairs for MJX, to be timed on the GPU first.
  * what can touch a rung is capsules and balls, never boxes: MJX's capsule-box
    test separates only on the box's face normals, so a 2.4 m rung passing a 12 mm
    plate at a slight angle "penetrates" it (50-75 mm contacts where C saw none,
    5000-12000 N where C had 200). Capsule-capsule is exact. So each hook plate is
    drawn in rods of 0.25 mm radius lying in its broad faces -- a comb across the
    hook's width and one along each end (`_round`) -- the box itself to 0.25 mm, and
    the rung finds it where it finds the box: on the edges of the contact patch.
    (Two 6 mm rods through each plate rounded the plate edges off by 6 mm, where a
    rung tilted against the hook actually bears: 35 % of the full robot's hook load
    at the same states.) The finger motor is a ring of rods tangent to it and flush
    with its ends (a capsule's cap stood 27.5 mm proud of the end the rung meets); the
    rest of the arm, link by link, a capsule fitted to its motor and one to each of its
    bracket plates (ARM) -- a row under the arms, the posture of every climb up, down or
    over, went through them before; base and torso a rod up each vertical edge, and the
    lidar cage its own capsules, as scene.xml has them. The lidars inside the cage meet
    nothing here.
  * one HAND against the other: the hooks' own boxes (box-box is a full SAT in MJX)
    and the three motors above each hook, as the cylinders they are. (Boxes round
    the motors met 9 mm deep in 11 % of the full robot's states on lesson 2, where
    the full robot's motors touched lightly in 1-2 %: the trailing hand could not
    come up beside the lead one, and win17 timed out on the wide gap.) Measured at
    the same states on lesson 2: the hands touch in 4.6 % of them on the full robot
    and 4.4 % on this one, 2.4 N mean both. MJX's cylinder-cylinder is an SDF
    descent, not exact: at those states it misses 27 % of C's contacts, all shallow,
    and is 1.3 mm off at p90 -- light, rare contacts, so kept. The exact
    alternatives cost more than they are worth: a convex hull per link (full SAT)
    needs ~100 GB of scratch for 4096 envs, and a 12-gon prism per motor (5.3 GB)
    misses 45 % of the full robot's hand contacts on its flat facets.
  * collisions are an explicit list of PAIRS, not everything-with-everything. Each
    pair gets the friction and softness MuJoCo would have mixed for those two geoms,
    so the pad still defines the hook's contact.
  * the lessons become DATA: N_RUNG rung slots and one extra wall slot whose
    geometry is set per env (`layout`). A slot a lesson does not use is parked far
    away. That is what lets one compiled model train every lesson at once.

Measured against the full robot at its own states along win17's gait (lessons 1,
1.85, 2; `rl.mjx_check model`, seeded): one control step apart by 0.55-2.4e-3 rad
(p50) where the first version was 1.1e-2, the body by 0.08-0.24 mm where it was 8; a
hook on its rung in one robot and not the other in at most 2.4 % of the states on
lessons 1 and 2, and 6.5 % on 1.85 -- a hook off a rung's end, still on its cap; the
finger motor on a rung in this robot alone 0.0 %, in the full one alone at most
3.2 %; and win17 on the full robot, this one in C and this one in MJX, 8 episodes on
each of its eleven lessons: 76, 73 and 76 of 88 (70, 74 and 70 once a hold half off a
rung's end stopped counting, climb_task.SEAT_TOL); 24 each on lessons 1, 1.5, 1.7,
1.85 and 2: 90, 94 and 83 of 120, lesson 2 alone 10, 9 and 3 of 24 (the gap lesson,
hooks on rung ends: the way to watch -- though the full robot alone moved 97 -> 90 of
120 across an unrelated scene change, so +-7 is noise here). One known difference: a
49 N pull along the rung slides the full robot's hooks 20-24 mm in 6 s and these
14-20 -- more contacts in parallel creep less. It errs toward the dragging this
task charges for, so it is left, not tuned away.

Masses, inertias, joints, actuators and the keyframe are scene.xml's, untouched:
the geoms carried no mass (every body has an explicit <inertial>).

The same builder writes each lesson as a plain MuJoCo scene (`c_path`), so the
simplified robot can be run and cross-checked in C MuJoCo, where the full robot
also lives. `rl.mjx_check` does that.
"""
import hashlib, os
import numpy as np, mujoco
from . import SCENE
from .lessons import LESSONS, OUT, PAD, canon_axis, lesson_tag, xml_path

N_RUNG = max(len(v['rungs']) for v in LESSONS.values())     # 6: lesson 6
PARK = 50.0              # m: where an unused slot waits, out of everyone's reach
# The solver, which is ~80 % of an MJX step. C MuJoCo allows scene.xml 100 Newton
# iterations and, under win17's policy, uses 6-7 on every step to reach its 1e-8
# tolerance. A vmapped MJX batch runs to the cap on every env, so the cap is the
# cost. Measured as one control step from the same C state against C, joint error
# p50 / p90 and the worst contact force: 8 iterations 3e-6 / 9e-6, 8 N; 6 the same,
# 53 N; 4 2e-5 / 3e-4, 182 N; 2 6e-4 / 2e-3, 280 N.
ITERATIONS, LS_ITERATIONS = 6, 10
# MJX computes every pair's contacts every substep (its arrays are static), but it
# hands the solver only the deepest MAX_CONTACTS of them. C, which keeps them all,
# never had more than 17 at once on this robot: 600 steps of random actions on every
# lesson and of win17's policy on its eleven, 17 400 control steps, p99.9 13-14
# (`rl.mjx_check physics` asserts it). 24 keeps 40 % above that, and a batch counts
# every substep where all 24 are in use (`capped` in the training log), which is
# the sign the cap is dropping real contacts. Each one kept is four rows of the
# solver's system.
MAX_CONTACTS = 24

SIDES = 'LR'
# what each body keeps as boxes (walls, the other hand). 'hull': round all its
# collision meshes; 'motor': round its actuator's cylinder; 'web': round its plates
# with 'web' in the name; 'shank' / 'lip': the hook's own two boxes.
KEEP = {'base_link': ('hull',), 'torso_upper': ('hull',),
        'shoulder_%s': ('motor',), 'upper_arm_%s': ('motor',),
        'humerus_rot_%s': ('motor',), 'fore_arm_%s': ('motor',),
        'forearm_rot_%s': ('motor',), 'wrist_%s': ('motor', 'web'),
        'wrist_dev_%s': ('motor', 'web'), 'hook_%s': ('shank', 'lip')}
# what can meet a rung, all of it round (`_round`): the body climbing past a row,
# and the hand. (What can meet a WALL is every collision geom of scene.xml's robot,
# as it is: see `build`.)
BODY_ROD = 0.02          # m, the radius of the rods up the body's edges
HOOK_EDGE = 0.00025      # m, the radius of the rods a hook plate is drawn in
HOOK_COMB = 5            # rods across the hook's 65 mm width, per broad face (16 mm apart)
MOTOR_RODS = 24          # rods round the finger motor, 15 deg apart: 0.24 mm deep between two
# the arm, link by link from the shoulder to the wrist: two capsules each, one fitted
# to the link's motor and one to its brackets (`_arm`). A robot hanging with a row under
# its arms -- every climb up, down or over -- rests that row against them, and without
# these the arms went through it. Measured as the stand-in's surface against the real
# link's near a rung (rl.mjx_check model, rows below): one capsule a link missed the
# motors set off its axis (the forearm roll's by 17 mm, median); two, at ARM_Q 75, are
# unbiased to 3 mm and within -7..+5 mm (p10..p90) for every link but the upper arm,
# whose thin side plate a round capsule over-covers by up to 20 mm in its worst tenth.
# Tried and worse: a ball on each motor's axis (misses its rim by 9 mm), and rings of
# rods on the motors (exact round the side, but a rung lying on a motor's flat face
# sinks up to 7 mm before it meets a rod end, and these faces turn to the wall).
ARM = ('shoulder', 'upper_arm', 'humerus_rot', 'fore_arm', 'forearm_rot', 'wrist')
ARM_Q = 75               # percentile of a part's points' distance from its axis: the radius
# (and every `<link>:arm<i>` capsule `_round` fits to the arm: how many a link gets is
# how many parts scene.xml gives it, so `build` takes those names from the fit)
RUNG = [f'{b}:rod{i}' for b in ('base_link', 'torso_upper') for i in range(4)] + [
    n % sd for sd in SIDES for n in (
        [f'hook_%s:{t}_e{k}' for t in ('shank', 'lip') for k in range(2 * HOOK_COMB + 4)]
        + [f'wrist_dev_%s:rod{i}' for i in range(MOTOR_RODS)])]
# one hand against the other: the hooks' own boxes (box-box is exact in MJX), and the
# motor cylinder of each of the three links above a hook against each of the other
# hand's -- what the full robot's hands meet with (measured: motor on motor, and
# rarely a motor on the other's bracket).
DISTAL = ('forearm_rot', 'wrist', 'wrist_dev')
SELF = [(f'hook_L:{a}', f'hook_R:{b}') for a in ('shank', 'lip') for b in ('shank', 'lip')]
# sensors the RL task never reads, and jointlimitfrc, which MJX does not have
DROP_SENSORS = ('lim_', 'imu_quat', 'imu_vel', 'imu_angvel', 'imu_angacc',
                'imu_linacc', 'torso_pos', 'com_')

# the grip pad (`lessons.PAD`), read back from its patch so it cannot drift from it
_pad = mujoco.MjSpec.from_string(
    '<mujoco><worldbody><body><freejoint/>' + PAD[0][1].replace('class="col" ', '')
    + '</body></worldbody></mujoco>')
_pg = _pad.worldbody.first_body().first_geom()
PAD_PARAMS = dict(priority=_pg.priority, friction=np.array(_pg.friction),
                  solref=np.array(_pg.solref), solimp=np.array(_pg.solimp))


def _full():
    """scene.xml compiled without its visuals: where the collision shapes come from"""
    s = mujoco.MjSpec.from_file(SCENE)
    s.compiler.discardvisual = True
    return s.compile()


def _boxes(m):
    """'body:tag' -> (pos, quat, half sizes) in the body's frame, from the full model"""
    out = {}
    for pattern, tags in KEEP.items():
        for sd in (SIDES if '%' in pattern else ' '):
            name = pattern % sd if '%' in pattern else pattern
            b = m.body(name).id
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]
            for tag in tags:
                if tag == 'motor':
                    (g,) = [g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
                    r, h = m.geom_size[g][:2]
                    out[f'{name}:motor'] = (m.geom_pos[g].copy(), m.geom_quat[g].copy(),
                                            np.array([r, r, h]))
                elif tag in ('shank', 'lip'):
                    bx = [g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
                    g = bx[0 if tag == 'shank' else 1]
                    out[f'{name}:{tag}'] = (m.geom_pos[g].copy(), m.geom_quat[g].copy(),
                                            m.geom_size[g].copy())
                else:
                    vs = []
                    for g in gs:
                        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or 'lidar' in m.mesh(m.geom_dataid[g]).name:
                            continue
                        if tag == 'web' and 'web' not in m.mesh(m.geom_dataid[g]).name:
                            continue
                        vs.append(_verts(m, g))
                    v = np.concatenate(vs)
                    lo, hi = v.min(0), v.max(0)
                    out[f'{name}:{tag}'] = ((lo + hi) / 2, np.array([1.0, 0, 0, 0]), (hi - lo) / 2)
    return out


def _verts(m, g):
    """a mesh geom's vertices in its body's frame"""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ R.reshape(3, 3).T + m.geom_pos[g]


def _cylinder(m, body):
    """(centre, unit axis, radius, half length) of a body's actuator housing"""
    b = m.body(body).id
    (g,) = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return m.geom_pos[g].copy(), R.reshape(3, 3)[:, 2], m.geom_size[g][0], m.geom_size[g][1]


def _round(m, boxes):
    """'body:tag' -> (geom type, pos, quat, size): what a rung can touch"""
    cap, ball = mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE
    out = {}
    for body in ('base_link', 'torso_upper'):          # a rod up each vertical edge
        c, _, h = boxes[f'{body}:hull']
        r = BODY_ROD
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            out[f'{body}:rod{i}'] = (cap, c + [sx * (h[0] - r), sy * (h[1] - r), 0.0],
                                     np.array([1.0, 0, 0, 0]), [r, h[2] - r, 0.0])
    for sd in SIDES:
        # Each hook plate is the box itself, drawn in rods of radius HOOK_EDGE lying
        # in its two broad faces: a comb of HOOK_COMB rods square to the rung across
        # the hook's width (y, the rung's direction when seated), the outer two on its
        # edges, and one along each end. Every rod is tangent to the faces it lies on,
        # so nothing is outside the box and its edges are rounded by HOOK_EDGE only.
        # A rung touches a plate along the boundary of its contact patch: its edges
        # when the rung crosses the whole hook, the rung's END when the hook hangs
        # over it -- which the comb samples every 16 mm. (C MuJoCo puts its
        # box-cylinder contacts on the same boundary.) The 12 mm thickness is closed
        # by the rods themselves: no rung fits between the two faces.
        for tag in ('shank', 'lip'):
            c, _, h = boxes[f'hook_{sd}:{tag}']
            r, k = HOOK_EDGE, 0
            thin = int(np.argmin(h))
            along = 3 - thin - 1                   # the plate's long axis that is not y
            for st in (-1, 1):
                for y in np.linspace(-(h[1] - r), h[1] - r, HOOK_COMB):
                    off = np.zeros(3)
                    off[thin], off[1] = st * (h[thin] - r), y
                    out[f'hook_{sd}:{tag}_e{k}'] = (cap, c + off, _quat_z_to(np.eye(3)[along]),
                                                     [r, h[along] - r, 0.0])
                    k += 1
                for sa in (-1, 1):
                    off = np.zeros(3)
                    off[thin], off[along] = st * (h[thin] - r), sa * (h[along] - r)
                    out[f'hook_{sd}:{tag}_e{k}'] = (cap, c + off, _quat_z_to(np.eye(3)[1]),
                                                     [r, h[1] - r, 0.0])
                    k += 1
        # The finger motor and the plates on its end: a ring of MOTOR_RODS rods tangent
        # to the motor from inside, each running where the hand's surface is at the
        # motor's radius at its own angle -- the motor, the disc on its end, and on one
        # side the web's half cylinder beyond it -- and flush with where that ends. The
        # rung meets this motor on its free end's rim (168 of the full robot's 205
        # touches along win17's gait). One capsule round the motor and the web's box,
        # as it was, stood a radius (27.5 mm) proud of that rim and 10-22 mm past where
        # the web is round: it touched the rung in 0.4-4.5 % of lesson 1's states where
        # the full robot was 1-9 mm clear.
        p, a, r, h = _cylinder(m, f'wrist_dev_{sd}')
        b = m.body(f'wrist_dev_{sd}').id
        v = np.concatenate([_verts(m, g) for g in range(m.ngeom) if m.geom_bodyid[g] == b
                            and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
        u = (v - p) @ a
        w = (v - p) - np.outer(u, a)
        e1 = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        on = np.abs(np.linalg.norm(w, axis=1) - r) < 0.002          # at the motor's radius
        ang = np.arctan2(w @ e2, w @ e1)
        rr = HOOK_EDGE
        for i in range(MOTOR_RODS):
            t = 2 * np.pi * i / MOTOR_RODS
            mine = on & (np.abs(np.angle(np.exp(1j * (ang - t)))) <= np.pi / MOTOR_RODS)
            lo, hi = min(-h, u[mine].min(initial=-h)), max(h, u[mine].max(initial=h))
            out[f'wrist_dev_{sd}:rod{i}'] = (
                cap, p + (lo + hi) / 2 * a + (r - rr) * (np.cos(t) * e1 + np.sin(t) * e2),
                _quat_z_to(a), [rr, (hi - lo) / 2 - rr, 0.0])
        for body in ARM:
            for i, (c, a, r, h) in enumerate(_arm(m, f'{body}_{sd}')):
                out[f'{body}_{sd}:arm{i}'] = (cap, c, _quat_z_to(a), [r, h, 0.0])
    return out


def _arm(m, body, q=None):
    """capsules for an arm link, in its body's frame, as [(centre, axis, radius, half
    length)]: one fitted to its motor cylinder (points on its rims and waist), and one
    to each of its bracket meshes (its vertices) -- a link's plates face different ways
    (the upper arm's: a long side plate, and a square one across its elbow end whose
    corners reach 68 mm from the arm's axis), and one capsule round all of them was 4 mm
    short of the upper arm where a row meets it"""
    b = m.body(body).id
    motor, brackets = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or not m.geom_contype[g]:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            brackets.append(_verts(m, g))
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            R = R.reshape(3, 3)
            r, h = m.geom_size[g][:2]
            t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            ring = np.stack([r * np.cos(t), r * np.sin(t), np.zeros_like(t)], 1)
            for z in (-h, 0.0, h):
                motor.append((ring + [0, 0, z]) @ R.T + m.geom_pos[g])
    return [_capsule(p, ARM_Q if q is None else q) for p in [np.concatenate(motor)] + brackets]


def _capsule(pts, q):
    """(centre, axis, radius, half length): along the points' principal axis, its radius
    the q-th percentile of their distance from it, its caps inside their ends"""
    c = pts.mean(0)
    a = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    t = (pts - c) @ a
    r = float(np.percentile(np.linalg.norm((pts - c) - np.outer(t, a), axis=1), q))
    lo, hi = t.min() + r, t.max() - r
    if hi < lo:                                   # shorter than it is wide: a ball
        lo = hi = (t.min() + t.max()) / 2
    return c + (lo + hi) / 2 * a, a, r, max((hi - lo) / 2, 1e-4)   # a ball: MuJoCo wants a length


def _mix(a, b):
    """contact parameters for an explicit pair, by the rule MuJoCo applies to a
    dynamic one: the higher priority geom's own, else friction max and the rest
    averaged (every geom here has solmix 1). A <pair> takes nothing from its geoms,
    so without this every pair would silently get the defaults."""
    if a['priority'] != b['priority']:
        w = a if a['priority'] > b['priority'] else b
        fr, sr, si = w['friction'], w['solref'], w['solimp']
    else:
        fr = np.maximum(a['friction'], b['friction'])
        sr = (a['solref'] + b['solref']) / 2
        si = (a['solimp'] + b['solimp']) / 2
    return dict(friction=fr[[0, 0, 1, 2, 2]], solref=sr, solimp=si)


def lesson_layout(lesson):
    """the lesson as data, read off its C scene: rungs (centre, axis, extent, radius),
    walls (a point on the inner face, the normal toward the robot), goals"""
    spec = LESSONS[lesson]
    m = mujoco.MjModel.from_xml_path(xml_path(lesson))
    rungs, walls = [], []
    for g in range(m.ngeom):
        if m.geom(g).name.startswith('rung_'):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            rungs.append((m.geom(g).name, m.geom_pos[g].copy(), canon_axis(R.reshape(3, 3)[:, 2]),
                          float(m.geom_size[g][1]), float(m.geom_size[g][0])))
    for g in range(m.ngeom):
        if m.geom(g).name.startswith('wall_'):
            c, sz = m.geom_pos[g], m.geom_size[g]
            thin = int(np.argmin(sz))
            n = np.zeros(3)
            # the robot's side of the wall: the rungs stand proud of the face they
            # belong to
            near = min((r[1] for r in rungs), key=lambda p: np.linalg.norm(p - c))
            n[thin] = np.sign(near[thin] - c[thin]) or 1.0
            walls.append((m.geom(g).name, c + n * sz[thin], n))
    assert len(rungs) <= N_RUNG and walls[0][0] == 'wall_front' and len(walls) <= 2, lesson
    return dict(rungs=rungs, walls=walls, goals=np.asarray(spec['goals'], float),
                goal_rungs=[i for i, r in enumerate(rungs) if r[0] in spec['goal_rungs']],
                timeout=float(spec['timeout']))


def _quat_z_to(v):
    """unit quaternion turning local z onto v"""
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, np.asarray(v, float))
    return q


def slots(layout):
    """per-slot geometry for this lesson: rung capsules (pos, quat, size) and the
    extra wall plane (pos, quat); what `build` puts in the model and what the MJX
    env writes into each env's copy of the model"""
    pos = np.tile([0.0, 0.0, -PARK], (N_RUNG, 1)) - np.arange(N_RUNG)[:, None] * [0, 0, 1.0]
    quat = np.tile([1.0, 0, 0, 0], (N_RUNG, 1))
    size = np.tile([0.015, 0.1, 0.0], (N_RUNG, 1))
    for i, (_, c, a, h, r) in enumerate(layout['rungs']):
        pos[i], quat[i], size[i] = c, _quat_z_to(a), [r, h, 0.0]
    walls = layout['walls']
    wpos, wquat = np.array([0.0, 0.0, -PARK]), np.array([1.0, 0, 0, 0])   # faces up, far below
    if len(walls) > 1:
        wpos, wquat = walls[1][1], _quat_z_to(walls[1][2])
    return dict(rung_pos=pos, rung_quat=quat, rung_size=size,
                wall_pos=wpos, wall_quat=wquat,
                front_pos=walls[0][1], front_quat=_quat_z_to(walls[0][2]))


def _pad(g):
    g.priority, g.friction = PAD_PARAMS['priority'], PAD_PARAMS['friction']
    g.solref, g.solimp = PAD_PARAMS['solref'], PAD_PARAMS['solimp']


def build(layout=None, n_rung=N_RUNG, wall2=True):
    """the MjSpec: the robot as MJX can collide it, N_RUNG rung slots, two wall planes.

    layout=None parks every slot (the MJX model; each env places its own). A lesson's
    layout fills them and names them after that lesson's rungs and walls (the C
    model: `ClimbTask` finds rungs by name).

    `n_rung`, `wall2`: collide only the first n_rung rung slots and, with wall2, the
    extra wall -- every geom is still there. A slot no wall in a batch fills is parked
    PARK m away and never touched, but MJX tests every pair of it every substep
    whatever the distance: the pairs of rung slots 3..5 and of the extra wall are
    42 % of a substep's memory traffic, and families 1..3 use neither
    (`rl.mjx_task.Batch`'s `fit`)."""
    full = _full()
    boxes = _boxes(full)
    spec = mujoco.MjSpec.from_file(SCENE)
    spec.modelname = 'mjx_' + (lesson_tag(layout['lesson']) if layout else 'robot')
    # absolute, so the XML can be read from anywhere, from a string included
    spec.meshdir = os.path.join(os.path.dirname(SCENE), spec.meshdir)
    # What meets a WALL is the robot's own collision geometry, every body's, exactly as
    # scene.xml draws it: a plane against a convex mesh (its vertices) or a cylinder is
    # exact in MJX, and cheap. Named '<body>:col<i>'; a hook's two boxes, which carry
    # the grip pad, 'hook_<s>:shank' and ':lip'. The visuals go, and their meshes.
    wall, meshes, motor, guard = [], set(), {}, []
    for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
        n = nbox = 0
        for g in list(body.geoms):
            if not (g.contype or g.conaffinity):
                spec.delete(g)
                continue
            # the lidars and the roof under the head ones sit inside the cage, which
            # meets the walls and the rungs before they could
            if g.name.startswith('lidar_') or (g.type == mujoco.mjtGeom.mjGEOM_MESH
                                               and 'lidar' in g.meshname):
                spec.delete(g)
                continue
            if g.name.startswith('guard_'):
                guard.append(g.name)          # capsules already: exact against a rung
            elif body.name.startswith('hook_') and g.type == mujoco.mjtGeom.mjGEOM_BOX:
                g.name = f"{body.name}:{('shank', 'lip')[nbox]}"
                nbox += 1
                _pad(g)
            else:
                g.name = f'{body.name}:col{n}'
                n += 1
                if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    motor.setdefault(body.name, g.name)
            g.contype = g.conaffinity = 0
            wall.append(g.name)
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                meshes.add(g.meshname)
    for g in list(spec.worldbody.geoms):
        spec.delete(g)
    for me in list(spec.meshes):
        if me.name not in meshes:
            spec.delete(me)
    # what meets a RUNG or the other hand is added: round shapes and a few boxes. Only
    # what some pair uses -- C's discardvisual drops an unpaired geom and would
    # renumber the rest against MJX's copy
    shapes = {k: (mujoco.mjtGeom.mjGEOM_BOX,) + v for k, v in boxes.items()}
    shapes.update(_round(full, boxes))
    rung = RUNG + sorted(k for k in shapes if ':arm' in k) + guard
    used = set(rung) | {g for p in SELF for g in p}
    for key, (typ, pos, quat, size) in shapes.items():
        if key not in used or key in wall:
            continue
        body, tag = key.split(':')
        g = spec.body(body).add_geom(name=key, type=typ, pos=pos, quat=quat, size=size,
                                     contype=0, conaffinity=0,
                                     friction=[1.0, 0.02, 0.001], rgba=[0.5, 0.6, 0.9, 0.6])
        if tag.startswith(('shank', 'lip')):
            _pad(g)
    sl = slots(layout) if layout else slots(dict(rungs=[], walls=[
        ('wall_front', np.array([0.5, 0.3, 1.0]), np.array([-1.0, 0, 0]))]))
    rung_names = ([r[0] for r in layout['rungs']] if layout else [])
    rung_names += [f'park_rung{i}' for i in range(len(rung_names), N_RUNG)]
    for i, nm in enumerate(rung_names):
        spec.worldbody.add_geom(name=nm if layout else f'slot_rung{i}',
                                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                pos=sl['rung_pos'][i], quat=sl['rung_quat'][i],
                                size=sl['rung_size'][i], contype=0, conaffinity=0,
                                friction=[1.2, 0.02, 0.001], rgba=[0.85, 0.55, 0.15, 1])
    extra = layout['walls'][1][0] if layout and len(layout['walls']) > 1 else 'park_wall'
    for nm, p, q in (('wall_front', sl['front_pos'], sl['front_quat']),
                     (extra if layout else 'slot_wall', sl['wall_pos'], sl['wall_quat'])):
        spec.worldbody.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_PLANE, pos=p, quat=q,
                                size=[2.0, 2.0, 0.05], contype=0, conaffinity=0,
                                friction=[1.0, 0.005, 0.0001], rgba=[0.34, 0.36, 0.40, 1])
    world = [g.name for g in spec.worldbody.geoms]
    rungs, walls = world[:N_RUNG], world[N_RUNG:]
    geom = {g.name: g for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)
            for g in b.geoms}
    geom.update({g.name: g for g in spec.worldbody.geoms})
    par = {k: dict(priority=g.priority, friction=np.array(g.friction),
                   solref=np.array(g.solref), solimp=np.array(g.solimp))
           for k, g in geom.items()}

    def pair(a, b):
        spec.add_pair(geomname1=a, geomname2=b, condim=3, **_mix(par[a], par[b]))

    for w in rungs[:n_rung]:
        for r in rung:
            pair(r, w)
    for w in walls[:2 if wall2 else 1]:
        for r in wall:
            pair(r, w)
    for a, b in SELF + [(motor[f'{a}_L'], motor[f'{b}_R']) for a in DISTAL for b in DISTAL]:
        pair(a, b)
    for s in list(spec.sensors):
        if s.name.startswith(DROP_SENSORS):
            spec.delete(s)
    spec.option.iterations, spec.option.ls_iterations = ITERATIONS, LS_ITERATIONS
    spec.add_numeric(name='max_contact_points', data=[MAX_CONTACTS])
    return spec


def _stamp():
    """what the simplified scenes are generated FROM (see `lessons._stamp`)"""
    h = hashlib.sha256()
    for f in (SCENE, __file__, os.path.join(os.path.dirname(__file__), 'lessons.py')):
        h.update(open(f, 'rb').read())
    h.update(os.path.abspath(SCENE).encode())     # the meshes are read by absolute path
    return h.hexdigest()[:16]


def c_path(lesson):
    """the simplified robot on this lesson's rungs, as a scene file C MuJoCo (and
    `ClimbTask(xml=...)`) can load; written on demand when missing or stale"""
    p = os.path.join(OUT, f'mjx_{lesson_tag(lesson)}_{LESSONS[lesson]["name"]}.xml')
    want = f'<!-- stamp {_stamp()} -->'
    if not os.path.exists(p) or want not in open(p).read(3000):
        lay = lesson_layout(lesson)
        lay['lesson'] = lesson
        xml = build(lay).to_xml()
        xml = xml.replace('<mujoco model=', f'{want}\n<mujoco model=', 1)
        os.makedirs(OUT, exist_ok=True)
        tmp = f'{p}.{os.getpid()}.tmp'
        open(tmp, 'w').write(xml)
        os.replace(tmp, p)
    return p


def model(n_rung=N_RUNG, wall2=True):
    """the MJX model: every slot parked. Through the same XML text `c_path` writes --
    the writer keeps 6 significant digits, and the two must be the same robot to the
    last one for `rl.mjx_check task` to compare them exactly."""
    return mujoco.MjModel.from_xml_string(build(n_rung=n_rung, wall2=wall2).to_xml())


def main():
    from urdf.climbing_env import ClimbingEnv
    full = _full()
    m = model()
    kinds = [mujoco.mjtGeom(m.geom_type[g]).name[7:].lower() for g in range(m.ngeom) if m.geom_bodyid[g]]
    print(f"simplified robot: " + ", ".join(f"{kinds.count(k)} {k}" for k in sorted(set(kinds)))
          + f" (scene.xml: {sum(1 for g in range(full.ngeom) if full.geom_bodyid[g] and full.geom_contype[g])} "
          f"collision geoms, {full.nmesh} of them meshes), {m.npair} contact pairs, "
          f"{m.nsensordata} sensor values (scene.xml {full.nsensordata})")
    assert np.allclose(m.body_mass, full.body_mass) and np.allclose(m.body_inertia, full.body_inertia)
    assert np.allclose(m.key_qpos, full.key_qpos) and m.nu == full.nu == 17
    for k in sorted(LESSONS):
        e = ClimbingEnv(xml=c_path(k), success_z=1e9)
        e.reset()
        p0 = e.data.xpos[e.base].copy()
        for _ in range(int(1.0 / e.model.opt.timestep)):
            e.step(e.data.ctrl.copy())
        drift = np.linalg.norm(e.data.xpos[e.base] - p0) * 1000.0
        print(f"lesson {k:>5} {LESSONS[k]['name']:>24}: keyframe drift {drift:.2f} mm, "
              f"{e.data.ncon} contacts")
        assert drift < 5.0, f"lesson {k}: the simplified robot drifted {drift:.1f} mm off its keyframe"


if __name__ == '__main__':
    main()
