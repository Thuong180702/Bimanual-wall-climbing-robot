"""The robot as MJX can simulate it, and the lessons as data instead of scene files.

    python -m rl.mjx_scene      # build it, write urdf/lessons/mjx_*.xml, check every lesson

The CPU path simulates `urdf/scene.xml` as it is: every CAD part's convex hull
collides with everything, and each lesson is its own scene file. MJX can run
neither. Its JAX backend has no box-cylinder collision (hook against rung), runs
mesh hulls slowly, has no `jointlimitfrc` sensor, and compiles ONE model for every
env of a batch -- so a batch cannot mix lesson scenes.

So, measured before it was drawn -- win17's best policy on lessons -2..2, 6800
control steps, which body pairs touched and how often (`python -m rl.mjx_check
contacts` repeats it):

    hook - rung            48-70 %   ~220 N      the hold itself
    wrist_dev_R - rung        36 %     58 N      the finger motor resting on it
    base_link - wall          30 %     32 N      the body leans on the wall
    wrist_L - wall            16 %     44 N
    hook_L - hook_R          8.7 %     31 N
    arm - wall, arm L - arm R   < 2.3 % each
    arm - torso, arm - base       never

and every contact on an arm was on an actuator housing, not a bracket plate. Hence:

  * what can touch a WALL, or the other hand, is boxes: each actuator's cylinder
    becomes the box round it (exact on its end faces and its four sides, 0.41 r too
    big toward the edges), base and torso the box round their plates, the plates
    beside the hand (wrist, finger motor) their own box, and the hooks keep their
    two boxes, pad and all. The hook's knuckle, the waist and the other plates never
    touched anything.
  * what can touch a RUNG is capsules and balls, never boxes, because MJX's
    capsule-box test is wrong for a capsule as long as a rung: it separates only on
    the box's face normals, so a 2.4 m rung passing a 12 mm plate at a slight angle
    "penetrates" it. Measured on win17's lesson-1 gait: 50-75 mm contacts between a
    rung and a hook shank where C MuJoCo saw no contact at all, 5000-12000 N where C
    had 200, and the policy's lesson 1 went from 7/8 in C to 0/8 in MJX. Capsule
    against capsule is exact (the closest points of two segments). So each hook
    plate is two rods square to the rung at its edges -- the two ends of the line
    contact --, the finger motor and the plate beside it one capsule along the
    motor's axis (the rung is parallel to it when the hook is seated), the wrist and
    forearm motors a ball each, and base and torso a rod up each vertical edge.
  * rungs are capsules. MJX has no cylinder-box either. The cap's tip sits on the
    cylinder's end, so a rung is never longer than the real one, and
    `lessons.rung_segments` reads a capsule's extent as its half-length plus radius.
  * walls are planes: the robot only ever touches a wall's inner face, and a plane
    is the cheapest collision there is. ponytail: a plane does not end -- the front
    wall does, at y = -1.05 and 1.65; nothing reaches past it in any lesson today.
  * collisions are an explicit list of PAIRS (the table above, plus the same parts on
    the other hand), not everything-with-everything. Each pair gets the friction and
    softness MuJoCo would have mixed for those two geoms, so the pad still defines
    the hook's contact.
  * the lessons become DATA: N_RUNG rung slots and one extra wall slot whose
    geometry is set per env (`layout`). A slot a lesson does not use is parked far
    away. That is what lets one compiled model train every lesson at once.

Masses, inertias, joints, actuators and the keyframe are scene.xml's, untouched:
the geoms carried no mass (every body has an explicit <inertial>).

The same builder writes each lesson as a plain MuJoCo scene (`c_path`), so the
simplified robot can be run and cross-checked in C MuJoCo, where the full robot
also lives. `rl.mjx_check` does that.
"""
import hashlib, os
import numpy as np, mujoco
from controller import SCENE
from .lessons import LESSONS, OUT, PAD, lesson_tag, xml_path

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
# never had more than 9 at once: 600 steps of random actions on every lesson, and
# win17's policy on its eleven (`rl.mjx_check physics` asserts it). Each one kept is
# four rows of the solver's system: 32 cost 22 % more per step than 16.
MAX_CONTACTS = 16

SIDES = 'LR'
# what each body keeps as boxes (walls, the other hand). 'hull': round all its
# collision meshes; 'motor': round its actuator's cylinder; 'web': round its plates
# with 'web' in the name; 'shank' / 'lip': the hook's own two boxes.
KEEP = {'base_link': ('hull',), 'torso_upper': ('hull',),
        'shoulder_%s': ('motor',), 'upper_arm_%s': ('motor',),
        'humerus_rot_%s': ('motor',), 'fore_arm_%s': ('motor',),
        'forearm_rot_%s': ('motor',), 'wrist_%s': ('motor', 'web'),
        'wrist_dev_%s': ('motor', 'web'), 'hook_%s': ('shank', 'lip')}
# what can meet a wall: the body and every housing out to the hook
WALL = ['base_link:hull', 'torso_upper:hull'] + [
    n % sd for sd in SIDES for n in (
        'shoulder_%s:motor', 'upper_arm_%s:motor', 'humerus_rot_%s:motor',
        'fore_arm_%s:motor', 'forearm_rot_%s:motor', 'wrist_%s:motor',
        'wrist_dev_%s:motor', 'hook_%s:shank', 'hook_%s:lip')]
# what can meet a rung, all of it round (`_round`): the body climbing past a row,
# and the hand
BODY_ROD = 0.02          # m, the radius of the rods up the body's edges
RUNG = [f'{b}:rod{i}' for b in ('base_link', 'torso_upper') for i in range(4)] + [
    n % sd for sd in SIDES for n in (
        'hook_%s:shank0', 'hook_%s:shank1', 'hook_%s:lip0', 'hook_%s:lip1',
        'wrist_dev_%s:finger', 'wrist_%s:ball', 'forearm_rot_%s:ball')]
# one hand against the other: every pair measured touching, and the hooks
SELF = [('hook_L:shank', 'hook_R:shank'), ('hook_L:shank', 'hook_R:lip'),
        ('hook_L:lip', 'hook_R:shank'), ('hook_L:lip', 'hook_R:lip'),
        ('forearm_rot_L:motor', 'forearm_rot_R:motor'),
        ('wrist_L:motor', 'wrist_R:motor'), ('wrist_L:motor', 'wrist_R:web'),
        ('wrist_L:web', 'wrist_R:motor'), ('wrist_L:motor', 'wrist_dev_R:motor'),
        ('wrist_dev_L:motor', 'wrist_R:motor'),
        ('wrist_dev_L:motor', 'wrist_dev_R:motor')]
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
                        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                            continue
                        mid = m.geom_dataid[g]
                        if tag == 'web' and 'web' not in m.mesh(mid).name:
                            continue
                        R = np.zeros(9)
                        mujoco.mju_quat2Mat(R, m.geom_quat[g])
                        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                        vs.append(v @ R.reshape(3, 3).T + m.geom_pos[g])
                    v = np.concatenate(vs)
                    lo, hi = v.min(0), v.max(0)
                    out[f'{name}:{tag}'] = ((lo + hi) / 2, np.array([1.0, 0, 0, 0]), (hi - lo) / 2)
    return out


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
        # each hook plate: two rods at its edges, along its long side square to the
        # rung (the rung lies along the hook's y), as thick as the plate
        for tag, axis in (('shank', 2), ('lip', 0)):
            c, _, h = boxes[f'hook_{sd}:{tag}']
            r = h.min()
            for i, sy in enumerate((-1, 1)):
                out[f'hook_{sd}:{tag}{i}'] = (cap, c + [0.0, sy * (h[1] - r), 0.0],
                                              _quat_z_to(np.eye(3)[axis]),
                                              [r, h[axis] - r, 0.0])
        # the finger motor and the plate beside it, along the motor's axis
        p, a, r, h = _cylinder(m, f'wrist_dev_{sd}')
        wc, _, wh = boxes[f'wrist_dev_{sd}:web']
        reach = wh @ np.abs(a)
        lo = min(p @ a - h, wc @ a - reach)
        hi = max(p @ a + h, wc @ a + reach)
        out[f'wrist_dev_{sd}:finger'] = (cap, p + ((lo + hi) / 2 - p @ a) * a,
                                         _quat_z_to(a), [r, (hi - lo) / 2, 0.0])
        for body in (f'wrist_{sd}', f'forearm_rot_{sd}'):   # never seen on a rung
            p, a, r, h = _cylinder(m, body)
            out[f'{body}:ball'] = (ball, p, np.array([1.0, 0, 0, 0]), [r, 0.0, 0.0])
    return out


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
            rungs.append((m.geom(g).name, m.geom_pos[g].copy(), R.reshape(3, 3)[:, 2],
                          float(m.geom_size[g][1]), float(m.geom_size[g][0])))
    for g in range(m.ngeom):
        if m.geom(g).name.startswith('wall_'):
            c, sz = m.geom_pos[g], m.geom_size[g]
            thin = int(np.argmin(sz))
            n = np.zeros(3)
            # the robot's side of the wall: the rungs stand proud of the face they
            # belong to (controller.kinematics.wall_planes reads it the same way)
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
        pos[i], quat[i], size[i] = c, _quat_z_to(a), [r, h - r, 0.0]
    walls = layout['walls']
    wpos, wquat = np.array([0.0, 0.0, -PARK]), np.array([1.0, 0, 0, 0])   # faces up, far below
    if len(walls) > 1:
        wpos, wquat = walls[1][1], _quat_z_to(walls[1][2])
    return dict(rung_pos=pos, rung_quat=quat, rung_size=size,
                wall_pos=wpos, wall_quat=wquat,
                front_pos=walls[0][1], front_quat=_quat_z_to(walls[0][2]))


def build(layout=None):
    """the MjSpec: the simplified robot, N_RUNG rung slots and two wall planes.

    layout=None parks every slot (the MJX model; each env places its own). A lesson's
    layout fills them and names them after that lesson's rungs and walls (the C
    model: `ClimbTask` finds rungs by name)."""
    full = _full()
    boxes = _boxes(full)
    spec = mujoco.MjSpec.from_file(SCENE)
    spec.modelname = 'mjx_' + (lesson_tag(layout['lesson']) if layout else 'robot')
    # every geom goes, visuals too, and with them every mesh: the file this writes is
    # physics only and stands on its own
    for g in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        spec.delete(g)
    for me in list(spec.meshes):
        spec.delete(me)
    shapes = {k: (mujoco.mjtGeom.mjGEOM_BOX,) + v for k, v in boxes.items()}
    shapes.update(_round(full, boxes))
    # only what some pair uses: C's discardvisual drops an unpaired geom and would
    # renumber the rest against MJX's copy
    used = set(WALL) | set(RUNG) | {g for p in SELF for g in p}
    for key, (typ, pos, quat, size) in shapes.items():
        if key not in used:
            continue
        body, tag = key.split(':')
        g = spec.body(body).add_geom(name=key, type=typ, pos=pos, quat=quat, size=size,
                                     contype=0, conaffinity=0,
                                     friction=[1.0, 0.02, 0.001], rgba=[0.5, 0.6, 0.9, 0.6])
        if tag.startswith(('shank', 'lip')):
            g.priority, g.friction = PAD_PARAMS['priority'], PAD_PARAMS['friction']
            g.solref, g.solimp = PAD_PARAMS['solref'], PAD_PARAMS['solimp']
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

    for w in rungs:
        for r in RUNG:
            pair(r, w)
    for w in walls:
        for r in WALL:
            pair(r, w)
    for a, b in SELF:
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


def model():
    """the MJX model: every slot parked. Through the same XML text `c_path` writes --
    the writer keeps 6 significant digits, and the two must be the same robot to the
    last one for `rl.mjx_check task` to compare them exactly."""
    return mujoco.MjModel.from_xml_string(build().to_xml())


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
