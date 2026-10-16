"""The benchmark: nine lesson scenes, one per skill, in the order they are learnt.

    python -m rl.lessons      # writes urdf/lessons/*.xml and smoke-checks each one

    1 traverse         along one continuous rung
    2 traverse_gaps    along a broken rung, across its gaps
    3 climb_up         up rows standing out from the wall
    4 climb_down       down rows standing in toward it (3 backwards)
    5 staircase_down   down short rungs, each lower and further along
    6 step_up          diagonally up onto a tier of short rungs
    7 wall_corner      onto the side wall's rung, in the corner
    8 wall_behind      onto the rung on the wall behind the back
    9 swing_gap        across a gap past the static reach, by swinging

Each scene is generated from ../urdf/scene.xml: the robot, actuators, sensors and the
verified hanging keyframe are kept verbatim; the walls and rungs are replaced per
lesson. Every lesson starts from that keyframe on a start rung at x = 0.44, z = 1.00,
spanning at least y = +-0.16 -- leaning on the front wall, or pitched to hang free
where there is no wall to lean on (lesson 4).

Training does not draw these scenes: it draws generated walls (rl.walls), one family
per lesson, each lesson inside its family's last range.
"""
import hashlib
import os
import re
import numpy as np
from . import SCENE, URDF

OUT = os.path.join(URDF, 'lessons')

# scene.xml's walls, 8 m long and 6 m deep: every layout here fits, mirrored too --
# lesson 5's staircase ends 2.45 m along and 1.15 m down
WALL_FRONT = '    <geom name="wall_front" type="box" material="wall" pos="0.550 0.00 0.50" size="0.05 4.00 3.00"/>'
# ...and 0.40 m further back, behind lesson 4's lowest row (its face at 0.90)
WALL_FRONT_FAR = '    <geom name="wall_front" type="box" material="wall" pos="0.950 0.00 0.50" size="0.05 4.00 3.00"/>'
WALL_BACK = '    <geom name="wall_back"  type="box" material="wall" pos="-0.300 0.00 0.50" size="0.05 4.00 3.00"/>'
WALL_SIDE = '    <geom name="wall_side"  type="box" material="wall" pos="0.125 -1.050 0.50" size="0.425 0.05 3.00"/>'

# The keyframe leans on the front wall: its centre of mass is 65 mm behind the start
# rung. With no wall there it swings +-0.13 m. Pitched about the rung by -6.4 deg the
# keyframe's centre of mass is under it, and the arms then sag; -7.5 deg is the stillest
# free hang measured (26 mm of sway over 3 s, against 44 mm at -6.4 and 260 unpitched).
FREE_PITCH = np.radians(-7.5)
START_RUNG = np.array([0.44, 0.0, 1.00])     # the start rung's axis passes through here, along y


def hang_qpos(pitch=0.0):
    """scene.xml's keyframe qpos, the whole robot turned `pitch` rad about the start rung's
    axis (+y): the hooks stay on the rung, the body swings round it"""
    q = np.array(re.search(r'<key name="hang_on_rung"\s+qpos="([^"]+)"', open(SCENE).read())
                 .group(1).split(), float)
    if pitch:
        c, s = np.cos(pitch), np.sin(pitch)
        p = q[:3] - START_RUNG
        q[:3] = START_RUNG + [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]]
        w, x, y, z = q[3:7]
        a, b = np.cos(pitch / 2), np.sin(pitch / 2)          # (a, 0, b, 0) * (w, x, y, z)
        q[3:7] = [a * w - b * y, a * x + b * z, a * y + b * w, a * z - b * x]
    return q


# --- the grip pad: what makes the hook hold along its rung --------------------
# The hook is a slot round a smooth cylinder, so along the rung axis nothing holds
# it: 40 N sideways moved it ~100 mm (measured, `slide_test` below), and PPO
# solved lesson 1 by exploiting exactly that instead of climbing. The fix belongs on
# the HAND, not the wall -- a climber chalks up and wears rubber, they do not groove
# the rock -- and a real climber can still slip, so the aim is to make the slide
# small and honest rather than to abolish it. Collars on the rung were tried and
# rejected for exactly that reason.
#
# Raising the hook's friction alone does NOT work, and the reason is worth keeping:
# the contacts are far too soft, so more friction just funnels more force into the
# contact and the hook sinks INTO the rung. Measured penetration against friction --
# mu 1.0: 3.3 mm, 1.4: 7.1, 1.8: 18.3, 2.0: 21.7 and the hook drops straight
# through and the robot falls. Rubber is only available once the contact is stiff
# enough to carry it. With solref 0.005 and solimp d_max 0.99 the same mu 3.0 sits
# at 1.4 mm of penetration and holds.
#
# `priority="1"` makes the hand's pad define the contact outright rather than being
# combined with the rung's parameters -- that is the physical statement too: the
# soft high-friction part is the pad, and it is what deforms.
#
# Measured on a smooth rung, hook travel under a 40 N sideways pull:
#   bare hook, soft contact (today)      +77 mm
#   mu 1.0, stiff contact                +35 mm     <- most of the win is stiffness
#   mu 2.0, stiff contact + hard solimp  +20 mm
#   mu 3.0, stiff contact + hard solimp  +15 mm
#   mu 5.0                               falls -- 10 mm of penetration again
# mu 3.0 is at the optimistic end for rubber on metal; 2.0 is the safer claim and
# costs 5 mm. `scene.xml` keeps the bare hook so the classical controller's measured
# numbers stay valid; only these RL scenes get the pad.
PAD_FRICTION = 2.0
PAD = [
    # both hook boxes, on both hands -- these four lines exist nowhere else
    ('<geom class="col" type="box" size="0.00600 0.03250 0.05500" '
     'pos="0.00000 0.00000 -0.05500"/>',
     '<geom class="col" type="box" size="0.00600 0.03250 0.05500" '
     'pos="0.00000 0.00000 -0.05500" priority="1" friction="%.1f 0.02 0.001" '
     'solref="0.005 1" solimp="0.9 0.99 0.001 0.5 2"/>' % PAD_FRICTION),
    ('<geom class="col" type="box" size="0.02000 0.03250 0.00600" '
     'pos="0.02600 0.00000 -0.10400"/>',
     '<geom class="col" type="box" size="0.02000 0.03250 0.00600" '
     'pos="0.02600 0.00000 -0.10400" priority="1" friction="%.1f 0.02 0.001" '
     'solref="0.005 1" solimp="0.9 0.99 0.001 0.5 2"/>' % PAD_FRICTION),
]


def _ry(name, y0, y1, z, x=0.44):
    """a front/back-wall rung, axis along y"""
    return (f'    <geom name="rung_{name}" class="hold" size="0.0150" '
            f'fromto="{x:.4f} {y0:.4f} {z:.4f} {x:.4f} {y1:.4f} {z:.4f}"/>')


def _rx(name, x0, x1, z, y=-0.94):
    """a side-wall rung, axis along x"""
    return (f'    <geom name="rung_{name}" class="hold" size="0.0150" '
            f'fromto="{x0:.4f} {y:.4f} {z:.4f} {x1:.4f} {y:.4f} {z:.4f}"/>')


# Goals are one world point per hand, [L, R], always ON a goal rung's axis. Success =
# both hooks latched on a goal rung within tolerance of their point (climb_task.py).
# Reach, measured on the robot hanging still from one hook (urdf/_reach.py's
# constraints): 0.56 m up, 0.64-0.68 m along a rung, 0.65-0.88 m down at 0.15-0.45 m
# aside; along a rung with momentum at most 1.50-1.55 m. Rows stay <= 0.50 m apart
# and steps within the static reach, except lesson 9, which is past it on purpose.
# Each lesson goes toward the robot's left (+y); `mirrored` gives it toward its right.
#
# A row straight below or above the hang is inside the robot: its arms and body fill
# the rung plane 0.10-1.10 m under the hooks (59-107 mm off it). So rows above stand
# OUT from the wall (lesson 3), rows below stand IN toward it (lesson 4: lesson 3 run
# backwards), and the diagonal moves step along as they go (lessons 5 and 6).
LESSONS = {
    1: dict(name='traverse', timeout=60.0,
            title='traverse sideways along one continuous rung',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 1.50, 1.00)],
            goal_rungs=['rung_start'],
            goals=[[0.44, 1.257, 1.00], [0.44, 0.943, 1.00]]),
    2: dict(name='traverse_gaps', timeout=60.0,
            title='traverse a broken rung: segments with 0.25 m gaps',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.45, 0.35, 1.00),
                   _ry('seg1', 0.60, 0.95, 1.00),
                   _ry('seg2', 1.20, 1.65, 1.00)],
            goal_rungs=['rung_seg2'],
            goals=[[0.44, 1.55, 1.00], [0.44, 1.24, 1.00]]),
    # Two rows 0.40 m apart straight above, each standing OUT 0.20 m further from the
    # wall than the one below (rl.walls.OUT): hanging from the top one, the rows below are
    # 92-164 mm clear of the robot. 0.40, not 0.45: reaching out costs the free hook
    # height (0.52 m up 0.15-0.25 m out).
    3: dict(name='climb_up', timeout=90.0,
            title='climb up two rows, each standing 0.20 m further out from the wall',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 1.50, 1.00),
                   _ry('up1', -0.90, 1.50, 1.40, x=0.24),
                   _ry('up2', -0.90, 1.50, 1.80, x=0.04)],
            goal_rungs=['rung_up2'],
            goals=[[0.04, 0.157, 1.80], [0.04, -0.157, 1.80]]),
    # Lesson 3 backwards: from the top row down two, each 0.20 m nearer the wall. The
    # start row stands 0.46 m off the wall, so the robot hangs free instead of leaning on
    # it: the keyframe is pitched about the start rung to its stillest free hang (`pitch`).
    4: dict(name='climb_down', timeout=90.0,
            title='climb down two rows, each standing 0.20 m nearer the wall',
            walls=[WALL_FRONT_FAR], pitch=FREE_PITCH,
            rungs=[_ry('start', -0.90, 1.50, 1.00),
                   _ry('down1', -0.90, 1.50, 0.60, x=0.64),
                   _ry('down2', -0.90, 1.50, 0.20, x=0.84)],
            goal_rungs=['rung_down2'],
            goals=[[0.84, 0.157, 0.20], [0.84, -0.157, 0.20]]),
    # Down a staircase of short rungs, each lower and further along: the way a climber
    # steps down beside the holds.
    5: dict(name='staircase_down', timeout=120.0,
            title='down a staircase of short rungs, each lower and further along',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 0.40, 1.00),
                   _ry('step1', 0.60, 1.10, 0.60),
                   _ry('step2', 1.25, 1.75, 0.25),
                   _ry('step3', 1.95, 2.45, -0.15)],
            goal_rungs=['rung_step3'],
            goals=[[0.44, 2.33, -0.15], [0.44, 2.02, -0.15]]),
    # One tier of two short rungs, each at its own height, the tier starting 0.15 m past
    # the end of the start row: a step up taken diagonally, in the wall's own plane.
    6: dict(name='step_up', timeout=90.0,
            title='step up diagonally onto a tier of short rungs, each at its own height',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 0.40, 1.00),
                   _ry('t1a', 0.55, 0.90, 1.42),
                   _ry('t1b', 1.15, 1.50, 1.50)],
            goal_rungs=['rung_t1a', 'rung_t1b'],
            goals=[[0.44, 1.325, 1.50], [0.44, 0.725, 1.42]]),
    7: dict(name='wall_corner', timeout=120.0,
            title='transfer to the perpendicular wall in the corner',
            walls=[WALL_FRONT, WALL_SIDE],
            rungs=[_ry('start', -0.90, 0.90, 1.00),
                   _rx('side', -0.18, 0.43, 1.00)],
            goal_rungs=['rung_side'],
            goals=[[0.35, -0.94, 1.00], [0.04, -0.94, 1.00]]),
    # The hands keep their order across (L at +y). Facing the back wall that is the
    # crossed arrangement -- turning round would put L at -y -- and K_SEP, on the start
    # rung's axis and sign, agrees with it. A posture the robot has: with
    # urdf/_reach.py's constraints plus this rung and the wall behind it, a
    # least-squares search from 100 random starts found legal two-hook hangs 0.314 m
    # apart crossed 8 times and the normal way round 7.
    8: dict(name='wall_behind', timeout=120.0,
            title='transfer to the parallel wall behind the back',
            walls=[WALL_FRONT, WALL_BACK],
            rungs=[_ry('start', -0.90, 1.50, 1.00),
                   _ry('back', -0.90, 1.50, 1.00, x=-0.19)],
            goal_rungs=['rung_back'],
            goals=[[-0.19, 0.157, 1.00], [-0.19, -0.157, 1.00]]),
    # One gap past the static reach (0.58 m of gap is the most a still hang crosses):
    # one hook holds while the robot swings the other across. The one motion found
    # across it catches the far rung 1.10 m from the other hook, all the reach there
    # is, and has the swing carry the loaded hook 0.18 m on to make room for the other.
    9: dict(name='swing_gap', timeout=60.0,
            title='a 0.90 m gap, past the static reach: swing across',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.45, 0.35, 1.00),
                   _ry('far', 1.25, 1.75, 1.00)],
            goal_rungs=['rung_far'],
            goals=[[0.44, 1.63, 1.00], [0.44, 1.32, 1.00]]),
}


def mirrored(spec):
    """the lesson reflected y -> -y: every wall and rung, and each hand given the
    other's goal so the left hand stays on the left (rl.walls.mirror's rule)"""
    def flip(m):
        v = [float(x) for x in m.group(2).split()]
        for i in range(1, len(v), 3):
            v[i] = -v[i]
        return f'{m.group(1)}="{" ".join(f"{x:.4f}" for x in v)}"'
    xml = lambda lines: [re.sub(r'(pos|fromto)="([^"]+)"', flip, g) for g in lines]
    goals = [[x, -y, z] for x, y, z in spec['goals']][::-1]
    return dict(spec, walls=xml(spec['walls']), rungs=xml(spec['rungs']), goals=goals,
                title=spec['title'] + ' (mirrored)')


def spec_of(lesson, mirror=False):
    return mirrored(LESSONS[lesson]) if mirror else LESSONS[lesson]


def lesson_key(x):
    """'2' -> 2"""
    return int(str(x).strip())


def lesson_tag(lesson):
    """'lesson01' -- the file and model name stem"""
    return f"lesson{lesson:02d}"


def _path(lesson, mirror=False):
    return os.path.join(OUT, f"{lesson_tag(lesson)}_{LESSONS[lesson]['name']}"
                             f"{'_mirror' if mirror else ''}.xml")


def _stamp(lesson, mirror=False):
    """what this lesson's scene is generated FROM, as one short hash"""
    h = hashlib.sha256()
    h.update(open(SCENE, 'rb').read())
    h.update(repr(sorted(spec_of(lesson, mirror).items(), key=lambda kv: kv[0])).encode())
    h.update(repr(PAD).encode())
    return h.hexdigest()[:16]


def xml_path(lesson, mirror=False):
    """the lesson's scene file, written on demand -- if it is missing OR stale.

    `urdf/lessons/` is generated from `scene.xml` and gitignored, so a fresh clone
    -- or a pull that adds a lesson -- has no scene for it, and MuJoCo reports that
    as a bare ParseXML error once per worker process.

    Staleness is why the check is a hash and not just `os.path.exists`. `win13`
    trained 7.6M steps on the robot as it was BEFORE the hook was re-mounted: the
    scenes were already on disk from an earlier run, a pull brought the new
    `scene.xml`, and nothing regenerated them. It was only caught by replaying the
    checkpoint -- 12/12 on the old model, 0/12 on the new one -- which is an
    expensive way to learn that a model edit did not take. The stamp covers
    `scene.xml`, this lesson's spec and the patch lists, so any of them moving
    rewrites the file.
    """
    p = _path(lesson, mirror)
    want = _stamp(lesson, mirror)
    if not os.path.exists(p) or f'<!-- stamp {want} -->' not in open(p).read(2000):
        write(lesson, mirror)
    return p


def scene_xml(spec, model, stamp=''):
    """scene.xml with its walls and rungs replaced by `spec`'s, the grip pad on, for a
    file in OUT"""
    xml = open(SCENE).read()
    xml = xml.replace('<mujoco model="bimanual_wall_climber_scene">',
                      f'<mujoco model="{model}">\n  <!-- stamp {stamp} -->')
    xml = xml.replace('meshdir="meshes/"', 'meshdir="../meshes/"')
    for old, new in PAD:
        assert old in xml, f"scene.xml changed; patch target not found: {old}"
        xml = xml.replace(old, new)
    head, rest = xml.split('    <!-- ===== walls ===== -->')
    _, tail = rest.split('    <!-- ===== robot ===== -->')
    mid = '\n'.join([f'    <!-- {spec["title"]} -->'] + spec['walls'] + spec['rungs'])
    xml = head + mid + '\n\n    <!-- ===== robot ===== -->' + tail
    if spec.get('pitch'):
        q = ' '.join(f'{v:.6f}' for v in hang_qpos(spec['pitch']))
        xml = re.sub(r'(<key name="hang_on_rung"\s+qpos=")[^"]+', lambda m: m.group(1) + q, xml)
    return xml


def write(lesson, mirror=False):
    spec = spec_of(lesson, mirror)
    out = scene_xml(spec, f'{lesson_tag(lesson)}_{spec["name"]}', _stamp(lesson, mirror))
    # via a temp file and an atomic rename: the 8 SubprocVecEnv workers build their
    # environments at the same moment, and without it one of them parses the file
    # while another is still writing it
    os.makedirs(OUT, exist_ok=True)
    p = _path(lesson, mirror)
    tmp = f'{p}.{os.getpid()}.tmp'
    open(tmp, 'w').write(out)
    os.replace(tmp, p)
    return p


def canon_axis(a):
    """A rung is a line, not an arrow: its axis turned so its largest component is
    positive -- +y along the front and back walls, +x along the side one.

    MuJoCo points a `fromto` geom's z from the second point to the first, so every rung
    here, written lowest coordinate first, read -y (-x) off its scene, while the
    generated walls training draws (rl.walls) are +y. The policy sees the nearest
    rungs' axes and the distances to each one's two ends, so the benchmark's rungs
    reached it reversed: gpu1's pilot let go of its own training walls on the full
    robot (3/24 on family 1, 21/24 in MJX) and was 21/24 there with the axes the same."""
    a = np.asarray(a, float)
    return -a if a[np.argmax(np.abs(a))] < 0 else a


def rung_segments(model):
    """rung name -> (centre, unit axis (canon_axis), half length), straight from the
    model. A capsule rung (`rl.mjx_scene`) is the real cylinder with a cap past each
    end, so its half length is the cylinder's too."""
    import mujoco
    out = {}
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if nm.startswith('rung_'):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            out[nm] = (model.geom_pos[g].copy(), canon_axis(R.reshape(3, 3)[:, 2]),
                       float(model.geom_size[g][1]))
    return out


GRIP_HARD = np.radians(-38)   # a load-bearing hand gripping down to the edge of the
                              # measured band while the other one is off the wall


def slide_test(path, force=40.0):
    """phase 7's measurement: hook_R's travel along its rung under a sideways pull.

    Both fingers shut, then R grips hard (GRIP_HARD) while L lets go.
    """
    import mujoco
    from climbing_env import ClimbingEnv, GRIP_SET
    e = ClimbingEnv(xml=path, free_fall_s=1e9, timeout_s=1e9, success_z=1e9)
    e.reset()
    ctrl = e.data.ctrl.copy()
    aL = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'act_finger_L')
    aR = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'act_finger_R')
    for i in range(200):
        ctrl[aL] = ctrl[aR] = GRIP_SET * (i + 1) / 200
        e.step(ctrl)
    for _ in range(300):
        e.step(ctrl)
    for i in range(200):
        ctrl[aR] = GRIP_SET + (GRIP_HARD - GRIP_SET) * (i + 1) / 200
        ctrl[aL] = GRIP_SET + (0.30 - GRIP_SET) * (i + 1) / 200
        e.step(ctrl)
    for _ in range(400):
        e.step(ctrl)
    y0 = e.data.xpos[e.hook['R']][1]
    e.data.xfrc_applied[e.base] = [0, force, 0, 0, 0, 0]
    for _ in range(1200):
        e.step(ctrl)
    return (e.data.xpos[e.hook['R']][1] - y0) * 1000


def main():
    from climbing_env import ClimbingEnv
    for k, mirror in [(k, m) for m in (False, True) for k in sorted(LESSONS)]:
        spec = spec_of(k, mirror)
        path = write(k, mirror)
        e = ClimbingEnv(xml=path, success_z=1e9)
        e.reset()
        p0 = e.data.xpos[e.base].copy()
        ctrl = e.data.ctrl.copy()
        for _ in range(int(1.0 / e.model.opt.timestep)):
            e.step(ctrl)
        drift = np.linalg.norm(e.data.xpos[e.base] - p0) * 1000.0
        # 0.7 mm on the 21.5 kg robot leaning on its wall. It read 3.8 while the hooks
        # hung back to front and the arms were jammed into the torso -- a drift budget
        # is a symptom detector, so it stays where it was rather than following one.
        # Hanging free (FREE_PITCH) it sways: 36 mm in this first second with the fingers
        # still open, 14 mm once ClimbTask has shut them
        budget = 60.0 if spec.get('pitch') else 5.0
        assert drift < budget, f"lesson {k}: keyframe drifted {drift:.1f} mm"
        assert e.latched('L')[0] and e.latched('R')[0], f"lesson {k}: a hook came off"
        segs = rung_segments(e.model)
        for nm in spec['goal_rungs']:
            assert nm in segs, f"lesson {k}: goal rung {nm} missing"
        for gp in spec['goals']:
            gp = np.asarray(gp)
            d = min(np.linalg.norm(gp - (c + np.clip((gp - c) @ a, -h, h) * a))
                    for c, a, h in segs.values())
            assert d < 0.02, f"lesson {k}: goal {gp} is {d*1000:.0f} mm off every rung"
        pad = [g for g in range(e.model.ngeom)
               if e.model.geom_bodyid[g] in (e.hook['L'], e.hook['R'])
               and e.model.geom_priority[g] == 1]
        assert len(pad) == 4, f"lesson {k}: grip pad missing from the hooks"
        print(f"lesson {k}{' mirrored' if mirror else ''} {spec['name']:>20}: drift "
              f"{drift:.2f} mm, {len(spec['rungs'])} rungs, OK")

    # phase 7, on a lesson scene: pull the hanging robot sideways and see how far
    # the loaded hook travels along its rung. The pad is meant to make that small,
    # not zero -- a climber slips too -- so this asserts a ceiling, not a floor.
    d = slide_test(xml_path(1))
    print(f"\nphase 7 on lesson 1: 40 N sideways moves the loaded hook "
          f"{d:+.0f} mm along its rung (the bare hook gives ~+107)")
    assert abs(d) < 60.0, (f"the hook slid {d:+.0f} mm: the grip pad is not doing "
                           f"its job and lesson 1 is solvable by sliding again")
    print("all lessons written to", OUT)


if __name__ == '__main__':
    main()
