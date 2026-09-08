"""Curriculum lesson scenes for PPO, one scene per task, ordered by difficulty.

Each scene is generated from ../urdf/scene.xml: the robot, actuators, sensors and the
verified hanging keyframe are kept verbatim; the walls and rungs are replaced per
lesson. Every lesson therefore starts from the same keyframe, which pins the start
rung at x = 0.44, z = 1.00, spanning at least y = +-0.16.

The lessons use the WIDE-SHOULDER model: abduction opened 60 deg outward on each
side, the same two-attribute edit as scene_wide_shoulders.xml. A human climber pulls
up by flaring the shoulders out in the wall plane and flexing the elbows; the
original ranges start at the hang posture and only travel inward, so that motion
does not exist in scene.xml. The classical controller in ../controller is tuned
against the narrow ranges (its solvers search the ctrlrange) and runs on scene.xml
only; PPO learns from scratch, so it gets the corrected model from day one.

    python -m rl.lessons      # writes urdf/lessons/*.xml and smoke-checks each one
"""
import os
import numpy as np
from controller import SCENE, URDF

OUT = os.path.join(URDF, 'lessons')

# the walls of scene.xml, verbatim -- big enough for every rung layout used here
WALL_FRONT = '    <geom name="wall_front" type="box" material="wall" pos="0.550 0.30 1.00" size="0.05 1.35 2.00"/>'
WALL_BACK = '    <geom name="wall_back"  type="box" material="wall" pos="-0.300 0.30 1.00" size="0.05 1.35 2.00"/>'
WALL_SIDE = '    <geom name="wall_side"  type="box" material="wall" pos="0.125 -1.050 1.00" size="0.425 0.05 2.00"/>'

# widening the shoulders: identical to the scene.xml -> scene_wide_shoulders.xml diff
WIDEN = [
    ('<joint name="shoulder_L_abduction" type="hinge" axis="1 0 0" range="0.00000 3.14159"/>',
     '<joint name="shoulder_L_abduction" type="hinge" axis="1 0 0" range="-1.04720 3.14159"/>'),
    ('<joint name="shoulder_R_abduction" type="hinge" axis="1 0 0" range="-3.14159 0.00000"/>',
     '<joint name="shoulder_R_abduction" type="hinge" axis="1 0 0" range="-3.14159 1.04720"/>'),
    ('joint="shoulder_L_abduction" kp="800" dampratio="1" ctrlrange="0.00000 3.14159"',
     'joint="shoulder_L_abduction" kp="800" dampratio="1" ctrlrange="-1.04720 3.14159"'),
    ('joint="shoulder_R_abduction" kp="800" dampratio="1" ctrlrange="-3.14159 0.00000"',
     'joint="shoulder_R_abduction" kp="800" dampratio="1" ctrlrange="-3.14159 1.04720"'),
]


# --- the grip pad: what makes the hook hold along its rung --------------------
# The hook is a slot round a smooth cylinder, so along the rung axis nothing holds
# it: `controller/test_climb.py` phase 7 measures 40 N moving it ~100 mm, and PPO
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


# A second touch zone on each hook: the LIP, the underside of the mouth. The one
# thing on the wall map a free hand can feel is the END of a rung, and it touches
# the hook there and only there -- measured over 252 end contacts, every one at
# z = -110 mm in the hook frame, while every seat, loaded or light, touches at
# z >= -98 (the top face of the lip). Same sensor as the grip pad, split in two:
# the throat says "on a rung", the lip says "against the end of one". The
# estimator uses it to find out how far the holding hook has crept (`_end_contact`).
TIP = [
    ('<site name="s_grip_L" type="box" pos="0.02600 0 -0.05900" size="0.02400 0.03250 0.06300"/>',
     '<site name="s_grip_L" type="box" pos="0.02600 0 -0.05900" size="0.02400 0.03250 0.06300"/>\n'
     '                            <site name="s_tip_L" type="box" pos="0.01000 0 -0.11100" size="0.02200 0.03500 0.00600"/>'),
    ('<site name="s_grip_R" type="box" pos="0.02600 0 -0.05900" size="0.02400 0.03250 0.06300"/>',
     '<site name="s_grip_R" type="box" pos="0.02600 0 -0.05900" size="0.02400 0.03250 0.06300"/>\n'
     '                            <site name="s_tip_R" type="box" pos="0.01000 0 -0.11100" size="0.02200 0.03500 0.00600"/>'),
    ('<touch  name="grip_L"      site="s_grip_L"/>',
     '<touch  name="grip_L"      site="s_grip_L"/>\n    <touch  name="tip_L"       site="s_tip_L"/>'),
    ('<touch  name="grip_R"      site="s_grip_R"/>',
     '<touch  name="grip_R"      site="s_grip_R"/>\n    <touch  name="tip_R"       site="s_tip_R"/>'),
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
# Row spacing stays <= 0.50 m (the measured 0.671 m reach x 75 %) except lesson 9,
# which is deliberately beyond static reach.
LESSONS = {
    # Three rungs below lesson 1, because lesson 1 cannot be found from scratch any
    # more. Once dragging a loaded hook and jumping stopped paying, the only paid
    # move left was lift-and-place, and 1.1 m of it is five cycles per hand: the
    # 15M-step win5 run never completed one, settled into hanging from one hook
    # and fluttering the other, and its action noise collapsed from 1.0 to 0.13.
    #
    # -2: the LEFT hand takes one step of 0.30 m toward the goal side; the right
    #     hand's goal is where it already is, so all it owes is one supported
    #     let-go and re-seat for MIN_GRABS. Nothing is in anyone's way.
    # -1: the RIGHT hand takes one step of 0.30 m -- the other way, because 0.30 m
    #     toward the left hand would end 14 mm from it. Same skill, mirrored: this
    #     arm lifts and places, that hand carries. Without it the 2M local run
    #     reached lesson 1 with a right hand that had never lifted, and dragged it.
    #  0: both hands take one step of 0.40 m. Not 0.30: the first version of this
    #     put the right hand's goal 14 mm from where the left hand starts, and two
    #     65 mm hooks cannot share that span -- measured on a 600k-step policy, the
    #     right hook reached its goal region on 42 % of steps and hovered there
    #     100 mm (min 44) from the left hook, while the left hook, which had to
    #     move first, never let go. 0.40 m puts the goal 86 mm past the other
    #     hand's start.
    -2: dict(name='step_left_hand', timeout=30.0,
             title='the left hand alone takes one 0.30 m step along a continuous rung',
             walls=[WALL_FRONT],
             rungs=[_ry('start', -0.90, 1.50, 1.00)],
             goal_rungs=['rung_start'],
             goals=[[0.44, 0.457, 1.00], [0.44, -0.157, 1.00]]),
    -1: dict(name='step_right_hand', timeout=30.0,
             title='the right hand alone takes one 0.30 m step, away from the left',
             walls=[WALL_FRONT],
             rungs=[_ry('start', -0.90, 1.50, 1.00)],
             goal_rungs=['rung_start'],
             goals=[[0.44, 0.157, 1.00], [0.44, -0.457, 1.00]]),
    0: dict(name='step_both_hands', timeout=30.0,
            title='both hands, one step of 0.40 m along a continuous rung',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 1.50, 1.00)],
            goal_rungs=['rung_start'],
            goals=[[0.44, 0.557, 1.00], [0.44, 0.243, 1.00]]),
    1: dict(name='traverse_continuous', timeout=60.0,
            title='traverse sideways along one continuous rung',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 1.50, 1.00)],
            goal_rungs=['rung_start'],
            goals=[[0.44, 1.257, 1.00], [0.44, 0.943, 1.00]]),
    # 1.5 sits between the continuous rung and the 0.25 m gaps because 0.25 was
    # never crossed once in 2M steps of exposure, in three local runs. Measured on
    # the last of them: when the free hand came closest to the far ledge it was
    # 120-160 mm short, the hands were 627-689 mm apart -- the static reach limit
    # -- and the HOLDING hand was still 0.47-0.54 m from the edge of its segment.
    # The order a climber uses -- lead hand to the edge, trailing hand up beside
    # it, then reach -- was never found because it was never once rewarded, and
    # the edge itself was not in the observation (it is now, see `_near_rungs`).
    # A 0.12 m gap is crossed by the ordinary 0.30 m step the policy already has,
    # so the first crossings happen by accident and the credit does the rest.
    1.5: dict(name='traverse_small_gaps', timeout=60.0,
              title='traverse a broken rung: segments with 0.12 m gaps',
              walls=[WALL_FRONT],
              rungs=[_ry('start', -0.45, 0.35, 1.00),
                     _ry('seg1', 0.47, 0.95, 1.00),
                     _ry('seg2', 1.07, 1.65, 1.00)],
              goal_rungs=['rung_seg2'],
              goals=[[0.44, 1.55, 1.00], [0.44, 1.24, 1.00]]),
    2: dict(name='traverse_gaps', timeout=60.0,
            title='traverse a broken rung: segments with 0.25 m gaps',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.45, 0.35, 1.00),
                   _ry('seg1', 0.60, 0.95, 1.00),
                   _ry('seg2', 1.20, 1.65, 1.00)],
            goal_rungs=['rung_seg2'],
            goals=[[0.44, 1.55, 1.00], [0.44, 1.24, 1.00]]),
    3: dict(name='climb_up', timeout=90.0,
            title='climb up two rows of continuous rungs',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 1.50, 1.00),
                   _ry('up1', -0.90, 1.50, 1.45),
                   _ry('up2', -0.90, 1.50, 1.90)],
            goal_rungs=['rung_up2'],
            goals=[[0.44, 0.157, 1.90], [0.44, -0.157, 1.90]]),
    4: dict(name='climb_over', timeout=120.0,
            title='up, across the top, down the far side',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 0.35, 1.00),
                   _ry('top', -0.90, 1.50, 1.45),
                   _ry('far', 1.00, 1.50, 1.00)],
            goal_rungs=['rung_far'],
            goals=[[0.44, 1.41, 1.00], [0.44, 1.10, 1.00]]),
    5: dict(name='climb_down', timeout=90.0,
            title='climb down one row',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 1.50, 1.00),
                   _ry('low', -0.90, 1.50, 0.55)],
            goal_rungs=['rung_low'],
            goals=[[0.44, 0.157, 0.55], [0.44, -0.157, 0.55]]),
    6: dict(name='staggered', timeout=120.0,
            title='climb staggered short rungs, each at its own height',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 1.50, 1.00),
                   _ry('stag1', -0.60, -0.25, 1.40),
                   _ry('stag2', 0.00, 0.35, 1.50),
                   _ry('stag3', 0.60, 0.95, 1.42),
                   _ry('top1', -0.35, 0.00, 1.88),
                   _ry('top2', 0.30, 0.65, 1.95)],
            goal_rungs=['rung_top1', 'rung_top2'],
            goals=[[0.44, 0.48, 1.95], [0.44, -0.18, 1.88]]),
    7: dict(name='wall_corner', timeout=120.0,
            title='transfer to the perpendicular wall in the corner',
            walls=[WALL_FRONT, WALL_SIDE],
            rungs=[_ry('start', -0.90, 0.90, 1.00),
                   _rx('side', -0.18, 0.43, 1.00)],
            goal_rungs=['rung_side'],
            goals=[[0.35, -0.94, 1.00], [0.04, -0.94, 1.00]]),
    8: dict(name='wall_behind', timeout=120.0,
            title='transfer to the parallel wall behind the back',
            walls=[WALL_FRONT, WALL_BACK],
            rungs=[_ry('start', -0.90, 1.50, 1.00),
                   _ry('back', -0.90, 1.50, 1.00, x=-0.19)],
            goal_rungs=['rung_back'],
            goals=[[-0.19, 0.157, 1.00], [-0.19, -0.157, 1.00]]),
    9: dict(name='far_reach', timeout=120.0,
            title='a row 0.75 m up, beyond the 0.671 m static reach',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 1.50, 1.00),
                   _ry('top', -0.90, 1.50, 1.75)],
            goal_rungs=['rung_top'],
            goals=[[0.44, 0.157, 1.75], [0.44, -0.157, 1.75]]),
}


def lesson_key(x):
    """'2' -> 2, '-1' -> -1, '1.5' -> 1.5: keys are ints except the half-steps"""
    f = float(str(x).strip())
    return int(f) if f == int(f) else f


def lesson_tag(lesson):
    """'lesson01', 'lesson-2', 'lesson1.5' -- the file and model name stem"""
    return f"lesson{lesson:02d}" if isinstance(lesson, int) else f"lesson{lesson:g}"


def _path(lesson):
    return os.path.join(OUT, f"{lesson_tag(lesson)}_{LESSONS[lesson]['name']}.xml")


def xml_path(lesson):
    """the lesson's scene file, written on demand if it is not there.

    `urdf/lessons/` is generated from `scene.xml` and gitignored, so a fresh clone
    -- or a pull that adds a lesson -- has no scene for it, and MuJoCo reports that
    as a bare ParseXML error once per worker process. Nothing is regenerated that
    already exists: edit `scene.xml` or a spec and you still run `rl.lessons`.
    """
    p = _path(lesson)
    if not os.path.exists(p):
        write(lesson)
    return p


def write(lesson):
    spec = LESSONS[lesson]
    xml = open(SCENE).read()
    xml = xml.replace('<mujoco model="bimanual_wall_climber_scene">',
                      f'<mujoco model="{lesson_tag(lesson)}_{spec["name"]}">')
    xml = xml.replace('meshdir="meshes/"', 'meshdir="../meshes/"')
    for old, new in WIDEN + PAD + TIP:
        assert old in xml, f"scene.xml changed; patch target not found: {old}"
        xml = xml.replace(old, new)
    head, rest = xml.split('    <!-- ===== walls ===== -->')
    _, tail = rest.split('    <!-- ===== robot ===== -->')
    mid = '\n'.join(
        [f'    <!-- lesson {lesson}: {spec["title"]} -->']
        + spec['walls'] + spec['rungs'])
    out = head + mid + '\n\n    <!-- ===== robot ===== -->' + tail
    # via a temp file and an atomic rename: the 8 SubprocVecEnv workers build their
    # environments at the same moment, and without it one of them parses the file
    # while another is still writing it
    os.makedirs(OUT, exist_ok=True)
    p = _path(lesson)
    tmp = f'{p}.{os.getpid()}.tmp'
    open(tmp, 'w').write(out)
    os.replace(tmp, p)
    return p


def rung_segments(model):
    """rung name -> (centre, unit axis, half length), straight from the model"""
    import mujoco
    out = {}
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if nm.startswith('rung_'):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            out[nm] = (model.geom_pos[g].copy(), R.reshape(3, 3)[:, 2],
                       float(model.geom_size[g][1]))
    return out


def slide_test(path, force=40.0):
    """phase 7's measurement: hook_R's travel along its rung under a sideways pull.

    Deliberately not built on `controller.moves`: this has to run on a lesson scene,
    whose rung names the controller's HANG table does not know. Grip sequence matches
    phase 7 -- both fingers shut, then R grips hard while L lets go.
    """
    import mujoco
    from climbing_env import ClimbingEnv, GRIP_SET
    from controller.moves import GRIP_HARD
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
    import mujoco
    from climbing_env import ClimbingEnv
    for k in sorted(LESSONS):
        spec = LESSONS[k]
        path = write(k)
        e = ClimbingEnv(xml=path, success_z=1e9)
        e.reset()
        p0 = e.data.xpos[e.base].copy()
        ctrl = e.data.ctrl.copy()
        for _ in range(int(1.0 / e.model.opt.timestep)):
            e.step(ctrl)
        drift = np.linalg.norm(e.data.xpos[e.base] - p0) * 1000.0
        assert drift < 5.0, f"lesson {k}: keyframe drifted {drift:.1f} mm"
        assert e.latched('L')[0] and e.latched('R')[0], f"lesson {k}: a hook came off"
        segs = rung_segments(e.model)
        for nm in spec['goal_rungs']:
            assert nm in segs, f"lesson {k}: goal rung {nm} missing"
        for gp in spec['goals']:
            gp = np.asarray(gp)
            d = min(np.linalg.norm(gp - (c + np.clip((gp - c) @ a, -h, h) * a))
                    for c, a, h in segs.values())
            assert d < 0.02, f"lesson {k}: goal {gp} is {d*1000:.0f} mm off every rung"
        lo = e.model.jnt_range[e.model.joint('shoulder_L_abduction').id][0]
        assert lo < -1.0, "shoulders were not widened"
        pad = [g for g in range(e.model.ngeom)
               if e.model.geom_bodyid[g] in (e.hook['L'], e.hook['R'])
               and e.model.geom_priority[g] == 1]
        assert len(pad) == 4, f"lesson {k}: grip pad missing from the hooks"
        print(f"lesson {k} {spec['name']:>20}: drift {drift:.2f} mm, "
              f"{len(spec['rungs'])} rungs, OK")

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
