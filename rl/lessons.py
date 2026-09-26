"""Curriculum lesson scenes for PPO, one scene per task, ordered by difficulty.

Each scene is generated from ../urdf/scene.xml: the robot, actuators, sensors and the
verified hanging keyframe are kept verbatim; the walls and rungs are replaced per
lesson. Every lesson therefore starts from the same keyframe, which pins the start
rung at x = 0.44, z = 1.00, spanning at least y = +-0.16.

A human climber pulls up by flaring the shoulders out in the wall plane and
flexing the elbows. The shoulder is pitch-first (cad/src/lib/spec.py), so from the
hang that flare is positive abduction on the left, and scene.xml has all of it -
the lessons used to widen the range, which the abduction-first shoulder had only
inward of the hang.

    python -m rl.lessons      # writes urdf/lessons/*.xml and smoke-checks each one
"""
import hashlib
import os
import re
import numpy as np
from controller import SCENE, URDF

OUT = os.path.join(URDF, 'lessons')

# scene.xml's walls, 8 m long and 6 m deep: every layout here fits, mirrored too --
# lesson 5's far side is 3.3 m along, lesson 4's staircase ends 1.15 m down
WALL_FRONT = '    <geom name="wall_front" type="box" material="wall" pos="0.550 0.00 0.50" size="0.05 4.00 3.00"/>'
WALL_BACK = '    <geom name="wall_back"  type="box" material="wall" pos="-0.300 0.00 0.50" size="0.05 4.00 3.00"/>'
WALL_SIDE = '    <geom name="wall_side"  type="box" material="wall" pos="0.125 -1.050 0.50" size="0.425 0.05 3.00"/>'

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
# Lessons 1..9 are the benchmark; each goes toward the robot's left (+y), and
# `mirrored` gives the same lesson toward its right.
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
    # Stop at the end. Every seat on a continuous rung is as good as the next, so
    # nothing before this asks a hand to land SHORT of somewhere -- and at a gap that
    # is the whole move: lead hand to the last seat, trailing hand up beside it, then
    # across (the user's description of the climb). Four local runs at 0 % on lesson
    # 1.5 each ended with the lead hook past its last seat, brushing the end, never
    # retreating the 5-10 cm to a seat, the trailing hand never moving. Here the rung
    # ends 3 cm past the left hand's goal, and its goal is one ordinary opener from
    # its start (mac19's hops: p50 226 mm): land there, or hang half off the end.
    1.1: dict(name='stop_at_the_end', timeout=30.0,
              title='one step each along a rung that ends just past the goal',
              walls=[WALL_FRONT],
              rungs=[_ry('start', -0.90, 0.42, 1.00)],
              goal_rungs=['rung_start'],
              goals=[[0.44, 0.3875, 1.00], [0.44, 0.0755, 1.00]]),
    # Then the gaps, in the order of the reach they ask of a hand that crosses from
    # the last seat with its partner still at the start (0.54 m + gap, against the
    # 0.67 m static limit): 0.58, 0.62, 0.66 for lesson 1.5 -- the mac16 policy
    # pushed its hook into the far segment's end 38 mm short there -- and 0.79 for
    # lesson 2, which cannot be lunged at all: the trailing hand has to come up first.
    # (with the map describing SEATS, half a hook-width in from each end --
    # climb_task.SEAT_IN -- the seat-to-seat jumps are 105, 145, 185 and 315 mm)
    # ONE gap each, goals on the segment just across it (0.6 m of travel, not 1.4 m
    # over two): the first version asked for the full traverse and the policy that
    # had just learnt to stop short crawled it in 5-9 cm steps -- the climber's
    # order visible on every replay, the goals never reached in 60 s, 0/40 for
    # 3.3M steps, and the crawl spread to lesson 1 (95 % -> 20 %). Teach the
    # crossing on its own; lesson 1.5 asks for two gaps and the distance.
    1.2: dict(name='cross_a_hairline_gap', timeout=60.0,
              title='cross one 0.04 m gap: to the last seat, the other hand up, over',
              walls=[WALL_FRONT],
              rungs=[_ry('start', -0.45, 0.35, 1.00),
                     _ry('seg1', 0.39, 0.95, 1.00),
                     _ry('seg2', 0.99, 1.65, 1.00)],
              goal_rungs=['rung_seg1'],
              goals=[[0.44, 0.76, 1.00], [0.44, 0.45, 1.00]]),
    1.35: dict(name='cross_a_narrow_gap', timeout=60.0,
               title='cross one 0.08 m gap',
               walls=[WALL_FRONT],
               rungs=[_ry('start', -0.45, 0.35, 1.00),
                      _ry('seg1', 0.43, 0.95, 1.00),
                      _ry('seg2', 1.03, 1.65, 1.00)],
               goal_rungs=['rung_seg1'],
               goals=[[0.44, 0.80, 1.00], [0.44, 0.49, 1.00]]),
    1.5: dict(name='traverse_small_gaps', timeout=60.0,
              title='traverse a broken rung: segments with 0.12 m gaps',
              walls=[WALL_FRONT],
              rungs=[_ry('start', -0.45, 0.35, 1.00),
                     _ry('seg1', 0.47, 0.95, 1.00),
                     _ry('seg2', 1.07, 1.65, 1.00)],
              goal_rungs=['rung_seg2'],
              goals=[[0.44, 1.55, 1.00], [0.44, 1.24, 1.00]]),
    # The last rung before lesson 2, and the measurement that says why it is here:
    # a crossing is a reach from the last SEAT of one segment to the first seat of
    # the next -- 185 mm on lesson 1.5, 315 mm on lesson 2, against hops of 226 mm
    # at the median and 307 at p90 (mac19 on lesson 1). So lesson 2's crossing sits
    # past the p90 of the step this gait produces, with nothing in between: 0/40
    # through every run that reached it. 0.18 m makes the reach 245 mm, one
    # ordinary hop, and it is a single crossing like 1.2 and 1.35 -- the two
    # lessons that turned the gap skill from "never" into 90-100 %.
    1.7: dict(name='cross_a_wide_gap', timeout=60.0,
              title='cross one 0.18 m gap',
              walls=[WALL_FRONT],
              rungs=[_ry('start', -0.45, 0.35, 1.00),
                     _ry('seg1', 0.53, 0.95, 1.00),
                     _ry('seg2', 1.13, 1.65, 1.00)],
              goal_rungs=['rung_seg1'],
              goals=[[0.44, 0.90, 1.00], [0.44, 0.59, 1.00]]),
    # Between 1.7 and 2 the step used to change two things at once -- the gap goes
    # 0.18 -> 0.25 m AND one gap becomes two -- which is the double step the rest of
    # this curriculum exists to avoid. Measured on win15 3's best, that is exactly
    # where it breaks: in 6 of 10 lesson-2 episodes the lead hook crossed the gap
    # while the trailing one was still 246-313 mm short of its own seat edge, leaving
    # the arms at 583-651 mm of a ~671 mm span -- full stretch, the free hook only
    # 3-25 mm onto the next seat, and the trailing hand then too far back to follow.
    # All six failed; the four that kept the trailing hand within 64 mm of the edge
    # hold both successes. So 1.85 isolates the width: one 0.25 m gap, nothing else
    # new, with the goals on seg1 the way 1.7 has them.
    1.85: dict(name='cross_a_wide_gap_again', timeout=60.0,
               title='cross one 0.25 m gap',
               walls=[WALL_FRONT],
               rungs=[_ry('start', -0.45, 0.35, 1.00),
                      _ry('seg1', 0.60, 0.95, 1.00),
                      _ry('seg2', 1.20, 1.65, 1.00)],
               goal_rungs=['rung_seg1'],
               goals=[[0.44, 0.90, 1.00], [0.44, 0.66, 1.00]]),
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
    # Climbing down comes before climbing over, which needs both directions. A row
    # straight below the hang is inside the robot (its arms and body fill 0.25-1.10 m
    # under the start rung), so the way down is a staircase: each step lower and
    # further along, the way a climber steps down beside the holds, not onto them.
    4: dict(name='climb_down', timeout=120.0,
            title='down a staircase of short rungs, each lower and further along',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 0.40, 1.00),
                   _ry('step1', 0.60, 1.10, 0.60),
                   _ry('step2', 1.25, 1.75, 0.25),
                   _ry('step3', 1.95, 2.45, -0.15)],
            goal_rungs=['rung_step3'],
            goals=[[0.44, 2.33, -0.15], [0.44, 2.02, -0.15]]),
    # The break is past what the robot can bridge even with momentum (1.55 m hook to
    # hook), so the only way across is up, along the top and down.
    5: dict(name='climb_over', timeout=120.0,
            title='up, across the top, down the far side of a 2.2 m break',
            walls=[WALL_FRONT],
            rungs=[_ry('start', -0.90, 0.35, 1.00),
                   _ry('top', -0.90, 3.30, 1.45),
                   _ry('far', 2.55, 3.05, 1.00)],
            goal_rungs=['rung_far'],
            goals=[[0.44, 2.93, 1.00], [0.44, 2.62, 1.00]]),
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
    # one hook holds while the robot swings the other across. Thought well inside the
    # 1.47 m kinematics allow a swing; the one motion found across it (rl/README.md,
    # 2026-09-26) catches the far rung 1.10 m from the other hook, all the reach there
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
    """'2' -> 2, '-1' -> -1, '1.5' -> 1.5: keys are ints except the half-steps"""
    f = float(str(x).strip())
    return int(f) if f == int(f) else f


def lesson_tag(lesson):
    """'lesson01', 'lesson-2', 'lesson1.5' -- the file and model name stem"""
    return f"lesson{lesson:02d}" if isinstance(lesson, int) else f"lesson{lesson:g}"


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
    return head + mid + '\n\n    <!-- ===== robot ===== -->' + tail


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
    for k, mirror in [(k, False) for k in sorted(LESSONS)] + [(k, True) for k in range(1, 10)]:
        spec = spec_of(k, mirror)
        path = write(k, mirror)
        e = ClimbingEnv(xml=path, success_z=1e9)
        e.reset()
        p0 = e.data.xpos[e.base].copy()
        ctrl = e.data.ctrl.copy()
        for _ in range(int(1.0 / e.model.opt.timestep)):
            e.step(ctrl)
        drift = np.linalg.norm(e.data.xpos[e.base] - p0) * 1000.0
        # 0.7 mm on the 21.5 kg robot. It read 3.8 while the hooks hung back to
        # front and the arms were jammed into the torso -- a drift budget is a
        # symptom detector, so it stays where it was rather than following one.
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
