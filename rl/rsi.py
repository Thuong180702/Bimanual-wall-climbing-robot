"""Reference-state initialisation for family 3 (rl.walls): a share of its walls start partway
up a climb that is known to work, so the policy meets the end of a climb before it can find
the start of one.

    python -m rl.rsi                    # rebuild rl/fixtures/rsi_family3.npz from the climbs, and check it
    python -m rl.rsi --teach out/RUN    # record the TAUGHT climbs with RUN's policy first

Why (2026-10-06): gpu2 resumed onto family 3's rows standing out (35M steps, 7792 episodes,
39 % of its steps) and never once touched the row above. Not the resumed policy: on the
same walls in MJX a fresh SB3 policy, uniform random actions and gpu2's own noise all
touched it in 0 of 32 episodes, their throats getting no nearer than 191-195 mm of a goal
283 mm off. A climb is 15-20 s of swing, reach out and up, seat, follow -- a chain nothing
random strings together -- and with no success there is nothing for PPO to follow.

The climbs are joint-setpoint sequences found by CEM (out/climb_demo/plan_chain.py, HILL=2:
PLAN=f3x20 seed 42; f3x25w seed 43 and f3x30w seed 47 warm-started from it, INIT=) on the
robot MJX trains -- the simplified one, `XML=` that robot's scene -- each on one wall:
the start row and one row SPACING up and `walls.OUT` out, full length, goals straight
above the hang. Replayed here through ClimbTask.step on that robot, every control step's
state becomes a row, up to a second before the climb succeeds, and every EVERY of them a
start, booked as one and begun from any of its rows (`mix`): the joints and
their speeds, the setpoints, and the holds each hand has really taken by then (success
asks each hand for one). Family 3's rows are the same full-length rungs whatever its
goals and mirroring, so a start fits every family-3 wall of its spacing.

None may climb more than HILL_MAX of live potential on the way (the barrier test, run on the
motion itself, `build_fixture`): the first climbs, found without HILL, swung the free arm
across the body up 0.40-2.44 m of it, and gpu3 (2026-10-06, 60M steps) learnt every start
after the swing (38/40) and none on it (0/80) -- going back to the start rung paid more.

Those episodes are booked as family 3 + 100 (rl.train_ppo.Walls): they never count toward
family 3's gate or its ADR probes, which stay what the policy does from the hang.

Which start a wall gets is drawn toward each climb's FRONTIER -- the latest start along it
the policy does not yet finish from (`frontier`, from Walls' books per start) -- and not
evenly: gpu4 (2026-10-07, 64M steps) finished every 0.20 m start from 5.0 s on, the hook
already over the row above, and none before, at 50M as at 64M; the start at 4.8 s, the
hook 30 mm under that row and rising, had seen ~128 episodes in the whole run (~2 per 1M
steps) and the policy pulled it straight into the rung's underside every time. Three
control steps of the climb from there and it finished 8/8; the climb from there with the
policy's own noise on every joint, 4/8. A start one step back from what is learnt is
learnable; drawn 1 time in 66 it is never learnt. FOCUS of the starts now go to the BACK + 1
starts ending at the frontier, the rest evenly, so what is learnt is still visited.

A start booked fewer than START_MIN times is drawn as one at the frontier. Drawn evenly, cold
books (a resume, a climb ADR lets in) put the frontier at the climb's last start, and it
walked down through every start the policy already finishes from, BACK + 1 of them per
TIMEOUT lag: for gpu4, 41 of them, 13-25M steps simulated before it reached 4.8 s; 6-9M now.

Where the books say the policy does not finish, rl.train_ppo.Clone also clones the climb's own
action there (`action`): gpu6 (2026-10-10, 61M steps) finished every row from 4.88 s and none
before, 0/64 at 4.84 s, the catch there 7-8 sigma of its noise away. Only at each climb's
frontier and the BACK starts before it (`cloned`, 2026-10-14): every start not passed was
gpu11's family 5 entering, 2786 rows, and a rollback.

Two rows (`TAUGHT`, `teach`): gpu7 (2026-10-10, 173M steps) climbed one row 24/24 and two
0/24, stopped on the middle row by the start row in its view (see `teach`). Those climbs are
the policy's own, recorded with the rows below both its hands out of its observation, and
join the found ones as climbs of their own: starts, books and frontier per climb, and
`mix` draws each on a family-3 wall of its rows and spacing, offered by spacing alone (`allowed`).
What Clone clones on them is the teacher's mean action at each state, not the step it took.

A start's episodes begin from any of its EVERY control steps, not its first only: gpu4's
policy finished 0/192 from 4.8 s, the first step of a start, and every one from 5.0 s, the
next start's -- a cliff with nothing between to learn from, however often it is drawn.

Family 6 too (`STEPS6`, 2026-10-11): gpu8 (162M steps) passed families 1-4 and stayed at
0 % on 5 for 50M steps and on 6 for 43M, the same in both: the policy goes to the start
row's end and hangs there, its hook never more than 2-22 cm up (the rung above 30-50 cm).
Not the reward (lifting there is downhill all the way, 0.0 mm of hill) nor the map (the
lidar has every rung from the first step): family 3's rows stand OUT from the wall, and the
diagonal step up onto a rung in the wall's own plane, past the end of the one below, was
never taught -- family 3 in gpu1-gpu6 again. A family-6 climb is found on one wall
(`wall6`) and fits only that one, each way round (`side`), so an episode from its start is
on that wall, mirrored for the climb found the other way, booked as family 6 + BOOK.

Family 4 (`TAUGHT4`, 2026-10-13): gpu9 (ea972f3, 179M steps) sat at 45 % on family 4 from 51M
on, ADR never off its first tops, because it only stepped down toward -y: on 24 walls every
one toward +y (the benchmark's own way) timed out, no hook ever more than 6 cm down, and
10/11 toward -y succeeded in 3-7 s. gpu7 sat the same way for 90M steps; gpu8 got out, and
at 162M stepped down both ways (+y 12/13, -y 9/11). Which side a policy learns first is
chance; that it learns the other is not given. Each way round, gpu8's own step down on one
wall (`wall4`), recorded as the two-row climbs are (`teach4`), booked as family 4 + BOOK.

BACK6 (2026-10-13): a start's episode is booked only once every episode begun before it can
have ended, its family's timeout later -- family 6's 60 s, 6.2M steps at 4096 envs -- so the
frontier moves at most BACK + 1 starts per lag. gpu9's family-6 frontier, at BACK 2, moved
0.2 s of climb per 2.6M steps (-y) and per 5.4M (+y), the -y at ~80 % of that cap, and its
policy at 179M finished from 29.0 s 70 % and 30.0 s 90 % (-y; the books: 30 %, 25 %) and
from 30.0 s 77-80 % (+y; books 15 %), where the books still put the frontiers at 30.4 s and
33.0 s. Family 6's climbs are ~37 s, 187 starts each; family 3's and 4's, 4-15 s and 30 s
(3.1M steps) a lag, keep BACK: spread three times wider their hardest start, the one the
frontier waits at, would be drawn a third as often (gpu4's: 1 time in 66 was never learnt).

Families 8 and 9 (`STEPS8`, `STEPS9`, 2026-10-13): gpu10 had both at ~0 % for 50M+ steps.
On family 8 its left hook reached the back rung and loaded it, and its right never followed.
A free search (plan_chain.py) did the same: a hook holds a rung behind the robot only turned
round, its tip toward the back wall. Pressed on the rung's front with its back face, it
"holds" 127 N and nothing once the other hand lets go. That grip is the arm raised sideways
over the head (shoulder abduction ~3 rad, flexion -2.8 -> -0.4) and nothing a search's noise
finds from the hang. Least squares found it clear of every rung and wall, with the right
still seated on the start row, and it seeded the search (TURN, IKQ). The move is 16 s, its
open loop 27/27 on Warp. Family 9 (gap 0.55) is a climber's crossing at 0.83 m hook to hook,
static, ~40 s, each way round.

Family 5 (`STEPS5`) and family 9 past its first top (2026-10-13), found before either was stuck.
Family 5 at its first break (1.80) is family 6's step up (its ten phases searched again, seeded
by family 6's climb), three steps along the top, and family 4's step down. That is 55.9 s, under
the walls' 60 s from the hang. Family 9 has a climb every 0.1 m of gap (`STEPS9`): 0.60 is still
crossed statically, and 0.70, 0.80 and 0.90 by lesson 9's swing, searched again on the simplified
robot (seeded by its full-robot chain) with no hill weight. A HILL weight stopped the catch at
1.01 m; this one catches 1.11 m from the other hook. The swing's pendulum is a 3.5-4.4 m hill
of live potential (HILL_SWING). The climb still ends at +16.4.

The simplified robot is its own mirror image (out/climb_demo/mirror_fixture.py: joint map,
forward kinematics exact). So a static climb found one way round is mirrored to the other
(family 5, family 9 at 0.60), and replays the same. A swing does not: mirrored, 0.90 fell, so
each swing's other way round was searched, seeded by the mirrored chain.

Open loop on Warp, from each climb's starts, +y / -y: f5 23/26 / 23/26; f9 0.60 26/26 / 26/26,
0.70 27/27 / 20/27, 0.80 26/27 / 25/27; 0.90 only 13/27 / 17/27. The 0.90 swings finish from
every start after the catch and from few before it: the catch C MuJoCo found sits millimetres
inside the far rung's seat, and Warp's physics misses it. 0.90 is family 9's last top, so it
can be searched again before ADR gets there.

Family 7 (`TAUGHT7`, 2026-10-14): gpu11 (fd642d2) sat at ~50 % on family 7 from ~100M to 226M,
stalled, and it took 16-25 % of all steps. It only turned the corner rl.walls' own way (the side
wall toward -y): 15/15 that way at 210M, 0/9 mirrored, every one a timeout. Family 4 in gpu9
again. gpu10's policy (215M) turns it both ways (22/24 at side 0.60). Its own move each way round
on one wall (`wall7`), recorded as family 4's are (`teach7`), booked as family 7 + BOOK.

Family 8 past 0.50 (`STEPS8B`, the same day): gpu11 climbed ADR to 0.45-0.50 on the 0.35 m climb
alone (0.50 5/64 from the hang, 0.55 0/64; lesson 8 is 0.63). Found as 0.35's was, with a hill
weight (HILL=2: without it the left arm's swing overhead climbed 1.7-4.3 m of live potential), at
0.60 from IK poses and at 0.50, 0.55 and 0.70 from 0.60's climb: 16.8-17.4 s, hills 8-12 mm,
open loop on Warp 25/29, 26/29, 26/29 and 23/28 of their starts.

Family 3's wider two rows (`TAUGHT3`, the same day): at its last tops (2 rows, 0.45) gpu11 went
from 41/48 at 100M to 34/48 at 200M and 226M on the same walls. All of the loss was two rows
0.35-0.45 apart (11/15 -> 6/15, released), where no climb was offered: TAUGHT's two rows stop at
0.25. Its 100M policy's own climbs at 0.35 and 0.40 (57/64, 32/64; `teach3`), lesson 3's own
spacing, are its references.

Family 6's two tiers (`TAUGHT6`, the same day): one tier 19/24 (210M), two tiers 0/16 (226M),
never off the start row. The far goal stops it, not the map (`teach6`). Shown tier 1's end as its
goals until both hands are on it, the same policy climbed both tiers on 3/16. Its climb that way
on one wall (`wall6x2`, +y 12/64), and that climb's mirror image for -y (`mirror_taught`: -y
itself reached tier 1 28/32 and tier 2 never), are offered with the one-tier climbs (by rise),
before ADR's tiers reach 2.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, 'fixtures', 'rsi_family3.npz')
# spacing -> the found climb's setpoints (ctrl0, then one row per control step)
CLIMBS = {s: os.path.join(HERE, 'fixtures', f'climb_f3_s{round(s * 100)}.npz')
          for s in (0.20, 0.25, 0.30)}
# (spacing, rows) -> a 2-row climb from the hang a trained policy made (`teach`): its states and
# actions, recorded in MJX, the policy's own climb up the first row and a reference for the second
TAUGHT = {(s, 2): os.path.join(HERE, 'fixtures', f'climb_f3x2_s{round(s * 100)}.npz')
          for s in (0.20, 0.25)}
# (rise, side) -> family 6's found climb onto one tier `rise` up (`wall6`), side 1 toward +y
# (PLAN=f6), -1 toward -y (PLAN=f6m): setpoints as CLIMBS'
STEPS6 = {(r, side): os.path.join(HERE, 'fixtures', f'climb_f6{"m" if side < 0 else ""}_r{round(r * 100)}.npz')
          for r in (0.30, 0.40) for side in (1, -1)}
# (drop, side) -> family 4's step down `drop` (`wall4`), side 1 toward +y, a trained policy's own
# (`teach4`): recorded as TAUGHT's
TAUGHT4 = {(d, side): os.path.join(HERE, 'fixtures', f'climb_f4{"m" if side < 0 else ""}_d{round(d * 100)}.npz')
           for d in (0.30,) for side in (1, -1)}
GAP4 = 0.10          # m on past the start row's end that `wall4`'s step starts: family 4's first top
# (side, way) -> family 7's move onto the side rung `side` out (`wall7`), way 1 rl.walls' own (the
# corner toward -y), -1 mirrored: a trained policy's own (`teach7`), recorded as TAUGHT4's
TAUGHT7 = {(s, way): os.path.join(HERE, 'fixtures', f'climb_f7{"m" if way < 0 else ""}_s{round(s * 100)}.npz')
           for s in (0.60,) for way in (1, -1)}
# (spacing, rows) -> family 3's two rows past TAUGHT's spacings, to lesson 3's 0.40: an earlier
# policy's own climb (`teach3`), as it climbed them before it forgot (gpu11 100M, 2026-10-14).
# Not 0.45, the last top: 2/64 there, the faster 23 s with six re-grips, nothing to clone
TAUGHT3 = {(s, 2): os.path.join(HERE, 'fixtures', f'climb_f3x2_s{round(s * 100)}.npz')
           for s in (0.35, 0.40)}
# (rise, side) -> family 6's climb up two tiers (`wall6x2`), side 1 toward +y: a trained policy's own,
# shown tier 1's end as its goals until both its hands are off the start row (`teach6`)
TAUGHT6 = {(r, side): os.path.join(HERE, 'fixtures', f'climb_f6x2{"m" if side < 0 else ""}_r{round(r * 100)}.npz')
           for r in (0.30,) for side in (1, -1)}
# (behind, side) -> family 8's found move onto the back rung `behind` (`wall8`), and (gap, side) ->
# family 9's across one gap (`wall9`), side 1 toward +y: setpoints as CLIMBS' (plan_chain.py PLAN=f8,
# f9g55 / f9mg55). Family 8's one way round only: both its rows run full length, and mirrored
# its wall is the same wall
STEPS8 = {(b, 1): os.path.join(HERE, 'fixtures', f'climb_f8_b{round(b * 100)}.npz') for b in (0.35,)}
# ...and past 0.50 (2026-10-14), after every climb before them so a run's books are kept: gpu11 rose to
# 0.45 on 0.35's alone, 0.50 5/64 and 0.55 0/64 from the hang, lesson 8 at 0.63. plan_chain f8b60 with
# TURN, HILL=2 and IKQ from ik_f8.py at 0.60 (hill 12 mm; without HILL 1.7-4.3 m), then f8b50, f8b55
# and f8b70 from that climb (INIT): their own IK poses' searches did not get the left hook over
STEPS8B = {(b, 1): os.path.join(HERE, 'fixtures', f'climb_f8_b{round(b * 100)}.npz') for b in (0.50, 0.55, 0.60, 0.70)}
# Family 9 every 0.1 m from its first top to its last (0.90, lesson 9's): a climb is offered once
# the gap's top reaches it, so each top past 0.55 is at most 0.05 past a climb. Past the static
# reach the crossing is lesson 9's swing (plan_chain f9s)
STEPS9 = {(g, side): os.path.join(HERE, 'fixtures', f'climb_f9{"m" if side < 0 else ""}_g{round(g * 100)}.npz')
          for g in (0.55, 0.60, 0.70, 0.80, 0.90) for side in (1, -1)}
# (break, side) -> family 5's found climb over a break (`wall5`): family 6's step up, along the
# top, family 4's step down (plan_chain f5b180 / f5mb180)
STEPS5 = {(b, side): os.path.join(HERE, 'fixtures', f'climb_f5{"m" if side < 0 else ""}_b{round(b * 100)}.npz')
          for b in (1.80,) for side in (1, -1)}
LONG = (5, 6, 9)     # families whose found climbs take ~40 s or more: TIMEOUT6 and BACK6
SLOW = (5, 6, 8, 9)  # families whose hang runs to its 60 s timeout until learnt: SHARE6_HI
TOP = {3: 'spacing', 4: 'drop', 5: 'break', 6: 'rise', 7: 'side', 8: 'behind', 9: 'gap'}   # the top a family's climbs are offered by (`allowed`): their height
SHARE = 0.25         # of a family's walls started partway up once it has passed its gate, new
SHARE_HI = 0.75      # climbs of its own or not (train_ppo.Walls.share), and until then. From the hang
                     # family 3 teaches nothing until the starts' frontier reaches it: gpu4 gave it
                     # 44 % of all steps for 0 %, the starts 6 %. A quarter stays from the hang, for
                     # its gate and ADR. Past the gate three in four: gpu11's families 6 and 8, given
                     # harder climbs and fewer hangs, forgot the walls they had passed.
SHARE6_HI = 0.95     # ...SLOW families': the hang runs 60 s to a timeout until the frontier is near
                     # it, five starts' worth -- at 0.75, 63 % of gpu10's family-6 steps (150-215M)
                     # went to the hang's timeouts. One in twenty still gives its gate ~650 a 10M
TIMEOUT = 30.0       # s an episode from a start may run: twice the longest rest of a climb
                     # (14.2 s, 0.20 m from the hang). A start not finished by then is not
                     # going to be; gpu4's failed starts idled out the family's 60 s.
TIMEOUT6 = 60.0      # ...LONG families' (`timeout`): their found climbs take ~40 s from the hang
FOCUS = 0.75         # of the starts drawn: at each climb's frontier...
BACK = 2             # ...the frontier start and the BACK before it, evenly: BACK + 1 a lag
BACK6 = 8            # ...LONG families' (above)
START_WINDOW = 20    # a start's book: its last episodes...
START_MIN = 10       # ...at least this many before it can count as passed...
PASS = 0.7           # ...at this rate (train_ppo.UNLOCK_AT, the families' gate)
SLIP = 0.5           # ...and holds the frontier only under this (`slipped`): a start learnt at 0.85
                     # reads under PASS over 20 episodes 7 % of the time, and with 100+ of them past a
                     # climb's cliff one always did. gpu11's frontier sat on such a start, not at the
                     # cliff, at 76-100 % of its checkpoints 500-700M (f9 +y at 700M: 34.8 s at 0.65,
                     # the cliff 23.6-24.8 s at 0.15-0.45), and FOCUS and Clone went with it
EVERY = 5            # control steps a start spans along a climb (rows: one a control step)
MARGIN = 25          # control steps (1 s) before the climb's success: the last start
MARGIN6 = 5          # ...family 6's (0.2 s): its last hand's catch and seat are its last second.
                     # gpu9 (122M) finished the -y climb from 34.2 s, its catch inside the rows,
                     # and the +y one 45 % from its last start: L in the air 0.42 m from its goal
                     # there, the catch ~0.6 s on -- no start, nothing cloned. Its mean reached
                     # t1b on time from 33-35 s, slipped off and went back to t1a, 0/9.
                     # Family 4's too: a step down is ~5-8 s, its second hand's seat its last second
BOOK = 100          # its episodes report family + BOOK
SEEN = 0.08          # m: a row is a start's only if each hook is seen at most this far from
                     # where it is when an episode begins there (`_seen`; rl.mjx_check task holds
                     # every start under 0.1 on noise draws of its own, 2 cm of them above this).
                     # The found climbs see 78 mm at most (family 6's 78, since a start has
                     # its hooks' rungs end to end: MjxClimb.start); the taught ones, fast, reach for a
                     # rung with one hook loaded off its seat or none (0.25 m x2: up to 187 mm
                     # off for 0.2 s), where a climb from the hang carries what it knew and a
                     # start knows nothing -- gpu5 learnt to finish on such a view (2026-10-09)
HILL_SWING = 5.0     # ...family 9's past STATIC9: a swing's pendulum is its hill, 3.5-4.4 m in the
                     # swings found at 0.70-0.90 each way round. Lesson 9's (gap 0.90,
STATIC9 = 0.65       # simplified robot): the left hand down to z 0.25 to swing (+1.6 m, the return
                     # -3.1 at its lowest) and caught 1.11 m from the right, then the right let go
                     # and down to z -0.38 while the left slid on (+2.5 m, -5.7); each paid back
                     # within ~6 s, the 0.995 discount's 8 s, and the climb ends +16.4. A crossing
                     # without a swing was found at 0.60 (0.58, rl.lessons' still-hang estimate)
HILL_MAX = 0.05      # m the live potential (both hands, what K_REACH pays) may rise above
                     # its lowest so far along a climb. The first climbs swung the free arm
                     # across the body up a 0.40-2.44 m hill of it before the reach paid;
                     # gpu3 (60M steps) never learnt that stretch from these starts -- going
                     # back to the start rung paid more -- and stayed at 0 % from the hang.


def wall(spacing, rows=1):
    """the wall a climb was found on, as a rl.walls layout: the start row and `rows` rows
    `spacing` up and OUT out each, all full length, goals on the top one straight above the
    hang -- rl.walls' family 3 with d = 0"""
    from .walls import _row, _front, X, Y0, Y1, Z0, HY, OUT, TIMEOUT
    z, x = Z0 + rows * spacing, X - rows * OUT
    rungs = [_row(Y0, Y1, Z0 + k * spacing, x=X - k * OUT) for k in range(rows + 1)]
    return dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
                walls=[_front()], goals=np.array([[x, HY, z], [x, -HY, z]]), goal_rungs=[rows],
                timeout=TIMEOUT, family=3, probe=None, probe_val=0.0)


def wall6(rise, side=1):
    """the wall family 6's climbs were found on (out/climb_demo/plan_chain.py, PLAN=f6): the
    start row ending 0.37 m out, one tier `rise` up of two short rungs, the first 0.47-0.87 m,
    the second 1.12-1.47 m, goals on their middles (R the first's, L the second's) -- rl.walls'
    family 6, one tier, scatter 0 -- mirrored (rl.walls.mirror) for `side` -1"""
    from .walls import _row, _front, mirror, X, Z0
    z = Z0 + rise
    rungs = [_row(-0.90, 0.37, Z0), _row(0.47, 0.87, z), _row(1.12, 1.47, z)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front()], goals=np.array([[X, 1.295, z], [X, 0.67, z]]), goal_rungs=[1, 2],
               timeout=TIMEOUT6, family=6, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def wall6x2(rise, side=1):
    """`wall6` with a second tier `rise` above the first, two short rungs on past its end (rl.walls'
    family 6 at tiers 2, its middle draws), the goals on the second -- mirrored for `side` -1"""
    from .walls import _row, _front, mirror, X, Z0
    z1, z2 = Z0 + rise, Z0 + 2 * rise
    rungs = [_row(-0.90, 0.37, Z0), _row(0.47, 0.87, z1), _row(1.12, 1.47, z1),
             _row(1.595, 1.97, z2), _row(2.22, 2.595, z2)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front()], goals=np.array([[X, 2.4075, z2], [X, 1.7825, z2]]), goal_rungs=[3, 4],
               timeout=TIMEOUT6, family=6, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def wall4(drop, side=1):
    """the wall family 4's step down is taught on (`teach4`): rl.walls' family 4 with one step
    `drop` down, GAP4 on past the start row's end, that end 0.21 m past the left hook and the
    goals at the middle of rl.walls' draws -- mirrored (rl.walls.mirror) for `side` -1"""
    from .walls import _row, _front, mirror, X, Z0, HY, SEAT, TIMEOUT
    y = HY + 0.21
    y0, z = y + GAP4, Z0 - drop
    yR = y0 + SEAT + 0.03
    yL = yR + 0.31
    rungs = [_row(-0.90, y, Z0), _row(y0, yL + SEAT + 0.08, z)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front()], goals=np.array([[X, yL, z], [X, yR, z]]), goal_rungs=[1],
               timeout=TIMEOUT, family=4, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def wall7(s, side=1):
    """the wall family 7's move is taught on (`teach7`): rl.walls' family 7 with the side wall `s`
    out, the start row to 1.2 m and the goals at the middle of rl.walls' draws -- mirrored
    (rl.walls.mirror) for `side` -1, the corner then toward +y"""
    from .walls import _row, _front, mirror, Z0, TIMEOUT
    ys = -s + 0.06
    rungs = [_row(-(s - 0.10), 1.2, Z0), (0.125, np.array([0.125, ys, Z0]), np.array([1.0, 0.0, 0.0]), 0.305)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front(), ('wall_side', np.array([0.125, -s, 1.0]), np.array([0.0, 1.0, 0.0]))],
               goals=np.array([[0.35, ys, Z0], [0.04, ys, Z0]]), goal_rungs=[1],
               timeout=TIMEOUT, family=7, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def wall8(behind, side=1):
    """the wall family 8's move was found on (plan_chain.py PLAN=f8): rl.walls' family 8 with d = 0,
    the back rung `behind` the start row at its height, both full length, the back wall 0.06 m
    behind it, goals straight behind the hang"""
    from .walls import _row, _front, mirror, X, Y0, Y1, Z0, HY, TIMEOUT
    xb = X - behind
    rungs = [_row(Y0, Y1, Z0), _row(Y0, Y1, Z0, x=xb)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front(), ('wall_back', np.array([xb - 0.06, 0.30, 1.0]), np.array([1.0, 0.0, 0.0]))],
               goals=np.array([[xb, HY, Z0], [xb, -HY, Z0]]), goal_rungs=[1],
               timeout=TIMEOUT, family=8, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def wall9(gap, side=1):
    """the wall family 9's crossing was found on (plan_chain.py f9): rl.walls' family 9, the start
    row from -0.52 to 0.21 m past the left hook, the far rung `gap` on, goals at the middle of
    _goals_from's draws -- mirrored for `side` -1"""
    from .walls import _row, _front, mirror, X, Z0, HY, SEAT, TIMEOUT
    e = HY + 0.21
    y0 = e + gap
    yR = y0 + SEAT + 0.03
    yL = yR + 0.31
    rungs = [_row(-0.52, e, Z0), _row(y0, yL + SEAT + 0.08, Z0)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front()], goals=np.array([[X, yL, Z0], [X, yR, Z0]]), goal_rungs=[1],
               timeout=TIMEOUT, family=9, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def wall5(brk, side=1):
    """the wall family 5's climb was found on (plan_chain.py f5): rl.walls' family 5, the start row
    ending 0.21 m past the left hook, the top row 0.425 m up from 0.125 m past it to 0.125 m short
    of the far row, the far row `brk` on, goals at the middle of _goals_from's draws"""
    from .walls import _row, _front, mirror, X, Z0, HY, SEAT, TIMEOUT
    e = HY + 0.21
    f0 = e + brk
    yR = f0 + SEAT + 0.03
    yL = yR + 0.31
    rungs = [_row(-0.90, e, Z0), _row(e + 0.125, f0 - 0.125, Z0 + 0.425), _row(f0, yL + SEAT + 0.08, Z0)]
    lay = dict(rungs=[(f'rung_g{i}', c, a, h, 0.015) for i, (_, c, a, h) in enumerate(rungs)],
               walls=[_front()], goals=np.array([[X, yL, Z0], [X, yR, Z0]]), goal_rungs=[2],
               timeout=TIMEOUT, family=5, probe=None, probe_val=0.0)
    return mirror(lay) if side < 0 else lay


def family(ref):
    """each climb's family (a fixture from before family 6's: all 3)"""
    return ref['climb_family'] if 'climb_family' in ref else np.full(len(ref['climb_spacing']), 3)


def found(ref):
    """the climbs a search found, replayed from their setpoints (CLIMBS, STEPS6/8/9/5), not a policy's
    own (TAUGHT*, family 6's two tiers): their actions are their rows' steps"""
    return np.r_[np.arange(len(CLIMBS)), np.nonzero(np.isin(family(ref), (5, 6, 8, 9)) & (ref['climb_rows'] == 1))[0]]


def timeout(f):
    """s an episode from one of family f's starts may run"""
    return TIMEOUT6 if f in LONG else TIMEOUT


def climb_wall(ref, c):
    """the wall climb `c` was found or taught on"""
    s = float(ref['climb_spacing'][c])
    if family(ref)[c] == 6:
        return (wall6x2 if ref['climb_rows'][c] == 2 else wall6)(s, int(ref['climb_side'][c]))
    if family(ref)[c] == 4:
        return wall4(s, int(ref['climb_side'][c]))
    if family(ref)[c] == 7:
        return wall7(s, int(ref['climb_side'][c]))
    if family(ref)[c] == 8:
        return wall8(s, int(ref['climb_side'][c]))
    if family(ref)[c] == 9:
        return wall9(s, int(ref['climb_side'][c]))
    if family(ref)[c] == 5:
        return wall5(s, int(ref['climb_side'][c]))
    return wall(s, int(ref['climb_rows'][c]))


def start_wall(ref, c, rng):
    """a wall an episode from one of climb c's starts runs on: family 3's any wall of its rows and
    spacing (its rows are full length: any goals and mirroring fit), family 6's its own"""
    f = int(family(ref)[c])
    if f == 3:
        from . import walls as W
        lay = W.sample(3, dict(rows=int(ref['climb_rows'][c]), spacing=float(ref['climb_spacing'][c])),
                       rng, ('rows', 'spacing'))
    else:
        lay = climb_wall(ref, c)
    return dict(lay, family=f + BOOK, probe=None, probe_val=0.0, timeout=timeout(f))


def teach(run, spacings=tuple(s for s, _ in TAUGHT), tries=64, seed=0):
    """record each TAUGHT climb, rl.walls' family 3 with 2 rows from the hang, as a trained
    policy (`run`) does it when it is shown no rung below the row both its hands hold.

    Why (gpu7, 2026-10-10, 173M steps): it climbed one row 24/24 and two 0/24, every time both
    hooks on the middle row to the timeout. From a hang there, the start row in its near-rung
    observation alone stopped it -- 0/16 as it is, 8-11/16 with that row out of the
    observation (still there in the physics), and 10/16 -> 0/16 with one added to the
    observation of a wall that has none; reordering the rungs did not help. In every climb it
    had finished, nothing was below. Shown the climb it makes without that row as a reference
    (rl.train_ppo.Clone), it is taught to make it with the row in view.
    What is cloned at each of its states is the teacher's MEAN action there (`target`), not the
    step it took: a sampled climb's steps are 4 sigma of noise each, and cloned they took gpu7's
    family 1 from 13/16 to 5/16 offline (out/gpu7_diag/bc_probe2b.log).
    -> {spacing: the attempts' successes}; the fastest success of each saved to TAUGHT, the
    mean's own if it has one"""
    import jax.numpy as jp
    from . import mjx_task as MT

    class Teacher(MT.MjxClimb):
        def _obs(self, T, st, sen, m=None):
            m = self._map(st)
            under = jp.arange(m.rung_valid.shape[0]) < st.hold_idx.min()    # rows below both hands
            return super()._obs(T, st, sen, m._replace(rung_valid=m.rung_valid & ~under))

    return {s: _record(run, Teacher(tries), dict(wall(s, 2), family=3, probe=None, probe_val=0.0),
                       tries, seed, TAUGHT[(s, 2)], f"{s:.2f} m, 2 rows") for s in spacings}


def teach4(run, tries=32, seed=0):
    """record each TAUGHT4 step down, rl.walls' family 4 from the hang on `wall4`, as a trained
    policy (`run`, one that steps down both ways: gpu8's) does it -> {(drop, side): successes}"""
    from . import mjx_task as MT
    return {k: _record(run, MT.MjxClimb(tries), wall4(*k), tries, seed, path,
                       f"family 4's {k[0]:.2f} m step down{' the other way' if k[1] < 0 else ''}")
            for k, path in TAUGHT4.items()}


def teach7(run, tries=32, seed=0):
    """record each TAUGHT7 move onto the side rung, rl.walls' family 7 from the hang on `wall7`, as a
    trained policy (`run`, one that turns the corner both ways: gpu10's) does it -> {(side, way): successes}"""
    from . import mjx_task as MT
    return {k: _record(run, MT.MjxClimb(tries), wall7(*k), tries, seed, path,
                       f"family 7's {k[0]:.2f} m corner{' the other way' if k[1] < 0 else ''}")
            for k, path in TAUGHT7.items()}


def mirror_joints(rows):
    """joint rows (rl.estimator.JOINTS' order: qpos[7:], qvel[6:], ctrl, actions) y -> -y: each arm's
    joints the other's, those turning about an axis that mirrors onto its own reverse negated
    (out/climb_demo/mirror_fixture.py; the simplified robot is its own mirror image)"""
    from .estimator import JOINTS
    neg = ('waist_yaw', 'abduction', 'rotation', 'roll', 'deviation')
    rows = np.asarray(rows, float)
    out = np.empty_like(rows)
    for i, n in enumerate(JOINTS):
        src = n.replace('_L', '_X').replace('_R', '_L').replace('_X', '_R')
        out[..., i] = rows[..., JOINTS.index(src)] * (-1.0 if any(k in n for k in neg) else 1.0)
    return out


def mirror_taught(src, dst):
    """a recorded climb (`_record`'s npz) y -> -y, the climb on the mirrored wall: the floating base
    reflected (position y, quaternion (w, -x, y, -z), angular velocity -x, -z: a pseudo-vector),
    the joints by `mirror_joints`, the hands' holds swapped"""
    z = dict(np.load(src))
    q, v = z['qpos'].copy(), z['qvel'].copy()
    q[:, 1] *= -1
    q[:, 4] *= -1
    q[:, 6] *= -1
    q[:, 7:] = mirror_joints(q[:, 7:])
    v[:, 1] *= -1
    v[:, 3] *= -1
    v[:, 5] *= -1
    v[:, 6:] = mirror_joints(v[:, 6:])
    np.savez_compressed(dst, **dict(z, qpos=q, qvel=v, ctrl=mirror_joints(z['ctrl']), grabs=z['grabs'][:, ::-1],
                                    action=mirror_joints(z['action']), target=mirror_joints(z['target']),
                                    run=f"{z['run']} mirrored"))


def teach6(run, tries=64, seed=0):
    """record each TAUGHT6 climb, rl.walls' family 6 with two tiers from the hang on `wall6x2`, as a
    trained policy (`run`) does it when it is shown tier 1's last two rungs as its goals until both
    its hands are off the start row, the real goals after.

    Why (gpu11, 2026-10-14, 226M steps): it climbed one tier (19/24 at 210M) and on two never left
    the start row, 0/16, hooks never 8 cm up. The same walls with the goals on tier 1 instead, 8/16;
    with tier 2 out of its observation, 0/16 still; with the goals on tier 1 until both hands are on
    it, 3/16 up both tiers. Its goals there are 2.7-3.3 m off and 0.6 m up, where no wall it learnt
    from had one (family 4's: 2.6 m, down) -> {(rise, side): successes}"""
    import jax.numpy as jp
    from . import mjx_task as MT

    class Teacher(MT.MjxClimb):
        def _obs(self, T, st, sen, m=None):
            m = self._map(st)
            z, n = st.wall.slot_pos[:, 2], st.wall.slot_pos.shape[0]
            t1 = st.wall.rung_valid & (jp.abs(z - z[1]) < 0.1) & (jp.arange(n) >= 1)    # tier 1
            b = jp.max(jp.where(t1, jp.arange(n), 0))
            ca, cb = m.rung_c[b - 1], m.rung_c[b]
            way = jp.where(cb[1] > ca[1], jp.stack([cb, ca]), jp.stack([ca, cb]))   # rl.walls' L, R
            goals = jp.where(st.hold_idx.min() >= 1, m.goals, way)
            return super()._obs(T, st, sen, m._replace(goals=goals))

    # side -1 is side 1's mirror image: shown the same way round, the teacher reached tier 1 28/32
    # and tier 2 never (0/64), +y 12/64
    won = {k: _record(run, Teacher(tries), wall6x2(*k), tries, seed, path, f"family 6's {k[0]:.2f} m, two tiers")
           for k, path in TAUGHT6.items() if k[1] > 0}
    for (r, side), path in TAUGHT6.items():
        if side < 0 and os.path.exists(TAUGHT6[(r, 1)]):
            mirror_taught(TAUGHT6[(r, 1)], path)
            print(f"rsi.teach: -> {path}, the {r:.2f} m two tiers mirrored")
    return won


def teach3(run, tries=64, seed=0):
    """record each TAUGHT3 climb, rl.walls' family 3 with 2 rows from the hang, as a trained policy
    (`run`) does it with nothing hidden -> {(spacing, rows): successes}"""
    from . import mjx_task as MT
    return {k: _record(run, MT.MjxClimb(tries), dict(wall(*k), family=3, probe=None, probe_val=0.0),
                       tries, seed, path, f"{k[0]:.2f} m, {k[1]} rows")
            for k, path in TAUGHT3.items()}


def _record(run, task, lay, tries, seed, path, what):
    """`tries` episodes of `run`'s policy on the wall `lay` from the hang on `task`, half its mean
    and half sampled, every state recorded -> {try: steps to its success}; the fastest of the
    mean's (else of any: a sampled climb's steps carry its noise) saved to `path`"""
    import pickle
    import jax
    import jax.numpy as jp
    from stable_baselines3 import PPO
    from . import mjx_task as MT
    model = PPO.load(os.path.join(run, 'model.zip'), device='cpu')
    with open(os.path.join(run, 'vecnorm.pkl'), 'rb') as f:
        vn = pickle.load(f)
    L = len(MT.LESSON_KEYS)
    b = MT.Batch(tries, bank=tries, task=task)
    b.set_bank([lay] * tries)
    b.state, o = b._reset(b.task.tables, jp.asarray(L + np.arange(tries)),
                          jax.random.split(jax.random.PRNGKey(seed), tries))
    o, det = np.asarray(o), np.arange(tries) % 2 == 0          # half the mean, half sampled
    rec, won, done = {k: [] for k in ('qpos', 'qvel', 'ctrl', 'grabs', 'action', 'target')}, {}, np.zeros(tries, bool)
    for n in range(int(1.01 * lay['timeout'] / 0.04)):
        st = b.state
        mean = model.predict(vn.normalize_obs(o), deterministic=True)[0]
        a = np.where(det[:, None], mean, model.predict(vn.normalize_obs(o), deterministic=False)[0])
        for k, v in (('qpos', st.d.qpos), ('qvel', st.d.qvel), ('ctrl', st.ctrl), ('grabs', st.grabs),
                     ('action', a), ('target', mean)):
            rec[k].append(np.asarray(v))
        o, _, _, te, tr, info = b.step(a)
        o, end = np.asarray(o), np.asarray(te | tr) & ~done
        for i in np.nonzero(end)[0]:
            if MT.REASONS[int(info['reason'][i])] == 'success':
                won[int(i)] = n + 1
        done |= end
        if done.all():
            break
    print(f"rsi.teach: {what}: {len(won)}/{tries} succeed "
          f"({sum(i % 2 == 0 for i in won)}/{tries // 2} the mean), steps {sorted(won.values())}")
    if won:
        mine = {i: v for i, v in won.items() if det[i]} or won
        i = min(mine, key=mine.get)
        np.savez_compressed(path, won=won[i], run=run, **{k: np.stack(v)[:won[i], i] for k, v in rec.items()})
        print(f"rsi.teach: -> {path}, {won[i]} steps ({won[i] * 0.04:.1f} s)")
    return won


def load():
    """{spacing, rows, qpos, qvel, ctrl, grabs, action, start, climb, startable}, one row a
    control step, `start` the one it is booked as, `action` the climb's own step from it and
    `startable` whether an episode may begin there (SEEN); start_climb, one
    a start; climb_spacing and climb_rows, one a climb. None before
    `python -m rl.rsi`"""
    if not os.path.exists(FILE):
        return None
    z = np.load(FILE)
    return {k: z[k] for k in z.files}


def signature(ref):
    """the fixture's starts and their states, hashed: a run's books are by start number, and
    hold only for the fixture they were kept on (train_ppo.Walls drops them otherwise)"""
    import hashlib
    return hashlib.sha1(b''.join(np.ascontiguousarray(ref[k]).tobytes()
                                 for k in ('start', 'climb', 'qpos'))).hexdigest()[:16]


def kept(ref, sig):
    """how many of `ref`'s starts a run's books, kept on the fixture signed `sig`, still hold for:
    all of that fixture's, when it is the beginning of this one (climbs are only ever added after
    the others, so theirs keep their numbers: families 8 and 9 joined gpu10 at 218.8M), else 0"""
    ends = np.r_[np.nonzero(np.diff(ref['climb']))[0] + 1, len(ref['climb'])]
    for n in ends[::-1]:
        if signature({k: ref[k][:n] for k in ('start', 'climb', 'qpos')}) == sig:
            return int(ref['start'][n - 1]) + 1
    return 0


def label(ref, c):
    """climb `c`'s name in the log: '0.2 m', or '0.2 m x2' over two rows; family 6's 'f6 0.3 m',
    'f6 0.3 m -y' the other way round (family 4's 'f4 ...' the same; family 7's own way is toward
    -y, so its other way round is 'f7 0.6 m +y')"""
    s, r = float(ref['climb_spacing'][c]), int(ref['climb_rows'][c])
    if family(ref)[c] == 7:
        return f"f7 {s:g} m" + (' +y' if ref['climb_side'][c] < 0 else '')
    if family(ref)[c] in (4, 5, 6, 8, 9):
        return f"f{family(ref)[c]} {s:g} m" + (f" x{r}" if r > 1 else '') + (' -y' if ref['climb_side'][c] < 0 else '')
    return f"{s:g} m" + (f" x{r}" if r > 1 else '')


def allowed(ref, tops):
    """the starts of the climbs the tops `tops` ({family: its tops}, the families in the pool)
    let in, each with a row an episode may begin from (`startable`): a family's climbs no higher
    than its top (TOP). By height only: a family-3 climb over more rows is offered as soon as
    its spacing is, so the policy meets ADR's next row ready -- gpu7 met it at 28M with nothing
    to learn it from, and once ADR may step a count back (rl.walls.draw), a top of one row
    would take the references away in the middle of learning them"""
    c, some = ref['start_climb'], np.bincount(ref['start'], weights=ref['startable']) > 0
    lim = np.array([tops[f][TOP[f]] if f in tops else -np.inf for f in family(ref)])[c]
    return np.nonzero((ref['climb_spacing'][c] <= lim + 1e-9) & some)[0]


def unpassed(books, ref, tops):
    """the startable fixture rows of every start not yet passed in `books`, over the climbs
    the tops `tops` let in: where the policy does not finish the climb (rl.train_ppo.Walls.learning)"""
    bad = [int(j) for j in allowed(ref, tops) if not passed(books.get(int(j), []))]
    return np.nonzero(np.isin(ref['start'], bad) & ref['startable'])[0]


def cloned(books, ref, tops):
    """the startable fixture rows of the starts at each climb's frontier and the BACK (BACK6, LONG
    families) before and after it not yet passed: where the policy learns its climbs now, the window `weights`
    focuses on (rl.train_ppo.Clone). Not every start not passed (`unpassed`): a family that has just
    entered has all of them, family 5's 2786 rows in gpu11 at 258M, its references' actions 2.8
    sigma off the policy's mean, and in 12M steps families 8, 3, 4 and 6 fell 93 -> 2, 70 -> 34,
    80 -> 59, 73 -> 46 % to a rollback. Family 6's ~2000 at 33M took 3 and 4 down 20-30 points;
    13-75 rows (200-250M) took nothing; gpu6's catch was 60 rows behind its frontier"""
    ok, sc, fam = allowed(ref, tops), ref['start_climb'], family(ref)
    hot = []
    for c, f in frontier(books, sc, ok).items():
        if f is None:
            continue
        back = BACK6 if fam[c] in LONG else BACK
        hot += [int(j) for j in ok[(sc[ok] == c) & (ok <= f + back) & (ok >= f - back)] if not passed(books.get(int(j), []))]
    return np.nonzero(np.isin(ref['start'], hot) & ref['startable'])[0]


def passed(h):
    """a start's book (its episodes' successes, oldest first): passed?"""
    h = h[-START_WINDOW:]
    return len(h) >= START_MIN and sum(h) >= PASS * len(h)


def slipped(h):
    """a start's book: failing for the frontier -- booked fewer than START_MIN times, or under SLIP.
    Between SLIP and PASS a start is not passed (`unpassed`, `cloned`) but does not hold the frontier"""
    h = h[-START_WINDOW:]
    return len(h) < START_MIN or sum(h) < SLIP * len(h)


def frontier(books, climb, ok=None):
    """{climb: the latest start along it (its index) `slipped` in `books` ({start: [success,
    ...]}), None once none has}, over the starts `ok`; `climb`: each start's.
    The latest, not the earliest: one that slips behind the frontier is drawn first again"""
    ok = np.arange(len(climb)) if ok is None else np.asarray(ok)
    out = {}
    for c in np.unique(climb[ok]):
        bad = [int(j) for j in ok[climb[ok] == c] if slipped(books.get(int(j), []))]
        out[int(c)] = max(bad) if bad else None
    return out


def where(books, tops):
    """{climb's label: s into it of `frontier`'s start, None once every start has passed},
    over the climbs the tops `tops` let in -- for the log and rl.status"""
    ref = load()
    if ref is None:
        return {}
    sc = ref['start_climb']
    return {label(ref, c): None if f is None else round((f - int(np.argmax(sc == c))) * EVERY * 0.04, 2)
            for c, f in frontier(books, sc, allowed(ref, tops)).items()}   # 0.04 s: climb_task.CONTROL_DT


def weights(books, climb, ok, back=BACK):
    """draw probabilities over the starts `ok`: FOCUS evenly over the starts to learn from
    -- each climb's `back` + 1 ending at its frontier (and the ones up to `back` after it not
    passed), and every start booked fewer than
    START_MIN times -- and the rest evenly over every start"""
    ok = np.asarray(ok)
    hot = np.array([len(books.get(int(j), [])) < START_MIN for j in ok])
    learnt = np.array([passed(books.get(int(j), [])) for j in ok])
    for f in frontier(books, climb, ok).values():
        if f is not None:       # ...and the starts after it not yet passed, between SLIP and PASS
            hot |= (climb[ok] == climb[f]) & (ok >= f - back) & ((ok <= f) | ((ok <= f + back) & ~learnt))
    w = np.full(len(ok), (1.0 - FOCUS) / len(ok))
    w[hot] += FOCUS / max(hot.sum(), 1)
    return w / w.sum()


def mix(layouts, rng, tops, share=SHARE, books=None):
    """`layouts` (rl.walls.draw's) with `share` ({family: share}, or one for all) of each
    family's walls started partway up one of its climbs: a start of a climb the tops `tops`
    let in, drawn toward the frontiers in `books` (`weights`) among the family's own, from any
    of its rows, on a wall it fits (`start_wall`) cut to `timeout`, booked as family + BOOK
    and probing nothing"""
    ref = load()
    if ref is None:
        return layouts
    from .lessons import LESSONS
    L = len(LESSONS)                     # rl.mjx_task.LESSON_KEYS: the rows follow them
    sc, ok = ref['start_climb'], allowed(ref, tops)
    fam = family(ref)[sc[ok]]
    ok = {int(f): ok[fam == f] for f in np.unique(fam)}
    p = {f: weights(books or {}, sc, o, BACK6 if f in LONG else BACK) for f, o in ok.items()}
    out = []
    for lay in layouts:
        f = lay.get('family')
        if f in ok and rng.random() < (share.get(f, 0.0) if isinstance(share, dict) else share):
            j = int(rng.choice(ok[f], p=p[f]))
            row = int(rng.choice(np.nonzero((ref['start'] == j) & ref['startable'])[0]))
            lay = dict(start_wall(ref, sc[j], rng), d0_row=L + row)
        out.append(lay)
    return out


def _replay(spacing, path):
    """the family-3 climb through ClimbTask.step on the simplified robot -> (states, success step)"""
    from .lessons import _ry
    lay = wall(spacing)
    (_, c1, _, _, _) = lay['rungs'][1]
    return _replay_on(3, [_ry('start', -0.90, 1.50, 1.00), _ry('up1', -0.90, 1.50, float(c1[2]), x=float(c1[0]))],
                      ['rung_up1'], lay['goals'], path, f"the {spacing} m climb")


def _replay6(rise, side, path):
    """family 6's climb (`wall6`) through ClimbTask.step on the simplified robot, on lesson 6
    with its rungs as plan_chain.py's PLAN=f6 / f6m names them"""
    from .lessons import _ry
    lay = wall6(rise, side)
    rungs = [_ry(n, float(c[1] - h), float(c[1] + h), float(c[2]), x=float(c[0]))
             for n, (_, c, _, h, _) in zip(('start', 't1a', 't1b'), lay['rungs'])]
    return _replay_on(6, rungs, ['rung_t1a', 'rung_t1b'], lay['goals'], path,
                      f"family 6's {rise} m climb{' the other way' if side < 0 else ''}")


def _replay8(behind, side, path):
    """family 8's move (`wall8`) on lesson 8 with its rungs and back wall as plan_chain.py's PLAN=f8"""
    from .lessons import _ry, WALL_FRONT
    lay = wall8(behind, side)
    rungs = [_ry(n, float(c[1] - h), float(c[1] + h), float(c[2]), x=float(c[0]))
             for n, (_, c, _, h, _) in zip(('start', 'back'), lay['rungs'])]
    back = (f'    <geom name="wall_back"  type="box" material="wall" '
            f'pos="{float(lay["walls"][1][1][0]) - 0.05:.4f} 0.00 0.50" size="0.05 4.00 3.00"/>')
    return _replay_on(8, rungs, ['rung_back'], lay['goals'], path, f"family 8's {behind} m move back",
                      walls=[WALL_FRONT, back])


def _replay9(gap, side, path):
    """family 9's crossing (`wall9`) on lesson 9 with its rungs as plan_chain.py's PLAN=f9g / f9mg"""
    from .lessons import _ry
    lay = wall9(gap, side)
    rungs = [_ry(n, float(c[1] - h), float(c[1] + h), float(c[2]), x=float(c[0]))
             for n, (_, c, _, h, _) in zip(('start', 'far'), lay['rungs'])]
    return _replay_on(9, rungs, ['rung_far'], lay['goals'], path,
                      f"family 9's {gap} m crossing{' the other way' if side < 0 else ''}")


def _replay5(brk, side, path):
    """family 5's climb (`wall5`) on lesson 5 with its rungs as plan_chain.py's PLAN=f5b / f5mb"""
    from .lessons import _ry
    lay = wall5(brk, side)
    rungs = [_ry(n, float(c[1] - h), float(c[1] + h), float(c[2]), x=float(c[0]))
             for n, (_, c, _, h, _) in zip(('start', 'top', 'far'), lay['rungs'])]
    return _replay_on(5, rungs, ['rung_far'], lay['goals'], path,
                      f"family 5's {brk} m climb over{' the other way' if side < 0 else ''}")


def _replay_on(lesson, rungs, goal_rungs, goals, path, what, walls=None):
    """a found climb (`path`: its setpoints) through ClimbTask.step on the simplified robot, on
    lesson `lesson` with those rungs and goals (and `walls`, else the lesson's) -> (states, success
    step, live potential's hill)"""
    import tempfile
    from . import lessons
    from .climb_task import ClimbTask, ACT_SCALE
    from .mjx_scene import lesson_layout, build
    saved = lessons.LESSONS[lesson]
    lessons.LESSONS[lesson] = dict(saved, rungs=rungs, goal_rungs=goal_rungs,
                                   goals=np.asarray(goals).tolist(), timeout=90.0,
                                   **({'walls': walls} if walls else {}))
    try:
        ll = lesson_layout(lesson)
        ll['lesson'] = lesson
        with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False) as f:
            f.write(build(ll).to_xml())
        t = ClimbTask(lesson, visual=False, xml=f.name)
        t.rng = np.random.default_rng(0)
        t.reset()
    finally:
        lessons.LESSONS[lesson] = saved
        os.remove(f.name)
    z = np.load(path)
    assert np.allclose(t.ctrl, z['ctrl0'], atol=1e-6), "the climb does not start from this task's hang"
    d, out, won = t.env.data, [], None
    cmin, hill = float(t._live().sum()), 0.0
    for k, c in enumerate(z['ctrl']):
        a = np.clip((c - t.ctrl) / ACT_SCALE, -1, 1)
        out.append(dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=t.ctrl.copy(),
                        grabs=t.grabs.copy(), action=a))
        _, _, te, tr, info = t.step(a)
        hill = max(hill, info['live'] - cmin)
        cmin = min(cmin, info['live'])
        if te or tr:
            won = k + 1 if info.get('reason') == 'success' else None
            assert won, f"{what} ended '{info.get('reason')}' on replay"
            break
    assert won, f"{what} did not succeed on replay"
    return out, won, hill


def _seen(z):
    """each row's farther hook from its estimate at an episode's first sense there, on its
    climb's wall (rl.mjx_task's start: the estimator from the hooks' holds and the torso)"""
    import jax
    import jax.numpy as jp
    from . import mjx_task as MT
    n, L, out = 256, len(MT.LESSON_KEYS), []
    b = MT.Batch(n, bank=n)
    thr = jax.jit(jax.vmap(b.task._throat_true))
    walls = [climb_wall(z, c) for c in range(len(z['climb_spacing']))]
    fam = family(z)
    js = np.arange(len(z['climb']))
    for i in range(0, len(js), n):
        part = js[i:i + n]
        pad = np.r_[part, np.full(n - len(part), part[-1])]
        b.set_bank([dict(walls[z['climb'][j]], family=int(fam[z['climb'][j]]) + BOOK, d0_row=L + int(j)) for j in pad])
        st, _ = b._reset(b.task.tables, jp.asarray(L + np.arange(n)), jax.random.split(jax.random.PRNGKey(i), n))
        out.append(np.linalg.norm(np.asarray(st.est_thr) - np.asarray(thr(st.d)), axis=2).max(1)[:len(part)])
    return np.concatenate(out)


def build_fixture():
    rows, j0, climbs = [], 0, []
    for s, path in sorted(CLIMBS.items()):
        states, won, hill = _replay(s, path)
        keep = states[:won - MARGIN]
        rows += [dict(st, spacing=s, rows=1, start=j0 + k // EVERY, climb=len(climbs)) for k, st in enumerate(keep)]
        climbs.append((s, 1))
        j0 = rows[-1]['start'] + 1
        print(f"rsi: the {s:.2f} m climb succeeds at step {won} ({won * 0.04:.1f} s); "
              f"{len(keep)} rows to {MARGIN} steps before it, a start every {EVERY}; "
              f"holds taken by the last: {keep[-1]['grabs']}; live potential's hill {hill * 1000:.0f} mm")
        assert hill <= HILL_MAX, (f"the {s:.2f} m climb climbs {hill:.2f} m of live potential before "
                                  f"it pays (HILL_MAX {HILL_MAX}): a policy is paid to go back from its starts")
    # after the found ones, so their starts keep their numbers (a run's books are by start).
    # Recorded in MJX by the policy itself, no replay; no hill test: the policy's own climb
    for (s, r), path in sorted(TAUGHT.items()):
        if not os.path.exists(path):
            print(f"rsi: no {path} (python -m rl.rsi --teach RUN): the {s:.2f} m climb over {r} rows left out")
            continue
        z = np.load(path)
        won = int(z['won'])
        rows += [dict(qpos=z['qpos'][k], qvel=z['qvel'][k], ctrl=z['ctrl'][k], grabs=z['grabs'][k],
                      action=z['target'][k], spacing=s, rows=r, start=j0 + k // EVERY, climb=len(climbs))
                 for k in range(won - MARGIN)]          # `action`: the teacher's mean there (`teach`)
        climbs.append((s, r))
        j0 = rows[-1]['start'] + 1
        print(f"rsi: the {s:.2f} m climb over {r} rows ({z['run']}) succeeds at step {won} ({won * 0.04:.1f} s); "
              f"{won - MARGIN} rows; holds taken by the last: {z['grabs'][won - MARGIN - 1]}")
    # family 6's after both, the same way: family 3's starts keep their numbers
    for (rise, side), path in sorted(STEPS6.items()):
        if not os.path.exists(path):
            print(f"rsi: no {path}: family 6's {rise:.2f} m climb (side {side}) left out")
            continue
        states, won, hill = _replay6(rise, side, path)
        keep = states[:won - MARGIN6]
        rows += [dict(st, spacing=rise, rows=1, start=j0 + k // EVERY, climb=len(climbs)) for k, st in enumerate(keep)]
        climbs.append((rise, 1, 6, side))
        j0 = rows[-1]['start'] + 1
        print(f"rsi: family 6's {rise:.2f} m climb (side {side:+d}) succeeds at step {won} ({won * 0.04:.1f} s); "
              f"{len(keep)} rows; holds taken by the last: {keep[-1]['grabs']}; live potential's hill {hill * 1000:.0f} mm")
        assert hill <= HILL_MAX, (f"family 6's {rise:.2f} m climb climbs {hill:.2f} m of live potential before "
                                  f"it pays (HILL_MAX {HILL_MAX})")
    def taught(fam, table, flag, n_rows=1):
        """a policy's own climbs (`table`: teach3's, 4's, 6's, 7's), as TAUGHT's: its mean action the
        reference. Family 3's are keyed (spacing, rows), the others (height, side), over `n_rows`"""
        nonlocal rows, j0
        margin = MARGIN if fam == 3 else MARGIN6
        for (h, k2), path in sorted(table.items()):
            r, side = (k2, 0) if fam == 3 else (n_rows, k2)
            if not os.path.exists(path):
                print(f"rsi: no {path} (python -m rl.rsi {flag} RUN): family {fam}'s {h:.2f} m ({k2}) left out")
                continue
            z = np.load(path)
            won = int(z['won'])
            rows += [dict(qpos=z['qpos'][k], qvel=z['qvel'][k], ctrl=z['ctrl'][k], grabs=z['grabs'][k],
                          action=z['target'][k], spacing=h, rows=r, start=j0 + k // EVERY, climb=len(climbs))
                     for k in range(won - margin)]
            climbs.append((h, r, fam, side))
            j0 = rows[-1]['start'] + 1
            print(f"rsi: family {fam}'s {h:.2f} m ({k2}, {z['run']}) succeeds at step {won} "
                  f"({won * 0.04:.1f} s); {won - margin} rows; holds taken by the last: {z['grabs'][won - margin - 1]}")

    # family 4's after those
    taught(4, TAUGHT4, '--teach4')
    # families 8's, 9's and 5's found ones after all of those, as family 6's
    def replayed(fam, steps, replay):
        nonlocal rows, j0
        for (h, side), path in sorted(steps.items()):
            if not os.path.exists(path):
                print(f"rsi: no {path}: family {fam}'s {h:.2f} m climb (side {side}) left out")
                continue
            states, won, hill = replay(h, side, path)
            keep = states[:won - MARGIN6]
            rows += [dict(st, spacing=h, rows=1, start=j0 + k // EVERY, climb=len(climbs)) for k, st in enumerate(keep)]
            climbs.append((h, 1, fam, side))
            j0 = rows[-1]['start'] + 1
            print(f"rsi: family {fam}'s {h:.2f} m climb (side {side:+d}) succeeds at step {won} ({won * 0.04:.1f} s); "
                  f"{len(keep)} rows; holds taken by the last: {keep[-1]['grabs']}; live potential's hill {hill * 1000:.0f} mm")
            top = HILL_SWING if fam == 9 and h > STATIC9 else HILL_MAX
            assert hill <= top, (f"family {fam}'s {h:.2f} m climb climbs {hill:.2f} m of live potential "
                                 f"before it pays (limit {top})")

    for fam, steps, replay in ((8, STEPS8, _replay8), (9, STEPS9, _replay9), (5, STEPS5, _replay5)):
        replayed(fam, steps, replay)
    # family 7's, family 3's wider two rows, family 6's two tiers and family 8's farther back rungs
    # last of all (2026-10-14):
    # every start before keeps its number, so a run's books on the fixture before are kept (`kept`)
    taught(7, TAUGHT7, '--teach7')
    taught(3, TAUGHT3, '--teach3')
    taught(6, TAUGHT6, '--teach6', n_rows=2)
    replayed(8, STEPS8B, _replay8)                   # family 8's past 0.50
    climbs = [c if len(c) == 4 else c + (3, 0) for c in climbs]     # (height, rows, family, side)
    z = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    first = np.unique(z['start'], return_index=True)[1]
    z['start_climb'] = z['climb'][first]
    z['climb_spacing'], z['climb_rows'] = np.array([c[0] for c in climbs]), np.array([c[1] for c in climbs])
    z['climb_family'], z['climb_side'] = np.array([c[2] for c in climbs]), np.array([c[3] for c in climbs])
    # on disk first: _seen's batch (rl.mjx_task's tables) begins its episodes from the FILE's rows.
    # Measured before it was written, a row new to the fixture was measured at a stale file's row
    z['startable'] = np.ones(len(rows), bool)
    np.savez_compressed(FILE, **z)
    err = _seen(z)
    z['startable'] = err < SEEN
    for c in range(len(climbs)):
        e = err[z['climb'] == c]
        print(f"rsi: the {label(z, c)} climb: hooks seen at most {e.max() * 1000:.0f} mm off at a start; "
              f"{int((e >= SEEN).sum())} of its {len(e)} rows past SEEN, no episode begins there")
    np.savez_compressed(FILE, **z)
    print(f"rsi: {len(rows)} rows, {j0} starts, {len(climbs)} climbs -> {FILE}")


if __name__ == '__main__':
    import sys
    if '--teach' in sys.argv:            # python -m rl.rsi --teach out/<run>: TAUGHT first
        teach(sys.argv[sys.argv.index('--teach') + 1])
    if '--teach4' in sys.argv:           # python -m rl.rsi --teach4 out/<run>: TAUGHT4 first
        teach4(sys.argv[sys.argv.index('--teach4') + 1])
    if '--teach7' in sys.argv:           # python -m rl.rsi --teach7 out/<run>: TAUGHT7 first
        teach7(sys.argv[sys.argv.index('--teach7') + 1])
    if '--teach3' in sys.argv:           # python -m rl.rsi --teach3 out/<run>: TAUGHT3 first
        teach3(sys.argv[sys.argv.index('--teach3') + 1])
    if '--teach6' in sys.argv:           # python -m rl.rsi --teach6 out/<run>: TAUGHT6 first
        teach6(sys.argv[sys.argv.index('--teach6') + 1])
    build_fixture()
    from .climb_task import ACT_SCALE
    z = load()
    assert z['qpos'].shape[0] == z['spacing'].shape[0] == z['start'].shape[0] == z['action'].shape[0] \
        and np.isfinite(z['qpos']).all()
    assert (np.diff(z['start']) >= 0).all() and np.bincount(z['start']).max() == EVERY
    assert (z['start_climb'][z['start']] == z['climb']).all(), "a start spans two climbs"
    assert z['grabs'].max() >= 1, "no start has a hold taken: the starts never reach the row above"
    assert (np.bincount(z['start'], weights=z['startable']) > 0).mean() > 0.9, "most starts have no row to begin from"
    for c in found(z):                                                  # a found climb's action is
        j = np.nonzero(z['climb'] == c)[0]                              # the step to its next row
        err = np.abs(z['ctrl'][j[1:]] - z['ctrl'][j[:-1]] - ACT_SCALE * z['action'][j[:-1]]).max()
        assert err < 1e-9, f"climb {label(z, c)}: its actions are not its rows' steps ({err:g})"
    assert np.isfinite(z['action']).all() and np.abs(z['action']).max() <= 1.0, "an action outside [-1, 1]"
    print("rsi: OK")
