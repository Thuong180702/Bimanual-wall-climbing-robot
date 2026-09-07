# PPO curriculum

The frame for training PPO on the climbing tasks, as twelve lessons ordered by
difficulty instead of the mixed scene.

```bash
python -m rl.lessons        # regenerate urdf/lessons/*.xml + smoke-check each
python -m rl.climb_task     # smoke-check the task API on every lesson
python -m rl.bench          # where the time actually goes on this machine
python -m rl.train_ppo --self-check          # curriculum logic, trains nothing

python -m rl.train_ppo --name climber        # train: all 12, curriculum on
# Ctrl-C any time, then:
python -m rl.train_ppo --name climber --resume

# from a second terminal, while it trains:
python -m rl.status --name climber --watch
```

`torch`, `gymnasium` and `stable-baselines3` are installed in this repo's `.venv`
(torch 2.13 / SB3 2.9 on Python 3.14 — no separate venv needed).

| file | what it is |
|---|---|
| `lessons.py` | the 12 lesson specs + scene generator (writes `urdf/lessons/`) |
| `climb_task.py` | Gymnasium-shaped env: obs / action / reward / termination |
| `train_ppo.py` | thin SB3 PPO entry point |
| `bench.py` | simulator throughput vs env count, and cpu vs mps for the policy |
| `render.py` | PNGs of the lesson scenes → `out/scenes/` (ffmpeg writes them, no image dep) |
| `status.py` | what a run is doing, read from its checkpoint — safe to run while it trains |
| `eval.py` | score a checkpoint per lesson against a do-nothing control |

## Lessons

−2. left hand, one step (0.30 m) → −1. right hand, one step (0.30 m, the other way)
→ 0. both hands, one step (0.40 m) → 1. traverse, continuous rung → 2. traverse,
gaps → 3. climb up → 4. up-over-down → 5. climb down → 6. staggered heights →
7. perpendicular wall → 8. wall behind the back → 9. row beyond static reach (needs
a dynamic move).

Lessons −2, −1 and 0 are one lift-and-place cycle. They exist because lesson 1 —
five such cycles per hand — stopped being findable from scratch once dragging and
jumping stopped paying; the `win5` section below has the numbers, and why the step
is one hand at a time first, each hand in turn, and 0.40 m rather than 0.30 after.

Every lesson keeps the robot, sensors, actuators and the verified hanging keyframe
of `urdf/scene.xml` verbatim and swaps only walls + rungs, so all start states are
the same known hang.

## One policy, not nine

The lessons are a curriculum, not nine separate jobs. **One shared policy trains on
all of them mixed**, and the curriculum only controls which lessons are in the
sampling pool — nothing ever leaves it.

Why not the two alternatives:

- **Nine specialists + a dispatcher** trains easily and isolates failures, but the
  nine share almost everything that is hard — hanging without falling, moving weight
  onto a hand, seating a hook in a ±25 mm window — and each would learn it again
  from scratch. It also leaves the seams untrained: the posture a traverse ends in
  is not in the climb specialist's training distribution.
- **A meta-policy calling nine sub-policies** (hierarchical RL) would replace
  something that already works — `controller/holds.py` picks the next hold with A\*
  over the wall map, deterministically, today. Learning that choice is harder than
  the problem it solves.

- **One policy over all lessons** needs no dispatcher, because the goal is already
  in the observation: it is *goal-conditioned*, not lesson-conditioned, and at
  deploy time the classical planner supplies the goal exactly as a lesson spec does.

**Lessons are mixed, never sequential.** Fine-tuning lesson 1 → 2 → … → 9 in order
is the obvious reading of "curriculum" and it is the one thing to avoid: training on
lesson 9 erases lesson 1 (catastrophic forgetting). Each env draws a fresh lesson
from the active pool on every reset instead, and a lesson unlocks when the newest
one clears 60 % success over 20+ episodes. `--self-check` asserts that a lesson
never leaves the pool.

## Pausing and resuming

`Ctrl-C` (or `SIGTERM`) saves and exits 0. A checkpoint is three files in
`out/<name>/`, always written together — a model restored against stale observation
statistics sees a different input scale than it trained on and reads as silently
degraded:

| file | what it holds |
|---|---|
| `model.zip` | policy + value net **and the Adam optimizer state** |
| `vecnorm.pkl` | running observation mean/variance |
| `state.json` | timestep count, unlocked pool, per-lesson success history |

`--resume` restores all three and continues the timestep counter. On resume,
`--steps` means *additional* steps, not a new total. Periodic checkpoints every
`--ckpt-every` steps (default 100 000) on top of the Ctrl-C one.

## The grip pad — why the lesson scenes differ from `scene.xml`

The hook is a slot round a smooth cylinder, so nothing holds it along the rung axis:
`controller/test_climb.py` phase 7 measures ~107 mm of travel under 40 N. The first
two PPO runs both solved lesson 1 by exploiting that instead of climbing — the hooks
never left the rung (throat-to-rung gap never exceeded 43 mm on a 50 mm throat) and
the robot simply shoved itself 1.1 m sideways.

**The fix goes on the hand, not the wall.** A climber chalks up and wears rubber;
they do not groove the rock. And a real climber can still slip, so the aim is to make
the slide small and honest, not to abolish it — what is left is the policy's problem.
(Collars on the rungs were built and measured first, and rejected on exactly this
argument: they worked, at the price of turning the wall into a jig.)

Raising the hook's friction alone does **not** work, and the reason is the useful
part: the contacts are far too soft, so extra friction just funnels more force into
the contact and the hook sinks *into* the rung. Penetration against µ, measured:

| µ | 1.0 | 1.4 | 1.8 | 2.0 |
|---|---|---|---|---|
| penetration | 3.3 mm | 7.1 mm | 18.3 mm | 21.7 mm → **falls through** |

Rubber only becomes available once the contact is stiff enough to carry it. The pad
is therefore `friction 2.0` **plus** `solref 0.005` and `solimp` d_max 0.99, with
`priority="1"` so the hand's pad defines the contact outright instead of being
averaged with the rung — which is the physical statement too, since the pad is the
soft part that deforms.

Phase 7 re-measured under its own grip conditions, hook travel along the rung:

| pull | bare hook | grip pad |
|---|---|---|
| 20 N | +42 mm | **+5 mm** |
| 40 N | +107 mm | **+16 mm** |
| 60 N | +2914 mm, off | +2162 mm, off |
| 80 N | +9987 mm, off | +5686 mm, off |

At 60 N and above the hook is levered off the rung either way — a different failure
on a different axis, which the pad neither helps nor hurts. That is the "a climber
slips too" regime, and it is left in on purpose. The hang keyframe is unaffected
(drift 1.75 mm, both hooks seated).

Most of the win is the stiffness, not the rubber: µ 1.0 with the stiff contact
already gives +35 mm. µ 3.0 would give +15 mm instead of +20, but 2.0 is the safer
claim for rubber on metal and costs 5 mm.

`scene.xml` keeps the bare hook so the classical controller's measured numbers stay
valid. Only `urdf/lessons/*` get the pad.

## Design decisions

- **Wide shoulders.** Lessons open shoulder abduction 60° outward (the
  `scene_wide_shoulders.xml` edit). A human climbs by flaring the shoulders in the
  wall plane and flexing the elbows; the original ranges only travel inward, so the
  posture the user asked for cannot exist without this. The classical controller is
  tuned against the narrow ranges and still runs on `scene.xml` only.
- **Obs are hardware-legal** (encoders, raw IMU, F/T, pads, own setpoints, goal
  relative to the anchor-rooted estimate, and the 3 nearest rungs per hand).
  **Reward and success use ground truth** — a training-time oracle, per the model
  README. Nothing in obs needs SLAM: the wall map is known in advance, as it would
  be from a perception module.
- **The policy can see the nearby rungs.** Without them it knows where the goal is
  and nothing about what lies between, so lesson 2 (gaps) and 6 (staggered) are
  guesswork and one shared policy could only memorise each fixed layout. Three
  rungs per hand is what a hand chooses between; the map supplies them.
- **Action** = setpoint deltas on all 17 position servos at 25 Hz, ±1.5 rad/s.
  Smoothness comes from the action-rate and joint-velocity penalties.
- **No SLAM / full autonomy layer.** Goals come from the map; the planner layer
  (which hold next) stays classical (`controller/holds.py`) and can feed goals to
  a trained policy later.

## Training on this Mac — measured, not assumed

`--device mps` runs the policy on the Mac GPU and works, but it is **2x slower**.
Same 40960-step run on an M4 Pro (12 CPU cores, 16-core GPU), lesson 1, 8 envs:

| device | fps |
|---|---|
| `cpu` (default) | **4107** |
| `mps` (Mac GPU) | 1994 |

The network is the wrong shape for a GPU: 99 → 64 → 64 → 17 is far too small to
cover Metal's per-kernel launch overhead (`rl.bench`: 2183 updates/s on cpu vs 1726
on mps), and MuJoCo runs on the CPU regardless, so every rollout step pays a
host↔device copy on top. **The simulator is the bottleneck** — ~10 000 control-steps/s
across 12 processes, while the policy update costs under a fifth of wall clock.

So the lever is parallel envs, not the GPU. Measured end-to-end fps by `--envs`:

| envs | 6 | 8 | 10 | **12** | 16 | 20 | 32 |
|---|---|---|---|---|---|---|---|
| fps | 3413 | 3950 | 4064 | **4330** | 4681 | 4977 | 5259 |

Default is 12 (the core count). It keeps creeping up past that because envs idle
during each policy update, but 32 buys 21 % over 12 for 2.7× the processes and an
8192-sample rollout. Re-run `python -m rl.bench` on any other machine.

This flips if the policy grows a vision encoder — then measure again.

**A real GPU path exists but is a rewrite**: MJX (`mujoco-mjx` 3.12) runs the
simulator itself on the GPU with thousands of envs batched, which is where a GPU
would actually pay off. It needs the env rewritten in JAX — on CUDA that is a well
-trodden path, on Apple silicon it needs `jax-metal` (0.1.1, still experimental).
Not worth it until a policy trains at all.

## Moving to a Linux + CUDA machine

Nothing here is Mac-specific. The RL path imports no OpenGL, so it runs headless.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # see the file for CUDA index-url options
python -m rl.lessons                   # regenerate the scenes on this machine
python -m rl.bench                     # measure BEFORE picking a device
python -m rl.train_ppo --name climber --device cuda
```

- `--envs` defaults to the machine's usable cores via `sched_getaffinity`, which
  respects `taskset` and cgroup limits — so it does not oversubscribe a shared box
  or a container the way `os.cpu_count()` would. On a 64-core server try more than
  the default; envs idle during each policy update. Budget RAM at **~66 MB per env**
  (each holds all 9 scenes).
- **A checkpoint moves between machines and devices.** Verified cpu → mps → cpu with
  the timestep counter continuing; cuda uses the same torch `map_location` path. So
  you can start a run here, copy `out/<name>/` to the Linux box, and carry on with
  `--resume --device cuda`.
- **Do not assume `--device cuda` is faster.** The measurement above is Metal's, but
  the reason is not Metal-specific: a 135 → 64 → 64 → 17 MLP is small enough that
  kernel-launch overhead dominates, and MuJoCo stays on the CPU either way. CUDA has
  lower launch overhead than Metal so it may well come out ahead — this has not been
  measured on CUDA hardware. `python -m rl.bench` prints the CUDA device name and
  its updates/s next to the CPU's; decide from that, not from this paragraph.

## The reward, and the three exploits it had to be rebuilt around

Each term below exists because a policy broke the previous version. `rl.climb_task`
asserts all of it.

| term | why |
|---|---|
| `K_BANK` on progress credited **relative to the holding hand** | paying live world distance paid for the phase-7 slide: the first policy crossed 0.75 m with a hook that never opened its finger. A per-hand *anchor* — bank how far the hold travelled in the world between supported grabs — fixed that and was beaten by the drag (the `win5` section): a free hand is now credited its approach **minus the holder's own forward motion**, a loaded hand approaching is credited nothing, and every retreat debits in full. Credit never exceeds the world approach, so over any closed loop it sums to ≤ 0 — a drag, a tap, a jump and a slide all earn exactly nothing |
| anchor moves only after the hand got `CLEAR_OF` the rung | without it, a wrist-load dip counted as taking a new hold — on the 5.1M policy **every** re-anchor was that, with the hook never leaving the rung |
| both terms charged **both ways** | paying only in the good direction is a pump: measured **+27.28 paid for 0.000 m of travel** and 99 release/reseat cycles in one episode |
| `K_DANGLE` with a `DANGLE_GRACE` | the flat version taxed the release climbing is made of, −15.25 against +5.57 of progress |
| `K_SEP` | the distance term is a **sum** over both hands, so it does not care which hand works: policies froze one hand and crossed the other over it. Pricing only *crossed* then parked both hands on the same point (mean separation 1 mm), so it is a floor, not a sign test |
| `K_DANGLE` over a cumulative budget | charging a *continuous* stretch off the wall was evadable: 88 short hops kept every stretch under the grace and a hand off 79 % of the episode paid 0.13 |
| `K_STALE` | across every run, **the left hand never once let go**. The hands deadlock — the right is airborne 79–84 % of the time, so the left never gets a safe moment. `K_DANGLE` says do not stay off, `K_STALE` says do not stay on. Now **0.001**, not 0.01: at 0.01 two hands hanging for 60 s cost 28.6 — three times a fall — and that is the landscape `win5` settled into (below). Nudges must never add up to more than the fall they exist to prevent; the incentive test asserts a full idle episode (−7.4) stays above `FAIL_R` |
| finger inside `GRIP_BAND` gates `_holding` and `_success` | nothing ever asked a finger to close, and the simulator lets an open hook carry the whole robot. The 10M policy held with the finger at +20° for most of an episode. There are **two** ways out of the band and it used both — the left hand opened, the right ratcheted −36 → −46 → −37, past the −45° cliff where pull-off drops from >1200 N to 72 N |
| every hold test reads **contact force** (`_contact`) | see the section below — this replaced `ON_RUNG`, `CLEAR_OF` and `env.latched`, all three of which were proxies a policy could satisfy without holding, or *fail* while holding |
| `STALE_AFTER` 8 s → **2.5 s** | the threshold was longer than the episode it policed. The winning lesson-1 run is 5.8 s, so the hand that never released was charged exactly 0.00 |
| `MIN_GRABS` in `_success` | the task was completable with **neither hand ever leaving the rung**: 1 m traversed with the right hook never further than 24 mm from the rail and the left hand carrying under 10 N for 89 % of the episode. Alternating grips is not one way of doing this task, it is the task |
| finger setpoint clamped at `GRIP_CLIFF` | the joint runs to −90° but the hook levers itself off past −45°, where pull-off drops from >1200 N to 72 N. The 20.8M policy drove the right finger to **−83°** and the hook walked out of the throat while carrying 100 N. Nothing above the cliff is useful, so it is a mechanical stop — 45° of pure self-destruct removed from the search. Opening is untouched |
| `K_FLIGHT`, and a jump earns nothing | the 21M policy's whole lesson-1 gait was jumps: the right hook let go and re-caught within 1–5 control steps while the left hung in the air — off **96 %** of the time, **0.0** load swaps per episode — and each brief tap of the left hand then banked what the jump had covered. A control step in which, for even one physics substep, no hook bore `HOLD_LOAD` is a jump: charged, nobody is loaded so nothing is credited, and the landing is no cycle. Replayed: that policy drops from 5/5 to **0/12** |
| `K_SLIDE` | sliding stays legal — the rung is smooth on purpose — but a **loaded** hook dragged along it pays 10/m, because it is friction work and because it is invisible: measured in shadow on the alternating policy, the anchored hook was dragged **100–724 mm per tenure**, both hooks seated (next section). This is *not* what makes dragging unprofitable — the relative credit is. Raising it to 20 was tried when the drag appeared and it stalled learning outright: 3M steps at 30 % on the very first lesson, std 0.42, because the same charge lands on the few centimetres the holding hook settles by during every honest reach. It only has to beat the live shaping term: `assert K_SLIDE > 2 K_REACH` |
| `K_EFFORT`, `K_FIGHT`, `K_TIME` | the gait was legal but nothing like climbing. Measured against a still hang (29.6 N·m, 107 N): the stuck episode ran 145 N·m and **202 N of grip on a 109 N robot** — ~95 N of the two arms pulling against each other. All three price the *excess* over hanging still, never the absolute, so existing stays free |

Replayed on the audited policies the set does what it should. The 10.4M run's
99-step success scored +13.23 against its 1490-step stuck episode's −51.17, with no
term charging the healthy episode more than 0.30. The 10M run's ratcheting exploit,
which scored **+13.58 and `success`** under the old rules, now scores **−4.98 and
`fell`** with 0 real grabs from either hand.

A single-joint sweep says crossing is not necessary: the leading hand gains **+459 mm**
toward the goal on its 60° of outward travel, against **+64 mm** for crossing the
trailing one. Crossing was a local optimum, and lesson 1 — solvable by sliding both
hands — kept reinforcing it.

**Do not train on a narrowed pool.** Dropping lesson 1 to force lesson 2 was tried and
it is plain sequential fine-tuning: lesson 1 went from 12/12 to falling 4/4, and
lesson 2 got worse too. Keep the pool mixed, as the section above says.

## Contact force, and why every distance threshold before it was wrong

The 20.8M-step `win3` run reported **2.5 %** on lesson 1 and never unlocked lesson 2.
The policy was not the problem. Three separate proxies for *"is this hand on its
hold"* disagreed with each other and with the physics:

| test | what it measured | how it failed |
|---|---|---|
| `env.latched` | rung inside a box round the nominal seat | read False on **both** hands for 32 % of an episode — 89 % of those with a hook carrying more than 20 N |
| seat gap `< ON_RUNG` | radial distance, throat point to rung axis | a seated hook reads 2 mm hanging still but **31 mm under a 50 N side pull**, and a hand merely hovering reads 42 mm. The two states overlap, so no threshold separates them |
| wrist F/T | force through the wrist | a hand pressed flat on a wall reads 86 N with the finger shut on nothing |

They were also wired to *different* places: the reward used the gap, and the episode
was ended by `env.latched`. So a one-handed phase — the thing a climbing gait is made
of — was 0.6 s from a spurious `released` and a −10, while the reward said the hand
was fine. `ON_RUNG` was tightened to 20 mm on a measurement taken from a **static**
two-handed hang, which does not survive contact with a loaded hook.

MuJoCo already knows the answer. `_contact` sums the normal force between a hook's
collision geoms and any rung:

| | holding | free / fallen |
|---|---|---|
| contact force | **72 – 229 N** | **0.0 N exactly** |

No overlap, across static hangs, one-handed hangs, a 50 N side pull and a 200 N
downward pull; no flicker (1200 steps of still hang, minimum 40 N); and it costs
nothing (1 % *faster* than `latched`, which does matrix work per rung). It now
answers every one of those questions — held, let go, still attached, success —
so all three proxies and their thresholds are gone.

Re-measuring the same `win3` policy against contact instead of the gap:

| | reported | actually |
|---|---|---|
| lesson 1 success | 2.5 % | **40 %** (16/40) |
| real grab cycles per episode | L 0, R 2 | **L 8.1, R 10.3** |
| episodes with both hands cycling | — | **38/40** |

The alternating gait was already there. The measurement could not see it, and
`MIN_GRABS` — which reads the same broken counter — was rejecting the successes.

## Where the robot thought it was

The observation is map-relative: goal and rungs are given relative to the hands, and
the hands come from an estimator rooted in the hand that holds — encoders, IMU and
the wall map, nothing else. The 21M `win4` run showed what that costs when the gait
is wrong:

| `win4`, 21.2M steps | lesson 1 | lesson 2 (0.25 m gaps) |
|---|---|---|
| success | 95 % (best), 60 % (last) | **0 %** for the 16M steps it was in the pool |
| gait | right hook jumps, left hangs 96 % off | left hangs 95 % off |
| rungs touched in 30 episodes | — | `rung_start` only — **never `seg1`, never `seg2`** |
| estimate drift within an episode | 0.6 – 1.2 m | 0.3 – 2.8 m; `seg1` truly 134 mm away, seen at 417 |

Lesson 2 was not hard. The policy could not see it: every jump moves the robot along
the rung with no hand to root the estimate in, and the drift is the sum of the jumps.

Where the error comes from, measured on the alternating `win3` policy over 815 steps
and 43 grabs, with the free hand's estimate rooted at the anchor's **true** hold:

| attitude source | free hand, all steps | along the rung, at a grab |
|---|---|---|
| IMU triad (accel + mag) | 148 mm mean, 1425 max | 29 mm mean, 250 max |
| rung frame from the map, IMU only for the swing | 154 / 1426 | **16 / 139** |
| perfect | **0 / 0** | **0 / 0** |

The kinematics are exact. All of it is the attitude, which the body's own
acceleration bends. So holds are now placed by **reach** (`_reach_point`): the
distance between the two throats is exact from the encoders whatever the attitude,
both throats sit on rung axes the map knows, and the new hold is the point on a rung
at that distance from the anchor's hold — a sphere cut by a line. The map-rooted
guess picks the side; after that the branch stays put. Exact in the unit test from a
guess 157 mm off, across the gap onto `seg1` as well.

What reach cannot fix, measured in shadow on the trained gait: the anchor itself
moving. With both hooks seated the anchored hook was dragged 100–724 mm per tenure,
and every jump adds more. `K_SLIDE` and `K_FLIGHT` exist so that the gait the policy
is paid for is the one the sensors can track — one hand holds still while the other
moves, which is also the one that was asked for.

## `win5`: the map became exact, and nothing moved

15.3M steps under the three rules above, lesson 1 only, **0 / 40** — the best
checkpoint is the first one. Not falling, either: all 30 audited episodes ran the
full 60 s.

| `win5` latest, lesson 1 | |
|---|---|
| gait | left hook on 100 % of the time; right hook flutters — let go and re-touch in place, 9–20 times per episode, enough for its `MIN_GRABS`; body moves 0.18 m |
| pose estimate | anchor **3–20 mm** off through the whole episode, free hand 9 mm mean (p90 16) on approach — reach odometry with a still anchor is as exact as promised |
| action noise (std) | 1.0 → **0.13** (win4 held 0.47): exploration was gone |
| reward, per episode | stale −14.7, time −4.5, effort −1.3, slide −0.8, dangle −0.8; bank **+0.9**, reach +0.3; total −22 |

Doing nothing cost −33 (stale −28.6, time −4.5) against −10 for a fall. The policy
found the cheapest place in that landscape — hang from one hook, flutter the other to
dodge its share of the stale charge — and its noise collapsed there. The +7/m for
progress was never large against the nudges, and in earlier runs it had not needed
to be: a random shove along the rung was paid. Now a shove is a charged drag, and
the one paid move — lift, carry, re-seat — is a sequence, and 1.1 m of it is five
cycles per hand.

Two changes. `K_STALE` 0.01 → 0.001, so a full idle episode costs −7.4 and a fall
never becomes the cheap way out (asserted). And a first rung that random behaviour
brushes against: random policies produce a supported grab cycle from **both** hands
in about a third of episodes, and those are paid — that is the gradient.

The first draft of that rung was 0.30 m per hand, and it deadlocked by geometry: the
hands start 0.314 m apart, so the right hand's goal landed 14 mm from where the left
hand starts, and two 65 mm hooks cannot share that span. A 600k-step policy on it
had the right hook in its goal region on 42 % of steps, hovering 100 mm (min 44)
from the left hook, banking +0.96 and never seating; the left hook, which had to
move first, never let go. Hence lessons −2 and −1 (one hand alone moves 0.30 m — the left toward the goal,
then the right the other way, so each arm learns to lift and place and each hand
learns to carry; the other hand owes only a supported re-grip where it is) and
lesson 0 (both hands 0.40 m, the goal 86 mm past the other hand's start).

Checked locally before shipping (8 envs on the M4 Pro, then numbered −1/0): the
left-hand step at 52 % by 250k and 75 % by 400k, both-hands unlocked at about 300k
with its first success by 400k, action std still 0.90. By 2M the pool had opened to
lesson 2 — and lesson 1 was being solved by a drag: the right hook, which no lesson
had yet asked to lift, pulled along the rung 864 mm per episode while the left hand
tapped its way forward, 20 % success and the map 0.2–0.8 m off again. Decomposed:
bank +7.9, reach +3.3 against slide −8.8 — every tap re-seated the left hand where it
already was *relative* to the right and banked what the drag had carried, so one
drag advanced both anchors and the slide charge could not keep up. Doubling the
charge did not help, it hurt: the run at 20/m never left the first lesson in 3M
steps (30 %, std 0.42), because a holding hook settles by a few centimetres during
every honest reach and paid for it. The fix is the credit itself: progress is
measured the way the sensors measure it, relative to the hand that holds — and a
lesson that makes the right hand lift too.

Skipped: domain randomization, best-checkpoint keeping. Reward weights are
first-guess knobs — `rl.eval` plus the diagnostics are how you tell.
