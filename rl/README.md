# PPO curriculum

The frame for training PPO on the climbing tasks, as sixteen lessons ordered by
difficulty instead of the mixed scene.

```bash
python -m rl.lessons        # regenerate urdf/lessons/*.xml + smoke-check each
                            # (training writes a missing one itself; run this after
                            #  editing scene.xml or a lesson spec)
python -m rl.climb_task     # smoke-check the task API on every lesson
python -m rl.bench          # where the time actually goes on this machine
python -m rl.train_ppo --self-check          # curriculum logic, trains nothing

python -m rl.train_ppo --name climber        # train: all 13, curriculum on
# Ctrl-C any time, then:
python -m rl.train_ppo --name climber --resume

# from a second terminal, while it trains:
python -m rl.status --name climber --watch
```

`torch`, `gymnasium` and `stable-baselines3` are installed in this repo's `.venv`
(torch 2.13 / SB3 2.9 on Python 3.14 — no separate venv needed).

| file | what it is |
|---|---|
| `lessons.py` | the 16 lesson specs + scene generator (writes `urdf/lessons/`) |
| `climb_task.py` | Gymnasium-shaped env: obs / action / reward / termination |
| `train_ppo.py` | thin SB3 PPO entry point |
| `bench.py` | simulator throughput vs env count, and cpu vs mps for the policy |
| `render.py` | PNGs of the lesson scenes → `out/scenes/` (ffmpeg writes them, no image dep) |
| `status.py` | what a run is doing, read from its checkpoint — safe to run while it trains |
| `eval.py` | score a checkpoint per lesson against a do-nothing control |

## Lessons

−2. left hand, one step (0.30 m) → −1. right hand, one step (0.30 m, the other way)
→ 0. both hands, one step (0.40 m) → 1. traverse, continuous rung → 1.1. stop at the
end of a rung → 1.2 / 1.35 / 1.5. traverse gaps of 0.04 / 0.08 / 0.12 m → 2. traverse,
gaps (0.25 m) → 3. climb up → 4. up-over-down → 5. climb down → 6. staggered heights →
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
| `K_BANK` on progress credited **relative to the holding hand** | paying live world distance paid for the phase-7 slide: the first policy crossed 0.75 m with a hook that never opened its finger. A per-hand *anchor* — bank how far the hold travelled in the world between supported grabs — fixed that and was beaten by the drag (the `win5` section): a light hand (under `HOLD_LOAD`) whose partner is loaded is credited its approach **minus the partner's own motion toward that goal, both ways**; a loaded hook is credited nothing either way (its motion along the rung is a slide, `K_SLIDE`'s business); a hand nobody carries is credited nothing forward and debited in full back. A drag, a tap and a jump earn exactly nothing; the one loop that would pay — reach free, get dragged back loaded — costs `K_SLIDE > K_BANK`. The first draft was one-sided to be safe (every retreat debited, only the partner's *forward* motion subtracted) and it charged a known-good one-hand step **−220 mm per episode** for ordinary settling: −7.5 m of debits against +6.3 m of credit over ten episodes. One-sided rules charge for noise |
| anchor moves only after the hand got `CLEAR_OF` the rung | without it, a wrist-load dip counted as taking a new hold — on the 5.1M policy **every** re-anchor was that, with the hook never leaving the rung |
| both terms charged **both ways** | paying only in the good direction is a pump: measured **+27.28 paid for 0.000 m of travel** and 99 release/reseat cycles in one episode |
| `K_DANGLE` with a `DANGLE_GRACE` | the flat version taxed the release climbing is made of, −15.25 against +5.57 of progress |
| `K_SEP` | the distance term is a **sum** over both hands, so it does not care which hand works: policies froze one hand and crossed the other over it. Pricing only *crossed* then parked both hands on the same point (mean separation 1 mm), so it is a floor, not a sign test |
| `K_DANGLE` over a cumulative budget | charging a *continuous* stretch off the wall was evadable: 88 short hops kept every stretch under the grace and a hand off 79 % of the episode paid 0.13 |
| `K_STALE` | across every run, **the left hand never once let go**. The hands deadlock — the right is airborne 79–84 % of the time, so the left never gets a safe moment. `K_DANGLE` says do not stay off, `K_STALE` says do not stay on. Now **0.001**, not 0.01: at 0.01 two hands hanging for 60 s cost 28.6 — three times a fall — and that is the landscape `win5` settled into (below). Nudges must never add up to more than the fall they exist to prevent; the incentive test asserts a full idle episode (−7.4) stays above `FAIL_R` |
| finger inside `GRIP_BAND` gates `_holding` and `_success` | nothing ever asked a finger to close, and the simulator lets an open hook carry the whole robot. The 10M policy held with the finger at +20° for most of an episode. There are **two** ways out of the band and it used both — the left hand opened, the right ratcheted −36 → −46 → −37, past the −45° cliff where pull-off drops from >1200 N to 72 N |
| every hold test reads **contact force** (`_contact`) | see the section below — this replaced `ON_RUNG`, `CLEAR_OF` and `env.latched`, all three of which were proxies a policy could satisfy without holding, or *fail* while holding |
| progress measured **at the rung** (`_route`, `_cost`) | the straight-line distance to a goal 1.4 m along the wall paid the `win9` policy +0.3–0.4 m for dropping half a metre under the far segment — the nearest it could get to the goal with the other hand parked — and it hung there for 58 s in 20/20 episodes: lesson 1.5 at 0 % for 5.3M steps with every earlier lesson above 80 % (the `win9` section). The bank now measures each hand at its **rung point** (nearest point on any rung) to the goal, which a loaded hook can move only by sliding, and sliding is charged; the live reach adds the detour off the rung, both hands, both ways. Identical along a rung (lesson 1 replays to the same reward); a dive is charged live and refunded on the way back up. The detour must stay out of the bank: a loaded hook's throat sits 30–80 mm off its line under load, uncharged, and with the detour banked that was refunded on every re-seat — the first check run banked 56 m per episode on it. Route test: the same dive is +137 mm straight-line, −146 mm at the rung; a loaded hook pulled 30 mm off its line moves the bank 0 |
| the map describes **seats**, not geometry (`SEAT_IN`) | the nearest-rung offsets and the end distances the policy sees, and the potential it is paid on, clip each rung half a hook-width (32.5 mm) in from its ends: a hook centred on the geometric end hangs half off it. On the `win9` video the hook ratcheted to the very end of its segment and let go — the user's reading of it |
| the anchor follows the **SLAM** position along its rung (`SLAM_TAU`) | the loaded hook creeps along the rung while it holds — 85–210 mm per episode on `mac16`, 100 mm by the first hand-over — and nothing on the arm can see it: two hooks on a smooth rung have no world reference. At a gap the far segment then read 60–210 mm too far and the hook was pushed into its end face 38 mm short. The real robot's lidar SLAM sees the body move against the wall; simulated as the base position with `SLAM_SIG` (2 cm) of noise, low-passed over 1 s, it moves the anchor's map point along its rung. Replayed on the mac16 policy: anchor error p50 20 mm / p90 34 everywhere (was 51 / 149 on lesson 1.5 and 145–179 / 300 on lesson 1), −18 ± 27 mm at the moment the hook meets the far segment (was +89, 60–210). The slam test creeps the loaded hook 25 mm under a pull and asserts the anchor follows. A second touch zone on the hook's lip was tried first and was exact (every end contact at z = −110 mm in the hook frame, every seat at z ≥ −98) — but the robot has no such sensor, and it is gone |
| draws by share of **steps**, not episodes | a stuck lesson-1.5 episode runs 1490 steps against ~200 for a one-step lesson, so at a quarter of the draws it was 60 % of the samples, and what it learnt there — left hand sweeping the rung, right hand still — was learnt in the states lesson −1 starts from, which needs the opposite: −1 fell from 90 % to 0–2 % in both check runs. Each lesson's share is divided by its mean episode length. `mac18`: −1 at 92 % |
| `STALE_AFTER` 8 s → **2.5 s** | the threshold was longer than the episode it policed. The winning lesson-1 run is 5.8 s, so the hand that never released was charged exactly 0.00 |
| `MIN_GRABS` in `_success` | the task was completable with **neither hand ever leaving the rung**: 1 m traversed with the right hook never further than 24 mm from the rail and the left hand carrying under 10 N for 89 % of the episode. Alternating grips is not one way of doing this task, it is the task |
| finger setpoint stopped at the **band's edge** | the joint runs to −90° but the hook levers itself off past −45°, where pull-off drops from >1200 N to 72 N. The 20.8M policy drove the right finger to **−83°** and the hook walked out of the throat while carrying 100 N, so the stop went to the cliff. That left a 5° trap: a policy that squeezes as hard as it can sits at −40…−44° under load — a fine hold physically (98–179 N) that neither the reward nor the estimator recognised, because both ask for `GRIP_BAND`. On the 10.8M `win8` policy **90 %** of loaded steps on lesson 1.5 (34 % on lesson 1) were "not gripping" for that reason alone; the estimator kept its anchor on a hand that was moving and saw the far ledge 240–500 mm from where it was. The stop is now the band's edge, −40°: squeeze as hard as you like, it is in band (375/382 loaded steps while swinging, asserted) |
| `K_FLIGHT`, and a jump earns nothing | the 21M policy's whole lesson-1 gait was jumps: the right hook let go and re-caught within 1–5 control steps while the left hung in the air — off **96 %** of the time, **0.0** load swaps per episode — and each brief tap of the left hand then banked what the jump had covered. A control step in which, for even one physics substep, no hook bore `HOLD_LOAD` is a jump: charged, nobody is loaded so nothing is credited, and the landing is no cycle. Replayed: that policy drops from 5/5 to **0/12** |
| `K_SLIDE` | sliding stays legal — the rung is smooth on purpose — but a **loaded** hook dragged along it pays 10/m, because it is friction work and because it is invisible: measured in shadow on the alternating policy, the anchored hook was dragged **100–724 mm per tenure**, both hooks seated (next section). This is *not* what makes dragging unprofitable — the relative credit is. Raising it to 20 was tried when the drag appeared and it stalled learning outright: 3M steps at 30 % on the very first lesson, std 0.42, because the same charge lands on the few centimetres the holding hook settles by during every honest reach. It has to beat the live shaping term and the bank: `assert K_SLIDE > K_BANK and K_SLIDE > 2 K_REACH` |
| a hold counts only while the **other hand is loaded** | three points of contact. A hold taken with nothing else on the wall was a jump's landing or a hand-over with a gap in it, and the gap is where the map goes wrong: 86 of 129 anchor switches on the 3M policy had no step with both hands confirmed and baked in 62 mm each. Gated, the same curriculum unlocks at the same pace (lesson 2 at 0.94M against 0.89M), hand-overs with overlap go from 30 % to 53 %, and the anchor's error at the end of a lesson-1 traverse drops from 540–980 mm to 27–320 mm. What is paid is what can be seen |
| credit classified by the load at **both ends** of the step | the pump the finger-stop fix uncovered: with every touch now in band, a 3M local policy took **578 holds per episode** with the right hand — 97 % of its lifts one or two steps long — and banked **20 m** of credit on a body that moved 0.4 m, +78 per episode against +25 for actually crossing the gap. Lift (free at the end of the step): the 30 mm forward credited. Touch down (loaded at the end): the 30 mm back credited nothing, and not a slide either. Two closures were tried first and were the wrong tools — charging a slide with load at *either* end taxed every seat and unseat of an honest gait (the 95 % lesson-1 policy paid 1.5 m per episode of it and fell to 50 %), and requiring three steps clear before a hold refused the 1–2-step hops honest hands take 70–90 % of the time, so the first lesson stayed at 0 % for 3M steps. Classified by the load at the *start* of the step, the lift is worth 0 and the touch-down debits the 30 mm; and the partner's motion is subtracted only while the partner is loaded at both ends. Replayed: the 95 % policy is back to 15/16 and 13/16, the pumping policy nets −122. That closure had a mirror, and `mac14` found it: classified by the start, the lift is the free step, so it rocked the hook on the rung at 12.5 Hz — throat 32 mm *back* on the lift (ignored), 31 mm *forward* on the touch-down (credited): 698 holds and 23 m per episode, +241. Classifying by either end of a step leaves the other end open. Now a step is a slide's business only with load at **both** ends of it — the slide charge's own test — and every other step is a potential difference, both ways; so every motion of a rung point is either a charged slide or a two-way credit, and no cycle between the two pays. Both one-step cycles are in the jump/drag test and net exactly nothing; the rocking policy replays to −38, the two 95 % lesson-1 policies to 11/12 |
| `K_GRAB`, once per hand per episode | seating a hook from the air is the one skill the reach credit cannot teach by itself. Under the relative credit the 600k policy carried the free hand to within 40 mm of its goal and **hovered there for the whole episode** — 0.2 grabs per episode, paying the dangle charge rather than come down, because the grab that would have banked its credit was a rare event with a small payoff. The earlier world-anchored reward had taught seating by accident, by paying shuffles *on* the rung first. One bonus for each hand's first supported hold: flutter cannot farm it, and it is a partial payment of `MIN_GRABS`, not a goal of its own |
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
acceleration bends — the bare triad (accelerometer + magnetometer, no state) reads
the swing as tilt and is **26.5° rms** off while the body moves. The robot has a
gyro, so the torso attitude is now a complementary filter: the gyro carries it
across each control step, the triad pins the drift with a 2 s time constant
(`AHRS_TAU`). Measured on the same trajectories: 3.6° / 2.1° rms on lessons 1 / 2,
free hand **16 / 12 mm** against 91 / 130 for the triad; the pure gyro (no
correction) drifts to 83 mm, and a swing-angle model of the seated hook — rung
frame from the map, one angle from the IMU — was tried first and came out *worse*
than the triad (181 mm even with the true angle: a loaded hook does not sit still
in its throat). Holds are placed by **reach** (`_reach_point`): the
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

## `win9`: everything before lesson 1.5 passes, and the reach paid for hanging below the rung

5.3M steps with the draw weights and the start-of-step credit: −2 80 %, −1 92 %,
0 92 %, lesson 1 88 % (10/12 in audit, two hands alternating, 1.0 m in 480 steps),
no collapse — and lesson 1.5 at 0/40 the whole way, best checkpoint at 1.8M. Not
blindness this time: the estimate at the far ledge is right to 30–50 mm. Not the
pool: the four lessons behind it held. The action std is 0.62.

What the policy does, every episode (20/20 in audit, and 7/8 for the mac13 policy
of the previous section): the left hook hops along the start segment as it does on
lesson 1 (first hop 87 mm, then 26 mm) until it is past the end, level with it at
y 0.39 — the segment ends at 0.35 and the far one begins at 0.47 — and then, within
one second, drops 0.34–0.47 m below the far segment and hangs there for the
remaining 58 s, 0.9 m from the right hand, which never moves. Off the wall 98 % of
the episode, −13.8 of `K_DANGLE`, −24 per episode against −7.4 for hanging still.

The drop was *paid*. Progress was the straight-line distance from the hook to a
goal 1.4 m along the wall, and from level with the gap a hand hanging half a metre
under the far segment is 0.14 m nearer that goal than one level with it:
+0.30–0.44 m of "approach" over the dive on replay, +0.6–0.9 of reach inside three
seconds, for arriving in a state no lesson had taught it to leave. Every earlier
lesson has one continuous rung, so straight-line and along-the-rung had always
agreed, and the seam only opened at the first gap.

Progress is now measured at the rung. A hand's rung point is the nearest point on
any rung (`_route`); the bank measures that point to the goal, and the live reach
adds the detour off the rung, both hands, both ways (`_cost`). Along a rung nothing
changes — 0.30 m of reach still pays 0.30 and lesson 1 replays to the same reward.
Off the rung the detour is charged and refunded on the way back; it is a function
of the state, so it telescopes and cannot be pumped. The route test asserts the
win9 dive itself: +137 mm in a straight line, −146 mm at the rung.

The first check run of it (`mac14`, 3M) scored **+241 per episode on lesson 1.5**
with a body that never moved — the right hand taking 698 "holds" per episode and
banking 56 m — and it taught two things. The detour cannot be in the bank: a loaded
hook's throat is pulled 30–80 mm off its line by the load, uncharged, and that was
refunded on every re-seat. And under it a second seam that had been there since the
win8 fix: the credit was classified by the load at the *start* of the step, which
leaves the lift-off step free, so the hook rocked on the rung at 12.5 Hz — throat
32 mm back on the lift (ignored), 31 mm forward on the touch-down (credited). The
win8 pump was the same cycle the other way round, and classifying by either end of
a step leaves the other end open. Now a step is a slide's business only with load
at *both* ends — the slide charge's own test — and every other step is a potential
difference both ways, so every motion of a rung point is either a charged slide or
a two-way credit and no cycle between the two pays. Both one-step cycles are in the
jump/drag test and net exactly nothing. Replayed: the rocking policy nets −38, the
win9 and win8 lesson-1 policies 11/12 at +19 and +17, win9 on lesson 1.5 −25 as
before.

Checked locally with both (`mac16`, 3M steps, 8 envs): the dive is gone — the left
hook spends 0–6 steps below the far segment instead of 1400 — and it now reaches the
far segment in every episode and touches it for 13–295 steps, but never takes a
seated hold there; lesson 1.5 stays at 0/40. Measured at those contacts: the throat
is 38 mm short of the segment's end (min 24, max 48), 30 mm below its axis, the tip
pressing the end face with 7–22 N — and the estimator has the hand 60–210 mm
*behind* where it is, so the policy believes the segment is still that far away and
keeps pushing. The whole of that error is the anchor: the loaded hook **creeps along
the rung while it holds**, 85–210 mm per episode and 100 mm already at the first
hand-over, and the estimator's anchor point does not move. Two hooks on a smooth
rung have no world reference, so this creep is unobservable on a continuous rung
(where it does not matter) and fatal at a gap. It is not the solver's tangential
slip: `noslip_iterations` removes the drift under a static 25 N pull (25 → 0 mm) but
the policy's own loading still slides the hook 489 mm per episode with it on. Three
things followed, each in the reward table: the anchor's map point following the
lidar SLAM position along its rung (the only sensor the real robot has that sees
the wall; a lip touch zone was tried first, was exact, and was removed because the
robot has no such sensor); the map and the potential describing seats rather than
geometry; and the curriculum drawing by share of steps, because lesson −1 had
collapsed to 0–2 % in both check runs of the both-ends rule — a stuck 1490-step
lesson-1.5 episode was 60 % of the samples at 25 % of the draws, and its behaviour
was learnt in lesson −1's own states.

With the seats, the step draws and the (since removed) lip landmark (`mac18`,
3M): −2 100 %, **−1 92 %**, 0 92 %, 1 60 %, 1.5 0/40 — but the far segment was
reached and *held*, in 12 of 16 episodes, by the wrong hand: the right hook crossed
over the left, which never moved, and hung on the far segment for ~1000 steps paying
`K_SEP` −41 per episode for the crossing. The events lesson 1.5 needs are happening
and the credit is paying them.

With the SLAM tracking in place of the lip pad (`mac19`, 3M): −2, −1 and 0 at
**100 %**, lesson 1 at **95 %** (11/12 in audit, 1.0 m in 184 steps, 106 mm of
loaded slide — the cleanest gait so far), the anchor right to p50 4 mm / p90 16 mm
on lesson 1.5 — and lesson 1.5 at 0/40 in a new basin: the left hook goes to the
end of its segment, past its last seat, and hovers there brushing the end (−11 per
episode, no dangle, no contact with the far segment), the right hand never moving.
Two numbers explain it. This gait's hops are long — p50 226 mm, p90 307 on lesson 1
— so the opener lands past the last seat (317 mm) into the gap every time, and the
hand never retreats the 5–10 cm to a seat: no lesson had ever asked a hand to land
*short* of somewhere. And crossing from the last seat with the partner still at its
start asks the arms for 0.54 m plus the gap — 0.66 m on lesson 1.5 against the
0.67 m static limit, which is why the mac16 hook stopped 38 mm short of the far
segment, and 0.79 m on lesson 2, which cannot be lunged at all. The climb the user
described — lead hand to its last seat, trailing hand up beside it, then across — is
the only way through lesson 2, and it starts with stopping. Three lessons were
added: **1.1**, one continuous rung ending 3 cm past the left hand's goal, which is
one ordinary opener from its start (land on the last seat or hang half off the
end); and **1.2** and **1.35**, gaps of 0.04 and 0.08 m, so the reach a lunge needs
rises 0.58 → 0.62 → 0.66 → 0.79 instead of jumping from a continuous rung to the
limit.

## `win8`: lesson 1 passed, and a 5° trap under the far ledge

10.8M steps on the curriculum with lessons −2 … 1.5. The best checkpoint, at 2.66M:
−2 95 %, −1 98 %, 0 95 %, **lesson 1 95 %** — two hands alternating, 18/20 in audit,
1.0 m in 194 steps — and lesson 1.5 at 1/40. Then 8M steps in which lesson 1.5 stayed
at 0 % and, drawn as often as the rest, pulled lesson 1 down to 32 % and the action
std to a floor of 0.04: the collapse the previous section warned about, at full
scale. Two things were wrong, one in the task and one in the curriculum.

On lesson 1.5 the free hand reached the far ledge — 35–78 mm from its axis by step
25–47 — and then hovered there for the rest of the episode, 96 % of the time off the
wall, while the estimate said the ledge was 200–320 mm away. The estimate was wrong
because the estimator's anchor had stayed on a hand that was moving: the other hand
was carrying 48–159 N and was not *confirmed*, and it was not confirmed because its
finger read −40…−44°. The finger stop sat at the cliff, −45°, and a policy that
squeezes as hard as it can lands just under the band's −40° edge — 90 % of loaded
steps on that lesson, 34 % on lesson 1. Physically a fine hold; to the reward and
the estimator, not a hold at all. The stop is now the band's edge.

And a freshly unlocked lesson is drawn `NEW_SHARE` (25 %) of the time until it is
being passed at `WARM_RATE` (20 %), so a lesson that fails for a while is a quarter
of the gradient, not half, and the pool keeps what it has.

The finger-stop fix then uncovered a pump that the −45° stop had been hiding: with
every touch in band, the 3M check run learnt to lift the right hand for one step
(credited its 30 mm of relative approach) and set it down loaded (the 30 mm back
neither debited nor, since the slide charge wanted load at both ends of the step,
charged) — 578 "holds" and 20 m of bank per episode. It is closed where it lives:
the credit is classified by the load at the *start* of the step, so the lift is
worth nothing and the touch-down debits its 30 mm (which left the mirror open — the
`win9` section). Two harsher closures were tried
first and each broke an honest gait — the reward table has them. Every gate a policy
can saturate its way past, it will; and every fine aimed at a pump lands on the
physics too.

Checked locally with all of it (3M steps, 8 envs): −1 at 0.14M, 0 at 0.25M, 1 at
0.46M, 1.5 at 0.68M; at 3M lesson 1 at 88 % in training and **12/12** in audit —
1.03 m in 138 steps, both hands alternating — with no collapse through 2.3M steps of
lesson 1.5 in the pool. On lesson 1.5 the pump is gone (−21 per episode, 2 holds by
the right hand), the estimate at the far ledge is right to ~50 mm, a hand seats on
the ledge in 11 of 20 episodes and one episode crossed both gaps. The holding hand
still sits 335–520 mm from its edge when the free hand reaches — the climber's order
is not learnt yet — but the events the credit needs are now happening.

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

With the relative credit, the right-hand step and `K_GRAB` (3M local steps): −2 at
100 %, −1 at 73 %, 0 at 63 %, **lesson 1 at 80 %** with both hands lifting — right
hook off the rung 28 % of the time against 2 % in the drag, 42 load swaps per
episode, 87 % success in audit — and lesson 2 unlocked at about 2.5M. What remained
was the map: 447 mm of loaded slide per episode, two thirds of it millimetre-scale
settling under load rather than dragging, plus the triad's 25° — enough to put
`seg1` at 230–630 mm in the estimate when it was 65–120 mm away. Hence the attitude
filter above, trusting a seated-but-unloaded anchor through the hand-over, and
counting a hold only while the other hand is loaded. With those three (1.5–3M local
steps): lesson 2 unlocks at 0.9M in every run; on lesson 2 the anchor holds to
9–70 mm and the free hand to 31 mm on approach — the map is readable there — and on
lesson 1 a traverse ends 27–320 mm off instead of 0.5–1.0 m. Lesson 2 itself was still
0/40 after ~2M steps of exposure on the Mac, and the reason is a climber's rule the
policy had no way to find. When the free hand came closest to the far ledge it was
120–160 mm short, the hands were 627–689 mm apart — the static reach limit — and the
**holding hand was still 0.47–0.54 m from the edge of its own segment**. Lead hand to
the edge, trailing hand up beside it, *then* reach: from a holder at y ≈ 0.25 the
free hand reaches ~0.9 and the ledge (0.60–0.95) is in static reach, no swing needed.
That order was never once rewarded, because a 0.25 m gap is never crossed by
accident — and the edge of a rung was not in the observation at all (the three
nearest rungs were given as nearest-point offsets, which read 0 anywhere along the
rung you are on). Two changes, neither a script: each hand now sees the signed
distance to both ends of its nearest rung, and lesson 1.5 puts a 0.12 m gap in the
way first — one an ordinary 0.30 m step crosses, so the first crossings happen with
the gait the policy already has and the credit does the rest. The body swing is a
lesson-9 matter (0.75 m, beyond static reach), not a lesson-2 one.

Checked locally (1.5M steps, 8 envs): with the end distances in the observation
the pool opened -1 @ 0.10M, 0 @ 0.19M, 1 @ 0.37M, 1.5 @ 0.78M — lesson 1 at 60 % and the first 0.12 m crossings by 1.5M;
without them, -1 @ 0.14M, 0 @ 0.24M, 1 @ 0.38M, 1.5 @ 0.54M, lesson 1 still at 0 %. One run of the same configuration
collapsed instead: lesson −1 reached 70 % at 0.44M, lesson 0 unlocked and stayed at
0 %, and by 3M both were at 0 % with the left hand fluttering 48 grabs per episode.
A newly unlocked lesson that fails for long is half the pool's gradient, and a
policy can lose what it knew to it. `rl.status` shows it: an earlier lesson's rate
falling while the newest sits at 0 % for more than ~1M steps. If that happens,
restart — the runs that get past it do so early.

Skipped: domain randomization, best-checkpoint keeping. Reward weights are
first-guess knobs — `rl.eval` plus the diagnostics are how you tell.
