# PPO curriculum

The frame for training PPO on the climbing tasks, as eighteen lessons ordered by
difficulty instead of the mixed scene.

```bash
python -m rl.lessons        # regenerate urdf/lessons/*.xml + smoke-check each
                            # (training writes a missing one itself; run this after
                            #  editing scene.xml or a lesson spec)
python -m rl.climb_task     # smoke-check the task API on every lesson
python -m rl.bench          # where the time actually goes on this machine
python -m rl.train_ppo --self-check          # curriculum logic, trains nothing

python -m rl.train_ppo --name climber        # train: every lesson, curriculum on
# Ctrl-C any time, then:
python -m rl.train_ppo --name climber --resume

# from a second terminal, while it trains:
python -m rl.status --name climber --watch

python -m rl.hardware       # GPU or CPU: which simulator training will use here
python -m rl.mjx_check      # the GPU path against this one (see "The GPU path")
```

`torch`, `gymnasium` and `stable-baselines3` are installed in this repo's `.venv`
(torch 2.13 / SB3 2.9 on Python 3.14 — no separate venv needed), and so are `jax`
and `mujoco-mjx` for the GPU path.

| file | what it is |
|---|---|
| `lessons.py` | the 18 lesson specs + scene generator (writes `urdf/lessons/`) |
| `climb_task.py` | Gymnasium-shaped env: obs / action / reward / termination |
| `train_ppo.py` | thin SB3 PPO entry point, on either simulator |
| `hardware.py` | is there an NVIDIA GPU JAX can use — picks the simulator |
| `mjx_scene.py` | the robot MJX can simulate, and the lessons as data (writes `urdf/lessons/mjx_*`) |
| `mjx_task.py` | `climb_task` ported to JAX, thousands of envs in one GPU batch |
| `mjx_check.py` | the GPU path cross-checked against this one: task, physics, a trained policy |
| `bench.py` | simulator throughput vs env count, and cpu vs mps for the policy |
| `render.py` | PNGs of the lesson scenes → `out/scenes/` (ffmpeg writes them, no image dep) |
| `status.py` | what a run is doing, read from its checkpoint — safe to run while it trains |
| `eval.py` | score a checkpoint per lesson against a do-nothing control |

## The real-actuator robot (Sept 21 2026)

`urdf/` is now the 21.9 kg robot generated from `cad/` — 17 CubeMars actuators,
masses from the solids, torque limits off datasheets. **Every policy trained
before it learned the 11.1 kg robot and is void: start a new run, do not
`--resume` an old one.** The shapes still match, so such a resume would load and
quietly be wrong; checkpoints now record which robot they learned (a hash of
`scene.xml`) and `--resume` refuses a different one unless given `--new-robot`.

**Then the shoulder was rebuilt pitch-first** (`cad/src/lib/spec.py`). The user
saw the raised arm grow out of a pivot *below* the abduction motor and fold back
up past it; ten of twelve published humanoids put the pitch actuator in the
torso and the roll actuator beside its output instead, and so does this robot
now. It is 21.5 kg; each arm's joints run flexion, then abduction, in `qpos` and
`ctrl`; and from the hang, positive abduction on the left is the outward flare,
which `scene.xml` has all of (−12…180°) — the lessons no longer widen anything.
Policies trained before it are void as well, and the robot stamp refuses them.
Canary on this Mac, 1.5M steps with 10 envs: lesson −2 passed before 100k, −1 at
~456k, and 0 was at 57 % with lesson 1 unlocked by the end.

The swap loaded cleanly and was broken in four ways, all found by this
directory's own checks — `python -m rl.climb_task` failed its incentive test —
and all fixed at the source:

- **The hooks hung back to front.** The re-mount in the elbow section below lived
  only in `scene.xml`, and regenerating the robot from the CAD dropped it: the
  throat faced away from the wall, the shank hung between rung and wall, and
  both wrists leaned on the wall. It is in the CAD now (`spec.MOUNT_ROT`), so a
  regeneration cannot lose it again. The hands start at ±155 mm again, which is
  what every lesson's goals assume.
- **The raised arms were jammed into the torso.** MuJoCo collides each mesh's
  convex hull, and the torso frame exported as one mesh was one hull — a block
  out to the shoulder plates — that the raised upper arms sank 10–22 mm into,
  18–52 mm from any real metal. The hang held four shoulder actuators at 88 % of
  peak torque against 1500 N of contact. Each CAD part is its own hull now; the
  hang holds on ≤3.2 N·m per joint, and of the self-contacts reported in the
  climbing envelope 14 % were hull artefacts, down from 71 % (2 % since the
  chest became a box of plates, pitch-first) — the flare the lessons exist for
  is no longer blocked by air.
- **The two idle baselines were the old robot's.** `TAU_IDLE` 30 N·m and
  `WEIGHT` 109 N were typed in; the new robot hangs still on ~61 N·m and weighs
  215 N, so standing still was charged as effort and as fighting: −58 over
  lesson 1 against −10 for a fall, the `win5` trap exactly. Both are read off the
  model now (`ClimbTask.tau_idle`, `.weight`); a still hang costs −7.89.
- **A training env held 1.5 GB.** The actuators' visual meshes are 168k vertices
  a model and an env holds every lesson. Training loads the scenes without them
  (`MultiLesson(visual=False)`; the physics is bit-for-bit the same) and stores
  collision meshes as hulls: 0.33 GB per env, ~0.55 GB per worker process once
  torch is imported. Budget `--envs` x 0.6 GB. Only `rl.render` loads the visuals.

The self-checks' side pulls (25 N, 60 N) are fractions of body weight now: a hook
is held along its rung by friction, which scales with the load, so the same
newtons prove nothing on a robot twice as heavy.

**What is really different, measured:** moving costs about 4x the torque above
idle it did (77 against 18 N·m under the same random actions) — mostly the
reflected rotor inertia, `armature` 0.21 kg·m² on every AK80-64 joint where the
old robot had 0.01. `K_EFFORT` is unchanged; if the early curriculum stalls, it
is the first knob to A/B. And the estimate's still-hang anchor wander is ~20 mm
(was ~12): the lever from the anchor hook to the base is 1.5x longer.

**Canaries, fresh runs on this Mac.** On an intermediate collision model (the
wishbone still one hull) two runs passed lesson −2 at ~175k and ~150k steps, and
one reached lesson 1.2 by 2M. On the final model one run passed −2 at 950k and
another was still at 0 % after 1M, parked in the `win5` basin (hanging to the
timeout, fluttering both hooks, the left hand never travelling). The physics is
not what blocks it: the stuck policy fails on both models, and trained policies
transfer between them both ways (20/20 and 16/20 on lesson −2). The early
curriculum's seed variance is large; **if lesson −2 has not passed by ~1M steps,
start again under a new `--name`** rather than waiting it out. Policies trained
on the abduction-first shoulder rested their arms at the −170° flexion stop,
leaning on a real 1.4 mm contact with the torso; pitch-first there is nothing
there to lean on.

## Lessons

−2. left hand, one step (0.30 m) → −1. right hand, one step (0.30 m, the other way)
→ 0. both hands, one step (0.40 m) → 1. traverse, continuous rung → 1.1. stop at the
end of a rung → 1.2 / 1.35. cross one gap of 0.04 / 0.08 m → 1.5. traverse, two gaps
of 0.12 m → 1.7. cross one 0.18 m gap → 2. traverse, gaps (0.25 m) → 3. climb up → 4. up-over-down → 5. climb down → 6. staggered heights →
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
from the active pool on every reset instead, and a lesson unlocks when **every**
lesson in the pool clears 70 % success over 20+ episodes (it was the newest alone
at 60 % until `win17`). `--self-check` asserts that a lesson never leaves the pool.

## Pausing and resuming

`Ctrl-C` (or `SIGTERM`) saves and exits 0. A checkpoint is three files in
`out/<name>/`, always written together — a model restored against stale observation
statistics sees a different input scale than it trained on and reads as silently
degraded:

| file | what it holds |
|---|---|
| `model.zip` | policy + value net **and the Adam optimizer state** |
| `vecnorm.pkl` | running observation mean/variance |
| `state.json` | timestep count, unlocked pool, per-lesson success history, the best score and its lessons, the learning rate, rollbacks so far |

`--resume` restores all three and continues the timestep counter. On resume,
`--steps` means *additional* steps, not a new total. Periodic checkpoints every
`--ckpt-every` steps (default 100 000) on top of the Ctrl-C one.

`Ctrl-C` on Windows usually reaches the worker processes first, so the trainer
sees a closed pipe rather than a `KeyboardInterrupt`; that path saves too (as
does a worker killed by the OS), because `win17` lost the 55k steps since its
last checkpoint to it at 41.7M. For a run that should survive that on its own,
loop the resume: `for /L %i in (1,1,50) do python -m rl.train_ppo --name win17
--resume --lessons=... --steps 30000000`.

**A run looks after itself.** It does not need someone watching for a drift and
resuming from best by hand: if the best checkpoint's lessons fall a whole lesson
below that best for a million steps, the run goes back to `best_*` by itself and
carries on at half the learning rate (the `win17` section has the rule and why).
The update size is kept in range as well: each action dimension's std is held in
[0.05, 1] and the learning rate moves to keep every update's KL near 0.02
(`win17 2`). `python -m rl.status` shows the current rate and how many times it
has gone back, and everything the run prints is appended to `out/<name>/train.log`.

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

- **The flare is in the model.** A human climbs by flaring the shoulders in the
  wall plane and flexing the elbows. The abduction-first shoulder only travelled
  inward from the hang, so the lessons used to open it 60° outward; the
  pitch-first one flares with positive abduction on the left, and the lessons
  take `scene.xml`'s range (−12…180°) as it is.
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

**The real GPU path is the simulator, not the policy**: MJX runs the physics and
the whole task on the GPU with thousands of envs in one batch. It exists now —
see "The GPU path" below — and needs an NVIDIA card; on Apple silicon it would
need `jax-metal` (still experimental), so this Mac trains on the CPU path.

## Moving to a Linux + CUDA machine

Nothing here is Mac-specific. The RL path imports no OpenGL, so it runs headless.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # see the file for CUDA index-url options
pip install -U "jax[cuda12]"           # the GPU path; skip it for a CPU-only box
python -m rl.hardware                  # says which path training will take, and why
python -m rl.lessons                   # regenerate the scenes on this machine
python -m rl.mjx_check                 # the GPU path against the CPU one, ~15 min
python -m rl.mjx_task --envs 4096      # GPU control-steps/s, before a long run
python -m rl.train_ppo --name climber  # on the GPU by itself if the card is usable
```

The rest of this section is the CPU path (`--backend cpu`, or no usable GPU).

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

## The GPU path: MJX

```bash
python -m rl.hardware                   # what this machine will train on, and why
python -m rl.train_ppo --name g1        # GPU if JAX can use one, else the CPU path
python -m rl.train_ppo --name g1 --backend mjx    # force it (runs, slowly, on a CPU)
python -m rl.mjx_task --envs 4096       # control-steps/s of one GPU batch
python -m rl.mjx_check                  # task, physics, policy: GPU path vs this one
python -m rl.mjx_check policy --run out/g1        # a GPU run's policy on all three robots
```

`rl.hardware` answers one question — is there an NVIDIA GPU that JAX can put an
array on, with `mujoco-mjx` installed — and `rl.train_ppo` takes the GPU path when
the answer is yes. On a machine with no NVIDIA card JAX is not even imported and
nothing about the CPU path changes. On Windows CUDA JAX only exists under WSL2, and
`rl.hardware` says so by name rather than "no GPU".

**What moves to the GPU is the simulator and the task, not the learner.** Every
earlier measurement here says the simulator is the bottleneck, so that is what
runs on the card: `rl.mjx_task`, 4096 envs (`MJX_ENVS`) in one jitted batch. The
learner stays SB3's PPO with the curriculum, the guard, the adaptive rate, the std
bounds and the checkpoints of the CPU path — the same code, fed by `MjxVecEnv`
instead of a `SubprocVecEnv`, with the policy on the card too because 4096-row
batches are what a GPU is for. Only the update's shape follows the batch
(`MJX_PPO`: 32 steps a rollout, 4 minibatches, 5 epochs — rsl_rl's). A checkpoint
records its backend and resumes on the other one.

**The task is a port, checked line for line.** `rl.mjx_task` computes the same
145 observations (the anchor-rooted estimator with its gyro filter, SLAM-tracked
anchor and reach odometry; the nearest rungs), the same reward books, the same
success test and the same terminations as `climb_task`, reading every constant
from it. What had to change is structural:

- **Lessons are data.** A batch compiles one model, so the eighteen scene files
  become one robot with six rung slots and one extra wall slot, and each env
  writes its own lesson's rungs and wall into its copy of those geoms. The
  curriculum is unchanged in what it decides; it now moves the lesson draw weights
  of a jitted step, and a finished env resets inside that same step onto a lesson
  drawn from them — no recompile when a lesson unlocks.
- **A reset is a copy.** `ClimbTask.reset` hangs the robot from the keyframe and
  closes the fingers for 0.4 s, deterministically, so that runs once per lesson at
  start-up and a reset copies it (the SLAM noise is still drawn fresh).

**The robot is simplified, and the simplification was measured before it was
drawn** (`rl.mjx_scene`'s docstring has the table: which parts touch, how often,
how hard, under win17's policy). MJX's JAX backend has no box-cylinder collision,
runs mesh hulls slowly and has no `jointlimitfrc` sensor. So what can touch a wall
or the other hand is boxes round the CAD parts, walls are planes, rungs are
capsules — and what can touch a **rung** is capsules and balls only. That last rule
is a measured bug, not taste: MJX's capsule-box test separates only on the box's
face normals, so a 2.4 m rung passing a 12 mm hook plate at a slight angle
"penetrates" it. On win17's lesson-1 gait it invented 50–75 mm contacts where C
MuJoCo saw none, 5000–12000 N where C had 200, and took the policy's lesson 1 from
7/8 in C to 0/8 in MJX. Capsule-capsule is exact, so each hook plate became two
rods square to the rung and the finger motor a capsule along its axis; the step
also got 40 % cheaper. Collisions are an explicit list of 183 pairs with the
friction and softness MuJoCo would have mixed, so the grip pad still defines the
hook's contact. Masses, joints, actuators and the keyframe are untouched.

The solver is ~80 % of an MJX step, so its two caps are measured, not guessed:
**6 Newton iterations** (C uses 6–7 on every step to its 1e-8 tolerance; at 4 the
one-step contact error is 4x larger, at 2 it is tens of newtons) and **16
contacts** handed to the solver (C never had more than 9 at once, on any lesson).

**Cross-checked three ways** (`python -m rl.mjx_check`, ~15 min on this Mac):

| check | what it compares | result |
|---|---|---|
| `task` | ClimbTask on C MuJoCo, its state handed to the JAX task every step, float64 | 780 steps on lessons 1, 1.5, 7 incl. win17's gait: observations equal to 2e-15, reward to 4e-10, every book and termination the same |
| `physics` | the same simplified robot in C and in MJX, float32 | hang loads 138/138 N both; a 49 N pull slides the hook 19.2 mm in both; one control step from C's own state: joints 4e-6 rad off (p50), hook loads 0.08 N |
| `policy` | win17's best, deterministic, on the full robot, the simplified one in C, and MJX | 70 / 68 / 67 of 88 episodes (below). With box-shaped hooks MJX had scored 46, lesson 1 0/8 |

| lesson (8 episodes each) | full robot, C | simplified, C | simplified, MJX |
|---|---|---|---|
| −2 one left-hand step | 5 | 7 | 7 |
| −1, 0, 1, 1.1 | 31 of 32 | 31 | 32 |
| 1.2, 1.35 one gap | 14 | 13 | 12 |
| 1.5 two gaps | 7 | 4 | 6 |
| 1.7, 1.85 wide gap | 13 | 12 | 10 |
| 2 two wide gaps | 0 | 1 | 0 |

Eight episodes a lesson is a coarse ruler — the full robot's own column moves by
±2 between runs, from the SLAM noise alone — so what it shows is the same lessons
easy and the same ones hard on all three, not a ranking.

**Unmeasured, because there is no NVIDIA card here:** GPU throughput. On this
Mac's CPU the same batch runs ~870 control-steps/s at 256 envs, against ~6400 for
C MuJoCo in 12 processes (`rl.bench`) — which is why a machine without a GPU stays
on the CPU path. Run `python -m rl.mjx_task --envs 4096` on the card before a long
run, and treat the first GPU run as the canary the CPU path has always had: lesson
−2 passed by ~1M steps, or start again. `MJX_ENVS`, `MJX_PPO` (`rl.train_ppo`) and
`ITERATIONS` / `MAX_CONTACTS` (`rl.mjx_scene`) are the knobs, in that order.

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
| the pool's **steps go where the failures are** | a stuck lesson-1.5 episode runs 1490 steps against ~200 for a one-step lesson, so drawing by episode gave the newest lesson 96 % of the samples, and what it learnt while stuck was learnt in the states lesson −1 starts from: −1 fell to 0–2 % in three runs (mac14, mac16, mac17). Each lesson's draw probability is now its share of the steps divided by its mean episode length, and the share is how much it still fails — `max(1 − rate, MIN_SHARE)` — so a lesson at 0 % is fed whether it is the frontier or the one behind it that regressed. Four rules were measured on 2.5–4.5M-step runs: by episode (above); a fixed 25 % for the frontier, which starved it — lesson 1 got a quarter of the samples `win9` gave it and `win10` was still 0 % at 1.4M; a fixed 60 %, which reached **lesson 2 in 2M steps** (mac26, the best run there has been) but left lesson −1 at 0 % from 0.6M on; and this one, which held every lesson at 85–100 % through 4.5M (mac27) with a slower frontier. Squaring the need concentrates harder (66 % of the steps for a stuck frontier instead of 49 %) but starves the early curriculum while several lessons are half-learnt, and lesson 1 stalled for 2M (mac28). A slow frontier is fixed by more steps; a forgotten lesson is not |
| the SLAM **attitude** — measured, available, *not* used for training | measured on the mac20 policy *during the climb*: the triad-referenced filter was 4.0° p50 / 6.3° p90 / 15° max off (the triad itself 18–30°: every hand-over is an impact), which on the 0.5 m lever from the anchor to the hand was 35–55 mm. Referenced to the SLAM attitude instead it is 0.2° / 1–3°, the free hand's error falls from 36 / 80 mm to 13 / 40 on lesson 1.2, and a trained policy runs fine under it — but six fresh runs with it never learnt lesson −1 (the bisect table in the `win9` section), so the policy trains under the triad-referenced filter and `SLAM_TAU` 1.0 s, and `ClimbingEnv.slam()[1]` waits for a policy that has passed the gap lessons |
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

Checked (`mac20`, 4M, 8 envs): −2 100 %, −1 95 %, 0 92 %, 1 90 %, **1.1 at 98 %**
within 0.4M steps of its unlock (the left hook lands at 0.35–0.41 for a last seat
at 0.388), and lesson 1.2, unlocked at 3.7M, at 0/40 in training but with the climb
the user described on the replay after 0.3M steps of exposure: left hand to the
last seat (+0.32), right hand up beside it (+0.02), left across the gap (+0.39,
+0.47, +0.57), right following (+0.08, +0.20, +0.28) — one full success in 10, a
hand seated on the far segment in 3 of 12. Out of steps, not stuck — or so it
looked. Resumed to 7M: 1.2 still 0/40 and **lesson 1 down to 20–38 %**. On the
replay the climber's order is there in every episode (left to the last seat +0.32,
right up to +0.02, left over +0.44, +0.51, +0.60, right +0.10, +0.16, +0.25) — in
steps of 5–9 cm. The policy that had just learnt to land short took short steps
everywhere: lesson 1.2 asked for 1.4 m over two gaps and it managed 0.5 m in 60 s;
lesson 1 took 1490 steps for its 1.1 m and timed out too, the right hand re-seating
57 times per episode. A lesson that never passes keeps its warm-up share, so the
crawl kept teaching. Lessons 1.2 and 1.35 now ask for one crossing each — goals on
the segment just across the gap, 0.6 m of travel — and leave the two-gap traverse to
lesson 1.5.

Resumed to 10M with that: −2 92 %, −1 98 %, 0 98 %, lesson 1 back to **85 %**,
1.1 95 %, **1.2 at 90 %** (10/12 and 22/24 in two audits, 216 steps per crossing,
the climber's order in every episode: left to the last seat +0.35, right up to
+0.11, left over +0.39 → +0.48, right following +0.27), 1.35 unlocked at 12 %. The
user's read of the slow video — a hand grabbing the gap between the segments, a hand
reaching without lifting — asked why a robot with SLAM, IMU and encoders does that.
Measured at those moments: half is the policy (it *knew* the hook was in the gap,
16 mm of error, and lifted 14–23 mm, not enough to clear the end), and half was the
estimate, which during the climb — not on a still hang — was 36 / 80 mm off and put
the hand "on a seat" while it was in the gap 7.5 % of the time. Split into parts,
the free hand's error was the attitude (see the reward table): the IMU triad the
filter leaned on reads 18–30° under the impacts of a climb. Referenced to the SLAM
attitude instead: 0.2° / 1–3°, the hand 13 / 40 mm, the false seat 2.7 % — on
replay. What that did to training is the bisect below.

Resumed to 13M with that estimator: 1.35 unlocked and reached 28 % — and then
lesson 1 fell from 92 % to **5 %** in 2M steps, 1.2 from 90 to 52 %, the third
such swing in one run. Not the estimate (15 / 25 mm, 0.2° on this policy) and not
one lesson's exploit: on lesson 1 the right hand was off the wall 68 % of the time
taking the 40–66 mm lifts lesson 1.35 was teaching, and half the episodes fell.
Seven lessons of a 145-dim observation in SB3's default 64×64 network, updated on
2048 steps at a time (one to four episodes of a 60 s lesson), and the policy swings
between the lessons instead of holding them. The two obvious stabilisers were
tried and both killed the early curriculum: 8192 steps per update left lesson −1
at 0 % for 4M steps (`mac21`), a 256×256 network left lesson −2 at 0 % for 1.5M
(`mac22`), against 0.14M and 0.1M with the defaults. What is in now is the
standard PPO recipe the runs had been missing — the learning rate decays linearly
to zero over the run — so the early lessons learn at full speed and the late
updates, where the pool is large, are small; and the `best_` checkpoint keeps the
peak (mac20's, at 11.1M: −2 100, −1 98, 0 90, 1 92, 1.1 100, 1.2 90, 1.35 22 %).
And then the estimator that was right on replay turned out to be one a policy
cannot learn *from*: six fresh runs with the SLAM attitude and/or `SLAM_TAU` 0.3 s
never learnt lesson −1 (0 % for 1–4M steps), while the same code with the triad
attitude and 1 s passes it in 0.3–0.4M. The bisect, 1M steps each:

| estimator | lesson −1 |
|---|---|
| A: triad attitude, `SLAM_TAU` 0.3 | 0 % at 1M |
| B: SLAM attitude, `SLAM_TAU` 1.0 | 0 % at 1M |
| C: triad attitude, `SLAM_TAU` 1.0 | passed at 0.3M, lesson 1 by 1M |
| D: as B + a variance floor in VecNormalize | 0 % at 1M |
| E: as C + the floor | passed at 0.4M, **lesson 1.2 unlocked by 1M** |

A trained policy (mac19) solves lesson −1 12/12 under all of them with the same
reward; the reward is ground truth in all of them; the failing policies move the
left hand forward and dangle the right, lesson −2's behaviour. The variance floor
(1 cm; a feature whose std the estimator's accuracy had shrunk to 0.6 mm was
being normalised to the ±10 clip) was the first hypothesis and is not it, though
it stays, since E was the fastest early curriculum of any run. The mechanism is
open. The training estimator is C's; the SLAM attitude remains available in
`ClimbingEnv.slam()` and is the thing to try once a policy has passed the gap
lessons, the way mac20 took it at 10M.

With the sampling fixed, the curriculum finally runs: `mac26` (2M steps, 8 envs)
went −2 → **lesson 2** — 1.1 at 98 %, 1.2 at 98 %, 1.35 at 92 %, **1.5 at 78 %**
after fifteen million steps of 0 % in earlier runs, lesson 2 unlocked and at 20 % —
and on lesson 1.5 the audit is 11/12, both hooks seating on both far segments. It
was the run that left lesson −1 behind, which is what the shipped rule fixes.

Two more things the 13M replays showed,
both the policy's:
on lesson 1.2 the left hand sometimes seats on the very edge of the far segment
(0.40–0.45) and holds, which leaves no room for the right hand's goal at 0.45
behind it, and the pair waits out the episode; and on 1.35 the left hand reaches
across (83 % off the wall) without landing — the right hand up beside it first,
which it does on 1.2, it does not yet do on 1.35.

## `win11`: the best policy this project has had, and the run that ate it

15.3M steps on the pool through lesson 2, and the two checkpoints tell opposite
stories. At **8.0M** (`best_model.zip`): −1 98 %, 0 100 %, 1 98 %, 1.1 98 %,
**1.2 100 %**, **1.35 92 %** (10/10 in audit), **1.5 52 %** — a hand on both far
segments, the climb the user described, and the first policy to get there. At
15.3M: 1.5 and 2 at 0 %, 1.35 at 12 %, −1 at 28 %.

It is not undertraining, and the action std is the thread. 0.48 at the peak, 1.33
at the end, 4.24 on one joint — and the *deterministic* policy fails too (0/10 on
lesson 1.5, 2/10 on lesson 1), so what degraded is the mean, not the sampling.
Once most of the pool is failing, every action is about as bad as every other, the
advantage signal flattens, and `ent_coef` is the only term left with an opinion:
it pays for noise, the noisier policy fails more, and the loop ran for the last
7M steps. SB3 starts log_std at 0 (std 1.0) and every good run here came down from
it to 0.3–0.6, so nothing legitimate needs more: `BoundExploration` clamps log_std
to at most its starting value after every rollout. Asserted (std forced to 4.06,
back to 1.00 after one update), and checked by resuming the 8.0M checkpoint with
it for 1.2M steps: std 0.46–0.48 with the cap binding at 1.00, lesson 1.2 at
100 %, 1.35 at 90–98 %, 1.5 oscillating 20–52 %, lesson 2 unlocked again, and no
collapse. The swings between lessons are still there; the spiral is not.

After a run has degraded, `--resume --from-best` continues from `best_model.zip`
rather than from the wreck.

## `win11 2`: the cap holds, the pool does not, and a crossing nothing had taught

Four million steps resumed from win11's 8.0M peak, with `BoundExploration`. It
worked as designed — std 0.46 → 0.52 with the cap binding at 1.00, no spiral — and
the run still did not improve: the best score stayed at 7.30 for the whole 4M and
lesson 1.5 went 55 % → 8 %, lesson 0 100 % → 40 %. What is left is interference:
nine lessons through one 64×64 network, and what one lesson gains another pays.
Narrowing the pool to the four that matter (`--resume --lessons 1.2,1.35,1.5,2`)
for 1.5M steps holds them instead — 1.2 95 %, 1.35 98 %, 1.5 oscillating 25–60 %
and ending at 57 % rather than collapsing.

Lesson 2 stayed at 0 % throughout. A crossing is a reach from the last SEAT of one
segment to the first seat of the next: 185 mm on lesson 1.5 and **315 mm on lesson
2**. The policy that has to attempt it never produces a step that long — 447 hops
measured on the 8.0M checkpoint, median 43 mm, p90 185 mm, longest 300 mm, and
**not one** at 315 mm. That is a statement about the gait it has learnt, NOT about
what the robot can do: the arms span 671 mm, the hands would be 435 mm apart on
landing, and pulling up on the holding arm, shifting the torso across, or swinging
the body all extend the reach further. (An earlier version of this section called
lesson 2 geometrically blocked. It is not; the user was right to reject that.) The
lesson is that a curriculum step should sit inside the distribution the current
gait actually samples, which is what lessons 1.2 and 1.35 did below it — they went
from never to 90–100 %. Lesson **1.7** (one 0.18 m gap, a 245 mm crossing, made by
1–2 % of this policy's steps) now sits between 1.5 and 2.

## The elbow pointed inward, and why the hook was the reason

The user watching a pull-up: the elbows fold *inward*, across the body, where a
person's go out or forward. Measured on the 8.0M policy over the steps where a hand
is loaded and the torso is rising (lesson 1.35): the right elbow sits **44 mm
inboard** of the shoulder-to-wrist line on average, 107 mm at p10, inward on 68 %
of steps — and the shoulder rotation reads **±90°, its own limit, on every one of
them**.

That last number is the thread. Raising the arm overhead is 160° of shoulder
flexion, which carries the hook round with it until its throat faces away from the
wall; bringing it back takes **175° about the arm's axis**. The forearm roll has
±85° (a human's pronation/supination, and correct), so the other 90° has to come
from the shoulder rotation, which is therefore pinned at its stop. And the shoulder
rotation is exactly what aims the elbow's hinge: at ±90° that hinge points at the
wall, so flexing the elbow sweeps the forearm sideways and the elbow bulges toward
the midline. Searching every posture whose hook orientation is within 15° of the
one that latches, the most outward elbow available was still 88 mm *inboard*: with
that mounting the human posture did not exist.

The fix is the mounting, not the joint limits: the hook bodies are rotated 180°
about the forearm axis (`quat="0 0 0 1"`), and the hang keyframe recomputed —
shoulder rotation 0° and forearm roll ∓5° reproduce the old hook pose to 0.2°, with
the throat still on the rung to 2 mm. Everything the code reads in hook-local
coordinates (`THROAT_PT`, `THROAT_X/Z`, the pads, the finger) rides along with it
and is unchanged. After it:

| | before | after |
|---|---|---|
| elbow hinge, in the torso frame | (0.94, 0, −0.34) — pointing at the wall | **(0, 1, 0)** — lateral, like a human's |
| flexing the elbow 120° moves it | 114 mm **inboard** | 172 mm **forward**, 8 mm sideways |
| shoulder rotation while hanging | ±90°, at the stop | **0°**, mid-range |

and the policy can now choose: rotation 0 puts the elbows forward, −30 to −60 puts
them out to the sides. Both are postures a climber uses. A fresh 1.5M-step run on
the fixed model unlocks −2 → 1.2 at the usual pace, so nothing about learning
changed.

**What it costs:** `controller/test_climb` went from failing one phase (8) to
failing five (3, 4, 5, 6, 8). The classical controller's servo weights were swept
empirically against the old arm posture — its own comment records the sweep — and
the new posture changes the Jacobian. Three values of `W_ALONG_COARSE` were tried
and none converged, so this is a re-tune of that stack, not a constant. The RL
side is unaffected and its own checks all pass.

## `win13`: 7.6M steps on the robot as it was before the fix

The run looked healthy — best at 3.79M with −1 100 %, 0 98 %, 1 95 %, 1.1 98 %,
**1.2 98 %**, 1.35 88 %, 1.5 22 %, faster than any run before it — and then every
replay of that checkpoint fell 12/12, on lessons it was scoring 98 % on. The
checkpoint was not the problem. `urdf/lessons/*.xml` is generated and gitignored,
`xml_path` wrote a scene only when the file was **missing**, and those files were
already on disk from the previous run. Pulling the re-mounted hook changed
`scene.xml` and nothing regenerated them: the run trained 7.6M steps on the old
robot, elbows and all. Replaying the same checkpoint against the pre-fix scene
scores **12/12** on lesson 1.2 and confirms it.

The scenes now carry a stamp — a hash of `scene.xml`, the lesson's own spec and
the patch lists — and `xml_path` rewrites any file whose stamp does not match, so
a model edit cannot silently fail to reach training again. Worth saying plainly:
the only reason this was caught is that a checkpoint was replayed and the numbers
disagreed with the training log. A generated artefact that is only checked for
existence is a place where a fix goes to die.

## `win15`: the fixed robot climbs, and the pool is what holds it back

11.5M steps on the re-mounted hook with the wrist back at ±25°, and the model
edits did what they were for. In the learnt gait on lesson 1.2 the shoulder
rotation sits at −1° (left) and +58° (right) — mid-range, where the old mounting
pinned it at ±90° — the elbows point outward on **100 %** of loaded, rising steps,
and the wrists sit at +21° / +8° with no jamming (`win14`'s right wrist had lived
at its +80° stop). Replaying the best checkpoint: lesson 1.2 **10/12**, lesson
1.35 **10/12**, lesson 1.5 2/12 with both hooks reaching the second far segment
and a hand closing to 40 mm of its goal. On 1.5 the seats alternate the way the
user described — left +0.29, right +0.20, left +0.48, right +0.24, left +0.57 —
so the technique is there; the finishes are not.

What the run also shows is where the ceiling is. The best checkpoint is at 6.78M
and **4.7M further steps did not beat it**: lesson 1.5 went 10 % → 8 %, lesson −1
fell to 48 %. Resuming that checkpoint on the hard lessons only
(`--lessons 1.2,1.35,1.5,1.7,2`) moved 1.5 from 10 % to **22 % in 0.8M steps**
before swinging back to 2 % — eight lessons through one 64×64 network is the
limit, not the number of steps.

That experiment also found a hole in the safety net. `score` sums the success
rates over the **active** lessons, so a best of 6.35 earned over eight of them can
never be beaten by three: `best_model.zip` was frozen for the whole narrowed run
and the 22 % policy was thrown away. Narrowing the pool on a resume now resets the
bar, so the peak of a narrowed run is kept.

## `win15 2`: the narrowed pool doubles lesson 1.5, then 7.7M steps eat it

Resumed from `win15`'s best (6.78M) with the pool cut to 1.2 / 1.35 / 1.5, 9M
further steps, ending at 15.77M. The best-score reset works: `best_model.zip` was
written this time, at **8.03M**, scoring 2.475 over the three lessons — 1.2 at
100 %, 1.35 at 100 %, **lesson 1.5 at 47 %**, against 10 % when the eight-lesson
pool was training it. Replaying that checkpoint: 12/12, 11/12, **5/12**.

Then it went backwards. 7.7M steps after the peak lesson 1.5 reads 17 % and
replays **1/12** — six of the twelve now time out. The mechanism is in the action
std: at the peak the mean is 0.469 with no dimension at the `BoundExploration`
ceiling; at the end it is 0.610 with **3 of 17 dimensions pinned at 1.0** and ten
above 0.8. The cap stops `win11`'s explosion to 4.24, it does not stop the drift,
and the policy degrades as it drifts. `ent_coef` is the term pushing it there.

The 53 % that fail at the peak fail one way, and it is not a bad grab. The
**trailing hand plants and will not let go**: over six episodes the right hook is
loaded and rising for 1683 steps against the left's 399. In five of eight episodes
the left hand walks out to full span and then re-lands on the *same seat* over and
over — +0.43 four times, +0.60 six times — while the right sits at −0.11 on the
start rung under 146–162 N with the finger closed, for all 1490 steps. The
successes look completely different: left +0.18 +0.29 +0.44 +0.60 +0.73 +0.91
+1.04, right −0.10 +0.19 +0.36 +0.43 +0.45 +0.64 +0.77 — proper hand-over-hand.
The policy is bimodal, not confused. The two outright falls both happen at the
last gap, at `rung_seg2` y ≈ 1.03–1.26, with the trailing hook catching the edge
rather than a seat, 42–49 mm off the rung axis.

The obvious suspect for a freeze is that timing out had become the cheap exit —
`win5` all over again. Measured on the peak checkpoint over 14 episodes, it has
not: success pays **+16.6**, a timeout **−17.0** and a fall **−18.8**. Freezing is
worth 1.8 points more than falling and 34 less than climbing. The shaping is doing
its job; what is missing is that the policy will not spend the exploration to
unweight a 160 N anchor. That is an optimization problem, so the next run turns
`--ent-coef` down to 3e-4 rather than touching a reward constant.

The model edits still hold at 15.77M: elbows outward on **100 %** of loaded rising
steps on both arms, shoulder rotation −43° / +64°, and no wrist within reach of
either stop under load (0.3–5.6 % of steps past ±25° while loaded).

## `win17 2`: the log names the drift — the action noise collapses and every update grows with it

Resumed from the 9.87M peak with the guard and the 70 % gate, to 20.0M, with the
console kept for the first time. The rules did their jobs. At 3e-4 the pool
drifted again inside 1.7M steps (lesson 1.5 62 % → 15–45 %); the guard went back
to the peak at 12.77M and halved the rate. From there nothing was forgotten:
1.85 reached 70 %, lesson 2 unlocked at 15.4M, and the best rose to 9.38 over all
eleven lessons at 19.44M. Deterministic, 20 each:

| lesson | 9.87M | 14.8M (old run) | 20.0M |
|---|---|---|---|
| −2 … 1.35 (seven lessons) | 134/140 | 130/140 | 138/140 |
| 1.5 | 13/20 | 11/20 | 12/20 |
| 1.7 | 20/20 | 15/20 | 16/20 |
| 1.85 | 12/20 | 12/20 | **17/20** |
| 2 | 0/20 | 2/20 | 1/20 (12–25 % sampled) |

The log is what the checkpoints could not show. **Every update was rewriting the
policy**: approx_kl 0.2–1.1 per update and a clip fraction of 0.5–0.8, against
the 0.01–0.03 and 0.1–0.2 of a usual PPO run. KL climbed in step with the policy
entropy, which fell in a straight line and never levelled: +2.5 → −3.5 nats before
the rollback, +2.4 → −7.7 after it. The halved rate slowed the slide and did not
stop it. The cause was in the checkpoint: six action dimensions had collapsed to a
std of 0.02, the right shoulder's three joints, its forearm roll and finger, and
the left wrist deviation. They were 0.09 at the peak. A Gaussian's KL goes as
1/std², so a tiny change to those means reads as an enormous update. Most samples
then hit the clip, and the gradient comes from the few that do not. By 20M the run
sat at KL ~1, the regime that had come just before the drift at 12M.

The fix is the one robot-learning PPO uses (`train_ppo.py`, both in the
self-check):

- **a floor on the action std**, 0.05 (`STD_MIN`), on top of the existing 1.0
  ceiling. That is 0.003 rad of setpoint noise per step. Sampled with the floor,
  the 19.44M policy did as well as without it: 1.5 14/20 against 13/20, 1.85
  14/20 against 10/20.
- **a KL-adaptive learning rate** (`AdaptiveLR`, rsl_rl's rule): ÷1.5 when an
  update's KL passes twice the run's KL target, ×1.5 under half, never above 3e-4.
  The target starts at `KL_START` 0.1 — where `win17` spent 2.5M steps improving
  and forgetting nothing — and **the guard halves it on every rollback**, down to
  `KL_MIN` 0.01. Small updates are what stops the forgetting and they are also
  what makes a lesson cost millions of steps, so a run takes the biggest step
  that has not yet cost it a lesson instead of a small one forever. (The guard's
  `LR_DECAY` alone no longer sticks: with an adaptive rate, a halved rate is
  pulled back up within a few updates. The target is the knob that lasts.)

Resumed from 19.44M on this Mac, the floor lifted the entropy from −7.5 to −3.3
and it stayed flat. At the 0.1 target the rate settles at 6.7e-5 with KL 0.09–0.12
and a clip fraction of 0.49; at 0.02 it would settle at 2e-5, KL 0.03, clip 0.25.
Over 0.5M steps at the tighter of the two nothing slipped — lesson 1 92 %, 1.5
60 %, 1.85 68 %, lesson 2 25 %, no guard trip. On a fresh run KL is 0.017 at
3e-4, so the rate never moves there and the early curriculum trains as before.

**Next:** resume from the best and leave it:

```
python -m rl.train_ppo --name win17 --resume --from-best --lessons=-2,-1,0,1,1.1,1.2,1.35,1.5,1.7,1.85,2 --envs 12 --steps 30000000
```

### How long the rest of the curriculum takes, and what makes it shorter

Measured on the user's Windows box: **1484 steps/s** with 12 envs, 10.1M steps in
2.0 h. So 50M steps is 9 h, 100M is 19 h, 200M is 37 h. The worry that this needs
billions of steps does not survive the arithmetic — the risk is a lesson that
stalls, not the clock. For scale, `win16` went from scratch to lesson 2 at 57 % in
5.75M steps on the old robot; `win17` needed 20M to get there because the drift ate
its progress three times over and the pitch-first arm has 0.56 m of reach instead
of the 0.68 m that let the old one lunge.

Three levers, in the order they are worth pulling:

1. **The update size, which now tunes itself.** The guard only tightens `KL_TARGET`
   when the pool has actually lost a lesson, so the run keeps the largest step it
   can get away with. Nothing to do.
2. **Network capacity.** Every run here has used SB3's default 64×64 for both nets,
   and eleven lessons through it is the standing suspect for the interference the
   guard keeps catching — rsl_rl's locomotion policies are 512-256-128. `--net
   256,256` is nearly free in wall clock, because the simulator is the bottleneck
   (cutting 90 % of the gradient work moved throughput by 0 %). It needs a fresh
   run, so it is an arm to put beside `win17`, not a change to it.
3. **Throughput.** More envs first (`python -m rl.bench` on that machine — the Mac
   gained 20 % from 12 → 20 envs). Then the real jump: MJX runs the physics on the
   GPU with thousands of envs in parallel and is where a 10–100× would come from.
   This model is two edits away from being MJX-ready (capsule collision geoms, drop
   the `jointlimitfrc` sensors), and it needs Linux or WSL2 with CUDA.

## `win17`: the pitch-first robot climbs, stalls at lesson 2, and runs now look after themselves

The first run on the rebuilt shoulder. At 14.8M its best checkpoint was 5M steps
back (9.87M, 9.03 over lessons −2…1.85), and those steps had bought nothing:

| lesson (deterministic, 20 each) | best 9.87M | latest 14.8M |
|---|---|---|
| −2 one left-hand step | 20/20 | **13/20** |
| −1 … 1.35 | 18–20/20 | 19–20/20 |
| 1.5 two 0.12 m gaps | 13/20 | 11/20 |
| 1.7 one 0.18 m gap | 20/20 | **15/20** |
| 1.85 one 0.25 m gap | 12/20 | 12/20 |
| 2 two 0.25 m gaps | 0/20 | 2/20 |

The gait is honest. Every hook leaves the rung (0 N) before it moves and seats
again with the finger in band (−33°), the holder does not slide, one hook always
carries load, and lesson 1 takes six grabs a hand. It is fast, 0.3 s a step, and
hard on the hooks: 480–640 N at the catch, 2.3–3× body weight.

Lesson 2 stalls at the deadlock described below: the lead hand reaches while the
trailing one is still ~0.2 m back from the edge, the hands span 0.46–0.49 m, and
the far segment stays out of reach. The robot can make that crossing — lesson 1.85
is the same gap and passes 60 % — the policy has not learnt to walk up first.
Meanwhile lesson 2 held ~30 % of the steps at 2 %, and the pool began to forget: on
lesson −2 the latest policy sometimes moves the right hand too, walks the whole
rung like lesson 1, and falls. It is not the action noise. Waist, left finger,
right elbow and left wrist flexion sit at or near the 1.0 cap, but sampling with
those four turned down to 0.2 did no better (lesson 1.5: 8/20 against 12/20).

**Two rules, so this needs no one watching** (`train_ppo.py`, both in the
self-check):

- The next lesson unlocks when **every** lesson in the pool is at 70 %
  (`UNLOCK_AT`), not the newest at 60 %. `win17` unlocked lesson 2 with 1.5 and
  1.85 at ~60 %.
- **The guard.** If the best checkpoint's own lessons score a whole lesson below
  that best (`GUARD_DROP`) for a million steps (`GUARD_STEPS`), the run goes back to
  the best checkpoint — policy and optimizer, statistics, curriculum — and carries
  on at half the learning rate (`LR_DECAY`, never below `LR_MIN`, 1e-5). A lesson
  unlocked since the best does not count, so a new lesson's first failures cannot
  trip it. Resumed on a copy of `win17` with the guard shortened, it went back to
  the 9.87M state, left lesson 2 parked (1.5 and 1.85 were under 70 % there) and
  carried on at 1.5e-4. Update size is the untested suspect for every drift since
  `win11`; halving it on each rollback is the experiment `win16 3` asked for, run
  by the trainer instead of by hand.

Every run also appends its console to `out/<name>/train.log`, with the
`approx_kl`, `clip_fraction` and `explained_variance` that question has been
waiting on.

**Next:** resume from the peak and leave it. Lesson 2 comes back once 1.5 and 1.85
are both at 70 %:

```
python -m rl.train_ppo --name win17 --resume --from-best --lessons=-2,-1,0,1,1.1,1.2,1.35,1.5,1.7,1.85,2 --envs 8 --steps 30000000
```

## `win16 3`: the same decay a third time, and three 1M-step arms that all lose ground

Resumed from `win16 2`'s peak with the new −37° finger stop. The best checkpoint
landed at **6.098M — 5k steps after the resume** — and the 4.2M steps after it went
the same way as the two runs before:

| lesson | best 6.098M | latest 10.33M |
|---|---|---|
| 0 | 9/10 | 9/10 |
| 1 traverse | 6/10 | **10/10** |
| 1.2 hairline gap | 9/10 | 6/10 |
| 1.5 two 0.12 m gaps | 6/10 | 2/10 |
| 2 two 0.25 m gaps | 3/10 | **0/10**, 7 timeouts |

The grip fix was in effect and did what it was for: the stop reads −37°, the loaded
finger sits at −34° (p50), and holds count as holds on **99–100 %** of loaded steps
at the peak against 40–70 % before. The collapse happened anyway, which is what the
commit that made the fix already said it could not rule out.

**The decay is selective, in both of the last two runs.** Lessons with no gap hold or
improve — lesson 1 *ended* at 10/10 — while every lesson with a gap rots. The
collapsed gait is the same freeze: on lesson 1.5 the left hook carries load on 29 %
of steps and keeps landing at y 0.42–0.43, in the gap between seats.

**Three arms, 1M steps each, from that same 6.098M peak.** One keeps the learning
rate at 3e-4, one drops it to 5e-5, one keeps 3e-4 but narrows the pool to the six
gap lessons:

| replay after 1M | start | lr 3e-4 | lr 5e-5 | gap lessons only |
|---|---|---|---|---|
| 1 traverse | 6/10 | 9/10 | 7/10 | — |
| 1.2 | 9/10 | 8/10 | 8/10 | 9/10 |
| 1.5 | 6/10 | 7/10 | 7/10 | 7/10 |
| 2 | 3/10 | 3/10 | 6/10 | 2/10 |

Three things come out of it. **1M steps cannot see this failure**: no arm collapsed,
and all three sit at or above the checkpoint they started from, while `win16 3` took
4.2M steps to reach 0/10. So the lr question is still open — 6/10 against 3/10 on
lesson 2 is one arm of ten episodes, not a result. **Narrowing the pool did not
protect anything**: lesson 2's logged rate fell 80 % → 48 % and lesson 1.5 72 % →
48 %, so `win15 2`'s narrowed-pool lift did not replicate. And **every arm reached
its best score within 2–6k steps of resuming and never beat it again** — gapC's best
is stamped at step 6,097,988, six steps in. Counting the user's runs, that is now six
resumes in a row where continuing from a peak buys nothing.

One more detail worth keeping: the logged (stochastic) rates fall inside 1M steps in
every arm while the deterministic replays hold. The training metric goes first and
the mean policy follows, which is why a run looks broken in `rl.status` long before
a replay agrees.

**Next:** the experiment has to match the timescale. From `win16 3`'s best, two arms
of **4M steps**, 3e-4 against 5e-5, on the machine that can run them, and this time
keep the console output — `approx_kl`, `clip_fraction` and `explained_variance` per
update are the one piece of evidence a checkpoint cannot reconstruct:

```
python -m rl.train_ppo --name w16lrA --resume --from-best --lessons=-2,-1,0,1,1.1,1.2,1.35,1.5,1.7,1.85,2 --envs 8 --steps 4000000 > w16lrA.log 2>&1
python -m rl.train_ppo --name w16lrB --resume --from-best --lessons=-2,-1,0,1,1.1,1.2,1.35,1.5,1.7,1.85,2 --envs 8 --steps 4000000 --lr 5e-5 > w16lrB.log 2>&1
```

Copy `win16 3/best_*` into `out/w16lrA` and `out/w16lrB` first, so both arms start
from the same policy. Until that says otherwise, the policy to keep and to show is
`win16 3/best_model.zip`.

## `win16 2`: without lesson 3 it still collapses, and a finger stop with no margin

This run resumed from `win16`'s 5.75M checkpoint with the pool capped at lesson 2,
so lesson 3 never entered it. The best checkpoint came 343k steps in, at **6.09M**,
and is the best policy yet: lesson 2 at 78 % logged and 7/10 in replay, lesson 1.5
at 7/10. By 11.36M the gap lessons had collapsed anyway:

| lesson | best 6.09M logged | best replay | latest 11.36M logged | latest replay |
|---|---|---|---|---|
| 0 | 92 % | 10/10 | 92 % | 9/10 |
| 1 traverse | 90 % | 9/10 | 52 % | 7/10 |
| 1.2 hairline gap | 100 % | 7/10 | 18 % | 2/10, 5 timeouts |
| 1.5 two 0.12 m gaps | 72 % | 7/10 | 0 % | 0/10, all timeouts |
| 2 two 0.25 m gaps | 78 % | 7/10 | 0 % | 0/10, all timeouts |

That settles what the `win16` A/B left open: **lesson 3 is not the cause**. Nor is
the input normalisation this time. No input drifted by more than 0.26 sd, the best
model with the latest statistics still passes lesson 1.5 4/5, and the latest model
with the best statistics passes 0/5.

**The collapsed policy freezes.** On lesson 1.5 the right hook carries the load on
100 % of steps and lets go 0–2 times an episode. The left hook carries it on 0–4 %:
it slaps at y 0.33–0.54, the edge and gap between `rung_start`'s last seat (0.32)
and `seg1`'s first (0.50), and never seats. The critic is flat at −4.4. The same
freeze is already present at the peak, in 1 episode of 8. On those left-hand touches
the finger is **open**: +6.5° on average against −32.5° at the peak, in band on 11 %
of loaded steps, and left-hand grabs fell from 14.6 to 0.2 per episode. The finger
is a symptom, though. Driving only the latest policy's left finger with the peak
policy's command brings left grabs back (0.4 → 7.2) but not one success in 5, and
giving the peak policy the latest's left finger still passes 4/5.

**What the collapse did expose is a real defect in the hold test.** The finger
setpoint stopped at −40°, exactly on `GRIP_BAND`'s closed edge. The table above says
that makes a full squeeze count as a grip, but there was no margin for it. A hook
whose setpoint sits on the stop reads 0.5–2.5° past it whenever the rung is not
pressing the finger open. On the collapsed policies the right hook read −41 to
−42.5° on 71–78 % of loaded steps, with 93 N of contact and 0.69 mm of slide per
step against 97 N and 0.91 mm in band. Those are real holds, yet `_holding`,
`_confirmed` and `_touching` all said no: no grab, no bank, no success, no anchor.
Even the peak policy lost 18 % of its left hook's loaded steps on lesson 2 this way.
It reproduces with no policy at all: a 60 N push along the rung hands one hook's
weight to the other, and the unloaded hook on a −40° stop leaves the band on 20
steps out of 300.

The stop is now **−37°**, still deep in the >1200 N band. The same push reads 0
steps out of band at −38° and at −37°, and a new self-check holds it there. With the
new stop, the collapsed policies' loaded holds count 100 % (up from 40–70 %). The
peak policy goes 5/8 → 6/8 on lesson 1.5 and 7/8 → 5/8 on lesson 2, within noise
either way. The fix does not revive a collapsed policy, which still succeeds 0–1
times in 5. It corrects what the reward and the map see; it has not been shown to
prevent the drift.

**Still open:** why every long run from a peak (`win15 2`, `win16`, `win16 2`)
drifts into a failure mode that was already a minority at the peak. It is not a
lesson, not the normalisation, and not entropy: the left finger's std is 0.07. The
untested candidate is update size, a constant learning rate of 3e-4 with no KL limit,
applied for millions of steps to a policy that is already good. SB3's console log
records `approx_kl`, `clip_fraction` and `explained_variance` on every update, and
that is the evidence to look at next.

**Next:** resume the 6.09M checkpoint with the new stop and lesson 3 still out. Keep
the run short, so the peak is checked before it can drift far:

```
python -m rl.train_ppo --name "win16 2" --resume --from-best --lessons=-2,-1,0,1,1.1,1.2,1.35,1.5,1.7,1.85,2 --envs 8 --steps 3000000
```

## `win16`: from scratch to the best policy yet in 5.75M steps, then 9.5M steps undo it

A fresh run on the full pool with lesson 1.85 in it. Its best checkpoint, at
**5.75M**, is the best policy this project has had — and it got there in a third of
the steps `win15` needed. The 9.5M steps after it destroyed it:

| lesson | best 5.75M, logged | best, replay | latest 15.26M, logged | latest, replay |
|---|---|---|---|---|
| −1 step right hand | 100 % | 10/10 | 38 % | 0/10, 7 timeouts |
| 1 traverse | 82 % | 8/10 | **3 %** | 0/10, all timeouts |
| 1.85 one 0.25 m gap | 82 % | 10/10 | 20 % | 1/10 |
| 2 two 0.25 m gaps | **57 %** | **5/10** | 0 % | 0/10 |
| 3 climb up | not unlocked | 0/10, 9 falls | 0 % | 0/10 |

The replays are deterministic, so the mean policy degraded, not the sampling. On
lesson −1 the latest policy ends 0.69 m from its goal against 0.30 m for doing
nothing at all. This is not under-training: the best checkpoint is 9.5M steps
behind the end and nothing after it came close.

**What changed between them is lesson 3.** The best checkpoint's pool stops at
lesson 2; lesson 2 crossed 60 % shortly after and unlocked `climb_up`, the first
lesson with a goal *above* the robot. Two mechanisms looked responsible.
A 600k-step A/B from the best checkpoint says neither explains the collapse.

**1. On lesson 3, letting go became the rewarded action.** The critic of the best
checkpoint, read along lesson-3 episodes: every state where the robot is still
holding on is worth **−18.5**, the fall is **−10**. A release that ends in a fall is
therefore scored at an advantage of about **+8.5** — PPO is paid to let go. The
per-term split of a lesson-3 episode says why: an attempt that does not fall pays
−58 to −118 over its 90 s, mostly loaded-hook slide (−20), grip fight (−12) and
effort (−11), while a fall pays −25. Raising `FAIL_R` would not have fixed this. On
a lesson the policy cannot yet pass, falling now instead of falling later saves
exactly the costs it would pay before falling. That saving does not depend on the
size of the fall penalty. The incentive self-check never saw it because it only
checks lesson 1. Hanging perfectly still to the timeout already costs **−11.96 on
lesson 3 and −16.0 on lessons 4–9**, all past `FAIL_R`, because every lesson from 3
on runs 90–120 s instead of 60.

**2. The input statistics drifted, but not the way it first looked.** `VecNormalize`
keeps one running mean and variance per observation, shared by all lessons. Between
the checkpoints only inputs 93 and 96 — the vertical offset from each hand to its
goal — moved by more than one standard deviation, their sd stretched 4× by lesson 3.
That looked like the culprit, and it is not. Swapping the statistics block by block
on the best model (deterministic, 8 episodes each):

| normalisation | lesson 1 | lesson 2 |
|---|---|---|
| best, as trained | 6/8 | 4/8 |
| latest, all of it | 3/8 | 0/8 |
| best, only goal-z from latest | **8/8** | **5/8** |
| best, layout block (91–142) from latest | 3/8 | 6/8 |
| latest, layout block from best | 6/8 | 2/8 |

The goal-z stretch on its own changes nothing. The damage is spread thin: the
layout block costs lesson 1 and the proprioceptive block costs lesson 2. A swap also
drops 9.5M steps of drift onto a frozen policy at once, where training lets the
policy follow it. The latest model with the best statistics still fails lesson 1
(0/8), so the policy degraded on its own. Pinning one block of statistics would
not have protected this run.

The collapsed gait is not a freeze. On lesson 1 the right hook lets go **70.8 times
an episode** against 6.5 at the peak, and the left carries the load 77 % of the
time. It is a flutter that times out. The action std went 0.359 → 0.477, with 4 of
17 dimensions pinned at the cap against none.

**What a controlled test says.** Both mechanisms came from snapshots, so both were
tested from the same 5.75M checkpoint: 600k steps per arm, lesson 3 in the pool (it
unlocked early in both arms). The fix tried for mechanism 1 charged a fall the
discounted cost of the steps it skipped, `FAIL_R − cost_ema · 1/(1−γ)`. On paper
that makes the timing of a fall irrelevant at any cost rate.

| after 600k steps | control, reward unchanged | with the fall charge |
|---|---|---|
| lesson 1 replay | **10/10** | 3/10, 7 falls |
| lesson 2 replay | 3/10 | 6/10 |
| lesson 1.5 in training | 25 % | 60 % |
| lesson 1: free hook more than 150 mm above the rung | 0 % | 38.6 % of free-hand steps |
| lesson 3 critic: still holding vs a fall | −3.9 vs −10 | −8.2 vs −19.3 |

The unchanged reward climbed out of the trap on its own. Its critic went from −18.5
to −3.9 per holding state, above the fall, within 600k steps, and nothing collapsed:
lesson 1 stayed at 10/10. The −18.5 at unlock was a transient of first exposure,
not what took lessons 1 and 2 apart over the following 9M steps. The charge helped
nowhere it could be shown. It doubled the cost of a fall on lessons that already
worked (−21.0 on lesson 1 against −10). In its run lesson 1 fell 7 times in 10 and
the free hook reached up past the rung on 38.6 % of steps, which the control never
did. One run per arm cannot call that harm, but it is not enough to ship. The
change is parked on the local branch `fall-charge`.

So what undid `win16` is still open. It is not an immediate effect of unlocking
lesson 3, because the control held every flat lesson for 600k steps after the
unlock. It is not the goal-z stretch. It took millions of steps. Three candidates
fit and are untested: twelve lessons through one 64×64 network (the swings
`train_ppo` already documents), lesson 3 sitting at 0 % on 15 % of the steps for
9M steps, and the action std drifting to the cap (4 dimensions pinned by the end).

**Next:** keep lesson 3 out of the pool and push lesson 2 up from the 5.75M
checkpoint. The pool matches that checkpoint's active set, so the 9.775 bar stays.
On Windows the pool must be written with `=`:

```
python -m rl.train_ppo --name win16 --resume --from-best --lessons=-2,-1,0,1,1.1,1.2,1.35,1.5,1.7,1.85,2 --envs 8 --steps 10000000
```

The experiment that would settle lesson 3 is the long version of the A/B above: the
same checkpoint with and without lesson 3 in the pool, several million steps each,
watching lessons 1 and 2.

## `win15 3`: lesson 2 crossed, and a flag that was never doing anything

10M steps from `win15 2`'s best (8.03M), ending at 18.03M with the best at
**16.84M**. It is the furthest this project has got, by a distance:

| lesson | logged | replay of the best |
|---|---|---|
| 1.2 hairline gap | 97 % | 10/12 |
| 1.35 narrow gap | 85 % | 10/12 |
| 1.5 two 0.12 m gaps | **60 %** | 7/12 |
| 1.7 one 0.18 m gap | **88 %** | 7/12 |
| 2 two 0.25 m gaps | **35 %** | **3/12** |

Lesson 1.5 finally cleared the 60 % bar, which unlocked 1.7, which cleared it too
and unlocked **lesson 2 — passed for the first time in this project**. It is worth
being precise about what that claims, because `_success` is strict: both hooks
seated on a *goal* rung, each within `LATCH_TOL` = 150 mm of its own goal point,
both holding under contact force, and each hand having taken at least one real
hold along the way. On lesson 2 the goal rung is `rung_seg2`, so a success is both
hands across **two 0.25 m gaps** and seated at the far end. There is no version of
that a ratchet or a one-handed drag can fake.

**The freeze is breaking.** `win15 2` failed lesson 1.5 by parking — the trailing
hook planted under 146–162 N while the lead hand re-landed on the same seat for
1490 steps. Measured on lesson 1.5, best against best: steps with **both hands
confirmed** went 12 % → **22 %**, anchor switches per episode 10.8 → 12.8, and
timeouts went 1 → **0** deterministic and 3 → **0** sampled. The policy stopped
parking and started attempting; some attempts fall, which is why lesson 1.5's
losses moved from timeouts into falls. That trade is the progress.

The estimator looks worse and mostly is not. Mean throat error on lesson 1.5 went
42 mm → 75 mm and the per-episode peaks 441 mm → 940 mm, which the end-of-episode
snapshots make look alarming — but the peaks are the fall itself, not its cause.
The number that decides a fall is whether the FREE hand's estimate puts it over a
seat when it is really over an edge, because that is the step where the robot
commits its weight. That rate is **flat: 4.1 % → 3.9 %**. The growth is all in the
harmless direction — a real seat read as an edge, 1.4 % → 3.4 % — which costs
hesitation, not a fall. The estimator is not what stands between 35 % and lesson 2.

**What does need fixing is that the last run's one experiment never ran.** The plan
after `win15 2` was `--ent-coef 3e-4`, to relieve an action std pressed against the
`BoundExploration` cap. Reading it back out of the checkpoint: `ent_coef = 0.001`.
`PPO.load` restores the saved value and only the *fresh* branch ever read the flag,
so on a resume `--ent-coef` did nothing at all, silently, for 10M steps. So the
whole of `win15 3`'s gain is more steps on the narrowed pool, and the entropy
hypothesis is still untested — std sits at 0.621 at the peak and 0.634 at the end,
with 3 and 5 of 17 dimensions pinned at the cap. The flag now applies on a resume
and the resume line prints the value it will actually train with.

One thing that did not happen this time: the run did not eat its own peak. The
latest checkpoint is 1.2M steps past the best and still within a few points of it,
against `win15 2`'s 7.7M-step slide from 47 % to 17 %.

## Why the lead hook crosses before the trailing one has walked up

Watching `win15 3`'s lesson-2 video the user asked the right question: in the
failed attempts the robot reaches for the next segment *before* advancing along
the one it is on, so the free arm stretches to its limit and still comes up short
— why not walk to the edge first, which is both closer and cheaper?

It is doing exactly that, and the split is clean. Taking each episode's best reach
— the step where the free hook comes nearest the next seat while exactly one hand
carries the load — over ten episodes of the best checkpoint:

| | holding hand vs its own seat edge | hand separation | free hook onto the next seat | outcome |
|---|---|---|---|---|
| 4 episodes | within **64 mm** of the edge | 417–573 mm | 115–235 mm | **both successes** |
| 6 episodes | **246–313 mm short** of it | 583–651 mm | 3–25 mm | 5 timeouts, 1 fall |

The arm span is about 671 mm, so the second group is reaching at full stretch, and
it arrives 3–25 mm onto a seat — the near edge, the catch that slips. Worse, it
strands the other hand: with the lead hook 4 mm onto `seg1` at +0.636 and the
trailing one back at +0.03, the trailing hand is 0.6 m from the gap it now has to
cross, and nothing can move. That is the deadlock behind every one of those
timeouts.

The energy half of the question is right too. Reaching at full stretch costs
**104.2 N·m** of total joint torque against **71.9** from close, 45 % more. But
`K_EFFORT` prices that difference at 0.0016 per step — about **0.24** over a
150-step reach — against 1.25 for the 250 mm of rung the hand left behind and a
20-point swing between success and failure. The robot's own effort term already
measures the waste; it is charged at roughly 1 % of what the decision is worth.

The fix is not a new penalty. Between 1.7 and 2 the curriculum changes two things
at once — the gap goes 0.18 → 0.25 m *and* one gap becomes two — which is the
double step the rest of these lessons exist to avoid, and the measurement says the
width is what breaks. **Lesson 1.85 is one 0.25 m gap and nothing else new.**

Adding it surfaced a second bug. `Curriculum` unlocked `pool[len(active)]`, an
index, so inserting a lesson into the middle of the pool on a resume left the
restored active set out of step with pool positions: with `active` =
[1.2, 1.35, 1.5, 1.7, 2] and 1.85 inserted, the next unlock was lesson **2 again**,
already active, and 1.85 would never have been trained. It now unlocks the first
pool lesson that is not yet active.

## The wrist: a hook's wrist is its grip, so it stays at ±25°

Asked to give the wrist a human range (−70° extension … +80° flexion, against the
±25° it had), the measurement decided both ends — and then a 20M-step run decided
the rest. Hanging on a rung and turning one wrist while reading the hook's contact
force:

| flexion (+) | 0° | +25° | +50° | +80° |
|---|---|---|---|---|
| hold | 44 N | 123 N | 167 N | 143 N |

| extension (−) | −25° | −40° | −55° | −60° |
|---|---|---|---|---|
| hold | 44 N | 20 N | 13 N | **0 N — off the rung** |

Extension unloads the hook — 70° of it is 70° of letting go, and a 1.5M-step run
with the full human range never passed the first lesson. So the extension stop went
to −30°. Flexion looked free: it wraps the hook tighter. It is not free, because
tighter is the problem. With the wrist flexed and the finger opened, the hook
should come off the rung; measured:

| wrist | hold | finger opened | arm lifted |
|---|---|---|---|
| 0° | 45 N | 0 N | 0 N — released |
| +25° | 58 N | 0 N | 0 N — released |
| +50° | 118 N | 0 N | **54 N — re-caught** |
| +80° | 118 N | **47 N** | **57 N — cannot let go** |

Past +25° the wrist locks the hook onto the rung, and `win14` (5.5M steps on
−30…+80) found the lock: on lesson 1 its right wrist sat at **+77° mean, at the
+80° stop 95 % of the time**, the right hand took **zero** holds in twelve
episodes, and lesson 1 stayed at 0 % for the 3.5M steps after it unlocked while
the left hand shuffled 37 times per episode. Every +80° configuration tried
stalled in the early curriculum (−70…+80, −30…+80, −25…+80); the one ±25° run on
the same fixed robot went −2 → lesson 1.2 in 1.5M. A human wrist has 150° of
travel because fingers do the gripping. A hook's wrist *is* the grip, and the
range where it can both hold and let go is the ±25° it had. Back to ±25°.

The elbow fix, measured in the same run, did what it was for: on lesson 0 the
shoulder rotation sits at −45° (left) and −4° (right) — mid-range, where the old
mounting pinned it at ±90° — and the elbows point outward on 94–100 % of loaded,
rising steps.

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
