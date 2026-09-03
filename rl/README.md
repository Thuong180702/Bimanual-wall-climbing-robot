# PPO curriculum

The frame for training PPO on the climbing tasks, as nine lessons ordered by
difficulty instead of the mixed scene.

```bash
python -m rl.lessons        # regenerate urdf/lessons/*.xml + smoke-check each
python -m rl.climb_task     # smoke-check the task API on every lesson
python -m rl.bench          # where the time actually goes on this machine
python -m rl.train_ppo --self-check          # curriculum logic, trains nothing

python -m rl.train_ppo --name climber        # train: all 9, curriculum on
# Ctrl-C any time, then:
python -m rl.train_ppo --name climber --resume

# from a second terminal, while it trains:
python -m rl.status --name climber --watch
```

`torch`, `gymnasium` and `stable-baselines3` are installed in this repo's `.venv`
(torch 2.13 / SB3 2.9 on Python 3.14 — no separate venv needed).

| file | what it is |
|---|---|
| `lessons.py` | the 9 lesson specs + scene generator (writes `urdf/lessons/`) |
| `climb_task.py` | Gymnasium-shaped env: obs / action / reward / termination |
| `train_ppo.py` | thin SB3 PPO entry point |
| `bench.py` | simulator throughput vs env count, and cpu vs mps for the policy |
| `render.py` | PNGs of the lesson scenes → `out/scenes/` (ffmpeg writes them, no image dep) |
| `status.py` | what a run is doing, read from its checkpoint — safe to run while it trains |
| `eval.py` | score a checkpoint per lesson against a do-nothing control |

## Lessons

1. traverse, continuous rung → 2. traverse, gaps → 3. climb up → 4. up-over-down →
5. climb down → 6. staggered heights → 7. perpendicular wall → 8. wall behind the
back → 9. row beyond static reach (needs a dynamic move).

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
| `K_BANK` on a per-hand **anchor** | paying live distance paid for the phase-7 slide: the first policy crossed 0.75 m with a hook that never opened its finger |
| anchor moves only after the hand got `CLEAR_OF` the rung | without it, a wrist-load dip counted as taking a new hold — on the 5.1M policy **every** re-anchor was that, with the hook never leaving the rung |
| both terms charged **both ways** | paying only in the good direction is a pump: measured **+27.28 paid for 0.000 m of travel** and 99 release/reseat cycles in one episode |
| `K_DANGLE` with a `DANGLE_GRACE` | the flat version taxed the release climbing is made of, −15.25 against +5.57 of progress |
| `K_CROSS` | the distance term is a **sum** over both hands, so it does not care which hand works: policies froze one hand and crossed the other over it, 100 % of steps in the crossed shoulder band and 0 % outward |

A single-joint sweep says crossing is not necessary: the leading hand gains **+459 mm**
toward the goal on its 60° of outward travel, against **+64 mm** for crossing the
trailing one. Crossing was a local optimum, and lesson 1 — solvable by sliding both
hands — kept reinforcing it.

**Do not train on a narrowed pool.** Dropping lesson 1 to force lesson 2 was tried and
it is plain sequential fine-tuning: lesson 1 went from 12/12 to falling 4/4, and
lesson 2 got worse too. Keep the pool mixed, as the section above says.

Skipped: domain randomization, best-checkpoint keeping. Reward weights are
first-guess knobs — `rl.eval` plus the diagnostics are how you tell.
