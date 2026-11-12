# Learning Versatile Bimanual Wall-Rung Climbing from Hardware-Available Sensing with Search-Seeded Reverse Curricula

This repository holds the code, robot model, benchmark, reference climbs and evaluation
scripts of the paper above (manuscript in preparation). Everything the paper reports can be
regenerated from it: the robot from its CAD sources, the training runs from their names, and
every table from the trained runs.

## Overview

Robots that climb by grasping holds can reach structures where wheels, suction and magnets
fail, yet learned climbing has so far addressed one kind of structure at a time, with one
skill or a fixed sequence of skills. Here a 24.5 kg robot with two seven-joint arms and
hooked hands hangs from rungs mounted on walls. It must traverse a rung, cross gaps, climb
rows up and down, descend a staircase, step up to a higher tier, and move round a corner
onto a perpendicular wall.

- **One goal-conditioned policy** learns all seven skills with PPO.
- **The robot is buildable.** Every joint is a purchasable CubeMars actuator, and every
  mass, inertia, envelope and torque limit comes from CAD solids or a datasheet.
- **The policy reads only hardware-available sensors**, including a rung map built from
  four simulated Livox Mid-360 LiDARs. Its dynamics are randomized in training.
- **Training** combines procedural walls whose difficulty grows by automatic domain
  randomization (ADR), reference climbs used only as episode starts drawn toward the
  frontier of what the policy can finish, behaviour cloning restricted to that frontier,
  and a two-way route potential with holds decided by contact force.
- **Evaluation** happens on the full robot in C MuJoCo: a fixed seven-lesson benchmark,
  unseen walls from the training families, ablations, reward exploits, robustness and
  sim-to-sim transfer.
- **A simulated LiDAR-inertial odometry study** (FAST-LIO2) explains why the state
  estimate trusts odometry along a rung only where a second wall makes that direction
  observable.

## Repository layout

| directory | contents |
|---|---|
| [`cad/`](cad/README.md) | the mechanical design: vendor actuator and LiDAR CAD, the brackets, pelvis and chest bay, and the tools that check them. The robot model is generated from here |
| [`urdf/`](urdf/README.md) | the generated robot (URDF and MJCF), the scene with 17 actuators and 86 sensors, and the measured properties of all of it |
| [`rl/`](rl/README.md) | the task, reward, sensor and LiDAR-map models, the benchmark, the wall families, the reference climbs, PPO on MJX, and the evaluation |
| [`slam/`](slam/README.md) | the odometry test rig: simulated Mid-360 point clouds through FAST-LIO2 in Docker |
| [`demo/`](demo/README.md) | videos from an earlier policy (see [Demo](#demo)) |

### Where each part of the paper lives

| paper | code |
|---|---|
| The robot | `cad/src/lib/spec.py`, `cad/tools/gen_urdf.py`, `urdf/_gen_scene.py`; checks in `cad/tools/verify.py`, `cad/tools/torque.py` |
| Sensing available to the policy | `rl/climb_task.py` (noise model), `rl/estimator.py` (hand estimate), `rl/lidar.py` (LiDAR map) |
| The task | `rl/climb_task.py` (`step`, `_obs`, `_success`, `_holding`) |
| Benchmark and held-out walls | `rl/lessons.py`, `rl/walls.py`, `rl/eval.py --walls` |
| Reward | `rl/climb_task.py`, ported to JAX in `rl/mjx_task.py` |
| Curriculum over procedural walls | `rl/walls.py`, `Curriculum` in `rl/train_ppo.py` |
| Search-seeded reverse curriculum | `rl/rsi.py` (frontier), `rl/plan.py` (CEM search, climb provenance), `Clone` in `rl/train_ppo.py` |
| Guard and checkpoint selection | `Checkpoint` in `rl/train_ppo.py` |
| Simulation and its fidelity | `rl/mjx_scene.py`, `rl/mjx_task.py`, `rl/mjx_check.py` |
| Training runs and ablations | `rl/paper_run.sh`, `rl/gpu_run.sh`, `ABLATIONS` in `rl/train_ppo.py` |
| Evaluation | `rl/paper_eval.py`, `rl/eval.py`, `rl/mjx_check.py policy` |
| LiDAR-inertial odometry on the wall | `slam/`, `rl/slam_test.py` |

## The robot

| | |
|---|---|
| joints | waist yaw, two 7-DoF arms (shoulder flexion, abduction, rotation; elbow; forearm roll; wrist flexion, deviation), a finger closing each hook: 17 position servos |
| actuators | CubeMars AK80-64 (shoulders, elbows), AK60-39 (waist, forearms, wrists), AK45-36 (fingers) |
| mass | 21.86 kg with a 48 V battery, NVIDIA Jetson Orin NX, IMU and power electronics in the chest; 24.50 kg with four Mid-360 LiDARs and their cage; actuators 52 % |
| proportions | a 1.40 m person: upper arm 0.26 m, forearm 0.21 m, waist yaw 0.305 m below the shoulders; the pelvis carries mounts for two hip pitch actuators where legs would attach |
| reach | 0.56 m vertical with one hook anchored; hook throat 65 mm |

The torque each joint must hold over its whole range stays at least 1.35 times below its
peak torque, but exceeds the continuous rating in one-handed phases (the elbow needs 78 N m
against 48 N m rated), so one-handed phases must last seconds. Full measurements are in
[`urdf/README.md`](urdf/README.md).

## Task and sensing

The policy acts at 25 Hz (500 Hz physics) with 17 setpoint increments. Its 145-number
observation holds joint positions, velocities, torques and setpoints, the IMU, wrist
force/torque and grip pads, the goal relative to each estimated hand, and the three nearest
rungs of the LiDAR map. Every reading has white noise and a per-episode offset. The map
holds only what the four Mid-360 have seen past the robot's own arms and cage. Hand
positions come from forward kinematics rooted in the holding hand, with a complementary
filter for the torso attitude. Ground truth enters only the reward and the success test.

A hand holds when its hook presses on a rung with more than 20 N of normal force, its finger
is in the grip band, and it is seated on the rung. An episode succeeds when both hands hold
their goal rungs within 0.15 m of their goal points, each after at least one real grab.

## The benchmark

Seven lessons, one skill each, all starting from a two-hook hang. Each is scored toward the
robot's left and mirrored, on the full robot in C MuJoCo, with nominal dynamics. Training
draws walls from one family per lesson, never the benchmark walls themselves.

| # | lesson | wall | family range (first to last level) | benchmark |
|---|---|---|---|---|
| 1 | traverse | one continuous rung | shift per hand 0.30 to 1.40 m | 1.10 m |
| 2 | traverse gaps | rung broken by gaps | gaps 1 to 2, width 0.04 to 0.30 m | 2 x 0.25 m |
| 3 | climb up | rows above, each 0.20 m further out | rows 1 to 2, spacing 0.20 to 0.45 m | 2 rows, 0.40 m |
| 4 | climb down | rows below, each 0.20 m further in | rows 1 to 2, spacing 0.20 to 0.45 m | 2 rows, 0.40 m |
| 5 | staircase down | short rungs, each lower and further on | steps 1 to 3, drop 0.30 to 0.45, gap 0.10 to 0.30 m | 3, 0.40, 0.20 m |
| 6 | step up | a tier of two short rungs past the row's end | rise 0.30 to 0.50, scatter 0.02 to 0.08 m | 0.46, 0.04 m |
| 7 | wall corner | rung on the perpendicular wall | side wall 0.60 to 1.10 m | 1.00 m |

Generalization is measured on 200 unseen walls per family, drawn as training draws them
from seeds no run uses, and on walls one step past the last level of training.

## The method

- **Wall families and ADR.** Each family's parameters start easy and advance by ADR
  probes. A family enters training once its prerequisites pass their gate.
- **Reference climbs as starts only.** The 19 climbs of families 3 to 7 (1481 starts,
  7372 states) are nine found by CEM search, two mirror images of searched climbs, and
  eight recorded from policies trained earlier in this project, for moves the search did
  not find. None comes from a human demonstration. `rl/plan.py` records the provenance of
  each. Episode starts are drawn toward each climb's frontier, the latest start the policy
  does not yet finish, so that what it learns near a climb's end extends toward its
  beginning.
- **Cloning at the frontier.** At unpassed starts, the policy mean is pulled toward the
  climb's action, with the gradient scaled to a fraction of PPO's.
- **Reward.** The shaping term is a two-way potential over a route graph through the
  rungs, so a retreat costs what an approach pays. Holds are decided by contact force.
  Each rule answers an exploit found during development, and the `x_` runs train with
  the weaker rule to show it.
- **Dynamics randomization** of masses, contact friction, torque limits and setpoint lag.
- **Guard.** The best checkpoint is chosen by ADR progress, never by the benchmark. A
  lasting drop rolls training back to it.
- **Simulation.** Training runs a simplified robot (capsules and spheres) in MJX with its
  Warp backend on 4096 environments. Before every session, `rl.mjx_check` holds the JAX
  task to the C task state by state, MJX's physics (JAX and Warp) to C MuJoCo's, and each
  environment's randomized dynamics to its own physics. The policy's transfer to the full
  robot is measured at evaluation.

## Reproducing the paper

### 1. Install and check

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m rl.lessons                 # generate the benchmark scenes (not in git)
python -m rl.climb_task              # the task, with every reward and sensor invariant asserted
python -m rl.train_ppo --self-check  # the curriculum, guard and cloning logic
python -m rl.mjx_check task          # the JAX task against the C task (slow on a CPU)
```

The CAD tools (`cad/tools/*.py`) also need cadgen ([`cad/README.md`](cad/README.md)).
The generated robot is in git, so training does not.

### 2. Train

Training needs a Linux machine with an NVIDIA GPU. Every run starts from the tag
`paper-v1` and is launched by its name alone. The name sets the families, steps, seed,
ablation and reward weights. A run checks its GPU before training (about 30 minutes),
resumes by itself after a crash, and refuses to resume on code other than it started on.

```bash
git clone --branch paper-v1 https://github.com/Thuong180702/Bimanual-wall-climbing-robot
cd Bimanual-wall-climbing-robot && bash rl/gpu_setup.sh
mkdir -p out && nohup bash rl/paper_run.sh p_s0 > out/p_s0.console 2>&1 &
DRY=1 bash rl/paper_run.sh a_rfcl_s1          # what a name means, without running it
```

| runs | names | steps each |
|---|---|---|
| the method, 5 seeds | `p_s0` to `p_s4` | 1.5B |
| the method under the ablations' budget | `a_none_s0` to `a_none_s2` | 0.75B |
| ablations, 3 seeds each | `a_<x>_s0..2`, x = `rsi`, `frontier`, `rfcl`, `clone`, `cloneall`, `curriculum`, `taught`, `route`, `hold` | 0.75B |
| PPO alone | `a_plain_s0..2` | 0.75B |
| one specialist per family | `b<f>_s0..2`, f = 1 to 7 | 0.75B |
| reward-weight sensitivity | `w_bank05_s0`, `w_bank2_s0`, `w_reach05_s0`, `w_reach2_s0`, `w_slide05_s0`, `w_slide2_s0` | 0.75B |
| reward exploits, families 1 to 3 | `x_none`, `x_oneway`, `x_world`, `x_grab`, `x_refund`, `x_route`, `x_hold` | 100M |

`python -m rl.train_ppo -h` describes each ablation. The full set is about 1,000 GPU-hours.
Earlier runs on an RTX 3090 trained at 10 to 14 thousand environment steps per second.

### 3. Evaluate

Evaluation runs on CPUs. Put each run's `out/<name>/` back in `out/`, then:

```bash
python -m rl.paper_eval --list       # every evaluation, by run name
python -m rl.paper_eval --jobs 8     # run them: one core per job, resumable
```

This writes `out/<name>/paper.jsonl` for each run: the benchmark both ways round, unseen
and beyond-range walls, distractor rungs, robustness conditions, joint torques, the
reference climbs open loop, and the same policy on the full robot, the simplified robot
and MJX.

## Results

The paper's runs start from `paper-v1` and are in progress. Results will be added here when
they finish, with the trained checkpoints and logs archived.

## Demo

These videos are from an **earlier version** of the project: the policy gpu14 (2.2B
steps), trained on the previous body with a nine-lesson curriculum. Lessons 8 (onto the
wall behind) and 9 (swing across a gap) are not in the paper's benchmark, and the code
keeps them for future work. Left: the full robot in MuJoCo. Right: the map its four
Mid-360 LiDARs build as it climbs, simulated point by point along the sensor's scan
pattern. More in [`demo/`](demo/README.md).

| climb down rows standing in (lesson 4) | step up a tier (lesson 6) |
|---|---|
| [![lesson 4](demo/preview/lesson04_climb_down.gif)](demo/lesson04_climb_down.mp4) | [![lesson 6](demo/preview/lesson06_step_up.gif)](demo/lesson06_step_up.mp4) |

| swing across a 0.90 m gap (lesson 9, earlier curriculum) |
|---|
| [![lesson 9](demo/preview/lesson09_swing_gap.gif)](demo/lesson09_swing_gap.mp4) |

## Design principles

**The policy reads only what real hardware has.** Ground truth (`data.xpos`,
`env.latched`) appears in tests and in the reward, which is a training-time oracle, and
nowhere else. Actor and critic read the same noisy observation.

**Every number has a source.** Nothing in `urdf/` is hand-written: `cad/tools/gen_urdf.py`
and `urdf/_gen_scene.py` write it from the CAD and datasheets, and `cad/tools/verify.py`
checks it against them. In the subsystem READMEs, most design choices are followed by the
measurement that forced them.

**Measure what the reward pays for.** Each behaviour the task rewarded that was not climbing
is an assertion in `python -m rl.climb_task`. Examples are a hook sliding along a smooth
rung, a one-way reward that can be pumped, and a sum over two hands that ignores which hand
works.

## Platforms

Training runs on an NVIDIA GPU, with physics, task and policy on the card.
`requirements-gpu.txt` pins that stack (Python 3.12, CUDA 12, a V100 or newer). Everything
else (the checks, the evaluation, the figures) runs on macOS, Linux and Windows. MJX also
runs on a CPU, slowly. `ffmpeg` is needed only by `rl.render`, and Docker only by the SLAM
rig.

## Citation and license

To cite this repository, see [`CITATION.cff`](CITATION.cff). The paper's reference will be
added on publication.

Copyright (c) 2026 Thuong Nguyen Van. All rights reserved: public to read, run and review,
not to reuse or publish from without permission ([`LICENSE`](LICENSE)).
