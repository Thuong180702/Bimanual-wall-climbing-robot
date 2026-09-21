# Bimanual wall-climbing robot

A 21.9 kg two-armed robot that hangs from wall rungs by hooked hands, in MuJoCo.
Two subsystems share one model:

| | |
|---|---|
| [`cad/`](cad/README.md) | the mechanical design: vendor actuator CAD, the brackets between them, and the tools that check both — the robot is generated from here |
| [`urdf/`](urdf/README.md) | the generated robot, the scene, 86 sensors, and the measured properties of both |
| [`controller/`](controller/README.md) | a classical controller: anchor-rooted state estimate, relative Jacobians, hold graph, A\* routing |
| [`rl/`](rl/README.md) | a PPO curriculum: seventeen lesson scenes ordered by difficulty, and the task/reward around them |

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m controller.test_climb      # the classical controller, phase by phase
python -m rl.lessons                 # generate the lesson scenes (not in git)
python -m rl.climb_task              # the RL task, plus every reward invariant
python -m rl.train_ppo --name run1   # train
python -m rl.render --video run1     # watch what it learned
```

## Real actuators, not blocks

Every joint is a purchasable CubeMars AK-series actuator, every mass and
inertia comes from the CAD solids, and every torque limit off a datasheet.
Nothing in `urdf/` is a plausible-looking number, and nothing there is
hand-written: `cad/tools/gen_urdf.py` and `urdf/_gen_scene.py` write it.

The robot this replaced — aluminium cylinders with masses derived from
geometry, 11.13 kg, fine in simulation and impossible to build — is in git
history. The cost of being real is **21.89 kg instead of 11.13**, more than
half of it in the actuators. The two things that did not change are the ones
everything else rests on: the kinematics, and the vertical reach that set the
rung spacing — 0.676 m, against v1's 0.671.

**The controller has not been retuned for it.** Its state estimate, hold map
and Jacobians check out on the new robot; its motion phases are tuned for
11 kg and time out or fall at 21.9. See [`urdf/README.md`](urdf/README.md).

## Where it stands

**The classical controller** traverses and moves down a row on both hands, landing
13 mm from the hold. Going back up still falls. Its README is a log of what was
measured on this robot, and most of it is load-bearing for anything built on top.

**The RL side** trains, and its main output so far is a list of things the task
rewarded that were not climbing — the hook slides along a smooth rung, a reward that
pays one direction can be pumped, a sum over two hands does not care which hand
works. Each is now an assertion in `python -m rl.climb_task`. No policy climbs a
gapped rung yet.

## Platforms

Trains on macOS, Linux and Windows; wheels exist for every dependency on all three.
The simulator is CPU-only, so **CPU cores are what set the speed** — a GPU can help
only the policy network, and that is under the noise floor here: cutting 90 % of the
gradient work moved throughput by 0 % (4850 vs 4797 fps). `--device cuda` exists and
works, it simply has nothing to do. The GPU path that would pay is MJX, which runs
the physics itself on the GPU; the model is two small edits away from being MJX-ready
(capsule collision geoms, and dropping the `jointlimitfrc` sensors — no sensor the
task uses is lost).

`ffmpeg` is needed only by `rl.render` and `controller.record`, never for training.

## Two rules that shaped everything here

**The control loop reads only what real hardware has** — encoders, IMU, force/torque,
grip pads, plus a wall map known in advance the way vision would supply it. Ground
truth (`data.xpos`, `env.latched`) appears in tests and in the RL reward, which is a
training-time oracle, and nowhere else.

**Measure, then decide.** Nearly every design choice in the two subsystem READMEs is
followed by the number that forced it, and several of them overturned an idea that
looked obviously right first — including a pinned-base reach calculation that was
wrong by 2x, and a "fix" for hand crossing that quietly made the hands overlap
instead.
