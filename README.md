# Bimanual wall-climbing robot

A 21.5 kg two-armed robot that hangs from wall rungs by hooked hands, in MuJoCo.
Two subsystems share one model:

| | |
|---|---|
| [`cad/`](cad/README.md) | the mechanical design: vendor actuator CAD, the brackets between them, and the tools that check both — the robot is generated from here |
| [`urdf/`](urdf/README.md) | the generated robot, the scene, 86 sensors, and the measured properties of both |
| [`controller/`](controller/README.md) | a classical controller: anchor-rooted state estimate, relative Jacobians, hold graph, A\* routing |
| [`rl/`](rl/README.md) | a PPO curriculum: eighteen lesson scenes ordered by difficulty, and the task/reward around them |

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
history. The cost of being real is **21.52 kg instead of 11.13**, more than
half of it in the actuators, and 24.16 kg with the four Livox Mid-360 lidars and
the tube cage that guards them (`urdf/README.md`, "The lidars"). The arm's root, the arm and the hands are where v1
had them. The shoulder is pitch-first, like ten of the twelve humanoids it was
checked against: the first real-actuator version stacked it abduction-first and
folded the raised arm up past its own abduction motor. Its vertical reach is
0.56 m, lower than v1's 0.671 and that version's 0.676 - both came from poses
no real shoulder makes - and still above the 0.45 m the rung rows are spaced.

**The classical controller has not been retuned for it.** Its state estimate,
hold map, Jacobians and the right hand's single step pass on the new robot
(phases 0, 1, 2, 4 and 7). The left hand's step lands true relative to the
anchored hand, but that bare hook slides ~150 mm along the rung meanwhile (3);
the traverse, the climb down and the route (5, 6, 8) fall, because their gains
were swept at 11 kg. The RL task needed no retuning, only its two baselines read
off the model instead of typed in. Every policy trained before this is void: it
learned the old robot. See [`urdf/README.md`](urdf/README.md).

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
