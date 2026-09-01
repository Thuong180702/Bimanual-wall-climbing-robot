# Bimanual wall-climbing robot

An 11 kg two-armed robot that hangs from wall rungs by hooked hands, in MuJoCo.
Two subsystems share one model:

| | |
|---|---|
| [`urdf/`](urdf/README.md) | the robot, the scene, 86 sensors, and the measured properties of both — start here |
| [`controller/`](controller/README.md) | a classical controller: anchor-rooted state estimate, relative Jacobians, hold graph, A\* routing |
| [`rl/`](rl/README.md) | a PPO curriculum: nine lesson scenes ordered by difficulty, and the task/reward around them |

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m controller.test_climb      # the classical controller, phase by phase
python -m rl.lessons                 # generate the lesson scenes (not in git)
python -m rl.climb_task              # the RL task, plus every reward invariant
python -m rl.train_ppo --name run1   # train
python -m rl.render --video run1     # watch what it learned
```

## Where it stands

**The classical controller** traverses and moves down a row on both hands, landing
13 mm from the hold. Going back up still falls. Its README is a log of what was
measured on this robot, and most of it is load-bearing for anything built on top.

**The RL side** trains, and its main output so far is a list of things the task
rewarded that were not climbing — the hook slides along a smooth rung, a reward that
pays one direction can be pumped, a sum over two hands does not care which hand
works. Each is now an assertion in `python -m rl.climb_task`. No policy climbs a
gapped rung yet.

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
