# rl: the climbing task and its policy

One goal-conditioned PPO policy, trained on MJX (MuJoCo on the GPU) over procedurally
generated walls with an automatic curriculum, and scored on a fixed benchmark of nine
lessons on the full robot in C MuJoCo.

```bash
python -m rl.lessons                 # the benchmark scenes (urdf/lessons/, not in git) + their checks
python -m rl.walls                   # the training walls: every family at its first and last levels
python -m rl.climb_task              # the task: every reward, sensor and termination invariant
python -m rl.rsi                     # rebuild the reference-climb fixture and check it
python -m rl.train_ppo --self-check  # curriculum, ADR, guard, update size, cloning
python -m rl.mjx_check               # the GPU task against the C one: task, physics, model, policy

bash rl/gpu_setup.sh && bash rl/gpu_run.sh gpu1      # a rented GPU box: preflight, train, score
python -m rl.status --name gpu1 --watch              # what a run is doing (safe while it trains)
python -m rl.eval --name gpu1 --from-best            # the benchmark, against doing nothing
python -m rl.curve --name gpu1 && python -m rl.plot --name gpu1     # learning curve, figures
```

| file | what it is |
|---|---|
| `lessons.py` | the nine benchmark lessons and their scene generator |
| `walls.py` | the training walls: nine families, their ranges, the order they enter, the wall check |
| `climb_task.py` | the task in C MuJoCo: observation, action, reward, termination, sensor models |
| `estimator.py`, `lidar.py` | what the robot knows of itself and of the wall, from its own sensors |
| `mjx_scene.py`, `mjx_task.py` | the robot as MJX can simulate it, and the task ported to JAX |
| `rsi.py`, `plan.py` | the reference climbs: how they are used, and how they were made |
| `train_ppo.py` | PPO, the curriculum, the guard; `gpu_run.sh` runs it unattended |
| `mjx_check.py` | the GPU path cross-checked against the C one |
| `eval.py`, `curve.py`, `plot.py`, `status.py`, `render.py` | scoring, figures, watching |
| `HISTORY.md` | the development journal: every run, and the measurement behind each choice |

## The benchmark

Each lesson goes toward the robot's left; every one is also scored mirrored. The robot
starts hanging from both hooks on a rung at x = 0.44 m, z = 1.00 m, leaning on the wall --
or, where the start rung stands off it (lesson 4), pitched about the rung to its
stillest free hang.

| | lesson | the wall |
|---|---|---|
| 1 | `traverse` | one continuous rung, the goals 1.10 m along it |
| 2 | `traverse_gaps` | a rung broken by two 0.25 m gaps |
| 3 | `climb_up` | two rows 0.40 m apart above, each 0.20 m further out from the wall |
| 4 | `climb_down` | lesson 3 backwards: two rows below, each 0.20 m nearer the wall |
| 5 | `staircase_down` | three short rungs, each up to 0.40 m lower and 0.20 m on |
| 6 | `step_up` | a tier of two short rungs ~0.46 m up, past the end of the start row |
| 7 | `wall_corner` | onto a rung on the perpendicular wall, 1.00 m to the side |
| 8 | `wall_behind` | onto a rung 0.63 m behind, on the wall at the robot's back |
| 9 | `swing_gap` | a 0.90 m gap, past the static reach (0.58 m): crossed by swinging |

No row ever sits straight above or below the hang in the rung plane: the arms and body
fill that plane 0.10-1.10 m under the hooks. So rows above stand out from the wall, rows
below stand in toward it, and the diagonal moves step along as they climb.

## The training walls

Training never draws a benchmark lesson as such. It draws walls from nine families
(`walls.py`), one per lesson, each that lesson's structure with a few parameters free.
Each parameter has a range and a **level**: walls draw it uniformly up to its level,
and a quarter of a family's walls put one parameter at its **candidate**, one step up.
When 200 walls at a candidate are passed at 70 %, it becomes the level (automatic
domain randomization). Training therefore never draws past what has been passed, and a
level that fails costs a quarter of the family's walls, never the rest.

| family | parameters: first level -> last | the lesson's |
|---|---|---|
| 1 | each hand's shift along the row, 0.30 -> 1.40 m | 1.10 |
| 2 | gaps 1 -> 2; width 0.04 -> 0.30 m | 2 x 0.25 |
| 3 | rows 1 -> 2; spacing 0.20 -> 0.45 m | 2 x 0.40 |
| 4 | rows 1 -> 2; spacing 0.20 -> 0.45 m | 2 x 0.40 |
| 5 | steps 1 -> 3; drop 0.30 -> 0.45 m; gap 0.10 -> 0.30 m | 3, 0.40, 0.20 |
| 6 | rise 0.30 -> 0.50 m; scatter 0.02 -> 0.08 m | 0.46, 0.04 |
| 7 | side wall 0.60 -> 1.10 m | 1.00 |
| 8 | back rung 0.35 -> 0.70 m | 0.63 |
| 9 | gap 0.55 -> 0.90 m | 0.90 |

Every wall is checked before it is drawn (`walls.Checker`): its goals on seats, and no
rung or extra wall within 2 cm of the robot at its start hang or at the goals.

## The curriculum

One policy learns every family at once: each reset draws from the pool, and nothing
ever leaves it, because training families one after another forgets the earlier ones.

- **Entry.** Families enter one at a time, in order, once every family in the pool
  passes its gate -- 70 % over its last 200 walls from the hang -- and the families it
  is built from have passed (`walls.NEEDS`). A family that has not passed once in 50M
  steps stops holding back the families that do not need it.
- **Step shares.** Each family's share of the steps is how much it still fails (10 %
  at least), so the frontier and anything that regresses get the samples.
- **Reference climbs** (`rsi.py`). Families whose skill is a long chain of moves --
  climbing a row, stepping up a tier, the back rung, the swing -- are never found from
  the hang by PPO's noise. A share of their walls (three in four until the gate is
  passed, one in four after) begin partway up a reference climb, drawn toward each
  climb's frontier: the latest point along it the policy does not yet finish from. In
  that window the policy's mean action is also pulled toward the climb's (behaviour
  cloning at 0.3x PPO's gradient). The climbs are searched by CEM on the simulated
  robot or recorded from a trained policy (`plan.py`, which lists where each came from).
- **Update size.** Each action dimension's std stays in [0.05, 1], and the learning
  rate keeps every update's KL near a target (rsl_rl's rule).
- **Guard.** The best checkpoint is kept, scored by ADR's progress (0..1 a family). If
  the families it had passed fall a whole family below what they held there for 325
  updates, the run goes back to it and halves its KL target.

The console prints one line every 5M steps -- steps, time, speed, each lesson's success
from the hang, ADR's progress, the best -- and one line per event: a lesson entering, a
level raised, a rollback. Everything else is in `out/<name>/` for `rl.plot`.

## The GPU path

`mjx_task.py` is `climb_task.py` in JAX, the same observations, reward books and
terminations, on `mjx_scene.py`'s simplified robot: walls are planes, rungs capsules,
and whatever meets a rung is capsules and balls, because those are the pairs MJX
computes exactly. `mjx_check.py` holds it to the original: the JAX task equal to
ClimbTask observation by observation, MJX's physics against C MuJoCo's on the same
robot, the simplified robot against the full one at the same states, and a trained
policy climbing alike on all three. `gpu_run.sh` runs these on each new card before
training, and scores the best checkpoint on the full robot afterwards.

## The robot it trains

The 24.2 kg robot generated from `cad/`, with its sensors as real ones read: encoder,
IMU, F/T and pad noise with per-episode offsets, a wall map holding only what the four
Mid-360 have seen, and a lidar SLAM that drifts and is believed along a rung only where
a wall faces that way (`slam/`). A checkpoint records which robot it learned, and a
resume onto another is refused.
