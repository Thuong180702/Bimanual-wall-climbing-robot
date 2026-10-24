# Demo videos

The full robot in MuJoCo climbing each lesson's hardest wall the policy climbs (left), and the
map its four lidars build as it goes (right): each Livox Mid-360 simulated point by point along
its own non-repetitive scan pattern, 200 000 points/s, the robot's own returns filtered out, the
rest kept in a 3 cm voxel map coloured by height. The pink line is the path of the body.

| video | lesson | wall | seeds climbed |
|---|---|---|---|
| [lesson01_traverse.mp4](lesson01_traverse.mp4) | 1, along one rung | goals 1.40 m along | 5/8 |
| [lesson02_traverse_gaps.mp4](lesson02_traverse_gaps.mp4) | 2, across a broken rung | 2 gaps of 0.30 m | 6/8 |
| [lesson03_climb_up.mp4](lesson03_climb_up.mp4) | 3, up rows standing out | 2 rows, 0.45 m apart | 5/8 |
| [lesson04_climb_down.mp4](lesson04_climb_down.mp4) | 4, down rows standing in | 2 rows, 0.45 m apart | 8/8 |
| [lesson05_staircase_down.mp4](lesson05_staircase_down.mp4) | 5, down a staircase | 3 steps, 0.45 m down, 0.30 m on | 1/8 |
| [lesson06_step_up.mp4](lesson06_step_up.mp4) | 6, diagonally up a tier | 0.50 m up, rungs ±0.08 m | 4/8 |
| [lesson07_wall_corner.mp4](lesson07_wall_corner.mp4) | 7, round the corner | side wall 1.10 m away | 7/8 |
| [lesson08_wall_behind.mp4](lesson08_wall_behind.mp4) | 8, onto the wall behind | back rung 0.50 m behind | 6/8 |
| [lesson09_swing_gap.mp4](lesson09_swing_gap.mp4) | 9, swing across a gap | a 0.90 m gap | 3/8 |

The policy is `gpu14`'s best checkpoint (2.195B steps). Each wall puts every parameter of its
lesson's family (`rl.walls.FAMILIES`) at the run's ADR level, or at the step above where the
policy climbs that too (lessons 6 and 9; lesson 8 at 0.55 m was 0/8); every other draw and
the side are the seed's. Every wall but lesson 8's is its family's last level (lesson 8's
runs to 0.70 m). Each video is the first of 8 seeds the deterministic
policy climbs; the column is how many of the 8 it climbed, not a benchmark score.

Made with `rl.demo`'s `Film` and `episode` on walls from `rl.walls.sample(f, level, rng,
exact=level)` built by `rl.mjx_scene.c_scene`; lessons 2 and 5 re-encoded (crf 31) to under
20 MB.
