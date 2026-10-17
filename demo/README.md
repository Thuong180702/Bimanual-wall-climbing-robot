# Demo videos

The full robot in MuJoCo climbing the benchmark lessons (left), and the map its four
lidars build as it goes (right): each Livox Mid-360 simulated point by point along its own
non-repetitive scan pattern, 200 000 points/s, the robot's own returns filtered out, the
rest kept in a 3 cm voxel map coloured by height. The pink line is the path of the body.

| video | lesson | driven by |
|---|---|---|
| [lesson01_traverse.mp4](lesson01_traverse.mp4) | 1, along one rung | the policy |
| [lesson02_traverse_gaps.mp4](lesson02_traverse_gaps.mp4) | 2, across a broken rung | the policy |
| [lesson03_climb_up_mirror.mp4](lesson03_climb_up_mirror.mp4) | 3, up rows standing out, toward the right | the policy |
| [lesson04_climb_down.mp4](lesson04_climb_down.mp4) | 4, down rows standing in | a reference climb (CEM, open loop) |
| [lesson05_staircase_down.mp4](lesson05_staircase_down.mp4) | 5, down a staircase | the policy |
| [lesson06_step_up.mp4](lesson06_step_up.mp4) | 6, diagonally up a tier | the policy |
| [lesson07_wall_corner.mp4](lesson07_wall_corner.mp4) | 7, round the corner | the policy |

The policy is `gpu11`'s best checkpoint (789M steps), trained before the lessons were
renumbered on 2026-10-16 and never on lesson 4, whose video plays the CEM climb
`rl/fixtures/climb_f4x2_r40.npz` instead. Each video is one successful episode, and
that is not its success rate: on 4 episodes each it passed lessons 1, 5, 6 and 7 4/4 and
lessons 2 and 3 0/4 (lesson 3 mirrored 1/4). Lesson 2's video is the first success of 10
seeds, lesson 3's of 2.

Made with `python -m rl.demo --run <run> --lessons <k>` (`--reference <climb>` for lesson 4),
re-encoded to under 20 MB each.
