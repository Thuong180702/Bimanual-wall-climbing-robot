# The SLAM test rig

The policy trains on error models of the SLAM and the wall map (`rl/climb_task.py`:
`slam_drift`, `noisy_map`; `rl/lidar.py`: what the lidars can see). This rig tests
against the real thing. The four Mid-360s are simulated point by point, **FAST-LIO2**
builds the pose and the map from their points, and the task reads that instead.

```bash
docker build -t climber-slam slam/                        # once, ~10 min: FAST-LIO2 on ROS Noetic
python -m rl.slam_test --lesson 1 2 7 --steps 250         # arms swinging: SLAM and map errors
python -m rl.slam_test --lesson 1 --still                 # hanging still
python -m rl.slam_test --lesson 1 --posts                 # a post at each rung end, lidar-only
python -m rl.slam_test --lesson 2 --run runs/NAME         # a checkpoint climbing on it
```

`rl.slam_test` starts the container itself and reuses it:

```bash
docker run -d --rm --name climber-slam -p 5555:5555 -v $PWD/slam:/sim climber-slam bash /sim/start.sh
```

## What is where

| | |
|---|---|
| `Dockerfile` | ROS Noetic (`ros:noetic-perception`, arm64 or amd64), FAST_LIO pinned at `7cc4175`, built with catkin |
| `livox_ros_driver/` | only the driver's two message definitions, `CustomMsg` and `CustomPoint`, which FAST-LIO reads. The simulated sensors publish exactly what the real driver does, without the SDK that talks to hardware |
| `sim_mid360.yaml` | FAST-LIO's parameters: its own `mid360.yaml` and launch file, with the extrinsics set to identity (the cloud arrives in the IMU frame), `blind` 0.05, and 0.1 m voxels |
| `bridge.py` | inside the container: a TCP server (port 5555) the simulation talks to in lock-step. It publishes `/livox/imu` and `/livox/lidar`, waits for FAST-LIO's `/Odometry` and `/cloud_registered` for that frame, and sends them back. It restarts FAST-LIO on every reset |
| `start.sh` | inside the container: roscore, the parameters, the bridge |
| `vendor/` | Livox's Mid-360 scan pattern, fetched and checksummed by `rl/slam_test.py`; gitignored |

## What the simulation gives FAST-LIO (`rl/slam_test.py`)

- **Scan pattern.** Each Mid-360 fires 200 000 points/s along a real sensor's recorded
  non-repetitive pattern: 800 000 directions, 37.8–97.2° off the axis, from
  `mid-360-real-centr.csv` in [fratopa/Mid360_simulation_plugin](https://github.com/fratopa/Mid360_simulation_plugin).
- **Ray casting.** MuJoCo casts the rays (`mj_multiRay`) into the walls, the rungs and the
  robot's collision shapes, 4 ms of points at a time as the robot moves.
- **Frames.** A 100 ms frame is 25 of those slices, so it is bent by the motion the way a
  real frame is.
- **Point noise, from the datasheet.** Range error 3 cm (1σ) at 0.2 m and 2 cm at 10 m;
  the beam 0.15° off from where the point is reported; nothing inside 0.1 m or past 40 m.
- **One merged cloud.** The four lidars are merged into one cloud in the IMU's frame,
  using the CAD's extrinsics. `line` says which lidar a point came from.
- **IMU.** The torso's IMU at 250 Hz, read through the training noise model.
- **Self-filter.** The robot's own returns are dropped by what they hit, so the
  self-filter is perfect. A real one masks the robot's shape at its encoder angles.
- **Before the episode.** The robot hangs still for 2 s. FAST-LIO initialises and settles
  in that time.
- **Map frame.** Fixed where the robot hangs at the end of those 2 s, as FAST-LIO has it
  then. On hardware you know where you hung it. Everything after that is FAST-LIO's.
- **Pose timing.** A pose lands `SLAM_LAG` control steps after its frame ends, as in
  training, and is held against the estimate of that moment.

**The rung map** (`planes`, `rungs`, `merge`) is built from FAST-LIO's registered points
in four steps:
1. **Walls.** RANSAC finds the big planes. A plane parallel to one already found and
   within 20 cm of it is that wall's own noise, not another wall.
2. **Excess in front of each wall.** The points 2–14 cm in front of a wall, less the
   mirror band behind it, are what stands proud of it. The wall's own noise is
   symmetric, which is why the band behind it is the background.
3. **Rungs.** The excess is summed over 18 × 6 cm, since the holds are horizontal, and
   judged at 5σ. At 3σ a 10⁵-cell wall's noise drew rungs of its own. A connected run at
   least 10 cm long, within 5° of level and 4–14 cm off its wall, is a rung.
4. **Ends and axis.** A rung's ends are where the plain excess along it falls away. Its
   axis is behind the points' centroid by 2r/π, away from where the points were seen
   from, because a lidar sees the near half.

A rung a hand hides the middle of is two pieces: a lidar cannot tell that from a gap.

Checked against a perfect SLAM (the truth in FAST-LIO's place), lessons 1, 2, 4, 6, 7
and 8 after 2.4 s of swinging:
- every rung is mapped, with none mapped where there is no rung;
- each rung is off by 1.4–9 mm across and 0.01–1.8° in tilt;
- its ends are off by −87…+42 mm.

## What FAST-LIO2 does on the lesson scenes (2026-09-30)

10 s per lesson with the arms swinging, sensors with their noise. The SLAM error is the
base's position against the truth, since the map frame was fixed. `along` is along the
start rung: the one component `ClimbTask`'s estimator takes from the SLAM.

| lesson | the scene | SLAM rms | along the rung | worst | the rung map |
|---|---|---:|---:|---:|---|
| 1 | one wall, one 2.4 m rung | 1339 mm | 1310 mm | 1851 mm | smeared: the rung not found where it is |
| 2 | one wall, three rungs with gaps | 1606 | 1602 | 3594 | smeared |
| 4 | one wall, four rungs, down | 883 | 875 | 2428 | 1 of 4 |
| 6 | one wall, six short rungs | 2488 | 2487 | 5216 | 2 of 6, 19 segments on no rung |
| **7** | **a corner: front wall and side wall** | **32** | **5** | 72 | **2 of 2: 12–18 mm across, tilt < 0.3°, ends −65…+38 mm** |

(This table is FAST-LIO's pose as it comes, before the fix below.)
| 8 | two parallel walls | 202 | 196 | 274 | 1 of 2 |

With the robot hanging still, lesson 1 drifts 0.4–0.8 m along the rung in 10 s, and
lesson 7 stays within 1 mm along it (5.8 mm rms overall). With the noise off, lesson 1
does not stay put: it diverges, 23 m in 10 s. Posts at the rung ends (`--posts`) do not
help: they are 1–1.5 m away and 2 cm thick, and FAST-LIO's 0.1 m voxels thin them to
nothing.

**What that says.** A flat wall with smooth rungs does not tell a lidar where it is
*along* the wall. Every point-to-plane match on the wall and on the rungs constrains the
two directions across them and none along. So the one direction the estimator asks the
SLAM for is the one it cannot give. The IMU integrates its offset into that direction
unopposed: 0.1 m/s² (`SEN_BIAS`) is 1.8 m in 6 s. A second wall at an angle (lesson 7)
closes it, and there the SLAM is better than the training model assumes (5 mm along the
rung, against `SLAM_SIG` 20 mm).

**Decided (2026-09-30):** the robot has to work on any flat wall, so the estimator trusts
the SLAM only where it can see. The fix along a rung is taken only when a wall faces
along it (`climb_task.along_seen`), in training and here alike. Here the frames are also
registered with the estimator's pose where FAST-LIO is blind: along a flat wall, and up
and down. The map then stays within 39–69 mm of the truth where FAST-LIO drifted
0.2–0.9 m; the full table is in `rl/README.md`.

The lesson scenes are empty on purpose for the physics: there is no floor, no ceiling
and no clutter, and `scene.xml` has no floor by request. A lidar sees all of that. The
two ways this could have gone:
- Model the real test site in the lesson scenes as lidar-only geometry (floor, ceiling,
  edges, brackets), then measure again and set `SLAM_*` / `MAP_*` from that.
- Or keep the flat wall and make the estimator trust the SLAM only in the directions
  FAST-LIO can observe. A long flat facade is exactly this case, and that is how
  degeneracy-aware LIO handles it.

Speed: 3–6× slower than real time here (Apple M-series, Docker Desktop, 12 cores).
