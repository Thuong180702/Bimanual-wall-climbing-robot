# Bimanual wall-climbing robot — URDF export

Generated from the Fusion design `3d-climbing-robot`.

## Files

| file | what it is |
|---|---|
| `climbing_robot.urdf` | **fixed base** — `base_link` welded to the world. Standard URDF for ROS/RViz. |
| `climbing_robot_floating.urdf` | **floating base** — adds `world` + `floating_base` joint. **Use this one for MuJoCo climbing sim.** |
| `meshes/*.stl` | 19 binary STL meshes, one per link, in each link's own frame, **in millimetres** (URDF applies `scale="0.001"`). |
| `_gen_urdf.py` | regenerates the URDF from the two JSON dumps |
| `_linkdata.json`, `_jointdata.json` | raw data pulled out of Fusion (mass, COM, inertia, joint offsets/axes/limits) |

## Conventions

- Units: metres, kg, radians. Inertia in kg·m² **about each link's COM**, axes parallel to the link frame.
- Frame: **+X forward (toward the wall), +Y left, +Z up**.
- Joint signs: about **Y, NEGATIVE = flexion / forward / closing**. About **X, positive = toward +Y**.
- Every link frame has identity rotation w.r.t. the root, so all joint `rpy` are `0 0 0`.

## Load in MuJoCo

```python
import mujoco
m = mujoco.MjModel.from_xml_path("climbing_robot_floating.urdf")
d = mujoco.MjData(m)
```

`nq=24` (7 free-joint + 17 hinge), `nv=23`, total mass **11.1316 kg**.

Note: `torso_upper` is joined to `waist_link` by a **fixed** joint, so MuJoCo welds them into one
body of 4.8852 kg. That is correct — a fixed joint carries no DOF.

## Collision

Collision uses **primitives**, not the meshes, on purpose: MuJoCo takes the convex hull of a mesh,
which would fill in the hook's opening and destroy the gripping function. Each hook is therefore
two boxes (palm plate + finger) — an exact convex decomposition of the L.

MuJoCo automatically disables contacts between bodies connected by a joint, which covers all 18
adjacent pairs. **Leave self-collision on for everything else** — the arm-vs-torso collisions are
inherently multi-joint (e.g. shoulder abduction 90° turns the flexion axis into world Z, which then
sweeps the arm through the chest) and no set of independent joint limits can express them.

## Things you must supply yourself

- `effort=` values are geometry-derived static-hold estimates (g x distal mass x reach), **not
  actuator specs**. Replace with your real motor/gearbox data.
- `velocity=3.0` rad/s and `damping=0.1` are placeholders.
- No actuators are defined. Add `<position>` or `<motor>` actuators in an MJCF wrapper.
- No wall/ground geometry — add it to your scene.

---

# MuJoCo scene — `scene.xml`

Ready to run. Contains the robot (floating base), three walls, 16 rungs, 17 position
actuators, 3 cameras, and a verified hanging keyframe.

```python
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)   # "hang_on_rung"
mujoco.viewer.launch(m, d)
```

## Walls (no floor, by request)

| geom | inner face | note |
|---|---|---|
| `wall_front` | x = 0.50 | the wall the robot faces |
| `wall_back` | x = -0.25 | **parallel**, 0.75 m gap - within arm reach, cross over to it |
| `wall_side` | y = -1.00 | **perpendicular**, closes the corner, spans the whole corridor |

The 0.75 m gap is not arbitrary. Measured by sweeping the right arm from the hanging
keyframe: the hook tip reaches back to **x = -0.346 m**, i.e. 0.690 m behind the shoulder.
The back-wall rungs sit at x = -0.19, leaving ~0.15 m of margin so a transfer is a real
stretch but not a singular full-extension pose. At the old 1.20 m gap the robot missed the
opposite wall by 0.29 m.

## Holds - 39 rungs, radius 15 mm

Each rung stands **60 mm off its wall face** so the hook's finger can pass over it and the
palm plate can hang down in front. A hold flush with the wall cannot be hooked.

Three kinds of run, all present on every wall:

| kind | group suffix | example |
|---|---|---|
| **continuous straight rung** | `_line`, `_low` | `rung_front_line_0`, one bar from y=-0.90 to 1.50 |
| **separated but on one straight line** (same height) | `_aligned` | 4 holds all at exactly z = 1.45 |
| **separated, each at its own height** | `_stagA/B/C`, `_stag` | z = 1.82 / 1.84 / 1.92 / 1.95 |

Rows alternate their lateral phase by half a period (hold 0.35 m, gap 0.25 m), so from any
hold there is always one up-left and one up-right.

### Vertical spacing is derived, not guessed

Measured by optimising over all 13 arm/waist joints: with one arm locked off on a hold and
the other stretched up (both hooks kept in a latchable orientation, target within 0.30 m
sideways), the free hand rises at most **0.671 m** above the anchored one. The locked-off arm
is at abduction +180 deg, flexion -177 deg, elbow -114 deg; the reaching arm is straight up.

Every consecutive row is therefore **<= 0.50 m apart** - 75 % of that reach. Verified
programmatically: for all 39 holds except the topmost of each wall, there is another hold
0.05-0.50 m above and within 0.30 m sideways. Front wall runs 0.55 -> 2.80 m, back wall
0.55 -> 2.36 m, side wall 1.00 -> 2.34 m.

## Actuators

17 `<position>` servos, `act_<jointname>`, `dampratio="1"`, `ctrlrange` = the joint limits.
`forcerange` is sized for the worst climbing case — **the entire 11.13 kg robot hanging off
that joint's distal chain** at its maximum lever arm: 68 N·m at the shoulders, 49 at the waist,
39 at the elbows, down to 13 at the fingers. Still placeholders; replace with real motor data.

## Verified

- Loads in MuJoCo 3.12. `nbody=20, nq=24, nv=23, ngeom=91, nu=17`, mass 11.1316 kg.
- Keyframe `hang_on_rung`: both hooks latched on `rung_front_line_0`, servos holding.
  Replayed 5 s — **base drift 0.0 mm**, 8 contacts, load path
  `hook_L`/`hook_R` -> `rung_front_line_0`.

## Posture note (matters for your controller)

The hook only latches with **the arm raised**: at ~-160 deg shoulder flexion the hand inverts,
the finger ends up on top pointing at the wall, and the throat opens downward onto the rung.
That pose also swings the body **0.22 m back** from the hand, which keeps the 150 mm-deep torso
clear of the rungs below. With the arm straight up (-180 deg) the torso hangs right against the
wall and rams the lower rung — it gets knocked off the hold. Keep the arms angled.

## Regenerating

```bash
python3 _gen_urdf.py     # URDF from the JSON dumps
python3 _gen_scene.py    # body tree + actuator list
python3 _gen_scene2.py   # assembles scene.xml  (note: this drops the keyframe)
```

---

# Sensors

86 sensors, 119 channels, all in `scene.xml`.

## Drives - complete servo-drive feedback, 4 signals per motor

| sensor | what it is |
|---|---|
| `enc_<joint>` | shaft angle (rad) |
| `vel_<joint>` | shaft rate (rad/s) |
| `tau_<joint>` | motor torque - stands in for current sensing |
| `lim_<joint>` | hard-stop constraint torque, 0 unless the joint is jammed on its limit |

`lim_*` verified: pushing the elbow onto its -120 deg stop with -10 / -30 / -60 N.m reads
5.9 / 18.0 / 35.2 N.m and the joint sinks 0.5 / 1.4 / 2.8 deg into the soft stop.

## IMU on the upper torso (site `s_imu`, mid-chest)

**Raw 9-DOF, exactly what real hardware outputs:**

| sensor | |
|---|---|
| `imu_acc` | 3-axis accelerometer (specific force, body frame) |
| `imu_gyro` | 3-axis rate gyro (body frame) |
| `imu_mag` | 3-axis magnetometer - world field is `(0, -0.5, 0)` |

**Derived / ground truth - free in simulation, needs an estimator on hardware:**

`imu_quat` (attitude), `imu_vel` (linear velocity), `imu_angvel`, `imu_angacc`, `imu_linacc`,
`torso_pos`.

A real IMU gives you none of that second group directly: attitude has to come out of a
gyro+accel+mag filter and velocity out of integration. If you plan to port the controller to
hardware, build it on the raw group and treat the rest as an oracle for training only.

Verified on a free-floating spin (1 rad/s about X, 2 rad/s about Z): gyro tracks the commanded
rates, the quaternion sweeps roll 0 -> 165 deg, and the magnetometer swings across its full
range - so heading is observable from the raw signals alone, with no external reference.

## Whole-body state

`com_pos`, `com_vel`, `com_angmom` - centre of mass position, velocity and angular momentum
of the entire robot. This is what actually has to be steered: the hook only holds while the
load vector stays inside its throat.

## Hands

| sensor | |
|---|---|
| `ft_force_L/R`, `ft_torque_L/R` | 6-axis wrench through the hand root |
| `grip_L/R` | touch pad filling the hook throat |

Cross-check while hanging: the two hand force sensors read 53.9 + 53.6 = **107.5 N** against a
robot weight of 109.2 N, the 1.7 N balance going through the torso resting on the wall.

# Grip: flexing the finger does tighten it, but only up to -40 deg

Measured by ramping an outward pull on the torso until the hooks let go:

| finger angle | throat pad force | pull-off force |
|---|---|---|
| 0 deg | 115 N | 764 N |
| -5 deg | 130 N | 1024 N |
| **-10 .. -40 deg** | 125-151 N | **> 1200 N** |
| -45 deg | 155 N | **72 N** |
| -50 deg | 155 N | 60 N |
| -55 deg | 154 N | 48 N |

There is a cliff at **-45 deg**. Past it the hook levers itself off: the rung migrates from
23 mm to 19 mm along the throat and pops out, even though the pad force keeps rising.

Two consequences for the controller:

1. Hold on at **finger = -25 deg** (`GRIP_SET`), which sits 15-20 deg clear of both edges.
2. **Do not use pad force as the grip-quality signal** - it peaks exactly where the grip fails.
   Use the seat position (`ClimbingEnv.seat()`, rung's local x inside the throat) or the
   finger angle band instead.

# Capture window

For the hook to seat rather than merely perch on top of a hold, the hand must arrive within:

| | tolerance |
|---|---|
| fore/aft | **+-25 mm** |
| height above the rung axis | **-10 .. +70 mm** |

Approaching from above is far more forgiving than approaching sideways, so plan the reach as
"go high, then drop onto the hold".

# Episode termination - `climbing_env.py`

```python
from climbing_env import ClimbingEnv, GRIP_SET
env = ClimbingEnv()          # loads scene.xml
env.reset()                  # keyframe "hang_on_rung"
while True:
    env.step(ctrl)
    done, why = env.terminated()
    if done: break
```

| reason | trigger |
|---|---|
| `blowup` | non-finite state, or max abs(qvel) > 100 rad/s |
| `fell` | base_link below z = -1.2 m |
| `released` | neither hook seated for 0.6 s |
| `success` | base_link above z = 2.00 m |
| `timeout` | 60 s |

All five verified to fire: holding the grip survives 12 s untouched; commanding the arms
straight down reports `released` at t = 1.03 s.

Also provided: `env.encoders()`, `env.imu()`, `env.hand('L')`, `env.latched('L')`,
`env.grip_ok('L')`, `env.seat('L', geom)`.
