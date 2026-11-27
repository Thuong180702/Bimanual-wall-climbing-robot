# Legs for the climbing robot: design

2026-11-24. Status: for approval.

## Intent

The robot gets two legs so it can **walk on the ground like any ordinary humanoid**.
Climbing does not change: the **legs hang free while it climbs, and the two arms carry the
whole robot, legs included, through every climb** the lessons ask for (lessons 1-9, the
swings included). The user chose this over legs that stand on rungs: the climbing method
and its RL stay as they are, and the price is paid in the arms.

## Goals

1. Two 6-DOF legs, G1-class, drawn in the CAD to the same standard as the arms (vendor
   actuator STEPs, connectors bolted to real motor faces, `check_one` / `check_pair` /
   `check_fit` clean), on the hip sockets the pelvis already has (`spec.HIPS`).
2. Every leg joint holds at least **1.3x peak torque** over its load cases (below).
3. Every arm joint still holds at least **1.3x peak torque** over the full joint box AND
   the climbing envelope (`cad/tools/torque.py`, `verify.py` section 6) **at the final mass
   of the whole robot with legs**, re-actuated where it does not.
4. The legs have a **climbing stow pose** (knees and hips folded) that clears every
   lesson's walls and rungs, held by the leg actuators without overheating.
5. Human proportions (the arms are a 1.40 m person's).

## Non-goals (this round)

- No legs in the training simulation or RL: `urdf/scene.xml`, the lessons, `rl/` and the
  paper's robot stay as they are. The full robot is a separate model (see Constraints).
- No walking controller, no gait, no locomotion policy.
- No legs standing on rungs while climbing.

## Anti-goals (failure even if it "works")

- An arm joint under 1.3x anywhere in its box or envelope at the final mass.
- Legs that need the arms to lift them clear of a rung row below (the stow pose must be
  the legs' own job).
- Breaking the paper's current robot or its reference climbs.
- Shapes that cannot be built (wrapping motors, plates wider than the motor face,
  unbolted parts): the rules in the CAD connector workflow hold for the legs too.

## Constraints and what exists

- Pelvis (`base_link`) has a socket for each hip PITCH, drawn for an AK80-64 (G1 order:
  pitch, roll, yaw): it is **redrawn for RobStride 04** (its own face, bolt circle and
  body; vendor STEP fetched and checked like the others), hips about 146 mm apart, axis 425 mm under the shoulders, range -150..60 deg
  (`spec.HIPS`, `HIP_Y 0.073`, `HIP_Z -0.020`); the harness path from the legs (window
  over each hip, the pelvis's open top, a loop under the chest deck) is planned.
- Load path when climbing: hooks -> arms -> chest -> **waist actuator (AK60-39)** -> pelvis
  -> legs. The waist bearing then carries the legs hanging under it.
- Robot today: 24.50 kg, worst arm margins 1.36x elbow, 1.44x finger, 1.50x wrist flexion
  (`urdf/t_box.json`, `t_env.json`). A one-hook hang's joint torques scale with total
  mass, so **any** added mass takes the elbow under 1.3x.
- Actuators in use: CubeMars AK80-64 (120 N.m peak, 0.85 kg, 75 rpm no-load at 48 V),
  AK60-39 (72 N.m, 0.75 kg), AK45-36 (24 N.m, 0.34 kg); one CAN protocol, one driver.
  CubeMars has nothing stronger than AK80-64. Outside it: RobStride 04 (120 N.m, 1.42 kg,
  200 rpm), MyActuator RMD-X12-320 (320 N.m, ~2.4 kg).
- Battery: 48 V 6 Ah in the chest drawer (spec.BAY).
- `spec.JOINTS` feeds `gen_urdf.py`, `_gen_scene.py` and the RL's 17-action robot; adding
  joints there changes the training robot.

## Prior art (looked at)

- Unitree G1: 6 DOF a leg (hip 3, knee 1, ankle 2), hip pitch -> roll -> yaw, ankle
  pitch+roll by two motors on the calf driving rods (parallel ankle), knee 90 N.m (EDU 120),
  thigh + calf 0.6 m, ~35 kg, 1.27-1.32 m.
- Booster T1: 6 DOF legs, joints up to 130 N.m, ~30 kg, 1.18 m.
- Unitree H1 (knee 360 N.m, 47 kg), Berkeley Humanoid Lite, ToddlerBot: same 6-DOF layout.
- Ladder-climbing humanoids (HRP-2, DRC-Hubo) put their feet on rungs; this robot will not.

## Approaches considered

| | 1. G1-class legs (chosen) | 2. Light legs |
|---|---|---|
| hip pitch, hip roll, knee | AK80-64 | AK80-64 |
| hip yaw | AK60-39 | AK45-36 |
| ankle (2 motors, rods) | 2 x AK60-39 | 2 x AK45-36 |
| a leg | ~6 kg | ~4.9 kg |
| robot with arm upgrades | ~41 kg, ~43 with RobStride 04 (estimate) | ~35 kg (estimate) |
| arm joints to re-actuate | shoulder roll, shoulder rotation, elbow, wrist flexion, finger (wrist deviation on the edge) | elbow, wrist flexion, finger |

The user chose **1**: legs as strong as an ordinary humanoid's, whatever the arms then
need. (2 was the lighter path with weaker ankles and fewer arm changes.)

Then, for walking speed, the user chose **RobStride 04 at hip pitch and knee** (120 N.m
peak, 200 rpm no-load, 1.42 kg, 48 V, CAN) over AK80-64 there (75 rpm: a slow walk of
about 0.4-0.5 m/s): a walk like an ordinary humanoid's (about 1 m/s and up, estimate), at
about +2.3 kg (robot ~43 kg) and a second actuator vendor and CAN protocol in the legs.
Hip roll and yaw, which do not swing fast in a walk, stay CubeMars.

## The design

### Leg kinematics (per leg, mirrored)

| joint | axis | actuator | range (start; set by `check_fit`) |
|---|---|---|---|
| hip_pitch | y | **RobStride 04** (socket redrawn for it) | -150..60 deg |
| hip_roll | x | AK80-64 | G1-like, about -30..+90 outward |
| hip_yaw | z | AK60-39 | about -45..+45 |
| knee | y | **RobStride 04** | 0..150 deg |
| ankle_pitch | y | AK60-39 on the calf, rod | about -45..+30 |
| ankle_roll | x | AK60-39 on the calf, rod | about -25..+25 |

- Hip pitch -> roll -> yaw, the roll offset sideways from the pitch output at nearly the
  same height, as the shoulder (and G1). Yaw runs down the thigh.
- Lengths (my call, a 1.40 m person scaled toward G1): thigh (hip to knee) **0.32 m**,
  calf (knee to ankle) **0.32 m**, ankle axis **0.06 m** over the sole. Hip height ~0.70 m,
  shoulders ~1.13 m, top of the lidar cage ~1.32 m.
- Ankle: the two ankle motors sit high on the calf, just under the knee, and drive ankle
  pitch + roll through two rods (G1's parallel ankle): the mass stays near the knee.
- Foot: flat, about 0.20 x 0.09 m, a rubber sole, ankle axis over a point about 40 % from
  the heel (my call; the builder sizes it from the stance check).

### Load cases and checks

- **Legs** (each joint >= 1.3x peak): standing on both legs at a 90 deg knee squat; standing
  on one leg at 30 deg knee; the stow pose held while hanging (hip pitch and knee holding
  the folded leg up); ankle on one leg with the COM over the toe or the heel. Expected
  (41 kg): knee ~64 N.m one-legged at 30 deg -> 1.9x on AK80-64; ankle ~40 N.m one-legged
  shared by two AK60-39.
- **Legs walking** (my call): one leg carrying 1.5x the robot's weight at 20 deg knee
  (a walking stance with its landing), each joint >= 1.3x peak.
- **Arms, static**: `torque.py` box and envelope on a model of the WHOLE robot (legs in the
  stow pose and hanging straight), each joint >= 1.3x. Re-actuate until it passes, then
  iterate: heavier arm actuators raise the load on everything above them.
- **Arms, dynamic** (my call): every reference climb's recorded trajectory (`rl/fixtures`,
  lessons 3-9, the swings, and the lesson 8-9 climbs in `rsi.DROPPED` too: their recorded
  motion still says what a swing asks of a joint) is played through inverse dynamics
  (`mj_inverse`) on the full model, legs stowed: each joint's torque >= 1.3x under peak and
  its speed under the actuator's (a 2:1 stage halves it) at every step. No climb is searched
  again; this measures what today's climbs ask of the heavier robot.
- **Waist actuator (AK60-39)**: check its bearing against the legs' weight and moment
  hanging under it (axial and tilting load ratings from the CubeMars datasheet), and its
  yaw torque while walking; upgrade if short.
- **Stow pose clearance**: the folded legs meet no wall or rung over every reference climb's
  states on its own wall (`rl.rsi` states, `rsi.climb_wall`) and every lesson's hang.
- **Self-collision**: `check_fit` with the legs (named poses + random sweep), legs vs arms
  hanging down, legs vs each other.
- **Lidar** (my call): with the legs stowed, at every lesson's hang each rung keeps the
  cells the four lidars see today (`ClimbTask(k).reset()` seen per rung, the 2026-11-24
  method); the stow pose is chosen to keep them. Standing: the chest lidar's view of the
  ground at the feet, now through the thighs, reported.
- **Electronics** (my call): a mass allowance of 0.5 kg for the legs' CAN interface and
  harness, inside the chest bay or pelvis; the battery stays 48 V 6 Ah and the walking
  runtime it gives is reported, not designed for.

### Arm re-actuation (rule, my call)

Target: every arm joint >= 1.3x at the final mass. Estimated needs at ~43 kg (static
scaling of today's tables, to be measured): elbow ~154 N.m (need ~200 peak), shoulder roll
~105 (need ~137), shoulder rotation ~100 (need ~130), wrist flexion ~84 (need ~110), wrist
deviation ~58 (need ~76), forearm roll ~52 (need ~67, AK60-39 on the edge), finger ~29
(need ~38). Shoulder flexion (~59, need ~77) keeps its AK80-64.

- Stay on CubeMars where one of its actuators reaches the need: wrist flexion and wrist
  deviation AK60-39 -> AK80-64; finger AK45-36 -> AK60-39; forearm roll -> AK80-64 if the
  measured margin is under 1.3x.
- Where no CubeMars actuator reaches it (elbow, shoulder roll, shoulder rotation): an
  AK80-64 with a 2:1 toothed-belt stage (240 N.m, ~+0.25 kg, half the speed), the belt and
  its tensioner drawn as real parts. If a stage cannot fit a joint's geometry, one
  outside actuator family for those joints only (RobStride / MyActuator class), chosen by
  peak torque per kg.
- Every arm geometry change is checked as the arms were (`check_pair`, `check_fit`,
  `balance.py`, reach).

### Arm re-actuation, as built (user's choices, 2026-11-24/25)

The belt stages were not needed for the arms: the measured needs at ~64 kg are far past
the estimates above, and outside actuators reach them directly.

| joint | actuator | why |
|---|---|---|
| shoulder flexion, abduction, rotation; elbow | MyActuator RMD-X12-P20-320 (320 N.m, 2.37 kg, 48 V) | |
| forearm roll | AK80-64 | |
| wrist flexion, deviation | AK80-64 each, as a gimbal (the two axes cross) | brings shoulder -> hook to ~0.73 m, v2's; the hand had grown to 0.33 m (a person's 0.15) |
| finger | AK60-39 | |

The first build (X15-450 elbows on a 72 V boost branch, X12 wrist flexion, AKE90-8 wrist
deviation, 69.2 kg) gave way to this one when the user asked why the arms hung past the
knees and why the elbows alone ran at 72 V. With the shorter hand the X12 holds the elbow
(1.37x), so **every actuator runs on the 48 V battery**, no converter.

In the gimbal the flexion actuator points outboard (inboard it met the hip yaw actuators
with the arms down) and the deviation actuator points away from the wall (toward it, its
rim met a wall 60 mm behind the rung and levered the hook off: the hang fell). The left
wrist deviation holds 1.29x over the joint box, 1.30x over the climbing envelope:
accepted by the user (2026-11-25).

### Knee, as built (user's choice, 2026-11-25)

RobStride 04 on the thigh driving the knee through an 8MGT belt, 30 -> 45 teeth (1.5:1),
21 mm wide, 66-tooth 528 mm belt, 112.4 mm centres, the motor slotted to tension it: 180
N.m, 133 rpm. Direct, the RS04 (120 N.m) was 0.93x on a stair step-up at +0.3 g. The big
pulley is pocketed to a 5 mm web (0.40 kg). The belt's rating at 3.1 kN pull (peak) is
still to be read off Gates' tables.

### Models and files (my call)

- The full robot is a **variant**: `spec` gets the legs and the upgraded arm actuators
  behind a switch (e.g. `ROBOT = "v2" | "full"`), default `"v2"`. `"full"` writes
  `cad/STEP/robot_full.step`, `urdf/climbing_robot_full.urdf` and a hang/stand check
  scene; `"v2"` writes exactly what it writes today, byte for byte. The training robot moves
  to the full robot only when the user decides, as its own change.
- New CAD parts follow the connector workflow: **one connector at a time, rendered and
  confirmed before the next**, starting at the hip roll bracket on the existing pitch
  socket and going down the leg; then the arm upgrades.

## Left to the builder

- Exact ranges, offsets and bracket shapes, within the checks above.
- Foot size and ankle position from the stance and tipping checks.
- The ankle rod geometry (lengths, ball joints) from the ankle's range and torque.
- Belt stage layout per joint.

## Open

Nothing. Walking speed was settled by the user (RobStride 04 at hip pitch and knee).
