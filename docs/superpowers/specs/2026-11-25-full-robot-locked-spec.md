# Full robot (real build): locked kinematics and actuators

Status: **LOCKED by the user, 2026-11-25.** Replaces the
actuator choices in `2026-11-24-legs-design.md` for the robot that will be built. The
simulation robot of the paper (v2) is unchanged.

The user's requirements (2026-11-25): climb by hand as v2 does and pull the whole body up;
walk, **run fast, jump**, stairs up and down, sit down / stand up, climb onto platforms;
**off-the-shelf actuators only**, **one 48 V battery**, no mass ceiling; after the lock, CAD
in the style of current humanoids (enclosed, tight to the actuators, cooling, cable
routing, a clean look).

## 1. Actuators (all 48 V)

Chosen by `cad/tools/size_actuators.py` (lightest catalogue actuator per joint meeting
peak, continuous and speed needs, the robot's mass re-solved until it converges),
restricted to ENCOS plus CubeMars where ENCOS has nothing fast enough. Margins are peak
torque over the joint's need at the converged mass.

| Joint (x2) | Actuator | Transmission | Peak / rated N.m (at joint) | Speed rad/s | Need N.m | Margin | kg |
|---|---|---|---|---|---|---|---|
| shoulder flexion | ENCOS EC-A10020-P2-24 | direct | 330 / 100 | 12.9 | 253 | 1.30x | 2.18 |
| shoulder abduction | ENCOS EC-A13720-P1-11.4 | direct | 400 / 120 | 14.6 | 305 | 1.31x | 3.11 |
| shoulder rotation | ENCOS EC-A13720-P1-11.4 | direct | 400 / 120 | 14.6 | 274 | 1.46x | 3.11 |
| elbow | **MyActuator RMD-X15-P20-450** (user ruling 2026-11-26; a 72 V motor run at 48 V; was EC-A13720 direct, before that A10020-P2-24 + 1.5 belt, see 3.2) | direct | 450 / 145 | ~8.3 at 48 V (5.2 needed) | 337 at 101.8 kg (envelope) | 1.34x | 3.50 |
| forearm roll | ENCOS EC-A10020-P1-12 | direct | 150 / 50 | 14.7 | 89 | 1.68x | 1.36 |
| wrist flexion | CubeMars AKE90-8 | direct (gimbal) | 170 / 55 | 22.0 | 127 | 1.34x | 1.45 |
| wrist deviation | CubeMars AKE90-8 | direct (gimbal) | 170 / 55 | 22.0 | 128 | 1.33x | 1.45 |
| finger | CubeMars AK60-39 | direct | 72 / 24 | 10.3 | 49 | 1.47x | 0.75 |
| hip yaw | ENCOS EC-A8112-P1-18 | direct | 90 / 30 | 16.4 | 62 | 1.46x | 0.86 |
| hip roll | ENCOS EC-A10020-P2-24 | crank-rod-lever, >= 1.4:1 over the range (was a 1.25 belt, see 3.1) | 462 / 140 | 9.2 | 338 | 1.37x | 2.78 |
| hip pitch | ENCOS EC-A13720-P1-11.4 | direct | 400 / 120 | 14.6 | 259 | 1.54x | 3.11 |
| knee | ENCOS EC-A13720-P1-11.4 | four-bar 1.04-1.08:1 over knee 15-80 deg (>= 1.04 everywhere, 2.2 at 135), motor high on the thigh (cad/src/real/thigh.py) | 416 / 125 | 14.0 | 309 | 1.35x | 3.11 |
| ankle (pitch + roll) | 2 x CubeMars AKE90-8 | two rods (37 mm cranks square to the rods, foot ball joints 26 mm under the ankle axis), 1.43-1.46:1 at neutral (was 40 mm / 1.35, user ruling 2026-11-26), motors high on the calf 35 mm ahead of the leg line | 491 / 159 (pitch) | 15.1 | 425 at 101.8 kg | 1.16x (target 1.25) | 2.90 |
| waist yaw (x1) | ENCOS EC-A8112-P1-18 + crossed roller bearing (120 x 150 x 16) taking the tilt (was AK60-39: torque enough, its bearing not the ~120-300 N.m tilt) | direct | 90 / 30 | 16.4 | ~29 (running yaw) | 3.1x | 0.86 + ~0.35 |

Needs: arms = the one-hook hang torques measured on the CAD robot, scaled to the mass, with
a 1.3 target margin. Until 2026-11-26 the need was the worst pose of the whole joint box; by the
user's ruling (2026-11-26) it is the worst pose of the CLIMBING ENVELOPE (urdf/full/t_env.json:
elbow 211 vs 233 N.m on 63.85 kg, flexion 154 vs 184, abduction 164 vs 222, rotation 152 vs 197),
and the controller keeps a hanging arm inside that envelope (software limits). Margins at the
drawn 101.8 kg (2026-11-26): elbow 1.34, shoulder flexion 1.34, abduction 1.53, rotation 1.65,
forearm roll 1.70, finger 1.28, wrist flexion 1.18 / deviation 1.16 (under 1.3), hip pitch
1.33, hip roll 1.30, hip yaw 1.26, knee 1.16, ankle 1.16 (under 1.25). The arm needs are from the
63.85 kg simulation model scaled per kg: re-measure them on a model built from this CAD before
buying actuators; legs = running at 5 m/s (Schache et al. 2011, Table 1) x
1.25 for jumping and landing, hip roll from this robot's own stance model. Sources:
ENCOS joint-module datasheet V3.10 (rated/peak torque, speed, mass, measured load-speed
tables), CubeMars AKE90-8 / AK60-39 parameter sheets.

## 2. Mass budget (converged)

| | kg |
|---|---|
| actuators (31 on joints, 2 per ankle) | 57.9 |
| brackets / links (0.25 kg per kg of limb actuator) | 14.5 |
| fixed: torso structure, battery (1.6), Jetson, IMU, CAN, 4 lidars + cage, head riser, feet, harness | 14.8 |
| waist actuator | 0.75 |
| **total** | **87.9** |

## 3. Kinematics

**Legs** (KAIST/H1-style): hip **yaw -> roll -> pitch**, the three axes crossing at the hip
centre; hip centres **200 mm** apart (was 170, see 3.1); thigh 0.34 m, calf 0.34 m, ankle axis 0.06 m over the
sole; foot 0.22 m (toe 0.15 ahead, heel 0.07 behind the ankle). The knee actuator sits high
on the thigh (near the hip axis) driving the knee through a four-bar; both ankle actuators
sit high on the calf driving the foot through two rods (pitch together, roll against).

| Joint (left; right mirrored) | Range deg |
|---|---|
| hip yaw (+ = toe out) | -45 .. 35; both feet toe-out at once: at most 15 each (software) |
| hip roll (+ out) | -20 .. 45 |
| hip pitch (- = thigh forward) | -125 .. 30; past -100 the hip roll stays <= 15 (the knee actuator swings in under the pelvis: cad/src/real/pelvis.py) |
| knee | 0 .. 135 |
| ankle pitch (+ = toe down) | -50 .. 40 |
| ankle roll | -30 .. 30 for ankle pitch -10..40; -20 .. 25 when the toe is up past -20 (the rods meet the calf: cad/src/real/calf.py) |

**Waist**: yaw, +-45; the AK60-39 sits on the pelvis deck between the hip-yaw actuators, its output 227 mm over the hip centres (210 mm under the shoulders).

**Arms**: v2's order and ranges (shoulder flexion -> abduction -> rotation -> elbow ->
forearm roll -> wrist flexion/deviation as a gimbal (axes crossing) -> finger + hook), on
the full robot's 111 mm head riser. The joint offsets are re-spaced for the larger bodies by
v2's own rules (radius + air + web) when the arm modules are drawn; the arm needs above
already hold at the converged mass.

**Shoulder width (user ruling 2026-11-26, cad/src/real/arm.py).** The abduction axis sits at
y = +-225 mm (136 mm out from the flexion output face), not the 195 the barrel alone needs:
with the rotation A13720 (170 mm across) hanging beside the chest, at 195 the arm turned only
2-8 deg inward before it met the chest, the lidar guard or the pack, and v2's range (-12) is
used by a third of the reference climbs' states (`rl.rsi`, every flexion angle). At 225 it turns
12 deg inward at every flexion angle; the shoulder bracket's hub is a cup (floor on the
flexion output, wall out to y 139). The guard's front hoops moved in to y +-88 (lidar targets
unchanged: footholds -2 points, nothing else). Width over the abduction actuators: 620 mm.

**Hanging arm vs hips (user ruling 2026-11-26: software limit).** With the arm down (shoulder
flexion -40..+60) the elbow sits level with the hips and meets the hip-yaw link's outboard web
(z up to 200, 192 mm off the centre line), the hip-roll crank and rod or the pelvis. The
smallest clear abduction (deg, + = out; cad/src/real/arm.py, legs at zero, 3 deg steps):

| waist | flexion | rotation -70 | -35 | 0 | +45 | +90 |
|---|---|---|---|---|---|---|
| 0 | -30 | -12 | -12 | -12 | -12 | 0 |
| 0 | -10 | -9 | -12 | -6 | 0 | 12 |
| 0 | +10 | 9 | 0 | -3 | 9 | 12 |
| 0 | +30 | 12 | 6 | -3 | 0 | 12 |
| 0 | +50 | 6 | 0 | -3 | -12 | -9 |
| +45 toward the arm | -30 | -12 | -12 | 0 | 3 | 21 |
| +45 toward the arm | -10 | 12 | 6 | 9 | 15 | 21 |
| +45 toward the arm | +10 | 21 | 15 | 9 | 0 | 15 |
| +45 toward the arm | +30 | 9 | 0 | -9 | -12 | -12 |
| -45 | -10..+30 | -3 | -6 | -12 | -6 | 0 |

Raised arms (flexion <= -50) keep the full -12. The controller holds the abduction at or over
this table (interpolated, +3 deg margin); the hips stay as drawn.

### 3.2 Elbow drive (2026-11-26, when the arm was drawn)

The locked 1.5:1 belt carries ~13 kN at 495 N.m on a 75 mm pulley (more than 8M carbon
belts take, as at the hip roll). A crank-rod-lever cannot replace it: over the elbow's 107 deg
the best of 259k searched (A10020 behind the elbow) falls to 1.18 mid-range with 15-16 kN in
the rod. A 1.25 belt on the A13720 (500 N.m) fails the same way: 6.7 kN even on a 150 mm pulley.
No 48 V joint module near 500 N.m was found (2026-11-26). The elbow is a direct MyActuator
RMD-X15-P20-450 (user ruling 2026-11-26; was a direct EC-A13720, 400 N.m, 1.08x at 101 kg):
450 N.m peak / 145 rated, 3.5 kg, 166 x 69 mm (+ the 124 connector boss to 90), from its 2D drawing
(output 12 x M5 on 100, housing front 16 x M5 on 154, rear face 12 x M5 on 157). Rated 72 V; at 48 V
its no-load speed is ~8.3 rad/s (29.9 V/krpm, ratio 20.25; 7.3 on an empty 42 V pack) against the 5.2
the climbs need. Its driver takes 20-80 V (user, from MyActuator, 2026-11-27), so it runs on the one
48 V bus with everything else. Coaxial, hung
by its rear face from the humerus L (12 mm plate with a bore for the boss, an outer lip along the top,
the front reaching x 75 outboard of the forearm: 1.03 kg, FEA 226 MPa at 450 N.m), its axis 97 mm
under the rotation output; the forearm's hub is a ring on its output (6 of the 12 x M5).

### 3.1 Hip packaging change (user ruling 2026-11-25, after the B layout sweep)

Hip pitch and knee actuators sit inside the thigh volume (option B). The hip roll actuator
(A10020 + belt) can only sit behind the hip (the roll axis runs through the hip-pitch
actuator, the thigh sweeps the front), so it swings toward the other leg when the hip yaws.
`cad/tools/real_legs_b.py` (every body pair, 3 mm clearance) at 170 mm hip centres: one leg
toe-out 27 deg, both 10 deg; at **200 mm**: one leg 38 deg out / 45 in, both 15 deg. The user
chose 200 mm. The roll lever grows by 14 mm (0.143 -> 0.157 m in the stance model): need 3.50 ->
3.85 N.m/kg = 338 N.m at 87.9 kg; belt 1.25 (412 N.m, 1.22x) -> **1.4** (462 N.m, 1.37x; rated
140 vs 135 continuous; 9.2 rad/s vs 6 needed). A belt at that torque carries ~9 kN on a 100 mm pulley,
more than the 8M carbon belt data found supports, so the stage is a crank (40 mm, on the
actuator) - rod - lever (56 mm, on the roll shaft) behind the hip, like the knee and ankle;
its ratio rises toward the range ends (~1.4 mid, ~1.7 at the ends). The elbow's 1.5 belt
(495 N.m) failed the same check: see 3.2. Width: 318 mm over the thighs. The hip-yaw
output face sits 110 mm over the hip centre (the pitch actuator's roll sweep), the pelvis
bridges in front of the roll actuators and is cut open behind them.

## 4. Evidence per task

| Task | Status | How |
|---|---|---|
| hang / climb by hand (static hold, any arm pose) | verified (1.3x) on the 63.85 kg CAD robot; scaled to 87.9 kg | `check_full.py arms_static`, then the sizing |
| arm speeds over the 27 reference climbs | verified (>= 2.8x) | `check_full.py arms_dynamic` |
| sit-to-stand 42 cm | verified, knee 12 %, hip 16 % of the envelope | `run_check.py sit_stand` (inverse dynamics, centre of pressure on the sole) |
| stairs 18 cm up / down | verified, knee 173 N.m (43 %) | `run_check.py stairs_up/down` |
| step onto a 40 cm box | verified, knee 183 N.m (46 %), knee 112 deg | `run_check.py box` |
| squat jump | verified **~0.25 m** clean (42 V); 20-27 cm at 55-67 V | `gait_sim.py jump` (contact sim, torques clamped to the envelopes, CEM) |
| running 3.5 m/s | **estimated, not proven**: per kg like KAIST (knee 400 N.m @ ~12 rad/s vs 320 @ 10; ankle 2x170 @ 22 vs 2x176 @ 20) but 88 vs 75 kg; a hand-tuned controller falls even with 5x actuators (controller-limited) | proof needs an RL policy or CasADi collocation (deferred by the user) |

## 5. Open items (to settle before / during CAD)

- **Battery (user ruling 2026-11-26)**: a backpack, 13S4P 21700 high-drain 5 Ah cells, 20 Ah / ~0.96 kWh, ~4.1 kg with case and BMS, on rails on the chest's back plate (cad/src/real/torso.py). Was 13S6P 30 Ah / 6.1 kg (2026-11-25), before that 288 Wh (13S 6 Ah)
- **Lidars (user ruling 2026-11-25)**: three RoboSense Airy (360 x 90 hemisphere): chest front +20 deg up and -30 deg down (proud of the front face), one on a mast at the back of the top plate +20 deg up toward the back; chosen by cad/tools/lidar_airy_study.py over every 1-4 combination; no guard cage
- **Mass and strength (2026-11-26)**: every structural part checked at peak loads with a voxel FEA
  (cad/tools/real_fea.py, real_stress.py: 3 x weight on one leg + every actuator at peak, element stress x 1.4
  <= 200 MPa = 7075 yield / 2.5). The first drawings failed (261-403 MPa: loads went through single flat
  plates off-centre); the fixed parts pass (thigh 180, hip-yaw link 176, roll yoke 158, pelvis 118, calf 118,
  foot 110, chest 52 MPa) but weigh more: one leg's structure 9.5 kg = 0.78 x its actuators (budget 0.25).
  Projected robot ~108 kg against 87.9: margins fall to ankle 1.02x, shoulder flexion 1.06x, knee 1.10x, hip
  roll 1.12x. Not acceptable as is; options: a lighter structure method (carbon tubes for thigh / calf,
  shells, FEA-guided pocketing, a standard criterion: rare peaks <= 0.6 x yield, running loads vs fatigue),
  or re-sizing the actuators at the real structure ratio.
  User ruling 2026-11-26: lighter structure. Criterion moved to the usual one (rare peaks, surface stress
  <= 260 MPa = 0.52 x yield; the running stride's ~0.6 x then under 7075's ~160 MPa fatigue strength).
  Shelled / ribbed / pocketed to it: thigh 1.98 kg (239 MPa), hip-yaw link 1.60 (235), roll yoke 1.08 (164),
  calf 1.60 (130), foot 1.18 (110), pelvis 1.63 (226), chest 3.25 (52). One leg 8.6 kg = 0.71 x its
  actuators; robot ~106 kg; margins ankle 1.04x, shoulders 1.09x, wrist 1.11x, knee 1.12x, hip roll 1.14x.
  Machined 7075 plate construction does not reach the 0.25 x the sizing assumed.
  Arms drawn (2026-11-26, cad/src/real/arm.py, every part one solid, FEA <= 260 MPa): shoulder bracket
  0.84 kg (165 MPa), upper arm 1.23 (167), humerus 1.18 (245; an L on the elbow's rear ring, the user's
  call -- the U on both rings was 1.05 kg, 203 MPa), forearm 0.89 (221), wrist link 0.48 (174), gimbal
  0.33 (181), hand 0.38 (187), hook 0.35 (45): 5.67 kg per arm = 0.34 x its actuators (with the flexion
  one). Wrist (user's call): the deviation AKE90 centred on the wrist centre under the forearm, output
  forward at x 27; the flexion one outboard (output y 302); the finger 97 mm under W (v2 93.5).
  Robot as drawn 101.5 kg (legs 17.2 + 25.5 actuators, arms 11.3 + 28.7 + 4.4 shoulder
  flexion, chest/head/guard 3.8, pack and electronics 7.8, pelvis/waist 2.8); with harness, fasteners,
  bearings and pins (not drawn, ~3-4 kg) ~105 kg against 87.9. Margins at 105 kg: elbow (direct A13720)
  and ankle 1.05x, shoulder flexion 1.09x, abduction 1.10x, wrist 1.11-1.12x, knee 1.13x, hip roll 1.15x,
  rotation / hip yaw 1.22x, finger 1.23x, hip pitch 1.29x, forearm roll 1.41x.
  13S 15-20 Ah pack (~720-960 Wh, ~4-5 kg) is the realistic choice; it adds 2.5-3.5 kg the
  budget above does not carry.
  Mass round (user ruling 2026-11-26: no material change -- 7075 is already the lightest common
  metal for these strength-limited parts; AZ31B / ZK60 magnesium, carbon plate and titanium were
  compared): FEA-guided trims, each drawn and FEA'd again: chest 2.92 -> 2.29 kg (63 MPa), pelvis
  1.63 -> 1.50 (240), calf 1.57 -> 1.28 (knee hub 80 -> 60 mm, 142 with the 4.6 kN ankle rods), foot
  1.18 -> 1.08 (toe / heel / edge 121 / 35 / 88), upper arm 1.23 -> 1.15 (183), shoulder bracket 0.84
  -> 0.79 (168), hook 0.35 -> 0.27 (palm and lip 8 mm, the throat unchanged; one-hand 3 g catch
  232 MPa). Thigh, hip-yaw link, roll yoke, forearm, wrist link, gimbal, hand and the linkage parts
  kept: their idle material is bolted clamp round the actuator outputs or they carry 40-115 MPa
  through (the thigh also deflects 10.4 mm at the knee at peak load). With the 20 Ah pack and the
  X15 elbow: robot drawn 98.3 kg, ~101.8 kg with harness, fasteners and bearings.
- **ENCOS**: hollow-shaft bore (only the P2-24's 7 mm is published), maximum bus voltage,
  mounting drawings (the datasheet's dimension pages are images) -- ask ENCOS.
- **AKE90-8**: torque-speed curve (assumed flat to 60 % of no-load), driver choice.
- **Cooling**: rated torques are continuous at the vendors' heat-sink conditions; the shells
  must leave the housings' faces open to air or carry fins.
- Belt stages (elbow 1.5:1, hip roll 1.25:1): width/tooth profile against the vendors'
  rating tables.
