# Real robot CAD: style and process

For the locked robot (`2026-11-25-full-robot-locked-spec.md`). New code lives in
`cad/src/real/` and `cad/tools/real_*.py`; v2 and the `ROBOT=full` study stay as they are.

## What market robots do (and we copy)

- **Structure**: CNC-machined 6061/7075 aluminium links that carry the loads and hold each
  actuator by its own bolt circles (front and rear faces), topology-lightened; steel only
  where impacts concentrate (Unitree G1 teardowns: machined billet, steel shin).
- **Covers**: thin smooth shells (2.5-3 mm PA12/PC, printed or moulded) over the
  structure, linkages and harness; limbs taper from hip to ankle; each joint shows as a
  clean ring where the actuator's end cap sits flush (G1, H1).
- **Cooling**: the actuators' finned barrels stay in moving air (ENCOS A10020/A13720 barrels
  are finned); covers carry vent slots in line with the fins; joint heat also goes into the
  aluminium links through the bolted faces (G1: copper spreader at the knee, fans at the
  hip and the control board).
- **Cables**: through the actuators' hollow centres where they have one (ENCOS A10020: an
  8 mm cable hole; A13720: Ø29 centre), otherwise in covered channels along the links,
  clamped on both sides of every joint with a service loop sized to the joint's range; one
  48 V + CAN trunk per limb.
- **Ankle**: both actuators high on the calf, two rods behind the calf to a universal joint
  (G1, KAIST).
- **Knee**: actuator high on the thigh, a four-bar to the knee (KAIST).

## Process

1. **Layout** (`cad/tools/real_layout.py`): every actuator as its true-size body at its
   joint in the locked kinematics; overall widths, proportions, clearances in the extreme
   poses. Reviewed before any part is drawn.
2. **Modules**, one at a time, each rendered and confirmed: hip (yaw-roll-pitch) + thigh +
   knee linkage; calf + ankle linkage + foot; pelvis + waist; torso + battery bay; shoulder;
   upper arm + elbow belt; forearm + wrist gimbal; hand/hook; head + lidars.
3. Per module: structure first (load path, bolt circles, bearings), then covers, vents and
   cable channels; checks: no interference over the joint ranges, mass vs the budget,
   section stress at peak torque.

## Vendor interfaces read so far (ENCOS datasheet V3.10 drawings)

| Actuator | Body | Front (output) face | Rear face | Centre |
|---|---|---|---|---|
| EC-A13720-P1-11.4 | Ø170 x 84.5, finned barrel 50 | housing ring 12 x M4 on Ø155; output star 3 x M5 + 3 x Ø6 dowels on Ø40/50, Ø70 pilot | 20 x M4 on Ø155, Ø124/Ø95 boss | Ø29 |
| EC-A10020-P2-24 | Ø124 x 85, finned barrel 66 | 10 x M4 on Ø116.5; output 6 x M5, 3 x Ø6 dowels, Ø62/Ø30 | connectors, Ø94-100 | Ø8 cable hole |
| EC-A8112-P1-18 | Ø105 x 53 | 8 x M4 on Ø97; output 6 x M4 on Ø50, Ø54 pilot | 8 x M4 on Ø97, connectors | -- |
