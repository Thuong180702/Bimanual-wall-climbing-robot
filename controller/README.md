> **Out of date since the hook re-mount (Sept 2026).** The hook bodies were rotated
> 180° about the forearm axis so the elbow flexes the way a human's does (see
> `rl/README.md`), which changes the arm posture every move here was tuned against:
> `test_climb` now fails phases 3, 4, 5, 6 and 8, where before the change it failed
> only phase 8. The servo weights in `moves.py` need re-sweeping against the new
> posture. Nothing structural is known to be wrong: phases 0, 1, 2 and 7 — keyframe,
> hold map, estimator, slide resistance — still pass.

# Climbing controller

Classical controller for the bimanual wall-climbing robot in `../urdf/scene.xml`. The
control loop reads only what real hardware has — encoders, IMU, F/T, grip pad — plus a
wall map known in advance, the way vision or SLAM would supply it. Ground truth
(`data.xpos`, `env.latched`) is used in tests only.

```bash
python -m controller.test_climb            # every phase
python -m controller.test_climb --phase 3
python -m controller.record descent out/descent.mp4   # or: climb
```

| file | what it is |
|---|---|
| `holds.py` | wall map: 39 rungs → 178 grasp points, reachability graph, A\* route |
| `kinematics.py` | anchor-rooted estimator and relative Jacobians |
| `moves.py` | `secure`, `_pull_up`, `reach`, `climb`, `traverse`, `route_to`, and the safety interlocks |
| `test_climb.py` | one assert-based check per build phase |
| `record.py` | writes a move to mp4 — `mujoco.Renderer` into `ffmpeg` on stdin, no video dependency |
| `holds.json` | built offline by `holds.build`; the runtime only reads it |

## State

| phase | | |
|---|---|---|
| 0 | keyframe replay is stable | **passes** — 0.00 mm base drift over 5 s, reproducing the model README |
| 1 | hold map + planner | **passes** — 178 grasp points, 2118 edges; every rung except the 7 top-of-wall ones has a hold 0.05–0.50 m above it, exactly as the model README claims. Vertical moves are now priced by how much rung is left at both ends, and **one pair of rungs in the whole scene** survives that: `rung_front_low_0` (0.55) and `rung_front_line_0` (1.00) |
| 2 | estimator + Jacobians | **passes** — free hand predicted to 0.58 mm from encoders + IMU alone, attitude to 0.28°, all 15 relative-Jacobian columns match finite differences to 3e-7 |
| 3 | one hand takes the next hold | **passes** — the left hand steps along a rung and latches: seat (20.0, 3.4, −83.1) mm, near the middle of the throat, finger in the band that holds, the anchored hand never goes slack |
| 4 | the other hand does the same step | **passes** — the right hand arrives 1.6 mm fore/aft (window ±25) and 2.4 mm above the rung (window −10…+70), 12° off in yaw |
| 5 | traverse, four alternating steps | **passes** — the robot side-steps along a rung and finishes with both hands on: 98 mm of travel in 255 s of simulated time |
| 6 | climb, on the slide-safe pair | **passes** — the left hand changes row: 450 mm down onto the continuous rung, landing **13 mm from the hold** (+5 mm fore/aft against a ±25 window, −1 mm in height against −10…+70), in **51 s**. Two seconds later it carries **108 N of the robot's 109** — the weight transfers on its own, with no separate commit step needed. The same phase also checks that a hold on the segmented row above is *refused* with the reason. The 144 mm it drifts along the rung is the axis the move deliberately ignores |
| 7 | the hook's restraint along its rung | **passes** as a characterisation: it records the sliding, so that changing the hook geometry shows up here |
| 8 | route: the whole robot down a row and back | **half of it works.** `route_to` walks the map's route and the robot **moves down a row on both hands** — `hands on ['rung_front_low_0', 'rung_front_low_0']`, base −295 mm, 74 s, nothing slack. It then falls on the way back up. Going down is done; going up is not |
| — | ascent, wall transfers, full route | not reached |

## How a reach works

1. **Pull up on both hands.** The free hand cannot rise relative to its own hold, so the
   height for a reach is won while both hooks are still on.
2. **Grip harder with the hand about to take the load**, then open the other finger.
3. **Servo the hand across on a learned Jacobian.** Position with both arms, then aim the
   hook with the free arm, then drop on. Nothing is planned open loop.
4. **Close the finger**, re-anchor the state estimate, restore the resting grip, verify
   the latch from load and finger angle — not from geometry.

Two ideas carry the whole thing:

**Everything is relative to the hand that is holding on.** That hand is latched on a rung
whose position the map gives, so its pose is known and the rest follows from the encoders.
The same framing gives an exact Jacobian for the pinned chain:

```
J_rel = (J_free - J_anchor)[:, joints] + skew(p_free - p_anchor) @ J_rot_anchor
```

Differencing removes the floating base; the skew term removes the anchor frame's own
rotation. Without that second term a waist yaw looks like it swings the free hand, when in
truth it only spins the body under a pinned hand — phase 2 catches this.

**The Jacobian is then corrected from the robot's own response.** The analytic one is
right about kinematics and wrong about the robot: command the arm up and the body turns on
the hook and cancels it. The servo maps *setpoints* to hand pose and updates that map from
what actually happens (Broyden), which absorbs servo droop, the hook turning on its rung
and the body re-hanging, with no contact model.

## What was measured on this model

Numbers, not guesses — each one changed the design.

| | measurement |
|---|---|
| Reach, anchored arm frozen | the free hand reaches 0.10 m sideways and **cannot reach a single point above its own hold**. Every reach must move both arms — as the model README's 0.671 m figure also assumed ("optimising over all 13 arm/waist joints") |
| Reach, however driven | the free hand still plateaus **~50 mm below** a hold at its own height. Hence the pull-up: from 0.73 m below the hands a 0.20 m lift puts the torso at 0.50 m, and that is the posture a reach captures from |
| Latchable orientations | only **1.8 %** of uniformly sampled configurations put the free hook in a latchable orientation |
| Joint-space branches | every posture that reaches a hold sits at **abduction 85–176°**, the arm flipped over the top; the hang keyframe's abduction = 0 is a different branch and no local IK crosses between them |
| Hang keyframe | **six joints already sit on their stops**, so an IK without limit handling diverges rather than stalls |
| Latched hook orientation | local **z down, x into the wall, y along the rung** — one formula for all three walls. Resting seat is **(23, 0, −85) mm**, not the centre of the detection box |
| Aiming the hook | the wrist has no authority over a 60° yaw, and turning the hand with the whole arm at the hold threw a converged 26 mm approach out to **334 mm**. The turn has to happen early, far from the hold |
| Seating | stopping with the throat point on the rung axis leaves the rung at the mouth (seat x 17.7 mm against a throat starting at 18). Backing off **8 mm** puts it at 18.3; 25 mm misses entirely |
| Anchor drift | mid-reach the anchored hook turned **17°, nearly all of it off the rung axis** — no pendulum model predicts it, and a posture solved before the move is 122 mm wrong by the time the robot arrives |
| Body lean | hanging on one hook the COM sits **70 mm** off plumb: the hook carries ~7.6 N·m and is not a frictionless pin |
| Sag | a kinematic plan lands the hand **~150 mm low** under 11 kg at kp = 800 |
| Swing on the hold | rotating the body on its rung walks the rung along the throat; geometry gives **−17.8°…+3.2°** before it leaves, and runs fell off at ~20° |
| IMU quiescence | attitude comes from TRIAD on gravity, so it is only valid when the accelerometer *is* reading gravity. After a pull-up the gyro had fallen to 0.24 rad/s — quiet by any threshold — while the accelerometer read 6.7 m/s² and the attitude was **19.7° out**, putting the far hand 120 mm from where it was |
| `env.latched` | a tight box around the nominal seat. A hook that has rotated falls outside it while still carrying the robot: with the other arm retracted entirely, one such hook held for six seconds, **167 N** through the wrist, dropping 21 mm. The controller judges a latch by load and finger angle instead |
| Along-rung observability | with both hooks on one straight rung, position **along** it is unobservable from the map and encoders — sliding the anchor slides the other hand identically. It becomes observable again once the hands are on different rungs |
| Base and wall | `base_link`'s front face sits exactly on the wall, so mesh penetration there measures estimator error, not posture |
| IMU filtering | waiting for the gyro to go quiet is not enough and waiting for a fixed time is not either: a sweep found the left hand needed 0.20 s of settle and the right 0.30 s, with no value serving both. Averaging only the samples that already read like gravity brings both hands to **2.5°**, where before it was 6.6° for one and 0.4° for the other |
| Servo saturation | during a right-hand reach **8–10 of 15 joints sat on their stops**, and the requests against them were silently clipped — so a third of every command vanished and the Broyden update learned a response that was not there |
| Arms are not in the same state | after a pull-up the left arm has **2 joints on their stops** (both wrists) and the right has **5**, including shoulder, elbow and forearm roll. Freeing an arm that is only slightly stuck costs more than it gains: doing it unconditionally made the working left-hand reach fall off the wall |
| The frame `body_shift` solves in | it should be the hand about to **move**, not the one that keeps holding. In `ik_body` the frame hand's arm is free to reconfigure while every other anchor is pinned by a zero-motion row that drives it onto its stops: choosing the mover takes both arms to 2 jammed joints, choosing the holder leaves 2 and 5 |
| The hook does not resist sliding along its rung | 20 N sideways (18 % of body weight) slides it 30 mm; 40 N, 150 mm; 60 N, 310 mm; 80 N runs it off the rung. The model README specifies pull-off force (>1200 N) and the capture window but never this axis. **This is what blocks climbing** |
| The plan's error *is* the slide | measured waypoint by waypoint, the gap between where the model says the hand is and where it actually is tracks the anchor's slide almost exactly: 33/7, 162/-142, 265/-225, 330/-330, 570/-591 mm. Not droop -- joint tracking held at 0.06 rad -- and not attitude |
| Tightening the route does not help | confining the hand to a 0.4 m corridor between the anchor and the hold worked as specified (the hand stayed inside it) and the anchor still slid 591 mm. The slide comes from the forces of reconfiguring the body, not from swinging the hand sideways |
| Open loop is safe, closed loop is accurate, neither is both | playing the planned joint path open loop keeps the robot inside the set the planner validated -- 11 of 11 waypoints, hook loaded at 99-150 N throughout -- but the hand lands 374 mm from the plan's prediction. Servoing to the same route's hand positions instead tracks each waypoint to 10-60 mm and falls, because the servo reaches those positions through joint configurations the planner never checked. **This is the open question** |
| Costing by hand travel works | a height floor plus edges priced by how far the hand moves takes the route's lowest point from z = 0.106 to z = 0.798, against a hold at 1.45. Priced by joint distance the route was valid and wild |
| Long legs must be split, and splitting is free | the roadmap's own edges move the hand up to 0.347 m, more than the servo crosses in one step. Every edge was checked feasible along its whole length, so points interpolated inside one need no re-checking: densifying to 0.146 m costs nothing |
| The space is connected | a roadmap of 3000 hangable postures (from 257 536 samples, **1.2 %** feasible) with 12-nearest-neighbour edges finds a route from the hang to a climbing posture: 12 waypoints, 22.4 rad, longest leg 2.70. An RRT with the identical feasibility test failed 12 times running — 4 targets x 3 settings x 12 000 iterations. **At 1.2 % density in 15 dimensions, uniform sampling is the problem, not the geometry** |
| The feasible set is 1.2 %, not 4.4 % | the earlier figure omitted self-collision. Filtering on walls alone also passed a "solution" with the humerus 21 mm inside base_link |
| The hook's rotation band is not the blocker | widening it from 40 deg to 70, 110 and 180 changes neither the density nor the outcome. It was the obvious suspect and it is innocent |
| Big reconfigurations are executable | a planned 4.7 rad route ran to completion with the hook loaded throughout, where the straight line to the same posture fell off the wall at 0.15 and 0.06 rad/s alike. The planner, not the speed, is what made it survivable |
| A climbing posture exists | solving directly for one, with the other hand anchored: **0.0 mm and 0°, collision-free** — but 1 start in 300 finds it, and it lies **5.18 rad** from the hanging posture. Driving straight there falls off the wall at 0.15 rad/s and at 0.06 rad/s alike: the straight line leaves the feasible set, so slowing down does not help |
| Lock-off tops out | postures keeping both hooks on one rung, collision-free and with the hands' separation unchanged, put the shoulder at best **0.36 m below** the hands — hand reach z ≈ 1.22 against a hold at 1.45. Raising the body with both hands on is not enough on its own |
| Lift is a criterion, not a term | putting "get the shoulder high" into the same least squares as the two-hook constraints gives **0 of 60** feasible answers; solving the constraints alone gives **52 of 60**, and picking the highest afterwards works |
| Climbing is not a reach problem | sampling 120 000 postures, the free hand can sit **1.128 m above** the anchored one, and 13 % of random postures put it at least 0.45 m up. The row above is reachable; the servo cannot find those postures from the hanging one |
| Shoulder flexion cannot be given up | freezing it and leaving the solvers abduction, rotation, the elbow, the roll, both wrists and the waist — 13 joints instead of 15 — makes the hold below **geometrically unreachable**: `solve_hanging_pose` finds a holdable posture on **5 of 5 seeds with flexion and 0 of 5 without**, and phases 3, 4, 5, 6 and 8 all fall. Abduction turns about x, the wall normal, so it only swings the arm *within* the plane of the wall; flexion turns about y, along the rung, and is the only shoulder joint that moves the hand toward or away from the wall. Left free, the solvers use it for nearly everything — over one traverse it sweeps **−162° to −75°** while abduction moves 2°, and over one climb it reaches −28° |
| Which way the elbow points is not a choice | at the hang, `shoulder_rotation` sits at **+90° of a −70…+90 range** and `forearm_roll` at **+85° of ±85** — both on their stops. With the elbow straight the two are redundant, so turning the shoulder to face the elbow outward turns the hook off the rung degree for degree (10° → 10°, 20° → 20°, 30° → 30°) and the roll that would cancel it has no travel left. The roll chain is entirely spent orienting the hook onto a horizontal rung. The mirrored configuration (rotation −70, roll −85) is worse, not better: 53.6° against 45.0 |
| Forward kinematics with the base pinned is a trap | held at the keyframe pose, FK says the elbow turns the hook off its rung one-for-one and abduction cancels it — elbow −30 alone gives 30.1°, at 30° of abduction it gives 10.2°, against a throat that tolerates ~21°. Both readings are true of a pinned base and neither survives a floating one: the body re-hangs instead of the hook turning. Capping the elbow bend at the abduction on that reasoning took the landing from **13 mm to 128 mm**, outside the capture window. Wrist load is the only test of this that has been right |
| Widening the shoulder range breaks the controller | `urdf/scene_wide_shoulders.xml` is the model with abduction opened 60° outward on each side (joint range **and** actuator ctrlrange). The arms do spread — 309 mm at 0°, **467 at −20°, 602 at −40°, 696 at −60°** — and the climber's flare finally exists. It also fails everything: with the outward flare the robot is on the floor in 9 s, and with the old inward one **phases 3, 4, 5 and 8 still fail** and the descent falls from 13 mm to 28 mm with the load split 52/58 instead of 108/1. The ctrlrange is what bounds every solver here — the IK, the joint-limit locking, the posture sampler, both planners — so widening it moves the search space every measurement in this table was tuned against. The model change is right and it needs the controller re-tuned behind it; `scene.xml` stays as it was |
| The shoulders cannot open outwards at all | `shoulder_L_abduction` runs **0…180°** and the hang sits at **0**, its lower edge; the right mirrors it. So the only travel the joint has swings the arm *across* the body. Measured with the elbows bent 45°, the two elbows go from **310 mm apart at 0° to 57 mm at 30° and 239 mm crossed at 90°**. Opening outward — a climber flaring their elbows — needs negative abduction on the left, which is out of range; it would give **638 mm of spread at −45°**. Changing this is a two-attribute edit to `scene.xml` and nothing else here can substitute for it |
| The pull-up that works closes the shoulders, not opens them | `_tuck` tucks the arms across the body and bends the elbows, and it beats the forward swing by a distance: the climb after it lands **13 mm** from the hold in 51 s, against 144 mm in 90 s, and the trailing hand's descent, which had never once completed, completes. It also transfers the weight by itself — 108 N against 1 N — where the forward swing needed `_commit` to push it and still only reached 53/62. It was written to open the shoulders outward and does the opposite; the result is real, the reason it was expected to work is not |
| ...and it lifts less, which is the point | tucked: torso 0.294, wall gap 67 mm. Forward: torso 0.500, gap 193 mm. The one that lifts 200 mm more and sits 126 mm further off the wall is the one that lands 10x worse. Height was never what the pull-up was for -- it is which **branch** the arm ends up in. Every posture that takes a hold sits at abduction 85-176 deg, the hang keyframe sits at 0, and no local IK crosses between them; the tuck crosses it |
| The two pull-ups are not interchangeable | the tuck drops the robot on all three reach phases, and the forward swing falls off the wall on the trailing hand's descent. `reach` keeps `_pull_up`, `climb` uses `_tuck` |
| The tuck is what makes the elbow bend available | bent alone, the elbows pull both hooks off by **−30°**; with the shoulders tucked they go to **−60°**. Abduction is commanded at 90° and the closed chain stops it near 20–30° — two hands pinned on one rung can only cross so far — so the command is deliberately past what is reachable |
| Straightening the shoulder is the wrong direction | it looks folded in the hang keyframe, and swinging it *further* is what helps. Commanding both shoulders and letting the robot settle, hooks on: **−180° → torso 0.258, 43 mm off the rung plane; −160° (the keyframe) → 0.272, 67 mm; −120° → 0.350, 145 mm; −80° → 0.471, 189 mm**, both hooks still latched. At −60° they come off. The elbow is already dead straight at 0° and on its stop, so the folded look is entirely the shoulder |
| The pull-up already does it | `_pull_up` takes the shoulders from **−160° to −86°/−91°**, the torso from z = 0.272 to 0.500 and the wall gap from **67 mm to 193 mm** — within 4 mm of the best the direct sweep found. Every move starts with it, so the pressed-against-the-wall posture only ever exists in the keyframe |
| A human pull-up is not available to this arm | bending both elbows with both hands on pulls the hooks off at **−30°** (latched at 0° and −15°, gone by −30°): the hands are pinned 0.314 m apart on the rung and bending draws them together. A human flares the elbows out to compensate — shoulder abduction — and this robot cannot: from the hang, abduction only travels *inward* |
| The pull-up saturates | commanding 0.20 m or 0.65 m both leave the torso ~0.45 m below the hands, with **both elbows dead straight against their stops**. It raises the body by swinging the shoulders, not by bending the arms |
| Bending the elbows is not allowed | biasing the pull-up towards bent arms does bend them — and pulls the second hook 0.29 m off its rung every time. Bending both elbows with both hands pinned a fixed distance apart is not something the geometry permits |
| Nullspace projectors need their own damping | sharing the task solve's damping makes `I − J⁺J` not a projector, and the posture bias leaks into the task rows; dropping the damping to make it one makes the solve unstable. Extra weighted rows do neither |
| Spending the anchored arm | both arms are needed (the approach fails outright with the free arm alone) but moving the anchored one moves the body: on a right-hand reach it carried the shoulder from 0.540 m from the hold to **0.638 m**, past the 0.581 m the arm spans. Pricing its joints at 0.15 against the free arm's holds the shoulder at 0.570 m and both hands then latch |
| Anchor slide | over one reach the anchored hook skates **180 mm along its own rung**, and the estimate error tracks it almost exactly. It is the one direction that is both unobservable and unimportant, so the servo now weights it down instead of chasing it |
| The whole estimate error *is* the anchor's seat offset | logged component by component through a climb: estimate error `[9, 97, −25] mm` against anchor seat offset `[−9, −97, 25] mm`, every sample, while TRIAD attitude read **0.0°**. The estimator pins its tree so the anchor's `THROAT_PT` sits on the rung axis, so wherever the rung really rides in that throat translates the entire world estimate by exactly that much. Along the rung it is 97 mm and harmless; **across it is 27 mm against a ±25 mm capture window** |
| A climb is a reach with the along-rung axis let go | `climb` used to aim at all three axes at the hold and landed 25 mm out and 19 mm low. Sharing `reach`'s approach-and-seat tail, with the anchor's rung axis deweighted — which the slide-margin guard is what licenses — lands it at +3 mm and −5 mm |
| The RRT was not failing, it was underfunded | at 1.5 % feasible, 6000 iterations is ~90 usable samples and finds nothing. 250000 connects the 4.6 rad climb in **3 s**, and a hopeless call still gives up in half a minute. Drawing half the samples from a corridor around the straight line between the endpoints keeps phase 6 working but does not crack the second hand's descent |
| Which hand takes a climb | the *nearer* one. Sending the far hand meant crossing the body to a hold 0.45 m below and 0.2 m past the anchor; the hand that moved seated correctly, and relaxing the anchor's grip at the end of the move levered **it** off the rung instead. Sideways is the opposite: the trailing hand goes |
| Route from a hold a hand is on | starting the route at the midpoint between the hands looks harmless. The midpoint is its own grasp point, so A\* inserted a step to reach it, and the climb it then planned went 100 mm across the body. From the leading hand's own hold the same goal is one edge — the one phase 6 passes |
| The roadmap works, and was not the blocker | 2500 nodes with corridor sampling and lazily validated edges, priced by hand travel: **7 s**, and paths far better conditioned than the RRT's — longest leg **0.72 rad against 2.57**. It is wired in as the fallback when the RRT gives up. It did not make the second hand descend, because what stopped that was never the search |
| A 3.6 mm self-contact stopped both planners | `Hanging` rejected any posture with a self-penetration past 2 mm, and after the leading hand's row change the robot's **own current posture** has one at 3.6 mm. Both planners return at `if not (ok(q_a) and ok(q_b))`, which reads as "no route" and is not one. The feasible set there is **17 %**, not 1.5 %, and the goal connected to 20 of its 20 nearest neighbours. The reference posture's own penetration is now the baseline, exactly as its wall-violation count already was |
| The trajectory pause had no bound | `_follow` waits while the body swings or the anchored hand unloads, and the wait did not increment its counter. With the anchor hanging at 1 N it waited **2006 s of simulated time**, standing still, until the episode timed out. Bounded at 3 s |
| Taking a hold below does not load it | the arms are stiff position servos, so the load follows the setpoints and not the statics — the torso already hangs half a metre below both holds and it changes nothing. After the descent the split is 53/62 N; by the next move it is **1 N on the new hand and 111 N on the old one**. Opening a finger does not unhook anything either: the hook is a slot round a cylinder, and it goes on carrying the robot wide open |
| The order is the climber's, and it works | put the weight on the lower hand while both are still on, *then* take the upper hook off, then move it. Traced: 59/48 N → 65/56 → **108/1**, hook lifted clear and staying clear at 114/1. Doing it the other way round — open the finger, plan, go — leaves the upper hook carrying the robot through the entire move. The transfer has to be pushed past parity to work: at 54 N against 57 N the hook would not come off |
| Which way to move the body to transfer | **down**, and it is not a guess. With the hands a row apart the torso sits level with the lower hold, so that arm is folded up and can pull on nothing; letting the body down puts it back at the extension it works at. It took 0.3-0.5 m of descent, in 0.12 m steps, checked by load each time |
| `seat_est` is blind for the hand that matters | the estimator pins the anchor's throat point onto the recorded rung position, so `est.throat(anchor) == anchor_pos` by construction and `seat_est(anchor)` returns the nominal seat forever. Through the whole descent it read a constant **(23, −85) mm** while the rung really walked 15 → 33 mm and left. The hold-integrity check in `tick()` therefore cannot fire for the anchoring hand — the one hand it exists to protect |
| The pivot band was too wide, and that was not the whole story | `Hanging` allowed the hook to turn **0.70 rad (40°)** where the measured geometry gives −17.8°…+3.2°, 21° in total, before the rung leaves the throat. Narrowed to 0.30 rad: phase 6 still passes, and the descent still walks its rung out. So body rotation about the rung is one mechanism and not the only one |
| Unhooking before planning does not transfer it either | adding the lift a reach uses, before the climb plans, moves the posture the planner solves from and the hand lands **58 mm** off fore/aft instead of 2. The seat servo caught it and refused rather than closing on nothing |
| Pressing on a wall reads exactly like holding | with the hook 26 mm past the rung axis and the finger shut on nothing, the pad read >20 N and the wrist 86 N, so `latch_confirmed` said yes. The servo already knew it had not arrived; its return value was simply being dropped |

## Where it stands

Both hands take holds, the robot traverses, and **it moves down a row on both hands** —
`route_to` walks the map's own route, the leading hand lands 13 mm from its hold carrying
108 N of the robot's 109, and the trailing hand follows. Going back up still falls.

Two things got it there, and neither was a tuning. The first is routing the climb where
the hardware's one unrestrained degree of freedom does not matter: the hook slides along
its rung under 40 N (phase 7), so the map prices every vertical move by how much rung is
left at both ends, and exactly one pair in the scene survives — `rung_front_low_0` at 0.55
and `rung_front_line_0` at 1.00, both 2.4 m unbroken. Everywhere else `climb` refuses and
says why.

The second is how the body gets set before the hand moves. Asking IK for 0.20 m of torso
lift swings the shoulders *forward*, lifts 200 mm more and sits 126 mm further off the
wall — and lands 144 mm from the hold. Tucking the shoulders *across the body* and bending
the elbows lifts almost nothing and lands 13 mm. Height was never the point: the reach
postures live at shoulder abduction 85–176° and the hang keyframe sits at 0, with no local
IK crossing between them, so every climb until now was servoing towards somewhere it could
not get to from where it was. That credit is the user's, not mine — I had the branch fact
in this table the whole time and did not connect it to the pull-up.

What is still weak:

- **The ascent grinds, then falls.** Hanging one-handed off the *low* rung with the free
  arm reaching 0.45 m up, the body never goes quiet: every `_follow` spends its full 3 s
  brake wait and every `_settle` its full budget, so one climb burns **3276 s of simulated
  time** (498 s of wall clock) and ends on the floor. The same pair descends in 51 s.
  Nothing about the move is different except which hand is below.
- **`env.latched` is no longer usable as a pass/fail.** It calls both hooks off while one
  of them is carrying 108 N. Phase 6 now asserts the model's own capture window plus the
  load through the hand, which is the physical statement; the tight box was never one.
- **The rung walks along the anchor's throat** during a long one-handed reconfiguration —
  15 → 33 mm and then out. The tuck made the descent short enough that it no longer
  matters there. It will come back on any move that stays one-handed for long.
- **Nothing can see that happen.** The estimator pins the anchor's throat onto the recorded
  rung position, so `seat_est` of the anchoring hand is a constant by construction, and
  the wrist load gives no warning either — 90–113 N right up to the moment it let go.
- **The traverse creeps.** Each step aims about 100 mm along the rung and gains roughly a
  quarter of that, on a coarse/fine along-weight split found by sweeping, not derived.

What I would do next, in order:

1. **Trace the ascent.** The descent works and the two differ in one thing: which hand is
   below. Everything the descent needed — the tuck, the transfer, the release — has a
   mirror image going up, and none of it has been checked in that direction.
2. **Give the shoulders outward travel.** Two attributes in `scene.xml` — the abduction
   ranges start at the hang posture, so the joint can only close across the body, and the
   climber's flare is simply not in the model. Everything above works around that.
3. **Then try the tuck, or a real flare, on `reach`.** It drops the robot as a straight substitution,
   but the reach was tuned around the forward swing for months of measurements; the branch
   argument applies just as much to a sideways step.
4. **A tug test before committing to a new hold.** Pull the hand outward along the wall
   normal and watch the wrist load: a hooked hand resists (>1200 N pull-off), a hand
   pressed on the wall goes slack immediately.
5. **Give the hook something to hold along the rung** — a deeper throat, a higher-friction
   pad, a second contact, or a grooved rung. The routing above is working around this, and
   phase 7 records the current behaviour so a change shows up there.
6. Then the wall transfers and the full route.

A note to whoever picks this up, me included. Three times now the thing needed next was
something deleted in an earlier pass as "superseded": `balance_pivot`, the RRT, and the
roadmap. They were superseded for the *approach*, and I generalised from that. Evaluating
a hypothetical posture without hanging it at its balance angle is meaningless -- the hang
keyframe itself then reads as 0.315 m off plumb and driven into the wall -- and every
large move needs a planner. Delete narrowly.

And a fourth: this pass added `_unjam` to `climb` on a hunch that the moving arm was
sitting on its stops. The evidence never supported it, and the table above already
recorded that calling it unconditionally breaks the left hand. It broke the left hand.
The hunch cost three runs; the table had the answer before the first one.
