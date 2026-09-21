# cad — the mechanical design

Parametric CAD for the v2 robot. Vendor actuators are the manufacturers' own
STEP files; everything between them is drawn here. The same spec drives the
CAD, the collision primitives and the URDF, so the model and the simulation
cannot drift apart.

```
src/
  lib/spec.py      WHERE every joint is, WHICH actuator is on it, WHY each
                   offset differs from v1. The single source of truth.
  lib/motors.py    the actuator catalogue - datasheet numbers, nothing invented
  lib/connect.py   connector primitives: output_plate, face_plate, cheek,
                   bolt_access, and the measured keep-out each is checked on
  lib/build.py     the connectors themselves, one function per link
  purchased/       the three vendor STEP files put in the canonical joint frame
  robot.py         the assembly - run this
tools/
  verify.py        check every number against its source. Run after any change.
  check_one.py     ONE connector: one solid? clears both actuators? bolts on?
  check_pair.py    two neighbouring connectors, swept over their joint's range
  check_swing.py   a link against an actuator two joints away, swept over both
  balance.py       where the mass actually sits, per link and for the robot
  preview.py       write tmp/<name>.step and the same bolted between its motors
  arm_view.py      every arm connector so far, bolted into one chain
  robot_view.py    the whole robot, walked from base_link
  profile.py       read a vendor STEP's faces and cylinders -> the IFACE numbers
  gen_envelope.py  tessellate a vendor STEP -> its real radial keep-out
  check_fit.py     do the real actuator bodies fit? boolean intersection on solids
  torque.py        per-joint hold torque, measured on the built model
  massprops.py     mass, COM, inertia per link
  gen_urdf.py      writes ../urdf/
vendor/            downloaded vendor STEP, sha256 verified against step.parts
STEP/              outputs
```

```bash
python src/robot.py        # build STEP/robot.step
python tools/verify.py     # 18 checks
cadgen viewer --json       # review in the browser
```

## Conventions

**Canonical actuator frame.** Every actuator, whatever the vendor file does:
`+Z` is the joint axis, `z = 0` is the output flange face, the body occupies
`z < 0`. `src/purchased/*.py` holds the one transform each vendor file needs.

**Link frame.** URDF convention: origin on the proximal joint axis, identity
rotation at q = 0. `BODY_DIR[joint]` points from the joint origin toward that
actuator's body, so the child structure grows the other way. `MOUNT_AT[joint]`
slides the actuator along its own axis — kinematically free clearance.

**One connector pattern for the whole arm.** Every link is two plates and a
web. One plate lies flat on the proximal actuator's OUTPUT flange, the other on
ONE housing face of the distal actuator; each is exactly the diameter of the
face it sits on and carries a clearance hole for every one of that face's
bolts, continued on through the web so a driver can reach them. The web is a
single plate whose outline is tangent to both rims, so it follows the two faces
instead of butting onto them. Nothing wraps an actuator: each one is held by
the bolts on a single face and cantilevers off it, the way the open-source arms
on this class of actuator do it.

**Except at a corner.** Where the two axes are perpendicular to each other AND
to the link - both wrist links - the two faces lie in perpendicular planes and
no single plate sits flat on both. There it is one leg per face, and the two
share ONE SPINE where they meet: each leg's thickness is the other leg's
width, and each leg's taper is centred on the other leg rather than on its own
joint axis. Sized any other way the legs stand proud of each other and leave
two wings and a rib that carry nothing - which is exactly what the first two
attempts looked like.

The spine is a rectangle, not a square, because its two dimensions do not cost
the same. Across the arm is open space; fore-and-aft is where the NEXT link's
bracket has to pass, so a millimetre there costs the next link a millimetre of
stand-off. 16 x 14 mm is the smallest pair that keeps both section moduli over
522 mm3, which is what the AK60-39's 72 N.m stall needs at half of 6061-T6's
yield. `wrist_dev_L` then stands its plate 12 mm out, because `wrist_L`'s
16 mm leg reaches 10 mm past the flange it bolts to - measured at 2181 mm3 of
overlap before the stand-off was added, by `check_pair.py`.

**Balance is a layout rule, not an afterthought.** Every bracket cantilevers
off one face and every actuator's body hangs to one side of its own joint, so
each link has a lateral offset. All four of the arm's cross-axis actuators
used to have their bodies inboard - shoulder flexion, elbow, wrist flexion,
finger - and none of it cancelled: `balance.py` measured the arm's centre of
mass 13.5 mm inboard of the axis it hangs on. The elbow and the finger are
turned round, so the four now go in, out, in, out and the offset is 4.3 mm.
It is what OpenArm does with its joint origins, whose lateral offsets read
-30.1, +30.1, +31.5, -31.5, +37.5, -37.5.

`MOUNT_AT` for the shoulder came down 40 -> 30 mm at the same time. It is
there so the raised arm's rotation actuator sweeps past the abduction one, and
40 was a guess; `check_swing.py` sweeps that pair over both joints between
them and puts the clash below 27 mm. Thirty keeps a margin and brings 1.7 kg
of shoulder actuator 10 mm back toward the middle.

**The three torso links are frames, not connectors**, so they are panels and
decks rather than cheeks - but the rule they are drawn to is the same: every
actuator is held by a plate lying on a face it really has, with a clearance
hole for every one of its bolts. `base_link` is a stadium-section shell (the
140 x 180 box cut back to R90, because the waist turns the upper body over it
and the elbow actuator sweeps a 93 mm circle) with the waist actuator nose-up
on a deck in its top. `waist_link` is a cone that takes the Ø54 output flange
out to the frame's Ø88 bolt circle in one surface. `torso_upper` is a yoke:
two bands, each tangent to the Ø97 face its abduction actuator offers, laid
down the INBOARD side of it - the side the arm never sweeps - onto a deck,
with two side panels for fore-and-aft stiffness. The hull the arm links use
would have been a solid triangle there, and the arm swings straight through
the middle of it: 10 135 mm3, measured, before the bands replaced it.

**Which of the child's two flanges to bolt** is a real choice, not a rule.
The near one leaves the bracket on the same side as that actuator's output,
which is where the NEXT link has to grow, so the next link pays a stand-off;
the far one puts the bracket behind the actuator and leaves that side clear.
`wrist_dev_L` pays the stand-off because reaching behind the deviation
actuator would mean going 48 mm back past it. `shoulder_L` takes the far
flange instead, which put it on the inboard side, 31 mm clear of the torso,
and saved `upper_arm_L` an 11 mm stand-off. `FACE` in `build.py` records the
overrides.

**Lightening** is worked out from the web, not typed in: `_span_holes()` takes
the clear run left between the cable bore and the child's pilot boss and fills
it with as many holes as fit at 50 mm pitch. The forearm's 47 mm of run takes
one; the upper arm's 97 mm takes two.

**The hook** is the one link whose far end is not a flange. Its palm, lip and
65 mm width are the numbers the controller and the RL work were built on -
`urdf/climbing_robot.urdf` carries them as two collision boxes - so they are
copied, not redesigned; only the attachment is drawn. v1 never attached it: it
laid the palm flat against the plate's outer face and, to clear the finger
actuator, slid the whole hook 36.5 mm sideways, which is why its mesh and its
collision boxes disagreed about where the hand was. The palm is back on the
axis and the TOP of the palm is cut away instead, which costs nothing - the
actuator only reaches 27.5 mm down and the rung never touches there.

**Plate diameter** is the mating face, unless the actuator's bolt circle sits
so close to the rim that a socket head hangs over the edge, and then it is as
wide as the heads need and never wider than the actuator's own body. Only the
AK45-36 needs it: M3 at Ø48 on a Ø53 face leaves 0.8 mm of metal outside the
clearance hole, so its plates go out to the Ø55 barrel and finish flush.

The web's thickness is not free. Where the distal actuator's two mating faces
are close together, the web has to fit between them — the elbow gives 7 mm
between its front face and its output flange, and a 12 mm web there drove
straight into the forearm's own web (853 mm³, measured). `check_pair.py` is
what catches this; `check_one.py` cannot see the link next door.

## Interfaces, measured off the vendor files

Not from a drawing — `tools/` walked the STEP topology and found the bolt
circles by their cylindrical faces and z-positions.

| | output BC | output bolts | housing face z | housing BC | housing bolts | bore |
|---|---|---|---|---|---|---|
| AK80-64 | Ø28 | 6 × M4 | −8.0 | Ø89 | 8 × M3 | Ø21 |
| AK60-39 | Ø30 | 6 × M4 | −9.0 | Ø71 | 8 × M3 | Ø17 |
| AK45-36 | Ø27 | 6 × M2.5 | −4.0 | Ø48 | 6 × M3 | Ø8 |

## Two traps worth remembering

**Axis-distance arithmetic does not decide clearance for perpendicular
cylinders.** The first version of `check_fit.py` computed the distance between
axis lines minus the two radii; for perpendicular axes that is neither an upper
nor a lower bound, and it reported eight clashes that did not exist while
missing ones that did. It is a boolean intersection on real solids now, and it
costs about half a second per pose.

**`GProp_GProps.MatrixOfInertia()` is about the centre of mass, not the origin.**
Applying a parallel-axis correction to it produced negative principal inertias,
which is how the mistake surfaced. Verified against an offset box.
