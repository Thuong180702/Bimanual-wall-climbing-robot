"""RobStride 04, vendor STEP (step.parts robstride_04), put in the joint frame.

Canonical joint frame (as ak80_64.py): +Z = the joint axis, z = 0 = the OUTPUT
face, the body at z < 0. The vendor file already has its axis on +Z with the
output (the planetary carrier: 9 x M5 and 3 dowel pins on a 36 mm circle) facing
+Z at z = 25.5 and the driver board at the back, so it is only lowered 25.5 mm.
"""
import os
from cadgen import build123d as bd
from cadgen import step, read_step

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "vendor", "robstride_04.step")


@step(out="../../STEP/purchased/rs04.step")
def rs04():
    s = read_step(VENDOR)
    s = bd.Pos(0, 0, -25.5) * s
    s.label = "RobStride 04"
    return s


if __name__ == "__main__":
    rs04()
