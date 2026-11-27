"""MyActuator RMD-X15-P20-450, vendor STEP (step.parts myactuator_rmd_x15_450), put in the
joint frame: +Z the axis, z = 0 the OUTPUT face, the body at z < 0 (as ak80_64.py).

The vendor file has its axis on +Y with the output face at y = 93 (its Ø144 cover's face
at 90, the rear connector hub down to y = 3), so it is turned +90 deg about X (+y -> +z)
and lowered 93 mm.
"""
import os
from cadgen import build123d as bd
from cadgen import step, read_step

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "vendor", "myactuator_rmd_x15_450.step")


@step(out="../../STEP/purchased/x15_450.step")
def x15_450():
    s = read_step(VENDOR)
    s = bd.Pos(0, 0, -93.0) * (bd.Rot(90, 0, 0) * s)
    s.label = "MyActuator RMD-X15-P20-450"
    return s


if __name__ == "__main__":
    x15_450()
