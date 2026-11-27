"""MyActuator RMD-X12-P20-320, vendor STEP (step.parts myactuator_rmd_x12_320), put in the
joint frame: +Z the axis, z = 0 the OUTPUT face, the body at z < 0 (as ak80_64.py).

The vendor file has its axis on Y with the output flange facing -Y at y = -8.5, so it
is turned -90 deg about X (-y -> +z) and lowered 8.5 mm.
"""
import os
from cadgen import build123d as bd
from cadgen import step, read_step

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "vendor", "myactuator_rmd_x12_320.step")


@step(out="../../STEP/purchased/x12_320.step")
def x12_320():
    s = read_step(VENDOR)
    s = bd.Pos(0, 0, -8.5) * (bd.Rot(-90, 0, 0) * s)
    s.label = "MyActuator RMD-X12-P20-320"
    return s


if __name__ == "__main__":
    x12_320()
