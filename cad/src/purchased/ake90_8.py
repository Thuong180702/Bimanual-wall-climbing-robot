"""CubeMars AKE90-8 KV35, vendor STEP (step.parts cubemars_ake90_8_kv35), put in the
joint frame: +Z the axis, z = 0 the OUTPUT face, the body at z < 0 (as ak80_64.py).

The vendor file has its axis on Y with the output disc facing -Y at y = -11, so it is
turned -90 deg about X (-y -> +z) and lowered 11 mm. The driver board that goes behind
it is not in the file.
"""
import os
from cadgen import build123d as bd
from cadgen import step, read_step

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "vendor", "cubemars_ake90_8_kv35.step")


@step(out="../../STEP/purchased/ake90_8.step")
def ake90_8():
    s = read_step(VENDOR)
    s = bd.Pos(0, 0, -11.0) * (bd.Rot(-90, 0, 0) * s)
    s.label = "CubeMars AKE90-8 KV35"
    return s


if __name__ == "__main__":
    ake90_8()
