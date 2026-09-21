"""CubeMars AK45-36 KV80, vendor STEP, put in the joint frame.

Canonical joint frame: +Z = joint axis, z=0 = output flange face, body at z<0.
The vendor file has its axis on Y, spanning -17.0 .. +37.0 with the output end
at +Y: rotate +Y onto +Z, then drop the flange to z=0.
"""
import os
from cadgen import build123d as bd
from cadgen import step, read_step

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "vendor", "cubemars_ak45_36_kv80.step")


@step(out="../../STEP/purchased/ak45_36.step")
def ak45_36():
    s = read_step(VENDOR)
    s = bd.Rot(90, 0, 0) * s
    s = bd.Pos(0, 0, -37.0) * s
    s.label = "CubeMars AK45-36 KV80"
    return s


if __name__ == "__main__":
    ak45_36()
