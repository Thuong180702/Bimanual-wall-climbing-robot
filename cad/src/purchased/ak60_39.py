"""CubeMars AK60-39 V3.0 KV80, vendor STEP, put in the joint frame.

Canonical joint frame: +Z = joint axis, z=0 = output flange face, body at z<0.
The vendor file already has its axis on Z, spanning -29.0 .. +38.0 with the
output end at +Z, so it is translated down by 38.0.
"""
import os
from cadgen import build123d as bd
from cadgen import step, read_step

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "vendor", "cubemars_ak60_39_v3_kv80.step")


@step(out="../../STEP/purchased/ak60_39.step")
def ak60_39():
    s = read_step(VENDOR)
    s = bd.Pos(0, 0, -38.0) * s
    s.label = "CubeMars AK60-39 V3.0 KV80"
    return s


if __name__ == "__main__":
    ak60_39()
