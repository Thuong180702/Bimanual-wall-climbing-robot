"""CubeMars AK80-64 KV80, vendor STEP, put in the joint frame.

Canonical joint frame for every actuator in this project:
    +Z  = the joint axis
    z=0 = the OUTPUT FLANGE mating face (the child link bolts here)
    body occupies z < 0 (the parent link bolts to the housing)

The vendor file arrives with its axis on +X and the flange at x=0, so one
-90 deg rotation about Y puts +X on +Z and leaves the flange at z=0.
"""
import os
from cadgen import build123d as bd
from cadgen import step, read_step

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "vendor", "cubemars_ak80_64_kv80.step")


@step(out="../../STEP/purchased/ak80_64.step")
def ak80_64():
    s = read_step(VENDOR)
    s = bd.Rot(0, -90, 0) * s
    s.label = "CubeMars AK80-64 KV80"
    return s


if __name__ == "__main__":
    ak80_64()
