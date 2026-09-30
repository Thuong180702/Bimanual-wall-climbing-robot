"""Livox Mid-360, Livox's own STEP, put in the sensor frame.

Sensor frame: +Z = the sensor's axis (away from the mounting face), z=0 = the
bottom mounting face, +X = toward the M12 connector. The vendor file has its
axis on +Y with the bottom face at y = -25.92 and the connector toward +X, so it
is turned +90 deg about X (y -> z, z -> -y) and lifted 25.92.
"""
import os
from cadgen import build123d as bd
from cadgen import step, read_step

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "vendor", "livox_mid360.step")


@step(out="../../STEP/purchased/mid360.step")
def mid360():
    s = read_step(VENDOR)
    s = bd.Pos(0, 0, 25.92) * (bd.Rot(90, 0, 0) * s)
    s.label = "Livox Mid-360"
    return s


if __name__ == "__main__":
    mid360()
