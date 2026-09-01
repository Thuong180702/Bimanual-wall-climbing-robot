"""Climbing controller for the bimanual wall-climbing robot.

Layering:
    holds.py       wall map (known in advance, like vision/SLAM would give) + route planner
    kinematics.py  anchor-rooted state estimate + relative Jacobians
    moves.py       secure / pull-up / reach, and the safety interlocks

The control loop reads only what real hardware has: encoders, IMU, F/T, grip pad.
Ground truth (data.xpos, env.latched) is used in tests only.
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, 'urdf')
SCENE = os.path.join(URDF, 'scene.xml')
HOLDS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holds.json')

if URDF not in sys.path:            # climbing_env.py lives next to scene.xml
    sys.path.insert(0, URDF)
