"""How far above the anchored hand can the free hand reach? Re-derived for v2.

v1 measured 0.671 m and set every rung row <= 0.50 m apart on the strength of
it. v2's arm is 116 mm longer at the wrist and its actuators are fatter, so the
number has to be measured again - and this time self-collision is ENFORCED,
which v1's optimisation did not do (its own reported best pose, abduction 180 /
flexion -177 / elbow -114, puts the upper arm through the torso).

Constraints, same as v1:
  - one hook anchored, the other free
  - the whole-body COM at least 50 mm BELOW the anchored hook (it is a hang)
  - both hooks in a latchable orientation (hook opening within 35 deg of down)
  - the free hand within 0.30 m sideways of the anchor
  - no self-collision
"""
import math, os, sys
import numpy as np, mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "climbing_robot_floating.urdf"))
d = mujoco.MjData(m)
HINGE = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
NAMES = [m.joint(i).name for i in HINGE]
QA = np.array([m.jnt_qposadr[i] for i in HINGE])
LO = np.array([m.jnt_range[i][0] for i in HINGE])
HI = np.array([m.jnt_range[i][1] for i in HINGE])
HK = {s: m.body("hook_%s" % s).id for s in "LR"}
MAXTILT = math.radians(35.0)


def evaluate(q, anchor="L"):
    d.qpos[QA] = q
    mujoco.mj_forward(m, d)
    if d.ncon:                                     # self-collision
        return None
    free = "R" if anchor == "L" else "L"
    pa, pf = d.xpos[HK[anchor]], d.xpos[HK[free]]
    for s in "LR":                                 # hook must still open downward
        R = d.xmat[HK[s]].reshape(3, 3)
        if math.acos(max(-1.0, min(1.0, -R[2, 2]))) > MAXTILT:
            return None
    if abs(pf[1] - pa[1]) > 0.30:
        return None
    # and it has to be a HANG: the body's centre of mass below the anchor,
    # otherwise the "reach" is measured from a hand the robot is standing on.
    com = (d.xipos[1:] * m.body_mass[1:, None]).sum(0) / m.body_mass[1:].sum()
    if com[2] > pa[2] - 0.05:
        return None
    return float(pf[2] - pa[2])


def search(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    best, bq = -9.9, None
    tried = ok = 0
    for _ in range(n):
        q = LO + rng.random(len(HINGE)) * (HI - LO)
        tried += 1
        v = evaluate(q)
        if v is None:
            continue
        ok += 1
        if v > best:
            best, bq = v, q.copy()
    # local polish
    step = (HI - LO) * 0.08
    for _ in range(12):
        improved = False
        for _ in range(4000):
            q = np.clip(bq + rng.normal(0, 1, len(HINGE)) * step, LO, HI)
            v = evaluate(q)
            if v is not None and v > best:
                best, bq, improved = v, q, True
        if not improved:
            step *= 0.5
    return best, bq, ok, tried


if __name__ == "__main__":
    best, bq, ok, tried = search(int(sys.argv[1]) if len(sys.argv) > 1 else 40000)
    print("free hand rises %.3f m above the anchored one" % best)
    print("%d of %d random poses were legal (%.1f%% rejected by self-collision"
          " or hook orientation)" % (ok, tried, 100 * (1 - ok / tried)))
    print("pose:")
    for n, v in zip(NAMES, bq):
        if abs(v) > 1e-3:
            print("   %-24s %7.1f deg" % (n, math.degrees(v)))
