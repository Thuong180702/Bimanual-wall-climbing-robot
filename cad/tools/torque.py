"""Per-joint static hold torque, measured on the real kinematics.

Three load cases:

  HANG_EQ  one hook anchored, and the body hangs where gravity puts it - the
           whole robot rotated about the anchor until the COM is straight
           below it. This is the load the servos actually hold while climbing.
  HANG_ANY same anchor, but the pose is held anywhere in the joint box,
           swung out. The controller has to command this; it is the ceiling.
  REACH    base fixed, free arm out. Load = the distal subtree only.

No contacts, no inverse dynamics: FK from MuJoCo, moments computed here.
tau_j = axis . sum_i (r_i - p_j) x m_i g   over the load side of the cut.
"""
import json, os, sys
import numpy as np, mujoco

G = np.array([0.0, 0.0, -9.81])
NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
XML = sys.argv[2] if len(sys.argv) > 2 else \
    "/Users/admin/MyProject/Bimanual-wall-climbing-robot/urdf/scene.xml"
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "torque_req.json")

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

HINGE = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
NAMES = [m.joint(i).name for i in HINGE]
QADR = np.array([m.jnt_qposadr[i] for i in HINGE])
LO = np.array([m.jnt_range[i][0] for i in HINGE])
HI = np.array([m.jnt_range[i][1] for i in HINGE])

base = m.body("base_link").id
ROBOT = [b for b in range(1, m.nbody) if m.body_rootid[b] == m.body_rootid[base]]
PARENT = {b: m.body_parentid[b] for b in ROBOT}
MASS = np.array([m.body_mass[b] for b in range(m.nbody)])


def descendants(b):
    out, stack = set(), [b]
    while stack:
        x = stack.pop()
        out.add(x)
        stack += [c for c in ROBOT if PARENT[c] == x]
    return out


DISTAL = {i: descendants(m.jnt_bodyid[i]) for i in HINGE}
HOOK = {s: m.body("hook_%s" % s).id for s in "LR"}
ALL = set(ROBOT)
LOAD = {}                                   # anchored-hook load side of each cut
for s in "LR":
    for i in HINGE:
        LOAD[(s, i)] = (ALL - DISTAL[i]) if HOOK[s] in DISTAL[i] else DISTAL[i]
IDX = {k: np.fromiter(v, int) for k, v in LOAD.items()}
IDXD = {i: np.fromiter(DISTAL[i], int) for i in HINGE}
RIDX = np.fromiter(ALL, int)


def settle(xipos, xanchor, xaxis, p):
    """Rotate everything about p so the robot COM ends up straight below p.

    A body hanging off one hook is a pendulum; this is where it stops.
    One rotation about the horizontal axis that takes the COM offset onto -Z.
    """
    w = MASS[RIDX]
    com = (xipos[RIDX] * w[:, None]).sum(0) / w.sum()
    v = com - p
    n = np.linalg.norm(v)
    if n < 1e-9:
        return xipos, xanchor, xaxis
    a, b = v / n, np.array([0.0, 0.0, -1.0])
    k = np.cross(a, b)
    s, c = np.linalg.norm(k), float(a @ b)
    if s < 1e-9:
        return xipos, xanchor, xaxis
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return (p + (xipos - p) @ R.T, p + (xanchor - p) @ R.T, xaxis @ R.T)


CASES = ("HANG_EQ", "HANG_ANY", "REACH")
peak = {c: np.zeros(len(HINGE)) for c in CASES}
argq = {c: [None] * len(HINGE) for c in CASES}

# Optional climbing envelope: the hook only latches with the arm RAISED (v1
# README), so a climber lives in flexion -170..-70 with the elbow no more than
# half closed. Pass --envelope to size against that instead of the joint box.
ELO, EHI = LO.copy(), HI.copy()
if "--envelope" in sys.argv:
    for k, n in enumerate(NAMES):
        if n.endswith("_flexion") and "shoulder" in n:
            ELO[k], EHI[k] = np.radians(-170), np.radians(-70)
        elif n.startswith("elbow"):
            ELO[k], EHI[k] = np.radians(-90), 0.0
        elif n.endswith("_abduction"):
            lo, hi = LO[k], HI[k]
            ELO[k], EHI[k] = (np.radians(-45), np.radians(45)) if hi > 0                 else (np.radians(-45), np.radians(45))
            ELO[k], EHI[k] = max(lo, ELO[k]), min(hi, EHI[k])
            if ELO[k] >= EHI[k]:
                ELO[k], EHI[k] = lo, min(hi, lo + np.radians(45))

rng = np.random.default_rng(0)
for _ in range(NSAMP):
    q = ELO + rng.random(len(HINGE)) * (EHI - ELO)
    d.qpos[QADR] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    xi0, xa0, ax0 = d.xipos.copy(), d.xanchor.copy(), d.xaxis.copy()

    for k, i in enumerate(HINGE):
        r = xi0[IDXD[i]] - xa0[i]
        w = MASS[IDXD[i]][:, None] * G
        t = abs(float(ax0[i] @ np.cross(r, w).sum(0)))
        if t > peak["REACH"][k]:
            peak["REACH"][k], argq["REACH"][k] = t, q.copy()

    for s in "LR":
        p = xi0[HOOK[s]]                    # anchor ~ the hook body frame
        xiE, xaE, axE = settle(xi0, xa0, ax0, p)
        for case, (xi, xa, ax) in (("HANG_ANY", (xi0, xa0, ax0)),
                                   ("HANG_EQ", (xiE, xaE, axE))):
            for k, i in enumerate(HINGE):
                idx = IDX[(s, i)]
                r = xi[idx] - xa[i]
                w = MASS[idx][:, None] * G
                t = abs(float(ax[i] @ np.cross(r, w).sum(0)))
                if t > peak[case][k]:
                    peak[case][k], argq[case][k] = t, q.copy()

M = float(m.body_subtreemass[base])
print("model %s   robot mass %.4f kg   %d samples" % (os.path.basename(XML), M, NSAMP))
print("\n%-24s %9s %9s %9s %9s" % ("joint", "HANG_EQ", "HANG_ANY", "REACH", "lever_eq"))
out = {}
for k, n in enumerate(NAMES):
    e, a, r = peak["HANG_EQ"][k], peak["HANG_ANY"][k], peak["REACH"][k]
    out[n] = dict(hang_eq=e, hang_any=a, reach=r,
                  need=max(e, r), lever_eq=e / (M * 9.81), lever_any=a / (M * 9.81))
    print("%-24s %9.2f %9.2f %9.2f %9.3f" % (n, e, a, r, e / (M * 9.81)))
json.dump({"mass": M, "xml": XML, "samples": NSAMP, "joints": out},
          open(OUT, "w"), indent=1)
print("\nwrote", OUT)
