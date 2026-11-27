"""Which lidar layout for the real robot: RoboSense Airy (hemispherical: 360 deg about its
axis x 90 deg, i.e. everything within 90 deg of the axis, no blind cone; 0.1-30 m @10 %;
60 x 63 mm, < 240 g) placed on the full simulated robot (urdf/full, legs + arms), ray-cast
against the robot's own geoms (mj_ray, static walls/rungs ignored; the old lidars and
guard cage left out).

Scored at the hang (keyframe hang_on_rung, hands on the rung at z 1.0) on what climbing
needs: the start rung under the hands, the rungs above, the rung and wall below (where the
feet go), the wall behind; and standing on the floor on what walking needs: the ground ring
0.3-3 m round the feet, the ground at the toes, the space 1.5 m away up to head height.

    python cad/tools/lidar_airy_study.py
"""
import itertools, math
import numpy as np, mujoco

M = mujoco.MjModel.from_xml_path("urdf/full/scene.xml")
D = mujoco.MjData(M)
SKIP = [i for i in range(M.ngeom) if any(s in (M.geom(i).name or "") for s in ("lidar", "guard"))]
M.geom_group[SKIP] = 5
GROUPS = np.array([1, 1, 1, 1, 1, 0], np.uint8)          # mj_ray sees groups 0-4 only
TORSO = M.body("torso_upper").id
FOV_COS, RNG = 0.0, (0.10, 30.0)

c45 = math.sqrt(0.5)
# candidate mounts in the torso_upper frame: (position, axis). head top z 0.51 (v2 full riser),
# chest/back faces at x +-0.09, z 0.17; pelvis underside (base_link) for a down-looker
P = {"head_up":     ((0.0, 0.0, 0.54), (0, 0, 1)),
     "head_front":  ((0.06, 0.0, 0.51), (1, 0, 0)),
     "head_back":   ((-0.06, 0.0, 0.51), (-1, 0, 0)),
     "head_f45":    ((0.063, 0.0, 0.509), (c45, 0, c45)),
     "head_b45":    ((-0.063, 0.0, 0.509), (-c45, 0, c45)),
     "chest_front": ((0.095, 0.0, 0.17), (1, 0, 0)),
     "chest_fdown": ((0.095, 0.0, 0.17), (math.cos(math.radians(30)), 0, -math.sin(math.radians(30)))),
     "back":        ((-0.095, 0.0, 0.17), (-1, 0, 0)),
     "chest_up_fr": ((0.095, 0.0, 0.30), (c45, 0, c45)),
     "chest_fup20": ((0.095, 0.0, 0.17), (math.cos(math.radians(20)), 0, math.sin(math.radians(20)))),
     "head_fdown20": ((0.06, 0.0, 0.51), (math.cos(math.radians(20)), 0, -math.sin(math.radians(20)))),
     "head_fdown40": ((0.06, 0.0, 0.51), (math.cos(math.radians(40)), 0, -math.sin(math.radians(40)))),
     "back_down30": ((-0.095, 0.0, 0.17), (-math.cos(math.radians(30)), 0, -math.sin(math.radians(30))))}
LAYOUTS = {
    "4 now (Mid-360 places)": ["head_f45", "head_b45", "chest_front", "back"],
    "1 head up": ["head_up"],
    "2 head front + head back": ["head_front", "head_back"],
    "2 head up + chest front": ["head_up", "chest_front"],
    "2 head up + chest front-down": ["head_up", "chest_fdown"],
    "3 head up + chest front + back": ["head_up", "chest_front", "back"],
    "3 head up + chest front-down + back": ["head_up", "chest_fdown", "back"],
    "2 head f45 + head b45": ["head_f45", "head_b45"],
    "3 head f45 + head b45 + chest front-down": ["head_f45", "head_b45", "chest_fdown"],
    "2 head b45 + chest front-down": ["head_b45", "chest_fdown"],
    "2 head b45 + chest front": ["head_b45", "chest_front"],
    "2 head front-down20 + head back": ["head_fdown20", "head_back"],
    "2 head front-down40 + head b45": ["head_fdown40", "head_b45"],
    "3 head b45 + chest up-front + chest front-down": ["head_b45", "chest_up_fr", "chest_fdown"],
    "3 head b45 + chest front + back-down": ["head_b45", "chest_front", "back_down30"],
    "4 head f45 + head b45 + chest front-down + back": ["head_f45", "head_b45", "chest_fdown", "back"],
    "upper bound: all mounts": list(P),
}


def world_mounts(names):
    R = D.xmat[TORSO].reshape(3, 3)
    p0 = D.xpos[TORSO]
    return [(p0 + R @ np.array(P[n][0]), R @ (np.array(P[n][1]) / np.linalg.norm(P[n][1]))) for n in names]


def seen(points, mounts):
    out = np.zeros(len(points), bool)
    gid = np.zeros(1, np.int32)
    for i, q in enumerate(points):
        for o, ax in mounts:
            v = q - o
            dist = np.linalg.norm(v)
            if not (RNG[0] <= dist <= RNG[1]) or v @ ax / dist < FOV_COS:
                continue
            h = mujoco.mj_ray(M, D, o, v / dist, GROUPS, 0, TORSO, gid)   # its own torso: the hemisphere already faces out
            if h < 0 or h > dist - 1e-3:
                out[i] = True
                break
    return out


def rung_pts(name, ylim=None, n=32):
    g = M.geom(name).id
    c, half = D.geom_xpos[g], M.geom_size[g][1]
    ax = D.geom_xmat[g].reshape(3, 3)[:, 2]
    ts = np.linspace(-half, half, n)
    pts = np.array([c + t * ax for t in ts])
    if ylim:
        pts = pts[(pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1])]
    return pts


def hang_targets():
    mujoco.mj_resetDataKeyframe(M, D, M.key("hang_on_rung").id)
    mujoco.mj_forward(M, D)
    up = [n for n in ("rung_front_aligned_1", "rung_front_aligned_2", "rung_front_stagA_0", "rung_front_stagA_1", "rung_front_stagA_2")]
    wall_below = np.array([(0.42, y, z) for y in np.linspace(-0.5, 0.5, 11) for z in np.linspace(-0.7, 0.4, 12)])
    behind = np.vstack([rung_pts(n) for n in ("rung_back_line_0", "rung_back_aligned_1", "rung_back_aligned_2", "rung_back_low_0", "rung_back_low_1")])
    return {"start rung (hands)": rung_pts("rung_front_line_0", (-0.5, 0.5)),
            "rungs above": np.vstack([rung_pts(n) for n in up]),
            "rung below + wall at the feet": np.vstack([rung_pts("rung_front_low_0", (-0.5, 0.5)), wall_below]),
            "wall behind": behind}


def stand_targets():
    mujoco.mj_resetData(M, D)
    mujoco.mj_forward(M, D)
    zmin = min(D.geom_xpos[g][2] - M.geom_rbound[g] * 0 for g in range(M.ngeom) if M.geom_bodyid[g] in
               (M.body("foot_L").id, M.body("foot_R").id))
    D.qpos[2] -= zmin + 0.02 - 0.0                 # roughly feet to the floor (z ~ 0)
    D.qpos[0:2] = (5.0, 5.0)                       # away from the walls (ignored anyway)
    mujoco.mj_forward(M, D)
    c = D.xpos[M.body("base_link").id].copy()
    ring = np.array([(c[0] + r * math.cos(a), c[1] + r * math.sin(a), 0.0) for r in np.linspace(0.3, 3.0, 10) for a in np.linspace(0, 2 * math.pi, 36, endpoint=False)])
    toes = np.array([(c[0] + x, c[1] + y, 0.0) for x in np.linspace(0.15, 0.6, 8) for y in np.linspace(-0.3, 0.3, 7)])
    space = np.array([(c[0] + 1.5 * math.cos(a), c[1] + 1.5 * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 24, endpoint=False) for z in np.linspace(0.1, 1.8, 8)])
    return {"ground ring 0.3-3 m": ring, "ground ahead of the toes": toes, "space at 1.5 m": space}


if __name__ == "__main__":
    rows = {}
    for setup in (hang_targets, stand_targets):
        T = setup()
        for name, lay in LAYOUTS.items():
            mounts = world_mounts(lay)
            for t, pts in T.items():
                rows.setdefault(name, {})[t] = 100.0 * seen(pts, mounts).mean()
    cols = list(next(iter(rows.values())).keys())
    print("%-40s" % "layout (Airy)" + "".join("%11s" % c[:10] for c in cols))
    for name, r in rows.items():
        print("%-40s" % name + "".join("%10.0f%%" % r[c] for c in cols))
    print("columns:", cols)


def search(k_max=4, top=4):
    """every combination of 1..k_max mounts from a tilt sweep; visibility per mount computed
    once per point, a layout's = their OR. Weighted score: climbing columns count more."""
    cand = {"head_up": ((0.0, 0.0, 0.54), 0, 90)}
    for t in (45, 20, 0, -20, -40):
        cand["head_front%+d" % t] = ((0.06, 0.0, 0.51), 0, t)
    for t in (70, 45, 20, 0):
        cand["head_back%+d" % t] = ((-0.06, 0.0, 0.51), 180, t)
    for t in (45, 20, 0, -15, -30, -45):
        cand["chest_front%+d" % t] = ((0.095, 0.0, 0.17), 0, t)
    for t in (0, -30):
        cand["back%+d" % t] = ((-0.095, 0.0, 0.17), 180, t)
    for n, (p, az, el) in cand.items():
        a, e = math.radians(az), math.radians(el)
        P[n] = (p, (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    W = {"start rung (hands)": 2.0, "rungs above": 1.5, "rung below + wall at the feet": 1.5, "wall behind": 0.5,
         "ground ring 0.3-3 m": 1.0, "ground ahead of the toes": 1.0, "space at 1.5 m": 0.5}
    vis = {}
    for setup in (hang_targets, stand_targets):
        T = setup()
        for n in cand:
            mts = world_mounts([n])
            for t, pts in T.items():
                vis[(n, t)] = seen(pts, mts)
    cols = list(W)
    for k in range(1, k_max + 1):
        res = []
        for combo in itertools.combinations(cand, k):
            pct = {t: 100.0 * np.logical_or.reduce([vis[(n, t)] for n in combo]).mean() for t in cols}
            res.append((sum(W[t] * pct[t] for t in cols) / sum(W.values()), combo, pct))
        res.sort(key=lambda r: -r[0])
        print("\n%d lidar(s):" % k)
        for sc, combo, pct in res[:top]:
            print("  %5.1f  %-58s %s" % (sc, " + ".join(combo), " ".join("%3.0f" % pct[t] for t in cols)))
    print("\ncolumns:", cols)


if __name__ == "__main__" and True:
    search()


def check_real(head_heights=(0.40, 0.45, 0.51)):
    """the chosen 3-Airy layout at mounts the real chest can hold: the two front ones stacked
    (z 0.129 / 0.199 in the sim torso frame = 300 / 370 mm over the hip centres), sunk to x 0.08;
    the head-back one at a few mast heights"""
    W = ["start rung (hands)", "rungs above", "rung below + wall at the feet", "wall behind",
         "ground ring 0.3-3 m", "ground ahead of the toes", "space at 1.5 m"]
    def ax(az, el):
        a, e = math.radians(az), math.radians(el)
        return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    P["chest_hi"] = ((0.08, 0.0, 0.199), ax(0, 20))
    P["chest_lo"] = ((0.08, 0.0, 0.129), ax(0, -30))
    for h in head_heights:
        P["head_back_%.2f" % h] = ((-0.06, 0.0, h), ax(180, 20))
    res = {}
    for setup in (hang_targets, stand_targets):
        T = setup()
        for h in head_heights:
            mts = world_mounts(["head_back_%.2f" % h, "chest_hi", "chest_lo"])
            for t, pts in T.items():
                res.setdefault(h, {})[t] = 100.0 * seen(pts, mts).mean()
    for h, r in res.items():
        print("head-back at sim z %.2f (real %.0f mm over the hips): %s" % (h, 1000 * h + 171, " ".join("%3.0f" % r[t] for t in W)))
