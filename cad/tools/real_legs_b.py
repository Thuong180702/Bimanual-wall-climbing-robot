"""B packaging of the legs as an MJCF (joint frames, true-size bodies) + a sweep that
reports the closest approach between bodies over the joint ranges."""
import itertools, math, sys
import numpy as np, mujoco

Z_HIP, HIP_Y = 0.74, 0.100
P = dict(ZY=0.110,      # yaw motor output face over the hip centre
         XF=-0.065,     # roll motor front face (motor body behind it, output at the rear)
         ZR=0.140,      # roll motor axis over the hip centre
         S=0.015,       # roll motor shift outboard
         ZK=-0.180,     # knee motor below the hip centre
         ZP=0.207)      # pelvis underside (round sockets reach down to the yaw motors)
R13, H13 = 0.085, 0.04225   # A13720
R10, H10 = 0.062, 0.0425    # A10020
R81, L81 = 0.0525, 0.053    # A8112
RAK, HAK = 0.05375, 0.02725 # AKE90-8
GAP = 0.0025
RH = 0.035

def cy(name, a, b, r):
    return f'<geom name="{name}" type="cylinder" fromto="{" ".join("%.5f"%v for v in (*a,*b))}" size="{r}"/>'
def bx(name, lo, hi):
    c = [(l+h)/2 for l, h in zip(lo, hi)]; s = [(h-l)/2 for l, h in zip(lo, hi)]
    return f'<geom name="{name}" type="box" pos="{" ".join("%.5f"%v for v in c)}" size="{" ".join("%.5f"%v for v in s)}"/>'

def leg(sg, p):
    s = "L" if sg > 0 else "R"
    Y = lambda y: sg * y                      # mirror lateral coordinates
    def cyy(n, a, b, r): return cy(n+"_"+s, (a[0], Y(a[1]), a[2]), (b[0], Y(b[1]), b[2]), r)
    def bxy(n, lo, hi):
        ys = sorted((Y(lo[1]), Y(hi[1]))); return bx(n+"_"+s, (lo[0], ys[0], lo[2]), (hi[0], ys[1], hi[2]))
    ZY, XF, ZR, S, ZK = p["ZY"], p["XF"], p["ZR"], p["S"], p["ZK"]
    xw = XF - 2*H10                            # rear wall front face
    yaw = [cyy("yaw_plate", (0, 0, ZY-0.008), (0, 0, ZY-GAP/2), 0.055),
           cyy("roll_motor", (XF, S, ZR), (xw, S, ZR), R10),
           cyy("rear_ring", (xw, S, ZR), (xw-0.010, S, ZR), 0.066),
           bxy("rear_strip", (xw-0.010, S/2-0.032, 0), (xw, S/2+0.032, ZR)),
           cyy("rear_hubwall", (xw, 0, 0), (xw-0.010, 0, 0), 0.040),
           cyy("pulley_m", (xw-0.010, S, ZR), (xw-0.034, S, ZR), 0.042),
           cyy("pulley_r", (xw-0.010, 0, 0), (xw-0.034, 0, 0), 0.050),
           bxy("belt_band", (xw-0.034, S/2-0.04, 0), (xw-0.010, S/2+0.04, ZR)),
           cyy("roll_hub", (xw, 0, 0), (-0.105, 0, 0), RH),
           bxy("side_web", (xw, S+R10+0.003, ZR-0.05), (-0.035, S+R10+0.011, ZY))]
    roll = [cyy("pitch_motor", (0, -H13, 0), (0, H13, 0), R13),
            cyy("yoke_disc", (0, -H13-GAP, 0), (0, -H13-0.011, 0), 0.080),
            bxy("yoke_boom", (-0.100, -H13-0.011, -0.03), (-0.070, 0.02, 0.03))]
    y0 = H13 + GAP
    pitch = [cyy("thigh_disc", (0, y0, 0), (0, y0+0.010, 0), 0.050),
             bxy("thigh_spine", (-0.04, y0, ZK), (0.04, y0+0.010, 0)),
             cyy("knee_motor", (0, -H13, ZK), (0, H13, ZK), R13),
             cyy("knee_plate", (0, y0, ZK), (0, y0+0.010, ZK), 0.083),
             cyy("crank", (0, -H13-GAP, ZK), (0, -H13-0.012, ZK), 0.032),
             bxy("thigh_low", (-0.015, -0.047, -0.318), (0.045, y0+0.010, ZK-R13-0.005)),
             cyy("knee_clevis_o", (0, 0.030, -0.34), (0, y0+0.010, -0.34), 0.042),
             cyy("knee_clevis_i", (0, -0.030, -0.34), (0, -0.047, -0.34), 0.042)]
    calf = [cyy("calf_top", (0, -0.0295, 0), (0, 0.0295, 0), 0.040),
            bxy("calf", (-0.035, -0.03, -0.30), (0.035, 0.03, -0.04)),
            cyy("ankle_a", (-0.03, -HAK, -0.090), (-0.03, HAK, -0.090), RAK),
            cyy("ankle_b", (-0.03, -HAK, -0.205), (-0.03, HAK, -0.205), RAK),
            bxy("foot", (-0.07, -0.045, -0.34), (0.15, 0.045, -0.31))]
    ax = lambda v: " ".join(str(x) for x in v)
    return f'''
<body name="yaw_{s}" pos="0 {Y(HIP_Y)} {Z_HIP}"><joint name="yaw_{s}" axis="0 0 {sg}" range="-45 45"/>{"".join(yaw)}
 <body name="roll_{s}"><joint name="roll_{s}" axis="{sg} 0 0" range="-20 45"/>{"".join(roll)}
  <body name="pitch_{s}"><joint name="pitch_{s}" axis="0 1 0" range="-125 30"/>{"".join(pitch)}
   <body name="knee_{s}" pos="0 0 -0.34"><joint name="knee_{s}" axis="0 1 0" range="0 135"/>{"".join(calf)}</body>
  </body>
 </body>
</body>'''

def model(p=P):
    ZY, ZP = p["ZY"], p["ZP"]
    pel = "".join(cy(f"yaw_motor_{s}", (0, sg*HIP_Y, Z_HIP+ZY), (0, sg*HIP_Y, Z_HIP+ZY+L81), R81) +
                  cy(f"socket_{s}", (0, sg*HIP_Y, Z_HIP+ZY+L81), (0, sg*HIP_Y, Z_HIP+ZP), R81+0.005)
                  for sg, s in ((1, "L"), (-1, "R")))
    zb = Z_HIP + ZY + L81          # pelvis bridge from the yaw motors' tops, rear cut open for the roll motors
    pel = "".join(cy(f"yaw_motor_{s}", (0, sg*HIP_Y, Z_HIP+ZY), (0, sg*HIP_Y, zb), R81) +
                  cy(f"socket_{s}", (0, sg*HIP_Y, zb), (0, sg*HIP_Y, zb+0.035), R81+0.005)
                  for sg, s in ((1, "L"), (-1, "R")))
    pel += bx("pelvis", (p.get("XB", -0.02), -HIP_Y-0.06, zb), (0.07, HIP_Y+0.06, zb+0.035))
    pel += cy("waist_motor", (0, 0, zb+0.035), (0, 0, zb+0.035+0.067), 0.0395)
    xml = f'<mujoco><compiler angle="degree"/><worldbody>{pel}{leg(1, p)}{leg(-1, p)}</worldbody></mujoco>'
    return mujoco.MjModel.from_xml_string(xml)

SKIP = {("calf", "knee_clevis_o"), ("calf", "knee_clevis_i"), ("roll_hub", "yoke_boom"), ("yaw_plate", "yaw_motor"), ("thigh_disc", "pitch_motor"),
        ("knee_clevis_o", "calf_top"), ("knee_clevis_i", "calf_top"), ("yoke_disc", "pitch_motor")}

def pairs(m):
    out = []
    for i, j in itertools.combinations(range(m.ngeom), 2):
        if m.geom_bodyid[i] == m.geom_bodyid[j]: continue
        a, b = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, k).rsplit("_", 1)[0] for k in (i, j))
        if (a, b) in SKIP or (b, a) in SKIP: continue
        out.append((i, j))
    return np.array(out)

def closest(m, d, pr, thr=0.002):
    c = d.geom_xpos
    near = np.linalg.norm(c[pr[:, 0]] - c[pr[:, 1]], axis=1) < m.geom_rbound[pr[:, 0]] + m.geom_rbound[pr[:, 1]] + thr
    hits = []
    for i, j in pr[near]:
        dist = mujoco.mj_geomDistance(m, d, int(i), int(j), thr, None)
        if dist < thr: hits.append((dist, i, j))
    return hits

def name(m, k): return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(k))

def sweep(p=P, step=10, yaw_pairs=True):
    m = model(p); d = mujoco.MjData(m); pr = pairs(m)
    q = lambda j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
    worst = {}
    rng = lambda a, b: np.radians(np.arange(a, b + 1e-9, step))
    # one leg through its whole range, the other standing
    for y, r, pt, k in itertools.product(rng(-45, 45), rng(-20, 45), rng(-125, 30), rng(0, 135)):
        d.qpos[:] = 0; d.qpos[[q("yaw_L"), q("roll_L"), q("pitch_L"), q("knee_L")]] = y, r, pt, k
        mujoco.mj_kinematics(m, d); mujoco.mj_collision  # noqa
        for dist, i, j in closest(m, d, pr):
            key = tuple(sorted((name(m, i), name(m, j))))
            if key not in worst or dist < worst[key][0]:
                worst[key] = (dist, np.degrees([y, r, pt, k]).round().astype(int).tolist())
    out = {"one leg": worst}
    if yaw_pairs:   # both legs yawed (toe-out +), the rest standing
        lim = {}
        for a, b in itertools.product(range(0, 46, 5), repeat=2):
            d.qpos[:] = 0; d.qpos[q("yaw_L")] = math.radians(a); d.qpos[q("yaw_R")] = math.radians(b)
            mujoco.mj_kinematics(m, d)
            for dist, i, j in closest(m, d, pr):
                key = tuple(sorted((name(m, i), name(m, j))))
                if "_L" in key[0] and "_R" in key[1] or "_R" in key[0] and "_L" in key[1]:
                    lim.setdefault(key, []).append((a, b, round(dist*1000, 1)))
        out["toe-out both"] = lim
    return out

if __name__ == "__main__":
    import json
    for k, v in sweep().items():
        print("==", k)
        for kk, vv in sorted(v.items(), key=lambda t: str(t[1])):
            print(" ", kk, vv if k != "toe-out both" else vv[:6])
