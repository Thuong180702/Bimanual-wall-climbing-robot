"""Write urdf/scene.xml: the v2 robot in the same corridor v1 used.

Walls, rungs, cameras and the 86 sensors are carried over from urdf/scene.xml
unchanged - the environment did not change, the robot did. What IS new:

  forcerange   the actuator's datasheet PEAK torque, per joint, not a
               geometry-derived guess
  ctrlrange    the joint limits, with shoulder flexion clipped to -170 deg
  armature     N^2 * J_rotor, the rotor inertia the gearbox reflects into the
               joint. v1 used armature="0.01" for every joint; a 64:1 AK80-64
               reflects ~0.2 kg m^2, twenty times that, and a 36:1 AK45-36
               ~0.008. Getting this wrong makes a geared joint feel massless in
               simulation and sluggish on the bench - it is the single largest
               sim2real term added here. J_rotor itself is ESTIMATED (see
               rotor_inertia below) and is the first thing to identify on
               hardware.
  frictionloss scaled from the one published backdrive figure (AK45-36, 0.8 N.m
               at 36:1). Also to be identified.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cad", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "cad", "tools"))
from lib.build import BUILDERS, CHILD_OF, PJOINT
from lib.collide import link_collision, named_collision
from lib.motors import MOTORS
from lib.spec import FIXED, JOINTS, LIDARS, MID360, MOTOR_OF, MOUNT_ROT, TORSO_UP, HOOK

A = json.load(open(os.path.join(HERE, "_actuators.json")))
PROPS = A["links"]

# Position-servo gains, carried over from v1 by joint role.
KP = {"waist_yaw": 800}
for s in ("L", "R"):
    KP.update({"shoulder_%s_abduction" % s: 800, "shoulder_%s_flexion" % s: 800,
               "shoulder_%s_rotation" % s: 500, "elbow_%s" % s: 500,
               "forearm_%s_roll" % s: 200, "wrist_%s_flexion" % s: 150,
               "wrist_%s_deviation" % s: 150, "finger_%s" % s: 200})


def rotor_inertia(key):
    """J of the rotor about the motor axis, kg m^2 - an ESTIMATE.

    No AK datasheet publishes it. Modelled as a disc holding 20 % of the
    actuator's mass at half its outer radius; calibrated so AK80-9 lands at
    2.4e-3 kg m^2 reflected, which is the figure people measure on that unit.
    Identify it on hardware before trusting a transient.
    """
    d = MOTORS[key]
    m = 0.20 * d["mass_kg"]
    r = 0.5 * (d["dia_mm"] / 2 * 1e-3)
    return 0.5 * m * r * r


def armature(key):
    return MOTORS[key]["ratio"] ** 2 * rotor_inertia(key)


def frictionloss(key):
    return 0.8 * MOTORS[key]["ratio"] / 36.0        # scaled from AK45-36's 0.8 N.m


def geom_xml(kind, size, pos, zdir, xhint, cls="col"):
    import numpy as np
    if kind == "capsule":
        (a, b), r = pos, size[0]
        return ('<geom class="%s" type="capsule" size="%.5f" '
                'fromto="%.5f %.5f %.5f %.5f %.5f %.5f"/>' % (cls, r, *a, *b))
    z = np.array(zdir, float)
    z /= np.linalg.norm(z)
    if kind == "cylinder":
        r, L = size
        a = np.array(pos) - z * L / 2
        b = np.array(pos) + z * L / 2
        return ('<geom class="%s" type="cylinder" size="%.5f" '
                'fromto="%.5f %.5f %.5f %.5f %.5f %.5f"/>'
                % (cls, r, a[0], a[1], a[2], b[0], b[1], b[2]))
    if kind == "mesh":
        return '<geom class="col" type="mesh" mesh="%s" pos="0 0 0"/>' % size[0]
    x = np.array(xhint if xhint else ((1, 0, 0) if abs(z[2]) < 0.9 else (0, 1, 0)), float)
    if abs(x @ z) > 0.9:
        x = np.array((0.0, 0.0, 1.0)) if abs(z[2]) < 0.9 else np.array((1.0, 0.0, 0.0))
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = _quat(np.column_stack([x, y, z]))
    g = ('<geom class="%s" type="box" size="%.5f %.5f %.5f" pos="%.5f %.5f %.5f"'
         % (cls, size[0] / 2, size[1] / 2, size[2] / 2, pos[0], pos[1], pos[2]))
    # an identity quaternion is noise, and rl/lessons.py patches these lines by
    # exact text - the hook's two boxes are axis-aligned and must stay quiet
    if max(abs(q[1]), abs(q[2]), abs(q[3])) > 1e-9:
        g += ' quat="%.6f %.6f %.6f %.6f"' % q
    return g + '/>' 


def _quat(R):
    import numpy as np
    t = R.trace()
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1], q[j + 1], q[k + 1] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return tuple(q)


def _frame_quat(z, x=None):
    """the quaternion of a frame whose z is `z` and whose x is as near `x` as it can be"""
    import numpy as np
    z = np.array(z, float) / np.linalg.norm(z)
    x = np.array(x if x is not None else ((1, 0, 0) if abs(z[0]) < 0.9 else (0, 1, 0)), float)
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return _quat(np.column_stack([x, np.cross(z, x), z]))


def occluders(link):
    """What a lidar cannot see through, as sites (`rl.lidar` casts its rays at them):
    one per collision geom of the link, the same shape where a site can take it -- a
    motor's cylinder, a hook's box, a tube's capsule -- and a part's hull as its
    tightest box. Not the head lidars' roof: it lies under their field of view, and
    its box would take them in. -> [(name, type, size, pos, quat)] in the link frame"""
    import numpy as np, trimesh
    out = []
    for i, (kind, size, pos, zdir, xhint) in enumerate(link_collision(link)):
        nm = "occ_%s_%d" % (link, i)
        if kind == "mesh" and "lidar" in size[0]:
            continue
        if kind == "capsule":
            (a, b), r = np.array(pos, float), size[0]
            out.append((nm, "capsule", (r, np.linalg.norm(b - a) / 2), (a + b) / 2, _frame_quat(b - a)))
        elif kind == "cylinder":
            out.append((nm, "cylinder", (size[0], size[1] / 2), pos, _frame_quat(zdir)))
        elif kind == "box":
            out.append((nm, "box", tuple(v / 2 for v in size), pos, _frame_quat(zdir, xhint)))
        else:
            hull = trimesh.load(os.path.join(HERE, "meshes", "%s.stl" % size[0]))
            ob = hull.bounding_box_oriented
            T = ob.primitive.transform
            out.append((nm, "box", tuple(ob.primitive.extents * 0.0005), T[:3, 3] * 0.001,
                        _quat(T[:3, :3])))
    return out


def _turn(a):
    """the quat attribute for a link turned by `a` about its own z (spec.MOUNT_ROT)"""
    if not a:
        return ""
    return ' quat="%g 0 0 %g"' % (round(math.cos(a / 2), 12), round(math.sin(a / 2), 12))


def body_tree():
    import numpy as np
    from massprops import link_props
    kids = {}
    for j in JOINTS:
        kids.setdefault(j[1], []).append(j)
    for f in FIXED:
        kids.setdefault(f[1], []).append((None, f[1], f[2], None, f[3], None, None))
    acts = []

    def emit(name, ind, joint):
        p = "  " * ind
        M, C, I, _ = link_props(name)
        if joint is None:
            L = ['%s<body name="%s" pos="0 0 0.30">' % (p, name),
                 '%s  <freejoint name="floating_base"/>' % p]
        else:
            jn, _, _, ax, off, lim, _bd = joint
            L = ['%s<body name="%s" pos="%.6f %.6f %.6f"%s>' % (
                p, name, *off, _turn(MOUNT_ROT.get(name, 0.0)))]
            if jn is not None:
                key = MOTOR_OF[jn]
                L.append('%s  <joint name="%s" type="hinge" axis="%d %d %d" '
                         'range="%.5f %.5f" armature="%.6g" frictionloss="%.4g"/>'
                         % (p, jn, ax[0], ax[1], ax[2], lim[0], lim[1],
                            armature(key), frictionloss(key)))
                acts.append((jn, key, lim[0], lim[1]))
        L.append('%s  <inertial pos="%.6f %.6f %.6f" mass="%.6f" '
                 'fullinertia="%.9g %.9g %.9g %.9g %.9g %.9g"/>'
                 % (p, C[0], C[1], C[2], M, I[0][0], I[1][1], I[2][2],
                    I[0][1], I[0][2], I[1][2]))
        L.append('%s  <geom class="vis" mesh="%s"/>' % (p, name))
        for gn, g in named_collision(name):
            x = geom_xml(*g)
            L.append("%s  %s" % (p, x.replace('<geom ', '<geom name="%s" ' % gn, 1) if gn else x))
        if name == "torso_upper":
            L.append('%s  <site name="s_imu" type="box" pos="0 0 %.5f" '
                     'size="0.02 0.02 0.01"/>' % (p, TORSO_UP[2] / 2))
        # each lidar's optical origin, z its axis and x toward its plug (rl.lidar)
        for ln, (host, o, z, x) in LIDARS.items():
            if host == name:
                c = [o[i] + z[i] * MID360["origin_z"] for i in range(3)]
                L.append('%s  <site name="%s" type="sphere" size="0.006" group="4" '
                         'pos="%.5f %.5f %.5f" quat="%.6f %.6f %.6f %.6f"/>'
                         % (p, ln, *c, *_frame_quat(z, x)))
        for nm, typ, sz, pos, q in occluders(name):
            L.append('%s  <site name="%s" type="%s" group="5" size="%s" pos="%.5f %.5f %.5f" '
                     'quat="%.6f %.6f %.6f %.6f"/>'
                     % (p, nm, typ, " ".join("%.5f" % v for v in sz), *pos, *q))
        if name.startswith("hook_"):
            sd, ht, hb, hl, hw = (name[-1], HOOK["thk"], HOOK["back"],
                                  HOOK["lip"], HOOK["width"])
            L += ['%s  <site name="s_ft_%s" type="sphere" pos="0 0 0" size="0.008"/>'
                  % (p, sd),
                  '%s  <site name="s_grip_%s" type="box" pos="%.5f 0 %.5f" '
                  'size="%.5f %.5f %.5f"/>'
                  % (p, sd, ht / 2 + hl / 2, -(hb - ht) / 2 - 0.010,
                     hl / 2 + 0.004, hw / 2, (hb - ht) / 2 + 0.014)]
        for j in kids.get(name, []):
            L += emit(j[2], ind + 1, j)
        L.append("%s</body>" % p)
        return L

    return emit("base_link", 3, None), acts


def main():
    tree, acts = body_tree()
    tree = "\n".join(tree)
    LINKS = list(BUILDERS)

    # ------------------------------------------------------------
    R, SO = 0.015, 0.06          # rung radius, stand-off from the wall face
    XF, XB, YS = 0.5, -0.25, -1.0    # inner faces: front wall, back wall, side wall
    AF, AB, AS = XF - SO, XB + SO, YS + SO
    SX0, SX1 = XB - 0.05, XF + 0.05  # side wall spans the corridor

    rungs, posts = [], []
    def rowY(x, segs, tag, facex):
        for i, (y0, y1, z) in enumerate(segs):
            rungs.append(('%s_%d' % (tag, i), R, (x, y0, z), (x, y1, z)))
            for y in (y0, y1):
                posts.append(((x, y, z), (facex, y, z)))
    def rowX(y, segs, tag, facey):
        for i, (x0, x1, z) in enumerate(segs):
            rungs.append(('%s_%d' % (tag, i), R, (x0, y, z), (x1, y, z)))
            for x in (x0, x1):
                posts.append(((x, y, z), (x, facey, z)))

    # Vertical spacing rule: measured max reach when one arm is locked off on a hold and
    # the other stretches up is 0.671 m (|dy| <= 0.30 m). Every consecutive row here is
    # <= 0.50 m apart, i.e. 75 % of that, so the robot always has a hold it can actually take.

    # --- FRONT WALL -------------------------------------------------------------
    rowY(AF, [(-0.90, 1.50, 0.55)], 'front_low',  XF)   # continuous straight rung
    rowY(AF, [(-0.90, 1.50, 1.00)], 'front_line', XF)   # continuous - keyframe hangs here
    # separated holds but all on ONE straight line (same height)
    rowY(AF, [(-0.90,-0.55, 1.45), (-0.30, 0.05, 1.45),
              ( 0.30, 0.65, 1.45), ( 0.90, 1.25, 1.45)], 'front_aligned', XF)
    # separated holds, each at its own height, rows offset half a period sideways
    rowY(AF, [(-0.60,-0.25, 1.82), ( 0.00, 0.35, 1.95),
              ( 0.60, 0.95, 1.84), ( 1.20, 1.50, 1.92)], 'front_stagA', XF)
    rowY(AF, [(-0.90,-0.55, 2.26), (-0.30, 0.05, 2.38),
              ( 0.30, 0.65, 2.24), ( 0.90, 1.25, 2.34)], 'front_stagB', XF)
    rowY(AF, [(-0.60,-0.25, 2.70), ( 0.00, 0.35, 2.80),
              ( 0.60, 0.95, 2.68)], 'front_stagC', XF)

    # --- BACK WALL (parallel, within arm reach) ---------------------------------
    rowY(AB, [(-0.30, 0.05, 0.55), ( 0.60, 0.95, 0.55)], 'back_low', XB)
    rowY(AB, [(-0.90, 1.50, 1.00)], 'back_line', XB)
    rowY(AB, [(-0.90,-0.55, 1.45), (-0.30, 0.05, 1.45),
              ( 0.30, 0.65, 1.45), ( 0.90, 1.25, 1.45)], 'back_aligned', XB)
    rowY(AB, [(-0.60,-0.25, 1.86), ( 0.00, 0.35, 1.94),
              ( 0.60, 0.95, 1.82), ( 1.20, 1.50, 1.90)], 'back_stag', XB)
    rowY(AB, [(-0.90,-0.55, 2.28), (-0.30, 0.05, 2.36),
              ( 0.30, 0.65, 2.26), ( 0.90, 1.25, 2.32)], 'back_stagB', XB)

    # --- SIDE WALL (perpendicular, closes the corner) ---------------------------
    rowX(AS, [(-0.18, 0.43, 1.00)], 'side_line', YS)
    rowX(AS, [(-0.18, 0.05, 1.45), ( 0.18, 0.43, 1.45)], 'side_aligned', YS)
    rowX(AS, [(-0.15, 0.10, 1.88), ( 0.15, 0.43, 1.96)], 'side_stagA', YS)
    rowX(AS, [(-0.12, 0.13, 2.34), ( 0.18, 0.43, 2.28)], 'side_stagB', YS)

    # ------------------------------------------------------------
    L = ['<mujoco model="bimanual_wall_climber_scene">',
    '  <!-- Frame: +X toward the front wall, +Y left, +Z up. No floor (by request). -->',
    '  <!-- Rungs stand %g mm off each wall face so the hook finger can pass behind them. -->' % (SO*1000),
    '  <compiler angle="radian" meshdir="meshes/" autolimits="true"/>',
    '  <!-- Robot: 17 CubeMars AK-series actuators, %.4f kg total. -->' % A["total_mass"],
    '  <option timestep="0.002" integrator="implicitfast"/>',
    '  <statistic center="0 0.3 1.0" extent="3.5"/>',
    '  <visual><global azimuth="150" elevation="-15" offwidth="1600" offheight="1200"/><quality shadowsize="4096"/></visual>',
    '',
    '  <asset>',
    '    <material name="alu"  rgba="0.75 0.76 0.78 1"/>',
    '    <material name="wall" rgba="0.34 0.36 0.40 1"/>',
    '    <material name="rung" rgba="0.85 0.55 0.15 1"/>']
    for n in LINKS:
        L.append('    <mesh name="%s" file="%s.stl" scale="0.001 0.001 0.001"/>' % (n, n))
        # the bracket's parts, which the collision geoms use; MuJoCo hulls each
        for kind, size, *_ in link_collision(n):
            if kind == "mesh":
                L.append('    <mesh name="%s" file="%s.stl"'
                         ' scale="0.001 0.001 0.001"/>' % (size[0], size[0]))
    L += ['  </asset>', '',
    '  <default>',
    '    <default class="vis">',
    '      <geom type="mesh" material="alu" contype="0" conaffinity="0" group="1"/>',
    '    </default>',
    '    <default class="col">',
    '      <geom group="3" rgba="0.5 0.6 0.9 0.35" friction="1.0 0.02 0.001"/>',
    '    </default>',
    '    <default class="hold">',
    '      <geom type="cylinder" material="rung" friction="1.2 0.02 0.001"/>',
    '    </default>',
    '    <!-- armature and frictionloss are PER JOINT, on the joint itself: a 64:1 -->',
    '    <!-- unit and a 36:1 unit do not reflect the same rotor inertia. -->',
    '    <joint damping="0.15"/>',
    '  </default>', '',
    '  <worldbody>',
    '<camera name="front_face2" pos="-2.40 0.30 1.30" xyaxes="0 -1 0 0 0 1" fovy="62"/>',
    '<camera name="back_face2"  pos="2.60 0.30 1.30"  xyaxes="0 1 0 0 0 1"  fovy="62"/>',
    '<camera name="side_face2"  pos="0.12 3.20 1.30"  xyaxes="-1 0 0 0 0 1" fovy="58"/>',
    '<camera name="iso2" pos="4.2000 4.6000 3.6000" fovy="32" xyaxes="-0.72374 0.69008 0.00000 -0.24912 -0.26128 0.93256"/>',
    '<camera name="iso3" pos="-3.2000 4.4000 2.8000" fovy="34" xyaxes="-0.77901 -0.62701 0.00000 0.17186 -0.21352 0.96170"/>',
    '<camera name="front_face" pos="-0.60 0.30 1.20" xyaxes="0 -1 0 0 0 1" fovy="98"/>',
    '<camera name="back_face"  pos="0.42 0.30 1.20"  xyaxes="0 1 0 0 0 1"  fovy="98"/>',
    '<camera name="side_face"  pos="-0.10 0.90 1.20" xyaxes="-1 0 0 0 0 1" fovy="95"/>',
    '<camera name="iso"        pos="3.60 4.20 3.40" mode="targetbody" target="base_link" fovy="40"/>',
    '<camera name="top"        pos="-0.10 0.30 5.60" xyaxes="1 0 0 0 1 0" fovy="42"/>',
    '    <camera name="overview" pos="-0.10 5.20 1.60" fovy="34" mode="targetbody" target="base_link"/>',
    '    <camera name="closeup"  pos="0.05 0.72 1.06" fovy="30" mode="targetbody" target="hook_L"/>',
    '    <camera name="corner"   pos="2.60 3.40 1.90" fovy="38" mode="targetbody" target="base_link"/>',
    '    <light pos="0 1 3.5" dir="0 -0.3 -1" directional="true" diffuse="0.7 0.7 0.7"/>',
    '    <light pos="-1 -1 2.5" dir="0.5 0.5 -1" diffuse="0.3 0.3 0.35"/>',
    '',
    '    <!-- ===== walls ===== -->',
    '    <geom name="wall_front" type="box" material="wall" pos="%.3f 0.30 1.00" size="0.05 1.35 2.00"/>' % (XF+0.05),
    '    <geom name="wall_back"  type="box" material="wall" pos="%.3f 0.30 1.00" size="0.05 1.35 2.00"/>' % (XB-0.05),
    '    <geom name="wall_side"  type="box" material="wall" pos="%.3f %.3f 1.00" size="%.3f 0.05 2.00"/>'
         % ((SX0+SX1)/2, YS-0.05, (SX1-SX0)/2),
    '',
    '    <!-- ===== holds: horizontal rungs the hooks latch onto ===== -->']
    for nm, r, a, b in rungs:
        L.append('    <geom name="rung_%s" class="hold" size="%.4f" fromto="%.4f %.4f %.4f %.4f %.4f %.4f"/>'
                 % (nm, r, a[0], a[1], a[2], b[0], b[1], b[2]))
    L.append('    <!-- cosmetic stubs tying each rung to its wall (no collision) -->')
    for i, (a, b) in enumerate(posts):
        L.append('    <geom name="stub_%d" type="cylinder" size="0.010" material="wall" group="1" '
                 'contype="0" conaffinity="0" fromto="%.4f %.4f %.4f %.4f %.4f %.4f"/>'
                 % (i, a[0], a[1], a[2], b[0], b[1], b[2]))

    # ------------------------------------------------------------
    L += ['', '    <!-- ===== robot ===== -->', tree, '  </worldbody>', '',
          '  <actuator>']
    for jn, key, lo, hi in acts:
        eff = MOTORS[key]["peak_Nm"]
        L.append('    <position name="act_%s" joint="%s" kp="%g" dampratio="1" '
                 'ctrlrange="%.5f %.5f" forcerange="-%g %g"/>   <!-- %s -->'
                 % (jn, jn, KP[jn], lo, hi, eff, eff, key))
    L += ['  </actuator>', '', '  <sensor>']
    L.append('    <!-- joint encoders: position + velocity, one pair per motor -->')
    for jn, key, lo, hi in acts:
        L.append('    <jointpos name="enc_%s" joint="%s"/>' % (jn, jn))
    for jn, key, lo, hi in acts:
        L.append('    <jointvel name="vel_%s" joint="%s"/>' % (jn, jn))
    L.append('    <!-- motor torque feedback (current sensing on the real drive) -->')
    for jn, key, lo, hi in acts:
        L.append('    <actuatorfrc name="tau_%s" actuator="act_%s"/>' % (jn, jn))
    L.append('    <!-- hard-stop detection: non-zero when a joint is jammed on its limit -->')
    for jn, key, lo, hi in acts:
        L.append('    <jointlimitfrc name="lim_%s" joint="%s"/>' % (jn, jn))
    L.append('    <!-- IMU on the upper torso -->')
    L.append('    <!-- physically realisable IMU channels (9-DOF: accel + gyro + mag) -->')
    L += ['    <accelerometer name="imu_acc"  site="s_imu"/>',
          '    <gyro          name="imu_gyro" site="s_imu"/>',
          '    <magnetometer  name="imu_mag"  site="s_imu"/>']
    L.append('    <!-- derived / ground-truth: free in sim, needs an estimator on hardware -->')
    L += ['    <framequat   name="imu_quat"    objtype="site" objname="s_imu"/>',
          '    <velocimeter name="imu_vel"     site="s_imu"/>',
          '    <frameangvel name="imu_angvel"  objtype="site" objname="s_imu"/>',
          '    <frameangacc name="imu_angacc"  objtype="site" objname="s_imu"/>',
          '    <framelinacc name="imu_linacc"  objtype="site" objname="s_imu"/>',
          '    <framepos    name="torso_pos"   objtype="site" objname="s_imu"/>']
    L.append('    <!-- whole-body state: COM is what the climbing controller must steer -->')
    L += ['    <subtreecom    name="com_pos"    body="base_link"/>',
          '    <subtreelinvel name="com_vel"    body="base_link"/>',
          '    <subtreeangmom name="com_angmom" body="base_link"/>']
    L.append('    <!-- hand force sensing: 6-axis wrench + throat contact pad -->')
    for sd in ('L', 'R'):
        L += ['    <force  name="ft_force_%s"  site="s_ft_%s"/>' % (sd, sd),
              '    <torque name="ft_torque_%s" site="s_ft_%s"/>' % (sd, sd),
              '    <touch  name="grip_%s"      site="s_grip_%s"/>' % (sd, sd)]
    L += ['  </sensor>', '</mujoco>', '']
    p = os.path.join(HERE, 'scene.xml')
    open(p, 'w').write('\n'.join(L))
    print('wrote %s  (%d lines, %d rungs, %d actuators)' % (p, len(L), len(rungs), len(acts)))


if __name__ == "__main__":
    main()
