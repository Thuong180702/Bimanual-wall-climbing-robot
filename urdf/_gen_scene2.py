import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
tree = open(os.path.join(OUT, '_bodytree.txt')).read()
acts = json.load(open(os.path.join(OUT, '_acts.json')))
LINKS = list(json.load(open(os.path.join(OUT, '_linkdata.json')))['links'].keys())

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

L = ['<mujoco model="bimanual_wall_climber_scene">',
'  <!-- Frame: +X toward the front wall, +Y left, +Z up. No floor (by request). -->',
'  <!-- Rungs stand %g mm off each wall face so the hook finger can pass behind them. -->' % (SO*1000),
'  <compiler angle="radian" meshdir="meshes/" autolimits="true"/>',
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
'    <joint damping="0.15" armature="0.01"/>',
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
L += ['', '    <!-- ===== robot ===== -->', tree, '  </worldbody>', '',
      '  <actuator>']
for jn, kp, eff, lo, hi in acts:
    L.append('    <position name="act_%s" joint="%s" kp="%g" dampratio="1" '
             'ctrlrange="%.5f %.5f" forcerange="-%g %g"/>' % (jn, jn, kp, lo, hi, eff, eff))
L += ['  </actuator>', '', '  <sensor>']
L.append('    <!-- joint encoders: position + velocity, one pair per motor -->')
for jn, kp, eff, lo, hi in acts:
    L.append('    <jointpos name="enc_%s" joint="%s"/>' % (jn, jn))
for jn, kp, eff, lo, hi in acts:
    L.append('    <jointvel name="vel_%s" joint="%s"/>' % (jn, jn))
L.append('    <!-- motor torque feedback (current sensing on the real drive) -->')
for jn, kp, eff, lo, hi in acts:
    L.append('    <actuatorfrc name="tau_%s" actuator="act_%s"/>' % (jn, jn))
L.append('    <!-- hard-stop detection: non-zero when a joint is jammed on its limit -->')
for jn, kp, eff, lo, hi in acts:
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
p = os.path.join(OUT, 'scene.xml')
open(p, 'w').write('\n'.join(L))
print('wrote %s  (%d lines, %d rungs, %d actuators)' % (p, len(L), len(rungs), len(acts)))
