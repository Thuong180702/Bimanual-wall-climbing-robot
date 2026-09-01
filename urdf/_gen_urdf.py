import json, math, os

OUT = os.path.dirname(os.path.abspath(__file__))
G = 9.81

LD = json.load(open(os.path.join(OUT, '_linkdata.json')))
JD = json.load(open(os.path.join(OUT, '_jointdata.json')))
links, scale, P, joints = LD['links'], LD['scale'], JD['params'], JD['joints']

NAME = {('base_link','waist_link'): 'waist_yaw',
        ('waist_link','torso_upper'): 'torso_fixed'}
for s in ('L','R'):
    NAME.update({
      ('torso_upper','shoulder_%s'%s):        'shoulder_%s_abduction'%s,
      ('shoulder_%s'%s,'upper_arm_%s'%s):     'shoulder_%s_flexion'%s,
      ('upper_arm_%s'%s,'humerus_rot_%s'%s):  'shoulder_%s_rotation'%s,
      ('humerus_rot_%s'%s,'fore_arm_%s'%s):   'elbow_%s'%s,
      ('fore_arm_%s'%s,'forearm_rot_%s'%s):   'forearm_%s_roll'%s,
      ('forearm_rot_%s'%s,'wrist_%s'%s):      'wrist_%s_flexion'%s,
      ('wrist_%s'%s,'wrist_dev_%s'%s):        'wrist_%s_deviation'%s,
      ('wrist_dev_%s'%s,'hook_%s'%s):         'finger_%s'%s})

def cyl(r, l, oz, rpy='0 0 0', oxy=(0.0, 0.0)):
    return [('cylinder', dict(radius=r, length=l), (oxy[0], oxy[1], oz), rpy)]
def box(sx, sy, sz, o, rpy='0 0 0'):
    return [('box', dict(size=(sx, sy, sz)), o, rpy)]

tld, tlw, tlh = P['torso_low_d'], P['torso_low_w'], P['torso_low_h']
tud, tuw, tuh = P['torso_up_d'], P['torso_up_w'], P['torso_up_h']
wdia, wh = P['waist_dia'], P['waist_h']
sdia, soff = P['shoulder_dia'], P['shoulder_off']
ldia, rdia, rl = P['link_dia'], P['roll_dia'], P['roll_len']
ual, fal = P['upper_arm_len'], P['fore_arm_len']
wrdia, wrl = P['wrist_dia'], P['wrist_len']
hb, hl, ht, hw = P['hook_back'], P['hook_lip'], P['hook_thk'], P['hook_width']
RX90 = '%.6f 0 0' % (-math.pi/2)

COL = {'base_link':   box(tld, tlw, tlh, (0, 0, tlh/2)),
       'waist_link':  cyl(wdia/2, wh, wh/2),
       'torso_upper': box(tud, tuw, tuh, (0, 0, tuh/2))}
for s, sg in (('L', 1), ('R', -1)):
    COL['shoulder_%s'%s]    = cyl(sdia/2, soff, 0.0, RX90, (0.0, sg*soff/2))
    COL['upper_arm_%s'%s]   = cyl(ldia/2, ual-rl, -(ual-rl)/2)
    COL['humerus_rot_%s'%s] = cyl(rdia/2, rl, -rl/2)
    COL['fore_arm_%s'%s]    = cyl(ldia/2, fal-rl, -(fal-rl)/2)
    COL['forearm_rot_%s'%s] = cyl(rdia/2, rl, -rl/2)
    COL['wrist_%s'%s]       = cyl(wrdia/2, wrl/2, -wrl/4)
    COL['wrist_dev_%s'%s]   = cyl(wrdia/2, wrl/2, -wrl/4)
    COL['hook_%s'%s] = (box(ht, hw, hb, (0, 0, -hb/2)) +
                        box(hl, hw, ht, (ht/2 + hl/2, 0, -hb + ht/2)))

kids = {}
for j in joints:
    kids.setdefault(j['parent'], []).append(j['child'])
def subtree(n):
    out = [n]
    for c in kids.get(n, []):
        out += subtree(c)
    return out

M_TOTAL = sum(l['mass'] for l in links.values())

def effort_for(j):
    # Worst case for a climbing robot: the WHOLE robot hangs off this joint's
    # distal chain, at the largest lever arm that chain can reach.
    D = subtree(j['child'])
    o0 = links[j['child']]['org']
    reach = max(math.dist(links[d]['org'], o0) for d in D) + hb
    return max(5.0, math.ceil(G * M_TOTAL * reach))

L = ['<?xml version="1.0"?>',
     '<!-- Bimanual wall-climbing robot. Generated from Fusion "3d-climbing-robot". -->',
     '<!-- Units: m, kg, rad. Frame: +X forward (toward the wall), +Y left, +Z up. -->',
     '<!-- Joint sign convention: about Y, NEGATIVE = flexion / forward / closing. -->',
     '<!-- effort= values are geometry-derived static-hold estimates, NOT actuator specs. -->',
     '<robot name="bimanual_wall_climber">',
     '  <mujoco>',
     '    <compiler meshdir="." balanceinertia="true" discardvisual="false"/>',
     '  </mujoco>',
     '  <material name="aluminium">',
     '    <color rgba="0.75 0.76 0.78 1.0"/>',
     '  </material>']

sc = '%g %g %g' % (scale, scale, scale)
for n, d in links.items():
    m = d['mass']; cx, cy, cz = d['com']
    ixx, iyy, izz, ixy, iyz, ixz = d['I']
    L += ['', '  <link name="%s">' % n,
          '    <inertial>',
          '      <origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>' % (cx, cy, cz),
          '      <mass value="%.6f"/>' % m,
          '      <inertia ixx="%.9g" ixy="%.9g" ixz="%.9g" iyy="%.9g" iyz="%.9g" izz="%.9g"/>'
          % (ixx, ixy, ixz, iyy, iyz, izz),
          '    </inertial>',
          '    <visual>',
          '      <origin xyz="0 0 0" rpy="0 0 0"/>',
          '      <geometry><mesh filename="meshes/%s.stl" scale="%s"/></geometry>' % (n, sc),
          '      <material name="aluminium"/>',
          '    </visual>']
    for kind, g, o, rpy in COL[n]:
        if kind == 'box':
            gs = '<box size="%.6f %.6f %.6f"/>' % g['size']
        else:
            gs = '<cylinder radius="%.6f" length="%.6f"/>' % (g['radius'], g['length'])
        L += ['    <collision>',
              '      <origin xyz="%.6f %.6f %.6f" rpy="%s"/>' % (o[0], o[1], o[2], rpy),
              '      <geometry>%s</geometry>' % gs,
              '    </collision>']
    L += ['  </link>']

for j in joints:
    nm = NAME[(j['parent'], j['child'])]
    x, y, z = j['xyz']
    L += ['', '  <joint name="%s" type="%s">' % (nm, j['type']),
          '    <parent link="%s"/>' % j['parent'],
          '    <child link="%s"/>' % j['child'],
          '    <origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>' % (x, y, z)]
    if j['type'] == 'revolute':
        L += ['    <axis xyz="%d %d %d"/>' % tuple(j['axis']),
              '    <limit lower="%.6f" upper="%.6f" effort="%g" velocity="3.0"/>'
              % (j['lower'], j['upper'], effort_for(j)),
              '    <dynamics damping="0.1" friction="0.0"/>']
    L += ['  </joint>']

L += ['', '</robot>', '']
p = os.path.join(OUT, 'climbing_robot.urdf')
open(p, 'w').write('\n'.join(L))
print('wrote', p, '(%d lines)' % len(L))
