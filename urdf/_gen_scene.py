import json, math, os
OUT = os.path.dirname(os.path.abspath(__file__))
LD = json.load(open(os.path.join(OUT, '_linkdata.json')))
JD = json.load(open(os.path.join(OUT, '_jointdata.json')))
links, P, joints = LD['links'], JD['params'], JD['joints']

NAME = {('base_link','waist_link'):'waist_yaw', ('waist_link','torso_upper'):None}
for s in ('L','R'):
    NAME.update({('torso_upper','shoulder_%s'%s):'shoulder_%s_abduction'%s,
      ('shoulder_%s'%s,'upper_arm_%s'%s):'shoulder_%s_flexion'%s,
      ('upper_arm_%s'%s,'humerus_rot_%s'%s):'shoulder_%s_rotation'%s,
      ('humerus_rot_%s'%s,'fore_arm_%s'%s):'elbow_%s'%s,
      ('fore_arm_%s'%s,'forearm_rot_%s'%s):'forearm_%s_roll'%s,
      ('forearm_rot_%s'%s,'wrist_%s'%s):'wrist_%s_flexion'%s,
      ('wrist_%s'%s,'wrist_dev_%s'%s):'wrist_%s_deviation'%s,
      ('wrist_dev_%s'%s,'hook_%s'%s):'finger_%s'%s})

tld,tlw,tlh = P['torso_low_d'],P['torso_low_w'],P['torso_low_h']
tud,tuw,tuh = P['torso_up_d'],P['torso_up_w'],P['torso_up_h']
wdia,wh = P['waist_dia'],P['waist_h']
sdia,soff = P['shoulder_dia'],P['shoulder_off']
ldia,rdia,rl = P['link_dia'],P['roll_dia'],P['roll_len']
ual,fal = P['upper_arm_len'],P['fore_arm_len']
wrdia,wrl = P['wrist_dia'],P['wrist_len']
hb,hl,ht,hw = P['hook_back'],P['hook_lip'],P['hook_thk'],P['hook_width']

def C(r,x1,y1,z1,x2,y2,z2):
    return '<geom class="col" type="cylinder" size="%.5f" fromto="%.5f %.5f %.5f %.5f %.5f %.5f"/>'%(r,x1,y1,z1,x2,y2,z2)
def B(sx,sy,sz,ox,oy,oz):
    return '<geom class="col" type="box" size="%.5f %.5f %.5f" pos="%.5f %.5f %.5f"/>'%(sx/2,sy/2,sz/2,ox,oy,oz)

COL = {'base_link':[B(tld,tlw,tlh,0,0,tlh/2)],
       'waist_link':[C(wdia/2,0,0,0,0,0,wh)],
       'torso_upper':[B(tud,tuw,tuh,0,0,tuh/2)]}
for s,sg in (('L',1),('R',-1)):
    COL['shoulder_%s'%s]=[C(sdia/2,0,0,0,0,sg*soff,0)]
    COL['upper_arm_%s'%s]=[C(ldia/2,0,0,0,0,0,-(ual-rl))]
    COL['humerus_rot_%s'%s]=[C(rdia/2,0,0,0,0,0,-rl)]
    COL['fore_arm_%s'%s]=[C(ldia/2,0,0,0,0,0,-(fal-rl))]
    COL['forearm_rot_%s'%s]=[C(rdia/2,0,0,0,0,0,-rl)]
    COL['wrist_%s'%s]=[C(wrdia/2,0,0,0,0,0,-wrl/2)]
    COL['wrist_dev_%s'%s]=[C(wrdia/2,0,0,0,0,0,-wrl/2)]
    COL['hook_%s'%s]=[B(ht,hw,hb,0,0,-hb/2), B(hl,hw,ht,ht/2+hl/2,0,-hb+ht/2)]

KP = {'waist_yaw':800}
for s in ('L','R'):
    KP.update({'shoulder_%s_abduction'%s:800,'shoulder_%s_flexion'%s:800,
               'shoulder_%s_rotation'%s:500,'elbow_%s'%s:500,
               'forearm_%s_roll'%s:200,'wrist_%s_flexion'%s:150,
               'wrist_%s_deviation'%s:150,'finger_%s'%s:200})

kids={}
for j in joints: kids.setdefault(j['parent'],[]).append(j)
def sub(n):
    o=[n]
    for j in kids.get(n,[]): o+=sub(j['child'])
    return o
M_TOTAL=sum(l['mass'] for l in links.values())
def effort(j):
    D=sub(j['child']); o0=links[j['child']]['org']
    return max(5.0, math.ceil(9.81*M_TOTAL*(max(math.dist(links[d]['org'],o0) for d in D)+hb)))

acts=[]
def emit(name, ind):
    d=links[name]; p='  '*ind
    L=[]
    jj=[j for j in joints if j['child']==name]
    if jj:
        j=jj[0]; x,y,z=j['xyz']
        L.append('%s<body name="%s" pos="%.6f %.6f %.6f">'%(p,name,x,y,z))
        jn=NAME[(j['parent'],name)]
        if jn:
            L.append('%s  <joint name="%s" type="hinge" axis="%d %d %d" range="%.5f %.5f"/>'
                     %(p,jn,j['axis'][0],j['axis'][1],j['axis'][2],j['lower'],j['upper']))
            acts.append((jn, KP[jn], effort(j), j['lower'], j['upper']))
    else:
        L.append('%s<body name="%s" pos="0 0 0.30">'%(p,name))
        L.append('%s  <freejoint name="floating_base"/>'%p)
    cx,cy,cz=d['com']; ixx,iyy,izz,ixy,iyz,ixz=d['I']
    L.append('%s  <inertial pos="%.6f %.6f %.6f" mass="%.6f" fullinertia="%.9g %.9g %.9g %.9g %.9g %.9g"/>'
             %(p,cx,cy,cz,d['mass'],ixx,iyy,izz,ixy,ixz,iyz))
    L.append('%s  <geom class="vis" mesh="%s"/>'%(p,name))
    for g in COL[name]: L.append('%s  %s'%(p,g))
    if name == 'torso_upper':          # IMU mounting point, mid-chest
        L.append('%s  <site name="s_imu" type="box" pos="0 0 %.5f" size="0.02 0.02 0.01"/>'
                 % (p, tuh/2))
    if name.startswith('hook_'):
        sd = name[-1]
        # 6-axis F/T at the hand root: wrench passing through the finger joint
        L.append('%s  <site name="s_ft_%s" type="sphere" pos="0 0 0" size="0.008"/>' % (p, sd))
        # touch pad filling the hook throat - non-zero only when a hold is seated
        L.append('%s  <site name="s_grip_%s" type="box" pos="%.5f 0 %.5f" size="%.5f %.5f %.5f"/>'
                 % (p, sd, ht/2+hl/2, -(hb-ht)/2 - 0.010,
                    hl/2 + 0.004, hw/2, (hb-ht)/2 + 0.014))
    for j in kids.get(name,[]): L+=emit(j['child'], ind+1)
    L.append('%s</body>'%p)
    return L

open(os.path.join(OUT,'_bodytree.txt'),'w').write('\n'.join(emit('base_link',3)))
json.dump(acts, open(os.path.join(OUT,'_acts.json'),'w'))
print('body tree + %d actuators prepared' % len(acts))
