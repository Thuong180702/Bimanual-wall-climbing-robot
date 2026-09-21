"""Write urdf/ from the CAD: masses, inertias and envelopes all come from the
solids, and every effort/velocity limit is a number off a datasheet.

Outputs
  urdf/climbing_robot.urdf           fixed base
  urdf/climbing_robot_floating.urdf  free base, for MuJoCo
  urdf/meshes/<link>.stl             visual, mm, link frame, actuators included
  urdf/meshes/<link>_col_<part>.stl   one bracket part each, for collision
  urdf/_actuators.json               the motor table the scene generator reads
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "urdf"))

from cadgen import build123d as bd
from cadgen import read_step
import trimesh

from lib.build import BUILDERS, CHILD_OF, fuse, mount_plane
from lib.collide import col_parts, link_collision
from lib.motors import MOTORS
from lib.spec import FIXED, JOINTS, MOTOR_OF, MOUNT_ROT, PICK
from massprops import link_props

MM = 0.001
# The vendor solids carry every screw head; a fine tessellation of 17 of them
# makes a quarter-gigabyte of STL that nothing benefits from. These meshes are
# VISUAL only - the exact geometry lives in cad/STEP/robot.step.
TOL, ANG = 0.6, 0.6
RPM = math.pi / 30.0
STEPS = {"AK80-64": "ak80_64", "AK60-39": "ak60_39", "AK45-36": "ak45_36"}
PURCHASED = os.path.abspath(os.path.join(HERE, "..", "STEP", "purchased"))
_cache = {}


def motor_solid(key):
    """The vendor part already put in the canonical joint frame, in mm."""
    if key not in _cache:
        s = bd.import_step(os.path.join(PURCHASED, STEPS[key] + ".step"))
        # The AK60-39 file models every screw: 51 solids, 2428 faces, and a
        # 2 MB STL in every link that carries one. The eight largest solids are
        # the housing, rotor and covers - they hold the envelope to 0.1 mm and
        # halve the mesh. Exact geometry stays in cad/STEP/.
        sols = s.solids()
        if len(sols) > 8:
            keep = sorted(sols, key=lambda x: -x.volume)[:8]
            s = bd.Compound(children=list(keep))
        _cache[key] = s
    return _cache[key]


def _loc(origin_mm, z_dir, x_hint=(1, 0, 0)):
    z, x = bd.Vector(*z_dir), bd.Vector(*x_hint)
    if abs(x.dot(z)) > 0.9:
        x = bd.Vector(0, 0, 1)
    x = x - z * (x.dot(z) / z.dot(z))
    return bd.Location(bd.Plane(origin=origin_mm, x_dir=(x.X, x.Y, x.Z), z_dir=z_dir))


def link_visual(link):
    """Link solid plus every actuator it carries, in the link frame, mm."""
    kids = [fuse(BUILDERS[link]()).scale(1000.0)]
    for cj in CHILD_OF.get(link, []):
        m = mount_plane(cj, cj[4])
        kids.append(bd.Location(bd.Plane(
            origin=tuple(v * 1000 for v in m.origin),
            x_dir=m.x_dir, z_dir=m.z_dir)) * motor_solid(MOTOR_OF[cj[0]]))
    # a compound, not a boolean union: this mesh is only ever looked at, and
    # fusing a 2400-face vendor solid into a bracket buys nothing.
    return bd.Compound(children=kids)


def rpy(z_dir, x_hint):
    z = bd.Vector(*z_dir)
    z = z / z.length
    x = bd.Vector(*(x_hint or ((1, 0, 0) if abs(z.Z) < 0.9 else (0, 1, 0))))
    if abs(x.dot(z)) > 0.9:                       # hint parallel to the axis
        x = bd.Vector(0, 0, 1) if abs(z.Z) < 0.9 else bd.Vector(1, 0, 0)
    x = x - z * (x.dot(z))
    x = x / x.length
    y = z.cross(x)
    R = [[x.X, y.X, z.X], [x.Y, y.Y, z.Y], [x.Z, y.Z, z.Z]]
    sy = math.hypot(R[0][0], R[1][0])
    if sy > 1e-9:
        return (math.atan2(R[2][1], R[2][2]), math.atan2(-R[2][0], sy),
                math.atan2(R[1][0], R[0][0]))
    return (math.atan2(-R[1][2], R[1][1]), math.atan2(-R[2][0], sy), 0.0)


def main():
    os.makedirs(os.path.join(OUT, "meshes"), exist_ok=True)
    props = {}
    for link in BUILDERS:
        props[link] = link_props(link)
        bd.export_stl(link_visual(link),
                      os.path.join(OUT, "meshes", "%s.stl" % link),
                      tolerance=TOL, angular_tolerance=ANG)
        # the bracket's parts are the collision meshes, one each - MuJoCo hulls
        # every mesh (see lib/collide.py) - and the actuators are separate
        # cylinders that are exact
        if not link.startswith("hook"):
            for name, solid in col_parts(link):
                # wrapped, not scaled bare: a part placed by `plane * shape`
                # keeps that Location, and .scale() on it moved upper_arm_L's
                # face plate 200 mm up its own arm
                p = os.path.join(OUT, "meshes", "%s.stl" % name)
                bd.export_stl(bd.Compound(children=[solid]).scale(1000.0), p,
                              tolerance=TOL, angular_tolerance=ANG)
                # MuJoCo collides a mesh's HULL anyway, so store only that: the
                # same geometry in 5.8k triangles instead of 66k, and the BVH
                # MuJoCo builds over the triangles was most of a model's memory
                trimesh.load(p).convex_hull.export(p)

    total = sum(p[0] for p in props.values())
    L = ['<?xml version="1.0"?>',
         '<!-- Bimanual wall-climbing robot, v2: real actuators. -->',
         '<!-- Generated by cad/tools/gen_urdf.py from the CAD assembly - do not hand-edit. -->',
         '<!-- Units: m, kg, rad. Frame: +X forward (toward the wall), +Y left, +Z up. -->',
         '<!-- Joint sign convention: about Y, NEGATIVE = flexion / forward / closing. -->',
         '<!-- effort = the actuator PEAK torque from its datasheet. -->',
         '<!-- velocity = the actuator NO-LOAD speed from its datasheet. -->',
         '<!-- Total mass %.4f kg. -->' % total,
         '<robot name="bimanual_wall_climber_v2">',
         '  <mujoco>',
         '    <compiler meshdir="." balanceinertia="true" discardvisual="false"/>',
         '    <visual><global offwidth="1600" offheight="1200"/></visual>',
         '  </mujoco>',
         '  <material name="aluminium"><color rgba="0.75 0.76 0.78 1.0"/></material>',
         '  <material name="actuator"><color rgba="0.25 0.27 0.30 1.0"/></material>']

    for link in BUILDERS:
        m, c, I, _ = props[link]
        L += ['', '  <link name="%s">' % link,
              '    <inertial>',
              '      <origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>' % c,
              '      <mass value="%.6f"/>' % m,
              '      <inertia ixx="%.9g" ixy="%.9g" ixz="%.9g" iyy="%.9g" '
              'iyz="%.9g" izz="%.9g"/>'
              % (I[0][0], I[0][1], I[0][2], I[1][1], I[1][2], I[2][2]),
              '    </inertial>',
              '    <visual>',
              '      <origin xyz="0 0 0" rpy="0 0 0"/>',
              '      <geometry><mesh filename="meshes/%s.stl" scale="0.001 0.001 0.001"/></geometry>' % link,
              '      <material name="aluminium"/>',
              '    </visual>']
        for kind, size, pos, zdir, xhint in link_collision(link):
            r, p_, y_ = rpy(zdir, xhint)
            if kind == "cylinder":
                g = '<cylinder radius="%.6f" length="%.6f"/>' % size
                pos = tuple(pos)
            elif kind == "box":
                g = '<box size="%.6f %.6f %.6f"/>' % size
            else:
                g = '<mesh filename="meshes/%s.stl" scale="0.001 0.001 0.001"/>' % size[0]
            L += ['    <collision>',
                  '      <origin xyz="%.6f %.6f %.6f" rpy="%.6f %.6f %.6f"/>'
                  % (pos[0], pos[1], pos[2], r, p_, y_),
                  '      <geometry>%s</geometry>' % g,
                  '    </collision>']
        L += ['  </link>']

    acts = {}
    for name, par, ch, ax, off, lim, _bdir in JOINTS:
        lo, hi = lim
        key = MOTOR_OF[name]
        d = MOTORS[key]
        eff, vel = d["peak_Nm"], d["noload_rpm"] * RPM
        acts[name] = dict(motor=key, peak_Nm=eff, rated_Nm=d["rated_Nm"],
                          noload_rad_s=vel, rated_rad_s=d["rated_rpm"] * RPM,
                          lower=lo, upper=hi, axis=list(ax))
        L += ['', '  <joint name="%s" type="revolute">' % name,
              '    <parent link="%s"/>' % par,
              '    <child link="%s"/>' % ch,
              '    <origin xyz="%.6f %.6f %.6f" rpy="0 0 %.9g"/>'
              % (*off, MOUNT_ROT.get(ch, 0.0)),
              '    <axis xyz="%d %d %d"/>' % ax,
              '    <limit lower="%.6f" upper="%.6f" effort="%g" velocity="%.4f"/>'
              % (lo, hi, eff, vel),
              '    <dynamics damping="0.1" friction="0.0"/>',
              '  </joint>']
    for nm, par, ch, off in FIXED:
        L += ['', '  <joint name="%s" type="fixed">' % nm,
              '    <parent link="%s"/>' % par,
              '    <child link="%s"/>' % ch,
              '    <origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>' % off,
              '  </joint>']
    L += ['', '</robot>', '']

    open(os.path.join(OUT, "climbing_robot.urdf"), "w").write("\n".join(L))

    F = L[:]
    i = F.index('<robot name="bimanual_wall_climber_v2">') + 1
    F[i:i] = ['', '  <link name="world"/>',
              '  <joint name="floating_base" type="floating">',
              '    <parent link="world"/>', '    <child link="base_link"/>',
              '    <origin xyz="0 0 0" rpy="0 0 0"/>', '  </joint>']
    open(os.path.join(OUT, "climbing_robot_floating.urdf"), "w").write("\n".join(F))

    json.dump(dict(total_mass=total, actuators=acts,
                   links={k: dict(mass=v[0], com=list(v[1])) for k, v in props.items()},
                   picks=PICK, motors=MOTORS),
              open(os.path.join(OUT, "_actuators.json"), "w"), indent=1)
    print("total mass %.4f kg, %d links, %d joints" % (total, len(BUILDERS), len(JOINTS)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
