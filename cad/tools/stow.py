"""The full robot's climbing stow pose: where its legs fold while the arms climb.

    ROBOT=full python cad/tools/stow.py            # search hip pitch x knee, write nothing
    python cad/tools/check_full.py stow            # the acceptance check on spec.STOW

The legs must (1) meet no wall or rung over every reference climb's states on its own
wall and every lesson's hang (`clearance`), and (2) hide nothing the four lidars see of
the rungs at each lesson's hang today (`seen`, the v2 robot the reference). Worlds are
the lesson / reference-climb scenes' walls and rungs copied into urdf/full/scene.xml;
the state's base and arm joints are set by name and the legs at the candidate. The full
robot's arms are longer than v2's, so the same joint angles put its hooks elsewhere: the
base is then moved so the two grip pads' mid-point is where v2's is in that state (the
hooks on the same rung, as _gen_keyframe.place() hangs it).
"""
import itertools, os, sys
import numpy as np, mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
FULL = os.path.join(ROOT, "urdf", "full", "scene.xml")
LEG = ("hip_", "hip_roll_", "thigh_", "calf_", "ankle_", "foot_")
EVERY = 4                       # every 4th reference row


def _world_geoms(m):
    """(type, size, pos, quat, name, contact params) of a compiled scene's walls and rungs:
    the contact params too, or a hook slides off a rung that lost its friction"""
    return [(int(m.geom_type[g]), m.geom_size[g].copy(), m.geom_pos[g].copy(), m.geom_quat[g].copy(),
             m.geom(g).name, dict(friction=m.geom_friction[g].copy(), condim=int(m.geom_condim[g]),
                                  solref=m.geom_solref[g].copy(), solimp=m.geom_solimp[g].copy(),
                                  margin=float(m.geom_margin[g]), gap=float(m.geom_gap[g]),
                                  priority=int(m.geom_priority[g]),
                                  contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g])))
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == 0 and m.geom_contype[g] | m.geom_conaffinity[g]]


def _full_on(world):
    """urdf/full/scene.xml with its corridor replaced by `world`'s walls and rungs"""
    sp = mujoco.MjSpec.from_file(FULL)
    for g in list(sp.worldbody.geoms):
        sp.delete(g)
    for t, size, pos, quat, nm, kw in world:
        sp.worldbody.add_geom(name=nm, type=mujoco.mjtGeom(t), size=size, pos=pos, quat=quat, **kw)
    return sp.compile()


class World:
    """one wall: the full robot on it, its v2 states, and the leg/world geom pairs"""

    def __init__(self, world, states, names):
        self.m = _full_on(world)
        self.d = mujoco.MjData(self.m)
        self.states, self.names = states, names
        self.adr = {n: self.m.joint(n).qposadr[0] for n in names if n != "floating_base"}
        legs = {b for b in range(self.m.nbody) if self.m.body(b).name.startswith(LEG)
                and self.m.body(b).name[-2:] in ("_L", "_R")}
        lg = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in legs and self.m.geom_contype[g]]
        wg = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == 0]
        self.pairs = [(a, b) for a in lg for b in wg]
        self.grip = [_v2_grip(q) for q in states]

    def set(self, q, stow):
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[:7] = q[:7]
        for i, n in enumerate(self.names[1:]):
            d.qpos[self.adr[n]] = q[7 + i]
        for n, v in stow.items():
            d.qpos[self.m.joint(n).qposadr[0]] = v
        mujoco.mj_kinematics(self.m, d)
        g = self.grip[self.states_index(q)] if self.grip else None
        if g is not None:
            d.qpos[:3] += g - _grip_mid(self.m, d)
            mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)

    def states_index(self, q):
        for i, s_ in enumerate(self.states):
            if s_ is q or np.array_equal(s_, q):
                return i
        return None

    def clearance(self, stow, cap=0.05):
        worst = cap
        ft = np.zeros(6)
        for q in self.states:
            self.set(q, stow)
            for a, b in self.pairs:
                worst = min(worst, mujoco.mj_geomDistance(self.m, self.d, a, b, cap, ft))
        return worst


_V2 = []


def _grip_mid(m, d):
    return 0.5 * (d.site_xpos[m.site("s_grip_L").id] + d.site_xpos[m.site("s_grip_R").id])


def _v2_grip(q):
    """the v2 robot's grip-pad mid-point in state q (world frame)"""
    if not _V2:
        m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "urdf", "scene.xml"))
        _V2.extend([m, mujoco.MjData(m)])
    m, d = _V2
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return _grip_mid(m, d).copy()


def v2_names():
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "urdf", "scene.xml"))
    return ["floating_base"] + [m.joint(i).name for i in range(1, m.njnt)]


def worlds(every=EVERY):
    """every reference climb's wall with its rows, and every lesson's hang"""
    from rl import rsi
    from rl.mjx_scene import c_scene
    from rl.lessons import LESSONS
    from rl.climb_task import ClimbTask
    names, out = v2_names(), []
    z = rsi.load()
    for c in range(len(z["climb_family"])):
        xml, _ = c_scene(rsi.climb_wall(z, c), robot="full")
        try:
            w = _world_geoms(mujoco.MjModel.from_xml_path(xml))
        finally:
            os.remove(xml)
        out.append(("climb %d" % c, World(w, z["qpos"][z["climb"] == c][::every], names)))
    for k in sorted(LESSONS):        # the robot is its own mirror image: one side does
        t = ClimbTask(k, noise=False)
        t.reset()
        out.append(("lesson %d" % k, World(_world_geoms(t.env.model), [t.env.data.qpos.copy()], names)))
    return out


def settle(k, stow, seconds=10.0):
    """lesson k's hang with the FULL robot, simulated: hung by its hooks where v2's are
    with the legs at hip +30 (clear of the wall from v2's lean), the legs then moved to
    `stow` over 2 s and held while it swings to rest -> dict(hooked, drift (m / 2 s),
    legs (leg/world contacts), seen (cells per rung, the four lidars), v2 (v2's own))"""
    from rl import lidar as L
    from rl.climb_task import ClimbTask
    t = ClimbTask(k, noise=False)
    t.reset()
    q = t.env.data.qpos.copy()
    v2 = L.visible(np, L.Rig(t.env.model), t.env.data.site_xpos, t.env.data.site_xmat, *t._rungs).sum(1)
    w = World(_world_geoms(t.env.model), [q], v2_names())
    m, d = w.m, w.d
    w.set(q, stow_of(np.radians(30), 0.0))
    mujoco.mj_forward(m, d)
    adr = [m.joint(m.actuator_trnid[i][0]).qposadr[0] for i in range(m.nu)]
    ctrl = d.qpos[adr].copy()
    a0 = ctrl.copy()
    act = {m.joint(m.actuator_trnid[i][0]).name: i for i in range(m.nu)}
    def hold(sec):
        for _ in range(int(round(sec / m.opt.timestep))):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
    hold(1.0)
    for i in range(1, 101):
        for n, v in stow.items():
            ctrl[act[n]] = a0[act[n]] + (v - a0[act[n]]) * i / 100
        hold(0.02)
    hold(seconds - 2.0)
    p0 = d.body("base_link").xpos.copy()
    hold(2.0)
    nm = lambda g: m.body(m.geom_bodyid[g]).name
    legs = sorted({nm(c.geom1 if m.geom_bodyid[c.geom2] == 0 else c.geom2) for c in d.contact[:d.ncon]
                   if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
                   and nm(c.geom1 if m.geom_bodyid[c.geom2] == 0 else c.geom2).startswith(LEG)})
    return dict(hooked=all(np.linalg.norm(d.sensor("ft_force_%s" % s).data) > 50 for s in "LR"),
                drift=float(np.linalg.norm(d.body("base_link").xpos - p0)), legs=legs,
                seen=[int(v) for v in L.visible(np, L.Rig(m), d.site_xpos, d.site_xmat, *t._rungs).sum(1)],
                v2=[int(v) for v in v2])


def stow_of(hip, knee, ankle=0.0):
    """both legs: hip pitch, knee and ankle pitch (rad), the rest at zero"""
    return {**{"hip_%s_pitch" % s: hip for s in "LR"}, **{"knee_%s" % s: knee for s in "LR"},
            **{"ankle_%s_pitch" % s: ankle for s in "LR"}}


def seen(stow, robot_xml=FULL):
    """lesson -> seen cells per rung at its hang, the four lidars, legs at `stow`"""
    from rl import lidar as L
    from rl.climb_task import ClimbTask
    from rl.lessons import LESSONS
    out = {}
    names = v2_names()
    for k in sorted(LESSONS):
        t = ClimbTask(k, noise=False)
        t.reset()
        w = World(_world_geoms(t.env.model), [t.env.data.qpos.copy()], names) if robot_xml == FULL else None
        if w is None:
            m, d = t.env.model, t.env.data
        else:
            w.set(t.env.data.qpos, stow)
            m, d = w.m, w.d
        rig = L.Rig(m)
        out[k] = [int(v) for v in L.visible(np, rig, d.site_xpos, d.site_xmat, *t._rungs).sum(1)]
    return out


if __name__ == "__main__":
    W = worlds(every=20)
    best = None
    for hip, knee in itertools.product(np.radians(np.arange(0, 61, 10)), np.radians(np.arange(40, 111, 10))):
        st = stow_of(hip, knee, np.radians(30))
        c = min(w.clearance(st) for _, w in W)
        print("hip %+4.0f knee %4.0f: clearance %.1f mm" % (np.degrees(hip), np.degrees(knee), c * 1000), flush=True)
        if best is None or c > best[0]:
            best = (c, hip, knee)
    print("best: hip %+.0f knee %.0f, %.1f mm" % (np.degrees(best[1]), np.degrees(best[2]), best[0] * 1000))
