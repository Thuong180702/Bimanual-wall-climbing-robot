"""ClimbTask in JAX, thousands of envs at once on MJX: the GPU path of rl.train_ppo.

    python -m rl.mjx_task --envs 1024       # control-steps/s on this machine
    python -m rl.mjx_check                   # is it the same task as rl.climb_task

A line-for-line port of `climb_task.ClimbTask` -- the same 145 observations (the
anchor-rooted estimator, the SLAM-tracked anchor, reach odometry, the nearest
rungs), the same reward books, the same success test and the same terminations
-- on `rl.mjx_scene`'s simplified robot. Every constant is read from climb_task,
so a reward change there is a reward change here. The comments that say WHY a rule
is what it is live there, next to the numpy version; this file only says where the
port had to differ.

What had to change, and nothing else:

  * one model for every wall. A wall -- a lesson, or a generated one (rl.walls) -- is
    a row of the tables (rung slots, walls, goals, timeout); an env copies its row
    into its own state at reset (`Wall`) and writes its rungs and extra wall into its
    own copy of the model's slot geoms (`_env_model`). Generated walls are a bank of
    rows after the lessons' that `Batch.set_bank` refills without a recompile.
  * a reset is a copy. ClimbTask.reset hangs the robot from the keyframe and closes
    the fingers over 0.4 s, which is deterministic, so that is done once per lesson
    at build time (`_secure`) and a reset only copies it and senses (SLAM noise is
    drawn fresh).
  * branches are selects: `jp.where` both ways instead of `if`.
  * a finished env resets inside the same jitted step, onto a lesson drawn from the
    pool's weights, which the curriculum moves without a recompile.
"""
import inspect, os, time, warnings
# MJX shrinks a mesh face of more than 20 coplanar vertices to 20 for its contact
# clipping, and says so for 34 of ours. It changes nothing here: every mesh meets only
# a wall plane (rl.mjx_scene), and plane-mesh reads the vertices, not the faces.
warnings.filterwarnings('ignore', message=r'Mesh ".*" has a coplanar face with more than')
# JAX takes 75 % of the GPU up front by default, and torch -- the policy -- lives on
# the same card
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from typing import NamedTuple
import numpy as np
import jax, jax.numpy as jp
# The physics in full float32 on every card. On Ampere and newer XLA lowers a
# float32 matmul to TF32 (a 10-bit mantissa) by default; the cross-checks against C
# were made without it. A V100 has no TF32, so there this changes nothing.
jax.config.update('jax_default_matmul_precision', 'highest')
import mujoco
from mujoco import mjx
from controller.kinematics import JOINTS, THROAT_PT, MAG_WORLD
from climbing_env import ClimbingEnv, GRIP_BAND, GRIP_SET, SLAM_SIG
from . import climb_task as ct
from .lessons import LESSONS, canon_axis
from .mjx_scene import N_RUNG, lesson_layout, model as scene_model, slots

LESSON_KEYS = sorted(LESSONS)
N_PTS = 256        # route points a wall carries (climb_task.route_table): 180 on the
                   # largest family walls, lesson 3's three full rows
FAR_PT = 1e3       # padding: a point, and a route distance, no hand ever takes
REASONS = ('', 'success', 'timeout', 'fell', 'released', 'blowup')
_ENV = {k: v.default for k, v in inspect.signature(ClimbingEnv).parameters.items()}
FALL_Z, FREE_FALL_S, QVEL_MAX = _ENV['fall_z'], _ENV['free_fall_s'], _ENV['qvel_max']
FINGERS = [JOINTS.index('finger_L'), JOINTS.index('finger_R')]
MINVAL = 1e-15           # mujoco's mjMINVAL, for rotvec's zero-angle case


# --- small rotation helpers, as climb_task / controller.kinematics compute them ------
def _skew(v):
    z = jp.zeros_like(v[0])
    return jp.stack([jp.stack([z, -v[2], v[1]]), jp.stack([v[2], z, -v[0]]),
                     jp.stack([-v[1], v[0], z])])


def _rotm(v):
    """climb_task._rotm: rotation vector -> matrix (Rodrigues)"""
    th = jp.sqrt(v @ v)
    small = th < 1e-12
    K = _skew(v / jp.where(small, 1.0, th))
    R = jp.eye(3) + jp.sin(th) * K + (1.0 - jp.cos(th)) * K @ K
    return jp.where(small, jp.eye(3), R)


def _mat2quat(R):
    """mju_mat2Quat, branch for branch"""
    m = R.reshape(9)
    tr = m[0] + m[4] + m[8]
    s = lambda x: 0.5 * jp.sqrt(jp.maximum(x, MINVAL))
    a0 = s(1 + tr)
    qa = jp.stack([a0, 0.25 * (m[7] - m[5]) / a0, 0.25 * (m[2] - m[6]) / a0, 0.25 * (m[3] - m[1]) / a0])
    b1 = s(1 + m[0] - m[4] - m[8])
    qb = jp.stack([0.25 * (m[7] - m[5]) / b1, b1, 0.25 * (m[1] + m[3]) / b1, 0.25 * (m[2] + m[6]) / b1])
    c2 = s(1 - m[0] + m[4] - m[8])
    qc = jp.stack([0.25 * (m[2] - m[6]) / c2, 0.25 * (m[1] + m[3]) / c2, c2, 0.25 * (m[5] + m[7]) / c2])
    d3 = s(1 - m[0] - m[4] + m[8])
    qd = jp.stack([0.25 * (m[3] - m[1]) / d3, 0.25 * (m[2] + m[6]) / d3, 0.25 * (m[5] + m[7]) / d3, d3])
    q = jp.where(tr > 0, qa, jp.where((m[0] > m[4]) & (m[0] > m[8]), qb,
                                      jp.where(m[4] > m[8], qc, qd)))
    return q / jp.sqrt(q @ q)


def _rotvec(R):
    """controller.kinematics.rotvec: mju_mat2Quat then mju_quat2Vel(dt=1)"""
    q = _mat2quat(R)
    n = jp.sqrt(q[1:] @ q[1:])
    ax = jp.where(n < MINVAL, jp.array([1.0, 0.0, 0.0]), q[1:] / jp.where(n < MINVAL, 1.0, n))
    speed = 2.0 * jp.arctan2(n, q[0])
    return ax * jp.where(speed > jp.pi, speed - 2.0 * jp.pi, speed)


def _frame(a, b):
    x = a / jp.sqrt(a @ a)
    z = jp.cross(x, b)
    z = z / jp.sqrt(z @ z)
    return jp.stack([x, jp.cross(z, x), z], axis=1)


def _triad(up, mag):
    """controller.kinematics.triad"""
    return _frame(jp.array([0.0, 0.0, 1.0]), jp.asarray(MAG_WORLD)) @ _frame(up, mag).T


# --- the state one env carries ------------------------------------------------------
class State(NamedTuple):
    d: mjx.Data
    lesson: jax.Array          # index into LESSON_KEYS
    ctrl: jax.Array            # (17,) setpoints
    key: jax.Array
    no_grip_t: jax.Array       # ClimbingEnv._no_grip_t: s since any hook touched a rung
    # the estimator (climb_task._sense)
    Rf: jax.Array              # (3, 3) filtered world<-torso
    anchor: jax.Array          # 0 = L, 1 = R
    hold_idx: jax.Array        # (2,) rung slot of each hand's hold
    hold_pt: jax.Array         # (2, 3) its point on that rung's axis
    both_conf: jax.Array
    anchor_fresh: jax.Array
    est_thr: jax.Array         # (2, 3) estimated throats, what the observation reads
    # the reward's books (climb_task.reset / step)
    prev_d: jax.Array
    prev_live: jax.Array
    credit: jax.Array
    held: jax.Array
    prev_thr: jax.Array
    prev_loaded: jax.Array
    let_go: jax.Array
    grabs: jax.Array
    cross_axis: jax.Array
    cross_sign: jax.Array
    off_t: jax.Array
    held_t: jax.Array
    slide_rung: jax.Array      # (2,) each hook's loaded run: its rung slot (-1: none),
    slide_ref: jax.Array       # where along it the run began,
    slide_net: jax.Array       # and how far from there it is now
    n_steps: jax.Array
    wall: 'Wall'               # this episode's wall, copied at its reset: a new bank
                               # (Batch.set_bank) never moves a rung under a robot


class Wall(NamedTuple):
    """one wall: a lesson, or a generated one (rl.walls)"""
    rung_c: jax.Array          # (N_RUNG, 3) the map: centre, axis, half extent
    rung_a: jax.Array
    rung_h: jax.Array
    rung_valid: jax.Array
    slot_pos: jax.Array        # (N_RUNG, 3) what the slot geoms become
    slot_quat: jax.Array
    slot_size: jax.Array
    wall_pos: jax.Array        # (3,) the extra wall plane
    wall_quat: jax.Array
    goals: jax.Array           # (2, 3)
    goal_mask: jax.Array       # (N_RUNG,)
    pts: jax.Array             # (N_PTS, 3) climb_task.route_table's points, padded far away
    pts_g: jax.Array           # (2, N_PTS) their route distance to each goal
    timeout: jax.Array
    family: jax.Array          # rl.walls family, -1 for a lesson
    probe: jax.Array           # which of its parameters sits at its top, -1 none
    probe_val: jax.Array       # ...and that top, so ADR counts a probe only at its own
    d0_row: jax.Array          # the lesson whose secured hang it starts from


class Tables(NamedTuple):
    """the walls an env can draw, leading axis = the rows: LESSON_KEYS, then a bank of
    generated walls (rl.walls) that `Batch.set_bank` refills without a recompile"""
    walls: Wall                # (R, ...)
    tau_idle: jax.Array        # (L,) climb_task's idle-hang torque, per lesson
    fight_idle: jax.Array      # (L,) ...and its internal force
    d0: mjx.Data               # (L,) the secured hang, per lesson
    ctrl0: jax.Array
    ngt0: jax.Array


class MjxClimb:
    """the task over every lesson, one model. Pure functions of (tables, state) for
    one env; `Batch` vmaps them."""

    def __init__(self, bank=0):
        self.bank = bank           # generated-wall rows after the lessons' (rl.walls)
        self.mc = scene_model()
        m = self.mc
        self.mx = mjx.put_model(m, impl='jax')
        self.dt = float(m.opt.timestep)
        self.lo = m.actuator_ctrlrange[:, 0].copy()
        self.hi = m.actuator_ctrlrange[:, 1].copy()
        self.lo[FINGERS] = ct.GRIP_STOP
        sen = lambda n: np.arange(m.sensor(n).adr[0], m.sensor(n).adr[0] + m.sensor(n).dim[0])
        self.s_q = np.array([m.sensor('enc_' + j).adr[0] for j in JOINTS])
        self.s_dq = np.array([m.sensor('vel_' + j).adr[0] for j in JOINTS])
        self.s_tau = np.array([m.sensor('tau_' + j).adr[0] for j in JOINTS])
        self.s_acc, self.s_gyro, self.s_mag = sen('imu_acc'), sen('imu_gyro'), sen('imu_mag')
        self.s_f = np.stack([sen('ft_force_' + s) for s in 'LR'])
        self.s_t = np.stack([sen('ft_torque_' + s) for s in 'LR'])
        self.s_pad = np.array([m.sensor('grip_' + s).adr[0] for s in 'LR'])
        self.ft_site = np.array([m.site('s_ft_' + s).id for s in 'LR'])
        self.hook = np.array([m.body('hook_' + s).id for s in 'LR'])
        self.base, self.torso = m.body('base_link').id, m.body('torso_upper').id
        self.rung_geoms = np.array([m.geom(f'slot_rung{i}').id for i in range(N_RUNG)])
        # each hook's geoms that are paired with a rung: what climb_task._contact sums
        on_rung = {g for a, b in zip(m.pair_geom1, m.pair_geom2)
                   for g in (a, b) if {a, b} & set(self.rung_geoms)}
        self.hook_geoms = np.array([sorted(g for g in on_rung if m.geom_bodyid[g] == h)
                                    for h in self.hook])
        self.wall_geom = m.geom('slot_wall').id
        self.weight = float(m.body_subtreemass[self.base] * -m.opt.gravity[2])
        d = mjx.make_data(self.mx)
        con = d._impl.contact
        assert (con.dim == 3).all(), "every pair is condim 3 (rl.mjx_scene._mix)"
        # the four pyramid rows of each contact; their sum is its normal force
        self.efc4 = con.efc_address[:, None] + np.arange(4)
        self.tables = self._tables()

    # ---- per-lesson tables ------------------------------------------------------
    @staticmethod
    def rows(layouts):
        """lesson layouts (rl.mjx_scene.lesson_layout's, or rl.walls') -> the per-row
        arrays of Tables. A generated wall starts from lesson 1's secured hang: its
        start rung holds the keyframe's hooks and nothing else of it is within
        rl.walls.CLEAR of the robot there (rl.walls.Checker)."""
        from .walls import FAMILIES
        out = []
        for lay in layouts:
            sl = slots(lay)
            n = len(lay['rungs'])
            c = np.zeros((N_RUNG, 3)); a = np.zeros((N_RUNG, 3)); h = np.zeros(N_RUNG)
            for i, (_, ci, ai, hi, _) in enumerate(lay['rungs']):
                c[i], a[i], h[i] = ci, canon_axis(ai), hi     # whatever wrote the layout
            mask = np.zeros(N_RUNG, bool)
            mask[lay['goal_rungs']] = True
            P, G = ct.route_table([(c[i], a[i], h[i]) for i in range(n)], lay['goals'])
            assert len(P) <= N_PTS, f"{len(P)} route points: raise N_PTS"
            pts = np.full((N_PTS, 3), FAR_PT)
            pts[:len(P)] = P
            pts_g = np.full((2, N_PTS), FAR_PT)
            pts_g[:, :len(P)] = G
            f, pr = lay.get('family', -1), lay.get('probe')
            probe = -1 if pr is None else [s[0] for s in FAMILIES[f]].index(pr)
            out.append(dict(rung_c=c, rung_a=a, rung_h=h, rung_valid=np.arange(N_RUNG) < n,
                            slot_pos=sl['rung_pos'], slot_quat=sl['rung_quat'],
                            slot_size=sl['rung_size'], wall_pos=sl['wall_pos'],
                            wall_quat=sl['wall_quat'], goals=lay['goals'], goal_mask=mask,
                            pts=pts, pts_g=pts_g,
                            timeout=lay['timeout'], family=f, probe=probe,
                            probe_val=lay.get('probe_val', 0.0),
                            d0_row=lay.get('d0_row', LESSON_KEYS.index(1))))
        return {k: np.stack([r[k] for r in out]) for k in out[0]}

    def _tables(self):
        layouts = [dict(lesson_layout(k), d0_row=i) for i, k in enumerate(LESSON_KEYS)]
        layouts += [dict(lesson_layout(1), family=1)] * self.bank   # until a bank is set
        w = Wall(**{f: jp.asarray(v) for f, v in self.rows(layouts).items()})
        L = len(LESSON_KEYS)
        d0, ctrl0, ngt0 = jax.jit(jax.vmap(self._secure))(
            w.slot_pos[:L], w.slot_quat[:L], w.slot_size[:L], w.wall_pos[:L], w.wall_quat[:L])
        tau_idle = jax.vmap(lambda d: jp.abs(d.sensordata[self.s_tau]).sum())(d0)
        fight_idle = jax.vmap(lambda d: self._internal(d, self._sensors(d)))(d0)
        return Tables(walls=w, tau_idle=tau_idle, fight_idle=fight_idle, d0=d0, ctrl0=ctrl0, ngt0=ngt0)

    @staticmethod
    def wall(T, row):
        """row `row` of the tables' walls"""
        return jax.tree_util.tree_map(lambda x: x[row], T.walls)

    def _env_model(self, w):
        """the shared model with this wall's rungs and extra wall in the slot geoms"""
        return self._slot_model(w.slot_pos, w.slot_quat, w.slot_size, w.wall_pos, w.wall_quat)

    def _slot_model(self, pos, quat, size, wpos, wquat):
        mx = self.mx
        return mx.replace(
            geom_pos=mx.geom_pos.at[self.rung_geoms].set(pos).at[self.wall_geom].set(wpos),
            geom_quat=mx.geom_quat.at[self.rung_geoms].set(quat).at[self.wall_geom].set(wquat),
            geom_size=mx.geom_size.at[self.rung_geoms].set(size))

    def _secure(self, pos, quat, size, wpos, wquat):
        """ClimbTask.reset's physics: the keyframe, then 0.4 s closing the fingers to
        GRIP_SET. Deterministic, so it runs once per lesson."""
        m, mc = self._slot_model(pos, quat, size, wpos, wquat), self.mc
        d = mjx.make_data(self.mx).replace(
            qpos=jp.asarray(mc.key_qpos[0]), qvel=jp.asarray(mc.key_qvel[0]),
            ctrl=jp.asarray(mc.key_ctrl[0]), time=jp.asarray(mc.key_time[0]))
        d = mjx.forward(m, d)
        n = int(0.4 / self.dt)
        ctrl0 = jp.asarray(mc.key_ctrl[0])

        def one(carry, i):
            d, ngt = carry
            ctrl = ctrl0.at[np.array(FINGERS)].set(GRIP_SET * (i + 1) / n)
            d = mjx.step(m, d.replace(ctrl=ctrl))
            con, _ = self._contact(d)
            return (d, jp.where((con > 0.0).any(), 0.0, ngt + self.dt)), None

        (d, ngt), _ = jax.lax.scan(one, (d, jp.zeros(())), jp.arange(n))
        return d, d.ctrl, ngt

    # ---- measurements -----------------------------------------------------------
    def _contact(self, d):
        """climb_task._contact for both hands: (N of normal force between each hook
        and the rungs (2,), which rung slots each hook touches (2, N_RUNG))"""
        c = d._impl.contact
        fn = d._impl.efc_force[self.efc4].sum(-1)
        g1, g2 = c.geom1[:, None], c.geom2[:, None]
        rung1 = (g1 == self.rung_geoms[None]).any(-1)
        rung2 = (g2 == self.rung_geoms[None]).any(-1)
        on = []
        for s in (0, 1):
            h1 = (g1 == self.hook_geoms[s][None]).any(-1)
            h2 = (g2 == self.hook_geoms[s][None]).any(-1)
            on.append((h1 & rung2) | (h2 & rung1))
        on = jp.stack(on)                                                 # (2, ncon)
        force = jp.where(on, jp.abs(fn)[None], 0.0).sum(-1)
        slot = (c.geom1[:, None] == self.rung_geoms[None]) | (c.geom2[:, None] == self.rung_geoms[None])
        touched = (on[:, :, None] & slot[None] & (c.dist < 0.0)[None, :, None]).any(1)
        return force, touched

    def _sensors(self, d):
        sd = d.sensordata
        return dict(q=sd[self.s_q], dq=sd[self.s_dq], tau=sd[self.s_tau],
                    acc=sd[self.s_acc], gyro=sd[self.s_gyro], mag=sd[self.s_mag],
                    f=sd[self.s_f], t=sd[self.s_t], pad=sd[self.s_pad])

    def _throat_true(self, d):
        """(2, 3): the oracle's throats (reward and success only)"""
        return d.xpos[self.hook] + jp.einsum('sij,j->si', d.xmat[self.hook], jp.asarray(THROAT_PT))

    @staticmethod
    def _gripping(sen):
        ang = sen['q'][np.array(FINGERS)]
        return (GRIP_BAND[0] <= ang) & (ang <= GRIP_BAND[1])

    def _confirmed(self, sen):
        fn = jp.sqrt((sen['f'] ** 2).sum(-1))
        return (sen['pad'] > 20.0) & self._gripping(sen) & (fn > 12.0)

    def _touching(self, sen):
        return (sen['pad'] > 2.0) & self._gripping(sen)

    # ---- the wall map (jp.argmin keeps the FIRST minimum, as climb_task's
    # `if d < best` loops over the rungs do) --------------------------------------
    @staticmethod
    def _seg_point(p, c, a, h):
        """nearest point on every rung's axis (clipped to +-h), and the distance"""
        s = ((p[None] - c) * a).sum(-1)
        q = c + jp.clip(s, -h, h)[:, None] * a
        return q, jp.sqrt(((p[None] - q) ** 2).sum(-1)), s

    def _nearest_rung(self, w, p):
        """climb_task._nearest_rung: (found, slot, point)"""
        q, dist, _ = self._seg_point(p, w.rung_c, w.rung_a, w.rung_h)
        dist = jp.where(w.rung_valid, dist, jp.inf)
        i = jp.argmin(dist)
        return dist[i] < ct.RUNG_TOL, i, q[i]

    def _reach_point(self, w, anchor_pt, L, guess):
        """climb_task._reach_point: (found, slot, point)"""
        c, a, h = w.rung_c, w.rung_a, w.rung_h
        v = c - anchor_pt[None]
        b = (v * a).sum(-1)
        disc = b * b - ((v * v).sum(-1) - L * L)
        r = jp.sqrt(jp.maximum(disc, 0.0))
        s = jp.stack([-b - r, -b + r], axis=1)                            # (N_RUNG, 2)
        ok = w.rung_valid[:, None] & (disc >= 0.0)[:, None] & (jp.abs(s) <= h[:, None] + 0.03)
        q = c[:, None] + jp.clip(s, -h[:, None], h[:, None])[..., None] * a[:, None]
        err = jp.where(ok, jp.sqrt(((q - guess) ** 2).sum(-1)), jp.inf).reshape(-1)
        i = jp.argmin(err)
        return err[i] < ct.REACH_TOL, i // 2, q.reshape(-1, 3)[i]

    def _near_rungs(self, w, p):
        """climb_task._near_rungs"""
        c, a, h = w.rung_c, w.rung_a, w.rung_h
        hs = jp.maximum(h - ct.SEAT_IN, 0.0)
        q, dist, s = self._seg_point(p, c, a, hs)
        dist = jp.where(w.rung_valid, dist, jp.inf)
        order = jp.argsort(dist, stable=True)[:ct.N_NEAR]
        have = jp.isfinite(dist[order])[:, None]
        out = jp.concatenate([jp.where(have, q[order] - p[None], jp.asarray(ct.FAR)[None]),
                              jp.where(have, a[order], 0.0)], axis=1).reshape(-1)
        i = order[0]
        return jp.concatenate([out, jp.stack([hs[i] - s[i], hs[i] + s[i]])])

    def _dists(self, w, thr):
        """(climb_task._hand_dists, climb_task._live) for both hands: the bank's from
        the NEAREST rung's point (climb_task._route), the live one along the routes
        through the rungs (climb_task._cost)"""
        hs = jp.maximum(w.rung_h - ct.SEAT_IN, 0.0)
        d, live = [], []
        for s in (0, 1):
            q, dist, _ = self._seg_point(thr[s], w.rung_c, w.rung_a, hs)
            dist = jp.where(w.rung_valid, dist, jp.inf)
            d.append(jp.sqrt(((q[jp.argmin(dist)] - w.goals[s]) ** 2).sum()))
            dv = w.pts - thr[s][None]
            air = jp.sqrt((dv ** 2).sum(-1))
            reach = jp.sqrt((dv[:, 0] ** 2 + dv[:, 1] ** 2) / ct.HOP ** 2 + dv[:, 2] ** 2 / ct.HOP_UP ** 2)
            out = air * jp.maximum(1.0 - 1.0 / jp.maximum(reach, 1e-9), 0.0)
            below = jp.where(w.rung_valid, jp.maximum(q[:, 2] - thr[s][2], 0.0), jp.inf).min()
            live.append((air + ct.K_OVER * out + w.pts_g[s]).min() + ct.AIR * below)
        return jp.stack(d), jp.stack(live)

    def _internal(self, d, sen):
        """climb_task._internal: N of force the two wrists' (world-frame) forces cancel"""
        F = jp.einsum('sij,sj->si', d.site_xmat[self.ft_site], sen['f'])
        n = jp.sqrt((F ** 2).sum(-1))
        return 0.5 * (n.sum() - jp.sqrt((F.sum(0) ** 2).sum()))

    @staticmethod
    def _order(w, st, a):
        """climb_task._order: which way round the hands belong along axis a"""
        g = (w.goals[0] - w.goals[1]) @ a
        c = st.cross_axis @ a
        return jp.where(jp.abs(g) > 0.05, jp.sign(g),
                        jp.where(jp.abs(c) > 0.5, st.cross_sign * jp.sign(c), 0.0))

    # ---- the estimator ----------------------------------------------------------
    def _tree(self, d, q):
        """encoder-only kinematics with the free joint at the origin: throats, base
        and torso in the tree's own frame (Estimator.update's first mj_kinematics)"""
        dd = mjx.kinematics(self.mx, d.replace(qpos=jp.concatenate(
            [jp.zeros(3), jp.array([1.0, 0.0, 0.0, 0.0]), q])))
        thr = dd.xpos[self.hook] + jp.einsum('sij,j->si', dd.xmat[self.hook], jp.asarray(THROAT_PT))
        return thr, dd.xpos[self.base], dd.xmat[self.torso]

    @staticmethod
    def _place(tree, Rf, side, pos):
        """Estimator.update: the tree placed with its `side` throat on `pos`, rotated
        by the filtered attitude -> (world throats (2, 3), world base)"""
        thr, base, Rt = tree
        R = Rf @ Rt.T
        p = pos - R @ thr[side]
        return thr @ R.T + p, R @ base + p

    def _sense(self, T, st, d, w_sum, noise):
        """climb_task._sense. `w_sum`: the gyro summed over the step's substeps;
        `noise`: the SLAM's position error this step."""
        sen = self._sensors(d)
        dt = ct.FRAME_SKIP * self.dt
        Rf = st.Rf @ _rotm(w_sum * self.dt)
        p_slam = d.xpos[self.base] + noise
        R_tri = _triad(sen['acc'], sen['mag'])
        Rf = Rf @ _rotm((dt / ct.AHRS_TAU) * _rotvec(Rf.T @ R_tri))
        tree = self._tree(d, sen['q'])
        a, w = st.anchor, st.wall
        hold_idx, hold_pt = st.hold_idx, st.hold_pt
        _, base = self._place(tree, Rf, a, hold_pt[a])
        ax = w.rung_a[hold_idx[a]]
        hold_pt = hold_pt.at[a].add((dt / ct.SLAM_TAU) * ((p_slam - base) @ ax) * ax)
        thr, _ = self._place(tree, Rf, a, hold_pt[a])
        o = 1 - a
        conf, touch = self._confirmed(sen), self._touching(sen)
        cA, cB = conf[a], conf[o]
        both = cB & (cA | st.anchor_fresh | touch[a])
        pA = hold_pt[a]
        guess = jp.where(st.both_conf, hold_pt[o], thr[o])
        ok1, i1, q1 = self._reach_point(w, pA, jp.sqrt(((thr[o] - thr[a]) ** 2).sum()), guess)
        ok2, i2, q2 = self._nearest_rung(w, thr[o])
        hit = both & (ok1 | ok2)
        hold_idx = hold_idx.at[o].set(jp.where(hit, jp.where(ok1, i1, i2), hold_idx[o]))
        hold_pt = hold_pt.at[o].set(jp.where(hit, jp.where(ok1, q1, q2), hold_pt[o]))
        switch = ~cA & cB
        anchor = jp.where(switch, o, a)
        thr2, _ = self._place(tree, Rf, o, hold_pt[o])
        thr = jp.where(switch, thr2, thr)
        return st._replace(Rf=Rf, anchor=anchor, hold_idx=hold_idx, hold_pt=hold_pt,
                           both_conf=both, anchor_fresh=cA | switch, est_thr=thr), sen

    def _obs(self, T, st, sen):
        """climb_task._obs"""
        g = st.wall.goals
        thr = st.est_thr
        return jp.concatenate([
            sen['q'], sen['dq'], sen['tau'], st.ctrl, sen['acc'], sen['gyro'], sen['mag'],
            sen['f'][0], sen['t'][0], sen['pad'][:1], sen['f'][1], sen['t'][1], sen['pad'][1:],
            (g - thr).reshape(-1),
            self._near_rungs(st.wall, thr[0]), self._near_rungs(st.wall, thr[1]),
            thr[1] - thr[0], thr[0] - thr[1],
            self._confirmed(sen).astype(thr.dtype)]).astype(jp.float32)

    # ---- episode ----------------------------------------------------------------
    def reset(self, T, lesson, key):
        """climb_task.reset, from the lesson's secured hang -> (state, obs)"""
        r0 = T.walls.d0_row[lesson]
        d = jax.tree_util.tree_map(lambda x: x[r0], T.d0)
        key, k = jax.random.split(key)
        return self.start(T, lesson, d, T.ctrl0[r0], T.ngt0[r0], key,
                          SLAM_SIG * jax.random.normal(k, (3,)))

    def start(self, T, lesson, d, ctrl, ngt, key, noise):
        """the rest of climb_task.reset once the fingers are shut: the holds, the
        attitude, the first sense (`noise`: its SLAM error) and the books"""
        w = self.wall(T, lesson)
        sen = self._sensors(d)
        thr = self._throat_true(d)
        holds = [self._nearest_rung(w, thr[s]) for s in (0, 1)]
        con, _ = self._contact(d)
        z2 = jp.zeros(2)
        st = State(d=d, lesson=lesson, ctrl=ctrl, key=key, no_grip_t=ngt,
                   Rf=_triad(sen['acc'], sen['mag']), anchor=jp.array(0),
                   hold_idx=jp.stack([h[1] for h in holds]),
                   hold_pt=jp.stack([h[2] for h in holds]),
                   both_conf=jp.array(False), anchor_fresh=jp.array(True), est_thr=thr,
                   prev_d=z2, prev_live=z2, credit=z2, held=jp.zeros(2, bool),
                   prev_thr=thr, prev_loaded=con > ct.HOLD_LOAD, let_go=jp.zeros(2, bool),
                   grabs=jp.zeros(2, jp.int32), cross_axis=jp.zeros(3), cross_sign=jp.array(1.0),
                   off_t=z2, held_t=z2, slide_rung=jp.full(2, -1), slide_ref=z2, slide_net=z2,
                   n_steps=jp.array(0), wall=w)
        st, sen = self._sense(T, st, d, jp.zeros(3), noise)
        dd, live = self._dists(w, thr)
        _, seat_d, _ = self._seg_point(thr[0], w.rung_c, w.rung_a, w.rung_h)
        axis = w.rung_a[jp.argmin(jp.where(w.rung_valid, seat_d, jp.inf))]
        sign = jp.sign((thr[0] - thr[1]) @ axis)
        st = st._replace(prev_d=dd, prev_live=live, held=self._holding(con, sen),
                         cross_axis=axis, cross_sign=jp.where(sign == 0, 1.0, sign))
        return st, self._obs(T, st, sen)

    @staticmethod
    def _holding(con, sen):
        return (con > ct.HOLD_LOAD) & MjxClimb._gripping(sen)

    def physics(self, T, st, action):
        """ctrl from the action, then FRAME_SKIP substeps with climb_task._attached's
        per-substep books -> (state, flight substeps, gyro sum)"""
        a = jp.clip(action, -1.0, 1.0)
        ctrl = jp.clip(st.ctrl + ct.ACT_SCALE * a, self.lo, self.hi)
        m = self._env_model(st.wall)

        def sub(carry, _):
            d, flight, ngt, w, capped = carry
            d = mjx.step(m, d.replace(ctrl=ctrl))
            con, _ = self._contact(d)
            return (d, flight + ~(con > ct.HOLD_LOAD).any(),
                    jp.where((con > 0.0).any(), 0.0, ngt + self.dt),
                    w + d.sensordata[self.s_gyro],
                    capped + (d._impl.contact.dist <= 0.0).all()), None

        (d, flight, ngt, w, capped), _ = jax.lax.scan(
            sub, (st.d, jp.array(0), st.no_grip_t, jp.zeros(3), jp.array(0)), None,
            length=ct.FRAME_SKIP)
        return st._replace(d=d, ctrl=ctrl, no_grip_t=ngt), flight, w, capped

    def update(self, T, st, action, flight, w_sum, noise):
        """everything climb_task.step does after the physics -> (state, obs, reward,
        terminated, truncated, info). Separate from `physics` so `rl.mjx_check` can
        feed it C MuJoCo's own states and compare it with ClimbTask exactly."""
        a = jp.clip(action, -1.0, 1.0)
        d, lesson, w = st.d, st.lesson, st.wall
        st, sen = self._sense(T, st, d, w_sum, noise)
        touch, on = self._contact(d)
        thr = self._throat_true(d)
        dd, live = self._dists(w, thr)
        free, held = touch <= 0.0, self._holding(touch, sen)
        dp = thr - st.prev_thr
        loaded = touch > ct.HOLD_LOAD
        first_on = jp.argmax(on, axis=1)
        ax = w.rung_a[first_on]
        # climb_task: NET over each loaded run
        run = loaded & st.prev_loaded & on.any(1)
        new = run & (st.slide_rung != first_on)
        ref = jp.where(new, (st.prev_thr * ax).sum(-1), st.slide_ref)
        net = jp.abs((thr * ax).sum(-1) - ref)
        slide = jp.where(run, net - jp.where(new, 0.0, st.slide_net), 0.0).sum()
        slide_rung = jp.where(run, first_on, -1)
        slide_ref, slide_net = jp.where(run, ref, 0.0), jp.where(run, net, 0.0)
        was = st.prev_loaded
        n_steps = st.n_steps + 1
        off_t = st.off_t + free
        held_t = jp.where(free, 0.0, st.held_t + 1)
        grace = ct.DANGLE_GRACE / (ct.FRAME_SKIP * self.dt)
        parked = (free & (off_t > grace) & (off_t > ct.DANGLE_MAX_FRAC * n_steps)).sum()
        stale = ((held_t > ct.STALE_AFTER / (ct.FRAME_SKIP * self.dt)) & loaded[::-1]).sum()
        flight = flight > 0
        let_go = st.let_go | (free & ~flight)
        took = held & ~st.held & let_go & loaded[::-1]
        let_go = let_go & ~took
        grabs = st.grabs + took
        first = took & (grabs == 1)
        approach = st.prev_d - dd
        both = was & loaded
        g = w.goals
        credit = []
        for k in (0, 1):
            o = 1 - k
            u = g[k] - thr[k]
            u = u / jp.maximum(jp.sqrt(u @ u), 1e-9)
            rel = approach[k] - dp[o] @ u
            credit.append(jp.where(both[k], 0.0, jp.where(both[o], rel, jp.minimum(0.0, approach[k]))))
        credit = jp.stack(credit)
        acc = st.credit + credit
        bank = jp.where(took, acc, 0.0).sum()
        acc = jp.where(took, 0.0, acc)
        shared = on[0] & on[1]                     # climb_task._sep_error: one rung only
        sa = w.rung_a[jp.argmax(shared)]
        sep, o = (thr[0] - thr[1]) @ sa, self._order(w, st, sa)
        sep_err = jp.where(shared.any(), jp.maximum(0.0, ct.MIN_SEP - jp.where(o != 0, o * sep, jp.abs(sep))), 0.0)
        effort = jp.maximum(0.0, jp.abs(sen['tau']).sum() - T.tau_idle[w.d0_row])
        fight = jp.maximum(0.0, self._internal(d, sen) - T.fight_idle[w.d0_row])
        r = (ct.K_BANK * bank + ct.K_GRAB * first.sum()
             + ct.K_REACH * (st.prev_live - live).sum()
             - ct.K_DANGLE * parked - ct.K_STALE * stale - ct.K_SEP * sep_err
             - ct.K_EFFORT * effort - ct.K_FIGHT * fight - ct.K_TIME - ct.K_FLIGHT * flight
             - ct.K_SLIDE * slide
             - ct.K_ACT * (a @ a) - ct.K_VEL * (sen['dq'] @ sen['dq']))
        st = st._replace(held=held, prev_d=dd, prev_live=live, prev_thr=thr, prev_loaded=loaded,
                         let_go=let_go, grabs=grabs, credit=acc, off_t=off_t,
                         held_t=held_t, slide_rung=slide_rung, slide_ref=slide_ref,
                         slide_net=slide_net, n_steps=n_steps)
        # climb_task._success
        near = jp.sqrt(((thr - g) ** 2).sum(-1)) <= ct.LATCH_TOL
        on_goal = (on & w.goal_mask[None]).any(1)
        success = (grabs >= ct.MIN_GRABS).all() & (held & on_goal & near).all()
        obs = self._obs(T, st, sen)
        # ClimbingEnv.terminated, in its order, with ClimbTask's addition: an
        # observation that is not finite is a blowup too
        finite = jp.isfinite(d.qpos).all() & jp.isfinite(d.qvel).all() & jp.isfinite(obs).all()
        blowup = ~finite | (jp.abs(d.qvel).max() > QVEL_MAX)
        fell = d.xpos[self.base][2] < FALL_Z
        released = st.no_grip_t >= FREE_FALL_S
        timeout = d.time > w.timeout
        why = jp.where(success, 1, jp.where(blowup, 5, jp.where(fell, 3, jp.where(
            released, 4, jp.where(timeout, 2, 0)))))
        truncated = why == 2
        terminated = (why != 0) & ~truncated
        r = jp.where(why == 5, ct.FAIL_R,       # a blowup is FAIL_R alone, as in ClimbTask
                     r + jp.where(success, ct.SUCCESS_R, jp.where(terminated, ct.FAIL_R, 0.0)))
        info = dict(reason=why, lesson=lesson, family=w.family, probe=w.probe,
                    probe_val=w.probe_val,
                    steps=n_steps, dist=dd.sum(), bank=bank,
                    slide=slide, flight=flight.astype(r.dtype), parked=parked, stale=stale,
                    sep_err=sep_err, effort=effort, fight=fight, first=first.sum(),
                    credit=credit, banked=acc.sum(), d=dd, touch=touch, grabs=grabs,
                    live=live.sum(), off=free.sum(), loaded_start=was)
        return st, jp.where(jp.isfinite(obs), obs, 0.0), r, terminated, truncated, info

    def step(self, T, st, action):
        """one env, one control step"""
        st, flight, w, capped = self.physics(T, st, action)
        key, k = jax.random.split(st.key)
        st, obs, r, term, trunc, info = self.update(T, st._replace(key=key), action, flight, w,
                                                    SLAM_SIG * jax.random.normal(k, (3,)))
        # substeps in which the solver's MAX_CONTACTS were all in use (rl.mjx_scene)
        return st, obs, r, term, trunc, dict(info, capped=capped)


class Batch:
    """N envs on MJX with the auto-reset a VecEnv needs; the curriculum sets the
    lesson draw weights (`probs`, over LESSON_KEYS) without a recompile."""

    def __init__(self, n, seed=0, task=None, bank=0):
        self.task = task or MjxClimb(bank)
        self.n = n
        self.key = jax.random.PRNGKey(seed)
        t = self.task
        self._reset = jax.jit(jax.vmap(t.reset, in_axes=(None, 0, 0)))

        def step(T, st, actions, probs, key):
            st, obs, r, term, trunc, info = jax.vmap(t.step, in_axes=(None, 0, 0))(T, st, actions)
            done = term | trunc
            k1, k2 = jax.random.split(key)
            lessons = jax.random.choice(k1, probs.shape[0], (n,), p=probs)
            fresh, fresh_obs = jax.vmap(t.reset, in_axes=(None, 0, 0))(T, lessons, jax.random.split(k2, n))
            pick = lambda f, s: jp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), f, s)
            st = jax.tree_util.tree_map(pick, fresh, st)
            return st, jp.where(done[:, None], fresh_obs, obs), obs, r, term, trunc, info

        self._step = jax.jit(step)
        self.set_probs({k: 1.0 for k in LESSON_KEYS})

    def set_probs(self, weights):
        """{lesson key: weight}; lessons not named are never drawn, nor is the bank"""
        p = np.array([float(weights.get(k, 0.0)) for k in LESSON_KEYS] + [0.0] * self.task.bank)
        self.probs = jp.asarray(p / p.sum())

    def set_bank(self, layouts):
        """refill the generated rows with `layouts` (rl.walls) and draw only from
        them, uniformly: the family mix and the probes are in the bank itself"""
        L, K = len(LESSON_KEYS), self.task.bank
        assert len(layouts) == K, (len(layouts), K)
        new = self.task.rows(layouts)
        W = self.task.tables.walls
        self.task.tables = self.task.tables._replace(walls=W._replace(
            **{f: getattr(W, f).at[L:].set(jp.asarray(v)) for f, v in new.items()}))
        self.probs = jp.asarray(np.r_[np.zeros(L), np.full(K, 1.0 / K)])

    def reset(self):
        self.key, k1, k2 = jax.random.split(self.key, 3)
        lessons = jax.random.choice(k1, self.probs.shape[0], (self.n,), p=self.probs)
        self.state, obs = self._reset(self.task.tables, lessons, jax.random.split(k2, self.n))
        return obs

    def step(self, actions):
        """-> (next obs, terminal obs, reward, terminated, truncated, info), all jax"""
        self.key, k = jax.random.split(self.key)
        self.state, obs, obs_end, r, term, trunc, info = self._step(
            self.task.tables, self.state, jp.asarray(actions), self.probs, k)
        return obs, obs_end, r, term, trunc, info


def bench(n, steps=20):
    print(f"{n} envs on {jax.devices()[0].device_kind}: building and compiling first")
    b = Batch(n)
    t0 = time.perf_counter()
    b.reset()
    a = np.zeros((n, 17), np.float32)
    jax.block_until_ready(b.step(a))
    print(f"{n} envs: build + compile {time.perf_counter() - t0:.0f} s on {jax.devices()[0]}")
    t0 = time.perf_counter()
    for _ in range(steps):
        out = b.step(a)
    jax.block_until_ready(out)
    dt = time.perf_counter() - t0
    print(f"  {n * steps / dt:,.0f} control-steps/s ({dt / steps * 1000:.0f} ms per batch step, "
          f"{ct.FRAME_SKIP} physics substeps each)")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', type=int, default=64)
    ap.add_argument('--steps', type=int, default=20)
    a = ap.parse_args()
    bench(a.envs, a.steps)
