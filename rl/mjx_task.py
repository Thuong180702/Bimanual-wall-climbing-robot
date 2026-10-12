"""ClimbTask in JAX, thousands of envs at once on MJX: the GPU path of rl.train_ppo.

    python -m rl.mjx_task --envs 1024       # control-steps/s on this machine
    python -m rl.mjx_check                   # is it the same task as rl.climb_task

A line-for-line port of `climb_task.ClimbTask` -- the same 145 observations (the
anchor-rooted estimator, the SLAM-tracked anchor, reach odometry, the nearest
rungs), read through the same sensor, SLAM and map noise (`climb_task.sensed`,
`slam_drift`, `noisy_map`) and the same lidars (`rl.lidar`, the same lines on jax
arrays: the map holds what they have seen), the same reward books, the same success
test and the same terminations -- on `rl.mjx_scene`'s simplified robot. Every constant is read from climb_task,
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
    at build time (`_secure`) and a reset only copies it and senses (the noise is
    drawn fresh).
  * branches are selects: `jp.where` both ways instead of `if`.
  * a finished env resets inside the same jitted step, onto a lesson drawn from the
    pool's weights, which the curriculum moves without a recompile.

Two MJX backends step it (`impl`): JAX, which tests every listed pair every substep
however far apart, and Warp, whose broadphase tests only the near ones. Warp keeps
the batch's contacts in one buffer, each with its world, so its step runs over the
whole batch (`MjxClimb.step_batch`, `_contacts`); it reads a world's own geoms --
the rungs -- from the data, not the model (`_place_static`); and a reset copies the
start into the env's own data (`reset`'s d_like). `rl.mjx_check warp` decides which
one a run trains on: Warp only where it is C MuJoCo's physics by the same checks
JAX passes, and faster.
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
from .estimator import JOINTS, THROAT_PT, MAG_WORLD
from climbing_env import ClimbingEnv, GRIP_BAND, GRIP_SET
from . import climb_task as ct, lidar, rsi
from .lessons import LESSONS, canon_axis
from .mjx_scene import N_RUNG, PARK, lesson_layout, model as scene_model, slots

LESSON_KEYS = sorted(LESSONS)
N_PTS = 256        # route points a wall carries (climb_task.route_table): 180 on the
                   # largest family walls, lesson 3's three full rows
FAR_PT = 1e3       # padding: a point, and a route distance, no hand ever takes
REASONS = ('', 'success', 'timeout', 'fell', 'released', 'blowup')
_ENV = {k: v.default for k, v in inspect.signature(ClimbingEnv).parameters.items()}
FALL_Z, FREE_FALL_S, QVEL_MAX = _ENV['fall_z'], _ENV['free_fall_s'], _ENV['qvel_max']
FINGERS = [JOINTS.index('finger_L'), JOINTS.index('finger_R')]
MINVAL = 1e-15           # mujoco's mjMINVAL, for rotvec's zero-angle case
# MJX's Warp backend (`impl='warp'`, rl.mjx_check warp decides): it keeps all of a world's
# contacts, not the deepest MAX_CONTACTS, in one buffer for the whole batch, and gives
# up any past these -- counted, as `capped`, when it does. C never had more than 17.
WARP_CONTACTS = 48       # contacts a world, on average over the batch
WARP_EFC = 256           # constraint rows a world: four a contact, and the joint limits


# --- small rotation helpers, as climb_task / rl.estimator compute them -------------
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
    """rl.estimator.rotvec: mju_mat2Quat then mju_quat2Vel(dt=1)"""
    q = _mat2quat(R)
    n = jp.sqrt(q[1:] @ q[1:])
    ax = jp.where(n < MINVAL, jp.array([1.0, 0.0, 0.0]), q[1:] / jp.where(n < MINVAL, 1.0, n))
    speed = 2.0 * jp.arctan2(n, q[0])
    return ax * jp.where(speed > jp.pi, speed - 2.0 * jp.pi, speed)


def _quat_z(q):
    """the z axis of a unit quaternion's rotation: a plane's normal"""
    w, x, y, z = q
    return jp.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def _frame(a, b):
    x = a / jp.sqrt(a @ a)
    z = jp.cross(x, b)
    z = z / jp.sqrt(z @ z)
    return jp.stack([x, jp.cross(z, x), z], axis=1)


def _triad(up, mag):
    """rl.estimator.triad"""
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
    map_c: jax.Array           # (N_RUNG, 3) the lidar's map of this wall (climb_task.noisy_map):
    map_a: jax.Array           # centre, axis, half extent
    map_h: jax.Array
    map_goals: jax.Array       # (2, 3) the goals on it
    map_z: jax.Array           # (N_RUNG, 8) its errors, drawn at the reset
    seen: jax.Array            # (N_RUNG, lidar.K) the cells of each rung the lidars have seen
    bias: dict                 # this episode's sensor offsets (climb_task.SEN_BIAS)
    slam_err: jax.Array        # (3,) the SLAM's drifting error
    k: jax.Array               # senses this episode
    buf_true: jax.Array        # (SLAM_LAG, 3) the base, and its estimate, the last
    buf_est: jax.Array         # SLAM_LAG senses: what a late SLAM fix is held against
    # the reward's books (climb_task.reset / step)
    prev_d: jax.Array
    prev_live: jax.Array
    live_att: jax.Array        # (2,) the live cost when a hook last touched a rung
    credit: jax.Array
    held: jax.Array
    prev_thr: jax.Array
    prev_loaded: jax.Array
    let_go: jax.Array
    grabs: jax.Array
    grab_paid: jax.Array       # (2,) climb_task.grab_paid: K_GRAB paid this episode
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
    tau_idle: jax.Array        # (L + S,) climb_task's idle-hang torque, per start
    fight_idle: jax.Array      # (L + S,) ...and its internal force
    d0: mjx.Data               # (L + S,) the secured hang per lesson, then rl.rsi's starts
    ctrl0: jax.Array
    ngt0: jax.Array
    grabs0: jax.Array          # (L + S, 2) holds each hand has taken by then: 0 at a hang
    con0: jax.Array            # (L + S, 2) `_contact` at each start: what Warp's reset reads
    on0: jax.Array             # (L + S, 2, N_RUNG)


class MjxClimb:
    """the task over every lesson, one model. Pure functions of (tables, state) for
    one env; `Batch` vmaps them."""

    def __init__(self, bank=0, impl='jax'):
        self.bank = bank           # generated-wall rows after the lessons' (rl.walls)
        self.impl = impl           # what steps the physics: 'jax', or 'warp' (`step_batch`)
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
        (self.front_n,) = ct.wall_normals(m)            # the front wall; the slot is its own
        self.weight = float(m.body_subtreemass[self.base] * -m.opt.gravity[2])
        self.rig = lidar.Rig(m)
        d = mjx.make_data(self.mx)
        con = d._impl.contact
        assert (con.dim == 3).all(), "every pair is condim 3 (rl.mjx_scene._mix)"
        # the four pyramid rows of each contact; their sum is its normal force
        self.efc4 = con.efc_address[:, None] + np.arange(4)
        self.tables = self._tables()
        self.pairs = (N_RUNG, True)      # what `mx` collides (set_pairs); the tables use all
        # the tables are made, and the estimator's kinematics run, on JAX whatever steps
        self.mxj, self._kd = self.mx, mjx.make_data(self.mx)
        self.public = [f for f in vars(self._kd) if f != '_impl']   # mjx.Data's own fields
        if impl == 'warp':
            import warp
            warp.config.quiet = True     # a line for every kernel it loads, otherwise
            self.mx = mjx.put_model(m, impl='warp')

    def set_pairs(self, n_rung, wall2):
        """step with the collision pairs of the first `n_rung` rung slots and, with
        `wall2`, of the extra wall only (rl.mjx_scene.build): the same geoms, sensors
        and state shapes, the tables untouched -- a wall whose rungs and walls all lie
        inside moves exactly as with every pair (`rl.mjx_check physics` asserts it)"""
        if (n_rung, wall2) == self.pairs:
            return
        mc = scene_model(n_rung, wall2)
        mx = mjx.put_model(mc, impl='jax')
        shape = lambda x: [a.shape for a in jax.tree_util.tree_leaves(mjx.make_data(x))]
        assert shape(mx) == shape(self.mxj), "a pair set that changes the state's shapes"
        self.mx = mx if self.impl == 'jax' else mjx.put_model(mc, impl='warp')
        self.pairs = (n_rung, wall2)

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
        grabs0 = jp.zeros((L, 2), jp.int32)
        ref = rsi.load()
        if ref is not None:
            # rl.rsi's starts after the lessons' hangs (a wall's d0_row picks one): each
            # state forwarded on the wall it was found on, its clock at 0; the effort and
            # fight baselines stay the secured hang's (lesson 1's), as on every wall
            c = ref['climb']                                            # one wall per climb
            sl = {k: v[c] for k, v in self.rows([rsi.climb_wall(ref, i)
                                                  for i in range(len(ref['climb_spacing']))]).items()}
            base = mjx.make_data(self.mx)
            ds = jax.jit(jax.vmap(lambda p, qt, sz, wp, wq, q, v, c: mjx.forward(
                self._slot_model(p, qt, sz, wp, wq), base.replace(qpos=q, qvel=v, ctrl=c, time=jp.zeros(())))))(
                *(jp.asarray(sl[k]) for k in ('slot_pos', 'slot_quat', 'slot_size', 'wall_pos', 'wall_quat')),
                jp.asarray(ref['qpos']), jp.asarray(ref['qvel']), jp.asarray(ref['ctrl']))
            r1, S = LESSON_KEYS.index(1), len(ref['spacing'])
            cat = lambda a, b: jp.concatenate([a, b])
            d0 = jax.tree_util.tree_map(cat, d0, ds)
            ctrl0, ngt0 = cat(ctrl0, jp.asarray(ref['ctrl'])), cat(ngt0, jp.zeros(S))
            tau_idle, fight_idle = cat(tau_idle, jp.full(S, tau_idle[r1])), cat(fight_idle, jp.full(S, fight_idle[r1]))
            grabs0 = cat(grabs0, jp.asarray(ref['grabs'], jp.int32))
        con0, on0 = jax.jit(jax.vmap(self._contact))(d0)
        return Tables(walls=w, tau_idle=tau_idle, fight_idle=fight_idle, d0=d0, ctrl0=ctrl0,
                      ngt0=ngt0, grabs0=grabs0, con0=con0, on0=on0)

    @staticmethod
    def wall(T, row):
        """row `row` of the tables' walls"""
        return jax.tree_util.tree_map(lambda x: x[row], T.walls)

    def _env_model(self, w):
        """the shared model with this wall's rungs and extra wall in the slot geoms"""
        return self._slot_model(w.slot_pos, w.slot_quat, w.slot_size, w.wall_pos, w.wall_quat)

    def _slot_model(self, pos, quat, size, wpos, wquat):
        mx = self.mx
        mx = mx.replace(
            geom_pos=mx.geom_pos.at[self.rung_geoms].set(pos).at[self.wall_geom].set(wpos),
            geom_quat=mx.geom_quat.at[self.rung_geoms].set(quat).at[self.wall_geom].set(wquat),
            geom_size=mx.geom_size.at[self.rung_geoms].set(size))
        if mx.impl == mjx.Impl.WARP:
            # Warp's broadphase culls a pair by each geom's bounds, the parked slot's
            # (a 0.2 m rung) unless they are the rung's: a hook off its middle fell through
            r, h = size[:, 0], size[:, 1]
            box = jp.concatenate([jp.zeros((len(r), 3)), jp.stack([r, r, r + h], -1)], -1)
            mx = mx.replace(geom_rbound=mx.geom_rbound.at[self.rung_geoms].set(r + h),
                            geom_aabb=mx.geom_aabb.at[self.rung_geoms].set(
                                box.reshape((len(r),) + mx.geom_aabb.shape[1:])))
        return mx

    def _place_static(self, d, w):
        """the wall's rungs and extra wall where Warp reads them: in the data. Its
        kinematics leaves the world's own geoms where `make_data` put them -- parked,
        and no hook ever touched a rung -- where JAX's places them from the model"""
        from mujoco.mjx._src.math import quat_to_mat
        g = np.r_[self.rung_geoms, self.wall_geom]
        quat = jp.concatenate([w.slot_quat, w.wall_quat[None]])
        return d.replace(geom_xpos=d.geom_xpos.at[g].set(jp.concatenate([w.slot_pos, w.wall_pos[None]])),
                         geom_xmat=d.geom_xmat.at[g].set(jax.vmap(quat_to_mat)(quat)))

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

    def _noisy_map(self, w, z, ext):
        """climb_task.noisy_map on a wall's rows and the spans seen of them (`ext`,
        rl.lidar.extent) -> (centres, axes, half extents, goals)"""
        a = w.rung_a
        R = jax.vmap(_rotm)(ct.MAP_ROT_SIG * z[:, 3:6])
        a2 = jp.einsum('rij,rj->ri', R, a)
        big = jp.take_along_axis(a2, jp.argmax(jp.abs(a2), axis=1)[:, None], axis=1)
        a2 = jp.where(big < 0.0, -a2, a2)                                   # canon_axis
        n = ct.MAP_SIG * z[:, :3]
        lo, hi = ct.MAP_END_SIG * z[:, 6], ct.MAP_END_SIG * z[:, 7]
        rigid = w.rung_c + (n - (n * a).sum(-1, keepdims=True) * a) + (0.5 * (lo + hi))[:, None] * a2
        u0, u1 = ext
        c2 = rigid + (0.5 * (u0 + u1))[:, None] * jp.einsum('rij,rj->ri', R, a)
        h2 = jp.maximum(0.5 * (u1 - u0) + 0.5 * (hi - lo), 0.0)
        own = [jp.argmin(jp.where(w.rung_valid, self._seg_point(g, w.rung_c, a, w.rung_h)[1], jp.inf))
               for g in w.goals]
        g2 = jp.stack([rigid[i] + R[i] @ (g - w.rung_c[i]) for i, g in zip(own, w.goals)])
        return c2, a2, h2, g2

    def _look(self, st, d, seen):
        """climb_task._look: this frame's cells join `seen`, and the map is redrawn
        from it -> (seen, centres, half extents)"""
        w = st.wall
        seen = seen | lidar.visible(jp, self.rig, d.site_xpos, d.site_xmat, w.rung_c, w.rung_a,
                                    w.rung_h, w.slot_size[:, 0], w.rung_valid)
        mc, _, mh, _ = self._noisy_map(w, st.map_z, lidar.extent(jp, seen, w.rung_h))
        return seen, mc, mh

    @staticmethod
    def _map(st):
        """this episode's wall as the robot's map has it: only the rungs it has seen"""
        return st.wall._replace(rung_c=st.map_c, rung_a=st.map_a, rung_h=st.map_h, goals=st.map_goals,
                                rung_valid=st.wall.rung_valid & st.seen.any(-1))

    @staticmethod
    def _normals(key, shapes):
        """standard normals in `shapes` (a dict, nested: climb_task.STEP_NOISE, episode_noise)"""
        leaves, tree = jax.tree_util.tree_flatten(shapes, is_leaf=lambda x: isinstance(x, tuple))
        return tree.unflatten([jax.random.normal(k, s) for k, s in zip(jax.random.split(key, len(leaves)), leaves)])

    def _rung_pt(self, w, p):
        """climb_task._route's point: the nearest rung's, inside its seats"""
        q, dist, _ = self._seg_point(p, w.rung_c, w.rung_a, jp.maximum(w.rung_h - ct.SEAT_IN, 0.0))
        return q[jp.argmin(jp.where(w.rung_valid, dist, jp.inf))]

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
        q = jp.concatenate([jp.zeros(3), jp.array([1.0, 0.0, 0.0, 0.0]), q])
        dd = (mjx.kinematics(self.mx, d.replace(qpos=q)) if self.impl == 'jax' else
              mjx.kinematics(self.mxj, self._kd.replace(qpos=q)))     # JAX's, not a Warp call
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

    def _sense(self, T, st, d, w_sum, z):
        """climb_task._sense -> (state, what the sensors would read if perfect -- the
        reward's --, what they do read). `w_sum`: the gyro summed over the step's
        substeps; `z`: this sense's draws (climb_task.STEP_NOISE)."""
        fix = ct.scan_due(st.k)                    # a lidar frame, and the SLAM fix it carries
        seen, mc, mh = self._look(st, d, st.seen)
        st = st._replace(seen=jp.where(fix, seen, st.seen), map_c=jp.where(fix, mc, st.map_c),
                         map_h=jp.where(fix, mh, st.map_h))
        true = self._sensors(d)
        sen = ct.sensed(true, z['sen'], st.bias)
        dt = ct.FRAME_SKIP * self.dt
        first = st.k == 0
        Rf = jp.where(first & (st.wall.d0_row < len(LESSON_KEYS)),     # an rl.rsi start's: `start`
                      _triad(sen['acc'], sen['mag']), st.Rf)
        Rf = Rf @ _rotm(ct.sensed_rates(w_sum, z['w'], st.bias['gyro']) * self.dt)
        R_tri = _triad(sen['acc'], sen['mag'])
        Rf = Rf @ _rotm((dt / ct.AHRS_TAU) * _rotvec(Rf.T @ R_tri))
        tree = self._tree(d, sen['q'])
        a, w = st.anchor, self._map(st)
        hold_idx, hold_pt = st.hold_idx, st.hold_pt
        _, base_est = self._place(tree, Rf, a, hold_pt[a])
        base = d.xpos[self.base]
        buf_true = jp.where(first, jp.tile(base, (ct.SLAM_LAG, 1)), st.buf_true)
        buf_est = jp.where(first, jp.tile(base_est, (ct.SLAM_LAG, 1)), st.buf_est)
        slam_err = jp.where(fix, ct.slam_drift(st.slam_err, z['slam']), st.slam_err)
        ax = w.rung_a[hold_idx[a]]
        dp = buf_true[0] + slam_err - buf_est[0]
        # climb_task.along_seen: a wall facing along the rung -- the front one, or the
        # extra one when this wall has it (a parked slot faces up, far below)
        wq = st.wall
        along = (jp.abs(jp.asarray(self.front_n) @ ax) > ct.FACING) | (
            (wq.wall_pos[2] > -PARK / 2) & (jp.abs(_quat_z(wq.wall_quat) @ ax) > ct.FACING))
        hold_pt = hold_pt.at[a].add(jp.where(fix & along, ct.SLAM_GAIN * (dp @ ax), 0.0) * ax)
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
        _, base_est = self._place(tree, Rf, anchor, hold_pt[anchor])
        return st._replace(Rf=Rf, anchor=anchor, hold_idx=hold_idx, hold_pt=hold_pt,
                           both_conf=both, anchor_fresh=cA | switch, est_thr=thr,
                           slam_err=slam_err, k=st.k + 1,
                           buf_true=jp.concatenate([buf_true[1:], base[None]]),
                           buf_est=jp.concatenate([buf_est[1:], base_est[None]])), true, sen

    def _obs(self, T, st, sen, m=None):
        """climb_task._obs, from what the sensors read (`sen`) and the map (`m`: another one,
        rl.rsi.teach's)"""
        m = self._map(st) if m is None else m
        g = m.goals
        thr = st.est_thr
        return jp.concatenate([
            sen['q'], sen['dq'], sen['tau'], st.ctrl, sen['acc'], sen['gyro'], sen['mag'],
            sen['f'][0], sen['t'][0], sen['pad'][:1], sen['f'][1], sen['t'][1], sen['pad'][1:],
            (g - thr).reshape(-1),
            self._near_rungs(m, thr[0]), self._near_rungs(m, thr[1]),
            thr[1] - thr[0], thr[0] - thr[1],
            self._confirmed(sen).astype(thr.dtype)]).astype(jp.float32)

    # ---- episode ----------------------------------------------------------------
    def reset(self, T, lesson, key, d_like=None):
        """climb_task.reset, from the lesson's secured hang -> (state, obs). `d_like`
        (Warp): the env's own data, which takes the start's state -- its fields, not
        Warp's, which the next step recomputes and one buffer of which the batch shares"""
        r0 = T.walls.d0_row[lesson]
        if d_like is None:
            d, con = jax.tree_util.tree_map(lambda x: x[r0], T.d0), None
        else:
            d = d_like.replace(**{f: getattr(T.d0, f)[r0] for f in self.public})
            con = (T.con0[r0], T.on0[r0])
        key, k1, k2 = jax.random.split(key, 3)
        return self.start(T, lesson, d, T.ctrl0[r0], T.ngt0[r0], key,
                          self._normals(k1, ct.episode_noise(N_RUNG)), self._normals(k2, ct.STEP_NOISE),
                          con)

    def start(self, T, lesson, d, ctrl, ngt, key, ep, z, con=None):
        """the rest of climb_task.reset once the fingers are shut: the map, the holds,
        the first sense and the books. `ep`: the episode's draws
        (climb_task.episode_noise(N_RUNG)); `z`: the first sense's; `con`: `_contact`
        at `d`, when it is not to be read from it (Warp)"""
        w = self.wall(T, lesson)
        thr = self._throat_true(d)
        seen = lidar.visible(jp, self.rig, d.site_xpos, d.site_xmat, w.rung_c, w.rung_a, w.rung_h,
                             w.slot_size[:, 0], w.rung_valid)       # what it sees from the hang
        on, idx = zip(*[self._nearest_rung(w, thr[s])[:2] for s in (0, 1)])
        idx = jp.stack(idx)                                                  # climb_task._hung
        # ...and an rl.rsi start, partway up a climb, has the rungs its hooks are on end to end,
        # as a climb that got there has seen them on its way: from its own first frame alone
        # the map can lack the very rung it holds -- family 6's climb, both hooks at the start
        # row's end behind the arms, had them placed 0.35-0.60 m along it, clipped to the
        # span seen (2026-10-11). Not a second lidar.visible from the hang: that doubled the
        # step's largest reduction (4 lidars x rungs x cells x occluders, ~1.8e9 elements
        # at 4096 envs), and gpu9's GPU compiles of it stalled 16-18+ min (2026-10-12)
        held = ((jp.arange(N_RUNG)[:, None] == idx[None]) & jp.stack(on)[None]).any(-1) & w.rung_valid
        seen = jp.where(w.d0_row >= len(LESSON_KEYS), seen | held[:, None], seen)
        mc, ma, mh, mg = self._noisy_map(w, ep['map'], lidar.extent(jp, seen, w.rung_h))
        u = jp.clip(((thr - mc[idx]) * ma[idx]).sum(-1), -mh[idx], mh[idx])
        con, on = self._contact(d) if con is None else con
        z2 = jp.zeros(2)
        # the anchor: the left hand, as ClimbTask's, unless only the right carries load
        # (an rl.rsi start with the left in the air)
        anchor = jp.where(~(con[0] > ct.HOLD_LOAD) & (con[1] > ct.HOLD_LOAD), 1, 0)
        # the attitude filter: from a hang, `_sense` starts it from one accelerometer and
        # magnetometer reading, as ClimbTask does. An rl.rsi start is mid-swing, where that
        # reading is the swing's as much as gravity's: 16-37 deg off around the catch, the
        # free hook seen 0.1-0.3 m from where it is, a different way at every row -- gpu5
        # (2026-10-09) finished from 4.80 s on that and 0/12 there with the true attitude
        # or after the climb itself from the hang. So it starts where a climb's filter
        # has got to by then: the torso's attitude (2-3 deg off along the 0.20 m climb)
        rsi_start = w.d0_row >= len(LESSON_KEYS)
        st = State(d=d, lesson=lesson, ctrl=ctrl, key=key, no_grip_t=ngt,
                   Rf=jp.where(rsi_start, d.xmat[self.torso], jp.eye(3)), anchor=anchor, hold_idx=idx,
                   hold_pt=mc[idx] + u[:, None] * ma[idx],
                   both_conf=jp.array(False), anchor_fresh=jp.array(True), est_thr=thr,
                   map_c=mc, map_a=ma, map_h=mh, map_goals=mg, map_z=ep['map'], seen=seen,
                   bias=ep['bias'],
                   slam_err=ct.SLAM_SIG * ep['slam'], k=jp.array(0),
                   buf_true=jp.zeros((ct.SLAM_LAG, 3)), buf_est=jp.zeros((ct.SLAM_LAG, 3)),
                   prev_d=z2, prev_live=z2, live_att=z2, credit=z2, held=jp.zeros(2, bool),
                   prev_thr=thr, prev_loaded=con > ct.HOLD_LOAD, let_go=jp.zeros(2, bool),
                   grabs=T.grabs0[w.d0_row], grab_paid=T.grabs0[w.d0_row] > 0,
                   cross_axis=jp.zeros(3), cross_sign=jp.array(1.0),
                   off_t=z2, held_t=z2, slide_rung=jp.full(2, -1), slide_ref=z2, slide_net=z2,
                   n_steps=jp.array(0), wall=w)
        st, true, sen = self._sense(T, st, d, jp.zeros(3), z)
        dd, live = self._dists(w, thr)
        _, seat_d, _ = self._seg_point(thr[0], w.rung_c, w.rung_a, w.rung_h)
        axis = w.rung_a[jp.argmin(jp.where(w.rung_valid, seat_d, jp.inf))]
        sign = jp.sign((thr[0] - thr[1]) @ axis)
        st = st._replace(prev_d=dd, prev_live=live, live_att=live,
                         held=self._holding(w, con, on, thr, true),
                         cross_axis=axis, cross_sign=jp.where(sign == 0, 1.0, sign))
        return st, self._obs(T, st, sen)

    @staticmethod
    def _holding(w, con, on, thr, sen):
        """climb_task._holding: loaded, gripping, and on a SEAT of a rung it touches --
        the throat at most SEAT_TOL past the last full one"""
        u = jp.abs(((thr[:, None, :] - w.rung_c[None]) * w.rung_a[None]).sum(-1))   # (2, N_RUNG)
        seat = (on & (u <= w.rung_h[None] - ct.SEAT_IN + ct.SEAT_TOL)).any(-1)
        return (con > ct.HOLD_LOAD) & MjxClimb._gripping(sen) & seat

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

    def update(self, T, st, action, flight, w_sum, z, con=None):
        """everything climb_task.step does after the physics -> (state, obs, reward,
        terminated, truncated, info). Separate from `physics` so `rl.mjx_check` can
        feed it C MuJoCo's own states and compare it with ClimbTask exactly. `con`:
        `_contact` at the state, when it is not to be read from it (Warp)"""
        a = jp.clip(action, -1.0, 1.0)
        d, lesson, w = st.d, st.lesson, st.wall
        st, true, sen = self._sense(T, st, d, w_sum, z)
        touch, on = self._contact(d) if con is None else con
        thr = self._throat_true(d)
        dd, live = self._dists(w, thr)
        live_att = jp.where((touch > 0.0).any(), live, st.live_att)
        free, held = touch <= 0.0, self._holding(w, touch, on, thr, true)
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
        approach = st.prev_d - dd
        both = was & loaded
        g = w.goals
        credit = []
        for k in (0, 1):
            o = 1 - k
            u = g[k] - self._rung_pt(w, thr[k])       # climb_task: the partner's slide
            u = u / jp.maximum(jp.sqrt(u @ u), 1e-9)
            rel = approach[k] - (dp[o] @ ax[o]) * (ax[o] @ u)
            credit.append(jp.where(both[k], 0.0, jp.where(both[o], rel, jp.minimum(0.0, approach[k]))))
        credit = jp.stack(credit)
        acc = st.credit + credit
        first = took & ~st.grab_paid & (acc >= ct.GRAB_MIN)     # climb_task: K_GRAB's hold
        bank = jp.where(took, acc, 0.0).sum()
        acc = jp.where(took, 0.0, acc)
        shared = on[0] & on[1]                     # climb_task._sep_error: one rung only
        sa = w.rung_a[jp.argmax(shared)]
        sep, o = (thr[0] - thr[1]) @ sa, self._order(w, st, sa)
        sep_err = jp.where(shared.any(), jp.maximum(0.0, ct.MIN_SEP - jp.where(o != 0, o * sep, jp.abs(sep))), 0.0)
        effort = jp.maximum(0.0, jp.abs(true['tau']).sum() - T.tau_idle[w.d0_row])
        fight = jp.maximum(0.0, self._internal(d, true) - T.fight_idle[w.d0_row])
        r = (ct.K_BANK * bank + ct.K_GRAB * first.sum()
             + ct.K_REACH * (st.prev_live - live).sum()
             - ct.K_DANGLE * parked - ct.K_STALE * stale - ct.K_SEP * sep_err
             - ct.K_EFFORT * effort - ct.K_FIGHT * fight - ct.K_TIME - ct.K_FLIGHT * flight
             - ct.K_SLIDE * slide
             - ct.K_ACT * (a @ a) - ct.K_VEL * (true['dq'] @ true['dq']))
        st = st._replace(held=held, prev_d=dd, prev_live=live, live_att=live_att,
                         prev_thr=thr, prev_loaded=loaded,
                         let_go=let_go, grabs=grabs, grab_paid=st.grab_paid | first,
                         credit=acc, off_t=off_t,
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
        # a fall (fell, released) gets back the live term charged since a hook last
        # touched a rung, as in ClimbTask
        refund = jp.where((why == 3) | (why == 4), ct.K_REACH * (live - live_att).sum(), 0.0)
        r = jp.where(why == 5, ct.FAIL_R,       # a blowup is FAIL_R alone, as in ClimbTask
                     r + jp.where(success, ct.SUCCESS_R, jp.where(terminated, ct.FAIL_R, 0.0)) + refund)
        info = dict(reason=why, lesson=lesson, family=w.family, probe=w.probe,
                    probe_val=w.probe_val, row=w.d0_row,     # the start: rl.rsi's at L + j
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
                                                    self._normals(k, ct.STEP_NOISE))
        # substeps in which the solver's MAX_CONTACTS were all in use (rl.mjx_scene)
        return st, obs, r, term, trunc, dict(info, capped=capped)

    # ---- Warp: the whole batch at once ------------------------------------------
    def _contacts(self, d):
        """`_contact` for every env from Warp's one contact buffer (each contact
        carries its world) -> (force (n, 2), touched (n, 2, N_RUNG), substeps over
        WARP_CONTACTS or WARP_EFC (n,): contacts or rows Warp gave up)"""
        c, n = d._impl, d.qpos.shape[0]
        na, nj = c.contact__dist.shape[0], c.efc__force.shape[1]
        live = jp.arange(na) < c.nacon[0]
        wid = jp.where(live, c.contact__worldid, n)                 # the rest: segment n
        g1, g2 = c.contact__geom[:, 0], c.contact__geom[:, 1]
        adr = c.contact__efc_address                                 # (na, 4): its pyramid rows
        fn = jp.where(adr >= 0, c.efc__force[jp.minimum(wid, n - 1)[:, None], jp.maximum(adr, 0)], 0.0).sum(-1)
        rung1 = (g1[:, None] == self.rung_geoms[None]).any(-1)
        rung2 = (g2[:, None] == self.rung_geoms[None]).any(-1)
        on = jp.stack([(((g1[:, None] == h[None]).any(-1) & rung2) | ((g2[:, None] == h[None]).any(-1) & rung1))
                       for h in self.hook_geoms])                    # (2, na)
        force = jax.ops.segment_sum(jp.where(on, jp.abs(fn)[None], 0.0).T, wid, n + 1)[:n]
        slot = (g1[:, None] == self.rung_geoms[None]) | (g2[:, None] == self.rung_geoms[None])
        hit = on[:, :, None] & slot[None] & (c.contact__dist < 0.0)[None, :, None]   # (2, na, N_RUNG)
        touched = jax.ops.segment_max(hit.transpose(1, 0, 2).astype(jp.int32), wid, n + 1)[:n] > 0
        over = (c.nacon[0] > na) | (c.nefc > nj)
        return force, touched, over.astype(jp.int32)

    def step_batch(self, T, st, actions):
        """`step` for every env at once, as Warp needs it: its contacts are read across
        the batch (`_contacts`), so the substeps' books are kept here and `update` is
        handed the contacts. The same books as `physics`, the same update after."""
        n = actions.shape[0]
        a = jp.clip(actions, -1.0, 1.0)
        ctrl = jp.clip(st.ctrl + ct.ACT_SCALE * a, self.lo, self.hi)
        d = jax.vmap(self._place_static)(st.d, st.wall).replace(ctrl=ctrl)
        # each env's model made inside the vmap, as `physics` does: made outside, every
        # field of it would come out one a world, the meshes too
        one = jax.vmap(lambda w, d: mjx.step(self._env_model(w), d))

        def sub(carry, _):
            d, flight, ngt, w, capped = carry
            d = one(st.wall, d)
            con, _, over = self._contacts(d)
            return (d, flight + ~(con > ct.HOLD_LOAD).any(-1),
                    jp.where((con > 0.0).any(-1), 0.0, ngt + self.dt),
                    w + d.sensordata[:, self.s_gyro], capped + over), None

        z = jp.zeros(n, jp.int32)
        (d, flight, ngt, w, capped), _ = jax.lax.scan(
            sub, (d, z, st.no_grip_t, jp.zeros((n, 3)), z), None, length=ct.FRAME_SKIP)
        st = st._replace(d=d, ctrl=ctrl, no_grip_t=ngt)
        con, on, _ = self._contacts(d)

        def upd(st, action, flight, w, con, on):
            key, k = jax.random.split(st.key)
            return self.update(T, st._replace(key=key), action, flight, w,
                               self._normals(k, ct.STEP_NOISE), (con, on))

        st, obs, r, term, trunc, info = jax.vmap(upd)(st, actions, flight, w, con, on)
        return st, obs, r, term, trunc, dict(info, capped=capped)


class Batch:
    """N envs on MJX with the auto-reset a VecEnv needs; the curriculum sets the
    lesson draw weights (`probs`, over LESSON_KEYS) without a recompile."""

    def __init__(self, n, seed=0, task=None, bank=0, fit=False, impl='jax'):
        self.task = task or MjxClimb(bank, impl)
        self.n = n
        self.fit = fit
        self.key = jax.random.PRNGKey(seed)
        if self.task.impl == 'warp':     # every env's data, whose state each reset replaces
            self.dw = jax.vmap(lambda _: mjx.make_data(
                self.task.mc, impl='warp', naconmax=n * WARP_CONTACTS, njmax=WARP_EFC))(jp.arange(n))
        self._compile()
        self.set_probs({k: 1.0 for k in LESSON_KEYS})

    def _compile(self):
        t, n = self.task, self.n
        warp = t.impl == 'warp'
        reset = jax.vmap(t.reset, in_axes=(None, 0, 0, 0) if warp else (None, 0, 0))
        self._reset = jax.jit(reset)

        def step(T, st, actions, probs, key):
            if warp:
                st, obs, r, term, trunc, info = t.step_batch(T, st, actions)
            else:
                st, obs, r, term, trunc, info = jax.vmap(t.step, in_axes=(None, 0, 0))(T, st, actions)
            done = term | trunc
            k1, k2 = jax.random.split(key)
            lessons = jax.random.choice(k1, probs.shape[0], (n,), p=probs)
            keys = jax.random.split(k2, n)
            fresh, fresh_obs = reset(T, lessons, keys, st.d) if warp else reset(T, lessons, keys)
            pick = lambda f, s: jp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), f, s)
            if warp:      # the data's own fields only: Warp's are shared by the batch
                d = st.d.replace(**{f: pick(getattr(fresh.d, f), getattr(st.d, f)) for f in t.public})
                st = jax.tree_util.tree_map(pick, fresh._replace(d=None), st._replace(d=None))._replace(d=d)
            else:
                st = jax.tree_util.tree_map(pick, fresh, st)
            return st, jp.where(done[:, None], fresh_obs, obs), obs, r, term, trunc, info

        self._step = jax.jit(step)

    def _fit(self):
        """with `fit`, collide only the rung slots and walls that a wall which can be
        drawn (probs > 0) or is being climbed (a live env's) fills: MJX tests every
        pair every substep however far, and while families 1..3 train, rung slots
        3..5 and the extra wall are 42 % of a substep's memory traffic, which is what
        a batch step waits on (rl.mjx_scene.build). Grows before such a wall can be
        drawn, shrinks once none is live; each change recompiles the step."""
        if not self.fit:
            return
        W = self.task.tables.walls
        rows = np.nonzero(np.asarray(self.probs) > 0)[0]
        valid, wall_z = [np.asarray(W.rung_valid)[rows]], [np.asarray(W.wall_pos)[rows, 2]]
        if hasattr(self, 'state'):
            valid.append(np.asarray(self.state.wall.rung_valid))
            wall_z.append(np.asarray(self.state.wall.wall_pos)[:, 2])
        need = (max(1, max(int(v.sum(-1).max()) for v in valid)),
                any(bool((z > -PARK / 2).any()) for z in wall_z))
        if need != self.task.pairs:
            print(f"[mjx] collision pairs: rung slots {need[0]} of {N_RUNG}, extra wall "
                  f"{'yes' if need[1] else 'no'} (was {self.task.pairs[0]}, "
                  f"{'yes' if self.task.pairs[1] else 'no'}); recompiling the step", flush=True)
            self.task.set_pairs(*need)
            self._compile()

    def set_probs(self, weights):
        """{lesson key: weight}; lessons not named are never drawn, nor is the bank"""
        p = np.array([float(weights.get(k, 0.0)) for k in LESSON_KEYS] + [0.0] * self.task.bank)
        self.probs = jp.asarray(p / p.sum())
        self._fit()

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
        self._fit()

    def reset(self):
        self.key, k1, k2 = jax.random.split(self.key, 3)
        lessons = jax.random.choice(k1, self.probs.shape[0], (self.n,), p=self.probs)
        args = (self.dw,) if self.task.impl == 'warp' else ()
        self.state, obs = self._reset(self.task.tables, lessons, jax.random.split(k2, self.n), *args)
        self._fit()
        return obs

    def step(self, actions):
        """-> (next obs, terminal obs, reward, terminated, truncated, info), all jax"""
        self.key, k = jax.random.split(self.key)
        self.state, obs, obs_end, r, term, trunc, info = self._step(
            self.task.tables, self.state, jp.asarray(actions), self.probs, k)
        return obs, obs_end, r, term, trunc, info


def bench(n, steps=20, impl='jax'):
    """control-steps/s of one batch: colliding every pair (every lesson), then fitted to
    lessons 1..3 -- the collision pairs families 1..3 need, what training runs early"""
    print(f"{n} envs on {jax.devices()[0].device_kind}, {impl}: building and compiling first")
    b = Batch(n, impl=impl)
    a = np.zeros((n, 17), np.float32)
    for fit in (False, True):
        if fit:
            b.fit = True
            b.set_probs({k: 1.0 for k in (1, 2, 3)})
        t0 = time.perf_counter()
        b.reset()                      # fitted: the envs on lessons 1..3, the pairs cut
        jax.block_until_ready(b.step(a))
        r, w = b.task.pairs
        label = f"{r} rung slots{' and the extra wall' if w else ''} collided"
        print(f"{n} envs, {label}: build + compile {time.perf_counter() - t0:.0f} s on {jax.devices()[0]}")
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
    ap.add_argument('--impl', default='jax', choices=['jax', 'warp'])
    a = ap.parse_args()
    bench(a.envs, a.steps, a.impl)
