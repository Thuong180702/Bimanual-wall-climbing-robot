"""Anchor-rooted state estimate and relative-Jacobian differential IK.

Nothing here reads the simulator's ground truth. The estimate is built from
    encoders  -> relative kinematics of the whole tree (exact, it is a rigid chain)
    IMU       -> world attitude, via TRIAD on gravity + magnetic field
    hold map  -> where the anchored hand is in the world
which is exactly what the hardware would have.

The IK works on *relative* poses, free hand w.r.t. anchored hand:

    J_rel = (J_free - J_anchor)[:, joints between them]

Differencing kills the floating-base columns, so those joint columns are the exact
derivative of the relative pose -- no base pose needed, no welding, no fixed-base model.
"""
import heapq
import numpy as np, mujoco

# 17 drives, in actuator order (== qpos[7:] order)
JOINTS = ['waist_yaw',
          'shoulder_L_abduction', 'shoulder_L_flexion', 'shoulder_L_rotation', 'elbow_L',
          'forearm_L_roll', 'wrist_L_flexion', 'wrist_L_deviation', 'finger_L',
          'shoulder_R_abduction', 'shoulder_R_flexion', 'shoulder_R_rotation', 'elbow_R',
          'forearm_R_roll', 'wrist_R_flexion', 'wrist_R_deviation', 'finger_R']
IDX = {n: i for i, n in enumerate(JOINTS)}
WAIST = [0]
ARM = {s: [IDX['shoulder_%s_abduction' % s], IDX['shoulder_%s_flexion' % s],
           IDX['shoulder_%s_rotation' % s], IDX['elbow_%s' % s],
           IDX['forearm_%s_roll' % s], IDX['wrist_%s_flexion' % s],
           IDX['wrist_%s_deviation' % s]] for s in 'LR'}
FINGER = {s: IDX['finger_%s' % s] for s in 'LR'}


# measured: where a hanging rung actually settles inside the throat (README table)
THROAT_PT = np.array([0.023, 0.0, -0.085])
MAG_WORLD = np.array([0.0, -0.5, 0.0])      # scene.xml magnetic field
WALL_MARGIN = 0.03        # how far a link origin must stay on the robot's side of a
                          # wall face
W_ROT = 0.08              # metres per radian: how much orientation counts against
                          # position. A latched hook sits ~5 deg off the nominal
                          # grasp frame in yaw and holds perfectly well, so this
                          # is a preference, not a constraint.
DOWN = np.array([0.0, 0.0, -1.0])


# --- small helpers ----------------------------------------------------------
def wall_planes(model):
    """[(point on the inner face, unit normal pointing at the robot)] for each wall.

    Which side the robot lives on is read off the rungs: they stand proud of the face
    they belong to, so the wall centre -> rung direction is the inward normal.
    """
    walls, rungs = [], []
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if nm.startswith('wall_'):
            walls.append((model.geom_pos[g].copy(), model.geom_size[g].copy()))
        elif nm.startswith('rung_'):
            rungs.append(model.geom_pos[g].copy())
    out = []
    for c, sz in walls:
        thin = int(np.argmin(sz))
        near = min(rungs, key=lambda r: np.linalg.norm(r - c))
        n = np.zeros(3)
        n[thin] = np.sign(near[thin] - c[thin]) or 1.0
        out.append((c + n * sz[thin], n))
    return out


def _frame(a, b):
    """orthonormal frame from two non-parallel vectors, a is kept exactly"""
    x = a / np.linalg.norm(a)
    z = np.cross(x, b)
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def triad(up_body, mag_body):
    """world<-body rotation from an accelerometer and a magnetometer. No filter, no state."""
    return _frame(np.array([0., 0., 1.]), MAG_WORLD) @ _frame(up_body, mag_body).T


def rotvec(R):
    """rotation matrix -> rotation vector (axis * angle)"""
    q, v = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q, R.flatten())
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def grasp_frame(normal):
    """World orientation a hook must have to latch on a wall with this outward normal:
    local z points down, local x into the wall, so local y ends up along the rung."""
    x = -np.asarray(normal, float)
    z = DOWN
    return np.column_stack([x, np.cross(z, x), z])


def balance_pivot(c_rel, axis):
    """Angle to swing about `axis` so the centre of mass hangs directly below the hold.

    A latched hook turns freely on its rung, so the body is a pendulum on that axis and
    its swing is set by gravity, not by the joints. Rotating rigidly about the axis rotates
    the COM offset exactly, so the equilibrium angle is closed form -- no extra unknown and
    no iteration. Evaluating a hypothetical posture without it is meaningless: pinned at
    the nominal grasp frame instead, the hang keyframe itself reads as 0.315 m off plumb
    and driven into the wall.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    v = c_rel - (c_rel @ a) * a                     # COM offset, in the swing plane
    t = np.array([0., 0., -1.])
    t = t - (t @ a) * a                             # straight down, in the swing plane
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(t) < 1e-6:
        return 0.0
    return float(np.arctan2(np.cross(v, t) @ a, v @ t))


def rot_about(axis, angle):
    q, R = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def skew(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def dls(J, err, lam=0.08):
    """damped least squares, one solve"""
    return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(err)), err)


def pinv(J, lam=0.08):
    return J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(J.shape[0]))


# --- estimator --------------------------------------------------------------
class Estimator:
    """Scratch copy of the model driven by encoders only.

    Relative poses inside a scratch tree are correct whatever the base pose is, so we run
    it with the free joint at the origin, work out where the tree must be, and then re-run
    it placed there -- after which xpos / xmat are world estimates and collision queries
    mean something.
    """

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.hook = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hook_' + s)
                     for s in 'LR'}
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_upper')
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.walls = wall_planes(model)
        self.links = [b for b in range(model.nbody)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or '')
                      not in ('world', '')]
        self.anchor = 'L'

    def update(self, q, imu, anchor_side, anchor_pos, anchor_R=None, R_torso=None):
        """q: 17 encoder angles. imu: dict(accel=, mag=). anchor_pos: world point on the
        rung axis where the anchored hook's throat sits.

        Attitude comes from the IMU when you are estimating the state you are in, and from
        `anchor_R` -- the anchored hook's world orientation -- when you are evaluating a
        configuration you are not in yet. The latched hook's orientation is fixed by the
        rung, so it is the right invariant to hang a hypothetical posture off; reusing the
        IMU reading for a different joint configuration puts the model somewhere physically
        impossible (in practice: the torso a hundred millimetres inside the wall).

        """
        d = self.d
        d.qpos[:3] = 0.0
        d.qpos[3:7] = (1., 0., 0., 0.)
        d.qpos[7:] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.m, d)

        if anchor_R is not None:
            R_off = anchor_R @ d.xmat[self.hook[anchor_side]].reshape(3, 3).T
        elif R_torso is not None:
            # a filtered world<-torso attitude (the RL task's gyro complementary
            # filter); the bare triad below is what the classical controller uses
            R_off = R_torso @ d.xmat[self.torso].reshape(3, 3).T
        else:
            R_off = triad(imu['accel'], imu['mag']) @ d.xmat[self.torso].reshape(3, 3).T
        p_off = np.asarray(anchor_pos) - R_off @ self._throat(anchor_side)

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R_off @ d.xmat[self.base].reshape(3, 3)).flatten())
        d.qpos[:3] = p_off + R_off @ d.xpos[self.base]
        d.qpos[3:7] = quat
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        self.anchor = anchor_side

        return self

    # -- world estimates ----------------------------------------------------
    def _throat(self, side):
        b = self.hook[side]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ THROAT_PT

    def throat(self, side):
        return self._throat(side)

    def hook_R(self, side):
        return self.d.xmat[self.hook[side]].reshape(3, 3)

    def body_pos(self, body_id):
        return self.d.xpos[body_id].copy()

    def com(self):
        """whole-robot centre of mass (base_link's subtree is the whole robot)"""
        return self.d.subtree_com[self.base].copy()

    # -- clearance ----------------------------------------------------------
    def hang(self, q, imu, anchor_side, anchor_pos, anchor_R, axis):
        """Place a hypothetical posture the way gravity would hang it, and say how far
        the hook had to turn on its rung to get there."""
        self.update(q, imu, anchor_side, anchor_pos, anchor_R)
        th = balance_pivot(self.com() - np.asarray(anchor_pos), axis)
        self.update(q, imu, anchor_side, anchor_pos, rot_about(axis, th) @ anchor_R)
        return th

    def wall_violations(self, margin=WALL_MARGIN):
        """[(body, face point, inward normal, how far past)] for links behind a wall"""
        out = []
        for b in self.links:
            p = self.d.xpos[b]
            for face, nrm in self.walls:
                dist = float(nrm @ (p - face))
                if dist < margin:
                    out.append((b, face, nrm, margin - dist))
        out.sort(key=lambda v: -v[3])
        return out

    def clear(self, q, imu, anchor_side, anchor_pos, anchor_R=None, margin=WALL_MARGIN):
        """True if this configuration is not driving through anything.

        Self-collision uses real geometry. Walls do not: the body rides against the wall
        by design (base_link's front face sits exactly on it) and the estimate is only
        good to a centimetre or two out at the base, so mesh penetration there says more
        about estimator error than about the posture. A link *origin* past the wall face,
        on the other hand, is unambiguous.
        """
        self.update(q, imu, anchor_side, anchor_pos, anchor_R)
        mujoco.mj_collision(self.m, self.d)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            n = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
                 for g in (c.geom1, c.geom2)]
            if any(x.startswith(('rung_', 'wall_')) for x in n):
                continue
            if c.dist < -0.002:
                return False
        return not self.wall_violations(margin=margin)

    # -- jacobians ----------------------------------------------------------
    def _jac_point(self, body, world_pt):
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jp, jr, world_pt, body)
        return jp, jr

    def _rel(self, body, pt_world, side_anchor, joints):
        """Motion of a point as seen from the anchored hand, which is treated as pinned
        in both position and orientation.

        Differencing the two Jacobians removes the floating base. The skew term then
        removes the anchor frame's own rotation -- without it a waist yaw looks like it
        swings the free hand, when in truth it just spins the body under a pinned hand.
        """
        dofs = [6 + j for j in joints]
        jpx, jrx = self._jac_point(body, pt_world)
        a_pt = self._throat(side_anchor)
        jpa, jra = self._jac_point(self.hook[side_anchor], a_pt)
        d = pt_world - a_pt
        return ((jpx - jpa)[:, dofs] + skew(d) @ jra[:, dofs], (jrx - jra)[:, dofs])

    def jac_rel(self, side_free, side_anchor, joints):
        """d(free throat pose relative to the anchored throat) / d(joints), world axes"""
        return self._rel(self.hook[side_free], self._throat(side_free),
                         side_anchor, joints)

    def jac_body_rel(self, body_id, side_anchor, joints):
        """same, for an arbitrary body (used to haul the torso around)"""
        return self._rel(body_id, self.d.xpos[body_id], side_anchor, joints)

    def jac_com_rel(self, side_anchor, joints):
        """d(COM relative to the anchored throat) / d(joints), world axes"""
        dofs = [6 + j for j in joints]
        jc = np.zeros((3, self.m.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, jc, self.base)
        a_pt = self._throat(side_anchor)
        jpa, jra = self._jac_point(self.hook[side_anchor], a_pt)
        return (jc - jpa)[:, dofs] + skew(self.com() - a_pt) @ jra[:, dofs]


# --- closed-loop IK steps ---------------------------------------------------
def ik_body(est, body_id, anchors, joints, p_goal, gain=0.6, max_step=0.06,
            q=None, q_ref=None, k_null=0.4):
    """Move a body (the torso, usually) while 1-2 hands stay put. With two anchors the
    second hand is pinned as an extra constraint, which is what keeps a bridge between
    two walls from tearing itself apart.

    `q_ref` steers the redundancy, and is off by default. Biasing a pull-up towards bent
    elbows was tried, to get the torso higher than the 0.45 m below the hands where the
    least-norm answer saturates with both elbows dead straight. It does bend them -- and
    it pulls the second hook off its rung every time, because bending both elbows with
    both hands pinned a fixed distance apart is not something the geometry allows.
    """
    a0 = anchors[0]
    Jp, _ = est.jac_body_rel(body_id, a0, joints)
    J, err = Jp, p_goal - est.body_pos(body_id)
    for a in anchors[1:]:
        Jp2, _ = est.jac_rel(a, a0, joints)
        J = np.vstack([J, Jp2])
        err = np.concatenate([err, np.zeros(3)])
    if q_ref is not None and q is not None:
        # Steer the redundancy with extra rows, not with a nullspace projector. The
        # projector needs an almost undamped pseudo-inverse to be a projector at all:
        # share the task solve's damping and the bias leaks into the task rows and pulls
        # the second hand 0.29 m off its rung; drop the damping and the solve goes
        # unstable. Extra rows are just a weighted least squares and do neither.
        J = np.vstack([J, k_null * np.eye(len(joints))])
        err = np.concatenate([err, k_null * np.array([q_ref[j] - q[j] for j in joints])])
    dq = gain * dls(J, err)
    n = np.max(np.abs(dq))
    return dq * (max_step / n) if n > max_step else dq


# --- planning a move the robot can hold all the way through --------------------
# Climbing needs a big reconfiguration: every posture that both reaches a hold on the row
# above and can actually be held is at least 4.4 rad from the hanging one, and the
# straight line to it falls off the wall at any speed. So the path has to be planned
# inside the set of postures the robot can hang in -- which is 4.4 % of joint space,
# dense enough to sample directly.

class Hanging:
    """The set of postures the robot can hang in, from one hook on a known rung."""

    def __init__(self, est, imu, anchor, anchor_pos, anchor_R, axis, q_ref,
                 pivot_band=0.30):
        # 0.30 rad, not the 0.70 it was. Turning the body on its hook walks the rung along
        # the throat, and the geometry gives -17.8..+3.2 deg before it leaves -- 21 deg
        # in total against the 40 this band used to allow. Measured through the trailing
        # hand's descent: the rung crept 15 -> 33 mm along the throat over waypoints the
        # planner had all passed, and then left. Widening this band was tried once and
        # changed nothing, which is not evidence that narrowing it does nothing.
        self.est, self.imu, self.anchor = est, imu, anchor
        self.pos, self.R, self.axis = anchor_pos, anchor_R, axis
        self.band = pivot_band
        self.th0 = est.hang(q_ref, imu, anchor, anchor_pos, anchor_R, axis)
        mujoco.mj_collision(est.m, est.d)
        self.max_walls = len(est.wall_violations())   # the hang keyframe's own count
        # ...and its own deepest self-contact, for the same reason. The posture the robot
        # is actually in has to pass this test or no planner can plan out of it: after the
        # leading hand's row change two links rest 3.6 mm into each other, and a flat
        # -2 mm rule then called the robot's own current posture infeasible. Both planners
        # returned at their first line, which reads as "no route" and is not.
        self.max_pen = min(-0.002, self._pen())

    def _pen(self):
        """deepest self-penetration in the current scratch pose; rungs and walls excluded"""
        est, worst = self.est, 0.0
        for i in range(est.d.ncon):
            c = est.d.contact[i]
            n = [mujoco.mj_id2name(est.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
                 for g in (c.geom1, c.geom2)]
            if any(x.startswith(('rung_', 'wall_')) for x in n):
                continue
            worst = min(worst, float(c.dist))
        return worst

    def __call__(self, q):
        est = self.est
        th = est.hang(q, self.imu, self.anchor, self.pos, self.R, self.axis)
        if abs(th - self.th0) > self.band:            # the hook would turn off its rung
            return False
        mujoco.mj_collision(est.m, est.d)
        if len(est.wall_violations()) > self.max_walls:
            return False
        return self._pen() >= self.max_pen - 1e-9


def solve_hanging_pose(est, ok, side, anchor, anchor_pos, anchor_R, axis, imu, joints,
                       lo, hi, q_seed, target, R_goal, tries=1500, w_rot=0.10, seed=0):
    """A posture that puts the free hook on `target` and that the robot can hold there.

    Every candidate is evaluated hung at its own balance angle and then passed through
    `ok`, which is the same test the planner uses. Filtering on reach alone is not enough:
    the first "solution" found that way turned 74 deg when hung and buried the robot in
    the wall, and a later one had the humerus 21 mm inside base_link.
    """
    rng = np.random.default_rng(seed)
    best = None
    for t in range(tries):
        q = np.array(q_seed, float)
        if t:
            for j in joints:
                q[j] = rng.uniform(lo[j], hi[j])
        for _ in range(500):
            est.hang(q, imu, anchor, anchor_pos, anchor_R, axis)
            ep = target - est.throat(side)
            er = rotvec(R_goal @ est.hook_R(side).T)
            a = np.asarray(axis, float) / np.linalg.norm(axis)
            er = er - (er @ a) * a
            Jp, Jr = est.jac_rel(side, anchor, joints)
            dq = dls(np.vstack([Jp, w_rot * Jr]),
                     np.concatenate([ep, w_rot * er]), lam=0.05)
            n = np.max(np.abs(dq))
            if n > 0.05:
                dq *= 0.05 / n
            for i, j in enumerate(joints):
                q[j] = np.clip(q[j] + dq[i], lo[j], hi[j])
        est.hang(q, imu, anchor, anchor_pos, anchor_R, axis)
        er = rotvec(R_goal @ est.hook_R(side).T)
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        if (np.linalg.norm(target - est.throat(side)) < 0.012
                and np.linalg.norm(er - (er @ a) * a) < 0.25 and ok(q)):
            d = np.linalg.norm([q[j] - q_seed[j] for j in joints])
            if best is None or d < best[0]:
                best = (d, q.copy())
    return None if best is None else best[1]


def _edge_ok(ok, q_a, q_b, joints, res=0.10):
    d = np.linalg.norm([q_b[j] - q_a[j] for j in joints])
    for u in np.linspace(0.0, 1.0, max(2, int(d / res) + 1))[1:]:
        q = np.array(q_a, float)
        for j in joints:
            q[j] = q_a[j] + u * (q_b[j] - q_a[j])
        if not ok(q):
            return False
    return True


def roadmap_hanging(ok, q_a, q_b, joints, lo, hi, side=None, n=2500, k=12, seed=0,
                    corridor=0.5, sigma=0.35, leg=0.6):
    """PRM through the postures the robot can hang in. Waypoints or None.

    An RRT grows two trees by extending from whatever it samples, so it needs the straight
    segment from a tree to a new sample to be feasible. At 1.5 % density in 15 dimensions
    those segments almost all fail, and the trees stay where they started -- which is what
    the second hand's descent does: the posture solver finds the goal, the RRT then reports
    no route through 250000 iterations. A roadmap spends the same sampling once and lets
    every node be tried against twelve neighbours, so a corridor built out of many short
    hops gets found even when no long one exists.

    Edges are validated lazily, only when the search relaxes them, and priced by how far
    the *hand* travels rather than the joints: priced by joint distance the routes came out
    valid and wild. Long legs are then split, which is free -- every edge was checked
    feasible along its whole length, so points interpolated inside one need no re-checking.
    """
    if not (ok(q_a) and ok(q_b)):
        return None
    if _edge_ok(ok, q_a, q_b, joints):
        return [q_b]

    rng = np.random.default_rng(seed)
    hand = (lambda: ok.est.throat(side).copy()) if side else (lambda: np.zeros(3))
    Q, P = [np.array(q_a, float)], [hand()]
    ok(q_b)
    Q.append(np.array(q_b, float))
    P.append(hand())
    while len(Q) < n:
        q = np.array(q_a, float)
        if rng.random() < corridor:
            u = rng.random()
            for j in joints:
                q[j] = np.clip(q_a[j] + u * (q_b[j] - q_a[j]) + rng.normal(0.0, sigma),
                               lo[j], hi[j])
        else:
            for j in joints:
                q[j] = rng.uniform(lo[j], hi[j])
        if ok(q):
            Q.append(q)
            P.append(hand())

    X = np.array([[q[j] for j in joints] for q in Q])
    d2 = (X * X).sum(1)
    D = np.sqrt(np.maximum(d2[:, None] + d2[None, :] - 2.0 * (X @ X.T), 0.0))
    nbr = np.argpartition(D, k + 1, axis=1)[:, :k + 1]
    Pw = np.array(P)
    price = ((lambda i, j: float(np.linalg.norm(Pw[i] - Pw[j]))) if side
             else (lambda i, j: float(D[i, j])))

    # lazy Dijkstra: an edge is only checked along its length when it is first useful
    best, came, bad = {0: 0.0}, {0: -1}, set()
    openq = [(0.0, 0)]
    while openq:
        c, i = heapq.heappop(openq)
        if i == 1:
            break
        if c > best.get(i, 1e18):
            continue
        for j in nbr[i]:
            j = int(j)
            if j == i or (min(i, j), max(i, j)) in bad:
                continue
            nc = c + price(i, j)
            if nc >= best.get(j, 1e18):
                continue
            if not _edge_ok(ok, Q[i], Q[j], joints):
                bad.add((min(i, j), max(i, j)))
                continue
            best[j], came[j] = nc, i
            heapq.heappush(openq, (nc, j))
    if 1 not in came:
        return None

    idx, i = [], 1
    while i >= 0:
        idx.append(i)
        i = came[i]
    path = [Q[i] for i in idx[::-1]]

    for _ in range(60):                               # shortcut
        if len(path) < 3:
            break
        a = int(rng.integers(0, len(path) - 2))
        b = int(rng.integers(a + 2, len(path)))
        if _edge_ok(ok, path[a], path[b], joints):
            path = path[:a + 1] + path[b:]

    out = []                                          # then split the long legs
    for u, v in zip(path, path[1:]):
        d = np.linalg.norm([v[j] - u[j] for j in joints])
        for t in np.linspace(0.0, 1.0, max(2, int(d / leg) + 1))[1:]:
            w = np.array(u, float)
            for j in joints:
                w[j] = u[j] + t * (v[j] - u[j])
            out.append(w)
    return out


def plan_hanging(ok, q_a, q_b, joints, lo, hi, step=0.45, iters=250000, seed=0,
                 corridor=0.5, sigma=0.35):
    """Bidirectional RRT through the postures the robot can hang in. Waypoints or None.

    The budget is large because the set is thin: a uniform sample of the 15 joints is a
    holdable posture 1.5 % of the time, so an iteration usually ends at the `ok(s)`
    guard without touching a tree. At 6000 iterations that is ~90 usable samples and it
    finds nothing; at 250000 it connects the 4.6 rad climb below in 3 s. Sampling runs
    at ~8000/s, so a hopeless call still gives up in half a minute.

    Half the samples come from a corridor around the straight line between the two
    postures rather than from the whole box. Both ends are feasible and the set is thin,
    so that is where a path can be, while uniform draws spend themselves in corners of
    the box no path goes near. The second hand's descent found nothing in 250000 uniform
    iterations.
    """
    if not (ok(q_a) and ok(q_b)):
        return None
    if _edge_ok(ok, q_a, q_b, joints):
        return [q_b]
    rng = np.random.default_rng(seed)
    trees = [[(np.array(q_a, float), -1)], [(np.array(q_b, float), -1)]]

    def extend(tree, target):
        i = min(range(len(tree)),
                key=lambda k: np.linalg.norm([tree[k][0][j] - target[j] for j in joints]))
        d = np.array([target[j] - tree[i][0][j] for j in joints])
        n = np.linalg.norm(d)
        q = np.array(tree[i][0], float)
        for c, j in enumerate(joints):
            q[j] += d[c] * (1.0 if n <= step else step / n)
        if not ok(q) or not _edge_ok(ok, tree[i][0], q, joints):
            return None
        tree.append((q, i))
        return len(tree) - 1

    for it in range(iters):
        s = np.array(q_a, float)
        if rng.random() < corridor:
            u = rng.random()
            for j in joints:
                s[j] = np.clip(q_a[j] + u * (q_b[j] - q_a[j]) + rng.normal(0.0, sigma),
                               lo[j], hi[j])
        else:
            for j in joints:
                s[j] = rng.uniform(lo[j], hi[j])
        if not ok(s):
            continue
        A, B = trees[it % 2], trees[(it + 1) % 2]
        ia = extend(A, s)
        if ia is None:
            continue
        ib = extend(B, A[ia][0])
        if ib is None:
            continue
        if _edge_ok(ok, A[ia][0], B[ib][0], joints):
            pa, k = [], ia
            while k >= 0:
                pa.append(A[k][0]); k = A[k][1]
            pb, k = [], ib
            while k >= 0:
                pb.append(B[k][0]); k = B[k][1]
            path = pa[::-1] + pb if A is trees[0] else pb[::-1] + pa
            for _ in range(60):                       # shortcut
                if len(path) < 3:
                    break
                i = rng.integers(0, len(path) - 2)
                k = rng.integers(i + 2, len(path))
                if _edge_ok(ok, path[i], path[k], joints):
                    path = path[:i + 1] + path[k:]
            return path[1:]
    return None
