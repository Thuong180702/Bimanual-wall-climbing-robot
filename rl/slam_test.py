"""The SLAM test: the four Mid-360 simulated point by point, FAST-LIO2 building the map
and the pose from their points, and the task -- the policy -- reading that instead of
the error models it trains on (climb_task.noisy_map, slam_drift, rl.lidar).

    docker build -t climber-slam slam/              # once: FAST-LIO2 on ROS Noetic
    python -m rl.slam_test --lesson 1               # the arms swinging: the SLAM's and
                                                    # the map's errors against the model's
    python -m rl.slam_test --lesson 2 --run NAME    # a checkpoint climbing on it

What the lidars give, as the Livox driver would:
  * each Mid-360 fires 200 000 points/s along Livox's non-repetitive pattern -- a real
    sensor's, recorded (`mid360-real-centr.csv`, fratopa/Mid360_simulation_plugin,
    800 000 points: 4 s, 37.8-97.2 deg off the axis) -- and MuJoCo casts them
    (mj_multiRay) into the walls, the rungs and the robot's collision shapes, 4 ms of
    points at a time as the robot moves. A frame is 100 ms of them, bent by the motion
    the way a real frame is, and FAST-LIO undistorts it with the IMU as it would.
  * each return off by the datasheet: 3 cm (1 sigma) of range at 0.2 m, 2 cm at 10 m,
    the beam 0.15 deg from where the point is reported; nothing inside 0.1 m or past 40 m.
  * the four merged into one cloud in the IMU's frame (the CAD's extrinsics; `line` says
    which lidar) with the torso's IMU at 250 Hz, read through the training noise model
    (climb_task.SEN_SIG / SEN_BIAS).
  * ponytail: the robot's own returns are dropped by what they hit -- a perfect
    self-filter. A real one masks the robot's shape at its encoder angles, padded, and
    loses some rung points next to the hands with it.
  * the lesson scenes have walls and rungs and nothing else: no posts holding the rungs,
    no floor, no clutter. Along a smooth rung on a flat wall the only thing that says
    where the robot is are the rungs' ends.

FAST-LIO2 (hku-mars/FAST_LIO @7cc4175) runs in Docker (slam/), lock-step with the
simulation: a frame is handed over as it ends and the physics waits for its answer.
The pose lands SLAM_LAG control steps after the frame ends, as in training, and is held
against the estimate of that moment. The map frame is fixed where the robot was hung
(on hardware you know where you hung it); everything after that is FAST-LIO's.

The rung map from FAST-LIO's registered points (`rungs`): the walls are the big planes;
a rung is what stands 2-14 cm proud of a wall more often than the wall's own noise
stands behind it -- that noise is symmetric, so the band behind the wall is the
background to subtract -- connected and long; its axis is the points' mean depth less
2r/pi, a lidar seeing the near half of a 15 mm rung.
"""
import argparse, hashlib, io, os, socket, struct, subprocess, time, urllib.request
import numpy as np, mujoco
from scipy import ndimage
from . import climb_task as ct, lidar
from .climb_task import ClimbTask

HERE = os.path.dirname(os.path.abspath(__file__))
SLAM_DIR = os.path.abspath(os.path.join(HERE, '..', 'slam'))
PATTERN = os.path.join(SLAM_DIR, 'vendor', 'mid360-real-centr.csv')
PATTERN_URL = ('https://raw.githubusercontent.com/fratopa/Mid360_simulation_plugin/'
               'aae8ee3a0f16e2f21c480da9b858dd702a2cd13a/livox_laser_simulation/scan_mode/'
               'mid360-real-centr.csv')
PATTERN_SHA = 'aa486c66092af45eab59fccb1119421470ba01981434b75d04367dd374576553'
NAMES = ('lidar_head_front', 'lidar_head_back', 'lidar_chest', 'lidar_back')
RATE = 200_000           # points/s, each lidar
SLICE = 2                # physics steps a slice of points (and an IMU reading) spans: 4 ms
FRAME = 50               # physics steps a frame spans: 100 ms, 10 Hz
RANGE = (0.10, 40.0)     # m
RANGE_SIG = ((0.2, 0.03), (10.0, 0.02))   # m (1 sigma) at m, from the datasheet
ANGLE_SIG = np.radians(0.15)
R_RUNG = 0.015           # m: the rungs the hook is made for
STANDOFF = (0.04, 0.14)  # m a rung's axis stands off its wall: the hook's finger has to fit
                         # behind it (the lessons: 0.06)
PREROLL = 2.0            # s the robot hangs still after it is secured, before the episode:
                         # the SLAM has run a while before a climb starts, on hardware too
PT = np.dtype([('t', '<u4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
               ('r', 'u1'), ('tag', 'u1'), ('line', 'u1')])     # livox CustomPoint, 19 bytes
IMAGE, PORT = 'climber-slam', 5555


def pattern():
    """Livox's scan directions in the sensor frame (z its axis), (800 000, 3)"""
    npy = PATTERN[:-4] + '.npy'
    if not os.path.exists(npy):
        if not os.path.exists(PATTERN):
            os.makedirs(os.path.dirname(PATTERN), exist_ok=True)
            print(f"[slam] fetching the Mid-360 scan pattern from {PATTERN_URL}", flush=True)
            urllib.request.urlretrieve(PATTERN_URL, PATTERN)
        sha = hashlib.sha256(open(PATTERN, 'rb').read()).hexdigest()
        assert sha == PATTERN_SHA, f"{PATTERN}: sha256 {sha}, expected {PATTERN_SHA}"
        a = np.loadtxt(PATTERN, delimiter=',', skiprows=1)
        az, zen = np.radians(a[:, 1]), np.radians(a[:, 2])
        np.save(npy, np.stack([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)], 1))
    return np.load(npy)


# --- the container -------------------------------------------------------------------
def _send(conn, **kw):
    f = io.BytesIO()
    np.savez(f, **{k: np.asarray(v) for k, v in kw.items()})
    conn.sendall(struct.pack('<Q', f.tell()) + f.getvalue())


def _recv(conn):
    n = struct.unpack('<Q', _exact(conn, 8))[0]
    return dict(np.load(io.BytesIO(_exact(conn, n))))


def _exact(conn, n):
    b = bytearray()
    while len(b) < n:
        c = conn.recv(min(1 << 20, n - len(b)))
        if not c:
            raise ConnectionError('the SLAM container closed the connection')
        b += c
    return bytes(b)


class Slam:
    """FAST-LIO2 in its container, through slam/bridge.py; started if it is not running"""

    def __init__(self, port=PORT):
        self.port = port
        self.conn = self._connect()

    def _connect(self):
        for i in range(120):
            try:
                c = socket.create_connection(('127.0.0.1', self.port), timeout=120)
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return c
            except OSError:
                if i == 0:
                    self._start()
                time.sleep(1.0)
        raise RuntimeError(f"no SLAM bridge on port {self.port}: `docker logs {IMAGE}`")

    def _start(self):
        up = subprocess.run(['docker', 'ps', '-q', '-f', f'name=^{IMAGE}$'], capture_output=True, text=True)
        if up.stdout.strip():
            return
        print(f"[slam] starting the {IMAGE} container", flush=True)
        subprocess.run(['docker', 'run', '-d', '--rm', '--name', IMAGE, '-p', f'{self.port}:{PORT}',
                        '-v', f'{SLAM_DIR}:/sim', IMAGE, 'bash', '/sim/start.sh'], check=True,
                       capture_output=True)

    def call(self, **kw):
        _send(self.conn, **kw)
        return _recv(self.conn)


# --- the map ---------------------------------------------------------------------------
class VoxelMap:
    """the registered points, as the mean of each `size` voxel and how many fell in it"""

    def __init__(self, size=0.01):
        self.size = size
        self.key = np.zeros(0, np.int64)
        self.n = np.zeros(0)
        self.sum = np.zeros((0, 3))

    def add(self, P):
        i = np.floor(P / self.size).astype(np.int64) + (1 << 20)
        k = (i[:, 0] << 42) | (i[:, 1] << 21) | i[:, 2]
        key, inv = np.unique(np.r_[self.key, k], return_inverse=True)
        n = np.bincount(inv, np.r_[self.n, np.ones(len(k))], len(key))
        S = np.r_[self.sum, P]
        s = np.stack([np.bincount(inv, S[:, i], len(key)) for i in range(3)], 1)
        self.key, self.n, self.sum = key, n, s

    def points(self):
        return self.sum / self.n[:, None], self.n


def planes(P, w, rng, band=0.04, frac=0.05, span=0.6, tries=300, sample=6000):
    """the walls: up to three planes, each holding at least `frac` of ALL the points
    within `band` and `span` m across both ways (RANSAC on a sample, refined by weighted
    least squares on all) -> [(point, unit normal)]. Of all, not of what is left: the
    rungs of a wall lie in one plane too, a thin one. A voxel counts for at most 3 points:
    the wall right in front of a lidar gets thousands a voxel, and any plane through that
    patch, at any tilt, outweighed the rest of the wall"""
    w = np.minimum(w, 3.0)
    out, left, total = [], np.ones(len(P), bool), w.sum()
    for _ in range(3):
        idx = np.nonzero(left)[0]
        if len(idx) < 100:
            break
        sub = rng.choice(idx, min(sample, len(idx)), replace=False)
        best = None
        for _ in range(tries):
            p = P[rng.choice(sub, 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            score = w[sub][np.abs((P[sub] - p[0]) @ n) < band].sum()
            if best is None or score > best[0]:
                best = (score, p[0], n)
        if best is None or best[0] < frac * total * len(sub) / len(P):
            break
        _, c, n = best
        for _ in range(3):                                   # refine on every point
            inl = idx[np.abs((P[idx] - c) @ n) < band]
            ww = w[inl] / w[inl].sum()
            c = ww @ P[inl]
            n = np.linalg.svd((P[inl] - c) * np.sqrt(ww)[:, None], full_matrices=False)[2][2]
        inl = idx[np.abs((P[idx] - c) @ n) < band]
        left[inl] = False
        # the band either side of a wall is that wall's own noise -- 2-3 cm a point along
        # the ray piles up +-10 cm off a wall hit at a slant -- not another wall
        if any(abs(n @ n0) > 0.95 and abs((c - c0) @ n0) < 0.2 for c0, n0 in out):
            continue
        ext = np.linalg.svd(P[inl] - c, full_matrices=False)[2][:2]
        if (np.ptp((P[inl] - c) @ ext.T, axis=0) >= span).all():
            out.append((c, n))
    return out


def rungs(P, w, walls, toward, cell=0.02, near=(0.02, 0.14), sig=5.0, kern=(9, 3), tilt=5.0):
    """rung segments [(centre, unit axis, half length)] standing proud of `walls`, on the
    side of `toward` (where the robot is). The holds are horizontal rungs, and the IMU
    says which way is up: per wall, on a `cell` grid in its plane (u across, v up), the
    points 2-14 cm in front of it less the mirror band behind it -- the wall's own noise,
    which is symmetric -- is what stands proud of it. That excess is judged summed over
    `kern` cells, 18 cm along and 6 cm up (a far rung is a few points a cell, under its
    cell's noise; a range error along a slanting ray spreads it 2-3 cm up and down), at
    `sig` sigma: a wall is ~10^5 cells, and at 3 sigma the noise itself drew rungs. A
    connected run of it at least 10 cm long, within `tilt` deg of level, is a rung; its
    ends are where the plain excess along it falls away, its axis the points' mean depth
    less 2r/pi. The other walls' own points are not in front of this one: in a corner the
    side wall stands proud of the front one all the way up."""
    out = []
    K = np.ones(kern)
    for i, (c, n) in enumerate(walls):
        mine = np.ones(len(P), bool)
        for k, (c2, n2) in enumerate(walls):
            if k != i:
                mine &= np.abs((P - c2) @ n2) > 0.14
        if (toward - c) @ n < 0:
            n = -n
        u = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 0.5:                        # a floor or a ceiling: no rungs
            continue
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        d = P - c
        s, a, b = d @ n, d @ u, d @ v
        F = (s > near[0]) & (s < near[1]) & mine
        B = (s < -near[0]) & (s > -near[1]) & mine
        if F.sum() < 20:
            continue
        lo = np.array([a[F].min(), b[F].min()]) - 5 * cell
        ia = np.floor((a - lo[0]) / cell).astype(int)
        ib = np.floor((b - lo[1]) / cell).astype(int)
        shape = (ia[F].max() + 6, ib[F].max() + 6)
        ok = (ia >= 0) & (ib >= 0) & (ia < shape[0]) & (ib < shape[1])
        hf, hb = np.zeros(shape), np.zeros(shape)
        np.add.at(hf, (ia[F & ok], ib[F & ok]), w[F & ok])
        np.add.at(hb, (ia[B & ok], ib[B & ok]), w[B & ok])
        ex = hf - hb
        E = ndimage.convolve(ex, K, mode='constant')
        V = ndimage.convolve(hf + hb, K, mode='constant')
        lab, nl = ndimage.label(E > sig * np.sqrt(np.maximum(V, 1.0)), structure=np.ones((3, 3)))
        for k in range(1, nl + 1):
            ci = np.argwhere(lab == k)
            wt = E[ci[:, 0], ci[:, 1]]
            # the row it runs along (a slight tilt allowed): b on a, weighted
            A = np.c_[ci[:, 0], np.ones(len(ci))] * np.sqrt(wt)[:, None]
            slope, icpt = np.linalg.lstsq(A, ci[:, 1] * np.sqrt(wt), rcond=None)[0]
            j = np.arange(ci[:, 0].min() - kern[0], ci[:, 0].max() + kern[0] + 1)
            j = j[(j >= 0) & (j < shape[0])]
            row = slope * j + icpt
            # the plain excess along it, its row and the one either side
            prof = np.array([ex[jj, max(0, int(round(r)) - 1):int(round(r)) + 2].sum()
                             for jj, r in zip(j, row)])
            run = np.convolve(prof, np.ones(3), 'same')
            core = run[(j >= ci[:, 0].min()) & (j <= ci[:, 0].max())]
            on = run > 0.25 * max(np.median(core), 1e-9)
            ang = np.arctan(slope)                            # cells are square
            if abs(np.degrees(ang)) > tilt:
                continue
            e = np.array([np.cos(ang), np.sin(ang)])
            # each stretch where it stands proud is a rung: the 18 cm sum bridges a gap
            # between two, the plain excess does not
            edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            for r0, r1 in zip(edges[::2], edges[1::2]):
                j0, j1 = j[r0], j[r1 - 1] + 1                 # cell edges
                if (j1 - j0) * cell < 0.10:
                    continue
                p0 = lo + cell * np.array([j0, slope * j0 + icpt + 0.5])
                p1 = lo + cell * np.array([j1, slope * j1 + icpt + 0.5])
                cp = np.c_[a, b] - p0
                along, across = cp @ e, cp @ np.array([-e[1], e[0]])
                inl = ok & (np.abs(across) <= 1.5 * cell) & (along >= 0) & (along <= np.linalg.norm(p1 - p0))
                inf, inb = F & inl, B & inl
                wn = max(w[inf].sum() - w[inb].sum(), 1e-9)
                depth = (w[inf] @ s[inf] + w[inb] @ s[inb]) / wn
                mid = (p0 + p1) / 2
                seen = c + n * depth + mid[0] * u + mid[1] * v   # the middle of what was seen
                # a lidar sees the near half of a rung: its axis is 2r/pi behind the points,
                # away from where they were seen from
                look = toward - seen
                look -= (look @ (e[0] * u + e[1] * v)) * (e[0] * u + e[1] * v)
                axis = e[0] * u + e[1] * v
                centre = seen - 2.0 * R_RUNG / np.pi * look / np.linalg.norm(look)
                # a hold stands off its wall far enough for a hook's finger to pass behind
                # it. Nearer, it is the wall itself seen at a slant: a range error moves a
                # point along its ray, and on a ray that grazes the wall that is mostly
                # sideways, so the near side's noise piles up toward the lidar and the far
                # side's away from it -- the one place the band behind is not the background
                if not STANDOFF[0] <= (centre - c) @ n <= STANDOFF[1]:
                    continue
                out.append((centre, ct.canon_axis(axis / np.linalg.norm(axis)), np.linalg.norm(p1 - p0) / 2))
    return merge(out)


def merge(segs, gap=0.0):
    """one segment for pieces of one rung that overlap -- two runs of a rung either side
    of a hand both grow to its full length -- and never across a gap"""
    segs = list(segs)
    done = False
    while not done:
        done = True
        for i in range(len(segs)):
            for k in range(i + 1, len(segs)):
                (c1, a1, h1), (c2, a2, h2) = segs[i], segs[k]
                off = c2 - c1
                t = off @ a1
                if (abs(a1 @ a2) > np.cos(np.radians(5)) and np.linalg.norm(off - t * a1) < 0.02
                        and abs(t) < h1 + h2 - gap):
                    lo_, hi_ = min(-h1, t - h2), max(h1, t + h2)
                    a = a1 if a1 @ a2 > 0 else a1
                    segs[i] = (c1 + a1 * (lo_ + hi_) / 2, a, (hi_ - lo_) / 2)
                    del segs[k]
                    done = False
                    break
            if not done:
                break
    return segs


def with_posts(lesson, r=0.01):
    """the lesson's scene with a post from each rung end back to its wall, for the lidars
    only (contype 0) -> the path of that scene"""
    from .lessons import OUT, lesson_tag, xml_path
    from .mjx_scene import lesson_layout
    spec = mujoco.MjSpec.from_file(xml_path(lesson))
    lay = lesson_layout(lesson)
    for nm, c, a, h, _ in lay['rungs']:
        for sgn in (-1, 1):
            p = c + sgn * h * a
            back = [((p - w) @ n, n) for _, w, n in lay['walls'] if (p - w) @ n > 0]
            dist, n = min(back, key=lambda x: x[0])
            spec.worldbody.add_geom(name=f'post_{nm}_{sgn}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    fromto=np.r_[p, p - n * dist], size=[r, 0, 0],
                                    contype=0, conaffinity=0, rgba=[0.34, 0.36, 0.40, 1])
    path = os.path.join(OUT, f'slam_{lesson_tag(lesson)}_posts.xml')
    open(path, 'w').write(spec.to_xml())
    return path


# --- the task on it --------------------------------------------------------------------
class SlamTask(ClimbTask):
    """ClimbTask with the map and the SLAM fixes from FAST-LIO on simulated Mid-360s.
    Everything else -- encoders, IMU, F/T, the estimator, the reward -- is ClimbTask's."""

    def __init__(self, lesson=1, slam=None, posts=False, **kw):
        """`posts`: a post from each rung end back to its wall, as scene.xml's cosmetic
        stubs and every real rung have -- something for the lidar to see along a wall;
        they touch nothing"""
        self.slam = slam or Slam()
        self.dirs = pattern()
        super().__init__(lesson, visual=False, xml=with_posts(lesson) if posts else None, **kw)

    def _install(self):
        """the lidar and IMU readings ride on the physics steps"""
        m = self.env.model
        self.lid = [m.site(n).id for n in NAMES]
        self.imu_site = m.site('s_imu').id
        self.own = [[m.geom(n + x).id for x in ('_body', '_window')] for n in NAMES]
        adr = lambda n: m.sensor_adr[m.sensor(n).id]
        self.s_acc, self.s_gyro = adr('imu_acc'), adr('imu_gyro')
        # what a ray can hit: the world (group 0) and the robot's collision shapes (3)
        self.groups = np.array([1, 0, 0, 1, 0, 0], np.uint8)
        step = self.env.step

        def stepped(ctrl):
            out = step(ctrl)
            self._tick()
            return out

        self.env.step = stepped
        self.env.timeout_s += PREROLL                    # the lesson's own time is untouched
        self._installed = True

    def reset(self, seed=None):
        if not getattr(self, '_installed', False):
            self._install()
        self.slam.call(op='reset')
        g = self.rng
        self.phase = g.integers(0, len(self.dirs), len(NAMES))       # where each lidar's pattern is
        self.imu_bias = {k: g.standard_normal(3) if self.noise else np.zeros(3) for k in ('acc', 'gyro')}
        self._sub, self._t0, self._imu, self._slices = 0, None, [], []
        self._landing, self._est_t, self._pending = [], [], []
        self._align, self.vmap, self._walls, self._dirty = None, VoxelMap(), [], True
        self.map, self.map_seen, self._nid = {}, set(), 0
        self.results, self._t_align = [], None
        self.frames = [0, 0]                             # handed to FAST-LIO, answered
        self._preroll = True
        return super().reset(seed)

    # ---- the sensors, riding on the physics -------------------------------------------
    def _tick(self):
        self._sub += 1
        if self._sub % SLICE:
            return
        m, d = self.env.model, self.env.data
        t = float(d.time)
        if self._t0 is None:
            self._t0 = t
        z = lambda n: self.rng.standard_normal(n) if self.noise else np.zeros(n)
        acc = d.sensordata[self.s_acc:self.s_acc + 3] + ct.SEN_SIG['acc'] * z(3) + ct.SEN_BIAS['acc'] * self.imu_bias['acc']
        gyro = d.sensordata[self.s_gyro:self.s_gyro + 3] + ct.SEN_SIG['gyro'] * z(3) + ct.SEN_BIAS['gyro'] * self.imu_bias['gyro']
        self._imu.append(np.r_[t, gyro, acc])
        xi, Ri = d.site_xpos[self.imu_site], d.site_xmat[self.imu_site].reshape(3, 3)
        k = int(RATE * SLICE * m.opt.timestep)
        for li, s in enumerate(self.lid):
            idx = (self.phase[li] + np.arange(k)) % len(self.dirs)
            self.phase[li] = (self.phase[li] + k) % len(self.dirs)
            nominal = self.dirs[idx]
            beam = nominal + ANGLE_SIG * z((k, 3)).reshape(k, 3) if self.noise else nominal
            beam /= np.linalg.norm(beam, axis=1, keepdims=True)
            R = d.site_xmat[s].reshape(3, 3)
            o = d.site_xpos[s].copy()
            gid, dist = np.zeros(k, np.int32), np.zeros(k)
            m.geom_group[self.own[li]] = 4                   # it looks out of its own window
            mujoco.mj_multiRay(m, d, o, (beam @ R.T).ravel(), self.groups, 1, -1, gid, dist, None, k, RANGE[1])
            m.geom_group[self.own[li]] = 3
            keep = (dist >= RANGE[0]) & (m.geom_bodyid[gid] == 0)          # the self-filter
            r = dist[keep]
            if self.noise:
                r = r + np.interp(r, *zip(*RANGE_SIG)) * z(len(r))
            p = o + (nominal[keep] @ R.T) * r[:, None]                     # where it says
            q = (p - xi) @ Ri                                              # in the IMU's frame
            pt = np.zeros(len(q), PT)
            # 1 us early: FAST-LIO takes a frame once an IMU reading reaches its end, and
            # the last one is stamped at the same instant -- by float rounding, sometimes
            # a nanosecond before it, and the frame would wait for the next one
            pt['t'] = max(0, int(round((t - self._t0) * 1e9)) - 1000)
            pt['x'], pt['y'], pt['z'] = q.T
            pt['r'], pt['line'] = 100, li
            self._slices.append(pt)
        if self._sub % FRAME == 0:
            self._frame()

    def _frame(self):
        """hand the frame that just ended to FAST-LIO, and keep what it says"""
        d = self.env.data
        pts = np.concatenate(self._slices) if self._slices else np.zeros(0, PT)
        imu = np.array(self._imu)
        t0 = self._t0
        self._t0, self._imu, self._slices = None, [], []
        self.frames[0] += 1
        a = self.slam.call(op='frame', t0=t0, imu=imu, pts=pts.view(np.uint8), n=len(pts))
        if not int(a['ok']):
            return
        self.frames[1] += 1
        xi, Ri = d.site_xpos[self.imu_site].copy(), d.site_xmat[self.imu_site].reshape(3, 3).copy()
        Ro = np.zeros(9)
        mujoco.mju_quat2Mat(Ro, a['quat'])
        base = d.xpos[self.env.base].copy()
        # in FAST-LIO's own frame until the map frame is fixed (`_look`); `rel`: the base
        # from the IMU, which the waist's encoder gives; `imu`: the truth, for fixing it
        self.results.append(dict(t=float(a['t']), pos=a['pos'], R=Ro.reshape(3, 3), rel=Ri.T @ (base - xi),
                                 imu=(xi, Ri), base=base, cloud=a['cloud']))
        self._landing.append(len(self.results) - 1)
        self._pending.append(len(self.results) - 1)     # into the map once it has a pose (`_place`)

    def _base(self, r):
        """where a result puts the base, in the map frame"""
        Ra, ta = self._align
        return Ra @ r['pos'] + ta + Ra @ r['R'] @ r['rel']

    @property
    def track(self):
        """(t, FAST-LIO's base, the true base) for every result since the map frame was fixed"""
        if self._align is None:
            return []
        return [(r['t'], self._base(r), r['base']) for r in self.results if r['t'] >= self._t_align]

    @property
    def track_fused(self):
        """(t, the base the map was built with, the true base): FAST-LIO's where it can see,
        the estimator's where it cannot (`_blind`)"""
        return [(r['t'], r['fused'], r['base']) for r in self.results if 'fused' in r]

    def _blind(self):
        """the projector onto what FAST-LIO cannot see here: up and down -- no floor, no
        ceiling, only rungs, and it drifts 3-7 cm on them -- and, along a flat wall, the
        wall (climb_task.along_seen: no wall faces that way)"""
        D = np.outer([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        ns = [n for _, n in self._walls]
        if not ns:
            return np.eye(3)
        u = np.cross(ns[0], [0.0, 0.0, 1.0])
        u /= np.linalg.norm(u)
        if not ct.along_seen(u, ns):
            D += np.outer(u, u)
        return D

    def _place(self, idx, delta_fl=None):
        """a result's points into the map. Where FAST-LIO cannot see, the frame is put
        where the estimator has the body -- its kinematics, from the hold -- and not where
        FAST-LIO drifted it: degeneracy-aware registration. `delta_fl`: before the map
        frame is fixed, the shift in FAST-LIO's frame that holds the hanging robot still"""
        r = self.results[idx]
        Ra, ta = self._align
        if delta_fl is not None:
            shift = Ra @ delta_fl
        else:
            T, E = zip(*self._est_t)
            base = np.array([np.interp(r['t'], T, np.asarray(E)[:, i]) for i in range(3)])
            Rm = Ra @ r['R']
            imu_slam, imu_kin = Ra @ r['pos'] + ta, base - Rm @ r['rel']
            shift = self._blind() @ (imu_kin - imu_slam)
        r['fused'] = self._base(r) + shift
        if len(r['cloud']):
            self.vmap.add(r['cloud'] @ Ra.T + ta + shift)
            self._dirty = True
        r['cloud'] = None

    # ---- what ClimbTask reads -------------------------------------------------------------
    def _slam_fix(self):
        now = float(self.env.data.time)
        ready = [i for i in self._landing if self.results[i]['t'] + ct.SLAM_LAG * ct.CONTROL_DT <= now + 1e-9]
        if not ready:
            return None
        self._landing = self._landing[len(ready):]
        if not self._est_t or self._align is None:
            return None
        nm = self.hold[self.anchor][0]
        if nm not in self.map or not ct.along_seen(self.map[nm][1], [n for _, n in self._walls]):
            return None                     # it cannot see along this rung (climb_task.along_seen)
        t_end, slam_base = self.results[ready[-1]]['t'], self._base(self.results[ready[-1]])
        T, E = zip(*self._est_t)
        est = np.array([np.interp(t_end, T, np.asarray(E)[:, i]) for i in range(3)])
        return slam_base - est

    def _sense(self):
        super()._sense()
        now = float(self.env.data.time)
        self._est_t.append((now, self.est.body_pos(self.est.base).copy()))
        self._est_t = self._est_t[-20:]
        if self._align is not None:
            for i in [i for i in self._pending if self.results[i]['t'] <= now + 1e-9]:
                self._place(i)
            self._pending = [i for i in self._pending if self.results[i]['t'] > now + 1e-9]

    def _look(self):
        """the rung map from what FAST-LIO has registered so far"""
        if self._preroll:                    # the reset's: hang still PREROLL s first
            self._preroll = False
            for _ in range(int(round(PREROLL / self.dt))):
                self.env.step(self.ctrl)
            # the map frame: where it hangs, as FAST-LIO has it at the end of the wait --
            # not at its first frame, which it moves off by 2-3 cm while it settles
            assert self.results, "FAST-LIO answered no frame in the wait: `docker logs climber-slam`"
            r = self.results[-1]
            (xi, Ri), Ra = r['imu'], r['imu'][1] @ r['R'].T
            self._align, self._t_align = (Ra, xi - Ra @ r['pos']), r['t']
            # it hung still all along: every frame of the wait from where it hangs now
            for i in self._pending:
                self._place(i, delta_fl=r['pos'] - self.results[i]['pos'])
            self._pending = []
        if self._dirty and len(self.vmap.n):
            P, w = self.vmap.points()                     # in the map frame (`_place`)
            ta = self._align[1]
            if not self._walls or self.frames[1] % 10 == 0:
                self._walls = planes(P, w, np.random.default_rng(0))
            toward = self.results[0]['imu'][0] if self.results else ta     # where it was hung
            self._update(self._bridge(rungs(P, w, self._walls, toward)))
            self._dirty = False
        self.map_goals = [self._snap(g) for g in self.goals]

    def _bridge(self, segs, longest=0.4):
        """One piece for two in line whose gap the robot's own body hides: a hand on a
        rung hides the span under it from every lidar, and the detector ends a rung
        where the points stop, so without this every held rung read as two with an
        end at each hand -- the policy read the nearer end 120-250 mm off, the training
        model 40-60. A gap the lidars can see is left a gap, seen empty or not: that is
        rl.lidar's rule, where a rung spans from its first seen cell to its last. The
        robot knows its own shape from its encoders; ponytail: the true pose stands in
        for that here, and the gap is judged where the robot is now, not over time."""
        segs, d = list(segs), self.env.data
        done = False
        while not done:
            done = True
            for i in range(len(segs)):
                for k in range(len(segs)):
                    if i == k:
                        continue
                    (c1, a1, h1), (c2, a2, h2) = segs[i], segs[k]
                    off = c2 - c1
                    t = off @ a1
                    gap = abs(t) - h1 - h2
                    if (t <= 0 or abs(a1 @ a2) < np.cos(np.radians(5))
                            or np.linalg.norm(off - t * a1) > 0.03 or not 0.0 < gap < longest):
                        continue
                    g0 = c1 + a1 * (h1 + gap / 2)          # the gap, as a rung of its own
                    seen = lidar.visible(np, self.rig, d.site_xpos, d.site_xmat, g0[None], a1[None],
                                         np.array([gap / 2]), np.array([R_RUNG]), np.array([True]))
                    if seen.any():
                        continue
                    lo_, hi_ = -h1, t + h2
                    segs[i] = (c1 + a1 * (lo_ + hi_) / 2, a1, (hi_ - lo_) / 2)
                    del segs[k]
                    done = False
                    break
                if not done:
                    break
        return segs

    def _update(self, dets):
        """keep each rung's name from one map to the next: a hold refers to it by name"""
        new, taken = {}, set()
        for c, a, h in dets:
            nm = None
            for old, (c0, a0, h0) in self.map.items():
                off = c - c0
                if (old not in taken and abs(a @ a0) > np.cos(np.radians(15))
                        and np.linalg.norm(off - (off @ a0) * a0) < 0.05
                        and abs(off @ a0) < h + h0):
                    nm = old
                    break
            if nm is None:
                nm, self._nid = f'det_{self._nid}', self._nid + 1
            taken.add(nm)
            new[nm] = (c, a, h)
        for nm, _ in getattr(self, 'hold', {}).values():  # a rung in a hand stays mapped
            if nm not in new and nm in self.map:
                new[nm] = self.map[nm]
        self.map, self.map_seen = new, set(new)

    def _snap(self, g):
        """a goal is a point the operator picks on a mapped rung: the nearest one within
        0.1 m, or the goal as given if none is mapped there yet"""
        best = None
        for c, a, h in self.map.values():
            q = c + np.clip((g - c) @ a, -h, h) * a
            if best is None or np.linalg.norm(g - q) < np.linalg.norm(g - best):
                best = q
        return best if best is not None and np.linalg.norm(g - best) < 0.10 else np.asarray(g, float)

    def _hung(self, side):
        """the rung the hook was hung on: the mapped one nearest its throat (a hang you
        set up yourself). None mapped there -- the SLAM has not seen it -- is a failure
        of the rig, and says so."""
        p = self._throat_true(side)
        best = None
        for nm, (c, a, h) in self.map.items():
            q = c + np.clip((p - c) @ a, -h, h) * a
            if best is None or np.linalg.norm(p - q) < best[0]:
                best = (np.linalg.norm(p - q), nm, q)
        if best is None or best[0] > ct.RUNG_TOL:
            nm = f'hung_{side}'                              # the true rung, marked as such
            c, a, h = self.segs[self._seat(side)[0]]
            self.map[nm], self.map_seen = (c, a, h), self.map_seen | {nm}
            print(f"[slam] the {side} hook's rung is not in the map at the start: "
                  f"{'nothing mapped' if best is None else f'nearest {best[0]*1e3:.0f} mm'}", flush=True)
            return nm, c + np.clip((p - c) @ a, -h, h) * a
        return best[1], best[2]


# --- how it did ------------------------------------------------------------------------------
def map_errors(t):
    """each true rung against the mapped pieces on its line (within 3 cm and 5 deg of its
    axis) -> ([(name, pieces, across m, tilt deg, end errors m (2,), share mapped)], the
    mapped segments on no rung's line). A rung a hand hides the middle of is two pieces:
    a lidar cannot tell that from a gap, so the map keeps them apart."""
    out, used = [], set()
    for nm, (c, a, h) in t.segs.items():
        on = []
        for k, (c2, a2, h2) in t.map.items():
            off = c2 - c
            if (abs(a @ a2) > np.cos(np.radians(5)) and np.linalg.norm(off - (off @ a) * a) < 0.03
                    and abs(off @ a) < h + h2):
                on.append((k, c2, a2, h2, off @ a))
        if not on:
            out.append((nm, 0, np.nan, np.nan, (np.nan, np.nan), 0.0))
            continue
        used |= {k for k, *_ in on}
        span = [(m_ - h2, m_ + h2) for _, _, _, h2, m_ in on]
        cover = np.zeros(400)                              # the rung in 400 pieces
        grid = np.linspace(-h, h, 400)
        for lo_, hi_ in span:
            cover[(grid >= lo_) & (grid <= hi_)] = 1
        across = np.mean([np.linalg.norm((c2 - c) - ((c2 - c) @ a) * a) for _, c2, _, _, _ in on])
        tilt = np.mean([np.degrees(np.arccos(min(1.0, abs(a @ a2)))) for _, _, a2, _, _ in on])
        ends = (min(x for x, _ in span) + h, max(y for _, y in span) - h)
        out.append((nm, len(on), across, tilt, ends, cover.mean()))
    return out, [k for k in t.map if k not in used]


def reading(t):
    """What the policy reads about the rungs, against what it would read if every sensor
    were exact -- per hand: the offset from its throat to the nearest seat on the
    nearest mapped rung (rl.climb_task._near_rungs: estimate and map) against the one
    from the true throat to the true nearest rung, and the distance to that rung's
    nearer end. -> [(offset error (3,), along-rung component, end error, wrong rung)]"""
    out = []
    for sd in 'LR':
        p_est, p = t.est.throat(sd), t._throat_true(sd)

        def near(segs, x):
            best = None
            for c, a, h in segs:
                hs = max(h - ct.SEAT_IN, 0.0)
                u = float((x - c) @ a)
                q = c + np.clip(u, -hs, hs) * a
                if best is None or np.linalg.norm(x - q) < best[0]:
                    best = (np.linalg.norm(x - q), q, a, hs - abs(u))
            return best
        mapped = near([t.map[k] for k in t.map if k in t.map_seen], p_est)
        true = near(list(t.segs.values()), p)
        if mapped is None:
            out.append((np.full(3, np.nan), np.nan, np.nan, True))
            continue
        e = (mapped[1] - p_est) - (true[1] - p)
        wrong = np.linalg.norm((mapped[1] - p_est) - (true[1] - p)) > 0.10
        out.append((e, float(e @ true[2]), mapped[3] - true[3], wrong))
    return out


def episode(lesson, act, steps, slam, seed=0, noise=True, posts=False, task=None):
    """`task`: a ClimbTask class to run instead of SlamTask on the same actions -- the
    training model, for `--compare`"""
    t = SlamTask(lesson, slam=slam, noise=noise, posts=posts) if task is None else \
        task(lesson, visual=False, noise=noise)
    t.rng = np.random.default_rng(seed)
    t0 = time.time()
    o, _ = t.reset()
    info, n, reads = {}, 0, []
    for i in range(steps):
        o, r, te, tr, info = t.step(act(o, i))
        reads += reading(t)
        n += 1
        if te or tr:
            break
    err = lambda tr: np.array([(s - b) for _, s, b in tr]) if tr else np.zeros((0, 3))
    slam_task = task is None
    return dict(t=t, steps=n, wall=time.time() - t0, reason=info.get('reason', ''),
                slam_err=err(t.track) if slam_task else None,
                fused_err=err(t.track_fused) if slam_task else None,
                maps=map_errors(t), frames=tuple(t.frames) if slam_task else None, reads=reads)


def report_reading(tag, reads):
    E = np.array([r[0] for r in reads if not r[3]])
    along = np.array([r[1] for r in reads if not r[3]])
    end = np.array([r[2] for r in reads if not r[3]])
    wrong = np.mean([r[3] for r in reads])
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v)))) * 1e3
    print(f"  what the policy reads, {tag}: the nearest rung's offset off by {rms(np.linalg.norm(E, axis=1)):.0f} "
          f"mm rms (p90 {np.nanpercentile(np.linalg.norm(E, axis=1), 90) * 1e3:.0f}), along the rung "
          f"{rms(along):.0f}, the nearer end {rms(end):.0f}; the wrong rung in {100 * wrong:.1f} % of readings")


def report(k, e):
    t, E = e['t'], e['slam_err']
    ax = t.segs[next(iter(t.segs))][1]
    print(f"lesson {k}: {e['steps']} steps ({e['steps'] * ct.CONTROL_DT:.1f} s simulated, "
          f"{e['wall']:.0f} s here), {e['frames'][0]} frames to FAST-LIO, {e['frames'][1]} "
          f"answered; ended {e['reason'] or 'running'}")
    blind = t._blind()
    print(f"  what FAST-LIO cannot see here: up/down{', and along the wall' if blind[0, 0] + blind[1, 1] > 0.5 else ''}"
          f" -- the SLAM fixes the anchor along a rung {'never' if ct.along_seen(ax, [n for _, n in t._walls]) is False else 'with every frame'}")
    for tag, E in (('FAST-LIO', E), ('the map was built with', e['fused_err'])):
        if not len(E):
            continue
        n = np.linalg.norm(E, axis=1)
        print(f"  {tag}: the base against the truth rms {np.sqrt((n ** 2).mean()) * 1e3:.1f} mm, "
              f"max {n.max() * 1e3:.1f}; along the start rung {np.sqrt(((E @ ax) ** 2).mean()) * 1e3:.1f} "
              f"mm rms, across it {np.sqrt((np.linalg.norm(E - np.outer(E @ ax, ax), axis=1) ** 2).mean()) * 1e3:.1f}")
    rows, stray = e['maps']
    print(f"  map at the end: {len(t.map)} segments for {len(t.segs)} rungs, {len(stray)} on no "
          f"rung (training: MAP_SIG {ct.MAP_SIG * 1e3:.0f} mm across, MAP_ROT_SIG "
          f"{np.degrees(ct.MAP_ROT_SIG):.1f} deg, MAP_END_SIG {ct.MAP_END_SIG * 1e3:.0f} mm an end)")
    for nm, pieces, across, ang, ends, cover in rows:
        print(f"    {nm:>22}: " + (f"{pieces} piece(s), across {across * 1e3:5.1f} mm, tilt {ang:4.2f} "
              f"deg, ends {ends[0] * 1e3:+5.0f} / {ends[1] * 1e3:+5.0f} mm, {cover * 100:3.0f} % mapped"
              if pieces else "not mapped"))
    for k in stray:
        c, a, h = t.map[k]
        print(f"    {'on no rung':>22}: {k} at {np.round(c, 3)}, {2 * h:.2f} m long")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lesson', type=int, nargs='+', default=[1])
    ap.add_argument('--run', help='a checkpoint (runs/NAME) to drive the robot; default: '
                                  'the arms swinging with the hooks shut')
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--exact', action='store_true', help='the sensors without their noise')
    ap.add_argument('--still', action='store_true', help='hang still: what the SLAM does with no motion')
    ap.add_argument('--posts', action='store_true', help="a post at each rung end, for the lidars to see")
    ap.add_argument('--compare', action='store_true',
                    help='also run the training model (ClimbTask) on the same actions, and compare what the policy reads')
    a = ap.parse_args()
    from .lessons import lesson_key
    slam = Slam()
    if a.still:
        act = lambda o, i: np.r_[np.zeros(8), -1.0, np.zeros(7), -1.0]
    elif a.run:
        from .mjx_check import _policy
        import rl.mjx_check as mc
        mc.RUN = a.run
        pol = _policy(a.run)
        assert pol, f"no checkpoint in {a.run}"
        act = lambda o, i: pol[0](o[None])[0]
    else:
        rng = np.random.default_rng(1)

        def act(o, i):                       # a slow swing of every joint, the fingers shut
            x = 0.4 * np.sin(2 * np.pi * (i / 60.0 + np.arange(17) / 17.0))
            x[[8, 16]] = -1.0
            return x + 0.1 * rng.standard_normal(17)
    for k in a.lesson:
        k = lesson_key(k)
        for s in range(a.episodes):
            acts = []                        # the same actions for both, recorded once
            rec = lambda o, i: acts[i] if i < len(acts) else acts.append(act(o, i)) or acts[i]
            e = episode(k, rec, a.steps, slam, seed=s, noise=not a.exact, posts=a.posts)
            report(k, e)
            report_reading('FAST-LIO and the rung map from its points', e['reads'])
            if a.compare:
                e2 = episode(k, rec, a.steps, slam, seed=s, noise=not a.exact, task=ClimbTask)
                report_reading('the training model (climb_task)', e2['reads'])


if __name__ == '__main__':
    main()
