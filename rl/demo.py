"""Demo videos: a trained policy climbing benchmark lessons on the full robot, with the point
cloud its four lidars build as it goes.

    python -m rl.demo --run out/gpu11 --lessons 1,2,3 --out demo/      # -> demo/lesson01_traverse.mp4 ...

The lidars are simulated point by point, as rl.slam_test does: each Mid-360 fires 200 000
points/s along Livox's own non-repetitive pattern (a real sensor's, recorded), MuJoCo casts
them into the walls, the rungs and the robot, and each return is off by the datasheet's
range noise. The robot's own returns are dropped (a self-filter). What the others hit is
kept in a 2.5 cm voxel map and drawn over the frame, coloured by height, the last moment's
points bright: the cloud a SLAM accumulates. The policy itself reads the rung map
ClimbTask builds from what the lidars see (rl.lidar), not this cloud.

An episode is first run without drawing, seed after seed, until one succeeds (at most
--tries); that seed is then replayed and drawn: the physics, the noise and the deterministic
policy repeat exactly.
"""
import argparse, os, pickle, subprocess
import numpy as np, mujoco

from .climb_task import CONTROL_DT
from .lessons import LESSONS, lesson_tag

W, H, FPS = 960, 720, 25           # each half of the split screen
VOXEL = 0.03                  # m: the map keeps one point a voxel
FRESH = 2                     # control steps a return is drawn bright for
VIEW = {7: (232.0, -24.0, 4.2), 8: (250.0, -26.0, 4.2)}   # lessons whose side or back wall needs an angle
WALLS = 5                     # the geom group the walls are moved to: cast into, never drawn


def load(run):
    from stable_baselines3 import PPO
    model = PPO.load(os.path.join(run, 'best_model.zip'), device='cpu')
    with open(os.path.join(run, 'best_vecnorm.pkl'), 'rb') as f:
        vn = pickle.load(f)
    return lambda o: model.predict(vn.normalize_obs(o[None].astype(np.float32)), deterministic=True)[0][0]


def reference(task, name):
    """an act() that plays a found climb's setpoints (rl.rsi's fixtures) open loop instead of a
    policy: a lesson no trained policy climbs yet, shown by the motion its references teach"""
    from .climb_task import ACT_SCALE
    from .rsi import HERE
    z = np.load(os.path.join(HERE, 'fixtures', name + '.npz'))
    rows, i = z['ctrl'], [0]

    def act(o):
        c = rows[min(i[0], len(rows) - 1)]
        i[0] += 1
        return np.clip((c - task.ctrl) / ACT_SCALE, -1, 1)
    act.restart = lambda: i.__setitem__(0, 0)
    return act


def episode(task, act, seed, each=None):
    """one episode from `seed` -> (reason, steps); `each(task, k, end)` after every control step"""
    task.rng = np.random.default_rng(seed)
    o, _ = task.reset()
    getattr(act, 'restart', lambda: None)()
    k = 0
    while True:
        o, _, te, tr, info = task.step(act(o))
        k += 1
        if each:
            each(task, k, info if (te or tr) else None)
        if te or tr:
            return info.get('reason', 'timeout'), k


class Scanner:
    """the four Mid-360 point by point, their world returns into a voxel map"""

    def __init__(self, task, seed=0):
        from .slam_test import NAMES, RANGE, RANGE_SIG, RATE, pattern
        m = task.env.model
        self.dirs, self.range, self.sig = pattern(), RANGE, RANGE_SIG
        self.lid = [m.site(n).id for n in NAMES]
        self.own = [[m.geom(n + x).id for x in ('_body', '_window')] for n in NAMES]
        self.k = int(RATE * CONTROL_DT)                         # points a lidar a control step
        self.groups = np.zeros(6, np.uint8)
        self.groups[[0, 3, WALLS]] = 1                          # rungs, the robot, the walls
        self.rng = np.random.default_rng(seed)
        self.phase = self.rng.integers(0, len(self.dirs), len(self.lid))
        self.keys = np.zeros(0, np.int64)
        self.pts = np.zeros((0, 3))
        self.born = np.zeros(0, int)
        self.on_rung = np.zeros(0, bool)
        self.rungs = set(task.env.rungs)

    def scan(self, task, step):
        m, d = task.env.model, task.env.data
        new, hit = [], []
        for li, s in enumerate(self.lid):
            idx = (self.phase[li] + np.arange(self.k)) % len(self.dirs)
            self.phase[li] = (self.phase[li] + self.k) % len(self.dirs)
            beam = self.dirs[idx] @ d.site_xmat[s].reshape(3, 3).T
            o = d.site_xpos[s].copy()
            gid, dist = np.zeros(self.k, np.int32), np.zeros(self.k)
            m.geom_group[self.own[li]] = 4                      # it looks out of its own window
            mujoco.mj_multiRay(m, d, o, beam.ravel(), self.groups, 1, -1, gid, dist, None, self.k, self.range[1])
            m.geom_group[self.own[li]] = 3
            keep = (dist >= self.range[0]) & (gid >= 0)
            keep[keep] &= m.geom_bodyid[gid[keep]] == 0         # the self-filter
            r = dist[keep] + np.interp(dist[keep], *zip(*self.sig)) * self.rng.standard_normal(keep.sum())
            new.append(o + beam[keep] * r[:, None])
            hit.append(np.isin(gid[keep], list(self.rungs)))
        P, R = np.concatenate(new), np.concatenate(hit)
        v = np.floor(P / VOXEL).astype(np.int64) + 2 ** 20
        key = (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]
        key, first = np.unique(key, return_index=True)
        fresh = ~np.isin(key, self.keys)
        self.born[np.isin(self.keys, key)] = step               # hit again: drawn bright again
        self.keys = np.r_[self.keys, key[fresh]]
        self.pts = np.r_[self.pts, P[first[fresh]]]
        self.born = np.r_[self.born, np.full(fresh.sum(), step)]
        self.on_rung = np.r_[self.on_rung, R[first[fresh]]]


def _colour(z, lo, hi):
    """height -> orange, yellow, green, cyan, as a lidar map is drawn"""
    stops = np.array([[0.95, 0.45, 0.10], [0.95, 0.85, 0.15], [0.35, 0.90, 0.30], [0.15, 0.85, 0.85]])
    x = np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f


class Film:
    """one lesson's episode into a split-screen mp4: on the left the robot climbing in the
    scene; on the right the map its four lidars have built so far, from above and aside, the
    robot moving through it, its path behind it and a 1 m reference grid under it"""

    def __init__(self, task, path, title, speed=1.0, who='the trained policy'):
        m = task.env.model
        self.walls = [g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('wall_')]
        m.geom_group[self.walls] = WALLS
        m.geom_rgba[self.walls] = [0.55, 0.57, 0.60, 0.35]
        m.vis.headlight.ambient[:] = 0.40
        m.vis.headlight.diffuse[:] = 0.70
        m.vis.global_.offwidth, m.vis.global_.offheight = W, H
        self.r = mujoco.Renderer(m, height=H, width=W)
        self.scene_opt = mujoco.MjvOption()                  # the left: everything
        self.map_opt = mujoco.MjvOption()                    # the right: the robot alone
        self.map_opt.geomgroup[0] = self.map_opt.geomgroup[WALLS] = 0
        az, el, dist = VIEW.get(task.lesson, (200.0, -8.0, 3.2))
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth, self.cam.elevation, self.cam.distance = az, el, dist
        self.map_cam = mujoco.MjvCamera()
        self.map_cam.azimuth, self.map_cam.elevation, self.map_cam.distance = 325.0, -36.0, 5.0
        self.every, self.title, self.look, self.path = max(1, int(round(speed))), title, None, []
        self.who = who
        c, a, h = task._rungs[:3]
        ends = np.r_[c + h[:, None] * a, c - h[:, None] * a]
        self.zlim = (ends[:, 2].min() - 0.8, ends[:, 2].max() + 0.5)
        self.floor = ends[:, 2].min() - 1.6                 # the reference grid: under the robot's feet
        self.scan = Scanner(task)
        self.p = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2 * W}x{H}',
             '-r', str(FPS), '-i', '-', '-an', '-vcodec', 'libx264', '-crf', '26', '-preset', 'slow',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path], stdin=subprocess.PIPE)
        from PIL import Image, ImageDraw, ImageFont
        ttf = '/System/Library/Fonts/Helvetica.ttc'
        self.pil = (Image, ImageDraw, ImageFont.truetype(ttf, 24) if os.path.exists(ttf) else ImageFont.load_default())

    def step(self, task, k, end=None):
        self.scan.scan(task, k)
        self.path.append(task.env.data.xpos[task.env.base].copy())
        if k % self.every and end is None:
            return
        d = task.env.data
        base = d.xpos[task.env.base] + np.array([0.0, 0.0, 0.75])
        self.look = base if self.look is None else 0.92 * self.look + 0.08 * base   # steady cameras
        self.cam.lookat[:] = self.map_cam.lookat[:] = self.look
        left, right = self.scene(task), self.map(task, k)
        img = self.caption(np.concatenate([left, right], 1), task, k, end)
        for _ in range(FPS if end is not None else 1):          # a second on the last frame
            self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def _goals(self, task):
        sc = self.r.scene
        for g in task.goals:
            if sc.ngeom < sc.maxgeom:
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.035, 0, 0]),
                                    np.asarray(g, float), np.eye(3).ravel(), np.array([1.0, 0.15, 0.15, 0.9], np.float32))
                sc.ngeom += 1

    def scene(self, task):
        self.scene_opt.geomgroup[WALLS] = 1
        self.r.update_scene(task.env.data, self.cam, self.scene_opt)
        self._goals(task)
        return self.r.render().copy()

    def _project(self, P):
        """world points -> (u, v, depth along the view) through the map camera, as drawn"""
        gc = self.r.scene.camera[0]
        pos, fwd, up = np.asarray(gc.pos), np.asarray(gc.forward), np.asarray(gc.up)
        Q = np.asarray(P) - pos
        z = Q @ fwd
        th = np.tan(np.radians(self.r.model.vis.global_.fovy) / 2)
        zz = np.maximum(z, 1e-6)
        return ((Q @ np.cross(fwd, up)) / zz / (th * W / H) + 1) * W / 2, (1 - (Q @ up) / zz / th) * H / 2, z

    def map(self, task, k):
        self.r.update_scene(task.env.data, self.map_cam, self.map_opt)
        # the robot alone; every map point is drawn where the robot is not (a depth test needs
        # ARB_clip_control, which not every GL has). The mask before the goals go in: geoms
        # added to the scene carry no segmentation id
        self.r.enable_segmentation_rendering()
        bg = self.r.render()[..., 0] < 0
        self.r.disable_segmentation_rendering()
        self._goals(task)
        img = self.r.render().copy()
        img[bg] = (12, 14, 18)
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, 'RGBA')
        c = self.look
        for i in range(-6, 7):                                   # a 1 m grid on the floor plane
            for a, b in (((c[0] + i, c[1] - 6), (c[0] + i, c[1] + 6)), ((c[0] - 6, c[1] + i), (c[0] + 6, c[1] + i))):
                u, v, z = self._project([[round(a[0]), round(a[1]), self.floor], [round(b[0]), round(b[1]), self.floor]])
                if (z > 0.1).all():
                    dr.line([(u[0], v[0]), (u[1], v[1])], fill=(90, 95, 110, 110), width=1)
        img = np.asarray(im).astype(np.float32) / 255.0
        u, v, z = self._project(self.scan.pts)
        ok = (z > 0.05) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        sel = np.nonzero(ok)[0]
        sel = sel[np.argsort(-z[sel])]                           # far first: near points on top
        u, v, z = u[sel].astype(int), v[sel].astype(int), z[sel]
        P, born, rung = self.scan.pts[sel], self.scan.born[sel], self.scan.on_rung[sel]
        col = _colour(P[:, 2], *self.zlim) * np.clip(1.5 - 0.08 * z, 0.45, 1.0)[:, None]
        col[~rung] *= 0.75                                        # the walls behind, the rungs in front
        new = (k - born) < FRESH
        col[new] = 0.4 * col[new] + 0.6                          # this moment's scan, bright
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):   # rungs 6 px, walls 2
            pick = (rung | ((du, dv) in ((0, 0), (1, 1)))) & (v + dv >= 0) & (u + du >= 0)
            vis = pick & bg[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)]
            img[v[vis] + dv, u[vis] + du] = col[vis]
        out = Image.fromarray((img * 255).astype(np.uint8))
        dr = ImageDraw.Draw(out, 'RGBA')
        if len(self.path) > 1:                                    # where the body has been
            u, v, z = self._project(np.array(self.path) + [0.0, 0.0, 0.75])
            pts = [(float(a), float(b)) for a, b, c in zip(u, v, z) if c > 0.1]
            dr.line(pts, fill=(255, 60, 200, 230), width=3)
        return np.asarray(out)

    def caption(self, img, task, k, end):
        Image, ImageDraw, font = self.pil
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, 'RGBA')
        dr.rectangle([0, 0, 2 * W, 40], fill=(0, 0, 0, 170))
        text = f"t = {k * CONTROL_DT:4.1f} s" + (f"    -> {end.get('reason', 'timeout')}" if end else "")
        dr.text((16, 7), text, fill=(255, 255, 255, 255), font=font)
        dr.text((W + 16, 7), f"lidar map: {len(self.scan.pts):,} points", fill=(255, 255, 255, 255), font=font)
        dr.line([(W, 0), (W, H)], fill=(80, 80, 90, 255), width=2)
        dr.rectangle([0, H - 38, 2 * W, H], fill=(0, 0, 0, 170))
        dr.text((16, H - 33), f"the full robot in MuJoCo, driven by {self.who}   red: goals", fill=(220, 220, 220, 255), font=font)
        dr.text((W + 16, H - 33), "4 x Livox Mid-360, 200k pts/s each   colour: height   pink: path",
                fill=(220, 220, 220, 255), font=font)
        return np.asarray(im)

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        self.r.close()


def main():
    from .climb_task import ClimbTask
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a run directory with best_model.zip')
    ap.add_argument('--lessons', default='1,2,3')
    ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--tries', type=int, default=8, help='seeds tried for a successful episode')
    ap.add_argument('--speed', type=float, default=1.0, help='simulated seconds a second of video')
    ap.add_argument('--out', default='demo')
    ap.add_argument('--reference', default=None,
                    help="drive the robot with this reference climb (rl/fixtures/<name>.npz) instead of the policy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    policy = load(a.run)
    for k in (int(x) for x in a.lessons.split(',')):
        task = ClimbTask(k, visual=True, mirror=a.mirror)
        act = reference(task, a.reference) if a.reference else policy
        who = 'a reference climb (CEM, open loop)' if a.reference else 'the trained policy'
        seed = next((s for s in range(a.tries) if episode(task, act, s)[0] == 'success'), None)
        if seed is None:
            print(f"lesson {k}: no success in {a.tries} episodes -- no video")
            continue
        name = f"{lesson_tag(k)}_{LESSONS[k]['name']}{'_mirror' if a.mirror else ''}.mp4"
        film = Film(task, os.path.join(a.out, name), LESSONS[k]['name'], a.speed, who)
        why, n = episode(task, act, seed, film.step)
        film.close()
        print(f"lesson {k}: seed {seed}, {why} at {n * CONTROL_DT:.1f} s -> {os.path.join(a.out, name)}")


if __name__ == '__main__':
    main()
