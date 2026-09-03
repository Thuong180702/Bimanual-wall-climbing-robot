"""Record a move to mp4.

No video dependency: ffmpeg reads raw RGB frames on stdin. `mujoco.Renderer` draws them.

    python -m controller.record traverse out/traverse.mp4 --after-pullup
    python -m controller.record down     out/down.mp4     --after-pullup
"""
import subprocess, sys, argparse, numpy as np, mujoco
from . import SCENE


def _need_ffmpeg():
    """ffmpeg writes the frames; it is a separate program, not a pip
    package, and Windows has none by default."""
    import shutil
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH -- it writes the frames.\n"
            "  macOS:   brew install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg  (then reopen the shell)\n"
            "  Linux:   apt install ffmpeg\n"
            "Training and evaluation do not need it.")


class Video:
    """Frames in, mp4 out. `speed` is simulated seconds played per second of video."""

    def __init__(self, path, model, w=960, h=540, fps=30, speed=6.0, azimuth=315.0):
        # The robot hangs in a 0.75 m corridor between the front and back walls, so every
        # camera outside it is looking at the back of a slab. Render the walls nearly
        # transparent instead of trying to find an angle that does not exist.
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            if nm.startswith('wall_'):
                model.geom_rgba[g] = [0.45, 0.47, 0.50, 0.10]
        model.vis.headlight.ambient[:] = 0.45
        model.vis.headlight.diffuse[:] = 0.75
        self.r = mujoco.Renderer(model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.9, azimuth, -10.0
        self.every = max(1, int(round(speed / (fps * model.opt.timestep))))
        self.n = 0
        _need_ffmpeg()
        self.p = subprocess.Popen(
            ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
             '-r', str(fps), '-i', '-', '-an', '-vcodec', 'libx264', '-crf', '23',
             '-pix_fmt', 'yuv420p', path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def tick(self, data, lookat=None):
        self.n += 1
        if self.n % self.every:
            return
        if lookat is not None:
            self.cam.lookat[:] = lookat
        self.r.update_scene(data, self.cam)
        self.p.stdin.write(np.ascontiguousarray(self.r.render()).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        self.r.close()


def run(scene, path, speed=6.0, after_pullup=False, azimuth=315.0):
    from .holds import HoldMap
    from .moves import (make_env, Ctx, secure, climb, traverse, route_to, _pull_up,
                        _flare, MoveFailed)

    e = make_env(timeout_s=8000.0)
    e.reset()
    ctx = Ctx(e, HoldMap.load()).start_hanging()
    pick = lambda r: min((h for h in ctx.holds.grasps.values() if h.hold == r),
                         key=lambda h: abs(h.pos[1] - ctx.grip['L'][1][1]))
    for _ in secure(ctx):
        pass

    # (what to run before the camera starts, what to film)
    scenes = {
        'traverse': (None, lambda: traverse(ctx, [0., 1., 0.], 4)),
        'climb':    (None, lambda: climb(ctx, 'L', pick('rung_front_low_0').key)),
        'down':     (None, lambda: route_to(ctx, pick('rung_front_low_0').key)),
        'up':       (lambda: route_to(ctx, pick('rung_front_low_0').key),
                     lambda: route_to(ctx, pick('rung_front_line_0').key)),
    }
    setup, move = scenes[scene]
    if setup is not None:                     # get to where the scene starts, unfilmed
        try:
            for _ in setup():
                pass
        except MoveFailed as ex:
            print(f"{path}: could not reach the start of '{scene}': {ex}")
            return
    if after_pullup:
        # Start the camera on the posture the move works in, not the keyframe. A traverse
        # swings the shoulders forward, a climb opens them outward -- and the two do not
        # compose: running one and then the other drops the robot in five seconds.
        pre = _pull_up if scene == 'traverse' else _flare
        try:
            for _ in pre(ctx, 'R', mover='L'):
                pass
        except MoveFailed as ex:
            print(f"{path}: fell before the camera started: {ex}")
            return

    vid = Video(path, e.model, speed=speed, azimuth=azimuth)
    base = ctx.est.base
    tick0 = ctx.tick

    def tick_rec():
        tick0()
        vid.tick(e.data, lookat=e.data.xpos[base] + np.array([0., 0., 0.35]))
    ctx.tick = tick_rec

    t0, why = e.data.time, None
    try:
        for _ in move():
            pass
    except MoveFailed as ex:
        why = str(ex)
    for _ in range(600):                      # a beat on the final state
        try:
            tick_rec()
        except MoveFailed:
            break
    vid.close()
    print(f"{path}: {e.data.time - t0:.0f} s of simulated time"
          + (f" | stopped: {why}" if why else ""))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('scene', choices=['traverse', 'climb', 'down', 'up'])
    ap.add_argument('path')
    ap.add_argument('--speed', type=float, default=6.0)
    ap.add_argument('--azimuth', type=float, default=315.0)
    ap.add_argument('--after-pullup', action='store_true',
                    help='start the camera once the body is up, not at the keyframe')
    a = ap.parse_args()
    run(a.scene, a.path, a.speed, a.after_pullup, a.azimuth)
