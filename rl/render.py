"""Render the lesson scenes to PNG.

    python -m rl.render                    # a sheet of all seventeen + close-ups
    python -m rl.render --lesson 6         # just one, bigger

No image dependency: PNGs are written by piping a raw frame to ffmpeg, the same
trick `controller/record.py` uses for video.
"""
import argparse, os, subprocess
import numpy as np, mujoco
from .lessons import LESSONS, xml_path, lesson_key, lesson_tag

OUT = 'out/scenes'


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


def png(path, img):
    _need_ffmpeg()
    h, w = img.shape[:2]
    p = subprocess.run(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
         '-i', '-', '-frames:v', '1', path],
        input=np.ascontiguousarray(img).tobytes(),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-400:])
    return path


def _frame(m, d, w, h, margin=1.25):
    """lookat + distance that fit every rung and the robot into the picture.

    Hand-picked framing does not survive nine different layouts: lesson 6 spans
    0.95 m of height and lesson 1 is a single bar, so one distance either crops the
    tall ones or loses the robot in the wide ones.
    """
    pts = []
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if nm.startswith(('rung_', 'collar_')):
            half = m.geom_size[g][1] if nm.startswith('rung_') else 0.0
            ax = np.zeros(3)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            ax = R.reshape(3, 3)[:, 2]
            pts += [m.geom_pos[g] + half * ax, m.geom_pos[g] - half * ax]
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    pts.append(d.xpos[base])
    P = np.array(pts)
    lo, hi = P.min(0), P.max(0)
    centre = (lo + hi) / 2
    span = hi - lo
    fov = np.radians(m.vis.global_.fovy or 45.0)
    # the view is nearly along +x, so the picture's width is the y span (or x span
    # for a side wall) and its height is the z span
    wide = max(span[0], span[1])
    dist = max(span[2] / (2 * np.tan(fov / 2)),
               wide / (2 * np.tan(fov / 2) * (w / h)))
    return centre, max(0.6, dist * margin)


def shot(lesson, w=760, h=520, azimuth=196.0, elevation=-6.0, distance=None,
         lookat=None, wall_alpha=0.10, collar_rgba=(0.95, 0.45, 0.12, 1.0)):
    """the lesson at its hang keyframe, seen from in front of the wall.

    Azimuth ~196 deg puts the camera on the robot's side of the front wall looking
    back at it, so the rungs read as horizontal lines and the layout of the lesson
    is legible. The walls are faded because the robot hangs in a 0.75 m corridor and
    any camera outside it is looking at the back of a slab.
    """
    m = mujoco.MjModel.from_xml_path(xml_path(lesson))
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if nm.startswith('wall_'):
            m.geom_rgba[g] = [0.45, 0.47, 0.50, wall_alpha]
        elif nm.startswith('collar_'):
            m.geom_rgba[g] = list(collar_rgba)
    m.vis.headlight.ambient[:] = 0.5
    m.vis.headlight.diffuse[:] = 0.75
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=h, width=w)
    cam = mujoco.MjvCamera()
    auto_c, auto_d = _frame(m, d, w, h)
    cam.distance = distance if distance is not None else auto_d
    cam.azimuth, cam.elevation = azimuth, elevation
    cam.lookat[:] = lookat if lookat is not None else auto_c
    r.update_scene(d, cam)
    img = r.render().copy()
    r.close()
    return img


def sheet(imgs, cols=3, pad=6, bg=28):
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), bg, np.uint8)
    for i, im in enumerate(imgs):
        y, x = divmod(i, cols)
        out[pad + y * (h + pad):pad + y * (h + pad) + h,
            pad + x * (w + pad):pad + x * (w + pad) + w] = im
    return out


def video(run, lesson, path, episodes=3, fps=15, stride=1, cap=600,
          deterministic=True, azimuth=200.0, distance=1.9, ckpt='', mirror=False):
    """Roll a saved policy out on a lesson and write it to mp4.

    Reuses `controller.record.Video`, which already fades the walls and pipes raw
    frames to ffmpeg. Its `every` is speed/(fps*timestep), so passing
    speed = fps*timestep makes it write on every call -- one frame per call here,
    which is one per control step.

    The hooks are tinted per frame by whether the finger is actually gripping:
    GREEN inside the band the model README measures at >1200 N of pull-off, RED
    outside it (764 N at 0 deg, 72 N past -45). Policies hold the robot with an open
    hand for 13-55 % of their load-bearing time and it is invisible otherwise -- the
    hook looks the same either way from any camera angle.
    """
    import mujoco as mj
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from controller.record import Video
    from .train_ppo import MultiLesson
    from climbing_env import GRIP_BAND

    pre = f'out/{run}/{ckpt}'
    venv = VecNormalize.load(pre + 'vecnorm.pkl',
                             DummyVecEnv([lambda: MultiLesson([lesson], visual=True, mirror=mirror)]))
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(pre + 'model', env=venv, device='cpu')
    obs = venv.reset()
    t = venv.envs[0].cur
    m, d = t.env.model, t.env.data
    # the hook alone is a few centimetres of mesh and reads as a grey speck at this
    # framing, so the whole forearm-roll/wrist/hook chain carries the colour
    import mujoco as _mj
    chain = ('forearm_rot_%s', 'wrist_%s', 'wrist_dev_%s', 'hook_%s')
    hook_geoms = {}
    for sd in 'LR':
        ids = {_mj.mj_name2id(m, _mj.mjtObj.mjOBJ_BODY, n % sd) for n in chain}
        hook_geoms[sd] = [g for g in range(m.ngeom)
                          if m.geom_bodyid[g] in ids and m.geom_group[g] == 1]
    vid = Video(path, m, w=960, h=640, fps=fps, speed=fps * m.opt.timestep,
                azimuth=azimuth)
    vid.cam.distance = distance
    base = t.env.base
    done_n, n, reasons, marks = 0, 0, [], []
    while done_n < episodes and n < cap * episodes:
        act, _ = model.predict(obs, deterministic=deterministic)
        obs, r, dn, info = venv.step(act)
        n += 1
        if n % stride == 0:
            for sd in 'LR':
                ang = t.env.sensor('enc_finger_' + sd)[0]
                grip = GRIP_BAND[0] <= ang <= GRIP_BAND[1]
                for g in hook_geoms[sd]:
                    m.geom_rgba[g] = ([0.25, 0.85, 0.35, 1] if grip
                                      else [0.95, 0.25, 0.20, 1])
            vid.tick(d, lookat=d.xpos[base] + np.array([0., 0., 0.45]))
        if dn[0]:
            done_n += 1
            reasons.append(info[0].get('reason'))
            marks.append((n // stride) / fps)
    vid.close()
    # a multi-episode clip is unwatchable without an index: where each episode ends,
    # in playback seconds, so a reader can jump straight to the one being discussed
    at, idx = 0.0, []
    for r, end in zip(reasons, marks):
        idx.append(f"{int(at)//60}:{int(at)%60:02d} {r}")
        at = end
    print(f"{path}: {episodes} episodes, {n} control steps, {at/60:.1f} min")
    print("   " + " | ".join(idx))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lesson', type=lesson_key, default=None)
    ap.add_argument('--width', type=int, default=560)
    ap.add_argument('--video', default=None, metavar='RUN',
                    help='render a saved policy from out/RUN to mp4')
    ap.add_argument('--best', action='store_true',
                    help='render best_model.zip instead of the latest checkpoint')
    a = ap.parse_args()
    if a.video:
        os.makedirs(OUT, exist_ok=True)
        ck = 'best_' if a.best else ''
        tag = f"{a.video}{'_best' if a.best else ''}"
        video(a.video, 1, f'{OUT}/{tag}_lesson1.mp4', episodes=4, ckpt=ck)
        video(a.video, 2, f'{OUT}/{tag}_lesson2.mp4', episodes=2,
              stride=2, cap=1600, ckpt=ck)
        return
    os.makedirs(OUT, exist_ok=True)

    if a.lesson:
        p = png(f"{OUT}/{lesson_tag(a.lesson)}_{LESSONS[a.lesson]['name']}.png",
                shot(a.lesson, w=1200, h=820))
        print(p)
        return

    # Lessons 7 and 8 are the two-wall ones, and head-on they read as a single wall
    # with extra rungs: the side wall projects to a diagonal and the back wall lands
    # right behind the front one. Swing round and look down so the corner and the
    # 0.75 m corridor are visible as depth.
    view = {7: dict(azimuth=232.0, elevation=-26.0),
            8: dict(azimuth=250.0, elevation=-30.0)}
    imgs = [shot(k, w=a.width, h=int(a.width * 0.68), **view.get(k, {}))
            for k in sorted(LESSONS)]
    print(png(f'{OUT}/all_lessons.png', sheet(imgs)))

    # close-up of a hook on the rung: the rungs are smooth, so what there is to see
    # is the hook wrapping the bar and the pad face that now carries the grip
    m = mujoco.MjModel.from_xml_path(xml_path(1))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hook_L')
    throat = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0.023, 0.0, -0.085])
    print(png(f'{OUT}/hook_closeup.png',
              shot(1, w=1100, h=760, distance=0.34, azimuth=205.0, elevation=-14.0,
                   lookat=throat, wall_alpha=0.22)))
    print(png(f'{OUT}/hook_rung.png',
              shot(1, w=1200, h=560, distance=1.05, azimuth=200.0, elevation=-9.0,
                   lookat=throat + [0, 0.18, 0.02], wall_alpha=0.18)))


if __name__ == '__main__':
    main()
