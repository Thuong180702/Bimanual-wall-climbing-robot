"""Drive the trained policy by hand across one wall that mixes every lesson: you pick the
rung, the policy climbs to it.

    .venv/bin/mjpython -m rl.play --run out/gpu14 [--seed 3] [--mean]

The wall is drawn from the run's own ADR levels (best_state.json), one wall of each family
(rl.walls.sample) laid out round family 3's start row:

  * above the start: family 3's rows standing out (and down them again: lesson 4)
  * on the robot's left (+y), in a random order, each from where the last one ends: 2 (gaps),
    9 (a swing gap), 5 (a staircase down), 6 (a tier up)
  * on its right (-y), each past a gap: a row with family 8's rung and wall behind it, then a
    row ending in family 7's corner
  * along any row: lesson 1

Each rung carries its number in the viewer. Type in the terminal:

    5          both hands to rung 5, at its point nearest the robot (a far rung: its near end)
    5 0.4      ...0.4 m further along it (+y, or +x on the side rung)
    r          the robot back to the start
    q          quit

The goals are put as training puts them: on one rung, ~0.31 m apart, the left hand's on the
robot's left. A target is one move a lesson teaches -- a rung several moves away, or one no
lesson reaches from where the robot is, is outside what the policy learnt. Two more limits,
measured on gpu14 (2.2B steps): the policy is shown only the move's own rungs (`attach`), and
lesson 8 starts only from the start hang -- after two traverses on its own wall 2/8 (mean) or
4/8 (sampled) against 8/8, and 0/4 here after the moves to its row.
"""
import argparse, json, os, queue, sys, threading, time, types
import numpy as np, mujoco, mujoco.viewer

from . import demo, walls
from .climb_task import ClimbTask, CONTROL_DT, FRAME_SKIP, route_table, _rung_graph, _rung_points
from .lessons import scene_xml, _rx, _ry
from .mjx_scene import OUT
from .walls import FACE, HY, SEAT, X, Z0

LIMIT = 60.0          # s a target is given, as an episode (walls.TIMEOUT)
SIDE_X = (0.125 - 0.305, 0.125 + 0.305)   # family 7's side rung, along x (rl.walls.sample)


def _box(name, pos, size):
    return (f'    <geom name="{name}" type="box" material="wall" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}"/>')


def _plus_y(lay):
    """a family's wall going +y (rl.walls.sample mirrors half of them)"""
    return walls.mirror(lay) if lay['rungs'][-1][1][1] < 0 else lay


def world(levels, rng):
    """-> (spec for ClimbTask, [(rung number, what it is)])"""
    rows, about = [], []             # (y0, y1, z, x) along y; side rungs as ('x', x0, x1, z, y)

    def row(y0, y1, z, x=X, what=''):
        rows.append((y0, y1, z, x))
        about.append(what)

    # Every rung as long as training's: the policy sees how far each end of its rung is, and
    # on a 5.7 m start row (the corner 4.3 m off) it climbed family 3's rows 1/6, against
    # 6/6 on Y0..Y1. So the start row and the rows above are family 3's own, and the -y side
    # is two more rows, each past a gap of lesson 2's
    def gap():
        return rng.uniform(0.04, levels[2]['width'])

    row(walls.Y0, walls.Y1, Z0, what='start row (lesson 1 along it)')
    lay3 = walls.sample(3, levels[3], rng)
    for k, (_, c, _, _, _) in enumerate(lay3['rungs'][1:], 1):
        row(walls.Y0, walls.Y1, c[2], x=c[0], what=f'row {k} above, {c[2] - Z0:.2f} m up (lesson 3; down: 4)')
    # the right (-y): a row with family 8's rung behind it, then family 7's corner
    a1 = walls.Y0 - gap()
    row(a1 - 2.0, a1, Z0, what='row on the right (-y), past a gap (lesson 2)')
    back = (a1 - 1.9, a1 - 0.1)        # behind all of it: lesson 8's goals are straight behind
    if 8 in levels:                    # a run of families 1-7 (the paper's) has no wall behind
        xb = walls.sample(8, levels[8], rng)['rungs'][1][1][0]
        row(*back, Z0, x=xb, what=f'behind that row, {X - xb:.2f} m (lesson 8)')
    s = abs(walls.sample(7, levels[7], rng)['walls'][1][1][1])
    yc = a1 - 2.0 - gap() - 0.9      # where the robot hangs to turn the corner
    row(yc - (s - 0.10), yc + 0.9, Z0, what='corner row, past a gap')
    side = ('x', *SIDE_X, Z0, yc - s + 0.06)
    rows.append(side)
    about.append(f'round the corner, side wall {s:.2f} m along (lesson 7)')
    # the left (+y): each family from where the last ended
    y_end, z = walls.Y1, Z0
    for f in rng.permutation([f for f in (2, 5, 6, 9) if f in levels]):
        lay = _plus_y(walls.sample(int(f), levels[int(f)], rng))
        (_, c0, _, h0, _), *rest = lay['rungs']
        dy, dz = y_end - (c0[1] + h0), z - Z0
        for i, (_, c, _, h, _) in enumerate(rest):
            y0, y1 = c[1] - h + dy, c[1] + h + dy
            if i == len(rest) - 1:       # room to hang on before the next one
                y1 = max(y1, y0 + 1.0)
                y_end, z = y1, c[2] + dz
            row(y0, y1, c[2] + dz, x=c[0], what=f'lesson {f}, rung {i + 1} of {len(rest)}')

    ys = [v for r in rows for v in ((r[1], r[2]) if r[0] != 'x' else (r[4],))]
    lo, hi = min(ys) - 1.0, max(ys) + 1.0
    w = [_box('wall_front', (X + FACE + 0.05, (lo + hi) / 2, 0.5), (0.05, (hi - lo) / 2, 3.5)),
         _box('wall_side', (0.125, side[4] - 0.06 - 0.05, 0.5), (0.425, 0.05, 3.0))]
    if 8 in levels:
        w.append(_box('wall_back', (xb - FACE - 0.05, sum(back) / 2, Z0 - 0.5),
                      (0.05, (back[1] - back[0]) / 2 + 0.2, 2.0)))
    xml = [_rx(str(i), r[1], r[2], r[3], y=r[4]) if r[0] == 'x' else _ry(str(i), r[0], r[1], r[2], x=r[3])
           for i, r in enumerate(rows)]
    spec = dict(name='wall', title='every lesson on one wall', walls=w, rungs=xml,
                goals=[[X, HY, Z0], [X, -HY, Z0]], timeout=1e9, goal_rungs=['rung_0'])
    return spec, list(enumerate(about))


def aim(task, rung, off=0.0):
    """both goals onto `rung`, as training puts them -> the goals [L, R]. A rung too short
    for both hands (family 6's tier) takes one, at its middle, and the rung nearest it the
    other, 0.15 m in from its near end"""
    c, a, h = task.segs[rung]
    thr = np.array([task._throat_true(sd) for sd in 'LR'])
    mid = thr.mean(0)
    half = min(HY, h - SEAT)
    on = [rung]
    if half >= walls.MIN_W / 2:
        room = h - SEAT - half
        u = np.clip(float((mid - c) @ a) + off, -room, room)
        g1, g2 = c + (u + half) * a, c + (u - half) * a
    else:
        def seat(nm, p):
            c2, a2, h2 = task.segs[nm]
            return c2 + np.clip(float((p - c2) @ a2), SEAT - h2, h2 - SEAT) * a2
        nm = min((n for n in task.segs if n != rung), key=lambda n: np.linalg.norm(seat(n, c) - c))
        q = seat(nm, c)
        if np.linalg.norm(q - c) > 1.0:
            raise ValueError(f'{rung} is too short for both hands, and no rung is near it')
        a2 = task.segs[nm][1]
        g1, g2, u = c, seat(nm, q + 0.15 * np.sign(float((q - c) @ a2)) * a2), 0.0
        on.append(nm)
    left = thr[0] - thr[1]
    if abs(left @ a) < 0.3 * np.linalg.norm(left):   # a rung across the hands: by where it is
        f = c + u * a - mid
        f[2] = 0.0
        left = np.cross([0.0, 0.0, 1.0], f - (f @ a) * a)
    gL, gR = (g1, g2) if (g1 - g2) @ left > 0 else (g2, g1)
    task.goals = [gL, gR]
    task.spec['goal_rungs'] = on
    task.shown = way(task) | set(on)
    task._pts, task._route_g = route_table(list(task.segs.values()), task.goals)
    task._look()                                  # the map's goals
    task.grabs[:] = 0
    task.credit[:] = 0.0
    task.prev_d, task.prev_live = task._hand_dists(), task._live()
    return task.goals


def way(task):
    """the rungs on each hand's shortest way to its goal through the rungs (climb_task's
    reach graph), the ones it holds and the goal's included"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    names = list(task.segs)
    P, R = _rung_points(list(task.segs.values()))
    G = csr_matrix(_rung_graph(P, R))
    out = set()
    for sd, g in zip('LR', task.goals):
        src = int(np.argmin(np.linalg.norm(P - task._throat_true(sd), axis=1)))
        k = int(np.argmin(np.linalg.norm(P - g, axis=1)))
        out |= {names[R[src]], names[R[k]]}
        _, pred = dijkstra(G, indices=src, return_predecessors=True)
        while k != src and pred[k] >= 0:     # unreachable: the two ends alone
            out.add(names[R[k]])
            k = pred[k]
    return out


def attach(task):
    """show the policy only `task.shown`'s rungs (None: all it has seen). It reads the three
    rungs nearest each hand, however far: on its training walls those are the move's own,
    and lesson 8's wall has two -- one more anywhere in view, 1.0 to 5.0 m off, and it
    climbed lesson 8's own walls 0/8 against 8/8"""
    task._look, task.shown = types.MethodType(_look_shown, task), None   # a method: deepcopy-able


def _look_shown(self):
    ClimbTask._look(self)
    if self.shown is not None:
        self.map_seen &= self.shown


def hold(task):
    """a control step with the setpoints kept: hanging where it is"""
    for _ in range(FRAME_SKIP):
        task.env.step(task.ctrl)
    task._sense()


def marks(v, task, goals):
    """each rung's number at its middle, the goals in red"""
    sc = v.user_scn
    sc.ngeom = 0
    for nm, (c, _, _) in task.segs.items():
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
                            np.asarray(c, float) - [0.04, 0, 0], np.eye(3).ravel(),
                            np.array([0.2, 0.6, 1.0, 0.8], np.float32))
        g.label = nm[5:]
        sc.ngeom += 1
    for p in goals or ():
        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.035, 0, 0]),
                            np.asarray(p, float), np.eye(3).ravel(), np.array([1.0, 0.15, 0.15, 0.9], np.float32))
        sc.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a run directory with best_model.zip and best_state.json')
    ap.add_argument('--seed', type=int, default=None, help='the wall (default: a new one)')
    ap.add_argument('--mean', action='store_true', help="the policy's mean action (default: sampled, as it trained)")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else int(np.random.default_rng().integers(1 << 30))
    levels = {int(f): v for f, v in json.load(open(os.path.join(a.run, 'best_state.json')))['curriculum']['levels'].items()}
    spec, about = world(levels, np.random.default_rng(seed))
    os.makedirs(OUT, exist_ok=True)
    xml = os.path.join(OUT, f'_play_{seed}.xml')
    open(xml, 'w').write(scene_xml(spec, 'wall'))
    task = ClimbTask(1, visual=True, xml=xml, spec=spec)
    os.remove(xml)
    task.env.timeout_s = 1e9
    attach(task)
    pol = demo.load(a.run, deterministic=a.mean)
    o, _ = task.reset()

    print(f"wall seed {seed}, levels from {a.run}")
    for i, what in about:
        print(f"  {i:2d}  {what}")
    print("type a rung number (and an offset along it, m), r to restart, q to quit")
    cmds = queue.Queue()
    threading.Thread(target=lambda: [cmds.put(ln.strip()) for ln in sys.stdin], daemon=True).start()

    m, d = task.env.model, task.env.data
    going, t0, goals = False, 0.0, None
    for w in ('wall_back', 'wall_side'):            # seen through from the front: drawn only
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, w) >= 0:
            m.geom_rgba[m.geom(w).id] = [0.55, 0.57, 0.60, 0.25]
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.type, v.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, task.env.base
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 330.0, -15.0, 4.5
        marks(v, task, goals)
        while v.is_running():
            tic = time.time()
            while not cmds.empty():
                c = cmds.get().split()
                if not c:
                    continue
                if c[0] == 'q':
                    return
                if c[0] == 'r':
                    task.shown = None
                    o, _ = task.reset()
                    going, goals = False, None
                    print("back at the start")
                    continue
                try:
                    goals = aim(task, f'rung_{int(c[0])}', float(c[1]) if len(c) > 1 else 0.0)
                except (ValueError, KeyError) as err:
                    print(f"  ? {err}")
                    continue
                o = task._obs()
                going, t0 = True, d.time
                print(f"-> rung {c[0]}: L {np.round(goals[0], 2)}, R {np.round(goals[1], 2)}")
            if going:
                o, _, te, tr, info = task.step(pol(o))
                why = info.get('reason') if te or tr else ('timeout' if d.time - t0 > LIMIT else None)
                if why:
                    going = False
                    print(f"   {why} after {d.time - t0:.1f} s" +
                          ("" if why in ('success', 'timeout') else "  (r to restart)"))
            elif task.env.terminated()[1] not in ('fell', 'released', 'blowup'):
                hold(task)
            with v.lock():
                marks(v, task, goals)
            v.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - tic)))


if __name__ == '__main__':
    main()
