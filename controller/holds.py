"""The wall map: where the holds are and which ones you can move between.

Built once from the model (offline), then written to holds.json. At run time the
controller reads the json only -- on hardware this map would come from vision/SLAM,
not from a simulator, so the control loop must never reach back into the model for it.
"""
import json, heapq, numpy as np, mujoco
from dataclasses import dataclass, asdict
from . import HOLDS_JSON

# --- reach envelope, from the README's measured numbers ---------------------
MAX_UP      = 0.50     # rows are spaced <= 0.50 m (75 % of the measured 0.671 m reach)
MAX_LATERAL = 0.30     # sideways offset the measurement held at
MAX_CROSS   = 0.68     # front->back span is 0.63 m; hook tip reaches 0.69 m behind
CROSS_DZ    = 0.25     # cross to a wall at roughly the same height
# Measured on this model: with the anchored arm frozen the free hand reaches only 0.10 m
# sideways. With the anchored arm free to swing the body -- which is what a climber does --
# solutions exist out to 0.30 m, but they get scarce. So: long side-steps are legal but
# priced, and the planner strings together short ones unless a gap forces a long one.
MAX_SIDESTEP = 0.28
EASY_SIDESTEP = 0.10
GRASP_STEP  = 0.10     # spacing of candidate grasp points along a rung
END_MARGIN  = 0.05     # keep the hook this far from a rung's end

BASE_COST = {'same_rung': 1.0, 'gap': 1.5, 'up': 1.0, 'down': 1.0,
             'cross_back': 4.0, 'cross_side': 4.0}

# Measured in phase 7: the hook is a slot around a cylinder, so along the rung axis it is
# held by friction alone -- 40 N sideways slides it 150 mm, 60 N slides it 310 mm. A
# vertical move reshuffles the whole body against the anchored hand and drags it that far,
# and the free hand's target slides with it. On a rung that runs out first, the hand
# arrives over a gap. So a climb wants SLIDE metres of rung to spare at BOTH ends.
SLIDE = 0.30
SLIDE_COST = 8.0       # a vertical move with no margin costs more than four side-steps


def cost(a, b, kind):
    """Long reaches are legal but expensive, so a route prefers several easy steps
    over one that sits at the edge of the measured envelope."""
    d = b.p() - a.p()
    lateral = float(np.linalg.norm(d[:2]))
    c = BASE_COST[kind] + 3.0 * max(0.0, lateral - EASY_SIDESTEP) / MAX_SIDESTEP
    if kind in ('up', 'down'):
        c += SLIDE_COST * max(0.0, SLIDE - min(a.margin, b.margin)) / SLIDE
    return c


@dataclass
class Grasp:
    key: str
    hold: str
    pos: list      # world point on the rung axis
    axis: list     # unit vector along the rung
    normal: list   # unit vector from the wall face toward the robot
    wall: str
    margin: float  # m of rung left between this point and the nearer end of its rung

    def p(self):
        return np.asarray(self.pos)

    def n(self):
        return np.asarray(self.normal)

    def a(self):
        return np.asarray(self.axis)


def build(model):
    """Extract every rung_* geom into grasp points. Wall + normal come from geometry,
    not from the geom name, so adding a wall to the scene needs no code change."""
    walls = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g), model.geom_pos[g].copy(),
              model.geom_size[g].copy())
             for g in range(model.ngeom)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('wall_')]
    grasps = {}
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if not nm.startswith('rung_'):
            continue
        c = model.geom_pos[g].copy()
        axis = model.geom_quat[g]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        axis = R.reshape(3, 3)[:, 2]                    # cylinder axis is local +z
        half = float(model.geom_size[g][1])

        # nearest wall, and the direction off its face that the robot lives on
        wname, wc, ws = min(walls, key=lambda w: np.linalg.norm(c - w[1]))
        thin = int(np.argmin(ws))                       # box's thinnest axis == its normal
        n = np.zeros(3)
        n[thin] = np.sign(c[thin] - wc[thin]) or 1.0

        usable = half - END_MARGIN
        k = max(1, int(round(2 * usable / GRASP_STEP)))
        for i, s in enumerate(np.linspace(-usable, usable, k + 1)):
            key = f"{nm}#{i}"
            grasps[key] = Grasp(key, nm, list(c + s * axis), list(axis), list(n), wname,
                                float(usable - abs(s)))
    return HoldMap(grasps)


def _kind(a: Grasp, b: Grasp):
    """Which climbing move gets you from a to b, or None if it is out of reach."""
    d = b.p() - a.p()
    dz = d[2]
    lateral = float(np.linalg.norm(d - dz * np.array([0., 0., 1.])))

    if a.wall != b.wall:
        if abs(dz) > CROSS_DZ or np.linalg.norm(d) > MAX_CROSS:
            return None
        # crossing to a wall facing the same way (parallel) is the bridge move
        parallel = abs(float(a.n() @ b.n())) > 0.9
        return 'cross_back' if parallel else 'cross_side'

    if abs(dz) < 0.02:
        if lateral > MAX_SIDESTEP:
            return None
        return 'same_rung' if a.hold == b.hold else 'gap'
    if abs(dz) > MAX_UP or lateral > MAX_LATERAL:
        return None
    return 'up' if dz > 0 else 'down'


class HoldMap:
    def __init__(self, grasps):
        self.grasps = grasps
        self._edges = None

    # ---- persistence ------------------------------------------------------
    def save(self, path=HOLDS_JSON):
        json.dump({k: asdict(v) for k, v in self.grasps.items()}, open(path, 'w'), indent=1)
        return path

    @staticmethod
    def load(path=HOLDS_JSON):
        return HoldMap({k: Grasp(**v) for k, v in json.load(open(path)).items()})

    # ---- graph ------------------------------------------------------------
    def edges(self):
        if self._edges is None:
            self._edges = {k: [] for k in self.grasps}
            ks = list(self.grasps)
            for i, ka in enumerate(ks):
                a = self.grasps[ka]
                for kb in ks[i + 1:]:
                    b = self.grasps[kb]
                    for (x, y, u, v) in ((a, b, ka, kb), (b, a, kb, ka)):
                        t = _kind(x, y)
                        if t:
                            self._edges[u].append((v, t))
        return self._edges

    def neighbors(self, key):
        return self.edges()[key]

    def nearest(self, pos, wall=None):
        cand = [g for g in self.grasps.values() if wall in (None, g.wall)]
        return min(cand, key=lambda g: np.linalg.norm(g.p() - np.asarray(pos))).key

    def route(self, start, goal):
        """A* over grasp points. Returns [(key, kind), ...] starting with (start, None)."""
        gp = self.grasps[goal].p()
        h = lambda k: float(np.linalg.norm(self.grasps[k].p() - gp)) / MAX_UP
        openq = [(h(start), 0.0, start)]
        came, best = {start: (None, None)}, {start: 0.0}
        while openq:
            _, c, k = heapq.heappop(openq)
            if k == goal:
                out = []
                while k:
                    out.append((k, came[k][1]))
                    k = came[k][0]
                return out[::-1]
            if c > best[k]:
                continue
            for nb, kind in self.neighbors(k):
                nc = c + cost(self.grasps[k], self.grasps[nb], kind)
                if nc < best.get(nb, 1e9):
                    best[nb], came[nb] = nc, (k, kind)
                    heapq.heappush(openq, (nc + h(nb), nc, nb))
        raise ValueError(f"no route {start} -> {goal}")
