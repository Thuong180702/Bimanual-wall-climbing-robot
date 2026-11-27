"""Voxel finite elements for the real robot's parts (no FEA package on this Mac): the part is
voxelised (trimesh), each voxel an 8-node linear hexahedron, assembled into one sparse
stiffness and solved with conjugate gradients (scipy). Linear elastic, 7075-T6
(E 71.7 GPa, nu 0.33). Supports = nodes in a region, fixed; loads = a force + a moment spread
over the nodes of a region (each node takes F/n + M x r / sum r^2 about the region's centroid).

Voxel stresses run a little low in thin bending (2-3 elements through a plate) and spike at
the load / support nodes, so the reported figure is the 99.5th percentile of the element
von Mises stresses outside 4 mm of any loaded or fixed node.

    python cad/tools/real_fea.py            # the cantilever check against beam theory
"""
import os, sys, tempfile
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

E7075, NU = 71.7e3, 0.33           # MPa (N/mm2), mm units throughout
YIELD = 503.0                      # 7075-T6, MPa


def _ke(E=E7075, nu=NU):
    """8-node hex, unit cube, 2x2x2 Gauss: 24 x 24 stiffness and the centre B matrix"""
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    Dm = np.zeros((6, 6))
    Dm[:3, :3] = lam
    Dm[np.arange(3), np.arange(3)] += 2 * mu
    Dm[np.arange(3, 6), np.arange(3, 6)] = mu
    corners = np.array([(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)], float)
    def B_at(p):
        s = corners * 2 - 1
        dN = np.zeros((8, 3))
        for a in range(8):
            fx = (1 + s[a, 0] * (2 * p[0] - 1)) / 2
            fy = (1 + s[a, 1] * (2 * p[1] - 1)) / 2
            fz = (1 + s[a, 2] * (2 * p[2] - 1)) / 2
            dN[a] = (s[a, 0] * fy * fz, fx * s[a, 1] * fz, fx * fy * s[a, 2])
        B = np.zeros((6, 24))
        for a in range(8):
            B[0, 3 * a], B[1, 3 * a + 1], B[2, 3 * a + 2] = dN[a]
            B[3, 3 * a], B[3, 3 * a + 1] = dN[a, 1], dN[a, 0]
            B[4, 3 * a + 1], B[4, 3 * a + 2] = dN[a, 2], dN[a, 1]
            B[5, 3 * a], B[5, 3 * a + 2] = dN[a, 2], dN[a, 0]
        return B
    g = (1 - 1 / np.sqrt(3)) / 2
    K = np.zeros((24, 24))
    for p in [(a, b, c) for a in (g, 1 - g) for b in (g, 1 - g) for c in (g, 1 - g)]:
        B = B_at(p)
        K += B.T @ Dm @ B / 8
    return K, B_at((0.5, 0.5, 0.5)), Dm, corners.astype(int)


KE, BC, DM, CORNERS = _ke()


def voxelize(part, pitch):
    """part (build123d, mm) -> filled voxel centres (n x 3) on a grid of `pitch`"""
    from cadgen import build123d as bd
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "p.stl")
        bd.export_stl(part, f, tolerance=0.05, angular_tolerance=0.2)
        m = trimesh.load(f)
    lo, hi = m.bounds
    xs = [np.arange(lo[i] + pitch / 2, hi[i], pitch) for i in range(3)]
    # parity along z, column by column: every triangle's crossing of every grid column under it
    hits = {}
    T = m.triangles
    for tri in T:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-12:
            continue
        i0, i1 = np.searchsorted(xs[0], [min(x0, x1, x2), max(x0, x1, x2)])
        j0, j1 = np.searchsorted(xs[1], [min(y0, y1, y2), max(y0, y1, y2)])
        if i0 >= i1 or j0 >= j1:
            continue
        X, Y = np.meshgrid(xs[0][i0:i1], xs[1][j0:j1], indexing="ij")
        a = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / det
        b_ = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / det
        c = 1 - a - b_
        ok = (a >= 0) & (b_ >= 0) & (c >= 0)
        if not ok.any():
            continue
        Z = a * z0 + b_ * z1 + c * z2
        for (ii, jj), z in zip(np.argwhere(ok), Z[ok]):
            hits.setdefault((i0 + ii, j0 + jj), []).append(z)
    cells = []
    for (i, j), zs in hits.items():
        zs = np.unique(np.round(np.sort(zs), 6))
        for k in range(0, len(zs) - 1, 2):
            sel = (xs[2] > zs[k]) & (xs[2] < zs[k + 1])
            for z in xs[2][sel]:
                cells.append((xs[0][i], xs[1][j], z))
    return np.array(cells)


class Model:
    def __init__(self, centres, pitch):
        self.h = pitch
        c = np.round((centres - centres.min(0)) / pitch).astype(int)
        self.o = centres.min(0) - pitch / 2
        node_ijk = (c[:, None, :] + CORNERS[None]).reshape(-1, 3)
        uniq, inv = np.unique(node_ijk, axis=0, return_inverse=True)
        self.nodes = self.o + uniq * pitch
        self.conn = inv.reshape(-1, 8)
        self.centres = centres
        dof = (3 * self.conn[:, :, None] + np.arange(3)).reshape(-1, 24)
        rows = np.repeat(dof, 24, axis=1).ravel()
        cols = np.tile(dof, (1, 24)).ravel()
        vals = np.tile((KE * pitch).ravel(), len(dof))
        n = 3 * len(self.nodes)
        self.K = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        self.dof = dof

    def region(self, pred):
        return np.where(pred(self.nodes))[0]

    def solve(self, fixed, loads, tol=1e-7, maxiter=20000):
        """fixed: node ids; loads: [(node ids, F (3,), M (3,))] -> displacement (n x 3)"""
        n = len(self.nodes)
        f = np.zeros((n, 3))
        for ids, F, M in loads:
            P = self.nodes[ids]
            r = P - P.mean(0)
            r2 = (r ** 2).sum()
            f[ids] += np.asarray(F, float) / len(ids)
            if np.any(M):
                f[ids] += np.cross(np.asarray(M, float), r) / r2 * 1.0
        free = np.ones(3 * n, bool)
        free[(3 * np.asarray(fixed)[:, None] + np.arange(3)).ravel()] = False
        Kf = self.K[free][:, free]
        d = Kf.diagonal()
        Mi = sp.diags(1.0 / d)
        uf, info = spla.cg(Kf, f.ravel()[free], rtol=tol, maxiter=maxiter, M=Mi)
        u = np.zeros(3 * n)
        u[free] = uf
        self.info = info
        self.marked = np.concatenate([np.asarray(fixed)] + [np.asarray(i) for i, _, _ in loads])
        return u.reshape(-1, 3)

    def von_mises(self, u):
        ue = u[self.conn].reshape(len(self.conn), 24)
        eps = ue @ (BC / self.h).T
        s = eps @ DM.T
        sx, sy, sz, txy, tyz, tzx = s.T
        return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))

    def report(self, u, keep_out=4.0, q=99.5):
        vm = self.von_mises(u)
        from scipy.spatial import cKDTree
        d, _ = cKDTree(self.nodes[self.marked]).query(self.centres)
        far = d > keep_out
        s = np.percentile(vm[far], q)
        return dict(stress=s, peak_at=self.centres[far][np.argmax(np.where(vm[far] >= s, vm[far], 0))],
                    defl=np.abs(u).max(), elems=len(self.conn), cg=self.info)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from cadgen import build123d as bd
    L, b, t, F = 200.0, 20.0, 10.0, 200.0          # cantilever: 200 x 20 x 10, tip load 200 N down
    beam = bd.Pos(L / 2, 0, 0) * bd.Box(L, b, t)
    for h in (2.5, 1.67):
        m = Model(voxelize(beam, h), h)
        u = m.solve(m.region(lambda P: P[:, 0] < 1e-6), [(m.region(lambda P: P[:, 0] > L - 1e-6), (0, 0, -F), (0, 0, 0))])
        vm = m.von_mises(u)
        root = vm[(m.centres[:, 0] > 20) & (m.centres[:, 0] < 20 + h)].max()
        theory_root = F * (L - 20) * (t / 2) / (b * t ** 3 / 12)
        print("pitch %.2f: tip deflection %.3f (beam theory %.3f) mm; stress 20 mm from the root %.0f (theory at the fibre %.0f) MPa; elems %d, cg %d"
              % (h, -u[:, 2].min(), F * L ** 3 / (3 * E7075 * b * t ** 3 / 12), root, theory_root, len(m.conn), m.info))
