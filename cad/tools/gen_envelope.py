"""Derive each actuator's real radial envelope from its vendor STEP.

A hand-typed stepped profile missed 0.5-0.8 % of the vendor volume - connector
bosses and cable chamfers that a bracket would have collided with. This reads
the geometry instead: for every face, the largest radius anywhere in its
z-span, accumulated into 0.5 mm bands. Bounding boxes over-estimate slightly,
which is the right direction for a keep-out.

Writes src/lib/_envelope.json, which lib.shapes.keepout() builds from.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cadgen import build123d as bd
from lib.spec import IFACE

STEPS = {"AK80-64": "ak80_64", "AK60-39": "ak60_39", "AK45-36": "ak45_36"}
STEP = 0.5


def envelope(path):
    """Max radius per 0.5 mm band, from the tessellated surface.

    Face bounding boxes are not usable here: a full cylinder's box corner is
    r*sqrt(2), so a 98 mm actuator reads as 139. Mesh vertices lie ON the
    surface, so their radius is the real one.
    """
    s = bd.import_step(path)
    verts, tris = s.tessellate(0.25)
    zs = [v.Z for v in verts]
    z0, z1 = min(zs), max(zs)
    n = int(math.ceil((z1 - z0) / STEP)) + 1
    r = [0.0] * n
    # per TRIANGLE, not per vertex: one triangle on a long smooth barrel spans
    # many bands, and binning only its corners leaves the bands between empty.
    for t in tris:
        p3 = [verts[i] for i in t]
        rad = max(math.hypot(v.X, v.Y) for v in p3)
        i0 = int((min(v.Z for v in p3) - z0) / STEP)
        i1 = int((max(v.Z for v in p3) - z0) / STEP)
        for i in range(max(0, i0), min(n, i1 + 1)):
            if rad > r[i]:
                r[i] = rad
    bands = []
    for i, v in enumerate(r):
        zlo, zhi = z0 + i * STEP, z0 + (i + 1) * STEP
        if bands and abs(bands[-1][2] - v) < 0.05:
            bands[-1][1] = zhi
        else:
            bands.append([zlo, zhi, v])
    return dict(z0=z0, z1=z1, bands=[[round(a, 3), round(b, 3), round(c, 3)]
                                     for a, b, c in bands])


def clip(bands, d):
    """Make the envelope exact at the two mating faces.

    A 0.5 mm band that straddles a face picks up that face's full diameter and
    bleeds it across, which would push the brackets off the very faces they
    bolt to. The measured face data says what is really on each side: nothing
    but the spigot above the output face, nothing but the boss and rotor
    between the two front faces, and nothing but the pilot boss behind the
    rear flange.
    """
    oz, hz, rz = d["out_z"], d["hou_z"], d["rear_z"]
    # Locating pins are NOT swept into a ring here: three Ø4 pins on a circle
    # leave most of that circle free, and the plate they locate is meant to
    # occupy it. keepout() adds them as three discrete cylinders instead.
    above_out = d["out_spigot"] / 2 if d["out_spigot"] > 0 else 0.0
    # The band that STARTS on the front face carries that face's own radius -
    # a triangle lying in the face plane bins into it. Where a boss is
    # declared, it is the cap. Where none is, take the largest radius the mesh
    # reports strictly between the two faces instead: that is the real rotor
    # shoulder, and it is what a plate bolted to the face has to miss. Capping
    # at hou_id there made the AK60-39 look 1.7 mm fatter than it is and
    # pushed every plate off the metal its own bolts need.
    if d["hou_boss_h"] > 0:
        between = max(d["hou_boss"], d["out_od"]) / 2
    else:
        inner = [r for lo, hi, r in bands if lo >= hz + 1e-6 and hi <= oz + 1e-6]
        between = max(inner) if inner else d["out_od"] / 2
    below_rear = d["rear_pilot"] / 2 if d["rear_pilot"] > 0 else 1e9
    out = []
    for lo, hi, r in bands:
        for a, b, cap in ((oz, 1e9, above_out), (hz, oz, between),
                          (rz, hz, r), (-1e9, rz, below_rear)):
            s0, s1 = max(lo, a), min(hi, b)
            if s1 - s0 > 1e-6:
                out.append([round(s0, 3), round(s1, 3), round(min(r, cap), 3)])
    return sorted(out)


if __name__ == "__main__":
    out = {}
    for key, f in STEPS.items():
        p = os.path.join(HERE, "..", "STEP", "purchased", f + ".step")
        out[key] = envelope(p)
        out[key]["bands"] = clip(out[key]["bands"], IFACE[key])
        bands = out[key]["bands"]
        print("%-9s z %.1f..%.1f, %d bands, max r %.1f mm"
              % (key, out[key]["z0"], out[key]["z1"], len(bands),
                 max(b[2] for b in bands)))
    dst = os.path.join(HERE, "..", "src", "lib", "_envelope.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)
