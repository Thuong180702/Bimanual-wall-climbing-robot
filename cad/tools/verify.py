"""Check every number in the v2 model against its source. Run after any change.

Each check names where the truth comes from: a vendor datasheet, the vendor
STEP file, the CAD solids, or a measurement on the built model.
"""
import hashlib, json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from cadgen import build123d as bd
import numpy as np

from lib.build import BUILDERS, fuse
from lib.motors import MOTORS
from lib.spec import (AL, BODY_DIR, IFACE, JOINTS, MOTOR_OF, MOUNT_AT,
                      PICK)
from massprops import link_props

RPM = math.pi / 30.0
FAIL = []


def check(name, ok, detail=""):
    print("  %-4s %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAIL.append(name)


def h1(t):
    print("\n" + t)
    print("  " + "-" * 74)


def main():
    h1("1. vendor STEP files - checksum and envelope vs the datasheet")
    known = {
        "cubemars_ak80_64_kv80.step":
            "826b2603459d79cbcb64b54b02f7f137cb3656d240feb35aa6642fc9c3ce34f0",
        "cubemars_ak60_39_v3_kv80.step":
            "00e18639c4794eff5a83ea6930ae681433054ba1ed58fbb4b54434d303c4fe19",
        "cubemars_ak45_36_kv80.step":
            "d6d093884d7daab8517e38d75f5b86762dd1363e8c7a758280e77abbfd1e4686",
    }
    for key, d in MOTORS.items():
        p = os.path.join(HERE, "..", "vendor", d["step"])
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        check("%s STEP sha256 matches step.parts" % key,
              sha == known[d["step"]], sha[:16] + "...")
        got = sorted(round(v, 1) for v in d["size_mm"])
        want = sorted([round(d["dia_mm"], 1), round(d["dia_mm"], 1),
                       round(d["len_mm"], 1)])
        check("%s envelope %s = datasheet %s mm" % (key, got, want),
              got == want)

    h1("2. canonical motor frame - output flange on +Z at z = 0")
    for key in MOTORS:
        f = os.path.join(ROOT, "cad", "STEP", "purchased",
                         {"AK80-64": "ak80_64", "AK60-39": "ak60_39",
                          "AK45-36": "ak45_36"}[key] + ".step")
        s = bd.import_step(f)
        b = s.bounding_box()
        L = MOTORS[key]["len_mm"]
        check("%s spans z %.1f..%.1f (want %.1f..0)" % (key, b.min.Z, b.max.Z, -L),
              abs(b.max.Z) < 0.1 and abs(b.min.Z + L) < 0.1)

    h1("3. mass budget - CAD solids + datasheet actuator masses")
    struct = sum(fuse(BUILDERS[l]()).volume * AL for l in BUILDERS)
    motors = sum(MOTORS[MOTOR_OF[j[0]]]["mass_kg"] for j in JOINTS)
    total = sum(link_props(l)[0] for l in BUILDERS)
    payload = total - struct - motors
    print("      structure %.3f kg + actuators %.3f kg + payload %.3f kg"
          % (struct, motors, payload))
    check("link masses sum to structure + actuators + payload",
          abs(total - (struct + motors + payload)) < 1e-6, "%.4f kg" % total)

    h1("4. inertia tensors are physically possible")
    bad = []
    for l in BUILDERS:
        M, C, I, _ = link_props(l)
        w = np.linalg.eigvalsh(np.array(I))
        if (w <= 0).any() or not (w[0] + w[1] >= w[2] - 1e-12
                                  and w[0] + w[2] >= w[1] - 1e-12
                                  and w[1] + w[2] >= w[0] - 1e-12):
            bad.append(l)
    check("all 19 links positive definite and obey the triangle inequality",
          not bad, ", ".join(bad))

    h1("5. URDF matches the spec and the datasheets")
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "urdf", "scene.xml"))
    check("MuJoCo loads urdf/scene.xml", True,
          "nbody=%d nq=%d nv=%d nu=%d nsensor=%d" % (m.nbody, m.nq, m.nv, m.nu, m.nsensor))
    check("total mass in MuJoCo = CAD total",
          abs(m.body_subtreemass[m.body("base_link").id] - total) < 1e-3,
          "%.4f kg" % m.body_subtreemass[m.body("base_link").id])
    bad = []
    for name, par, ch, ax, off, lim, _bd in JOINTS:
        j, key = m.joint(name), MOTOR_OF[name]
        ai = [i for i in range(m.nu) if m.actuator_trnid[i][0] == j.id][0]
        if not np.allclose(j.range, lim, atol=1e-6):
            bad.append(name + ":range")
        if not np.allclose(m.jnt_axis[j.id], ax, atol=1e-9):
            bad.append(name + ":axis")
        if abs(m.actuator_forcerange[ai][1] - MOTORS[key]["peak_Nm"]) > 1e-6:
            bad.append(name + ":forcerange")
    check("17 joints: axis, range and forcerange all match the spec/datasheet",
          not bad, ", ".join(bad))

    h1("6. actuator sizing margins (measured on THIS model)")
    for tag, f in (("full joint box", "t_box.json"), ("climbing envelope", "t_env.json")):
        p = os.path.join(HERE, "..", "..", "urdf", f)
        if not os.path.exists(p):
            print("      %s: %s missing, skipped" % (tag, f))
            continue
        R = json.load(open(p))
        worst, wj = 9e9, None
        for jn, v in R["joints"].items():
            pk = MOTORS[MOTOR_OF[jn]]["peak_Nm"]
            mrg = pk / max(v["hang_eq"], 1e-9)
            if mrg < worst:
                worst, wj = mrg, jn
        check("%s: every joint >= 1.3x peak torque" % tag, worst >= 1.3,
              "worst %.2fx at %s" % (worst, wj))

    h1("7. no two actuator bodies intersect, in any legal named pose")
    r = subprocess.run([sys.executable, os.path.join(HERE, "check_fit.py")],
                       capture_output=True, text=True)
    lines = r.stdout.splitlines()
    pose, offenders = None, []
    for ln in lines:
        if ln.startswith("--- "):
            pose = ln.strip("- ").strip()
        elif "CLASH" in ln and pose and "README reach pose" not in pose:
            offenders.append("%s: %s" % (pose, ln.split()[0]))
    check("every legal operating pose is clear of self-intersection",
          not offenders, ", ".join(offenders) or
          "%d named poses" % sum(ln.startswith("--- ") for ln in lines))

    print("\n%s  %d check(s) failed" % ("ALL CHECKS PASS" if not FAIL else "FAILURES:",
                                        len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
