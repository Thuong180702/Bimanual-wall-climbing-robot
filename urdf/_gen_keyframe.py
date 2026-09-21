"""Find and verify the v2 hanging keyframe, the way v1's was verified.

Put both hooks on rung_front_line_0 with the arms in the latching posture,
drop the robot onto them, let the servos settle it, then hold for 5 s and
measure the drift. Writes the settled qpos/ctrl back into scene.xml.
"""
import os, sys
import numpy as np, mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "scene.xml")
FLEX, WR = np.radians(-160.0), np.radians(5.0)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
J = [m.joint(i).name for i in range(m.njnt)
     if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
QA = {n: m.joint(n).qposadr[0] for n in J}

POSE = {"shoulder_L_flexion": FLEX, "shoulder_R_flexion": FLEX,
        "wrist_L_flexion": -WR, "wrist_R_flexion": WR}


def place():
    mujoco.mj_resetData(m, d)
    for n, v in POSE.items():
        d.qpos[QA[n]] = v
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    # bring the mid-point of the two grip pads onto the rung axis
    g = m.geom("rung_front_line_0")
    rung = d.geom_xpos[g.id].copy()
    # ...at y = 0, not at the rung's own centre. Every lesson scene in rl/ is
    # laid out about y = 0 and its start rung reaches only a little past it;
    # settling at the rung's midpoint (y = 0.30) put the left hook 47 mm off
    # the end of lesson 1.35's start rung and the robot fell.
    rung[1] = 0.0
    pads = 0.5 * (d.site_xpos[m.site("s_grip_L").id]
                  + d.site_xpos[m.site("s_grip_R").id])
    d.qpos[0:3] += rung - pads
    d.qpos[2] -= 0.004                      # drop the throat onto the bar
    mujoco.mj_forward(m, d)


def hold(seconds, ctrl):
    n = int(seconds / m.opt.timestep)
    for _ in range(n):
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)


if __name__ == "__main__":
    place()
    ctrl = np.array([d.qpos[QA[m.joint(m.actuator_trnid[i][0]).name]]
                     for i in range(m.nu)])
    hold(35.0, ctrl)   # the pendulum needs ~20 s to stop swinging; after that it
                       # holds to 0.04 mm per 5 s, which is what v1 reported as 0.0
    q0 = d.qpos.copy()
    p0 = d.body("base_link").xpos.copy()
    hold(5.0, ctrl)
    p1 = d.body("base_link").xpos.copy()
    drift = np.linalg.norm(p1 - p0)
    ncon = d.ncon
    print("settled base at (%.4f %.4f %.4f)" % tuple(p0))
    print("drift over 5 s: %.4f mm      contacts: %d" % (drift * 1000, ncon))
    for s in "LR":
        f = d.sensor("ft_force_%s" % s).data
        print("  hand %s force %7.2f N   grip pad %6.2f N"
              % (s, np.linalg.norm(f), d.sensor("grip_%s" % s).data[0]))
    print("  weight %.2f N" % (m.body_subtreemass[m.body("base_link").id] * 9.81))
    if "--write" in sys.argv and drift < 0.01:
        key = ('  <keyframe>\n'
               '    <!-- Both hooks latched on rung_front_line_0 (z = 1.00). -->\n'
               '    <key name="hang_on_rung"\n'
               '         qpos="%s"\n'
               '         ctrl="%s"/>\n'
               '  </keyframe>\n'
               % (" ".join("%.6f" % v for v in q0),
                  " ".join("%.6f" % v for v in ctrl)))
        s = open(XML).read()
        import re
        s = re.sub(r"  <keyframe>.*?</keyframe>\n", "", s, flags=re.S)
        s = s.replace("</mujoco>", key + "</mujoco>")
        open(XML, "w").write(s)
        print("keyframe written into scene.xml")
