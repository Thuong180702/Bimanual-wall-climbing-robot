"""The anchor-rooted state estimate the policy observes through.

Nothing here reads the simulator's ground truth. The estimate is built from
    encoders  -> relative kinematics of the whole tree (exact, it is a rigid chain)
    IMU       -> world attitude (rl.climb_task filters it; the bare TRIAD is here)
    hold map  -> where the anchored hand is in the world
which is exactly what the hardware would have.

What is left of the classical controller (removed 2026-09-30; its IK, planner and
routing are in git history): the one piece the RL task was built on.
"""
import numpy as np, mujoco

# 17 drives, in actuator order (== qpos[7:] order)
JOINTS = ['waist_yaw',
          'shoulder_L_flexion', 'shoulder_L_abduction', 'shoulder_L_rotation', 'elbow_L',
          'forearm_L_roll', 'wrist_L_flexion', 'wrist_L_deviation', 'finger_L',
          'shoulder_R_flexion', 'shoulder_R_abduction', 'shoulder_R_rotation', 'elbow_R',
          'forearm_R_roll', 'wrist_R_flexion', 'wrist_R_deviation', 'finger_R']

# measured on the hang: where a rung settles inside the hook's throat
THROAT_PT = np.array([0.023, 0.0, -0.085])
MAG_WORLD = np.array([0.0, -0.5, 0.0])      # scene.xml magnetic field


def _frame(a, b):
    """orthonormal frame from two non-parallel vectors, a is kept exactly"""
    x = a / np.linalg.norm(a)
    z = np.cross(x, b)
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def triad(up_body, mag_body):
    """world<-body rotation from an accelerometer and a magnetometer. No filter, no state."""
    return _frame(np.array([0., 0., 1.]), MAG_WORLD) @ _frame(up_body, mag_body).T


def rotvec(R):
    """rotation matrix -> rotation vector (axis * angle)"""
    q, v = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q, R.flatten())
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class Estimator:
    """Scratch copy of the model driven by encoders only.

    Relative poses inside a scratch tree are correct whatever the base pose is, so we run
    it with the free joint at the origin, work out where the tree must be, and then re-run
    it placed there -- after which xpos / xmat are world estimates and collision queries
    mean something.
    """

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        # everything here indexes qpos[7:] by JOINTS; a model whose tree order
        # differs (the shoulder going pitch-first swapped two) scrambles it silently
        order = [model.joint(i).name for i in range(model.njnt)
                 if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        assert order == JOINTS, "model joint order %s is not JOINTS" % order
        self.hook = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hook_' + s)
                     for s in 'LR'}
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_upper')
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.anchor = 'L'

    def update(self, q, imu, anchor_side, anchor_pos, anchor_R=None, R_torso=None):
        """q: 17 encoder angles. imu: dict(accel=, mag=). anchor_pos: world point on the
        rung axis where the anchored hook's throat sits.

        Attitude comes from `R_torso` (the task's filtered attitude), else the bare IMU
        triad, else `anchor_R` -- the anchored hook's world orientation -- for a
        configuration the robot is not in: the latched hook's orientation is fixed by
        the rung, so it is the right invariant to hang a hypothetical posture off.

        """
        d = self.d
        d.qpos[:3] = 0.0
        d.qpos[3:7] = (1., 0., 0., 0.)
        d.qpos[7:] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.m, d)

        if anchor_R is not None:
            R_off = anchor_R @ d.xmat[self.hook[anchor_side]].reshape(3, 3).T
        elif R_torso is not None:
            # a filtered world<-torso attitude (climb_task's gyro complementary filter)
            R_off = R_torso @ d.xmat[self.torso].reshape(3, 3).T
        else:
            R_off = triad(imu['accel'], imu['mag']) @ d.xmat[self.torso].reshape(3, 3).T
        p_off = np.asarray(anchor_pos) - R_off @ self._throat(anchor_side)

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R_off @ d.xmat[self.base].reshape(3, 3)).flatten())
        d.qpos[:3] = p_off + R_off @ d.xpos[self.base]
        d.qpos[3:7] = quat
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        self.anchor = anchor_side

        return self

    # -- world estimates ----------------------------------------------------
    def _throat(self, side):
        b = self.hook[side]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ THROAT_PT

    def throat(self, side):
        return self._throat(side)

    def hook_R(self, side):
        return self.d.xmat[self.hook[side]].reshape(3, 3)

    def body_pos(self, body_id):
        return self.d.xpos[body_id].copy()

    def com(self):
        """whole-robot centre of mass (base_link's subtree is the whole robot)"""
        return self.d.subtree_com[self.base].copy()
