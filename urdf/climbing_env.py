"""Runtime helpers for the bimanual wall-climbing robot in MuJoCo.

Sensor access, latch detection and episode-termination logic.
All thresholds here were measured on this model, not guessed - see README.
"""
import numpy as np, mujoco

SLAM_SIG = 0.02   # m, one sigma per axis per reading: what the lidar SLAM's base position
                  # is off by. ponytail: white noise; a real SLAM drifts slowly instead,
                  # calibrate this (and the RL task's SLAM_TAU) against the real stack

# --- measured on this model -------------------------------------------------
GRIP_BAND = (np.radians(-40), np.radians(-10))   # finger angle where pull-off > 1200 N
GRIP_SET  = np.radians(-25)                      # recommended hold-on setpoint
GRIP_CLIFF = np.radians(-45)                     # beyond this the hook levers itself off
THROAT_X = (18.0, 48.0)                          # rung seat inside the hook, mm
THROAT_Z = (-92.0, -68.0)
CAPTURE_DX = 0.025                               # placement tolerance, m
CAPTURE_DZ = (-0.010, 0.070)


class ClimbingEnv:
    def __init__(self, xml='scene.xml', fall_z=-1.2, free_fall_s=0.6,
                 success_z=2.00, timeout_s=60.0, qvel_max=100.0, held_fn=None):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.fall_z, self.free_fall_s = fall_z, free_fall_s
        # What counts as "still attached to the wall", for the `released` timer only.
        # Defaults to `latched`, which is what the classical controller wants. But
        # `latched` is a tight box round the nominal seat, and measured over an
        # episode it reads False on 32 % of the steps -- 89 % of those with a hook
        # carrying more than 20 N. That kills any one-handed phase, which is exactly
        # the phase a climbing gait is made of, so the RL task passes its own.
        self.held_fn = held_fn or (lambda: self.latched('L')[0] or self.latched('R')[0])
        self.success_z, self.timeout_s, self.qvel_max = success_z, timeout_s, qvel_max
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self._slam_rng = np.random.default_rng()
        self.hook = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hook_'+s)
                     for s in 'LR'}
        self.rungs = [g for g in range(self.model.ngeom)
                      if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                          or '').startswith('rung_')]
        self._no_grip_t = 0.0

    # ---- sensors ----------------------------------------------------------
    def sensor(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if i < 0:
            raise KeyError(name)
        a = self.model.sensor_adr[i]
        return self.data.sensordata[a:a + self.model.sensor_dim[i]].copy()

    def encoders(self):
        """joint angle / rate / motor torque for all 17 drives"""
        out = {}
        for i in range(self.model.nu):
            j = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)[4:]
            out[j] = (self.sensor('enc_'+j)[0], self.sensor('vel_'+j)[0],
                      self.sensor('tau_'+j)[0])
        return out

    def slam(self):
        """where the lidar SLAM says the base is (world, m): the true position with
        SLAM_SIG of noise. The one world-referenced sensor on the real robot -- and
        without one, two hooks on a smooth rung cannot tell that the loaded hook is
        creeping along it (rl/climb_task.py, `_sense`)."""
        return self.data.xpos[self.base] + self._slam_rng.normal(0.0, SLAM_SIG, 3)

    def imu(self, raw_only=False):
        """9-DOF IMU. raw_only=True returns just what real hardware outputs."""
        out = dict(accel=self.sensor('imu_acc'), gyro=self.sensor('imu_gyro'),
                   mag=self.sensor('imu_mag'))
        if not raw_only:                      # ground truth - needs an estimator on hardware
            out.update(quat=self.sensor('imu_quat'), vel=self.sensor('imu_vel'),
                       angvel=self.sensor('imu_angvel'), angacc=self.sensor('imu_angacc'),
                       linacc=self.sensor('imu_linacc'), pos=self.sensor('torso_pos'))
        return out

    def com(self):
        """whole-robot centre of mass - the quantity a climbing controller steers"""
        return dict(pos=self.sensor('com_pos'), vel=self.sensor('com_vel'),
                    angmom=self.sensor('com_angmom'))

    def at_limit(self, tol=1.0):
        """joints currently jammed against a hard stop -> {joint: constraint torque}"""
        out = {}
        for i in range(self.model.nu):
            j = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)[4:]
            f = self.sensor('lim_'+j)[0]
            if abs(f) > tol:
                out[j] = f
        return out

    def hand(self, side):
        return dict(force=self.sensor('ft_force_'+side),
                    torque=self.sensor('ft_torque_'+side),
                    pad=self.sensor('grip_'+side)[0])

    # ---- latch detection --------------------------------------------------
    def seat(self, side, geom):
        """nearest point of a rung's axis, in hook-local mm"""
        d, mo = self.data, self.model
        p = d.geom_xpos[geom]; ax = d.geom_xmat[geom].reshape(3, 3)[:, 2]
        half = mo.geom_size[geom][1]; hp = d.xpos[self.hook[side]]
        t = np.clip(np.dot(hp - p, ax), -half, half)
        return (d.xmat[self.hook[side]].reshape(3, 3).T @ ((p + t * ax) - hp)) * 1000.0

    def latched(self, side):
        """(bool, geom_id) - is a hold seated inside this hook's throat?"""
        for g in self.rungs:
            s = self.seat(side, g)
            if THROAT_X[0] <= s[0] <= THROAT_X[1] and THROAT_Z[0] <= s[2] <= THROAT_Z[1]:
                return True, g
        return False, -1

    def grip_ok(self, side):
        """seated AND the finger is inside the band that actually holds"""
        ok, _ = self.latched(side)
        a = self.sensor('enc_finger_'+side)[0]
        return ok and GRIP_BAND[0] <= a <= GRIP_BAND[1]

    # ---- episode control --------------------------------------------------
    def reset(self, keyframe=0):
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe)
        mujoco.mj_forward(self.model, self.data)
        self._no_grip_t = 0.0
        return self.data

    def step(self, ctrl=None):
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        held = self.held_fn()
        self._no_grip_t = 0.0 if held else self._no_grip_t + self.model.opt.timestep
        return self.data

    def terminated(self):
        """(done, reason). Reasons: blowup, fell, released, timeout, success."""
        d = self.data
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            return True, 'blowup'
        if np.max(np.abs(d.qvel)) > self.qvel_max:
            return True, 'blowup'
        if d.xpos[self.base][2] < self.fall_z:
            return True, 'fell'
        if self._no_grip_t >= self.free_fall_s:
            return True, 'released'
        if d.xpos[self.base][2] > self.success_z:
            return True, 'success'
        if d.time > self.timeout_s:
            return True, 'timeout'
        return False, ''
