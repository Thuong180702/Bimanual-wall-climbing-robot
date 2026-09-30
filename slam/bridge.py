#!/usr/bin/env python3
"""The simulated Mid-360s' side of FAST-LIO, inside the container (Python 3.8, rospy).

A TCP server on port 5555 that the simulation (rl/slam_test.py, on the host) talks to in
lock-step, one frame at a time. It publishes what the Livox driver publishes -- the
IMU on /livox/imu and each 100 ms frame on /livox/lidar as a livox_ros_driver/CustomMsg
-- then waits for FAST-LIO's answer for that frame and sends it back: the IMU's pose
(/Odometry) and the frame's points in the map (/cloud_registered). FAST-LIO itself is a
child process, restarted on every reset: a SLAM starts from nothing each episode.

Every message both ways is an 8-byte little-endian length and an .npz:
  -> op=reset                                  <- ok
  -> op=frame, t0 (s, the frame's first point), imu (M, 7: t, gyro, accel),
     pts (19-byte Livox points, raw), n       <- ok, t (the frame's end), pos (3,),
                                                 quat (4, w x y z), cloud (K, 3)
"""
import io, os, signal, socket, struct, subprocess, threading, time
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from livox_ros_driver.msg import CustomMsg

PORT = 5555
T_BASE = 1000.0          # s added to the simulation's clock: ROS time 0 means "unset"
WAIT = 5.0               # s to wait for FAST-LIO's answer to a frame; none comes for the
                         # first frame and while its IMU initialises


def recv(conn):
    n = struct.unpack('<Q', _exact(conn, 8))[0]
    return dict(np.load(io.BytesIO(_exact(conn, n))))


def _exact(conn, n):
    b = bytearray()
    while len(b) < n:
        c = conn.recv(min(1 << 20, n - len(b)))
        if not c:
            raise ConnectionError('closed')
        b += c
    return bytes(b)


def send(conn, **kw):
    f = io.BytesIO()
    np.savez(f, **{k: np.asarray(v) for k, v in kw.items()})
    conn.sendall(struct.pack('<Q', f.tell()) + f.getvalue())


class RawCustomMsg(CustomMsg):
    """a CustomMsg whose points are already serialised: 80 000 CustomPoint objects a
    frame are most of a second of Python, the bytes are the same"""
    __slots__ = ['raw', 'n']

    def serialize(self, buff):
        h = self.header
        fid = h.frame_id.encode()
        buff.write(struct.pack('<3I', h.seq, h.stamp.secs, h.stamp.nsecs))
        buff.write(struct.pack('<I', len(fid)) + fid)
        buff.write(struct.pack('<QIB3BI', self.timebase, self.n, 0, 0, 0, 0, self.n))
        buff.write(self.raw)


class Bridge:
    def __init__(self):
        rospy.init_node('sim_bridge')
        self.imu = rospy.Publisher('/livox/imu', Imu, queue_size=None)       # synchronous
        self.lidar = rospy.Publisher('/livox/lidar', CustomMsg, queue_size=None)
        self.got = {}
        self.cv = threading.Condition()
        rospy.Subscriber('/Odometry', Odometry, self._odom, queue_size=100)
        rospy.Subscriber('/cloud_registered', PointCloud2, self._cloud, queue_size=100)
        self.proc = None

    def _put(self, stamp, key, val):
        with self.cv:
            self.got.setdefault(round(stamp.to_sec(), 6), {})[key] = val
            self.cv.notify_all()

    def _odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self._put(m.header.stamp, 'pose', (np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z])))

    def _cloud(self, m):
        off = {f.name: f.offset for f in m.fields}
        raw = np.frombuffer(m.data, np.uint8).reshape(-1, m.point_step)
        xyz = np.stack([raw[:, off[c]:off[c] + 4].copy().view('<f4')[:, 0] for c in 'xyz'], 1)
        self._put(m.header.stamp, 'cloud', xyz.astype(np.float64))

    def reset(self):
        if self.proc:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        with self.cv:
            self.got.clear()
        self.proc = subprocess.Popen(['rosrun', 'fast_lio', 'fastlio_mapping'],
                                     stdout=open('/tmp/fastlio.log', 'w'), stderr=subprocess.STDOUT)
        t = time.time()
        while self.lidar.get_num_connections() == 0 or self.imu.get_num_connections() == 0:
            if time.time() - t > 30:
                raise RuntimeError('FAST-LIO never subscribed; see /tmp/fastlio.log')
            time.sleep(0.05)

    def frame(self, t0, imu, pts, n):
        for r in imu:
            m = Imu()
            m.header.stamp = rospy.Time.from_sec(T_BASE + r[0])
            m.header.frame_id = 'body'
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = r[1:4]
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = r[4:7]
            m.orientation_covariance[0] = -1.0
            self.imu.publish(m)
        m = RawCustomMsg()
        m.header.stamp = rospy.Time.from_sec(T_BASE + t0)
        m.header.frame_id = 'livox_frame'
        m.timebase = int(round((T_BASE + t0) * 1e9))
        m.raw, m.n = pts.tobytes(), int(n)
        m.point_num = m.n
        self.lidar.publish(m)
        last = int(np.frombuffer(m.raw[:19 * m.n], np.uint8).reshape(-1, 19)[-1, :4].copy().view('<u4')[0]) if m.n else 0
        end = round(rospy.Time.from_sec(T_BASE + t0).to_sec() + last / 1e9, 6)
        deadline = time.time() + WAIT
        with self.cv:
            while True:
                hit = [k for k, v in self.got.items() if abs(k - end) < 2e-4 and len(v) == 2]
                if hit or time.time() > deadline:
                    break
                self.cv.wait(0.05)
            if not hit:
                return dict(ok=0, t=end - T_BASE)
            v = self.got.pop(hit[0])
            for k in [k for k in self.got if k < end]:           # nothing older is wanted
                del self.got[k]
        return dict(ok=1, t=hit[0] - T_BASE, pos=v['pose'][0], quat=v['pose'][1], cloud=v['cloud'])

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', PORT))
        srv.listen(1)
        rospy.loginfo('sim bridge listening on %d', PORT)
        while not rospy.is_shutdown():
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while True:
                    q = recv(conn)
                    op = str(q['op'])
                    if op == 'reset':
                        self.reset()
                        send(conn, ok=1)
                    elif op == 'frame':
                        send(conn, **self.frame(float(q['t0']), q['imu'], q['pts'], int(q['n'])))
                    else:
                        send(conn, ok=0, error=op)
            except ConnectionError:
                pass
            finally:
                conn.close()


if __name__ == '__main__':
    Bridge().serve()
