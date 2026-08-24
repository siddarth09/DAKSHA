"""Live Rerun view of everything a demo records: cameras, joints, and fingertip forces.

    ros2 run zero_control rerun_viewer --ros-args \
        --params-file install/zero_bringup/share/zero_bringup/config/rebot_control.yaml

WHY A VIEWER AND NOT JUST RVIZ. What matters while teleoperating is whether the DATA is good, and
that is a different question from whether the sim looks right. Three things go wrong invisibly:
a camera silently stops publishing (mujoco_ros2_control binds cameras to URDF <sensor> blocks BY
NAME, and a mismatch produces no error and no images); the arm stops tracking the operator, so the
recorded action is not what the arm did; and a grasp loads one pad and not the other, which is how
the tool-point bug presented. All three are obvious in a time series and invisible in a 3D view.

Everything is logged on one shared timeline, so a force spike can be lined up against the frame
that produced it -- which is the whole point of looking at them together.
"""

from __future__ import annotations

import numpy as np
import rclpy
import rerun as rr
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from zero_control.action import SIDES, unpack


def _stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class RerunViewer(Node):
    def __init__(self) -> None:
        super().__init__("zero_rerun_viewer")
        self.declare_parameter("cameras", ["front", "left_wrist", "right_wrist"])
        self.declare_parameter("ft_sensors", [""])
        self.declare_parameter("depth", False)
        self.declare_parameter("arm_joints", [""])       # flat list, for labelling the plots

        cams = list(self.get_parameter("cameras").value)
        fts = [s for s in self.get_parameter("ft_sensors").value if s]
        self.show_depth = bool(self.get_parameter("depth").value)

        rr.init("zero_teleop", spawn=True)
        # A blueprint would pin the layout, but the default auto-layout already groups by entity
        # path, so keep the paths meaningful instead: camera/<name>, joint/<name>, force/<name>.
        rr.log("/", rr.TextDocument(
            "ZERO teleop monitor\n"
            f"cameras: {', '.join(cams)}\n"
            f"fingertip F/T: {len(fts)}\n"
            "joint/* = measured position, force/* = fingertip wrench magnitude"),
            static=True)

        self.t0: float | None = None
        for cam in cams:
            self.create_subscription(
                Image, f"/zero/{cam}/image_raw",
                lambda m, c=cam: self._on_image(m, c), 5)
            if self.show_depth:
                self.create_subscription(
                    Image, f"/zero/{cam}/depth",
                    lambda m, c=cam: self._on_depth(m, c), 5)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        for s in fts:
            self.create_subscription(
                WrenchStamped, f"/{s}_broadcaster/wrench",
                lambda m, k=s: self._on_wrench(m, k), 10)
        self.create_subscription(Float64MultiArray, "/zero/eef_target", self._on_target, 10)
        self.create_subscription(Float64MultiArray, "/zero/ik_status", self._on_status, 10)
        self.get_logger().info(
            f"rerun viewer up: {len(cams)} cameras, {len(fts)} F/T sensors, "
            f"depth={'on' if self.show_depth else 'off'}")

    # ---------------------------------------------------------------- timeline
    def _t(self, header=None) -> None:
        """One shared timeline for every stream, so spikes line up with frames."""
        t = _stamp_s(header) if header is not None else \
            self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = t
        rr.set_time("ros", duration=t - self.t0)

    # ---------------------------------------------------------------- callbacks
    def _on_image(self, msg: Image, cam: str) -> None:
        self._t(msg.header)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
        rr.log(f"camera/{cam}", rr.Image(img))

    def _on_depth(self, msg: Image, cam: str) -> None:
        self._t(msg.header)
        # 32FC1 metres is what mujoco_ros2_control publishes; 16UC1 would be millimetres.
        dtype = np.float32 if msg.encoding == "32FC1" else np.uint16
        d = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        rr.log(f"camera/{cam}_depth", rr.DepthImage(d, meter=1.0 if dtype is np.float32 else 1000))

    def _on_joints(self, msg: JointState) -> None:
        self._t(msg.header)
        for name, pos in zip(msg.name, msg.position):
            rr.log(f"joint/{name}", rr.Scalars(float(pos)))

    def _on_wrench(self, msg: WrenchStamped, sensor: str) -> None:
        self._t(msg.header)
        f = msg.wrench.force
        v = np.array([f.x, f.y, f.z])
        # Magnitude is the number that tells you whether a pad is loaded; the axes are there for
        # when it matters which way.
        rr.log(f"force/{sensor}", rr.Scalars(float(np.linalg.norm(v))))
        for axis, val in zip("xyz", v):
            rr.log(f"force_axis/{sensor}/{axis}", rr.Scalars(float(val)))

    def _on_target(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 20:
            return
        self._t()
        poses, grips = unpack(np.asarray(msg.data, dtype=float))
        for side in SIDES:
            pos, _ = poses[side]
            for axis, val in zip("xyz", pos):
                rr.log(f"target/{side}/{axis}", rr.Scalars(float(val)))
            rr.log(f"target/{side}/grip", rr.Scalars(float(grips[side])))

    def _on_status(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 6:
            return
        self._t()
        # Tracking error. If this is not near zero the recorded action is not what the arm did,
        # which makes the frame worse than useless -- it is mislabelled.
        for i, side in enumerate(SIDES):
            rr.log(f"ik/{side}/pos_err_mm", rr.Scalars(float(msg.data[3 * i] * 1000)))
            rr.log(f"ik/{side}/rot_err_deg",
                   rr.Scalars(float(np.degrees(msg.data[3 * i + 1]))))


def main() -> None:
    rclpy.init()
    node = RerunViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
