"""Servo-accuracy regression test: command a known EEF offset, report the residual.

    ros2 run zero_control servo_test --ros-args \
        --params-file install/zero_bringup/share/zero_bringup/config/rebot_control.yaml \
        -p dz:=0.04 -p seconds:=6.0

WHY THIS EXISTS AS A SCRIPT AND NOT AS A SHELL ONE-LINER. The first time I checked servo accuracy
I hand-packed the 20-value target on the command line and got a 574 mm residual on the right arm
that no offline test could reproduce -- the solver, the joint limits and the MuJoCo closed loop
were all provably fine. A hand-built action vector is an untested code path that only ever runs
once, so any error in it is indistinguishable from a controller fault. This builds the target with
`action.pack`, the same function the teleop and the dataset writer use, so what gets tested is the
path that ships.

The target is CURRENT measured FK plus an offset, so it is reachable by construction and the
residual is attributable to the servo rather than to the goal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from zero_control.action import SIDES, pack, rot_to_6d
from zero_control.ik import ArmIK


class ServoTest(Node):
    def __init__(self) -> None:
        super().__init__("zero_servo_test")
        self.declare_parameter("robot", "rebot")
        self.declare_parameter("eef_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("seconds", 6.0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("grip", 1.0)
        # Which hands actually get the offset. The action vector always carries BOTH hands, so
        # eef_control always commands both -- moving one arm at a time means sending the other
        # its own current pose, not omitting it.
        self.declare_parameter("only", "both")
        # Absolute goals. "" = use the dx/dy/dz delta instead; "pick" / "place" servo to the task
        # landmarks, whose positions arrive as params measured from the compiled scene, so no
        # coordinate is ever restated here.
        self.declare_parameter("left_goal", "")
        self.declare_parameter("right_goal", "")
        self.declare_parameter("goal_dz", 0.0)
        self.declare_parameter("pick_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("place_xyz", [0.0, 0.0, 0.0])
        for axis in ("dx", "dy", "dz"):
            self.declare_parameter(axis, 0.0)
        for side in SIDES:
            self.declare_parameter(f"{side}_arm_joints", [""])
            self.declare_parameter(f"{side}_eef_frame", "")

        key = self.get_parameter("robot").value
        urdf = str(Path(get_package_share_directory("zero_description"))
                   / "urdf" / f"zero_{key}.urdf")
        off = tuple(self.get_parameter("eef_offset").value)
        self.delta = np.array([self.get_parameter(a).value for a in ("dx", "dy", "dz")])

        self.ik = {s: ArmIK(urdf, self.get_parameter(f"{s}_eef_frame").value,
                            list(self.get_parameter(f"{s}_arm_joints").value), off)
                   for s in SIDES}
        self.q = np.zeros(self.ik["left"].model.nq)
        self.target: np.ndarray | None = None
        self.start: dict[str, pin.SE3] = {}
        self.goal: dict[str, np.ndarray] = {}
        self.trace: list[tuple[float, list[float]]] = []
        self.t0 = 0.0

        self.pub = self.create_publisher(Float64MultiArray, "/zero/eef_target", 10)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(Float64MultiArray, "/zero/ik_status", self._on_status, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.deadline = float(self.get_parameter("seconds").value)

    def _on_js(self, msg: JointState) -> None:
        model = self.ik["left"].model
        for name, position in zip(msg.name, msg.position):
            jid = model.getJointId(name)
            if jid < model.njoints:
                self.q[model.joints[jid].idx_q] = position
        if self.target is None:
            self._build_target()

    def _build_target(self) -> None:
        poses, grips = {}, {}
        grip = float(self.get_parameter("grip").value)
        only = str(self.get_parameter("only").value)
        landmark = {"pick": np.asarray(self.get_parameter("pick_xyz").value, dtype=float),
                    "place": np.asarray(self.get_parameter("place_xyz").value, dtype=float)}
        dz = float(self.get_parameter("goal_dz").value)
        for side in SIDES:
            T = self.ik[side].fk(self.q)
            self.start[side] = T
            move = only in ("both", side)
            named = str(self.get_parameter(f"{side}_goal").value)
            if named and move:
                # Absolute goal. Orientation is held at the measured one: the home poses are
                # already solved to point the gripper down at the task volume, so re-aiming here
                # would only fight that.
                goal = landmark[named] + np.array([0.0, 0.0, dz])
                note = f"   [{named}{f' +{dz*100:.0f}cm' if dz else ''}]"
            else:
                goal = T.translation + (self.delta if move else np.zeros(3))
                note = "" if move else "   (HOLD)"
            poses[side] = (goal, T.rotation)
            grips[side] = grip
            self.goal[side] = goal
            self.get_logger().info(
                f"{side:5} measured {np.round(T.translation, 4)} -> "
                f"target {np.round(goal, 4)}{note}")
        self.target = pack(poses, grips)
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

    def _on_status(self, msg: Float64MultiArray) -> None:
        if self.target is not None:
            t = self.get_clock().now().nanoseconds * 1e-9 - self.t0
            self.trace.append((t, list(msg.data)))

    def _tick(self) -> None:
        if self.target is None:
            return
        self.pub.publish(Float64MultiArray(data=[float(v) for v in self.target]))
        if self.get_clock().now().nanoseconds * 1e-9 - self.t0 > self.deadline:
            self.report()
            raise SystemExit(0)

    def report(self) -> None:
        cmd = np.linalg.norm(self.delta)
        print(f"\ncommanded offset {np.round(self.delta, 4)} = {cmd * 1000:.1f} mm\n")
        print(f"{'t[s]':>6} {'Lpos[mm]':>9} {'Lrot[deg]':>10} {'Lclamp':>7} "
              f"{'Rpos[mm]':>9} {'Rrot[deg]':>10} {'Rclamp':>7}")
        for t, s in self.trace[:: max(1, len(self.trace) // 12)]:
            print(f"{t:6.2f} {s[0]*1000:9.2f} {np.degrees(s[1]):10.3f} {s[2]:7.0f} "
                  f"{s[3]*1000:9.2f} {np.degrees(s[4]):10.3f} {s[5]:7.0f}")
        if not self.trace:
            print("NO ik_status RECEIVED -- is eef_control running?")
            return
        final = self.trace[-1][1]
        # Achieved displacement, measured independently of the IK's own residual.
        for i, side in enumerate(SIDES):
            T = self.ik[side].fk(self.q)
            moved = T.translation - self.start[side].translation
            gap = np.linalg.norm(T.translation - self.goal[side]) * 1000
            print(f"\n{side:5} residual {final[3*i]*1000:7.2f} mm / "
                  f"{np.degrees(final[3*i+1]):6.3f} deg   "
                  f"achieved {np.round(moved, 4)} ({np.linalg.norm(moved)*1000:.1f} mm)"
                  f"\n      reached {np.round(T.translation, 4)}  goal "
                  f"{np.round(self.goal[side], 4)}  gap {gap:.2f} mm")
        worst = max(final[0], final[3])
        print(f"\n{'PASS' if worst < 0.010 else 'FAIL'}  worst residual {worst*1000:.2f} mm "
              f"(tol 10 mm)")


def main() -> None:
    rclpy.init()
    node = ServoTest()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
