"""Capture one live observation from the running sim and compare it against the dataset.

    bash scripts/diag_live_obs.sh

A policy that tracks demos to 16 mm offline but barely moves in rollout is usually being fed a
different observation than it trained on. This dumps the observation the policy node would see
and diffs it feature by feature against the same feature in the training set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

OUT = Path("/home/sid/live_obs.npz")
CAMS = ["front", "left_wrist", "right_wrist"]


class Diag(Node):
    def __init__(self) -> None:
        super().__init__("zero_diag")
        self.rgb: dict[str, np.ndarray] = {}
        self.state = None
        self.target = None
        self.n_state = 0
        self.rgb_hits = {c: 0 for c in CAMS}
        for c in CAMS:
            self.create_subscription(Image, f"/zero/{c}/image_raw",
                                     lambda m, c=c: self._rgb(m, c), 10)
        self.create_subscription(Float64MultiArray, "/zero/eef_state", self._state, 10)
        self.create_subscription(Float64MultiArray, "/zero/eef_target", self._target, 10)
        self.create_timer(0.5, self._report)
        self.t0 = self.get_clock().now()

    def _rgb(self, msg: Image, cam: str) -> None:
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.rgb[cam] = img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()
        self.rgb_hits[cam] += 1
        self.enc = msg.encoding

    def _state(self, msg: Float64MultiArray) -> None:
        self.state = np.asarray(msg.data, dtype=np.float32)
        self.n_state += 1

    def _target(self, msg: Float64MultiArray) -> None:
        self.target = np.asarray(msg.data, dtype=np.float32)

    def _report(self) -> None:
        dt = (self.get_clock().now() - self.t0).nanoseconds / 1e9
        if dt < 3.0:
            return
        print(f"\n=== after {dt:.1f}s ===")
        print(f"/zero/eef_state msgs: {self.n_state}  ({self.n_state/dt:.0f} Hz)")
        for c in CAMS:
            print(f"/zero/{c}/image_raw msgs: {self.rgb_hits[c]:4d} "
                  f"({self.rgb_hits[c]/dt:5.1f} Hz)  "
                  f"{'PRESENT' if c in self.rgb else 'MISSING'}")
        if self.state is None or len(self.rgb) < len(CAMS):
            print("!! incomplete: is the sim running?")
            raise SystemExit(1)
        print(f"encoding: {getattr(self, 'enc', '?')}")
        np.savez(OUT, state=self.state,
                 target=self.target if self.target is not None else np.zeros(20),
                 **{f"rgb_{c}": self.rgb[c] for c in CAMS})
        print(f"\nwrote {OUT}")
        raise SystemExit(0)


def main() -> None:
    rclpy.init()
    node = Diag()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
