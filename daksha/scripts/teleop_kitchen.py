"""
DAKSHA — teleop the flat-terrain RL locomotion policy INSIDE the RoboCasa kitchen.

Reuses bheema_rl (the trained policy + Teleop) unchanged, with ONE adaptation:
in the combined kitchen+G1 scene the G1 floating base is no longer the first joint
(RoboCasa fixtures/objects come first), so we locate `floating_base_joint` by name
instead of assuming qpos[0:7] / qvel[0:6]. Everything else (99-dim obs, per-joint
PD/action-scale, name-based joint mapping) is inherited from bheema_rl.RLBridge.

Controls (same as bheema): arrows = walk/strafe, Z/X = turn, SPACE = stop, ESC = quit.

Run (needs a display):
    source ~/bheema_rl_env/bin/activate
    python teleop_kitchen.py
Headless self-check (no window):
    python teleop_kitchen.py --headless --seconds 4
"""
import os, sys, time, argparse
from pathlib import Path
import numpy as np
import mujoco as mj

# ── make bheema_rl importable and reuse it as-is ──
BHEEMA_RL = "/home/sid/projects25/src/bheema/bheema_rl"
sys.path.insert(0, BHEEMA_RL)
from rl_bridge import RLBridge, POLICY_JOINTS, DEFAULT_QPOS   # noqa: E402
from teleop import Teleop                                     # noqa: E402
from huggingface_hub import hf_hub_download                   # noqa: E402

HERE = Path(__file__).resolve().parent
SCENE = str(HERE.parent / "robots" / "unitree_g1_mjcf" / "kitchen_g1.xml")
SIM_HZ = 1000
POLICY_HZ = 50.0


class KitchenRLBridge(RLBridge):
    """RLBridge that reads the base state from `floating_base_joint` (by name),
    since the kitchen's joints precede G1's free joint in qpos/qvel."""

    def __init__(self, checkpoint_path, model, policy_hz=POLICY_HZ):
        super().__init__(checkpoint_path, model, policy_hz)
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if jid == -1:
            raise RuntimeError("floating_base_joint not found")
        self.base_qadr = model.jnt_qposadr[jid]   # pos [adr:adr+3], quat [adr+3:adr+7]
        self.base_vadr = model.jnt_dofadr[jid]     # linvel [adr:adr+3], angvel [adr+3:adr+6]
        print(f"[KitchenRLBridge] base qpos@{self.base_qadr} qvel@{self.base_vadr}")

    def build_observation(self, data, velocity_cmd):
        q, v = self.base_qadr, self.base_vadr
        w, x, y, z = data.qpos[q + 3:q + 7]
        R_bw = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
        R_wb = R_bw.T
        base_lin_vel = (R_wb @ data.qvel[v:v + 3]).astype(np.float32)
        base_ang_vel = data.qvel[v + 3:v + 6].astype(np.float32)
        proj_grav = (R_wb @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
        joint_pos = self._read_joint_pos(data) - DEFAULT_QPOS
        joint_vel = self._read_joint_vel(data)
        return np.concatenate([base_lin_vel, base_ang_vel, proj_grav,
                               joint_pos, joint_vel, self.last_action,
                               np.array(velocity_cmd, dtype=np.float32)])


def init_stand(model, data, rl):
    """Reset, place G1 base at its default (kitchen open-floor) pose, joints at DEFAULT_QPOS."""
    mj.mj_resetData(model, data)
    for i in range(29):
        data.qpos[rl.qpos_indices[i]] = DEFAULT_QPOS[i]
    mj.mj_forward(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="run without viewer (self-check)")
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()

    print("Loading scene:", SCENE)
    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    model.opt.timestep = 1.0 / SIM_HZ
    base_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")

    ckpt = hf_hub_download(repo_id="Siddarth09/bheema_locomotion", filename="model_59999.pt")
    rl = KitchenRLBridge(ckpt, model, policy_hz=POLICY_HZ)

    init_stand(model, data, rl)
    zad = rl.base_qadr + 2
    print(f"init base z = {data.qpos[zad]:.3f}")

    # warmup: hold stand with cmd=0
    for _ in range(int(0.5 * SIM_HZ)):
        rl.step(data, [0.0, 0.0, 0.0], data.time)
        rl.apply_control(data)
        mj.mj_step(model, data)
    print(f"after warmup: base z = {data.qpos[zad]:.3f}")

    if args.headless:
        # scripted check: 1s stand, then walk forward, print base z + xy drift
        x0 = data.qpos[rl.base_qadr:rl.base_qadr + 2].copy()
        n = int(args.seconds * SIM_HZ)
        for k in range(n):
            cmd = [0.0, 0.0, 0.0] if k < SIM_HZ else [0.6, 0.0, 0.0]
            rl.step(data, cmd, data.time)
            rl.apply_control(data)
            mj.mj_step(model, data)
            if k % (SIM_HZ // 2) == 0:
                p = data.qpos[rl.base_qadr:rl.base_qadr + 3]
                print(f"  t={data.time:4.1f}  cmd={cmd}  base z={p[2]:.3f}  "
                      f"xy=({p[0]:.2f},{p[1]:.2f})")
        dx = data.qpos[rl.base_qadr:rl.base_qadr + 2] - x0
        print(f"walked dxy = ({dx[0]:+.2f},{dx[1]:+.2f}) m; final z={data.qpos[zad]:.3f} "
              f"({'UPRIGHT' if data.qpos[zad] > 0.5 else 'FELL'})")
        return

    # interactive teleop
    import mujoco.viewer as mjv
    teleop = Teleop(max_vx=1.5, max_vy=0.5, max_yaw_rate=0.6, ramp_rate=2.0)
    teleop.start()
    print("\nRunning kitchen teleop — arrows to walk, ESC to quit\n")
    t0 = time.perf_counter()
    with mjv.launch_passive(model, data) as viewer:
        viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_bid
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -20
        while viewer.is_running() and teleop.is_running():
            vx, vy, _, yaw = teleop.get_cmd()
            rl.step(data, [vx, vy, yaw], data.time)
            rl.apply_control(data)
            mj.mj_step(model, data)
            if int(data.time * SIM_HZ) % int(SIM_HZ / 60) == 0:
                viewer.sync()
                ahead = data.time - (time.perf_counter() - t0)
                if ahead > 0:
                    time.sleep(ahead)
    teleop.stop()


if __name__ == "__main__":
    main()
