import mujoco, numpy as np
from PIL import Image
import zero_layout as L
m = mujoco.MjModel.from_xml_path(str(L.OUT_MJCF)); d = mujoco.MjData(m)
for side in L.SIDES:
    for jb, q in zip(L.ALL_JOINTS, L.HOME_QPOS + L.GRIPPER_OPEN):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{jb}")
        d.qpos[m.jnt_qposadr[j]] = q
mujoco.mj_forward(m, d)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
tiles = []
with mujoco.Renderer(m, height=360, width=480) as r:
    for n in names:
        r.update_scene(d, camera=n)
        img = np.array(Image.fromarray(r.render()))
        # label bar
        img[:26] = (18, 24, 46)
        tiles.append((n, img))
W = 480 * 3
grid = np.zeros((360 * 2, W, 3), np.uint8)
for k, (n, img) in enumerate(tiles):
    rr, cc = divmod(k, 3)
    grid[rr*360:(rr+1)*360, cc*480:(cc+1)*480] = img
Image.fromarray(grid).save(str(L.ROOT / "scenes" / "cameras.png"))
print("cameras:", names)
print("wrote", L.ROOT / "scenes" / "cameras.png")
