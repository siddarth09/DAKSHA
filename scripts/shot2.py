import mujoco, numpy as np, sys
from PIL import Image
import zero_layout as L
key=sys.argv[1]
m=mujoco.MjModel.from_xml_path(str(L.PKG/"mjcf"/f"zero_{key}.xml")); d=mujoco.MjData(m)
r=L.ROBOTS[key]
for side in L.SIDES:
    for jb,q in zip(L.robot_all_joints(key), tuple(r["home"][side]) + (r["grip_range"][1],)*len(r["gripper_joints"])):
        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"{side}_{jb}")
        d.qpos[m.jnt_qposadr[j]]=q
mujoco.mj_forward(m,d)
cam=mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m,cam)
cam.lookat[:]=[0.2,0,0.85]; cam.distance,cam.azimuth,cam.elevation=2.4,140,-20
with mujoco.Renderer(m,height=560,width=900) as ren:
    ren.update_scene(d,cam); over=np.array(ren.render())
tiles=[]
with mujoco.Renderer(m,height=280,width=373) as ren:
    for n in ("top","left_wrist","right_wrist"):
        ren.update_scene(d,camera=n); tiles.append(np.array(ren.render()))
strip=np.concatenate(tiles,axis=1)
h=over.shape[0]+strip.shape[0]; w=max(over.shape[1],strip.shape[1])
out=np.zeros((h,w,3),np.uint8); over=np.pad(over,((0,0),(0,w-over.shape[1]),(0,0))); out[:over.shape[0]]=over
out[over.shape[0]:,:strip.shape[1]]=strip
Image.fromarray(out).save(str(L.ROOT/"scenes"/f"{key}_task.png"))
print("wrote", L.ROOT/"scenes"/f"{key}_task.png")
