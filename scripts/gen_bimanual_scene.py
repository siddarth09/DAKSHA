"""Build the two-arm Seeed reBot DevArm MJCF that mujoco_ros2_control loads, and render it.

Two copies of the menagerie arm are attached with `MjSpec.attach`, which prefixes every
joint/body/geom/actuator name. MuJoCo's `<include>` does NOT namespace anything, so two
includes of one model collide on every name -- attach is the only clean route.

All names and mount poses come from `zero_layout`, shared with `gen_bimanual_urdf.py`, because
mujoco_ros2_control binds the URDF and the MJCF together by joint name and a mismatch fails
quietly rather than loudly.

Run:  MUJOCO_GL=egl python scripts/gen_bimanual_scene.py
Out:  zero_description/mjcf/zero_bimanual.xml  +  scenes/bimanual.png
"""

from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image

import zero_layout as L

LEG_R = 0.03


def build() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "zero_bimanual"
    spec.compiler.degree = False  # radians, matching the arm model

    # ⚠️ ABSOLUTE meshdir on purpose: the MJCF lands in zero_description/mjcf/ while the meshes
    # live in robots/seeed_rebot_devarm/assets/, and a relative path breaks as soon as the file
    # is opened from a different working directory. Regenerate if the project moves.
    spec.meshdir = str(L.MENAGERIE_ARM.parent / "assets")

    # Offscreen framebuffer defaults to 640x480; larger renders silently fail without this.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    spec.add_texture(
        name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7], rgb2=[0.0, 0.0, 0.0], width=512, height=3072,
    )
    spec.add_texture(
        name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8], width=300, height=300,
    )
    spec.add_material(name="groundplane", texrepeat=[5, 5], reflectance=0.2).textures[
        mujoco.mjtTextureRole.mjTEXROLE_RGB
    ] = "groundplane"
    spec.add_material(name="table_mat", rgba=[0.72, 0.60, 0.44, 1.0])
    spec.add_material(name="leg_mat", rgba=[0.25, 0.25, 0.28, 1.0])

    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
        material="groundplane",
    )
    # NB: mujoco 3.8 dropped `directional` in favour of a `type` enum.
    # offset from straight-overhead: a light at (0,0,2.5) shone directly into the `top`
    # camera and blew the whole frame out.
    spec.worldbody.add_light(pos=[-0.6, 0.7, 2.2], dir=[0.3, -0.3, -1],
                             type=mujoco.mjtLightType.mjLIGHT_SPOT)
    spec.worldbody.add_light(pos=[0.9, 0.9, 2.0], dir=[-0.4, -0.4, -1],
                             type=mujoco.mjtLightType.mjLIGHT_SPOT)

    # --- table: thin top at TABLE_TOP_Z + four legs (a solid 0.75 m block read as a crate) ---
    cx, cy = L.TABLE_CENTER_XY
    hx, hy, hz = L.TABLE_HALF
    top = spec.worldbody.add_body(name="table", pos=[cx, cy, L.TABLE_TOP_Z - hz])
    top.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[hx, hy, hz], material="table_mat")
    inset = 0.05
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        lz = (L.TABLE_TOP_Z - 2 * hz) / 2
        spec.worldbody.add_body(
            name=f"leg{i}",
            pos=[cx + sx * (hx - inset), cy + sy * (hy - inset), lz],
        ).add_geom(name=f"leg{i}_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[LEG_R, lz], material="leg_mat")

    # --- two arms, prefixed ---
    for side in L.SIDES:
        # ⚠️ RELOAD per side. Reusing one MjSpec for two attaches fails with "incompatible id
        # in exclude array" -- this model ships <contact><exclude> pairs for adjacent links
        # whose meshes interpenetrate, and attaching rewrites the child's element ids, so the
        # second attach sees stale references.
        arm = mujoco.MjSpec.from_file(str(L.MENAGERIE_ARM))
        frame = spec.worldbody.add_frame(pos=list(L.MOUNTS[side]), quat=[1, 0, 0, 0])
        spec.attach(arm, prefix=f"{side}_", frame=frame)

    return spec



def _lookat(eye, target, up=(0.0, 0.0, 1.0)):
    """MJCF `xyaxes` for a camera at `eye` looking at `target`.

    MuJoCo cameras look along their own -z with +y up, so: z_cam = normalize(eye - target),
    x_cam = normalize(up x z_cam), y_cam = z_cam x x_cam. Returned flat as [x_cam, y_cam],
    which is what the `xyaxes` attribute wants. Computing it beats hand-writing nine numbers
    and getting a sign wrong -- the previous project burned time on exactly that.
    """
    eye, target, up = np.array(eye, float), np.array(target, float), np.array(up, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    assert nx > 1e-6, "camera axis is parallel to `up`; pick a different up vector"
    x /= nx
    y = np.cross(z, x)
    return [*x, *y]


def add_cameras(spec: mujoco.MjSpec) -> None:
    """Three fixed scene cameras + one per wrist.

    Scene cams all aim at the Phase-0.5 task centroid. Wrist cams look along the gripper's
    local +z, which is its approach axis (the eef site sits at +0.10 along it).
    """
    for name, (eye, fovy) in L.SCENE_CAMS.items():
        spec.worldbody.add_camera(
            name=name, pos=list(eye), fovy=fovy,
            xyaxes=_lookat(eye, L.LOOK_AT),
            resolution=list(L.CAM_RES),
        )
    for side in L.SIDES:
        # -z_cam must equal +z_body, so z_cam = -z_body: xyaxes = [1,0,0, 0,-1,0]
        spec.body(f"{side}_gripper_end").add_camera(
            name=f"{side}_wrist", pos=list(L.WRIST_CAM_POS), fovy=L.WRIST_CAM_FOVY,
            xyaxes=[1, 0, 0, 0, -1, 0], resolution=list(L.CAM_RES),
        )


def add_eef_sites(spec: mujoco.MjSpec) -> None:
    """The menagerie model ships ZERO sites, so there is no end-effector reference frame.

    Anything pose-based -- IK, an ee_to_object observation, a reach reward -- needs one. Placed
    ahead of `gripper_end` so it sits near the pinch point rather than at the wrist.
    """
    for side in L.SIDES:
        spec.body(f"{side}_gripper_end").add_site(
            name=f"{side}_eef", pos=[0, 0, 0.10], size=[0.008] * 3,
            group=4,  # group 4 -> not drawn in the default view
        )


def main() -> None:
    spec = build()
    add_eef_sites(spec)
    add_cameras(spec)
    model = spec.compile()
    data = mujoco.MjData(model)

    for side in L.SIDES:
        for jbase, q in zip(L.ALL_JOINTS, L.HOME_QPOS + L.GRIPPER_OPEN):
            name = L.prefixed(side, jbase)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert jid >= 0 and aid >= 0, f"missing joint/actuator {name}"
            data.qpos[model.jnt_qposadr[jid]] = q
            data.ctrl[aid] = q
    mujoco.mj_forward(model, data)

    L.OUT_MJCF.parent.mkdir(parents=True, exist_ok=True)
    L.OUT_MJCF.write_text(spec.to_xml())

    print(f"nq={model.nq} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom} "
          f"ncam={model.ncam}")
    print("  cameras:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                         for i in range(model.ncam)])
    for side in L.SIDES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
        print(f"  {side}_eef @ {np.round(data.site_xpos[sid], 4)}")
    print(f"wrote {L.OUT_MJCF}")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.15, 0.0, 0.85]
    cam.distance, cam.azimuth, cam.elevation = 1.9, 135, -22
    # This binding has no `mjr_writePNG`; Renderer.render() gives a top-down HWC uint8 array.
    L.OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=720, width=1280) as r:
        r.update_scene(data, cam)
        Image.fromarray(r.render()).save(L.OUT_PNG)
    print(f"wrote {L.OUT_PNG}")


if __name__ == "__main__":
    main()
