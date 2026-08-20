"""Build a two-arm MJCF for ANY registered embodiment, with identical table and cameras.

    MUJOCO_GL=egl python scripts/gen_scene.py rebot
    MUJOCO_GL=egl python scripts/gen_scene.py panda

WHY ONE GENERATOR. The cross-embodiment chain is reBot -> Panda -> G1, and the whole study only
means anything if the TASK IS IDENTICAL across embodiments: same table, same object position,
same camera poses, same look-at point. Duplicating a per-robot script guarantees those drift.
Everything that must be shared comes from `zero_layout`; only what genuinely differs (joint
names, reach, mount separation, home pose, aperture) lives in the per-robot registry entry.

Out:  zero_description/mjcf/zero_<robot>.xml  +  scenes/<robot>_cameras.png
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from PIL import Image

import zero_layout as L

LEG_R = 0.03


def add_task(spec: mujoco.MjSpec) -> None:
    """Attach the robocasa pick object and place target -- identical for every embodiment.

    Attached with MjSpec.attach and prefixed, same as the arms, so their mesh/material/geom names
    cannot collide with the robot's. The pick object gets a FREE JOINT (it must be liftable); the
    tray is left welded to the world so it cannot be nudged out of position, which keeps the
    place target fixed across episodes. Give the tray a freejoint later if sliding it becomes
    part of the task.
    """
    for label, path, pos, free in (
        ("obj", L.PICK_OBJECT, L.PICK_POS, True),
        ("plate", L.PLACE_TARGET, L.PLACE_POS, False),
    ):
        child = mujoco.MjSpec.from_file(str(path))
        # ⚠️ ABSOLUTISE the child's asset paths. A compiled model has ONE global meshdir, which we
        # point at the ROBOT's assets; the robocasa object's meshes are relative to its own
        # folder, so after `to_xml()` they resolve against the wrong directory and the model fails
        # to load with "Error opening file .../seeed_rebot_devarm/assets/visual/LemonWedge001.obj".
        # In-process compilation happens to work, which makes this an easy trap to miss.
        for coll in (child.meshes, child.textures):
            for asset in coll:
                if asset.file and not asset.file.startswith("/"):
                    asset.file = str((path.parent / asset.file).resolve())
        # robocasa nests the mesh as <worldbody><body><body name="object">, so `object` is two
        # levels down. A free joint is only legal on a direct child of worldbody, so it has to be
        # added to the child's TOP body BEFORE attaching -- after attach that body becomes a
        # worldbody child of ours. Adding it to `object` post-attach fails with
        # "free joint can only be used on top level".
        top = child.worldbody.bodies[0]
        top.name = "root"
        top.pos = list(pos)          # position carried on the body, so the frame stays identity
        if free:
            top.add_freejoint()
        # attach() insists on a frame or site; an IDENTITY frame keeps the pose on the body and
        # avoids double-counting it (frames are flattened at compile, so the free joint still
        # ends up on a top-level body).
        spec.attach(child, prefix=f"{label}_",
                    frame=spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0]))


def lookat(eye, target, up=(0.0, 0.0, 1.0)):
    """MJCF `xyaxes` for a camera at `eye` aimed at `target`.

    MuJoCo cameras look down their own -z with +y up: z_cam = normalize(eye-target),
    x_cam = normalize(up x z_cam), y_cam = z_cam x x_cam. Computed rather than hand-written --
    nine numbers with a sign error is a classic silent framing bug.
    """
    eye, target, up = np.array(eye, float), np.array(target, float), np.array(up, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    assert n > 1e-6, "camera axis parallel to `up`"
    x /= n
    return [*x, *np.cross(z, x)]


def build(key: str) -> mujoco.MjSpec:
    r = L.ROBOTS[key]
    mounts = L.robot_mounts(key)
    spec = mujoco.MjSpec()
    spec.modelname = f"zero_{key}"
    spec.compiler.degree = False
    # Absolute meshdir: the MJCF is written to zero_description/mjcf/ but the meshes live under
    # robots/<robot>/assets, and a relative path breaks the moment the cwd changes.
    spec.meshdir = str(r["mjcf"].parent / "assets")
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720

    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4],
                     rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name="groundplane", texrepeat=[5, 5], reflectance=0.2).textures[
        mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    spec.add_material(name="table_mat", rgba=[0.72, 0.60, 0.44, 1])
    spec.add_material(name="leg_mat", rgba=[0.25, 0.25, 0.28, 1])

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], material="groundplane")
    # Offset from straight overhead: a light at (0,0,z) shines into the `top` camera and blows
    # the frame out.
    # diffuse well below 1.0: two full-strength spots blew out the pale table and Panda's white
    # meshes in the `top` view. Ambient carries the fill instead.
    for pos, dr in (([-0.6, 0.7, 2.2], [0.3, -0.3, -1]), ([0.9, -0.7, 2.0], [-0.4, 0.3, -1])):
        spec.worldbody.add_light(pos=pos, dir=dr, type=mujoco.mjtLightType.mjLIGHT_SPOT,
                                 diffuse=[0.35, 0.35, 0.35], specular=[0.05, 0.05, 0.05],
                                 ambient=[0.25, 0.25, 0.25])

    cx, cy = L.TABLE_CENTER_XY
    hx, hy, hz = L.TABLE_HALF
    spec.worldbody.add_body(name="table", pos=[cx, cy, L.TABLE_TOP_Z - hz]).add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
        material="table_mat")
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        lz = (L.TABLE_TOP_Z - 2 * hz) / 2
        spec.worldbody.add_body(
            name=f"leg{i}", pos=[cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), lz]
        ).add_geom(name=f"leg{i}_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[LEG_R, lz], material="leg_mat")

    for side in L.SIDES:
        # Reload per side: reusing one MjSpec for two attaches fails with "incompatible id in
        # exclude array", because these models ship <contact><exclude> pairs and attaching
        # rewrites the child's element ids.
        arm = mujoco.MjSpec.from_file(str(r["mjcf"]))
        frame = spec.worldbody.add_frame(pos=list(mounts[side]), quat=[1, 0, 0, 0])
        spec.attach(arm, prefix=f"{side}_", frame=frame)

    add_task(spec)

    # eef sites + cameras. Neither menagerie model ships a site, and nothing pose-based (IK, an
    # ee_to_object observation, a reach reward) can work without one.
    for side in L.SIDES:
        body = spec.body(f"{side}_{r['eef_body']}")
        body.add_site(name=f"{side}_eef", pos=list(r["eef_offset"]),
                      size=[0.008] * 3, group=4)
        body.add_camera(name=f"{side}_wrist", pos=list(r["wrist_cam_pos"]),
                        fovy=L.WRIST_CAM_FOVY, xyaxes=list(r["wrist_cam_xyaxes"]),
                        resolution=list(L.CAM_RES))
    for name, (eye, fovy) in L.SCENE_CAMS.items():
        spec.worldbody.add_camera(name=name, pos=list(eye), fovy=fovy,
                                  xyaxes=lookat(eye, L.LOOK_AT), resolution=list(L.CAM_RES))
    return spec


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    if key not in L.ROBOTS:
        raise SystemExit(f"unknown robot {key!r}; known: {list(L.ROBOTS)}")
    r = L.ROBOTS[key]
    spec = build(key)
    model = spec.compile()
    d = mujoco.MjData(model)

    joints = L.robot_all_joints(key)
    for side in L.SIDES:
        pose = tuple(r["home"][side]) + tuple(r["grip_open"])
        for jb, q in zip(joints, pose):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{jb}")
            assert jid >= 0, f"missing joint {side}_{jb}"
            d.qpos[model.jnt_qposadr[jid]] = q
    mujoco.mj_forward(model, d)

    # ⚠️ WRITE A `home` KEYFRAME. Without it mujoco_ros2_control starts the sim at qpos=0 --
    # arms bolt upright, which is what the launch actually showed. The generator was only applying
    # the home pose for its own render. qpos here is the FULL vector (both arms + the free-jointed
    # object), captured after mj_forward so the object sits where add_task placed it.
    key_ = spec.add_key(name="home")
    key_.qpos = d.qpos.copy().tolist()
    key_.ctrl = [0.0] * model.nu
    for side in L.SIDES:
        for jb, q in zip(joints, tuple(r["home"][side]) + tuple(r["grip_open"])):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_{jb}")
            if aid >= 0:
                key_.ctrl[aid] = q

    out = L.PKG / "mjcf" / f"zero_{key}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    cams = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    print(f"[{key}] nq={model.nq} nu={model.nu} nbody={model.nbody} ncam={model.ncam} "
          f"neq={model.neq}")
    print(f"  cameras: {cams}")
    for side in L.SIDES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
        print(f"  {side}_eef @ {np.round(d.site_xpos[sid], 3)}")
    print(f"  wrote {out}")

    # contact sheet, so framing is verified rather than assumed
    tiles = []
    with mujoco.Renderer(model, height=360, width=480) as ren:
        for n in cams:
            ren.update_scene(d, camera=n)
            img = np.array(ren.render())
            img[:24] = (18, 24, 46)
            tiles.append(img)
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((360 * rows, 480 * cols, 3), np.uint8)
    for i, img in enumerate(tiles):
        rr, cc = divmod(i, cols)
        grid[rr * 360:(rr + 1) * 360, cc * 480:(cc + 1) * 480] = img
    png = L.ROOT / "scenes" / f"{key}_cameras.png"
    Image.fromarray(grid).save(png)
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
