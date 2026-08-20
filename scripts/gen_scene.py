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


def rest_offset(path) -> float:
    """Height of an object's TOP-BODY ORIGIN above the surface it rests on.

    Compiles the robocasa object on its own and measures the lowest point of its COLLISION
    geometry relative to its top body's frame, from `mesh_vert` -- the actual vertices. Never from
    `geom_rbound`, which is a bounding-sphere radius and overestimates badly on a flat tray.

    WHY THIS IS COMPUTED. The placement z used to be a hand-guessed constant (`TABLE_TOP_Z +
    0.015`). Measured, that left the tray 12.9 mm in the air -- and the tray is WELDED to the
    world, so unlike the free-jointed lemon it can never fall and settle. It just hangs there
    forever, which is exactly what it looked like. A guessed offset is only ever right by luck;
    each robocasa asset has its own origin convention.
    """
    child = mujoco.MjSpec.from_file(str(path))
    for coll in (child.meshes, child.textures):
        for asset in coll:
            if asset.file and not asset.file.startswith("/"):
                asset.file = str((path.parent / asset.file).resolve())
    m = child.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    top_z = d.xpos[1][2]                       # body 1 = the child's top body (0 is world)
    lo = np.inf
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[mid]:
                            m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].reshape(-1, 3)
            W = (d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g]
            lo = min(lo, float(W[:, 2].min()))
        else:
            lo = min(lo, float(d.geom_xpos[g][2] - m.geom_size[g][2]))
    assert np.isfinite(lo), f"{path} has no collision geometry to rest on"
    return top_z - lo


def add_task(spec: mujoco.MjSpec) -> None:
    """Attach the robocasa pick object and place target -- identical for every embodiment.

    Attached with MjSpec.attach and prefixed, same as the arms, so their mesh/material/geom names
    cannot collide with the robot's.

    BOTH GET A FREE JOINT. The pick object obviously must be liftable. The tray was initially
    welded so the place target could not be nudged, but that buys determinism with physics: a
    welded body is infinitely rigid, so brushing it gives an unphysical contact response and the
    object bounces off a tray a real 2.8 kg one would have shifted. That response is part of what
    transfers, and the two embodiments reach into the tray differently. Read the place target from
    the tray's MEASURED pose -- it is already in the observation -- rather than bolting it down.
    A free tray also settles instead of hanging in mid-air when its placement is off.
    """
    # 0.5 mm of clearance rather than exact contact, so nothing starts the episode already
    # interpenetrating the table and getting pushed out by the contact solver.
    CLEARANCE = 0.0005
    for label, path, pos, free in (
        ("obj", L.PICK_OBJECT, L.PICK_POS, True),
        ("plate", L.PLACE_TARGET, L.PLACE_POS, True),
    ):
        # x,y come from the layout; z is MEASURED so the object rests ON the table. The z in
        # PICK_POS/PLACE_POS stays the nominal aim point for the home-pose search.
        pos = (pos[0], pos[1], L.TABLE_TOP_Z + rest_offset(path) + CLEARANCE)
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
        # ⚠️ GEOM GROUPS: match the robots' convention or the scene renders wrong. MuJoCo's
        # viewer shows groups 0/1/2 and hides 3+, and menagerie uses 2 for visual and 3 for
        # collision so the collision hulls stay hidden. The robocasa objects instead ship their
        # COLLISION meshes in group 0, which is visible -- so the grey convex hulls were drawn
        # over the textured visual meshes, and the object looked like an untextured blob until
        # you pressed `0` to hide the group (which also hid the table, since it lives there too).
        # `*_reg_bbox` are robocasa's annotation boxes, not geometry; park them in 4 so they are
        # hidden and do not clutter the collision view either.
        for geom in child.geoms:
            if geom.name.endswith("reg_bbox"):
                geom.group = 4
            elif geom.contype or geom.conaffinity:
                geom.group = 3
            else:
                geom.group = 2

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
    # ⚠️ SCENE PHYSICS IS CANONICAL, NOT INHERITED PER ROBOT. MjSpec.attach does not bring the
    # child's <option>, so a fresh MjSpec silently falls back to MuJoCo's defaults (Euler,
    # pyramidal cone) -- which left Panda's kp=4500 joints oscillating hard enough to pin seven
    # actuators at their force limits while holding a STATIONARY pose.
    #
    # But copying the settings from each ARM's source model is also wrong, and more subtly so:
    # the reBot ships cone="elliptic" impratio="10" and Panda does not, so the SAME lemon wedge
    # drifted 0.15 deg per 28 s in one scene and 5.65 deg in the other. The environment must be
    # identical across embodiments or a contact-solver difference shows up as a transfer result,
    # which is precisely the confound this project exists to measure. Pinned here, both scenes
    # give 0.06 mm / 0.15 deg. Elliptic friction with a high impratio is also MuJoCo's own
    # recommendation for grasping, and both source models ask for implicitfast anyway.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    # Timestep stays the finest either arm was tuned for, rather than a number invented here.
    spec.option.timestep = min(
        mujoco.MjModel.from_xml_path(str(rb["mjcf"])).opt.timestep for rb in L.ROBOTS.values())

    # MULTI-POINT CONVEX CONTACT. MuJoCo's convex-convex narrowphase returns a SINGLE contact
    # point per geom pair, so a mesh object resting on the table is balanced on one point and
    # rocks forever: the lemon wedge accumulated 24-58 deg of net rotation while sitting still,
    # which looks like a mass/inertia or friction bug and is neither -- timestep, solver
    # iterations, integrator and friction all made no difference (some made it worse), because
    # nothing there adds the missing contact points. multiccd generates several, taking the
    # wedge to 0.2 deg over 4 s and 2.8 deg over 30 s. It matters beyond cosmetics: object pose
    # is part of the recorded observation, so a drifting object is label noise in every demo.
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
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

        # ⚠️ DROP THE ARM'S OWN LIGHTS. Both source models ship an overhead spot intended for
        # viewing that arm ALONE. Attaching two arms brings two more lights on top of the two
        # tuned here, and their diffuse is 0.7 against our 0.35, so total diffuse hit 2.1 and the
        # `top` camera rendered a white frame. Lighting is a property of the scene, not of a
        # component, and these frames are recorded observations -- a blown-out view is unusable
        # training data, not just an ugly preview.
        for light in list(arm.lights):
            arm.delete(light)

        # Marker sites must not render into the camera images. Sid's panda_ros2.xml carries a
        # `panda_ee` site in group 0 (visible by default, solid red), which put a red ball in
        # the middle of both wrist views -- straight into the recorded observations. Group 4 is
        # hidden by default; a site's group affects rendering only, never its use as a frame.
        for site in arm.sites:
            site.group = 4

        frame = spec.worldbody.add_frame(pos=list(mounts[side]), quat=[1, 0, 0, 0])
        spec.attach(arm, prefix=f"{side}_", frame=frame)

    # ⚠️ GRAVITY COMPENSATION, and it MUST be set here rather than on the compiled model.
    # MuJoCo counts gravcomp bodies at compile time (`ngravcomp`) and skips the whole force path
    # when that count is zero, so assigning `model.body_gravcomp` at runtime does nothing at all
    # and reports no error -- it measured byte-identical to no compensation.
    #
    # WHY compensate: a position servo can only generate holding torque from standing error, so
    # every joint sits gravity_torque/kp away from its command forever. That is the entire
    # residual -- with gravity off, Panda's hold error is exactly 0.000000 rad. Left uncorrected
    # it means the pose we RECORD as the action is not the pose the arm reaches, which is label
    # noise in every demo and differs per embodiment, i.e. exactly the confound this project
    # exists to measure. Compensating the arm bodies removes it. The manipulated object is NOT
    # compensated, so grasping still has to hold real weight.
    for side in L.SIDES:
        for body in spec.bodies:
            if body.name.startswith(f"{side}_"):
                body.gravcomp = 1.0

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
        pose = tuple(r["home"][side]) + (r["grip_range"][1],) * len(r["gripper_joints"])
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

    # ⚠️ RESOLVE ACTUATORS BY TRANSMISSION TARGET, NEVER BY JOINT NAME. This used to be
    # mj_name2id(mjOBJ_ACTUATOR, f"{side}_{jb}"), which finds an actuator only when it happens to
    # share its joint's name. The reBot's do; Panda's are `actuator1..7` / `gripper_actuator`, so
    # EVERY lookup returned -1, the `if aid >= 0` swallowed it, and Panda shipped a home keyframe
    # whose ctrl was all zeros. The sim then drove every joint from the home qpos toward 0 the
    # instant it loaded: four actuators pinned at their force limits, the fingers slammed through
    # each other by 17 mm, and the arm could not track an EEF command to better than 40 mm. It
    # reads as an IK or a controller fault -- the IK was in fact exact (0.0000 mm open-loop).
    # `check_parity.py` resolves actuators the same way, and for the same reason.
    act_of_joint: dict[int, int] = {}
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            act_of_joint[int(model.actuator_trnid[i, 0])] = i

    n_set = 0
    for side in L.SIDES:
        pose = tuple(r["home"][side]) + (r["grip_range"][1],) * len(r["gripper_joints"])
        for jb, q in zip(joints, pose):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{jb}")
            aid = act_of_joint.get(jid, -1)
            if aid >= 0:
                key_.ctrl[aid] = q
                n_set += 1
    # A keyframe whose ctrl does not command the qpos it stores is a spring-loaded model, so
    # assert every actuator got a value rather than trusting the loop.
    assert n_set == model.nu, (
        f"home keyframe set {n_set}/{model.nu} actuators -- an unset actuator holds 0.0 and "
        f"will drive its joint away from the home pose on load")

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
