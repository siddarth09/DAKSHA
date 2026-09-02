#!/usr/bin/env bash
# Fetch the upstream description the ViperX 300s target needs, and flatten its xacro.
#
#   bash scripts/fetch_vendor.sh
#
# robots/_vendor is gitignored: this is an upstream tree, not ours to version. The apt package
# ros-jazzy-... holds the same files, but cloning needs no root and pins the branch we chose.
#
# Trossen's own ROS 2 description, jazzy branch. menagerie's vx300s MJCF was derived from it, so
# the two agree on forward kinematics to 0.0000 mm over 400 configurations, which is the property
# the Panda never had and the reason this arm replaced it.
#
# The G1 is NOT fetched here. robots/unitree_g1_mjcf/ is checked in, and its URDF is derived from
# its MJCF (the *_jointbody links are the converter's signature), so the pair agrees to 0.0000 mm.
# The upstream unitreerobotics/unitree_ros clone is 1.6 GB, and its MJCF ships bare torque motors
# with no standing keyframe, so it is strictly worse for this scene.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V="$ROOT/robots/_vendor"
mkdir -p "$V"

if [ -d "$V/interbotix/.git" ]; then
  echo "interbotix already present"
else
  git clone --depth 1 -b jazzy \
    https://github.com/Interbotix/interbotix_ros_manipulators.git "$V/interbotix"
fi

# --- flatten the ViperX xacro into a plain URDF -------------------------------------------------
# Three things have to be corrected on the way out, all of them Interbotix packaging choices:
#   * every link is namespaced `vx300s/`, which would collide with our own per-side prefixing and
#     put a '/' inside a link name;
#   * it carries its own <ros2_control> block, and gen_urdf.py writes ours;
#   * `interbotix_black.png` is referenced as a mesh filename, so it travels with the STLs.
DESC="$V/interbotix/interbotix_ros_xsarms/interbotix_xsarm_descriptions"
OV="$(mktemp -d)"
mkdir -p "$OV/share/ament_index/resource_index/packages"
ln -sfn "$DESC" "$OV/share/interbotix_xsarm_descriptions"
touch "$OV/share/ament_index/resource_index/packages/interbotix_xsarm_descriptions"
AMENT_PREFIX_PATH="$OV:${AMENT_PREFIX_PATH:-}" \
  xacro "$DESC/urdf/vx300s.urdf.xacro" \
    robot_model:=vx300s robot_name:=vx300s hardware_type:=fake > "$V/vx300s_raw.urdf"
python3 - "$V/vx300s_raw.urdf" "$V/vx300s_flat.urdf" <<'PY'
import re, sys, pathlib
t = pathlib.Path(sys.argv[1]).read_text()
t = re.sub(r'<ros2_control\b.*?</ros2_control>', '', t, flags=re.S)
t = t.replace('vx300s/', '')
pathlib.Path(sys.argv[2]).write_text(t)
print(f"flattened -> {sys.argv[2]}")
PY
rm -f "$V/vx300s_raw.urdf"; rm -rf "$OV"

echo
echo "vendored into $V:"
ls -1 "$V"
