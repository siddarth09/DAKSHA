cmake_minimum_required(VERSION 3.16)
project(zero_description)

find_package(ament_cmake REQUIRED)

# urdf/ and mjcf/ are GENERATED (scripts/gen_bimanual_urdf.py, scripts/gen_bimanual_scene.py)
# but are installed like normal assets so `package://zero_description/...` resolves at runtime
# for both robot_state_publisher and mujoco_ros2_control's `mujoco_model` param.
install(
  DIRECTORY urdf mjcf meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
