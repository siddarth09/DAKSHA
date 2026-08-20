from setuptools import find_packages, setup

package_name = "zero_control"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sid",
    maintainer_email="dayasagar.s@northeastern.edu",
    description="SE(3) end-effector control + DLS IK for ZERO",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "eef_control = zero_control.eef_control_node:main",
            "servo_test = zero_control.servo_test:main",
        ],
    },
)
