from setuptools import find_packages, setup

package_name = "axis_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/axis_detector.yaml"]),
        ("share/" + package_name + "/launch", ["launch/axis_detector.launch.py"]),
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "axis_detector_node = axis_perception.axis_detector_node:main",
        ],
    },
    zip_safe=True,
    maintainer="Competition Team",
    maintainer_email="user@example.com",
    description="Depth-rule perception for the fixed vertical assembly axis.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
