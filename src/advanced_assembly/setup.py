from setuptools import find_packages, setup

package_name = "advanced_assembly"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/advanced_task.yaml"]),
        ("share/" + package_name + "/launch", ["launch/advanced_task.launch.py"]),
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "advanced_nut_detector = advanced_assembly.nut_detector_node:main",
            "assembly_task_controller = advanced_assembly.task_controller:main",
        ],
    },
    zip_safe=True,
    maintainer="Competition Team",
    maintainer_email="user@example.com",
    description="Four-nut handover and vertical-axis assembly task.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
