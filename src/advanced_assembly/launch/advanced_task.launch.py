"""Launch advanced perception and the optional task controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory("advanced_assembly")
    axis_share = get_package_share_directory("axis_perception")
    config = LaunchConfiguration("config")
    axis_config = LaunchConfiguration("axis_config")
    run_task = LaunchConfiguration("run_task")

    return LaunchDescription([
        DeclareLaunchArgument(
            "config", default_value=[share, "/config/advanced_task.yaml"],
            description="Advanced detector and task controller parameters"),
        DeclareLaunchArgument(
            "axis_config", default_value=[axis_share, "/config/axis_detector.yaml"],
            description="Fixed-axis detector parameters"),
        DeclareLaunchArgument("run_task", default_value="false",
                              description="Also start the motion task controller"),
        DeclareLaunchArgument("execute_task", default_value="false",
                              choices=["true", "false"],
                              description="Allow controller motion commands"),
        DeclareLaunchArgument("task_mode", default_value="validate",
                              choices=["validate", "single", "full"]),
        DeclareLaunchArgument("target_id", default_value="1",
                              choices=["1", "2", "3", "4"]),
        Node(
            package="advanced_assembly", executable="advanced_nut_detector",
            name="advanced_nut_detector", output="screen", parameters=[config]),
        Node(
            package="axis_perception", executable="axis_detector_node",
            name="axis_detector", output="screen", parameters=[axis_config]),
        Node(
            package="advanced_assembly", executable="assembly_task_controller",
            name="advanced_assembly_task", output="screen",
            condition=IfCondition(run_task),
            parameters=[config,
                        {"execute_task": LaunchConfiguration("execute_task"),
                         "task_mode": LaunchConfiguration("task_mode"),
                         "target_id": LaunchConfiguration("target_id")}]),
    ])
