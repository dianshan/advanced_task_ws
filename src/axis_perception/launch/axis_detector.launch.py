from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = LaunchConfiguration("config")
    use_tf = LaunchConfiguration("use_tf")

    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value=[
                get_package_share_directory("axis_perception"),
                "/config/axis_detector.yaml",
            ],
        ),
        DeclareLaunchArgument(
            "use_tf",
            default_value="true",
            choices=["true", "false"],
        ),
        Node(
            package="axis_perception",
            executable="axis_detector_node",
            name="axis_detector",
            output="screen",
            parameters=[config, {"use_tf": use_tf}],
        ),
    ])
