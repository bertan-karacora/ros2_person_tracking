from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    args_launch = [
        DeclareLaunchArgument("name_config", description="Name of the config", default_value="constant_acceleration"),
        DeclareLaunchArgument("threshold_confidence_detection", description="Confidence threshold for detections to be considered valid", default_value="0.5"),
        DeclareLaunchArgument("threshold_association", description="Similarity threshold for associations to be considered valid", default_value="0.5"),
        DeclareLaunchArgument("use_transform_frame", description="Usage of transform to target frame", default_value="True"),
        DeclareLaunchArgument("name_frame_target", description="Name of the target frame for detections", default_value="map"),
        DeclareLaunchArgument("name_topic_detections_2d_lidar", description="Name of the detections topic from the 2D LiDAR (for subscriber)", default_value="/person_detection_2d_lidar/detections"),
        DeclareLaunchArgument("name_topic_detections_camera", description="Name of the detections topic from the camera (for publisher)", default_value="/person_detection/detections"),
        DeclareLaunchArgument("name_topic_detections_tracked", description="Name of the tracked detections topic (for publisher)", default_value="/person_tracking/detections"),
        DeclareLaunchArgument("name_topic_marker", description="Name of the marker topic (for publisher)", default_value="/person_tracking/marker"),
        DeclareLaunchArgument("use_service_only", description="Usage of service-only mode", default_value="False"),
    ]
    launch_description = LaunchDescription(args_launch)

    action_person_tracking = Node(
        package="ros2_person_tracking",
        namespace="",
        executable="spin",
        name="ros2_person_tracking",
        output="screen",
        parameters=[
            {
                "name_config": LaunchConfiguration("name_config"),
                "threshold_confidence_detection": LaunchConfiguration("threshold_confidence_detection"),
                "threshold_association": LaunchConfiguration("threshold_association"),
                "use_transform_frame": LaunchConfiguration("use_transform_frame"),
                "name_frame_target": LaunchConfiguration("name_frame_target"),
                "name_topic_detections_2d_lidar": LaunchConfiguration("name_topic_detections_2d_lidar"),
                "name_topic_detections_camera": LaunchConfiguration("name_topic_detections_camera"),
                "name_topic_detections_tracked": LaunchConfiguration("name_topic_detections_tracked"),
                "name_topic_marker": LaunchConfiguration("name_topic_marker"),
                "use_service_only": LaunchConfiguration("use_service_only"),
            }
        ],
        respawn=True,
        respawn_delay=1.0,
    )
    launch_description.add_action(action_person_tracking)

    return launch_description
