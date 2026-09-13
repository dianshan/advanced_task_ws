import json
from collections import deque
from dataclasses import asdict
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException

from .axis_detector import AxisObservation, DepthAxisDetector, DetectorConfig


class AxisDetectorNode(Node):
    def __init__(self):
        super().__init__("axis_detector")

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("depth_info_topic", "/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("use_tf", True)
        self.declare_parameter("axis_pose_topic", "/axis_detection/pose")
        self.declare_parameter("axis_debug_topic", "/axis_detection/debug")
        self.declare_parameter("axis_status_topic", "/axis_detection/status")
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("synchronize_slop", 0.10)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("lock_after_confirmations", 5)
        self.declare_parameter("max_center_jitter_m", 0.008)
        self.declare_parameter("max_top_height_jitter_m", 0.015)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("workspace_min_x_m", -0.10)
        self.declare_parameter("workspace_max_x_m", 0.10)
        self.declare_parameter("workspace_min_y_m", -0.10)
        self.declare_parameter("workspace_max_y_m", 0.10)
        self.declare_parameter("plane_inlier_threshold_m", 0.006)
        self.declare_parameter("min_axis_height_m", 0.105)
        self.declare_parameter("max_axis_height_m", 0.185)
        self.declare_parameter("min_top_points", 60)
        self.declare_parameter("max_top_depth_m", 0.012)
        self.declare_parameter("cylinder_radius_m", 0.015)
        self.declare_parameter("cylinder_radius_tolerance_m", 0.008)
        self.declare_parameter("min_circularity", 0.65)
        self.declare_parameter("min_top_inlier_points", 80)
        self.declare_parameter("top_radius_quantile", 0.95)
        self.declare_parameter("top_plane_tolerance_m", 0.008)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.info_topic = self.get_parameter("depth_info_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.use_tf = bool(self.get_parameter("use_tf").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.lock_after_confirmations = int(self.get_parameter("lock_after_confirmations").value)
        self.max_center_jitter_m = float(self.get_parameter("max_center_jitter_m").value)
        self.max_top_height_jitter_m = float(
            self.get_parameter("max_top_height_jitter_m").value
        )

        config = DetectorConfig(
            workspace_min_x_m=float(self.get_parameter("workspace_min_x_m").value),
            workspace_max_x_m=float(self.get_parameter("workspace_max_x_m").value),
            workspace_min_y_m=float(self.get_parameter("workspace_min_y_m").value),
            workspace_max_y_m=float(self.get_parameter("workspace_max_y_m").value),
            plane_inlier_threshold_m=float(
                self.get_parameter("plane_inlier_threshold_m").value
            ),
            min_axis_height_m=float(self.get_parameter("min_axis_height_m").value),
            max_axis_height_m=float(self.get_parameter("max_axis_height_m").value),
            min_top_points=int(self.get_parameter("min_top_points").value),
            max_top_depth_m=float(self.get_parameter("max_top_depth_m").value),
            cylinder_radius_m=float(self.get_parameter("cylinder_radius_m").value),
            cylinder_radius_tolerance_m=float(
                self.get_parameter("cylinder_radius_tolerance_m").value
            ),
            min_circularity=float(self.get_parameter("min_circularity").value),
            min_top_inlier_points=int(
                self.get_parameter("min_top_inlier_points").value
            ),
            top_radius_quantile=float(
                self.get_parameter("top_radius_quantile").value
            ),
            top_plane_tolerance_m=float(
                self.get_parameter("top_plane_tolerance_m").value
            ),
        )
        self.detector = DepthAxisDetector(config)
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion: Optional[np.ndarray] = None
        self.observations: deque[AxisObservation] = deque(maxlen=self.lock_after_confirmations)
        self.locked_observation: Optional[AxisObservation] = None
        self.last_debug_image: Optional[np.ndarray] = None
        self.last_detection_count = 0

        pose_topic = self.get_parameter("axis_pose_topic").value
        debug_topic = self.get_parameter("axis_debug_topic").value
        status_topic = self.get_parameter("axis_status_topic").value
        self.pose_publisher = self.create_publisher(PoseStamped, pose_topic, 10)
        self.debug_publisher = self.create_publisher(Image, debug_topic, 10)
        self.status_publisher = self.create_publisher(String, status_topic, 10)

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        color_subscriber = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=sensor_qos)
        depth_subscriber = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        info_subscriber = message_filters.Subscriber(
            self, CameraInfo, self.info_topic, qos_profile=sensor_qos
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_subscriber, depth_subscriber, info_subscriber],
            int(self.get_parameter("queue_size").value),
            float(self.get_parameter("synchronize_slop").value),
        )
        self.synchronizer.registerCallback(self.on_images)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(1.0 / self.publish_rate_hz, self.publish_latest)
        self.get_logger().info("Depth axis detector started")

    @staticmethod
    def image_to_cv(image: Image) -> np.ndarray:
        if image.encoding == "mono8":
            return np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.height, image.width
            )
        if image.encoding in ("rgb8", "bgr8"):
            return np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.height, image.width, 3
            )
        if image.encoding == "rgba8":
            return np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )[:, :, :3]
        raise ValueError(f"Unsupported color image encoding: {image.encoding}")

    @staticmethod
    def depth_to_cv(image: Image) -> np.ndarray:
        if image.encoding == "16UC1":
            return np.frombuffer(image.data, dtype=np.uint16).reshape(
                image.height, image.width
            )
        if image.encoding == "32FC1":
            return np.frombuffer(image.data, dtype=np.float32).reshape(
                image.height, image.width
            )
        raise ValueError(f"Unsupported depth image encoding: {image.encoding}")

    def on_images(self, color: Image, depth: Image, info: CameraInfo):
        try:
            color_image = self.image_to_cv(color)
            depth_image = self.depth_to_cv(depth)
        except (ValueError, TypeError) as error:
            self.get_logger().warning(str(error), throttle_duration_sec=5.0)
            return

        if color_image.shape[:2] != depth_image.shape:
            self.get_logger().warning("Color and depth dimensions differ", throttle_duration_sec=5.0)
            return

        camera_matrix = np.array(info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.array(info.d, dtype=np.float64)
        if self.camera_matrix is None or self.last_detection_count == 0:
            self.camera_matrix = camera_matrix
            self.distortion = distortion
        else:
            if not np.allclose(self.camera_matrix, camera_matrix):
                self.get_logger().warning("Camera intrinsics changed", throttle_duration_sec=5.0)
                self.camera_matrix = camera_matrix
                self.distortion = distortion

        if depth_image.dtype == np.uint16:
            metric_depth = depth_image.astype(np.float32) * self.depth_scale
        else:
            metric_depth = depth_image.astype(np.float32)

        observation = self.detect_from_images(
            metric_depth, color_image, camera_matrix, distortion, depth.header.frame_id
        )
        if observation is None:
            self.last_debug_image = color_image
            self.last_detection_count = 0
            return

        self.last_debug_image = self.draw_debug(color_image, observation, camera_matrix)
        self.last_detection_count = 1
        self.observations.append(observation)
        self.update_locked_observation()

    def detect_from_images(
        self,
        depth: np.ndarray,
        color: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        frame_id: str,
    ) -> Optional[AxisObservation]:
        del color
        valid_mask = np.isfinite(depth) & (depth > 0.10) & (depth < 3.0)
        ys, xs = np.nonzero(valid_mask)
        if len(xs) < 100:
            return None

        pixel_points = np.column_stack((xs, ys)).astype(np.float32)
        undistorted_points = cv2.undistortPoints(
            pixel_points, camera_matrix, distortion
        ).reshape(-1, 2)
        rays = np.column_stack((undistorted_points, np.ones_like(undistorted_points[:, :1])))
        camera_points = rays * depth[ys, xs, np.newaxis]
        camera_points = camera_points[:, :3]

        observation = self.detector.detect(camera_points)
        if observation is None or not self.use_tf:
            return observation

        if frame_id == self.target_frame:
            return observation

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )
        except TransformException:
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rotation_matrix = self.rotation_to_matrix(rotation)
        transformed_center = rotation_matrix @ np.array([
            observation.center_x_m,
            observation.center_y_m,
            observation.top_z_m,
        ]) + np.array([translation.x, translation.y, translation.z])
        return AxisObservation(
            center_x_m=float(transformed_center[0]),
            center_y_m=float(transformed_center[1]),
            top_z_m=float(transformed_center[2]),
            radius_m=observation.radius_m,
            circularity=observation.circularity,
            inlier_count=observation.inlier_count,
        )

    @staticmethod
    def rotation_to_matrix(rotation) -> np.ndarray:
        x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def update_locked_observation(self):
        if self.locked_observation is not None or len(self.observations) < self.lock_after_confirmations:
            return

        observations = list(self.observations)
        centers = np.array([[item.center_x_m, item.center_y_m] for item in observations])
        heights = np.array([item.top_z_m for item in observations])
        if np.std(centers, axis=0).max() > self.max_center_jitter_m:
            return
        if np.std(heights) > self.max_top_height_jitter_m:
            return

        self.locked_observation = observations[len(observations) // 2]
        self.get_logger().info(f"Locked axis at {self.locked_observation}")

    @staticmethod
    def observation_to_pose(observation: AxisObservation) -> Pose:
        pose = Pose()
        pose.position = Point(
            x=float(observation.center_x_m),
            y=float(observation.center_y_m),
            z=float(observation.top_z_m),
        )
        pose.orientation.w = 1.0
        return pose

    def draw_debug(
        self,
        color: np.ndarray,
        observation: AxisObservation,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        debug = color.copy()
        if debug.ndim == 2:
            debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
        elif debug.shape[2] == 3 and self.color_topic.endswith("rgb8"):
            debug = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)

        point_3d = np.array([
            observation.center_x_m,
            observation.center_y_m,
            observation.top_z_m,
        ])
        projected, _ = cv2.projectPoints(
            point_3d.reshape(1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera_matrix,
            self.distortion,
        )
        center = tuple(np.round(projected[0, 0]).astype(int))
        radius_pixels = int(
            observation.radius_m
            * camera_matrix[0, 0]
            / max(point_3d[2], 0.1)
        )
        cv2.circle(debug, center, radius_pixels, (0, 255, 0), 2)
        cv2.circle(debug, center, 4, (0, 0, 255), -1)
        cv2.putText(
            debug,
            f"axis r={observation.radius_m * 1000:.0f}mm c={observation.circularity:.2f}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        return debug

    def publish_latest(self):
        observation = self.locked_observation
        if observation is None and self.observations:
            observation = self.observations[-1]
        if observation is None:
            status = {"state": "searching", "confirmations": 0}
            self.status_publisher.publish(String(data=json.dumps(status)))
            return

        pose = PoseStamped()
        pose.header.frame_id = self.target_frame if self.use_tf else "camera_optical_frame"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose = self.observation_to_pose(observation)
        self.pose_publisher.publish(pose)

        status = {
            "state": "locked" if self.locked_observation is not None else "candidate",
            "confirmations": len(self.observations),
            **asdict(observation),
        }
        self.status_publisher.publish(String(data=json.dumps(status)))

        if self.last_debug_image is not None:
            message = self.create_cv_image(self.last_debug_image)
            self.debug_publisher.publish(message)

    @staticmethod
    def create_cv_image(image: np.ndarray) -> Image:
        message = Image()
        message.height, message.width = image.shape[:2]
        if image.ndim == 2:
            message.encoding = "mono8"
            message.step = message.width
            message.data = image.tobytes()
        else:
            bgr_image = image
            message.encoding = "bgr8"
            message.step = message.width * 3
            message.data = bgr_image.tobytes()
        return message


def main(args=None):
    rclpy.init(args=args)
    node = AxisDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
