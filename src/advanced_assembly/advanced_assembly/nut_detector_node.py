"""RGB-D perception node for the four advanced-task nuts."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from std_srvs.srv import Trigger
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .nut_detector import AdvancedNutDetector, DetectorConfig
from .nut_sequence import AdvancedNutSequence, Observation, SequenceConfig


class AdvancedNutDetectorNode(Node):
    def __init__(self):
        super().__init__("advanced_nut_detector")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("use_tf", True)
        self.declare_parameter("detection_topic", "/advanced_nut_detections")
        self.declare_parameter("sequence_topic", "/advanced_nut_sequence")
        self.declare_parameter("sequence_status_topic", "/advanced_nut_sequence/status")
        self.declare_parameter("debug_image_topic", "/advanced_nut_detection/debug")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("frame_hold_s", 8.0)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_nut_clearance_m", 0.003)
        self.declare_parameter("depth_min_m", 0.15)
        self.declare_parameter("depth_max_m", 5.0)
        self.declare_parameter("max_image_age_s", 0.5)
        self.declare_parameter("max_depth_age_s", 0.1)
        self.declare_parameter("synchronize_slop", 0.10)
        self.declare_parameter("queue_size", 10)

        self.declare_parameter("frame_size_m", 0.200)
        self.declare_parameter("black_v_max", 190)
        self.declare_parameter("debug_black_frame", True)
        self.declare_parameter("min_frame_area_ratio", 0.01)
        self.declare_parameter("max_frame_area_ratio", 0.10)
        self.declare_parameter("frame_inner_scale", 1.0)
        self.declare_parameter("min_nut_radius_px", 5.0)
        self.declare_parameter("max_nut_radius_px", 40.0)
        self.declare_parameter("min_nut_area_px", 50.0)
        self.declare_parameter("max_nut_area_px", 100000.0)
        self.declare_parameter("adaptive_block_size", 31)
        self.declare_parameter("adaptive_c", 8.0)
        self.declare_parameter("blackhat_kernel_size", 31)
        self.declare_parameter("blackhat_threshold", 25.0)
        self.declare_parameter("nut_mask_open_kernel_size", 3)
        self.declare_parameter("nut_mask_close_kernel_size", 3)
        self.declare_parameter("min_nut_solidity", 0.45)
        self.declare_parameter("min_nut_circularity", 0.25)
        self.declare_parameter("min_nut_aspect_ratio", 0.30)
        self.declare_parameter("silver_gray_min", 120.0)
        self.declare_parameter("silver_black_margin", 15.0)
        self.declare_parameter("m45_min_size_m", 0.075)
        self.declare_parameter("m45_max_size_m", 0.125)
        self.declare_parameter("m33_min_size_m", 0.052)
        self.declare_parameter("m33_max_size_m", 0.075)
        self.declare_parameter("m27_min_size_m", 0.038)
        self.declare_parameter("m27_max_size_m", 0.052)
        self.declare_parameter("duplicate_center_factor", 0.55)

        self.declare_parameter("sequence_stable_frames", 3)
        self.declare_parameter("sequence_confirmation_window_s", 3.0)
        self.declare_parameter("sequence_max_center_shift_px", 40.0)
        self.declare_parameter("sequence_max_radius_change_ratio", 0.20)
        self.declare_parameter("sequence_max_gap_s", 1.0)

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.use_tf = bool(self.get_parameter("use_tf").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.max_image_age_s = float(self.get_parameter("max_image_age_s").value)
        self.max_depth_age_s = float(self.get_parameter("max_depth_age_s").value)
        self.min_nut_clearance_m = float(self.get_parameter("min_nut_clearance_m").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)

        p = lambda name: self.get_parameter(name).value
        detector_config = DetectorConfig(
            frame_size_m=float(p("frame_size_m")),
            black_v_max=int(p("black_v_max")),
            debug_black_frame=bool(p("debug_black_frame")),
            min_frame_area_ratio=float(p("min_frame_area_ratio")),
            max_frame_area_ratio=float(p("max_frame_area_ratio")),
            frame_inner_scale=float(p("frame_inner_scale")),
            min_nut_radius_px=float(p("min_nut_radius_px")),
            max_nut_radius_px=float(p("max_nut_radius_px")),
            min_nut_area_px=float(p("min_nut_area_px")),
            max_nut_area_px=float(p("max_nut_area_px")),
            adaptive_block_size=int(p("adaptive_block_size")),
            adaptive_c=float(p("adaptive_c")),
            blackhat_kernel_size=int(p("blackhat_kernel_size")),
            blackhat_threshold=float(p("blackhat_threshold")),
            nut_mask_open_kernel_size=int(p("nut_mask_open_kernel_size")),
            nut_mask_close_kernel_size=int(p("nut_mask_close_kernel_size")),
            min_nut_solidity=float(p("min_nut_solidity")),
            min_nut_circularity=float(p("min_nut_circularity")),
            min_nut_aspect_ratio=float(p("min_nut_aspect_ratio")),
            silver_gray_min=float(p("silver_gray_min")),
            silver_black_margin=float(p("silver_black_margin")),
            m45_min_size_m=float(p("m45_min_size_m")),
            m45_max_size_m=float(p("m45_max_size_m")),
            m33_min_size_m=float(p("m33_min_size_m")),
            m33_max_size_m=float(p("m33_max_size_m")),
            m27_min_size_m=float(p("m27_min_size_m")),
            m27_max_size_m=float(p("m27_max_size_m")),
            duplicate_center_factor=float(p("duplicate_center_factor")),
        )
        sequence_config = SequenceConfig(
            stable_frames=int(p("sequence_stable_frames")),
            confirmation_window_s=float(p("sequence_confirmation_window_s")),
            max_center_shift_px=float(p("sequence_max_center_shift_px")),
            max_radius_change_ratio=float(p("sequence_max_radius_change_ratio")),
            max_gap_s=float(p("sequence_max_gap_s")),
            max_observation_age_s=self.max_image_age_s,
            max_depth_age_s=self.max_depth_age_s,
        )
        self.detector = AdvancedNutDetector(detector_config)
        self.sequence = AdvancedNutSequence(sequence_config)
        self.latest_color: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.latest_info: Optional[CameraInfo] = None
        self.last_processed_stamp_ns = -1
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion: Optional[np.ndarray] = None

        self.detections_publisher = self.create_publisher(
            PoseArray, str(p("detection_topic")), 10)
        self.sequence_publisher = self.create_publisher(
            String, str(p("sequence_topic")), 10)
        self.status_publisher = self.create_publisher(
            String, str(p("sequence_status_topic")), 10)
        self.debug_publisher = self.create_publisher(
            Image, str(p("debug_image_topic")), 2)

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        color_subscriber = message_filters.Subscriber(
            self, Image, str(p("color_topic")), qos_profile=sensor_qos)
        depth_subscriber = message_filters.Subscriber(
            self, Image, str(p("depth_topic")), qos_profile=sensor_qos)
        info_subscriber = message_filters.Subscriber(
            self, CameraInfo, str(p("camera_info_topic")), qos_profile=sensor_qos)
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_subscriber, depth_subscriber, info_subscriber],
            int(p("queue_size")), float(p("synchronize_slop")))
        synchronizer.registerCallback(self.on_images)
        self.tf_buffer = Buffer()
        self._sequence_event_lock = threading.Lock()
        self._last_event_sequence = 0
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sequence_event_services = {
            action: self.create_service(
                Trigger, str(p(f"sequence_{action}_service")),
                lambda request, response, action=action: self._on_sequence_event(
                    request, response, action))
            for action in ("start", "complete", "retry", "reset")
        }
        self.create_timer(1.0 / max(0.1, self.publish_rate_hz), self.process)
        self.get_logger().info("Advanced four-nut detector started; no motion commands are sent")

    @staticmethod
    def image_to_cv(image: Image) -> np.ndarray:
        if image.encoding == "rgb8":
            data = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        if image.encoding == "bgr8":
            return np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
        if image.encoding == "rgba8":
            data = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 4)
            return cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
        raise ValueError(f"unsupported color encoding {image.encoding}")

    @staticmethod
    def depth_to_cv(image: Image) -> np.ndarray:
        if image.encoding == "16UC1":
            return np.frombuffer(image.data, dtype=np.uint16).reshape(image.height, image.width)
        if image.encoding == "32FC1":
            return np.frombuffer(image.data, dtype=np.float32).reshape(image.height, image.width)
        raise ValueError(f"unsupported depth encoding {image.encoding}")

    def on_images(self, color: Image, depth: Image, info: CameraInfo):
        self.latest_color = color
        self.latest_depth = depth
        self.latest_info = info

    @staticmethod
    def cv_to_image(image: np.ndarray, header, encoding: str = "bgr8") -> Image:
        message = Image()
        message.header = header
        message.height, message.width = image.shape[:2]
        message.encoding = encoding
        message.step = message.width * (1 if image.ndim == 2 else image.shape[2])
        message.data = image.tobytes()
        return message

    @staticmethod
    def transform_to_matrix(transform) -> tuple[np.ndarray, np.ndarray]:
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        x, y, z, w = (transform.transform.rotation.x, transform.transform.rotation.y,
                      transform.transform.rotation.z, transform.transform.rotation.w)
        rotation = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ])
        return rotation, translation

    def sample_depth(self, depth: np.ndarray, pixel: tuple[float, float]) -> float:
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            return float("nan")
        values = []
        for y in range(max(0, v - 3), min(depth.shape[0], v + 4)):
            for x in range(max(0, u - 3), min(depth.shape[1], u + 4)):
                value = float(depth[y, x])
                if self.depth_scale > 0 and depth.dtype == np.uint16:
                    value *= self.depth_scale
                if np.isfinite(value) and self.depth_min_m <= value <= self.depth_max_m:
                    values.append(value)
        return float(np.median(values)) if values else float("nan")

    def pixel_to_camera(self, pixel: tuple[float, float], depth: float) -> np.ndarray:
        undistorted = cv2.undistortPoints(
            np.array([[pixel]], dtype=np.float32), self.camera_matrix, self.distortion).reshape(1, 2)
        return np.array([undistorted[0, 0] * depth, undistorted[0, 1] * depth, depth])

    def transform_point(self, camera_point: np.ndarray, header):
        if not self.use_tf or self.target_frame == header.frame_id:
            return camera_point, header.frame_id
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, header.frame_id, rclpy.time.Time())
            rotation, translation = self.transform_to_matrix(transform)
            return rotation @ camera_point + translation, self.target_frame
        except TransformException:
            return camera_point, header.frame_id

    def _on_sequence_event(self, request, response, action: str):
        """Record a task event and return whether the sequence accepted it."""
        with self._sequence_event_lock:
            event_sequence = self._last_event_sequence + 1
            target_id = 0 if action == "reset" else self.sequence.snapshot.current_target_id
            accepted, reason = self.sequence.event(
                self.sequence.snapshot.round_id, event_sequence, target_id, action)
            if accepted:
                self._last_event_sequence = event_sequence
            self.publish_state({})
        response.success = bool(accepted)
        response.message = reason
        return response

    def process(self):
        if self.latest_color is None or self.latest_depth is None or self.latest_info is None:
            return
        color, depth, info = self.latest_color, self.latest_depth, self.latest_info
        stamp_ns = rclpy.time.Time().from_msg(color.header.stamp).nanoseconds()
        if stamp_ns == self.last_processed_stamp_ns:
            return
        now_ns = self.get_clock().now().nanoseconds()
        if now_ns - stamp_ns > int(self.max_image_age_s * 1e9):
            self.sequence.observe(stamp_ns, False, ())
            self.publish_state({})
            return
        if abs((rclpy.time.Time().from_msg(color.header.stamp) -
                rclpy.time.Time().from_msg(depth.header.stamp)).nanoseconds()) > int(self.max_depth_age_s * 1e9):
            self.sequence.observe(stamp_ns, False, ())
            self.publish_state({})
            return
        try:
            color_image = self.image_to_cv(color)
            depth_image = self.depth_to_cv(depth)
            self.camera_matrix = np.array(info.k, dtype=np.float64).reshape(3, 3)
            self.distortion = np.array(info.d, dtype=np.float64)
        except (ValueError, TypeError) as error:
            self.get_logger().warning(f"image conversion failed: {error}", throttle_duration_sec=5.0)
            return
        if color_image.shape[:2] != depth_image.shape:
            self.get_logger().warning("color and depth dimensions differ", throttle_duration_sec=5.0)
            return
        self.last_processed_stamp_ns = stamp_ns
        detection = self.detector.detect(color_image, reuse_frame=True)
        if self.debug_publisher.get_subscription_count():
            self.debug_publisher.publish(self.cv_to_image(detection.annotated, color.header))
        observations = tuple(
            Observation(candidate.center, candidate.radius_px, candidate.size_m, candidate.mean_gray)
            for candidate in detection.candidates if candidate.accepted
        )
        snapshot = self.sequence.observe(stamp_ns, detection.frame_found, observations)
        if not snapshot.observation_valid:
            self.publish_state({})
            return
        positions = {}
        poses = PoseArray()
        poses.header = color.header
        for target_id, target in enumerate(snapshot.targets, start=1):
            if not target.visible:
                continue
            depth = self.sample_depth(depth_image, target.pixel)
            if not np.isfinite(depth):
                self.publish_state({})
                return
            point, frame_id = self.transform_point(self.pixel_to_camera(target.pixel, depth), color.header)
            positions[target_id] = {
                "label": target.label, "x": float(point[0]), "y": float(point[1]),
                "z": float(point[2]), "frame": frame_id,
            }
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, point)
            pose.orientation.w = 1.0
            poses.poses.append(pose)
            target.position = (float(point[0]), float(point[1]), float(point[2]))
            target.frame_id = frame_id
        if len(positions) != snapshot.expected_count:
            self.publish_state({})
            return
        poses.header.frame_id = self.target_frame if self.use_tf else color.header.frame_id
        self.detections_publisher.publish(poses)
        self.publish_state(positions)

    def publish_state(self, positions: dict):
        snapshot = self.sequence.snapshot
        state = {
            "status": snapshot.status,
            "round_id": snapshot.round_id,
            "initialized": snapshot.initialized,
            "observation_valid": snapshot.observation_valid,
            "observed_count": snapshot.observed_count,
            "expected_count": snapshot.expected_count,
            "current_target_id": snapshot.current_target_id,
            "targets": [
                {"id": target.id, "label": target.label, "state": target.state,
                 "visible": target.visible, "position": positions.get(target.id)}
                for target in snapshot.targets
            ],
        }
        self.sequence_publisher.publish(String(data=json.dumps(state)))
        self.status_publisher.publish(String(data=json.dumps({"status": state["status"], "count": state["observed_count"]})))


def main():
    rclpy.init()
    node = AdvancedNutDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
