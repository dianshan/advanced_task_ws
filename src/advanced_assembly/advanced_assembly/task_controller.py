"""Task controller for four left-to-right handovers and vertical-axis assembly.

The controller is deliberately split into small stage methods so one nut can be
run independently while all four run through the same code path in full mode.
All motion is disabled by default and requires calibration flags plus an
explicit execute flag.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from std_msgs.msg import String

from .motion_geometry import arm_tip_for_tcp
from .robot_interface import Arm, ArmMotionOptions, DeviceResult

ORDER = ("m45_silver", "m45_black", "m33", "m27")


class TaskMode(str, Enum):
    VALIDATE = "validate"
    SINGLE = "single"
    FULL = "full"


@dataclass
class Axis:
    x: float
    y: float
    top_z: float
    frame_id: str = "base_link"


@dataclass
class Target:
    id: int
    label: str
    x: float
    y: float
    z: float


@dataclass
class TaskResult:
    success: bool
    message: str


class AssemblyTaskController(Node):
    def __init__(self):
        super().__init__("advanced_assembly_task")
        self.declare_parameter("execute_task", False)
        self.declare_parameter("task_mode", "validate")
        self.declare_parameter("target_id", 1)
        self.declare_parameter("robot_namespace", "/robot1")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("vision_calibrated", False)
        self.declare_parameter("left_route_calibrated", False)
        self.declare_parameter("right_route_calibrated", False)
        self.declare_parameter("hand_calibrated", False)
        self.declare_parameter("axis_calibrated", False)
        self.declare_parameter("middle_position_calibrated", False)
        self.declare_parameter("joint_limit_margin_rad", 0.05)
        self.declare_parameter("joint_speed", 0.08)
        self.declare_parameter("joint_acceleration", 0.08)
        self.declare_parameter("cartesian_speed", 0.03)
        self.declare_parameter("cartesian_acceleration", 0.03)
        self.declare_parameter("service_timeout_s", 30.0)
        self.declare_parameter("grip_settle_s", 1.0)
        self.declare_parameter("vision_wait_timeout_s", 30.0)
        self.declare_parameter("axis_wait_timeout_s", 30.0)
        self.declare_parameter("sequence_topic", "/advanced_nut_sequence")
        self.declare_parameter("sequence_start_service", "/advanced_nut_sequence/event/start")
        self.declare_parameter("sequence_complete_service", "/advanced_nut_sequence/event/complete")
        self.declare_parameter("axis_pose_topic", "/axis_detection/pose")
        self.declare_parameter("status_topic", "/advanced_assembly/status")
        self.declare_parameter("hand_speed", 40)
        self.declare_parameter("hand_force", 30)

        self.declare_parameter("left_joint_min", [-2.91, -0.07, -2.72, -2.05, -2.69, -1.59, -1.59])
        self.declare_parameter("left_joint_max", [2.91, 3.20, 2.69, 2.05, 2.69, 1.59, 1.59])
        self.declare_parameter("right_joint_min", [-2.91, -3.20, -2.69, -2.05, -2.69, -1.59, -1.59])
        self.declare_parameter("right_joint_max", [2.91, 0.07, 2.69, 2.05, 2.69, 1.59, 1.59])
        self.declare_parameter("left_enter_route.names", [])
        self.declare_parameter("left_home_route.names", [])
        self.declare_parameter("right_enter_route.names", [])
        self.declare_parameter("right_home_route.names", [])
        self.declare_parameter("middle_position", [0.0, 0.0, 0.0])
        self.declare_parameter("axis_above_height", 0.080)
        self.declare_parameter("axis_descend_height", 0.020)
        self.declare_parameter("axis_release_lift_height", 0.100)
        self.declare_parameter("middle_lift_height", 0.080)
        self.declare_parameter("left_grasp_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("right_grasp_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("left_tcp_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("right_tcp_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("hand_open", [255, 40, 255, 255, 255, 255])
        self.declare_parameter("hand_ready", [240, 30, 180, 180, 180, 180])
        self.declare_parameter("hand_closed_m45", [0, 30, 0, 0, 0, 0])
        self.declare_parameter("hand_closed_m33", [0, 30, 0, 0, 0, 0])
        self.declare_parameter("hand_closed_m27", [0, 30, 0, 0, 0, 0])

        self.execute_task = bool(self.get_parameter("execute_task").value)
        mode_text = str(self.get_parameter("task_mode").value)
        self.task_mode = TaskMode(mode_text)
        self.base_frame = str(self.get_parameter("base_frame").value)
        namespace = str(self.get_parameter("robot_namespace").value)
        self.motion_options = ArmMotionOptions(
            float(self.get_parameter("joint_speed").value),
            float(self.get_parameter("joint_acceleration").value))
        self.cartesian_options = ArmMotionOptions(
            float(self.get_parameter("cartesian_speed").value),
            float(self.get_parameter("cartesian_acceleration").value))
        self.service_timeout_s = float(self.get_parameter("service_timeout_s").value)
        self.grip_settle_s = float(self.get_parameter("grip_settle_s").value)
        self.hand_speed = int(self.get_parameter("hand_speed").value)
        self.hand_force = int(self.get_parameter("hand_force").value)
        self.axis: Optional[Axis] = None
        self.target: Optional[Target] = None
        self.targets: dict[int, Target] = {}
        self.motion_started = False
        self.lock = threading.Lock()

        left_min = tuple(map(float, self.get_parameter("left_joint_min").value))
        left_max = tuple(map(float, self.get_parameter("left_joint_max").value))
        right_min = tuple(map(float, self.get_parameter("right_joint_min").value))
        right_max = tuple(map(float, self.get_parameter("right_joint_max").value))
        self.left = Arm(self, namespace, self.base_frame, left_min, left_max,
                        float(self.get_parameter("joint_limit_margin_rad").value))
        self.right = Arm(self, namespace, self.base_frame, right_min, right_max,
                         float(self.get_parameter("joint_limit_margin_rad").value))

        self.sequence_subscription = self.create_subscription(
            String, str(self.get_parameter("sequence_topic").value),
            self._on_sequence, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.axis_subscription = self.create_subscription(
            PointStamped, str(self.get_parameter("axis_pose_topic").value),
            self._on_axis, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10)
        self.sequence_event_clients = {
            action: self.create_client(
                Trigger, str(self.get_parameter(f"sequence_{action}_service").value))
            for action in ("start", "complete")
        }
        self.executor: Optional[MultiThreadedExecutor] = None
        self.spin_thread: Optional[threading.Thread] = None

    def _on_sequence(self, message: String):
        try:
            state = json.loads(message.data)
            if not state.get("observation_valid"):
                return
            for item in state.get("targets", []):
                position = item.get("position")
                if position is None:
                    continue
                target = Target(
                    int(item["id"]), str(item["label"]), float(position["x"]),
                    float(position["y"]), float(position["z"]))
                with self.lock:
                    self.targets[target.id] = target
                    self.target = target if item.get("id") == state.get("current_target_id") else self.target
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning(f"invalid sequence state: {error}", throttle_duration_sec=2.0)

    def _sequence_event(self, action: str) -> TaskResult:
        client = self.sequence_event_clients[action]
        if not client.wait_for_service(timeout_sec=self.service_timeout_s):
            return TaskResult(False, f"sequence event service unavailable ({action})")
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + self.service_timeout_s
        while not future.done() and time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.02)
        if not future.done():
            return TaskResult(False, f"sequence event {action} timeout")
        response = future.result()
        if response is None:
            return TaskResult(False, f"sequence event {action} returned no response")
        if not response.success:
            return TaskResult(False, f"sequence event {action} rejected: {response.message}")
        return TaskResult(True, f"sequence event {action} accepted")

    def _on_axis(self, message: PointStamped):
        if message.header.frame_id != self.base_frame:
            self.get_logger().warning(
                f"axis frame {message.header.frame_id} is not {self.base_frame}",
                throttle_duration_sec=2.0)
            return
        with self.lock:
            self.axis = Axis(message.pose.position.x, message.pose.position.y,
                             message.pose.position.z, message.header.frame_id)

    def _joint_route(self, arm: Arm, side: str, route_name: str) -> list[tuple[str, tuple[float, ...]]]:
        names = [str(value) for value in self.get_parameter(f"{side}_{route_name}.names").value]
        route = []
        for name in names:
            values = tuple(map(float, self.get_parameter(f"{side}_{route_name}.{name}").value))
            if len(values) != 7:
                raise ValueError(f"{side}_{route_name}.{name} must contain 7 joint radians")
            route.append((name, values))
        return route

    def _execute_route(self, arm: Arm, side: str, route_name: str, reverse: bool = False) -> TaskResult:
        route = self._joint_route(arm, side, route_name)
        if not route:
            return TaskResult(False, f"{side}_{route_name} is empty")
        if reverse:
            route = list(reversed(route))
        for name, joints in route:
            label = f"{side} {route_name} {name}"
            result = arm.move_joints(joints, self.motion_options, label, self.service_timeout_s)
            if not result.success:
                return TaskResult(False, f"{label} failed: {result.message}")
            time.sleep(0.2)
        return TaskResult(True, f"{side} {route_name} completed")

    def _wait_for_target(self, timeout_s: float, target_id: int) -> TaskResult:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            with self.lock:
                target = self.targets.get(target_id)
            if target is not None:
                return TaskResult(True, f"target {ORDER[target_id - 1]} captured at ({target.x:.4f},{target.y:.4f},{target.z:.4f})")
            time.sleep(0.05)
        return TaskResult(False, f"timed out waiting for target {target_id} ({ORDER[target_id - 1]})")

    def _wait_for_axis(self, timeout_s: float) -> TaskResult:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            with self.lock:
                axis = self.axis
            if axis is not None:
                return TaskResult(True, f"axis captured at ({axis.x:.4f},{axis.y:.4f},{axis.top_z:.4f})")
            time.sleep(0.05)
        return TaskResult(False, "timed out waiting for axis")

    def _rpy(self, name: str) -> tuple[float, float, float]:
        values = [float(value) for value in self.get_parameter(name).value]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} must contain three finite [roll, pitch, yaw] values")
        return values[0], values[1], values[2]

    def _tcp_offset(self, side: str) -> tuple[float, float, float]:
        values = [float(value) for value in self.get_parameter(f"{side}_tcp_offset").value]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{side}_tcp_offset must contain three finite values")
        return values[0], values[1], values[2]

    def _set_hand(self, arm: Arm, parameter: str, label: str) -> TaskResult:
        values = [int(value) for value in self.get_parameter(parameter).value]
        result = arm.set_hand(values, self.hand_speed, self.hand_force)
        if not result.success:
            return TaskResult(False, f"{label} failed: {result.message}")
        time.sleep(self.grip_settle_s)
        return TaskResult(True, label)

    def _required_flags(self):
        flags = {
            "vision_calibrated": bool(self.get_parameter("vision_calibrated").value),
            "left_route_calibrated": bool(self.get_parameter("left_route_calibrated").value),
            "right_route_calibrated": bool(self.get_parameter("right_route_calibrated").value),
            "hand_calibrated": bool(self.get_parameter("hand_calibrated").value),
            "axis_calibrated": bool(self.get_parameter("axis_calibrated").value),
            "middle_position_calibrated": bool(self.get_parameter("middle_position_calibrated").value),
        }
        return flags

    def _validate(self):
        if self.task_mode not in (TaskMode.SINGLE, TaskMode.FULL):
            return TaskResult(True, "validation only; no motion requested")
        missing = [name for name, value in self._required_flags().items() if not value]
        if missing:
            return TaskResult(False, "calibration flags not enabled: " + ", ".join(missing))
        numeric_parameters = {
            "joint_speed": self.motion_options.speed,
            "joint_acceleration": self.motion_options.acceleration,
            "cartesian_speed": self.cartesian_options.speed,
            "cartesian_acceleration": self.cartesian_options.acceleration,
            "middle_lift_height": float(self.get_parameter("middle_lift_height").value),
            "axis_above_height": float(self.get_parameter("axis_above_height").value),
            "axis_descend_height": float(self.get_parameter("axis_descend_height").value),
            "axis_release_lift_height": float(self.get_parameter("axis_release_lift_height").value),
        }
        for name, value in numeric_parameters.items():
            if not math.isfinite(value) or value <= 0.0:
                return TaskResult(False, f"{name} must be finite and positive")
        if self.grip_settle_s < 0.0 or self.service_timeout_s <= 0.0:
            return TaskResult(False, "grip_settle_s must be non-negative and service_timeout_s positive")
        if not 0 <= self.hand_speed <= 255 or not 0 <= self.hand_force <= 255:
            return TaskResult(False, "hand_speed and hand_force must be in 0..255")
        for side in ("left", "right"):
            self._rpy(f"{side}_grasp_rpy")
            try:
                self._tcp_offset(side)
            except ValueError as error:
                return TaskResult(False, str(error))
        middle = [float(value) for value in self.get_parameter("middle_position").value]
        if len(middle) != 3 or not all(math.isfinite(value) for value in middle):
            return TaskResult(False, "middle_position must contain three finite values")
        axis_above = numeric_parameters["axis_above_height"]
        axis_descend = numeric_parameters["axis_descend_height"]
        if axis_above <= axis_descend:
            return TaskResult(False, "axis_above_height must be greater than axis_descend_height")
        for hand_parameter in ("hand_open", "hand_ready", "hand_closed_m45",
                               "hand_closed_m33", "hand_closed_m27"):
            values = [int(value) for value in self.get_parameter(hand_parameter).value]
            if len(values) != 6 or any(value < 0 or value > 255 for value in values):
                return TaskResult(False, f"{hand_parameter} must contain six values in 0..255")
        for side in ("left", "right"):
            for route_name in ("enter_route", "home_route"):
                for _, joints in self._joint_route(getattr(self, side), side, route_name):
                    if not getattr(self, side).joints_within_limits(joints):
                        return TaskResult(False, f"{side} {route_name} violates configured joint limits")
        for arm in (self.left, self.right):
            state = arm.wait_for_state(self.service_timeout_s)
            if not state.success:
                return TaskResult(False, state.message)
        return TaskResult(True, "configuration and arm feedback validation passed")

    def _left_grasp_to_middle(self, target: Target) -> TaskResult:
        roll, pitch, yaw = self._rpy("left_grasp_rpy")
        tcp_offset = self._tcp_offset("left")
        lift = float(self.get_parameter("middle_lift_height").value)
        approach_tcp = (target.x, target.y, target.z + lift)
        grasp_tcp = (target.x, target.y, target.z)
        middle_tcp = tuple(float(value) for value in self.get_parameter("middle_position").value)
        approach_arm = arm_tip_for_tcp(*approach_tcp, roll, pitch, yaw, tcp_offset)
        grasp_arm = arm_tip_for_tcp(*grasp_tcp, roll, pitch, yaw, tcp_offset)
        lift_arm = arm_tip_for_tcp(
            target.x, target.y, target.z + lift, roll, pitch, yaw, tcp_offset)
        middle_arm = arm_tip_for_tcp(*middle_tcp, roll, pitch, yaw, tcp_offset)
        label = f"left grasp {target.label}"

        result = self._set_hand(self.left, "hand_ready", "left hand ready")
        if not result.success:
            return result
        result = self.left.move_pose(
            *approach_arm, roll, pitch, yaw, self.cartesian_options,
            f"{label} approach", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        result = self.left.move_linear(
            *grasp_arm, roll, pitch, yaw, self.cartesian_options,
            f"{label} descent", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        close_parameter = f"hand_closed_{target.label.split('_')[0]}"
        result = self._set_hand(self.left, close_parameter, f"left close {target.label}")
        if not result.success:
            return result
        result = self.left.move_linear(
            *lift_arm, roll, pitch, yaw, self.cartesian_options,
            f"{label} lift", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        result = self.left.move_pose(
            *middle_arm, roll, pitch, yaw, self.cartesian_options,
            f"left place {target.label} at middle position", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        result = self._set_hand(self.left, "hand_open", f"left release {target.label}")
        if not result.success:
            return result
        return TaskResult(True, f"{target.label} moved to middle position")

    def _right_middle_to_axis(self, target: Target) -> TaskResult:
        axis = self.axis
        if axis is None:
            return TaskResult(False, "axis is not captured")
        roll, pitch, yaw = self._rpy("right_grasp_rpy")
        tcp_offset = self._tcp_offset("right")
        lift = float(self.get_parameter("middle_lift_height").value)
        middle_tcp = tuple(float(value) for value in self.get_parameter("middle_position").value)
        approach_arm = arm_tip_for_tcp(
            middle_tcp[0], middle_tcp[1], middle_tcp[2] + lift, roll, pitch, yaw, tcp_offset)
        grasp_arm = arm_tip_for_tcp(*middle_tcp, roll, pitch, yaw, tcp_offset)
        lift_arm = arm_tip_for_tcp(
            middle_tcp[0], middle_tcp[1], middle_tcp[2] + lift, roll, pitch, yaw, tcp_offset)
        above_arm = arm_tip_for_tcp(
            axis.x, axis.y, axis.top_z + float(self.get_parameter("axis_above_height").value),
            roll, pitch, yaw, tcp_offset)
        descend_arm = arm_tip_for_tcp(
            axis.x, axis.y, axis.top_z + float(self.get_parameter("axis_descend_height").value),
            roll, pitch, yaw, tcp_offset)
        retreat_arm = arm_tip_for_tcp(
            axis.x, axis.y, axis.top_z + float(self.get_parameter("axis_descend_height").value) +
            float(self.get_parameter("axis_release_lift_height").value),
            roll, pitch, yaw, tcp_offset)

        result = self._set_hand(self.right, "hand_ready", "right hand ready")
        if not result.success:
            return result
        result = self.right.move_pose(
            *approach_arm, roll, pitch, yaw, self.cartesian_options,
            f"right approach {target.label} at middle", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        result = self.right.move_linear(
            *grasp_arm, roll, pitch, yaw, self.cartesian_options,
            f"right grasp descent {target.label}", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        close_parameter = f"hand_closed_{target.label.split('_')[0]}"
        result = self._set_hand(self.right, close_parameter, f"right close {target.label}")
        if not result.success:
            return result
        result = self.right.move_linear(
            *lift_arm, roll, pitch, yaw, self.cartesian_options,
            f"right lift {target.label}", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        result = self.right.move_pose(
            *above_arm, roll, pitch, yaw, self.cartesian_options,
            f"axis above {target.label}", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        result = self.right.move_linear(
            *descend_arm, roll, pitch, yaw, self.cartesian_options,
            f"axis threading descent {target.label}", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        result = self._set_hand(self.right, "hand_open", f"axis release {target.label}")
        if not result.success:
            return result
        result = self.right.move_linear(
            *retreat_arm, roll, pitch, yaw, self.cartesian_options,
            f"axis retreat {target.label}", self.service_timeout_s)
        if not result.success:
            return TaskResult(False, result.message)
        return TaskResult(True, f"{target.label} released above axis")

    def run_single(self, target_id: int) -> TaskResult:
        validation = self._validate()
        if not validation.success:
            return validation
        if not 1 <= target_id <= 4:
            return TaskResult(False, "target_id must be 1..4")
        axis_result = self._wait_for_axis(float(self.get_parameter("axis_wait_timeout_s").value))
        if not axis_result.success:
            return axis_result
        target_result = self._wait_for_target(
            float(self.get_parameter("vision_wait_timeout_s").value), target_id)
        if not target_result.success:
            return target_result
        with self.lock:
            target = self.targets[target_id]
        if self.task_mode == TaskMode.FULL:
            event = self._sequence_event("start")
            if not event.success:
                return event
        left_route = self._execute_route(self.left, "left", "enter_route")
        if not left_route.success:
            return left_route
        self.motion_started = True
        result = self._left_grasp_to_middle(target)
        if not result.success:
            return result
        home = self._execute_route(self.left, "left", "home_route", reverse=True)
        if not home.success:
            return home
        right_route = self._execute_route(self.right, "right", "enter_route")
        if not right_route.success:
            return right_route
        result = self._right_middle_to_axis(target)
        if not result.success:
            return result
        home = self._execute_route(self.right, "right", "home_route", reverse=True)
        if not home.success:
            return home
        if self.task_mode == TaskMode.FULL:
            event = self._sequence_event("complete")
            if not event.success:
                return event
            with self.lock:
                self.target = None
        return TaskResult(True, f"single {target.label} assembly completed")

    def run_full(self) -> TaskResult:
        for target_id in range(1, 5):
            result = self.run_single(target_id)
            if not result.success:
                return result
        return TaskResult(True, "all four nuts assembled in required order")

    def stop(self):
        if not self.motion_started:
            return
        for arm in (self.left, self.right):
            arm.set_hand([255, 40, 255, 255, 255, 255], self.hand_speed, self.hand_force, repeat=1)
            arm.emergency_stop(True, timeout_s=3.0)

    def start(self):
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def close(self):
        if self.executor is not None:
            self.executor.remove_node(self)
            self.executor.shutdown()
        if self.spin_thread is not None:
            self.spin_thread.join(timeout=2.0)
        self.destroy_node()

    def publish_status(self, state: str, detail: str = ""):
        self.status_publisher.publish(String(data=json.dumps(
            {"state": state, "detail": detail}, ensure_ascii=False)))


def main():
    rclpy.init()
    node = AssemblyTaskController()
    exit_code = 0
    try:
        node.start()
        if not node.execute_task:
            node.publish_status("disabled", "execute_task=false; no motion command sent")
            node.get_logger().info("execute_task=false; no motion command sent")
        else:
            node.publish_status("running", node.task_mode.value)
            if node.task_mode == TaskMode.FULL:
                result = node.run_full()
            elif node.task_mode == TaskMode.SINGLE:
                result = node.run_single(int(node.get_parameter("target_id").value))
            else:
                result = node._validate()
            if result.success:
                node.publish_status("complete", result.message)
                node.get_logger().info(result.message)
            else:
                exit_code = 1
                node.publish_status("failed", result.message)
                node.get_logger().error(result.message)
                node.stop()
    except Exception as error:  # noqa: BLE001 - top-level task boundary must always report.
        exit_code = 1
        node.get_logger().error(f"task exception: {error}")
        node.publish_status("failed", str(error))
        node.stop()
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)
