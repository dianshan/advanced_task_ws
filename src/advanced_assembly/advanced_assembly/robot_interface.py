"""Generic ROS adapter for the left/right Dexterous-Hand arms and O6 hands."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from lbot_arm_interfaces.srv import ForwardKinematics, InverseKinematics, MoveJ, MoveJP, MoveL, SetEmergency
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8MultiArray


@dataclass(frozen=True)
class ArmMotionOptions:
    speed: float
    acceleration: float
    block: bool = True


@dataclass
class DeviceResult:
    success: bool
    message: str = ""

    @classmethod
    def ok(cls, message: str = "") -> cls:
        return cls(True, message)

    @classmethod
    def fail(cls, message: str) -> cls:
        return cls(False, message)


class Arm:
    def __init__(self, node: Node, namespace: str, base_frame: str,
                 joint_min: tuple[float, ...], joint_max: tuple[float, ...],
                 joint_margin_rad: float):
        self.node = node
        self.base_frame = base_frame
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.joint_margin_rad = joint_margin_rad
        self.endpoint = lambda suffix: f"{namespace.rstrip('/')}/{suffix}"
        self.joints: Optional[tuple[float, ...]] = None
        self.joint_stamp = None
        self.pose: Optional[PoseStamped] = None
        self.joint_subscription = node.create_subscription(
            JointState, self.endpoint("joint_states"), self._on_joint_state, 10)
        self.pose_subscription = node.create_subscription(
            PoseStamped, self.endpoint("pose_states"), self._on_pose, 10)
        self.move_j = node.create_client(MoveJ, self.endpoint("move_joint"))
        self.move_jp = node.create_client(MoveJP, self.endpoint("move_pose"))
        self.move_l = node.create_client(MoveL, self.endpoint("move_linear"))
        self.ik = node.create_client(InverseKinematics, self.endpoint("inverse_kinematics"))
        self.fk = node.create_client(ForwardKinematics, self.endpoint("forward_kinematics"))
        self.emergency = node.create_client(SetEmergency, self.endpoint("set_emergency_stop"))
        self.speed_publisher = node.create_publisher(UInt8MultiArray, self.endpoint("hand/set_l6_speed"), 10)
        self.force_publisher = node.create_publisher(UInt8MultiArray, self.endpoint("hand/set_l6_force"), 10)
        self.hand_publisher = node.create_publisher(UInt8MultiArray, self.endpoint("hand/set_l6_joint"), 10)

    def _on_joint_state(self, message: JointState):
        if len(message.position) == 7:
            self.joints = tuple(float(value) for value in message.position)
            self.joint_stamp = self.node.get_clock().now().nanoseconds()

    def _on_pose(self, message: PoseStamped):
        self.pose = message

    def joints_within_limits(self, joints) -> bool:
        if joints is None or len(joints) != 7 or any(value is None for value in joints):
            return False
        return all(
            self.joint_min[index] + self.joint_margin_rad <= joints[index] <=
            self.joint_max[index] - self.joint_margin_rad for index in range(7))

    def wait_for_state(self, timeout_s: float) -> DeviceResult:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            if self.joints is not None:
                return DeviceResult.ok()
            time.sleep(0.02)
        return DeviceResult.fail(f"{self.endpoint('joint_states')} unavailable")

    def _call(self, client, request, label: str, timeout_s: float) -> DeviceResult:
        if not client.wait_for_service(timeout_sec=timeout_s):
            return DeviceResult.fail(f"{client.srv_name} service unavailable")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.02)
        response = future.result()
        if not getattr(response, "success", False):
            return DeviceResult.fail(f"{label} rejected by controller")
        return DeviceResult.ok(label)

    def move_joints(self, joints, options: ArmMotionOptions, label: str, timeout_s: float) -> DeviceResult:
        if not self.joints_within_limits(joints):
            return DeviceResult.fail(f"{label} violates configured joint limits")
        request = MoveJ.Request()
        request.joints = [float(value) for value in joints]
        request.speed, request.acce, request.block = options.speed, options.acceleration, options.block
        return self._call(self.move_j, request, label, timeout_s)

    def move_pose(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                  options: ArmMotionOptions, label: str, timeout_s: float) -> DeviceResult:
        request = MoveJP.Request()
        request.position.x, request.position.y, request.position.z = x, y, z
        request.euler.x, request.euler.y, request.euler.z = roll, pitch, yaw
        request.speed, request.acce, request.block = options.speed, options.acceleration, options.block
        return self._call(self.move_jp, request, label, timeout_s)

    def move_linear(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                    options: ArmMotionOptions, label: str, timeout_s: float) -> DeviceResult:
        request = MoveL.Request()
        request.position.x, request.position.y, request.position.z = x, y, z
        request.euler.x, request.euler.y, request.euler.z = roll, pitch, yaw
        request.speed = options.speed
        request.acce = options.acceleration
        request.block = options.block
        return self._call(self.move_l, request, label, timeout_s)

    def inverse_kinematics(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                           seed, timeout_s: float):
        request = InverseKinematics.Request()
        request.position.x, request.position.y, request.position.z = x, y, z
        request.euler.x, request.euler.y, request.euler.z = roll, pitch, yaw
        request.joints = [float(value) for value in seed] if seed else []
        if not self.ik.wait_for_service(timeout_sec=timeout_s):
            return None, DeviceResult.fail(f"{self.ik.srv_name} service unavailable")
        future = self.ik.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.02)
        response = future.result()
        if not response.success or len(response.joints) != 7:
            return None, DeviceResult.fail("inverse_kinematics failed")
        joints = tuple(float(value) for value in response.joints)
        if not self.joints_within_limits(joints):
            return None, DeviceResult.fail("inverse_kinematics result violates joint limits")
        return joints, DeviceResult.ok("inverse_kinematics accepted")

    def set_hand(self, values, speed: int, force: int, repeat: int = 3) -> DeviceResult:
        if len(values) != 6 or any(value < 0 or value > 255 for value in values):
            return DeviceResult.fail("hand values must contain six values in 0..255")
        speed_message, force_message, hand_message = UInt8MultiArray(), UInt8MultiArray(), UInt8MultiArray()
        speed_message.data = [int(speed)] * 6
        force_message.data = [int(force)] * 6
        hand_message.data = [int(value) for value in values]
        for _ in range(max(1, repeat)):
            self.speed_publisher.publish(speed_message)
            self.force_publisher.publish(force_message)
            self.hand_publisher.publish(hand_message)
        return DeviceResult.ok("hand command published (open-loop, no hand feedback available)")

    def emergency_stop(self, emergency: bool = True, timeout_s: float = 3.0) -> DeviceResult:
        request = SetEmergency.Request()
        request.emergency = emergency
        return self._call(self.emergency, request, "emergency_stop", timeout_s)
