"""Pure geometry helpers shared by the controller and unit tests."""

from __future__ import annotations

import math


def arm_tip_for_tcp(tcp_x: float, tcp_y: float, tcp_z: float,
                    roll: float, pitch: float, yaw: float,
                    offset_x: float, offset_y: float, offset_z: float) -> tuple[float, float, float]:
    """Convert a grasp TCP point to the driver Arm_Tip point.

    ``offset`` is the vector from Arm_Tip to TCP in the tool frame.  This is
    the same ZYX rotation used by the basic-task motion planner; the resulting
    base-frame displacement is subtracted from the TCP point.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    base_x = (cy * cp * offset_x + (cy * sp * sr - sy * cr) * offset_y +
              (cy * sp * cr + sy * sr) * offset_z)
    base_y = (sy * cp * offset_x + (sy * sp * sr + cy * cr) * offset_y +
              (sy * sp * cr - cy * sr) * offset_z)
    base_z = (-sp * offset_x + cp * sr * offset_y + cp * cr * offset_z)
    return tcp_x - base_x, tcp_y - base_y, tcp_z - base_z
