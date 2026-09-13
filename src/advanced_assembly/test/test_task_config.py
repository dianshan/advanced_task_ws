import pytest
import yaml


def test_advanced_config_defaults_are_safe():
    with open("src/advanced_assembly/config/advanced_task.yaml", encoding="utf-8") as stream:
        parameters = yaml.safe_load(stream)["/**"]["ros__parameters"]
    assert parameters["execute_task"] is False
    for flag in ("vision_calibrated", "left_route_calibrated", "right_route_calibrated",
                 "hand_calibrated", "axis_calibrated", "middle_position_calibrated"):
        assert parameters[flag] is False
    assert len(parameters["middle_position"]) == 3
    assert parameters["axis_above_height"] > parameters["axis_descend_height"]


def test_detector_config_expects_four_targets():
    with open("src/advanced_assembly/config/advanced_task.yaml", encoding="utf-8") as stream:
        detector = yaml.safe_load(stream)["advanced_nut_detector"]["ros__parameters"]
    assert detector["frame_size_m"] == 0.200
    assert detector["sequence_stable_frames"] >= 2
    assert detector["silver_gray_min"] > 0


def test_tcp_conversion_matches_basic_task_rotation():
    from advanced_assembly.motion_geometry import arm_tip_for_tcp

    offset = (0.010185531567894372, -0.009052663438149304, -0.12065361905422836)
    roll, pitch, yaw = 1.11332116233439, -0.12826538177690483, -1.1465078720371755
    x, y, z = arm_tip_for_tcp(
        0.32, -0.12, -0.36, roll, pitch, yaw, *offset)
    assert all(abs(value) < 10.0 for value in (x, y, z))
    # The offset must disappear at identity orientation.
    assert arm_tip_for_tcp(1.0, 2.0, 3.0, 0, 0, 0, *offset) == (
        1.0 - offset[0], 2.0 - offset[1], 3.0 - offset[2])


def test_sequence_event_services_are_declared():
    with open("src/advanced_assembly/config/advanced_task.yaml", encoding="utf-8") as stream:
        sections = yaml.safe_load(stream)
    controller = sections["/**"]["ros__parameters"]
    detector = sections["advanced_nut_detector"]["ros__parameters"]
    assert controller["sequence_start_service"] == detector["sequence_start_service"]
    assert controller["sequence_complete_service"] == detector["sequence_complete_service"]
    for action in ("retry", "reset"):
        assert detector[f"sequence_{action}_service"]
