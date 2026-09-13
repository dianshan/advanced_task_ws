import numpy as np
import pytest

from axis_perception.axis_detector import DepthAxisDetector, DetectorConfig


def make_scene(center, radius=0.015, height=0.150, count=180):
    rng = np.random.default_rng(42)
    plane = np.column_stack((
        rng.uniform(-0.10, 0.10, 1200),
        rng.uniform(-0.10, 0.10, 1200),
        rng.uniform(-0.001, 0.001, 1200),
    ))
    angles = np.linspace(0.0, 2.0 * np.pi, count)
    radii = radius * np.sqrt(rng.uniform(0.2, 1.0, count))
    top = np.column_stack((
        center[0] + radii * np.cos(angles),
        center[1] + radii * np.sin(angles),
        np.full(count, height),
    ))
    side = np.column_stack((
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
        rng.uniform(0.0, height, count),
    ))
    return np.vstack((plane, top, side))


def test_detects_expected_axis_center():
    detector = DepthAxisDetector(DetectorConfig())
    observation = detector.detect(make_scene((0.03, -0.02)))

    assert observation is not None
    assert observation.center_x_m == pytest.approx(0.03, abs=0.010)
    assert observation.center_y_m == pytest.approx(-0.02, abs=0.010)
    assert observation.top_z_m == pytest.approx(0.150, abs=0.005)
    assert observation.radius_m == pytest.approx(0.015, abs=0.008)


def test_rejects_flat_scene_without_axis():
    detector = DepthAxisDetector(DetectorConfig())
    assert detector.detect(make_scene((0.03, -0.02), height=0.02)) is None


def test_rejects_axis_with_wrong_radius():
    detector = DepthAxisDetector(DetectorConfig())
    assert detector.detect(make_scene((0.03, -0.02), radius=0.045)) is None
