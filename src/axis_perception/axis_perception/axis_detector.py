from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AxisObservation:
    center_x_m: float
    center_y_m: float
    top_z_m: float
    radius_m: float
    circularity: float
    inlier_count: int


@dataclass(frozen=True)
class DetectorConfig:
    workspace_min_x_m: float = -0.10
    workspace_max_x_m: float = 0.10
    workspace_min_y_m: float = -0.10
    workspace_max_y_m: float = 0.10
    plane_inlier_threshold_m: float = 0.006
    min_axis_height_m: float = 0.105
    max_axis_height_m: float = 0.185
    min_top_points: int = 60
    max_top_depth_m: float = 0.012
    cylinder_radius_m: float = 0.015
    cylinder_radius_tolerance_m: float = 0.008
    min_circularity: float = 0.65
    min_top_inlier_points: int = 80
    top_radius_quantile: float = 0.95
    top_plane_tolerance_m: float = 0.008


class DepthAxisDetector:
    """Detect a vertical cylinder from a local metric point cloud."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

    @staticmethod
    def _fit_plane(points: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        best_inliers = np.zeros(len(points), dtype=bool)
        best_count = 0
        for _ in range(100):
            indices = rng.choice(len(points), size=3, replace=False)
            first_vector = points[indices[1]] - points[indices[0]]
            second_vector = points[indices[2]] - points[indices[0]]
            normal = np.cross(first_vector, second_vector)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm
            if normal[2] < 0:
                normal = -normal
            offsets = (points - points[indices[0]]) @ normal
            inliers = np.abs(offsets) <= 0.004
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_count < 3:
            return np.array([0.0, 0.0, 1.0])

        centroid = np.mean(points[best_inliers], axis=0)
        centered = points[best_inliers] - centroid
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        normal = right_vectors[-1]
        if normal[2] < 0:
            normal = -normal
        return normal

    @staticmethod
    def _fit_circle(
        points: np.ndarray, radius_quantile: float = 0.95
    ) -> tuple[np.ndarray, float]:
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        radius = np.quantile(distances, radius_quantile)
        return center, radius

    @staticmethod
    def _circularity(points: np.ndarray, center: np.ndarray) -> float:
        if len(points) == 0:
            return 0.0
        distances = np.linalg.norm(points - center, axis=1)
        mean_radius = np.mean(distances)
        if mean_radius <= 1e-6:
            return 0.0
        return max(0.0, 1.0 - np.std(distances) / (2.0 * mean_radius))

    def detect(self, points: np.ndarray) -> Optional[AxisObservation]:
        if points is None or len(points) < 100:
            return None

        config = self.config
        workspace_points = points[
            (points[:, 0] >= config.workspace_min_x_m)
            & (points[:, 0] <= config.workspace_max_x_m)
            & (points[:, 1] >= config.workspace_min_y_m)
            & (points[:, 1] <= config.workspace_max_y_m)
        ]
        if len(workspace_points) < 100:
            return None

        normal = self._fit_plane(workspace_points)
        if normal[2] < 0.50:
            return None

        centroid = np.mean(workspace_points, axis=0)
        vertical_offsets = (workspace_points - centroid) @ normal
        floor_reference = centroid + normal * np.quantile(vertical_offsets, 0.15)
        axis_heights = (workspace_points - floor_reference) @ normal
        top_height = np.max(axis_heights)
        top_mask = np.abs(axis_heights - top_height) <= config.top_plane_tolerance_m
        top_points = workspace_points[top_mask]
        if len(top_points) < config.min_top_points:
            return None

        top_horizontal = top_points[:, :2] - floor_reference[:2]
        center_2d, radius = self._fit_circle(
            top_horizontal, config.top_radius_quantile
        )
        top_depths = (top_points - floor_reference) @ normal
        if not config.min_axis_height_m <= top_height <= config.max_axis_height_m:
            return None

        radii = np.linalg.norm(top_horizontal - center_2d, axis=1)
        inlier_mask = np.abs(radii - radius) <= 0.003
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < config.min_top_inlier_points:
            return None

        circularity = self._circularity(top_horizontal[inlier_mask], center_2d)
        if circularity < config.min_circularity:
            return None

        radius_min = config.cylinder_radius_m - config.cylinder_radius_tolerance_m
        radius_max = config.cylinder_radius_m + config.cylinder_radius_tolerance_m
        if not radius_min <= radius <= radius_max:
            return None

        center_3d = np.array([center_2d[0], center_2d[1], floor_reference[2]])
        center_3d = center_3d + normal * float(np.median(top_depths))
        return AxisObservation(
            center_x_m=float(center_3d[0]),
            center_y_m=float(center_3d[1]),
            top_z_m=float(center_3d[2]),
            radius_m=float(radius),
            circularity=float(circularity),
            inlier_count=inlier_count,
        )
