"""Reusable two-dimensional detector for the four advanced-task nuts.

The frame, inner-ROI, adaptive-threshold, black-hat, and watershed logic is a
Python port of the proven C++ implementation in linkerbot_ws.  The advanced
task removes the basic task's three-target cap and classifies the two equal
M45 nuts by measured brightness instead of size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    frame_size_m: float = 0.200
    black_v_max: int = 190
    debug_black_frame: bool = True
    min_frame_area_ratio: float = 0.01
    max_frame_area_ratio: float = 0.10
    frame_inner_scale: float = 1.0
    min_nut_radius_px: float = 5.0
    max_nut_radius_px: float = 40.0
    min_nut_area_px: float = 50.0
    max_nut_area_px: float = 100000.0
    adaptive_block_size: int = 31
    adaptive_c: float = 8.0
    blackhat_kernel_size: int = 31
    blackhat_threshold: float = 25.0
    nut_mask_open_kernel_size: int = 3
    nut_mask_close_kernel_size: int = 3
    min_nut_solidity: float = 0.45
    min_nut_circularity: float = 0.25
    min_nut_aspect_ratio: float = 0.30
    silver_gray_min: int = 120
    silver_black_margin: int = 15
    m45_min_size_m: float = 0.040
    m33_min_size_m: float = 0.030
    m27_min_size_m: float = 0.038
    m45_max_size_m: float = 0.055
    m33_max_size_m: float = 0.040
    m27_max_size_m: float = 0.030
    duplicate_center_factor: float = 0.55


@dataclass
class Candidate:
    contour: np.ndarray
    center: tuple[float, float]
    radius_px: float
    size_m: float
    mean_gray: float
    source: str
    reason: str = "rejected"
    accepted: bool = False


@dataclass
class Detection2D:
    frame: list[tuple[int, int]] = field(default_factory=list)
    frame_inner: list[tuple[int, int]] = field(default_factory=list)
    frame_found: bool = False
    frame_reused: bool = False
    circles: list[tuple[float, float, float]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    annotated: Optional[np.ndarray] = None
    black_frame_debug: Optional[np.ndarray] = None


class AdvancedNutDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.config: DetectorConfig
        self._frame: list[tuple[int, int]] = []
        self._frame_inner: list[tuple[int, int]] = []

    @staticmethod
    def _valid_config(config: DetectorConfig) -> None:
        if not 0.0 < config.frame_size_m <= 10.0:
            raise ValueError("frame_size_m must be in (0, 10] m")
        if not 0.0 < config.frame_inner_scale <= 1.0:
            raise ValueError("frame_inner_scale must be in (0, 1]")
        if not 0.0 < config.min_nut_radius_px < config.max_nut_radius_px:
            raise ValueError("min_nut_radius_px must be less than max_nut_radius_px")
        if not 0.0 < config.min_nut_area_px < config.max_nut_area_px:
            raise ValueError("min_nut_area_px must be less than max_nut_area_px")
        if config.adaptive_block_size < 3 or config.adaptive_block_size % 2 != 1:
            raise ValueError("adaptive_block_size must be an odd integer >= 3")
        if config.blackhat_kernel_size < 3 or config.blackhat_kernel_size % 2 != 1:
            raise ValueError("blackhat_kernel_size must be an odd integer >= 3")
        if not 0.0 < config.min_nut_solidity <= 1.0:
            raise ValueError("min_nut_solidity must be in (0, 1]")

    def _best_quadrilateral(self, mask: np.ndarray, gray: np.ndarray, min_area: float, max_area: float):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -np.inf
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not min_area <= area <= max_area:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) < 4 or len(approx) > 8 or not cv2.isContourConvex(approx):
                hull = cv2.convexHull(contour)
                if cv2.contourArea(hull) > area * 1.15:
                    continue
                approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
                if len(approx) < 4 or len(approx) > 8 or not cv2.isContourConvex(approx):
                    continue
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx) or cv2.contourArea(approx) < 0.75 * area:
                continue
            rect = cv2.minAreaRect(approx)
            if min(rect[1]) <= 1.0:
                continue
            rectangularity = area / (rect[1][0] * rect[1][1])
            aspect = max(rect[1]) / min(rect[1])
            if rectangularity < 0.55 or aspect > 4.0:
                continue
            interior = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(interior, approx, 255)
            cv2.erode(interior, interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            mean_gray = float(cv2.mean(gray, mask=interior)[0])
            if mean_gray < 90.0:
                continue
            score = rectangularity + (4.0 - aspect) / 3.0 + min(mean_gray / 255.0, 1.0)
            if score > best_score:
                best_score = score
                best = approx.reshape(4, 2)
        return best

    def _inner_quad(self, mask: np.ndarray, outer: np.ndarray):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None
        hierarchy = hierarchy[0]
        outer_area = float(cv2.contourArea(outer.astype(np.int32)))
        best = None
        best_area = outer_area * 1.05
        for index, contour in enumerate(contours):
            depth = 0
            parent = hierarchy[index][3]
            while parent >= 0:
                depth += 1
                parent = hierarchy[parent][3]
            if depth % 2 != 1:
                continue
            hull = cv2.convexHull(contour)
            quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            area = float(cv2.contourArea(quad))
            if area >= best_area or area < outer_area * 0.25 or area < cv2.contourArea(hull) * 0.75:
                continue
            if cv2.pointPolygonTest(outer.astype(np.float32), (int(quad[0][0][0]), int(quad[0][0][1])), True) < -1.5:
                continue
            best_area = area
            best = quad.reshape(4, 2)
        return best

    def _find_frame(self, bgr: np.ndarray, gray: np.ndarray):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, self.config.black_v_max))
        black_before_close = black_mask.copy()
        black_mask = cv2.morphologyEx(
            black_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        image_area = float(bgr.shape[0] * bgr.shape[1])
        area_range = (
            image_area * self.config.min_frame_area_ratio,
            image_area * self.config.max_frame_area_ratio,
        )
        frame = self._best_quadrilateral(black_mask, gray, *area_range)
        if frame is None:
            local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 10.0)
            local = cv2.morphologyEx(
                local, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            frame = self._best_quadrilateral(local, gray, *area_range)
        inner = None
        if frame is not None:
            inner = self._inner_quad(black_before_close, frame)
            if inner is None:
                inner = self._inner_quad(black_mask, frame)
            if inner is None:
                local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 15, 10.0)
                inner = self._inner_quad(local, frame)
        return frame, inner, black_mask

    def _plane_transform(self, frame: np.ndarray):
        square = np.array(
            [[0.0, 0.0], [self.config.frame_size_m, 0.0],
             [self.config.frame_size_m, self.config.frame_size_m], [0.0, self.config.frame_size_m]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(frame.astype(np.float32), square)

    def _split_oversized(self, contour: np.ndarray):
        area = cv2.contourArea(contour)
        if area <= self.config.min_nut_area_px * 2.0:
            return [contour]
        bbox = cv2.boundingRect(contour)
        max_x = min(bbox[0] + bbox[2], self._image_size[1])
        max_y = min(bbox[1] + bbox[3], self._image_size[0])
        bbox = (max(0, bbox[0]), max(0, bbox[1]),
                max(0, max_x - max(0, bbox[0])), max(0, max_y - max(0, bbox[1])))
        if bbox[2] <= 0 or bbox[3] <= 0:
            return [contour]
        local_mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
        shifted = contour - np.array([bbox[0], bbox[1]], dtype=contour.dtype)
        cv2.drawContours(local_mask, [shifted], 0, 255, cv2.FILLED)
        distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _, max_value, _, _ = cv2.minMaxLoc(distance)
        if max_value < 2.0:
            return [contour]
        markers = np.zeros(local_mask.shape, dtype=np.int32)
        seed = (distance > max_value * 0.45).astype(np.uint8)
        seed_contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(seed_contours) < 2:
            return [contour]
        for label, seed_contour in enumerate(seed_contours, start=1):
            cv2.drawContours(markers, [seed_contour], 0, label, cv2.FILLED)
        markers[local_mask == 0] = -1
        watershed_input = cv2.cvtColor(local_mask, cv2.COLOR_GRAY2BGR)
        cv2.watershed(watershed_input, markers)
        result = []
        for label in range(1, len(seed_contours) + 1):
            component = (markers == label).astype(np.uint8) * 255
            components, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for component_contour in components:
                shifted_back = component_contour + np.array([bbox[0], bbox[1]], dtype=component_contour.dtype)
                if cv2.contourArea(shifted_back) >= self.config.min_nut_area_px * 0.5:
                    result.append(shifted_back)
        return result

    def _mask_contours(self, gray: np.ndarray, roi: np.ndarray, edge_distance: np.ndarray):
        block_size = max(3, self.config.adaptive_block_size)
        if block_size % 2 == 0:
            block_size += 1
        masks = []
        for block in (block_size, max(3, block_size // 2 | 1)):
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block, self.config.adaptive_c,
            )
            adaptive &= roi
            kernel_open = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.nut_mask_open_kernel_size, self.config.nut_mask_open_kernel_size),
            )
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.nut_mask_close_kernel_size, self.config.nut_mask_close_kernel_size),
            )
            adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel_open)
            adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_close)
            masks.append(adaptive)
        blackhat_size = max(3, self.config.blackhat_kernel_size)
        if blackhat_size % 2 == 0:
            blackhat_size += 1
        blackhat = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_size, blackhat_size)),
        )
        blackhat = cv2.threshold(blackhat, self.config.blackhat_threshold, 255, cv2.THRESH_BINARY)[1]
        blackhat &= roi
        blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return [blackhat] + masks

    @staticmethod
    def _mean_gray(contour: np.ndarray, gray: np.ndarray) -> float:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)
        pixels = gray[mask > 0]
        return float(np.median(pixels)) if len(pixels) else 0.0

    def _classify(self, size_m: float, mean_gray: float) -> str:
        cfg = self.config
        if cfg.m45_min_size_m <= size_m <= cfg.m45_max_size_m:
            return "m45_silver" if mean_gray >= cfg.silver_gray_min else "m45_black"
        if cfg.m33_min_size_m <= size_m < cfg.m33_max_size_m:
            return "m33"
        if cfg.m27_min_size_m <= size_m < cfg.m27_max_size_m:
            return "m27"
        return "unknown"

    def detect(self, bgr: np.ndarray, reuse_frame: bool = True) -> Detection2D:
        self._valid_config(self.config)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("detector requires a non-empty BGR image")
        self._image_size = bgr.shape[:2]
        result = Detection2D()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        frame, inner, _ = self._find_frame(bgr, gray)
        if frame is None or inner is None and self._frame and self._frame_inner:
            pass
        if (frame is None or inner is None) and reuse_frame and self._frame and self._frame_inner:
            frame = np.array(self._frame, dtype=np.int32)
            inner = np.array(self._frame_inner, dtype=np.int32)
            result.frame_reused = True
        if frame is None or inner is None:
            result.annotated = bgr.copy()
            cv2.putText(result.annotated, "source frame not found", (16, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return result
        frame = np.asarray(frame, dtype=np.float32)
        inner = np.asarray(inner, dtype=np.float32)
        self._frame = [tuple(map(int, point)) for point in frame]
        self._frame_inner = [tuple(map(int, point)) for point in inner]
        result.frame = list(self._frame)
        result.frame_inner = list(self._frame_inner)
        result.frame_found = True

        plane_transform = self._plane_transform(frame)
        center = np.mean(inner, axis=0)
        scaled_inner = center + self.config.frame_inner_scale * (inner - center)
        roi = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(roi, np.round(scaled_inner).astype(np.int32), 255)
        cv2.erode(roi, roi, np.ones((1, 1), dtype=np.uint8))
        edge_distance = cv2.distanceTransform(roi, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        raw_gray = gray.copy()
        smooth_gray = cv2.GaussianBlur(raw_gray, (5, 5), 1.2)
        border = edge_distance <= 1.5
        smooth_gray[border] = raw_gray[border]
        candidate_by_center: dict[tuple[float, float], Candidate] = {}
        for mask_index, candidate_mask in enumerate(self._mask_contours(smooth_gray, roi, edge_distance)):
            source = "blackhat" if mask_index == 0 else f"adaptive_{mask_index}"
            contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            split_contours: list[np.ndarray] = []
            for contour in contours:
                split_contours.extend(self._split_oversized(contour))
            for contour in split_contours:
                area = float(cv2.contourArea(contour))
                if not self.config.min_nut_area_px <= area <= self.config.max_nut_area_px:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 1e-6:
                    continue
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                if circularity < self.config.min_nut_circularity:
                    continue
                hull = cv2.convexHull(contour)
                if cv2.contourArea(hull) <= 1e-6 or area / cv2.contourArea(hull) < self.config.min_nut_solidity:
                    continue
                rect = cv2.minAreaRect(contour)
                width, height = max(rect[1]), min(rect[1])
                if width <= 1.0 or height / width < self.config.min_nut_aspect_ratio:
                    continue
                clipped = sum(edge_distance[point[0][1], point[0][0]] <= 2.0 for point in contour)
                if clipped / len(contour) > 0.40:
                    continue
                moments = cv2.moments(contour)
                if abs(moments["m00"]) < 1e-6:
                    continue
                center_xy = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
                radius = width * 0.5
                if not self.config.min_nut_radius_px <= radius <= self.config.max_nut_radius_px:
                    continue
                center_pixel = (int(round(center_xy[0])), int(round(center_xy[1])))
                if not (0 <= center_pixel[0] < gray.shape[1] and 0 <= center_pixel[1] < gray.shape[0]):
                    continue
                if roi[center_pixel[1], center_pixel[0]] == 0:
                    continue
                rectified = cv2.perspectiveTransform(
                    contour.astype(np.float32).reshape(-1, 1, 2), plane_transform).reshape(-1, 2)
                area = float(abs(cv2.contourArea(rectified)))
                if area <= 1e-12:
                    continue
                size_m = float(np.sqrt(4.0 * area / np.pi))
                mean_gray = self._mean_gray(contour, raw_gray)
                candidate = Candidate(contour, center_xy, radius, size_m, mean_gray, source)
                key = (round(center_xy[0], 1), round(center_xy[1], 1))
                prior = candidate_by_center.get(key)
                if prior is None or candidate.radius_px > prior.radius_px:
                    candidate_by_center[key] = candidate

        candidates = sorted(candidate_by_center.values(), key=lambda item: item.radius_px, reverse=True)
        accepted: list[Candidate] = []
        for candidate in candidates:
            duplicate = False
            for other in accepted:
                distance = np.hypot(candidate.center[0] - other.center[0], candidate.center[1] - other.center[1])
                if distance < self.config.duplicate_center_factor * (candidate.radius_px + other.radius_px):
                    duplicate = True
                    break
            if duplicate:
                continue
            candidate.accepted = True
            candidate.reason = "accepted"
            accepted.append(candidate)
        result.candidates = candidates
        result.circles = [(item.center[0], item.center[1], item.radius_px) for item in accepted]

        result.annotated = bgr.copy()
        result.black_frame_debug = result.annotated.copy()
        cv2.polylines(result.annotated, [frame.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.polylines(result.annotated, [inner.astype(np.int32)], True, (0, 200, 200), 1)
        for item in accepted:
            point = tuple(map(int, item.center))
            cv2.drawMarker(result.annotated, point, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            label = self._classify(item.size_m, item.mean_gray)
            cv2.putText(result.annotated, f"{label} {item.size_m * 1000:.1f}mm gray={item.mean_gray:.0f}",
                        (point[0] + 6, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 0), 1)
        return result
