"""Stage-aware stable observation and explicit task-state feedback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ORDER = ("m45_silver", "m45_black", "m33", "m27")


@dataclass(frozen=True)
class SequenceConfig:
    stable_frames: int = 3
    confirmation_window_s: float = 3.0
    max_center_shift_px: float = 40.0
    max_radius_change_ratio: float = 0.20
    max_gap_s: float = 1.0
    max_observation_age_s: float = 0.5
    max_depth_age_s: float = 0.1


@dataclass
class NutTarget:
    id: int
    label: str
    state: str = "pending"
    visible: bool = False
    pixel: Optional[tuple[float, float]] = None
    radius_px: float = 0.0
    size_m: float = 0.0
    mean_gray: float = 0.0
    position: Optional[tuple[float, float, float]] = None
    frame_id: str = ""
    last_seen_ns: int = 0


@dataclass
class Snapshot:
    round_id: int = 1
    initialized: bool = False
    observation_valid: bool = False
    observed_count: int = 0
    expected_count: int = 4
    current_target_id: int = 1
    status: str = "waiting_for_four"
    targets: list[NutTarget] = field(default_factory=lambda: [
        NutTarget(index + 1, label) for index, label in enumerate(ORDER)
    ])


@dataclass(frozen=True)
class Observation:
    pixel: tuple[float, float]
    radius_px: float
    size_m: float
    mean_gray: float


class AdvancedNutSequence:
    """Stabilise the current stage, advancing only on explicit task events."""

    def __init__(self, config: Optional[SequenceConfig] = None):
        self.config = config or SequenceConfig()
        if self.config.stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self.snapshot = Snapshot()
        self._confirmations: deque[tuple[int, tuple[Observation, ...]]] = deque()
        self._last_stamp_ns = -1
        self._last_event_sequence = 0
        self._last_event = (0, "", "")

    @property
    def completed_count(self) -> int:
        return sum(target.state == "completed" for target in self.snapshot.targets)

    @staticmethod
    def _same_layout(first: tuple[Observation, ...], second: tuple[Observation, ...], tolerance: float) -> bool:
        if len(first) != len(second):
            return False
        used = [False] * len(second)
        for a in first:
            for index, b in enumerate(second):
                if used[index]:
                    continue
                if np.hypot(a.pixel[0] - b.pixel[0], a.pixel[1] - b.pixel[1]) <= tolerance:
                    used[index] = True
                    break
            else:
                return False
        return True

    @classmethod
    def classify(cls, observations: tuple[Observation, ...], start_id: int = 1):
        """Return target observations by ID, or None if identity is ambiguous."""
        candidates: list[tuple[Observation, int]] = []
        for observation in observations:
            if observation.size_m >= 0.045:
                candidates.append((observation, 1 if observation.mean_gray >= 120 else 2))
            elif observation.size_m >= 0.0325:
                candidates.append((observation, 3))
            elif observation.size_m >= 0.025:
                candidates.append((observation, 4))
            else:
                return None
        expected = list(range(start_id, start_id + len(observations)))
        ids = sorted(item[1] for item in candidates)
        if ids != expected:
            return None
        return {candidate[1]: candidate[0] for candidate in candidates}

    def observe(self, stamp_ns: int, frame_found: bool, observations: tuple[Observation, ...]) -> Snapshot:
        self.snapshot.observed_count = len(observations)
        self.snapshot.observation_valid = False
        for target in self.snapshot.targets:
            target.visible = False
        if stamp_ns <= self._last_stamp_ns:
            self.snapshot.status = "non_increasing_timestamp"
            return self.snapshot
        if frame_found and observations and self._last_stamp_ns >= 0:
            gap_s = (stamp_ns - self._last_stamp_ns) / 1e9
            if gap_s > self.config.max_gap_s and self.config.confirmation_window_s == 0.0:
                self._confirmations.clear()
        self._last_stamp_ns = stamp_ns
        self.snapshot.expected_count = 4 - self.completed_count
        self.snapshot.current_target_id = 0 if self.snapshot.expected_count == 0 else self.completed_count + 1
        if not frame_found:
            self.snapshot.status = "frame_not_found"
            self._confirmations.clear()
            return self.snapshot
        if len(observations) != self.snapshot.expected_count:
            self.snapshot.status = "observation_count_mismatch"
            self._confirmations.clear()
            return self.snapshot
        classified = self.classify(observations, self.completed_count + 1)
        if classified is None:
            self.snapshot.status = "ambiguous_nut_identity"
            self._confirmations.clear()
            return self.snapshot
        layout = tuple(classified[index] for index in sorted(classified))
        window_ns = int(self.config.confirmation_window_s * 1e9)
        if window_ns > 0:
            while self._confirmations and stamp_ns - self._confirmations[0][0] > window_ns:
                self._confirmations.popleft()
            self._confirmations.append((stamp_ns, layout))
            while len(self._confirmations) > self.config.stable_frames:
                self._confirmations.popleft()
            if not self._same_layout(self._confirmations[0][1], layout,
                                     self.config.max_center_shift_px):
                self.snapshot.status = "confirming_window"
                return self.snapshot
        else:
            if (not self._confirmations or not self._same_layout(self._confirmations[-1][1], layout,
                                                                self.config.max_center_shift_px)):
                self._confirmations.clear()
            self._confirmations.append((stamp_ns, layout))
            while len(self._confirmations) > self.config.stable_frames:
                self._confirmations.popleft()
        if len(self._confirmations) < self.config.stable_frames:
            self.snapshot.status = "confirming_window"
            return self.snapshot
        first = self.completed_count
        first_id = self.completed_count + 1
        for offset, target_id in enumerate(range(first_id, first_id + len(observations))):
            target = self.snapshot.targets[first + offset]
            observation = classified[target_id]
            target.visible = True
            target.pixel = observation.pixel
            target.radius_px = observation.radius_px
            target.size_m = observation.size_m
            target.mean_gray = observation.mean_gray
            target.last_seen_ns = stamp_ns
        self.snapshot.initialized = True
        self.snapshot.observation_valid = True
        self.snapshot.status = "target_in_progress" if self.snapshot.targets[first].state == "in_progress" else "ready"
        return self.snapshot

    def event(self, round_id: int, event_sequence: int, target_id: int, action: str):
        if event_sequence == self._last_event_sequence and (round_id, target_id, action) == self._last_event:
            return True, "already_applied"
        if round_id != self.snapshot.round_id:
            return False, "wrong_round"
        if event_sequence <= self._last_event_sequence:
            return False, "stale_event_sequence"
        if action == "reset":
            if target_id != 0:
                return False, "reset_requires_target_zero"
            self.snapshot = Snapshot(round_id=self.snapshot.round_id + 1)
            self._confirmations.clear()
            self._last_stamp_ns = -1
        else:
            if not self.snapshot.initialized or not 1 <= target_id <= 4:
                return False, "unknown_target"
            if target_id != self.completed_count + 1:
                return False, "order_violation"
            target = self.snapshot.targets[target_id - 1]
            if action == "start":
                if not self.snapshot.observation_valid or not target.visible:
                    return False, "target_not_currently_visible"
                if target.state != "pending":
                    return False, "start_requires_pending"
                target.state = "in_progress"
            elif action == "complete":
                if target.state != "in_progress":
                    return False, "complete_requires_in_progress"
                target.state = "completed"
                self.snapshot.observation_valid = False
                self.snapshot.status = "round_completed" if self.completed_count == 4 else "awaiting_next_observation"
            elif action == "retry":
                if target.state != "in_progress":
                    return False, "retry_requires_in_progress"
                target.state = "pending"
                self.snapshot.observation_valid = False
                self.snapshot.status = "awaiting_next_observation"
            else:
                return False, "unknown_action"
        self._last_event_sequence = event_sequence
        self._last_event = (round_id, target_id, action)
        return True, "applied"
