import pytest

from advanced_assembly.nut_sequence import AdvancedNutSequence, Observation, SequenceConfig


def observation(x, y, size, gray):
    return Observation((x, y), max(size * 20, 8), size, gray)


def test_classifies_equal_size_m45_by_brightness():
    observations = (
        observation(100, 100, 0.048, 180),
        observation(200, 120, 0.047, 40),
        observation(300, 140, 0.034, 40),
        observation(400, 160, 0.028, 40),
    )
    classified = AdvancedNutSequence.classify(observations)
    assert classified is not None
    assert classified[1].mean_gray == 180
    assert classified[2].mean_gray == 40
    assert classified[3].size_m == pytest.approx(0.034)
    assert classified[4].size_m == pytest.approx(0.028)


def test_requires_four_stable_matching_frames():
    sequence = AdvancedNutSequence(SequenceConfig(stable_frames=3, confirmation_window_s=3.0))
    layout = (
        observation(100, 100, 0.048, 180),
        observation(200, 120, 0.047, 40),
        observation(300, 140, 0.034, 40),
        observation(400, 160, 0.028, 40),
    )
    first = sequence.observe(1_000_000_000, True, layout)
    assert not first.observation_valid
    sequence.observe(1_100_000_000, True, layout)
    stable = sequence.observe(1_200_000_000, True, layout)
    assert stable.observation_valid
    assert stable.expected_count == 4
    assert stable.targets[0].label == "m45_silver"


def test_after_completion_expects_three_remaining_nuts():
    sequence = AdvancedNutSequence(SequenceConfig(stable_frames=1, confirmation_window_s=0.0))
    layout4 = (
        observation(100, 100, 0.048, 180),
        observation(200, 120, 0.047, 40),
        observation(300, 140, 0.034, 40),
        observation(400, 160, 0.028, 40),
    )
    sequence.observe(1_000_000_000, True, layout4)
    accepted, reason = sequence.event(1, 1, 1, "start")
    assert accepted, reason
    accepted, reason = sequence.event(1, 2, 1, "complete")
    assert accepted, reason
    layout3 = layout4[1:]
    snapshot = sequence.observe(2_000_000_000, True, layout3)
    assert snapshot.observation_valid
    assert snapshot.expected_count == 3
    assert snapshot.current_target_id == 2
    assert [target.label for target in snapshot.targets if target.visible] == ["m45_black", "m33", "m27"]
