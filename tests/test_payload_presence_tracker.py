"""Tests for the single commit-gated payload debounce authority."""

from use_cases.payload_presence_tracker import PayloadPresenceTracker


def test_presence_changes_only_after_committed_consecutive_samples():
    tracker = PayloadPresenceTracker(
        confirmation_samples=2, present_max_range_m=0.08
    )

    first = tracker.assess(0.02)
    repeated_without_commit = tracker.assess(0.02)
    tracker.commit(first)
    second = tracker.assess(0.03)

    assert first.payload_sample is True
    assert first.payload_present is None
    assert repeated_without_commit.consecutive_samples == 1
    assert second.payload_present is True


def test_no_hit_debounces_to_empty():
    tracker = PayloadPresenceTracker(confirmation_samples=2)

    first = tracker.assess(-1.0)
    tracker.commit(first)
    second = tracker.assess(-1.0)
    tracker.commit(second)

    assert tracker.stable_presence is False


def test_invalid_or_distant_reading_is_unknown_and_does_not_advance():
    tracker = PayloadPresenceTracker(
        confirmation_samples=2, present_max_range_m=0.08
    )
    first = tracker.assess(0.02)
    tracker.commit(first)

    invalid = tracker.assess(None)
    distant = tracker.assess(0.4)
    tracker.commit(invalid)
    after_invalid = tracker.assess(0.02)

    assert invalid.payload_sample is None
    assert invalid.payload_present is None
    assert distant.payload_sample is None
    assert after_invalid.consecutive_samples == 2
