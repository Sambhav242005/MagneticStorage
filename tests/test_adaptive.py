#!/usr/bin/env python3
"""Tests for adaptive threshold behavior."""

import threading
import time
import unittest
from collections import deque


class AdaptiveThreshold:
    MIN_THRESHOLD = 0.3
    MAX_THRESHOLD = 0.7
    INITIAL_THRESHOLD = 0.4
    PERCENTILE = 40

    def __init__(self, window_size=50):
        self.confidence_history = deque(maxlen=window_size)
        self.current_threshold = self.INITIAL_THRESHOLD
        self.lock = threading.Lock()

    def record_match(self, confidence: float, was_accepted: bool):
        with self.lock:
            self.confidence_history.append(
                {
                    "confidence": confidence,
                    "accepted": was_accepted,
                    "timestamp": time.time(),
                }
            )
            if len(self.confidence_history) >= 10 and len(self.confidence_history) % 10 == 0:
                self._update_threshold()

    def _update_threshold(self):
        if len(self.confidence_history) < 10:
            return

        confidences = sorted(item["confidence"] for item in self.confidence_history)
        idx = int(len(confidences) * (self.PERCENTILE / 100))
        new_threshold = confidences[idx]
        new_threshold = max(self.MIN_THRESHOLD, min(self.MAX_THRESHOLD, new_threshold))
        self.current_threshold = 0.7 * self.current_threshold + 0.3 * new_threshold

    def get_threshold(self) -> float:
        return self.current_threshold


class TestAdaptiveThreshold(unittest.TestCase):
    def test_initial_threshold(self):
        threshold = AdaptiveThreshold()
        self.assertEqual(threshold.get_threshold(), 0.4)

    def test_high_confidence_inputs_raise_threshold(self):
        threshold = AdaptiveThreshold()

        for _ in range(20):
            threshold.record_match(confidence=0.8, was_accepted=True)

        self.assertGreater(threshold.get_threshold(), 0.4)

    def test_low_confidence_inputs_lower_threshold(self):
        threshold = AdaptiveThreshold(window_size=50)
        threshold.current_threshold = 0.553

        for _ in range(50):
            threshold.record_match(confidence=0.2, was_accepted=False)

        self.assertLess(threshold.get_threshold(), 0.4)

    def test_threshold_respects_safety_bounds(self):
        threshold = AdaptiveThreshold(window_size=50)

        for _ in range(50):
            threshold.record_match(1.0, True)
        self.assertLessEqual(threshold.get_threshold(), 0.701)

        for _ in range(50):
            threshold.record_match(0.0, False)
        self.assertGreaterEqual(threshold.get_threshold(), 0.299)


if __name__ == "__main__":
    unittest.main()
