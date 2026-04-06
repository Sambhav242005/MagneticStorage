#!/usr/bin/env python3
"""Tests for terminal-safe performance and visualization output."""

import unittest

from core.performance_tracker import PerformanceTracker, VisualDisplay


class TestPerformanceTrackerDisplay(unittest.TestCase):
    def test_memory_visual_zero_state_reports_zero_total(self):
        output = VisualDisplay.memory_stats_visual(0, 0, 0)

        self.assertIn("Total Items: 0", output)
        self.assertIn("Cells", output)
        self.assertNotIn("Total Items: 1", output)

    def test_display_stats_is_ascii_friendly(self):
        tracker = PerformanceTracker()
        output = tracker.display_stats()

        self.assertIn("Performance Metrics", output)
        self.assertTrue(output.isascii())


if __name__ == "__main__":
    unittest.main()
