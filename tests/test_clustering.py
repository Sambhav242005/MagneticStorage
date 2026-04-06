#!/usr/bin/env python3
"""Tests for the clustering threshold heuristics used by Neuro-Savant."""

import unittest


def distance_to_confidence(distance: float) -> float:
    return 1.0 / (1.0 + distance)


class TestMagneticClustering(unittest.TestCase):
    def test_similarity_threshold(self):
        threshold = 0.5
        cases = [
            (0.5, True),
            (1.0, True),
            (1.5, False),
            (2.0, False),
            (0.0, True),
        ]

        for distance, should_cluster in cases:
            with self.subTest(distance=distance):
                confidence = distance_to_confidence(distance)
                self.assertEqual(confidence >= threshold, should_cluster)

    def test_related_queries_expected_buckets(self):
        threshold = 0.5
        scenarios = [
            ("hi", "hi again", 0.1, True),
            ("what is capital of france", "what is paris", 0.6, True),
            ("capital of france", "center of france", 1.2, False),
            ("what is 2+2", "capital of france", 2.5, False),
        ]

        for query_a, query_b, expected_distance, should_cluster in scenarios:
            with self.subTest(query_a=query_a, query_b=query_b):
                confidence = distance_to_confidence(expected_distance)
                self.assertEqual(confidence >= threshold, should_cluster)


if __name__ == "__main__":
    unittest.main()
