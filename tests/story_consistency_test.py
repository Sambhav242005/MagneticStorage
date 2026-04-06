"""Tests for story consistency tracking and infinite generation integration."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("ollama", MagicMock())

from tools.story_registry import ConsistencyType, StoryConsistencyRegistry


class TestStoryConsistencyRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = StoryConsistencyRegistry()

    def test_character_physical_conflict_detection(self):
        self.registry._merge_facts(
            {
                "characters": [{"name": "Elena", "physical": {"eyes": "blue", "hair": "auburn"}, "traits": ["brave"]}],
                "world_rules": {},
                "events": [],
                "locations": {},
            }
        )

        conflicts = self.registry.validate_characters(
            [{"name": "Elena", "physical": {"eyes": "green"}, "traits": []}],
            chunk_id=2,
        )

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].entity, "elena")
        self.assertEqual(conflicts[0].field, "physical.eyes")
        self.assertEqual(conflicts[0].old_value, "blue")
        self.assertEqual(conflicts[0].new_value, "green")

    def test_character_trait_contradiction(self):
        self.registry._merge_facts(
            {
                "characters": [{"name": "Kira", "physical": {}, "traits": ["brave", "kind"]}],
                "events": [],
                "world_rules": {},
                "locations": {},
            }
        )

        conflicts = self.registry.validate_characters(
            [{"name": "Kira", "physical": {}, "traits": ["cowardly"]}],
            chunk_id=3,
        )

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConsistencyType.CHARACTER)

    def test_world_rule_contradiction(self):
        self.registry._merge_facts(
            {
                "characters": [],
                "world_rules": {"magic source": "Magic comes from crystals that must be recharged"},
                "events": [],
                "locations": {},
            }
        )

        conflicts = self.registry.validate_world_rules(
            {"magic source": "Magic is innate and unlimited"},
            chunk_id=4,
        )

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConsistencyType.WORLD)

    def test_event_outcome_contradiction(self):
        self.registry._merge_facts(
            {
                "characters": [],
                "world_rules": {},
                "events": [{"id": "battle_1", "description": "The siege of Ironhold", "outcome": "Defenders won"}],
                "locations": {},
            }
        )

        conflicts = self.registry.validate_events(
            [{"id": "battle_1", "description": "The siege", "outcome": "Attackers conquered the city"}],
            chunk_id=5,
        )

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConsistencyType.PLOT)

    def test_no_false_positives_on_additions(self):
        self.registry._merge_facts(
            {
                "characters": [{"name": "Mara", "physical": {"eyes": "brown"}, "traits": []}],
                "events": [],
                "world_rules": {},
                "locations": {},
            }
        )

        conflicts = self.registry.validate_characters(
            [{"name": "Mara", "physical": {"hair": "black"}, "traits": ["wise"]}],
            chunk_id=2,
        )

        self.assertEqual(len(conflicts), 0)

    def test_case_insensitive_matching(self):
        self.registry._merge_facts(
            {
                "characters": [{"name": "ELENA", "physical": {"eyes": "blue"}, "traits": []}],
                "events": [],
                "world_rules": {},
                "locations": {},
            }
        )

        conflicts = self.registry.validate_characters(
            [{"name": "elena", "physical": {"eyes": "brown"}, "traits": []}],
            chunk_id=2,
        )

        self.assertEqual(len(conflicts), 1)


class TestStoryConsistencyStress(unittest.TestCase):
    def test_multi_chunk_consistency(self):
        chunks = [
            "Chapter 1. Elena stood at the gates of Ironhold. Elena had blue eyes. Elena was brave. The ancient magic flowed through the land.",
            "Chapter 2. Elena traveled to the mountains. Elena met Kira. Kira was wise. Kira had green eyes. The magic here felt stronger.",
            "Chapter 3. Elena gazed into the mirror. Elena's brown eyes stared back at her. Elena was confused. The sacred texts spoke of forbidden power.",
            "Chapter 4. Elena and Kira pressed forward. The power of the ancients guided them. Elena was determined to find the truth.",
            "Chapter 5. Elena drew her sword. Kira stood beside her. Elena was strong in combat. They arrived at the Crystal Tower.",
        ]

        registry = StoryConsistencyRegistry()
        for index, chunk in enumerate(chunks, start=1):
            registry.process_chunk(index, chunk, None)

        eye_conflicts = [conflict for conflict in registry.conflicts if "eyes" in conflict.field]

        self.assertEqual(len(registry.history), 5)
        self.assertGreaterEqual(len(eye_conflicts), 1)
        self.assertIn("elena", registry.facts.characters)
        self.assertIn("kira", registry.facts.characters)

    def test_perfect_consistency(self):
        chunks = [
            "Hero walked through the forest. Hero had blue eyes. Hero was brave.",
            "Hero continued the journey. Hero saw a mountain. Hero was brave and strong.",
            "Hero reached the castle. Hero prepared for battle. Hero was brave as always.",
        ]

        registry = StoryConsistencyRegistry()
        for index, chunk in enumerate(chunks, start=1):
            registry.process_chunk(index, chunk, None)

        eye_conflicts = [conflict for conflict in registry.conflicts if "eyes" in conflict.field]
        self.assertEqual(len(eye_conflicts), 0)


class TestIntegrationWithInfiniteLoop(unittest.TestCase):
    @patch("ollama.chat")
    def test_infinite_loop_with_registry(self, mock_chat):
        from tools.infinite import InfiniteLoopTool

        registry = StoryConsistencyRegistry()
        infinite_tool = InfiniteLoopTool()
        infinite_tool.active = True
        infinite_tool.chunk_limit = 3

        generation_chunks = iter(
            [
                "Elena stood tall. Elena had blue eyes. Elena was brave.",
                "Elena drew her sword. Elena was brave and calm.",
            ]
        )

        def side_effect(model, messages):
            prompt = messages[-1]["content"]
            if "Analyze the following story/world description" in prompt:
                return {"message": {"content": "COMPLETE"}}
            return {"message": {"content": next(generation_chunks)}}

        mock_chat.side_effect = side_effect

        full_text, chunks = infinite_tool.generate_sequence(
            "test-model",
            "You are a storyteller",
            "Write a fantasy story",
            consistency_tracker=registry,
        )

        self.assertEqual(len(chunks), 2)
        self.assertIn("Elena", full_text)
        self.assertEqual(len(registry.history), 2)
        self.assertIn("elena", registry.facts.characters)


if __name__ == "__main__":
    unittest.main()
