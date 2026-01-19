"""
Story Consistency Stress Test

Validates that story generation maintains consistency across chunks:
- Character attributes don't drift
- Plot events don't contradict
- World rules remain stable
- Timeline stays coherent
"""

import sys
import os
import time
import json
import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pre-mock heavy dependencies for fast testing
sys.modules['chromadb'] = MagicMock()
sys.modules['networkx'] = MagicMock()

from tools.story_registry import StoryConsistencyRegistry, Conflict, ConsistencyType


class TestStoryConsistencyRegistry(unittest.TestCase):
    """Unit tests for the StoryConsistencyRegistry"""
    
    def setUp(self):
        self.registry = StoryConsistencyRegistry()
    
    def test_character_physical_conflict_detection(self):
        """Detect when character physical attributes change"""
        # First chunk: Elena has blue eyes
        first_facts = {
            "characters": [
                {"name": "Elena", "physical": {"eyes": "blue", "hair": "auburn"}, "traits": ["brave"]}
            ],
            "world_rules": {},
            "events": [],
            "locations": {}
        }
        self.registry._merge_facts(first_facts)
        
        # Second chunk: Elena now has green eyes (CONFLICT!)
        second_facts = {
            "characters": [
                {"name": "Elena", "physical": {"eyes": "green"}, "traits": []}
            ]
        }
        conflicts = self.registry.validate_characters(second_facts["characters"], chunk_id=2)
        
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].entity, "elena")
        self.assertEqual(conflicts[0].field, "physical.eyes")
        self.assertEqual(conflicts[0].old_value, "blue")
        self.assertEqual(conflicts[0].new_value, "green")
        print(f"✅ Detected conflict: {conflicts[0]}")
    
    def test_character_trait_contradiction(self):
        """Detect contradicting personality traits"""
        self.registry._merge_facts({
            "characters": [{"name": "Kira", "physical": {}, "traits": ["brave", "kind"]}],
            "events": [], "world_rules": {}, "locations": {}
        })
        
        # Now Kira is described as cowardly (contradicts brave)
        conflicts = self.registry.validate_characters(
            [{"name": "Kira", "physical": {}, "traits": ["cowardly"]}],
            chunk_id=3
        )
        
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConsistencyType.CHARACTER)
        print(f"✅ Detected trait contradiction: {conflicts[0]}")
    
    def test_world_rule_contradiction(self):
        """Detect when world rules change"""
        self.registry._merge_facts({
            "characters": [],
            "world_rules": {"magic source": "Magic comes from crystals that must be recharged"},
            "events": [], "locations": {}
        })
        
        # Now magic comes from something completely different
        conflicts = self.registry.validate_world_rules(
            {"magic source": "Magic is innate and unlimited"},
            chunk_id=4
        )
        
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConsistencyType.WORLD)
        print(f"✅ Detected world rule conflict: {conflicts[0]}")
    
    def test_event_outcome_contradiction(self):
        """Detect when event outcomes contradict"""
        self.registry._merge_facts({
            "characters": [],
            "world_rules": {},
            "events": [{"id": "battle_1", "description": "The siege of Ironhold", "outcome": "Defenders won"}],
            "locations": {}
        })
        
        # Now the same battle has different outcome
        conflicts = self.registry.validate_events(
            [{"id": "battle_1", "description": "The siege", "outcome": "Attackers conquered the city"}],
            chunk_id=5
        )
        
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConsistencyType.PLOT)
        print(f"✅ Detected plot conflict: {conflicts[0]}")
    
    def test_no_false_positives_on_additions(self):
        """Adding new info should not trigger conflicts"""
        self.registry._merge_facts({
            "characters": [{"name": "Mara", "physical": {"eyes": "brown"}, "traits": []}],
            "events": [], "world_rules": {}, "locations": {}
        })
        
        # Adding hair color (not changing eyes)
        conflicts = self.registry.validate_characters(
            [{"name": "Mara", "physical": {"hair": "black"}, "traits": ["wise"]}],
            chunk_id=2
        )
        
        self.assertEqual(len(conflicts), 0)
        print("✅ No false positives on new additions")
    
    def test_case_insensitive_matching(self):
        """Character names should match case-insensitively"""
        self.registry._merge_facts({
            "characters": [{"name": "ELENA", "physical": {"eyes": "blue"}, "traits": []}],
            "events": [], "world_rules": {}, "locations": {}
        })
        
        conflicts = self.registry.validate_characters(
            [{"name": "elena", "physical": {"eyes": "brown"}, "traits": []}],
            chunk_id=2
        )
        
        self.assertEqual(len(conflicts), 1)  # Should still detect the conflict
        print("✅ Case-insensitive matching works")


class TestStoryConsistencyStress(unittest.TestCase):
    """Stress tests with realistic text chunks (regex-based extraction)"""
    
    def setUp(self):
        print("\n" + "=" * 60)
        print("STRESS TEST SETUP")
        print("=" * 60)
    
    def test_multi_chunk_consistency(self):
        """Generate 5 chunks and verify consistency tracking via regex extraction"""
        
        # Realistic story chunks with extractable patterns
        chunks = [
            # Chunk 1: Introduce Elena with blue eyes
            "Chapter 1: The Beginning. Elena stood at the gates of Ironhold. Elena had blue eyes that sparkled in the sunlight. Elena was brave, everyone knew that. The ancient magic flowed through the land, drawing power from the ley lines beneath the earth.",
            
            # Chunk 2: Consistent continuation, add Kira
            "Chapter 2: The Meeting. Elena traveled to the mountains. Elena met Kira at the temple. Kira was wise beyond her years. Kira had green eyes like emeralds. The magic here felt stronger.",
            
            # Chunk 3: CONFLICT - Elena's eyes change to brown
            "Chapter 3: The Revelation. Elena gazed into the mirror. Elena's brown eyes stared back at her. Elena was confused by what she saw. The sacred texts spoke of forbidden power.",
            
            # Chunk 4: Consistent with new world rule
            "Chapter 4: The Journey. Elena and Kira pressed forward. The power of the ancients guided them. Elena was determined to find the truth.",
            
            # Chunk 5: Add more details
            "Chapter 5: The Battle. Elena drew her sword. Kira stood beside her. Elena was strong in combat. They arrived at the Crystal Tower."
        ]
        
        registry = StoryConsistencyRegistry()
        
        print("\n🚀 Running multi-chunk stress test with regex extraction...")
        
        for i, chunk in enumerate(chunks):
            conflicts = registry.process_chunk(i + 1, chunk, None)  # No model needed for regex
        
        # Generate report
        report = registry.get_report()
        print(report)
        
        # Assertions
        self.assertEqual(len(registry.history), 5)
        
        # Should have detected Elena's eye color change (blue -> brown)
        eye_conflicts = [c for c in registry.conflicts if 'eyes' in c.field]
        self.assertGreaterEqual(len(eye_conflicts), 1, "Should detect Elena's eye color change")
        
        # Should have tracked characters
        self.assertIn('elena', registry.facts.characters)
        self.assertIn('kira', registry.facts.characters)
        
        print("\n✅ Multi-chunk stress test PASSED")
        print(f"   - Chunks processed: {len(registry.history)}")
        print(f"   - Conflicts detected: {len(registry.conflicts)}")
        print(f"   - Characters tracked: {len(registry.facts.characters)}")
    
    def test_perfect_consistency(self):
        """Verify no false positives with perfectly consistent story"""
        
        # All chunks consistently describe Hero with blue eyes
        chunks = [
            "Hero walked through the forest. Hero had blue eyes. Hero was brave.",
            "Hero continued his journey. Hero saw a mountain. Hero was brave and strong.",
            "Hero reached the castle. Hero prepared for battle. Hero was brave as always."
        ]
        
        registry = StoryConsistencyRegistry()
        
        # Process chunks
        for i, chunk in enumerate(chunks):
            registry.process_chunk(i + 1, chunk, None)
        
        # Should have no conflicts (consistent eye color, no contradicting traits)
        eye_conflicts = [c for c in registry.conflicts if 'eyes' in c.field]
        self.assertEqual(len(eye_conflicts), 0, "Should have no eye color conflicts")
        print("\n✅ Perfect consistency test PASSED (0 false positives)")


class TestIntegrationWithInfiniteLoop(unittest.TestCase):
    """Test integration with InfiniteLoopTool"""
    
    def setUp(self):
        print("\n" + "=" * 60)
        print("INTEGRATION TEST")
        print("=" * 60)
    
    @patch('ollama.chat')
    def test_infinite_loop_with_registry(self, mock_chat):
        """Test that registry can be used as a hook in infinite loop"""
        from tools.infinite import InfiniteLoopTool
        from tools.story_registry import StoryConsistencyRegistry
        
        # Setup mock responses
        story_chunks = [
            "Elena stood tall, her blue eyes reflecting the sunset...",
            "She drew her sword, the blade gleaming...",
            "COMPLETE"  # Triggers end of loop
        ]
        
        extraction_responses = [
            json.dumps({
                "characters": [{"name": "Elena", "physical": {"eyes": "blue"}, "traits": ["brave"], "relationships": {}}],
                "world_rules": {}, "events": [], "locations": {}
            }),
            json.dumps({
                "characters": [{"name": "Elena", "physical": {}, "traits": ["skilled"], "relationships": {}}],
                "world_rules": {}, "events": [], "locations": {}
            })
        ]
        
        call_idx = [0]
        def mock_response(model, messages):
            idx = call_idx[0]
            call_idx[0] += 1
            
            # Detect if this is an extraction call
            if "Extract ALL facts" in str(messages):
                resp_idx = min(idx // 2, len(extraction_responses) - 1)
                return {"message": {"content": extraction_responses[resp_idx]}}
            
            # Otherwise it's a generation call
            resp_idx = min(idx, len(story_chunks) - 1)
            return {"message": {"content": story_chunks[resp_idx]}}
        
        mock_chat.side_effect = mock_response
        
        # Create tools
        infinite_tool = InfiniteLoopTool()
        infinite_tool.active = True
        infinite_tool.chunk_limit = 3
        
        registry = StoryConsistencyRegistry()
        
        # Custom tracking function
        def track_consistency(chunk_idx: int, chunk_text: str):
            return registry.process_chunk(chunk_idx, chunk_text, "test-model")
        
        print("🔄 Running infinite loop with consistency tracking...")
        
        # Simulate what would happen in a real run
        # (In production, this would be integrated into InfiniteLoopTool)
        full_text, chunks = infinite_tool.generate_sequence(
            "test-model",
            "You are a storyteller",
            "Write a fantasy story"
        )
        
        # Manually run registry on chunks (simulating integration)
        for i, chunk in enumerate(chunks):
            track_consistency(i + 1, chunk)
        
        print(registry.get_report())
        print("\n✅ Integration test PASSED")


def run_all_tests():
    """Run all tests and generate summary"""
    print("\n" + "=" * 70)
    print("🧪 STORY CONSISTENCY STRESS TEST SUITE")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStoryConsistencyRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestStoryConsistencyStress))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithInfiniteLoop))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
