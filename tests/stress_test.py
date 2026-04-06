"""Workflow stress test for the story subsystem."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("ollama", MagicMock())

from tools.agent_behavior import AgentBehaviorTool
from tools.example import ExampleTool
from tools.infinite import InfiniteLoopTool
from tools.storyline_agent import StorylineAgent


class FakeBrain:
    def __init__(self):
        self.config = type("Config", (), {"model_name": "mock-model"})()
        self.ingest = MagicMock()
        self.tools = {
            "behavior": AgentBehaviorTool(),
            "example": ExampleTool(),
            "infinite": InfiniteLoopTool(),
        }

        self.behavior_tool = self.tools["behavior"]
        self.example_tool = self.tools["example"]
        self.infinite_tool = self.tools["infinite"]


class TestStorylineSystem(unittest.TestCase):
    def setUp(self):
        self.brain = FakeBrain()
        self.agent = StorylineAgent(self.brain)

    @patch("ollama.chat")
    def test_world_config_parser_accepts_python_style_payload(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": """```python
                {
                    'magic_system': {'source': 'Aether', 'cost': 'Heat', 'hard_restriction': 'No teleportation'},
                    'biomes': {'name': 'Glass Desert', 'visual_prompt': 'Mirror dunes', 'hazard_level': 8},
                    'civilizations': {'settlement_style': 'Nomadic', 'defense_strategy': 'Decoys'}
                }
                ```"""
            }
        }

        config = self.agent._create_world_config("Mirror Empire")

        self.assertEqual(config["magic_system"]["source"], "Aether")
        self.assertEqual(len(config["biomes"]), 1)
        self.assertEqual(config["civilization"]["settlement_style"], "Nomadic")
        self.assertNotIn("_warning", config)

    @patch("ollama.chat")
    def test_world_config_retries_placeholder_payloads(self, mock_chat):
        mock_chat.side_effect = [
            {
                "message": {
                    "content": """```json
                    {
                        "magic_system": {"source": "...", "cost": "...", "hard_restriction": "..."},
                        "biomes": [{"name": "...", "visual_prompt": "...", "hazard_level": 1}],
                        "civilization": {"settlement_style": "...", "defense_strategy": "..."}
                    }
                    ```"""
                }
            },
            {
                "message": {
                    "content": """{
                        "magic_system": {"source": "Steam rites", "cost": "Copper dust", "hard_restriction": "No time reversal"},
                        "biomes": [{"name": "Brass Marsh", "visual_prompt": "Foggy gears", "hazard_level": 6}],
                        "civilization": {"settlement_style": "Tiered foundries", "defense_strategy": "Automaton walls"}
                    }"""
                }
            },
        ]

        config = self.agent._create_world_config("Clockwork Republic")

        self.assertEqual(config["magic_system"]["source"], "Steam rites")
        self.assertEqual(config["civilization"]["defense_strategy"], "Automaton walls")
        self.assertEqual(mock_chat.call_count, 2)

    @patch("ollama.chat")
    def test_full_workflow_stress(self, mock_chat):
        self.brain.infinite_tool.execute("on")
        self.brain.infinite_tool.execute("set_chunks 5")
        self.brain.behavior_tool.execute("set critic")
        self.brain.example_tool.execute("load technical")

        generation_context_sizes = []

        def side_effect(model, messages):
            first_content = messages[0]["content"]

            if "JSON generator" in first_content:
                return {
                    "message": {
                        "content": """```json
                        {
                            "magic_system": { "source": "Void", "cost": "Sanity", "hard_restriction": "No resurrection" },
                            "biomes": [
                                {"name": "Crystal Wastes", "visual_prompt": "Shiny", "hazard_level": 9},
                                {"name": "Iron Forests", "visual_prompt": "Rusty", "hazard_level": 5},
                                {"name": "Neon Slums", "visual_prompt": "Cyberpunk", "hazard_level": 3}
                            ],
                            "civilization": { "settlement_style": "Vertical", "defense_strategy": "Shields" }
                        }
                        ```"""
                    }
                }

            if "Summarize this text" in first_content:
                return {"message": {"content": "SUMMARY: This section discusses complex political maneuvers."}}

            if "Analyze the following story/world description" in first_content:
                return {"message": {"content": "Main Characters, Landscapes, World Systems"}}

            generation_context_sizes.append(len(messages))
            return {"message": {"content": "GENERATED CONTENT. " + ("Lore " * 150)}}

        mock_chat.side_effect = side_effect

        facts = self.agent.execute_workflow("The Collapse of the Galactic Senate")

        self.assertIsInstance(facts, dict)
        self.assertEqual(self.brain.ingest.call_count, 18)
        self.assertEqual(len(generation_context_sizes), 15)
        self.assertTrue(all(size <= 5 for size in generation_context_sizes))
        self.assertEqual(mock_chat.call_count, 31)


if __name__ == "__main__":
    unittest.main()
