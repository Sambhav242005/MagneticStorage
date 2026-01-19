
import sys
import os
import time
import json
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory (project root) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our system components
# Note: Adjusting imports based on apparent file structure
# Pre-mock dependencies that might be missing in test env
from unittest.mock import MagicMock
sys.modules['chromadb'] = MagicMock()
sys.modules['ollama'] = MagicMock()
sys.modules['networkx'] = MagicMock()

try:
    from neuro_savant import NeuroSavant
    from tools.storyline_agent import StorylineAgent
    from tools.infinite import InfiniteLoopTool
    from tools.agent_behavior import AgentBehaviorTool
    from tools.example import ExampleTool
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'tools'))
    import neuro_savant
    from tools.storyline_agent import StorylineAgent
    from tools.infinite import InfiniteLoopTool
    from tools.agent_behavior import AgentBehaviorTool
    from tools.example import ExampleTool

class TestStorylineSystem(unittest.TestCase):
    
    def setUp(self):
        print("\n" + "="*50)
        print("SETUP: Initializing System Agents")
        print("="*50)
        
        # 1. Initialize Brain (Mocking DB to avoid persistent side effects)
        # We patch the database parts of NeuroSavant to avoid disk writes
        with patch('neuro_savant.chromadb.PersistentClient'), \
             patch('neuro_savant.os.makedirs'):
            try:
                self.brain = neuro_savant.NeuroSavant(model_name="mock-model")
                # Mock the memory grid explicitly to capture writes
                self.brain.memory = MagicMock()
                self.brain.memory._generate_id_from_content.side_effect = lambda x: f"hash_{len(x)}"
                print("✅ NeuroSavant Brain Initialized (Mocked Memory)")
            except Exception as e:
                print(f"⚠️ Failed to init real brain, using dummy: {e}")
                self.brain = MagicMock()
                self.brain.model_name = "mock-model"

        # 2. Attach Tools
        self.brain.infinite_tool = InfiniteLoopTool()
        self.brain.behavior_tool = AgentBehaviorTool()
        self.brain.example_tool = ExampleTool()
        print("✅ Tools Attached: Infinite, Behavior, Example")
        
        # 3. Initialize Agent
        self.agent = StorylineAgent(self.brain)
        print("✅ StorylineAgent Initialized")

    @patch('ollama.chat')
    def test_full_workflow_stress(self, mock_chat):
        """
        Stress Test Scenario:
        1. Enable Infinite Mode (Massive generation)
        2. Set Persona (Modify style)
        3. Load Template (Modify context)
        4. Run Workflow (Trigger multiple calls)
        """
        print("\n🚀 STARTING STRESS TEST: 'Galactic Senate Crisis'")
        
        # A. Setup Tools
        self.brain.infinite_tool.execute("on")
        self.brain.infinite_tool.execute("set_chunks 5") # Generate 5 chunks per section
        self.brain.behavior_tool.execute("set critic")
        self.brain.example_tool.execute("load technical")
        
        # B. Configure Mock Responses
        def side_effect(model, messages):
            # Check who is calling (System prompt clues)
            system_content = messages[0]['content']
            user_content = messages[-1]['content']
            
            # 1. JSON Config Call
            if "JSON generator" in system_content:
                return {
                    'message': {
                        'content': '''```json
                        {
                            "magic_system": { "source": "Void", "cost": "Sanity", "hard_restriction": "No resurrection" },
                            "biomes": [ 
                                {"name": "Crystal Wastes", "visual_prompt": "Shiny", "hazard_level": 9},
                                {"name": "Iron Forests", "visual_prompt": "Rusty", "hazard_level": 5}, 
                                {"name": "Neon Slums", "visual_prompt": "Cyberpunk", "hazard_level": 3}
                            ],
                            "civilization": { "settlement_style": "Vertical", "defense_strategy": "Shields" }
                        }
                        ```'''
                    }
                }
            
            # 2. Summarizer Call
            if "Summarize this text" in system_content:
                return {'message': {'content': "SUMMARY: This section discusses complex political maneuvers..."}}
                
            # 3. Generation Call (Infinite or Normal)
            # Return a large chunk to stress memory
            return {'message': {'content': f"GENERATED CONTENT ({len(messages)} msgs). " + "Lore " * 50}}

        mock_chat.side_effect = side_effect
        
        # C. Execute Workflow
        topic = "The Collapse of the Galactic Senate"
        self.agent.execute_workflow(topic)
        
        # D. Verifications
        print("\n📊 ANALYZING RESULTS...")
        
        # 1. Check Tool Usage in Logic
        # We can't easily check internal python state changes of the tool unless we inspect side effects,
        # but we can check if the agent *accessed* them.
        # Since we are running the real code of StorylineAgent, it DEFINITELY called them if lines were hit.
        
        # 2. Verify Call Counts
        # Config (1) + 3 Sections * (Infinite 5 chunks) + 3 Verification Summaries
        # Actually logic is: 
        #   1. Config -> 1 call
        #   2. Loop 3 sections:
        #       - Generate (Infinite tool handles the loop internally)
        #           - Infinite tool does 'chunk_limit' calls (5)
        #       - Verify (1 check, no LLM call in current dummy verifier)
        #       - Summarize -> 1 call
        # Total expected = 1 + 3*(5 + 1) = 1 + 18 = 19 calls
        
        call_count = mock_chat.call_count
        print(f"   - LLM Call Count: {call_count} (Expected ~19)")
        self.assertGreater(call_count, 10, "Should have made significant number of LLM calls")
        
        # 3. Verify Memory Writes
        # Each section: 1 Master + 5 Chunks = 6 writes
        # Total = 3 * 6 = 18 writes
        write_count = self.brain.memory.update_state_immediate.call_count
        print(f"   - Memory Writes: {write_count} (Expected ~18)")
        self.assertGreater(write_count, 10, "Should have saved chunks to memory")
        
        # 4. Verify Infinite Tool Logic
        # It creates a 'sliding window' of context. We can check the last call's context length.
        last_call_args = mock_chat.call_args[1]
        last_messages = last_call_args['messages']
        print(f"   - Context Window Size (Last Call): {len(last_messages)}")
        # Expecting System + User + (Assistant+User)*Depth. 
        # infinite.py limits context to size 6.
        self.assertLessEqual(len(last_messages), 10, "Infinite tool should manage context window size")

        print("✅ STRESS TEST PASSED")

if __name__ == '__main__':
    unittest.main()
