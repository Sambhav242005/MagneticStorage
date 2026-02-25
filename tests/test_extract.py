import os
import sys
from neuro_savant import NeuroSavant, Config
from tools.storyline_agent import StorylineAgent

os.environ["USE_MOCK_ENCODER"] = "true"
config = Config(db_path="./test_story_db", model_name="qwen2.5:3b")
brain = NeuroSavant(config=config)
agent = StorylineAgent(brain)

if 'infinite' in brain.tools:
    del brain.tools['infinite']

content_full, chunks, extracted_facts = agent._generate_section("The Magic System", "A cyberpunk rebellion.")
print("RAW FACTS:", repr(extracted_facts))
