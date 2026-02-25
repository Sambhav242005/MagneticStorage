import os
import shutil
from neuro_savant import NeuroSavant, Config
from tools.storyline_agent import StorylineAgent

if os.path.exists("./test_story_db"):
    shutil.rmtree("./test_story_db")

os.environ["USE_MOCK_ENCODER"] = "true"

config = Config(db_path="./test_story_db", model_name="qwen2.5:3b")
brain = NeuroSavant(config=config)
agent = StorylineAgent(brain)
if 'infinite' in brain.tools:
    del brain.tools['infinite']

print("\n--- Starting Story Test ---")
facts = agent.execute_workflow("A cyberpunk rebellion in Neo-Tokyo")
print("\n--- Resulting Facts ---")
print(facts)
