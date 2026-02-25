from neuro_savant import NeuroSavant, Config
import os
import shutil

if os.path.exists("./test_db"):
    shutil.rmtree("./test_db")
    
os.environ["USE_MOCK_ENCODER"] = "true"

config = Config(db_path="./test_db")
brain = NeuroSavant(config=config)

texts = ["The quick brown fox jumps over the lazy dog."]*50 + ["A completely different topic about space exploration."]*50
brain.batch_ingest(texts)
print(f"Groups before: {brain.groups.count()}")
brain.consolidate_memory()
print(f"Groups after: {brain.groups.count()}")
