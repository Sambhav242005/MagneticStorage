
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import shutil

print("Importing SentenceTransformer...")
from sentence_transformers import SentenceTransformer
print("Loading ST model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded ST model successfully.")
except Exception as e:
    print(f"Failed to load ST: {e}")

print("Importing NeuroSavantV2...")
from neuro_savant_v2 import NeuroSavantV2, Config

# Unique DB path
db_path = f"./debug_db_{int(time.time())}"
Config.db_path = db_path

print(f"Initializing NeuroSavantV2 with db={db_path}...")
ns = NeuroSavantV2()
print("Initialized NeuroSavantV2")

text = """
The Omega Protocol password is 'Azure-99-Gamma'.
The frequency of the Ghost Signal is 142.8 MHz.
Commander Reyes betrayed Section 9.
The launch code for Project Titan is 'Titan-Alpha-One'.
The rebel base is located on the Moon of Endor.

Section 9 is a top secret unit.
Project Titan is a massive weapon.
The Ghost Signal is a mystery.
Commander Reyes is a hero.
"""

print("Ingesting...")
ns.ingest(text)
print("Ingested")

queries = [
    "Who is the traitor in Section 9?",
    "What is the launch code for Project Titan?",
    "What is the frequency of the Ghost Signal?"
]

for q in queries:
    print(f"\n❓ Q: {q}")
    res = ns.query(q)
    print(f"📝 Result:\n{res}")
