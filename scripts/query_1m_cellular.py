"""
Query Cellular Memory at 1M Token Scale
=======================================

Queries the existing database to verify recall.
"""

import time
import os
from neuro_savant import NeuroSavant

def main():
    print("Initializing CellularMemory (Existing DB)...")
    # Ensure we use the same encoder!
    os.environ["USE_MOCK_ENCODER"] = "true"
    ns = NeuroSavant()
    
    print(f"Total Groups: {ns.groups.count()}")
    print(f"Total Cells: {ns.cells.count()}")
    print(f"Total Entity Index Entries: {ns.entity_index.count()}")
    
    print("\nVerifying Recall...")
    # Debug Entity Extraction on Needle Text
    needle_text = "The Omega Protocol password is 'Azure-99-Gamma'. Do not share."
    extracted = ns.extractor.extract(needle_text)
    print(f"\nDebug Extraction on Needle Text:\nText: '{needle_text}'\nExtracted: {extracted}")
    
    # Debug specific query
    q = "What is the Omega Protocol password?"
    print(f"\nQuery: {q}")
    res = ns.query(q)
    print(f"Result Context:\n{res}")
    
    if "Azure-99-Gamma" in res:
        print("✅ Found answer in context!")
    else:
        print("❌ Answer NOT found in context.")

if __name__ == "__main__":
    main()
