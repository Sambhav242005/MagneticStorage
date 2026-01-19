"""
Verify Sleep Mode (Consolidation)
=================================

Tests if `consolidate_memory` correctly merges similar groups.
"""

import os
import shutil
import time
from neuro_savant import NeuroSavant, Config

def main():
    # 1. Setup
    if os.path.exists("./neuro_savant_memory"):
        shutil.rmtree("./neuro_savant_memory")
        
    # Use MockEncoder for deterministic testing of merging logic
    # (We want to control similarity easily)
    os.environ["USE_MOCK_ENCODER"] = "true"
    
    # Set low similarity threshold so they form separate groups initially (dist < 0.01 to join)
    # Set low merge threshold so they merge later (sim > 0.5 to merge)
    config = Config(
        similarity_threshold=0.01, # Strict: Only join if almost identical. Forces split.
        merge_threshold=0.5        # Loose: Merge if vaguely similar. Forces merge.
    )
    
    print("Initializing CellularMemory...")
    ns = NeuroSavant(config)
    
    # 2. Ingest Similar Data
    # These should be similar enough to merge, but different enough to split initially if threshold is high
    texts = [
        "The quick brown fox jumps over the dog.",
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over the canine.",
        "Completely unrelated text about space.",
        "Another space text about stars."
    ]
    
    print("Ingesting data...")
    for t in texts:
        ns.ingest(t)
        
    initial_groups = ns.groups.count()
    print(f"Initial Groups: {initial_groups}")
    
    # Debug: Check similarities
    vecs = ns.encoder.encode(texts)
    import numpy as np
    sims = np.dot(vecs, vecs.T)
    print("Pairwise Similarities:")
    print(sims)
    
    # Expectation: 5 groups (since similarity_threshold is 0.99)
    if initial_groups < 3:
        print("WARNING: Groups already merged during ingest? (Check MockEncoder)")
        
    # 3. Run Sleep Mode
    print("\nRunning Sleep Mode...")
    ns.consolidate_memory()
    
    final_groups = ns.groups.count()
    print(f"Final Groups: {final_groups}")
    
    # 4. Verify
    if final_groups < initial_groups:
        print(f"✅ [PASS] Groups merged ({initial_groups} -> {final_groups})")
    else:
        print(f"❌ [FAIL] Groups did not merge ({initial_groups} -> {final_groups})")
        
    # Check if cells are still accessible
    print("\nVerifying Data Integrity...")
    q = "fox jumps dog"
    res = ns.query(q)
    if "quick brown fox" in res:
        print("✅ [PASS] Data still retrievable after merge")
    else:
        print("❌ [FAIL] Data lost after merge")
        print(f"Got: {res}")

if __name__ == "__main__":
    main()
