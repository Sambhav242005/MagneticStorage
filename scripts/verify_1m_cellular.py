"""
Verify Cellular Memory at 1M Token Scale
========================================

Generates ~1M tokens of synthetic data and tests retrieval.
Target: 5000 chunks (~200 tokens each).
"""

import os
import time
import random
import string
import shutil
from neuro_savant import NeuroSavant, Config

def generate_random_text(num_words=150):
    words = []
    for _ in range(num_words):
        word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        words.append(word)
    return " ".join(words)

def main():
    # 1. Setup
    if os.path.exists("./neuro_savant_memory"):
        shutil.rmtree("./neuro_savant_memory")
    
    print("Initializing CellularMemory...")
    ns = NeuroSavant()
    
    # 2. Generate Data
    NUM_CHUNKS = 5000
    print(f"Generating {NUM_CHUNKS} chunks (~1M tokens)...")
    
    data = []
    # Add filler
    for i in range(NUM_CHUNKS):
        text = f"Chunk {i}: " + generate_random_text()
        data.append(text)
        
    # Add Needles (Facts)
    needles = [
        "The Omega Protocol password is 'Azure-99-Gamma'.",
        "The frequency of the Ghost Signal is 142.8 MHz.",
        "Commander Reyes betrayed Section 9.",
        "The launch code for Project Titan is 'Titan-Alpha-One'.",
        "The rebel base is located on the Moon of Endor."
    ]
    
    # Insert needles at random positions
    needle_indices = []
    for needle in needles:
        idx = random.randint(0, NUM_CHUNKS - 1)
        data[idx] = needle + " " + generate_random_text(50) # Mix with noise
        needle_indices.append(idx)
        print(f"Inserted needle at index {idx}: {needle[:30]}...")
        
    # 3. Ingest
    print("Ingesting data (Batch Mode)...")
    start_time = time.time()
    
    # Process in batches of 1000 to be safe with memory
    BATCH_SIZE = 1000
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        ns.batch_ingest(batch)
        print(f"Ingested {min(i+BATCH_SIZE, len(data))}/{NUM_CHUNKS}...", end='\r')
            
    ingest_time = time.time() - start_time
    print(f"\nIngestion complete in {ingest_time:.2f}s ({NUM_CHUNKS/ingest_time:.1f} chunks/s)")
    
    # 4. Query & Verify
    print("\nVerifying Recall...")
    queries = [
        ("What is the Omega Protocol password?", "Azure-99-Gamma"),
        ("What is the frequency of the Ghost Signal?", "142.8 MHz"),
        ("Who is the traitor in Section 9?", "Commander Reyes"),
        ("What is the launch code for Project Titan?", "Titan-Alpha-One"),
        ("Where is the rebel base located?", "Moon of Endor")
    ]
    
    passed = 0
    total_latency = 0
    
    for q, expected in queries:
        t0 = time.time()
        result = ns.query(q)
        latency = (time.time() - t0) * 1000
        total_latency += latency
        
        if expected in result:
            print(f"✅ [PASS] {q} ({latency:.1f}ms)")
            passed += 1
        else:
            print(f"❌ [FAIL] {q} ({latency:.1f}ms)")
            print(f"   Expected: {expected}")
            print(f"   Got: {result[:100]}...")
            
    print(f"\nResults: {passed}/{len(queries)} Passed")
    print(f"Avg Latency: {total_latency/len(queries):.1f}ms")

if __name__ == "__main__":
    main()
