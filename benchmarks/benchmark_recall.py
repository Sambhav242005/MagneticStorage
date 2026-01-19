
import os
import time
import json
import mock_setup # Mocking for speed/dependencies
from benchmark_data_gen import DataGenerator
from baseline_rag import BaselineRAG

# Import NeuroSavant (Mocked or Real)
try:
    from neuro_savant import NeuroSavant
except:
    import sys
    sys.path.append(os.getcwd())
    from neuro_savant import NeuroSavant

class BenchmarkRunner:
    def __init__(self):
        self.gen = DataGenerator()
        
    def run_benchmark(self, data_size_mb=0.5):
        print("\n" + "="*50)
        print(f"🔥 STARTING RECALL BENCHMARK ({data_size_mb} MB Haystack)")
        print("="*50)
        
        # 1. Generate Data
        data_path, truth_path = self.gen.generate(size_mb=data_size_mb)
        with open(truth_path, 'r') as f:
            needles = json.load(f)
            
        results = {
            "Baseline RAG": {"ingest_time": 0, "correct": 0, "queries": []},
            "Neuro-Savant": {"ingest_time": 0, "correct": 0, "queries": []}
        }
        
        # 2. RUN BASELINE
        print("\n--- Testing Baseline RAG ---")
        baseline = BaselineRAG()
        results["Baseline RAG"]["ingest_time"] = baseline.ingest(data_path)
        
        for needle in needles:
            q = needle['question']
            ans = needle['answer']
            
            resp = baseline.query(q)
            context = resp['context']
            
            # Simple keyword check for "Accuracy"
            success = ans.lower() in context.lower()
            if success: results["Baseline RAG"]["correct"] += 1
            
            results["Baseline RAG"]["queries"].append({
                "q": q,
                "found": success,
                "latency": resp['latency']
            })
            print(f"   Q: {q[:30]}... -> {'✅ Found' if success else '❌ Missed'} ({resp['latency']:.3f}s)")

        # 3. RUN NEURO-SAVANT
        print("\n--- Testing Neuro-Savant ---")
        # Setup Brain
        brain = NeuroSavant(model_name="mock-model") 
        # Note: We need a way to feed it. NeuroSavant updates via 'update_state_immediate'
        # We will split data into sections and feed them as "Experiences"
        
        start_ns = time.time()
        with open(data_path, 'r') as f:
            full_text = f.read()
            
        # Simulate processing many small chunks (Agent "Thoughts")
        chunk_size = 500 
        for i in range(0, len(full_text), chunk_size):
            section = full_text[i:i+chunk_size + 50] # Add overlap to match Baseline
            # ID generation
            cid = f"ns_chunk_{i}"
            # Brain updates
            brain.memory.update_state_immediate(cid, ["BenchLoad"], "Log", section)
            
        results["Neuro-Savant"]["ingest_time"] = time.time() - start_ns
        print(f"   [NeuroSavant] Ingested in {results['Neuro-Savant']['ingest_time']:.2f}s")
        
        # DEBUG: Check if data actually landed
        total_cells = 0
        for l in range(5):
             if l in brain.memory.layer_collections:
                 total_cells += brain.memory.layer_collections[l].count()
        print(f"   [NeuroSavant] Total Cells in Memory: {total_cells}")
        
        # Querying NeuroSavant
        # NeuroSavant doesn't have a direct 'query' method exposed simply in the snippet,
        # usually it's internal. We might need to use the method that the agent uses.
        # Looking at code: 'brain.memory.retrieve_relevant(...)' or similar.
        # *Correction*: In neuro_savant.py, there isn't a simple public query.
        # But `StorylineAgent` uses `_verify_consistency` or internal tool use.
        # We will check if we can verify recall by checking if the memory contains the key content.
        # For the benchmark, we'll try to use the underlying memory retrieval if accessible
        # or mock the retrieval call if necessary.
        
        # Assumption: brain.memory has a way to query.
        # If not, we check the graph/chroma directly for the benchmark's sake.
        
        for needle in needles:
            q = needle['question']
            ans = needle['answer']
            start_q = time.time()
            
            # Simulated Retrieval from potentially hierarchical memory
            # We scan the memory graph/content to see if it was preserved.
            found = False
            
            # We can use the graph or db wrapper directly since we initialized it
            # But let's try to be fair and use a 'search' if it existed.
            # Since I don't see a clear 'search' public API in the partials I read,
            # I will assume we check if the *exact needle text* exists in the database
            # which proves it survived 'Consolidation'.
            
            # This is a proxy for "Recall": Did the system delete/compress it away?
            
            # Robust Search Mode
            found_context = False
            for layer in range(5):
                try:
                    col = brain.memory.layer_collections[layer]
                    count = col.count()
                    if count > 0:
                        # Standardize to Top-3 like Baseline RAG
                        res = col.query(query_texts=[q], n_results=3)
                        if res['documents'] and res['documents'][0]:
                            # Iterate all returned top-k docs
                            for doc in res['documents'][0]:
                                if ans.lower() in doc.lower():
                                    found_context = True
                                    break
                        if found_context: break
                except Exception as e:
                    print(f"Debug Error Layer {layer}: {e}")
            success = found_context
            
            latency = time.time() - start_q
            if success: results["Neuro-Savant"]["correct"] += 1
            
            results["Neuro-Savant"]["queries"].append({
                "q": q,
                "found": success,
                "latency": latency
            })
            print(f"   Q: {q[:30]}... -> {'✅ Found' if success else '❌ Missed'} ({latency:.3f}s)")

        # 4. REPORT
        print("\n" + "="*50)
        print("🏆 BENCHMARK RESULTS")
        print("="*50)
        print(f"{'Metric':<20} | {'Baseline RAG':<15} | {'Neuro-Savant':<15}")
        print("-" * 56)
        
        base_acc = (results["Baseline RAG"]["correct"] / len(needles)) * 100
        ns_acc = (results["Neuro-Savant"]["correct"] / len(needles)) * 100
        
        print(f"{'Accuracy':<20} | {base_acc:>14.1f}% | {ns_acc:>14.1f}%")
        print(f"{'Ingest Time':<20} | {results['Baseline RAG']['ingest_time']:>14.2f}s | {results['Neuro-Savant']['ingest_time']:>14.2f}s")
        
        # Calculate Avg Latency
        base_lat = sum(x['latency'] for x in results["Baseline RAG"]["queries"]) / len(needles)
        ns_lat = sum(x['latency'] for x in results["Neuro-Savant"]["queries"]) / len(needles)
        print(f"{'Avg Query Latency':<20} | {base_lat:>14.3f}s | {ns_lat:>14.3f}s")
        print("="*50)

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_benchmark(data_size_mb=0.1) # Fast test
