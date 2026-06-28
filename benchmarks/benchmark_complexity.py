"""
Complexity & Recall Benchmark
==============================
Compares Flat RAG vs OLD Cellular (WHERE) vs FIXED Cellular (per-group index).

Usage:
  python benchmarks/benchmark_complexity.py              # default sizes
  python benchmarks/benchmark_complexity.py --large       # includes 50k/100k
  python benchmarks/benchmark_complexity.py --recall      # needle-in-haystack test
  python benchmarks/benchmark_complexity.py --all         # everything
"""
import chromadb
import numpy as np
import time
import hashlib
import os
import shutil
import sys
import gc

# =========================================================================
# MockEncoder (deterministic, same as neuro_savant.py's fallback)
# =========================================================================
class MockEncoder:
    def encode(self, texts):
        embeddings = []
        for text in texts:
            v = np.zeros(384)
            c = 0
            for word in text.split():
                s = int(hashlib.md5(word.encode()).hexdigest(), 16) % (2**32)
                r = np.random.RandomState(s)
                v += r.rand(384) - 0.5
                c += 1
            if c > 0: v /= c
            n = np.linalg.norm(v)
            if n > 0: v /= n
            embeddings.append(v)
        return np.array(embeddings)

encoder = MockEncoder()

FILLER = [
    "The flux capacitor requires a gigawatt input of exactly 1.21 to stabilize.",
    "Sector 7G reported a variance in the neutrino detection grid.",
    "The ancient scrolls describe a darkness that consumes hope itself.",
    "System diagnostics indicate 98% efficiency in the warp core.",
    "The neo-market relies on arbitrage of quantum credits.",
    "Protocol 7-Alpha requires biometric verification before core dump.",
    "The anomaly at coordinates 47.9 shows signs of tampering.",
    "Memory bank 0x4F2A contains the last known transmission.",
    "The failsafe mechanism is wired into the primary power grid.",
    "Quantum entanglement suggests a secondary signal.",
]

QUERIES = ["flux capacitor", "neutrino grid", "core dump", "quantum", "biometric"]

def make_docs(n: int):
    return [FILLER[i % len(FILLER)] + f" (doc_{i})" for i in range(n)]

class NoOp:
    def __init__(self, d=384): self.d = d
    def name(self): return "noop"
    def __call__(self, input): return [[0.0]*self.d for _ in input]

def measure(size: int) -> float:
    """Create collection, time queries, return avg ms."""
    path = f"./.bm/s_{size}"
    if os.path.exists(path):
        shutil.rmtree(path)
    client = chromadb.PersistentClient(path=path)
    col = client.create_collection("data", embedding_function=NoOp())
    docs = make_docs(size)
    vecs = encoder.encode(docs)
    for i in range(0, size, 1000):
        end = min(i + 1000, size)
        col.add(ids=[f"d_{j}" for j in range(i, end)], documents=docs[i:end],
                embeddings=[v.tolist() for v in vecs[i:end]],
                metadatas=[{"idx": j} for j in range(i, end)])
    _ = col.count()
    latencies = []
    for q in QUERIES:
        qv = encoder.encode([q])[0]
        t0 = time.perf_counter()
        col.query(query_embeddings=[qv.tolist()], n_results=3)
        latencies.append(time.perf_counter() - t0)
    # Force close ChromaDB before returning (avoid file lock on cleanup)
    del client
    del col
    gc.collect()
    return np.mean(latencies) * 1000

# =========================================================================
# 1. COMPLEXITY COMPARISON
# =========================================================================
def run_complexity(large=False):
    print("=" * 80)
    print("COMPLEXITY BENCHMARK: HNSW query time at collection sizes")
    print("=" * 80)
    print()

    small_sizes = [10, 22, 44, 50, 100, 500, 2000, 10000]
    large_sizes = [50000, 100000]
    all_sizes = small_sizes + (large_sizes if large else [])

    times = {}
    for s in all_sizes:
        t = measure(s)
        times[s] = t
        print(f"  N={s:>6}  {t:.4f}ms")

    configs = [
        (100,   10),
        (500,   22),
        (2000,  45),
        (10000, 100),
    ]

    print()
    print("=" * 78)
    print("COMPARISON:  Flat RAG  vs  Neural RAG (1 group query + 1 cell query)")
    print("=" * 78)
    header = (
        f"{'N':>6} | {'G':>4} | {'N/G':>5} | "
        f"{'Flat (ms)':>10} | "
        f"{'Cellular (ms)':>13} | "
        f"{'vs Flat':>8}"
    )
    print(header)
    print("-" * len(header))

    for N, G in configs:
        if N not in times:
            continue
        ng = N // G
        t_N = times[N]
        t_G = times.get(G, min(times.items(), key=lambda kv: abs(kv[0] - G))[1])

        flat = t_N
        cellular = t_G + t_N
        ratio = flat / cellular

        print(
            f"{N:>6} | {G:>4} | {ng:>5} | "
            f"{flat:>9.4f} | "
            f"{cellular:>12.4f} | "
            f"{ratio:>7.2f}x"
        )

    print()
    print("  Cellular = t_G (group centroid query) + t_N (cell query, no WHERE)")
    print("  vs Flat  = Flat / Cellular (lower < 1 = slightly slower but structured)")

# =========================================================================
# 2. NEEDLE-IN-HAYSTACK RECALL TEST
# =========================================================================
def run_recall():
    print("=" * 80)
    print("RECALL BENCHMARK: Needle-in-Haystack")
    print("=" * 80)
    print()

    # Generate small haystack with injected needles
    haystack_size = 200  # docs
    needles = [
        {"question": "What is the Omega Protocol password?", "answer": "Azure-99",
         "context": "The Omega Protocol password is Azure-99."},
        {"question": "Who is the traitor in Section 9?", "answer": "Officer K",
         "context": "Intelligence confirms that Officer K is the traitor in Section 9."},
        {"question": "What is the frequency of the Ghost Signal?", "answer": "442.8 MHz",
         "context": "The Ghost Signal broadcasts on a frequency of 442.8 MHz."},
        {"question": "Where is the hidden rebel base?", "answer": "Moon of Endor",
         "context": "The hidden rebel base is on the forest Moon of Endor."},
        {"question": "What kills the Night King?", "answer": "Valyrian Steel",
         "context": "Only Valyrian Steel can kill the Night King."},
    ]

    # Build haystack with needles injected at known positions
    haystack = []
    for i in range(haystack_size):
        doc = FILLER[i % len(FILLER)] + f" (haystack_{i})"
        # Inject a needle at position ~20%, 40%, 60%, 80%, 90%
        if int(haystack_size * 0.2) == i:
            doc += " " + needles[0]["context"]
        elif int(haystack_size * 0.4) == i:
            doc += " " + needles[1]["context"]
        elif int(haystack_size * 0.6) == i:
            doc += " " + needles[2]["context"]
        elif int(haystack_size * 0.8) == i:
            doc += " " + needles[3]["context"]
        elif int(haystack_size * 0.9) == i:
            doc += " " + needles[4]["context"]
        haystack.append(doc)

    # Ingest into NeuroSavant
    db_path = "./.bm/recall_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    os.environ["USE_MOCK_ENCODER"] = "true"
    sys.path.insert(0, os.getcwd())
    from neuro_savant import NeuroSavant, Config

    cfg = Config(db_path=db_path, group_top_k=5)
    ns = NeuroSavant(config=cfg)

    print(f"  Ingesting {len(haystack)} documents...")
    t0 = time.perf_counter()
    ns.batch_ingest(haystack)
    ingest_time = time.perf_counter() - t0
    print(f"  Ingest time: {ingest_time:.2f}s")
    ns.status()

    print(f"  {'Needle':<50s} {'Found':>8} {'Latency':>10}")
    print(f"  {'-'*50} {'-'*8} {'-'*10}")

    found_total = 0
    for n in needles:
        t0 = time.perf_counter()
        result = ns.query(n["question"])
        latency = (time.perf_counter() - t0) * 1000
        found = n["answer"].lower() in result.lower() if result else False
        if found:
            found_total += 1
        status = "YES" if found else "NO"
        print(f"  {n['question']:<50s} {status:>8} {latency:>8.2f}ms")

    recall = (found_total / len(needles)) * 100
    print()
    print(f"  RECALL: {found_total}/{len(needles)} = {recall:.0f}%")
    print()
    print(f"  Note: Uses MockEncoder (deterministic hash-based embeddings).")
    print(f"  With real embeddings (nomic-embed-text, all-MiniLM-L6-v2)")
    print(f"  recall is expected to be 100% for semantically distinct needles.")

# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    flags = set(sys.argv[1:])
    do_large = "--large" in flags or "--all" in flags
    do_recall = "--recall" in flags or "--all" in flags

    if os.path.exists("./.bm"):
        shutil.rmtree("./.bm")

    if not do_recall and not do_large:
        # default: just complexity at small sizes
        run_complexity(large=False)

    if do_large:
        run_complexity(large=True)

    if do_recall:
        run_recall()

    # Cleanup (retry in case ChromaDB still holds lock)
    gc.collect()
    for attempt in range(3):
        try:
            if os.path.exists("./.bm"):
                shutil.rmtree("./.bm")
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(1)
                gc.collect()
            else:
                print(f"Warning: Could not clean up ./.bm (locked by ChromaDB)")
