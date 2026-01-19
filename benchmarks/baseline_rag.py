
import chromadb
import os
import time
from typing import List, Dict

class BaselineRAG:
    """
    Standard RAG implementation:
    - Fixed Chunking
    - Flat Vector Store (ChromaDB)
    - Cosine Similarity Retrieval
    """
    
    def __init__(self, collection_name="baseline_rag"):
        # Use ephemeral client for benchmark isolation if possible, or local persistence
        self.client = chromadb.PersistentClient(path="./benchmark_baseline_db")
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.client.create_collection(name=collection_name)
        self.chunk_size = 500 # chars ~ 100-150 tokens
        
    def ingest(self, file_path: str):
        """Reads file, chunks it, and adds to ChromaDB"""
        print("   [Baseline] Ingesting...")
        start_time = time.time()
        
        with open(file_path, 'r') as f:
            text = f.read()
            
        # Naive sliding window chunking
        chunks = []
        ids = []
        metadatas = []
        
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i+self.chunk_size + 50] # Slight overlap
            chunks.append(chunk)
            ids.append(f"chunk_{i}")
            metadatas.append({"source": file_path, "offset": i})
            
        # Batch add to Chroma
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                documents=chunks[i:end],
                ids=ids[i:end],
                metadatas=metadatas[i:end]
            )
            
        duration = time.time() - start_time
        print(f"   [Baseline] Ingested {len(chunks)} chunks in {duration:.2f}s")
        return duration

    def query(self, question: str) -> Dict:
        """Retrieve context and return 'answer' (simulation)"""
        start_time = time.time()
        
        results = self.collection.query(
            query_texts=[question],
            n_results=3 # Standard top-k
        )
        
        context = ""
        if results['documents'] and results['documents'][0]:
            context = "\n".join(results['documents'][0])
            
        return {
            "context": context,
            "latency": time.time() - start_time
        }
