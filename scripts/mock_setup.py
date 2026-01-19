
import sys
import types
from unittest.mock import MagicMock

# 1. Mock Heavy Dependencies
sys.modules['ollama'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['pysqlite3'] = MagicMock()

# 2. Functional Mock for ChromaDB
class MockCollection:
    def __init__(self, name):
        self.name = name
        self.data = [] # List of (doc, metadata, id)
        
    def add(self, documents, metadatas=None, ids=None):
        for i, doc in enumerate(documents):
            meta = metadatas[i] if metadatas else {}
            id_ = ids[i] if ids else f"id_{len(self.data)}"
            self.data.append({"doc": doc, "meta": meta, "id": id_})
            
    def query(self, query_texts, n_results=1, **kwargs):
        # Naive "Keyword" search to simulate vector similarity
        results = {"documents": [], "ids": [], "metadatas": [], "distances": []}
        
        for q in query_texts:
            # Sort by primitive overlap
            q_lower = q.lower()
            scored = []
            for item in self.data:
                score = 0
                if q_lower in item['doc'].lower():
                    score = 10
                # Boost specific keywords
                for word in q_lower.split():
                    if word in item['doc'].lower():
                        score += 1
                scored.append((score, item))
            
            # Top K
            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = scored[:n_results]
            
            # Format as Chroma expects: List of Lists
            results["documents"].append([x[1]['doc'] for x in top_k])
            results["ids"].append([x[1]['id'] for x in top_k])
            
            # Simulate distances (0.0 = perfect match, 1.0 = far)
            # Our score is roughly 0-15. 
            # Conf = 1 / (1 + dist)  <=>  dist = (1/Conf) - 1
            # We need dist < 0.1 to hit conf > 0.9
            # Let's map score 10 -> dist 0.05
            dists = []
            for x in top_k:
                score = x[0]
                # Inverse mapping with stronger bias
                dist = max(0.01, 1.0 - (score / 12.0))
                if score >= 10: dist = 0.05 # Force good match for keywords
                dists.append(dist)
            results["distances"].append(dists)
            
        return results
        
    def count(self):
        return len(self.data)
        
    def get(self, ids=None, **kwargs):
        # Used by NeuroSavant
        return {"documents": ["Mock Content"], "ids": ids}
        
    def delete(self, ids=None):
        pass
        
    def upsert(self, **kwargs):
        # Alias to add for mock simplicity
        self.add(kwargs.get('documents', []), kwargs.get('metadatas', []), kwargs.get('ids', []))

class MockClient:
    def __init__(self, path=None, **kwargs):
        self.collections = {}
        
    def create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]
        
    def delete_collection(self, name):
        if name in self.collections:
            del self.collections[name]
            
    def get_collection(self, name):
        return self.create_collection(name) # Auto create for mock

    def get_or_create_collection(self, name, metadata=None):
        return self.create_collection(name)

# Create the mock module structure for 'chromadb'
mock_chroma = types.ModuleType("chromadb")
mock_chroma.PersistentClient = MockClient
mock_chroma.Client = MockClient

# Mock 'chromadb.config' which is imported by NeuroSavant
mock_config = types.ModuleType("chromadb.config")
mock_config.Settings = MagicMock()
mock_chroma.config = mock_config

# Inject into sys.modules
sys.modules['chromadb'] = mock_chroma
sys.modules['chromadb.config'] = mock_config
