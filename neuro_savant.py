"""
NeuroSavant 2.0: Cellular Memory Architecture
=============================================

A dynamic, biological-inspired memory system.
- **Cells**: Individual memory units (text + vector).
- **Groups**: Dynamic clusters of cells with a Representation Vector (Centroid).
- **Layer 0**: Fast retrieval of relevant Groups.
- **Reasoning**: Entity-aware processing on retrieved Cells.

Goal: 100% Accuracy, Perfect Recall, Low Latency.
"""

import os
import time
import re
import hashlib
import numpy as np
import threading
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
import chromadb
import argparse
import sys
import requests
import json
HAS_OLLAMA = True

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

# Import performance tracker
try:
    from core.performance_tracker import PerformanceTracker, VisualDisplay
    HAS_PERF_TRACKER = True
except ImportError:
    HAS_PERF_TRACKER = False

# Import tools
# Import tools (individually to avoid one failure breaking all)
HAS_BEHAVIOR_TOOL = False
HAS_EXAMPLE_TOOL = False
HAS_INFINITE_TOOL = False
HAS_STORYLINE_TOOL = False
HAS_INGEST_TOOL = False



try:
    from tools.agent_behavior import AgentBehaviorTool
    HAS_BEHAVIOR_TOOL = True
except ImportError:
    pass

try:
    from tools.example import ExampleTool
    HAS_EXAMPLE_TOOL = True
except ImportError:
    pass

try:
    from tools.infinite import InfiniteLoopTool
    HAS_INFINITE_TOOL = True
except ImportError:
    pass

try:
    from tools.storyline_agent import StorylineAgent
    HAS_STORYLINE_TOOL = True
except ImportError:
    pass

try:
    from tools.github_ingest import GitHubIngestTool
    HAS_INGEST_TOOL = True
except ImportError:
    pass


# Try to import SentenceTransformer, fall back to MockEncoder if it fails/hangs
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    db_path: str = "./neuro_savant_memory"
    model_name: str = "deepseek-r1:1.5b"
    embed_model: str = "nomic-embed-text"
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    use_agentic: bool = True
    similarity_threshold: float = 0.4
    merge_threshold: float = 0.85
    group_top_k: int = 3
    cell_top_k: int = 200
    entity_boost: float = 0.5
    filename_boost: float = 2.0
    file_meta_boost: float = 0.3

# =============================================================================
# RETRY WRAPPER
# =============================================================================

def _db_retry(fn, max_retries: int = 2, backoff: float = 0.5):
    """Retry a ChromaDB call on failure with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff * (attempt + 1))

# =============================================================================
# AGENTIC TOOLS DEFINITION
# =============================================================================

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search the vector database for relevant information. Use this when you need to retrieve context about the user's projects, files, or previous conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_memory",
            "description": "Store new information in memory. Use this when the user shares important facts, project details, preferences, or anything worth remembering for future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to store in memory"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what is being stored (optional)"
                    }
                },
                "required": ["content"]
            }
        }
    }
]

# =============================================================================
# MOCK ENCODER (Fallback)
# =============================================================================

class MockEncoder:
    """
    Deterministic dense encoder fallback.
    """
    def __init__(self, device='cpu'):
        pass
        
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Dense deterministic embedding
            final_vec = np.zeros(384)
            count = 0
            for word in text.split():
                # Generate dense vector for word
                seed = int(hashlib.md5(word.encode()).hexdigest(), 16) % (2**32)
                rng = np.random.RandomState(seed)
                word_vec = rng.rand(384) - 0.5 
                final_vec += word_vec
                count += 1
            
            if count > 0:
                final_vec /= count
            
            # Normalize
            norm = np.linalg.norm(final_vec)
            if norm > 0:
                final_vec = final_vec / norm
            embeddings.append(final_vec)
        return np.array(embeddings)

# =============================================================================
# OLLAMA ENCODER (Primary - uses nomic-embed-text)
# =============================================================================

class OllamaEncoder:
    """
    Uses Ollama's embedding API with nomic-embed-text model.
    Produces 768-dimensional embeddings.
    """
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.dimension = 768  # nomic-embed-text produces 768-dim vectors
        print(f"INFO: Using Ollama encoder with model: {model}")
        
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text}
                )
                if response.status_code == 200:
                    embedding = response.json().get('embedding', [])
                    embeddings.append(np.array(embedding))
                else:
                    # Fallback to zero vector on error
                    print(f"WARNING: Embedding API error: {response.status_code}")
                    embeddings.append(np.zeros(self.dimension))
            except Exception as e:
                print(f"WARNING: Embedding failed: {e}")
                embeddings.append(np.zeros(self.dimension))
        return np.array(embeddings)

# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

class EntityExtractor:
    def __init__(self):
        self.patterns = [
            # Files and paths (HIGHEST PRIORITY)
            r'\[File: ([^\]]+)\]',              # Files in metadata headers
            r'\b([\w-]+\.(?:py|js|ts|java|cpp|c|h|go|rs|md|txt|yaml|yml|json|xml|html|css))\b',  # Common code files
            r'\b([\w-]+/[\w-]+(?:/[\w-]+)*\.\w+)\b',  # File paths
            r'\b([\w-]+\.(?:config|conf|cfg))\b',  # Config files
            
            # Generic Entity Extractors
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', # Generic capitalized phrase
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', # Emails
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', # URLs
        ]
        self.compiled = [re.compile(p) for p in self.patterns]

    def extract(self, text: str) -> List[str]:
        entities = set()
        for pattern in self.compiled:
            matches = pattern.findall(text)
            entities.update(matches)
        return list(entities)

# =============================================================================
# CELLULAR MEMORY
# =============================================================================

class NeuroSavant:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.db_path)
        use_mock = os.environ.get("USE_MOCK_ENCODER", "false").lower() == "true"
        
        # Load Encoder - Try Ollama first, then SentenceTransformer, then Mock
        try:
            test_response = requests.get(f"{self.ollama_api_base}/api/tags", timeout=2)
            if test_response.status_code == 200:
                self.encoder = OllamaEncoder(model=config.embed_model, base_url=self.ollama_api_base)
                print(f"INFO: Using Ollama embeddings: {config.embed_model} @ {self.ollama_api_base}")
            else:
                raise Exception("Ollama not responding")
        except:
            # Fallback to SentenceTransformer
            use_mock = os.environ.get("USE_MOCK_ENCODER", "false").lower() == "true"
            if HAS_ST and not use_mock:
                try:
                    print("INFO: Ollama unavailable. Loading SentenceTransformer...")
                    self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                except Exception as e:
                    print(f"WARNING: SentenceTransformer failed ({e}). Using MockEncoder.")
                    self.encoder = MockEncoder()
            else:
                print("WARNING: Using MockEncoder (no Ollama or SentenceTransformer).")
                self.encoder = MockEncoder()
        if use_mock:
            print("WARNING: Overriding encoder with MockEncoder (USE_MOCK_ENCODER=true).")
            self.encoder = MockEncoder()

        self.extractor = EntityExtractor()
        
        # Get encoder dimension
        encoder_dim = getattr(self.encoder, 'dimension', 384)  # Default to 384 if not specified
        
        # Create a no-op embedding function to silence ChromaDB warnings
        # We always provide pre-computed embeddings, so this is never actually used
        class NoOpEmbeddingFunction:
            def __init__(self, dim):
                self.dim = dim
            def name(self):
                return "default"
            def __call__(self, input):
                # This should never be called since we always pass embeddings directly
                return [[0.0] * self.dim for _ in input]
        
        noop_ef = NoOpEmbeddingFunction(encoder_dim)
        
        # Collections (with no-op embedding function to silence warnings)
        self.cells = self.client.get_or_create_collection("ns_cells", embedding_function=noop_ef)
        self.groups = self.client.get_or_create_collection("ns_groups", embedding_function=noop_ef)
        self.entity_index = self.client.get_or_create_collection("ns_entity_index", embedding_function=noop_ef)
        
        # In-memory cache for group updates (thread-safe)
        self.group_cache = {}
        self._cache_lock = threading.Lock()
        self._load_groups()
        
        # Initialize performance tracker
        if HAS_PERF_TRACKER:
            self.perf_tracker = PerformanceTracker()
            self.visual = VisualDisplay()
        else:
            self.perf_tracker = None
            self.visual = None
        
        # Initialize tools
        self.tools = {}
        loaded_tools = []
        self.behavior_tool = None
        self.example_tool = None
        self.infinite_tool = None
        self.ingest_tool = None
        
        if HAS_BEHAVIOR_TOOL:
            self.tools['behavior'] = AgentBehaviorTool()
            self.behavior_tool = self.tools['behavior']
            loaded_tools.append('behavior')
        
        if HAS_EXAMPLE_TOOL:
            self.tools['example'] = ExampleTool()
            self.example_tool = self.tools['example']
            loaded_tools.append('example')
        
        if HAS_INFINITE_TOOL:
            self.tools['infinite'] = InfiniteLoopTool()
            self.infinite_tool = self.tools['infinite']
            loaded_tools.append('infinite')
        
        if HAS_INGEST_TOOL:
            self.tools['ingest'] = GitHubIngestTool(memory_grid=self)
            self.ingest_tool = self.tools['ingest']
            loaded_tools.append('ingest')
        
        if loaded_tools:
            print(f"INFO: Tools loaded: {', '.join(loaded_tools)}")

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @model_name.setter
    def model_name(self, value: str):
        self.config.model_name = value

    @property
    def ollama_api_base(self) -> str:
        return f"http://{self.config.ollama_host}:{self.config.ollama_port}"

    def _db_retry(self, fn, max_retries=2, backoff=0.5):
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff * (attempt + 1))

    def _cell_id(self, text: str) -> str:
        return f"cell_{hashlib.md5(text.encode()).hexdigest()}"

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            return vector
        return vector / norm

    def _get_existing_ids(self, collection, ids: List[str]) -> Set[str]:
        if not ids:
            return set()

        try:
            existing = collection.get(ids=ids)
        except Exception:
            return set()

        return set(existing.get('ids', []) or [])

    def _load_groups(self):
        try:
            existing = self._db_retry(lambda: self.groups.get(include=['embeddings', 'metadatas']))
            if existing['ids']:
                with self._cache_lock:
                    for i, gid in enumerate(existing['ids']):
                        emb = existing['embeddings'][i]
                        meta = existing['metadatas'][i]
                        self.group_cache[gid] = {
                            'centroid': np.array(emb),
                            'count': meta.get('count', 1)
                        }
        except:
            pass

    def ingest(self, text: str) -> bool:
        cell_id = self._cell_id(text)
        if cell_id in self._get_existing_ids(self.cells, [cell_id]):
            return False

        vector = self.encoder.encode([text])[0]

        best_group_id: Optional[str] = None
        min_dist = float('inf')

        with self._cache_lock:
            if self.group_cache:
                for gid, data in self.group_cache.items():
                    dist = 1 - np.dot(vector, data['centroid'])
                    if dist < min_dist:
                        min_dist = dist
                        best_group_id = gid

        if best_group_id and min_dist < self.config.similarity_threshold:
            self._update_group(best_group_id, vector)
            group_id = best_group_id
        else:
            group_id = f"group_{int(time.time()*1000)}_{hash(text)%1000}"
            self._create_group(group_id, vector)

        self.cells.upsert(
            ids=[cell_id],
            documents=[text],
            embeddings=[vector.tolist()],
            metadatas=[{"group_id": group_id, "creation_timestamp": time.time()}]
        )

        entities = self.extractor.extract(text)
        if entities:
            entity_vectors = self.encoder.encode(entities)
            for entity, entity_vector in zip(entities, entity_vectors):
                eid = f"idx_{hashlib.md5((entity + group_id).encode()).hexdigest()}"
                self.entity_index.upsert(
                    ids=[eid],
                    documents=[entity],
                    embeddings=[entity_vector.tolist()],
                    metadatas=[{"group_id": group_id, "entity": entity}]
                )

        return True

    def batch_ingest(self, texts: List[str]) -> None:
        """
        Optimized batch ingestion.
        """
        if not texts:
            return

        deduped_texts = []
        deduped_cell_ids = []
        seen_cell_ids = set()

        for text in texts:
            cell_id = self._cell_id(text)
            if cell_id in seen_cell_ids:
                continue
            seen_cell_ids.add(cell_id)
            deduped_texts.append(text)
            deduped_cell_ids.append(cell_id)

        existing_cell_ids = self._get_existing_ids(self.cells, deduped_cell_ids)
        pending_items = [
            (text, cell_id)
            for text, cell_id in zip(deduped_texts, deduped_cell_ids)
            if cell_id not in existing_cell_ids
        ]

        if not pending_items:
            return

        vectors = self.encoder.encode([text for text, _ in pending_items])
        
        # Prepare batch data
        cell_ids = []
        cell_docs = []
        cell_embs = []
        cell_metas = []
        
        group_upserts = {} # gid -> {centroid, count}
        
        entity_ids = []
        entity_docs = []
        entity_metas = []
        entity_embs = []
        
        # 2. Process each text
        for i, (text, cell_id) in enumerate(pending_items):
            vector = vectors[i]
            
            # Vectorized Group Finding (Online Clustering)
            best_group_id = None
            min_dist = float('inf')
            
            with self._cache_lock:
                if self.group_cache:
                    gids = list(self.group_cache.keys())
                    centroids = np.array([self.group_cache[gid]['centroid'] for gid in gids])

                    similarities = np.dot(centroids, vector)

                    best_idx = np.argmax(similarities)
                    max_sim = similarities[best_idx]
                    min_dist = 1.0 - max_sim

                    if min_dist < self.config.similarity_threshold:
                        best_group_id = gids[best_idx]

            with self._cache_lock:
                if best_group_id and min_dist < self.config.similarity_threshold:
                    data = self.group_cache[best_group_id]
                    n = data['count']
                    old_centroid = data['centroid']
                    new_centroid = (old_centroid * n + vector) / (n + 1)
                    new_centroid = self._normalize_vector(new_centroid)

                    data['centroid'] = new_centroid
                    data['count'] = n + 1

                    group_upserts[best_group_id] = {'centroid': new_centroid, 'count': n + 1}
                    group_id = best_group_id
                else:
                    group_id = f"group_{int(time.time()*1000)}_{i}_{hash(text)%1000}"
                    normalized = self._normalize_vector(vector)

                    self.group_cache[group_id] = {'centroid': normalized, 'count': 1}
                    group_upserts[group_id] = {'centroid': normalized, 'count': 1}
                group_id = group_id
                
            # Cell Data
            cell_ids.append(cell_id)
            cell_docs.append(text)
            cell_embs.append(vector.tolist())
            cell_metas.append({"group_id": group_id, "creation_timestamp": time.time()})
            
            # Entity Data
            entities = self.extractor.extract(text)
            for e_idx, entity in enumerate(entities):
                eid = f"idx_{hashlib.md5((entity + group_id).encode()).hexdigest()}_{i}_{e_idx}"
                entity_ids.append(eid)
                entity_docs.append(entity)
                entity_metas.append({"group_id": group_id, "entity": entity})
                
        # 3. Bulk Write
        self.cells.upsert(
            ids=cell_ids,
            documents=cell_docs,
            embeddings=cell_embs,
            metadatas=cell_metas
        )
        
        # Groups (Flush cache to DB)
        if group_upserts:
            g_ids = list(group_upserts.keys())
            g_embs = [group_upserts[gid]['centroid'].tolist() for gid in g_ids]
            g_metas = [{"count": group_upserts[gid]['count']} for gid in g_ids]
            
            self.groups.upsert(
                ids=g_ids,
                embeddings=g_embs,
                metadatas=g_metas
            )
        
        # Entities
        if entity_ids:
            unique_entities = {}
            for entity_id, entity_doc, entity_meta in zip(entity_ids, entity_docs, entity_metas):
                unique_entities[entity_id] = (entity_doc, entity_meta)

            entity_ids = list(unique_entities.keys())
            entity_docs = [unique_entities[entity_id][0] for entity_id in entity_ids]
            entity_metas = [unique_entities[entity_id][1] for entity_id in entity_ids]
            entity_vectors = self.encoder.encode(entity_docs)
            entity_embs = [vector.tolist() for vector in entity_vectors]
            self.entity_index.upsert(
                ids=entity_ids,
                documents=entity_docs,
                embeddings=entity_embs,
                metadatas=entity_metas
            )

    def _create_group(self, group_id: str, vector: np.ndarray):
        normalized = self._normalize_vector(vector)
        with self._cache_lock:
            self.group_cache[group_id] = {'centroid': normalized, 'count': 1}
        self._db_retry(lambda: self.groups.add(
            ids=[group_id],
            embeddings=[normalized.tolist()],
            metadatas=[{"count": 1}]
        ))

    def _update_group(self, group_id: str, new_vector: np.ndarray):
        with self._cache_lock:
            data = self.group_cache[group_id]
            n = data['count']
            old_centroid = data['centroid']

            new_centroid = (old_centroid * n + new_vector) / (n + 1)
            new_centroid = self._normalize_vector(new_centroid)

            data['centroid'] = new_centroid
            data['count'] = n + 1

        self._db_retry(lambda: self.groups.upsert(
            ids=[group_id],
            embeddings=[new_centroid.tolist()],
            metadatas=[{"count": n + 1}]
        ))

    def query(self, query_text: str) -> str:
        query_start = time.perf_counter()
        
        # 1. Extract Entities
        entities = self.extractor.extract(query_text)
        print(f"DEBUG: Query='{query_text}', Entities={entities}")
        
        # 2. Layer 0: Find relevant groups (Centroid Search)
        embed_start = time.perf_counter()
        query_vec = self.encoder.encode([query_text])[0]
        embed_time = (time.perf_counter() - embed_start) * 1000
        print(f"DEBUG: Embedding time: {embed_time:.1f}ms")
        
        # Search groups (with safety check for n_results)
        target_group_ids = set()
        group_count = self.groups.count()
        if group_count > 0:
            n_groups = min(self.config.group_top_k, group_count)
            groups = self.groups.query(
                query_embeddings=[query_vec.tolist()],
                n_results=n_groups
            )
            if groups['ids'] and groups['ids'][0]:
                target_group_ids.update(groups['ids'][0])
                print(f"DEBUG: Found Groups via centroid: {target_group_ids}")
        else:
            print("DEBUG: No groups in database.")
            
        # Stage 1.5: Entity Index Lookup (Guaranteed Recall)
        if entities:
            entity_count = self.entity_index.count()
            if entity_count > 0:
                n_entities = min(self.config.group_top_k, entity_count)
                for entity in entities:
                    e_vec = self.encoder.encode([entity])[0].tolist()
                    e_results = self.entity_index.query(
                        query_embeddings=[e_vec],
                        n_results=n_entities
                    )
                    print(f"DEBUG: Entity '{entity}' lookup results: {e_results['ids']}")
                    if e_results['ids'] and e_results['ids'][0]:
                        # Add all group_ids associated with the top-k entity results.
                        target_group_ids.update([m['group_id'] for m in e_results['metadatas'][0]])
                        print(f"DEBUG: Added Groups via entity '{entity}': {[m['group_id'] for m in e_results['metadatas'][0]]}")
            
        target_group_ids = list(target_group_ids)
            
        if not target_group_ids:
            return "No memory found."
            
        # Stage 2: Search Cells within these Groups
        # Fetch more candidates and filter in Python (more robust than Chroma $contains)
        # Note: ChromaDB 0.3.23 does not support $in operator, so we query iteratively.
        
        accumulated_docs = []
        accumulated_dists = []
        
        for gid in target_group_ids:
            try:
                # Chroma 0.3.23 throws error if n_results > count
                total_count = self.cells.count()
                k = min(self.config.cell_top_k, total_count)
                if k > 0:
                    res = self.cells.query(
                        query_embeddings=[query_vec.tolist()],
                        n_results=k,
                        where={"group_id": gid}
                    )
                    if res['documents'] and res['documents'][0]:
                        accumulated_docs.extend(res['documents'][0])
                        # Handle distances if present
                        if 'distances' in res and res['distances']:
                           accumulated_dists.extend(res['distances'][0])
                        else:
                           # Fallback if no distances (should not happen with embeddings)
                           accumulated_dists.extend([0.5] * len(res['documents'][0]))
            except Exception as e:
                print(f"DEBUG: Error querying group {gid}: {e}")

        # Construct candidate list from accumulated results
        candidates = []
        for i, doc in enumerate(accumulated_docs):
            dist = accumulated_dists[i] if i < len(accumulated_dists) else 1.0
            score = 1.0 / (1.0 + dist)
            candidates.append({'doc': doc, 'score': score})
        
        # Stage 3: Reasoning & Re-ranking (Entity Boosting + Filename Boosting)
        query_lower = query_text.lower()
        for cand in candidates:
            doc = cand['doc']
            doc_lower = doc.lower()
            
            for entity in entities:
                if entity.lower() in doc_lower:
                    cand['score'] += self.config.entity_boost

            query_words = query_text.split()
            for word in query_words:
                if '.' in word or '/' in word:
                    if word.lower() in doc_lower:
                        cand['score'] += self.config.filename_boost
                        print(f"DEBUG: Filename boost for '{word}' in doc")

            if '[File:' in doc:
                for word in query_words:
                    if word.lower() in doc_lower and len(word) > 3:
                        cand['score'] += self.config.file_meta_boost
                    
        # Sort and Return Top 3
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        final_docs = [c['doc'] for c in candidates[:3]]
        return "\n---\n".join(final_docs)

    def consolidate_memory(self) -> None:
        """
        Sleep Mode: Merges similar groups to compact the index.
        Uses greedy vectorized consolidation to avoid O(N^2) memory explosion.
        """
        print("Starting Sleep Mode Consolidation...")
        
        all_groups = self.groups.get(include=['embeddings', 'metadatas'])
        if not all_groups['ids']:
            print("No groups to consolidate.")
            return
            
        ids = all_groups['ids']
        embeddings = np.array(all_groups['embeddings'])
        metadatas = all_groups['metadatas']
        
        n_groups = len(ids)
        if n_groups < 2:
            return
        
        # Normalize embeddings just in case
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
        
        merged_indices = set()
        merges = 0
        
        # Batch similarities calculation (One-vs-Rest)
        for i in range(n_groups):
            if i in merged_indices:
                continue
                
            target_id = ids[i]
            vec_i = embeddings[i]
            count_i = metadatas[i].get('count', 1) if metadatas[i] else 1
            
            # (N, D) x (D,) -> (N,)
            sims = np.dot(embeddings, vec_i)
            
            for j in range(i + 1, n_groups):
                if j in merged_indices:
                    continue
                    
                if sims[j] > self.config.merge_threshold:
                    source_id = ids[j]
                    count_j = metadatas[j].get('count', 1) if metadatas[j] else 1
                    
                    print(f"Merging Group {source_id} -> {target_id} (Sim: {sims[j]:.3f})")
                    
                    source_cells = self._db_retry(lambda: self.cells.get(where={"group_id": source_id}))
                    if source_cells['ids']:
                        new_metas = [dict(m, group_id=target_id) for m in source_cells['metadatas']]
                        self._db_retry(lambda: self.cells.update(ids=source_cells['ids'], metadatas=new_metas))

                    source_entities = self._db_retry(lambda: self.entity_index.get(where={"group_id": source_id}))
                    if source_entities['ids']:
                        new_ent_metas = [dict(m, group_id=target_id) for m in source_entities['metadatas']]
                        self._db_retry(lambda: self.entity_index.update(ids=source_entities['ids'], metadatas=new_ent_metas))

                    vec_j = embeddings[j]
                    new_centroid = (vec_i * count_i + vec_j * count_j) / (count_i + count_j)
                    new_centroid = self._normalize_vector(new_centroid)

                    self._db_retry(lambda: self.groups.update(
                        ids=[target_id],
                        embeddings=[new_centroid.tolist()],
                        metadatas=[{"count": count_i + count_j}]
                    ))

                    self._db_retry(lambda: self.groups.delete(ids=[source_id]))
                    merged_indices.add(j)
                    
                    merged_indices.add(j)
                    merges += 1
                    
                    # Update iteration state
                    embeddings[i] = new_centroid
                    vec_i = new_centroid
                    count_i += count_j
                    if metadatas[i] is None: metadatas[i] = {}
                    metadatas[i]['count'] = count_i
                    
        print(f"Consolidation Complete. Merged {merges} groups.")
        with self._cache_lock:
            self.group_cache = {}
        self._load_groups()

    def _run_conflict_analysis_agent(self, cell_contents: List[str]) -> Optional[str]:
        """
        Uses an LLM to analyze a list of memory cells for contradictions.
        """
        if not HAS_OLLAMA or not self.config.use_agentic:
            return None

        # TODO: This needs a more robust implementation to handle large contexts
        # and parse the output reliably.
        
        system_prompt = """You are a meticulous AI analyst. Your task is to examine the following statements and identify any direct contradictions.

Respond in one of two ways:
1. If there are no contradictions, respond with "NO_CONFLICTS".
2. If you find contradictions, respond with a brief analysis explaining the conflicting points.

Statements:
"""
        
        full_prompt = system_prompt + "\n".join([f"- {c}" for c in cell_contents])

        try:
            response = requests.post(
                f"{self.ollama_api_base}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            if response.status_code == 200:
                analysis = response.json().get('response', '').strip()
                if analysis and "NO_CONFLICTS" not in analysis:
                    return analysis
        except Exception as e:
            print(f"ERROR: Conflict analysis agent failed: {e}")
        
        return None

    def agentic_consolidation(self) -> None:
        """
        Agentic Sleep Mode: Analyzes groups for semantic conflicts.
        """
        print("🧠 Starting Agentic Sleep Mode...")
        all_groups = self.groups.get()
        if not all_groups['ids']:
            print("No groups to analyze.")
            return

        conflicts_found = 0
        for group_id in all_groups['ids']:
            cells_in_group = self.cells.get(where={"group_id": group_id}, include=["documents"])
            
            # We only need to check for conflicts if there's more than one memory
            if len(cells_in_group['ids']) > 1:
                print(f"Analyzing group {group_id} with {len(cells_in_group['ids'])} cells...")
                
                documents = cells_in_group['documents']
                
                # Run agentic analysis
                conflict_analysis = self._run_conflict_analysis_agent(documents)

                if conflict_analysis:
                    conflicts_found += 1
                    print("\n" + "="*25 + " CONFLICT DETECTED " + "="*25)
                    print(f"Group ID: {group_id}")
                    print(f"Analysis: {conflict_analysis}")
                    print("Cells in conflict:")
                    for doc in documents:
                        print(f"  - {doc.strip()}")
                    print("="*70 + "\n")

        print(f"Agentic Sleep Mode finished. Found {conflicts_found} potential conflicts.")

    def _execute_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool call from the LLM"""
        success = False
        try:
            if tool_name == "search_memory":
                query = tool_args.get("query", "")
                max_results = tool_args.get("max_results", 3)
                print(f"DEBUG: LLM searching for: '{query}'")
                
                # Use existing query method
                result = self.query(query)
                
                # Limit results if needed
                if result and max_results < 3:
                    docs = result.split("\n---\n")
                    result = "\n---\n".join(docs[:max_results])
                
                success = True
                return result if result else "No results found."
            
            elif tool_name == "add_to_memory":
                content = tool_args.get("content", "")
                summary = tool_args.get("summary", "")
                
                if not content:
                    return "Error: No content provided to store."
                
                print(f"DEBUG: LLM storing: '{summary if summary else content[:50]}...'")
                
                # Create formatted content with summary if provided
                if summary:
                    formatted_content = f"[Summary: {summary}]\n\n{content}"
                else:
                    formatted_content = content
                
                # Store in memory using existing ingest method
                self.ingest(formatted_content)
                
                success = True
                return f"Successfully stored in memory: {summary if summary else 'information saved'}"
            
            else:
                return f"Unknown tool: {tool_name}"
        finally:
            if self.perf_tracker:
                self.perf_tracker.record_tool_usage(tool_name, success)
    
    def _streaming_chat(self, messages: list) -> str:
        """Non-agentic streaming chat call."""
        print("Assistant: ", end="", flush=True)
        response = requests.post(
            f"{self.ollama_api_base}/api/chat",
            json={
                "model": self.config.model_name,
                "messages": messages,
                "stream": True
            },
            stream=True
        )
        if response.status_code != 200:
            print(f"\nWARN: Ollama API error: {response.text}")
            return "Error connecting to Ollama."

        full_reply = ""
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                try:
                    chunk = json.loads(decoded)
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        print(content, end='', flush=True)
                        full_reply += content
                except:
                    pass
        print()
        return full_reply

    def _agentic_chat(self, messages: list) -> str:
        """Agentic chat with function calling loop. Returns final reply."""
        full_reply = ""
        tool_calls_made = 0
        max_tool_calls = 5

        truncated_context = messages[0]["content"][:3000]

        while tool_calls_made < max_tool_calls:
            response = requests.post(
                f"{self.ollama_api_base}/api/chat",
                json={
                    "model": self.config.model_name,
                    "messages": messages,
                    "tools": MEMORY_TOOLS,
                    "stream": False,
                },
            )
            if response.status_code != 200:
                print(f"\nWARN: Ollama API error: {response.text}")
                return "Error connecting to Ollama."

            result = response.json()
            message = result.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                messages.append(message)
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    tool_result = self._execute_tool_call(tool_name, tool_args)
                    messages.append({"role": "tool", "content": tool_result})
                    tool_calls_made += 1
            else:
                full_reply = message.get("content", "")
                break

        if tool_calls_made >= max_tool_calls:
            print("\nWARN: Max tool calls reached, getting final response...")
            response = requests.post(
                f"{self.ollama_api_base}/api/chat",
                json={
                    "model": self.config.model_name,
                    "messages": messages,
                    "stream": False,
                },
            )
            full_reply = response.json().get("message", {}).get("content", "")

        if full_reply:
            print(f"Assistant: {full_reply}")

        return full_reply

    def chat(self, user_input: str) -> str:
        """
        Interactive chat with agentic memory search.
        """
        total_start = time.perf_counter()
        print(f"Thinking... (Model: {self.config.model_name})")

        query_start = time.perf_counter()
        context = self.query(user_input)
        query_time = (time.perf_counter() - query_start) * 1000

        full_reply = ""

        if not HAS_OLLAMA:
            print("ERROR: Ollama not found. Cannot generate response.")
            print(f"Context found:\n{context}")
            return "Ollama not installed."

        ollama_start = time.perf_counter()
        try:
            if 'infinite' in self.tools and self.tools['infinite'].active:
                print(f"   Delegating to Infinite Loop Tool...")
                full_text, chunks = self.tools['infinite'].generate_sequence(
                    model_name=self.config.model_name,
                    system_prompt="You are a helpful AI assistant with access to a vast memory database.",
                    user_prompt=user_input
                )
                total_time = (time.perf_counter() - total_start) * 1000
                print(f"  Infinite Generation Complete | Total: {total_time:.0f}ms")
                return full_text

            messages = [
                {"role": "system", "content": f"You are a helpful AI assistant with advanced memory.\n\nRELEVANT MEMORY CONTEXT:\n{context[:4000]}"},
                {"role": "user", "content": user_input}
            ]

            if self.config.use_agentic:
                full_reply = self._agentic_chat(messages)
            else:
                full_reply = self._streaming_chat(messages)

        except Exception as e:
            print(f"\nWARN: Generation error: {e}")
            full_reply = "I apologize, but I encountered an error communicating with the model."

        ollama_time = (time.perf_counter() - ollama_start) * 1000

        ingest_start = time.perf_counter()
        interaction = f"User: {user_input}\nAssistant: {full_reply}"
        self.ingest(interaction)
        ingest_time = (time.perf_counter() - ingest_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        print(
            f"\nTiming: Query {query_time:.0f}ms | Ollama {ollama_time:.0f}ms | "
            f"Ingest {ingest_time:.0f}ms | Total {total_time:.0f}ms"
        )

        if self.perf_tracker:
            self.perf_tracker.metrics.query_times.append(query_time)
            self.perf_tracker.metrics.total_queries += 1

        return full_reply

    def status(self) -> None:
        """Display system status"""
        try:
            cell_count = self.cells.count()
            group_count = self.groups.count()
            entity_count = self.entity_index.count()
        except:
            cell_count = group_count = entity_count = 0
        
        print("\n" + "=" * 60)
        print("NEURO-SAVANT STATUS")
        print("=" * 60)
        print(f"  Database Path: {self.config.db_path}")
        print(f"  Model: {self.config.model_name}")
        print(f"  Total Cells: {cell_count}")
        print(f"  Total Groups: {group_count}")
        print(f"  Entity Index: {entity_count}")
        print(f"  Tools Loaded: {len(self.tools)}")
        print("=" * 60 + "\n")
        
        # Show visual if available
        if self.visual:
            print(self.visual.memory_stats_visual(cell_count, group_count, entity_count))

    def shutdown(self):
        """Clean shutdown"""
        print("Shutting down...")
        # No background threads to stop in this version yet
        
    def clear_memory(self):
        """Wipe all memory data (works while running)"""
        print("Cleaning memory...")
        try:
            # Clear all collections using ChromaDB's delete API
            collections = [self.cells, self.groups, self.entity_index]
            total_deleted = 0
            
            for collection in collections:
                try:
                    all_ids = collection.get()['ids']
                    if all_ids:
                        # Delete in batches to handle large collections
                        batch_size = 1000
                        for i in range(0, len(all_ids), batch_size):
                            batch = all_ids[i:i + batch_size]
                            collection.delete(ids=batch)
                            total_deleted += len(batch)
                except Exception as e:
                    print(f"  WARN: Error clearing collection: {e}")
            
            # Clear in-memory cache
            self.group_cache = {}
            
            print(f"  OK: Memory wiped. Deleted {total_deleted} items.")
            
        except Exception as e:
            print(f"ERROR: Failed to clear memory: {e}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Neuro-Savant: Cellular Memory Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek-r1:1.5b",
        help="Ollama LLM model (default: deepseek-r1:1.5b)"
    )
    parser.add_argument(
        "--embed", "-e",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model (default: nomic-embed-text)"
    )
    parser.add_argument(
        "--db", "-d",
        type=str,
        default="./neuro_savant_memory",
        help="Path to memory database (default: ./neuro_savant_memory)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.environ.get("OLLAMA_HOST", "localhost"),
        help="Ollama server host (default: localhost, env: OLLAMA_HOST)"
    )
    parser.add_argument(
        "--ollama-port",
        type=int,
        default=int(os.environ.get("OLLAMA_PORT", "11434")),
        help="Ollama server port (default: 11434, env: OLLAMA_PORT)"
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for group merging during sleep mode (default: 0.85)"
    )
    parser.add_argument(
        "--group-top-k",
        type=int,
        default=3,
        help="Number of groups to retrieve during query (default: 3)"
    )
    parser.add_argument(
        "--cell-top-k",
        type=int,
        default=200,
        help="Number of cells per group during query (default: 200)"
    )
    parser.add_argument(
        "--disable-agentic",
        action="store_true",
        help="Disable agentic function-calling mode"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("NEURO-SAVANT v2.0 - Cellular Memory Architecture")
    print("=" * 60 + "\n")
    
    print(f"LLM Model: {args.model}")
    print(f"Embed Model: {args.embed}")
    print(f"Database: {args.db}\n")
    
    try:
        config = Config(
            db_path=args.db,
            model_name=args.model,
            embed_model=args.embed,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            merge_threshold=args.merge_threshold,
            group_top_k=args.group_top_k,
            cell_top_k=args.cell_top_k,
            use_agentic=not args.disable_agentic,
        )
        agent = NeuroSavant(config)
    except Exception as e:
        print(f"ERROR: Init failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Type /help for all commands")
    print("=" * 60 + "\n")
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            if user_input == "/quit":
                agent.shutdown()
                break
            elif user_input == "/help":
                print("""
NEURO-SAVANT COMMANDS
---------------------
MEMORY
  /status                Show memory stats and visualization
  /clean                 Wipe all memory
  /visualize             Visual memory representation

MODELS
  /model <name>          Switch LLM model
  /embed <name>          Switch embedding model (wipes memory)

TOOLS
  /ingest <url>          Ingest GitHub repository
  /behavior <cmd>        Set AI persona (list/set <name>)
  /example <cmd>         Load template (list/load <name>)
  /infinite <cmd>        Infinite mode (on/off/set_chunks)
  /story <topic>         Generate story/world

PERFORMANCE
  /perf                  Show performance metrics

SYSTEM
  /quit                  Exit
""")
            elif user_input == "/status":
                agent.status()
            elif user_input == "/sleep":
                agent.agentic_consolidation()
            elif user_input == "/clean":
                confirm = input("WARN: Wipe all memory? (y/n): ").lower()
                if confirm == 'y':
                    agent.clear_memory()
            elif user_input == "/perf":
                if agent.perf_tracker:
                    print(agent.perf_tracker.display_stats())
                else:
                    print("WARN: Performance tracker not available")
            elif user_input == "/visualize":
                if agent.visual:
                    try:
                        cells = agent.cells.count()
                        groups = agent.groups.count()
                        entities = agent.entity_index.count()
                        print(agent.visual.memory_stats_visual(cells, groups, entities))
                        # Show group distribution
                        group_data = agent.groups.get(include=['metadatas'])
                        if group_data['ids']:
                            group_sizes = {gid: meta.get('count', 1) 
                                          for gid, meta in zip(group_data['ids'], group_data['metadatas'])}
                            print(agent.visual.group_distribution(group_sizes))
                    except Exception as e:
                        print(f"WARN: Visualization error: {e}")
                else:
                    print("WARN: Visualization not available")
            elif user_input.startswith("/model "):
                new_model = user_input[7:].strip()
                if new_model:
                    old_model = agent.config.model_name
                    agent.config.model_name = new_model
                    print(f"OK: LLM model switched: {old_model} -> {new_model}")
                else:
                    print(f"Current LLM model: {agent.config.model_name}")
            elif user_input.startswith("/embed "):
                new_embed = user_input[7:].strip()
                if new_embed:
                    print("\nWARN: Changing embedding model requires wiping all memory.")
                    print("      Embeddings from different models are incompatible.")
                    print(f"      Current: {agent.config.embed_model} -> New: {new_embed}\n")
                    confirm = input("WARN: Wipe memory and switch? (y/n): ").lower()
                    if confirm == 'y':
                        # Wipe memory first
                        agent.clear_memory()
                        # Switch encoder
                        old_embed = agent.config.embed_model
                        agent.config.embed_model = new_embed
                        agent.encoder = OllamaEncoder(model=new_embed)
                        print(f"OK: Embedding model switched: {old_embed} -> {new_embed}")
                    else:
                        print("Cancelled. Keeping current embedding model.")
                else:
                    print(f"Current embedding model: {agent.config.embed_model}")
            elif user_input == "/model" or user_input == "/embed":
                print(f"Current LLM: {agent.config.model_name}")
                print(f"Current Embedding: {agent.config.embed_model}")
            elif user_input.startswith("/ingest "):
                if 'ingest' in agent.tools:
                    url = user_input[8:].strip()
                    result = agent.tools['ingest'].execute(url=url)
                    if result['success']:
                        print(f"OK: Ingested {result['files_ingested']} files from {result['repository']}")
                    else:
                        print(f"ERROR: Ingest failed: {result.get('error', 'Unknown error')}")
                else:
                    print("WARN: Ingest tool not available")
            elif user_input.startswith("/behavior"):
                if 'behavior' in agent.tools:
                    cmd = user_input[9:].strip()
                    print(agent.tools['behavior'].execute(cmd))
                else:
                    print("WARN: Behavior tool not available")
            elif user_input.startswith("/example"):
                if 'example' in agent.tools:
                    cmd = user_input[8:].strip()
                    print(agent.tools['example'].execute(cmd))
                else:
                    print("WARN: Example tool not available")
            elif user_input.startswith("/infinite"):
                if 'infinite' in agent.tools:
                    cmd = user_input[9:].strip()
                    print(agent.tools['infinite'].execute(cmd))
                else:
                    print("WARN: Infinite tool not available")
            elif user_input.startswith("/story "):
                if HAS_STORYLINE_TOOL:
                    try:
                        topic = user_input[7:].strip()
                        story_agent = StorylineAgent(agent)
                        story_agent.execute_workflow(topic)
                    except Exception as e:
                        print(f"WARN: Story generation failed: {e}")
                else:
                    print("WARN: Story tool not available")
            elif user_input.startswith("/"):
                print(f"WARN: Unknown command: {user_input}. Type /help for available commands.")
            else:
                agent.chat(user_input)
                
    except KeyboardInterrupt:
        print("\nUsing /quit to exit...")
        agent.shutdown()

if __name__ == "__main__":
    main()
