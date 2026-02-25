

import json
import re
import requests
from typing import Tuple, List, Optional, Any

class InfiniteLoopTool:
    """
    Enables 'Infinite' generation mode (Rolling Context Loop) with:
    1. Planning Phase (Task breakdown)
    2. Dynamic Memory Access (Querying NeuroSavant)
    3. Auto-Save (Ingesting chunks)
    
    Usage: /infinite on | off
    """
    def __init__(self, memory_client: Any = None):
        self.active = False
        self.chunk_limit = 10 
        self.memory_client = memory_client # Reference to NeuroSavant instance
        self.api_url = "http://localhost:11434/api/chat"
        
    def _chat(self, model: str, messages: List[dict]) -> str:
        """Helper to call Ollama API directly via requests"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '')
        except Exception as e:
            print(f"   ⚠️  Ollama API error: {e}")
            return ""

    def execute(self, command: str) -> str:
        parts = command.split()
        if not parts:
            return f"Infinite Mode: {'ON' if self.active else 'OFF'}. Chunks: {self.chunk_limit}"
        
        action = parts[0].lower()
        if action == "on":
            self.active = True
            return "♾️  Infinite Generation: ENABLED (Planning + Memory Access)"
        elif action == "off":
            self.active = False
            return "Generations restricted to single-shot."
        elif action == "set_chunks":
            if len(parts) > 1 and parts[1].isdigit():
                self.chunk_limit = int(parts[1])
                return f"Chunk limit set to {self.chunk_limit}"
                
        return "Usage: /infinite on | off | set_chunks <N>"

    def generate_plan(self, model_name: str, user_prompt: str) -> List[dict]:
        """
        Generates a structured plan/todo list for the content.
        """
        print("\n📝 Generating Plan...")
        
        system_prompt = """
        You are an expert story architect and planner. 
        Your goal is to break down the user's request into a series of logical, detailed chunks (steps).
        
        Adapt the granularity and estimated length (e.g. "Chapter 1", "800 words") solely based on the user's request.
        
        Return a JSON array of objects, where each object has:
        - "step_id": integer
        - "title": string (short title)
        - "description": string (detailed instructions on what to write in this chunk)
        - "estimated_length": string (e.g. "800 words", "detailed scene")
        - "search_query": string (a search query to find relevant info in memory database. Empty if none needed.)
        
        Example:
        [
            {"step_id": 1, "title": "Chapter 1: The Call", "description": "Introduce the protagonist in their normal life, then disrupt it.", "estimated_length": "800 words", "search_query": "Protagonist backstory"},
            {"step_id": 2, "title": "Chapter 1: The Departure", "description": "The protagonist leaves home. Describe the environment in detail.", "estimated_length": "1000 words", "search_query": "Setting details"}
        ]
        """
        
        content = self._chat(model_name, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a detailed plan for: {user_prompt}"}
        ])
        
        if not content:
             print("   ⚠️  Planning failed: No response.")
             return []

        # extract json
        try:
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))
                print(f"   ✅ Plan created with {len(plan)} steps.")
                for step in plan:
                    print(f"      - {step.get('step_id')}. {step.get('title')} (Query: {step.get('search_query', 'None')})")
                return plan
            else:
                print("   ⚠️  Failed to parse plan as JSON. Falling back to linear generation.")
                return []
        except Exception as e:
            print(f"   ⚠️  Planning parsing failed: {e}")
            return []

    def generate_sequence(self, model_name: str, system_prompt: str, user_prompt: str, 
                          memory_check_fn=None, consistency_tracker=None) -> Tuple[str, List[str]]:
        """
        Executes the Infinite Loop generation.
        """
        if not self.active:
            # Fallback to single shot if somehow called while inactive
            return "Infinite mode is OFF.", []

        # 1. Generate Plan
        plan = self.generate_plan(model_name, user_prompt)
        
        # If plan failed or empty, fallback to simple chunks
        if not plan:
            # Create a dummy plan
            plan = [
                {"step_id": 1, "title": "Content", "description": "Write the response.", "estimated_length": "full", "search_query": user_prompt}
            ]

        chunks = []
        full_text_buffer = ""
        
        # Initial Context
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        print("\n🚀 Starting Infinite Generation Sequence...")
        
        for step in plan:
            step_id = step.get('step_id')
            title = step.get('title')
            desc = step.get('description')
            query = step.get('search_query')
            
            print(f"\n   👉 Step {step_id}: {title}")
            
            # 2. Retrieve Context (if configured)
            memory_context = ""
            if query and self.memory_client:
                # Handle if planner returns a list of queries
                if isinstance(query, list):
                    query = " ".join(query)
                    
                print(f"   🔍 Querying Memory: '{query}'...", end="", flush=True)
                try:
                    # We access the memory_client's query method
                    memory_context = self.memory_client.query(query)
                    if not memory_context or "No memory found" in memory_context:
                         print(" No results.")
                         memory_context = ""
                    else:
                         print(" Found context.")
                except Exception as e:
                    print(f" Error: {e}")

            # 3. Construct Prompt for this Chunk
            step_prompt = f"""
            CURRENT OBJECTIVE: Step {step_id} - {title}
            TARGET LENGTH: {step.get('estimated_length', 'Detailed')}
            INSTRUCTIONS: {desc}
            
            CONTEXT FROM MEMORY (GROUND TRUTH):
            {memory_context[:2000] if memory_context else "No external context."}
            
            PREVIOUS CONTENT SUMMARY:
            {full_text_buffer[-500:] if full_text_buffer else "Start of content."}
            
            Write this section now. 
            CRITICAL: 
            1. Write fully developed content. Do not summarize. Use dialogue, sensory details, and deep description.
            2. The CONTEXT FROM MEMORY is the absolute truth. Do not contradict it. Use it to ground your writing.
            
            Focus ONLY on this step, but ensure it flows from the previous text.
            """
            
            # 4. Generate
            try:
                generation_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original Request: {user_prompt}"},
                    {"role": "assistant", "content": f"I have written:\n\n{full_text_buffer}" if full_text_buffer else "I am ready to start."},
                    {"role": "user", "content": step_prompt}
                ]
                
                chunk_content = self._chat(model_name, generation_messages)
                
                if chunk_content:
                    print(f"\n{'-'*40}\n{chunk_content}\n{'-'*40}")
                    chunks.append(chunk_content)
                    full_text_buffer += f"\n\n## {title}\n{chunk_content}"
                    
                    # 5. Ingest/Store (Auto-Save)
                    if self.memory_client:
                        print("   💾 Saving chunk to memory...", end="", flush=True)
                        try:
                            self.memory_client.ingest(chunk_content)
                            print(" Done.")
                        except Exception as e:
                            print(f" Failed: {e}")
                else:
                    print("\n   ⚠️  Empty response for chunk.")
                    
            except Exception as e:
                print(f"\n   ❌ Error generating chunk: {e}")
                break
                
        print("\n✅ Generation Complete.")
        return full_text_buffer, chunks
