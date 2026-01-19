import ollama
from typing import Tuple, List

class InfiniteLoopTool:
    """
    Enables 'Infinite' generation mode (Rolling Context Loop).
    Allows tools to generate content exceeding standard window limits.
    Usage: /infinite on | off
    """
    def __init__(self):
        self.active = False
        self.chunk_limit = 5  # Generates up to 5 chunks by default
        
    def execute(self, command: str) -> str:
        parts = command.split()
        if not parts:
            return f"Infinite Mode: {'ON' if self.active else 'OFF'}. Chunks: {self.chunk_limit}"
        
        action = parts[0].lower()
        if action == "on":
            self.active = True
            return "♾️  Infinite Generation: ENABLED (Will chain outputs)"
        elif action == "off":
            self.active = False
            return "Generations restricted to single-shot."
        elif action == "set_chunks":
            if len(parts) > 1 and parts[1].isdigit():
                self.chunk_limit = int(parts[1])
                return f"Chunk limit set to {self.chunk_limit}"
                
        return "Usage: /infinite on | off"

    def generate_sequence(self, model_name: str, system_prompt: str, user_prompt: str, 
                          memory_check_fn=None, consistency_tracker=None) -> Tuple[str, List[str]]:
        """
        Generates -> (full_text, list_of_chunks)
        param memory_check_fn: Callable(topic) -> bool. Returns True if topic exists in DB.
        param consistency_tracker: Optional StoryConsistencyRegistry instance for tracking story consistency.
        """
        if not self.active:
            # Standard single shot
            response = ollama.chat(model=model_name, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            content = response['message']['content']
            
            # Track consistency even for single-shot
            if consistency_tracker:
                consistency_tracker.process_chunk(0, content, model_name)
                
            return content, [content]
        
        # Infinite Loop Mode
        chunks = []
        # Initial context
        current_context = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        full_text_buffer = ""
        safety_limit = 20 # Increased safety limit
        
        print(f"   ♾️  Looping generation (Goal-Directed, Max {safety_limit} chunks)...")
        
        for i in range(safety_limit):
            # 1. Generate Content
            response = ollama.chat(model=model_name, messages=current_context)
            chunk = response['message']['content']
            chunks.append(chunk)
            full_text_buffer += "\n" + chunk
            
            # Show full chunk content
            print(f"\n\n{'='*60}")
            print(f"📜 CHUNK {i+1}")
            print('='*60)
            print(chunk)
            print('='*60)
            
            # 1.5. Story Consistency Check
            if consistency_tracker:
                conflicts = consistency_tracker.process_chunk(i + 1, chunk, model_name)
                if conflicts:
                    critical_conflicts = [c for c in conflicts if c.severity == "critical"]
                    if critical_conflicts:
                        print(f"\n   ⚠️  CONSISTENCY WARNING: {len(critical_conflicts)} critical conflicts detected")
                        for c in critical_conflicts[:3]:  # Show first 3
                            print(f"      - {c}")
            
            # 2. Supervisor Check (Self-Reflection) using a separate context
            if i > 0: # Check after first chunk
                missing = self._detect_missing_elements(model_name, full_text_buffer, memory_check_fn)
                if not missing:
                    print("\n   ✅ Supervisor: World appears complete and detailed.")
                    break
                else:
                    print(f"\n   🔍 Supervisor: Missing {missing}. Steering...", end="", flush=True)
                    steering_prompt = f"Great. The narrative is taking shape, but we are missing detailed descriptions of: {', '.join(missing)}. Please write the next section focusing SPECIFICALLY on fleshing out these elements in high detail."
            else:
                missing = []
                steering_prompt = "Continue expounding on this world. Add more specific sections on characters and geography."
                
            # 3. Context Management
            current_context.append({"role": "assistant", "content": chunk})
            current_context.append({"role": "user", "content": steering_prompt})
            
            # Sliding Window (Keep system prompt + last 2 turns)
            if len(current_context) > 6:
                # [System, User_Original, ... last_assistant, last_steering]
                # Actually, keeping User_Original is good for grounding, but maybe we just keep System + Last 3 interactions
                current_context = [current_context[0]] + current_context[-4:]
        
        # Print final consistency report if tracker is active
        if consistency_tracker:
            print(consistency_tracker.get_report())
                
        return "\n\n".join(chunks), chunks


    def _detect_missing_elements(self, model: str, text: str, memory_check_fn=None) -> List[str]:
        """
        Analyzes text to see if it qualifies as a 'Complete World'.
        Returns list of missing aspects.
        Checks DB if memory_check_fn is provided.
        """
        # 1. LLM Analysis of Buffer
        prompt = f"""
        Analyze the following story/world description. 
        Does it contain DETAILED descriptions of:
        1. Main Characters (Names, appearances, personalities)
        2. Landscapes/Environments (Sensory details, geography)
        3. Rules/Systems (Magic, technology, or societal rules)
        
        TEXT:
        {text[-12000:]} 
        
        If ALL 3 are present in detail, output "COMPLETE".
        Otherwise, output a comma-separated list of what is missing (e.g. "Main Characters, Landscapes").
        Output ONLY the list or "COMPLETE".
        """
        
        candidates = []
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            result = response['message']['content'].strip()
            
            if "COMPLETE" in result.upper():
                return []
            
            lower_res = result.lower()
            if "character" in lower_res: candidates.append("Main Characters")
            if "landscape" in lower_res or "environment" in lower_res: candidates.append("Landscapes")
            if "rule" in lower_res or "system" in lower_res: candidates.append("World Systems")
        except:
            return [] 

        # 2. Memory DB Check (Deduplication)
        if not memory_check_fn or not candidates:
            return candidates
            
        real_missing = []
        for item in candidates:
            # Query the DB to see if we already know this
            # effectively asking: "Do I have Main Characters?"
            print(f" [DB Check: {item}]...", end="", flush=True)
            found_in_db = memory_check_fn(item)
            if found_in_db:
                 print("Found!", end="", flush=True)
            else:
                 print("Missing.", end="", flush=True)
                 real_missing.append(item)
                 
        return real_missing
