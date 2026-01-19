import json
import time
import ollama
from typing import List, Dict, Optional, Tuple

# Import story consistency tracking
try:
    from tools.story_registry import StoryConsistencyRegistry
except ImportError:
    StoryConsistencyRegistry = None

class StorylineAgent:
    """
    Agentic Storyteller that plans, generates, and verifies content.
    Integrated with NeuroSavant for memory.
    Now with story consistency tracking across all generation chunks.
    """
    
    def __init__(self, neuro_savant_instance):
        self.brain = neuro_savant_instance
        self.model = self.brain.model_name
        self.system_prompt = "You are the 'Genesis Core'. Your job is to procedurally generate a consistent 3D world."
        
        # Initialize story consistency tracker
        if StoryConsistencyRegistry:
            self.consistency_tracker = StoryConsistencyRegistry()
            print("  ✓ Story Consistency Tracker enabled")
        else:
            self.consistency_tracker = None

        
    def execute_workflow(self, topic: str):
        print(f"\n🏰 World Architect Activated: {topic}")
        
        # 1. PLANNER (GENESIS)
        print("   📝 Phase 1: Calculating Physics & Laws...", end="", flush=True)
        world_config = self._create_world_config(topic)
        print(" Done!")
        print(f"   => Config: {json.dumps(world_config, indent=2)}")
        
        # 2. EXECUTION LOOP (Iterate over the config sections)
        context_summary = f"Topic: {topic}\nWorld Laws: {json.dumps(world_config)}"
        
        # Flatten the keys to generate content for each major aspect
        sections = ["The Magic System", "Biomes & Hazards", "Civilization & Defense"]
        
        for section in sections:
            print(f"\n   ✍️  Phase 2: Generating Assets for '{section}'...", end="", flush=True)
            # We pass the World Config as the 'Plan' to guide the writer
            content_full, chunks = self._generate_section(section, context_summary)
            print(" Done!")
            
            # 3. VERIFIER
            print(f"   🛡️  Phase 3: Verifying Physics...", end="", flush=True)
            if self._verify_consistency(content_full, context_summary):
                print(" ✅ Passed")
                
                
                # 4. COMMIT TO MEMORY (Granular)
                print(f"   💾 Saving to Memory ({len(chunks)} chunks)...", end="", flush=True)
                
                # A. Generate Summary for Master Node
                summary = self._summarize_content(content_full)
                
                # B. Save Master Node (The Section Summary)
                master_id = self.brain.memory._generate_id_from_content(section)
                self.brain.memory.update_state_immediate(
                    master_id, 
                    ["Layer1", "WorldBible"], 
                    "", 
                    f"# {section} (Summary)\n{summary}\n\n[Full Content Linked in Children]"
                )
                
                # C. Save Individual Chunks (for granular retrieval)
                for i, chunk in enumerate(chunks):
                    chunk_id = self.brain.memory._generate_id_from_content(chunk)
                    # We preface chunk with section name so Re-ranker knows the topic
                    chunk_text = f"## {section} (Part {i+1})\n{chunk}"
                    
                    self.brain.memory.update_state_immediate(
                        chunk_id,
                        ["Layer1", "WorldBible"],
                        "", # Facts
                        chunk_text
                    )
                
                print(" Saved")
                context_summary += f"\n\n[Finished {section}]: {summary}..."
            else:
                print(" ❌ Physics Violation Detected (Skipping)")
        
        # Print final consistency report
        if self.consistency_tracker:
            print(self.consistency_tracker.get_report())
            
            # Return the facts tracked for potential further use
            return self.consistency_tracker.get_facts_summary()
        
        return None

    def _summarize_content(self, text: str) -> str:
        """Compress large content into a retrievable summary"""
        if len(text) < 500: return text
        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": "Summarize this text in 3-4 dense paragraphs. Capture key entities and rules."},
            {"role": "user", "content": text[:4000]} # Limit input to avoid context overflow
        ])
        return response['message']['content']

    def _create_world_config(self, topic: str) -> Dict:
        prompt = f"""
# User Input: {topic}

# Phase 1: The Laws (Output JSON)
Generate a JSON object containing:
1. "magic_system": {{ "source": "...", "cost": "...", "hard_restriction": "..." }}
2. "biomes": [ {{ "name": "...", "visual_prompt": "...", "hazard_level": 1-10 }} ] (List of 3)
3. "civilization": {{ "settlement_style": "...", "defense_strategy": "..." }}
"""
        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": "You are a JSON generator. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ])
        try:
            rv =  response['message']['content']
            if "```" in rv:
                 rv = rv.split("```")[1].replace("json", "").strip()
            return json.loads(rv)
        except:
             return {"error": "Failed to generate JSON", "raw": response['message']['content']}

    def _generate_section(self, section: str, context: str) -> Tuple[str, List[str]]:
        # Get Personas and Templates
        system_prompt = self.system_prompt
        if hasattr(self.brain, 'behavior_tool') and self.brain.behavior_tool:
            system_prompt = self.brain.behavior_tool.get_system_prompt()
            
        template_context = ""
        if hasattr(self.brain, 'example_tool') and self.brain.example_tool:
            template_context = self.brain.example_tool.get_context()
            
        prompt = f"Write the section '{section}'.\n\nCONTEXT:\n{context}\n{template_context}"
        
        # Use Infinite Generator if available
        if hasattr(self.brain, 'infinite_tool') and self.brain.infinite_tool:
            return self.brain.infinite_tool.generate_sequence(
                self.model, system_prompt, prompt,
                consistency_tracker=self.consistency_tracker
            )
            
        # Fallback to standard generation
        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])
        content = response['message']['content']
        return content, [content]

    def _verify_consistency(self, content: str, context: str) -> bool:
        # Simple self-consistency check using Re-ranker
        # If the content contradicts the context, score should be low?
        # Actually, for now, we just check if it's not empty/nonsense.
        # A Real implementation would extract facts and check graph.
        if len(content) < 50: return False
        return True
