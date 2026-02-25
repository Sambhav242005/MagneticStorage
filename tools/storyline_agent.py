import json
import time
import re
import requests
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
        self.model = self.brain.config.model_name
        self.system_prompt = "You are the 'Genesis Core'. Your job is to procedurally generate a consistent 3D world."
        
        # Initialize story consistency tracker
        if StoryConsistencyRegistry:
            self.consistency_tracker = StoryConsistencyRegistry()
            print("  ✓ Story Consistency Tracker enabled")
        else:
            self.consistency_tracker = None

    def _call_ollama(self, messages: List[Dict]) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json().get('message', {}).get('content', '')
            print(f"Ollama API Error: {response.status_code} - {response.text}")
            return ""
        except Exception as e:
            print(f"Ollama API Connect Error: {e}")
            return ""

        
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
        
        for section_idx, section in enumerate(sections):
            print(f"\n   ✍️  Phase 2: Generating Assets for '{section}'...", end="", flush=True)
            # We pass the World Config as the 'Plan' to guide the writer
            content_full, chunks, extracted_facts = self._generate_section(section, context_summary)
            print(" Done!")
            
            # 3. VERIFIER
            print(f"   🛡️  Phase 3: Verifying Physics...", end="", flush=True)
            if self._verify_consistency(content_full, context_summary, extracted_facts, section_idx):
                print(" ✅ Passed")
                
                
                # 4. COMMIT TO MEMORY (Granular)
                print(f"   💾 Saving to Memory ({len(chunks)} chunks)...", end="", flush=True)
                
                # A. Generate Summary for Master Node
                summary = self._summarize_content(content_full)
                
                # B. Save Master Node (The Section Summary)
                # B. Save Master Node (The Section Summary)
                self.brain.ingest(f"# {section} (Summary)\n{summary}\n\n[Full Content Linked in Children]")
                
                # C. Save Individual Chunks (for granular retrieval)
                for i, chunk in enumerate(chunks):
                    # We preface chunk with section name so Re-ranker knows the topic
                    chunk_text = f"## {section} (Part {i+1})\n{chunk}"
                    
                    self.brain.ingest(chunk_text)
                
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
        response_content = self._call_ollama([
            {"role": "system", "content": "Summarize this text in 3-4 dense paragraphs. Capture key entities and rules."},
            {"role": "user", "content": text[:4000]} # Limit input to avoid context overflow
        ])
        return response_content

    def _create_world_config(self, topic: str) -> Dict:
        prompt = f"""
# User Input: {topic}

# Phase 1: The Laws (Output JSON)
Generate a JSON object containing:
1. "magic_system": {{ "source": "...", "cost": "...", "hard_restriction": "..." }}
2. "biomes": [ {{ "name": "...", "visual_prompt": "...", "hazard_level": 1-10 }} ] (List of 3)
3. "civilization": {{ "settlement_style": "...", "defense_strategy": "..." }}
"""
        response_content = self._call_ollama([
            {"role": "system", "content": "You are a JSON generator. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ])
        try:
            rv = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
            if "```" in rv:
                 rv = rv.split("```")[1].replace("json", "").strip()
            return json.loads(rv)
        except:
             return {"error": "Failed to generate JSON", "raw": response_content}

    def _generate_section(self, section: str, context: str) -> Tuple[str, List[str], Optional[Dict]]:
        # Get Personas and Templates
        system_prompt = self.system_prompt
        if 'behavior' in self.brain.tools:
            system_prompt = self.brain.tools['behavior'].get_system_prompt()
            
        template_context = ""
        if 'example' in self.brain.tools:
            template_context = self.brain.tools['example'].get_context()
            
        prompt = f"Write the section '{section}'.\n\nCONTEXT:\n{context}\n{template_context}"
        
        json_prompt = prompt + """
        
IMPORTANT: Output ONLY a valid JSON object with this exact structure:
{
  "prose_text": "The actual narrative content you wrote for this section. Must be detailed and immersive.",
  "facts_extracted": {
    "characters": [{"name": "...", "physical": {"eyes": "...", "hair": "..."}, "traits": ["..."]}],
    "world_rules": {"rule_name_rule": "rule description"},
    "events": [{"id": "...", "outcome": "..."}],
    "locations": {"name": {"description": ""}}
  }
}
"""
        
        # Use Infinite Generator if available
        if 'infinite' in self.brain.tools:
            content, chunks = self.brain.tools['infinite'].generate_sequence(
                self.model, system_prompt, prompt,
                consistency_tracker=self.consistency_tracker
            )
            # We don't have facts in infinite loop directly without extra logic
            # but we can return {} so it doesn't default to regex extraction incorrectly
            return content, chunks, {}
            
        # Fallback to standard generation
        content_json = self._call_ollama([
            {"role": "system", "content": system_prompt + " You must output ONLY valid JSON."},
            {"role": "user", "content": json_prompt}
        ])

        prose_text = content_json
        extracted_facts = {}
        try:
            rv = re.sub(r'<think>.*?</think>', '', content_json, flags=re.DOTALL).strip()
            
            # Find JSON boundaries
            start_idx = rv.find('{')
            end_idx = rv.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                json_str = rv[start_idx:end_idx+1]
                data = json.loads(json_str, strict=False)
                prose_text = data.get("prose_text", prose_text)
                extracted_facts = data.get("facts_extracted", {})
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            pass

        return prose_text, [prose_text], extracted_facts

    def _verify_consistency(self, content: str, context: str, facts: Optional[Dict] = None, chunk_id: int = 0) -> bool:
        if len(content) < 50: return False
        
        if self.consistency_tracker:
            conflicts = self.consistency_tracker.process_chunk(chunk_id, content, self.model, pre_extracted_facts=facts)
            critical = [c for c in conflicts if c.severity == 'critical']
            if critical:
                return False
                
        return True
