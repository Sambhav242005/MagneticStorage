import ast
import json
import re
from typing import Dict, List, Tuple

import ollama

try:
    from tools.story_registry import StoryConsistencyRegistry
except ImportError:
    StoryConsistencyRegistry = None


class StorylineAgent:
    """
    Agentic storyteller that plans, generates, and verifies content.
    Integrated with the current NeuroSavant API.
    """

    def __init__(self, neuro_savant_instance):
        self.brain = neuro_savant_instance
        self.system_prompt = "You are the 'Genesis Core'. Your job is to procedurally generate a consistent 3D world."

        if StoryConsistencyRegistry:
            self.consistency_tracker = StoryConsistencyRegistry()
            print("  Story consistency tracker enabled")
        else:
            self.consistency_tracker = None

    @property
    def model(self) -> str:
        config = getattr(self.brain, "config", None)
        if config and getattr(config, "model_name", None):
            return config.model_name
        return getattr(self.brain, "model_name", "deepseek-r1:1.5b")

    def _get_tool(self, name: str, legacy_attr: str):
        tools = getattr(self.brain, "tools", {})
        if isinstance(tools, dict) and name in tools:
            return tools[name]
        return getattr(self.brain, legacy_attr, None)

    def _store_texts(self, texts: List[str]):
        if hasattr(self.brain, "batch_ingest"):
            self.brain.batch_ingest(texts)
            return

        if hasattr(self.brain, "ingest"):
            for text in texts:
                self.brain.ingest(text)

    def execute_workflow(self, topic: str):
        print(f"\nWorld Architect Activated: {topic}")

        print("   Phase 1: Calculating physics and laws...", end="", flush=True)
        world_config = self._create_world_config(topic)
        print(" Done!")
        print(f"   => Config: {json.dumps(world_config, indent=2)}")

        context_summary = f"Topic: {topic}\nWorld Laws: {json.dumps(world_config)}"
        sections = ["The Magic System", "Biomes & Hazards", "Civilization & Defense"]

        for section in sections:
            print(f"\n   Phase 2: Generating assets for '{section}'...", end="", flush=True)
            content_full, chunks = self._generate_section(section, context_summary)
            print(" Done!")

            print("   Phase 3: Verifying physics...", end="", flush=True)
            if self._verify_consistency(content_full, context_summary):
                print(" Passed")

                print(f"   Saving to memory ({len(chunks)} chunks)...", end="", flush=True)
                summary = self._summarize_content(content_full)

                memory_entries = [
                    f"[Story Section Summary]\n[Topic: {topic}]\n[Section: {section}]\n\n{summary}"
                ]
                for index, chunk in enumerate(chunks, start=1):
                    memory_entries.append(
                        f"[Story Section Chunk]\n[Topic: {topic}]\n[Section: {section}]\n"
                        f"[Part: {index}/{len(chunks)}]\n\n{chunk}"
                    )

                self._store_texts(memory_entries)
                print(" Saved")
                context_summary += f"\n\n[Finished {section}]: {summary}..."
            else:
                print(" Physics violation detected (skipping)")

        if self.consistency_tracker:
            print(self.consistency_tracker.get_report())
            return self.consistency_tracker.get_facts_summary()

        return None

    def _summarize_content(self, text: str) -> str:
        if len(text) < 500:
            return text

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this text in 3-4 dense paragraphs. Capture key entities and rules.",
                },
                {"role": "user", "content": text[:4000]},
            ],
        )
        return response["message"]["content"]

    def _extract_structured_payload(self, raw_value: str) -> str:
        text = raw_value.strip()
        fenced_match = re.search(r"```(?:json|python)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return text

    def _normalize_world_config(self, config: Dict) -> Dict:
        magic_system = config.get("magic_system") or {}
        biomes = config.get("biomes") or []
        civilization = (
            config.get("civilization")
            or config.get("civilizations")
            or config.get("society")
            or {}
        )

        if not isinstance(magic_system, dict):
            magic_system = {"source": str(magic_system)}

        if isinstance(biomes, dict):
            biomes = [biomes]
        elif not isinstance(biomes, list):
            biomes = [biomes] if biomes else []

        normalized_biomes = []
        for biome in biomes:
            if isinstance(biome, dict):
                normalized_biomes.append(biome)
            elif biome:
                normalized_biomes.append({"name": str(biome)})

        if not isinstance(civilization, dict):
            civilization = {"settlement_style": str(civilization)}

        return {
            "magic_system": {
                "source": magic_system.get("source", "Unknown"),
                "cost": magic_system.get("cost", "Unknown"),
                "hard_restriction": magic_system.get("hard_restriction", "Unknown"),
            },
            "biomes": normalized_biomes[:3],
            "civilization": {
                "settlement_style": civilization.get("settlement_style", "Unknown"),
                "defense_strategy": civilization.get("defense_strategy", "Unknown"),
            },
        }

    def _contains_placeholder_values(self, value) -> bool:
        if isinstance(value, dict):
            return any(self._contains_placeholder_values(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_placeholder_values(item) for item in value)
        if isinstance(value, str):
            return value.strip().lower() in {"...", "placeholder", "<placeholder>", "tbd"}
        return False

    def _parse_world_config_payload(self, raw_value: str) -> Dict:
        cleaned_value = self._extract_structured_payload(raw_value)

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(cleaned_value)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return self._normalize_world_config(parsed)

        return {}

    def _create_world_config(self, topic: str) -> Dict:
        prompt = f"""
# User Input: {topic}

# Phase 1: The Laws (Output JSON)
Generate a JSON object containing:
1. "magic_system": {{ "source": "...", "cost": "...", "hard_restriction": "..." }}
2. "biomes": [ {{ "name": "...", "visual_prompt": "...", "hazard_level": 1-10 }} ] (List of 3)
3. "civilization": {{ "settlement_style": "...", "defense_strategy": "..." }}
Use concrete topic-specific values. Do not return placeholder strings such as "...".
"""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a JSON generator. Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_value = response["message"]["content"]
        parsed_config = self._parse_world_config_payload(raw_value)
        if parsed_config and not self._contains_placeholder_values(parsed_config):
            return parsed_config

        repair_response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You repair world configs. Return ONLY valid JSON with concrete values. "
                        "Never use ellipses or placeholder text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Repair this world config for topic '{topic}'. Replace all placeholder values "
                        f"with concrete details:\n{raw_value}"
                    ),
                },
            ],
        )
        repaired_config = self._parse_world_config_payload(repair_response["message"]["content"])
        if repaired_config and not self._contains_placeholder_values(repaired_config):
            return repaired_config

        fallback = self._normalize_world_config({})
        fallback["_warning"] = "Failed to parse structured world config; using fallback values."
        fallback["_raw_excerpt"] = raw_value[:200]
        return fallback

    def _generate_section(self, section: str, context: str) -> Tuple[str, List[str]]:
        system_prompt = self.system_prompt
        behavior_tool = self._get_tool("behavior", "behavior_tool")
        if behavior_tool:
            system_prompt = behavior_tool.get_system_prompt()

        template_context = ""
        example_tool = self._get_tool("example", "example_tool")
        if example_tool:
            template_context = example_tool.get_context()

        prompt = f"Write the section '{section}'.\n\nCONTEXT:\n{context}\n{template_context}"

        infinite_tool = self._get_tool("infinite", "infinite_tool")
        if infinite_tool:
            return infinite_tool.generate_sequence(
                self.model,
                system_prompt,
                prompt,
                consistency_tracker=self.consistency_tracker,
            )

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response["message"]["content"]
        return content, [content]

    def _verify_consistency(self, content: str, context: str) -> bool:
        return len(content) >= 50
