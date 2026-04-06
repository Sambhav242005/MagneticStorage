"""
Story consistency registry.

Tracks and validates consistency across story generation chunks:
- Characters: names, traits, appearances, relationships
- Plot Events: what happened and when
- World Rules: magic systems, technology, geography
- Timeline: chronological ordering of events
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ConsistencyType(Enum):
    CHARACTER = "character"
    PLOT = "plot"
    WORLD = "world"
    TIMELINE = "timeline"


@dataclass
class Conflict:
    """Represents a detected inconsistency."""

    chunk_id: int
    conflict_type: ConsistencyType
    entity: str
    field: str
    old_value: str
    new_value: str
    severity: str = "warning"

    def __str__(self):
        return f"[{self.conflict_type.value}] {self.entity}.{self.field}: '{self.old_value}' -> '{self.new_value}'"


@dataclass
class StoryFacts:
    """Structured representation of story facts."""

    characters: Dict[str, Dict] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    world_rules: Dict[str, str] = field(default_factory=dict)
    locations: Dict[str, Dict] = field(default_factory=dict)
    timeline: List[Dict] = field(default_factory=list)


class StoryConsistencyRegistry:
    """
    Tracks story facts across generation chunks.
    Detects contradictions, drift, and inconsistencies.
    """

    NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
    EYES_PATTERN = re.compile(r"(\w+)(?:'s)?\s+(?:had|has|with|,)?\s*(\w+)\s+eyes", re.IGNORECASE)
    HAIR_PATTERN = re.compile(r"(\w+)(?:'s)?\s+(?:had|has|with|,)?\s*(\w+)\s+hair", re.IGNORECASE)
    TRAIT_PATTERN = re.compile(r"(\w+)\s+(?:was|is|seemed|appeared)\s+(\w+)", re.IGNORECASE)
    RULE_KEYWORDS = ["magic", "power", "rule", "law", "forbidden", "ancient", "sacred"]

    def __init__(self):
        self.facts = StoryFacts()
        self.history: List[Dict] = []
        self.conflicts: List[Conflict] = []

    def extract_facts(self, text: str, model: str = None) -> Optional[Dict]:
        extracted = {
            "characters": [],
            "world_rules": {},
            "events": [],
            "locations": {},
        }

        try:
            all_names = self.NAME_PATTERN.findall(text)
            name_counts = {}
            for name in all_names:
                name_lower = name.lower()
                if name_lower not in ["the", "and", "but", "she", "he", "they", "was", "were", "this", "that"]:
                    name_counts[name] = name_counts.get(name, 0) + 1

            character_names = [name for name, count in name_counts.items() if count >= 2]

            for name in character_names:
                char_data = {
                    "name": name,
                    "physical": {},
                    "traits": [],
                    "relationships": {},
                }

                for match in self.EYES_PATTERN.finditer(text):
                    if match.group(1).lower() == name.lower() or match.group(1).lower() in ["her", "his", "their"]:
                        char_data["physical"]["eyes"] = match.group(2)
                        break

                for match in self.HAIR_PATTERN.finditer(text):
                    if match.group(1).lower() == name.lower() or match.group(1).lower() in ["her", "his", "their"]:
                        char_data["physical"]["hair"] = match.group(2)
                        break

                trait_words = [
                    "brave",
                    "cowardly",
                    "kind",
                    "cruel",
                    "wise",
                    "foolish",
                    "strong",
                    "weak",
                    "honest",
                    "deceitful",
                    "calm",
                    "angry",
                    "young",
                    "old",
                    "tall",
                    "short",
                    "beautiful",
                    "handsome",
                ]

                for match in self.TRAIT_PATTERN.finditer(text):
                    subject = match.group(1).lower()
                    trait = match.group(2).lower()
                    if (subject == name.lower() or subject in ["she", "he", "they"]) and trait in trait_words:
                        if trait not in char_data["traits"]:
                            char_data["traits"].append(trait)

                extracted["characters"].append(char_data)

            sentences = re.split(r"[.!?]", text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for keyword in self.RULE_KEYWORDS:
                    if keyword in sentence_lower and len(sentence.strip()) > 20:
                        rule_key = f"{keyword}_rule"
                        if rule_key not in extracted["world_rules"]:
                            extracted["world_rules"][rule_key] = sentence.strip()[:200]
                        break

            location_pattern = re.compile(
                r"(?:in|at|to)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                re.IGNORECASE,
            )
            for match in location_pattern.finditer(text):
                location = match.group(1)
                if location.lower() not in ["the", "a", "an"] and len(location) > 2:
                    extracted["locations"][location] = {"description": "", "features": []}

            return extracted
        except Exception as exc:
            print(f"   Extraction error: {exc}")
            return None

    def validate_characters(self, new_chars: List[Dict], chunk_id: int) -> List[Conflict]:
        conflicts = []

        for char in new_chars:
            name = char.get("name", "").lower().strip()
            if not name:
                continue

            if name in self.facts.characters:
                existing = self.facts.characters[name]

                for attr, new_val in char.get("physical", {}).items():
                    if attr in existing.get("physical", {}):
                        old_val = existing["physical"][attr]
                        if old_val.lower() != new_val.lower():
                            conflicts.append(
                                Conflict(
                                    chunk_id=chunk_id,
                                    conflict_type=ConsistencyType.CHARACTER,
                                    entity=name,
                                    field=f"physical.{attr}",
                                    old_value=old_val,
                                    new_value=new_val,
                                    severity="critical",
                                )
                            )

                existing_traits = set(trait.lower() for trait in existing.get("traits", []))
                new_traits = set(trait.lower() for trait in char.get("traits", []))
                contradictions = [
                    ("brave", "cowardly"),
                    ("kind", "cruel"),
                    ("honest", "deceitful"),
                    ("calm", "angry"),
                    ("trusting", "suspicious"),
                    ("optimistic", "pessimistic"),
                ]

                for trait_a, trait_b in contradictions:
                    if (trait_a in existing_traits and trait_b in new_traits) or (
                        trait_b in existing_traits and trait_a in new_traits
                    ):
                        conflicts.append(
                            Conflict(
                                chunk_id=chunk_id,
                                conflict_type=ConsistencyType.CHARACTER,
                                entity=name,
                                field="traits",
                                old_value=str(existing_traits),
                                new_value="added contradicting trait",
                                severity="warning",
                            )
                        )

        return conflicts

    def validate_world_rules(self, new_rules: Dict[str, str], chunk_id: int) -> List[Conflict]:
        conflicts = []

        for rule_name, new_desc in new_rules.items():
            rule_key = rule_name.lower().strip()

            if rule_key in self.facts.world_rules:
                old_desc = self.facts.world_rules[rule_key]
                old_words = set(old_desc.lower().split())
                new_words = set(new_desc.lower().split())
                overlap = len(old_words & new_words) / max(len(old_words | new_words), 1)

                if overlap < 0.3:
                    conflicts.append(
                        Conflict(
                            chunk_id=chunk_id,
                            conflict_type=ConsistencyType.WORLD,
                            entity=rule_name,
                            field="description",
                            old_value=old_desc[:100],
                            new_value=new_desc[:100],
                            severity="critical",
                        )
                    )

        return conflicts

    def validate_events(self, new_events: List[Dict], chunk_id: int) -> List[Conflict]:
        conflicts = []
        existing_outcomes = {event.get("id", ""): event.get("outcome", "") for event in self.facts.events}

        for event in new_events:
            event_id = event.get("id", "")
            if event_id and event_id in existing_outcomes:
                old_outcome = existing_outcomes[event_id]
                new_outcome = event.get("outcome", "")

                if old_outcome and new_outcome and old_outcome.lower() != new_outcome.lower():
                    conflicts.append(
                        Conflict(
                            chunk_id=chunk_id,
                            conflict_type=ConsistencyType.PLOT,
                            entity=event_id,
                            field="outcome",
                            old_value=old_outcome,
                            new_value=new_outcome,
                            severity="critical",
                        )
                    )

        return conflicts

    def process_chunk(self, chunk_id: int, text: str, model: str) -> List[Conflict]:
        print(f"   Analyzing chunk {chunk_id} for consistency...", end="", flush=True)

        extracted = self.extract_facts(text, model)
        if not extracted:
            print(" (extraction failed)")
            self.history.append(
                {
                    "chunk_id": chunk_id,
                    "raw_text": text[:500],
                    "extracted": None,
                    "conflicts": [],
                }
            )
            return []

        all_conflicts = []
        all_conflicts.extend(self.validate_characters(extracted.get("characters", []), chunk_id))
        all_conflicts.extend(self.validate_world_rules(extracted.get("world_rules", {}), chunk_id))
        all_conflicts.extend(self.validate_events(extracted.get("events", []), chunk_id))

        self._merge_facts(extracted)

        self.history.append(
            {
                "chunk_id": chunk_id,
                "raw_text": text[:500],
                "extracted": extracted,
                "conflicts": [str(conflict) for conflict in all_conflicts],
            }
        )

        self.conflicts.extend(all_conflicts)

        status = f" OK ({len(extracted.get('characters', []))} chars, {len(all_conflicts)} conflicts)"
        print(status)

        return all_conflicts

    def _merge_facts(self, extracted: Dict):
        for char in extracted.get("characters", []):
            name = char.get("name", "").lower().strip()
            if name:
                if name not in self.facts.characters:
                    self.facts.characters[name] = {"physical": {}, "traits": [], "relationships": {}}

                for key, value in char.get("physical", {}).items():
                    if key not in self.facts.characters[name]["physical"]:
                        self.facts.characters[name]["physical"][key] = value

                existing_traits = set(self.facts.characters[name]["traits"])
                existing_traits.update(char.get("traits", []))
                self.facts.characters[name]["traits"] = list(existing_traits)

                self.facts.characters[name]["relationships"].update(char.get("relationships", {}))

        for rule, desc in extracted.get("world_rules", {}).items():
            rule_key = rule.lower().strip()
            if rule_key not in self.facts.world_rules:
                self.facts.world_rules[rule_key] = desc

        existing_ids = {event.get("id") for event in self.facts.events}
        for event in extracted.get("events", []):
            if event.get("id") not in existing_ids:
                self.facts.events.append(event)

        for location, details in extracted.get("locations", {}).items():
            loc_key = location.lower().strip()
            if loc_key not in self.facts.locations:
                self.facts.locations[loc_key] = details

    def get_report(self) -> str:
        lines = [
            "",
            "=" * 60,
            "STORY CONSISTENCY REPORT",
            "=" * 60,
            "",
            f"Chunks Analyzed: {len(self.history)}",
            f"Characters Tracked: {len(self.facts.characters)}",
            f"World Rules Tracked: {len(self.facts.world_rules)}",
            f"Events Tracked: {len(self.facts.events)}",
            f"Locations Tracked: {len(self.facts.locations)}",
            "",
            f"Total Conflicts Detected: {len(self.conflicts)}",
        ]

        if self.conflicts:
            lines.append("")
            lines.append("CONFLICTS:")
            for conflict in self.conflicts:
                severity = "CRITICAL" if conflict.severity == "critical" else "WARNING"
                lines.append(f"  [{severity}] Chunk {conflict.chunk_id}: {conflict}")

        total_checks = len(self.history) * 3
        failures = len(self.conflicts)
        score = max(0, 100 - (failures / max(total_checks, 1)) * 100)

        lines.append("")
        lines.append(f"Consistency Score: {score:.1f}%")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_facts_summary(self) -> Dict:
        return {
            "characters": self.facts.characters,
            "world_rules": self.facts.world_rules,
            "events": [event.get("description", "") for event in self.facts.events],
            "locations": list(self.facts.locations.keys()),
        }
