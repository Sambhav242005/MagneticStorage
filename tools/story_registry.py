"""
Story Consistency Registry

Tracks and validates consistency across story generation chunks:
- Characters: names, traits, appearances, relationships
- Plot Events: what happened and when  
- World Rules: magic systems, technology, geography
- Timeline: chronological ordering of events

Uses regex-based extraction (no LLM required for parsing).
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ConsistencyType(Enum):
    CHARACTER = "character"
    PLOT = "plot"
    WORLD = "world"
    TIMELINE = "timeline"


@dataclass
class Conflict:
    """Represents a detected inconsistency"""
    chunk_id: int
    conflict_type: ConsistencyType
    entity: str
    field: str
    old_value: str
    new_value: str
    severity: str = "warning"  # "warning" or "critical"
    
    def __str__(self):
        return f"[{self.conflict_type.value}] {self.entity}.{self.field}: '{self.old_value}' → '{self.new_value}'"


@dataclass 
class StoryFacts:
    """Structured representation of story facts"""
    characters: Dict[str, Dict] = field(default_factory=dict)
    # {name: {physical: {}, traits: [], relationships: {}}}
    
    events: List[Dict] = field(default_factory=list)
    # [{id, description, participants, outcome}]
    
    world_rules: Dict[str, str] = field(default_factory=dict)
    # {rule_name: rule_description}
    
    locations: Dict[str, Dict] = field(default_factory=dict)
    # {name: {description, features}}
    
    timeline: List[Dict] = field(default_factory=list)
    # [{event_id, before: [], after: []}]


class StoryConsistencyRegistry:
    """
    Tracks story facts across generation chunks.
    Detects contradictions, drift, and inconsistencies.
    """
    
    # Regex patterns for text-based extraction
    # Character names (capitalized words that appear multiple times)
    NAME_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b')
    
    # Physical descriptions: "X has/had/with [color] eyes/hair"
    EYES_PATTERN = re.compile(r'(\w+)(?:\'s)?\s+(?:had|has|with|,)?\s*(\w+)\s+eyes', re.IGNORECASE)
    HAIR_PATTERN = re.compile(r'(\w+)(?:\'s)?\s+(?:had|has|with|,)?\s*(\w+)\s+hair', re.IGNORECASE)
    
    # Traits: "X was/is [trait]" or "the [trait] X"  
    TRAIT_PATTERN = re.compile(r'(\w+)\s+(?:was|is|seemed|appeared)\s+(\w+)', re.IGNORECASE)
    
    # World rules: sentences with "magic", "power", "rule", "law"
    RULE_KEYWORDS = ['magic', 'power', 'rule', 'law', 'forbidden', 'ancient', 'sacred']

    def __init__(self):
        self.facts = StoryFacts()
        self.history: List[Dict] = []  # [{chunk_id, raw_text, extracted, conflicts}]
        self.conflicts: List[Conflict] = []
        
    def extract_facts(self, text: str, model: str = None) -> Optional[Dict]:
        """
        Extract facts using regex patterns (no LLM required).
        This is more robust than JSON parsing.
        """
        extracted = {
            "characters": [],
            "world_rules": {},
            "events": [],
            "locations": {}
        }
        
        try:
            # 1. Find potential character names (capitalized, appear 2+ times)
            all_names = self.NAME_PATTERN.findall(text)
            name_counts = {}
            for name in all_names:
                name_lower = name.lower()
                # Filter out common words
                if name_lower not in ['the', 'and', 'but', 'she', 'he', 'they', 'was', 'were', 'this', 'that']:
                    name_counts[name] = name_counts.get(name, 0) + 1
            
            # Names that appear 2+ times are likely characters
            character_names = [n for n, count in name_counts.items() if count >= 2]
            
            # 2. Extract physical attributes for each character
            for name in character_names:
                char_data = {
                    "name": name,
                    "physical": {},
                    "traits": [],
                    "relationships": {}
                }
                
                # Check for eye color
                for match in self.EYES_PATTERN.finditer(text):
                    if match.group(1).lower() == name.lower() or match.group(1).lower() in ['her', 'his', 'their']:
                        char_data["physical"]["eyes"] = match.group(2)
                        break
                
                # Check for hair color
                for match in self.HAIR_PATTERN.finditer(text):
                    if match.group(1).lower() == name.lower() or match.group(1).lower() in ['her', 'his', 'their']:
                        char_data["physical"]["hair"] = match.group(2)
                        break
                
                # Extract traits
                trait_words = ['brave', 'cowardly', 'kind', 'cruel', 'wise', 'foolish', 
                              'strong', 'weak', 'honest', 'deceitful', 'calm', 'angry',
                              'young', 'old', 'tall', 'short', 'beautiful', 'handsome']
                
                for match in self.TRAIT_PATTERN.finditer(text):
                    subj = match.group(1).lower()
                    trait = match.group(2).lower()
                    if (subj == name.lower() or subj in ['she', 'he', 'they']) and trait in trait_words:
                        if trait not in char_data["traits"]:
                            char_data["traits"].append(trait)
                
                extracted["characters"].append(char_data)
            
            # 3. Extract world rules (sentences with key words)
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for keyword in self.RULE_KEYWORDS:
                    if keyword in sentence_lower and len(sentence.strip()) > 20:
                        # Create a rule key from the keyword
                        rule_key = f"{keyword}_rule"
                        if rule_key not in extracted["world_rules"]:
                            extracted["world_rules"][rule_key] = sentence.strip()[:200]
                        break
            
            # 4. Extract locations (places mentioned with "in", "at", "the X")
            location_pattern = re.compile(r'(?:in|at|to)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE)
            for match in location_pattern.finditer(text):
                loc = match.group(1)
                if loc.lower() not in ['the', 'a', 'an'] and len(loc) > 2:
                    extracted["locations"][loc] = {"description": "", "features": []}
            
            return extracted
            
        except Exception as e:
            print(f"   ⚠️  Extraction error: {e}")
            return None
    
    def validate_characters(self, new_chars: List[Dict], chunk_id: int) -> List[Conflict]:
        """Check character consistency"""
        conflicts = []
        
        for char in new_chars:
            name = char.get("name", "").lower().strip()
            if not name:
                continue
                
            if name in self.facts.characters:
                existing = self.facts.characters[name]
                
                # Check physical attributes
                for attr, new_val in char.get("physical", {}).items():
                    if attr in existing.get("physical", {}):
                        old_val = existing["physical"][attr]
                        if old_val.lower() != new_val.lower():
                            conflicts.append(Conflict(
                                chunk_id=chunk_id,
                                conflict_type=ConsistencyType.CHARACTER,
                                entity=name,
                                field=f"physical.{attr}",
                                old_value=old_val,
                                new_value=new_val,
                                severity="critical"
                            ))
                
                # Check contradicting traits
                existing_traits = set(t.lower() for t in existing.get("traits", []))
                new_traits = set(t.lower() for t in char.get("traits", []))
                
                # Define contradicting trait pairs
                contradictions = [
                    ("brave", "cowardly"), ("kind", "cruel"), ("honest", "deceitful"),
                    ("calm", "angry"), ("trusting", "suspicious"), ("optimistic", "pessimistic")
                ]
                for t1, t2 in contradictions:
                    if (t1 in existing_traits and t2 in new_traits) or \
                       (t2 in existing_traits and t1 in new_traits):
                        conflicts.append(Conflict(
                            chunk_id=chunk_id,
                            conflict_type=ConsistencyType.CHARACTER,
                            entity=name,
                            field="traits",
                            old_value=str(existing_traits),
                            new_value=f"added contradicting trait",
                            severity="warning"
                        ))
                        
        return conflicts
    
    def validate_world_rules(self, new_rules: Dict[str, str], chunk_id: int) -> List[Conflict]:
        """Check world rule consistency"""
        conflicts = []
        
        for rule_name, new_desc in new_rules.items():
            rule_key = rule_name.lower().strip()
            
            if rule_key in self.facts.world_rules:
                old_desc = self.facts.world_rules[rule_key]
                # Simple check: if descriptions are very different
                old_words = set(old_desc.lower().split())
                new_words = set(new_desc.lower().split())
                overlap = len(old_words & new_words) / max(len(old_words | new_words), 1)
                
                if overlap < 0.3:  # Less than 30% word overlap suggests contradiction
                    conflicts.append(Conflict(
                        chunk_id=chunk_id,
                        conflict_type=ConsistencyType.WORLD,
                        entity=rule_name,
                        field="description",
                        old_value=old_desc[:100],
                        new_value=new_desc[:100],
                        severity="critical"
                    ))
                    
        return conflicts
    
    def validate_events(self, new_events: List[Dict], chunk_id: int) -> List[Conflict]:
        """Check for contradicting events"""
        conflicts = []
        
        existing_outcomes = {e.get("id", ""): e.get("outcome", "") for e in self.facts.events}
        
        for event in new_events:
            event_id = event.get("id", "")
            if event_id and event_id in existing_outcomes:
                old_outcome = existing_outcomes[event_id]
                new_outcome = event.get("outcome", "")
                
                if old_outcome and new_outcome and old_outcome.lower() != new_outcome.lower():
                    conflicts.append(Conflict(
                        chunk_id=chunk_id,
                        conflict_type=ConsistencyType.PLOT,
                        entity=event_id,
                        field="outcome",
                        old_value=old_outcome,
                        new_value=new_outcome,
                        severity="critical"
                    ))
                    
        return conflicts
    
    def process_chunk(self, chunk_id: int, text: str, model: str) -> List[Conflict]:
        """
        Main entry point: extract facts from chunk and validate consistency.
        Returns list of detected conflicts.
        """
        print(f"   📖 Analyzing chunk {chunk_id} for consistency...", end="", flush=True)
        
        # 1. Extract facts
        extracted = self.extract_facts(text, model)
        if not extracted:
            print(" (extraction failed)")
            self.history.append({
                "chunk_id": chunk_id,
                "raw_text": text[:500],
                "extracted": None,
                "conflicts": []
            })
            return []
        
        # 2. Validate against registry
        all_conflicts = []
        
        all_conflicts.extend(
            self.validate_characters(extracted.get("characters", []), chunk_id)
        )
        all_conflicts.extend(
            self.validate_world_rules(extracted.get("world_rules", {}), chunk_id)
        )
        all_conflicts.extend(
            self.validate_events(extracted.get("events", []), chunk_id)
        )
        
        # 3. Commit new facts to registry (even if conflicts, we track everything)
        self._merge_facts(extracted)
        
        # 4. Record history
        self.history.append({
            "chunk_id": chunk_id,
            "raw_text": text[:500],
            "extracted": extracted,
            "conflicts": [str(c) for c in all_conflicts]
        })
        
        self.conflicts.extend(all_conflicts)
        
        status = f" ✅ ({len(extracted.get('characters', []))} chars, {len(all_conflicts)} conflicts)"
        print(status)
        
        return all_conflicts
    
    def _merge_facts(self, extracted: Dict):
        """Merge new facts into the registry"""
        # Characters
        for char in extracted.get("characters", []):
            name = char.get("name", "").lower().strip()
            if name:
                if name not in self.facts.characters:
                    self.facts.characters[name] = {"physical": {}, "traits": [], "relationships": {}}
                
                # Merge physical (first value wins)
                for k, v in char.get("physical", {}).items():
                    if k not in self.facts.characters[name]["physical"]:
                        self.facts.characters[name]["physical"][k] = v
                
                # Merge traits (accumulate)
                existing_traits = set(self.facts.characters[name]["traits"])
                existing_traits.update(char.get("traits", []))
                self.facts.characters[name]["traits"] = list(existing_traits)
                
                # Merge relationships
                self.facts.characters[name]["relationships"].update(
                    char.get("relationships", {})
                )
        
        # World rules
        for rule, desc in extracted.get("world_rules", {}).items():
            rule_key = rule.lower().strip()
            if rule_key not in self.facts.world_rules:
                self.facts.world_rules[rule_key] = desc
        
        # Events
        existing_ids = {e.get("id") for e in self.facts.events}
        for event in extracted.get("events", []):
            if event.get("id") not in existing_ids:
                self.facts.events.append(event)
        
        # Locations
        for loc, details in extracted.get("locations", {}).items():
            loc_key = loc.lower().strip()
            if loc_key not in self.facts.locations:
                self.facts.locations[loc_key] = details
    
    def get_report(self) -> str:
        """Generate a human-readable consistency report"""
        lines = [
            "",
            "=" * 60,
            "📊 STORY CONSISTENCY REPORT",
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
            for c in self.conflicts:
                severity_icon = "🔴" if c.severity == "critical" else "🟡"
                lines.append(f"  {severity_icon} Chunk {c.chunk_id}: {c}")
        
        # Calculate consistency score
        total_checks = len(self.history) * 3  # 3 validation types per chunk
        failures = len(self.conflicts)
        score = max(0, 100 - (failures / max(total_checks, 1)) * 100)
        
        lines.append("")
        lines.append(f"Consistency Score: {score:.1f}%")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_facts_summary(self) -> Dict:
        """Return structured summary of all tracked facts"""
        return {
            "characters": self.facts.characters,
            "world_rules": self.facts.world_rules,
            "events": [e.get("description", "") for e in self.facts.events],
            "locations": list(self.facts.locations.keys())
        }
