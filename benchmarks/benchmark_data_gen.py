
import random
import json
import os

class DataGenerator:
    """
    Generates a massive dataset with 'needles' (specific facts) hidden inside.
    Simulates a '1 Million Token' load by repeating dense lore/technical text.
    """
    
    def __init__(self, output_dir="benchmark_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Filler text (Haystack)
        self.filler_templates = [
            "The flux capacitor requires a gigawatt input of exactly 1.21 to stabilize the temporal field. ",
            "Sector 7G has reported a variance in the neutrino detection grid, suggesting subspace interference. ",
            "The ancient scrolls of Azarath describe a darkness that consumes not just light, but hope itself. ",
            "System diagnostics indicate a 98% efficiency in the warp core, but the magnetic containment is fluctuating. ",
            "The economics of the neo-market rely heavily on the arbitrage of quantum credits across timelines. "
        ]
        
        # Needles (Facts to Retrieve)
        self.needles = [
            {"question": "What is the Omega Protocol password?", "answer": "Azure-99", "context": "The Omega Protocol password is 'Azure-99'."},
            {"question": "Who is the traitor in Section 9?", "answer": "Officer K", "context": "Intelligence confirms that Officer K is the traitor in Section 9."},
            {"question": "What is the frequency of the Ghost Signal?", "answer": "442.8 MHz", "context": "The Ghost Signal broadcasts on a frequency of 442.8 MHz."},
            {"question": "Where is the hidden rebel base?", "answer": "Moon of Endor", "context": "The hidden rebel base is located on the forest Moon of Endor."},
            {"question": "What kills the Night King?", "answer": "Valyrian Steel", "context": "Only Valyrian Steel or Dragonglass can kill the Night King."}
        ]

    def generate(self, size_mb=1):
        """
        Generates dataset.txt and needles.json
        size_mb: Approximate size in Megabytes of the haystack
        """
        print(f"Generizing {size_mb}MB dataset...")
        
        full_text = []
        target_bytes = size_mb * 1024 * 1024
        current_bytes = 0
        
        # 1. Generate Haystack
        while current_bytes < target_bytes:
            # Create a "Block" of varied text to avoid simple compression tricks
            block = ""
            for _ in range(50):
                block += random.choice(self.filler_templates)
            
            full_text.append(block)
            current_bytes += len(block)
            
        print(f"Haystack generated: {current_bytes} bytes")
        
        # 2. Inject Needles at specific depths
        # We ensure they are far apart
        total_blocks = len(full_text)
        injection_points = [
            int(total_blocks * 0.1),  # 10% depth
            int(total_blocks * 0.3),  # 30% depth
            int(total_blocks * 0.5),  # 50% depth
            int(total_blocks * 0.7),  # 70% depth
            int(total_blocks * 0.9)   # 90% depth
        ]
        
        # Shuffle needles to randomize which one goes where
        random.shuffle(self.needles)
        active_needles = self.needles[:5] # Take top 5
        
        ground_truth = []
        
        for i, point in enumerate(injection_points):
            if i >= len(active_needles): break
            needle = active_needles[i]
            
            # Inject
            # We add unique ID to context to make it truly unique if lines repeat
            unique_context = f"\n[CONFIDENTIAL LOG {random.randint(1000,9999)}]: {needle['context']}\n"
            full_text[point] += unique_context
            
            ground_truth.append({
                "id": i,
                "question": needle['question'],
                "answer": needle['answer'],
                "depth_percent": (point / total_blocks) * 100
            })
            
        # 3. Save
        dataset_path = os.path.join(self.output_dir, "dataset.txt")
        json_path = os.path.join(self.output_dir, "needles.json")
        
        with open(dataset_path, "w") as f:
            f.write("\n".join(full_text))
            
        with open(json_path, "w") as f:
            json.dump(ground_truth, f, indent=2)
            
        print(f"✅ Generated: {dataset_path} ({len(full_text)} blocks)")
        print(f"✅ Needles: {json_path} ({len(ground_truth)} injected)")
        return dataset_path, json_path

if __name__ == "__main__":
    # Test run
    gen = DataGenerator()
    gen.generate(size_mb=0.5) # Small test
