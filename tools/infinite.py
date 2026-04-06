from typing import List, Tuple

import ollama

from tools import BaseTool


class InfiniteLoopTool(BaseTool):
    """
    Enables infinite generation mode with a rolling context loop.
    Usage: /infinite on | off | set_chunks <n>
    """

    name = "infinite"
    description = "Chain multiple model generations with a rolling context window"
    command = "/infinite"

    def __init__(self, memory_grid=None):
        super().__init__(memory_grid)
        self.active = False
        self.chunk_limit = 5

    def execute(self, command: str = "", **kwargs) -> str:
        parts = command.split()
        if not parts:
            return f"Infinite Mode: {'ON' if self.active else 'OFF'}. Chunks: {self.chunk_limit}"

        action = parts[0].lower()
        if action == "on":
            self.active = True
            return "Infinite Generation: ENABLED (will chain outputs)"
        if action == "off":
            self.active = False
            return "Generations restricted to single-shot."
        if action == "set_chunks" and len(parts) > 1 and parts[1].isdigit():
            self.chunk_limit = int(parts[1])
            return f"Chunk limit set to {self.chunk_limit}"

        return "Usage: /infinite on | off | set_chunks <n>"

    def generate_sequence(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        memory_check_fn=None,
        consistency_tracker=None,
    ) -> Tuple[str, List[str]]:
        """
        Generates -> (full_text, list_of_chunks)
        """
        if not self.active:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response["message"]["content"]

            if consistency_tracker:
                consistency_tracker.process_chunk(0, content, model_name)

            return content, [content]

        chunks = []
        current_context = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_text_buffer = ""
        safety_limit = max(1, self.chunk_limit)

        print(f"   Looping generation (goal-directed, max {safety_limit} chunks)...")

        for i in range(safety_limit):
            response = ollama.chat(model=model_name, messages=current_context)
            chunk = response["message"]["content"]
            chunks.append(chunk)
            full_text_buffer += "\n" + chunk

            print(f"\n\n{'=' * 60}")
            print(f"CHUNK {i + 1}")
            print("=" * 60)
            print(chunk)
            print("=" * 60)

            if consistency_tracker:
                conflicts = consistency_tracker.process_chunk(i + 1, chunk, model_name)
                critical_conflicts = [c for c in conflicts if c.severity == "critical"]
                if critical_conflicts:
                    print(f"\n   Consistency warning: {len(critical_conflicts)} critical conflicts detected")
                    for conflict in critical_conflicts[:3]:
                        print(f"      - {conflict}")

            if i > 0:
                missing = self._detect_missing_elements(model_name, full_text_buffer, memory_check_fn)
                if not missing:
                    print("\n   Supervisor: world appears complete and detailed.")
                    break

                print(f"\n   Supervisor: missing {missing}. Steering...", end="", flush=True)
                steering_prompt = (
                    "Great. The narrative is taking shape, but we are missing detailed descriptions of: "
                    f"{', '.join(missing)}. Please write the next section focusing SPECIFICALLY on fleshing "
                    "out these elements in high detail."
                )
            else:
                steering_prompt = (
                    "Continue expounding on this world. Add more specific sections on characters and geography."
                )

            current_context.append({"role": "assistant", "content": chunk})
            current_context.append({"role": "user", "content": steering_prompt})

            if len(current_context) > 5:
                current_context = [current_context[0]] + current_context[-4:]

        if consistency_tracker:
            print(consistency_tracker.get_report())

        return "\n\n".join(chunks), chunks

    def _detect_missing_elements(self, model: str, text: str, memory_check_fn=None) -> List[str]:
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
            result = response["message"]["content"].strip()

            if "COMPLETE" in result.upper():
                return []

            lower_res = result.lower()
            if "character" in lower_res:
                candidates.append("Main Characters")
            if "landscape" in lower_res or "environment" in lower_res:
                candidates.append("Landscapes")
            if "rule" in lower_res or "system" in lower_res:
                candidates.append("World Systems")
        except Exception:
            return []

        if not memory_check_fn or not candidates:
            return candidates

        real_missing = []
        for item in candidates:
            print(f" [DB Check: {item}]...", end="", flush=True)
            found_in_db = memory_check_fn(item)
            if found_in_db:
                print("Found!", end="", flush=True)
            else:
                print("Missing.", end="", flush=True)
                real_missing.append(item)

        return real_missing
