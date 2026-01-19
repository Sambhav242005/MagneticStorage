class AgentBehaviorTool:
    """
    Manages agent persona and behavioral guidelines.
    Usage: /behavior set <persona>
    """
    def __init__(self):
        self.personas = {
            "default": "You are a helpful AI assistant.",
            "storyteller": "You are a master novelist. Use vivid imagery and show-dont-tell.",
            "critic": "You are a harsh literary critic. Find flaws in logic and pacing.",
            "coder": "You are a senior engineer. Prefer code over text."
        }
        self.current_persona = "default"

    def execute(self, command: str) -> str:
        parts = command.split()
        if not parts:
            return f"Current persona: {self.current_persona}. Available: {list(self.personas.keys())}"
            
        action = parts[0]
        if action == "set":
            if len(parts) > 1:
                name = parts[1]
                if name in self.personas:
                    self.current_persona = name
                    return f"Persona switched to: {name}"
                else:
                    # Allow custom persona
                    custom_persona = " ".join(parts[1:])
                    self.personas["custom"] = custom_persona
                    self.current_persona = "custom"
                    return f"Custom persona set: {custom_persona[:30]}..."
        elif action == "list":
            return f"Available: {list(self.personas.keys())}"
            
        return "Unknown command. Use: list, set <name/custom>"

    def get_system_prompt(self) -> str:
        return self.personas.get(self.current_persona, self.personas["default"])
