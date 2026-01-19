class ExampleTool:
    """
    Manages few-shot examples and templates to guide generation.
    Usage: /example load <template_name>
    """
    def __init__(self):
        self.templates = {
            "hero_journey": "1. Call to Adventure\n2. Refusal of Call\n3. Meeting Mentor...",
            "noir": "It was a raining night. Dimensions: 1920x1080...",
            "technical": "## Architecture\n- Components\n- Data Flow..."
        }
        self.current_template = None

    def execute(self, command: str) -> str:
        parts = command.split()
        if not parts:
            return f"Current template: {self.current_template}. Available: {list(self.templates.keys())}"
        
        action = parts[0]
        if action == "load" and len(parts) > 1:
            name = parts[1]
            if name in self.templates:
                self.current_template = self.templates[name]
                return f"Loaded template: {name}"
            return "Template not found."
        elif action == "list":
            return f"Available: {list(self.templates.keys())}"
            
        return "Unknown command. Use: list, load <name>"

    def get_context(self) -> str:
        if self.current_template:
            return f"\n[STYLE TEMPLATE]:\n{self.current_template}\n"
        return ""
