"""
Neuro-Savant tools framework.

Tools are modular extensions that can be added to Neuro-Savant.
Each tool should inherit from BaseTool and implement the execute() method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import glob
import importlib
import os


class BaseTool(ABC):
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool class"
    command: str = "/tool"

    def __init__(self, memory_grid=None):
        self.memory_grid = memory_grid

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool's main functionality."""

    def help(self) -> str:
        return f"{self.name}: {self.description}\n  Usage: {self.command}"


def load_tools(memory_grid=None) -> Dict[str, BaseTool]:
    """Auto-discover and load all BaseTool implementations from tools/."""

    tools = {}
    tools_dir = os.path.dirname(__file__)

    for filepath in glob.glob(os.path.join(tools_dir, "*.py")):
        filename = os.path.basename(filepath)
        if filename.startswith("_"):
            continue

        module_name = filename[:-3]

        try:
            module = importlib.import_module(f"tools.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseTool) and attr is not BaseTool:
                    tool_instance = attr(memory_grid)
                    tools[tool_instance.name] = tool_instance
                    print(f"  Loaded tool: {tool_instance.name}")
        except Exception as exc:
            print(f"  Failed to load tool from {filename}: {exc}")

    return tools


def get_tool_commands(tools: Dict[str, BaseTool]) -> Dict[str, BaseTool]:
    """Map CLI commands to tools."""

    return {tool.command: tool for tool in tools.values()}
