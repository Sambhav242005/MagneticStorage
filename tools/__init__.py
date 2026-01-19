"""
Neuro-Savant Tools Framework

Tools are modular extensions that can be added to Neuro-Savant.
Each tool should inherit from BaseTool and implement the execute() method.

Usage:
    from tools import load_tools
    tools = load_tools(memory_grid)
    tools['ingest'].execute(url="https://github.com/user/repo")
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import importlib
import glob


class BaseTool(ABC):
    """Base class for all tools"""
    
    name: str = "base_tool"
    description: str = "Base tool class"
    command: str = "/tool"  # CLI command to invoke this tool
    
    def __init__(self, memory_grid=None):
        """
        Initialize tool with optional memory grid reference
        
        Args:
            memory_grid: HierarchicalLiquidGrid instance for storing data
        """
        self.memory_grid = memory_grid
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool's main functionality
        
        Returns:
            Dict with 'success' bool and 'message' or 'error' string
        """
        pass
    
    def help(self) -> str:
        """Return help text for this tool"""
        return f"{self.name}: {self.description}\n  Usage: {self.command}"


def load_tools(memory_grid=None) -> Dict[str, BaseTool]:
    """
    Auto-discover and load all tools from the tools/ directory
    
    Args:
        memory_grid: HierarchicalLiquidGrid instance to pass to tools
        
    Returns:
        Dict mapping tool names to tool instances
    """
    tools = {}
    tools_dir = os.path.dirname(__file__)
    
    # Find all Python files in tools/ (except __init__.py)
    for filepath in glob.glob(os.path.join(tools_dir, "*.py")):
        filename = os.path.basename(filepath)
        if filename.startswith("_"):
            continue
        
        module_name = filename[:-3]  # Remove .py
        
        try:
            # Import the module
            module = importlib.import_module(f"tools.{module_name}")
            
            # Find tool classes (subclasses of BaseTool)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseTool) and 
                    attr is not BaseTool):
                    
                    # Instantiate and register
                    tool_instance = attr(memory_grid)
                    tools[tool_instance.name] = tool_instance
                    print(f"  ✓ Loaded tool: {tool_instance.name}")
                    
        except Exception as e:
            print(f"  ⚠️  Failed to load tool from {filename}: {e}")
    
    return tools


def get_tool_commands(tools: Dict[str, BaseTool]) -> Dict[str, BaseTool]:
    """Map CLI commands to tools"""
    return {tool.command: tool for tool in tools.values()}
