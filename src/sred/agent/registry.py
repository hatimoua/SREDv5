"""
Tool registry for the agent runner.

Each tool is registered with:
- name: unique identifier matching OpenAI function name
- description: for the LLM
- parameters: JSON Schema for the function arguments
- handler: callable(session, run_id, **kwargs) -> dict
"""
from typing import Callable, Dict, Any, List

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    handler: Callable,
):
    """Register a tool in the global registry."""
    _REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "handler": handler,
    }


def get_tool_handler(name: str) -> Callable:
    """Return the handler for a registered tool."""
    return _REGISTRY[name]["handler"]


def get_openai_tools_schema() -> List[Dict[str, Any]]:
    """Return the list of tool definitions in OpenAI function-calling format."""
    tools = []
    for entry in _REGISTRY.values():
        tools.append({
            "type": "function",
            "function": {
                "name": entry["name"],
                "description": entry["description"],
                "parameters": entry["parameters"],
            },
        })
    return tools


# Expose for convenience
TOOL_REGISTRY = _REGISTRY
