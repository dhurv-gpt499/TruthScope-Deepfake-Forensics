"""Core module for Aegis-X."""

# Don't import early_stopping here - it causes circular imports
# Import only the registry which has no dependencies on early_stopping
from .tools.registry import get_registry, reset_registry, ToolRegistry, ToolSpec, ToolCategory

__all__ = [
    "get_registry",
    "reset_registry", 
    "ToolRegistry",
    "ToolSpec",
    "ToolCategory",
]