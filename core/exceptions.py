"""Custom exceptions for the TruthScope system.

This module defines all custom exception classes used across the project to provide
more granular error handling and reporting.
"""

class TruthScopeError(Exception):
    """Base exception class for all custom TruthScope errors."""
    pass

class ModelLoadError(TruthScopeError):
    """Raised when an AI model or weight file fails to load properly."""
    pass

class PreprocessingError(TruthScopeError):
    """Raised when an error occurs during media preprocessing or data extraction."""
    pass

class ToolExecutionError(TruthScopeError):
    """Raised when a forensic tool encounters a fatal error during its execution."""
    pass
