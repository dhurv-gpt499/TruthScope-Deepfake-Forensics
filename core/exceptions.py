"""Custom exceptions for the Aegis-X system.

This module defines all custom exception classes used across the project to provide
more granular error handling and reporting.
"""

class AegisError(Exception):
    """Base exception class for all custom Aegis-X errors."""
    pass

class ModelLoadError(AegisError):
    """Raised when an AI model or weight file fails to load properly."""
    pass

class PreprocessingError(AegisError):
    """Raised when an error occurs during media preprocessing or data extraction."""
    pass

class ToolExecutionError(AegisError):
    """Raised when a forensic tool encounters a fatal error during its execution."""
    pass
