"""Core data types for the Aegis-X system.

This module contains the unified interface and payload contracts for all internal
forensic tools and orchestration components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass(init=False)
class ToolResult:
    """Standardized output payload from any forensic tool in Aegis-X."""
    
    tool_name: str
    success: bool
    fake_score: float
    confidence: float
    details: Dict[str, Any]
    error: bool
    error_msg: Optional[str]
    execution_time: float
    evidence_summary: str
    
    # Don't declare 'score' as field - it's a property alias for fake_score

    def __init__(
        self,
        tool_name: str,
        success: bool,
        score: Optional[float] = None,
        confidence: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        error: bool = False,
        error_msg: Optional[str] = None,
        execution_time: float = 0.0,
        evidence_summary: str = "",
        fake_score: Optional[float] = None,
        **kwargs
    ):
        self.tool_name = tool_name
        self.success = success
        
        # Handle backward compatibility: score vs fake_score
        if fake_score is not None:
            self.fake_score = fake_score
        elif score is not None:
            self.fake_score = score
        elif 'score' in kwargs:
            self.fake_score = kwargs['score']
        else:
            self.fake_score = 0.0
            
        self.confidence = confidence
        self.details = details if details is not None else {}
        self.error = error
        self.error_msg = error_msg
        self.execution_time = execution_time
        self.evidence_summary = evidence_summary

    @property
    def score(self) -> float:
        """Alias for fake_score (backward compatibility)."""
        return self.fake_score