"""Abstract base class for all Aegis-X forensic tools.

This module defines the structural requirements and execution wrapper required
to ensure smooth integration into the application's overall orchestration engine.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from core.data_types import ToolResult
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseForensicTool(ABC):
    """Abstract base class that all specialized forensic tools must extend."""
    
    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Returns the internal name of the tool."""
        pass
        
    @abstractmethod
    def setup(self) -> None:
        """Initialize models, verify paths, and prepare for inference."""
        pass
        
    @abstractmethod
    def _run_inference(self, input_data: Any) -> ToolResult:
        """Core algorithmic implementation of the tool.
        
        Args:
            input_data: The media data to be processed (image arrays, paths, etc.)
            
        Returns:
            ToolResult: The detailed findings of the tool.
        """
        pass
        
    def execute(self, input_data: Any) -> ToolResult:
        """Execution wrapper that safely triggers inference and captures metrics.
        
        This method acts as a mandatory firewall, guaranteeing that individual
        tool crashes do not crash the overarching Aegis-X application.
        
        Args:
            input_data: The media data to be processed.
            
        Returns:
            ToolResult: The findings of the tool or an empty Abstention ToolResult.
        """
        start_time = time.time()
        try:
            result = self._run_inference(input_data)
            
            # Ensure execution_time is correctly stamped if omitted or empty
            if not result.execution_time:
                result.execution_time = time.time() - start_time
                
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Tool {self.tool_name} crashed during execution: {error_msg}")
            
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg=error_msg,
                execution_time=execution_time,
                evidence_summary=f"Tool {self.tool_name} failed: {error_msg}"
            )
