"""
FreqNet Calibration Manager
---------------------------
Handles loading of frequency band baseline statistics.
Supports dual-mode: full calibration or fallback ratios.
"""

import torch
from pathlib import Path
from typing import Optional, Dict
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CalibrationManager:
    """
    Manages frequency band calibration data.
    
    Dual-mode operation:
    - FULL: Load from calibration file (Z-scores)
    - FALLBACK: Use default values (ratios only)
    """
    
    DEFAULT_CALIBRATION = {
        'mean_base': 0.70,  # Natural images: ~70% low frequency
        'std_base': 0.10,
        'mean_mid': 0.25,   # ~25% mid frequency
        'std_mid': 0.08,
        'mean_high': 0.05,  # ~5% high frequency
        'std_high': 0.03,
    }
    
    def __init__(self, calibration_path: str = "calibration/freqnet_fad_baseline.pt"):
        self.calibration_path = Path(calibration_path)
        self.calibration_data: Optional[Dict[str, float]] = None
        self.mode = "FALLBACK"
    
    def load(self) -> bool:
        """
        Attempt to load calibration data.
        
        Returns:
            True if calibration loaded successfully, False if using fallback
        """
        if self.calibration_path.exists():
            try:
                self.calibration_data = torch.load(self.calibration_path, map_location='cpu')
                
                # Validate required keys
                required_keys = ['mean_base', 'std_base', 'mean_mid', 'std_mid', 'mean_high', 'std_high']
                if all(k in self.calibration_data for k in required_keys):
                    self.mode = "FULL"
                    logger.info(f"✅ FreqNet calibration loaded from {self.calibration_path}")
                    logger.info(f"   Mode: FULL (Z-scores enabled)")
                    return True
                else:
                    logger.warning(f"Calibration file missing required keys. Using fallback.")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}. Using fallback.")
        
        # Fallback mode - still use default values
        self.calibration_data = self.DEFAULT_CALIBRATION.copy()
        self.mode = "FALLBACK"
        logger.info("⚠️  FreqNet calibration not found. Using fallback mode (ratio-based).")
        logger.info(f"   Mode: FALLBACK (Z-scores disabled)")
        logger.info(f"   To enable: run scripts/compute_fad_calibration.py")
        return False
    
    def get_data(self) -> Dict[str, float]:
        """
        Get calibration data.
        
        Returns:
            Dict with mean/std for each band (always returns data, never None)
        """
        # Always return calibration_data (either loaded or defaults)
        return self.calibration_data if self.calibration_data else self.DEFAULT_CALIBRATION.copy()
    
    def is_calibrated(self) -> bool:
        """Check if full calibration is available."""
        return self.mode == "FULL"
