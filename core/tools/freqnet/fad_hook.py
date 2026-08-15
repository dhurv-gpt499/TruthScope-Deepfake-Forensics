import torch.nn as nn
"""
FreqNet FAD Hook Module
-----------------------
Captures DCT coefficients and performs band analysis.
Implements JPEG zigzag ordering for frequency band separation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


# FIX #9: Hardcoded JPEG zigzag scan order
# This maps: zigzag_position → raster_position in 8×8 block
ZIGZAG_ORDER = [
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
]

# Frequency band definitions based on zigzag positions
# Base (low freq): DC + coarse structure
# Mid (mid freq): Medium texture
# High (high freq): Fine texture + artifacts
BAND_BASE = ZIGZAG_ORDER[0:21]    # coefficients 0-20 (low frequency)
BAND_MID = ZIGZAG_ORDER[21:42]    # coefficients 21-41 (mid frequency)
BAND_HIGH = ZIGZAG_ORDER[42:64]   # coefficients 42-63 (high frequency)


@dataclass
class BandAnalysis:
    """Results from frequency band analysis."""
    base_energy: float
    mid_energy: float
    high_energy: float
    total_energy: float
    base_ratio: float
    mid_ratio: float
    high_ratio: float
    z_base: Optional[float] = None
    z_mid: Optional[float] = None
    z_high: Optional[float] = None
    anomaly_detected: bool = False
    anomaly_type: Optional[str] = None
    interpretation: str = ""


class FADHook:
    """
    Forward hook for capturing DCT coefficients and performing band analysis.
    
    FIX #12: Properly detaches captured tensors to prevent graph retention.
    FIX #7: Hook cleanup via context manager pattern.
    """
    
    def __init__(self, calibration_data: Optional[Dict[str, float]] = None):
        """
        Args:
            calibration_data: Dict with mean/std for each band (for Z-score).
                             If None, uses ratio-based fallback.
        """
        self.calibration = calibration_data
        self.captured_features = None
        self.hook_handle = None
    
    def _capture_hook(self, module, input, output):
        """
        FIX #12: Capture and detach immediately to prevent graph retention.
        """
        # output shape: (B, 64, 28, 28)
        # FIX #12: detach().clone() to free from computation graph
        self.captured_features = output.detach().clone().cpu()
    
    def register(self, module: nn.Module):
        """Register hook on target module."""
        self.hook_handle = module.register_forward_hook(self._capture_hook)
        return self
    
    def remove(self):
        """FIX #7: Always remove hook to prevent VRAM leaks."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - guarantees hook cleanup."""
        self.remove()
    
    def _compute_band_energy(self, features: torch.Tensor, band_indices: List[int]) -> float:
        """
        Compute energy for a specific frequency band.
        
        Args:
            features: (B, 64, 28, 28) DCT coefficients
            band_indices: List of zigzag indices for this band
        
        Returns:
            energy: L2 norm of band coefficients
        """
        # Extract coefficients for this band
        # features[:, band_indices, :, :] → (B, len(band), 28, 28)
        band_features = features[:, band_indices, :, :]
        
        # Compute L2 energy (Frobenius norm)
        energy = torch.norm(band_features, p='fro').item()
        
        return energy
    
    def _safe_z_score(self, value: float, mean: float, std: float) -> float:
        """FIX #Upgrade C: Z-score with zero-std protection."""
        if std < 1e-8:
            return 0.0
        return (value - mean) / std
    
    def analyze(self) -> BandAnalysis:
        """
        Perform band analysis on captured features.
        
        Returns:
            BandAnalysis with energies, ratios, and Z-scores
        """
        if self.captured_features is None:
            return BandAnalysis(
                base_energy=0.0, mid_energy=0.0, high_energy=0.0,
                total_energy=0.0, base_ratio=0.0, mid_ratio=0.0, high_ratio=0.0,
                interpretation="No features captured"
            )
        
        # Compute energy for each band
        base_energy = self._compute_band_energy(self.captured_features, BAND_BASE)
        mid_energy = self._compute_band_energy(self.captured_features, BAND_MID)
        high_energy = self._compute_band_energy(self.captured_features, BAND_HIGH)
        
        total_energy = base_energy + mid_energy + high_energy + 1e-10
        
        # Compute ratios
        base_ratio = base_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
        
        # Compute Z-scores if calibration available
        z_base = z_mid = z_high = None
        anomaly_detected = False
        anomaly_type = None
        interpretation = ""
        
        if self.calibration is not None:
            # FIX #3: Dual-mode - use Z-scores when calibration exists
            z_base = self._safe_z_score(
                base_ratio,
                self.calibration.get('mean_base', base_ratio),
                self.calibration.get('std_base', 0.1)
            )
            z_mid = self._safe_z_score(
                mid_ratio,
                self.calibration.get('mean_mid', mid_ratio),
                self.calibration.get('std_mid', 0.1)
            )
            z_high = self._safe_z_score(
                high_ratio,
                self.calibration.get('mean_high', high_ratio),
                self.calibration.get('std_high', 0.1)
            )
            
            # Check for anomalies
            from utils.thresholds import FREQNET_Z_THRESHOLD
            
            if abs(z_high) > FREQNET_Z_THRESHOLD:
                anomaly_detected = True
                anomaly_type = "high_freq"
                interpretation = "GAN texture artifacts or diffusion upscaling detected"
            elif abs(z_base) > FREQNET_Z_THRESHOLD:
                anomaly_detected = True
                anomaly_type = "low_freq"
                interpretation = "Global illumination mismatch or face compositing detected"
        else:
            # FIX #3: Fallback mode - use ratio thresholds
            from utils.thresholds import FREQNET_HIGH_BAND_RATIO_THRESHOLD
            
            if high_ratio > FREQNET_HIGH_BAND_RATIO_THRESHOLD:
                anomaly_detected = True
                anomaly_type = "high_freq"
                interpretation = "Elevated high-frequency energy suggests synthetic texture"
        
        return BandAnalysis(
            base_energy=base_energy,
            mid_energy=mid_energy,
            high_energy=high_energy,
            total_energy=total_energy,
            base_ratio=base_ratio,
            mid_ratio=mid_ratio,
            high_ratio=high_ratio,
            z_base=z_base,
            z_mid=z_mid,
            z_high=z_high,
            anomaly_detected=anomaly_detected,
            anomaly_type=anomaly_type,
            interpretation=interpretation
        )