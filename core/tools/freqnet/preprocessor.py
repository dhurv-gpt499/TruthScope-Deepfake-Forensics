"""
FreqNet Preprocessor
--------------------
Handles frequency domain transformation via DCT Conv2d.
Implements BT.709 luma extraction with proper device tracking.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class DCTPreprocessor(nn.Module):
    """
    Converts RGB face crops to frequency domain via DCT-II.
    
    Architecture:
        RGB (3, 224, 224) → BT.709 Luma (1, 224, 224) → DCT Conv2d → (64, 28, 28)
    """
    
    def __init__(self):
        super().__init__()
        
        # FIX #6: BT.709 coefficients as buffer for device tracking
        self.register_buffer(
            'bt709_coeffs',
            torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        )
        
        # FIX: Create and store the DCT conv layer as a module attribute
        # This allows freqnet_tool.py to register hooks on it
        self._dct_conv = self._create_dct_conv2d()
        
        # Log compression constant
        self.register_buffer('log_epsilon', torch.tensor(1.0))
    
    def _create_dct_basis_vectorized(self) -> torch.Tensor:
        """
        FIX #2 & #10: Build 64 DCT-II basis functions as Conv2d kernels.
        Fully vectorized with proper orthonormal normalization.
        """
        # Spatial positions (0-7)
        n = torch.arange(8).float()
        # Frequency indices (0-7)
        k = torch.arange(8).float()
        
        # 1D DCT-II basis: C[k,n] = α(k) * cos(π(2n+1)k / 2N), N=8
        # FIX #10: Include normalization factors α(k)
        basis_1d = torch.cos(torch.pi * torch.outer(k, 2 * n + 1) / 16)  # (8, 8)
        
        # Normalization: α(0) = 1/√8, α(k>0) = √(2/8)
        alpha = torch.ones(8) * math.sqrt(2.0 / 8.0)
        alpha[0] = 1.0 / math.sqrt(8.0)
        
        # Apply normalization to both dimensions
        basis_1d = alpha.unsqueeze(1) * basis_1d  # (8, 8)
        
        # 2D basis = outer product of 1D bases
        # basis_2d[u,v,x,y] = basis_1d[u,x] * basis_1d[v,y]
        basis_2d = torch.einsum('ux,vy->uvxy', basis_1d, basis_1d)  # (8,8,8,8)
        
        # Reshape to Conv2d format: (64, 1, 8, 8)
        # Each of 64 output channels = one frequency basis function
        basis_2d = basis_2d.reshape(64, 1, 8, 8)
        
        # Verify orthonormality (optional debug check)
        # basis_flat = basis_2d.reshape(64, 64)
        # identity = torch.eye(64)
        # assert torch.allclose(basis_flat @ basis_flat.T, identity, atol=1e-5)
        
        return basis_2d
    
    def _create_dct_conv2d(self) -> nn.Conv2d:
        """Creates Conv2d with frozen DCT-II basis weights."""
        conv = nn.Conv2d(1, 64, kernel_size=8, stride=8, padding=0, bias=False)
        
        # Load pre-computed basis
        dct_basis = self._create_dct_basis_vectorized()
        
        # FIX #2: Freeze weights (no gradients)
        conv.weight = nn.Parameter(dct_basis, requires_grad=False)
        
        return conv
    
    def _rgb_to_luma(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB to BT.709 luma (grayscale).
        
        Args:
            rgb: (B, 3, H, W) tensor in [0, 1] range
        
        Returns:
            luma: (B, 1, H, W) tensor
        """
        # FIX #6: Use registered buffer for proper device handling
        luma = (rgb * self.bt709_coeffs).sum(dim=1, keepdim=True)
        return luma  # (B, 1, H, W)
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Full preprocessing pipeline: RGB → Luma → DCT → Log-compress.
        
        Args:
            rgb: (B, 3, H, W) tensor in [0, 1] range
        
        Returns:
            freq_features: (B, 64, H/8, W/8) log-compressed DCT coefficients
        """
        # Step 1: RGB to BT.709 luma
        luma = self._rgb_to_luma(rgb)  # (B, 1, H, W)
        
        # Step 2: DCT-II via frozen Conv2d
        # FIX: Use the module attribute instead of functional conv2d
        # This enables forward hooks to capture the output
        dct_coeffs = self._dct_conv(luma)
        
        # For 224×224 input: output is (B, 64, 28, 28)
        
        # Step 3: Log compression (FIX #Upgrade C: numerical stability)
        # log(|x| + 1) compresses dynamic range
        freq_features = torch.log1p(dct_coeffs.abs())
        
        return freq_features  # (B, 64, 28, 28)


class SpatialPreprocessor(nn.Module):
    """
    Standard ImageNet normalization for spatial stream.
    """
    
    def __init__(self):
        super().__init__()
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Normalize RGB tensor with ImageNet statistics.
        
        Args:
            rgb: (B, 3, H, W) tensor in [0, 1] range
        
        Returns:
            normalized: (B, 3, H, W) tensor
        """
        return (rgb - self.mean) / self.std