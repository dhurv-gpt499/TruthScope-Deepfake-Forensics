"""Image I/O utilities for Aegis-X.

Provides safe, type-hinted functions for loading and validating image files,
ensuring they are properly converted to RGB format for downstream inference.
OPTIMIZED: Ensures intermediate buffers are freed immediately to prevent RAM/VRAM pressure.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union

# Known valid image extensions
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def is_image(path: Union[str, Path]) -> bool:
    """Checks if the given file path has a valid image extension."""
    return Path(path).suffix.lower() in VALID_IMAGE_EXTENSIONS

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Loads an image from disk and converts it to RGB format.
    
    Args:
        path (Union[str, Path]): Path to the image file.
        
    Returns:
        np.ndarray: The loaded image as an RGB numpy array (H, W, 3) uint8.
        
    Raises:
        FileNotFoundError: If the file does not exist or fails to load.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {path}")
        
    img_array = None
    img_bgr = None
    try:
        # Load bytes into temporary array
        img_array = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise FileNotFoundError(f"Error reading file bytes at path {path}: {e}")
    finally:
        # FIX: Free the byte array immediately after decoding
        del img_array
    
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load or parse image at path: {path}")
    
    # FIX: Handle alpha channel (PNG/WebP can be 4-channel)
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]  # Drop alpha channel
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # FIX: Free BGR buffer immediately after conversion
    del img_bgr
    
    # Enforce uint8 dtype and (H, W, 3) shape
    if img_rgb.dtype != np.uint8:
        img_rgb = np.round(img_rgb).clip(0, 255).astype(np.uint8)
    
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise FileNotFoundError(f"Invalid image shape: {img_rgb.shape}, expected (H, W, 3)")
    
    return img_rgb