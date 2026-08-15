"""Illumination Tool — Shape-from-Shading Physics Check (V3 — FIXED).

Compares dominant illumination gradient of the face against immediate background
context. Diffusion models often composite photorealistic faces into scenes with
conflicting directional light sources.

V3 Fixes:
- FIXED: Context region now extracts from ORIGINAL FRAME (not face crop chin)
- FIXED: media_path now actually used via frames_30fps + trajectory_bboxes
- FIXED: Diffuse lighting now abstains (0.0, 0.0) instead of voting REAL
- FIXED: Confidence now includes sharpness + face size factors
- ADDED: 2D gradient direction (not just left/right)
- ADDED: Shadow-highlight consistency check

Spec Reference: Section 2.5 (CPU Tools — Zero VRAM, Zero Training Data)
"""

import time
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    ILLUMINATION_DIFFUSE_THRESHOLD,
    ILLUMINATION_GRADIENT_CONSISTENT_WEIGHT,
    ILLUMINATION_GRADIENT_MISMATCH_WEIGHT,
    ILLUMINATION_MISMATCH_BASE_PENALTY,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


class IlluminationTool(BaseForensicTool):
    """Tool for detecting illumination direction mismatches between face and scene."""

    @property
    def tool_name(self) -> str:
        return "run_illumination"

    def setup(self) -> None:
        """Initialize tool configuration from thresholds.py."""
        self.diffuse_threshold = ILLUMINATION_DIFFUSE_THRESHOLD
        self.consistent_weight = ILLUMINATION_GRADIENT_CONSISTENT_WEIGHT
        self.mismatch_weight = ILLUMINATION_GRADIENT_MISMATCH_WEIGHT
        self.mismatch_base_penalty = ILLUMINATION_MISMATCH_BASE_PENALTY

    # ---------------------------------------------------------
    # Helper: Extract Luma from RGB
    # ---------------------------------------------------------
    @staticmethod
    def _extract_luma(rgb_img: np.ndarray) -> np.ndarray:
        """
        Convert RGB to YCrCb and extract Y (luma) channel.
        
        Args:
            rgb_img: (H, W, 3) uint8 RGB
            
        Returns:
            (H, W) float32 luma channel
        """
        ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
        return ycrcb[:, :, 0].astype(np.float32)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Division with ZeroDivisionError protection."""
        return numerator / (denominator + 1e-6)

    # ---------------------------------------------------------
    # FIX 1: Extract Scene Context from Original Frame (NOT face crop)
    # ---------------------------------------------------------
    def _extract_scene_context(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Extract region BELOW the face bbox — the actual scene context (neck/shoulders).
        
        FIX: Returns 2D luma array so left/right split by columns works correctly.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        face_h = y2 - y1
        
        # Sample region BELOW the face (shoulders/neck area in full frame)
        ctx_y1 = min(y2, h - 1)
        ctx_y2 = min(y2 + int(face_h * 0.3), h)
        
        if ctx_y2 <= ctx_y1:
            return None  # Can't extract below face
        
        # Extract context region at SAME width as face bbox
        # This ensures left/right split aligns with face left/right
        ctx_region = frame[ctx_y1:ctx_y2, x1:x2]
        
        if ctx_region.size == 0:
            return None
        
        # Extract luma (keeps 2D structure)
        return self._extract_luma(ctx_region)  # Shape: (ctx_h, ctx_w)

    # ---------------------------------------------------------
    # FIX 2: 2D Gradient Direction (Not Just Left/Right)
    # ---------------------------------------------------------
    def _compute_gradient_direction(
        self, 
        luma: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute 2D gradient vector using Sobel operators.
        
        Returns:
            (angle_degrees, magnitude, confidence)
            angle: 0=right, 90=up, 180=left, 270=down
        """
        # Sobel gradients
        gx = cv2.Sobel(luma, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(luma, cv2.CV_64F, 0, 1, ksize=5)
        
        # Mean gradient direction (weighted by magnitude)
        magnitudes = np.sqrt(gx**2 + gy**2)
        weights = magnitudes / (magnitudes.sum() + 1e-10)
        
        mean_gx = (gx * weights).sum()
        mean_gy = (gy * weights).sum()
        
        angle = float(np.degrees(np.arctan2(mean_gy, mean_gx)))
        magnitude = float(np.sqrt(mean_gx**2 + mean_gy**2))
        
        # Normalize angle to 0-360
        if angle < 0:
            angle += 360
        
        return angle, magnitude, float(magnitudes.mean())

    # ---------------------------------------------------------
    # FIX 3: Shadow-Highlight Consistency Check
    # ---------------------------------------------------------
    def _check_shadow_highlight_consistency(
        self,
        luma: np.ndarray,
        midpoint_x: int,
    ) -> Tuple[bool, str]:
        """
        Real faces: shadow falls opposite to light source.
        Deepfakes: shadow direction often contradicts highlight direction.
        
        Returns:
            (consistent, detail_string)
        """
        if luma.size == 0:
            return True, "Insufficient data for shadow check"
        
        threshold_high = np.percentile(luma, 80)
        threshold_low = np.percentile(luma, 20)
        
        highlight_mask = luma > threshold_high
        shadow_mask = luma < threshold_low
        
        # Center of mass for highlights vs shadows
        h_cols = np.where(highlight_mask)[1]
        s_cols = np.where(shadow_mask)[1]
        
        if len(h_cols) == 0 or len(s_cols) == 0:
            return True, "Insufficient contrast for shadow check"
        
        highlight_x = float(np.mean(h_cols))
        shadow_x = float(np.mean(s_cols))
        
        # Highlights and shadows should be on opposite sides of midline
        consistent = (highlight_x < midpoint_x) != (shadow_x < midpoint_x)
        
        detail = f"Highlight center: {highlight_x:.0f}, Shadow center: {shadow_x:.0f}"
        return consistent, detail

    # ---------------------------------------------------------
    # FIX 4: Dynamic Confidence Calculation
    # ---------------------------------------------------------
    def _calculate_confidence(
        self,
        face_grad: float,
        crop_sharpness: float,
        face_width: int,
    ) -> float:
        """
        Confidence adjusts based on:
        - Gradient strength (stronger = more confident)
        - Crop sharpness (blurry = unreliable gradient)
        - Face size in pixels (smaller = noisier)
        """
        # Gradient strength
        grad_conf = min(0.9, face_grad * 10)
        
        # Sharpness penalty (Laplacian variance)
        if crop_sharpness < 50.0:
            grad_conf *= 0.6
        elif crop_sharpness < 100.0:
            grad_conf *= 0.8
        
        # Face size penalty (tiny face = noisy gradient)
        if face_width < 80:
            grad_conf *= 0.7
        elif face_width < 120:
            grad_conf *= 0.85
        
        return round(min(grad_conf, 0.95), 3)

    # ---------------------------------------------------------
    # MAIN INFERENCE
    # ---------------------------------------------------------
    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run illumination analysis on all tracked faces."""
        start_time = time.time()

        # --- Guard: Missing Input ---
        tracked_faces = input_data.get("tracked_faces", [])
        frames = input_data.get("frames_30fps", [])

        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,  # Graceful abstention
                score=0.0,
                confidence=0.0,
                details={
                    "illumination_score": 0.0,
                    "face_gradient": 0.0,
                    "lighting_consistent": None,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Illumination analysis skipped: No tracked faces provided.",
            )

        if not frames:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={
                    "illumination_score": 0.0,
                    "face_gradient": 0.0,
                    "lighting_consistent": None,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Illumination analysis skipped: No frames provided for scene context.",
            )

        # --- Process Each Face ---
        face_results = []

        for face in tracked_faces:
            face_crop = face.get("face_crop_224")
            trajectory_bboxes = face.get("trajectory_bboxes", {})
            landmarks = face.get("landmarks")

            # Guard: Missing face crop
            if face_crop is None:
                continue

            face_crop = np.array(face_crop, dtype=np.uint8)

            # Guard: Invalid crop shape
            if face_crop.ndim != 3 or face_crop.shape[0] != 224 or face_crop.shape[1] != 224:
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"Invalid crop shape {face_crop.shape}, expected (224, 224, 3)"
                )
                continue

            # --- Get Best Frame Index for Context Extraction ---
            best_frame_idx = face.get("best_frame_idx", 0)
            if best_frame_idx >= len(frames):
                best_frame_idx = 0
            
            original_frame = frames[best_frame_idx]

            # --- Get Face Bbox from Trajectory for Context Extraction ---
            bbox = None
            if trajectory_bboxes and best_frame_idx in trajectory_bboxes:
                bbox = trajectory_bboxes[best_frame_idx]
            elif landmarks is not None:
                # Fallback: compute bbox from landmarks
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
            
            if bbox is None:
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"No bounding box available for scene context extraction"
                )
                continue

            # --- Extract Face Luma ---
            face_luma = self._extract_luma(face_crop)  # (224, 224)

            # --- Calculate Face Lighting Gradient (2D) ---
            midpoint_x = 112  # Exact center of 224×224 crop

            # Simple left/right for backward compatibility
            face_l = face_luma[:, :midpoint_x].mean()
            face_r = face_luma[:, midpoint_x:].mean()
            face_grad = self._safe_divide(abs(face_l - face_r), face_l + face_r)

            # 2D gradient direction (enhancement)
            face_angle, face_magnitude, _ = self._compute_gradient_direction(face_luma)

            # --- FIX 3: Check for Diffuse Lighting ---
            if face_grad < self.diffuse_threshold:
                # FIX: Diffuse lighting = NO SIGNAL = abstain (not vote REAL)
                face_results.append({
                    "identity_id": face.get("identity_id", 0),
                    "fake_score": 0.0,  # FIX: Was 0.2, now 0.0 (abstain)
                    "confidence": 0.0,  # FIX: Was 0.3, now 0.0
                    "face_gradient": face_grad,
                    "face_dom": "diffuse",
                    "ctx_dom": "none",
                    "lighting_consistent": None,
                    "interpretation": "Diffuse lighting detected — no directional signal available. Abstaining.",
                })
                continue

            # --- FIX 1: Extract Scene Context from Original Frame ---
            context_luma = self._extract_scene_context(original_frame, bbox)

            if context_luma is None or context_luma.size == 0:
                # Can't extract context — abstain
                face_results.append({
                    "identity_id": face.get("identity_id", 0),
                    "fake_score": 0.0,
                    "confidence": 0.0,
                    "face_gradient": face_grad,
                    "face_dom": "none",
                    "ctx_dom": "none",
                    "lighting_consistent": None,
                    "interpretation": "Scene context extraction failed — abstaining.",
                })
                continue

            # Calculate context gradient
            ctx_midpoint_x = context_luma.shape[1] // 2
            ctx_l = context_luma[:, :ctx_midpoint_x].mean()
            ctx_r = context_luma[:, ctx_midpoint_x:].mean()
            ctx_grad = self._safe_divide(abs(ctx_l - ctx_r), ctx_l + ctx_r)


            # --- Determine Dominant Lighting Direction ---
            face_dom = "left" if face_l > face_r else "right"
            ctx_dom = "left" if ctx_l > ctx_r else "right"

            # --- FIX 3: Shadow-Highlight Consistency ---
            shadow_consistent, shadow_detail = self._check_shadow_highlight_consistency(
                face_luma, midpoint_x
            )

            # --- Calculate Mismatch Penalty ---
            lighting_consistent = face_dom == ctx_dom

            if lighting_consistent and shadow_consistent:
                # Consistent lighting — low fake score
                fake_score = face_grad * self.consistent_weight
            elif lighting_consistent and not shadow_consistent:
                # Direction matches but shadow inconsistent — moderate penalty
                fake_score = 0.15 + (face_grad * self.consistent_weight)
            else:
                # MISMATCH — high fake score
                fake_score = self.mismatch_base_penalty + (face_grad * self.mismatch_weight)

            # Cap score at 1.0
            fake_score = min(fake_score, 1.0)

            # --- FIX 4: Calculate Confidence ---
            face_width = bbox[2] - bbox[0]
            crop_sharpness = cv2.Laplacian(
                cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY), 
                cv2.CV_64F
            ).var()
            
            confidence = self._calculate_confidence(face_grad, crop_sharpness, face_width)

            # --- Build Interpretation ---
            if lighting_consistent:
                interpretation = (
                    f"Consistent lighting found (face: {face_dom}, context: {ctx_dom}). "
                    f"Gradient: {face_grad:.3f}. {shadow_detail}"
                )
            else:
                interpretation = (
                    f"Face illumination direction mismatches environmental scene context "
                    f"(face: {face_dom}, context: {ctx_dom}). Gradient: {face_grad:.3f}. {shadow_detail}"
                )

            face_results.append({
                "identity_id": face.get("identity_id", 0),
                "fake_score": fake_score,
                "confidence": confidence,
                "face_gradient": face_grad,
                "face_angle": face_angle,
                "face_magnitude": face_magnitude,
                "face_dom": face_dom,
                "ctx_dom": ctx_dom,
                "lighting_consistent": lighting_consistent,
                "shadow_consistent": shadow_consistent,
                "interpretation": interpretation,
            })

        # --- No Valid Faces ---
        if not face_results:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={
                    "illumination_score": 0.0,
                    "face_gradient": 0.0,
                    "lighting_consistent": None,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Illumination analysis skipped: No valid face crops found.",
            )

        # --- Multi-Face: Return Highest Outlier Score ---
        best_face = max(face_results, key=lambda x: x["fake_score"])

        # --- Build Evidence Summary ---
        if best_face["lighting_consistent"] is None:
            summary = best_face["interpretation"]
        elif best_face["lighting_consistent"]:
            summary = f"Consistent lighting found for face {best_face['identity_id']} (out of {len(face_results)} analyzed)."
        else:
            summary = (
                f"Face illumination direction mismatches environmental scene context "
                f"for face {best_face['identity_id']} (out of {len(face_results)} analyzed)."
            )

        # --- Build Details Dict ---
        details = {
            "illumination_score": best_face["fake_score"],
            "face_gradient": float(best_face["face_gradient"]),
            "lighting_consistent": best_face["lighting_consistent"],
            "faces_analyzed": len(face_results),
            "face_dom": best_face.get("face_dom", "none"),
            "ctx_dom": best_face.get("ctx_dom", "none"),
            "worst_face_id": best_face["identity_id"],
            "shadow_consistent": best_face.get("shadow_consistent", None),
            "face_angle": best_face.get("face_angle", 0),
            "face_magnitude": best_face.get("face_magnitude", 0),
        }

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=float(best_face["fake_score"]),
            confidence=float(best_face["confidence"]),
            details=details,
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary=summary,
        )