"""Corneal Tool — Specular Reflection Consistency Check (V2 — FIXED).

Physics-based check specifically targeting diffusion models (Midjourney, DALL-E,
Stable Diffusion) that struggle to synthesize consistent specular highlights
(catchlights) in both eyes simultaneously.

V2 Fixes:
- FIXED: Removed mirror correction (real eyes have SAME offset, not mirrored)
- FIXED: Added absolute brightness threshold (catchlights must be >180 intensity)
- FIXED: Coordinate transformation from full-frame to face_crop_224 space
- FIXED: Divergence normalization uses geometric maximum (2.83, not 0.5)
- FIXED: Single catchlight now abstains (0.0, 0.0) instead of voting fake
- FIXED: Connected components for multi-blob catchlight detection
- FIXED: No padding on edge cases — reject instead of corrupt ROI

Spec Reference: Section 2.6 (CPU Tools — Zero VRAM, Zero Training Data)
"""

import time
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from core.base_tool  import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    CORNEAL_BOX_SIZE,
    CORNEAL_MAX_DIVERGENCE,
    CORNEAL_CONSISTENCY_THRESHOLD,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Catchlights must be objectively bright (not just top 2% of dark iris)
ABSOLUTE_MIN_BRIGHTNESS = 180.0  # 0-255 scale

# Maximum possible divergence in normalized offset space [-1,1]×[-1,1]
# Distance from (-1,-1) to (1,1) = sqrt(2² + 2²) = 2.83
MAX_GEOMETRIC_DIVERGENCE = np.sqrt(8)  # ≈ 2.83

# Eye region sanity check (iris should be in top half of face crop)
EYE_REGION_Y_MIN = 50   # Top of eye band in 224×224 crop
EYE_REGION_Y_MAX = 140  # Bottom of eye band


class CornealTool(BaseForensicTool):
    """Tool for detecting corneal reflection (catchlight) inconsistencies."""

    @property
    def tool_name(self) -> str:
        return "run_corneal"

    def setup(self) -> None:
        """Initialize tool configuration from thresholds.py."""
        self.box_size = CORNEAL_BOX_SIZE  # 15×15 pixels
        self.max_divergence = CORNEAL_MAX_DIVERGENCE  # 0.5 (sensitivity threshold)
        self.consistency_threshold = CORNEAL_CONSISTENCY_THRESHOLD  # 0.5

    # ---------------------------------------------------------
    # FIX 2: Transform Landmarks from Full-Frame to Crop Space
    # ---------------------------------------------------------
    def _transform_to_crop_coords(
        self,
        landmark: np.ndarray,
        bbox: Tuple[int, int, int, int],
        crop_size: int = 224,
    ) -> np.ndarray:
        """
        Convert full-frame landmark coordinates to face_crop_224 pixel space.
        
        Args:
            landmark: (2,) pixel coordinate in FULL FRAME space
            bbox: (x1, y1, x2, y2) of face in full frame
            crop_size: Target crop size (224)
            
        Returns:
            (2,) pixel coordinate in crop space
        """
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Scale factors from face bbox to 224×224 crop
        scale_x = crop_size / (face_w + 1e-10)
        scale_y = crop_size / (face_h + 1e-10)
        
        # Transform: subtract bbox origin, then scale
        crop_x = (landmark[0] - x1) * scale_x
        crop_y = (landmark[1] - y1) * scale_y
        
        return np.array([crop_x, crop_y], dtype=np.float32)

    # ---------------------------------------------------------
    # Helper: Extract Iris ROI from Face Crop
    # ---------------------------------------------------------
    def _extract_iris_roi(
        self,
        face_crop: np.ndarray,
        iris_landmark: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract 15×15 pixel box centered on iris landmark.
        
        FIX: No padding on edge cases — reject instead of corrupt ROI.
        """
        if face_crop is None or face_crop.size == 0:
            return None
        
        if iris_landmark is None or len(iris_landmark) < 2:
            return None
        
        h, w = face_crop.shape[:2]
        cx, cy = int(iris_landmark[0]), int(iris_landmark[1])
        
        # Validate landmark is within crop bounds
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        
        # FIX: Sanity check — iris should be in eye region (top half of face)
        if not (EYE_REGION_Y_MIN <= cy <= EYE_REGION_Y_MAX):
            logger.debug(f"Iris landmark at y={cy} outside expected eye region [{EYE_REGION_Y_MIN}-{EYE_REGION_Y_MAX}]")
            return None
        
        # Extract 15×15 box centered on iris
        half_box = self.box_size // 2
        x1 = max(0, cx - half_box)
        y1 = max(0, cy - half_box)
        x2 = min(w, cx + half_box + 1)
        y2 = min(h, cy + half_box + 1)
        
        roi = face_crop[y1:y2, x1:x2]
        
        # FIX: No padding — if box is clipped, reject (landmark likely wrong)
        if roi.shape[0] != self.box_size or roi.shape[1] != self.box_size:
            return None
        
        return roi

    # ---------------------------------------------------------
    # FIX 2: Detect Catchlight with Absolute Brightness Guard
    # ---------------------------------------------------------
    def _detect_catchlight_centroid(
        self,
        iris_roi: np.ndarray,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Detect specular highlight centroid within iris ROI.
        
        FIX: Requires absolute brightness > 180, not just top 2% relative.
        FIX: Uses connected components for multi-blob detection.
        """
        if iris_roi is None or iris_roi.size == 0:
            return None, 0.0
        
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(iris_roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # FIX 2: Absolute brightness guard — catchlights must be genuinely bright
        max_brightness = float(np.max(gray))
        if max_brightness < ABSOLUTE_MIN_BRIGHTNESS:
            return None, 0.0  # No real catchlight present
        
        # Relative threshold (top 2%)
        threshold = float(np.percentile(gray, 98))
        
        # FIX 2: Must ALSO exceed absolute minimum
        actual_threshold = max(threshold, ABSOLUTE_MIN_BRIGHTNESS)
        
        catchlight_mask = (gray >= actual_threshold).astype(np.uint8)
        
        # Check if any catchlight pixels found
        if catchlight_mask.sum() == 0:
            return None, 0.0
        
        # FIX: Use connected components to find largest blob (handles multi-light)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            catchlight_mask, connectivity=8
        )
        
        if num_labels < 2:  # Only background (label 0)
            return None, 0.0
        
        # Find largest non-background component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        centroid_x, centroid_y = centroids[largest_label]
        
        # Normalize to ROI center (0, 0 = center of 15×15 box)
        center_x, center_y = self.box_size / 2, self.box_size / 2
        offset_x = (centroid_x - center_x) / center_x  # -1.0 to +1.0
        offset_y = (centroid_y - center_y) / center_y  # -1.0 to +1.0
        
        # Catchlight strength (how bright relative to max)
        strength = max_brightness / 255.0
        
        return (float(offset_x), float(offset_y)), strength

    # ---------------------------------------------------------
    # FIX 1: REMOVED Mirror Correction — Real Eyes Have SAME Offset
    # ---------------------------------------------------------
    # _apply_mirror_correction() DELETED — physically incorrect

    # ---------------------------------------------------------
    # MAIN INFERENCE
    # ---------------------------------------------------------
    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run corneal reflection analysis on all tracked faces."""
        start_time = time.time()

        # --- Guard: Missing Input ---
        tracked_faces = input_data.get("tracked_faces", [])

        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,  # Graceful abstention
                score=0.0,
                confidence=0.0,
                details={
                    "corneal_score": 0.0,
                    "catchlights_detected": False,
                    "divergence": 0.0,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Corneal analysis skipped: No tracked faces provided.",
            )

        # --- Process Each Face ---
        face_results = []

        for face in tracked_faces:
            face_crop = face.get("face_crop_224")
            landmarks = face.get("landmarks")
            trajectory_bboxes = face.get("trajectory_bboxes", {})
            best_frame_idx = face.get("best_frame_idx", 0)

            # Guard: Missing face crop or landmarks
            if face_crop is None or landmarks is None:
                continue

            face_crop = np.array(face_crop, dtype=np.uint8)
            landmarks = np.array(landmarks, dtype=np.float32)

            # Guard: Invalid landmark shape (must have 478 points for iris nodes)
            if landmarks.shape[0] < 478:
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"Insufficient landmarks ({landmarks.shape[0]} < 478) for iris nodes. "
                    f"Preprocessor must use refine_landmarks=True."
                )
                continue

            # FIX 2: Get face bbox for coordinate transformation
            face_bbox = trajectory_bboxes.get(best_frame_idx)
            if face_bbox is None:
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"No bounding box available for coordinate transformation"
                )
                continue

            # --- Extract Iris Centers (Nodes 468, 473) ---
            left_iris_full = landmarks[468]  # Left iris center (FULL FRAME coords)
            right_iris_full = landmarks[473]  # Right iris center (FULL FRAME coords)

            # FIX 2: Transform to crop coordinate space
            left_iris_crop = self._transform_to_crop_coords(left_iris_full, face_bbox)
            right_iris_crop = self._transform_to_crop_coords(right_iris_full, face_bbox)

            # --- Extract Iris ROIs ---
            left_roi = self._extract_iris_roi(face_crop, left_iris_crop)
            right_roi = self._extract_iris_roi(face_crop, right_iris_crop)

            if left_roi is None or right_roi is None:
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"Failed to extract iris ROI (eyes may be closed or occluded)"
                )
                continue

            # --- Detect Catchlights ---
            left_centroid, left_strength = self._detect_catchlight_centroid(left_roi)
            right_centroid, right_strength = self._detect_catchlight_centroid(right_roi)

            # Guard: No catchlight detected in either eye → ABSTAIN
            if left_centroid is None and right_centroid is None:
                face_results.append({
                    "identity_id": face.get("identity_id", 0),
                    "fake_score": 0.0,
                    "confidence": 0.0,
                    "catchlights_detected": False,
                    "divergence": 0.0,
                    "consistent": None,
                    "interpretation": "No catchlights detected in either eye — abstaining.",
                })
                continue

            # FIX 6: Catchlight in only one eye → ABSTAIN (can't measure divergence)
            if left_centroid is None or right_centroid is None:
                face_results.append({
                    "identity_id": face.get("identity_id", 0),
                    "fake_score": 0.0,  # FIX: Was 0.5, now 0.0 (abstain)
                    "confidence": 0.0,  # FIX: Was 0.4, now 0.0
                    "catchlights_detected": False,
                    "divergence": 0.0,
                    "consistent": None,  # FIX: Was False, now None (unknown)
                    "interpretation": "Catchlight detected in only one eye — insufficient data to score.",
                })
                continue

            # --- Measure Divergence (NO MIRROR CORRECTION) ---
            # FIX 1: Real eyes have SAME offset for same light source
            divergence = float(np.linalg.norm(
                np.array(left_centroid) - np.array(right_centroid)
            ))

            # FIX 5: Normalize by geometric maximum first
            normalized_divergence = divergence / MAX_GEOMETRIC_DIVERGENCE
            
            # Then apply sensitivity threshold
            fake_score = min(1.0, normalized_divergence / self.max_divergence)
            consistent = bool(fake_score < self.consistency_threshold) 

            # --- Calculate Confidence ---
            # Higher catchlight strength = more confident in centroid detection
            avg_strength = (left_strength + right_strength) / 2
            confidence = min(0.9, 0.5 + (avg_strength * 0.4))

            # --- Build Interpretation ---
            if consistent:
                interpretation = (
                    f"Corneal reflections consistent between both eyes "
                    f"(divergence: {divergence:.3f}, score: {fake_score:.3f})."
                )
            else:
                interpretation = (
                    f"Asymmetric corneal reflections detected — specular highlights "
                    f"diverge between left and right eyes (divergence: {divergence:.3f}, "
                    f"score: {fake_score:.3f})."
                )

            face_results.append({
                "identity_id": face.get("identity_id", 0),
                "fake_score": fake_score,
                "confidence": confidence,
                "catchlights_detected": True,
                "divergence": divergence,
                "consistent": consistent,
                "left_offset": left_centroid,
                "right_offset": right_centroid,
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
                    "corneal_score": 0.0,
                    "catchlights_detected": False,
                    "divergence": 0.0,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Corneal analysis skipped: No valid iris regions found.",
            )

        # --- Multi-Face: Return Highest Outlier Score ---
        best_face = max(face_results, key=lambda x: x["fake_score"])

        # --- Build Evidence Summary ---
        if not best_face["catchlights_detected"]:
            summary = best_face["interpretation"]
        elif best_face["consistent"]:
            summary = f"Corneal reflections consistent for face {best_face['identity_id']} (out of {len(face_results)} analyzed)."
        else:
            summary = (
                f"Asymmetric corneal reflections detected for face {best_face['identity_id']} "
                f"(out of {len(face_results)} analyzed)."
            )

        # --- Build Details Dict ---
        details = {
            "corneal_score": best_face["fake_score"],
            "catchlights_detected": best_face["catchlights_detected"],
            "divergence": best_face["divergence"],
            "consistent": best_face["consistent"],
            "faces_analyzed": len(face_results),
            "worst_face_id": best_face["identity_id"],
            "left_offset": best_face.get("left_offset", None),
            "right_offset": best_face.get("right_offset", None),
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