import time
import numpy as np
import cv2  # FIX 3: Add cv2 for Laplacian variance check
from typing import Dict, Any, Tuple, Optional

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    RPPG_MIN_FRAMES,
    RPPG_HAIR_OCCLUSION_VARIANCE,
    RPPG_MIN_TEMPORAL_STD,
    RPPG_CARDIAC_BAND_MIN_HZ,
    RPPG_CARDIAC_BAND_MAX_HZ,
    RPPG_COHERENCE_THRESHOLD_HZ,
    RPPG_FFT_NFFT,
    RPPG_SNR_THRESHOLD
)
import mediapipe as mp


class RPPGTool(BaseForensicTool):
    """Tool for detecting biological liveness signals using POS rPPG algorithm and Spectral Coherence."""

    @property
    def tool_name(self) -> str:
        return "run_rppg"

    def setup(self) -> None:
        self._debug = False

    def _extract_roi(
        self, frame: np.ndarray, current_bbox: tuple, relative_box: tuple
    ) -> np.ndarray:
        x1, y1, x2, y2 = current_bbox
        w = x2 - x1
        h = y2 - y1
        rx_min, ry_min, rx_max, ry_max = relative_box
        fx1 = int(x1 + (w * rx_min))
        fy1 = int(y1 + (h * ry_min))
        fx2 = int(x1 + (w * rx_max))
        fy2 = int(y1 + (h * ry_max))
        frame_h, frame_w = frame.shape[:2]
        fx1 = max(0, min(fx1, frame_w - 1))
        fy1 = max(0, min(fy1, frame_h - 1))
        fx2 = max(0, min(fx2, frame_w))
        fy2 = max(0, min(fy2, frame_h))
        if fx1 >= fx2 or fy1 >= fy2:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[fy1:fy2, fx1:fx2]

    def _get_facial_rois(self, landmarks: np.ndarray) -> Dict[str, tuple]:
        # FIX 4: Use Spec Section 2.2 forehead landmarks [109, 10, 338, 297, 332, 284, 103, 67]
        rois = {
            "forehead": (0.2, 0.05, 0.8, 0.25),
            "left_cheek": (0.1, 0.5, 0.4, 0.85),
            "right_cheek": (0.6, 0.5, 0.9, 0.85),
        }
        if len(landmarks.shape) == 2 and landmarks.shape[0] == 478:
            face_min_x, face_max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
            face_min_y, face_max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
            face_w = face_max_x - face_min_x
            face_h = face_max_y - face_min_y
            if face_w > 0 and face_h > 0:
                def _rel(pts):
                    return (
                        (np.min(pts[:, 0]) - face_min_x) / face_w,
                        (np.min(pts[:, 1]) - face_min_y) / face_h,
                        (np.max(pts[:, 0]) - face_min_x) / face_w,
                        (np.max(pts[:, 1]) - face_min_y) / face_h,
                    )
                # FIX 4: Added index 109 to forehead polygon
                rois["forehead"] = _rel(landmarks[[109, 10, 338, 297, 332, 284, 103, 67]])
                rois["left_cheek"] = _rel(landmarks[[50, 205, 207, 215, 138, 135, 210]])
                rois["right_cheek"] = _rel(landmarks[[280, 425, 427, 435, 367, 364, 430]])
        return rois

    def _check_hair_occlusion(self, roi: np.ndarray) -> Tuple[bool, float]:
        """
        FIX 3: Hair occlusion guardrail per Spec Section 2.2.
        Returns (is_occluded, variance_score)
        """
        if roi.size == 0:
            return True, 0.0
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
        variance = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
        # Spec threshold: variance > 35.0 indicates hair texture, not smooth skin
        return variance > RPPG_HAIR_OCCLUSION_VARIANCE, variance

    def _extract_pos_signal(
        self, frames: list, trajectory: dict, relative_roi: tuple
    ) -> Tuple[Optional[np.ndarray], float, bool]:
        """Extracts the 1D POS pulse signal for a specific ROI.
        Returns (H, temporal_std, hair_occluded) or (None, 0.0, False) on failure.
        """
        rgb_means = []
        last_known_box = None
        hair_occluded = False

        for f_idx, frame in enumerate(frames):
            if f_idx in trajectory:
                curr_box = trajectory[f_idx]
                last_known_box = curr_box
            elif last_known_box is not None:
                curr_box = last_known_box
            else:
                continue

            roi = self._extract_roi(frame, curr_box, relative_roi)

            # FIX 3: Hair occlusion check on first valid frame
            if len(rgb_means) == 0 and roi.size > 0:
                is_occluded, variance = self._check_hair_occlusion(roi)
                if is_occluded:
                    hair_occluded = True

                # Darkness occlusion guard
                gray_roi = np.mean(roi, axis=2) if roi.ndim == 3 else roi
                if np.mean(gray_roi) < 50.0:
                    return None, 0.0, False

            if roi.size == 0:
                if rgb_means:
                    rgb_means.append(rgb_means[-1])
                else:
                    rgb_means.append(np.array([128.0, 128.0, 128.0]))
            else:
                spatial_mean = np.mean(roi, axis=(0, 1))
                rgb_means.append(spatial_mean)

        rgb_matrix = np.array(rgb_means, dtype=np.float64)
        if len(rgb_matrix) < RPPG_MIN_FRAMES:
            return None, 0.0, hair_occluded

        green_temporal_std = float(np.std(rgb_matrix[:, 1]))

        mean_rgb = np.maximum(np.mean(rgb_matrix, axis=0), 1.0)
        Cn = rgb_matrix / mean_rgb
        pos_weights = np.array([[0, 1, -1], [-2, 1, 1]])
        S = pos_weights @ Cn.T
        std_s0 = np.std(S[0])
        std_s1 = np.std(S[1]) + 1e-7
        h = S[0] + (std_s0 / std_s1) * S[1]
        h_mean = np.mean(h)
        h_std = np.std(h) + 1e-7
        H = (h - h_mean) / h_std
        return np.nan_to_num(H), green_temporal_std, hair_occluded

    def _calculate_signal_metrics(
        self, signal_1d: np.ndarray, fps: float = 30.0
    ) -> Dict[str, float]:
        if np.std(signal_1d) < 1e-5:
            return {"peak_hz": 0.0, "snr_db": -100.0, "spectral_concentration": 0.0}

        signal_centered = signal_1d.copy() - np.mean(signal_1d)
        windowed = signal_centered * np.hanning(len(signal_centered))

        n_fft = RPPG_FFT_NFFT
        psd = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

        band_mask = (freqs >= RPPG_CARDIAC_BAND_MIN_HZ) & (freqs <= RPPG_CARDIAC_BAND_MAX_HZ)
        band_psd = psd[band_mask]
        band_freqs = freqs[band_mask]

        if len(band_psd) == 0 or np.sum(band_psd) == 0:
            return {"peak_hz": 0.0, "snr_db": -100.0, "spectral_concentration": 0.0}

        peak_idx = int(np.argmax(band_psd))
        peak_hz = float(band_freqs[peak_idx])

        median_power = float(np.median(band_psd))
        peak_power = float(band_psd[peak_idx])
        spectral_concentration = peak_power / (median_power + 1e-10)

        start_idx = max(0, peak_idx - 2)
        end_idx = min(len(band_psd), peak_idx + 3)
        signal_power = np.sum(band_psd[start_idx:end_idx])
        noise_mask = (freqs >= 0.4)
        total_valid = np.sum(psd[noise_mask])
        noise_power = max(total_valid - signal_power, 1e-10)
        signal_power = max(signal_power, 1e-10)
        snr_db = float(10 * np.log10(signal_power / noise_power))

        return {
            "peak_hz": peak_hz,
            "snr_db": snr_db,
            "spectral_concentration": spectral_concentration,
        }

    def _evaluate_liveness(
        self,
        h_forehead: np.ndarray,
        h_left: np.ndarray,
        h_right: np.ndarray,
        quality_stds: list,
        hair_occluded: bool,
    ) -> dict:
        roi_labels = ["Forehead", "L_Cheek", "R_Cheek"]
        signals = [h_forehead, h_left, h_right]

        # FIX 3: Hair occlusion → AMBIGUOUS (not NO_PULSE)
        if hair_occluded:
            return {
                "label": "AMBIGUOUS",
                "score": 0.0,  # FIX 1: Abstention = 0.0
                "confidence": 0.0,
                "interpretation": (
                    "rPPG abstained: Forehead ROI occluded by hair (texture variance > 35.0). "
                    "Blood flow signal cannot be reliably extracted from this region."
                ),
            }

        all_flat = all(np.std(s) < 1e-5 for s in signals)
        if all_flat:
            return {
                "label": "NO_PULSE",
                "score": 1.0,
                "confidence": 0.90,
                "interpretation": (
                    "Biological liveness failed: All facial regions show zero temporal "
                    "variation — complete flatline. No blood flow modulation detected. "
                    "Strong indicator of synthetic or static media."
                ),
            }

        analyzable_count = sum(1 for s in quality_stds if s >= RPPG_MIN_TEMPORAL_STD)

        if analyzable_count < 2:
            return {
                "label": "SYNTHETIC_FLATLINE",
                "score": 1.0,  # FIX: Penalize flat pixel variance as FAKE
                "confidence": 0.85,
                "interpretation": (
                    "Biological liveness failed: Facial regions lack physiological temporal variance. "
                    "This perfectly smooth consistency indicates AI generation lacking human blood flow."
                ),
            }

        metrics = [self._calculate_signal_metrics(s) for s in signals]

        if getattr(self, "_debug", False):
            for i, label in enumerate(roi_labels):
                m = metrics[i]
                print(
                    f"       [DEBUG rPPG] {label}: "
                    f"SNR={m['snr_db']:+.2f} dB  "
                    f"SC={m['spectral_concentration']:.1f}x  "
                    f"peak={m['peak_hz']:.3f} Hz ({m['peak_hz']*60:.0f} BPM)  "
                    f"green_std={quality_stds[i]:.2f}"
                )

        good_mask = [m["spectral_concentration"] >= RPPG_SNR_THRESHOLD for m in metrics]
        n_good = sum(good_mask)

        if n_good == 0:
            max_sc = max(m["spectral_concentration"] for m in metrics)
            return {
                "label": "NO_PULSE",
                "score": 1.0,  # FIX: Always penalize 0 valid peaks as FAKE
                "confidence": 0.90,
                "interpretation": (
                    "Biological liveness failed: No cardiac peak detected in any facial region. "
                    f"Spectral concentration is flat (max {max_sc:.1f}x). Strong indicator of AI."
                ),
            }

        if n_good == 1:
            return {
                "label": "WEAK_PULSE_FAILED",
                "score": 1.0,  # FIX: Penalize partial failures as FAKE
                "confidence": 0.70,
                "interpretation": (
                    "Biological liveness failed: Unnatural pulse signal. "
                    "Only one facial region yielded a usable signal. Common in AI blending."
                ),
            }

        good_indices = [i for i, g in enumerate(good_mask) if g]
        good_peaks = [metrics[i]["peak_hz"] for i in good_indices]

        best_pair = None
        best_pair_diff = float("inf")

        for a in range(len(good_indices)):
            for b in range(a + 1, len(good_indices)):
                diff = abs(good_peaks[a] - good_peaks[b])
                if diff < best_pair_diff:
                    best_pair_diff = diff
                    best_pair = (good_indices[a], good_indices[b])

        if best_pair_diff <= RPPG_COHERENCE_THRESHOLD_HZ:
            pair_avg_hz = (metrics[best_pair[0]]["peak_hz"] + metrics[best_pair[1]]["peak_hz"]) / 2
            n_coherent = sum(
                1 for i in good_indices
                if abs(metrics[i]["peak_hz"] - pair_avg_hz) <= RPPG_COHERENCE_THRESHOLD_HZ
            )
            avg_bpm = pair_avg_hz * 60

            conf = 0.95 if n_coherent >= 3 else (0.70 if n_coherent == 2 else 0.50)

            return {
                "label": "PULSE_PRESENT",
                "score": 0.0,
                "confidence": conf,
                "interpretation": (
                    f"Biological liveness confirmed: Synchronous cardiac pulse detected "
                    f"across {n_coherent}/3 facial regions at ~{avg_bpm:.0f} BPM."
                ),
            }

        conf = 0.90 if n_good == 3 else 0.65
        return {
            "label": "INCOHERENT",
            "score": 1.0,
            "confidence": conf,
            "interpretation": (
                "Spoof detected: Pulse frequencies are unsynchronized across facial regions. "
                f"Best pair spread: {best_pair_diff*60:.1f} BPM."
            ),
        }

    def _lightweight_face_check(self, frames: list) -> bool:
        """Sample up to 5 evenly-spaced frames, run MediaPipe face_detection to verify face presence."""
        if not frames:
            return False
            
        mp_face_detection = mp.solutions.face_detection
        sample_indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
        detections_count = 0
        
        # frames are assumed to be RGB based on Aegis-X preprocessing
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
            for idx in sample_indices:
                frame = frames[idx]
                results = face_detection.process(frame)
                
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        area = (bboxC.width * w) * (bboxC.height * h)
                        if detection.score[0] > 0.6 and area > 2500:
                            detections_count += 1
                            break # Found at least one valid face in this frame
                            
        # Face must be detected in majority of sampled frames
        required_detections = min(3, len(sample_indices) // 2 + 1)
        return detections_count >= required_detections

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()

        # FIX 6: Check if this is a static image (rPPG requires video)
        media_type = input_data.get("original_media_type", "unknown")
        if media_type == "image":
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,  # FIX 1: Abstention = 0.0
                confidence=0.0,
                details={"liveness_label": "SKIPPED", "reason": "Static image"},
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="rPPG skipped: Static image has no temporal signal.",
            )

        if "frames_30fps" not in input_data or "tracked_faces" not in input_data:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,  # FIX 1 & 5: Align with Base class exception handler
                confidence=0.0,
                details={"liveness_label": "ERROR"},
                error=True,
                error_msg="Missing 'frames_30fps' or 'tracked_faces' in input_data",
                execution_time=time.time() - start_time,
                evidence_summary="Missing required input data.",
            )

        frames = input_data.get("frames_30fps", [])
        tracked_faces = input_data.get("tracked_faces", [])

        if len(frames) < RPPG_MIN_FRAMES:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={"liveness_label": "ABSTAIN", "reason": "INSUFFICIENT_TEMPORAL_DATA"},
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary=f"rPPG skipped: insufficient frames ({len(frames)} < {RPPG_MIN_FRAMES}) for rPPG analysis.",
            )
            
        # BACKUP GUARD: Check if preprocessor missed faces or tracking is somehow empty.
        # This prevents running purely on noise if invoked erroneously.
        if not tracked_faces:
            has_face = self._lightweight_face_check(frames)
            if not has_face:
                return ToolResult(
                    tool_name=self.tool_name,
                    success=True,
                    score=0.0,
                    confidence=0.0,
                    details={"liveness_label": "ABSTAIN", "reason": "NO_FACE_CONFIRMED"},
                    error=False,
                    error_msg=None,
                    execution_time=time.time() - start_time,
                    evidence_summary="rPPG skipped: Neither preprocessor nor internal backup detector found a valid face."
                )

        face_results = []

        for face in tracked_faces:
            trajectory = face.get("trajectory_bboxes", {})
            landmarks = face.get("landmarks", [])
            rois = self._get_facial_rois(landmarks)
            
            face_window = face.get("face_window", (0, 0))
            if face_window[1] > face_window[0]:
                start_offset = face_window[0]
                end_frame = face_window[1]
                target_frames = frames[start_offset:end_frame]
                sliced_trajectory = {k - start_offset: v for k, v in trajectory.items() if start_offset <= k < end_frame}
            else:
                face_results.append({
                    "score": 0.0,
                    "confidence": 0.0,
                    "label": "ABSTAIN",
                    "interpretation": "Face window could not be established. Skipping to avoid noise.",
                    "metrics": {}
                })
                continue

            h_forehead, std_f, hair_f = self._extract_pos_signal(target_frames, sliced_trajectory, rois["forehead"])
            h_left, std_l, hair_l = self._extract_pos_signal(target_frames, sliced_trajectory, rois["left_cheek"])
            h_right, std_r, hair_r = self._extract_pos_signal(target_frames, sliced_trajectory, rois["right_cheek"])

            # If tracking was dropped entirely mid-video
            if h_forehead is None or h_left is None or h_right is None:
                if hair_f:
                    # Legitimate hair occlusion -> Abstain
                    face_results.append(
                        {
                            "score": 0.0,
                            "confidence": 0.0,
                            "label": "ABSTAIN",
                            "interpretation": "One or more facial regions occluded by hair.",
                            "metrics": {}
                        }
                    )
                else:
                    # Tracking just failed structurally -> FAKE
                    face_results.append(
                        {
                            "score": 1.0,
                            "confidence": 0.85,
                            "label": "TRACKING_FAILED",
                            "interpretation": "Facial tracking was structurally unstable and lost mid-video. Strong indicator of temporal AI instability.",
                            "metrics": {}
                        }
                    )
                continue

            liveness = self._evaluate_liveness(
                h_forehead, h_left, h_right,
                quality_stds=[std_f, std_l, std_r],
                hair_occluded=hair_f,
            )
            
            # FIX 2: Capture metrics for details dict
            metrics = [self._calculate_signal_metrics(s) for s in [h_forehead, h_left, h_right]]
            face_results.append(
                {
                    "score": liveness["score"],
                    "confidence": liveness["confidence"],
                    "label": liveness["label"],
                    "interpretation": liveness["interpretation"],
                    "metrics": {
                        "forehead": metrics[0],
                        "left_cheek": metrics[1],
                        "right_cheek": metrics[2],
                    }
                }
            )

        if not face_results:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,  # FIX 1: Abstention = 0.0
                confidence=0.0,
                details={"liveness_label": "NO_FACES"},
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="All tracked faces yielded ambiguous tracking or were occluded.",
            )

        best = sorted(face_results, key=lambda x: x["confidence"], reverse=True)[0]

        # FIX 2: Populate details with ensemble routing data
        details = {
            "liveness_label": best["label"],  # ← Ensemble reads this for routing
            "peak_hz": best["metrics"].get("forehead", {}).get("peak_hz", 0.0),
            "spectral_concentration": best["metrics"].get("forehead", {}).get("spectral_concentration", 0.0),
            "faces_analyzed": len(face_results),
        }

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=best["score"],
            confidence=best["confidence"],
            details=details,  # FIX 2: Structured data for ensemble
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary=best["interpretation"],
        )