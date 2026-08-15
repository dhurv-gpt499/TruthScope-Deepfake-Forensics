# Aegis-X Dual-Pipeline Architecture Design

## Overview
Aegis-X leverages a dynamic dual-pipeline architecture (Face Pipeline vs. No-Face Pipeline) dynamically orchestrated by the `ForensicAgent`.

## Pipeline Segment A: CPU Phase
The CPU Phase executes immediately after preprocessing. It aims to rapidly identify clear signs of authenticity or structural synthetic instability using minimal resources.

- **Tools Run:**
  - `check_c2pa`: Validates cryptographic signatures. If AI-generation metadata is found (e.g., IPTC `trainedAlgorithmicMedia`, Gemini/Midjourney software agent strings), issues an immediate **FAKE** verdict. If authentic hardware capture is confirmed, issues an immediate **REAL** verdict.
  - `run_dct`: Always runs. Uses face crops when available, **falls back to raw image** (resized to 224×224) when no faces are detected.
  - **If Face Detected:**
    - `run_rppg`: Runs on the `face_window` (contiguous frame bounds) to extract biological signals unless `INSUFFICIENT_TEMPORAL_DATA` is flagged.
    - `run_geometry`, `run_illumination`, `run_corneal`: Configured dynamically using preprocessing heuristics fallbacks (`FACE_TOO_SMALL`, `LOW_LIGHT`, `MOTION_BLUR`, `OCCLUSION`).

- **CPU Tool Role:** CPU tools are **supporters only** — they contribute to the weighted average but cannot unilaterally override GPU specialist verdicts. Their weights are intentionally low (0.04–0.08) compared to GPU specialists (0.10–0.25).

## Pipeline Segment B: CPU->GPU Gate
After the CPU Phase, the system dynamically routes execution via Unison and Confidence Aggregation Rules to manage VRAM utilization optimally:

- **HALT**: (CPU Confidence > 0.93 AND Unison Agreement across domains). System skips GPU tools completely.
- **MINIMAL_GPU**: (CPU Confidence 0.80-0.93). Runs `run_univfd` only.
- **FULL_GPU**: (CPU Confidence < 0.80 or Unison disagreement). Entire GPU stack runs.

## Pipeline Segment C: GPU Phase
The GPU Phase processes deep-learning features. All network executions utilize strict `run_with_vram_cleanup` execution blocks sequentially.

- **For No-Face Media**: `run_freqnet` -> `run_univfd` -> `run_xception`
- **For Face Media**: `run_freqnet` -> `run_univfd` -> `run_xception` -> `run_sbi`

### Raw Image Fallback (No-Face Pipeline)
When `tracked_faces` is empty, all GPU tools fall back to analyzing the **raw image** loaded from `media_path`:

| Tool | Fallback Behavior |
|---|---|
| `run_freqnet` | Loads raw image, resizes to 224×224, runs CNNDetect + FAD frequency analysis |
| `run_univfd` | Loads raw image, resizes to 224×224, runs CLIP-ViT-L/14 + linear probe |
| `run_xception` | Loads raw image (original resolution), `_prepare_tensor` resizes to 299×299 internally |
| `run_dct` | Loads raw image, resizes to 224×224, runs DCT double-quantization analysis |
| `run_sbi` | **Not run** in no-face pipeline (requires face crops by design) |

This ensures no-face media (landscapes, objects, AI art without people) receives full GPU forensic coverage instead of all tools erroring with "No tracked faces found."

## Pipeline Segment D: Ensemble Scoring (v4.0 — Three-Pronged Anomaly Detection)

The ensemble aggregator applies a multi-layered scoring algorithm to prevent false negatives:

### Prong 1 — Suspicion Overdrive (Hard Max-Pool)
If any GPU specialist's implied fake probability exceeds `SUSPICION_OVERRIDE_THRESHOLD` (0.70):
- **GPU Conflict Check**: Before max-pooling, verify that GPU specialists don't contradict each other (spread > 0.30). If FreqNet says 70% fake but UnivFD says 30% fake, this is a noisy false positive (e.g., disco lights confusing frequency analysis). In this case, fall back to the weighted average.
- **Unanimous Override**: If GPU specialists agree (spread ≤ 0.30), the max detected fake probability becomes the ensemble's fake_score directly.

### Prong 2 — Borderline Consensus Detection
When ≥2 GPU specialists independently cluster in the "borderline zone" (0.35–0.55 fake probability):
- Their mean fake probability is boosted by `BORDERLINE_CONSENSUS_BOOST` (1.25×) to reflect that corroborated weak signals are statistically significant.
- Example: XceptionNet=49% + UnivFD=44% → mean=46.5% → boosted=58.1% fake.

### Prong 3 — GPU Coverage Degradation
When GPU specialists blind-spot out (e.g., SBI returns 0% fake and gets filtered by `SBI_BLIND_SPOT_THRESHOLD`), the system recognizes it has less evidence:
- Each abstained GPU specialist applies a `GPU_COVERAGE_DEGRADATION_FACTOR` (0.10) multiplicative boost to the fake_score.
- Disabled when GPU conflict is already detected (to prevent double-penalization).

### Abstention Handling
- Tools with `confidence = 0` are displayed as `[ABSTAINED] N/A` in both the UI and the LLM prompt.
- The LLM receives explicit instructions: `"Do NOT assume authenticity"` for abstained tools.
- Abstentions never contribute to the weighted average.

## Safety Routing
- `run_web.py` no longer rejects no-face media — it forwards all media to `ForensicAgent.analyze()` which handles routing internally.
- `run_rppg` is guarded by dual safety mechanisms checking video bounds and fallback MediaPipe validation if upstream preprocessing falters.
- Exceptions explicitly route as `ERROR` and degrade the ensemble confidence safely.
- C2PA tool recursively scans all historical manifests (including nested ingredients) for AI-generation indicators before issuing verdicts.
