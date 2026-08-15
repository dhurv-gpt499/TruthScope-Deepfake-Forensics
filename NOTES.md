# Aegis-X — Complete Repository Documentation (allfiles.md)

> Generated automatically. Includes full README, every file, every class, every function —
> what it does, what it **cannot** do, and how to counter its limitations.

---

## Part 1 — Original README

# Aegis-X

Aegis-X is a **Python** repository that contains the building blocks for a media/deepfake forensics pipeline, including:
- A tool interface (`BaseForensicTool`) and a tool registry (`ToolRegistry`)
- Multiple forensic tools (C2PA, rPPG, DCT-based analysis, geometry, illumination, corneal/catchlight checks, SBI, FreqNet, SigLIP adapter)
- An “agent” orchestration layer and early-stopping logic
- Utilities for preprocessing, video/image I/O, VRAM/device management, logging, and an Ollama client

> Note: This repository currently includes the **core pipeline modules and tools**, but it does **not** include runnable app entrypoints like `main.py` / Streamlit / Gradio scripts at the repo root.

---

## Repository Contents (Current)

Top-level files/directories currently present:

- `.env.example`
- `.gitignore`
- `NOTES.md`
- `README.md`
- `__init__.py`
- `day18_test_results.txt`
- `diagnostics_day14.py`
- `pyproject.toml`
- `requirements.txt`
- `core/`
- `utils/`
- `downloads/`
- `test_data/`

---

## Installation

### Requirements
- Python **>= 3.10** (as specified in `pyproject.toml`)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment variables
An example environment file is provided at `.env.example`. Common variables include:

- `AEGIS_MODEL_DIR` (default example: `models/`)
- `AEGIS_DEVICE` (example: `auto`)
- `OLLAMA_ENDPOINT` (example: `http://localhost:11434`)
- `AEGIS_VRAM_THRESHOLD` (example: `3.5`)

Copy it to `.env` and adjust as needed:

```bash
cp .env.example .env
```

---

## Project Structure

### `core/`
Core functionality:

- `core/base_tool.py`  
  Defines the base tool contract used by forensic tools.

- `core/tools/`  
  Contains tool implementations and the registry:
  - `registry.py`
  - `c2pa_tool.py`
  - `rppg_tool.py`
  - `dct_tool.py` (+ `dct_tool2.py`)
  - `geometry_tool.py`
  - `illumination_tool.py`
  - `corneal_tool.py`
  - `siglip_adapter_tool.py`
  - `sbi_tool.py`
  - `freqnet_tool.py`
  - `freqnet/` (subpackage used by FreqNet tool)

- `core/agent.py`  
  Orchestration logic (agent-style coordination of tools).

- `core/early_stopping.py`  
  Early stopping / gating logic to reduce unnecessary computation.

- `core/forensic_summary.py`  
  Produces summaries of tool outputs (useful for downstream reporting/LLM prompting).

- `core/config.py`, `core/data_types.py`, `core/exceptions.py`  
  Configuration/data types and custom exceptions.

### `utils/`
Utility modules used across the system:

- `utils/preprocessing.py`  
  Preprocessing helpers used before tool execution.

- `utils/video.py`, `utils/image.py`  
  Video and image IO helpers.

- `utils/vram_manager.py`  
  Device/VRAM management helpers.

- `utils/ensemble.py`  
  Ensemble/scoring utilities.

- `utils/thresholds.py`  
  Central place for thresholds/weights/constants used by scoring and tools.

- `utils/ollama_client.py`  
  HTTP client helpers for interacting with an Ollama server.

- `utils/logger.py`  
  Logging helpers.

### `downloads/`
Download helpers:
- `downloads/download_model.py`
- `downloads/download_freqnet_models.py`
- `downloads/siglip.py`

### `test_data/`
SQLite databases used for testing/development:
- `test_data/sig_test.db`
- `test_data/import_test.db`

---

## Tools Included

The repository contains implementations for multiple forensic tools under `core/tools/`, including:

- C2PA provenance checking (`c2pa_tool.py`)
- rPPG-based liveness analysis (`rppg_tool.py`)
- DCT/frequency-based analysis (`dct_tool.py`, `dct_tool2.py`)
- Landmark/geometry-based checks (`geometry_tool.py`)
- Illumination consistency checks (`illumination_tool.py`)
- Corneal/catchlight analysis (`corneal_tool.py`)
- A SigLIP adapter-based model tool (`siglip_adapter_tool.py`)
- SBI tool (`sbi_tool.py`)
- FreqNet tool (`freqnet_tool.py` + `core/tools/freqnet/` package)

A registry is provided in `core/tools/registry.py` to manage tool construction and execution.

---

## Configuration Notes

- There is **no** `config.yaml` currently in the repository root.  
- Configuration appears to be driven by Python modules (e.g., `core/config.py`, `utils/thresholds.py`) and environment variables (see `.env.example`).

---

## Development Notes

- `NOTES.md` contains additional project notes.
- `diagnostics_day14.py` and `day18_test_results.txt` provide diagnostics and recorded test outputs.

---

## License

No `LICENSE` file is currently present in the repository.  
If you intend the project to be open-source, consider adding a license file (MIT/Apache-2.0/GPL/etc.) and updating this section accordingly.


---

## Part 2 — Core Infrastructure

### File: `core/__init__.py`
Package initializer. Exports the core module namespace.
**Cannot do**: Nothing on its own. It is a marker file.

---

### File: `core/data_types.py`
Defines the standard data contracts used by every tool in the pipeline.

#### `class ToolResult`
The **universal output payload** returned by every forensic tool.

| Field | Type | Purpose |
|-------|------|---------|
| `tool_name` | str | Name of the tool that produced this result |
| `success` | bool | Whether the tool ran successfully |
| `score` / `fake_score` | float 0–1 | Deepfake probability (1.0 = definitely fake) |
| `confidence` | float 0–1 | How confident the tool is in its score |
| `details` | dict | Tool-specific breakdown data for the LLM |
| `error` | bool | Whether an error occurred |
| `error_msg` | str | Error description if `error=True` |
| `execution_time` | float | Seconds taken to run inference |
| `evidence_summary` | str | Human-readable 1-sentence summary |

**`def score(self)`** *(property)* — Alias for `fake_score` for backward compatibility.

> **Limitation**: `ToolResult` is a flat dataclass. It cannot carry nested multi-face per-frame timeseries natively.
> **Counter**: The `details` dict is open-ended and stores all nested info as JSON-safe dicts.

---

### File: `core/exceptions.py`
Custom exception hierarchy for granular error handling.

| Class | Inherits | Raised When |
|-------|----------|-------------|
| `AegisError` | `Exception` | Base: any Aegis-X internal error |
| `ModelLoadError` | `AegisError` | AI model file missing or corrupted |
| `PreprocessingError` | `AegisError` | MediaPipe/frame extraction fails |
| `ToolExecutionError` | `AegisError` | Fatal error inside a tool's inference |

> **Limitation**: Does not include network errors (for Ollama/C2PA). Those fall back to `AegisError`.
> **Counter**: Wrap Ollama/C2PA calls in specific `try/except` with descriptive messages.

---

### File: `core/config.py`
Master configuration system using typed Python dataclasses.

#### Classes
- **`ModelPaths`** — File paths to all model weights (FreqNet, SBI, SigLIP).
- **`AgentConfig`** — LLM timeout, max retries, Ollama model name.
- **`EnsembleWeights`** — Weights for each tool in the final scoring.
- **`ThresholdConfig`** — REAL/FAKE verdict boundaries.
- **`PreprocessingConfig`** — Face crop sizes, frame rates, max subjects.
- **`AegisConfig`** — Master class that groups all the above.

> **Limitation**: No hot-reload. Config is read at startup. Changing `.env` requires restart.
> **Counter**: Not yet countered. Plan: Add `watchdog` file watcher in future sprint.

---

### File: `core/base_tool.py`
Abstract base class that all forensic tools **must** extend.

#### `class BaseForensicTool(ABC)`

| Method | What it does |
|--------|-------------|
| `tool_name` *(abstract property)* | Returns the canonical tool ID string used by registry |
| `setup()` *(abstract)* | Load model weights, verify paths, prepare for inference |
| `_run_inference(input_data)` *(abstract)* | Core detection algorithm — pure tool logic |
| `execute(input_data)` *(concrete)* | **Firewall wrapper**: calls `_run_inference`, catches all exceptions, guarantees a `ToolResult` is always returned even on crash |

> **Limitation**: `execute()` catches ALL exceptions, meaning silent tool failures become `score=0.0` abstentions.
> **Counter**: Countered via the `error=True` flag and `error_msg` — the agent/ensemble checks these fields.

---

### File: `core/agent.py`
The main **orchestration brain** of Aegis-X. Coordinates CPU → GPU → Ensemble → LLM.

#### `class AgentEvent`
Lightweight progress event for real-time UI streaming via Python generators.
- `event_type`: `"tool_start"`, `"tool_complete"`, `"early_stop"`, `"llm_start"`, `"verdict"`
- `tool_name`: Which tool triggered the event
- `data`: Dict with score, confidence, evidence_summary

#### `class ForensicAgent`

| Method | What it does | Cannot do | Counter |
|--------|-------------|-----------|---------|
| `__init__(config)` | Initializes registry, ensemble aggregator, early stopping controller | Cannot dynamically add new tools at runtime | Re-instantiate agent after registry update |
| `_run_cpu_phase(preprocess_result)` | Runs all CPU tools sequentially; short-circuits entire pipeline if C2PA verified; gates on confidence threshold | Cannot run CPU tools in parallel (by design — sequential for determinism) | Acceptable tradeoff; parallelism would complicate early stopping |
| `_run_gpu_phase(preprocess_result)` | Runs GPU tools one-by-one with VRAM cleanup; skips SBI if SigLIP score >0.70 | Cannot run multiple GPU tools simultaneously | By design — 4GB VRAM constraint |
| `analyze(preprocess_result)` | Full pipeline: CPU → GPU → Ensemble → LLM synthesis. Yields `AgentEvent` for each step | Cannot process multiple media files in parallel | Caller must loop; agent is stateless per call |

---

### File: `core/early_stopping.py`
Implements **Evidential Subjective Logic** to decide when enough tools have run.

#### `class StopReason(Enum)`
Six possible reasons for stopping or continuing:
- `CONTINUE_AMBIGUOUS` — Not enough evidence yet
- `CONTINUE_ADVERSARIAL_CONFLICT` — Tools disagree too much (adversarial risk)
- `CONTINUE_SECURITY_REQUIRED` — High confidence but high-trust tool not yet run
- `HALT_C2PA_HARDWARE_SIGNED` — Cryptographic proof = real
- `HALT_LOCKED_FAKE` — Mathematically impossible for remaining tools to flip verdict
- `HALT_LOCKED_REAL` — Mathematically impossible for remaining tools to flip verdict

#### `class EarlyStoppingController`

| Method | What it does | Cannot do | Counter |
|--------|-------------|-----------|---------|
| `__init__(...)` | Pre-computes high-trust tool set, loads registry weights | Cannot reconfigure without reinstantiation | Singletons via `get_registry()` |
| `evaluate(tool_scores, completed_tools, c2pa_hw_verified)` | Runs 9-step logic: C2PA lock → empty check → validation → weighted mean → evidential conflict → math bounds → locked REAL → locked FAKE → default continue | Cannot detect adversarial manipulation of tool scores without conflict check | Conflict ratio check at step 5 guards against this |

> **Key insight**: The conflict threshold (default 0.35) means if tools disagree strongly, early stopping is **blocked** — this is the anti-adversarial safeguard.

---

### File: `core/forensic_summary.py`
Converts raw tool scores into a structured Phi-3 Mini prompt.

#### `def build_phi3_prompt(ensemble_score, tool_results, verdict)`
- **Does**: Formats each tool's evidence_summary + score into a grounded natural language prompt. Every LLM claim is anchored to a specific tool output.
- **Cannot do**: Generate explanations without tool outputs — it only summarizes what tools found.
- **Counter**: If a tool abstained, its entry is omitted from the prompt to avoid hallucination.

---

### File: `core/tools/registry.py`
**Single source of truth** for all tool metadata and runtime instances.

#### Tool Manifest (v4.0 — Decider/Supporter Hierarchy)

| Tool | Weight | Category | Role | Trust Tier |
|------|--------|----------|------|------------|
| `check_c2pa` | 0.05 | PROVENANCE | Gate | 2 |
| `run_rppg` | 0.06 | BIOLOGICAL | Supporter | 2 |
| `run_dct` | 0.04 | FREQUENCY | Supporter | 1 |
| `run_geometry` | 0.08 | GEOMETRIC | Supporter | 1 |
| `run_illumination` | 0.04 | FREQUENCY | Supporter | 1 |
| `run_corneal` | 0.04 | BIOLOGICAL | Supporter | 1 |
| `run_univfd` | 0.22 | SEMANTIC | Decider | 3 |
| `run_sbi` | 0.25 | GENERATIVE | Decider | 3 |
| `run_xception` | 0.15 | SEMANTIC | Decider | 2 |
| `run_freqnet` | 0.10 | FREQUENCY | Decider | 1 |

> **v4.0 change**: SigLIP adapter replaced by UnivFD (CLIP-ViT-L/14). GPU tools (Deciders) control ensemble verdict. CPU tools (Supporters) inform but cannot override. All GPU tools fall back to raw image analysis when no faces detected.

#### `class ToolRegistry`

| Method | What it does | Cannot do | Counter |
|--------|-------------|-----------|---------|
| `_register_all()` | Imports, instantiates, and calls `setup()` on all tools. Isolates each — one crash doesn't block others | Cannot hot-reload new tools | Call `reset_registry()` and reinstantiate |
| `execute_tool(name, input_data)` | Dispatches to tool, handles GPU OOM with one retry + cache clear | Cannot recover from double OOM | Tool returns error ToolResult; ensemble skips it |
| `get_viable_pending_tools(completed_tools)` | Returns registered tools not yet executed | Does not know which tools are GPU-only | Use `tool.requires_gpu` flag |
| `get_high_trust_tools()` | Returns tier-3 tools (`run_geometry`, `run_sbi`) | — | Required by early stopping for safety gate |
| `get_health_report()` | Full diagnostics: active, failed, call counts, total time | — | Use for monitoring dashboards |
| `shutdown()` | Calls cleanup on all tools, clears GPU cache | — | Called on application exit |


---

## Part 3 — CPU Forensic Tools (Zero VRAM)

These tools run purely on CPU, require no model weights, and execute first in the pipeline.

---

### File: `core/tools/c2pa_tool.py` — C2PA Provenance Tool {#c2pa}

#### What It Does
Reads **Content Credentials** (C2PA standard) embedded in image/video metadata.
If a camera or creation tool signed the file cryptographically, C2PA extracts the signer and timestamp.

#### `class C2PATool(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"check_c2pa"` |
| `setup()` | Attempts `import c2pa`; sets `_c2pa_available` flag |
| `_run_inference(input_data)` | Reads C2PA manifest from `media_path`; extracts `signer`, `timestamp`, `c2pa_verified`. Tries both new API (`read_file`) and legacy API (`Reader`) |
| `_no_c2pa_result(start_time)` | Returns a graceful abstention when no C2PA data is embedded |

#### ✅ What It CAN Detect
- Files signed by **hardware cameras** (Sony, Canon, Nikon with C2PA firmware)
- Files created by **Adobe tools** (Photoshop, Lightroom with Content Credentials)
- Files from **AI image generators** that *voluntarily* add C2PA (e.g. Adobe Firefly, Stable Diffusion with provenance plugin)

#### ❌ What It CANNOT Detect
- Files with **no C2PA metadata** — most deepfakes have none; tool simply abstains (score=0.0, confidence=0.0)
- **Stripped metadata** — adversary can remove EXIF/XMP/C2PA with a single `exiftool -all= image.jpg` command
- **Replayed C2PA** — copying a real C2PA manifest onto a fake file (requires cryptographic verification beyond basic field reading)
- **Unverified signers** — the tool only checks if a signer field exists, not if the certificate chain is trusted

#### 🔧 Counter / Mitigation
- **Already partially countered**: `c2pa_verified=True` triggers a **short-circuit** that skips remaining tools (saves compute). This is correct — if hardware-signed, trust it.
- **Not yet countered**: Certificate chain validation. Future sprint must add `c2pa.verify()` call.
- **Weight is low (0.05)**: Ensemble treats C2PA as soft evidence, not gospel.

---

### File: `core/tools/dct_tool.py` — DCT Frequency Analysis Tool {#dct}

#### What It Does
Detects **JPEG double-quantization artifacts** — the "fingerprint" left when an image is compressed, edited, then re-compressed. GAN-generated images have an unnaturally smooth DCT coefficient distribution.

#### `class DCTTool(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"run_dct"` |
| `setup()` | No-op (pure math, no models) |
| `_coerce_to_uint8(frame)` | Normalizes float images to 0–255 uint8 |
| `_to_gray(crop)` | RGB→grayscale via ITU-R BT.601 luminance formula |
| `_compute_video_hash(frames)` | MD5 of first 100×100 pixels for grid-alignment cache key |
| `_find_optimal_grid(gray, video_hash)` | Tries all 8×8 grid offsets (0–7 in both axes = 64 combinations); picks the one that maximizes the secondary autocorrelation peak. Caches result per video hash |
| `_compute_peak_ratio(gray, dy, dx)` | With known grid alignment: computes DCT of all 8×8 blocks; builds histogram of AC coefficients; autocorrelates; returns secondary/primary peak ratio |
| `_score_from_ratio(peak_ratio)` | Linear mapping: low ratio → 0.0 (real), high ratio → 1.0 (fake) |
| `_confidence_from_score(score)` | Confidence = min(cap, score + bump) — low-score results get low confidence |
| `_abstain(start_time, reason)` | Returns graceful abstention when no valid face crops |
| `_run_inference(input_data)` | Orchestrates: get crops → find optimal grid → compute peak ratios for all faces → average → score |

#### ✅ What It CAN Detect
- Images that were **JPEG-compressed twice** (editing artifacts)
- **GAN artifacts** that create unnaturally periodic DCT coefficient distributions
- **Spliced/composited** images where the inserted region has different compression history

#### ❌ What It CANNOT Detect
- **PNG or lossless images** — no JPEG quantization grid exists; tool scores near 0
- **Heavily compressed genuine images** (low quality JPEG) — may produce false positives
- **AI images saved directly as PNG** — no DCT artifacts
- **Post-processed GAN outputs** (denoised, upsampled) — artifacts may be smoothed out

#### 🔧 Counter / Mitigation
- **Compression discount in ensemble** (`DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD`): if DCT peak is very high, SBI and FreqNet weights are also reduced (they share the blind spot)
- **Grid caching** prevents re-scanning all 64 offsets per frame in videos (performance fix)
- ❗ **Not fully countered**: PNG/lossless blind spot. Future sprint: add separate PNG artifact detector

---

### File: `core/tools/geometry_tool.py` — Anthropometric Geometry Tool {#geometry}

#### What It Does
Checks **7 anatomical ratios** of the human face using MediaPipe's 478 facial landmarks. Real human faces obey consistent proportional rules that GAN/diffusion models frequently violate.

#### `class GeometryTool(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"run_geometry"` |
| `setup()` | Loads yaw skip threshold from config |
| `_dist(a, b)` | Euclidean distance between two 2D landmark points |
| `_safe_divide(n, d)` | Division with epsilon denominator guard |
| `_soft_check(value, low, high, name)` | Returns (passed, severity 0–1, detail). Severity = how far outside valid range |
| `_check_ipd_ratio(lm, face_width)` | **Check 1**: Inter-Pupillary Distance / jaw width. Very stable in real humans (weight: 2.0) |
| `_check_philtrum_ratio(lm)` | **Check 2**: Philtrum length / lower face length (weight: 1.5) |
| `_check_yaw_proxy(lm, face_width)` | **Gate**: Eye midpoint offset from nose tip. If >threshold, bilateral checks are skipped (not scored) |
| `_check_eye_asymmetry(lm, face_width)` | **Check 3**: Left vs right eye width asymmetry. Skipped at high yaw (weight: 1.0) |
| `_check_nose_width_ratio(lm, ipd)` | **Check 4**: Nose alar width / IPD. GANs frequently fail here (weight: 1.5) |
| `_check_mouth_width_ratio(lm, ipd)` | **Check 5**: Mouth width / IPD (weight: 1.0) |
| `_check_vertical_thirds(lm)` | **Check 6**: Upper/mid/lower face thirds must be roughly equal. Uses bounding box top (not hairline landmark) (weight: 2.0) |
| `_check_landmark_stability(trajectory, lm)` | **Bonus (video)**: IPD ratio variance across frames. Stub — needs multi-frame implementation |
| `_calculate_confidence(checks, yaw, face_width)` | Dynamic confidence: fewer checks = lower; high yaw = lower; small face = lower |
| `_weighted_score(violations, severities)` | Weighted sum of severe violations / MAX_WEIGHT |
| `_run_inference(input_data)` | Processes all tracked faces; returns worst-face result |

#### ✅ What It CAN Detect
- **GAN face generation** (StyleGAN, DALL-E, Midjourney) — wrong IPD, wrong nose width
- **Full-face synthesis** — all proportions must be correct simultaneously
- **Faces at roughly frontal pose** (yaw < threshold)

#### ❌ What It CANNOT Detect
- **Oblique poses** (high yaw) — bilateral checks are safely skipped, confidence drops
- **Face swaps that preserve geometry** — if only texture is swapped, ratios stay real
- **Small faces** (<80px face width) — MediaPipe landmarks are too noisy
- **Professional deepfakes** trained specifically to mimic anthropometric ratios

#### 🔧 Counter / Mitigation
- ✅ **Soft severity scoring** (V2): Instead of binary pass/fail, how far outside the ratio matters. Partially counters edge cases.
- ✅ **Weighted violations**: IPD and vertical thirds count more than eye asymmetry.
- ✅ **Dynamic confidence**: Low confidence on unreliable inputs rather than false strong signal.
- ❗ **Landmark stability check** is a stub. Needs multi-frame landmarks to be fully countered for video.
- ❗ **Face swap blind spot** not countered here — countered by SBI tool (which detects blend boundaries).

---

### File: `core/tools/illumination_tool.py` — Illumination Consistency Tool {#illumination}

#### What It Does
Compares the **dominant light direction** on the face against the **scene background** (shoulders/neck area). AI generators frequently composite a face from one lighting environment into a scene with different directional lighting.

#### `class IlluminationTool(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"run_illumination"` |
| `setup()` | Loads thresholds from config |
| `_extract_luma(rgb_img)` | Converts RGB → YCrCb, returns Y (luma) channel |
| `_safe_divide(n, d)` | Division with epsilon guard |
| `_extract_scene_context(frame, bbox)` | Extracts region **below** the face bbox in the original full frame (neck/shoulders area). Fixed in V3 to use original frame, not face crop |
| `_compute_gradient_direction(luma)` | Sobel operators → weighted mean gradient → angle (0–360°) and magnitude |
| `_check_shadow_highlight_consistency(luma, midpoint_x)` | Top 20% brightness = highlights, bottom 20% = shadows. Checks if they're on opposite sides — they should be for consistent lighting |
| `_calculate_confidence(face_grad, sharpness, face_width)` | Dynamic: strong gradient = high confidence; blurry face = lower; small face = lower |
| `_run_inference(input_data)` | Extracts face luma + scene luma; computes gradients; checks if dominant direction matches; applies mismatch penalty; returns worst face |

#### ✅ What It CAN Detect
- **AI-composited faces** (Midjourney, Stable Diffusion) where face has left-lit lighting but background is right-lit
- **Green screen / chroma-key composites** with mismatched lighting
- **GAN faces pasted onto stock photos**

#### ❌ What It CANNOT Detect
- **Diffuse lighting** (overcast, studio soft-box) — tool correctly abstains (score=0.0, confidence=0.0)
- Face images **without scene context** (cropped headshots with plain background) — can't extract scene luma
- **Skilled compositing** where the artist matches lighting direction exactly
- **Indoor portraits** with complex multi-source lighting

#### 🔧 Counter / Mitigation
- ✅ **Diffuse abstention (V3 fix)**: Diffuse lighting no longer votes REAL — it abstains. This prevents false confidence.
- ✅ **Shadow-highlight consistency check**: Added as secondary signal.
- ✅ **Context from original frame (V3 fix)**: Uses neck/shoulder area of full frame, not face crop chin. Much more reliable.
- ❗ **Multi-source lighting not countered**: Only left/right dominant direction is checked. Full 2D gradient angle is computed but not fully utilized yet.

---

### File: `core/tools/corneal_tool.py` — Corneal Reflection Tool {#corneal}

#### What It Does
Checks **specular highlights (catchlights)** in both eyes. Under the same light source, the catchlight should appear at the same relative position in both irises. Diffusion models frequently hallucinate inconsistent catchlights.

#### `class CornealTool(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"run_corneal"` |
| `setup()` | Loads box size (15px), max divergence, consistency threshold |
| `_transform_to_crop_coords(landmark, bbox, crop_size)` | Converts full-frame landmark coordinates → 224×224 face crop pixel space |
| `_extract_iris_roi(face_crop, iris_landmark)` | Extracts 15×15 pixel box centered on iris landmark (nodes 468, 473). Rejects if landmark outside eye band (y: 50–140 in 224px crop) |
| `_detect_catchlight_centroid(iris_roi)` | Finds specular highlight in iris ROI. Requires absolute brightness >180/255. Uses connected components to find the largest bright blob. Normalizes centroid offset to [-1, +1] |
| `_run_inference(input_data)` | For each face: extract both iris ROIs → detect catchlights → compute divergence between left and right offsets → score |

#### ✅ What It CAN Detect
- **Midjourney / DALL-E / Stable Diffusion** images — these models are notoriously bad at consistent catchlights
- **Low-quality deepfakes** where the face swap doesn't preserve the original catchlight positions

#### ❌ What It CANNOT Detect
- **Eyes with no catchlights** (indoor dim lighting, sunglasses) — tool correctly abstains
- **Single catchlight visible** (one eye occluded) — tool correctly abstains (V2 fix)
- **Skilled deepfakes** that happen to produce consistent catchlights by chance (~30% of cases)
- **High-quality diffusion outputs** (FLUX.1, SD 3.5) which have improved catchlight consistency

#### 🔧 Counter / Mitigation
- ✅ **Absolute brightness threshold (V2)**: Catchlights must be genuinely bright (>180/255), not just the brightest pixel in a dark iris.
- ✅ **Single-catchlight abstention (V2)**: No longer votes fake when only one eye has a catchlight. Prevents false positives on occluded eyes.
- ✅ **Connected components (V2)**: Handles multi-blob catchlights (multiple light sources).
- ❗ **Not countered**: Diffusion models getting better. Future: train a small CNN on catchlight position maps.


---

## Part 4 — GPU Forensic Tools (Require CUDA)

These tools use neural networks and run after the CPU phase, only if early stopping hasn't triggered.

---

### File: `core/tools/rppg_tool.py` — Remote Photoplethysmography Tool {#rppg}

#### What It Does
Detects **biological pulse signals** by analyzing microscopic color changes in facial skin caused by blood flow. Static images and most AI-generated videos have no heartbeat signal.

> ⚠️ **Photo Note**: rPPG is **automatically skipped** for static images (`media_type == "image"`). It is purely a **video tool**.

#### `class RPPGTool(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"run_rppg"` |
| `setup()` | Sets debug flag |
| `_extract_roi(frame, bbox, relative_box)` | Extracts a region of interest (forehead/cheek) from a frame using relative coordinates within the face bbox |
| `_get_facial_rois(landmarks)` | Returns 3 ROI boxes: forehead, left cheek, right cheek — using MediaPipe landmark indices |
| `_check_hair_occlusion(roi)` | Laplacian variance check: high texture = hair covering forehead = can't measure pulse |
| `_extract_pos_signal(frames, trajectory, roi)` | POS (Plane-Orthogonal-to-Skin) algorithm: extracts 1D pulse signal from RGB means across all frames. Normalizes the signal |
| `_calculate_signal_metrics(signal_1d, fps)` | FFT analysis: finds peak frequency in 0.7–3.5Hz cardiac band; calculates SNR in dB; spectral concentration (peak / median power ratio) |
| `_evaluate_liveness(h_forehead, h_left, h_right, stds, hair)` | Decision logic: if ≥2 ROI signals show a coherent cardiac peak (within 0.1Hz) → `PULSE_PRESENT`. If signals are flat → `NO_PULSE`. If incoherent → `INCOHERENT` (spoof) |
| `_run_inference(input_data)` | Orchestrates per-face rPPG; returns best face result |

#### ✅ What It CAN Detect
- **Purely GAN-generated video** (no heartbeat in generated pixels)
- **Static image rendered as video** (all pixel values constant)
- **Screen recordings of deepfake videos** played from a phone/monitor
- **Very low-quality face swaps** that break temporal color consistency

#### ❌ What It CANNOT Detect (Photo deepfakes — N/A, Video only)
- **Needs ≥90 frames** (3 seconds at 30fps) — short clips abstain
- **Hair over forehead** — abstains due to occlusion
- **Advanced temporal deepfakes** (e.g. FaceSwap with temporal smoothing) that preserve some color variation
- **Real video with poor skin visibility** (dark skin under yellow light may produce weak signal)

#### 🔧 Counter / Mitigation
- ✅ **Hair occlusion guard** — detects and abstains rather than producing wrong signal
- ✅ **Coherence check across 3 ROI regions** — requires BOTH forehead AND cheeks to agree
- ✅ **Static image skip** — correctly returns score=0.0 for images
- ❗ **Temporal deepfakes with residual color variation** — not fully countered. Future: use motion-compensated rPPG

---

### File: `core/tools/siglip_adapter_tool.py` — SigLIP Forensic Adapter {#siglip}

#### What It Does
Runs 6 multi-scale crops through a **frozen Google SigLIP-Base-Patch16-224 backbone**, then uses a **trainable forensic adapter** (Dynamic Spatial Pooling + Cross-Patch Attention) to score them for deepfake probability. This is the highest-weight semantic signal tool.

#### Architecture: 3 Stages
1. **Stage 1**: Extract features from 6 crops using frozen SigLIP backbone (hooks on layers 3, 6, 9, 11). Each crop produces (4 layers × 196 tokens × 768 dims).
2. **Stage 2**: Dynamic Spatial Pooling — learned query vectors attend over 196 spatial tokens per layer; layer fusion via learned weights → (6 crops × 768 dims)
3. **Stage 3**: Cross-Patch Attention — 6 crops attend to each other (captures cross-region inconsistencies); Zero-init residual for training stability
4. **Stage 3.5**: Scoring head (768→128→1) + Log-Sum-Exp pooling on logits → final score

#### `class SiglipForensicAdapter(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"run_siglip_adapter"` |
| `setup()` | Calls `_load_model()` |
| `_load_model()` | Loads SigLIP backbone from local `models/siglip-base-patch16-224/` (FP16, frozen). Registers forward hooks on layers 3,6,9,11 |
| `_register_hooks(layer_indices)` | Attaches `forward_hook` functions that capture hidden states from specified encoder layers |
| `_preprocess_crop(crop)` | Resizes to 224×224, normalizes to [-1,1] (manual, no HF Processor — preserves forensic artifacts) |
| `_extract_features(crops)` | Sequential per-crop backbone forward pass; stacks layer hook outputs → (6,4,196,768) |
| `_stage2_dynamic_pool(x)` | Einsum-based query attention over tokens per layer, then layer fusion → (6,768) |
| `_stage3_cross_attention(x)` | Multi-head attention (4 heads, 32 dim Q/K, 768 dim V) across 6 crops → (6,768) |
| `_stage35_score_lse(x)` | Score head → logits; LSE pooling with learnable temperature β; single sigmoid at end |
| `_forward_single_pass(crops)` | Chains stages 1→2→3→3.5 → scalar score |
| `run(crops)` | **TTA**: runs original + horizontally flipped crops; max-pools scores. Synchronizes VRAM between passes |
| `unload()` | Moves backbone to CPU, removes hooks, calls `gc.collect()`, clears CUDA cache |
| `_run_inference(input_data)` | Extracts face crops from tracked_faces; calls run(); returns ToolResult |

#### ✅ What It CAN Detect
- **Diffusion model outputs** (Stable Diffusion, FLUX, Midjourney) — very strong signal
- **GAN faces** (StyleGAN, InsightFace) — SigLIP features capture unnatural texture
- **Fully synthetic images** — highest weight tool (0.17)

#### ❌ What It CANNOT Detect
- **Trained adversarially against SigLIP** (adversarial examples targeting the ViT backbone)
- **Real images with unusual textures** (heavy makeup, prosthetics) may score false positives
- **Requires GPU** — falls back to abstention on CPU-only machines
- **Needs trained adapter weights** — without custom fine-tuning, adapter is random-init

#### 🔧 Counter / Mitigation
- ✅ **TTA (Test-Time Augmentation)** with horizontal flip reduces single-view bias
- ✅ **Mixed precision (FP16 backbone / FP32 adapter)** prevents NaN in LSE pooling
- ✅ **Strict offline mode** (`local_files_only=True`) — no internet calls at inference
- ❗ **Adapter not trained** (random init) = weak signal until fine-tuned on deepfake dataset. This is the biggest gap.

---

### File: `core/tools/sbi_tool.py` — Self-Blended Images Tool {#sbi}

#### What It Does
Detects **face-swap blend boundaries** — the seam where a fake face was composited onto a real head. Uses EfficientNet-B4 with GradCAM to localize which facial region has the artifact.

#### `class SBITool(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"run_sbi"` |
| `setup()` | Logs setup complete |
| `_load_model()` | Loads EfficientNet-B4 with binary head (1792→1). Loads weights from `models/sbi/efficientnet_b4.pth` if available |
| `_prepare_crop_and_landmarks(face_image, landmarks, scale)` | Pads face image by scale factor (1.15× or 1.25×) using BORDER_CONSTANT (black). Resizes to 380×380. Applies exact affine transform to landmarks to prevent coordinate drift |
| `_compute_gradcam(model, input_tensor)` | Hooks last EfficientNet features layer. Runs backward pass from score. Computes GradCAM map (380×380). Strict hook lifecycle — always removed in `finally` |
| `_map_regions(cam, landmarks)` | Samples GradCAM activation at 5 facial regions (jaw, hairline, cheeks, nose bridge). Returns highest activation region. 5% border clipping prevents BORDER_CONSTANT artifacts |
| `_run_inference(input_data)` | Two-pass: Pass 1 (no_grad) for fast scoring at 1.15× and 1.25×. Pass 2 (gradients) for GradCAM only if score > threshold. Returns boundary region |

#### ✅ What It CAN Detect
- **FaceSwap / DeepFaceLab composites** — clear blend boundaries at jaw/hairline
- **FaceApp / consumer face swaps** — medium-quality compositing artifacts
- **Face-swap videos** where blend boundary is visible in best frame

#### ❌ What It CANNOT Detect
- **Fully-synthetic images** (no face swap = no blend boundary). Agent skips SBI if SigLIP >0.70 for this reason.
- **Very high-quality professional swaps** (e.g. movie-grade VFX with feathered masks)
- **Currently uses random-init weights** (official SBI weights not included) — scores ~0.5 random in test mode

#### 🔧 Counter / Mitigation
- ✅ **Dual scale (1.15× + 1.25×)** — wider context helps detect hairline artifacts
- ✅ **GradCAM localization** — identifies exactly which region triggered the detection
- ✅ **5% border clipping** — prevents BORDER_CONSTANT false positives at image edges
- ❗ **Biggest gap**: Official SBI weights not present. Using random initialization. Scores are unreliable until official weights from `mapooon/SelfBlendedImages` are integrated.

---

### File: `core/tools/freqnet_tool.py` — FreqNet Frequency Tool {#freqnet}

#### What It Does
Dual-stream neural network: one stream analyzes **spatial RGB features**, the other analyzes **DCT frequency coefficients**. Together they detect GAN/diffusion frequency artifacts invisible to the human eye.

#### `class FreqNetDual(nn.Module)`
The underlying dual-stream architecture.

| Stream | Architecture | Input |
|--------|-------------|-------|
| Spatial | ResNet-50 (ImageNet pretrained) + Identity FC | Standard 224×224 RGB |
| Frequency | ResNet-50 with modified conv1 (64-channel DCT input) | DCTPreprocessor output |
| Fusion | concat(2048+2048) → Linear(4096→512) → ReLU → Dropout(0.5) → Linear(512→1) | Both streams |

#### `class FreqNetTool(BaseForensicTool)`

| Method | What it does |
|--------|-------------|
| `tool_name` | Returns `"run_freqnet"` |
| `setup()` | Loads calibration data (Z-score baseline or ratio fallback) |
| `_load_model()` | Loads FreqNetDual; tries `models/freqnet/freqnet_f3net.pth`; falls back to ImageNet backbone + random head |
| `_safe_expand_crop(face_image, landmarks, expansion)` | Expands face crop by 10% with boundary clamping. No padding artifacts |
| `_run_inference(input_data)` | For each face: expand crop → tensor → forward pass (return_dct=True) → FADHook band analysis → calibrated score |

#### Band Analysis (`freqnet/fad_hook.py`)
After DCT preprocessing, analyzes power in 3 frequency bands:
- **Base (low freq)**: Natural frequency of real images → should be dominant
- **Mid freq**: Where GAN artifacts typically hide
- **High freq**: Noise floor

> Anomaly detected if mid/high bands show statistically significant elevation (Z-score or ratio mode).

#### ✅ What It CAN Detect
- **GAN-generated faces** — irregular high-frequency patterns
- **Diffusion model outputs** — characteristic mid-band anomalies in DCT space
- **JPEG-compressed GAN outputs** — frequency fingerprint survives compression

#### ❌ What It CANNOT Detect
- **Post-processed deepfakes** (heavy blur, noise injection) — obscures frequency signature
- **Frequency-domain adversarial attacks** targeting this specific architecture
- **Real images with unusual sensors** (old cameras, phone cameras with heavy sharpening) — may score as fake

#### 🔧 Counter / Mitigation
- ✅ **Dual-mode calibration** (Z-score when full statistics available, ratio fallback otherwise)
- ✅ **FADHook cleanup** in `try/finally` — no hook leaks
- ✅ **Compression discount** in ensemble — if DCT tool already found compression artifacts, FreqNet weight is reduced
- ❗ **Custom FreqNet weights not present** — uses ImageNet backbone. Needs official F3Net weights for full performance.


---

## Part 5 — Utility Modules

---

### File: `utils/preprocessing.py` — MediaPipe Face Preprocessor {#preprocessing}

The **entry point** for all media. Every piece of media passes through here before any tool runs.

#### `class KalmanBoxTracker`
Kalman filter-based tracker for a single face bounding box across video frames.
- `predict()` — state prediction step
- `update(bbox)` — correction step when detection is available

#### `class SortTracker`
Multi-object tracker using Hungarian algorithm + IoU matching.
- `update(dets)` — matches new detections to existing tracks; spawns new tracks; returns active tracks
- `associate(detections, trackers, iou_threshold)` — Hungarian assignment via `scipy.optimize.linear_sum_assignment`

#### `class TrackedFace`
Data container for a single detected identity. Holds:
- `identity_id`, `landmarks` (478×2), `trajectory_bboxes` dict
- `face_crop_224` (for DCT/Geometry/Illumination/Corneal/FreqNet/SigLIP)
- `face_crop_380` (for SBI)
- 6 anatomical patches (periorbital, nasolabial, hairline, chin/jaw)

#### `class Preprocessor`

| Method | What it does |
|--------|-------------|
| `_get_landmarks(image)` | MediaPipe FaceMesh → 478 landmark coordinates. Validates that nose/jaw landmarks are in-frame. Returns sorted by face area (largest first) |
| `_crop_align(image, landmarks, size)` | Extracts face bounding box with 20% margin, resizes to target size via LANCZOS4 |
| `_extract_native_patches(image, landmarks)` | Extracts 6 anatomical patches using specific MediaPipe landmark indices |
| `_select_sharpest_frame(frames, trajectory)` | Samples up to 5 frames from trajectory; picks best by Laplacian variance (sharpness) |
| `process_media(path)` | **Main entry**: detects image vs video. For video: Phase 1 builds trajectories (SORT tracking), Phase 2 extracts crops for stable tracks. For image: single-frame landmark extraction |

---

### File: `utils/ensemble.py` — Ensemble Scoring Engine (v4.0 — Three-Pronged Anomaly Detection) {#ensemble}

Aggregates all tool results into a single deepfake probability using a multi-layered scoring algorithm.

#### Key Functions

| Function | What it does |
|----------|-------------|
| `calculate_ensemble_score(tool_results, ...)` | Three-pronged scoring: (1) weighted average as base, (2) Suspicion Overdrive max-pool with GPU conflict guard, (3) Borderline Consensus boost, (4) GPU Coverage Degradation penalty |
| `_route(result, context, ...)` | Tool-specific routing logic: rPPG routes based on PULSE/NO_PULSE label; SBI has blind-spot threshold; FreqNet has compression discount; C2PA always returns (0,0) — it's a gate not a scorer |
| `stream_ensemble_score(iterator, ...)` | Video streaming: processes per-subject, per-frame results; applies EMA temporal smoothing; hard-resets on tracking loss (scene cut) |
| `_compute_conflict_std(implied_probs)` | Population std dev of per-tool implied probabilities. High std = tools disagree = `has_conflict=True` |

#### v4.0 Three-Pronged Anomaly Detection
- **Prong 1 — Suspicion Overdrive**: If any GPU specialist's fake probability > 0.70, max-pool fires. BUT: GPU Conflict Guard checks spread (max - min) first. If spread > 0.30, specialists contradict each other → falls back to weighted average.
- **Prong 2 — Borderline Consensus**: If ≥2 GPU specialists cluster in [0.35, 0.55] zone, their mean is boosted 1.25× as a corroboration signal.
- **Prong 3 — GPU Coverage Degradation**: Each abstained GPU specialist applies +0.10 multiplicative boost to fake_score. Disabled when GPU conflict detected.

#### Notable Design Decisions
- **SBI blind-spot**: SBI scores below `SBI_BLIND_SPOT_THRESHOLD` are treated as abstentions (score=0, weight=0) — low-confidence SBI shouldn't downvote real images
- **C2PA visual contradiction check**: If C2PA says real but UnivFD/SBI/FreqNet say >70% fake, C2PA override is **not** applied (logs warning instead)
- **EMA smoothing** for video: `score_t = α × raw_t + (1-α) × score_{t-1}` per subject ID
- **Abstention transparency**: Tools with confidence=0 display `[ABSTAINED] N/A` in UI and LLM prompt

---

### File: `utils/vram_manager.py` — VRAM Lifecycle Manager {#vram}

Controls GPU memory for all GPU tools to prevent OOM.

#### Key Classes/Functions

| Name | What it does |
|------|-------------|
| `VRAMLifecycleManager` (context manager) | Wraps GPU model load/unload. On `__enter__`: calls loader fn, moves model to GPU. On `__exit__`: moves to CPU, `del model`, `torch.cuda.empty_cache()`, `gc.collect()` |
| `run_with_vram_cleanup(loader_fn, inference_fn, ...)` | Load model → run inference → unload. Returns ToolResult. Catches `torch.cuda.OutOfMemoryError` with one retry |

---

### File: `utils/thresholds.py` — Central Constants Registry {#thresholds}

All numerical thresholds and weights in ONE file. This is the single source of truth used by all tools and the ensemble.

Key groups:
- **GPU Decider weights**: `WEIGHT_UNIVFD=0.22`, `WEIGHT_XCEPTION=0.15`, `WEIGHT_SBI=0.25`, `WEIGHT_FREQNET=0.10`
- **CPU Supporter weights**: `WEIGHT_GEOMETRY=0.08`, `WEIGHT_DCT=0.04`, `WEIGHT_ILLUMINATION=0.04`, `WEIGHT_CORNEAL=0.04`
- **Ensemble thresholds**: `ENSEMBLE_REAL_THRESHOLD=0.50`, `SUSPICION_OVERRIDE_THRESHOLD=0.70`
- **v4.0 Anomaly constants**: `BORDERLINE_CONSENSUS_BOOST=1.25`, `GPU_COVERAGE_DEGRADATION_FACTOR=0.10`
- **Per-tool thresholds**: `GEOMETRY_IPD_RATIO_MIN/MAX`, `CORNEAL_MAX_DIVERGENCE`, `RPPG_SNR_THRESHOLD`, etc.
- **EMA smoothing**: `EMA_SMOOTHING_ALPHA`, `EMA_SMOOTHING_ENABLED`

---

### File: `utils/ollama_client.py` — Ollama LLM HTTP Client {#ollama}

HTTP client for communicating with a local Ollama server (Phi-3 Mini / any model).

#### Key Functions

| Function | What it does |
|----------|-------------|
| `generate(prompt, model, timeout)` | POST to `/api/generate`. Handles streaming responses by accumulating chunks until `done=True` |
| `check_health()` | GET `/api/tags` — returns True if Ollama is running |
| `pull_model(model_name)` | POST to `/api/pull` — downloads model if not present |

> **Limitation**: Fully synchronous. Blocks the analysis pipeline while LLM generates.
> **Counter**: Agent yields `AgentEvent("llm_start")` before calling, so UI knows to show a loading state.

---

### File: `utils/logger.py` — Logging Helpers {#logger}

#### `def setup_logger(name)`
Returns a Python logger with:
- Console handler (INFO level)
- Optional file handler if `LOG_FILE` env var set
- Consistent format: `[timestamp] [level] [module] message`

---

### File: `utils/video.py` — Video I/O Helpers {#video}

| Function | What it does |
|----------|-------------|
| `extract_frames(path, max_frames, fps)` | OpenCV frame extraction at target FPS. Returns list of RGB numpy arrays |
| `is_video_file(path)` | Checks extension against known video formats |

---

### File: `utils/image.py` — Image I/O Helpers {#image}

| Function | What it does |
|----------|-------------|
| `load_image(path)` | Loads image as RGB numpy array (handles EXIF rotation) |
| `is_image(path)` | Checks extension against known image formats |

---

### File: `downloads/download_model.py`
Script to download custom model weights from a URL or HuggingFace Hub into the `models/` directory.

### File: `downloads/download_freqnet_models.py`
Script specifically for downloading F3Net/FreqNet pretrained weights.

### File: `downloads/siglip.py`
Script to download the SigLIP-Base-Patch16-224 model from Google/HuggingFace for offline use.

---

### `test_data/` — SQLite Test Databases

| File | Purpose |
|------|---------|
| `sig_test.db` | SigLIP adapter integration tests — verifies adapter returns valid ToolResult |
| `import_test.db` | Import/schema validation tests — ensures all modules import without error |

> **Limitation**: These are local SQLite files; no CI/CD schema migration is set up.
> **Counter**: Future plan — use `alembic` for versioned schema migrations.

---

## Part 6 — FreqNet Subpackage: `core/tools/freqnet/`

This internal subpackage provides the frequency-domain components used by `FreqNetTool`.

---

### File: `core/tools/freqnet/__init__.py`
Package marker. Exports `DCTPreprocessor`, `SpatialPreprocessor`, `FADHook`, `BandAnalysis`, `CalibrationManager` for use by `freqnet_tool.py`.

---

### File: `core/tools/freqnet/preprocessor.py` — DCT Preprocessor

Converts face crops from RGB pixel-space to **frequency-domain coefficient maps** for the FreqNet dual-stream architecture.

#### `class DCTPreprocessor(nn.Module)`

| Method | What it does |
|--------|-------------|
| `__init__()` | Registers BT.709 luma coefficients and frozen DCT basis as `register_buffer` (device-agnostic) |
| `_create_dct_basis_vectorized()` | Constructs 64 orthonormal DCT-II basis functions as (64,1,8,8) tensor. Uses `torch.einsum('ux,vy->uvxy')` for 2D outer product. Includes α(k) normalization so DC and AC coefficients are on the same scale |
| `_create_dct_conv2d()` | Wraps the basis into a `nn.Conv2d(1, 64, kernel=8, stride=8)` with frozen weights (`requires_grad=False`) |
| `_rgb_to_luma(rgb)` | BT.709 weighted sum: `0.2126·R + 0.7152·G + 0.0722·B` using registered buffer |
| `forward(rgb)` | Pipeline: RGB → BT.709 luma → DCT Conv2d → log1p compression. Output: `(B, 64, 28, 28)` for 224×224 input |

> **What it CAN do**: Provides a mathematically exact, GPU-portable 2D-DCT transform with no external JPEG library.
> **What it CANNOT do**: Does not preserve sign information (uses `abs()` before `log1p`). Phase information is lost.
> **Counter**: For GAN detection, the magnitude spectrum is sufficient. Phase loss is acceptable.

#### `class SpatialPreprocessor(nn.Module)`

| Method | What it does |
|--------|-------------|
| `forward(rgb)` | Standard ImageNet normalization: `(x - mean) / std` using registered buffers for mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

> **What it CAN do**: Ensures the spatial ResNet-50 stream receives correctly normalized input matching its ImageNet pretraining.
> **What it CANNOT do**: Does not handle non-standard color spaces (e.g. YCbCr input).

---

### File: `core/tools/freqnet/fad_hook.py` — FAD (Frequency Analysis & Detection) Hook

Captures DCT coefficient tensors from the preprocessing layer and performs **frequency band analysis** using JPEG zigzag ordering.

#### Constants
- `ZIGZAG_ORDER`: 64-element list mapping zigzag scan position → raster position in an 8×8 DCT block (matches JPEG standard)
- `BAND_BASE`: Zigzag positions 0–20 (low frequency). **Real images should dominate here** (~70%).
- `BAND_MID`: Positions 21–41 (mid frequency). **GAN artifacts hide here**.
- `BAND_HIGH`: Positions 42–63 (high frequency). **Synthetic texture upscaling shows here**.

#### `@dataclass BandAnalysis`
Result container for band analysis.

| Field | Type | Meaning |
|-------|------|---------|
| `base/mid/high_energy` | float | L2 norm (Frobenius norm) of that band's coefficients |
| `base/mid/high_ratio` | float | Energy fraction out of total |
| `z_base/mid/high` | Optional[float] | Z-score vs calibration baseline (None if no calibration) |
| `anomaly_detected` | bool | True if any band exceeds threshold |
| `anomaly_type` | str | `"high_freq"` or `"low_freq"` |
| `interpretation` | str | Human-readable anomaly description |

#### `class FADHook`

| Method | What it does |
|--------|-------------|
| `__init__(calibration_data)` | Stores calibration dict; sets up capture slot |
| `_capture_hook(module, input, output)` | Forward hook function: `.detach().clone().cpu()` — frees from computation graph immediately |
| `register(module)` | Attaches hook to target PyTorch module |
| `remove()` | Removes hook handle; sets to None. Called in `try/finally` to guarantee no VRAM leaks |
| `_compute_band_energy(features, band_indices)` | Slices `features[:, band_indices, :, :]` → Frobenius norm |
| `_safe_z_score(value, mean, std)` | `(value-mean)/std` with `std < 1e-8` guard |
| `analyze()` | Computes energies + ratios for all 3 bands; if calibration present uses Z-scores; otherwise ratio thresholds |

> **What it CAN do**: Identifies when a face crop has unnaturally elevated high or mid-frequency energy.
> **What it CANNOT do**: Cannot isolate which spatial region of the face has the anomaly (only global energy per band).
> **Counter**: For localization, GradCAM in SBI tool provides spatial attention. FreqNet provides the frequency signal.

---

### File: `core/tools/freqnet/calibration.py` — Calibration Manager

Manages the reference statistics (mean ± std of band ratios for real images) used by `FADHook` for Z-score computation.

#### `class CalibrationManager`

| Method | What it does |
|--------|-------------|
| `__init__(calibration_path)` | Sets path to `calibration/freqnet_fad_baseline.pt` |
| `load()` | Tries to load `.pt` file; validates 6 required keys (`mean_base`, `std_base`, `mean_mid`, `std_mid`, `mean_high`, `std_high`); falls back to `DEFAULT_CALIBRATION` hardcoded values if file not found |
| `get_data()` | Returns calibration dict — always returns data, never `None` |
| `is_calibrated()` | True if FULL mode (loaded from file), False if FALLBACK |

**Default fallback values** (hardcoded from natural image statistics):
- Base band: mean=0.70 ± 0.10 (real images are ~70% low frequency)
- Mid band: mean=0.25 ± 0.08
- High band: mean=0.05 ± 0.03

> **What it CAN do**: Operates without any external file using sensible defaults.
> **What it CANNOT do**: Default fallback is not dataset-specific — a dataset with unusual images may drift.
> **Counter**: Run `scripts/compute_fad_calibration.py` on a real-image dataset to generate a proper `.pt` baseline file.

---

## Part 7 — Configuration & Meta Files

### File: `pyproject.toml`
Project metadata for `pip install -e .`:
- **Name**: `aegis-x`
- **Version**: `4.0.0`
- **Python requirement**: `>= 3.10`
- **Packages**: Includes `core*` and `utils*` only (downloads/test_data excluded from install)
- **Build backend**: `setuptools`

### File: `requirements.txt`
Full dependency list:

| Package | Purpose |
|---------|---------|
| `mediapipe==0.10.14` | Face landmark detection (478 nodes) |
| `torchcodec>=0.9.0` | GPU-accelerated video decoding |
| `insightface` | Alternative face detection backend |
| `c2pa-python` | C2PA Content Credentials reading |
| `scipy` | DCT, linear assignment (Hungarian), signal processing |
| `numpy>=1.24.0` | All numerical operations |
| `opencv-python>=4.8.0` | Image/video I/O, Laplacian sharpness |
| `streamlit` / `gradio` | Front-end UI options |
| `httpx` | Async HTTP client for Ollama |
| `pydantic` | Data validation for config |
| `python-dotenv` | `.env` file loading |
| `transformers` | SigLIP backbone loading from HuggingFace |
| `Pillow` | Image loading for SigLIP preprocessing |
| `torch>=2.0.0` / `torchvision>=0.15.0` | Neural network framework |

> **Note**: `numpy` and `opencv-python` appear twice — safe to deduplicate.

### File: `.env.example`
Template for environment variables. Users copy to `.env` and fill in:

| Variable | Example | Purpose |
|----------|---------|---------|
| `AEGIS_MODEL_DIR` | `models/` | Root directory for all model weight files |
| `AEGIS_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama LLM server URL |
| `AEGIS_VRAM_THRESHOLD` | `3.5` | Min free VRAM (GB) required to run GPU tools |

### File: `.gitignore`
Standard Python + ML gitignore. Excludes: `models/`, `__pycache__/`, `*.pth`, `*.pt`, `.env`, `test_data/*.db`, virtual envs.

### File: `NOTES.md`
Internal development notes tracking per-sprint design decisions and bug fixes.

### File: `day18_test_results.txt`
Output log from Day 18 integration testing. Captures ToolResult outputs, ensemble scores, and execution timing across all tools on sample media.

### File: `diagnostics_day14.py`
Standalone diagnostic script from Day 14. Run to verify:
- All tools import successfully
- ToolRegistry initializes without errors
- GPU/CPU device allocation works
- Ollama is reachable at configured endpoint

### File: `__init__.py` (root)
Root package marker. Makes the repo importable as a Python package.

---

## Part 8 — Complete Processing Breakdown

---

# 📸 SECTION 1: Photo Deepfake Detection — Full Step-by-Step Process

This section walks through **exactly** what happens when a static image is submitted to Aegis-X for deepfake analysis. Each step is described in the order it executes, with the tools that activate, what evidence they collect, what they miss, edge cases we have handled, and edge cases still open.

---

## Step 1 — Input & Preprocessing
**File**: `utils/preprocessing.py` → `Preprocessor.process_media(path)`

### What Happens:
1. The image is loaded as an RGB numpy array via `utils/image.py → load_image()`. EXIF rotation is applied.
2. MediaPipe `FaceMesh` (static mode, `refine_landmarks=True`) runs on the image to detect all faces and extract **478 facial landmarks** per face.
3. Landmarks are validated: nose tip, left jaw, right jaw must all be within image bounds.
4. Faces are sorted by area (largest first). Up to `max_subjects_to_analyze` (default: 2) are processed.
5. For each detected face:
   - `face_crop_224` is extracted: bounding box + 20% margin, resized to 224×224 (LANCZOS4). Used by DCT, Geometry, Illumination, Corneal, FreqNet, SigLIP.
   - `face_crop_380` is extracted: bounding box + 20% margin, resized to 380×380. Used by SBI.
   - 6 anatomical patches are extracted using specific landmark indices: left periorbital, right periorbital, nasolabial left, nasolabial right, hairline band, chin/jaw.
   - `trajectory_bboxes = {0: (x1,y1,x2,y2)}` (a single-frame trajectory for images).
6. Result is a `PreprocessResult` with `has_face=True`, `tracked_faces=[TrackedFace, ...]`, `original_media_type="image"`.

### ✅ What This CAN Do:
- Detect multiple faces in one image (multi-face forensics)
- Handle any image format (JPEG, PNG, WEBP, TIFF) via OpenCV + Pillow
- Produce a standardized payload `tracked_faces` that all tools consume identically

### ❌ What This CANNOT Do:
- Detect faces that are **too small** (<64px width) — MediaPipe returns no landmarks
- Handle **heavily obscured faces** (sunglasses covering >80% of face, heavy masks)
- Detect faces in **extreme profile** (>90° yaw) — landmark validation fails
- Process **non-face deepfakes** (full-body swaps, background manipulation) — only face region is analyzed

### 🟢 Edge Cases COVERED:
- **No face detected** → `has_face=False`; pipeline halts early with INCONCLUSIVE
- **Very small face** → minimum resolution check (`min_face_resolution=64`); rejected
- **Multiple faces** → all processed; worst-scoring face drives the verdict

### 🔴 Edge Cases NOT YET COVERED:
- **Upscaled/AI-enhanced images** where the original face was tiny — landmarks may be unreliable on upscaled textures
- **Cartoon / anime deepfakes** — MediaPipe fails on non-photorealistic faces
- **Non-face AI-generated content** (AI backgrounds, AI objects) — **now covered in v4.0** via raw image fallback (GPU tools analyze full image when no faces detected)

---

## Step 2 — Tool 1: C2PA Provenance Check
**File**: `core/tools/c2pa_tool.py` → `C2PATool._run_inference()`  
**Phase**: CPU | **Weight**: 0.05 | **Role**: Gate | **Trust Tier**: 2

### What Activates:
The tool reads the media file path directly from `input_data["media_path"]`. It does NOT analyze pixels — it reads embedded cryptographic metadata.

### What It Does Step-by-Step:
1. Tries to import `c2pa` Python library. If unavailable, abstains immediately.
2. Calls `c2pa.read_file(media_path)` (or legacy `c2pa.Reader(media_path)`) to extract the C2PA manifest.
3. If manifest found: extracts `signer`, `timestamp`, sets `c2pa_verified=True`.
4. Returns `ToolResult(score=0.0, confidence=0.9)` — C2PA says REAL, so score=0.
5. **Special behavior**: Agent checks `c2pa_verified` flag. If True → calls `EarlyStoppingController` with `c2pa_hw_verified=True` → immediately returns `HALT_C2PA_HARDWARE_SIGNED` → **pipeline stops, verdict = REAL**.

### ✅ Detects:
- Images signed by **hardware cameras** with C2PA firmware (Sony α7, Nikon Z-series, certain Canons)
- Images from **Adobe Content Credentials** (Photoshop, Lightroom, Adobe Firefly)
- Images from AI tools that voluntarily add C2PA (some versions of SD WebUI with provenance plugin)

### ❌ Cannot Detect:
- **No C2PA in file** → abstains (`score=0.0, confidence=0.0`). Most deepfakes have none.
- **Stripped metadata** — attacker runs `exiftool -all= image.jpg` → all C2PA removed in 1 second
- **Replayed manifests** — copying a valid C2PA block from a real image to a fake one
- **Unverified signer** — tool reads the signer name but does NOT verify the X.509 certificate chain

### 🟢 Edge Cases COVERED:
- **C2PA library not installed** → graceful abstain, no crash
- **C2PA verified but visual AI tools still say FAKE** → Ensemble has a visual contradiction check: if SigLIP/SBI/FreqNet collectively score >70% fake, the C2PA override is blocked and a warning is logged
- **Corrupt/unreadable manifest** → caught in try/except, abstains

### 🔴 Edge Cases NOT YET COVERED:
- Certificate chain validation (is the signer actually a trusted camera vendor or a spoofed cert?)
- Replay attack detection
- Partial C2PA manifests (manifest present but incomplete)

---

## Step 3 — Tool 2: DCT Frequency Artifact Check
**File**: `core/tools/dct_tool.py` → `DCTTool._run_inference()`  
**Phase**: CPU | **Weight**: 0.04 | **Role**: Supporter | **Trust Tier**: 1

> **v4.0**: DCT now falls back to raw image analysis when no tracked faces are available.

### What Activates:
Takes `tracked_faces[].face_crop_224` from the preprocessing result. No model needed — pure math.

### What It Does Step-by-Step:
1. For each face crop: converts to uint8, then to grayscale (BT.601 luminance).
2. Computes a **video hash** (MD5 of top-left 100×100 pixels) for caching across multiple faces.
3. **Grid alignment search**: tries all 64 possible (dy, dx) offsets (0–7 × 0–7). For each offset, divides the grayscale image into non-overlapping 8×8 blocks and computes the 2D DCT using `scipy.fft.dctn`. Extracts low-frequency AC coefficients (where i+j ≤ 5). Builds a histogram of AC values, autocorrelates it, and measures the secondary/primary peak ratio. Best offset = one that maximizes this ratio.
4. With the best grid offset, computes the final `peak_ratio` for the face crop.
5. Maps peak ratio → score linearly via `DCT_RATIO_THRESHOLD` and `DCT_RATIO_SCALE`.

### ✅ Detects:
- **Double-JPEG compression** — image was saved, opened, edited, saved again → AC histogram has periodic spikes at multiples of the quantization step
- **GAN images saved as JPEG** where the generator introduces unnatural coefficient distributions
- **Composited images** where pasted regions have different quantization history than the background

### ❌ Cannot Detect:
- **PNG / lossless images** — no JPEG quantization grid; peak ratio stays near 0; tool correctly scores near 0 but provides no real evidence
- **AI images saved as PNG** (most Midjourney, DALL-E outputs) — completely blind
- **Single-compression JPEG deepfakes** (saved only once) — no double-quant artifact
- **Heavily denoised / upsampled GAN outputs** — frequency artifacts washed out

### 🟢 Edge Cases COVERED:
- **No face crop available** → abstains (`score=0.0, confidence=0.0`)
- **Very small crop** → if h or w ≤ 0 after grid offset, peak ratio returns 0.0
- **Grid result shared across video frames** via hash-based cache (performance)
- **Ensemble compression discount**: if `peak_ratio > DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD`, the ensemble reduces SBI and FreqNet weights (they share the JPEG blind spot)

### 🔴 Edge Cases NOT YET COVERED:
- **PNG images need a separate artifact detector** (Error Level Analysis / ELA is the standard approach)
- **HEIC / AVIF** formats — not handled; OpenCV conversion may silently discard frequency info

---

## Step 4 — Tool 3: Anthropometric Geometry Check
**File**: `core/tools/geometry_tool.py` → `GeometryTool._run_inference()`  
**Phase**: CPU | **Weight**: 0.08 | **Role**: Supporter | **Trust Tier**: 1

> **v4.0 demotion**: Geometry was demoted from 0.20 (Decider) to 0.08 (Supporter) because noisy CPU heuristics were overriding GPU deep-learning models.

### What Activates:
Takes `tracked_faces[].landmarks` (478×2 array) directly. Does NOT use the face crop — works purely in landmark coordinate space.

### What It Does Step-by-Step:
1. **Yaw gate**: Computes `_check_yaw_proxy()` — eye midpoint horizontal offset from nose tip divided by face width. If yaw exceeds threshold → bilateral checks are **skipped entirely** (not failed), confidence drops.
2. Runs up to 6 anatomical ratio checks, each producing (passed, severity 0–1, detail string):
   - **IPD ratio** (weight 2.0): `inter_pupillary_distance / jaw_width`. Real range: [0.38, 0.52]. GANs frequently produce too-wide or too-narrow IPD.
   - **Philtrum ratio** (weight 1.5): `philtrum_length / lower_face_height`. Real range: [0.28, 0.42].
   - **Eye asymmetry** (weight 1.0, skipped at high yaw): `|left_eye_width - right_eye_width| / avg_eye_width`. Real range: [0.0, 0.15].
   - **Nose width ratio** (weight 1.5): `nose_alar_width / IPD`. Real range: [0.55, 0.75].
   - **Mouth width ratio** (weight 1.0): `mouth_width / IPD`. Real range: [0.95, 1.30].
   - **Vertical thirds** (weight 2.0): upper/mid/lower face thirds should each be ~33%. Real range per third: [0.25, 0.42].
3. Each failed check contributes `weight × severity` to a violation sum.
4. `weighted_score = violation_sum / MAX_WEIGHT`. Clamped to [0, 1].
5. Confidence is computed dynamically: fewer valid checks + high yaw + small face = lower confidence.
6. Processes ALL faces; returns the worst-scoring face result.

### ✅ Detects:
- **Diffusion model hallucinations**: wrong IPD (eyes too close/far), wrong nose alar width, vertical proportion errors
- **StyleGAN / GAN faces**: consistent anthropometric failures across multiple ratios simultaneously
- **Fully synthetic portraits** where all 6 checks can be evaluated

### ❌ Cannot Detect:
- **High-yaw poses** (>~30°) — bilateral checks skipped; only philtrum + vertical thirds run; score is low-evidence
- **Face swaps that preserve source geometry** — if the swapped face has real human proportions, all ratios pass
- **Small faces** (<80px landmark span) — MediaPipe landmarks are noisy at small scale
- **Faces with real proportional extremes** (very wide-set eyes, very narrow nose) — may false-positive

### 🟢 Edge Cases COVERED:
- **Soft severity** (V2): Not binary pass/fail — how FAR outside the range matters. A slightly off IPD scores less than a wildly off IPD.
- **Weighted violations**: IPD and vertical thirds carry 2× weight vs eye asymmetry — focus on the most reliable signals.
- **Dynamic confidence**: When yaw is high or face is small, confidence drops so ensemble doesn't over-weight an unreliable score.
- **Multi-face**: Returns the face with the worst anthropometric violations.

### 🔴 Edge Cases NOT YET COVERED:
- **Landmark stability across video frames** (`_check_landmark_stability`) is implemented as a stub — it exists but only gets single-frame landmarks today. Multi-frame IPD jitter is a strong video deepfake signal.
- **Professional face swaps preserving geometry** — this tool gives them a pass. The SBI tool is the designated counter for this case.
- **Cartoon/anime deepfakes** — MediaPipe fails entirely on non-photorealistic faces.

---

## Step 5 — Tool 4: Illumination Consistency Check
**File**: `core/tools/illumination_tool.py` → `IlluminationTool._run_inference()`  
**Phase**: CPU | **Weight**: 0.04 | **Role**: Supporter | **Trust Tier**: 1

### What Activates:
Takes `tracked_faces[].face_crop_224` AND the original full `frames_30fps[0]` (original image as a frame) and the face bounding box. Both are needed — the face and the scene context.

### What It Does Step-by-Step:
1. Extracts **face luma** from the 224×224 face crop (YCrCb → Y channel).
2. Extracts **scene context luma** from the region BELOW the face bbox in the original full frame (neck/shoulders area, 40px tall strip). If no scene region available (face at very bottom of image), abstains.
3. Checks if either the face or scene luma has **diffuse lighting** (gradient magnitude < threshold). If diffuse → abstains (`score=0.0, confidence=0.0`). Diffuse lighting gives no directional signal.
4. Computes **dominant gradient direction** for face and scene separately via Sobel operators → weighted circular mean of gradient angles.
5. Calculates angular difference between face and scene dominant directions.
6. Applies a mismatch penalty: difference > 45° → score increases toward 1.0.
7. **Shadow-highlight consistency check**: verifies bright pixels (highlights) and dark pixels (shadows) are on opposite horizontal sides of the face — they should be for consistent single-source lighting.

### ✅ Detects:
- **AI composite images** where the face was generated with left-side lighting but the scene is right-lit
- **Green screen / chroma-key face swaps** with unmatched lighting environments
- **GAN faces pasted onto stock photos** where lighting environments obviously mismatch

### ❌ Cannot Detect:
- **Diffuse / overcast lighting** — gradient too weak to determine direction; tool correctly abstains
- **Images cropped to face only** (passport photos, headsets, headshots on plain backgrounds) — no scene context below face
- **Multiple light sources** — tool only finds one dominant direction; multi-source rigs fool it
- **Skilled deepfakes** where the artist explicitly matched lighting direction

### 🟢 Edge Cases COVERED:
- **Diffuse abstention (V3 fix)**: Previously gave a small REAL vote on diffuse images — now correctly abstains
- **Context from original frame (V3 fix)**: Previously extracted scene from the face crop's bottom — nonsensical. Now extracts from original frame below the face bbox.
- **Shadow-highlight secondary check**: Added as a second independent signal for single-source lighting consistency.
- **Missing scene region** → abstains instead of scoring on face alone

### 🔴 Edge Cases NOT YET COVERED:
- **Multi-source lighting analysis** — only left/right dominant direction is checked; a top-lit face in a bottom-lit scene would be missed
- **Very large faces** (face fills the image) — no room below for scene context
- **Portrait-mode bokeh** — scene is intentionally blurred; gradient analysis unreliable on blurry backgrounds

---

## Step 6 — Tool 5: Corneal Reflection Check
**File**: `core/tools/corneal_tool.py` → `CornealTool._run_inference()`  
**Phase**: CPU | **Weight**: 0.04 | **Role**: Supporter | **Trust Tier**: 1

### What Activates:
Takes `tracked_faces[].face_crop_224`, `tracked_faces[].landmarks` (must have ≥478 points for iris nodes 468, 473), and `tracked_faces[].trajectory_bboxes` for coordinate transformation.

### What It Does Step-by-Step:
1. Checks landmark count ≥ 478 (requires MediaPipe `refine_landmarks=True` which gives iris nodes).
2. Gets face bounding box from `trajectory_bboxes[best_frame_idx]`.
3. Extracts iris center coordinates for LEFT iris (node 468) and RIGHT iris (node 473) — these are in full-frame pixel space.
4. **Transforms** both iris centers from full-frame space → 224×224 crop space (scale by `224/(face_w, face_h)`).
5. Validates transformed coordinates are within the expected eye region (y: 50–140 in 224×224 crop).
6. Extracts two **15×15 pixel ROIs** centered on each iris landmark.
7. For each ROI: checks if max brightness > 180/255 (absolute minimum for a real catchlight). If not → no catchlight present.
8. Finds the largest bright blob via connected components analysis. Normalizes its centroid to [-1, +1] range relative to the ROI center.
9. **Divergence**: Euclidean distance between left and right catchlight offset vectors. Normalizes by geometric max (√8 ≈ 2.83).
10. Maps normalized divergence → fake score. Consistent catchlights → score=0. Divergent → score→1.

### ✅ Detects:
- **Midjourney / DALL-E / Stable Diffusion 1.x–2.x** outputs — these models frequently generate catchlights at random positions in each eye
- **Low-quality deepfakes** where the composite disrupts the original catchlight pattern

### ❌ Cannot Detect:
- **No catchlights** (dim indoor lighting, dark irises) → both centroid detections return None → abstains
- **Single catchlight** (one eye occluded or closed) → abstains (cannot measure divergence with one data point)
- **Perfectly consistent accidental catchlights** (~30% of AI images pass by chance)
- **Modern FLUX.1 / SD 3.5 / Midjourney v6** — significantly improved catchlight consistency vs earlier models

### 🟢 Edge Cases COVERED:
- **Absolute brightness guard (V2)**: Previous version used top 2% relative threshold — a dark iris with max brightness 50/255 could still "find" a catchlight. Now requires >180/255 absolute.
- **Single-catchlight abstention (V2)**: Previously voted FAKE (score=0.5) when only one eye had a catchlight. Fixed to abstain (score=0.0) — you cannot measure divergence without both points.
- **Connected components (V2)**: Handles multi-light environments (two windows = two catchlight blobs). Uses the largest blob.
- **Coordinate transformation (V2 fix)**: Landmarks are in full-frame space; tool now correctly maps them to crop space instead of using raw landmark coordinates on the 224×224 crop (which produced wrong locations).
- **Mirror correction removed (V2 fix)**: Earlier version incorrectly applied a mirror transform — real eyes have the SAME directional offset for the same light source, not mirrored offsets.

### 🔴 Edge Cases NOT YET COVERED:
- **Modern diffusion models** (FLUX.1, Midjourney v6) are rapidly improving catchlight consistency — the signal will weaken over time
- **Synthetic corneal maps** trained to match real catchlight distributions — future adversarial attack vector
- Small face images where the iris ROI is smaller than 15×15 pixels → tool rejects the face (correct but uninformative)

---

## Early Stopping Check (After Each CPU Tool)
**File**: `core/early_stopping.py` → `EarlyStoppingController.evaluate()`

After each CPU tool completes, the agent calls `EarlyStoppingController.evaluate()` with:
- All tool scores collected so far
- List of completed tool names
- `c2pa_hw_verified` boolean

**9-step logic**:
1. C2PA hardware verified? → `HALT_C2PA_HARDWARE_SIGNED` immediately
2. No tools completed yet? → `CONTINUE_AMBIGUOUS`
3. Validate input shapes
4. Compute weighted mean of completed-tool scores
5. Compute evidential conflict (std of per-tool scores weighted by confidence) — if conflict > threshold → `CONTINUE_ADVERSARIAL_CONFLICT` (blocks early stopping)
6. Compute mathematical upper/lower bounds: best case if all remaining tools say FAKE vs REAL
7. Lower bound > FAKE threshold → `HALT_LOCKED_FAKE`
8. Upper bound < REAL threshold → `HALT_LOCKED_REAL`
9. Default → `CONTINUE_AMBIGUOUS`

> If early stopping fires as HALT, the GPU phase is skipped entirely.

---

## Step 7 — Tool 6: UnivFD (AI-Generated Image Specialist)
**File**: `core/tools/univfd_tool.py` → `UnivFDTool._run_inference()`  
**Phase**: GPU | **Weight**: 0.22 | **Role**: Decider | **Trust Tier**: 3

> **v4.0**: SigLIP adapter replaced by UnivFD (CLIP-ViT-L/14 linear probe, CVPR 2023). Falls back to raw image when no faces detected.

### What Activates:
Requires CUDA. Takes `tracked_faces[].face_crop_224` (or `face_crop_380`). Produces 6 copies of the best face crop as input (TTA: original + horizontal flip = 2 passes of 6 each).

### What It Does Step-by-Step:
1. **Loads SigLIP backbone** from `models/siglip-base-patch16-224/` (local, offline, FP16, frozen). Registers forward hooks on encoder layers 3, 6, 9, 11.
2. **Stage 1** — Feature extraction: For each of 6 crops, runs frozen SigLIP vision encoder. Each forward pass captures intermediate features from 4 hooked layers → (4, 196_tokens, 768_dims). Hook outputs are immediately cast to FP32 and detached.
3. **Stage 2** — Dynamic Spatial Pooling: Learned query vectors (4×768) attend over the 196 spatial tokens per layer using einsum attention. Layer fusion via learned softmax weights → (6_crops × 768).
4. **Stage 3** — Cross-Patch Attention: Treats 6 crops as a sequence, applies 4-head attention (cross-patch inconsistency capture) with zero-init residual projection → (6×768).
5. **Stage 3.5** — Score Head + LSE: Linear(768→128→1) → raw logits per crop. Log-Sum-Exp pooling with learnable temperature β (operates on logits, NOT probabilities, to avoid numerical issues). Single sigmoid at the end → scalar score.
6. **TTA**: Runs again with horizontally flipped crops. Final score = max(original_score, flipped_score).
7. Unloads backbone (CPU offload + del + gc.collect + cuda empty_cache).

### ✅ Detects:
- **Stable Diffusion / FLUX / Midjourney** — diffusion model texture is captured by SigLIP's ViT features
- **StyleGAN / GAN faces** — GAN fingerprint survives into mid-level ViT layers (layers 3, 6)
- **Fully synthetic portraits** — strongest signal tool for whole-image synthesis

### ❌ Cannot Detect:
- **Requires trained adapter weights** — the adapter (Stage 2, 3, 3.5) is currently **random-initialized**. Without fine-tuning on a deepfake dataset, scores are near-random.
- **Adversarial examples crafted against SigLIP** — ViT backbones are known to be vulnerable to targeted perturbations
- **CPU-only machines** — falls back to abstention (no CUDA)
- **Very low resolution faces** (<80px), heavy motion blur — preprocessing resizes but artifacts may confuse feature extraction

### 🟢 Edge Cases COVERED:
- **TTA with horizontal flip**: Reduces false signals from left/right image biases
- **Mixed precision (FP16 backbone / FP32 adapter)**: Prevents NaN in LSE pooling (a known problem when computing exp() on FP16 logits)
- **Strict offline mode** (`local_files_only=True`): Pipeline cannot accidentally call HuggingFace Hub at inference
- **VRAM cleanup** in `unload()`: Always called in `finally` block — even if inference crashes

### 🔴 Edge Cases NOT YET COVERED:
- **Adapter fine-tuning** is the biggest open gap. Random-init adapter = meaningless score. Must fine-tune on FaceForensics++, DFDC, or WildDeepfake.
- **Adversarial patches on clothing/background** that fool SigLIP features without touching the face
- **Very small adapter** (4 layers × queries) may miss subtle manipulation artifacts that appear only in layers 0–2

---

## Step 8 — Tool 7: SBI (Self-Blended Images) Blend Boundary Detector
**File**: `core/tools/sbi_tool.py` → `SBITool._run_inference()`  
**Phase**: GPU | **Weight**: 0.20 | **Trust Tier**: 3 (must-run for safety)

### What Activates:
First checks `context["siglip_score"]`. If SigLIP scored >0.70 (image is strongly fully-synthetic), **SBI is skipped** — face swaps don't happen on fully-synthetic images, so SBI would be irrelevant. Takes `tracked_faces[].face_crop_380` (380×380, wider context than 224×224 to capture hairline/jaw area where blend boundaries appear).

### What It Does Step-by-Step:
1. **Pass 1 (no_grad)**: For each face, creates two crops at 1.15× and 1.25× scale using `BORDER_CONSTANT` (black border padding). Applies exact affine landmark transformation to avoid coordinate drift. Normalizes (ImageNet stats). Runs EfficientNet-B4 with `torch.no_grad()` for fast scoring.
2. If max(score_1.15×, score_1.25×) > `SBI_FAKE_THRESHOLD` (0.60):
   - **Pass 2 (gradients enabled)**: Selects wider scale (1.25×) if both triggered, else whichever triggered. Runs GradCAM: hooks last EfficientNet features block, runs backward pass, computes weighted activation map.
   - **Region mapping**: Samples GradCAM at jaw, hairline, left cheek, right cheek, nose bridge landmark positions. 5% border clipping prevents false signals from BORDER_CONSTANT padding. Returns highest-activation region name.
3. Returns blend boundary region (jaw/hairline/cheek_l/cheek_r/nose_bridge) or "diffuse" if no localized region.

### ✅ Detects:
- **DeepFaceLab / FaceSwap composites** — strong blend boundary at jaw or hairline
- **FaceApp / consumer apps** — medium quality composite with detectable seam
- **Any face composite** where the pasted face region has different texture/frequency than the surrounding head

### ❌ Cannot Detect:
- **Fully synthetic images** (auto-skipped when SigLIP>0.70) — no composite to detect
- **Very high quality professional composites** (movie VFX, feathered masks) — boundary too smooth to detect
- **Requires trained SBI weights** — official weights from `mapooon/SelfBlendedImages` not included; using random init → scores near 0.5

### 🟢 Edge Cases COVERED:
- **Dual scale (1.15× + 1.25×)**: Wider crop context catches hairline artifacts that a tight crop misses
- **GradCAM localization**: Not just "is it fake" but "WHERE is the blend boundary" — useful for explainability
- **5% border clipping**: BORDER_CONSTANT (black) pads create artificial edges; sampling is clipped away from the border zone to avoid false positives
- **Exact affine landmark transform**: Padding shifts landmark positions; tool computes exact scale/translate to keep landmarks accurate post-padding
- **SigLIP skip optimization**: Avoids running expensive EfficientNet-B4 inference when SigLIP already found strong fully-synthetic evidence

### 🔴 Edge Cases NOT YET COVERED:
- **Official SBI weights not downloaded**: Biggest open gap. Random init makes this tool unreliable.
- **Inpainting-style deepfakes** (only eyes/mouth region replaced) — boundary is at a non-standard location not in the 5 defined regions

---

## Step 9 — Tool 8: FreqNet Dual-Stream Frequency Detector
**File**: `core/tools/freqnet_tool.py` → `FreqNetTool._run_inference()`  
**Phase**: GPU | **Weight**: 0.10 | **Trust Tier**: 1

### What Activates:
Takes `tracked_faces[].face_crop_224`. Applies a 10% safe expansion before processing.

### What It Does Step-by-Step:
1. **Loads FreqNetDual** from `models/freqnet/freqnet_f3net.pth` if available. Otherwise uses ImageNet ResNet-50 backbone + random head.
2. Expands face crop by 1.1× (`_safe_expand_crop`) to include some surrounding skin area (frequency artifacts can appear at face boundary).
3. Converts to tensor, sends to GPU.
4. **Spatial stream**: `SpatialPreprocessor` applies ImageNet normalization → ResNet-50 feature extraction → (2048,) vector.
5. **Frequency stream**: `DCTPreprocessor` → BT.709 luma → DCT-II conv (frozen 64 basis functions) → log1p → (64,28,28) feature map → Modified ResNet-50 (conv1 replaced for 64-channel input) → (2048,) vector.
6. **Fusion**: concat(spatial+freq) → Linear(4096→512) → ReLU → Dropout → Linear(512→1) → sigmoid → score.
7. Simultaneously, `FADHook` captures the (64,28,28) DCT coefficients from the preprocessor and runs **band analysis**: separates into 3 frequency bands (low/mid/high) using JPEG zigzag ordering. Computes Z-scores vs calibration baseline (or ratio thresholds if no calibration file).
8. Returns score + band anomaly interpretation.

### ✅ Detects:
- **GAN texture fingerprints** in mid/high frequency bands — GANs produce characteristic spectral patterns invisible to the human eye
- **Diffusion model upscaling artifacts** — elevated high-frequency energy from denoising steps
- **JPEG-compressed GAN outputs** — frequency fingerprint partially survives JPEG compression

### ❌ Cannot Detect:
- **Post-processed deepfakes** (heavy Gaussian blur, median filter, denoise) — frequency fingerprint is destroyed
- **Frequency-domain adversarial attacks** targeting this specific architecture (published attack methods exist)
- **Requires F3Net weights** — without the custom pretrained checkpoint, spatial/frequency fusion is meaningless (random head)
- **Calibration file missing** — falls back to ratio mode (less sensitive to subtle anomalies)

### 🟢 Edge Cases COVERED:
- **Dual-mode calibration**: FULL (Z-score vs dataset baseline) or FALLBACK (hardcoded ratio thresholds). Tool always runs in some mode — never crashes for missing calibration.
- **FADHook cleanup in `try/finally`**: Hook is always removed even if inference crashes. No VRAM leaks.
- **Safe crop expansion**: 10% boundary expansion with image clamping — no padding artifacts.
- **Compression discount in ensemble**: If DCT tool already detected heavy JPEG compression, FreqNet's effective weight in ensemble is reduced (they share the compression blind spot).

### 🔴 Edge Cases NOT YET COVERED:
- **F3Net custom weights not present**: Biggest open gap alongside SBI.
- **Calibration file not generated**: Need to run `scripts/compute_fad_calibration.py` on a real-image dataset to get dataset-specific Z-score baselines.

---

## Step 10 — Ensemble Scoring
**File**: `utils/ensemble.py` → `calculate_ensemble_score()`

### What Happens:
1. Deduplicates tool results (prevents double-counting if a tool appears twice).
2. Extracts context: DCT peak ratio, SigLIP score, compression flag.
3. **C2PA override check**: If C2PA verified AND visual tools don't contradict → return REAL immediately.
4. Routes each tool through `_route()`:
   - C2PA → always (0.0, 0.0) — it's a gate, not a scorer
   - rPPG → checks `liveness_label`: PULSE_PRESENT → contributes 0 toward FAKE; NO_PULSE → contributes strong FAKE signal; AMBIGUOUS → abstains
   - SBI → blind-spot check: score < 0.35 → abstains; score > 0.70 → full weight; in between → dynamic weight scaled by SigLIP score
   - FreqNet → if score < 0.40 → abstains (noise floor); applies compression discount if applicable
   - All others → `score × weight` with optional confidence weighting
5. `ensemble_score = sum(contributions) / sum(effective_weights)`
6. Computes `conflict_std` (std dev of per-tool implied probabilities). If > 0.35 → `has_conflict=True`.
7. Returns `{ensemble_score, tools_ran, abstentions, conflict_std, has_conflict}`.

### ✅ What the Ensemble Handles:
- Tools that abstained are correctly excluded from the weighted average
- C2PA can override all other tools
- Compression detected by DCT automatically reduces trust in SBI and FreqNet (both are affected by JPEG artifacts)
- rPPG provides a biological liveness signal independent of visual appearance

### ❌ What the Ensemble Cannot Do:
- Detect that a specific tool was adversarially manipulated (only detects statistical conflict between tools)
- Produce a score when ALL tools abstained (returns 0.5 INCONCLUSIVE)

---

## Step 11 — LLM Synthesis
**File**: `core/forensic_summary.py` + `utils/ollama_client.py`

### What Happens:
1. `build_phi3_prompt()` constructs a structured prompt listing each tool's `evidence_summary` and `score`. Only tool results with `confidence > 0` are included.
2. Prompt is sent to Phi-3 Mini (or configured model) via Ollama HTTP API.
3. LLM generates a natural language explanation GROUNDED in the tool evidence — it cannot fabricate new signals.
4. Verdict is already determined by ensemble_score before LLM runs. LLM only explains WHY.

---

## Step 12 — Final Verdict

| Ensemble Score | Verdict |
|---------------|---------|
| < REAL threshold | **REAL** — No significant deepfake evidence found |
| > FAKE threshold | **FAKE** — Multiple tools flag manipulated content |
| In between | **INCONCLUSIVE** — Evidence insufficient or conflicting |
| C2PA override | **REAL (Hardware Verified)** — Cryptographic provenance confirmed |

---

## 📊 Complete Edge Case Matrix — Photo Detection

| Scenario | Handled? | How |
|----------|----------|-----|
| No face detected | ✅ | `has_face=False` → INCONCLUSIVE |
| Face too small (<64px) | ✅ | Rejected in preprocessing |
| Multiple faces in image | ✅ | Each face analyzed; worst score drives verdict |
| Fully synthetic (Midjourney/DALL-E) | ✅ | SigLIP + FreqNet + Geometry all activate |
| Face swap (FaceSwap/DFL) | ✅ | SBI detects blend boundary |
| C2PA-signed authentic image | ✅ | C2PA short-circuits pipeline |
| C2PA stripped by attacker | ❌ | No counter — C2PA abstains |
| PNG deepfake (no JPEG) | ⚠️ | DCT blind but SigLIP + FreqNet still active |
| Diffuse lighting (no gradient) | ✅ | Illumination correctly abstains |
| High-yaw face pose | ✅ | Geometry bilateral checks skipped; confidence reduced |
| No catchlights in eyes | ✅ | Corneal correctly abstains |
| Single-eye catchlight only | ✅ | Corneal abstains (cannot measure divergence) |
| All tools conflict | ✅ | `conflict_std > 0.35` → INCONCLUSIVE flagged |
| GPU not available | ✅ | GPU tools skip; CPU tools still run |
| Ollama not available | ✅ | LLM synthesis skipped; verdict from ensemble only |
| SBI adapter weights missing | ❌ | Random init — SBI score unreliable |
| SigLIP adapter not fine-tuned | ❌ | Random init — SigLIP score unreliable |
| FreqNet F3Net weights missing | ❌ | ImageNet backbone only — weaker detection |
| Advanced adversarial attack on ViT | ❌ | No adversarial defense layer implemented |
| Anime / cartoon deepfake | ❌ | MediaPipe fails — no landmarks |
| Body/background manipulation (non-face) | ❌ | Out of scope; face-only pipeline |

---

---

# 🎬 SECTION 2: Video Deepfake Detection

> **Will be continued in a future sprint.**
>
> The architecture for video detection is already partially built inside the codebase:
> - rPPG tool (`core/tools/rppg_tool.py`) is complete and handles pulse detection across video frames.
> - SORT tracker (`utils/preprocessing.py`) builds `trajectory_bboxes` per identity across all frames.
> - EMA-smoothed ensemble (`utils/ensemble.py → stream_ensemble_score()`) is implemented.
> - Landmark stability check stub exists in `geometry_tool.py`.
>
> Full documentation of the video processing pipeline — step-by-step per tool, what activates, what each tool finds in video that it cannot find in photos, temporal consistency checks, rPPG coherence decision logic, and per-frame ensemble flow — will be written when video detection is implemented and tested end-to-end.

