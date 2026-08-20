# Truth-Scope: Interview Preparation Guide

This document is designed to help you prepare for technical interviews by structuring your knowledge of the **Truth-Scope** project into compelling narratives. It covers your system architecture, technical decisions, and provides STAR (Situation, Task, Action, Result) method answers for common interview questions.

---

## 1. The Elevator Pitch
**Interviewer:** *"Tell me about a recent project you worked on."*

**Your Pitch:**
"I recently developed **Truth-Scope**, a comprehensive deepfake forensics and media authentication platform. It analyzes images and videos using an ensemble of AI models and traditional forensic techniques to determine if the media is authentic or AI-generated. The system is built with a **FastAPI** backend and a **Vanilla JS/HTML/CSS** frontend, utilizing Server-Sent Events (SSE) to stream real-time analysis results to the user. To handle the massive computational load without crashing, I designed a unique **dual-environment architecture** that isolates heavy PyTorch GPU workloads into separate subprocesses, ensuring the main web server remains lightweight and memory-efficient."

---

## 2. System Architecture & Tech Stack

Be prepared to draw or explain this architecture:

*   **Frontend:** Vanilla JavaScript, HTML5, CSS3. Uses drag-and-drop file uploads and parses Server-Sent Events (SSE) to build dynamic, real-time UI cards as each forensic tool completes its analysis.
*   **Backend Server:** Python, FastAPI, Uvicorn. Operates in a lightweight virtual environment (`.venv_main`).
*   **GPU Workers:** Python, PyTorch, Insightface, MediaPipe. Operates in an isolated heavy environment (`.venv_gpu`).
*   **Forensic Ensemble:**
    *   **CPU Phase:** C2PA Cryptographic Signature validation, DCT (Discrete Cosine Transform) frequency analysis, Geometry/Illumination inconsistency checks, and rPPG (Remote Photoplethysmography) for micro-blood-flow detection in videos.
    *   **GPU Phase:** Heavy neural networks including FreqNet (ResNet50), SBI (EfficientNetB4), Xception, and UnivFD.
*   **Communication:** The lightweight API spawns isolated subprocesses (`subprocess_proxy.py`) to execute the GPU models, passing data via IPC (Inter-Process Communication). 

---

## 3. Key Architectural Decisions (The "Why")

**Why a Dual-Environment Architecture?**
*   **Reasoning:** PyTorch and heavy ML libraries notoriously suffer from memory leaks and GPU VRAM fragmentation when kept alive in long-running web servers. By isolating the web server (`.venv_main`) from the ML execution (`.venv_gpu`), the system can spawn a fresh subprocess for inference and completely release all VRAM when the process terminates. It also completely eliminated dependency conflicts between lightweight API tools and heavy CUDA libraries.

**Why Server-Sent Events (SSE) instead of WebSockets?**
*   **Reasoning:** Forensic analysis of video can take minutes. Standard HTTP requests would time out. While WebSockets support real-time data, they are bi-directional and overkill. SSE is a lightweight, unidirectional stream perfect for sending consecutive "tool completed" payloads from the backend to update the frontend progressively.

**Why Early Stopping?**
*   **Reasoning:** Running 8 different AI models is expensive. I implemented an `EarlyStoppingController`. If a decisive tool (like a cryptographically verified C2PA signature) confirms the media is AI-generated with 100% certainty in the first second, the pipeline halts immediately, saving immense computational resources.

---

## 4. STAR Method Interview Questions

### Question 1: "Tell me about a challenging bug you faced and how you solved it."
*   **Situation:** During development, my facial landmark detection (MediaPipe) suddenly started crashing with a cryptic C++ binding error, while at the same time, my FastAPI server was throwing `python-multipart` errors.
*   **Task:** I needed to stabilize the pipeline and fix the broken tools.
*   **Action:** I dove into the dependency trees and realized that running an ML installation script inside the wrong virtual environment had caused a "cross-pollination" dependency conflict. A package (`insightface`) pulled in `protobuf>=4.0`, which silently broke MediaPipe's C++ backend (which strictly requires `protobuf<4`). I isolated the environments, downgraded Protobuf in the main environment, added the missing `python-multipart` library for FastAPI file handling, and explicitly enforced `--index-url` flags to ensure PyTorch downloaded the massive 2.5GB CUDA wheels instead of falling back to CPU caches.
*   **Result:** The server stabilized, MediaPipe successfully initialized, and the models properly utilized GPU hardware acceleration, reducing inference time drastically. 

### Question 2: "How did you handle a situation where data was missing or an API failed?"
*   **Situation:** I noticed that the frontend was displaying "undefined" for the descriptions of my heavy GPU tools, making it look like the tools were failing.
*   **Task:** Identify why the descriptions were missing without breaking the streaming architecture.
*   **Action:** I traced the data flow from the ML execution script (`subprocess_proxy.py`), to the orchestrator (`agent.py`), and finally to the frontend parser (`script.js`). I discovered that the tools were actually executing perfectly and generating an `evidence_summary`, but the backend orchestrator was failing to serialize that specific field into the JSON payload emitted via SSE. I updated the dictionary payload to include `"evidence_summary": result.evidence_summary`.
*   **Result:** The frontend immediately began rendering the correct forensic explanations in real-time as each tool finished, vastly improving the user experience and trust in the system's output.

### Question 3: "How do you make decisions when different systems give conflicting answers?"
*   **Situation:** In deepfake forensics, one model might flag an image as fake (e.g., weird lighting), while another says it's real (e.g., standard frequency data). 
*   **Task:** I needed a way to provide a single, reliable verdict to the user.
*   **Action:** I built an `EnsembleAggregator`. Instead of a simple majority vote, I assigned specific baseline weights to each tool based on their historical accuracy (e.g., `run_rppg` at 0.35, `run_dct` at 0.15). The system normalizes these weights dynamically based on which tools successfully ran (skipping video tools if the input is an image). It computes a directional confidence score.
*   **Result:** The system intelligently aggregates the signals. If the aggregate confidence exceeds 80%, it can even confidently bypass further GPU testing, returning a highly accurate verdict while saving processing time.

---

## 5. Potential Technical Deep-Dives to Expect

If you are interviewing for a Backend or ML Engineer role, expect follow-up questions like:
1.  **"How does your subprocess proxy actually serialize and send image arrays to the GPU environment?"** 
    *   *Prep:* Be ready to discuss passing file paths, using JSON for metadata IPC, or using memory-mapped files if you did.
2.  **"If this application went viral and 1,000 users uploaded videos at once, how would you scale it?"**
    *   *Prep:* Mention replacing the local Python `ThreadPoolExecutor` and subprocesses with a message queue (like RabbitMQ or Redis/Celery) and deploying the GPU workers as separate microservices on Kubernetes nodes with dedicated GPUs.
3.  **"What is C2PA?"**
    *   *Prep:* Know that it stands for Coalition for Content Provenance and Authenticity. It's an open technical standard providing publishers, creators, and consumers the ability to trace the origin of different types of media using cryptographic hashes.
