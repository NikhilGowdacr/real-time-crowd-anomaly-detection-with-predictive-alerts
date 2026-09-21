# Real-Time Crowd Anomaly Detection with Predictive Alerts

An engineering research project implementing a modular, multi-stage AI surveillance architecture for real-time crowd monitoring, behavioral anomaly detection, weapon detection, fight detection, and predictive alert dispatching.

---

## 1. Project Overview

Surveillance systems in high-density public venues (transit terminals, stadiums, pedestrian plazas, festival grounds) require rapid, autonomous anomaly detection. Traditional security monitoring relies on human visual vigilance, which degrades rapidly under fatigue.

This project develops an end-to-end multi-modal AI framework combining:
- **Spatial Computer Vision**: Real-time person detection (YOLOv8/YOLO11), crowd density modeling, and localized congestion hotspot detection.
- **Temporal Tracking & Trajectories**: ByteTrack multi-object tracking, persistent track identities, velocity vectors, and direction angles.
- **Crowd Behavioral Dynamics (Phase 4)**: Macroscopic crowd kinematics (dispersion/scattering, gathering/mobbing, directional entropy) fused with Vision Transformer (ViT / Swin Transformer) spatiotemporal feature representations.
- **Weapon & Dangerous Object Detection**: Dedicated modular object detector for firearms, knives, and dangerous handheld objects with spatial track association.
- **Fight & Violent Altercation Detection**: Multi-frame temporal sliding window evaluating spatial convergence, reciprocal motion, and acceleration bursts.
- **Multi-Evidence Safety Fusion**: Calibrated risk scoring ($S_{\text{crowd}}, S_{\text{fight}}, S_{\text{weapon}}, S_{\text{behavior}}$) with evidence correlation boosters and false-positive suppression.
- **Acoustic Surveillance (Upcoming Phase 5)**: Environmental audio feature extraction (Mel Spectrograms) and sound event classification.

---

## 2. System Architecture

The complete end-to-end multi-modal AI surveillance pipeline links all 12 project phases:

```
VIDEO STREAM (CCTV / Webcam / File / Synthetic)
  │
  ▼
Person Detection (Phase 2: YOLOv8 / YOLO11)
  │
  ▼
Crowd Tracking & Trajectories (Phase 3: ByteTrack & Safety Analysis)
  │
  ▼
Crowd Density & Spatial Hotspots (Phase 2 & 3)
  │
  ▼
Behavioral Anomaly & Kinematics (Phase 4: Swin-T / ViT & Dispersion)
  │
        ┌───────────────────────────────────────┐
AUDIO ─►│ Multimodal Fusion Engine (Phase 6)    │◄── Video Evidence
(Ph. 5) │ (Temporal Alignment, Synergy, Denoise)│
        └───────────────────┬───────────────────┘
                            │
                            ▼
                  Decision Engine (Phase 7)
            (Finite State Machine, Hysteresis, Quotas)
                            │
                            ▼
                  Alert Manager (Phase 8)
            (Deduplication, Cooldown, Severity Policies)
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
  Explainable AI (Phase 9)          Operator Dashboard (Phase 10)
(Attribution & HUD Evidence)       (Real-Time Telemetry & Console)
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                            ▼
                  SQLite Persistence (Phase 11)
            (WAL Mode, Referential Integrity, Query API)
                            │
                            ▼
            Evaluation & Benchmarking (Phase 12)
          (Ablation, Robustness, Scenarios, Latency)
```

---

## 3. Project Directory Structure

```
crowd_anomaly_detection/
├── configs/
│   └── config.yaml           # Central parameter configuration
├── data/
│   ├── audio/                # Audio recordings and test samples (for Phase 5)
│   ├── datasets/             # Research benchmarks (ShanghaiTech, UCF-Crime, RWF-2000)
│   ├── processed/            # Extracted features and cached tensors
│   └── videos/               # CCTV clips and test videos
├── models/
│   ├── audio/                # Audio classification weights
│   ├── behavior/             # ViT / Swin behavior model weights
│   ├── detection/            # YOLO detection checkpoints (yolov8n.pt, weapon_model.pt)
│   └── tracking/             # ByteTrack tracker configurations
├── notebooks/                # Exploratory analysis and training notebooks
├── src/
│   ├── alerts/               # Alert manager, siren, notifications
│   ├── audio/                # Audio capture, spectrogram, classifier
│   ├── behavior/             # Advanced behavioral analysis subsystem (Phase 4)
│   │   ├── __init__.py
│   │   ├── temporal_buffer.py    # FIFO frame sequence buffer with subsampling
│   │   ├── behavior_features.py  # Crowd kinematics (scattering, gathering, entropy)
│   │   ├── behavior_model.py     # Swin / ViT deep feature representation interface
│   │   └── behavior_anomaly.py   # Kinematic + Deep score fusion & smoothing
│   ├── dashboard/            # Real-time monitoring dashboard & operator UI (Phase 10)
│   │   ├── __init__.py
│   │   ├── dashboard_data.py     # DashboardSnapshot telemetry schema & DashboardDataProvider
│   │   ├── dashboard_state.py    # Thread-safe bounded in-memory state & alert lifecycle
│   │   ├── dashboard_metrics.py  # Rolling FPS, latency stats, and telemetry summarization
│   │   ├── dashboard_theme.py    # Surveillance terminal dark palette, CSS, and status badges
│   │   ├── dashboard_components.py # Modular Streamlit telemetry panels & interactive controls
│   │   └── dashboard_app.py      # Streamlit application entrypoint & multi-scenario demo
│   ├── database/             # Persistent surveillance event database (Phase 11)
│   ├── alerts/               # Alert management & notification subsystem (Phase 8)
│   │   ├── __init__.py
│   │   ├── alert_record.py       # AlertSeverity, AlertStatus enums & AlertRecord schema
│   │   ├── alert_policy.py       # Configurable alert policies, filtering & severity mapping
│   │   ├── cooldown.py           # Cooldown debouncing, escalation bypass & emergency repeat
│   │   ├── deduplication.py      # Deterministic fingerprint hashing & duplicate suppression
│   │   ├── acknowledgement.py    # Operator lifecycle management (active/acknowledged/resolved)
│   │   ├── channels.py           # Notification channels (Console, HUD, and Email/SMS/Push stubs)
│   │   ├── local_alarm.py        # Fail-safe non-blocking audio alarm tone dispatcher
│   │   └── alert_manager.py      # Master coordinator ingesting Phase 7 DecisionEvents
│   ├── decision/             # Intelligent decision engine subsystem (Phase 7)
│   │   ├── __init__.py
│   │   ├── decision_record.py   # RiskState enum and DecisionEvent schema
│   │   ├── hysteresis.py        # Dual-threshold hysteresis & dynamic boundary adaptation
│   │   ├── confirmation.py      # Temporal persistence tracking & quotas
│   │   ├── state_machine.py     # Deterministic state machine with transition cooldown
│   │   └── decision_engine.py   # High-level facade, critical rules & objective reasoning
│   ├── xai/                  # Explainable AI & Evidence Visualization subsystem (Phase 9)
│   │   ├── __init__.py
│   │   ├── evidence_record.py        # Structured EvidenceRecord schema
│   │   ├── feature_attribution.py    # Deterministic normalized factor attributions
│   │   ├── evidence_extractor.py     # Causal factor extraction from multimodal telemetry
│   │   ├── gradcam_interface.py      # Architecture for future CNN/ViT Grad-CAM
│   │   ├── visual_explanation.py     # Motion vectors, dispersion arrows & OpenCV HUD overlay
│   │   ├── explanation_formatter.py  # Short, detailed, HUD, and alert-specific formatting
│   │   └── explanation_engine.py     # Master ExplanationEngine facade
│   ├── explainability/       # Backwards-compatible alias re-exporting src.xai
│   ├── fusion/               # Multimodal audio-visual fusion subsystem (Phase 6)
│   │   ├── __init__.py
│   │   ├── types.py              # ModalityEvidence, CrossModalSynergy, MultimodalAssessment
│   │   ├── temporal_aligner.py   # Rate synchronization (30 FPS video with 2 Hz audio)
│   │   └── multimodal_fusion.py  # 5 Video + 4 Audio feature fusion, synergy & noise suppression
│   ├── safety/               # Safety intelligence subsystem
│   │   ├── __init__.py
│   │   ├── weapon_detector.py # Custom YOLO weapon detector & track association
│   │   ├── pose_analyzer.py   # Kinematics, distance matrix, reciprocal motion
│   │   ├── fight_detector.py  # Temporal sliding window fight classifier
│   │   └── safety_fusion.py   # Multi-evidence fusion, correlation, cooldown
│   ├── utils/                # Config, logger, metrics helpers
│   └── video/                # Video capture, preprocessing, YOLO, density, ByteTrack
├── tests/
│   ├── test_dashboard.py     # Phase 10 monitoring dashboard unit tests (24 tests)
│   ├── test_xai.py           # Phase 9 explainable AI & attribution tests (24 tests)
│   ├── test_alerts.py        # Phase 8 alert management unit tests (22 tests)
│   ├── test_audio.py         # Phase 5 acoustic unit tests (16 tests)
│   ├── test_behavior.py      # Phase 4 behavioral unit tests (13 tests)
│   ├── test_decision.py      # Phase 7 decision engine unit tests (18 tests)
│   ├── test_fusion.py        # Phase 6 multimodal fusion unit tests (16 tests)
│   ├── test_safety.py        # Safety subsystem unit tests (11 tests)
│   └── test_video.py         # Video pipeline unit tests (8 tests)
├── .gitignore
├── main.py                   # Main CLI entry point (--mode alerts|decision|multimodal|audio|behavior|safety|tracking|video)
├── README.md
└── requirements.txt
```

---

## 4. Environment Setup & Installation (Windows / VS Code)

### Prerequisites
- Windows 10/11
- Python 3.12 (preferred)
- `uv` package manager (recommended) or standard `python -m venv`
- VS Code

### Setup using `uv` (Recommended)

1. Open PowerShell in the project root:
   ```powershell
   cd "c:\Users\Nikhil gowda c.r\OneDrive\Desktop\mainproj"
   ```

2. Create a virtual environment using Python 3.12 (copy mode avoids Windows symlink limitations):
   ```powershell
   uv venv --python 3.12 --link-mode copy .venv
   ```

3. Activate the virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```

4. Install dependencies:
   ```powershell
   uv pip install -r requirements.txt --python .venv\Scripts\python.exe
   ```

---

### 0. Real-Time Surveillance Monitoring Dashboard (`--mode dashboard`) [PHASE 10]
Launches the interactive Streamlit operator console providing live video/crowd dynamics, acoustic telemetry, multimodal fusion status, active alert cards with operator controls ([Acknowledge] / [Resolve]), XAI evidence summaries, and real-time time-series telemetry charts:
```powershell
# Launch surveillance operator dashboard via main CLI entry point
.venv\Scripts\python.exe main.py --mode dashboard

# Launch in standalone demo mode (cycles through NORMAL, SUSPICIOUS, HIGH_RISK, EMERGENCY, RECOVERY)
.venv\Scripts\python.exe main.py --demo-dashboard

# Direct Streamlit invocation
.venv\Scripts\python.exe -m streamlit run src/dashboard/dashboard_app.py --server.port 8501
```

### 1. Alert Management & Predictive Notification Mode (`--mode alerts`) [PHASE 8]
Executes the full surveillance stack with active Phase 8 alert generation, deduplication, cooldown debouncing, and multi-channel dispatch:
```powershell
# Real-time surveillance with Alert Management on synthetic video & audio
.venv\Scripts\python.exe main.py --mode alerts --synthetic

# Live multi-scenario demonstration exercising the 8 alert lifecycle and suppression scenarios
.venv\Scripts\python.exe main.py --demo-alerts

# Physical webcam + microphone surveillance with active alert manager
.venv\Scripts\python.exe main.py --mode alerts --source 0 --audio-source microphone
```

### 2. Intelligent Decision Engine Surveillance Mode (`--mode decision`) [PHASE 7]
Executes the full end-to-end pipeline: Video + Audio + Safety + Multimodal Assessment $\to$ Context-Aware Dynamic Thresholds $\to$ Dual-Threshold Hysteresis $\to$ Temporal Confirmation Quotas $\to$ 5-State Machine (`NORMAL`, `SUSPICIOUS`, `HIGH_RISK`, `EMERGENCY`, `RECOVERY`) with Cooldown Debouncing.
```powershell
# Real-time intelligent decision surveillance using synthetic stream
.venv\Scripts\python.exe main.py --mode decision --synthetic

# Live multi-scenario demonstration exercising dynamic state transitions and recovery
.venv\Scripts\python.exe main.py --mode decision --demo-multimodal

# Physical webcam + microphone surveillance with decision engine
.venv\Scripts\python.exe main.py --mode decision --source 0 --audio-source microphone
```

### 3. Multimodal Audio-Visual Fusion Surveillance Mode (`--mode multimodal`) [DEFAULT]
Executes the Phase 6 unified engine: Concurrent 30 FPS Video (YOLO, ByteTrack, Density, Kinematics, Behavior, Safety) + 2 Hz Audio (Capture, Preprocessing, Mel, Classification, Anomaly) $\to$ Temporal Stream Alignment $\to$ Cross-Modal Synergy Detection $\to$ False-Positive Noise Suppression $\to$ Dynamic Weighting.
```powershell
# Default multimodal surveillance using synthetic video and audio
.venv\Scripts\python.exe main.py --mode multimodal --synthetic

# Live multi-scenario demonstration (Scream+Scatter, Fight+Shouting, Noise Suppression, Weapon)
.venv\Scripts\python.exe main.py --mode multimodal --demo-multimodal

# Physical webcam + microphone surveillance
.venv\Scripts\python.exe main.py --mode multimodal --source 0 --audio-source microphone

# File playback (CCTV video + ambient audio)
.venv\Scripts\python.exe main.py --mode multimodal --source "data/videos/sample.mp4" --audio-source "data/audio/sample.wav"
```

### 2. Audio Anomaly Detection Mode (`--mode audio`)
Executes the Phase 5 acoustic pipeline: Audio Ingestion $\to$ Signal Preprocessing $\to$ Log-Mel Spectrogram Extraction $\to$ CNN/Spectral Classifier $\to$ Temporal Confirmation $\to$ Audio Anomaly Scoring.
```powershell
# Standard acoustic monitoring using synthetic stream
.venv\Scripts\python.exe main.py --mode audio --synthetic

# Live acoustic demonstration mode (cycles through screams, alarms, shouting, explosions)
.venv\Scripts\python.exe main.py --mode audio --demo-audio

# Physical microphone surveillance
.venv\Scripts\python.exe main.py --mode audio --source microphone

# Audio file playback (WAV, FLAC, etc.)
.venv\Scripts\python.exe main.py --mode audio --source "data/audio/sample.wav"
```

### 2. Advanced Behavioral Anomaly Detection (`--mode behavior`)
Executes the Phase 4 pipeline: YOLO Person Detection $\to$ ByteTrack $\to$ Crowd Kinematics $\to$ Temporal Frame Buffer $\to$ Swin/ViT Transformer Inference $\to$ Behavior Fusion.
```powershell
# Standard behavioral monitoring on synthetic crowd stream
.venv\Scripts\python.exe main.py --mode behavior --synthetic

# Academic demonstration mode (simulates sudden crowd scattering / turbulence)
.venv\Scripts\python.exe main.py --mode behavior --synthetic --demo-behavior

# Live webcam behavioral monitoring
.venv\Scripts\python.exe main.py --mode behavior --source 0

# Video file
.venv\Scripts\python.exe main.py --mode behavior --source "data/videos/sample.mp4"
```

### 3. Safety Intelligence Mode (`--mode safety`)
Executes the safety subsystem: Weapon Detection, Fight Detection, Pose Kinematics, and Multi-Evidence Safety Fusion.
```powershell
.venv\Scripts\python.exe main.py --mode safety --synthetic
.venv\Scripts\python.exe main.py --mode safety --synthetic --demo-safety
```

### 4. Tracking Mode (`--mode tracking`)
Executes ByteTrack multi-object tracking: persistent track IDs, velocity vectors, motion trajectories, and speed tagging.
```powershell
.venv\Scripts\python.exe main.py --mode tracking --synthetic
```

### 5. Video Processing Mode (`--mode video`)
Executes Phase 2 video ingestion, preprocessing, YOLO person detection, and crowd density heatmap estimation.
```powershell
.venv\Scripts\python.exe main.py --mode video --synthetic
```

### Interactive HUD Controls
- **`Q`** or **`ESC`**: Exit application cleanly
- **`G`**: Toggle spatial density heatmap grid overlay
- **`B`**: Toggle simulated abnormal crowd behavior (in `--mode behavior`)
- **`F`**: Toggle simulated fight interaction (in `--mode safety`)
- **`W`**: Toggle simulated weapon detection (in `--mode safety`)
- **`1` - `6`**: Toggle acoustic demo events: Normal, Scream, Shouting, Alarm, Crash, Explosion (in `--mode audio`)
- **`C`**: Toggle CLAHE low-light contrast enhancement

---

## 6. Phase 5 Audio Architecture & Acoustic Intelligence

### A. Unified Audio Capture (`src/audio/audio_capture.py`)
- **Multi-Source Support**: Seamlessly captures from physical microphone arrays (`sounddevice`), audio files (`soundfile`), or a zero-dependency synthetic audio generator.
- **Hardware Fault-Tolerance**: If no microphone is connected, gracefully falls back to synthetic crowd murmur with logged warnings. Video and tracking pipelines operate completely uninterrupted.
- **Sliding Window Slicing**: Slices continuous audio into overlapping analysis windows ($1.0\,\text{s}$ duration, $0.5\,\text{s}$ overlap at $16{,}000\,\text{Hz}$).

### B. Signal Preprocessing (`src/audio/audio_preprocessing.py`)
- **Mono Downmixing**: Averages multi-channel inputs to single-channel mono.
- **DC Offset Removal**: Subtracts mean amplitude to remove microphone rumble and baseline drift.
- **Dynamic Normalization**: Scales signal to target peak ($0.95$) without clipping ($[-1.0, 1.0]$) while preserving digital silence.
- **Signal Telemetry**: Calculates Root Mean Square (RMS) energy and decibels relative to full scale (dBFS).

### C. Mel Spectrogram Extraction (`src/audio/mel_spectrogram.py`)
- **Dual-Engine Implementation**: Native Librosa engine with pure NumPy/SciPy Mel filterbank fallback ($64$ Mel bands, $1024$ FFT window, $512$ hop length).
- **Logarithmic Compression**: Converts raw power to decibel scale ($dB = 10 \log_{10}(S + \epsilon)$).
- **Format**: Outputs 2D spectrogram $(64, T)$ and 4D PyTorch tensor $(1, 1, 64, T)$.

### D. Acoustic Event Classification & Research Integrity (`src/audio/audio_classifier.py`)
- **Target Event Categories**: `normal`, `speech`, `crowd_noise`, `shouting`, `scream`, `alarm`, `explosion_like`, `crash`, `other_abnormal`.
- **Modular CNN Interface**: PyTorch CNN/YAMNet model loader with softmax class probabilities.
- **Spectral-Loudness Baseline**: In absence of a fine-tuned deep checkpoint, executes an empirical spectral baseline analyzing spectral centroid, spectral rolloff, and peak impulse energy. Explicitly tagged as `model_mode: "spectral_baseline"` to preserve scientific integrity.

### E. Temporal Confirmation & Anomaly Scoring (`src/audio/audio_anomaly.py`)
- **Rolling Score Smoothing**: 5-window moving average dampens isolated transient acoustic glitches.
- **Multi-Window Confirmation**: Requires $N \ge 2$ consecutive abnormal windows before escalating to `SUSPICIOUS` or `EMERGENCY`.
- **Latency Benchmark**: Sub-$1.1\,\text{ms}$ total cycle latency per $1.0\,\text{s}$ window ($>900\times$ faster than real time).

---

## 7. Phase 6 Multimodal Audio-Visual Fusion Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                 VIDEO STREAM                 │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │                   Video Evidence                    │
             │  • Crowd Density (level, area density, count)       │
             │  • Movement Anomaly (velocity, dispersion, heading) │
             │  • Behavior Anomaly (ViT/Swin + Kinematics)         │
             │  • Fight Score (proximity, reciprocal motion)       │
             │  • Weapon Score (firearm, knife, dangerous obj)     │
             └──────────────────────────┬──────────────────────────┘
                                        │ (Composite Video Score)
                                        ▼
 ┌──────────────────────┐       ┌───────────────┐       ┌──────────────────────┐
 │     AUDIO STREAM     │──────▶│   MULTIMODAL  │◀──────│  TEMPORAL ALIGNER    │
 │  • Audio Anomaly     │       │ FUSION ENGINE │       │  • Sync 30fps Video  │
 │  • Audio Event Class │       │ (src/fusion/) │       │    with 2Hz Audio    │
 │  • Confidence        │──────▶│               │◀──────│  • Stale detection   │
 │  • Temporal Windows  │       └───────┬───────┘       └──────────────────────┘
 └──────────────────────┘               │
                                        ▼
                        ┌───────────────────────────────┐
                        │  Cross-Modal Synergy Analysis │
                        │  • Scream + Panic Scattering  │
                        │  • Shouting + Physical Fight  │
                        │  • Explosion + Dispersal      │
                        │  • Weapon + Distress Sound    │
                        │  False-Positive Suppression   │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │     Multimodal Assessment     │
                        │  • Multimodal Score [0.0-1.0] │
                        │  • Multimodal Confidence      │
                        │  • Status (NORMAL/SUSP/EMERG) │
                        │  • Detailed Evidence Breakdown│
                        └───────────────────────────────┘
```

### A. Rate Mismatch Resolution (`src/fusion/temporal_aligner.py`)
- **Temporal Synchronization**: Surveillance video processes at $\sim 30\,\text{FPS}$ ($\sim 33\,\text{ms}$ intervals), while audio analysis chunks arrive at $\sim 2\,\text{Hz}$ ($1.0\,\text{s}$ windows with $0.5\,\text{s}$ overlap).
- **Tolerance Window**: Aligns each incoming video frame with the most recent audio record within a configurable tolerance ($\tau = 1.5\,\text{s}$).
- **Stale Stream Detection**: If video frames or audio chunks stop arriving ($>2.5\,\text{s}$ timeout), the aligner automatically flags the stream as stale/inactive.

### B. Multi-Evidence Integration (`src/fusion/multimodal_fusion.py`)
- **5 Video Features**: Integrates Crowd Density ($15\%$), Movement Anomaly ($20\%$), Behavior Anomaly ($25\%$), Fight Score ($20\%$), and Weapon Detection ($20\%$) with high-severity safeguards (critical weapon or fight cannot be masked by low density).
- **4 Audio Features**: Integrates Audio Anomaly Score, Detected Sound Event, Classification Confidence, and Temporal Consecutive Window Confirmation count.
- **Dynamic Modality Re-Weighting**: Automatically re-normalizes weights ($w_{\text{video}} = 1.0, w_{\text{audio}} = 0.0$ or vice versa) during camera occlusion, mic disconnection, or sensor degradation without crashing.

### C. Cross-Modal Synergy Detection
Applies a configurable boost ($1.25\times$) when complementary cross-modal emergency signatures correlate:
- `SCREAM_AND_SCATTER`: Screaming audio combined with rapid crowd dispersion.
- `FIGHT_AND_SHOUTING`: Violent struggle kinematics combined with shouting/distress acoustics.
- `EXPLOSION_AND_DISPERSAL`: Sudden explosive acoustic impulse combined with rapid crowd flight.
- `WEAPON_AND_DISTRESS`: Visual weapon detection combined with vocal distress or screaming.

### D. False-Positive Noise Suppression
Dampens uncorroborated single-sensor transients by $50\%$:
- **Case A**: Acoustic impulse (dropped object, door slam) with calm crowd kinematics.
- **Case B**: Isolated visual motion spike (commuter running, tracking jitter) in nominal quiet audio.

### E. Latency Benchmark
- **Fusion Engine Latency**: **$0.035\,\text{ms}$** per frame ($>28{,}000\,\text{evaluations/second}$).
- **End-to-End Multimodal Pipeline**: Runs in real time at $>35\,\text{FPS}$ on CPU.

---

## 8. Phase 7 Intelligent Decision Engine Architecture

Phase 7 introduces the cognitive decision layer that transforms multimodal surveillance evidence into stable, actionable, and oscillation-free operational states.

```
                   ┌────────────────────────────────────────┐
                   │ Multimodal Assessment Record (Phase 6) │
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ Context-Aware Dynamic Thresholds       │
                   │  • Density offset (+0.05 max)          │
                   │  • Synergy reduction (-0.05 max)       │
                   │  • Sensor outage safety (+0.03 max)    │
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ Dual-Threshold Hysteresis Evaluator    │
                   │  • Suspicious: 0.45 enter / 0.35 exit  │
                   │  • High Risk:  0.65 enter / 0.55 exit  │
                   │  • Emergency:  0.85 enter / 0.70 exit  │
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ Temporal Confirmation Tracker          │
                   │  • Suspicious: 3 consecutive frames    │
                   │  • High Risk:  3 consecutive frames    │
                   │  • Emergency:  2 consecutive frames    │
                   │  • Recovery:   5 consecutive frames    │
                   │  • Critical Bypass (Weapon/Fight >=0.9)│
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ 5-State Finite State Machine (FSM)     │
                   │  NORMAL ⇄ SUSPICIOUS ⇄ HIGH_RISK       │
                   │            │              │            │
                   │            ▼              ▼            │
                   │             EMERGENCY                  │
                   │                 │                      │
                   │                 ▼                      │
                   │              RECOVERY                  │
                   │                 │                      │
                   │                 ▼                      │
                   │               NORMAL                   │
                   │  • 3.0s Transition Cooldown Debouncing │
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ DecisionEvent Record                   │
                   │  • Scientific objective rationale      │
                   │  • Separation: Score vs Confidence     │
                   │  • HUD Visual Status Cards & Badges    │
                   └────────────────────────────────────────┘
```

### A. 5-State Transition Model & Recovery Flow
- **`NORMAL`**: Baseline crowd movement and nominal acoustic ambient.
- **`SUSPICIOUS`**: Atypical visual agitation or acoustic spikes under scrutiny.
- **`HIGH_RISK`**: Sustained anomalous crowd dynamics, potential altercation, or cross-modal corroboration.
- **`EMERGENCY`**: Confirmed critical crowd crisis, severe panic scattering, or critical safety threat.
- **`RECOVERY`**: Controlled de-escalation buffer preventing premature return to `NORMAL`. Requires **5 consecutive frames** ($167\,\text{ms}$) of calm conditions before state clears.

### B. Dual-Threshold Hysteresis & Deadband Invariance
- **Separate Enter / Exit Thresholds**: Ensures that boundary oscillations (e.g. score fluctuating between 0.69 and 0.71) cannot induce rapid state flip-flopping.
- **Deadband Separation Invariant**: Enforces $T_{\text{enter}} > T_{\text{exit}}$ with $\ge 0.05$ separation under all conditions.

### C. Context-Aware Dynamic Thresholds
Dynamically adjusts thresholds based on operational context:
- **Crowd Density ($\ge 0.70$)**: Elevates enter thresholds by up to $+0.05$ to suppress false alarms from dense pedestrian friction.
- **Cross-Modal Corroboration**: Lowers enter thresholds by $-0.05$ to increase sensitivity when visual and acoustic cues agree.
- **Single-Modality Fallback**: Raises enter thresholds by $+0.03$ when a sensor stream is offline.
- **Safety Clamping**: Total adjustments bounded by $\pm 0.10$ and constrained to $[0.05, 0.98]$.

### D. Critical Safety Evidence Escalation
- **Weapon Detection ($\ge 0.90$)**: Triggers immediate transition to `EMERGENCY` bypassing multi-frame confirmation quotas.
- **Fight Altercation ($\ge 0.90$)**: Accelerates escalation to `HIGH_RISK` or `EMERGENCY`.
- **Multimodal Corroborated Emergency ($\ge 0.90$)**: Immediate emergency state declaration.

### E. Latency Benchmark
- **Decision Engine Latency**: **$0.0404\,\text{ms}$** per frame ($>24{,}700\,\text{evaluations/second}$).
- Zero computational overhead on edge or workstation deployment.

---

## 9. Phase 8 Alert Management & Predictive Notification Layer Architecture

Phase 8 creates the operational alerting layer that transforms confirmed Phase 7 `DecisionEvent` records into controlled, deduplicated, and lifecycle-managed surveillance alerts.

```
                    ┌────────────────────────────────────────┐
                    │  DecisionEvent Record (from Phase 7)   │
                    │  (Single Source of Truth for RiskState)│
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Event Validation & Policy Filter       │
                    │  • Confirmed state transitions only    │
                    │  • Minimum severity filter (WARNING)   │
                    │  • NORMAL -> Suppressed (Zero alerts)  │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Deterministic Fingerprint Deduplication│
                    │  • State, Modality, Category, Critical │
                    │  • Window-based (15.0s window)         │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Cooldown & Escalation Manager          │
                    │  • 10.0s cooldown per severity tier    │
                    │  • Escalation bypass: WARN -> HIGH -> CRIT
                    │  • Emergency repeat after 30.0s        │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ AlertRecord Dispatch & Lifecycle       │
                    │  • State: CREATED -> ACTIVE            │
                    │  • In-Memory Acknowledgement Registry  │
                    └───────────────────┬────────────────────┘
                                        │
         ┌──────────────┬───────────────┼───────────────┬──────────────┐
         ▼              ▼               ▼               ▼              ▼
   ┌───────────┐  ┌───────────┐   ┌───────────┐   ┌───────────┐  ┌───────────┐
   │  Console  │  │  HUD Box  │   │Local Alarm│   │Email Stub │  │ SMS Stub  │
   │  Output   │  │  Overlay  │   │(Disabled) │   │(Unconn.)  │  │(Unconn.)  │
   └───────────┘  └───────────┘   └───────────┘   └───────────┘  └───────────┘
```

### A. Operational Severity Mapping
Phase 7 remains the single source of truth for risk state. Phase 8 maps risk states to operational alert severities:

| Phase 7 RiskState | Alert Severity | Default Action | Trigger Condition |
| :--- | :--- | :--- | :--- |
| **`NORMAL`** | *None* | No alert | Standard surveillance baseline |
| **`SUSPICIOUS`** | **`WARNING`** | Elevated scrutiny | Confirmed movement/acoustic agitation |
| **`HIGH_RISK`** | **`HIGH`** | Dispatch personnel | Confirmed struggle or sustained crowd divergence |
| **`EMERGENCY`** | **`CRITICAL`** | Safety protocol | Confirmed panic dispersal, stampede, or weapon |
| **`RECOVERY`** | **`INFO`** | Post-crisis notice | Emitted only when `recovery_alert: true` |

### B. Cooldown Debouncing & Escalation Bypass
- **Severity Cooldown ($10.0\,\text{s}$)**: Prevents notification floods during persistent anomalous episodes.
- **Escalation Allowance**: If a situation escalates from `WARNING` $\to$ `HIGH` $\to$ `CRITICAL`, the higher severity alert immediately **bypasses active cooldowns**.
- **Periodic Emergency Reminder ($30.0\,\text{s}$)**: If an emergency situation persists, re-alerting is permitted after 30 seconds to maintain operator situational awareness.

### C. Fingerprint Deduplication
- Computes deterministic SHA-256 signatures based on: `(risk_state, severity, dominant_modality, reason_category, critical_flag, corroborated_flag)`.
- Explicitly excludes timestamps from the signature so temporal proximity within the duplicate window ($15.0\,\text{s}$) suppresses identical events.
- Differentiates materially distinct threats (e.g. visual weapon vs acoustic shouting) even within the same window.

### D. Operator Acknowledgement Lifecycle
Alerts follow a formal operational lifecycle without altering underlying Phase 7 risk state:
$$\text{CREATED} \longrightarrow \text{ACTIVE} \longrightarrow \text{ACKNOWLEDGED} \longrightarrow \text{RESOLVED}$$
Operators can acknowledge or resolve individual alerts while automated sensory tracking continues uninterrupted.

### E. Notification Channels & Academic Integrity
- **Console & HUD**: Deliver structured visual telemetry and terminal alert banners in real time.
- **Local Alarm**: Optional audible siren/tone with non-blocking execution and fail-safe error isolation.
- **Academic Integrity Note**: External notification providers (`email`, `sms`, `push`) are implemented as safe functional stubs returning `"UNCONNECTED_STUB"` until live cloud provider credentials and endpoints are supplied.

### F. Latency Benchmark
- **AlertManager Latency**: **$0.0516\,\text{ms}$** per event ($>19{,}300\,\text{events/second}$).
- Zero perceptible overhead on real-time surveillance processing.

---

## 10. Phase 9 — Explainable AI (XAI) & Evidence Visualization

The Explainable AI (XAI) subsystem translates multimodal sensor telemetry and decision factors into transparent, auditable, and human-interpretable evidence explanations. It bridges black-box anomaly scores with operator-facing causal justifications.

### A. Core Architectural Invariants
1. **Read-Only / Explanation-Only Invariant**: The XAI engine is strictly downstream and diagnostic. It **never** mutates `anomaly_score`, `current_state`, or `severity`.
2. **Academic Integrity & Transparent Attribution**: Strictly avoids fabricating artificial saliency maps or pseudo-SHAP heatmaps from unconfigured deep models. When deep models are in spectral or kinematic baseline mode, the system transparently reports `"Spectral baseline evidence"` or `"Model explanation unavailable (Grad-CAM unconfigured)"`.
3. **Rigorous Metric Separation**:
   - **Anomaly Score**: Severity and magnitude of observed physical disturbance ($[0.0 - 1.0]$).
   - **Model Confidence**: Sensory classification certainty ($[0.0 - 1.0]$).
   - **Decision Confidence**: Multi-frame temporal state machine certainty ($[0.0 - 1.0]$).
   - **Evidence Contribution**: Normalized relative attribution across observable factors ($[0.0 - 1.0]$, summing to $1.0$ or $0.0$).

### B. Normalized Factor Attribution
Relative evidence contributions are deterministically calculated and ranked across 7 observable factors:
- **Crowd Density**: Measured spatial packing relative to operating thresholds.
- **Rapid Movement**: Kinematic velocity exceeding baseline pedestrian velocity limits.
- **Behavioral Anomaly**: Radial crowd dispersion (scattering) and directional entropy.
- **Acoustic Anomaly**: Elevated sound energy or classified sound events (scream, alarm, explosion).
- **Physical Struggle**: Fight dynamics, reciprocal combat vectors, and spatial convergence.
- **Weapon Presence**: Dangerous handheld objects (firearm, knife).
- **Cross-Modal Corroboration**: Synchronized audio-visual synergy amplification.

### C. Visual Evidence Overlays
- **Motion Vectors**: Velocity heading arrows rendered from person tracking centroids, color-coded by speed (green: normal walking, yellow: brisk/agitated, red: rapid panic movement).
- **Radial Dispersion Indicators**: Outward scattering arrows projecting from crowd center of mass during panic stampedes.
- **XAI Surveillance HUD Card**: Semi-transparent telemetry card displaying evaluated state, score, confidence, ranked contribution bars, causal statements, and model explainability status.

### D. Multi-Format Text Representations
- **`format_short()`**: Single-sentence operator summary (e.g. *"SUSPICIOUS: Driven primarily by acoustic anomaly (51%) and rapid movement (28%) (score: 0.52, conf: 0.76)."*).
- **`format_detailed()`**: Comprehensive multi-section diagnostic report for security audits.
- **`format_hud()`**: Compact 3-5 line bullet points for video overlay.
- **`format_alert_rationale()`**: Formal justification explaining WHY an operational alert was emitted.

### E. Latency Benchmark
- **ExplanationEngine Latency**: **$0.0752\,\text{ms}$** per evaluation ($>13{,}200\,\text{evaluations/second}$), over 26x faster than the $2.0\,\text{ms}$ target quota.

---

## 11. Phase 10 — Real-Time Surveillance Monitoring Dashboard & Operator Interface

Phase 10 implements the professional operator interface that brings together all intelligence outputs from Phases 2–9 into a unified, high-performance, dark-mode terminal UI built on Streamlit.

```
+-----------------------------------------------------------------------------------+
|  [SHIELD] REAL-TIME CROWD ANOMALY MONITORING                   STATUS: LIVE       |
|  AI-Powered Multimodal Surveillance | Phase 10 Interface       TIME: 2026-09-10   |
+-----------------------------------------------------------------------------------+
|  * Video: ONLINE  * Detection: READY  * Tracking: READY  * Behavior: READY        |
|  * Audio: ONLINE  * Fusion: READY     * Decision: READY  * Alerts: READY          |
+-----------------------------------------------------------------------------------+
|                     SYSTEM STATE: EMERGENCY (CONFIDENCE: 96%)                     |
+-----------------------------------------+-----------------------------------------+
| LIVE VIDEO STREAM                       | ACOUSTIC SURVEILLANCE                   |
| [OpenCV Video Feed / Annotated BBoxes]  | Event: Scream | Sound Level: -8.4 dB    |
| Video Anomaly Score: [==========] 0.920 | Acoustic Engine: Spectral Baseline      |
|                                         | Audio Anomaly Score: [========] 0.950   |
| CROWD DYNAMICS                          | MULTIMODAL FUSION & DECISION            |
| People: 42 | Density: 2.90 /m2 (CRIT)   | Fused Score: 0.940 | Synergy: WEAPON    |
| Mean Speed: 18.5 px/f | Dispersion: High| Video / Audio Weights: 0.60 / 0.40      |
+-----------------------------------------+-----------------------------------------+
| ALERT MANAGEMENT (PHASE 8)              | EXPLAINABLE AI (XAI) EVIDENCE           |
| [CRITICAL] EMERGENCY ALERT              | Evidence Explanation:                   |
| Event: Firearm detected with scream     | "Driven by weapon presence (38%) and    |
| [ Acknowledge ]  [ Resolve ]            | acoustic distress with synergy."        |
|                                         | Recommended Operator Action:            |
| RECENT ALERT AUDIT LOG                  | "Initiate emergency broadcast, notify   |
| [Time] [ID] [State] [Severity] [Status] | rapid response units immediately."      |
+-----------------------------------------+-----------------------------------------+
| REAL-TIME TELEMETRY TIME-SERIES CHARTS (Synchronized 120-frame rolling histories) |
| [ Fused Anomaly Score  ---  Video Score  ---  Audio Score vs Time (s) ]          |
+-----------------------------------------------------------------------------------+
```

### A. Architectural Principles & Guarantees
1. **Downstream Visualization Only**: The dashboard never executes a second detection model or recalculates anomaly scores, risk states, or alert severities.
2. **Deterministic Multi-Scenario Demo Mode**: Supports immediate operator demonstration across 5 deterministic scenarios (`NORMAL`, `SUSPICIOUS`, `HIGH_RISK`, `EMERGENCY`, `RECOVERY`) with zero camera or microphone hardware dependencies.
3. **Bounded Memory Invariant**: Historical time-series telemetry queues are strictly capped at `maxlen=120` frames ($4\,\text{s}$ at $30\,\text{FPS}$) using `collections.deque` to guarantee zero memory leaks.
4. **Honest Sensor Provenance**: Offline cameras and microphones explicitly display `"OFFLINE"` / `"N/A"` without synthetic fabrication, and acoustic classification reports `"Spectral Baseline"`.
5. **Interactive Operator Actions**: Integrated `[Acknowledge]` and `[Resolve]` buttons update in-memory alert statuses in real time.

### B. High-Throughput Performance Benchmark
- **Snapshot Transformation Latency**: **$0.147\,\text{ms}$** per frame ($>6{,}700\,\text{snapshots/second}$), over **13x faster** than the $2.0\,\text{ms}$ latency budget.
- **Metrics Engine Latency**: **$0.501\,\text{ms}$** per summary evaluation.

---

---

## 12. SQLite Database & Historical Event Persistence (Phase 11)

Phase 11 establishes a robust, transactional local persistence layer for monitoring sessions, decision events, alerts, Explainable AI (XAI) factor evidence, and operational system metrics using standard library SQLite.

```
                  +----------------------------------------------+
                  |  STREAMING PIPELINE (Phases 1-10)            |
                  |  Video -> Tracking -> Audio -> Fusion -> ... |
                  +----------------------------------------------+
                                         │
                                         ▼ (Async / Safe Persistence)
                  +----------------------------------------------+
                  |  DatabaseManager (Thread-safe, WAL, FK ON)   |
                  +----------------------------------------------+
                                         │
        ┌───────────────────┬────────────┴───────────┬───────────────────┐
        ▼                   ▼                        ▼                   ▼
  [ sessions ]         [ events ]               [ alerts ]          [ evidence ]
  session_id (PK)      event_id (PK)            alert_id (PK)       evidence_id (PK)
  started_at           session_id (FK)          event_id (FK)       event_id (FK)
  status               timestamp (Indexed)      session_id (FK)     alert_id (FK)
  video_source         risk_state (Indexed)     status (Indexed)    factor, rank
                       anomaly_score            severity (Indexed)  contribution
```

### A. Core Architecture & Guarantees
1. **Downstream Persistence Invariant**: The database is strictly a storage and analytical query layer. It **never** acts as a detection source or alters `anomaly_score`, `risk_state`, dynamic thresholds, alert severities, or explanations.
2. **Standard Library Only**: Built entirely upon Python's standard `sqlite3` without external ORM dependencies.
3. **Write-Ahead Logging (WAL)**: Configured with `PRAGMA journal_mode = WAL;` and `PRAGMA busy_timeout = 5000;` allowing non-blocking concurrent reads during high-frequency writes.
4. **Referential Integrity**: Configured with `PRAGMA foreign_keys = ON;`. Deletion and retention pruning execute in strict child-before-parent order (`evidence` $\to$ `alerts` $\to$ `events` $\to$ `metrics`).
5. **Thread-Safe Connections**: Uses `threading.local()` connection pooling to guarantee complete isolation across background processing threads.
6. **Atomic Transactions**: Context-managed atomic transactions (`with db.transaction() as conn:`) with automatic rollback on error.
7. **Health & Corruption Resilience**: PRAGMA quick check validation reporting `ONLINE`, `OFFLINE`, or `DATABASE CORRUPTED` without crashing live surveillance.
8. **Dashboard Integration**: Modular Streamlit tab displays live connection status, persistent summary statistics, and interactive historical event and alert queries.

### B. Command-Line Usage
```powershell
# Inspect database health, table counts, and recent historical records
.venv\Scripts\python.exe main.py --mode database

# Run comprehensive synthetic database demonstration in isolated temporary storage
.venv\Scripts\python.exe main.py --demo-database

# Run performance and latency benchmark
.venv\Scripts\python.exe scratch/benchmark_database.py
```

### C. Performance Benchmark Results
- **Event Insert Latency**: **2.07 ms** per insert in WAL mode (Target: < 5.0 ms).
- **Event Insert Throughput**: **483.0 events/sec**.
- **Evidence Batch Insert**: **56,713 items/sec** (8.82 ms for 500 items).
- **Indexed Recent Events Query (50 items)**: **0.80 ms** (Target: < 2.0 ms).
- **Indexed State Query (EMERGENCY)**: **0.20 ms**.
- **Descriptive Statistics Aggregation**: **0.18 ms**.

---

---

## 13. Phase 12 — Final Academic Evaluation & Benchmarking Framework

Phase 12 provides a comprehensive academic evaluation and benchmarking framework for the complete surveillance system. It measures detection latency, sensor modality ablation, 10 failure-mode stress conditions, 13 controlled synthetic scenarios, and threshold sensitivities, exporting rigorous Markdown, JSON, CSV reports, and headless matplotlib visualizations.

```
                    ┌──────────────────────────────────────────────┐
                    │  SystemEvaluator Orchestration Pipeline      │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   ┌───────────┐                     ┌───────────┐                     ┌───────────┐
   │ Latency   │                     │ Ablation  │                     │ Robustness│
   │ Profiler  │                     │ Study     │                     │ Evaluator │
   │ 11 Modules│                     │ Video/Aud/│                     │ 10 Stress │
   │ Mean/P95  │                     │ Multimodal│                     │ Conditions│
   └─────┬─────┘                     └─────┬─────┘                     └─────┬─────┘
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Academic Deliverables Export Layer           │
                    │  • final_evaluation.md (Tables 1-5 + Claims) │
                    │  • evaluation_summary.json (Machine-readable)│
                    │  • CSV Deliverables (Metrics, Scen., Robust) │
                    │  • Headless Matplotlib Visualizations        │
                    └──────────────────────────────────────────────┘
```

### A. Academic Integrity & Non-Fabrication Invariants
1. **Zero Metric Fabrication**: In strict accordance with academic honesty, when ground-truth dataset labels are unconfigured, classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix) are explicitly reported as `NOT_AVAILABLE` with provenance `UNAVAILABLE`.
2. **Explicit Synthetic Provenance**: Controlled synthetic evaluation outcomes are tagged as `CONTROLLED_SYNTHETIC` and `SYNTHETIC_SCENARIO` to prevent confusion with real-world incident benchmarks.
3. **Production Isolation**: Evaluation experiments run strictly non-destructively in isolated temporary storage without altering production models, configuration thresholds, or the primary database (`data/crowd_anomaly.db`).

### B. Core Benchmarks & Findings
- **Real-Time Latency**: Full-pipeline core decision loop latency of **$0.40\,\text{ms}$** ($\sim 2{,}500\,\text{FPS}$), comfortably satisfying the $\le 33\,\text{ms}$ ($30\,\text{FPS}$) real-time deployment threshold.
- **Multimodal Ablation**: Multimodal fusion demonstrates cross-modal corroboration synergy during critical emergencies ($0.40$ corroboration rate) while suppressing single-sensor false alarms.
- **Robustness**: **10/10 Stress Conditions Passed (100%)** including camera disconnects, acoustic dropout, $0\,\text{dB}$ SNR noise spikes, and severe crowd occlusions.
- **13 Controlled Scenarios**: **13/13 Scenarios Passed (100%)** spanning normal crowd dynamics, physical altercations, stampedes, and panic dispersion.

### C. Command-Line Usage
```powershell
# Run the complete Phase 12 academic evaluation demonstration
.venv\Scripts\python.exe main.py --demo-evaluation

# Run full evaluation suite via mode selection
.venv\Scripts\python.exe main.py --mode evaluation

# View generated academic reports and charts
ls reports/evaluation/
ls reports/evaluation/plots/
```

---

## 14. Model Training Guide for Research Datasets

To fine-tune audio classifiers for academic surveillance evaluation:
1. **Benchmark Datasets**:
   - **ESC-50**: 2,000 environmental audio recordings covering alarms, shouting, crashes, and explosions.
   - **UrbanSound8K**: 8,732 urban sound excerpts across 10 classes (sirens, gunshots, drilling, street music).
   - **AudioSet**: Google's large-scale dataset with emergency categories (screaming, sirens, explosions).
2. **Audio Preprocessing**: Downmix to 16 kHz mono, compute 64-band Mel spectrograms.
3. **Training Objective**: Cross-Entropy loss with AdamW optimizer on ResNet-18 or CNN-14 backbones.
4. **Placement**: Save checkpoint to `models/audio/custom_audio_cnn.pth` and update `configs/config.yaml`.

---

## 15. Running Automated Unit Tests

Run the complete 221-test automated verification suite across all 12 project phases:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 221 tests pass with 0 failures and 0 regressions across all 11 subsystems:
- **Academic Evaluation & Benchmarking (34 tests)**: Metric result creation & defaults, unavailable metric handling with provenance, binary classification exact calculation, empty/unavailable handling, zero-division safety, multi-class macro and weighted metrics calculation, unconfigured multi-class fallback, classification result dict serialization, ground truth annotation dataclass, evaluation sample structure, evaluation dataset iteration & modality filtering, unconfigured dataset fallback, confusion matrix raw and normalized computation, confusion matrix unavailable fallback when GT missing, confusion matrix CSV export, ROC analysis AUC calculation, ROC unavailable fallback when GT missing, PR analysis AUC calculation, PR unavailable fallback when GT missing, threshold sensitivity sweep, inverse threshold-alert correlation, ablation study 3 configurations (Video Only, Audio Only, Multimodal Fusion), robustness evaluator 10 conditions passing 100%, robustness condition failure recovery details, end-to-end and module latency profiling, process memory resource usage & CPU profiling, bounded telemetry queue invariant, scenario evaluator 13 scenarios passing 100%, scenario explicit synthetic provenance tagging, master report generation in all formats (Markdown, JSON, CSV), headless visualization generation and safe GT skipping, system evaluator orchestration, production database isolation, and production threshold immutability.
- **SQLite Database & Persistence (35 tests)**: Database initialization & nested auto-directory creation, schema v1 creation & schema_version tracking, foreign keys enforcement, WAL mode activation, in-memory DB mode, session CRUD lifecycle, event insertion and retrieval, event recent query descending ordering, event query by risk state, event query by severity, event query by time range, alert insertion and retrieval, alert active status querying, alert acknowledgement and resolution timestamps, evidence batch insertion and event query, evidence query by alert, referential linkages across sessions/events/alerts/evidence, ISO-8601 serialization/parsing, boolean and enum serialization, JSON metadata serialization, EventRecord round-trip, AlertRecord round-trip, EvidenceRecord round-trip, SessionRecord round-trip, duplicate event idempotency (`INSERT OR IGNORE`), duplicate alert idempotency, transaction atomic rollback on failure, health check online reporting, corruption detection & safe reporting, system metrics sampled persistence, retention pruning with child-first referential integrity, concurrent thread connection safety, historical statistics aggregation, non-mutation immutability invariant, and missing directory auto-creation recovery.
- **Monitoring Dashboard (24 tests)**: Snapshot default creation, dictionary/JSON serialization, provider initialization, normal subsystem extraction, emergency subsystem extraction with XAI and alerts, 5-scenario demo snapshot verification (`NORMAL`, `SUSPICIOUS`, `HIGH_RISK`, `EMERGENCY`, `RECOVERY`), missing video fallback, missing audio fallback, missing XAI fallback, snapshot state ingestion, strictly bounded deque history (`maxlen=120`), active alert tracking, operator alert acknowledgement, operator alert resolution, state history clearing, rolling FPS calculation, mean/min/max/p95 latency statistics, full metrics summary compilation, terminal theme color tokens and HTML badges, and non-mutation immutability invariant.
- **Explainable AI & Evidence (24 tests)**: Normal explanation on calm baseline, suspicious explanation capturing agitation, multi-factor high-risk explanation, critical emergency rationale, deterministic evidence ranking, deterministic consistency across runs, movement velocity and dispersion statements, density evidence, honest acoustic baseline reporting, fight evidence, weapon evidence, cross-modal synergy explanation, single-modality reliance, conflicting sensor cue explanation, `EvidenceRecord` JSON/dict serialization, short explanation formatting, detailed audit report formatting, HUD compact formatting, Grad-CAM safe unconfigured state, Grad-CAM non-fabrication guarantee, decision state immutability, anomaly score immutability, alert severity immutability, and safe handling of missing/None evidence.
- **Alert Management (22 tests)**: Normal produces no alert, Suspicious creates WARNING, High Risk creates HIGH, Emergency creates CRITICAL, Cooldown suppresses duplicate, Escalation bypasses lower severity cooldown, Duplicate detection works, Materially different evidence creates separate alert, Acknowledgement works, Resolution works, Reset works, Local alarm disabled by default, Missing audio device fails safely without crash, Email stub does not falsely claim delivery, SMS stub does not claim delivery, Push stub does not claim delivery, Alert record serialization works, Alert history works, Active alerts retrieval works, Recovery behavior works, Policy minimum severity filtering works, Alert manager rejects invalid events safely.
- **Decision Engine (18 tests)**: Config defaults, initial normal state, normal-to-suspicious transition, confirmation quota reset on transient drop, hysteresis enter threshold respect, hysteresis exit threshold prevents premature de-escalation, score oscillation resistance, controlled recovery flow from emergency (5 frames quota), critical weapon evidence accelerated escalation, critical fight evidence, dynamic threshold density offset, dynamic threshold corroboration reduction, transition cooldown debouncing (3.0s), single-modality fallback handling, decision confidence separation, objective scientific reason generation, schema validation, and long-term stability across 200 consecutive evaluations.
- **Multimodal Fusion (16 tests)**: Config defaults, engine initialization, single-modality video only, single-modality audio only, dynamic weight re-normalization, 5 video feature extraction, 4 audio feature extraction, Scream+Scatter synergy, Fight+Shouting synergy, Weapon+Distress synergy, audio spike calm crowd suppression, visual spike quiet audio suppression, temporal aligner tolerance sync, stale stream timeout, rolling score smoothing stability, output dictionary schema.
- **Audio (16 tests)**: Config loading, capture initialization, mic fallback, WAV reading, preprocessing DC removal, stereo downmix, peak normalization, Mel spectrogram dimensions, silence handling, classifier baseline, missing model handling, confidence filtering, anomaly scoring, rolling smoothing, consecutive confirmation, output dictionary schema.
- **Behavior (13 tests)**: Temporal buffer, kinematics dispersion, directional entropy, ViT/Swin initialization, behavior score smoothing, temporal confirmation, fusion weighting.
- **Safety (11 tests)**: Weapon detector, track association, pose kinematics, fight detector, alert cooldown.
- **Video (8 tests)**: Video capture, synthetic crowd generator, preprocessing, density estimation, YOLO detector.

---

## 16. Research Implementation Roadmap

| Phase | Component | Status |
| :--- | :--- | :--- |
| **Phase 1** | Project setup, modular structure, Python 3.12 `.venv` | **Completed** |
| **Phase 2** | Video ingestion, YOLO person detection, crowd density | **Completed** |
| **Phase 3** | ByteTrack person tracking (persistent IDs, velocity, trajectories) | **Completed** |
| **Safety Feature** | Weapon detection, fight detection, spatial association, fusion | **Completed** |
| **Phase 4** | Advanced Video Behavioral Analysis (ViT/Swin, Crowd Kinematics) | **Completed** |
| **Phase 5** | Audio capture, Mel Spectrograms, CNN sound classification, anomaly | **Completed** |
| **Phase 6** | Multimodal audio-visual fusion engine | **Completed** |
| **Phase 7** | Intelligent Decision Engine (Hysteresis, FSM, Recovery, Cooldown) | **Completed** |
| **Phase 8** | Alert Management & Notification Layer (Deduplication, Cooldown, Stubs) | **Completed** |
| **Phase 9** | Explainable AI (XAI) & Evidence Visualization | **Completed** |
| **Phase 10** | Real-Time Monitoring Dashboard & Operator Interface | **Completed** |
| **Phase 11** | SQLite persistent surveillance event database | **Completed** |
| **Phase 12** | Final Academic Evaluation & Benchmarking Framework | **Completed** |

---

## 17. Academic Limitations & Experimental Non-Claims

In strict adherence to scientific rigor and academic integrity:
1. **Unconfigured Real-World Ground-Truth Labels**: Public benchmark datasets with ground-truth anomaly annotations (e.g., UCF-Crime, ShanghaiTech, RWF-2000) are not bundled inside this repository. Consequently, real-world classification accuracy, precision, recall, F1, ROC-AUC, and PR-AUC are honestly reported as `NOT_AVAILABLE` with provenance `UNAVAILABLE`.
2. **Controlled Synthetic Scenarios**: The 13 controlled scenarios (`SCEN-01` through `SCEN-13`) are deterministic validation benchmarks designed to verify algorithmic state-machine transitions and cross-modal fusion behaviors. They are explicitly tagged with provenance `CONTROLLED_SYNTHETIC` and must **not** be cited as real-world field accuracy.
3. **Hardware Environment Latency**: Latency benchmarks reported (e.g., sub-millisecond core pipeline execution) represent single-process runtime measurements on local workstation hardware and vary depending on GPU availability, CPU clock speed, RAM bandwidth, and operating system load.
4. **Pretrained Baseline Models**: The acoustic analysis pipeline operates in an algorithmic spectral-loudness baseline mode in the absence of a custom fine-tuned CNN checkpoint. Similarly, Swin-T operates in feature extraction mode.
5. **Notification Channels**: External cloud delivery endpoints (email, SMS, push notifications) are implemented as safe functional stubs returning `"UNCONNECTED_STUB"` until live cloud provider credentials are provided.
6. **Optical Flow & Grad-CAM**: Visual saliency heatmaps operate in downstream explanation mode and are strictly omitted when trained neural explanation models are unattached, preventing artificial visualization fabrication.
7. **Operational CCTV Variables**: Real-world deployment performance is subject to environmental variables including camera mounting angle, vantage height, illumination changes, weather conditions, sensor dropout, and acoustic reverberation.

---

## 18. Academic Presentation Checklist & Demonstration Order

Follow this standardized 14-step sequence for academic reviews, viva voce, and system demonstrations:

| Step | Action | Command / Target | Observable Academic Outcome |
| :---: | :--- | :--- | :--- |
| **1** | **Start Project** | `python main.py --help` | Verify clean CLI flags across all 12 phases and help documentation. |
| **2** | **Live/Synthetic Video Detection** | `python main.py --mode video --synthetic` | Real-time YOLOv8/YOLO11 person bounding boxes and detection counts. |
| **3** | **Person Count & Density** | Terminal HUD / Log output | Spatial grid cell density ($persons/m^2$) and localized congestion levels. |
| **4** | **Crowd Tracking** | `python main.py --mode tracking --synthetic` | ByteTrack persistent IDs, trajectory histories, and velocity vectors. |
| **5** | **Trigger Movement Anomaly** | `python main.py --mode behavior --synthetic --demo-behavior` | Sudden kinetic dispersion surge and directional divergence entropy. |
| **6** | **Trigger Audio Anomaly** | `python main.py --mode audio --synthetic --demo-audio` | High-energy acoustic spike and spectral centroid elevation. |
| **7** | **Show Multimodal Corroboration** | `python main.py --mode multimodal --synthetic --demo-multimodal` | Simultaneous visual dispersion + scream triggering cross-modal synergy boost. |
| **8** | **Show Risk-State Transition** | Decision FSM logs | Clean state escalation: `NORMAL` $\to$ `SUSPICIOUS` $\to$ `HIGH_RISK` $\to$ `EMERGENCY`. |
| **9** | **Show Alert Generation** | Alert logs | Multi-tier alert dispatch with fingerprint deduplication and cooldown debouncing. |
| **10**| **Show XAI Explanation** | `python main.py --demo-xai` | Diagnostic `EvidenceRecord` attributing contributions to density, velocity, audio. |
| **11**| **Show Operator Dashboard** | `python main.py --demo-dashboard` | Real-time Streamlit HUD with time-series charts, alert actions, and health. |
| **12**| **Show Historical Database** | `python main.py --mode database` | SQLite WAL database status, indexed events, alerts, and schema version. |
| **13**| **Show Evaluation Report** | `python main.py --demo-evaluation` | Academic Markdown report (`final_evaluation.md`) and 4 generated plots. |
| **14**| **Show 221-Test Verification**| `python -m pytest tests/ -v` | Complete 221-test passing suite confirming 0 failures and 0 regressions. |

