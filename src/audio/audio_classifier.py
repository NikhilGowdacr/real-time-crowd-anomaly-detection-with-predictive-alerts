"""
Audio Event Classifier Module.
Provides modular inference interfaces for custom audio CNN/YAMNet models
and an empirical Spectral-Loudness Baseline detector.

Academic & Research Integrity Notice:
Pre-trained computer vision or generic audio models cannot be claimed as
trained crowd-emergency classifiers without dedicated fine-tuning on datasets
like AudioSet, ESC-50, or UrbanSound8K. When custom weights are not provided,
this module explicitly operates in baseline spectral mode.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.audio.audio_preprocessing import PreprocessedAudio
from src.utils.logger import setup_logger

logger = setup_logger("audio_classifier")


@dataclass
class AudioClassificationResult:
    """
    Standardized classification output for an analyzed audio chunk.
    """
    event: str                                # e.g. "scream", "explosion_like", "normal", etc.
    confidence: float                         # [0.0 - 1.0]
    audio_anomaly_score: float                # Instantaneous anomaly indicator [0.0 - 1.0]
    timestamp: float                          # Epoch timestamp
    duration: float                           # Analyzed chunk duration in seconds
    is_custom_model: bool                     # True if custom fine-tuned weights were used
    model_mode: str                           # "custom_cnn", "spectral_baseline", or "disabled"
    status_message: str
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    inference_ms: float = 0.0

    def to_dict(self) -> Dict[str, Union[float, str]]:
        return {
            "event": self.event,
            "confidence": self.confidence,
            "audio_anomaly_score": self.audio_anomaly_score,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "model_mode": self.model_mode,
        }


class AudioClassifier:
    """
    Modular audio event classifier.
    Supports custom PyTorch CNN checkpoints or an empirical spectral/loudness baseline.
    """

    DEFAULT_CLASSES = [
        "normal",
        "speech",
        "crowd_noise",
        "shouting",
        "scream",
        "alarm",
        "explosion_like",
        "crash",
        "other_abnormal",
    ]

    # Mapping event categories to emergency severity weights
    SEVERITY_WEIGHTS = {
        "normal": 0.0,
        "speech": 0.05,
        "crowd_noise": 0.15,
        "shouting": 0.55,
        "alarm": 0.70,
        "scream": 0.85,
        "crash": 0.90,
        "explosion_like": 0.95,
        "other_abnormal": 0.60,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.50,
        classes: Optional[List[str]] = None,
        enabled: bool = True,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.classes = classes or self.DEFAULT_CLASSES
        self.enabled = enabled
        self.device = device

        self.is_custom_model = False
        self.model = None
        self.model_mode = "spectral_baseline"
        self.status_message = ""

        if not self.enabled:
            self.model_mode = "disabled"
            self.status_message = "Audio classification disabled in configuration."
            logger.info(self.status_message)
            return

        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        """Loads custom CNN weights or initializes empirical spectral baseline."""
        custom_file = Path(self.model_path) if self.model_path else None

        if custom_file and custom_file.exists():
            try:
                logger.info(f"Loading custom audio classifier model from '{custom_file}'...")
                self.model = torch.load(str(custom_file), map_location=self.device)
                self.model.eval()
                self.is_custom_model = True
                self.model_mode = "custom_cnn"
                self.status_message = f"Custom audio model loaded: {custom_file.name}"
                logger.info(self.status_message)
                return
            except Exception as e:
                logger.warning(f"Failed to load custom audio model from '{custom_file}': {e}. Using baseline.")

        # Spectral Baseline Fallback
        self.is_custom_model = False
        self.model_mode = "spectral_baseline"
        self.status_message = (
            "Audio classifier model not configured. Operating in Spectral-Loudness Baseline mode. "
            "(Academic notice: Evaluates acoustic energy and spectral centroid; not a trained deep classifier)."
        )
        logger.info(self.status_message)

    def classify(
        self,
        preprocessed: PreprocessedAudio,
        mel_spectrogram: np.ndarray,
        timestamp: Optional[float] = None,
        demo_event: Optional[str] = None,
    ) -> AudioClassificationResult:
        """
        Classifies an audio window using custom CNN or spectral baseline.

        Args:
            preprocessed: PreprocessedAudio container with normalized samples and RMS.
            mel_spectrogram: 2D log-Mel spectrogram array (n_mels, time_steps).
            timestamp: Epoch timestamp.
            demo_event: Optional override to simulate an event in test/demo mode.

        Returns:
            AudioClassificationResult dataclass.
        """
        curr_ts = timestamp if timestamp is not None else time.time()
        start_t = time.perf_counter()
        duration = len(preprocessed.samples) / 16000.0 if len(preprocessed.samples) > 0 else 1.0

        if not self.enabled:
            return AudioClassificationResult(
                event="normal",
                confidence=0.0,
                audio_anomaly_score=0.0,
                timestamp=curr_ts,
                duration=duration,
                is_custom_model=False,
                model_mode="disabled",
                status_message="Audio classifier disabled.",
            )

        # 1. Controlled Simulation / Test Trigger
        if demo_event and demo_event in self.SEVERITY_WEIGHTS:
            severity = self.SEVERITY_WEIGHTS.get(demo_event, 0.5)
            return AudioClassificationResult(
                event=demo_event,
                confidence=0.85,
                audio_anomaly_score=severity,
                timestamp=curr_ts,
                duration=duration,
                is_custom_model=False,
                model_mode="demo_simulation",
                status_message=f"[DEMO SIMULATION] Simulated sound event: {demo_event}",
                inference_ms=0.5,
            )

        # 2. Corrupt or Silent Audio Handling
        if not preprocessed.is_valid or preprocessed.samples.size == 0:
            return AudioClassificationResult(
                event="normal",
                confidence=0.0,
                audio_anomaly_score=0.0,
                timestamp=curr_ts,
                duration=duration,
                is_custom_model=self.is_custom_model,
                model_mode=self.model_mode,
                status_message="Silent or near-zero amplitude audio window.",
            )

        # 3. Custom CNN Inference Mode
        if self.is_custom_model and self.model is not None:
            try:
                # Format Mel spectrogram as tensor: (1, 1, n_mels, time_steps)
                tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                pred_idx = int(np.argmax(probs))
                pred_event = self.classes[min(pred_idx, len(self.classes) - 1)]
                confidence = float(probs[pred_idx])

                if confidence < self.confidence_threshold:
                    pred_event = "normal"
                    score = 0.0
                else:
                    severity = self.SEVERITY_WEIGHTS.get(pred_event, 0.5)
                    score = round(confidence * severity, 3)
                inference_ms = (time.perf_counter() - start_t) * 1000.0

                prob_dict = {cls_name: round(float(probs[i]), 3) for i, cls_name in enumerate(self.classes) if i < len(probs)}

                return AudioClassificationResult(
                    event=pred_event,
                    confidence=round(confidence, 3),
                    audio_anomaly_score=score,
                    timestamp=curr_ts,
                    duration=duration,
                    is_custom_model=True,
                    model_mode="custom_cnn",
                    status_message=f"CNN classified event: {pred_event} ({confidence:.2f})",
                    class_probabilities=prob_dict,
                    inference_ms=inference_ms,
                )
            except Exception as e:
                logger.error(f"Inference error in custom audio model: {e}")

        # 4. Empirical Spectral-Loudness Baseline Mode
        # Evaluates:
        # A. RMS energy and dBFS loudness
        # B. Spectral Centroid: center of gravity of frequencies
        #    - Screams / Shouts have high centroid (1800Hz - 4000Hz) and high RMS
        #    - Explosions / Crashes have low centroid (below 600Hz) and massive impulse peak
        #    - Ambient speech / crowd noise has moderate energy and mid centroid (600Hz - 1500Hz)

        # Calculate spectral centroid from Mel spectrogram energy distribution
        mel_energies = 10.0 ** (mel_spectrogram / 10.0)  # convert dB back to linear energy
        freq_weights = np.linspace(100.0, 7500.0, mel_spectrogram.shape[0])
        total_energy = np.sum(mel_energies)

        if total_energy > 1e-6:
            mean_spectrum = np.mean(mel_energies, axis=1)
            spectral_centroid = float(np.sum(mean_spectrum * freq_weights) / (np.sum(mean_spectrum) + 1e-6))
        else:
            spectral_centroid = 1000.0

        rms = preprocessed.rms
        peak = preprocessed.peak

        # Classification decision rules
        if peak >= 0.70 and spectral_centroid >= 2000.0:
            pred_event = "scream"
            confidence = min(1.0, (peak / 0.85) * (spectral_centroid / 3000.0))
        elif peak >= 0.60 and spectral_centroid >= 1400.0:
            pred_event = "shouting"
            confidence = min(1.0, peak * 1.2)
        elif peak >= 0.80 and spectral_centroid <= 750.0:
            pred_event = "explosion_like"
            confidence = min(1.0, peak)
        elif peak >= 0.70 and spectral_centroid <= 1100.0:
            pred_event = "crash"
            confidence = min(1.0, peak * 0.9)
        elif peak >= 0.40 and 1200.0 <= spectral_centroid <= 2200.0:
            pred_event = "alarm"
            confidence = 0.65
        elif rms >= 0.15:
            pred_event = "crowd_noise"
            confidence = 0.60
        elif rms >= 0.05:
            pred_event = "speech"
            confidence = 0.50
        else:
            pred_event = "normal"
            confidence = 0.90

        confidence = round(max(0.2, min(1.0, float(confidence))), 3)
        if confidence < self.confidence_threshold and pred_event != "normal":
            pred_event = "normal"

        severity = self.SEVERITY_WEIGHTS.get(pred_event, 0.0)

        # Acoustic anomaly score combines severity and confidence
        if pred_event in ("normal", "speech"):
            audio_score = 0.0
        else:
            audio_score = round(min(1.0, severity * confidence), 3)

        inference_ms = (time.perf_counter() - start_t) * 1000.0

        return AudioClassificationResult(
            event=pred_event,
            confidence=confidence,
            audio_anomaly_score=audio_score,
            timestamp=curr_ts,
            duration=duration,
            is_custom_model=False,
            model_mode="spectral_baseline",
            status_message=f"Spectral baseline detected '{pred_event}' (RMS={rms:.2f}, Centroid={spectral_centroid:.0f}Hz)",
            class_probabilities={pred_event: confidence},
            inference_ms=inference_ms,
        )
