"""
Audio Anomaly Detection and Temporal Smoothing Module.
Applies rolling score smoothing and consecutive-window temporal confirmation
to eliminate acoustic false alarms and produce calibrated audio anomaly scores.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Dict, List, Optional, Union
import numpy as np

from src.audio.audio_classifier import AudioClassificationResult
from src.utils.logger import setup_logger

logger = setup_logger("audio_anomaly")


@dataclass
class AudioEventRecord:
    """
    Structured audio anomaly assessment container.
    Formatted for direct compatibility with the Phase 6 Multimodal Fusion engine.
    """
    audio_event: str                          # e.g. "scream", "alarm", "normal"
    audio_confidence: float                   # [0.0 - 1.0]
    audio_anomaly_score: float                # Temporally smoothed anomaly score [0.0 - 1.0]
    audio_timestamp: float                    # Epoch timestamp
    duration: float                           # Window duration (seconds)
    is_confirmed: bool                        # True if confirmed across multiple temporal windows
    consecutive_abnormal_count: int
    status: str                               # "NORMAL", "SUSPICIOUS", "EMERGENCY"
    description: str

    def to_dict(self) -> Dict[str, Union[float, str, bool, int]]:
        return {
            "audio_event": self.audio_event,
            "audio_confidence": self.audio_confidence,
            "audio_anomaly_score": self.audio_anomaly_score,
            "audio_timestamp": self.audio_timestamp,
            "duration": self.duration,
            "is_confirmed": self.is_confirmed,
            "status": self.status,
            "description": self.description,
        }


class AudioAnomalyDetector:
    """
    Evaluates rolling temporal persistence of detected sound events.
    Enforces multi-window confirmation so that single audio glitches or brief coughs
    do not trigger emergency alerts.
    """

    def __init__(
        self,
        temporal_window: int = 5,
        confirmation_windows: int = 2,
        smoothing_window: int = 5,
        suspicious_threshold: float = 0.40,
        emergency_threshold: float = 0.70,
    ):
        self.temporal_window = max(2, int(temporal_window))
        self.confirmation_windows = max(1, int(confirmation_windows))
        self.smoothing_window = max(1, int(smoothing_window))
        self.suspicious_threshold = float(suspicious_threshold)
        self.emergency_threshold = float(emergency_threshold)

        # Buffers for temporal analysis
        self._score_history: Deque[float] = deque(maxlen=self.smoothing_window)
        self._event_history: Deque[str] = deque(maxlen=self.temporal_window)
        self.consecutive_abnormal_count: int = 0

        logger.info(
            f"AudioAnomalyDetector initialized: temporal_win={self.temporal_window}, "
            f"conf_win={self.confirmation_windows}, smoothing={self.smoothing_window}, "
            f"thresholds=(susp={self.suspicious_threshold}, emerg={self.emergency_threshold})."
        )

    def evaluate(self, classification: AudioClassificationResult) -> AudioEventRecord:
        """
        Processes an incoming classification result and applies temporal smoothing.

        Args:
            classification: AudioClassificationResult from AudioClassifier.

        Returns:
            AudioEventRecord dataclass.
        """
        raw_score = float(classification.audio_anomaly_score)
        event_type = classification.event

        # 1. Rolling Score Smoothing
        self._score_history.append(raw_score)
        self._event_history.append(event_type)

        smoothed_score = float(np.mean(self._score_history))
        smoothed_score = round(min(1.0, max(0.0, smoothed_score)), 3)

        # 2. Consecutive Window Confirmation
        is_abnormal_event = (event_type not in ("normal", "speech", "crowd_noise")) or (raw_score >= self.suspicious_threshold)

        if is_abnormal_event:
            self.consecutive_abnormal_count += 1
        else:
            self.consecutive_abnormal_count = max(0, self.consecutive_abnormal_count - 1)

        is_confirmed = self.consecutive_abnormal_count >= self.confirmation_windows

        # 3. Status Categorization
        if is_confirmed and smoothed_score >= self.emergency_threshold:
            status = "EMERGENCY"
            final_score = smoothed_score
            desc = f"Confirmed emergency sound event ({event_type.upper()}) sustained across {self.consecutive_abnormal_count} windows."
            logger.warning(
                f"[EMERGENCY AUDIO ALERT] {desc} (Confidence: {classification.confidence:.2f}, Score: {final_score:.2f})"
            )
        elif is_confirmed and smoothed_score >= self.suspicious_threshold:
            status = "SUSPICIOUS"
            final_score = smoothed_score
            desc = f"Confirmed acoustic anomaly ({event_type.upper()}) across {self.consecutive_abnormal_count} windows."
            logger.warning(
                f"[SUSPICIOUS AUDIO ALERT] {desc} (Confidence: {classification.confidence:.2f}, Score: {final_score:.2f})"
            )
        else:
            status = "NORMAL"
            final_score = round(smoothed_score * 0.5, 3) if not is_confirmed else smoothed_score
            if self.consecutive_abnormal_count > 0:
                desc = f"Potential {event_type} detected — awaiting temporal confirmation ({self.consecutive_abnormal_count}/{self.confirmation_windows} windows)."
            else:
                desc = f"Nominal acoustic environment ({event_type})."

        return AudioEventRecord(
            audio_event=event_type,
            audio_confidence=classification.confidence,
            audio_anomaly_score=final_score,
            audio_timestamp=classification.timestamp,
            duration=classification.duration,
            is_confirmed=is_confirmed,
            consecutive_abnormal_count=self.consecutive_abnormal_count,
            status=status,
            description=desc,
        )

    def reset(self) -> None:
        """Clears buffers and confirmation counts."""
        self._score_history.clear()
        self._event_history.clear()
        self.consecutive_abnormal_count = 0
