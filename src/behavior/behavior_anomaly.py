"""
Behavior Anomaly Detector & Fusion Module.
Combines crowd kinematics and deep vision transformer representations into
a smoothed video anomaly score with multi-frame confirmation.
"""

from collections import deque
from dataclasses import dataclass
import time
from typing import Deque, Dict, List, Optional
import numpy as np

from src.behavior.behavior_features import CrowdBehaviorMetrics
from src.behavior.behavior_model import BehaviorPrediction
from src.utils.logger import setup_logger

logger = setup_logger("behavior_anomaly")


@dataclass
class BehaviorAssessment:
    """
    Consolidated video behavioral anomaly evaluation for the current frame.
    """
    kinematic_score: float                    # Macroscopic crowd kinematic score [0.0 - 1.0]
    deep_score: float                         # Deep transformer representation score [0.0 - 1.0]
    raw_score: float                          # Instantaneous fused score before smoothing
    smoothed_score: float                     # Rolling-average smoothed score [0.0 - 1.0]
    final_video_score: float                  # Confirmed anomaly score [0.0 - 1.0]
    behavior_class: str                       # e.g., "normal", "scattering", "crowd_gathering", etc.
    status: str                               # "NORMAL", "SUSPICIOUS", "EMERGENCY"
    consecutive_abnormal_frames: int
    is_confirmed: bool
    description: str
    active_people: int
    mean_speed: float


class BehaviorAnomalyDetector:
    """
    Combines kinematic crowd dynamics and deep Transformer behavioral features.
    Applies rolling exponential smoothing and multi-frame consecutive confirmation
    to eliminate frame-level false alerts.
    """

    def __init__(
        self,
        kinematic_weight: float = 0.40,
        deep_model_weight: float = 0.60,
        normal_threshold: float = 0.35,
        emergency_threshold: float = 0.70,
        confirmation_frames: int = 5,
        smoothing_window: int = 10,
    ):
        self.kinematic_weight = float(kinematic_weight)
        self.deep_model_weight = float(deep_model_weight)
        self.normal_threshold = float(normal_threshold)
        self.emergency_threshold = float(emergency_threshold)
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.smoothing_window = max(2, int(smoothing_window))

        # Normalize weights
        total_w = self.kinematic_weight + self.deep_model_weight
        if total_w > 0:
            self.kinematic_weight /= total_w
            self.deep_model_weight /= total_w

        # Rolling score buffer for temporal smoothing
        self._score_history: Deque[float] = deque(maxlen=self.smoothing_window)
        self.consecutive_abnormal_count: int = 0

        logger.info(
            f"BehaviorAnomalyDetector initialized: weights=(kinematic={self.kinematic_weight:.2f}, "
            f"deep={self.deep_model_weight:.2f}), thresholds=(normal<{self.normal_threshold}, "
            f"emerg>={self.emergency_threshold}), conf_frames={self.confirmation_frames}, "
            f"smoothing_window={self.smoothing_window}."
        )

    def evaluate(
        self,
        kinematics: CrowdBehaviorMetrics,
        deep_prediction: Optional[BehaviorPrediction] = None,
    ) -> BehaviorAssessment:
        """
        Integrates kinematic metrics and deep transformer predictions.

        Args:
            kinematics: CrowdBehaviorMetrics from CrowdBehaviorAnalyzer.
            deep_prediction: Optional BehaviorPrediction from BehaviorModel.

        Returns:
            BehaviorAssessment dataclass.
        """
        s_kin = float(kinematics.kinematic_score)

        if deep_prediction is not None and deep_prediction.confidence > 0:
            s_deep = float(deep_prediction.behavior_score)
            raw_fused = (self.kinematic_weight * s_kin) + (self.deep_model_weight * s_deep)
            # Candidate classification priority: deep model if custom, else kinematic
            if deep_prediction.is_custom_model and deep_prediction.behavior_class != "normal":
                candidate_class = deep_prediction.behavior_class
            else:
                candidate_class = kinematics.candidate_class
        else:
            s_deep = 0.0
            # If deep model is unconfigured or awaiting buffer, rely 100% on kinematic baseline
            raw_fused = s_kin
            candidate_class = kinematics.candidate_class

        # 1. Rolling Average Temporal Smoothing
        self._score_history.append(raw_fused)
        smoothed_score = float(np.mean(self._score_history))
        smoothed_score = round(min(1.0, max(0.0, smoothed_score)), 3)

        # 2. Multi-Frame Consecutive Confirmation
        if smoothed_score >= self.normal_threshold:
            self.consecutive_abnormal_count += 1
        else:
            self.consecutive_abnormal_count = max(0, self.consecutive_abnormal_count - 1)

        is_confirmed = self.consecutive_abnormal_count >= self.confirmation_frames

        # 3. Categorize Status & Formulate Final Video Score
        if is_confirmed and smoothed_score >= self.emergency_threshold:
            status = "EMERGENCY"
            final_video_score = smoothed_score
            desc = f"Confirmed emergency crowd behavior ({candidate_class}) sustained over {self.consecutive_abnormal_count} frames."
        elif is_confirmed and smoothed_score >= self.normal_threshold:
            status = "SUSPICIOUS"
            final_video_score = smoothed_score
            desc = f"Confirmed suspicious crowd movement pattern ({candidate_class})."
        else:
            status = "NORMAL"
            final_video_score = round(smoothed_score * 0.5, 3) if not is_confirmed else smoothed_score
            if self.consecutive_abnormal_count > 0:
                desc = f"Awaiting temporal confirmation ({self.consecutive_abnormal_count}/{self.confirmation_frames} frames)."
            else:
                desc = kinematics.description

        return BehaviorAssessment(
            kinematic_score=round(s_kin, 3),
            deep_score=round(s_deep, 3),
            raw_score=round(raw_fused, 3),
            smoothed_score=smoothed_score,
            final_video_score=final_video_score,
            behavior_class=candidate_class,
            status=status,
            consecutive_abnormal_frames=self.consecutive_abnormal_count,
            is_confirmed=is_confirmed,
            description=desc,
            active_people=kinematics.active_people,
            mean_speed=kinematics.mean_velocity,
        )

    def reset(self) -> None:
        """Clears smoothing history and resets confirmation counters."""
        self._score_history.clear()
        self.consecutive_abnormal_count = 0
