"""
Fight and Abnormal Movement Detection Module.
Evaluates multi-frame temporal dynamics (sliding window) to detect violent struggles,
aggressive physical altercations, and sudden panic clustering.

Research & Safety Guideline:
Single anomalous frames never trigger an emergency alert.
Detection mandates temporal persistence and consecutive-frame confirmation.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.safety.pose_analyzer import KinematicFeatures, InteractingPair
from src.video.crowd_tracker import TrackedPerson
from src.utils.logger import setup_logger

logger = setup_logger("fight_detector")


@dataclass
class FightDetectionResult:
    """
    Result container for temporal fight and aggressive interaction analysis.
    """
    is_fight: bool
    is_suspicious: bool
    fight_score: float                        # Confidence score [0.0 - 1.0]
    involved_track_ids: List[int]             # Track IDs implicated in the struggle
    interaction_bbox: Optional[np.ndarray]    # [x1, y1, x2, y2] bounding box of struggle
    status: str                               # "NORMAL", "SUSPICIOUS", "EMERGENCY"
    consecutive_abnormal_frames: int          # Multi-frame confirmation counter
    reason: str


class FightDetector:
    """
    Temporal sliding-window classifier for physical altercations and violent movement.
    Requires continuous evidence over multiple frames to suppress false positives.
    """

    def __init__(
        self,
        enabled: bool = True,
        temporal_window: int = 20,
        suspicious_threshold: float = 0.50,
        emergency_threshold: float = 0.75,
        confirmation_frames: int = 5,
        cooldown_seconds: float = 10.0,
    ):
        self.enabled = enabled
        self.temporal_window = max(5, int(temporal_window))
        self.suspicious_threshold = float(suspicious_threshold)
        self.emergency_threshold = float(emergency_threshold)
        self.confirmation_frames = max(2, int(confirmation_frames))
        self.cooldown_seconds = float(cooldown_seconds)

        # Sliding window buffer of recent kinematic feature frames
        self.history: Deque[KinematicFeatures] = deque(maxlen=self.temporal_window)

        # Multi-frame temporal state
        self.consecutive_abnormal_count = 0
        self.last_emergency_time: float = 0.0

        logger.info(
            f"FightDetector initialized (window={self.temporal_window}, "
            f"conf_frames={self.confirmation_frames}, "
            f"susp_thresh={self.suspicious_threshold}, "
            f"emerg_thresh={self.emergency_threshold})."
        )

    def update(
        self,
        kinematics: KinematicFeatures,
        tracked_persons: List[TrackedPerson],
        timestamp: Optional[float] = None,
        demo_trigger: bool = False,
    ) -> FightDetectionResult:
        """
        Ingests current frame kinematics and computes temporal fight score.

        Args:
            kinematics: KinematicFeatures extracted from current frame.
            tracked_persons: List of currently active tracked persons.
            timestamp: Current epoch timestamp.
            demo_trigger: Flag to simulate fight dynamics for test/demo mode.

        Returns:
            FightDetectionResult dataclass.
        """
        curr_time = timestamp if timestamp is not None else time.time()

        if not self.enabled:
            return FightDetectionResult(
                is_fight=False,
                is_suspicious=False,
                fight_score=0.0,
                involved_track_ids=[],
                interaction_bbox=None,
                status="NORMAL",
                consecutive_abnormal_frames=0,
                reason="Fight detection disabled in configuration.",
            )

        self.history.append(kinematics)

        # Controlled Demo / Simulation Mode
        if demo_trigger and len(tracked_persons) >= 2:
            return self._generate_simulated_fight(tracked_persons, curr_time)

        # 1. Analyze Temporal Inter-Person Interactions across window
        # Track persistent interacting pairs across the window
        pair_occurrence_count: Dict[Tuple[int, int], int] = {}
        pair_reciprocal_sum: Dict[Tuple[int, int], float] = {}
        pair_latest_bbox: Dict[Tuple[int, int], np.ndarray] = {}

        for frame_feat in self.history:
            for pair in frame_feat.interacting_pairs:
                key = (pair.track_id_1, pair.track_id_2)
                pair_occurrence_count[key] = pair_occurrence_count.get(key, 0) + 1
                pair_reciprocal_sum[key] = pair_reciprocal_sum.get(key, 0.0) + pair.reciprocal_motion_score
                pair_latest_bbox[key] = pair.union_bbox

        # 2. Compute Fight Indicators
        highest_pair_score = 0.0
        most_critical_pair: Optional[Tuple[int, int]] = None
        critical_bbox: Optional[np.ndarray] = None

        for key, count in pair_occurrence_count.items():
            persistence_ratio = count / len(self.history)
            avg_reciprocal = pair_reciprocal_sum[key] / count

            # A fight exhibits high persistence (agents stay locked in close proximity)
            # combined with violent reciprocal motion
            pair_fight_indicator = (0.50 * persistence_ratio) + (0.50 * avg_reciprocal)

            if pair_fight_indicator > highest_pair_score:
                highest_pair_score = pair_fight_indicator
                most_critical_pair = key
                critical_bbox = pair_latest_bbox.get(key)

        # 3. Sudden Velocity Variance and Acceleration Outliers across window
        recent_max_accels = [f.max_individual_accel for f in self.history]
        mean_recent_accel = float(np.mean(recent_max_accels))
        accel_factor = min(1.0, mean_recent_accel / 12.0)

        # Combined temporal fight score [0.0 - 1.0]
        raw_fight_score = round(
            0.70 * highest_pair_score + 0.30 * accel_factor,
            3,
        )

        # 4. Consecutive-Frame Confirmation (False-Positive Suppression)
        if raw_fight_score >= self.suspicious_threshold:
            self.consecutive_abnormal_count += 1
        else:
            self.consecutive_abnormal_count = max(0, self.consecutive_abnormal_count - 1)

        # Threshold evaluation with temporal persistence check
        confirmed_abnormal = self.consecutive_abnormal_count >= self.confirmation_frames

        if confirmed_abnormal and raw_fight_score >= self.emergency_threshold:
            status = "EMERGENCY"
            is_fight = True
            is_suspicious = False
            reason = f"High-confidence fight confirmed over {self.consecutive_abnormal_count} consecutive frames."
        elif confirmed_abnormal and raw_fight_score >= self.suspicious_threshold:
            status = "SUSPICIOUS"
            is_fight = False
            is_suspicious = True
            reason = f"Persistent aggressive interaction detected ({self.consecutive_abnormal_count} frames)."
        else:
            status = "NORMAL"
            is_fight = False
            is_suspicious = False
            if self.consecutive_abnormal_count > 0:
                reason = f"Awaiting temporal confirmation ({self.consecutive_abnormal_count}/{self.confirmation_frames} frames)."
            else:
                reason = "No confirmed violent interactions."

        # Active involved track IDs
        involved_tracks = list(most_critical_pair) if (most_critical_pair and confirmed_abnormal) else []

        return FightDetectionResult(
            is_fight=is_fight,
            is_suspicious=is_suspicious,
            fight_score=raw_fight_score if confirmed_abnormal else round(raw_fight_score * 0.5, 3),
            involved_track_ids=involved_tracks,
            interaction_bbox=critical_bbox if confirmed_abnormal else None,
            status=status,
            consecutive_abnormal_frames=self.consecutive_abnormal_count,
            reason=reason,
        )

    def _generate_simulated_fight(
        self,
        tracked_persons: List[TrackedPerson],
        timestamp: float,
    ) -> FightDetectionResult:
        """Simulates fight dynamics between the first two tracked agents."""
        p1 = tracked_persons[0]
        p2 = tracked_persons[1]
        b1 = p1.current_bbox
        b2 = p2.current_bbox

        union_bbox = np.array([
            min(b1[0], b2[0]) - 10,
            min(b1[1], b2[1]) - 10,
            max(b1[2], b2[2]) + 10,
            max(b1[3], b2[3]) + 10,
        ], dtype=np.float32)

        self.consecutive_abnormal_count = self.confirmation_frames + 2

        return FightDetectionResult(
            is_fight=True,
            is_suspicious=False,
            fight_score=0.74,
            involved_track_ids=[p1.track_id, p2.track_id],
            interaction_bbox=union_bbox,
            status="EMERGENCY",
            consecutive_abnormal_frames=self.consecutive_abnormal_count,
            reason="[DEMO SIMULATION] Simulated violent physical struggle between Track IDs.",
        )

    def draw_fight_detections(
        self,
        frame: np.ndarray,
        result: FightDetectionResult,
    ) -> np.ndarray:
        """
        Draws highlighted bounding regions and status tags around active fight interactions.
        """
        if result.interaction_bbox is None or result.status == "NORMAL":
            return frame

        annotated = frame.copy()
        x1, y1, x2, y2 = result.interaction_bbox.astype(int)

        color = (0, 0, 230) if result.status == "EMERGENCY" else (0, 140, 255)
        tag_text = f"FIGHT {int(result.fight_score * 100)}%" if result.status == "EMERGENCY" else f"SUSPICIOUS INTERACTION {int(result.fight_score * 100)}%"

        # Draw outer glowing/thick rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Draw banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(tag_text, font, 0.55, 2)
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(annotated, tag_text, (x1 + 4, label_y), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Subtitle: Involved IDs
        if result.involved_track_ids:
            sub_text = f"Tracks: {result.involved_track_ids}"
            cv2.putText(annotated, sub_text, (x1 + 4, y2 + 16), font, 0.42, color, 1, cv2.LINE_AA)

        return annotated
