"""
Multi-Evidence Safety Fusion Module.
Combines Crowd Density, Movement Kinematics, Fight Scores, and Weapon Detections
into a unified safety risk score with correlation analysis and false-positive filtering.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.safety.weapon_detector import WeaponDetectionResult, DetectedObject
from src.safety.fight_detector import FightDetectionResult
from src.safety.pose_analyzer import KinematicFeatures
from src.video.crowd_density import DensityResult
from src.utils.logger import setup_logger

logger = setup_logger("safety_fusion")


class SafetyStatus(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    EMERGENCY = "EMERGENCY"


@dataclass
class SafetyEvent:
    """
    Structured event record designed for immediate logging and Phase 11 SQLite persistence.
    """
    timestamp: float
    event_type: str                           # e.g., "POTENTIAL_WEAPON_NEAR_STRUGGLE"
    confidence: float                         # [0.0 - 1.0]
    track_id: Optional[int]                   # Associated track ID (if spatially linked)
    object_type: Optional[str]                # Object class name (if applicable)
    crowd_score: float
    fight_score: float
    weapon_score: float
    movement_score: float
    safety_score: float
    status: str                               # "NORMAL", "SUSPICIOUS", "EMERGENCY"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "object_type": self.object_type,
            "crowd_score": self.crowd_score,
            "fight_score": self.fight_score,
            "weapon_score": self.weapon_score,
            "movement_score": self.movement_score,
            "safety_score": self.safety_score,
            "status": self.status,
            "description": self.description,
        }


@dataclass
class SafetyAssessment:
    """
    Consolidated safety telemetry for the current surveillance frame.
    """
    crowd_score: float
    fight_score: float
    weapon_score: float
    movement_score: float
    safety_score: float                       # Final composite score [0.0 - 1.0]
    status: SafetyStatus
    active_events: List[SafetyEvent] = field(default_factory=list)
    is_weapon_fight_correlated: bool = False
    cooldown_active: bool = False


class SafetyFusionEngine:
    """
    Multi-evidence decision fusion engine with configurable weighting,
    spatial-temporal correlation between weapons and violent altercations,
    and alert cooldown mechanisms.
    """

    def __init__(
        self,
        crowd_weight: float = 0.25,
        fight_weight: float = 0.35,
        weapon_weight: float = 0.40,
        normal_threshold: float = 0.35,
        emergency_threshold: float = 0.70,
        cooldown_seconds: float = 10.0,
        correlation_multiplier: float = 1.25,
    ):
        self.crowd_weight = float(crowd_weight)
        self.fight_weight = float(fight_weight)
        self.weapon_weight = float(weapon_weight)

        # Normalize weights so sum equals 1.0
        total_w = self.crowd_weight + self.fight_weight + self.weapon_weight
        if total_w > 0:
            self.crowd_weight /= total_w
            self.fight_weight /= total_w
            self.weapon_weight /= total_w

        self.normal_threshold = float(normal_threshold)
        self.emergency_threshold = float(emergency_threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.correlation_multiplier = float(correlation_multiplier)

        self.last_alert_time: float = 0.0
        self.event_history: List[SafetyEvent] = []

        logger.info(
            f"SafetyFusionEngine configured: weights=(crowd={self.crowd_weight:.2f}, "
            f"fight={self.fight_weight:.2f}, weapon={self.weapon_weight:.2f}), "
            f"thresholds=(normal<{self.normal_threshold}, emerg>={self.emergency_threshold}), "
            f"cooldown={self.cooldown_seconds}s."
        )

    def evaluate(
        self,
        density_result: DensityResult,
        kinematics: KinematicFeatures,
        fight_result: FightDetectionResult,
        weapon_result: WeaponDetectionResult,
        timestamp: Optional[float] = None,
    ) -> SafetyAssessment:
        """
        Integrates multi-modal evidence into a final composite safety risk score.

        Args:
            density_result: Crowd density output.
            kinematics: Motion kinematics output.
            fight_result: Fight detector output.
            weapon_result: Weapon detector output.
            timestamp: Current frame epoch timestamp.

        Returns:
            SafetyAssessment container with final score, status, and structured events.
        """
        curr_time = timestamp if timestamp is not None else time.time()

        s_crowd = float(density_result.density_score)
        s_fight = float(fight_result.fight_score)
        s_weapon = float(weapon_result.weapon_score)
        s_movement = float(kinematics.abnormal_movement_score)

        # 1. Base Linear Evidence Fusion
        base_safety_score = (
            (self.crowd_weight * s_crowd) +
            (self.fight_weight * s_fight) +
            (self.weapon_weight * s_weapon)
        )

        # 2. Weapon + Fight Correlation Analysis
        # Check if any dangerous object is spatially adjacent to persons involved in a fight
        is_correlated = False
        correlated_track: Optional[int] = None
        correlated_object: Optional[str] = None

        if weapon_result.detected_objects and fight_result.involved_track_ids:
            for obj in weapon_result.detected_objects:
                if obj.associated_track_id in fight_result.involved_track_ids:
                    is_correlated = True
                    correlated_track = obj.associated_track_id
                    correlated_object = obj.object_class
                    break

        # Priority boost if weapon detected near violent altercation
        final_safety_score = base_safety_score
        if is_correlated:
            final_safety_score = min(1.0, base_safety_score * self.correlation_multiplier)

        # 3. Categorize Status
        # Confirmed violent fight altercations or emergencies escalate status directly
        if (
            final_safety_score >= self.emergency_threshold
            or is_correlated
            or (s_fight >= 0.80 and s_weapon > 0)
            or fight_result.is_fight
            or fight_result.status == "EMERGENCY"
        ):
            status = SafetyStatus.EMERGENCY
            final_safety_score = max(final_safety_score, s_fight)
        elif (
            final_safety_score >= self.normal_threshold
            or s_fight >= 0.50
            or s_weapon >= 0.50
            or fight_result.is_suspicious
            or fight_result.status == "SUSPICIOUS"
        ):
            status = SafetyStatus.SUSPICIOUS
            final_safety_score = max(final_safety_score, round(s_fight * 0.85, 3))
        else:
            status = SafetyStatus.NORMAL

        final_safety_score = round(final_safety_score, 3)

        # 4. Check Alert Cooldown
        time_since_last_alert = curr_time - self.last_alert_time
        cooldown_active = (time_since_last_alert < self.cooldown_seconds)

        # 5. Generate Structured Events
        active_events: List[SafetyEvent] = []

        if status != SafetyStatus.NORMAL and not cooldown_active:
            self.last_alert_time = curr_time

            if is_correlated:
                event = SafetyEvent(
                    timestamp=curr_time,
                    event_type="POTENTIAL_WEAPON_NEAR_STRUGGLE",
                    confidence=max(s_weapon, s_fight),
                    track_id=correlated_track,
                    object_type=correlated_object,
                    crowd_score=s_crowd,
                    fight_score=s_fight,
                    weapon_score=s_weapon,
                    movement_score=s_movement,
                    safety_score=final_safety_score,
                    status=status.value,
                    description="Potential dangerous object detected in spatial vicinity of abnormal altercation.",
                )
                active_events.append(event)
                logger.warning(
                    f"[EMERGENCY ALERT] {event.description} "
                    f"(Object: {correlated_object}, Track: {correlated_track}, SafetyScore: {final_safety_score})"
                )

            elif s_weapon >= 0.50:
                first_obj = weapon_result.detected_objects[0]
                event = SafetyEvent(
                    timestamp=curr_time,
                    event_type="DANGEROUS_OBJECT_DETECTED",
                    confidence=first_obj.confidence,
                    track_id=first_obj.associated_track_id,
                    object_type=first_obj.object_class,
                    crowd_score=s_crowd,
                    fight_score=s_fight,
                    weapon_score=s_weapon,
                    movement_score=s_movement,
                    safety_score=final_safety_score,
                    status=status.value,
                    description=f"Potential dangerous object detected: {first_obj.object_class} ({int(first_obj.confidence * 100)}%).",
                )
                active_events.append(event)
                logger.warning(
                    f"[{status.value} ALERT] Potential dangerous object detected: {first_obj.object_class} "
                    f"(Confidence: {first_obj.confidence:.2f}, Associated Track: {first_obj.associated_track_id})"
                )

            elif s_fight >= 0.50:
                event = SafetyEvent(
                    timestamp=curr_time,
                    event_type="FIGHT_ALTERCATION_DETECTED",
                    confidence=s_fight,
                    track_id=fight_result.involved_track_ids[0] if fight_result.involved_track_ids else None,
                    object_type=None,
                    crowd_score=s_crowd,
                    fight_score=s_fight,
                    weapon_score=s_weapon,
                    movement_score=s_movement,
                    safety_score=final_safety_score,
                    status=status.value,
                    description=f"Physical altercation detected between tracks {fight_result.involved_track_ids}.",
                )
                active_events.append(event)
                logger.warning(
                    f"[{status.value} ALERT] Potential physical struggle detected "
                    f"(FightScore: {s_fight:.2f}, Tracks: {fight_result.involved_track_ids})"
                )

            elif s_crowd >= 0.80:
                event = SafetyEvent(
                    timestamp=curr_time,
                    event_type="OVERCROWDING_DETECTED",
                    confidence=s_crowd,
                    track_id=None,
                    object_type=None,
                    crowd_score=s_crowd,
                    fight_score=s_fight,
                    weapon_score=s_weapon,
                    movement_score=s_movement,
                    safety_score=final_safety_score,
                    status=status.value,
                    description=f"Critical crowd density threshold exceeded ({density_result.density_per_m2:.2f} pers/m2).",
                )
                active_events.append(event)

            # Store in history for Phase 11 DB compatibility
            for ev in active_events:
                self.event_history.append(ev)

        return SafetyAssessment(
            crowd_score=s_crowd,
            fight_score=s_fight,
            weapon_score=s_weapon,
            movement_score=s_movement,
            safety_score=final_safety_score,
            status=status,
            active_events=active_events,
            is_weapon_fight_correlated=is_correlated,
            cooldown_active=cooldown_active,
        )
