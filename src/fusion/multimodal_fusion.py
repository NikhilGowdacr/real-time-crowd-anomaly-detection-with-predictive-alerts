"""
==============================================================================
Multimodal Audio-Visual Decision Fusion Engine (Phase 6).
Fuses video features (crowd density, movement anomaly, behavior anomaly,
fight score, weapon evidence) and audio features (anomaly score, detected event,
confidence, temporal confirmation) into a calibrated multimodal risk assessment.
Includes cross-modal synergy detection, dynamic modality re-weighting,
and intelligent false-positive suppression.
==============================================================================
"""

from collections import deque
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np

from src.audio.audio_anomaly import AudioEventRecord
from src.behavior.behavior_anomaly import BehaviorAssessment
from src.behavior.behavior_features import CrowdBehaviorMetrics
from src.fusion.temporal_aligner import TemporalStreamAligner
from src.fusion.types import CrossModalSynergy, ModalityEvidence, MultimodalAssessment
from src.safety.safety_fusion import SafetyAssessment
from src.safety.weapon_detector import WeaponDetectionResult
from src.video.crowd_density import DensityResult
from src.utils.logger import setup_logger

logger = setup_logger("multimodal_fusion")


class MultimodalFusionEngine:
    """
    Multimodal Decision Fusion Engine combining Video and Audio sensory streams.
    Implements cross-modal synergy amplification, intelligent noise suppression,
    graceful degradation for missing sensor streams, and rolling temporal smoothing.
    """

    def __init__(
        self,
        video_weight: float = 0.60,
        audio_weight: float = 0.40,
        density_sub_weight: float = 0.15,
        movement_sub_weight: float = 0.20,
        behavior_sub_weight: float = 0.25,
        fight_sub_weight: float = 0.20,
        weapon_sub_weight: float = 0.20,
        synergy_boost: float = 1.25,
        suppression_factor: float = 0.50,
        normal_threshold: float = 0.35,
        emergency_threshold: float = 0.70,
        smoothing_window: int = 5,
        temporal_tolerance_seconds: float = 1.5,
        stale_timeout_seconds: float = 2.5,
    ):
        self.video_weight = float(video_weight)
        self.audio_weight = float(audio_weight)
        self.synergy_boost = float(synergy_boost)
        self.suppression_factor = float(suppression_factor)
        self.normal_threshold = float(normal_threshold)
        self.emergency_threshold = float(emergency_threshold)
        self.smoothing_window = max(1, int(smoothing_window))

        # Video sub-feature weighting
        self.density_sub_weight = float(density_sub_weight)
        self.movement_sub_weight = float(movement_sub_weight)
        self.behavior_sub_weight = float(behavior_sub_weight)
        self.fight_sub_weight = float(fight_sub_weight)
        self.weapon_sub_weight = float(weapon_sub_weight)

        self._normalize_weights()

        # Rolling score history for smoothing
        self._score_history: Deque[float] = deque(maxlen=self.smoothing_window)

        # Stream synchronizer
        self.aligner = TemporalStreamAligner(
            tolerance_seconds=temporal_tolerance_seconds,
            stale_timeout_seconds=stale_timeout_seconds,
        )

        logger.info(
            f"MultimodalFusionEngine initialized: weights=(video={self.video_weight:.2f}, "
            f"audio={self.audio_weight:.2f}), synergy_boost={self.synergy_boost}x, "
            f"thresholds=(normal<{self.normal_threshold}, emerg>={self.emergency_threshold}), "
            f"smoothing_window={self.smoothing_window}."
        )

    def _normalize_weights(self) -> None:
        """Ensures bimodal and video sub-weights sum to 1.0."""
        total_bimodal = self.video_weight + self.audio_weight
        if total_bimodal > 0:
            self.video_weight /= total_bimodal
            self.audio_weight /= total_bimodal

        total_sub = (
            self.density_sub_weight
            + self.movement_sub_weight
            + self.behavior_sub_weight
            + self.fight_sub_weight
            + self.weapon_sub_weight
        )
        if total_sub > 0:
            self.density_sub_weight /= total_sub
            self.movement_sub_weight /= total_sub
            self.behavior_sub_weight /= total_sub
            self.fight_sub_weight /= total_sub
            self.weapon_sub_weight /= total_sub

    def extract_video_features(
        self,
        density: Optional[DensityResult] = None,
        kinematics: Optional[CrowdBehaviorMetrics] = None,
        behavior_assessment: Optional[BehaviorAssessment] = None,
        safety_assessment: Optional[SafetyAssessment] = None,
        weapon_result: Optional[WeaponDetectionResult] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extracts and normalizes the 5 required video intelligence features:
        1. Crowd Density
        2. Movement Anomaly
        3. Behavior Anomaly (ViT/Swin + Kinematics)
        4. Fight Score
        5. Weapon / Dangerous Object Evidence

        Returns:
            Tuple of (composite_video_score [0-1], video_confidence [0-1], evidence_dict).
        """
        # 1. Crowd Density Score
        density_score = 0.0
        density_level = "LOW"
        is_overcrowded = False
        density_m2 = 0.0

        if density is not None:
            density_level = density.density_level
            is_overcrowded = density.is_overcrowded
            density_m2 = density.density_per_m2
            if is_overcrowded or density_level == "CRITICAL":
                density_score = 1.0
            elif density_level == "HIGH":
                density_score = 0.75
            elif density_level == "MODERATE":
                density_score = 0.40
            else:
                density_score = min(0.25, density_m2 / 1.5)

        # 2. Movement Anomaly Score (Kinematics)
        movement_score = 0.0
        dispersion_rate = 0.0
        mean_speed = 0.0
        direction_entropy = 0.0

        if kinematics is not None:
            movement_score = float(kinematics.kinematic_score)
            dispersion_rate = float(kinematics.dispersion_rate)
            mean_speed = float(getattr(kinematics, "mean_velocity", getattr(kinematics, "mean_speed", 0.0)))
            direction_entropy = float(getattr(kinematics, "direction_variance", getattr(kinematics, "direction_entropy", 0.0)))
        elif behavior_assessment is not None:
            movement_score = float(behavior_assessment.kinematic_score)
            mean_speed = float(behavior_assessment.mean_speed)
        elif safety_assessment is not None:
            movement_score = float(safety_assessment.movement_score)

        # 3. Behavior Anomaly Score (ViT / Swin / Temporal Buffer)
        behavior_score = 0.0
        behavior_class = "normal"
        behavior_confirmed = False

        if behavior_assessment is not None:
            behavior_score = float(behavior_assessment.deep_score or behavior_assessment.final_video_score)
            behavior_class = str(behavior_assessment.behavior_class)
            behavior_confirmed = bool(behavior_assessment.is_confirmed)

        # 4. Fight Score
        fight_score = 0.0
        if safety_assessment is not None:
            fight_score = float(safety_assessment.fight_score)

        # 5. Weapon / Dangerous Object Score
        weapon_score = 0.0
        detected_weapons: List[str] = []

        if weapon_result is not None and weapon_result.detected_objects:
            weapon_score = float(weapon_result.max_confidence)
            detected_weapons = [
                getattr(obj, "object_class", getattr(obj, "class_name", "weapon"))
                for obj in weapon_result.detected_objects
            ]
        elif safety_assessment is not None:
            weapon_score = float(safety_assessment.weapon_score)

        # Weighted Linear Combination of the 5 Video Features
        s_weighted_video = (
            (self.density_sub_weight * density_score)
            + (self.movement_sub_weight * movement_score)
            + (self.behavior_sub_weight * behavior_score)
            + (self.fight_sub_weight * fight_score)
            + (self.weapon_sub_weight * weapon_score)
        )

        # Severity Safeguard: A verified high-severity weapon or violent fight cannot be masked by low density
        critical_visual_threat = max(weapon_score, fight_score)
        if critical_visual_threat >= 0.70:
            composite_video_score = max(s_weighted_video, critical_visual_threat * 0.90)
        else:
            composite_video_score = s_weighted_video

        composite_video_score = round(min(1.0, max(0.0, composite_video_score)), 4)

        # Video Confidence Estimation
        vid_conf_candidates = []
        if behavior_assessment is not None:
            vid_conf_candidates.append(0.85 if behavior_confirmed else 0.65)
        if safety_assessment is not None:
            vid_conf_candidates.append(0.80)
        if weapon_result is not None and weapon_result.detected_objects:
            vid_conf_candidates.append(weapon_result.max_confidence)
        video_confidence = float(np.mean(vid_conf_candidates)) if vid_conf_candidates else 0.75

        evidence = {
            "density_score": round(density_score, 3),
            "density_level": density_level,
            "density_m2": round(density_m2, 2),
            "movement_score": round(movement_score, 3),
            "mean_speed": round(mean_speed, 1),
            "dispersion_rate": round(dispersion_rate, 2),
            "direction_entropy": round(direction_entropy, 2),
            "behavior_score": round(behavior_score, 3),
            "behavior_class": behavior_class,
            "behavior_confirmed": behavior_confirmed,
            "fight_score": round(fight_score, 3),
            "weapon_score": round(weapon_score, 3),
            "detected_weapons": detected_weapons,
        }

        return composite_video_score, video_confidence, evidence

    def evaluate_cross_modal_synergy(
        self,
        video_evidence: Dict[str, Any],
        audio_evidence: Dict[str, Any],
        video_score: float,
        audio_score: float,
    ) -> CrossModalSynergy:
        """
        Detects mutually reinforcing sensory patterns between video and audio streams.
        Returns CrossModalSynergy dataclass with boost multiplier.
        """
        audio_event = str(audio_evidence.get("audio_event", "normal")).lower()
        audio_confirmed = bool(audio_evidence.get("is_confirmed", False))
        behavior_class = str(video_evidence.get("behavior_class", "normal")).lower()
        movement_score = float(video_evidence.get("movement_score", 0.0))
        fight_score = float(video_evidence.get("fight_score", 0.0))
        weapon_score = float(video_evidence.get("weapon_score", 0.0))

        # Rule 1: Scream / Alarm + Panic Scattering / Rapid Movement
        if audio_event in ("scream", "alarm", "explosion_like") and (
            behavior_class in ("scattering", "rapid_movement") or movement_score >= 0.50
        ):
            return CrossModalSynergy(
                synergy_detected=True,
                synergy_type="SCREAM_AND_SCATTER",
                boost_multiplier=self.synergy_boost,
                description=f"Correlated panic stampede: {audio_event.upper()} sound aligned with crowd {behavior_class}.",
                correlated_events=[audio_event, behavior_class],
            )

        # Rule 2: Violent Altercation (Fight + Shouting / Distress)
        if (fight_score >= 0.45 or behavior_class == "possible_fight") and (
            audio_event in ("shouting", "scream", "crash") or audio_score >= 0.50
        ):
            return CrossModalSynergy(
                synergy_detected=True,
                synergy_type="FIGHT_AND_SHOUTING",
                boost_multiplier=self.synergy_boost,
                description="Correlated physical struggle: visual combat cues corroborated by acoustic disturbance.",
                correlated_events=["fight", audio_event],
            )

        # Rule 3: Explosion / Crash + Rapid Dispersal
        if audio_event in ("explosion_like", "crash") and (movement_score >= 0.40 or video_score >= 0.40):
            return CrossModalSynergy(
                synergy_detected=True,
                synergy_type="EXPLOSION_AND_DISPERSAL",
                boost_multiplier=min(1.40, self.synergy_boost * 1.10),
                description=f"Catastrophic acoustic impulse ({audio_event.upper()}) coupled with sudden visual movement.",
                correlated_events=[audio_event, "rapid_movement"],
            )

        # Rule 4: Dangerous Object / Weapon + Distress Audio
        if weapon_score >= 0.45 and (audio_event in ("scream", "shouting", "alarm") or audio_score >= 0.45):
            return CrossModalSynergy(
                synergy_detected=True,
                synergy_type="WEAPON_AND_DISTRESS",
                boost_multiplier=self.synergy_boost,
                description="Armed threat escalation: weapon presence confirmed by acoustic distress sounds.",
                correlated_events=["weapon", audio_event],
            )

        # Rule 5: Generic High-Confidence Cross-Modal Anomaly
        if video_score >= 0.60 and audio_score >= 0.60:
            return CrossModalSynergy(
                synergy_detected=True,
                synergy_type="BIMODAL_CONCURRENT_ANOMALY",
                boost_multiplier=self.synergy_boost,
                description="Concurrent high-severity video and audio anomalies independently confirmed.",
                correlated_events=["video_anomaly", "audio_anomaly"],
            )

        return CrossModalSynergy(synergy_detected=False)

    def evaluate_false_positive_suppression(
        self,
        video_evidence: Dict[str, Any],
        audio_evidence: Dict[str, Any],
        video_score: float,
        audio_score: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determines if uncorroborated single-sensor noise or glitches should be dampened.

        Returns:
            Tuple of (should_suppress: bool, reason: Optional[str]).
        """
        audio_event = str(audio_evidence.get("audio_event", "normal")).lower()
        audio_confirmed = bool(audio_evidence.get("is_confirmed", False))
        movement_score = float(video_evidence.get("movement_score", 0.0))
        fight_score = float(video_evidence.get("fight_score", 0.0))
        weapon_score = float(video_evidence.get("weapon_score", 0.0))
        behavior_score = float(video_evidence.get("behavior_score", 0.0))
        behavior_confirmed = bool(video_evidence.get("behavior_confirmed", False))

        # Case A: Acoustic transient spike in a completely calm, normal crowd
        # (e.g. dropped metal item, door slam, mic bump with zero crowd reaction)
        if (
            audio_score >= 0.50
            and not audio_confirmed
            and movement_score < 0.25
            and fight_score < 0.20
            and weapon_score < 0.20
            and video_score < 0.25
        ):
            return (
                True,
                f"Acoustic spike ({audio_event.upper()}) uncorroborated by crowd movement; calm kinematics preserved.",
            )

        # Case B: Isolated visual motion spike in a completely nominal, quiet acoustic environment
        # (e.g. single commuter running to catch bus, camera tracking jitter)
        if (
            (video_score >= 0.25 or movement_score >= 0.45 or behavior_score >= 0.45)
            and not behavior_confirmed
            and weapon_score < 0.20
            and fight_score < 0.20
            and audio_event in ("normal", "speech")
            and audio_score < 0.20
        ):
            return (
                True,
                "Isolated visual motion spike unconfirmed across time and unsupported by acoustic environment.",
            )

        return False, None

    def fuse(
        self,
        density: Optional[DensityResult] = None,
        kinematics: Optional[CrowdBehaviorMetrics] = None,
        behavior_assessment: Optional[BehaviorAssessment] = None,
        safety_assessment: Optional[SafetyAssessment] = None,
        weapon_result: Optional[WeaponDetectionResult] = None,
        audio_record: Optional[AudioEventRecord] = None,
        current_time: Optional[float] = None,
        video_active: Optional[bool] = None,
        video_assessment: Optional[Any] = None,
        audio_assessment: Optional[Any] = None,
    ) -> MultimodalAssessment:
        """
        Core multimodal fusion method. Integrates all 5 video features and 4 audio features.
        Supports both raw pipeline outputs and pre-computed assessment structures/dictionaries.
        """
        curr_t = current_time if current_time is not None else time.time()

        # Map positional dict arguments if passed as assessments
        if isinstance(density, dict) and video_assessment is None:
            video_assessment = density
            density = None
        if isinstance(kinematics, dict) and audio_assessment is None:
            audio_assessment = kinematics
            kinematics = None

        # Update stream synchronizer
        self.aligner.register_video(curr_t)
        if audio_record is not None:
            self.aligner.register_audio(audio_record, audio_record.audio_timestamp or curr_t)

        aligned_audio, audio_is_fresh = self.aligner.get_aligned_audio(curr_t)

        # 1. Extract Video Evidence (5 features)
        video_score, video_conf, video_evidence = self.extract_video_features(
            density=density,
            kinematics=kinematics,
            behavior_assessment=behavior_assessment,
            safety_assessment=safety_assessment,
            weapon_result=weapon_result,
        )

        video_provided = (
            density is not None
            or kinematics is not None
            or behavior_assessment is not None
            or safety_assessment is not None
            or weapon_result is not None
        )

        if video_assessment is not None:
            if isinstance(video_assessment, dict):
                video_score = float(video_assessment.get("anomaly_score", video_assessment.get("video_score", video_assessment.get("final_video_score", 0.0))))
                video_conf = float(video_assessment.get("confidence", 0.80))
                video_evidence.update(video_assessment)
                video_evidence["movement_score"] = float(video_assessment.get("movement_score", video_assessment.get("kinematic_score", 0.0)))
                video_evidence["fight_score"] = float(video_assessment.get("fight_score", 0.0))
                video_evidence["weapon_score"] = float(video_assessment.get("weapon_score", 0.0))
                video_evidence["density_score"] = float(video_assessment.get("density_score", video_assessment.get("crowd_density", 0.0)))
            else:
                video_score = getattr(video_assessment, "final_video_score", getattr(video_assessment, "anomaly_score", 0.0))
                video_conf = getattr(video_assessment, "confidence", 0.80)
            video_provided = True
            if video_active is None:
                video_active = True

        # 2. Extract Audio Evidence (4 features)
        if aligned_audio is not None and audio_is_fresh:
            audio_score = float(aligned_audio.audio_anomaly_score)
            audio_conf = float(aligned_audio.audio_confidence)
            audio_evidence = {
                "audio_event": aligned_audio.audio_event,
                "audio_confidence": round(aligned_audio.audio_confidence, 3),
                "audio_anomaly_score": round(audio_score, 3),
                "is_confirmed": aligned_audio.is_confirmed,
                "consecutive_abnormal_count": aligned_audio.consecutive_abnormal_count,
                "is_active": True,
            }
            audio_active = True
        else:
            audio_score = 0.0
            audio_conf = 0.0
            audio_evidence = {
                "audio_event": "none",
                "audio_confidence": 0.0,
                "audio_anomaly_score": 0.0,
                "is_confirmed": False,
                "consecutive_abnormal_count": 0,
                "is_active": False,
            }
            audio_active = False

        if audio_assessment is not None:
            if isinstance(audio_assessment, dict):
                audio_score = float(audio_assessment.get("audio_score", audio_assessment.get("audio_anomaly_score", 0.0)))
                audio_conf = float(audio_assessment.get("audio_confidence", audio_assessment.get("confidence", 0.80)))
                audio_event_name = audio_assessment.get("sound_event", audio_assessment.get("audio_event", "normal"))
                audio_evidence.update({
                    "audio_event": audio_event_name,
                    "audio_confidence": audio_conf,
                    "audio_anomaly_score": audio_score,
                    "is_confirmed": bool(audio_assessment.get("is_confirmed", False)),
                    "consecutive_abnormal_count": int(audio_assessment.get("consecutive_abnormal_count", 0)),
                    "is_active": True,
                })
            else:
                audio_score = getattr(audio_assessment, "audio_anomaly_score", getattr(audio_assessment, "audio_score", 0.0))
                audio_conf = getattr(audio_assessment, "audio_confidence", 0.80)
                audio_evidence["audio_event"] = getattr(audio_assessment, "audio_event", "normal")
                audio_evidence["audio_anomaly_score"] = audio_score
                audio_evidence["audio_confidence"] = audio_conf
                audio_evidence["is_active"] = True
            audio_active = True

        # 3. Dynamic Modality Re-Weighting (Graceful Single-Modality Fallback)
        if video_active is None:
            video_active = video_provided

        video_evidence["is_active"] = bool(video_active)

        if video_active and audio_active:
            w_v = self.video_weight
            w_a = self.audio_weight
        elif video_active and not audio_active:
            w_v = 1.0
            w_a = 0.0
        elif not video_active and audio_active:
            w_v = 0.0
            w_a = 1.0
        else:
            w_v = 0.5
            w_a = 0.5

        # 4. Cross-Modal Synergy Detection
        synergy = self.evaluate_cross_modal_synergy(
            video_evidence=video_evidence,
            audio_evidence=audio_evidence,
            video_score=video_score,
            audio_score=audio_score,
        )

        # 5. False-Positive Suppression
        suppress_fp, suppress_reason = self.evaluate_false_positive_suppression(
            video_evidence=video_evidence,
            audio_evidence=audio_evidence,
            video_score=video_score,
            audio_score=audio_score,
        )

        # 6. Score Fusion Computation
        if suppress_fp:
            # Dampen the uncorroborated single sensor score
            effective_video_score = video_score * (self.suppression_factor if w_a > 0 and audio_score < 0.2 else 1.0)
            effective_audio_score = audio_score * (self.suppression_factor if w_v > 0 and video_score < 0.2 else 1.0)
        else:
            effective_video_score = video_score
            effective_audio_score = audio_score

        raw_fused = (w_v * effective_video_score) + (w_a * effective_audio_score)

        if synergy.synergy_detected:
            raw_fused = min(1.0, raw_fused * synergy.boost_multiplier)

        # 7. Rolling Temporal Score Smoothing
        self._score_history.append(raw_fused)
        smoothed_score = float(np.mean(self._score_history))
        final_multimodal_score = round(min(1.0, max(0.0, smoothed_score)), 4)

        # 8. Joint Confidence Estimation
        joint_conf = (w_v * video_conf) + (w_a * audio_conf)
        if synergy.synergy_detected:
            joint_conf = min(1.0, joint_conf * 1.15)
        elif suppress_fp:
            joint_conf = max(0.40, joint_conf * 0.85)
        final_confidence = round(min(1.0, max(0.1, joint_conf)), 4)

        # 9. Dominant Modality Determination
        if w_a == 0.0:
            dominant_modality = "video"
        elif w_v == 0.0:
            dominant_modality = "audio"
        elif abs(video_score - audio_score) < 0.20:
            dominant_modality = "bimodal"
        elif video_score > audio_score:
            dominant_modality = "video"
        else:
            dominant_modality = "audio"

        # 10. Status Classification
        if final_multimodal_score >= self.emergency_threshold or (synergy.synergy_detected and final_multimodal_score >= 0.60):
            status = "EMERGENCY"
        elif final_multimodal_score >= self.normal_threshold:
            status = "SUSPICIOUS"
        else:
            status = "NORMAL"

        # Description Generation
        if synergy.synergy_detected:
            desc = f"[CROSS-MODAL SYNERGY] {synergy.description} (Fused Score: {final_multimodal_score:.2f})"
        elif suppress_fp:
            desc = f"[NOISE SUPPRESSED] {suppress_reason} (Filtered Score: {final_multimodal_score:.2f})"
        elif status == "EMERGENCY":
            desc = f"Critical emergency situation detected by {dominant_modality.upper()} modality (Score: {final_multimodal_score:.2f})."
        elif status == "SUSPICIOUS":
            desc = f"Suspicious activity observed by {dominant_modality.upper()} modality (Score: {final_multimodal_score:.2f})."
        else:
            desc = f"Nominal surveillance environment (Score: {final_multimodal_score:.2f})."

        combined_evidence = {
            "video": video_evidence,
            "audio": audio_evidence,
            "fusion_weights": {"video": round(w_v, 2), "audio": round(w_a, 2)},
        }

        return MultimodalAssessment(
            multimodal_score=final_multimodal_score,
            multimodal_confidence=final_confidence,
            video_score=video_score,
            audio_score=audio_score,
            status=status,
            dominant_modality=dominant_modality,
            synergy=synergy,
            false_positive_suppressed=suppress_fp,
            suppression_reason=suppress_reason,
            evidence_breakdown=combined_evidence,
            timestamp=curr_t,
            description=desc,
        )

    def reset(self) -> None:
        """Clears score buffers and stream aligner state."""
        self._score_history.clear()
        self.aligner.reset()
