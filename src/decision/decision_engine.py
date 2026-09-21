"""
==============================================================================
Intelligent Decision Engine (Phase 7).
Converts Phase 6 MultimodalAssessment evidence into stable surveillance risk
states using dynamic thresholds, dual-threshold hysteresis, temporal persistence,
recovery de-escalation, and critical safety evidence handling.
==============================================================================
"""

import time
from typing import Any, Dict, Optional, Tuple, Union

from src.decision.decision_record import DecisionEvent, RiskState
from src.decision.hysteresis import HysteresisManager
from src.decision.confirmation import TemporalConfirmationTracker
from src.decision.state_machine import DecisionStateMachine
from src.fusion.types import MultimodalAssessment
from src.utils.logger import setup_logger

logger = setup_logger("decision_engine")


class DecisionEngine:
    """
    Intelligent Decision Engine for real-time surveillance threat assessment.

    Pipeline:
      MultimodalAssessment -> Dynamic Hysteresis -> Temporal Confirmation ->
      State Machine Transition -> DecisionEvent Record
    """

    def __init__(
        self,
        enabled: bool = True,
        suspicious_enter: float = 0.45,
        suspicious_exit: float = 0.35,
        high_risk_enter: float = 0.65,
        high_risk_exit: float = 0.55,
        emergency_enter: float = 0.85,
        emergency_exit: float = 0.70,
        suspicious_confirmation: int = 3,
        high_risk_confirmation: int = 3,
        emergency_confirmation: int = 2,
        recovery_confirmation: int = 5,
        transition_cooldown_seconds: float = 3.0,
        critical_evidence_enabled: bool = True,
        critical_weapon_threshold: float = 0.90,
        critical_fight_threshold: float = 0.90,
        critical_multimodal_threshold: float = 0.90,
        dynamic_thresholds: bool = True,
        max_threshold_adjustment: float = 0.10,
        normal_threshold: Optional[float] = None,      # Legacy backward compatibility
        emergency_threshold: Optional[float] = None,   # Legacy backward compatibility
        suspicious_threshold: Optional[float] = None,
        high_risk_threshold: Optional[float] = None,
    ):
        self.enabled = bool(enabled)

        if suspicious_threshold is not None:
            suspicious_enter = float(suspicious_threshold)
            suspicious_exit = max(0.10, suspicious_enter - 0.10)
        if high_risk_threshold is not None:
            high_risk_enter = float(high_risk_threshold)
            high_risk_exit = max(0.20, high_risk_enter - 0.10)
        if emergency_threshold is not None:
            emergency_enter = float(emergency_threshold)
            emergency_exit = max(0.20, emergency_enter - 0.15)

        # Allow legacy thresholds to map cleanly if supplied
        if normal_threshold is not None and suspicious_enter == 0.45:
            suspicious_enter = float(normal_threshold)
            suspicious_exit = max(0.10, suspicious_enter - 0.10)
        if emergency_threshold is not None and emergency_enter == 0.85:
            emergency_enter = float(emergency_threshold)
            emergency_exit = max(0.20, emergency_enter - 0.15)

        # 1. Hysteresis & Dynamic Threshold Subsystem
        self.hysteresis = HysteresisManager(
            suspicious_enter=suspicious_enter,
            suspicious_exit=suspicious_exit,
            high_risk_enter=high_risk_enter,
            high_risk_exit=high_risk_exit,
            emergency_enter=emergency_enter,
            emergency_exit=emergency_exit,
            dynamic_thresholds=dynamic_thresholds,
            max_threshold_adjustment=max_threshold_adjustment,
        )

        # 2. Temporal Confirmation Subsystem
        self.confirmation = TemporalConfirmationTracker(
            suspicious_confirmation=suspicious_confirmation,
            high_risk_confirmation=high_risk_confirmation,
            emergency_confirmation=emergency_confirmation,
            recovery_confirmation=recovery_confirmation,
            critical_evidence_enabled=critical_evidence_enabled,
        )

        # 3. Decision State Machine
        self.state_machine = DecisionStateMachine(
            initial_state=RiskState.NORMAL,
            transition_cooldown_seconds=transition_cooldown_seconds,
        )

        # Critical Safety Evidence Config
        self.critical_evidence_enabled = bool(critical_evidence_enabled)
        self.critical_weapon_threshold = float(critical_weapon_threshold)
        self.critical_fight_threshold = float(critical_fight_threshold)
        self.critical_multimodal_threshold = float(critical_multimodal_threshold)

        logger.info(
            f"DecisionEngine initialized: enabled={self.enabled}, "
            f"critical_rules={self.critical_evidence_enabled} "
            f"(weapon>={self.critical_weapon_threshold}, fight>={self.critical_fight_threshold}, "
            f"multimodal>={self.critical_multimodal_threshold})."
        )

    @property
    def current_state(self) -> RiskState:
        return self.state_machine.current_state

    @property
    def suspicious_threshold(self) -> float:
        return self.hysteresis.base_suspicious_enter

    @property
    def high_risk_threshold(self) -> float:
        return self.hysteresis.base_high_risk_enter

    @property
    def emergency_threshold(self) -> float:
        return self.hysteresis.base_emergency_enter

    def check_critical_evidence(
        self,
        weapon_score: float,
        fight_score: float,
        multimodal_score: float,
        corroborated: bool,
    ) -> bool:
        """
        Evaluates whether severe, high-certainty threat signatures warrant
        accelerated emergency escalation.
        """
        if not self.critical_evidence_enabled:
            return False

        if weapon_score >= self.critical_weapon_threshold:
            return True
        if fight_score >= self.critical_fight_threshold:
            return True
        if multimodal_score >= self.critical_multimodal_threshold and corroborated:
            return True

        return False

    def compute_decision_confidence(
        self,
        model_confidence: float,
        anomaly_score: float,
        corroborated: bool,
        is_confirmed: bool,
        single_modality: bool,
    ) -> float:
        """
        Computes formal decision confidence, strictly separate from detector confidence.
        Conditioned on evidence consistency, temporal confirmation, and multi-sensor coverage.
        """
        # Base certainty from sensory models and score magnitude
        base = (0.65 * model_confidence) + (0.35 * min(1.0, anomaly_score + 0.15))

        # Modifiers
        if corroborated:
            base += 0.10
        if is_confirmed:
            base += 0.05
        if single_modality:
            base -= 0.10

        return float(max(0.10, min(0.98, round(base, 4))))

    def generate_reason(
        self,
        state: RiskState,
        score: float,
        weapon_score: float,
        fight_score: float,
        behavior_class: str,
        audio_event: str,
        corroborated: bool,
        is_critical: bool,
        synergy_type: Optional[str] = None,
    ) -> str:
        """
        Generates objective, scientifically grounded rationales.
        Does not assert certainty of real-world criminal intent.
        """
        if state == RiskState.NORMAL:
            return "Normal crowd activity with calm baseline kinematics and ambient acoustics."

        if state == RiskState.SUSPICIOUS:
            if audio_event in ("shouting", "scream", "alarm", "crash"):
                return f"Elevated acoustic anomaly ({audio_event.upper()}) detected; monitoring crowd reaction."
            if behavior_class in ("rapid_movement", "crowd_gathering"):
                return f"Elevated visual movement pattern ({behavior_class}) observed; temporal confirmation active."
            return "Atypical crowd movement or acoustic energy observed; increased surveillance scrutiny."

        if state == RiskState.HIGH_RISK:
            if fight_score >= 0.50 and audio_event in ("shouting", "distress"):
                return "Possible physical struggle with supporting acoustic indicators detected."
            if behavior_class == "scattering":
                return "High-speed crowd dispersion and directional divergence exceeding high-risk boundary."
            if corroborated:
                return "Multimodal corroboration of abnormal crowd activity across visual and audio streams."
            return "Persistent abnormal crowd behavioral divergence exceeding high-risk threshold."

        if state == RiskState.EMERGENCY:
            if is_critical and weapon_score >= self.critical_weapon_threshold:
                return f"Critical safety evidence: Potential dangerous-object detected ({weapon_score:.2f} confidence)."
            if is_critical and fight_score >= self.critical_fight_threshold:
                return f"Critical safety evidence: High-intensity physical altercation detected ({fight_score:.2f} fight score)."
            if synergy_type:
                return f"Potential emergency: Confirmed cross-modal crisis signature ({synergy_type})."
            if corroborated:
                return "Potential emergency: Strong multimodal corroboration of crisis event."
            return "Potential emergency: Confirmed critical crowd anomaly and rapid scattering behavior."

        if state == RiskState.RECOVERY:
            return f"Anomaly score decreasing below emergency boundary ({score:.2f}); entering recovery stabilization."

        return "Surveillance state evaluated."

    def evaluate(
        self,
        assessment: Optional[MultimodalAssessment] = None,
        current_time: Optional[float] = None,
        fused_assessment: Optional[MultimodalAssessment] = None,
    ) -> DecisionEvent:
        """
        Evaluates current MultimodalAssessment through the decision pipeline.

        Args:
            assessment: MultimodalAssessment instance from Phase 6.
            current_time: Reference epoch timestamp.
            fused_assessment: Alias for assessment.

        Returns:
            Structured DecisionEvent record.
        """
        curr_t = time.time() if current_time is None else float(current_time)
        assessment = assessment if assessment is not None else fused_assessment

        # Handle None or unconfigured assessment gracefully
        if assessment is None or not self.enabled:
            return DecisionEvent(
                timestamp=curr_t,
                previous_state=self.state_machine.previous_state.value,
                current_state=self.state_machine.current_state.value,
                score=0.0,
                confidence=0.80,
                reason="Nominal surveillance environment (engine disabled or stream empty).",
                dominant_modality="none",
                confirmed=True,
                transition=False,
            )

        # 1. Unpack multimodal evidence
        score = float(assessment.multimodal_score)
        model_conf = float(assessment.multimodal_confidence)
        dominant_modality = str(assessment.dominant_modality)
        video_score = float(assessment.video_score)
        audio_score = float(assessment.audio_score)

        breakdown = assessment.evidence_breakdown or {}
        v_ev = breakdown.get("video", {})
        a_ev = breakdown.get("audio", {})

        density_score = float(v_ev.get("density_score", 0.0))
        weapon_score = float(v_ev.get("weapon_score", 0.0))
        fight_score = float(v_ev.get("fight_score", 0.0))
        behavior_class = str(v_ev.get("behavior_class", "normal"))
        audio_event = str(a_ev.get("audio_event", "normal"))

        corroborated = bool(
            getattr(assessment.synergy, "synergy_detected", False)
            or (video_score >= 0.50 and audio_score >= 0.50)
        )
        single_modality = bool(
            dominant_modality in ("video", "audio")
            and (not v_ev.get("is_active", True) or not a_ev.get("is_active", True))
        )

        # 2. Critical Safety Evidence Evaluation
        is_critical = self.check_critical_evidence(
            weapon_score=weapon_score,
            fight_score=fight_score,
            multimodal_score=score,
            corroborated=corroborated,
        )

        # 3. Dynamic Hysteresis Evaluation
        current_state = self.state_machine.current_state
        candidate_state, effective_thresholds = self.hysteresis.evaluate_candidate_state(
            current_state=current_state,
            score=score,
            density_score=density_score,
            corroborated=corroborated,
            single_modality=single_modality,
        )

        # If critical evidence triggers, force candidate to EMERGENCY or HIGH_RISK
        if is_critical:
            if (
                weapon_score >= self.critical_weapon_threshold
                or (score >= self.critical_multimodal_threshold and corroborated)
                or (fight_score >= self.critical_fight_threshold and score >= 0.70)
            ):
                candidate_state = RiskState.EMERGENCY
            else:
                candidate_state = RiskState.HIGH_RISK

        # 4. Temporal Confirmation Tracking
        is_confirmed, consecutive_count, quota = self.confirmation.update(
            current_state=current_state,
            candidate_state=candidate_state,
            critical_evidence=is_critical,
        )

        # 5. State Machine Transition Execution
        state_changed = False
        transition_emitted = False
        prev_state = current_state

        if is_confirmed and candidate_state != current_state:
            state_changed, transition_emitted = self.state_machine.transition_to(
                target_state=candidate_state,
                current_time=curr_t,
                critical_evidence=is_critical,
            )
            new_state = self.state_machine.current_state

            # 6. Structured Transition Logging
            if state_changed:
                reason = self.generate_reason(
                    state=new_state,
                    score=score,
                    weapon_score=weapon_score,
                    fight_score=fight_score,
                    behavior_class=behavior_class,
                    audio_event=audio_event,
                    corroborated=corroborated,
                    is_critical=is_critical,
                    synergy_type=getattr(assessment.synergy, "synergy_type", None),
                )
                if new_state == RiskState.EMERGENCY:
                    logger.warning(
                        f"[WARNING] Potential emergency state transition: "
                        f"{prev_state.value} -> {new_state.value} | Score: {score:.2f} | "
                        f"Corroborated: {corroborated} | Reason: {reason}"
                    )
                else:
                    logger.info(
                        f"[INFO] Decision transition: {prev_state.value} -> {new_state.value} | "
                        f"Score: {score:.2f} | Reason: {reason}"
                    )
        else:
            new_state = self.state_machine.current_state
            reason = self.generate_reason(
                state=new_state,
                score=score,
                weapon_score=weapon_score,
                fight_score=fight_score,
                behavior_class=behavior_class,
                audio_event=audio_event,
                corroborated=corroborated,
                is_critical=is_critical,
                synergy_type=getattr(assessment.synergy, "synergy_type", None),
            )

        # 7. Distinct Decision Confidence Formulation
        decision_conf = self.compute_decision_confidence(
            model_confidence=model_conf,
            anomaly_score=score,
            corroborated=corroborated,
            is_confirmed=is_confirmed,
            single_modality=single_modality,
        )

        evidence_summary = {
            "density_score": round(density_score, 3),
            "behavior_class": behavior_class,
            "audio_event": audio_event,
            "synergy_detected": getattr(assessment.synergy, "synergy_detected", False),
            "synergy_type": getattr(assessment.synergy, "synergy_type", None),
            "false_positive_suppressed": assessment.false_positive_suppressed,
            "consecutive_confirmations": consecutive_count,
            "required_quota": quota,
        }

        return DecisionEvent(
            timestamp=curr_t,
            previous_state=self.state_machine.previous_state.value,
            current_state=new_state.value,
            score=round(score, 4),
            confidence=round(decision_conf, 4),
            reason=reason,
            dominant_modality=dominant_modality,
            video_score=round(video_score, 4),
            audio_score=round(audio_score, 4),
            weapon_score=round(weapon_score, 4),
            fight_score=round(fight_score, 4),
            corroborated=corroborated,
            critical_evidence=is_critical,
            confirmed=is_confirmed,
            transition=state_changed,
            dynamic_thresholds=effective_thresholds,
            evidence_summary=evidence_summary,
        )

    def reset(self) -> None:
        """Clears state machine, confirmation counters, and resets to baseline NORMAL."""
        self.state_machine.reset(RiskState.NORMAL)
        self.confirmation.reset()
