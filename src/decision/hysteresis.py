"""
==============================================================================
Hysteresis and Dynamic Threshold Management (Phase 7).
Enforces separate escalation and de-escalation thresholds to prevent rapid
state oscillation around decision boundaries.
==============================================================================
"""

from typing import Any, Dict, Optional, Tuple
from src.decision.decision_record import RiskState
from src.utils.logger import setup_logger

logger = setup_logger("decision_hysteresis")


class HysteresisManager:
    """
    Manages dual-threshold hysteresis boundaries and context-aware dynamic adjustments.
    
    Guarantees:
      1. T_enter > T_exit for all risk state transitions (deadband stability).
      2. Dynamic adjustments are bounded within [-max_adjustment, +max_adjustment].
      3. All computed thresholds are strictly clamped within [0.05, 0.98].
    """

    def __init__(
        self,
        suspicious_enter: float = 0.45,
        suspicious_exit: float = 0.35,
        high_risk_enter: float = 0.65,
        high_risk_exit: float = 0.55,
        emergency_enter: float = 0.85,
        emergency_exit: float = 0.70,
        dynamic_thresholds: bool = True,
        max_threshold_adjustment: float = 0.10,
    ):
        self.base_suspicious_enter = float(suspicious_enter)
        self.base_suspicious_exit = float(suspicious_exit)
        self.base_high_risk_enter = float(high_risk_enter)
        self.base_high_risk_exit = float(high_risk_exit)
        self.base_emergency_enter = float(emergency_enter)
        self.base_emergency_exit = float(emergency_exit)

        self.dynamic_thresholds = bool(dynamic_thresholds)
        self.max_adjustment = max(0.0, min(0.20, float(max_threshold_adjustment)))

        self._validate_base_thresholds()

        logger.info(
            f"HysteresisManager initialized: Suspicious=({self.base_suspicious_exit:.2f}/{self.base_suspicious_enter:.2f}), "
            f"HighRisk=({self.base_high_risk_exit:.2f}/{self.base_high_risk_enter:.2f}), "
            f"Emergency=({self.base_emergency_exit:.2f}/{self.base_emergency_enter:.2f}), "
            f"dynamic={self.dynamic_thresholds} (max_adj={self.max_adjustment:.2f})."
        )

    def _validate_base_thresholds(self) -> None:
        """Ensures base thresholds maintain the enter > exit invariant."""
        if self.base_suspicious_enter <= self.base_suspicious_exit:
            raise ValueError(
                f"Invalid hysteresis: suspicious_enter ({self.base_suspicious_enter}) "
                f"must be greater than suspicious_exit ({self.base_suspicious_exit})"
            )
        if self.base_high_risk_enter <= self.base_high_risk_exit:
            raise ValueError(
                f"Invalid hysteresis: high_risk_enter ({self.base_high_risk_enter}) "
                f"must be greater than high_risk_exit ({self.base_high_risk_exit})"
            )
        if self.base_emergency_enter <= self.base_emergency_exit:
            raise ValueError(
                f"Invalid hysteresis: emergency_enter ({self.base_emergency_enter}) "
                f"must be greater than emergency_exit ({self.base_emergency_exit})"
            )

    def get_effective_thresholds(
        self,
        density_score: float = 0.0,
        corroborated: bool = False,
        single_modality: bool = False,
    ) -> Dict[str, float]:
        """
        Computes active dynamic thresholds conditioned on environmental context.

        Args:
            density_score: Crowd density level [0.0 - 1.0].
            corroborated: True if cross-modal audio-visual synergy is verified.
            single_modality: True if one sensory stream is offline or missing.

        Returns:
            Dictionary containing active enter and exit thresholds.
        """
        delta = 0.0

        if self.dynamic_thresholds:
            # 1. High Crowd Density Offset: In dense crowds, normal friction elevates baseline kinematics.
            # Slight upward threshold adjustment prevents false alarms from pedestrian jostling.
            if density_score >= 0.70:
                delta += min(self.max_adjustment * 0.5, (density_score - 0.70) * 0.20)

            # 2. Cross-Modal Corroboration: Mutual confirmation between sight and sound reduces uncertainty.
            # Lowers enter thresholds to improve responsiveness to genuine corroborated emergencies.
            if corroborated:
                delta -= self.max_adjustment * 0.5

            # 3. Single-Modality Conservatism: If video or audio is absent, raise threshold slightly.
            if single_modality:
                delta += self.max_adjustment * 0.3

            # Clamp overall adjustment to [-max_adjustment, +max_adjustment]
            delta = max(-self.max_adjustment, min(self.max_adjustment, delta))

        # Apply bounded adjustment to enter thresholds
        s_enter = max(0.15, min(0.95, self.base_suspicious_enter + delta))
        hr_enter = max(s_enter + 0.05, min(0.96, self.base_high_risk_enter + delta))
        em_enter = max(hr_enter + 0.05, min(0.98, self.base_emergency_enter + delta))

        # Exit thresholds track enter thresholds while preserving deadband gap >= 0.05
        s_gap = self.base_suspicious_enter - self.base_suspicious_exit
        hr_gap = self.base_high_risk_enter - self.base_high_risk_exit
        em_gap = self.base_emergency_enter - self.base_emergency_exit

        s_exit = max(0.05, min(s_enter - 0.05, s_enter - s_gap))
        hr_exit = max(s_exit + 0.05, min(hr_enter - 0.05, hr_enter - hr_gap))
        em_exit = max(hr_exit + 0.05, min(em_enter - 0.05, em_enter - em_gap))

        return {
            "suspicious_enter": round(s_enter, 4),
            "suspicious_exit": round(s_exit, 4),
            "high_risk_enter": round(hr_enter, 4),
            "high_risk_exit": round(hr_exit, 4),
            "emergency_enter": round(em_enter, 4),
            "emergency_exit": round(em_exit, 4),
            "adjustment_delta": round(delta, 4),
        }

    def evaluate_candidate_state(
        self,
        current_state: RiskState,
        score: float,
        density_score: float = 0.0,
        corroborated: bool = False,
        single_modality: bool = False,
    ) -> Tuple[RiskState, Dict[str, float]]:
        """
        Determines the target risk state for a given score under active hysteresis boundaries.

        Returns:
            Tuple of (candidate_state, effective_thresholds_dict).
        """
        thresh = self.get_effective_thresholds(
            density_score=density_score,
            corroborated=corroborated,
            single_modality=single_modality,
        )

        s_enter = thresh["suspicious_enter"]
        s_exit = thresh["suspicious_exit"]
        hr_enter = thresh["high_risk_enter"]
        hr_exit = thresh["high_risk_exit"]
        em_enter = thresh["emergency_enter"]
        em_exit = thresh["emergency_exit"]

        score = float(score)

        # -------------------------------------------------------------
        # 1. State: EMERGENCY
        # -------------------------------------------------------------
        if current_state == RiskState.EMERGENCY:
            if score < em_exit:
                # Score dropped below emergency exit -> initiate recovery
                return RiskState.RECOVERY, thresh
            return RiskState.EMERGENCY, thresh

        # -------------------------------------------------------------
        # 2. State: RECOVERY
        # -------------------------------------------------------------
        if current_state == RiskState.RECOVERY:
            # Re-escalation if severe threat suddenly spikes again
            if score >= em_enter:
                return RiskState.EMERGENCY, thresh
            elif score >= hr_enter:
                return RiskState.HIGH_RISK, thresh
            elif score >= s_enter:
                return RiskState.SUSPICIOUS, thresh
            elif score < s_exit:
                return RiskState.NORMAL, thresh
            return RiskState.RECOVERY, thresh

        # -------------------------------------------------------------
        # 3. State: HIGH_RISK
        # -------------------------------------------------------------
        if current_state == RiskState.HIGH_RISK:
            if score >= em_enter:
                return RiskState.EMERGENCY, thresh
            elif score < hr_exit:
                # De-escalating from HIGH_RISK
                if score < s_exit:
                    return RiskState.NORMAL, thresh
                return RiskState.SUSPICIOUS, thresh
            # Deadband [hr_exit, em_enter): retain HIGH_RISK
            return RiskState.HIGH_RISK, thresh

        # -------------------------------------------------------------
        # 4. State: SUSPICIOUS
        # -------------------------------------------------------------
        if current_state == RiskState.SUSPICIOUS:
            if score >= em_enter:
                return RiskState.EMERGENCY, thresh
            elif score >= hr_enter:
                return RiskState.HIGH_RISK, thresh
            elif score < s_exit:
                return RiskState.NORMAL, thresh
            # Deadband [s_exit, hr_enter): retain SUSPICIOUS
            return RiskState.SUSPICIOUS, thresh

        # -------------------------------------------------------------
        # 5. State: NORMAL
        # -------------------------------------------------------------
        # Escalation checks from baseline
        if score >= em_enter:
            return RiskState.EMERGENCY, thresh
        elif score >= hr_enter:
            return RiskState.HIGH_RISK, thresh
        elif score >= s_enter:
            return RiskState.SUSPICIOUS, thresh

        return RiskState.NORMAL, thresh
