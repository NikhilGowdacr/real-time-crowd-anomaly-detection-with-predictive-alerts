"""
==============================================================================
Temporal Confirmation and Persistence Tracking (Phase 7).
Ensures anomalous evidence persists across multiple consecutive frames
before escalating or de-escalating surveillance risk states.
==============================================================================
"""

from typing import Optional, Tuple
from src.decision.decision_record import RiskState
from src.utils.logger import setup_logger

logger = setup_logger("decision_confirmation")


class TemporalConfirmationTracker:
    """
    Tracks temporal persistence of candidate risk state transitions.
    Prevents single-frame sensor noise, motion blurs, or acoustic transients
    from triggering premature state escalations.
    """

    def __init__(
        self,
        suspicious_confirmation: int = 3,
        high_risk_confirmation: int = 3,
        emergency_confirmation: int = 2,
        recovery_confirmation: int = 5,
        critical_evidence_enabled: bool = True,
    ):
        self.suspicious_quota = max(1, int(suspicious_confirmation))
        self.high_risk_quota = max(1, int(high_risk_confirmation))
        self.emergency_quota = max(1, int(emergency_confirmation))
        self.recovery_quota = max(1, int(recovery_confirmation))
        self.critical_evidence_enabled = bool(critical_evidence_enabled)

        # Internal state
        self._candidate_state: Optional[RiskState] = None
        self._consecutive_count: int = 0

        logger.info(
            f"TemporalConfirmationTracker initialized: Quotas=(suspicious={self.suspicious_quota}, "
            f"high_risk={self.high_risk_quota}, emergency={self.emergency_quota}, "
            f"recovery={self.recovery_quota}), critical_bypass={self.critical_evidence_enabled}."
        )

    @property
    def candidate_state(self) -> Optional[RiskState]:
        return self._candidate_state

    @property
    def consecutive_count(self) -> int:
        return self._consecutive_count

    def get_quota_for_state(
        self,
        target_state: RiskState,
        critical_evidence: bool = False,
    ) -> int:
        """Determines the required consecutive confirmation frames for a target state."""
        if critical_evidence and self.critical_evidence_enabled:
            # Accelerated confirmation on critical safety evidence
            return 1

        if target_state == RiskState.SUSPICIOUS:
            return self.suspicious_quota
        elif target_state == RiskState.HIGH_RISK:
            return self.high_risk_quota
        elif target_state == RiskState.EMERGENCY:
            return self.emergency_quota
        elif target_state == RiskState.NORMAL:
            # If de-escalating through recovery, quota applies
            return self.recovery_quota
        elif target_state == RiskState.RECOVERY:
            # Entering recovery phase from emergency triggers immediately
            return 1
        return 1

    def update(
        self,
        current_state: RiskState,
        candidate_state: RiskState,
        critical_evidence: bool = False,
    ) -> Tuple[bool, int, int]:
        """
        Updates temporal counters for the candidate state.

        Args:
            current_state: Current active state of the state machine.
            candidate_state: Target state suggested by hysteresis evaluation.
            critical_evidence: Flag indicating severe threat evidence presence.

        Returns:
            Tuple of (is_confirmed: bool, consecutive_count: int, required_quota: int).
        """
        # If candidate matches the current state, no transition is pending
        if candidate_state == current_state:
            self._candidate_state = candidate_state
            self._consecutive_count = 0
            return True, 0, 0

        # If candidate state changed from previous evaluation, reset counter
        if candidate_state != self._candidate_state:
            self._candidate_state = candidate_state
            self._consecutive_count = 1
        else:
            self._consecutive_count += 1

        required_quota = self.get_quota_for_state(
            candidate_state,
            critical_evidence=critical_evidence,
        )

        is_confirmed = self._consecutive_count >= required_quota
        return is_confirmed, self._consecutive_count, required_quota

    def reset(self) -> None:
        """Clears all pending candidate confirmation counters."""
        self._candidate_state = None
        self._consecutive_count = 0
