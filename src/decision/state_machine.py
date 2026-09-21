"""
==============================================================================
Decision State Machine (Phase 7).
Implements the formal surveillance risk state machine, enforcing legal transitions,
gradual de-escalation via RECOVERY, and transition cooldown debouncing.
==============================================================================
"""

import time
from typing import Optional, Set, Tuple
from src.decision.decision_record import RiskState
from src.utils.logger import setup_logger

logger = setup_logger("decision_state_machine")


class DecisionStateMachine:
    """
    State machine governing transitions between surveillance risk states:
      NORMAL -> SUSPICIOUS -> HIGH_RISK -> EMERGENCY -> RECOVERY -> NORMAL

    Features:
      - Validates and enforces directed transition graph.
      - Supports direct high-threat escalation on critical safety evidence.
      - Enforces controlled post-emergency stabilization via RECOVERY.
      - Provides transition cooldown / debouncing to prevent event spamming.
    """

    def __init__(
        self,
        initial_state: RiskState = RiskState.NORMAL,
        transition_cooldown_seconds: float = 3.0,
    ):
        self._current_state = initial_state
        self._previous_state = initial_state
        self.cooldown_seconds = max(0.0, float(transition_cooldown_seconds))
        self._last_transition_time: float = 0.0

        # Legal transition graph
        self._allowed_transitions: Set[Tuple[RiskState, RiskState]] = {
            # Normal progression
            (RiskState.NORMAL, RiskState.NORMAL),
            (RiskState.NORMAL, RiskState.SUSPICIOUS),
            (RiskState.NORMAL, RiskState.HIGH_RISK),
            (RiskState.NORMAL, RiskState.EMERGENCY),

            # Suspicious progression
            (RiskState.SUSPICIOUS, RiskState.SUSPICIOUS),
            (RiskState.SUSPICIOUS, RiskState.NORMAL),
            (RiskState.SUSPICIOUS, RiskState.HIGH_RISK),
            (RiskState.SUSPICIOUS, RiskState.EMERGENCY),

            # High Risk progression
            (RiskState.HIGH_RISK, RiskState.HIGH_RISK),
            (RiskState.HIGH_RISK, RiskState.SUSPICIOUS),
            (RiskState.HIGH_RISK, RiskState.NORMAL),
            (RiskState.HIGH_RISK, RiskState.EMERGENCY),

            # Emergency progression (de-escalation MUST pass through RECOVERY)
            (RiskState.EMERGENCY, RiskState.EMERGENCY),
            (RiskState.EMERGENCY, RiskState.RECOVERY),

            # Recovery progression
            (RiskState.RECOVERY, RiskState.RECOVERY),
            (RiskState.RECOVERY, RiskState.EMERGENCY),
            (RiskState.RECOVERY, RiskState.HIGH_RISK),
            (RiskState.RECOVERY, RiskState.SUSPICIOUS),
            (RiskState.RECOVERY, RiskState.NORMAL),
        }

        logger.info(
            f"DecisionStateMachine initialized: State={self._current_state.value}, "
            f"cooldown={self.cooldown_seconds}s."
        )

    @property
    def current_state(self) -> RiskState:
        return self._current_state

    @property
    def previous_state(self) -> RiskState:
        return self._previous_state

    @property
    def last_transition_time(self) -> float:
        return self._last_transition_time

    def is_transition_allowed(self, target_state: RiskState) -> bool:
        """Verifies if the transition from current_state to target_state is legal."""
        return (self._current_state, target_state) in self._allowed_transitions

    def can_emit_transition(
        self,
        target_state: RiskState,
        current_time: Optional[float] = None,
        critical_evidence: bool = False,
    ) -> bool:
        """
        Determines if a transition event may be emitted without violating cooldown.
        Emergencies and critical evidence always bypass cooldown.
        """
        curr_t = time.time() if current_time is None else float(current_time)
        if target_state == RiskState.EMERGENCY or critical_evidence:
            return True

        time_since_last = curr_t - self._last_transition_time
        return time_since_last >= self.cooldown_seconds

    def transition_to(
        self,
        target_state: RiskState,
        current_time: Optional[float] = None,
        critical_evidence: bool = False,
        force: bool = False,
    ) -> Tuple[bool, bool]:
        """
        Attempts to execute a state transition.

        Args:
            target_state: Proposed destination state.
            current_time: Reference epoch timestamp.
            critical_evidence: Severe threat flag.
            force: Bypass transition legality check if True.

        Returns:
            Tuple of (state_changed: bool, transition_emitted: bool).
        """
        curr_t = time.time() if current_time is None else float(current_time)

        if target_state == self._current_state:
            return False, False

        # Validate transition legality
        if not force and not self.is_transition_allowed(target_state):
            logger.warning(
                f"Blocked illegal state transition attempt: {self._current_state.value} -> {target_state.value}."
            )
            return False, False

        # Check debounce / cooldown
        allowed_emission = self.can_emit_transition(
            target_state=target_state,
            current_time=curr_t,
            critical_evidence=critical_evidence,
        )

        old_state = self._current_state
        self._previous_state = old_state
        self._current_state = target_state
        self._last_transition_time = curr_t

        return True, allowed_emission

    def reset(self, state: RiskState = RiskState.NORMAL) -> None:
        """Resets state machine back to specified baseline state."""
        self._current_state = state
        self._previous_state = state
        self._last_transition_time = 0.0
