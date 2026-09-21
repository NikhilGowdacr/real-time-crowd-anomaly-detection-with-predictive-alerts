"""
==============================================================================
Alert Cooldown and Debounce Management (Phase 8).
Prevents alert flood conditions while allowing legitimate severity escalations
and periodic emergency repeat reminders.
==============================================================================
"""

import time
from typing import Dict, Optional

from src.alerts.alert_record import AlertSeverity
from src.utils.logger import setup_logger

logger = setup_logger("alert_cooldown")


class AlertCooldownManager:
    """
    Manages temporal cooldowns and debouncing for surveillance alerts.
    Enforces that repeated events do not spam operators, while ensuring
    genuine severity escalations bypass cooldown immediately.
    """

    def __init__(
        self,
        cooldown_seconds: float = 10.0,
        escalation_enabled: bool = True,
        repeat_emergency_after_seconds: float = 30.0,
    ):
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.escalation_enabled = bool(escalation_enabled)
        self.repeat_emergency_after_seconds = max(0.0, float(repeat_emergency_after_seconds))

        # Temporal tracking
        self._last_alert_time_by_state: Dict[str, float] = {}
        self._last_alert_time_by_severity: Dict[AlertSeverity, float] = {}
        self._last_emergency_time: float = 0.0
        self._highest_active_severity: Optional[AlertSeverity] = None

        logger.info(
            f"AlertCooldownManager initialized: cooldown={self.cooldown_seconds}s, "
            f"escalation={self.escalation_enabled}, repeat_emergency={self.repeat_emergency_after_seconds}s."
        )

    def is_in_cooldown(
        self,
        severity: AlertSeverity,
        risk_state: str,
        current_time: Optional[float] = None,
    ) -> bool:
        """
        Checks whether a candidate alert is currently suppressed by cooldown.

        Args:
            severity: Mapped AlertSeverity.
            risk_state: Underlying Phase 7 RiskState string.
            current_time: Epoch timestamp.

        Returns:
            True if alert is suppressed by cooldown, False if permitted.
        """
        curr_t = time.time() if current_time is None else float(current_time)

        # 1. Emergency Repeat Allowance
        # If an emergency persists beyond repeat_emergency_after_seconds, permit re-alerting
        if severity == AlertSeverity.CRITICAL and self.repeat_emergency_after_seconds > 0:
            if self._last_emergency_time > 0 and (curr_t - self._last_emergency_time) >= self.repeat_emergency_after_seconds:
                logger.debug(
                    f"Emergency repeat window ({self.repeat_emergency_after_seconds}s) elapsed; "
                    f"permitting critical re-alert at t={curr_t:.2f}."
                )
                return False

        # 2. Severity Escalation Bypass
        # If the incoming alert is of higher severity than what was previously active,
        # immediately bypass cooldown to ensure urgent threats are alerted without delay.
        if self.escalation_enabled and self._highest_active_severity is not None:
            if severity > self._highest_active_severity:
                logger.info(
                    f"Severity escalation detected: {self._highest_active_severity.value} -> {severity.value}. "
                    "Bypassing cooldown window."
                )
                return False

        # 3. State-based Cooldown
        last_state_t = self._last_alert_time_by_state.get(risk_state, 0.0)
        if (curr_t - last_state_t) < self.cooldown_seconds:
            logger.debug(
                f"Alert suppressed: Risk state '{risk_state}' is in cooldown "
                f"({curr_t - last_state_t:.2f}s < {self.cooldown_seconds}s)."
            )
            return True

        # 4. Severity-based Cooldown
        last_sev_t = self._last_alert_time_by_severity.get(severity, 0.0)
        if (curr_t - last_sev_t) < self.cooldown_seconds:
            logger.debug(
                f"Alert suppressed: Severity '{severity.value}' is in cooldown "
                f"({curr_t - last_sev_t:.2f}s < {self.cooldown_seconds}s)."
            )
            return True

        return False

    def record_alert(
        self,
        severity: AlertSeverity,
        risk_state: str,
        current_time: Optional[float] = None,
    ) -> None:
        """
        Records an emitted alert, resetting cooldown timers for the corresponding
        state and severity tier.
        """
        curr_t = time.time() if current_time is None else float(current_time)

        self._last_alert_time_by_state[risk_state] = curr_t
        self._last_alert_time_by_severity[severity] = curr_t

        if severity == AlertSeverity.CRITICAL:
            self._last_emergency_time = curr_t

        # Track or update highest severity
        if self._highest_active_severity is None or severity >= self._highest_active_severity:
            self._highest_active_severity = severity

    def reset(self) -> None:
        """Clears all cooldown tracking and active severity levels."""
        self._last_alert_time_by_state.clear()
        self._last_alert_time_by_severity.clear()
        self._last_emergency_time = 0.0
        self._highest_active_severity = None
