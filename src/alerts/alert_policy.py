"""
==============================================================================
Alert Policy Configuration and Rule Evaluation (Phase 8).
Manages alert filtering, minimum severity thresholds, channel enablement,
and RiskState to AlertSeverity mapping.
==============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from src.alerts.alert_record import AlertSeverity
from src.decision.decision_record import RiskState
from src.utils.logger import setup_logger

logger = setup_logger("alert_policy")


@dataclass
class AlertPolicy:
    """
    Configurable rules and constraints governing surveillance alert generation.
    """
    enabled: bool = True
    minimum_severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: float = 10.0
    duplicate_window_seconds: float = 15.0
    recovery_alert: bool = False
    channels: Dict[str, bool] = field(default_factory=lambda: {
        "console": True,
        "hud": True,
        "local_alarm": False,
        "email": False,
        "sms": False,
        "push": False,
    })
    escalation_enabled: bool = True
    repeat_emergency_after_seconds: float = 30.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "AlertPolicy":
        """Factory initializing AlertPolicy from configuration dictionary."""
        if not cfg:
            return cls()

        alerts_cfg = cfg.get("alerts", {}) if "alerts" in cfg else cfg

        # Parse minimum severity
        min_sev_raw = alerts_cfg.get("minimum_severity", "WARNING")
        if isinstance(min_sev_raw, AlertSeverity):
            min_sev = min_sev_raw
        else:
            try:
                min_sev = AlertSeverity(str(min_sev_raw).upper())
            except ValueError:
                min_sev = AlertSeverity.WARNING

        # Parse channels
        chan_cfg = alerts_cfg.get("channels", {})
        channels = {
            "console": bool(chan_cfg.get("console", alerts_cfg.get("enable_console", True))),
            "hud": bool(chan_cfg.get("hud", alerts_cfg.get("enable_visual", True))),
            "local_alarm": bool(chan_cfg.get("local_alarm", alerts_cfg.get("enable_sound", False))),
            "email": bool(chan_cfg.get("email", alerts_cfg.get("enable_email", False))),
            "sms": bool(chan_cfg.get("sms", alerts_cfg.get("enable_sms", False))),
            "push": bool(chan_cfg.get("push", False)),
        }

        # Parse escalation block
        esc_cfg = alerts_cfg.get("escalation", {})
        escalation_enabled = bool(esc_cfg.get("enabled", True))
        repeat_emergency = float(esc_cfg.get("repeat_emergency_after_seconds", 30.0))

        return cls(
            enabled=bool(alerts_cfg.get("enabled", True)),
            minimum_severity=min_sev,
            cooldown_seconds=float(alerts_cfg.get("cooldown_seconds", 10.0)),
            duplicate_window_seconds=float(alerts_cfg.get("duplicate_window_seconds", 15.0)),
            recovery_alert=bool(alerts_cfg.get("recovery_alert", False)),
            channels=channels,
            escalation_enabled=escalation_enabled,
            repeat_emergency_after_seconds=repeat_emergency,
        )

    def map_state_to_severity(self, state: Union[RiskState, str]) -> Optional[AlertSeverity]:
        """
        Maps a Phase 7 RiskState to an operational AlertSeverity.

        Mapping Specification:
            NORMAL     -> None (no alert generated)
            SUSPICIOUS -> AlertSeverity.WARNING
            HIGH_RISK  -> AlertSeverity.HIGH
            EMERGENCY  -> AlertSeverity.CRITICAL
            RECOVERY   -> AlertSeverity.INFO (only if recovery_alert is True, else None)
        """
        state_str = state.value if isinstance(state, RiskState) else str(state).upper()

        if state_str == RiskState.NORMAL.value:
            return None
        elif state_str == RiskState.SUSPICIOUS.value:
            return AlertSeverity.WARNING
        elif state_str == RiskState.HIGH_RISK.value:
            return AlertSeverity.HIGH
        elif state_str == RiskState.EMERGENCY.value:
            return AlertSeverity.CRITICAL
        elif state_str == RiskState.RECOVERY.value:
            return AlertSeverity.INFO if self.recovery_alert else None
        else:
            logger.warning(f"Unrecognized RiskState '{state_str}'; mapping to None.")
            return None

    def should_emit(self, severity: Optional[AlertSeverity]) -> bool:
        """
        Determines if an alert with the given severity is permitted to emit
        under active policy constraints.
        """
        if not self.enabled or severity is None:
            return False
        return severity >= self.minimum_severity
