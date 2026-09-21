"""
==============================================================================
Alert Manager Master Facade (Phase 8).
Coordinates alert policy evaluation, deduplication, cooldown debouncing,
lifecycle registration, and multi-channel notification dispatch.
Phase 7 DecisionEvent remains the single source of truth for risk state.
==============================================================================
"""

import time
from typing import Any, Dict, List, Optional, Union

from src.alerts.acknowledgement import AlertAcknowledgementManager
from src.alerts.alert_policy import AlertPolicy
from src.alerts.alert_record import AlertRecord, AlertSeverity, AlertStatus
from src.alerts.channels import (
    ConsoleChannel,
    EmailChannel,
    HUDChannel,
    LocalAlarmChannel,
    NotificationChannel,
    PushChannel,
    SMSChannel,
)
from src.alerts.cooldown import AlertCooldownManager
from src.alerts.deduplication import AlertDeduplicator
from src.alerts.local_alarm import LocalAlarmManager
from src.decision.decision_record import DecisionEvent, RiskState
from src.utils.logger import setup_logger

logger = setup_logger("alert_manager")


class AlertManager:
    """
    Central alert management subsystem for real-time surveillance.
    Converts confirmed Phase 7 DecisionEvents into controlled, deduplicated,
    and severity-aware alerts across configured channels.
    """

    def __init__(
        self,
        policy: Optional[AlertPolicy] = None,
        cooldown_manager: Optional[AlertCooldownManager] = None,
        deduplicator: Optional[AlertDeduplicator] = None,
        acknowledgement_manager: Optional[AlertAcknowledgementManager] = None,
        alarm_manager: Optional[LocalAlarmManager] = None,
        channels: Optional[List[NotificationChannel]] = None,
    ):
        self.policy = policy or AlertPolicy()
        self.cooldown_manager = cooldown_manager or AlertCooldownManager(
            cooldown_seconds=self.policy.cooldown_seconds,
            escalation_enabled=self.policy.escalation_enabled,
            repeat_emergency_after_seconds=self.policy.repeat_emergency_after_seconds,
        )
        self.deduplicator = deduplicator or AlertDeduplicator(
            duplicate_window_seconds=self.policy.duplicate_window_seconds,
        )
        self.acknowledgement_manager = acknowledgement_manager or AlertAcknowledgementManager()
        self.alarm_manager = alarm_manager or LocalAlarmManager(
            enabled=self.policy.channels.get("local_alarm", False),
        )

        # Initialize distribution channels
        if channels is not None:
            self.channels: List[NotificationChannel] = channels
        else:
            self.channels = [
                ConsoleChannel(enabled=self.policy.channels.get("console", True)),
                HUDChannel(enabled=self.policy.channels.get("hud", True)),
                LocalAlarmChannel(
                    alarm_manager=self.alarm_manager,
                    enabled=self.policy.channels.get("local_alarm", False),
                ),
                EmailChannel(enabled=self.policy.channels.get("email", False)),
                SMSChannel(enabled=self.policy.channels.get("sms", False)),
                PushChannel(enabled=self.policy.channels.get("push", False)),
            ]

        self._alert_counter: int = 0
        logger.info(
            f"AlertManager initialized: policy_enabled={self.policy.enabled}, "
            f"min_severity={self.policy.minimum_severity.value}, "
            f"active_channels={[c.name for c in self.channels if c.enabled]}."
        )

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "AlertManager":
        """Factory instantiating AlertManager directly from configuration dictionary."""
        policy = AlertPolicy.from_config(config)
        cooldown = AlertCooldownManager(
            cooldown_seconds=policy.cooldown_seconds,
            escalation_enabled=policy.escalation_enabled,
            repeat_emergency_after_seconds=policy.repeat_emergency_after_seconds,
        )
        dedup = AlertDeduplicator(duplicate_window_seconds=policy.duplicate_window_seconds)
        ack = AlertAcknowledgementManager()
        alarm = LocalAlarmManager(enabled=policy.channels.get("local_alarm", False))

        channels = [
            ConsoleChannel(enabled=policy.channels.get("console", True)),
            HUDChannel(enabled=policy.channels.get("hud", True)),
            LocalAlarmChannel(alarm_manager=alarm, enabled=policy.channels.get("local_alarm", False)),
            EmailChannel(enabled=policy.channels.get("email", False)),
            SMSChannel(enabled=policy.channels.get("sms", False)),
            PushChannel(enabled=policy.channels.get("push", False)),
        ]

        return cls(
            policy=policy,
            cooldown_manager=cooldown,
            deduplicator=dedup,
            acknowledgement_manager=ack,
            alarm_manager=alarm,
            channels=channels,
        )

    def process_decision(
        self,
        event: Optional[Union[DecisionEvent, Dict[str, Any]]],
        current_time: Optional[float] = None,
    ) -> Optional[AlertRecord]:
        """
        Main ingress pipeline for Phase 7 DecisionEvents.

        Pipeline:
            1. Validate event structure
            2. Verify confirmation and transition requirements
            3. Map RiskState -> AlertSeverity
            4. Apply AlertPolicy filtering
            5. Evaluate duplicate suppression
            6. Evaluate cooldown and escalation bypass
            7. Formulate AlertRecord
            8. Dispatch across enabled notification channels
            9. Store in active acknowledgement registry
            10. Return emitted AlertRecord (or None if suppressed)
        """
        curr_t = time.time() if current_time is None else float(current_time)

        # 1. Validate event safely
        if event is None or not self.policy.enabled:
            return None

        # Unpack event attributes safely
        if isinstance(event, DecisionEvent):
            current_state = str(event.current_state)
            previous_state = str(event.previous_state)
            score = float(event.score)
            confidence = float(event.confidence)
            reason = str(event.reason)
            dominant_modality = str(event.dominant_modality)
            video_score = float(event.video_score)
            audio_score = float(event.audio_score)
            weapon_score = float(event.weapon_score)
            fight_score = float(event.fight_score)
            corroborated = bool(event.corroborated)
            critical_evidence = bool(event.critical_evidence)
            confirmed = bool(event.confirmed)
            transition = bool(event.transition)
            source_dict = event.to_dict()
        elif isinstance(event, dict):
            current_state = str(event.get("current_state", "NORMAL"))
            previous_state = str(event.get("previous_state", "NORMAL"))
            score = float(event.get("score", 0.0))
            confidence = float(event.get("confidence", 0.80))
            reason = str(event.get("reason", ""))
            dominant_modality = str(event.get("dominant_modality", "none"))
            video_score = float(event.get("video_score", 0.0))
            audio_score = float(event.get("audio_score", 0.0))
            weapon_score = float(event.get("weapon_score", 0.0))
            fight_score = float(event.get("fight_score", 0.0))
            corroborated = bool(event.get("corroborated", False))
            critical_evidence = bool(event.get("critical_evidence", False))
            confirmed = bool(event.get("confirmed", True))
            transition = bool(event.get("transition", False))
            source_dict = dict(event)
        else:
            logger.warning(f"Rejecting invalid event object of type {type(event).__name__}.")
            return None

        # 2. Check confirmation & transition requirements
        # Do not generate alerts for unconfirmed frames
        if not confirmed:
            return None

        # NORMAL conditions never generate alerts
        if current_state == RiskState.NORMAL.value:
            return None

        # RECOVERY produces alerts ONLY if recovery_alert policy is explicitly active and state changed
        if current_state == RiskState.RECOVERY.value:
            if not self.policy.recovery_alert or not transition:
                return None

        # For active threats (SUSPICIOUS, HIGH_RISK, EMERGENCY):
        # Must be either a verified state transition OR a permitted emergency repeat
        is_emergency_repeat = bool(
            current_state == RiskState.EMERGENCY.value
            and not transition
            and self.policy.repeat_emergency_after_seconds > 0
            and not self.cooldown_manager.is_in_cooldown(AlertSeverity.CRITICAL, current_state, curr_t)
        )

        if not transition and not is_emergency_repeat:
            return None

        # 3. Map RiskState -> AlertSeverity
        severity = self.policy.map_state_to_severity(current_state)
        if severity is None:
            return None

        # 4. Policy minimum severity filtering
        if not self.policy.should_emit(severity):
            logger.debug(
                f"Alert suppressed by policy: Severity '{severity.value}' is below "
                f"minimum '{self.policy.minimum_severity.value}'."
            )
            return None

        # 5. Duplicate suppression check
        signature = self.deduplicator.generate_signature(event, severity)
        if self.deduplicator.is_duplicate(signature, curr_t):
            logger.info(f"[INFO] Duplicate alert suppressed (signature: {signature}).")
            return None

        # 6. Cooldown & Escalation check
        if self.cooldown_manager.is_in_cooldown(severity, current_state, curr_t):
            logger.debug(f"Alert suppressed: Cooldown active for state '{current_state}'.")
            return None

        # 7. Formulate AlertRecord
        self._alert_counter += 1
        time_tag = time.strftime("%Y%m%d-%H%M%S", time.localtime(curr_t))
        alert_id = f"ALT-{time_tag}-{self._alert_counter:04d}"

        alert = AlertRecord(
            alert_id=alert_id,
            timestamp=curr_t,
            severity=severity,
            risk_state=current_state,
            score=score,
            confidence=confidence,
            reason=reason,
            dominant_modality=dominant_modality,
            video_score=video_score,
            audio_score=audio_score,
            weapon_score=weapon_score,
            fight_score=fight_score,
            corroborated=corroborated,
            critical_evidence=critical_evidence,
            source_event=source_dict,
            status=AlertStatus.ACTIVE,
        )

        # 8. Dispatch across enabled notification channels
        channel_status: Dict[str, str] = {}
        for channel in self.channels:
            if channel.enabled:
                try:
                    status_code = channel.send(alert)
                    channel_status[channel.name] = status_code
                except Exception as e:
                    logger.error(f"Failed to dispatch alert via channel '{channel.name}': {e}")
                    channel_status[channel.name] = f"ERROR: {e}"
            else:
                channel_status[channel.name] = "DISABLED"

        alert.channel_status = channel_status

        # 9. Record state and register alert in lifecycle manager
        self.deduplicator.record_signature(signature, curr_t)
        self.cooldown_manager.record_alert(severity, current_state, curr_t)
        self.acknowledgement_manager.register(alert)

        return alert

    # Lifecycle proxy methods
    def acknowledge(self, alert_id: str, timestamp: Optional[float] = None) -> bool:
        """Acknowledges an active alert."""
        return self.acknowledgement_manager.acknowledge(alert_id, timestamp=timestamp)

    def resolve(self, alert_id: str, timestamp: Optional[float] = None) -> bool:
        """Resolves an alert."""
        return self.acknowledgement_manager.resolve(alert_id, timestamp=timestamp)

    def reset_alert(self, alert_id: str) -> bool:
        """Resets an alert back to active status."""
        return self.acknowledgement_manager.reset(alert_id)

    def get_active_alerts(self) -> List[AlertRecord]:
        """Returns all currently active (unresolved) alerts."""
        return self.acknowledgement_manager.get_active_alerts()

    def get_history(self, limit: int = 100) -> List[AlertRecord]:
        """Returns recent alert history in reverse chronological order."""
        return self.acknowledgement_manager.get_history(limit=limit)

    def get_current_hud_alert(self) -> Optional[AlertRecord]:
        """Retrieves the active alert currently buffered by HUDChannel (if any)."""
        for channel in self.channels:
            if isinstance(channel, HUDChannel) and channel.enabled:
                return channel.current_alert
        return None
