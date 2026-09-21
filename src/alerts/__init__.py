"""
==============================================================================
Alert Management and Predictive Notification Subsystem (Phase 8).
Ingests Phase 7 DecisionEvents and generates controlled, deduplicated,
and lifecycle-managed surveillance alerts.
==============================================================================
"""

from src.alerts.acknowledgement import AlertAcknowledgementManager
from src.alerts.alert_manager import AlertManager
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

__all__ = [
    "AlertSeverity",
    "AlertStatus",
    "AlertRecord",
    "AlertPolicy",
    "AlertCooldownManager",
    "AlertDeduplicator",
    "AlertAcknowledgementManager",
    "LocalAlarmManager",
    "NotificationChannel",
    "ConsoleChannel",
    "HUDChannel",
    "LocalAlarmChannel",
    "EmailChannel",
    "SMSChannel",
    "PushChannel",
    "AlertManager",
]
