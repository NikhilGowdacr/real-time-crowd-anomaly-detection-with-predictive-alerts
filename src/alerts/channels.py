"""
==============================================================================
Notification Channel Abstraction and Channel Implementations (Phase 8).
Dispatches alerts to active channels: Console, HUD, Local Alarm,
and academic verification stubs for Email, SMS, and Push notifications.
==============================================================================
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.alerts.alert_record import AlertRecord, AlertSeverity
from src.alerts.local_alarm import LocalAlarmManager
from src.utils.logger import setup_logger

logger = setup_logger("alert_channels")


class NotificationChannel(ABC):
    """Abstract base class for all alert notification distribution channels."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = str(name)
        self.enabled = bool(enabled)

    @abstractmethod
    def send(self, alert: AlertRecord) -> str:
        """
        Dispatches the alert record through the channel.

        Returns:
            Delivery status string (e.g. 'DELIVERED', 'MUTED', 'UNCONNECTED_STUB').
        """
        pass


class ConsoleChannel(NotificationChannel):
    """Outputs structured, formatted surveillance alert banners to the terminal/logger."""

    def __init__(self, enabled: bool = True):
        super().__init__(name="console", enabled=enabled)

    def send(self, alert: AlertRecord) -> str:
        if not self.enabled:
            return "DISABLED"

        # Log with appropriate severity-specific logger level
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(f"\n{alert.formatted_message}")
        elif alert.severity == AlertSeverity.HIGH:
            logger.warning(f"\n{alert.formatted_message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.info(f"\n{alert.formatted_message}")
        else:
            logger.info(f"\n{alert.formatted_message}")

        return "DELIVERED"


class HUDChannel(NotificationChannel):
    """
    Maintains the most recent active alert record in an accessible memory buffer
    for real-time OpenCV HUD surveillance overlay rendering.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(name="hud", enabled=enabled)
        self._current_alert: Optional[AlertRecord] = None

    @property
    def current_alert(self) -> Optional[AlertRecord]:
        return self._current_alert

    def send(self, alert: AlertRecord) -> str:
        if not self.enabled:
            return "DISABLED"

        self._current_alert = alert
        return "DELIVERED"

    def clear(self) -> None:
        self._current_alert = None


class LocalAlarmChannel(NotificationChannel):
    """Dispatches audible alarm tones via LocalAlarmManager."""

    def __init__(self, alarm_manager: Optional[LocalAlarmManager] = None, enabled: bool = False):
        super().__init__(name="local_alarm", enabled=enabled)
        self.alarm_manager = alarm_manager or LocalAlarmManager(enabled=enabled)

    def send(self, alert: AlertRecord) -> str:
        if not self.enabled or not self.alarm_manager.enabled:
            return "MUTED"
        return self.alarm_manager.trigger(alert.severity)


class EmailChannel(NotificationChannel):
    """
    Academic verification stub for future SMTP / SendGrid / AWS SES email integration.
    Transparently reports unconnected status without claiming false delivery.
    """

    def __init__(self, enabled: bool = False):
        super().__init__(name="email", enabled=enabled)

    def send(self, alert: AlertRecord) -> str:
        if not self.enabled:
            return "DISABLED"
        logger.info(
            f"[INFO] Email notification channel active for Alert [{alert.alert_id}] ({alert.severity.value}), "
            "but external SMTP/API gateway is not connected (Academic Stub)."
        )
        return "UNCONNECTED_STUB"


class SMSChannel(NotificationChannel):
    """
    Academic verification stub for future Twilio / AWS SNS text messaging integration.
    Transparently reports unconnected status without claiming false delivery.
    """

    def __init__(self, enabled: bool = False):
        super().__init__(name="sms", enabled=enabled)

    def send(self, alert: AlertRecord) -> str:
        if not self.enabled:
            return "DISABLED"
        logger.info(
            f"[INFO] SMS notification channel active for Alert [{alert.alert_id}] ({alert.severity.value}), "
            "but external SMS gateway provider is not connected (Academic Stub)."
        )
        return "UNCONNECTED_STUB"


class PushChannel(NotificationChannel):
    """
    Academic verification stub for future Firebase FCM / Apple APNs push notifications.
    Transparently reports unconnected status without claiming false delivery.
    """

    def __init__(self, enabled: bool = False):
        super().__init__(name="push", enabled=enabled)

    def send(self, alert: AlertRecord) -> str:
        if not self.enabled:
            return "DISABLED"
        logger.info(
            f"[INFO] Push notification channel active for Alert [{alert.alert_id}] ({alert.severity.value}), "
            "but cloud push service is not connected (Academic Stub)."
        )
        return "UNCONNECTED_STUB"
