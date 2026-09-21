"""
==============================================================================
Alert Lifecycle and Operator Acknowledgement Management (Phase 8).
Maintains in-memory registry of active and historical alerts, handling
operator acknowledgement, resolution, and status querying.
==============================================================================
"""

import time
from typing import Dict, List, Optional

from src.alerts.alert_record import AlertRecord, AlertStatus
from src.utils.logger import setup_logger

logger = setup_logger("alert_acknowledgement")


class AlertAcknowledgementManager:
    """
    Manages operator interactions with alerts without mutating underlying
    Phase 7 risk states. Tracks alert lifecycle transitions:
        CREATED -> ACTIVE -> ACKNOWLEDGED -> RESOLVED
    """

    def __init__(self, max_history: int = 500):
        self.max_history = max(10, int(max_history))
        # Ordered storage of alerts by alert_id
        self._alerts: Dict[str, AlertRecord] = {}
        self._history: List[AlertRecord] = []

        logger.info(f"AlertAcknowledgementManager initialized: max_history={self.max_history}.")

    def register(self, alert: AlertRecord) -> None:
        """Registers a new alert into the active registry and history buffer."""
        self._alerts[alert.alert_id] = alert
        self._history.append(alert)

        # Prune oldest history if exceeding capacity
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def acknowledge(
        self,
        alert_id: str,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Marks an active alert as acknowledged by a surveillance operator.

        Args:
            alert_id: Unique alert identifier.
            timestamp: Optional epoch timestamp of operator acknowledgement.

        Returns:
            True if alert was found and updated, False otherwise.
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            logger.warning(f"Acknowledgement failed: Alert ID '{alert_id}' not found.")
            return False

        curr_t = time.time() if timestamp is None else float(timestamp)
        alert.acknowledged = True
        alert.acknowledged_at = curr_t
        alert.status = AlertStatus.ACKNOWLEDGED
        logger.info(f"[INFO] Alert [{alert_id}] ({alert.severity.value}) acknowledged by operator at t={curr_t:.2f}.")
        return True

    def resolve(
        self,
        alert_id: str,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Marks an alert as resolved, removing it from active operator attention.

        Args:
            alert_id: Unique alert identifier.
            timestamp: Optional epoch timestamp of resolution.

        Returns:
            True if alert was found and updated, False otherwise.
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            logger.warning(f"Resolution failed: Alert ID '{alert_id}' not found.")
            return False

        curr_t = time.time() if timestamp is None else float(timestamp)
        alert.resolved = True
        alert.resolved_at = curr_t
        alert.status = AlertStatus.RESOLVED
        logger.info(f"[INFO] Alert [{alert_id}] ({alert.severity.value}) marked resolved at t={curr_t:.2f}.")
        return True

    def reset(self, alert_id: str) -> bool:
        """
        Resets an alert back to ACTIVE status (clears acknowledged and resolved flags).

        Args:
            alert_id: Unique alert identifier.

        Returns:
            True if alert was found and reset, False otherwise.
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            logger.warning(f"Reset failed: Alert ID '{alert_id}' not found.")
            return False

        alert.acknowledged = False
        alert.acknowledged_at = None
        alert.resolved = False
        alert.resolved_at = None
        alert.status = AlertStatus.ACTIVE
        logger.info(f"[INFO] Alert [{alert_id}] reset back to ACTIVE status.")
        return True

    def get_alert(self, alert_id: str) -> Optional[AlertRecord]:
        """Retrieves an alert by its unique identifier."""
        return self._alerts.get(alert_id)

    def get_active_alerts(self) -> List[AlertRecord]:
        """
        Returns all alerts that have not yet been marked RESOLVED.
        Includes both unacknowledged ACTIVE and ACKNOWLEDGED alerts.
        """
        return [alert for alert in self._alerts.values() if not alert.resolved]

    def get_history(self, limit: int = 100) -> List[AlertRecord]:
        """Returns the most recent alert records in reverse chronological order."""
        limit = max(1, int(limit))
        return list(reversed(self._history[-limit:]))

    def clear(self) -> None:
        """Clears all active and historical alerts."""
        self._alerts.clear()
        self._history.clear()
