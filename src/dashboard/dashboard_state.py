"""
Dashboard State Management - Phase 10.

Maintains in-memory bounded rolling history and operator state for the
surveillance dashboard without database dependencies or memory leaks.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Set

from src.dashboard.dashboard_data import DashboardSnapshot

logger = logging.getLogger("crowd_anomaly.dashboard.state")


@dataclass
class DashboardState:
    """
    Thread-safe, bounded in-memory state for the surveillance dashboard.

    Maintains rolling telemetry histories (default maxlen=120 frames),
    recent alerts buffer (max 50 records), operator acknowledgement/resolution
    sets, and live system status flags.
    """

    max_history: int = 120
    max_recent_alerts: int = 50

    # Rolling metric queues
    history_timestamps: deque = field(default_factory=lambda: deque(maxlen=120))
    history_anomaly_scores: deque = field(default_factory=lambda: deque(maxlen=120))
    history_video_scores: deque = field(default_factory=lambda: deque(maxlen=120))
    history_audio_scores: deque = field(default_factory=lambda: deque(maxlen=120))
    history_crowd_counts: deque = field(default_factory=lambda: deque(maxlen=120))
    history_crowd_densities: deque = field(default_factory=lambda: deque(maxlen=120))
    history_risk_states: deque = field(default_factory=lambda: deque(maxlen=120))
    history_latencies_ms: deque = field(default_factory=lambda: deque(maxlen=120))

    # Alert state
    active_alert: Optional[Dict[str, Any]] = None
    recent_alerts: List[Dict[str, Any]] = field(default_factory=list)
    acknowledged_alert_ids: Set[str] = field(default_factory=set)
    resolved_alert_ids: Set[str] = field(default_factory=set)

    # Operator log
    operator_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Subsystem status
    subsystem_status: Dict[str, bool] = field(default_factory=lambda: {
        "video_capture": True,
        "person_detection": True,
        "crowd_tracking": True,
        "behavior_analysis": True,
        "audio_analysis": True,
        "multimodal_fusion": True,
        "decision_engine": True,
        "alert_manager": True,
        "xai_engine": True,
    })

    # Latest snapshot reference
    last_snapshot: Optional[DashboardSnapshot] = None
    paused: bool = False
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Ensure bounded queues respect max_history if custom value passed."""
        if self.history_timestamps.maxlen != self.max_history:
            self.history_timestamps = deque(maxlen=self.max_history)
            self.history_anomaly_scores = deque(maxlen=self.max_history)
            self.history_video_scores = deque(maxlen=self.max_history)
            self.history_audio_scores = deque(maxlen=self.max_history)
            self.history_crowd_counts = deque(maxlen=self.max_history)
            self.history_crowd_densities = deque(maxlen=self.max_history)
            self.history_risk_states = deque(maxlen=self.max_history)
            self.history_latencies_ms = deque(maxlen=self.max_history)

    def add_snapshot(self, snapshot: DashboardSnapshot) -> None:
        """
        Ingest a new snapshot into rolling history.

        If the dashboard is paused, the snapshot is recorded as `last_snapshot`
        for inspectability, but time-series progression is held.
        """
        if snapshot is None:
            return

        self.last_snapshot = snapshot

        if self.paused:
            return

        # Append to rolling histories
        self.history_timestamps.append(snapshot.timestamp)
        self.history_anomaly_scores.append(snapshot.anomaly_score)
        self.history_video_scores.append(snapshot.video_score)
        self.history_audio_scores.append(snapshot.audio_score)
        self.history_crowd_counts.append(snapshot.crowd_count)
        self.history_crowd_densities.append(snapshot.crowd_density)
        self.history_risk_states.append(snapshot.risk_state)
        self.history_latencies_ms.append(snapshot.pipeline_latency_ms)

        # Update subsystem statuses if present in snapshot
        if snapshot.system_health:
            for k, v in snapshot.system_health.items():
                self.subsystem_status[k] = bool(v)

        # Process active alert
        if snapshot.active_alert:
            alert = dict(snapshot.active_alert)
            alert_id = str(alert.get("alert_id") or f"alert_{int(snapshot.timestamp)}")
            alert["alert_id"] = alert_id

            # Apply operator lifecycle flags if already tracked
            if alert_id in self.acknowledged_alert_ids:
                alert["acknowledged"] = True
                alert["status"] = "ACKNOWLEDGED"
            if alert_id in self.resolved_alert_ids:
                alert["resolved"] = True
                alert["status"] = "RESOLVED"

            # Set active alert if not resolved
            if alert_id not in self.resolved_alert_ids:
                self.active_alert = alert
            elif self.active_alert and self.active_alert.get("alert_id") == alert_id:
                self.active_alert = None

            # Add to recent alerts buffer (deduplicating by alert_id)
            existing_idx = next(
                (i for i, a in enumerate(self.recent_alerts) if a.get("alert_id") == alert_id),
                None,
            )
            if existing_idx is not None:
                self.recent_alerts[existing_idx] = alert
            else:
                self.recent_alerts.insert(0, alert)
                if len(self.recent_alerts) > self.max_recent_alerts:
                    self.recent_alerts.pop()
        else:
            # If no active alert in snapshot and current active alert was resolved, clear it
            if self.active_alert and self.active_alert.get("alert_id") in self.resolved_alert_ids:
                self.active_alert = None

        # Synchronize recent alerts list from snapshot if snapshot provided a full list
        if snapshot.recent_alerts:
            for alt in snapshot.recent_alerts:
                aid = str(alt.get("alert_id") or "")
                if not aid:
                    continue
                if aid in self.acknowledged_alert_ids:
                    alt["acknowledged"] = True
                if aid in self.resolved_alert_ids:
                    alt["resolved"] = True
                    alt["status"] = "RESOLVED"
                if not any(a.get("alert_id") == aid for a in self.recent_alerts):
                    self.recent_alerts.append(alt)
            # Cap recent alerts
            if len(self.recent_alerts) > self.max_recent_alerts:
                self.recent_alerts = self.recent_alerts[: self.max_recent_alerts]

    def acknowledge_alert(self, alert_id: str, operator_id: str = "operator_1") -> bool:
        """
        Acknowledge an alert by ID.

        Returns True if successful, False if alert not found or already acknowledged.
        """
        if not alert_id:
            return False

        self.acknowledged_alert_ids.add(alert_id)

        action_record = {
            "timestamp": time.time(),
            "action": "ACKNOWLEDGE",
            "alert_id": alert_id,
            "operator_id": operator_id,
        }
        self.operator_actions.append(action_record)
        logger.info("Operator %s acknowledged alert %s", operator_id, alert_id)

        # Update in active alert
        if self.active_alert and self.active_alert.get("alert_id") == alert_id:
            self.active_alert["acknowledged"] = True
            self.active_alert["status"] = "ACKNOWLEDGED"

        # Update in recent alerts list
        for alt in self.recent_alerts:
            if alt.get("alert_id") == alert_id:
                alt["acknowledged"] = True
                alt["status"] = "ACKNOWLEDGED"

        return True

    def resolve_alert(
        self,
        alert_id: str,
        operator_id: str = "operator_1",
        reason: str = "Resolved by operator",
    ) -> bool:
        """
        Resolve an alert by ID.

        Returns True if successful, False if alert not found.
        """
        if not alert_id:
            return False

        self.resolved_alert_ids.add(alert_id)

        action_record = {
            "timestamp": time.time(),
            "action": "RESOLVE",
            "alert_id": alert_id,
            "operator_id": operator_id,
            "reason": reason,
        }
        self.operator_actions.append(action_record)
        logger.info("Operator %s resolved alert %s: %s", operator_id, alert_id, reason)

        # Clear from active alert if it was active
        if self.active_alert and self.active_alert.get("alert_id") == alert_id:
            self.active_alert["resolved"] = True
            self.active_alert["status"] = "RESOLVED"
            self.active_alert = None

        # Update in recent alerts list
        for alt in self.recent_alerts:
            if alt.get("alert_id") == alert_id:
                alt["resolved"] = True
                alt["status"] = "RESOLVED"

        return True

    def get_chart_data(self) -> Dict[str, List[Any]]:
        """
        Extract synchronized lists suitable for time-series charts.
        """
        return {
            "timestamps": list(self.history_timestamps),
            "anomaly_scores": list(self.history_anomaly_scores),
            "video_scores": list(self.history_video_scores),
            "audio_scores": list(self.history_audio_scores),
            "crowd_counts": list(self.history_crowd_counts),
            "crowd_densities": list(self.history_crowd_densities),
            "risk_states": list(self.history_risk_states),
            "latencies_ms": list(self.history_latencies_ms),
        }

    def clear_history(self) -> None:
        """Clear all historical rolling telemetry."""
        self.history_timestamps.clear()
        self.history_anomaly_scores.clear()
        self.history_video_scores.clear()
        self.history_audio_scores.clear()
        self.history_crowd_counts.clear()
        self.history_crowd_densities.clear()
        self.history_risk_states.clear()
        self.history_latencies_ms.clear()
        self.recent_alerts.clear()
        self.active_alert = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state summary to dictionary."""
        return {
            "history_length": len(self.history_timestamps),
            "max_history": self.max_history,
            "paused": self.paused,
            "active_alert": self.active_alert,
            "recent_alerts_count": len(self.recent_alerts),
            "acknowledged_count": len(self.acknowledged_alert_ids),
            "resolved_count": len(self.resolved_alert_ids),
            "operator_actions_count": len(self.operator_actions),
            "subsystem_status": dict(self.subsystem_status),
            "has_last_snapshot": self.last_snapshot is not None,
        }
