"""
Unit tests for Phase 10: Real-Time Monitoring Dashboard & Operator Interface.

Covers:
- DashboardSnapshot creation and JSON/dict serialization
- DashboardDataProvider subsystem telemetry extraction and demo scenarios
- Robust fallback when subsystems (video, audio, XAI) are offline/None
- DashboardState bounded memory management (deque maxlen=120)
- Alert operator lifecycle (acknowledgement and resolution)
- Telemetry metrics calculations (FPS, latency stats, score summaries)
- Visual theme color tokens and badge generation
- Non-mutation invariant (dashboard never mutates pipeline scores or states)
"""

import json
import time
import numpy as np
import pytest

from src.dashboard.dashboard_data import DashboardDataProvider, DashboardSnapshot
from src.dashboard.dashboard_metrics import DashboardMetrics
from src.dashboard.dashboard_state import DashboardState
from src.dashboard.dashboard_theme import DashboardTheme
from src.decision.decision_engine import DecisionEvent, RiskState
from src.alerts.alert_manager import AlertRecord, AlertSeverity, AlertStatus
from src.xai.explanation_engine import EvidenceRecord


# 1. Snapshot Creation Default
def test_snapshot_creation_default():
    snap = DashboardSnapshot(timestamp=time.time())
    assert snap.timestamp > 0
    assert snap.risk_state == "NORMAL"
    assert snap.anomaly_score == 0.0
    assert snap.confidence == 1.0
    assert snap.crowd_count == 0
    assert snap.active_alert is None
    assert isinstance(snap.weights, dict)
    assert isinstance(snap.system_health, dict)


# 2. Snapshot Serialization
def test_snapshot_serialization():
    snap = DashboardSnapshot(
        timestamp=1700000000.0,
        risk_state="HIGH_RISK",
        anomaly_score=0.72,
        confidence=0.88,
        crowd_count=15,
        crowd_density=0.45,
        video_score=0.78,
        audio_score=0.65,
        dominant_modality="video",
        active_alert={"alert_id": "ALT-101", "severity": "HIGH"},
    )
    d = snap.to_dict()
    assert d["timestamp"] == 1700000000.0
    assert d["risk_state"] == "HIGH_RISK"
    assert d["anomaly_score"] == 0.72
    assert d["active_alert"]["alert_id"] == "ALT-101"

    json_str = snap.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["risk_state"] == "HIGH_RISK"
    assert parsed["crowd_count"] == 15


# 3. Provider Initialization
def test_provider_initialization():
    provider = DashboardDataProvider()
    assert provider is not None
    assert provider.last_snapshot is None


# 4. Provider from Subsystems (Normal)
def test_provider_from_subsystems_normal():
    provider = DashboardDataProvider()
    evt = DecisionEvent(
        timestamp=time.time(),
        previous_state="NORMAL",
        current_state="NORMAL",
        score=0.12,
        confidence=0.95,
        reason="Normal crowd flow",
        dominant_modality="video",
        video_score=0.10,
        audio_score=0.05,
    )
    snap = provider.create_snapshot_from_subsystems(decision_event=evt)
    assert snap.risk_state == "NORMAL"
    assert snap.anomaly_score == 0.12
    assert snap.dominant_modality == "video"
    assert snap.active_alert is None


# 5. Provider from Subsystems (Emergency)
def test_provider_from_subsystems_emergency():
    provider = DashboardDataProvider()
    evt = DecisionEvent(
        timestamp=time.time(),
        previous_state="HIGH_RISK",
        current_state="EMERGENCY",
        score=0.92,
        confidence=0.98,
        reason="Critical weapon detected with crowd panic",
        dominant_modality="video",
        video_score=0.95,
        audio_score=0.88,
        corroborated=True,
        critical_evidence=True,
    )
    alert = AlertRecord(
        alert_id="ALT-EMERGENCY-01",
        timestamp=time.time(),
        risk_state="EMERGENCY",
        severity=AlertSeverity.CRITICAL,
        score=0.92,
        confidence=0.98,
        dominant_modality="video",
        reason="Critical weapon detected",
        critical_evidence=True,
    )
    xai = EvidenceRecord(
        timestamp=time.time(),
        risk_state="EMERGENCY",
        anomaly_score=0.92,
        decision_confidence=0.98,
        dominant_modality="video",
        short_explanation="High confidence weapon observed with crowd dispersion",
        alert_rationale="Dispatch tactical unit and sound building alarm",
    )
    snap = provider.create_snapshot_from_subsystems(
        decision_event=evt,
        alert_record=alert,
        evidence_record=xai,
    )
    assert snap.risk_state == "EMERGENCY"
    assert snap.anomaly_score == 0.92
    assert snap.corroborated is True
    assert snap.critical_evidence is True
    assert snap.active_alert is not None
    assert snap.active_alert["alert_id"] == "ALT-EMERGENCY-01"
    assert "weapon" in snap.xai_explanation.lower()
    assert "tactical" in snap.operator_action.lower()


# 6. Demo Snapshot Normal
def test_provider_demo_snapshot_normal():
    provider = DashboardDataProvider()
    snap = provider.create_demo_snapshot("NORMAL")
    assert snap.risk_state == "NORMAL"
    assert snap.anomaly_score <= 0.25
    assert snap.active_alert is None


# 7. Demo Snapshot Suspicious
def test_provider_demo_snapshot_suspicious():
    provider = DashboardDataProvider()
    snap = provider.create_demo_snapshot("SUSPICIOUS")
    assert snap.risk_state == "SUSPICIOUS"
    assert 0.40 <= snap.anomaly_score <= 0.65
    assert snap.active_alert is not None
    assert snap.active_alert["severity"] == "WARNING"


# 8. Demo Snapshot High Risk
def test_provider_demo_snapshot_high_risk():
    provider = DashboardDataProvider()
    snap = provider.create_demo_snapshot("HIGH_RISK")
    assert snap.risk_state == "HIGH_RISK"
    assert snap.anomaly_score >= 0.65
    assert snap.active_alert is not None
    assert snap.active_alert["severity"] == "HIGH"


# 9. Demo Snapshot Emergency
def test_provider_demo_snapshot_emergency():
    provider = DashboardDataProvider()
    snap = provider.create_demo_snapshot("EMERGENCY")
    assert snap.risk_state == "EMERGENCY"
    assert snap.anomaly_score >= 0.85
    assert snap.critical_evidence is True
    assert snap.active_alert is not None
    assert snap.active_alert["severity"] == "CRITICAL"


# 10. Demo Snapshot Recovery
def test_provider_demo_snapshot_recovery():
    provider = DashboardDataProvider()
    snap = provider.create_demo_snapshot("RECOVERY")
    assert snap.risk_state == "RECOVERY"
    assert snap.anomaly_score <= 0.35
    assert snap.active_alert is None


# 11. Missing Video Fallback
def test_provider_missing_video():
    provider = DashboardDataProvider()
    snap = provider.create_snapshot_from_subsystems(detection_result=None, frame=None)
    assert snap.video_frame is None
    assert snap.crowd_count == 0
    assert snap.system_health["video_capture"] is False


# 12. Missing Audio Fallback
def test_provider_missing_audio():
    provider = DashboardDataProvider()
    snap = provider.create_snapshot_from_subsystems(audio_record=None)
    assert snap.dominant_audio_event.lower() == "normal"
    assert snap.audio_score == 0.0
    assert snap.system_health["audio_analysis"] is False


# 13. Missing XAI Fallback
def test_provider_missing_xai():
    provider = DashboardDataProvider()
    snap = provider.create_snapshot_from_subsystems(evidence_record=None)
    assert snap.xai_explanation is not None
    assert len(snap.xai_explanation) > 0
    assert snap.operator_action is not None


# 14. State Ingestion
def test_state_add_snapshot():
    state = DashboardState(max_history=50)
    snap = DashboardSnapshot(
        timestamp=time.time(),
        risk_state="NORMAL",
        anomaly_score=0.15,
        video_score=0.12,
        audio_score=0.08,
        crowd_count=8,
        crowd_density=0.20,
    )
    state.add_snapshot(snap)
    assert len(state.history_timestamps) == 1
    assert state.history_anomaly_scores[0] == 0.15
    assert state.last_snapshot == snap


# 15. State Bounded History (Strict Memory Bound)
def test_state_bounded_history():
    state = DashboardState(max_history=10)
    for i in range(25):
        snap = DashboardSnapshot(
            timestamp=float(i),
            anomaly_score=i * 0.04,
            crowd_count=i,
        )
        state.add_snapshot(snap)

    assert len(state.history_timestamps) == 10
    assert len(state.history_anomaly_scores) == 10
    assert state.history_timestamps[0] == 15.0
    assert state.history_timestamps[-1] == 24.0


# 16. Active Alert Tracking in State
def test_state_active_alert_tracking():
    state = DashboardState()
    snap = DashboardSnapshot(
        timestamp=time.time(),
        risk_state="EMERGENCY",
        active_alert={
            "alert_id": "ALT-999",
            "severity": "CRITICAL",
            "state": "EMERGENCY",
            "message": "Emergency situation",
        },
    )
    state.add_snapshot(snap)
    assert state.active_alert is not None
    assert state.active_alert["alert_id"] == "ALT-999"
    assert len(state.recent_alerts) == 1


# 17. Alert Operator Acknowledgement
def test_state_acknowledge_alert():
    state = DashboardState()
    snap = DashboardSnapshot(
        timestamp=time.time(),
        risk_state="HIGH_RISK",
        active_alert={
            "alert_id": "ALT-555",
            "severity": "HIGH",
            "state": "HIGH_RISK",
        },
    )
    state.add_snapshot(snap)
    ok = state.acknowledge_alert("ALT-555", operator_id="officer_smith")
    assert ok is True
    assert "ALT-555" in state.acknowledged_alert_ids
    assert state.active_alert["acknowledged"] is True
    assert state.active_alert["status"] == "ACKNOWLEDGED"
    assert state.operator_actions[-1]["action"] == "ACKNOWLEDGE"


# 18. Alert Operator Resolution
def test_state_resolve_alert():
    state = DashboardState()
    snap = DashboardSnapshot(
        timestamp=time.time(),
        risk_state="EMERGENCY",
        active_alert={
            "alert_id": "ALT-777",
            "severity": "CRITICAL",
            "state": "EMERGENCY",
        },
    )
    state.add_snapshot(snap)
    ok = state.resolve_alert("ALT-777", operator_id="officer_smith", reason="Area secured")
    assert ok is True
    assert "ALT-777" in state.resolved_alert_ids
    assert state.active_alert is None  # Active alert cleared upon resolution
    assert state.recent_alerts[0]["resolved"] is True
    assert state.recent_alerts[0]["status"] == "RESOLVED"


# 19. State Clear History
def test_state_clear_history():
    state = DashboardState()
    snap = DashboardSnapshot(timestamp=time.time(), anomaly_score=0.5)
    state.add_snapshot(snap)
    assert len(state.history_timestamps) == 1

    state.clear_history()
    assert len(state.history_timestamps) == 0
    assert len(state.recent_alerts) == 0
    assert state.active_alert is None


# 20. Metrics FPS Calculation
def test_metrics_fps_calculation():
    # 10 timestamps spaced by 0.05 seconds => 20 FPS
    ts = [100.0 + i * 0.05 for i in range(10)]
    fps = DashboardMetrics.calculate_fps(ts)
    assert pytest.approx(fps, rel=0.05) == 20.0

    # Less than 2 timestamps should yield 0.0
    assert DashboardMetrics.calculate_fps([100.0]) == 0.0
    assert DashboardMetrics.calculate_fps([]) == 0.0


# 21. Metrics Latency Calculation
def test_metrics_latency_stats():
    latencies = [10.0, 15.0, 20.0, 25.0, 30.0]
    stats = DashboardMetrics.calculate_latency_stats(latencies)
    assert stats["mean_ms"] == 20.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 30.0
    assert stats["p95_ms"] >= 25.0

    empty_stats = DashboardMetrics.calculate_latency_stats([])
    assert empty_stats["mean_ms"] == 0.0


# 22. Metrics Summary Compilation
def test_metrics_summary():
    state = DashboardState(max_history=50)
    for i in range(5):
        snap = DashboardSnapshot(
            timestamp=100.0 + i * 0.1,
            anomaly_score=0.1 * i,
            crowd_count=10 + i,
            crowd_density=0.2 * i,
            pipeline_latency_ms=15.0 + i,
        )
        state.add_snapshot(snap)

    summary = DashboardMetrics.compute_summary(state)
    assert summary["history_frames"] == 5
    assert summary["peak_crowd_count"] == 14
    assert summary["peak_crowd_density"] == 0.8
    assert summary["latency"]["mean_ms"] == 17.0
    assert summary["anomaly"]["peak"] == 0.4


# 23. Theme Tokens and HTML Badges
def test_theme_tokens_and_badges():
    assert hasattr(DashboardTheme, "CUSTOM_CSS")
    assert "<style>" in DashboardTheme.CUSTOM_CSS
    assert DashboardTheme.get_state_color("NORMAL") == "#10B981"
    assert DashboardTheme.get_state_color("EMERGENCY") == "#EF4444"
    assert DashboardTheme.get_severity_color("CRITICAL") == "#EF4444"

    badge_html = DashboardTheme.get_state_badge_html("EMERGENCY", confidence=0.95)
    assert "EMERGENCY" in badge_html
    assert "95%" in badge_html

    health_html = DashboardTheme.get_health_badge_html("video_capture", True)
    assert "ONLINE" in health_html

    health_offline_html = DashboardTheme.get_health_badge_html("video_capture", False)
    assert "OFFLINE" in health_offline_html


# 24. Immutability Invariant: Dashboard Never Mutates Scores or States
def test_state_immutability_of_scores():
    original_score = 0.82
    original_state = "EMERGENCY"
    evt = DecisionEvent(
        timestamp=time.time(),
        previous_state="HIGH_RISK",
        current_state=original_state,
        score=original_score,
        confidence=0.90,
        reason="Significant crowd panic",
    )
    provider = DashboardDataProvider()
    snap = provider.create_snapshot_from_subsystems(decision_event=evt)
    state = DashboardState()
    state.add_snapshot(snap)

    # Ingesting, charting, and viewing MUST NEVER change the original decision event
    assert evt.score == original_score
    assert evt.current_state == original_state
    assert snap.anomaly_score == original_score
    assert snap.risk_state == original_state
