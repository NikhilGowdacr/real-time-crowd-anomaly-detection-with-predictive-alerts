"""
==============================================================================
Unit Tests for Phase 8 Alert Management & Notification Subsystem.
Verifies severity mapping, policy evaluation, cooldown debouncing,
duplicate fingerprint suppression, operator acknowledgement lifecycle,
channel dispatch stubs, local alarm fail-safety, and AlertRecord schema.
==============================================================================
"""

import time
import pytest
from typing import Any, Dict, Optional

from src.alerts.acknowledgement import AlertAcknowledgementManager
from src.alerts.alert_manager import AlertManager
from src.alerts.alert_policy import AlertPolicy
from src.alerts.alert_record import AlertRecord, AlertSeverity, AlertStatus
from src.alerts.channels import (
    ConsoleChannel,
    EmailChannel,
    HUDChannel,
    LocalAlarmChannel,
    PushChannel,
    SMSChannel,
)
from src.alerts.cooldown import AlertCooldownManager
from src.alerts.deduplication import AlertDeduplicator
from src.alerts.local_alarm import LocalAlarmManager
from src.decision.decision_record import DecisionEvent, RiskState
from src.utils.config import load_config


def make_decision_event(
    state: str = "SUSPICIOUS",
    previous_state: str = "NORMAL",
    score: float = 0.50,
    confidence: float = 0.85,
    reason: str = "Atypical visual movement observed.",
    dominant_modality: str = "video",
    video_score: float = 0.50,
    audio_score: float = 0.10,
    weapon_score: float = 0.0,
    fight_score: float = 0.0,
    corroborated: bool = False,
    critical_evidence: bool = False,
    confirmed: bool = True,
    transition: bool = True,
    timestamp: Optional[float] = None,
) -> DecisionEvent:
    """Helper creating calibrated DecisionEvent instances for alert testing."""
    curr_t = time.time() if timestamp is None else float(timestamp)
    return DecisionEvent(
        timestamp=curr_t,
        previous_state=previous_state,
        current_state=state,
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
        confirmed=confirmed,
        transition=transition,
        dynamic_thresholds={"suspicious_enter": 0.45, "emergency_enter": 0.85},
        evidence_summary={"modality": dominant_modality},
    )


# ============================================================================
# 1. State-to-Severity Mapping Tests
# ============================================================================

def test_normal_produces_no_alert():
    """Test 1: NORMAL surveillance state never produces an alert."""
    manager = AlertManager()
    event = make_decision_event(state="NORMAL", score=0.10)
    alert = manager.process_decision(event)
    assert alert is None


def test_suspicious_creates_warning():
    """Test 2: Confirmed transition to SUSPICIOUS creates a WARNING alert."""
    manager = AlertManager()
    event = make_decision_event(state="SUSPICIOUS", score=0.50)
    alert = manager.process_decision(event)

    assert alert is not None
    assert alert.severity == AlertSeverity.WARNING
    assert alert.risk_state == "SUSPICIOUS"
    assert alert.status == AlertStatus.ACTIVE
    assert "WARNING" in alert.formatted_message


def test_high_risk_creates_high():
    """Test 3: Confirmed transition to HIGH_RISK creates a HIGH alert."""
    manager = AlertManager()
    event = make_decision_event(state="HIGH_RISK", score=0.70, fight_score=0.65)
    alert = manager.process_decision(event)

    assert alert is not None
    assert alert.severity == AlertSeverity.HIGH
    assert alert.risk_state == "HIGH_RISK"
    assert alert.fight_score == 0.65


def test_emergency_creates_critical():
    """Test 4: Confirmed transition to EMERGENCY creates a CRITICAL alert."""
    manager = AlertManager()
    event = make_decision_event(
        state="EMERGENCY",
        score=0.92,
        critical_evidence=True,
        reason="Critical safety evidence: Potential dangerous-object detected.",
    )
    alert = manager.process_decision(event)

    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL
    assert alert.risk_state == "EMERGENCY"
    assert alert.critical_evidence is True
    assert "CRITICAL" in alert.formatted_message


# ============================================================================
# 2. Cooldown & Escalation Tests
# ============================================================================

def test_cooldown_suppresses_duplicate():
    """Test 5: Repeated events within cooldown window are suppressed."""
    policy = AlertPolicy(cooldown_seconds=10.0, duplicate_window_seconds=2.0)
    manager = AlertManager(policy=policy)
    t0 = 1000.0

    # First event emits alert
    e1 = make_decision_event(state="SUSPICIOUS", reason="Atypical movement", timestamp=t0)
    a1 = manager.process_decision(e1, current_time=t0)
    assert a1 is not None

    # Second event 3 seconds later (within 10s cooldown) is suppressed
    e2 = make_decision_event(state="SUSPICIOUS", reason="Different observation", timestamp=t0 + 3.0)
    a2 = manager.process_decision(e2, current_time=t0 + 3.0)
    assert a2 is None


def test_escalation_bypasses_lower_severity_cooldown():
    """Test 6: Genuine severity escalation bypasses active cooldown immediately."""
    policy = AlertPolicy(cooldown_seconds=15.0, escalation_enabled=True)
    manager = AlertManager(policy=policy)
    t0 = 1000.0

    # 1. WARNING alert emitted
    e1 = make_decision_event(state="SUSPICIOUS", score=0.50, timestamp=t0)
    a1 = manager.process_decision(e1, current_time=t0)
    assert a1 is not None
    assert a1.severity == AlertSeverity.WARNING

    # 2. Escalate to HIGH_RISK 2 seconds later (within 15s cooldown) -> Allowed!
    e2 = make_decision_event(state="HIGH_RISK", score=0.72, timestamp=t0 + 2.0)
    a2 = manager.process_decision(e2, current_time=t0 + 2.0)
    assert a2 is not None
    assert a2.severity == AlertSeverity.HIGH

    # 3. Escalate to EMERGENCY 3 seconds later -> Allowed!
    e3 = make_decision_event(state="EMERGENCY", score=0.91, timestamp=t0 + 3.0)
    a3 = manager.process_decision(e3, current_time=t0 + 3.0)
    assert a3 is not None
    assert a3.severity == AlertSeverity.CRITICAL


# ============================================================================
# 3. Deduplication Tests
# ============================================================================

def test_duplicate_detection_works():
    """Test 7: Identical event fingerprint within duplicate window is suppressed."""
    dedup = AlertDeduplicator(duplicate_window_seconds=15.0)
    e = make_decision_event(state="SUSPICIOUS", dominant_modality="video")
    sig = dedup.generate_signature(e, AlertSeverity.WARNING)

    assert dedup.is_duplicate(sig, current_time=100.0) is False
    dedup.record_signature(sig, current_time=100.0)

    # Identical signature 5 seconds later is duplicate
    assert dedup.is_duplicate(sig, current_time=105.0) is True

    # After duplicate window expires (16 seconds later), no longer duplicate
    assert dedup.is_duplicate(sig, current_time=116.0) is False


def test_materially_different_evidence_creates_separate_alert():
    """Test 8: Events with materially different evidence generate distinct signatures."""
    dedup = AlertDeduplicator(duplicate_window_seconds=15.0)
    t = 100.0

    # Event A: Visual scattering
    e_scatter = make_decision_event(state="HIGH_RISK", dominant_modality="video", reason="Crowd scattering")
    sig_a = dedup.generate_signature(e_scatter, AlertSeverity.HIGH)
    dedup.record_signature(sig_a, t)

    # Event B: Fight altercation in audio
    e_fight = make_decision_event(state="HIGH_RISK", dominant_modality="audio", reason="Violent fight altercation")
    sig_b = dedup.generate_signature(e_fight, AlertSeverity.HIGH)

    assert sig_a != sig_b
    assert dedup.is_duplicate(sig_b, t + 1.0) is False


# ============================================================================
# 4. Operator Acknowledgement Lifecycle Tests
# ============================================================================

def test_acknowledgement_works():
    """Test 9: Acknowledging an active alert updates status to ACKNOWLEDGED."""
    ack_mgr = AlertAcknowledgementManager()
    alert = AlertRecord(
        alert_id="ALT-001",
        timestamp=100.0,
        severity=AlertSeverity.HIGH,
        risk_state="HIGH_RISK",
        score=0.75,
        confidence=0.88,
        reason="High crowd divergence",
    )
    ack_mgr.register(alert)

    assert alert.acknowledged is False
    assert alert.status == AlertStatus.ACTIVE

    success = ack_mgr.acknowledge("ALT-001", timestamp=105.0)
    assert success is True
    assert alert.acknowledged is True
    assert alert.acknowledged_at == 105.0
    assert alert.status == AlertStatus.ACKNOWLEDGED


def test_resolution_works():
    """Test 10: Resolving an alert updates status to RESOLVED."""
    ack_mgr = AlertAcknowledgementManager()
    alert = AlertRecord(
        alert_id="ALT-002",
        timestamp=100.0,
        severity=AlertSeverity.WARNING,
        risk_state="SUSPICIOUS",
        score=0.50,
        confidence=0.80,
        reason="Atypical movement",
    )
    ack_mgr.register(alert)

    success = ack_mgr.resolve("ALT-002", timestamp=110.0)
    assert success is True
    assert alert.resolved is True
    assert alert.resolved_at == 110.0
    assert alert.status == AlertStatus.RESOLVED


def test_reset_works():
    """Test 11: Resetting an alert clears acknowledged and resolved flags."""
    ack_mgr = AlertAcknowledgementManager()
    alert = AlertRecord(
        alert_id="ALT-003",
        timestamp=100.0,
        severity=AlertSeverity.CRITICAL,
        risk_state="EMERGENCY",
        score=0.90,
        confidence=0.95,
        reason="Mass panic dispersal",
    )
    ack_mgr.register(alert)
    ack_mgr.acknowledge("ALT-003")
    ack_mgr.resolve("ALT-003")

    success = ack_mgr.reset("ALT-003")
    assert success is True
    assert alert.status == AlertStatus.ACTIVE
    assert alert.acknowledged is False
    assert alert.resolved is False


# ============================================================================
# 5. Local Alarm & Hardware Fail-Safety Tests
# ============================================================================

def test_local_alarm_disabled_by_default():
    """Test 12: Local audible alarm is disabled by default."""
    alarm = LocalAlarmManager()
    assert alarm.enabled is False
    # Trigger returns MUTED when disabled
    status = alarm.trigger(AlertSeverity.CRITICAL)
    assert status == "MUTED"


def test_missing_audio_device_does_not_crash():
    """Test 13: Local alarm playback with unconfigured WAV file does not crash pipeline."""
    alarm = LocalAlarmManager(
        enabled=True,
        emergency_sound_path="nonexistent_siren.wav",
    )
    # Triggering should execute cleanly without raising unhandled exceptions
    status = alarm.trigger(AlertSeverity.CRITICAL)
    assert status in ("ALARM_TRIGGERED", "MUTED")
    alarm.stop()


# ============================================================================
# 6. Notification Channel Stubs (Academic Integrity)
# ============================================================================

def test_email_stub_does_not_claim_delivery():
    """Test 14: Email channel explicitly reports UNCONNECTED_STUB without false delivery."""
    ch = EmailChannel(enabled=True)
    alert = AlertRecord(
        alert_id="ALT-004",
        timestamp=100.0,
        severity=AlertSeverity.CRITICAL,
        risk_state="EMERGENCY",
        score=0.92,
        confidence=0.90,
        reason="Verified emergency",
    )
    status = ch.send(alert)
    assert status == "UNCONNECTED_STUB"


def test_sms_stub_does_not_claim_delivery():
    """Test 15: SMS channel explicitly reports UNCONNECTED_STUB without false delivery."""
    ch = SMSChannel(enabled=True)
    alert = AlertRecord(
        alert_id="ALT-005",
        timestamp=100.0,
        severity=AlertSeverity.HIGH,
        risk_state="HIGH_RISK",
        score=0.75,
        confidence=0.85,
        reason="Physical struggle",
    )
    status = ch.send(alert)
    assert status == "UNCONNECTED_STUB"


def test_push_stub_does_not_claim_delivery():
    """Test 16: Push notification channel explicitly reports UNCONNECTED_STUB."""
    ch = PushChannel(enabled=True)
    alert = AlertRecord(
        alert_id="ALT-006",
        timestamp=100.0,
        severity=AlertSeverity.WARNING,
        risk_state="SUSPICIOUS",
        score=0.55,
        confidence=0.80,
        reason="Acoustic scream",
    )
    status = ch.send(alert)
    assert status == "UNCONNECTED_STUB"


# ============================================================================
# 7. Record Serialization & History Tests
# ============================================================================

def test_alert_record_serialization_works():
    """Test 17: AlertRecord serializes to dictionary and JSON with all required keys."""
    alert = AlertRecord(
        alert_id="ALT-TEST-001",
        timestamp=1726000000.0,
        severity=AlertSeverity.CRITICAL,
        risk_state="EMERGENCY",
        score=0.91,
        confidence=0.94,
        reason="Rapid crowd dispersion with corroborating distress audio.",
        dominant_modality="bimodal",
        video_score=0.88,
        audio_score=0.94,
        weapon_score=0.0,
        fight_score=0.85,
        corroborated=True,
        critical_evidence=True,
    )
    d = alert.to_dict()
    assert d["alert_id"] == "ALT-TEST-001"
    assert d["severity"] == "CRITICAL"
    assert d["risk_state"] == "EMERGENCY"
    assert d["score"] == 0.91
    assert d["corroborated"] is True
    assert d["critical_evidence"] is True
    assert "formatted_message" in d

    json_str = alert.to_json()
    assert "ALT-TEST-001" in json_str
    assert "CRITICAL" in json_str


def test_alert_history_works():
    """Test 18: AlertManager maintains an ordered history of emitted alerts."""
    manager = AlertManager(policy=AlertPolicy(cooldown_seconds=0.0, duplicate_window_seconds=0.0))
    for i in range(5):
        e = make_decision_event(
            state="SUSPICIOUS",
            score=0.45 + (i * 0.05),
            reason=f"Scrutiny check {i}",
            timestamp=100.0 + i,
        )
        manager.process_decision(e, current_time=100.0 + i)

    history = manager.get_history(limit=10)
    assert len(history) == 5
    # Most recent first
    assert history[0].score == pytest.approx(0.65)


def test_active_alerts_retrieval_works():
    """Test 19: Unresolved alerts are accurately tracked and filtered upon resolution."""
    manager = AlertManager(policy=AlertPolicy(cooldown_seconds=0.0, duplicate_window_seconds=0.0))
    e1 = make_decision_event(state="SUSPICIOUS", timestamp=100.0)
    e2 = make_decision_event(state="HIGH_RISK", timestamp=101.0)

    a1 = manager.process_decision(e1, current_time=100.0)
    a2 = manager.process_decision(e2, current_time=101.0)
    assert len(manager.get_active_alerts()) == 2

    # Acknowledge a1 -> still active
    manager.acknowledge(a1.alert_id)
    assert len(manager.get_active_alerts()) == 2

    # Resolve a1 -> only a2 remains active
    manager.resolve(a1.alert_id)
    active = manager.get_active_alerts()
    assert len(active) == 1
    assert active[0].alert_id == a2.alert_id


# ============================================================================
# 8. Policy Filtering & Edge Case Tests
# ============================================================================

def test_recovery_behavior_works():
    """Test 20: RECOVERY emits an INFO alert only when recovery_alert is enabled."""
    # Case A: Disabled recovery alerts (default)
    mgr_no_rec = AlertManager(policy=AlertPolicy(recovery_alert=False))
    e_rec = make_decision_event(state="RECOVERY", previous_state="EMERGENCY", score=0.25)
    assert mgr_no_rec.process_decision(e_rec) is None

    # Case B: Enabled recovery alerts
    mgr_rec = AlertManager(
        policy=AlertPolicy(
            recovery_alert=True,
            minimum_severity=AlertSeverity.INFO,
        )
    )
    alert = mgr_rec.process_decision(e_rec)
    assert alert is not None
    assert alert.severity == AlertSeverity.INFO
    assert alert.risk_state == "RECOVERY"


def test_policy_minimum_severity_works():
    """Test 21: Policy suppresses alerts strictly below the configured minimum severity."""
    # Policy requiring at least HIGH severity
    policy = AlertPolicy(minimum_severity=AlertSeverity.HIGH)
    manager = AlertManager(policy=policy)

    # SUSPICIOUS maps to WARNING -> Suppressed
    e_susp = make_decision_event(state="SUSPICIOUS", score=0.50)
    assert manager.process_decision(e_susp) is None

    # HIGH_RISK maps to HIGH -> Emitted
    e_high = make_decision_event(state="HIGH_RISK", score=0.70)
    alert = manager.process_decision(e_high)
    assert alert is not None
    assert alert.severity == AlertSeverity.HIGH


def test_alert_manager_rejects_invalid_event_safely():
    """Test 22: AlertManager handles None and malformed inputs gracefully without exceptions."""
    manager = AlertManager()

    # None event
    assert manager.process_decision(None) is None

    # Arbitrary non-event object
    assert manager.process_decision("invalid_string_object") is None  # type: ignore

    # Unconfirmed event
    e_unconf = make_decision_event(state="EMERGENCY", confirmed=False)
    assert manager.process_decision(e_unconf) is None
