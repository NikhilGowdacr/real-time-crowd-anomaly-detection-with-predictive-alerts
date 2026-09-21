"""
==============================================================================
Unit Tests for Phase 7 Intelligent Decision Engine.
Verifies state machine transitions, dual-threshold hysteresis, dynamic thresholds,
temporal confirmation quotas, recovery stabilization, critical safety rules,
cooldown debouncing, confidence separation, and DecisionEvent schema.
==============================================================================
"""

import time
import numpy as np
import pytest
from typing import Any, Dict, List, Optional

from src.decision.decision_record import DecisionEvent, RiskState
from src.decision.hysteresis import HysteresisManager
from src.decision.confirmation import TemporalConfirmationTracker
from src.decision.state_machine import DecisionStateMachine
from src.decision.decision_engine import DecisionEngine
from src.fusion.types import CrossModalSynergy, MultimodalAssessment
from src.utils.config import load_config


def make_assessment(
    score: float = 0.0,
    confidence: float = 0.80,
    video_score: float = 0.0,
    audio_score: float = 0.0,
    weapon_score: float = 0.0,
    fight_score: float = 0.0,
    density_score: float = 0.0,
    behavior_class: str = "normal",
    audio_event: str = "normal",
    synergy_type: Optional[str] = None,
    dominant_modality: str = "bimodal",
    is_video_active: bool = True,
    is_audio_active: bool = True,
    timestamp: Optional[float] = None,
) -> MultimodalAssessment:
    """Helper to formulate calibrated MultimodalAssessment instances for testing."""
    curr_t = time.time() if timestamp is None else float(timestamp)
    syn = CrossModalSynergy(
        synergy_detected=bool(synergy_type is not None),
        synergy_type=synergy_type,
        boost_multiplier=1.25 if synergy_type else 1.0,
        description=f"Synergy {synergy_type}" if synergy_type else "",
    )
    evidence_breakdown = {
        "video": {
            "density_score": density_score,
            "weapon_score": weapon_score,
            "fight_score": fight_score,
            "behavior_class": behavior_class,
            "is_active": is_video_active,
        },
        "audio": {
            "audio_event": audio_event,
            "audio_anomaly_score": audio_score,
            "is_active": is_audio_active,
        },
        "fusion_weights": {"video": 0.6, "audio": 0.4},
    }
    return MultimodalAssessment(
        multimodal_score=score,
        multimodal_confidence=confidence,
        video_score=video_score,
        audio_score=audio_score,
        status="NORMAL" if score < 0.35 else ("EMERGENCY" if score >= 0.70 else "SUSPICIOUS"),
        dominant_modality=dominant_modality,
        synergy=syn,
        false_positive_suppressed=False,
        evidence_breakdown=evidence_breakdown,
        timestamp=curr_t,
        description="Test surveillance record",
    )


# ============================================================================
# 1. Engine Initialization & Default Config
# ============================================================================
def test_decision_config_defaults():
    """Test 1: Config loading includes valid Phase 7 decision parameters."""
    cfg = load_config("configs/config.yaml")
    assert "decision" in cfg
    dec = cfg["decision"]
    assert dec["enabled"] is True
    assert dec["suspicious_enter"] == 0.45
    assert dec["suspicious_exit"] == 0.35
    assert dec["high_risk_enter"] == 0.65
    assert dec["high_risk_exit"] == 0.55
    assert dec["emergency_enter"] == 0.85
    assert dec["emergency_exit"] == 0.70
    assert dec["suspicious_confirmation"] == 3
    assert dec["high_risk_confirmation"] == 3
    assert dec["emergency_confirmation"] == 2
    assert dec["recovery_confirmation"] == 5
    assert dec["transition_cooldown_seconds"] == 3.0
    assert dec["critical_evidence_enabled"] is True
    assert dec["dynamic_thresholds"] is True


def test_initial_state_normal():
    """Test 2: Engine initializes in baseline NORMAL state."""
    engine = DecisionEngine()
    assert engine.current_state == RiskState.NORMAL
    event = engine.evaluate(None)
    assert event.current_state == RiskState.NORMAL.value
    assert event.previous_state == RiskState.NORMAL.value
    assert event.transition is False


# ============================================================================
# 2. State Transitions & Confirmation Persistence
# ============================================================================
def test_state_transition_normal_to_suspicious():
    """Test 3: Transition to SUSPICIOUS requires temporal confirmation (3 frames)."""
    engine = DecisionEngine(suspicious_confirmation=3, dynamic_thresholds=False)

    # Frame 1: Elevated score 0.50 (above suspicious_enter 0.45), but frame count = 1
    a1 = make_assessment(score=0.50)
    e1 = engine.evaluate(a1)
    assert e1.current_state == RiskState.NORMAL.value
    assert e1.transition is False
    assert e1.confirmed is False

    # Frame 2: Still elevated
    e2 = engine.evaluate(a1)
    assert e2.current_state == RiskState.NORMAL.value
    assert e2.confirmed is False

    # Frame 3: Confirmed (3 frames fulfilled) -> State transitions to SUSPICIOUS
    e3 = engine.evaluate(a1)
    assert e3.current_state == RiskState.SUSPICIOUS.value
    assert e3.previous_state == RiskState.NORMAL.value
    assert e3.transition is True
    assert e3.confirmed is True


def test_confirmation_quota_reset_on_transient_drop():
    """Test 4: Transient noise spike does not escalate if score drops before quota."""
    engine = DecisionEngine(suspicious_confirmation=3, dynamic_thresholds=False)

    # 2 frames of elevated score
    a_high = make_assessment(score=0.50)
    engine.evaluate(a_high)
    engine.evaluate(a_high)

    # Score drops back to nominal baseline (0.10) on frame 3
    a_low = make_assessment(score=0.10)
    e3 = engine.evaluate(a_low)
    assert e3.current_state == RiskState.NORMAL.value
    assert e3.transition is False
    assert engine.confirmation.consecutive_count == 0


# ============================================================================
# 3. Dual-Threshold Hysteresis Stability
# ============================================================================
def test_hysteresis_enter_threshold_respected():
    """Test 5: Score must strictly exceed enter threshold to initiate escalation."""
    engine = DecisionEngine(suspicious_enter=0.45, dynamic_thresholds=False)
    # Score 0.42 is below suspicious_enter 0.45 -> Remains NORMAL
    for _ in range(5):
        event = engine.evaluate(make_assessment(score=0.42))
    assert event.current_state == RiskState.NORMAL.value


def test_hysteresis_exit_threshold_prevents_premature_deescalation():
    """Test 6: Score below enter threshold but above exit threshold stays in current state."""
    engine = DecisionEngine(
        suspicious_enter=0.45,
        suspicious_exit=0.35,
        suspicious_confirmation=2,
        dynamic_thresholds=False,
    )

    # Escalate to SUSPICIOUS with 0.50 across 2 frames
    engine.evaluate(make_assessment(score=0.50))
    e_esc = engine.evaluate(make_assessment(score=0.50))
    assert e_esc.current_state == RiskState.SUSPICIOUS.value

    # Score drops to 0.40 (between exit 0.35 and enter 0.45: Hysteresis deadband)
    for _ in range(5):
        e_mid = engine.evaluate(make_assessment(score=0.40))
        assert e_mid.current_state == RiskState.SUSPICIOUS.value

    # Score drops below exit 0.35 (e.g. 0.30) for recovery confirmation -> de-escalates to NORMAL
    for _ in range(5):
        e_exit = engine.evaluate(make_assessment(score=0.30))
    assert e_exit.current_state == RiskState.NORMAL.value


def test_score_oscillation_resistance():
    """Test 7: Score oscillating across a single threshold does not cause rapid state flapping."""
    engine = DecisionEngine(
        suspicious_enter=0.45,
        suspicious_exit=0.35,
        suspicious_confirmation=2,
        dynamic_thresholds=False,
    )

    # Simulate fluctuating score between 0.38 and 0.44 around 0.40
    transitions = 0
    for i in range(20):
        score = 0.38 if i % 2 == 0 else 0.44
        event = engine.evaluate(make_assessment(score=score))
        if event.transition:
            transitions += 1

    # In a naive 0.40 single threshold system this would flap ~10 times.
    # With hysteresis [0.35, 0.45], it remains securely in NORMAL with 0 transitions!
    assert transitions == 0
    assert engine.current_state == RiskState.NORMAL


# ============================================================================
# 4. Recovery Phase and Controlled De-escalation
# ============================================================================
def test_recovery_flow_from_emergency():
    """Test 8: Emergency de-escalation must pass through RECOVERY with 5 confirmation frames."""
    engine = DecisionEngine(
        emergency_enter=0.85,
        emergency_exit=0.70,
        emergency_confirmation=1,
        recovery_confirmation=5,
        dynamic_thresholds=False,
    )

    # Escalate to EMERGENCY
    e_em = engine.evaluate(make_assessment(score=0.90))
    assert e_em.current_state == RiskState.EMERGENCY.value

    # Threat subsides: score drops to 0.20 (< emergency_exit 0.70)
    # Next evaluation enters RECOVERY immediately
    e_rec1 = engine.evaluate(make_assessment(score=0.20))
    assert e_rec1.current_state == RiskState.RECOVERY.value
    assert e_rec1.previous_state == RiskState.EMERGENCY.value

    # During recovery (frames 1-4 with score 0.20), remains in RECOVERY
    for _ in range(4):
        e_rec = engine.evaluate(make_assessment(score=0.20))
        assert e_rec.current_state == RiskState.RECOVERY.value

    # Frame 5 of sustained calm score: completes recovery confirmation (quota=5) -> de-escalates to NORMAL
    e_norm = engine.evaluate(make_assessment(score=0.20))
    assert e_norm.current_state == RiskState.NORMAL.value
    assert e_norm.previous_state == RiskState.RECOVERY.value


# ============================================================================
# 5. Critical Safety Evidence Rules
# ============================================================================
def test_critical_weapon_evidence_accelerated_escalation():
    """Test 9: Critical weapon detection (>=0.90) accelerates emergency escalation in 1 frame."""
    engine = DecisionEngine(
        critical_evidence_enabled=True,
        critical_weapon_threshold=0.90,
        emergency_confirmation=3,  # Standard emergency requires 3 frames
    )

    # Single frame with verified dangerous object confidence 0.92
    a_crit = make_assessment(
        score=0.80,
        weapon_score=0.92,
        behavior_class="normal",
    )
    event = engine.evaluate(a_crit)
    assert event.critical_evidence is True
    assert event.current_state == RiskState.EMERGENCY.value
    assert event.transition is True
    assert "critical safety evidence" in event.reason.lower()


def test_critical_fight_evidence():
    """Test 10: Critical fight evidence (>=0.90) accelerates escalation."""
    engine = DecisionEngine(
        critical_evidence_enabled=True,
        critical_fight_threshold=0.90,
        high_risk_confirmation=3,
    )

    a_fight = make_assessment(
        score=0.75,
        fight_score=0.92,
        audio_event="shouting",
    )
    event = engine.evaluate(a_fight)
    assert event.critical_evidence is True
    assert event.current_state in (RiskState.HIGH_RISK.value, RiskState.EMERGENCY.value)


# ============================================================================
# 6. Dynamic Threshold Adaptation
# ============================================================================
def test_dynamic_threshold_bounds_and_density_offset():
    """Test 11: High crowd density offsets enter thresholds within bounded margin."""
    mgr = HysteresisManager(
        suspicious_enter=0.45,
        suspicious_exit=0.35,
        dynamic_thresholds=True,
        max_threshold_adjustment=0.10,
    )

    # Baseline (density=0.0)
    t_base = mgr.get_effective_thresholds(density_score=0.0)
    assert t_base["suspicious_enter"] == 0.45

    # Dense crowd (density=0.95): thresholds dynamically elevated to prevent false alarms
    t_dense = mgr.get_effective_thresholds(density_score=0.95)
    assert t_dense["suspicious_enter"] > 0.45
    assert t_dense["suspicious_enter"] <= 0.45 + 0.10
    # Invariant: enter > exit gap preserved
    assert t_dense["suspicious_enter"] > t_dense["suspicious_exit"]


def test_dynamic_threshold_corroboration_reduction():
    """Test 12: Multimodal corroboration lowers enter thresholds for higher sensitivity."""
    mgr = HysteresisManager(
        suspicious_enter=0.45,
        dynamic_thresholds=True,
        max_threshold_adjustment=0.10,
    )
    t_corrob = mgr.get_effective_thresholds(corroborated=True)
    assert t_corrob["suspicious_enter"] < 0.45
    assert t_corrob["suspicious_enter"] >= 0.45 - 0.10


# ============================================================================
# 7. Transition Cooldown & Debounce
# ============================================================================
def test_transition_cooldown_debouncing():
    """Test 13: Transition cooldown prevents repeated transition event emissions within window."""
    sm = DecisionStateMachine(
        initial_state=RiskState.NORMAL,
        transition_cooldown_seconds=3.0,
    )
    t0 = 100.0

    # First transition: NORMAL -> SUSPICIOUS at t=100.0s
    changed, emitted = sm.transition_to(RiskState.SUSPICIOUS, current_time=t0)
    assert changed is True
    assert emitted is True

    # Immediate second transition attempt at t=101.0s (within 3.0s cooldown)
    changed2, emitted2 = sm.transition_to(RiskState.NORMAL, current_time=t0 + 1.0)
    assert changed2 is True
    assert emitted2 is False  # State changed internally but event emission debounced

    # After cooldown expires at t=104.5s
    changed3, emitted3 = sm.transition_to(RiskState.SUSPICIOUS, current_time=t0 + 4.5)
    assert changed3 is True
    assert emitted3 is True


# ============================================================================
# 8. Modality Availability & Confidence Separation
# ============================================================================
def test_single_modality_fallback_handling():
    """Test 14: Engine functions safely when one sensory stream is offline."""
    engine = DecisionEngine(dynamic_thresholds=True)

    # Video stream active, Audio stream offline
    a_vid_only = make_assessment(
        score=0.55,
        confidence=0.85,
        video_score=0.55,
        audio_score=0.0,
        dominant_modality="video",
        is_audio_active=False,
    )
    event = engine.evaluate(a_vid_only)
    assert event is not None
    assert event.dominant_modality == "video"
    # Single modality penalizes decision confidence slightly
    assert event.confidence < a_vid_only.multimodal_confidence


def test_confidence_separation():
    """Test 15: Decision confidence is computed distinctly from anomaly score and model conf."""
    engine = DecisionEngine()
    a = make_assessment(
        score=0.30,          # Moderate anomaly score
        confidence=0.90,     # High model confidence
        synergy_type="SCREAM_AND_SCATTER",
    )
    event = engine.evaluate(a)
    # Decision confidence integrates cross-modal corroboration (+0.10)
    assert event.confidence != a.multimodal_score
    assert event.confidence >= 0.70
    assert event.score == 0.30


# ============================================================================
# 9. Schema, Reasoning & Long-Term Stability
# ============================================================================
def test_objective_reason_generation():
    """Test 16: Reasons are objective and avoid claiming certainty of criminal activity."""
    engine = DecisionEngine(suspicious_confirmation=1)
    e_norm = engine.evaluate(make_assessment(score=0.0))
    assert "normal crowd activity" in e_norm.reason.lower()

    e_susp = engine.evaluate(make_assessment(score=0.50, audio_event="shouting"))
    assert "atypical" in e_susp.reason.lower() or "shouting" in e_susp.reason.lower()
    assert "crime" not in e_susp.reason.lower()
    assert "terrorist" not in e_susp.reason.lower()


def test_decision_event_to_dict_schema():
    """Test 17: DecisionEvent.to_dict() matches the required schema with all fields."""
    engine = DecisionEngine()
    event = engine.evaluate(make_assessment(score=0.20))
    d = event.to_dict()

    required_keys = [
        "timestamp",
        "previous_state",
        "current_state",
        "score",
        "confidence",
        "reason",
        "dominant_modality",
        "video_score",
        "audio_score",
        "weapon_score",
        "fight_score",
        "corroborated",
        "critical_evidence",
        "confirmed",
        "transition",
        "dynamic_thresholds",
        "evidence_summary",
    ]
    for key in required_keys:
        assert key in d, f"Missing key '{key}' in DecisionEvent.to_dict()"

    assert isinstance(d["timestamp"], float)
    assert isinstance(d["score"], float)
    assert isinstance(d["confidence"], float)
    assert isinstance(d["previous_state"], str)
    assert isinstance(d["current_state"], str)
    assert isinstance(d["corroborated"], bool)
    assert isinstance(d["transition"], bool)


def test_long_term_stability():
    """Test 18: Engine maintains state consistency over 500 continuous evaluations."""
    engine = DecisionEngine()
    # 500 frames of nominal baseline
    for _ in range(500):
        ev = engine.evaluate(make_assessment(score=0.10))
        assert ev.current_state == RiskState.NORMAL.value
    assert engine.current_state == RiskState.NORMAL
