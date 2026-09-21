"""
==============================================================================
Test Suite for Phase 9: Explainable AI (XAI) & Evidence Visualization.
Verifies all 24 required test cases covering transparent attribution,
factual statements, academic integrity, read-only guarantees, formatting,
and edge cases.
==============================================================================
"""

import json
import time
import pytest
import numpy as np

from src.alerts.alert_record import AlertRecord, AlertSeverity, AlertStatus
from src.decision.decision_record import DecisionEvent, RiskState
from src.fusion.types import CrossModalSynergy, MultimodalAssessment
from src.xai.evidence_extractor import EvidenceExtractor
from src.xai.evidence_record import EvidenceRecord
from src.xai.explanation_engine import ExplanationEngine
from src.xai.explanation_formatter import ExplanationFormatter
from src.xai.feature_attribution import FeatureAttributor
from src.xai.gradcam_interface import GradCAMInterface
from src.xai.visual_explanation import VisualExplainer


@pytest.fixture
def xai_engine():
    """Returns a standard ExplanationEngine instance."""
    return ExplanationEngine()


@pytest.fixture
def base_normal_event():
    """Generates a nominal baseline DecisionEvent."""
    return DecisionEvent(
        timestamp=time.time(),
        previous_state=RiskState.NORMAL.value,
        current_state=RiskState.NORMAL.value,
        score=0.12,
        confidence=0.85,
        reason="Nominal conditions.",
        dominant_modality="none",
        video_score=0.10,
        audio_score=0.10,
        weapon_score=0.0,
        fight_score=0.0,
        corroborated=False,
        critical_evidence=False,
        evidence_summary={
            "video": {
                "density_m2": 0.5,
                "density_score": 0.1,
                "mean_speed": 2.1,
                "movement_score": 0.1,
                "dispersion_rate": 1.2,
                "direction_entropy": 0.8,
                "behavior_score": 0.1,
                "behavior_class": "normal",
                "fight_score": 0.0,
                "weapon_score": 0.0,
            },
            "audio": {
                "audio_event": "normal",
                "audio_confidence": 0.90,
                "is_active": True,
            },
        },
    )


# --------------------------------------------------------------------------
# Test 1: Minimal evidence on normal calm baseline
# --------------------------------------------------------------------------
def test_normal_explanation(xai_engine, base_normal_event):
    record = xai_engine.explain_decision(decision_event=base_normal_event)
    assert record.risk_state == "NORMAL"
    assert record.anomaly_score == 0.12
    assert "NORMAL" in record.short_explanation
    assert any("nominal" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 2: Identifies atypical agitation / suspicious evidence
# --------------------------------------------------------------------------
def test_suspicious_explanation(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.48,
        confidence=0.75,
        reason="Elevated acoustic agitation and brisk movement.",
        dominant_modality="audio",
        video_score=0.35,
        audio_score=0.55,
        weapon_score=0.0,
        fight_score=0.0,
        corroborated=False,
        critical_evidence=False,
        evidence_summary={
            "video": {"movement_score": 0.45, "mean_speed": 9.2, "density_m2": 1.2},
            "audio": {"audio_event": "shouting", "audio_confidence": 0.70, "is_active": True},
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert record.risk_state == "SUSPICIOUS"
    assert "SUSPICIOUS" in record.short_explanation
    assert any("movement" in s.lower() or "speed" in s.lower() for s in record.top_evidence)
    assert any("acoustic" in s.lower() or "shouting" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 3: Multi-factor high-risk evidence
# --------------------------------------------------------------------------
def test_high_risk_explanation(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="SUSPICIOUS",
        current_state="HIGH_RISK",
        score=0.72,
        confidence=0.82,
        reason="Physical altercation observed with rapid dispersion.",
        dominant_modality="video",
        video_score=0.78,
        audio_score=0.45,
        weapon_score=0.0,
        fight_score=0.75,
        corroborated=False,
        critical_evidence=False,
        evidence_summary={
            "video": {
                "fight_score": 0.75,
                "movement_score": 0.65,
                "dispersion_rate": 8.5,
                "direction_entropy": 2.2,
                "density_m2": 2.1,
                "density_level": "HIGH",
            },
            "audio": {"audio_event": "shouting", "audio_confidence": 0.65, "is_active": True},
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert record.risk_state == "HIGH_RISK"
    assert record.fight_score == 0.75
    assert any("altercation" in s.lower() or "struggle" in s.lower() for s in record.top_evidence)
    assert any("dispersion" in s.lower() or "entropy" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 4: Critical emergency rationale
# --------------------------------------------------------------------------
def test_emergency_explanation(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="HIGH_RISK",
        current_state="EMERGENCY",
        score=0.92,
        confidence=0.94,
        reason="Critical armed threat confirmed with scream audio synergy.",
        dominant_modality="bimodal",
        video_score=0.90,
        audio_score=0.85,
        weapon_score=0.95,
        fight_score=0.50,
        corroborated=True,
        critical_evidence=True,
        evidence_summary={
            "synergy_type": "WEAPON_AND_DISTRESS",
            "video": {"weapon_score": 0.95, "detected_weapons": ["knife"], "movement_score": 0.80},
            "audio": {"audio_event": "scream", "audio_confidence": 0.92, "is_active": True},
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert record.risk_state == "EMERGENCY"
    assert record.critical_evidence is True
    assert record.corroborated is True
    assert any("weapon" in s.lower() for s in record.top_evidence)
    assert any("synergy" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 5: Evidence ranking orders highest contribution factors first
# --------------------------------------------------------------------------
def test_evidence_ranking():
    attributor = FeatureAttributor()
    contributions = attributor.compute_contributions(
        density_score=0.20,
        movement_score=0.50,
        behavior_score=0.10,
        audio_score=0.30,
        fight_score=0.0,
        weapon_score=0.90,
        corroborated=False,
    )
    ranked = attributor.rank_contributions(contributions, top_k=3)
    assert len(ranked) >= 2
    # Weapon presence has highest raw score and weight -> must rank #1
    assert ranked[0][0] == "weapon_presence"
    # Ranked weights must be strictly non-increasing
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i + 1][1]


# --------------------------------------------------------------------------
# Test 6: Deterministic ranking produces identical results on identical input
# --------------------------------------------------------------------------
def test_deterministic_ranking():
    attributor = FeatureAttributor()
    runs = []
    for _ in range(10):
        c = attributor.compute_contributions(
            density_score=0.40,
            movement_score=0.40,
            behavior_score=0.40,
            audio_score=0.40,
            fight_score=0.40,
            weapon_score=0.40,
            corroborated=True,
        )
        ranked = attributor.rank_contributions(c, top_k=5)
        runs.append(ranked)

    # Every single run must match the first run exactly
    for run in runs[1:]:
        assert run == runs[0]


# --------------------------------------------------------------------------
# Test 7: Movement evidence captures velocity and dispersion
# --------------------------------------------------------------------------
def test_movement_evidence(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.55,
        confidence=0.80,
        reason="Rapid movement with crowd scattering.",
        dominant_modality="video",
        video_score=0.60,
        audio_score=0.10,
        evidence_summary={
            "video": {
                "mean_speed": 14.5,
                "movement_score": 0.70,
                "dispersion_rate": 8.0,
                "direction_entropy": 2.4,
            },
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert any("velocity" in s.lower() or "speed" in s.lower() for s in record.top_evidence)
    assert any("dispersion" in s.lower() or "scattering" in s.lower() for s in record.top_evidence)
    assert any("entropy" in s.lower() or "divergence" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 8: Density evidence captures elevated crowd density
# --------------------------------------------------------------------------
def test_density_evidence(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.50,
        confidence=0.85,
        reason="Severe crowd congestion.",
        dominant_modality="video",
        video_score=0.55,
        audio_score=0.10,
        evidence_summary={
            "video": {
                "density_m2": 2.8,
                "density_score": 0.85,
                "density_level": "HIGH",
            },
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert any("density" in s.lower() for s in record.top_evidence)
    assert record.crowd_density == 2.8


# --------------------------------------------------------------------------
# Test 9: Audio evidence explains acoustic energy honestly (spectral baseline)
# --------------------------------------------------------------------------
def test_audio_evidence(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.58,
        confidence=0.72,
        reason="Loud acoustic impulse detected.",
        dominant_modality="audio",
        video_score=0.15,
        audio_score=0.68,
        evidence_summary={
            "audio": {
                "audio_event": "explosion_like",
                "audio_confidence": 0.75,
                "model_path": "",  # Empty = spectral baseline
                "is_active": True,
            },
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert record.audio_details["mode"] == "spectral_baseline"
    # Statement must transparently reflect spectral baseline evidence
    assert any("spectral baseline evidence" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 10: Fight evidence captures physical altercation dynamics
# --------------------------------------------------------------------------
def test_fight_evidence(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="NORMAL",
        current_state="HIGH_RISK",
        score=0.75,
        confidence=0.88,
        reason="Aggressive combat struggle observed.",
        dominant_modality="video",
        video_score=0.80,
        audio_score=0.30,
        fight_score=0.82,
        evidence_summary={
            "video": {"fight_score": 0.82, "movement_score": 0.60},
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert record.fight_score == 0.82
    assert any("altercation" in s.lower() or "struggle" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 11: Weapon evidence captures potential dangerous object
# --------------------------------------------------------------------------
def test_weapon_evidence(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="SUSPICIOUS",
        current_state="EMERGENCY",
        score=0.90,
        confidence=0.92,
        reason="Firearm detected in surveillance field.",
        dominant_modality="video",
        video_score=0.92,
        audio_score=0.20,
        weapon_score=0.95,
        critical_evidence=True,
        evidence_summary={
            "video": {"weapon_score": 0.95, "detected_weapons": ["firearm"]},
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert record.weapon_score == 0.95
    assert any("firearm" in s.lower() or "weapon" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 12: Corroboration explanation explains mutual audio-video synergy
# --------------------------------------------------------------------------
def test_corroboration_explanation(xai_engine):
    assessment = MultimodalAssessment(
        multimodal_score=0.88,
        multimodal_confidence=0.92,
        video_score=0.82,
        audio_score=0.85,
        status="EMERGENCY",
        dominant_modality="bimodal",
        synergy=CrossModalSynergy(
            synergy_detected=True,
            synergy_type="SCREAM_AND_SCATTER",
            boost_multiplier=1.25,
            description="Acoustic scream coupled with crowd scattering.",
        ),
    )
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="HIGH_RISK",
        current_state="EMERGENCY",
        score=0.88,
        confidence=0.92,
        reason="Corroborated scream and crowd stampede.",
        dominant_modality="bimodal",
        corroborated=True,
        evidence_summary={"synergy_type": "SCREAM_AND_SCATTER"},
    )
    record = xai_engine.explain_decision(decision_event=event, multimodal_assessment=assessment)
    assert record.corroborated is True
    assert any("cross-modal synergy" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 13: Single modality explanation explains single sensor reliance
# --------------------------------------------------------------------------
def test_single_modality_explanation(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.55,
        confidence=0.65,
        reason="Acoustic stream offline.",
        dominant_modality="video",
        video_score=0.55,
        audio_score=0.0,
        evidence_summary={
            "video": {"movement_score": 0.60, "is_active": True},
            "audio": {"is_active": False},
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert any("single modality reliance" in s.lower() and "acoustic" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 14: Conflicting modality explanation explains divergent cues
# --------------------------------------------------------------------------
def test_conflicting_modality_explanation(xai_engine):
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.42,
        confidence=0.55,
        reason="Visual agitation unsupported by calm acoustics.",
        dominant_modality="video",
        video_score=0.65,
        audio_score=0.08,
        evidence_summary={
            "video": {"movement_score": 0.65, "is_active": True},
            "audio": {"audio_event": "normal", "audio_score": 0.08, "is_active": True},
        },
    )
    record = xai_engine.explain_decision(decision_event=event)
    assert any("divergent" in s.lower() for s in record.top_evidence)


# --------------------------------------------------------------------------
# Test 15: Serialization to_dict() and to_json() schema verification
# --------------------------------------------------------------------------
def test_evidence_record_serialization(xai_engine, base_normal_event):
    record = xai_engine.explain_decision(decision_event=base_normal_event)
    d = record.to_dict()
    assert isinstance(d, dict)
    assert "timestamp" in d
    assert "risk_state" in d
    assert "anomaly_score" in d
    assert "decision_confidence" in d
    assert "top_evidence" in d
    assert "evidence_contributions" in d
    assert "transparency" in d

    raw_json = record.to_json()
    assert isinstance(raw_json, str)
    parsed = json.loads(raw_json)
    assert parsed["risk_state"] == "NORMAL"
    assert parsed["anomaly_score"] == 0.12


# --------------------------------------------------------------------------
# Test 16: Short explanation formatting
# --------------------------------------------------------------------------
def test_short_explanation_formatting():
    record = EvidenceRecord(
        timestamp=time.time(),
        risk_state="SUSPICIOUS",
        anomaly_score=0.52,
        decision_confidence=0.78,
        dominant_modality="audio",
        evidence_contributions={"acoustic_anomaly": 0.60, "rapid_movement": 0.40},
        ranked_contributions=[("acoustic_anomaly", 0.60), ("rapid_movement", 0.40)],
    )
    short = ExplanationFormatter.format_short(record)
    assert "SUSPICIOUS" in short
    assert "acoustic anomaly" in short
    assert "60%" in short


# --------------------------------------------------------------------------
# Test 17: Detailed explanation formatting
# --------------------------------------------------------------------------
def test_detailed_explanation_formatting():
    record = EvidenceRecord(
        timestamp=time.time(),
        risk_state="HIGH_RISK",
        anomaly_score=0.74,
        decision_confidence=0.85,
        dominant_modality="video",
        top_evidence=["Physical struggle detected.", "Radial crowd scattering detected."],
        evidence_contributions={"physical_struggle": 0.55, "rapid_movement": 0.45},
        ranked_contributions=[("physical_struggle", 0.55), ("rapid_movement", 0.45)],
    )
    detailed = ExplanationFormatter.format_detailed(record)
    assert "SURVEILLANCE XAI DIAGNOSTIC REPORT" in detailed
    assert "Physical Struggle" in detailed
    assert "Physical struggle detected." in detailed
    assert "Model explanation unavailable" in detailed


# --------------------------------------------------------------------------
# Test 18: HUD formatting returns compact bullet points
# --------------------------------------------------------------------------
def test_hud_formatting():
    record = EvidenceRecord(
        timestamp=time.time(),
        risk_state="EMERGENCY",
        anomaly_score=0.88,
        decision_confidence=0.90,
        ranked_contributions=[("weapon_presence", 0.70)],
        top_evidence=["Weapon detected in field.", "Radial crowd dispersion."],
    )
    hud_lines = ExplanationFormatter.format_hud(record, max_lines=4)
    assert isinstance(hud_lines, list)
    assert len(hud_lines) <= 4
    assert any("EMERGENCY" in line for line in hud_lines)
    assert any("Weapon Presence" in line for line in hud_lines)


# --------------------------------------------------------------------------
# Test 19: Grad-CAM safe unavailable state when unconfigured
# --------------------------------------------------------------------------
def test_gradcam_unavailable_state():
    gradcam = GradCAMInterface(model=None, target_layer_name=None)
    assert gradcam.is_available() is False


# --------------------------------------------------------------------------
# Test 20: Grad-CAM does not fabricate synthetic heatmaps
# --------------------------------------------------------------------------
def test_gradcam_does_not_fabricate_heatmaps():
    gradcam = GradCAMInterface()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = gradcam.generate_heatmap(dummy_frame)
    assert res["available"] is False
    assert res["heatmap"] is None
    assert "unavailable" in res["reason"].lower()

    # Overlay on None heatmap returns unchanged frame
    overlaid = gradcam.overlay_heatmap(dummy_frame, None)
    assert np.array_equal(dummy_frame, overlaid)


# --------------------------------------------------------------------------
# Test 21: XAI does not modify decision state (Immutability guarantee)
# --------------------------------------------------------------------------
def test_xai_does_not_modify_decision_state(xai_engine, base_normal_event):
    original_state = base_normal_event.current_state
    xai_engine.explain_decision(decision_event=base_normal_event)
    assert base_normal_event.current_state == original_state


# --------------------------------------------------------------------------
# Test 22: XAI does not modify anomaly score (Immutability guarantee)
# --------------------------------------------------------------------------
def test_xai_does_not_modify_anomaly_score(xai_engine, base_normal_event):
    original_score = base_normal_event.score
    assessment = MultimodalAssessment(
        multimodal_score=0.65,
        multimodal_confidence=0.80,
        video_score=0.60,
        audio_score=0.70,
        status="SUSPICIOUS",
        dominant_modality="bimodal",
    )
    xai_engine.explain_decision(decision_event=base_normal_event, multimodal_assessment=assessment)
    assert base_normal_event.score == original_score
    assert assessment.multimodal_score == 0.65


# --------------------------------------------------------------------------
# Test 23: XAI does not modify alert severity (Immutability guarantee)
# --------------------------------------------------------------------------
def test_xai_does_not_modify_alert_severity(xai_engine):
    alert = AlertRecord(
        alert_id="ALT-TEST-001",
        timestamp=time.time(),
        severity=AlertSeverity.WARNING,
        risk_state="SUSPICIOUS",
        score=0.55,
        confidence=0.80,
        reason="Suspicious movement.",
    )
    original_sev = alert.severity
    original_status = alert.status
    record = xai_engine.explain_alert(alert_record=alert)
    assert alert.severity == original_sev
    assert alert.status == original_status
    assert record.alert_rationale is not None


# --------------------------------------------------------------------------
# Test 24: Missing or None evidence handled gracefully
# --------------------------------------------------------------------------
def test_missing_evidence_handled_safely(xai_engine):
    # Calling explain_decision with all None arguments should never crash
    record = xai_engine.explain_decision(
        decision_event=None,
        multimodal_assessment=None,
        alert_record=None,
    )
    assert isinstance(record, EvidenceRecord)
    assert record.risk_state == "NORMAL"
    assert record.anomaly_score == 0.0
    assert record.decision_confidence == 1.0
    assert record.explanation != ""
    assert isinstance(record.hud_explanation, list)

    # Visual explainer with empty tracks or None frame
    explainer = VisualExplainer()
    assert explainer.draw_motion_vectors(None, []) is None
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert explainer.draw_motion_vectors(frame, []) is not None
    assert explainer.draw_dispersion_indicators(frame, [], dispersion_rate=10.0) is not None
    assert explainer.draw_xai_hud_card(frame, record) is not None
