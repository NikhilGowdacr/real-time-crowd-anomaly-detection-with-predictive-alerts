"""
==============================================================================
Real-Time Crowd Anomaly Detection with Predictive Alerts
Unit Tests for Phase 6: Multimodal Audio-Visual Fusion Engine
==============================================================================
"""

import time
import numpy as np
import pytest

from src.audio.audio_anomaly import AudioEventRecord
from src.behavior.behavior_anomaly import BehaviorAssessment
from src.behavior.behavior_features import CrowdBehaviorMetrics
from src.fusion import (
    CrossModalSynergy,
    ModalityEvidence,
    MultimodalAssessment,
    MultimodalFusionEngine,
    TemporalStreamAligner,
)
from src.safety.safety_fusion import SafetyAssessment, SafetyStatus
from src.safety.weapon_detector import DetectedObject, WeaponDetectionResult
from src.utils.config import load_config
from src.video.crowd_density import DensityResult


# ============================================================================
# 1. Config Loading & Engine Initialization
# ============================================================================
def test_fusion_config_defaults():
    """Test 1: Config loads with correct multimodal fusion weights and tolerances."""
    config = load_config()
    assert "fusion" in config, "Missing 'fusion' section in configuration"
    fusion_cfg = config["fusion"]

    assert fusion_cfg["video_weight"] == 0.60
    assert fusion_cfg["audio_weight"] == 0.40
    assert fusion_cfg["density_sub_weight"] == 0.15
    assert fusion_cfg["movement_sub_weight"] == 0.20
    assert fusion_cfg["behavior_sub_weight"] == 0.25
    assert fusion_cfg["fight_sub_weight"] == 0.20
    assert fusion_cfg["weapon_sub_weight"] == 0.20
    assert fusion_cfg["synergy_boost"] == 1.25
    assert fusion_cfg["suppression_factor"] == 0.50
    assert fusion_cfg["temporal_tolerance_seconds"] == 1.5
    assert fusion_cfg["stale_timeout_seconds"] == 2.5
    assert fusion_cfg["smoothing_window"] == 5


def test_fusion_engine_initialization():
    """Test 2: MultimodalFusionEngine initializes with normalized weights."""
    engine = MultimodalFusionEngine(video_weight=6.0, audio_weight=4.0)
    assert np.isclose(engine.video_weight, 0.60, atol=1e-5)
    assert np.isclose(engine.audio_weight, 0.40, atol=1e-5)
    assert np.isclose(
        engine.density_sub_weight
        + engine.movement_sub_weight
        + engine.behavior_sub_weight
        + engine.fight_sub_weight
        + engine.weapon_sub_weight,
        1.0,
        atol=1e-5,
    )


# ============================================================================
# 2. Single-Modality Fallback & Dynamic Re-weighting
# ============================================================================
def test_single_modality_video_only():
    """Test 3: Video-only input produces valid assessment without audio stream."""
    engine = MultimodalFusionEngine()
    behavior_assessment = BehaviorAssessment(
        kinematic_score=0.20,
        deep_score=0.25,
        raw_score=0.22,
        smoothed_score=0.22,
        final_video_score=0.22,
        behavior_class="normal",
        status="NORMAL",
        consecutive_abnormal_frames=0,
        is_confirmed=False,
        description="Nominal behavior",
        active_people=5,
        mean_speed=10.0,
    )

    assessment = engine.fuse(behavior_assessment=behavior_assessment, audio_record=None)
    assert assessment is not None
    assert assessment.dominant_modality == "video"
    assert assessment.audio_score == 0.0
    assert assessment.multimodal_score > 0.0
    assert assessment.status == "NORMAL"


def test_single_modality_audio_only():
    """Test 4: Audio-only input produces valid assessment without video stream."""
    engine = MultimodalFusionEngine()
    audio_record = AudioEventRecord(
        audio_event="scream",
        audio_confidence=0.88,
        audio_anomaly_score=0.85,
        audio_timestamp=time.time(),
        duration=1.0,
        is_confirmed=True,
        consecutive_abnormal_count=3,
        status="EMERGENCY",
        description="Confirmed scream",
    )

    # Empty video inputs
    assessment = engine.fuse(audio_record=audio_record)
    assert assessment is not None
    assert assessment.audio_score == 0.85
    assert assessment.status in ("SUSPICIOUS", "EMERGENCY")


def test_dynamic_weight_renormalization():
    """Test 5: Weights re-normalize dynamically when a stream is missing or inactive."""
    engine = MultimodalFusionEngine(video_weight=0.70, audio_weight=0.30)
    # Both active
    audio_rec = AudioEventRecord(
        audio_event="normal",
        audio_confidence=0.9,
        audio_anomaly_score=0.0,
        audio_timestamp=time.time(),
        duration=1.0,
        is_confirmed=False,
        consecutive_abnormal_count=0,
        status="NORMAL",
        description="",
    )
    res_bimodal = engine.fuse(audio_record=audio_rec, video_active=True)
    weights = res_bimodal.evidence_breakdown["fusion_weights"]
    assert np.isclose(weights["video"], 0.70, atol=0.05)
    assert np.isclose(weights["audio"], 0.30, atol=0.05)

    # Audio missing -> Video dynamically receives 1.0 weight
    engine.aligner.reset()
    res_video_only = engine.fuse(audio_record=None, video_active=True)
    weights_v = res_video_only.evidence_breakdown["fusion_weights"]
    assert weights_v["video"] == 1.0
    assert weights_v["audio"] == 0.0

    # Video missing -> Audio dynamically receives 1.0 weight
    res_audio_only = engine.fuse(audio_record=audio_rec, video_active=False)
    weights_a = res_audio_only.evidence_breakdown["fusion_weights"]
    assert weights_a["video"] == 0.0
    assert weights_a["audio"] == 1.0


# ============================================================================
# 3. Video & Audio Feature Integration
# ============================================================================
def test_video_feature_extraction_all_5_signals():
    """Test 6: Multi-evidence video weighting correctly integrates all 5 video features."""
    engine = MultimodalFusionEngine()

    density = DensityResult(
        person_count=12,
        density_per_m2=1.8,
        density_score=1.0,
        density_level="CRITICAL",
        grid_counts=np.zeros((3, 3), dtype=int),
        grid_densities=np.zeros((3, 3), dtype=float),
        hotspot_cells=[],
        is_overcrowded=True,
    )
    kinematics = CrowdBehaviorMetrics(
        active_people=12,
        mean_velocity=24.0,
        velocity_variance=5.0,
        direction_variance=0.8,
        rapid_mover_ratio=0.5,
        direction_change_ratio=0.4,
        dispersion_rate=3.0,
        convergence_rate=0.5,
        kinematic_score=0.75,
        candidate_class="scattering",
        description="Panic scattering kinematics",
    )
    behavior = BehaviorAssessment(
        kinematic_score=0.75,
        deep_score=0.80,
        raw_score=0.78,
        smoothed_score=0.78,
        final_video_score=0.78,
        behavior_class="scattering",
        status="EMERGENCY",
        consecutive_abnormal_frames=5,
        is_confirmed=True,
        description="Panic scattering",
        active_people=12,
        mean_speed=24.0,
    )
    safety = SafetyAssessment(
        crowd_score=0.80,
        fight_score=0.70,
        weapon_score=0.85,
        movement_score=0.75,
        safety_score=0.82,
        status=SafetyStatus.EMERGENCY,
        is_weapon_fight_correlated=True,
    )
    weapon = WeaponDetectionResult(
        is_configured=True,
        status_message="OK",
        detected_objects=[
            DetectedObject(
                object_class="firearm",
                confidence=0.85,
                bbox=np.array([10, 10, 50, 50]),
                timestamp=time.time(),
                center=(30.0, 30.0),
            )
        ],
        max_confidence=0.85,
        weapon_score=0.85,
        inference_ms=5.0,
    )

    v_score, v_conf, v_evidence = engine.extract_video_features(
        density=density,
        kinematics=kinematics,
        behavior_assessment=behavior,
        safety_assessment=safety,
        weapon_result=weapon,
    )

    assert v_score >= 0.70, f"Expected high composite video score, got {v_score}"
    assert v_evidence["density_score"] == 1.0
    assert v_evidence["movement_score"] == 0.75
    assert v_evidence["behavior_score"] == 0.80
    assert v_evidence["fight_score"] == 0.70
    assert v_evidence["weapon_score"] == 0.85
    assert "firearm" in v_evidence["detected_weapons"]


def test_audio_feature_extraction_all_4_signals():
    """Test 7: Audio features integrated correctly (anomaly score, event label, confidence, temporal confirmation)."""
    engine = MultimodalFusionEngine()
    curr_t = time.time()
    audio_rec = AudioEventRecord(
        audio_event="alarm",
        audio_confidence=0.85,
        audio_anomaly_score=0.70,
        audio_timestamp=curr_t,
        duration=1.0,
        is_confirmed=True,
        consecutive_abnormal_count=4,
        status="EMERGENCY",
        description="Confirmed fire alarm",
    )

    assessment = engine.fuse(audio_record=audio_rec, current_time=curr_t)
    audio_ev = assessment.evidence_breakdown["audio"]

    assert audio_ev["audio_event"] == "alarm"
    assert audio_ev["audio_confidence"] == 0.85
    assert audio_ev["audio_anomaly_score"] == 0.70
    assert audio_ev["is_confirmed"] is True
    assert audio_ev["consecutive_abnormal_count"] == 4


# ============================================================================
# 4. Cross-Modal Synergy Detection
# ============================================================================
def test_cross_modal_synergy_scream_and_scatter():
    """Test 8: Cross-modal synergy boost triggered for Scream + Panic Scattering."""
    engine = MultimodalFusionEngine(synergy_boost=1.25)
    curr_t = time.time()

    behavior = BehaviorAssessment(
        kinematic_score=0.65,
        deep_score=0.70,
        raw_score=0.68,
        smoothed_score=0.68,
        final_video_score=0.68,
        behavior_class="scattering",
        status="EMERGENCY",
        consecutive_abnormal_frames=5,
        is_confirmed=True,
        description="Panic scattering",
        active_people=10,
        mean_speed=22.0,
    )
    audio_rec = AudioEventRecord(
        audio_event="scream",
        audio_confidence=0.90,
        audio_anomaly_score=0.85,
        audio_timestamp=curr_t,
        duration=1.0,
        is_confirmed=True,
        consecutive_abnormal_count=3,
        status="EMERGENCY",
        description="Confirmed scream",
    )

    assessment = engine.fuse(behavior_assessment=behavior, audio_record=audio_rec, current_time=curr_t)
    assert assessment.synergy.synergy_detected is True
    assert assessment.synergy.synergy_type == "SCREAM_AND_SCATTER"
    assert assessment.synergy.boost_multiplier >= 1.25
    assert assessment.status == "EMERGENCY"


def test_cross_modal_synergy_fight_and_shouting():
    """Test 9: Cross-modal synergy boost triggered for Fight + Shouting/Distress."""
    engine = MultimodalFusionEngine(synergy_boost=1.25)
    curr_t = time.time()

    safety = SafetyAssessment(
        crowd_score=0.20,
        fight_score=0.65,
        weapon_score=0.0,
        movement_score=0.50,
        safety_score=0.55,
        status=SafetyStatus.SUSPICIOUS,
    )
    audio_rec = AudioEventRecord(
        audio_event="shouting",
        audio_confidence=0.80,
        audio_anomaly_score=0.55,
        audio_timestamp=curr_t,
        duration=1.0,
        is_confirmed=True,
        consecutive_abnormal_count=2,
        status="SUSPICIOUS",
        description="Confirmed shouting",
    )

    assessment = engine.fuse(safety_assessment=safety, audio_record=audio_rec, current_time=curr_t)
    assert assessment.synergy.synergy_detected is True
    assert assessment.synergy.synergy_type == "FIGHT_AND_SHOUTING"


def test_cross_modal_synergy_weapon_and_distress():
    """Test 10: Cross-modal synergy boost triggered for Weapon + Scream/Distress."""
    engine = MultimodalFusionEngine(synergy_boost=1.25)
    curr_t = time.time()

    weapon = WeaponDetectionResult(
        is_configured=True,
        status_message="OK",
        detected_objects=[
            DetectedObject(
                object_class="firearm",
                confidence=0.80,
                bbox=np.array([0, 0, 10, 10]),
                timestamp=curr_t,
                center=(5.0, 5.0),
            )
        ],
        max_confidence=0.80,
        weapon_score=0.80,
        inference_ms=4.0,
    )
    audio_rec = AudioEventRecord(
        audio_event="scream",
        audio_confidence=0.85,
        audio_anomaly_score=0.85,
        audio_timestamp=curr_t,
        duration=1.0,
        is_confirmed=True,
        consecutive_abnormal_count=2,
        status="EMERGENCY",
        description="Confirmed scream",
    )

    assessment = engine.fuse(weapon_result=weapon, audio_record=audio_rec, current_time=curr_t)
    assert assessment.synergy.synergy_detected is True
    assert assessment.synergy.synergy_type == "WEAPON_AND_DISTRESS"
    assert assessment.status == "EMERGENCY"


# ============================================================================
# 5. False Positive Noise Suppression
# ============================================================================
def test_false_positive_suppression_audio_spike_calm_crowd():
    """Test 11: Acoustic spike with calm crowd kinematics is dampened."""
    engine = MultimodalFusionEngine(suppression_factor=0.50, smoothing_window=1)
    curr_t = time.time()

    # Calm video
    behavior = BehaviorAssessment(
        kinematic_score=0.10,
        deep_score=0.10,
        raw_score=0.10,
        smoothed_score=0.10,
        final_video_score=0.10,
        behavior_class="normal",
        status="NORMAL",
        consecutive_abnormal_frames=0,
        is_confirmed=False,
        description="",
        active_people=6,
        mean_speed=5.0,
    )
    # Unconfirmed acoustic transient (dropped object, mic glitch)
    audio_rec = AudioEventRecord(
        audio_event="crash",
        audio_confidence=0.75,
        audio_anomaly_score=0.80,
        audio_timestamp=curr_t,
        duration=1.0,
        is_confirmed=False,
        consecutive_abnormal_count=1,
        status="NORMAL",
        description="",
    )

    assessment = engine.fuse(behavior_assessment=behavior, audio_record=audio_rec, current_time=curr_t)
    assert assessment.false_positive_suppressed is True
    assert "calm kinematics" in assessment.suppression_reason.lower()
    # Dampened score should prevent an unconfirmed emergency alert
    assert assessment.multimodal_score < 0.40


def test_false_positive_suppression_visual_spike_quiet_audio():
    """Test 12: Isolated visual motion spike with quiet audio is dampened."""
    engine = MultimodalFusionEngine(suppression_factor=0.50, smoothing_window=1)
    curr_t = time.time()

    # Sudden visual motion spike (unconfirmed)
    behavior = BehaviorAssessment(
        kinematic_score=0.60,
        deep_score=0.55,
        raw_score=0.58,
        smoothed_score=0.58,
        final_video_score=0.58,
        behavior_class="rapid_movement",
        status="NORMAL",
        consecutive_abnormal_frames=1,
        is_confirmed=False,
        description="",
        active_people=3,
        mean_speed=20.0,
    )
    # Completely quiet audio
    audio_rec = AudioEventRecord(
        audio_event="normal",
        audio_confidence=0.95,
        audio_anomaly_score=0.0,
        audio_timestamp=curr_t,
        duration=1.0,
        is_confirmed=False,
        consecutive_abnormal_count=0,
        status="NORMAL",
        description="",
    )

    assessment = engine.fuse(behavior_assessment=behavior, audio_record=audio_rec, current_time=curr_t)
    assert assessment.false_positive_suppressed is True
    assert "visual motion spike" in assessment.suppression_reason.lower()


# ============================================================================
# 6. Temporal Stream Aligner & Hysteresis
# ============================================================================
def test_temporal_aligner_sync_within_tolerance():
    """Test 13: Temporal aligner correctly pairs video frames with recent audio windows."""
    aligner = TemporalStreamAligner(tolerance_seconds=1.5, stale_timeout_seconds=2.5)
    t0 = 100.0
    audio_rec = AudioEventRecord(
        audio_event="alarm",
        audio_confidence=0.8,
        audio_anomaly_score=0.7,
        audio_timestamp=t0,
        duration=1.0,
        is_confirmed=True,
        consecutive_abnormal_count=2,
        status="EMERGENCY",
        description="",
    )

    aligner.register_audio(audio_rec, timestamp=t0)
    # Video frame 0.5s later -> within tolerance (1.5s)
    rec, is_fresh = aligner.get_aligned_audio(reference_time=t0 + 0.5)
    assert is_fresh is True
    assert rec.audio_event == "alarm"


def test_temporal_aligner_stale_timeout():
    """Test 14: Temporal aligner detects stale stream after timeout."""
    aligner = TemporalStreamAligner(tolerance_seconds=1.0, stale_timeout_seconds=2.0)
    t0 = 100.0
    audio_rec = AudioEventRecord(
        audio_event="alarm",
        audio_confidence=0.8,
        audio_anomaly_score=0.7,
        audio_timestamp=t0,
        duration=1.0,
        is_confirmed=True,
        consecutive_abnormal_count=2,
        status="EMERGENCY",
        description="",
    )

    aligner.register_audio(audio_rec, timestamp=t0)
    # Video frame 3.0s later -> exceeds stale timeout (2.0s)
    rec, is_fresh = aligner.get_aligned_audio(reference_time=t0 + 3.0)
    assert is_fresh is False


def test_rolling_score_smoothing_stability():
    """Test 15: Rolling smoothing stabilizes multimodal score across consecutive frames."""
    engine = MultimodalFusionEngine(smoothing_window=5)

    # 4 frames of nominal score 0.0
    for _ in range(4):
        engine.fuse(current_time=1.0)

    # 1 instantaneous spike frame of score ~0.80
    audio_spike = AudioEventRecord(
        audio_event="alarm",
        audio_confidence=0.9,
        audio_anomaly_score=0.80,
        audio_timestamp=2.0,
        duration=1.0,
        is_confirmed=True,
        consecutive_abnormal_count=2,
        status="EMERGENCY",
        description="",
    )
    assessment = engine.fuse(audio_record=audio_spike, current_time=2.0)

    # Rolling window of 5 dampens the sudden spike: mean of [0, 0, 0, 0, ~0.32] < 0.15
    assert assessment.multimodal_score < 0.20


# ============================================================================
# 7. Output Schema Validation
# ============================================================================
def test_multimodal_assessment_output_schema():
    """Test 16: MultimodalAssessment.to_dict() matches required schema."""
    engine = MultimodalFusionEngine()
    assessment = engine.fuse(current_time=time.time())
    output = assessment.to_dict()

    required_keys = [
        "timestamp",
        "multimodal_score",
        "multimodal_confidence",
        "video_score",
        "audio_score",
        "status",
        "dominant_modality",
        "synergy_detected",
        "synergy_type",
        "boost_multiplier",
        "false_positive_suppressed",
        "suppression_reason",
        "evidence_breakdown",
        "description",
    ]

    for key in required_keys:
        assert key in output, f"Missing key '{key}' in MultimodalAssessment.to_dict()"

    assert isinstance(output["multimodal_score"], float)
    assert isinstance(output["multimodal_confidence"], float)
    assert isinstance(output["video_score"], float)
    assert isinstance(output["audio_score"], float)
    assert isinstance(output["status"], str)
    assert isinstance(output["synergy_detected"], bool)
    assert isinstance(output["evidence_breakdown"], dict)
