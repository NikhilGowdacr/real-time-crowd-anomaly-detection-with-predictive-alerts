"""
Unit tests for Weapon Detection, Fight Detection, Pose Kinematics, and Safety Fusion.
Verifies unconfigured model handling, spatial track association, temporal sliding windows,
false-positive suppression, score fusion weighting, and alert cooldowns.
"""

import time
import numpy as np
import pytest

from src.video.crowd_tracker import TrackedPerson, ByteTrackCrowdTracker
from src.video.person_detector import DetectionResult
from src.video.crowd_density import DensityResult
from src.safety.weapon_detector import (
    WeaponDetector,
    DetectedObject,
    WeaponDetectionResult,
)
from src.safety.pose_analyzer import PoseMovementAnalyzer, KinematicFeatures, InteractingPair
from src.safety.fight_detector import FightDetector, FightDetectionResult
from src.safety.safety_fusion import (
    SafetyFusionEngine,
    SafetyAssessment,
    SafetyStatus,
    SafetyEvent,
)


# ==============================================================================
# 1. Weapon Detector Tests
# ==============================================================================

def test_weapon_detector_initialization_and_missing_model():
    """Verify weapon detector handles non-existent model files gracefully without crashing."""
    detector = WeaponDetector(
        model_path="models/detection/non_existent_weapon_model_9999.pt",
        enabled=True,
    )
    assert detector.is_configured is False
    assert "not found" in detector.status_message.lower() or "unconfigured" in detector.status_message.lower()

    # Inference should return empty result with clear status
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame)
    assert res.is_configured is False
    assert len(res.detected_objects) == 0
    assert res.weapon_score == 0.0


def test_weapon_detector_disabled():
    """Verify disabled detector returns safe zeros immediately."""
    detector = WeaponDetector(enabled=False)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame)
    assert res.weapon_score == 0.0
    assert len(res.detected_objects) == 0


def test_weapon_detection_track_association_and_uncertainty():
    """Verify spatial association between detected object and nearby person."""
    detector = WeaponDetector(enabled=True)

    # Simulate tracked person at [100, 100, 200, 300] (center ~ 150, 200)
    person_near = TrackedPerson(
        track_id=10,
        current_bbox=np.array([100, 100, 200, 300], dtype=np.float32),
        history=[(150.0, 200.0)],
    )

    # Object 1: Right next to person (cx: 160, cy: 220) -> should associate
    obj_near = DetectedObject(
        object_class="FIREARM",
        confidence=0.88,
        bbox=np.array([150, 210, 170, 230], dtype=np.float32),
        timestamp=time.time(),
        center=(160.0, 220.0),
    )

    # Object 2: Far away in corner (cx: 600, cy: 400) -> uncertain/isolated
    obj_far = DetectedObject(
        object_class="KNIFE",
        confidence=0.75,
        bbox=np.array([590, 390, 610, 410], dtype=np.float32),
        timestamp=time.time(),
        center=(600.0, 400.0),
    )

    detector._associate_with_tracks([obj_near, obj_far], [person_near])

    assert obj_near.associated_track_id == 10
    assert obj_near.association_status == "associated"
    assert obj_near.association_confidence > 0.50

    # Far object should NOT be falsely associated
    assert obj_far.associated_track_id is None or obj_far.association_status in ("uncertain", "isolated")


# ==============================================================================
# 2. Pose & Kinematic Analyzer Tests
# ==============================================================================

def test_pose_analyzer_empty_tracks():
    """Verify kinematic analyzer handles zero tracks safely."""
    analyzer = PoseMovementAnalyzer()
    feat = analyzer.extract_features([])
    assert feat.active_track_count == 0
    assert feat.abnormal_movement_score == 0.0
    assert len(feat.interacting_pairs) == 0


def test_pose_analyzer_pair_convergence_and_reciprocal_motion():
    """Verify detection of close proximity and opposing velocity vectors."""
    analyzer = PoseMovementAnalyzer(proximity_threshold=100.0)

    # Person 1 moving right: vx = +10
    p1 = TrackedPerson(
        track_id=1,
        current_bbox=np.array([50, 100, 90, 200], dtype=np.float32),
        history=[(60.0, 150.0), (70.0, 150.0)],
        speed=10.0,
        velocity=(10.0, 0.0),
        acceleration=2.0,
    )

    # Person 2 moving left directly at P1: vx = -10, distance ~ 20px
    p2 = TrackedPerson(
        track_id=2,
        current_bbox=np.array([95, 100, 135, 200], dtype=np.float32),
        history=[(110.0, 150.0), (90.0, 150.0)],
        speed=10.0,
        velocity=(-10.0, 0.0),
        acceleration=2.0,
    )

    feat = analyzer.extract_features([p1, p2])
    assert feat.active_track_count == 2
    assert len(feat.interacting_pairs) == 1
    pair = feat.interacting_pairs[0]
    assert pair.distance <= 100.0
    assert pair.reciprocal_motion_score > 0.0
    assert feat.abnormal_movement_score > 0.0


# ==============================================================================
# 3. Fight Detector Tests (Temporal Accumulation & False-Positive Prevention)
# ==============================================================================

def test_fight_detector_insufficient_history():
    """Verify fight alert is NEVER triggered on single frame or insufficient frames."""
    detector = FightDetector(temporal_window=20, confirmation_frames=5)

    # Frame with artificial high abnormal score
    mock_feat = KinematicFeatures(
        active_track_count=2,
        mean_speed=20.0,
        max_individual_speed=25.0,
        max_individual_accel=15.0,
        interacting_pairs=[],
        abnormal_movement_score=0.95,
    )

    res = detector.update(mock_feat, tracked_persons=[])
    # Must NOT trigger an emergency alert on frame 1
    assert res.is_fight is False
    assert res.status == "NORMAL"
    assert "confirmed" in res.reason.lower() or "confirmation" in res.reason.lower()


def test_fight_detector_temporal_confirmation():
    """Verify fight alert triggers only after multiple consecutive abnormal frames."""
    detector = FightDetector(
        temporal_window=10,
        suspicious_threshold=0.45,
        emergency_threshold=0.70,
        confirmation_frames=4,
    )

    # Interacting violent pair
    violent_pair = InteractingPair(
        track_id_1=1,
        track_id_2=2,
        distance=40.0,
        relative_speed=15.0,
        convergence_rate=8.0,
        reciprocal_motion_score=0.90,
        center_point=(100.0, 100.0),
        union_bbox=np.array([50, 50, 150, 150], dtype=np.float32),
    )

    violent_feat = KinematicFeatures(
        active_track_count=2,
        mean_speed=18.0,
        max_individual_speed=22.0,
        max_individual_accel=14.0,
        interacting_pairs=[violent_pair],
        abnormal_movement_score=0.85,
    )

    # Feed 2 frames -> should remain unconfirmed
    res1 = detector.update(violent_feat, tracked_persons=[])
    res2 = detector.update(violent_feat, tracked_persons=[])
    assert res1.is_fight is False
    assert res2.is_fight is False

    # Feed up to confirmation frames (4 frames total)
    detector.update(violent_feat, tracked_persons=[])
    res4 = detector.update(violent_feat, tracked_persons=[])

    # Now it should escalate after consecutive temporal confirmation
    assert res4.consecutive_abnormal_frames >= 4
    assert res4.status in ("SUSPICIOUS", "EMERGENCY")
    assert res4.fight_score > 0.40


def test_false_positive_single_spike_suppression():
    """Verify single-frame anomalous spike returns to normal when not sustained."""
    detector = FightDetector(confirmation_frames=5)

    normal_feat = KinematicFeatures(
        active_track_count=2,
        mean_speed=3.0,
        max_individual_speed=4.0,
        max_individual_accel=1.0,
        interacting_pairs=[],
        abnormal_movement_score=0.10,
    )

    # Normal history
    for _ in range(5):
        detector.update(normal_feat, tracked_persons=[])

    # One random spike (e.g. tracking glitch)
    spike_feat = KinematicFeatures(
        active_track_count=2,
        mean_speed=30.0,
        max_individual_speed=35.0,
        max_individual_accel=20.0,
        interacting_pairs=[],
        abnormal_movement_score=0.99,
    )
    res_spike = detector.update(spike_feat, tracked_persons=[])
    assert res_spike.is_fight is False
    assert res_spike.status == "NORMAL"

    # Immediately normal again
    res_post = detector.update(normal_feat, tracked_persons=[])
    assert res_post.is_fight is False
    assert res_post.status == "NORMAL"


# ==============================================================================
# 4. Multi-Evidence Safety Fusion Tests
# ==============================================================================

def test_safety_fusion_weighting():
    """Verify weighted linear fusion: crowd*w1 + fight*w2 + weapon*w3."""
    engine = SafetyFusionEngine(
        crowd_weight=0.20,
        fight_weight=0.40,
        weapon_weight=0.40,
        normal_threshold=0.35,
        emergency_threshold=0.70,
    )

    dummy_density = DensityResult(
        person_count=10,
        density_per_m2=0.2,
        density_score=0.50,
        density_level="MODERATE",
        grid_counts=np.zeros((3, 3), dtype=int),
        grid_densities=np.zeros((3, 3)),
        hotspot_cells=[],
        is_overcrowded=False,
    )

    dummy_kinematics = KinematicFeatures(
        active_track_count=2,
        mean_speed=5.0,
        max_individual_speed=5.0,
        max_individual_accel=1.0,
        interacting_pairs=[],
        abnormal_movement_score=0.20,
    )

    dummy_fight = FightDetectionResult(
        is_fight=False,
        is_suspicious=False,
        fight_score=0.20,
        involved_track_ids=[],
        interaction_bbox=None,
        status="NORMAL",
        consecutive_abnormal_frames=0,
        reason="Normal",
    )

    dummy_weapon = WeaponDetectionResult(
        is_configured=True,
        status_message="OK",
        detected_objects=[],
        weapon_score=0.0,
    )

    assessment = engine.evaluate(dummy_density, dummy_kinematics, dummy_fight, dummy_weapon)
    # Expected base score: 0.20*0.50 + 0.40*0.20 + 0.40*0.0 = 0.10 + 0.08 = 0.18
    assert pytest.approx(assessment.safety_score, 0.02) == 0.18
    assert assessment.status == SafetyStatus.NORMAL


def test_weapon_fight_correlation_synergy():
    """Verify that dangerous object spatially linked with active fight triggers priority boost."""
    engine = SafetyFusionEngine(
        crowd_weight=0.20,
        fight_weight=0.40,
        weapon_weight=0.40,
        correlation_multiplier=1.30,
    )

    dummy_density = DensityResult(
        person_count=5, density_per_m2=0.1, density_score=0.2,
        density_level="LOW", grid_counts=np.zeros((3, 3), dtype=int),
        grid_densities=np.zeros((3, 3)), hotspot_cells=[], is_overcrowded=False,
    )
    dummy_kinematics = KinematicFeatures(
        active_track_count=2, mean_speed=15.0, max_individual_speed=15.0,
        max_individual_accel=8.0, interacting_pairs=[], abnormal_movement_score=0.5,
    )

    # Active fight with track 5
    fight_res = FightDetectionResult(
        is_fight=True, is_suspicious=False, fight_score=0.75,
        involved_track_ids=[5, 6], interaction_bbox=np.array([100, 100, 200, 200]),
        status="EMERGENCY", consecutive_abnormal_frames=5, reason="Fight",
    )

    # Weapon linked to track 5
    correlated_obj = DetectedObject(
        object_class="FIREARM", confidence=0.85, bbox=np.array([110, 110, 130, 130]),
        timestamp=time.time(), center=(120.0, 120.0), associated_track_id=5,
        association_confidence=0.90, association_status="associated",
    )
    weapon_res = WeaponDetectionResult(
        is_configured=True, status_message="OK",
        detected_objects=[correlated_obj], weapon_score=0.85,
    )

    assessment = engine.evaluate(dummy_density, dummy_kinematics, fight_res, weapon_res)
    assert assessment.is_weapon_fight_correlated is True
    assert assessment.status == SafetyStatus.EMERGENCY
    assert len(assessment.active_events) > 0
    event = assessment.active_events[0]
    assert event.event_type == "POTENTIAL_WEAPON_NEAR_STRUGGLE"
    assert event.track_id == 5


def test_alert_cooldown():
    """Verify duplicate alerts are suppressed during cooldown window."""
    engine = SafetyFusionEngine(cooldown_seconds=10.0)

    dummy_density = DensityResult(
        person_count=5, density_per_m2=0.1, density_score=0.1,
        density_level="LOW", grid_counts=np.zeros((3, 3), dtype=int),
        grid_densities=np.zeros((3, 3)), hotspot_cells=[], is_overcrowded=False,
    )
    dummy_kinematics = KinematicFeatures(
        active_track_count=2, mean_speed=20.0, max_individual_speed=20.0,
        max_individual_accel=10.0, interacting_pairs=[], abnormal_movement_score=0.6,
    )
    fight_res = FightDetectionResult(
        is_fight=True, is_suspicious=False, fight_score=0.85,
        involved_track_ids=[1, 2], interaction_bbox=np.array([0, 0, 50, 50]),
        status="EMERGENCY", consecutive_abnormal_frames=5, reason="Fight",
    )
    weapon_res = WeaponDetectionResult(is_configured=True, status_message="OK", detected_objects=[], weapon_score=0.0)

    # First trigger at t = 100.0
    assess1 = engine.evaluate(dummy_density, dummy_kinematics, fight_res, weapon_res, timestamp=100.0)
    assert len(assess1.active_events) == 1
    assert assess1.cooldown_active is False

    # Second trigger at t = 103.0 (within 10s cooldown)
    assess2 = engine.evaluate(dummy_density, dummy_kinematics, fight_res, weapon_res, timestamp=103.0)
    assert len(assess2.active_events) == 0
    assert assess2.cooldown_active is True

    # Third trigger at t = 112.0 (after 10s cooldown)
    assess3 = engine.evaluate(dummy_density, dummy_kinematics, fight_res, weapon_res, timestamp=112.0)
    assert len(assess3.active_events) == 1
    assert assess3.cooldown_active is False
