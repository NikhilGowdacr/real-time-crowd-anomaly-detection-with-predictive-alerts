"""
Unit tests for Phase 4: Advanced Video Behavioral Anomaly Detection.
Verifies temporal frame buffer, crowd kinematics, ViT/Swin model interfaces,
rolling temporal score smoothing, multi-frame confirmation, and behavior fusion.
"""

import numpy as np
import pytest
import torch

from src.video.crowd_tracker import TrackedPerson
from src.behavior.temporal_buffer import TemporalFrameBuffer
from src.behavior.behavior_features import CrowdBehaviorAnalyzer, CrowdBehaviorMetrics
from src.behavior.behavior_model import BehaviorModel, BehaviorPrediction
from src.behavior.behavior_anomaly import BehaviorAnomalyDetector, BehaviorAssessment


# ==============================================================================
# 1. Temporal Frame Buffer Tests
# ==============================================================================

def test_temporal_buffer_initialization_and_empty():
    """Verify temporal frame buffer initializes with correct bounds and empty state."""
    buffer = TemporalFrameBuffer(sequence_length=8, sample_rate=2, target_size=(224, 224))
    assert buffer.sequence_length == 8
    assert buffer.sample_rate == 2
    assert buffer.current_length == 0
    assert buffer.is_ready() is False
    assert buffer.get_sequence_numpy() is None
    assert buffer.get_torch_tensor() is None


def test_temporal_buffer_frame_insertion_and_sample_rate():
    """Verify frames are subsampled according to sample_rate."""
    buffer = TemporalFrameBuffer(sequence_length=4, sample_rate=3, target_size=(224, 224))
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: not sampled (1 % 3 != 0)
    assert buffer.add_frame(dummy_frame) is False
    assert buffer.current_length == 0

    # Frame 2: not sampled (2 % 3 != 0)
    assert buffer.add_frame(dummy_frame) is False
    assert buffer.current_length == 0

    # Frame 3: sampled! (3 % 3 == 0)
    assert buffer.add_frame(dummy_frame) is True
    assert buffer.current_length == 1


def test_temporal_buffer_max_length_and_ready():
    """Verify buffer caps at sequence_length and returns valid PyTorch tensors."""
    seq_len = 4
    buffer = TemporalFrameBuffer(sequence_length=seq_len, sample_rate=1, target_size=(112, 112))
    dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 200

    for i in range(seq_len + 3):
        buffer.add_frame(dummy_frame)

    assert buffer.is_ready() is True
    assert buffer.current_length == seq_len

    # NumPy sequence shape: (T, H, W, 3)
    np_seq = buffer.get_sequence_numpy()
    assert np_seq is not None
    assert np_seq.shape == (seq_len, 112, 112, 3)

    # PyTorch sequence shape: (T, C, H, W)
    t_seq = buffer.get_torch_tensor(device="cpu")
    assert t_seq is not None
    assert t_seq.shape == (seq_len, 3, 112, 112)
    assert isinstance(t_seq, torch.Tensor)


def test_temporal_buffer_clear():
    """Verify clearing resets buffer cleanly."""
    buffer = TemporalFrameBuffer(sequence_length=3, sample_rate=1)
    dummy = np.zeros((50, 50, 3), dtype=np.uint8)
    buffer.add_frame(dummy)
    buffer.add_frame(dummy)
    assert buffer.current_length == 2

    buffer.clear()
    assert buffer.current_length == 0
    assert buffer.is_ready() is False


# ==============================================================================
# 2. Crowd Kinematics Feature Tests
# ==============================================================================

def test_crowd_behavior_analyzer_empty():
    """Verify crowd behavior analyzer handles empty tracks safely."""
    analyzer = CrowdBehaviorAnalyzer()
    metrics = analyzer.analyze([])
    assert metrics.active_people == 0
    assert metrics.kinematic_score == 0.0
    assert metrics.candidate_class == "normal"


def test_crowd_dispersion_and_scattering():
    """Verify crowd dispersion rate calculation when people expand outward."""
    analyzer = CrowdBehaviorAnalyzer(speed_threshold=15.0, dispersion_threshold=2.0)

    # Step 1: People initially clustered closely together
    p1 = TrackedPerson(track_id=1, current_bbox=np.zeros(4), history=[(100.0, 100.0)], speed=18.0)
    p2 = TrackedPerson(track_id=2, current_bbox=np.zeros(4), history=[(110.0, 100.0)], speed=18.0)
    analyzer.analyze([p1, p2])

    # Step 2: People scatter outwards rapidly (distance expands by 80px)
    p1_next = TrackedPerson(track_id=1, current_bbox=np.zeros(4), history=[(50.0, 100.0)], speed=22.0)
    p2_next = TrackedPerson(track_id=2, current_bbox=np.zeros(4), history=[(140.0, 100.0)], speed=22.0)
    metrics = analyzer.analyze([p1_next, p2_next])

    assert metrics.dispersion_rate > 0.0
    assert metrics.rapid_mover_ratio >= 0.50
    assert metrics.candidate_class == "scattering"
    assert metrics.kinematic_score > 0.30


def test_directional_variance_chaos():
    """Verify circular variance identifies opposing/conflicting directional vectors."""
    analyzer = CrowdBehaviorAnalyzer()

    # Two persons heading in exact opposite directions (0 deg vs 180 deg) at high speed
    p_east = TrackedPerson(track_id=1, current_bbox=np.zeros(4), history=[(100.0, 100.0)], speed=12.0, direction_angle=0.0)
    p_west = TrackedPerson(track_id=2, current_bbox=np.zeros(4), history=[(200.0, 100.0)], speed=12.0, direction_angle=180.0)

    metrics = analyzer.analyze([p_east, p_west])
    # Opposing vectors yield maximum circular variance (1.0)
    assert metrics.direction_variance > 0.80


# ==============================================================================
# 3. Behavior Model Interface Tests
# ==============================================================================

def test_behavior_model_initialization_and_missing_custom():
    """Verify behavior model handles non-existent custom path by falling back to feature extractor."""
    model = BehaviorModel(
        model_type="swin",
        model_path="models/behavior/non_existent_weights_9999.pth",
        enabled=True,
    )
    assert model.is_custom_model is False
    assert "feature extraction mode" in model.status_message.lower()


def test_behavior_model_predict_format():
    """Verify BehaviorModel.predict returns standard BehaviorPrediction structure."""
    model = BehaviorModel(model_type="swin", enabled=True)
    # Sequence of 4 frames: (T=4, C=3, H=224, W=224)
    dummy_seq = torch.randn(4, 3, 224, 224)

    pred = model.predict(dummy_seq)
    assert isinstance(pred, BehaviorPrediction)
    assert 0.0 <= pred.behavior_score <= 1.0
    assert pred.behavior_class in BehaviorModel.BEHAVIOR_CLASSES
    assert pred.inference_ms >= 0.0


def test_behavior_model_disabled():
    """Verify disabled behavior model returns safe zeros."""
    model = BehaviorModel(enabled=False)
    pred = model.predict(None)
    assert pred.behavior_score == 0.0
    assert pred.behavior_class == "normal"


# ==============================================================================
# 4. Anomaly Fusion, Temporal Smoothing, and Confirmation Tests
# ==============================================================================

def test_behavior_score_smoothing():
    """Verify rolling average smooths sudden single-frame spikes."""
    detector = BehaviorAnomalyDetector(
        kinematic_weight=0.50,
        deep_model_weight=0.50,
        smoothing_window=5,
        confirmation_frames=3,
    )

    low_metrics = CrowdBehaviorMetrics(
        active_people=5, mean_velocity=3.0, velocity_variance=1.0, direction_variance=0.1,
        rapid_mover_ratio=0.0, direction_change_ratio=0.0, dispersion_rate=0.0,
        convergence_rate=0.0, kinematic_score=0.10, candidate_class="normal", description="Nominal",
    )

    # Prime history with 4 normal frames
    for _ in range(4):
        detector.evaluate(low_metrics, deep_prediction=None)

    # Spike frame with score 1.0
    spike_metrics = CrowdBehaviorMetrics(
        active_people=5, mean_velocity=25.0, velocity_variance=10.0, direction_variance=0.8,
        rapid_mover_ratio=1.0, direction_change_ratio=1.0, dispersion_rate=5.0,
        convergence_rate=0.0, kinematic_score=1.00, candidate_class="scattering", description="Spike",
    )
    assess = detector.evaluate(spike_metrics, deep_prediction=None)

    # Raw score is 1.0, but smoothed score across window (0.1, 0.1, 0.1, 0.1, 1.0) is 0.28!
    assert assess.raw_score == 1.0
    assert assess.smoothed_score < 0.40
    # Must NOT trigger an emergency alert on a single spike
    assert assess.status == "NORMAL"
    assert assess.is_confirmed is False


def test_behavior_temporal_confirmation():
    """Verify anomaly status escalates only after consecutive confirmed frames."""
    detector = BehaviorAnomalyDetector(
        kinematic_weight=1.0,
        deep_model_weight=0.0,
        normal_threshold=0.35,
        emergency_threshold=0.70,
        confirmation_frames=3,
        smoothing_window=3,
    )

    high_metrics = CrowdBehaviorMetrics(
        active_people=10, mean_velocity=20.0, velocity_variance=8.0, direction_variance=0.8,
        rapid_mover_ratio=0.8, direction_change_ratio=0.5, dispersion_rate=4.0,
        convergence_rate=0.0, kinematic_score=0.85, candidate_class="scattering", description="Scattering",
    )

    # Frame 1: abnormal, count=1, unconfirmed
    a1 = detector.evaluate(high_metrics)
    assert a1.consecutive_abnormal_frames == 1
    assert a1.is_confirmed is False
    assert a1.status == "NORMAL"

    # Frame 2: abnormal, count=2, unconfirmed
    a2 = detector.evaluate(high_metrics)
    assert a2.consecutive_abnormal_frames == 2
    assert a2.is_confirmed is False

    # Frame 3: abnormal, count=3 >= confirmation_frames (3) -> Confirmed!
    a3 = detector.evaluate(high_metrics)
    assert a3.consecutive_abnormal_frames >= 3
    assert a3.is_confirmed is True
    assert a3.status in ("SUSPICIOUS", "EMERGENCY")
    assert a3.final_video_score > 0.50


def test_behavior_fusion_weighting():
    """Verify behavior fusion strictly respects kinematic_weight and deep_model_weight."""
    detector = BehaviorAnomalyDetector(
        kinematic_weight=0.40,
        deep_model_weight=0.60,
        smoothing_window=1,
        confirmation_frames=1,
    )

    metrics = CrowdBehaviorMetrics(
        active_people=5, mean_velocity=10.0, velocity_variance=2.0, direction_variance=0.2,
        rapid_mover_ratio=0.2, direction_change_ratio=0.1, dispersion_rate=0.0,
        convergence_rate=0.0, kinematic_score=0.50, candidate_class="normal", description="Test",
    )

    pred = BehaviorPrediction(
        behavior_score=0.80,
        behavior_class="scattering",
        confidence=0.80,
        is_custom_model=True,
        status_message="OK",
    )

    # Expected: 0.40 * 0.50 + 0.60 * 0.80 = 0.20 + 0.48 = 0.68
    assess = detector.evaluate(metrics, pred)
    assert pytest.approx(assess.raw_score, 0.01) == 0.68
    assert pytest.approx(assess.smoothed_score, 0.01) == 0.68
