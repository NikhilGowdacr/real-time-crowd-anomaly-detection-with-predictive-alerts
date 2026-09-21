"""
Unit tests for Video Processing Module (Phase 2).
Tests configuration, video capture, preprocessing, YOLO detector structure, and crowd density.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.utils.config import load_config, get_default_config
from src.utils.metrics import FPSCounter, LatencyTracker
from src.video.video_capture import VideoSource, SyntheticCrowdGenerator
from src.video.preprocessing import FramePreprocessor
from src.video.crowd_density import CrowdDensityEstimator, DensityResult
from src.video.person_detector import DetectionResult, YOLOPersonDetector


def test_config_loading():
    """Verify that configuration loads correctly and contains all required sections."""
    cfg = load_config()
    assert "video" in cfg
    assert "detection" in cfg
    assert "density" in cfg
    assert "tracking" in cfg
    assert "decision" in cfg
    assert cfg["video"]["width"] == 1280
    assert cfg["density"]["frame_area_m2"] > 0


def test_synthetic_crowd_generator():
    """Verify synthetic frame generation dimensions and types."""
    gen = SyntheticCrowdGenerator(width=640, height=480, num_people=10)
    frame = gen.generate_frame()

    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert np.mean(frame) > 0


def test_video_source_synthetic():
    """Verify VideoSource works seamlessly with synthetic mode."""
    source = VideoSource(source="synthetic", width=640, height=360, fps=30)
    assert source.is_opened is True

    ret, frame = source.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (360, 640, 3)

    source.release()


def test_video_source_missing_file():
    """Verify that non-existent video files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        VideoSource(source="non_existent_surveillance_tape_9999.mp4")


def test_frame_preprocessor():
    """Verify frame resizing and CLAHE enhancement."""
    preprocessor = FramePreprocessor(target_width=640, target_height=360, enable_clahe=True)

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    processed = preprocessor.preprocess(dummy_frame)

    assert processed.shape == (360, 640, 3)
    assert processed.dtype == np.uint8


def test_crowd_density_estimator():
    """Verify crowd density calculation and spatial hotspot localization."""
    estimator = CrowdDensityEstimator(
        frame_area_m2=50.0,
        grid_rows=3,
        grid_cols=3,
        threshold_low=0.3,
        threshold_moderate=0.8,
        threshold_high=1.5,
    )

    # 1. Test Low Density (5 people in 50 m^2 = 0.1 persons/m^2 -> LOW)
    points_low = np.array([[100, 100], [200, 150], [300, 200], [400, 250], [500, 300]], dtype=float)
    res_low = estimator.estimate(points_low, frame_width=1280, frame_height=720)

    assert res_low.person_count == 5
    assert pytest.approx(res_low.density_per_m2, 0.01) == 0.10
    assert res_low.density_level == "LOW"
    assert res_low.is_overcrowded is False

    # 2. Test High / Critical Density (80 people concentrated in 50 m^2)
    # Put 30 people all in the top-left cell (x: 50, y: 50)
    points_high = np.full((30, 2), 50.0)
    res_high = estimator.estimate(points_high, frame_width=1280, frame_height=720)

    assert res_high.person_count == 30
    assert (0, 0) in res_high.hotspot_cells
    assert res_high.is_overcrowded is True

    # 3. Test Overlay Generation
    dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    overlay = estimator.draw_density_overlay(dummy_frame, res_high, show_grid=True)
    assert overlay.shape == dummy_frame.shape


def test_fps_counter_and_latency():
    """Verify metrics utilities."""
    fps = FPSCounter(window_size=5)
    for _ in range(5):
        fps.update()
    assert fps.current_fps >= 0.0

    tracker = LatencyTracker()
    tracker.start("test_stage")
    elapsed = tracker.stop("test_stage")
    assert elapsed >= 0.0
    assert tracker.get_average("test_stage") >= 0.0


def test_yolo_person_detector():
    """Verify YOLOPersonDetector initialization and inference output format."""
    detector = YOLOPersonDetector(
        model_path="models/detection/yolov8n.pt",
        confidence=0.25,
        device="cpu",
    )
    # Synthetic frame (black canvas with white patch)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (200, 100), (300, 380), (255, 255, 255), -1)

    result = detector.detect(dummy_frame)
    assert isinstance(result, DetectionResult)
    assert isinstance(result.person_count, int)
    assert result.person_count >= 0
    assert result.inference_ms > 0
    assert len(result.boxes.shape) == 2
    assert len(result.centers.shape) == 2
