"""
Video Processing Module: Capture, Preprocessing, Detection, and Density.
"""

from src.video.video_capture import VideoSource
from src.video.preprocessing import FramePreprocessor
from src.video.person_detector import YOLOPersonDetector, DetectionResult
from src.video.crowd_density import CrowdDensityEstimator, DensityResult

__all__ = [
    "VideoSource",
    "FramePreprocessor",
    "YOLOPersonDetector",
    "DetectionResult",
    "CrowdDensityEstimator",
    "DensityResult",
]
