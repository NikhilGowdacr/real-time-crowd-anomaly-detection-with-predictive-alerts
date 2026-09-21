"""
Frame Preprocessing Module.
Provides frame resizing, aspect-ratio preservation, normalization, and contrast enhancement.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class FramePreprocessor:
    """
    Standard surveillance frame preprocessor.
    Prepares raw video frames for YOLO person detection and behavioral feature extraction.
    """

    def __init__(
        self,
        target_width: int = 1280,
        target_height: int = 720,
        enable_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.enable_clahe = enable_clahe

        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_grid_size,
            )
        else:
            self.clahe = None

    def resize(self, frame: np.ndarray, keep_aspect_ratio: bool = False) -> np.ndarray:
        """
        Resizes frame to target dimensions.

        Args:
            frame: Input BGR frame.
            keep_aspect_ratio: If True, uses letterboxing (padding with black borders).

        Returns:
            Resized image.
        """
        h, w = frame.shape[:2]
        if w == self.target_width and h == self.target_height:
            return frame

        if not keep_aspect_ratio:
            return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        # Aspect ratio preserving letterbox
        scale = min(self.target_width / w, self.target_height / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        top = (self.target_height - nh) // 2
        left = (self.target_width - nw) // 2
        padded[top : top + nh, left : left + nw] = resized
        return padded

    def enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE on luminance (L) channel in LAB color space to boost low-light details.
        """
        if self.clahe is None:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        merged = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for a surveillance frame.
        """
        processed = self.resize(frame, keep_aspect_ratio=False)
        if self.enable_clahe:
            processed = self.enhance_contrast(processed)
        return processed
