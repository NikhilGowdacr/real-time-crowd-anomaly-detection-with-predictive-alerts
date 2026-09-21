"""
Temporal Frame Buffer Module.
Maintains a sliding FIFO buffer of subsampled, preprocessed video frames
to prepare spatiotemporal sequences for Vision Transformer (ViT / Swin) inference.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from src.utils.logger import setup_logger

logger = setup_logger("temporal_buffer")


class TemporalFrameBuffer:
    """
    Sliding window buffer for collecting temporal sequences of video frames.
    Subsamples frames at a configurable stride (sample_rate) and resizes them
    to standard vision transformer dimensions (224x224).
    """

    def __init__(
        self,
        sequence_length: int = 16,
        sample_rate: int = 4,
        target_size: Tuple[int, int] = (224, 224),
        normalize_imagenet: bool = True,
    ):
        """
        Initialize the temporal frame buffer.

        Args:
            sequence_length: Number of sampled frames required for transformer inference.
            sample_rate: Stride between sampled frames (e.g., 4 = take 1 frame every 4 frames).
            target_size: Dimensions (width, height) to resize frames for transformer input.
            normalize_imagenet: If True, applies ImageNet mean and std normalization.
        """
        self.sequence_length = max(2, int(sequence_length))
        self.sample_rate = max(1, int(sample_rate))
        self.target_size = target_size
        self.normalize_imagenet = normalize_imagenet

        self.buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.frame_counter: int = 0

        # Standard ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        logger.info(
            f"TemporalFrameBuffer initialized: seq_len={self.sequence_length}, "
            f"sample_rate={self.sample_rate}, target_size={self.target_size}."
        )

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Processes an incoming video frame and adds it to the buffer if sample stride matches.

        Args:
            frame: OpenCV BGR frame (H, W, 3).

        Returns:
            bool: True if the frame was sampled and stored, False if skipped by sample_rate.
        """
        self.frame_counter += 1

        # Only sample every N-th frame
        if (self.frame_counter % self.sample_rate) != 0:
            return False

        # Preprocess frame: resize and convert BGR -> RGB
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Scale to [0.0, 1.0]
        norm_frame = rgb.astype(np.float32) / 255.0

        if self.normalize_imagenet:
            norm_frame = (norm_frame - self.mean) / self.std

        self.buffer.append(norm_frame)
        return True

    def is_ready(self) -> bool:
        """Returns True if the buffer contains a complete sequence of sampled frames."""
        return len(self.buffer) >= self.sequence_length

    @property
    def current_length(self) -> int:
        """Returns count of sampled frames currently in buffer."""
        return len(self.buffer)

    def get_sequence_numpy(self) -> Optional[np.ndarray]:
        """
        Returns buffered frames as a NumPy array of shape (sequence_length, H, W, 3).
        Returns None if buffer is not yet full.
        """
        if not self.is_ready():
            return None
        return np.stack(list(self.buffer), axis=0)

    def get_torch_tensor(self, device: str = "cpu") -> Optional[torch.Tensor]:
        """
        Converts buffered sequence into a PyTorch tensor formatted for Vision Transformers:
        Shape: (T, C, H, W) where T=sequence_length, C=3, H=224, W=224.

        Returns None if buffer is not yet full.
        """
        seq = self.get_sequence_numpy()
        if seq is None:
            return None

        # (T, H, W, C) -> (T, C, H, W)
        tensor = torch.from_numpy(seq).permute(0, 3, 1, 2).to(device)
        return tensor

    def clear(self) -> None:
        """Clears all stored frames from the buffer and resets counter."""
        self.buffer.clear()
        self.frame_counter = 0
