"""
==============================================================================
Temporal Stream Aligner Module (Phase 6).
Synchronizes asynchronous multi-rate sensor streams (30 FPS Video vs. 2 Hz Audio)
and detects stale or dropped sensor streams.
==============================================================================
"""

from collections import deque
import time
from typing import Any, Deque, Dict, Optional, Tuple

from src.audio.audio_anomaly import AudioEventRecord
from src.utils.logger import setup_logger

logger = setup_logger("temporal_aligner")


class TemporalStreamAligner:
    """
    Synchronizes asynchronous video frames and audio analysis windows.
    Maintains a sliding temporal window to pair high-rate video telemetry (~30 Hz)
    with the most recent acoustic assessment (~2 Hz) within an acceptable temporal tolerance.
    """

    def __init__(
        self,
        tolerance_seconds: float = 1.5,
        stale_timeout_seconds: float = 2.5,
        buffer_size: int = 30,
    ):
        self.tolerance_seconds = float(tolerance_seconds)
        self.stale_timeout_seconds = float(stale_timeout_seconds)
        self.buffer_size = max(5, int(buffer_size))

        self._audio_buffer: Deque[AudioEventRecord] = deque(maxlen=self.buffer_size)
        self._video_timestamps: Deque[float] = deque(maxlen=self.buffer_size)

        self.latest_audio: Optional[AudioEventRecord] = None
        self.latest_audio_time: float = 0.0
        self.latest_video_time: float = 0.0

        logger.info(
            f"TemporalStreamAligner initialized: tolerance={self.tolerance_seconds}s, "
            f"stale_timeout={self.stale_timeout_seconds}s, buffer_size={self.buffer_size}."
        )

    def register_audio(self, audio_record: AudioEventRecord, timestamp: Optional[float] = None) -> None:
        """
        Registers a newly processed acoustic analysis window.
        """
        curr_t = timestamp if timestamp is not None else (audio_record.audio_timestamp or time.time())
        self.latest_audio = audio_record
        self.latest_audio_time = curr_t
        self._audio_buffer.append(audio_record)

    def register_video(self, timestamp: Optional[float] = None) -> None:
        """
        Registers a newly processed video frame timestamp.
        """
        curr_t = timestamp if timestamp is not None else time.time()
        self.latest_video_time = curr_t
        self._video_timestamps.append(curr_t)

    def get_aligned_audio(self, reference_time: Optional[float] = None) -> Tuple[Optional[AudioEventRecord], bool]:
        """
        Retrieves the most recent audio record aligned with the given reference time.

        Args:
            reference_time: Reference timestamp (e.g. current video frame time).

        Returns:
            Tuple of (AudioEventRecord or None, is_valid_and_fresh: bool).
        """
        ref_t = reference_time if reference_time is not None else time.time()

        if self.latest_audio is None:
            return None, False

        time_delta = abs(ref_t - self.latest_audio_time)

        # Check for stream timeout / stale audio
        if time_delta > self.stale_timeout_seconds:
            return self.latest_audio, False

        # Within tolerance window -> fresh and aligned
        is_aligned = time_delta <= self.tolerance_seconds
        return self.latest_audio, is_aligned

    def is_audio_active(self, current_time: Optional[float] = None) -> bool:
        """Returns True if an audio record has been registered within the stale timeout window."""
        curr_t = current_time if current_time is not None else time.time()
        if self.latest_audio is None:
            return False
        return (curr_t - self.latest_audio_time) <= self.stale_timeout_seconds

    def is_video_active(self, current_time: Optional[float] = None) -> bool:
        """Returns True if a video frame has been registered within the stale timeout window."""
        curr_t = current_time if current_time is not None else time.time()
        if self.latest_video_time == 0.0:
            return False
        return (curr_t - self.latest_video_time) <= self.stale_timeout_seconds

    def is_bimodal_active(self, current_time: Optional[float] = None) -> bool:
        """Returns True if both video and audio streams are concurrently active and fresh."""
        return self.is_video_active(current_time) and self.is_audio_active(current_time)

    def reset(self) -> None:
        """Clears all stored stream histories and timestamps."""
        self._audio_buffer.clear()
        self._video_timestamps.clear()
        self.latest_audio = None
        self.latest_audio_time = 0.0
        self.latest_video_time = 0.0
