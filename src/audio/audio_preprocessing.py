"""
Audio Preprocessing Module.
Provides mono downmixing, DC offset removal, RMS energy calculation,
amplitude normalization, and clipping protection for acoustic surveillance streams.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("audio_preprocessing")


@dataclass
class PreprocessedAudio:
    """
    Container for preprocessed audio samples and basic acoustic telemetry.
    """
    samples: np.ndarray                       # 1D float32 array in range [-1.0, 1.0]
    rms: float                                # Root-Mean-Square energy [0.0 - 1.0]
    dbfs: float                               # Decibels relative to Full Scale (-inf to 0 dB)
    peak: float                               # Maximum absolute peak amplitude [0.0 - 1.0]
    is_valid: bool                            # False if audio was corrupt, NaN, or silent
    sample_count: int


class AudioPreprocessor:
    """
    Prepares raw audio chunks for Mel Spectrogram generation and acoustic event classification.
    Preserves sudden emergency amplitude spikes while cleaning DC rumble and preventing digital clipping.
    """

    def __init__(
        self,
        target_peak: float = 0.95,
        enable_dc_removal: bool = True,
        clipping_threshold: float = 1.0,
    ):
        self.target_peak = float(target_peak)
        self.enable_dc_removal = enable_dc_removal
        self.clipping_threshold = float(clipping_threshold)

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Converts multi-channel audio to single-channel mono."""
        if audio is None or audio.size == 0:
            return np.empty((0,), dtype=np.float32)
        if audio.ndim == 1:
            return audio.astype(np.float32)
        return np.mean(audio, axis=-1).astype(np.float32)

    def sanitize(self, audio: np.ndarray) -> np.ndarray:
        """Replaces NaNs and Infinities with zero."""
        if not np.all(np.isfinite(audio)):
            return np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        return audio

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Subtracts mean to eliminate DC bias and microphone baseline drift."""
        if audio.size == 0:
            return audio
        mean_val = np.mean(audio)
        return audio - mean_val

    def calculate_rms(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates RMS energy, dBFS, and peak amplitude.

        Returns:
            Tuple of (rms [0-1], dbfs [dB], peak [0-1]).
        """
        if audio.size == 0:
            return 0.0, -100.0, 0.0

        peak = float(np.max(np.abs(audio)))
        mean_square = float(np.mean(audio ** 2))
        rms = math.sqrt(max(0.0, mean_square))

        # dBFS = 20 * log10(RMS + epsilon)
        dbfs = 20.0 * math.log10(rms + 1e-7)
        dbfs = max(-100.0, min(0.0, dbfs))

        return round(rms, 4), round(dbfs, 2), round(peak, 4)

    def normalize(self, audio: np.ndarray, target_peak: Optional[float] = None) -> np.ndarray:
        """
        Scales audio to target peak if signal is audible.
        Avoids amplifying pure digital silence.
        """
        if audio.size == 0:
            return audio

        peak = np.max(np.abs(audio))
        if peak < 1e-4:  # Near silence
            return audio

        target = target_peak if target_peak is not None else self.target_peak
        scaled = audio * (target / peak)
        return np.clip(scaled, -self.clipping_threshold, self.clipping_threshold)

    def preprocess(self, raw_audio: Optional[np.ndarray]) -> PreprocessedAudio:
        """
        Complete preprocessing pipeline for an incoming audio chunk.

        Args:
            raw_audio: 1D or 2D NumPy array of audio samples.

        Returns:
            PreprocessedAudio container.
        """
        if raw_audio is None or raw_audio.size == 0:
            return PreprocessedAudio(
                samples=np.empty((0,), dtype=np.float32),
                rms=0.0,
                dbfs=-100.0,
                peak=0.0,
                is_valid=False,
                sample_count=0,
            )

        # 1. Downmix to mono and sanitize
        mono = self.to_mono(raw_audio)
        clean = self.sanitize(mono)

        # 2. DC offset removal
        if self.enable_dc_removal:
            clean = self.remove_dc_offset(clean)

        # 3. Clip protection
        clean = np.clip(clean, -self.clipping_threshold, self.clipping_threshold)

        # 4. Energy telemetry
        rms, dbfs, peak = self.calculate_rms(clean)

        # 5. Peak normalization
        normalized = self.normalize(clean)

        return PreprocessedAudio(
            samples=normalized.astype(np.float32),
            rms=rms,
            dbfs=dbfs,
            peak=peak,
            is_valid=(clean.size > 0 and peak > 1e-5),
            sample_count=len(clean),
        )
