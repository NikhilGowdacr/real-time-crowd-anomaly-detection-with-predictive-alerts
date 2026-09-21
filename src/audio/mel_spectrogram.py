"""
Mel Spectrogram Extraction Module.
Converts preprocessed audio waveforms into normalized 2D Mel Spectrograms.
Features a dual-engine architecture: uses Librosa when available, with a pure
NumPy/SciPy Mel filterbank fallback for zero-dependency reliability.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch

from src.utils.logger import setup_logger

logger = setup_logger("mel_spectrogram")


class MelSpectrogramProcessor:
    """
    Extracts log-power Mel Spectrograms from audio chunks.
    Standardized for 16000Hz audio with 64 Mel frequency bins.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: Optional[float] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if f_max is not None else float(self.sample_rate / 2.0)

        self._has_librosa = False
        try:
            import librosa
            self._librosa = librosa
            self._has_librosa = True
            logger.info("MelSpectrogramProcessor initialized with Librosa engine.")
        except ImportError:
            self._has_librosa = False
            logger.info("Librosa not installed. Initializing native NumPy/SciPy Mel filterbank engine.")
            self._mel_filterbank = self._build_mel_filterbank()

    def _hz_to_mel(self, hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Converts frequency in Hz to Mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Converts Mel scale to frequency in Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> np.ndarray:
        """Constructs triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
        n_freqs = self.n_fft // 2 + 1
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)

        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        weights = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus and k < n_freqs:
                    weights[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m and k < n_freqs:
                    weights[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return weights

    def to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Converts a 1D audio sample array into a log-power Mel Spectrogram.

        Args:
            audio: 1D float32 audio waveform.

        Returns:
            2D numpy array of shape (n_mels, time_steps) with values in decibels (dB).
        """
        if audio is None or len(audio) < self.n_fft:
            # Pad short audio to at least n_fft
            if audio is None or len(audio) == 0:
                audio = np.zeros(self.n_fft, dtype=np.float32)
            else:
                pad = np.zeros(self.n_fft - len(audio), dtype=np.float32)
                audio = np.concatenate([audio, pad])

        # 1. Librosa Engine
        if self._has_librosa:
            mel = self._librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
                power=2.0,
            )
            mel_db = self._librosa.power_to_db(mel, ref=np.max)
            return mel_db.astype(np.float32)

        # 2. Native NumPy / SciPy Engine
        # STFT via sliding Hanning window
        window = np.hanning(self.n_fft)
        n_samples = len(audio)
        n_frames = 1 + (n_samples - self.n_fft) // self.hop_length
        n_freqs = self.n_fft // 2 + 1

        stft_matrix = np.empty((n_freqs, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            start = i * self.hop_length
            segment = audio[start : start + self.n_fft] * window
            spectrum = np.fft.rfft(segment, n=self.n_fft)
            stft_matrix[:, i] = spectrum

        # Power spectrogram
        power_spec = np.abs(stft_matrix) ** 2

        # Apply Mel Filterbank: (n_mels, n_freqs) @ (n_freqs, n_frames) -> (n_mels, n_frames)
        mel_spec = np.dot(self._mel_filterbank, power_spec)

        # Convert to decibels relative to peak
        ref = np.max(mel_spec) if np.max(mel_spec) > 1e-9 else 1.0
        mel_db = 10.0 * np.log10(np.maximum(1e-9, mel_spec) / ref)
        mel_db = np.clip(mel_db, -80.0, 0.0)

        return mel_db.astype(np.float32)

    def to_torch_tensor(self, audio: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """
        Extracts Mel spectrogram and formats as 4D PyTorch tensor: (1, 1, n_mels, time_steps).
        """
        mel_db = self.to_mel_spectrogram(audio)
        # Normalize to [0.0, 1.0] range
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        tensor = torch.from_numpy(mel_norm).unsqueeze(0).unsqueeze(0).to(device)
        return tensor
