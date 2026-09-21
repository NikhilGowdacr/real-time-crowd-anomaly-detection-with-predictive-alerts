"""
Audio Capture Module.
Supports real-time microphone input, local WAV/audio file reading,
and synthetic acoustic waveform generation for testing and demonstrations.
"""

from collections import deque
from pathlib import Path
import time
from typing import Generator, Optional, Tuple, Union
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("audio_capture")


class SyntheticAudioGenerator:
    """
    Generates realistic synthetic surveillance audio waveforms (ambient crowd murmur,
    sporadic footsteps, synthetic screaming bursts, impulse crashes) for zero-dependency testing.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.time_offset = 0.0

    def generate_chunk(
        self,
        duration: float = 1.0,
        event_type: str = "normal",
    ) -> np.ndarray:
        """
        Generates a synthetic audio chunk of specified duration.

        Args:
            duration: Chunk duration in seconds.
            event_type: "normal", "scream", "shouting", "explosion_like", "alarm", "crash".

        Returns:
            1D float32 numpy array of audio samples in range [-1.0, 1.0].
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(self.time_offset, self.time_offset + duration, n_samples, endpoint=False)
        self.time_offset += duration

        # Base ambient noise: low-amplitude Brownian / pink-like noise (crowd murmur)
        white = np.random.normal(0, 0.04, n_samples)
        ambient = np.convolve(white, np.ones(8) / 8.0, mode="same").astype(np.float32)

        if event_type == "normal":
            return np.clip(ambient, -1.0, 1.0)

        elif event_type in ("scream", "shouting"):
            # High-frequency oscillating formant frequencies (800Hz - 2500Hz) with vibrato
            vibrato = 6.0 * np.sin(2 * np.pi * 5.0 * t)
            carrier1 = 0.45 * np.sin(2 * np.pi * (1100 + vibrato) * t)
            carrier2 = 0.35 * np.sin(2 * np.pi * (2200 + 2 * vibrato) * t)
            burst = (carrier1 + carrier2) * np.hanning(n_samples)
            return np.clip(ambient + burst.astype(np.float32), -1.0, 1.0)

        elif event_type in ("explosion_like", "crash"):
            # Sharp low-frequency impulse decay with wideband turbulence
            decay = np.exp(-4.0 * np.linspace(0, 1, n_samples))
            impulse = 0.85 * np.sin(2 * np.pi * 65.0 * t) * decay
            crack = 0.40 * np.random.normal(0, 1, n_samples) * decay
            return np.clip(ambient + (impulse + crack).astype(np.float32), -1.0, 1.0)

        elif event_type == "alarm":
            # Alternating warble tone (900Hz and 1500Hz)
            square_wave = np.sign(np.sin(2 * np.pi * 2.0 * t))
            freq = np.where(square_wave > 0, 1400.0, 950.0)
            tone = 0.60 * np.sin(2 * np.pi * freq * t)
            return np.clip(ambient + tone.astype(np.float32), -1.0, 1.0)

        return np.clip(ambient, -1.0, 1.0)


class AudioCapture:
    """
    Unified audio capture stream supporting Live Microphones, WAV files, and Synthetic modes.
    Slices stream into overlapping sliding analysis windows.
    """

    def __init__(
        self,
        source: Union[str, int] = "synthetic",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 1.0,
        overlap: float = 0.5,
    ):
        self.raw_source = str(source).strip()
        self.sample_rate = int(sample_rate)
        self.channels = max(1, int(channels))
        self.chunk_duration = float(chunk_duration)
        self.overlap = max(0.0, min(0.9, float(overlap)))

        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.step_samples = int(self.chunk_samples * (1.0 - self.overlap))

        self.is_synthetic = False
        self.is_file = False
        self.is_mic = False
        self.is_active = False

        self.synthetic_gen: Optional[SyntheticAudioGenerator] = None
        self.file_data: Optional[np.ndarray] = None
        self.file_pos: int = 0

        # Ring buffer for streaming chunks
        self._buffer: deque = deque()
        self._mic_stream = None

        self._initialize_source()

    def _initialize_source(self) -> None:
        """Opens audio input source with graceful fallback on hardware failure."""
        # 1. Synthetic Mode
        if self.raw_source.lower() in ("synthetic", "demo", "mock", "simulated"):
            logger.info("Initializing Synthetic Audio Generator (16000Hz ambient crowd stream)...")
            self.is_synthetic = True
            self.synthetic_gen = SyntheticAudioGenerator(sample_rate=self.sample_rate)
            self.is_active = True
            return

        # 2. Local WAV / Audio file
        file_path = Path(self.raw_source)
        if file_path.is_file() or self.raw_source.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found at '{file_path.resolve()}'. "
                    "Place your file in data/audio/ or configure source: 'synthetic'."
                )
            try:
                try:
                    import soundfile as sf
                    data, sr = sf.read(str(file_path.resolve()), dtype="float32")
                except ImportError:
                    from scipy.io import wavfile
                    sr, raw_data = wavfile.read(str(file_path.resolve()))
                    if raw_data.dtype == np.int16:
                        data = raw_data.astype(np.float32) / 32768.0
                    elif raw_data.dtype == np.int32:
                        data = raw_data.astype(np.float32) / 2147483648.0
                    elif raw_data.dtype == np.uint8:
                        data = (raw_data.astype(np.float32) - 128.0) / 128.0
                    else:
                        data = raw_data.astype(np.float32)
                logger.info(f"Loaded audio file: '{file_path.name}' ({len(data)} samples, {sr}Hz).")


                # Resample if sample rate doesn't match
                if sr != self.sample_rate:
                    from scipy.signal import resample
                    new_len = int(len(data) * self.sample_rate / sr)
                    data = resample(data, new_len).astype(np.float32)

                # Convert to mono if multichannel
                if data.ndim > 1:
                    data = np.mean(data, axis=-1)

                self.file_data = data
                self.file_pos = 0
                self.is_file = True
                self.is_active = True
                return

            except Exception as e:
                raise RuntimeError(f"Failed to read audio file '{file_path}': {e}")

        # 3. Live Microphone Input
        if self.raw_source.lower() in ("microphone", "mic", "default", "0"):
            logger.info("Connecting to physical audio input device (Microphone)...")
            try:
                import sounddevice as sd
                # Query devices to verify input capability
                devices = sd.query_devices()
                default_in = sd.default.device[0]

                if default_in < 0 or len(devices) == 0:
                    raise ConnectionError("No input audio devices detected on the host system.")

                dev_info = sd.query_devices(default_in, "input")
                logger.info(f"Connected to input device: '{dev_info.get('name', 'Default Microphone')}'.")

                def _mic_callback(indata, frames, time_info, status):
                    mono = np.mean(indata, axis=-1) if indata.ndim > 1 else indata.flatten()
                    self._buffer.extend(mono.astype(np.float32))

                self._mic_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    callback=_mic_callback,
                    blocksize=self.step_samples,
                )
                self._mic_stream.start()
                self.is_mic = True
                self.is_active = True
                return

            except Exception as e:
                logger.warning(
                    f"Microphone unavailable ({e}). Gracefully falling back to Synthetic Audio Generator. "
                    "Video operations will continue normally."
                )
                self.is_synthetic = True
                self.synthetic_gen = SyntheticAudioGenerator(sample_rate=self.sample_rate)
                self.is_active = True
                return

        # Unrecognized source -> fallback to synthetic
        logger.warning(f"Unrecognized audio source '{self.raw_source}'. Defaulting to synthetic audio.")
        self.is_synthetic = True
        self.synthetic_gen = SyntheticAudioGenerator(sample_rate=self.sample_rate)
        self.is_active = True

    def read_chunk(self, demo_event: str = "normal") -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the next overlapping audio chunk.

        Args:
            demo_event: Optional event type to simulate in synthetic mode ("normal", "scream", etc.).

        Returns:
            Tuple of (success, chunk_samples np.ndarray of length chunk_samples).
        """
        if not self.is_active:
            return False, None

        # A. Synthetic Generator
        if self.is_synthetic and self.synthetic_gen:
            chunk = self.synthetic_gen.generate_chunk(duration=self.chunk_duration, event_type=demo_event)
            return True, chunk

        # B. Audio File Playback
        if self.is_file and self.file_data is not None:
            if self.file_pos + self.chunk_samples > len(self.file_data):
                # Loop back to beginning
                self.file_pos = 0

            chunk = self.file_data[self.file_pos : self.file_pos + self.chunk_samples]
            self.file_pos += self.step_samples

            if len(chunk) < self.chunk_samples:
                pad = np.zeros(self.chunk_samples - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])

            return True, chunk

        # C. Live Microphone Streaming
        if self.is_mic:
            # Wait until buffer has accumulated at least chunk_samples
            max_wait = 1.2 * self.chunk_duration
            start_wait = time.time()
            while len(self._buffer) < self.chunk_samples:
                time.sleep(0.01)
                if time.time() - start_wait > max_wait:
                    break

            if len(self._buffer) >= self.chunk_samples:
                # Extract chunk
                all_samples = list(self._buffer)
                chunk = np.array(all_samples[: self.chunk_samples], dtype=np.float32)

                # Discard step_samples for sliding overlap
                for _ in range(min(len(self._buffer), self.step_samples)):
                    self._buffer.popleft()

                return True, chunk

            return False, None

        return False, None

    def release(self) -> None:
        """Closes audio streams and releases device handles."""
        self.is_active = False
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
                logger.info("Audio input stream successfully closed.")
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
            self._mic_stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
