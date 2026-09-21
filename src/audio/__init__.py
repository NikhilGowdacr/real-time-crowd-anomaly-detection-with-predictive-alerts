"""
Audio Processing Subsystem (Phase 5).
Provides real-time audio capture (microphone, WAV, synthetic),
signal preprocessing, Mel spectrogram extraction, audio event classification,
and temporal anomaly confirmation.
"""

from src.audio.audio_capture import AudioCapture, SyntheticAudioGenerator
from src.audio.audio_preprocessing import AudioPreprocessor, PreprocessedAudio
from src.audio.mel_spectrogram import MelSpectrogramProcessor
from src.audio.audio_classifier import AudioClassifier, AudioClassificationResult
from src.audio.audio_anomaly import AudioAnomalyDetector, AudioEventRecord

__all__ = [
    "AudioCapture",
    "SyntheticAudioGenerator",
    "AudioPreprocessor",
    "PreprocessedAudio",
    "MelSpectrogramProcessor",
    "AudioClassifier",
    "AudioClassificationResult",
    "AudioAnomalyDetector",
    "AudioEventRecord",
]
