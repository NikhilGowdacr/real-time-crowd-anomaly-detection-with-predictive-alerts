"""
==============================================================================
Real-Time Crowd Anomaly Detection with Predictive Alerts
Unit Tests for Phase 5: Audio Anomaly Detection Subsystem
==============================================================================
"""

import math
from pathlib import Path
import tempfile
import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None


from src.utils.config import load_config
from src.audio import (
    AudioCapture,
    SyntheticAudioGenerator,
    AudioPreprocessor,
    PreprocessedAudio,
    MelSpectrogramProcessor,
    AudioClassifier,
    AudioClassificationResult,
    AudioAnomalyDetector,
    AudioEventRecord,
)


# ============================================================================
# 1. Config Loading
# ============================================================================
def test_audio_config_defaults():
    """Test 1: Audio config loads with correct defaults."""
    config = load_config()
    assert "audio" in config, "Missing 'audio' section in configuration"
    audio_cfg = config["audio"]

    assert audio_cfg["sample_rate"] == 16000
    assert audio_cfg["channels"] == 1
    assert audio_cfg["chunk_duration"] == 1.0
    assert audio_cfg["overlap"] == 0.5
    assert audio_cfg["n_mels"] == 64
    assert audio_cfg["n_fft"] == 1024
    assert audio_cfg["hop_length"] == 512
    assert audio_cfg["confidence_threshold"] == 0.50
    assert audio_cfg["temporal_window"] == 5
    assert audio_cfg["confirmation_windows"] == 2


# ============================================================================
# 2. Audio Capture
# ============================================================================
def test_audio_capture_synthetic_initialization():
    """Test 2: Audio capture initializes without hardware error."""
    cap = AudioCapture(source="synthetic", sample_rate=16000, chunk_duration=1.0)
    assert cap.is_active is True
    assert cap.is_synthetic is True

    success, chunk = cap.read_chunk()
    assert success is True
    assert chunk is not None
    assert len(chunk) == 16000
    assert isinstance(chunk, np.ndarray)
    assert chunk.dtype == np.float32
    cap.release()
    assert cap.is_active is False


def test_audio_capture_missing_microphone_fallback():
    """Test 3: Missing microphone handled gracefully (fallback to synthetic)."""
    # Even if no mic device or permission issue occurs, capture falls back to synthetic
    cap = AudioCapture(source="microphone", sample_rate=16000, chunk_duration=0.5)
    assert cap.is_active is True
    # Can read chunks cleanly regardless of whether hardware mic exists
    success, chunk = cap.read_chunk()
    assert success is True
    assert chunk is not None
    assert len(chunk) == 8000
    cap.release()


def test_audio_capture_wav_file_loading():
    """Test 4: WAV file loading produces correct sample rate and channels."""
    # Create a temporary WAV file
    sample_rate = 16000
    duration = 1.5
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # 440 Hz tone
    samples = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        temp_wav_path = tf.name

    try:
        if sf is not None:
            sf.write(temp_wav_path, samples, sample_rate)
        else:
            from scipy.io import wavfile
            wavfile.write(temp_wav_path, sample_rate, (samples * 32767).astype(np.int16))


        cap = AudioCapture(source=temp_wav_path, sample_rate=sample_rate, chunk_duration=1.0, overlap=0.0)
        assert cap.is_active is True
        assert cap.is_file is True
        assert cap.file_data is not None
        assert cap.file_data.ndim == 1

        success, chunk = cap.read_chunk()
        assert success is True
        assert len(chunk) == 16000
        cap.release()
    finally:
        Path(temp_wav_path).unlink(missing_ok=True)


# ============================================================================
# 3. Audio Preprocessing
# ============================================================================
def test_audio_preprocessing_dc_offset_removal():
    """Test 5: Audio preprocessing produces zero-mean signal."""
    preprocessor = AudioPreprocessor(enable_dc_removal=True)
    # Signal with artificial DC offset +0.35
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    signal = (0.2 * np.sin(2 * np.pi * 300.0 * t) + 0.35).astype(np.float32)

    result = preprocessor.preprocess(signal)
    assert result.is_valid is True
    assert math.isclose(float(np.mean(result.samples)), 0.0, abs_tol=1e-5)


def test_audio_preprocessing_stereo_downmix():
    """Test 6: Stereo downmixed to mono correctly."""
    preprocessor = AudioPreprocessor()
    # 2-channel stereo: Left = 0.2, Right = 0.8 -> Mean = 0.5
    stereo = np.zeros((16000, 2), dtype=np.float32)
    stereo[:, 0] = 0.2
    stereo[:, 1] = 0.8

    mono = preprocessor.to_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == 16000
    assert np.allclose(mono, 0.5, atol=1e-6)


def test_audio_preprocessing_peak_normalization():
    """Test 7: Peak normalization scales to [-1.0, 1.0] without clipping."""
    preprocessor = AudioPreprocessor(target_peak=0.95, clipping_threshold=1.0)
    raw = np.array([0.1, -0.25, 0.4, -0.3], dtype=np.float32)

    normalized = preprocessor.normalize(raw, target_peak=0.95)
    assert math.isclose(float(np.max(np.abs(normalized))), 0.95, abs_tol=1e-5)
    assert np.all(normalized <= 1.0)
    assert np.all(normalized >= -1.0)


# ============================================================================
# 4. Mel Spectrogram Extraction
# ============================================================================
def test_mel_spectrogram_dimensions():
    """Test 8: Mel spectrogram produces correct dimensions (64 bins x T frames)."""
    processor = MelSpectrogramProcessor(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)
    waveform = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)

    mel_spec = processor.to_mel_spectrogram(waveform)
    assert mel_spec.ndim == 2
    assert mel_spec.shape[0] == 64, f"Expected 64 mel bands, got {mel_spec.shape[0]}"
    # For 16000 samples with n_fft=1024, hop=512: 1 + (16000-1024)//512 = 30 or 32 frames
    assert mel_spec.shape[1] >= 28, f"Expected ~30-32 time frames, got {mel_spec.shape[1]}"


def test_mel_spectrogram_silence_handling():
    """Test 9: Mel spectrogram handles silence / all-zeros without NaN or Inf."""
    processor = MelSpectrogramProcessor(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)
    silence = np.zeros(16000, dtype=np.float32)

    mel_spec = processor.to_mel_spectrogram(silence)
    assert np.all(np.isfinite(mel_spec)), "Mel spectrogram of silence contained NaNs or Infs"
    assert not np.isnan(mel_spec).any()
    assert not np.isinf(mel_spec).any()


# ============================================================================
# 5. Audio Classification
# ============================================================================
def test_audio_classifier_spectral_baseline_initialization():
    """Test 10: Classifier initializes with spectral baseline fallback."""
    classifier = AudioClassifier()
    assert classifier.model_mode == "spectral_baseline"
    assert classifier.is_custom_model is False
    assert "Spectral" in classifier.status_message


def test_audio_classifier_missing_model_fallback():
    """Test 11: Classifier gracefully handles missing model file."""
    classifier = AudioClassifier(model_path="nonexistent_checkpoint.pt")
    assert classifier.model_mode == "spectral_baseline"
    assert classifier.is_custom_model is False


def test_audio_classifier_confidence_threshold():
    """Test 12: Classifier predictions respect confidence threshold."""
    classifier = AudioClassifier(confidence_threshold=0.99)
    # Low energy ambient noise
    preprocessed = PreprocessedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        rms=0.01,
        dbfs=-40.0,
        peak=0.02,
        is_valid=True,
        sample_count=16000,
    )
    mel = np.zeros((64, 30), dtype=np.float32)

    result = classifier.classify(preprocessed, mel)
    # Low confidence should not trigger abnormal alarms
    assert result.event == "normal" or result.audio_anomaly_score == 0.0


def test_audio_classifier_anomaly_scores_by_class():
    """Test 13: Anomaly score calculated correctly for each class."""
    classifier = AudioClassifier()
    preprocessed = PreprocessedAudio(
        samples=np.ones(16000, dtype=np.float32) * 0.5,
        rms=0.5,
        dbfs=-6.0,
        peak=0.8,
        is_valid=True,
        sample_count=16000,
    )
    mel = np.ones((64, 30), dtype=np.float32)

    # Test controlled simulation triggers for each category
    normal_res = classifier.classify(preprocessed, mel, demo_event="normal")
    assert normal_res.audio_anomaly_score == 0.0

    scream_res = classifier.classify(preprocessed, mel, demo_event="scream")
    assert scream_res.audio_anomaly_score == classifier.SEVERITY_WEIGHTS["scream"]

    alarm_res = classifier.classify(preprocessed, mel, demo_event="alarm")
    assert alarm_res.audio_anomaly_score == classifier.SEVERITY_WEIGHTS["alarm"]

    explosion_res = classifier.classify(preprocessed, mel, demo_event="explosion_like")
    assert explosion_res.audio_anomaly_score == classifier.SEVERITY_WEIGHTS["explosion_like"]


# ============================================================================
# 6. Temporal Smoothing and Anomaly Confirmation
# ============================================================================
def test_audio_anomaly_rolling_smoothing():
    """Test 14: Rolling temporal smoothing dampens score spikes."""
    detector = AudioAnomalyDetector(smoothing_window=5, confirmation_windows=2)

    # 4 nominal windows with 0.0 anomaly score
    normal_result = AudioClassificationResult(
        event="normal",
        confidence=0.90,
        audio_anomaly_score=0.0,
        timestamp=0.0,
        duration=1.0,
        is_custom_model=False,
        model_mode="spectral_baseline",
        status_message="",
    )
    for _ in range(4):
        detector.evaluate(normal_result)

    # Single spike window with 1.0 anomaly score
    spike_result = AudioClassificationResult(
        event="scream",
        confidence=0.90,
        audio_anomaly_score=1.0,
        timestamp=4.0,
        duration=1.0,
        is_custom_model=False,
        model_mode="spectral_baseline",
        status_message="",
    )
    record = detector.evaluate(spike_result)

    # Smoothed score should be (0 + 0 + 0 + 0 + 1.0) / 5 = 0.20 (and since unconfirmed, halved to 0.10)
    assert record.audio_anomaly_score < 0.30, f"Expected dampened score, got {record.audio_anomaly_score}"
    assert record.is_confirmed is False


def test_audio_anomaly_consecutive_confirmation():
    """Test 15: Consecutive confirmation window requires N consecutive detections."""
    detector = AudioAnomalyDetector(temporal_window=5, confirmation_windows=2, smoothing_window=3)

    abnormal_result = AudioClassificationResult(
        event="scream",
        confidence=0.90,
        audio_anomaly_score=0.85,
        timestamp=1.0,
        duration=1.0,
        is_custom_model=False,
        model_mode="spectral_baseline",
        status_message="",
    )

    # Window 1: Abnormal event -> consecutive_abnormal_count = 1 -> is_confirmed = False
    rec1 = detector.evaluate(abnormal_result)
    assert rec1.consecutive_abnormal_count == 1
    assert rec1.is_confirmed is False

    # Window 2: Sustained abnormal event -> consecutive_abnormal_count = 2 -> is_confirmed = True
    rec2 = detector.evaluate(abnormal_result)
    assert rec2.consecutive_abnormal_count == 2
    assert rec2.is_confirmed is True
    assert rec2.status in ("SUSPICIOUS", "EMERGENCY")


# ============================================================================
# 7. Output Schema and Interface
# ============================================================================
def test_audio_event_record_output_schema():
    """Test 16: Output dictionary matches required schema."""
    detector = AudioAnomalyDetector()
    sample_res = AudioClassificationResult(
        event="alarm",
        confidence=0.82,
        audio_anomaly_score=0.70,
        timestamp=100.0,
        duration=1.0,
        is_custom_model=False,
        model_mode="spectral_baseline",
        status_message="",
    )

    record = detector.evaluate(sample_res)
    output_dict = record.to_dict()

    required_keys = [
        "audio_event",
        "audio_confidence",
        "audio_anomaly_score",
        "audio_timestamp",
        "duration",
        "is_confirmed",
        "status",
        "description",
    ]

    for key in required_keys:
        assert key in output_dict, f"Missing key '{key}' in AudioEventRecord.to_dict()"

    assert isinstance(output_dict["audio_event"], str)
    assert isinstance(output_dict["audio_confidence"], float)
    assert isinstance(output_dict["audio_anomaly_score"], float)
    assert isinstance(output_dict["audio_timestamp"], float)
    assert isinstance(output_dict["is_confirmed"], bool)
    assert isinstance(output_dict["status"], str)
