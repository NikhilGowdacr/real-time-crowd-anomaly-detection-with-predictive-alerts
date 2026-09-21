"""
Dataset Abstraction & Sample Representation - Phase 12.

Provides a modular dataset interface supporting multimodal (video+audio),
video-only, and audio-only streams, with optional ground-truth annotations.
If no dataset is configured, operates safely without crashing.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterator, List, Optional
import numpy as np

from src.evaluation.ground_truth import EvaluationSource, GroundTruthAnnotation, GroundTruthState


@dataclass(init=False)
class EvaluationSample:
    """Represents a single multimodal surveillance evaluation sample."""
    sample_id: str
    video_source: Optional[str] = None
    audio_source: Optional[str] = None
    video_frame: Optional[np.ndarray] = None
    audio_chunk: Optional[np.ndarray] = None
    ground_truth: Optional[GroundTruthAnnotation] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        sample_id: str,
        timestamp: float = 0.0,
        video_source: Optional[str] = None,
        audio_source: Optional[str] = None,
        video_frame: Optional[np.ndarray] = None,
        audio_chunk: Optional[np.ndarray] = None,
        ground_truth: Optional[GroundTruthAnnotation] = None,
        metadata: Optional[Dict[str, Any]] = None,
        frame: Optional[np.ndarray] = None,
        annotation: Optional[GroundTruthAnnotation] = None,
    ) -> None:
        self.sample_id = sample_id
        self.timestamp = float(timestamp) if timestamp else time.time()
        self.video_source = video_source
        self.audio_source = audio_source
        self.video_frame = frame if frame is not None else video_frame
        self.audio_chunk = audio_chunk
        self.ground_truth = annotation if annotation is not None else ground_truth
        self.metadata = dict(metadata or {})

    @property
    def has_video(self) -> bool:
        return self.video_frame is not None or self.video_source is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_chunk is not None or self.audio_source is not None

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth is not None and self.ground_truth.has_valid_ground_truth

    @property
    def ground_truth_label(self) -> Optional[str]:
        return self.ground_truth.state if self.ground_truth else None


class EvaluationDataset:
    """
    Modular surveillance dataset container.
    Gracefully handles empty or unconfigured datasets without crashing.
    """

    def __init__(
        self,
        name: str = "SurveillanceDataset",
        dataset_type: str = "none",
        samples: Optional[List[EvaluationSample]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.dataset_type = dataset_type
        self.samples: List[EvaluationSample] = samples or []
        self.metadata: Dict[str, Any] = metadata or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvaluationSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> EvaluationSample:
        return self.samples[idx]

    @property
    def has_ground_truth(self) -> bool:
        """Returns True only if at least one sample contains explicit ground truth."""
        return any(s.has_ground_truth for s in self.samples)

    @property
    def ground_truth_count(self) -> int:
        return sum(1 for s in self.samples if s.has_ground_truth)

    def get_ground_truth_labels(self) -> List[Optional[str]]:
        """Returns list of discrete risk-state ground-truth labels."""
        return [s.ground_truth.state if s.ground_truth else None for s in self.samples]

    def get_binary_ground_truth_labels(self) -> List[Optional[int]]:
        """Returns list of binary anomaly labels (0 = normal, 1 = anomalous)."""
        return [s.ground_truth.is_anomaly if s.ground_truth else None for s in self.samples]

    def filter_by_modality(self, modality: str = "bimodal") -> "EvaluationDataset":
        """
        Filter dataset samples by modality:
        - 'bimodal': samples with both video and audio
        - 'video_only': samples with video but without audio
        - 'audio_only': samples with audio but without video
        """
        filtered: List[EvaluationSample] = []
        for s in self.samples:
            if modality == "bimodal" and s.has_video and s.has_audio:
                filtered.append(s)
            elif modality == "video_only" and s.has_video and not s.has_audio:
                filtered.append(s)
            elif modality == "audio_only" and s.has_audio and not s.has_video:
                filtered.append(s)

        return EvaluationDataset(
            name=f"{self.name}_{modality}",
            dataset_type=self.dataset_type,
            samples=filtered,
            metadata={"source_dataset": self.name, "modality": modality},
        )

    @property
    def ground_truth_available(self) -> bool:
        return self.has_ground_truth

    def get_video_samples(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.has_video]

    def get_audio_samples(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.has_audio]

    def get_annotated_samples(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.has_ground_truth]

    @classmethod
    def create_empty(cls, name: str = "unconfigured") -> "EvaluationDataset":
        """Creates an explicit empty dataset container."""
        return cls(
            name=name,
            dataset_type="none",
            samples=[],
            metadata={"configured": False, "reason": "Ground-truth dataset not configured"},
        )

    @classmethod
    def unconfigured(cls) -> "EvaluationDataset":
        return cls.create_empty()

    @classmethod
    def create_synthetic_benchmark(
        cls,
        num_samples: int = 50,
        name: str = "ControlledSyntheticDataset",
    ) -> "EvaluationDataset":
        """
        Generates synthetic evaluation samples for controlled stress-testing.
        Explicitly labeled with SYNTHETIC_SCENARIO provenance to prevent real-world misattribution.
        """
        samples: List[EvaluationSample] = []
        states = [
            GroundTruthState.NORMAL.value,
            GroundTruthState.NORMAL.value,
            GroundTruthState.SUSPICIOUS.value,
            GroundTruthState.HIGH_RISK.value,
            GroundTruthState.EMERGENCY.value,
        ]

        t_base = time.time()
        for i in range(num_samples):
            st = states[i % len(states)]
            is_anom = 0 if st == GroundTruthState.NORMAL.value else 1
            gt = GroundTruthAnnotation(
                sample_id=f"SYNTH-{i:04d}",
                state=st,
                is_anomaly=is_anom,
                source=EvaluationSource.SYNTHETIC_SCENARIO.value,
                notes="Controlled synthetic evaluation pattern",
            )
            sample = EvaluationSample(
                sample_id=f"SYNTH-{i:04d}",
                video_source="synthetic",
                audio_source="synthetic",
                ground_truth=gt,
                timestamp=t_base + i * 0.033,
                metadata={"synthetic_index": i},
            )
            samples.append(sample)

        return cls(
            name=name,
            dataset_type="synthetic",
            samples=samples,
            metadata={
                "synthetic": True,
                "provenance": EvaluationSource.SYNTHETIC_SCENARIO.value,
                "notice": "Synthetic evaluation dataset. NOT real-world accuracy.",
            },
        )
