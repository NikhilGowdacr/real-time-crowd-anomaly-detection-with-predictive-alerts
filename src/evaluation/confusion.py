"""
Confusion Matrix Evaluator - Phase 12.

Calculates raw integer confusion matrices and row-normalized (recall)
confusion matrices across discrete risk states or event categories.
Returns explicit 'NOT_AVAILABLE' status when ground truth is missing.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.evaluation.ground_truth import EvaluationSource, GroundTruthState


@dataclass
class ConfusionMatrixResult:
    """Represents a computed confusion matrix and normalized matrix."""
    matrix: Any = field(default_factory=list)
    normalized_matrix: Any = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    sample_count: int = 0
    status: str = "AVAILABLE"
    source: str = EvaluationSource.UNAVAILABLE.value
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.matrix, list):
            self.matrix = np.array(self.matrix, dtype=int) if len(self.matrix) > 0 else np.zeros((0, 0), dtype=int)
        if isinstance(self.normalized_matrix, list):
            self.normalized_matrix = np.array(self.normalized_matrix, dtype=float) if len(self.normalized_matrix) > 0 else np.zeros((0, 0), dtype=float)

    @property
    def available(self) -> bool:
        return self.status == "AVAILABLE"

    @property
    def is_available(self) -> bool:
        return self.status == "AVAILABLE"

    def to_table_string(self) -> str:
        if self.matrix is None or (hasattr(self.matrix, "size") and self.matrix.size == 0) or not self.classes:
            return "Confusion Matrix: NOT_AVAILABLE"
        mat_list = self.matrix.tolist() if isinstance(self.matrix, np.ndarray) else self.matrix
        header = f"{'True \\ Pred':15s} | " + " | ".join(f"{c:>12s}" for c in self.classes)
        sep = "-" * len(header)
        rows = [header, sep]
        for c, row in zip(self.classes, mat_list):
            row_str = f"{c:15s} | " + " | ".join(f"{val:>12d}" for val in row)
            rows.append(row_str)
        return "\n".join(rows)

    def to_dict(self) -> Dict[str, Any]:
        mat_list = self.matrix.tolist() if isinstance(self.matrix, np.ndarray) else self.matrix
        norm_list = self.normalized_matrix.tolist() if isinstance(self.normalized_matrix, np.ndarray) else self.normalized_matrix
        return {
            "matrix": mat_list,
            "normalized_matrix": [[round(val, 4) for val in row] for row in norm_list],
            "classes": self.classes,
            "sample_count": self.sample_count,
            "status": self.status,
            "available": self.status == "AVAILABLE",
            "source": self.source,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def unavailable(
        cls,
        reason: str = "Ground-truth labels not configured",
        source: Any = EvaluationSource.UNAVAILABLE.value,
    ) -> "ConfusionMatrixResult":
        src_val = source.value if hasattr(source, "value") else str(source)
        return cls(
            matrix=[],
            normalized_matrix=[],
            classes=[],
            sample_count=0,
            status="NOT_AVAILABLE",
            source=src_val,
            reason=reason,
        )


class ConfusionMatrixEvaluator:
    """Evaluates multi-class prediction distributions against ground-truth labels."""

    STANDARD_CLASSES = [
        GroundTruthState.NORMAL.value,
        GroundTruthState.SUSPICIOUS.value,
        GroundTruthState.HIGH_RISK.value,
        GroundTruthState.EMERGENCY.value,
        GroundTruthState.RECOVERY.value,
    ]

    def __init__(self, classes: Optional[List[str]] = None) -> None:
        self.classes = classes

    def compute(
        self,
        y_true: Optional[Sequence[str]],
        y_pred: Optional[Sequence[str]],
        classes: Optional[List[str]] = None,
        source: Any = EvaluationSource.REAL_DATASET.value,
    ) -> ConfusionMatrixResult:
        """
        Compute confusion matrix.
        Returns NOT_AVAILABLE if ground-truth labels or predictions are absent.
        """
        src_val = source.value if hasattr(source, "value") else str(source)
        cls_list = classes or self.classes
        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            return ConfusionMatrixResult.unavailable(
                reason="Ground-truth labels not configured or empty input sequence",
                source=source,
            )

        if len(y_true) != len(y_pred):
            return ConfusionMatrixResult.unavailable(
                reason=f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}",
                source=source,
            )

        yt = [str(x) for x in y_true]
        yp = [str(x) for x in y_pred]

        classes_to_use = cls_list
        if not classes_to_use:
            # Preserve standard order for recognized states, append any others
            present_classes = set(yt) | set(yp)
            classes_to_use = [c for c in self.STANDARD_CLASSES if c in present_classes]
            remaining = sorted(list(present_classes - set(classes_to_use)))
            classes_to_use.extend(remaining)

        if not classes_to_use:
            classes_to_use = sorted(list(set(yt) | set(yp)))

        class_to_idx = {c: i for i, c in enumerate(classes_to_use)}
        k = len(classes_to_use)
        raw_mat = np.zeros((k, k), dtype=int)

        for t, p in zip(yt, yp):
            t_idx = class_to_idx.get(t)
            p_idx = class_to_idx.get(p)
            if t_idx is not None and p_idx is not None:
                raw_mat[t_idx, p_idx] += 1

        # Row-normalized matrix (recall per true class)
        row_sums = raw_mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_mat = np.where(row_sums > 0, raw_mat / row_sums, 0.0)

        return ConfusionMatrixResult(
            matrix=raw_mat,
            normalized_matrix=norm_mat,
            classes=classes_to_use,
            sample_count=len(yt),
            status="AVAILABLE",
            source=src_val,
        )

    evaluate = compute
