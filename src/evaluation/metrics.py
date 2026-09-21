"""
Classification Metrics & Performance Results - Phase 12.

Implements accuracy, precision, recall, and F1-score (macro, micro, weighted)
with native NumPy and Python standard library calculations.
Enforces zero-division safety and returns explicit 'NOT_AVAILABLE' status
whenever ground truth is missing or unconfigured.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from src.evaluation.ground_truth import EvaluationSource


class ClassMetricItem:
    """Wrapper providing both attribute and item access to per-class metrics."""
    def __init__(self, d: Dict[str, Any]) -> None:
        self.d = d
        self.precision = d.get("precision")
        self.recall = d.get("recall")
        self.f1 = d.get("f1")
        self.support = d.get("support")

    def __getitem__(self, key: str) -> Any:
        return self.d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.d.get(key, default)


@dataclass
class MetricResult:
    """Represents a single evaluated metric value with provenance."""
    name: str = ""
    value: Optional[float] = None
    unit: str = ""
    status: str = "AVAILABLE"  # "AVAILABLE", "NOT_AVAILABLE"
    source: str = EvaluationSource.UNAVAILABLE.value
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metric_name: Optional[str] = None
    available: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.metric_name and not self.name:
            self.name = self.metric_name
        if not self.metric_name:
            self.metric_name = self.name
        if self.available is not None:
            self.status = "AVAILABLE" if self.available else "NOT_AVAILABLE"
        else:
            self.available = (self.status == "AVAILABLE")
        if hasattr(self.source, "value"):
            self.source = self.source.value

    @property
    def is_available(self) -> bool:
        return self.status == "AVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric_name": self.name,
            "value": round(self.value, 4) if isinstance(self.value, (int, float)) else None,
            "unit": self.unit,
            "status": self.status,
            "available": self.status == "AVAILABLE",
            "source": self.source,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def unavailable(
        cls,
        name: str,
        reason: str = "Ground-truth labels not configured",
        unit: str = "",
    ) -> "MetricResult":
        return cls(
            name=name,
            metric_name=name,
            value=None,
            unit=unit,
            status="NOT_AVAILABLE",
            source=EvaluationSource.UNAVAILABLE.value,
            reason=reason,
            available=False,
        )


@dataclass
class ClassificationResult:
    """Comprehensive container for binary and multi-class classification metrics."""
    accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_weighted: Optional[float] = None
    tp: Optional[int] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    per_class: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    sample_count: int = 0
    status: str = "AVAILABLE"
    source: str = EvaluationSource.UNAVAILABLE.value
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    available: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.precision is not None and self.precision_macro is None:
            self.precision_macro = self.precision
        if self.recall is not None and self.recall_macro is None:
            self.recall_macro = self.recall
        if self.f1_score is not None and self.f1_macro is None:
            self.f1_macro = self.f1_score

        if self.precision_macro is not None and self.precision is None:
            self.precision = self.precision_macro
        if self.recall_macro is not None and self.recall is None:
            self.recall = self.recall_macro
        if self.f1_macro is not None and self.f1_score is None:
            self.f1_score = self.f1_macro

        if self.available is not None:
            self.status = "AVAILABLE" if self.available else "NOT_AVAILABLE"
        else:
            self.available = (self.status == "AVAILABLE")
        if hasattr(self.source, "value"):
            self.source = self.source.value

    @property
    def is_available(self) -> bool:
        return self.status == "AVAILABLE"

    @property
    def macro_precision(self) -> Optional[float]:
        return self.precision_macro

    @property
    def macro_recall(self) -> Optional[float]:
        return self.recall_macro

    @property
    def macro_f1(self) -> Optional[float]:
        return self.f1_macro

    @property
    def weighted_f1(self) -> Optional[float]:
        return self.f1_weighted

    @property
    def class_metrics(self) -> Dict[str, ClassMetricItem]:
        return {k: ClassMetricItem(v) for k, v in self.per_class.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "precision": round(self.precision_macro, 4) if self.precision_macro is not None else None,
            "recall": round(self.recall_macro, 4) if self.recall_macro is not None else None,
            "f1_score": round(self.f1_macro, 4) if self.f1_macro is not None else None,
            "precision_macro": round(self.precision_macro, 4) if self.precision_macro is not None else None,
            "recall_macro": round(self.recall_macro, 4) if self.recall_macro is not None else None,
            "f1_macro": round(self.f1_macro, 4) if self.f1_macro is not None else None,
            "precision_weighted": round(self.precision_weighted, 4) if self.precision_weighted is not None else None,
            "recall_weighted": round(self.recall_weighted, 4) if self.recall_weighted is not None else None,
            "f1_weighted": round(self.f1_weighted, 4) if self.f1_weighted is not None else None,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "per_class": self.per_class,
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
        source: str = EvaluationSource.UNAVAILABLE.value,
    ) -> "ClassificationResult":
        return cls(
            status="NOT_AVAILABLE",
            source=source.value if hasattr(source, "value") else str(source),
            reason=reason,
            sample_count=0,
            available=False,
        )


def calculate_binary_metrics(
    y_true: Optional[Sequence[int]],
    y_pred: Optional[Sequence[int]],
    source: str = EvaluationSource.REAL_DATASET.value,
) -> ClassificationResult:
    """
    Computes binary anomaly classification metrics (0 = Normal, 1 = Anomaly).
    Gracefully handles missing labels or zero division without crashing.
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        return ClassificationResult.unavailable(
            reason="Ground-truth labels not configured or empty input sequence",
            source=source,
        )

    if len(y_true) != len(y_pred):
        return ClassificationResult.unavailable(
            reason=f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}",
            source=source,
        )

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    n = len(yt)

    # Confusion counts
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return ClassificationResult(
        accuracy=acc,
        precision_macro=prec,
        recall_macro=rec,
        f1_macro=f1,
        precision_weighted=prec,
        recall_weighted=rec,
        f1_weighted=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        per_class={
            "normal": {
                "precision": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
                "recall": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "f1": (2.0 * (tn / (tn + fn) if (tn + fn) > 0 else 0.0) * (tn / (tn + fp) if (tn + fp) > 0 else 0.0))
                / ((tn / (tn + fn) if (tn + fn) > 0 else 0.0) + (tn / (tn + fp) if (tn + fp) > 0 else 0.0))
                if ((tn / (tn + fn) if (tn + fn) > 0 else 0.0) + (tn / (tn + fp) if (tn + fp) > 0 else 0.0)) > 0
                else 0.0,
                "support": int(np.sum(yt == 0)),
            },
            "anomalous": {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": int(np.sum(yt == 1)),
            },
        },
        sample_count=n,
        status="AVAILABLE",
        source=source,
    )


def calculate_multiclass_metrics(
    y_true: Optional[Sequence[str]],
    y_pred: Optional[Sequence[str]],
    classes: Optional[List[str]] = None,
    source: str = EvaluationSource.REAL_DATASET.value,
) -> ClassificationResult:
    """
    Computes multi-class classification metrics (Macro and Weighted Precision, Recall, F1).
    Handles arbitrary discrete risk states or event categories without external sklearn dependency.
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        return ClassificationResult.unavailable(
            reason="Ground-truth labels not configured or empty input sequence",
            source=source,
        )

    if len(y_true) != len(y_pred):
        return ClassificationResult.unavailable(
            reason=f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}",
            source=source,
        )

    yt = [str(x) for x in y_true]
    yp = [str(x) for x in y_pred]
    n = len(yt)

    # Determine unique classes if not explicitly provided
    if not classes:
        classes = sorted(list(set(yt) | set(yp)))

    # Overall accuracy
    correct = sum(1 for t, p in zip(yt, yp) if t == p)
    acc = correct / n if n > 0 else 0.0

    per_class: Dict[str, Dict[str, Optional[float]]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for c in classes:
        tp = sum(1 for t, p in zip(yt, yp) if t == c and p == c)
        fp = sum(1 for t, p in zip(yt, yp) if t != c and p == c)
        fn = sum(1 for t, p in zip(yt, yp) if t == c and p != c)
        sup = sum(1 for t in yt if t == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)

    # Macro averages
    num_classes = len(classes)
    macro_prec = sum(precisions) / num_classes if num_classes > 0 else 0.0
    macro_rec = sum(recalls) / num_classes if num_classes > 0 else 0.0
    macro_f1 = sum(f1s) / num_classes if num_classes > 0 else 0.0

    # Weighted averages
    total_support = sum(supports)
    if total_support > 0:
        weighted_prec = sum(p * s for p, s in zip(precisions, supports)) / total_support
        weighted_rec = sum(r * s for r, s in zip(recalls, supports)) / total_support
        weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support
    else:
        weighted_prec, weighted_rec, weighted_f1 = macro_prec, macro_rec, macro_f1

    return ClassificationResult(
        accuracy=acc,
        precision_macro=macro_prec,
        recall_macro=macro_rec,
        f1_macro=macro_f1,
        precision_weighted=weighted_prec,
        recall_weighted=weighted_rec,
        f1_weighted=weighted_f1,
        per_class=per_class,
        sample_count=n,
        status="AVAILABLE",
        source=source,
    )
