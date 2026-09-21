"""
Threshold Sensitivity Analysis - Phase 12.

Sweeps detection thresholds across candidate operational ranges to analyze
trade-offs between false alarms (FP) and missed incidents (FN).
Guarantees production safety: never mutates production configuration thresholds.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.evaluation.ground_truth import EvaluationSource


@dataclass
class ThresholdPoint:
    """Evaluation metrics at a single threshold value."""
    threshold: float
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tn: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    trigger_count: int = 0
    trigger_rate: float = 0.0
    alert_count: Optional[int] = None
    emergency_count: Optional[int] = None
    alert_rate: Optional[float] = None
    mean_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.alert_count is None:
            self.alert_count = self.trigger_count
        if self.emergency_count is None:
            self.emergency_count = 0
        if self.alert_rate is None:
            self.alert_rate = self.trigger_rate
        if self.mean_confidence is None:
            self.mean_confidence = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": round(self.threshold, 2),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1": round(self.f1, 4) if self.f1 is not None else None,
            "trigger_count": self.trigger_count,
            "trigger_rate": round(self.trigger_rate, 4),
            "alert_count": self.alert_count,
            "emergency_count": self.emergency_count,
            "alert_rate": round(self.alert_rate, 4) if self.alert_rate is not None else None,
            "mean_confidence": round(self.mean_confidence, 4) if self.mean_confidence is not None else None,
        }


@dataclass
class ThresholdSensitivityResult:
    """Comprehensive outcome of a threshold sweep experiment."""
    points: List[ThresholdPoint] = field(default_factory=list)
    baseline_threshold: float = 0.50
    has_ground_truth: bool = False
    status: str = "AVAILABLE"
    source: str = EvaluationSource.UNAVAILABLE.value
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_points(self) -> int:
        return len(self.points)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_points": len(self.points),
            "start_threshold": self.points[0].threshold if self.points else None,
            "stop_threshold": self.points[-1].threshold if self.points else None,
            "baseline_threshold": self.baseline_threshold,
            "has_ground_truth": self.has_ground_truth,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "baseline_threshold": self.baseline_threshold,
            "has_ground_truth": self.has_ground_truth,
            "status": self.status,
            "source": self.source,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ThresholdSensitivityEvaluator:
    """
    Evaluates sensitivity and operational behavior across candidate thresholds.
    Operates non-destructively without modifying production system configurations.
    """

    DEFAULT_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    def __init__(
        self,
        decision_engine: Optional[Any] = None,
        start_threshold: float = 0.30,
        stop_threshold: float = 0.90,
        step: float = 0.05,
    ) -> None:
        self.decision_engine = decision_engine
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.step = step

    def sweep(
        self,
        anomaly_scores: Optional[Sequence[float]] = None,
        ground_truth_binary: Optional[Sequence[int]] = None,
    ) -> ThresholdSensitivityResult:
        """Runs non-destructive sweep over configured range."""
        curr = self.start_threshold
        sweep_vals: List[float] = []
        while curr <= self.stop_threshold + 1e-6:
            sweep_vals.append(round(curr, 4))
            curr += self.step

        scores = anomaly_scores or [0.10, 0.25, 0.42, 0.55, 0.68, 0.82, 0.94]
        return self.evaluate(
            anomaly_scores=scores,
            ground_truth_binary=ground_truth_binary,
            threshold_sweep=sweep_vals,
        )

    def evaluate(
        self,
        anomaly_scores: Sequence[float],
        ground_truth_binary: Optional[Sequence[int]] = None,
        threshold_sweep: Optional[List[float]] = None,
        baseline_threshold: float = 0.50,
        source: str = EvaluationSource.CONTROLLED_TEST.value,
    ) -> ThresholdSensitivityResult:
        """
        Executes threshold sweep evaluation.
        If ground_truth_binary is None or empty: evaluates operational trigger rates without computing fake precision/recall.
        If ground_truth_binary is provided: calculates TP, FP, FN, Precision, Recall, F1 for each threshold.
        """
        sweep = threshold_sweep or self.DEFAULT_SWEEP
        scores = np.asarray(anomaly_scores, dtype=float)
        n = len(scores)

        if n == 0:
            return ThresholdSensitivityResult(
                points=[],
                baseline_threshold=baseline_threshold,
                has_ground_truth=False,
                status="NOT_AVAILABLE",
                source=source,
                reason="No anomaly scores provided for threshold evaluation",
            )

        has_gt = (
            ground_truth_binary is not None
            and len(ground_truth_binary) == n
            and len(ground_truth_binary) > 0
        )

        gt = np.asarray(ground_truth_binary, dtype=int) if has_gt else None
        points: List[ThresholdPoint] = []

        for th in sweep:
            pred_binary = (scores >= th).astype(int)
            triggers = int(np.sum(pred_binary))
            trig_rate = triggers / n if n > 0 else 0.0

            if has_gt and gt is not None:
                tp = int(np.sum((gt == 1) & (pred_binary == 1)))
                tn = int(np.sum((gt == 0) & (pred_binary == 0)))
                fp = int(np.sum((gt == 0) & (pred_binary == 1)))
                fn = int(np.sum((gt == 1) & (pred_binary == 0)))

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                point = ThresholdPoint(
                    threshold=th,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=tn,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                    trigger_count=triggers,
                    trigger_rate=trig_rate,
                )
            else:
                point = ThresholdPoint(
                    threshold=th,
                    trigger_count=triggers,
                    trigger_rate=trig_rate,
                )

            points.append(point)

        return ThresholdSensitivityResult(
            points=points,
            baseline_threshold=baseline_threshold,
            has_ground_truth=has_gt,
            status="AVAILABLE",
            source=source if has_gt else EvaluationSource.PERFORMANCE_BENCHMARK.value,
            reason=None if has_gt else "Ground-truth unconfigured: computed operational trigger rates only",
        )
