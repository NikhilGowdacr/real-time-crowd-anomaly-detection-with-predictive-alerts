"""
ROC and Precision-Recall Curve Analysis - Phase 12.

Implements Receiver Operating Characteristic (ROC) and Precision-Recall (PR)
curve generation and numerical Area Under the Curve (AUC) integration.
Strictly returns 'NOT_AVAILABLE' when ground-truth labels are absent.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.evaluation.ground_truth import EvaluationSource


@dataclass
class CurveResult:
    """Container for ROC and Precision-Recall curve evaluation."""
    curve_type: str  # "ROC" or "PR"
    fpr: List[float] = field(default_factory=list)
    tpr: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)
    auc: Optional[float] = None
    sample_count: int = 0
    status: str = "AVAILABLE"
    source: str = EvaluationSource.UNAVAILABLE.value
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "AVAILABLE"

    @property
    def is_available(self) -> bool:
        return self.status == "AVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curve_type": self.curve_type,
            "fpr": [round(x, 4) for x in self.fpr],
            "tpr": [round(x, 4) for x in self.tpr],
            "precision": [round(x, 4) for x in self.precision],
            "recall": [round(x, 4) for x in self.recall],
            "thresholds": [round(x, 4) for x in self.thresholds],
            "auc": round(self.auc, 4) if self.auc is not None else None,
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
        curve_type: str = "ROC",
        reason: str = "Ground-truth labels not configured",
        source: Any = EvaluationSource.UNAVAILABLE.value,
    ) -> "CurveResult":
        src_val = source.value if hasattr(source, "value") else str(source)
        return cls(
            curve_type=curve_type,
            status="NOT_AVAILABLE",
            source=src_val,
            reason=reason,
        )


class ROCAnalysis:
    """Computes ROC curve (FPR vs TPR) and ROC-AUC via numerical integration."""

    @staticmethod
    def compute(
        y_true_binary: Optional[Sequence[int]],
        y_scores: Optional[Sequence[float]],
        source: str = EvaluationSource.REAL_DATASET.value,
    ) -> CurveResult:
        """
        Compute ROC curve and AUC.
        Returns NOT_AVAILABLE if ground truth or scores are absent.
        """
        if y_true_binary is None or y_scores is None or len(y_true_binary) == 0 or len(y_scores) == 0:
            return CurveResult.unavailable(
                curve_type="ROC",
                reason="Ground-truth labels not configured",
                source=source,
            )

        if len(y_true_binary) != len(y_scores):
            return CurveResult.unavailable(
                curve_type="ROC",
                reason=f"Length mismatch: len(y_true)={len(y_true_binary)} vs len(y_scores)={len(y_scores)}",
                source=source,
            )

        yt = np.asarray(y_true_binary, dtype=int)
        ys = np.asarray(y_scores, dtype=float)

        num_pos = np.sum(yt == 1)
        num_neg = np.sum(yt == 0)

        # Requires both positive and negative samples for meaningful ROC
        if num_pos == 0 or num_neg == 0:
            return CurveResult.unavailable(
                curve_type="ROC",
                reason="Single-class ground truth (both positive and negative samples required for ROC)",
                source=source,
            )

        # Sort descending by score
        desc_order = np.argsort(-ys)
        yt_sorted = yt[desc_order]
        ys_sorted = ys[desc_order]

        # Distinct threshold evaluation
        distinct_indices = np.where(np.diff(ys_sorted))[0]
        threshold_indices = np.concatenate(([0], distinct_indices + 1, [len(ys_sorted)]))

        tpr_list: List[float] = [0.0]
        fpr_list: List[float] = [0.0]
        thresholds_list: List[float] = [float(ys_sorted[0] + 1.0)]

        for idx in threshold_indices[1:]:
            tp = np.sum(yt_sorted[:idx] == 1)
            fp = np.sum(yt_sorted[:idx] == 0)
            tpr_list.append(float(tp / num_pos))
            fpr_list.append(float(fp / num_neg))
            thresh_val = float(ys_sorted[idx - 1]) if idx <= len(ys_sorted) else 0.0
            thresholds_list.append(thresh_val)

        # Ensure endpoints
        if fpr_list[-1] != 1.0 or tpr_list[-1] != 1.0:
            fpr_list.append(1.0)
            tpr_list.append(1.0)
            thresholds_list.append(0.0)

        # Numerical integration using trapezoidal rule: AUC = sum((FPR[i] - FPR[i-1]) * (TPR[i] + TPR[i-1]) / 2)
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2.0
        auc = max(0.0, min(1.0, auc))

        return CurveResult(
            curve_type="ROC",
            fpr=fpr_list,
            tpr=tpr_list,
            thresholds=thresholds_list,
            auc=auc,
            sample_count=len(yt),
            status="AVAILABLE",
            source=source,
        )

    evaluate = compute


class PRAnalysis:
    """Computes Precision-Recall curve and Average Precision (PR-AUC)."""

    @staticmethod
    def compute(
        y_true_binary: Optional[Sequence[int]],
        y_scores: Optional[Sequence[float]],
        source: str = EvaluationSource.REAL_DATASET.value,
    ) -> CurveResult:
        """
        Compute PR curve and PR-AUC.
        Returns NOT_AVAILABLE if ground truth or scores are absent.
        """
        if y_true_binary is None or y_scores is None or len(y_true_binary) == 0 or len(y_scores) == 0:
            return CurveResult.unavailable(
                curve_type="PR",
                reason="Ground-truth labels not configured",
                source=source,
            )

        if len(y_true_binary) != len(y_scores):
            return CurveResult.unavailable(
                curve_type="PR",
                reason=f"Length mismatch: len(y_true)={len(y_true_binary)} vs len(y_scores)={len(y_scores)}",
                source=source,
            )

        yt = np.asarray(y_true_binary, dtype=int)
        ys = np.asarray(y_scores, dtype=float)
        num_pos = np.sum(yt == 1)

        if num_pos == 0:
            return CurveResult.unavailable(
                curve_type="PR",
                reason="No positive samples present in ground truth for PR curve",
                source=source,
            )

        # Sort descending by score
        desc_order = np.argsort(-ys)
        yt_sorted = yt[desc_order]
        ys_sorted = ys[desc_order]

        distinct_indices = np.where(np.diff(ys_sorted))[0]
        threshold_indices = np.concatenate(([0], distinct_indices + 1, [len(ys_sorted)]))

        precision_list: List[float] = [1.0]
        recall_list: List[float] = [0.0]
        thresholds_list: List[float] = [float(ys_sorted[0] + 1.0)]

        for idx in threshold_indices[1:]:
            tp = np.sum(yt_sorted[:idx] == 1)
            fp = np.sum(yt_sorted[:idx] == 0)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            rec = float(tp / num_pos)
            precision_list.append(prec)
            recall_list.append(rec)
            thresh_val = float(ys_sorted[idx - 1]) if idx <= len(ys_sorted) else 0.0
            thresholds_list.append(thresh_val)

        # Average Precision (PR-AUC) = sum((Recall[i] - Recall[i-1]) * Precision[i])
        pr_auc = 0.0
        for i in range(1, len(recall_list)):
            delta_rec = recall_list[i] - recall_list[i - 1]
            pr_auc += delta_rec * precision_list[i]
        pr_auc = max(0.0, min(1.0, pr_auc))

        return CurveResult(
            curve_type="PR",
            precision=precision_list,
            recall=recall_list,
            thresholds=thresholds_list,
            auc=pr_auc,
            sample_count=len(yt),
            status="AVAILABLE",
            source=source,
        )

    evaluate = compute
