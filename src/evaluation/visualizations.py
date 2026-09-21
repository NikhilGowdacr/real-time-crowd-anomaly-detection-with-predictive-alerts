"""
Academic Visualization Generator - Phase 12.

Generates publication-quality charts using Matplotlib with a headless backend:
1. Latency by pipeline module
2. Sensor ablation comparison
3. Scenario anomaly score distribution
4. Threshold sensitivity sweep
5. Confusion matrix heatmap (ONLY when valid ground truth exists)
6. ROC curve (ONLY when valid ground truth exists)
7. Precision-Recall curve (ONLY when valid ground truth exists)

Strict academic integrity: never generates fake ROC or confusion plots when ground truth is absent.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.ablation import AblationStudyResult
from src.evaluation.confusion import ConfusionMatrixResult
from src.evaluation.latency import EndToEndLatencyResult
from src.evaluation.roc_pr import CurveResult
from src.evaluation.scenarios import ScenarioResult
from src.evaluation.threshold_sensitivity import ThresholdSensitivityResult


class EvaluationVisualizer:
    """Generates academic visualization figures saved to reports/evaluation/plots/."""

    def __init__(self, output_dir: str = "reports/evaluation/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_latency_by_module(self, latency: Optional[EndToEndLatencyResult]) -> Optional[str]:
        """Generates horizontal bar chart of mean and p95 latencies across modules."""
        if not latency or not latency.module_latencies:
            return None

        modules = list(latency.module_latencies.keys())
        means = [latency.module_latencies[m].mean_ms for m in modules]
        p95s = [latency.module_latencies[m].p95_ms for m in modules]

        clean_labels = [m.replace("_", " ").title() for m in modules]
        y_pos = np.arange(len(modules))
        height = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(y_pos - height / 2, means, height, label="Mean Latency (ms)", color="#2563EB")
        ax.barh(y_pos + height / 2, p95s, height, label="P95 Latency (ms)", color="#F59E0B")

        ax.set_ylabel("Subsystem Component")
        ax.set_xlabel("Execution Latency (ms)")
        ax.set_title("Processing Latency by Surveillance Module")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(clean_labels)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / "latency_by_module.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return str(save_path)

    def plot_ablation_comparison(self, ablation: Optional[AblationStudyResult]) -> Optional[str]:
        """Generates comparative bar chart for unimodal vs multimodal configurations."""
        if not ablation:
            return None

        configs = ["Video-Only", "Audio-Only", "Multimodal Fusion"]
        results = [ablation.video_only, ablation.audio_only, ablation.multimodal_fusion]
        scores = [r.mean_score for r in results]
        alerts = [r.alert_count for r in results]
        emergencies = [r.emergency_count for r in results]

        x = np.arange(len(configs))
        width = 0.25

        fig, ax1 = plt.subplots(figsize=(9, 5))
        rects1 = ax1.bar(x - width, scores, width, label="Mean Anomaly Score", color="#3B82F6")

        ax2 = ax1.twinx()
        rects2 = ax2.bar(x, alerts, width, label="Alert Count", color="#F59E0B")
        rects3 = ax2.bar(x + width, emergencies, width, label="Emergency Count", color="#EF4444")

        ax1.set_ylabel("Anomaly Score [0.0 - 1.0]", color="#3B82F6")
        ax2.set_ylabel("Alert Frequency (Count)", color="#D97706")
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.set_title("Sensor Modality Ablation: Unimodal vs Multimodal Fusion")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        save_path = self.output_dir / "ablation_comparison.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return str(save_path)

    def plot_scenario_scores(self, scenarios: Sequence[ScenarioResult]) -> Optional[str]:
        """Plots observed anomaly scores across the 13 controlled scenarios."""
        if not scenarios:
            return None

        names = [s.name for s in scenarios]
        scores = [s.score for s in scenarios]
        states = [s.observed_state for s in scenarios]

        state_color_map = {
            "NORMAL": "#10B981",
            "SUSPICIOUS": "#F59E0B",
            "HIGH_RISK": "#EA580C",
            "EMERGENCY": "#EF4444",
            "RECOVERY": "#06B6D4",
        }
        bar_colors = [state_color_map.get(st, "#64748B") for st in states]

        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(scenarios))
        bars = ax.barh(y_pos, scores, color=bar_colors, edgecolor="#1E293B")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Observed Composite Anomaly Score")
        ax.set_title("Anomaly Score & State Classification across 13 Controlled Scenarios")
        ax.set_xlim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")

        # Threshold lines
        ax.axvline(0.40, color="#F59E0B", linestyle=":", label="Suspicious Threshold (0.40)")
        ax.axvline(0.70, color="#EA580C", linestyle="--", label="High-Risk Threshold (0.70)")
        ax.axvline(0.85, color="#EF4444", linestyle="-.", label="Emergency Threshold (0.85)")
        ax.legend(loc="lower right")

        plt.tight_layout()
        save_path = self.output_dir / "scenario_anomaly_scores.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return str(save_path)

    def plot_threshold_sensitivity(self, result: Optional[ThresholdSensitivityResult]) -> Optional[str]:
        """Plots operational trigger rates across the threshold sweep."""
        if not result or not result.points:
            return None

        thresholds = [p.threshold for p in result.points]
        trigger_rates = [p.trigger_rate * 100.0 for p in result.points]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, trigger_rates, marker="o", linewidth=2, color="#2563EB", label="Alert Trigger Rate (%)")

        if result.has_ground_truth:
            f1s = [p.f1 * 100.0 if p.f1 is not None else 0.0 for p in result.points]
            ax.plot(thresholds, f1s, marker="s", linewidth=2, linestyle="--", color="#10B981", label="F1-Score (%)")

        ax.axvline(result.baseline_threshold, color="#EF4444", linestyle=":", label=f"Baseline Threshold ({result.baseline_threshold})")
        ax.set_xlabel("Decision Anomaly Threshold")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Threshold Sensitivity & Operational Trigger Trade-off")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        save_path = self.output_dir / "threshold_sensitivity.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return str(save_path)

    def plot_confusion_matrix(self, cm_res: Optional[ConfusionMatrixResult]) -> Optional[str]:
        """Generates heatmap ONLY when ground truth is verified and matrix is AVAILABLE."""
        if cm_res is None or cm_res.status != "AVAILABLE" or cm_res.matrix is None:
            # Respect academic integrity: skip plot when GT is missing
            return None

        matrix = np.array(cm_res.matrix)
        classes = cm_res.classes

        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)
        fig.colorbar(cax)

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(matrix[i, j]), va="center", ha="center", color="black", fontsize=11)

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="left")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted State")
        ax.set_ylabel("True Ground-Truth State")
        ax.set_title("State Classification Confusion Matrix", pad=20)

        plt.tight_layout()
        save_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return str(save_path)

    def plot_roc_curve(self, roc_res: Optional[CurveResult]) -> Optional[str]:
        """Plots ROC curve ONLY when ground truth is verified."""
        if roc_res is None or roc_res.status != "AVAILABLE" or roc_res.fpr is None or len(roc_res.fpr) == 0:
            return None

        fig, ax = plt.subplots(figsize=(6, 6))
        auc_str = f"{roc_res.auc:.3f}" if roc_res.auc is not None else "N/A"
        ax.plot(roc_res.fpr, roc_res.tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {auc_str})")
        ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", label="Random Classifier (AUC = 0.500)")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / "roc_curve.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return str(save_path)

    def plot_pr_curve(self, pr_res: Optional[CurveResult]) -> Optional[str]:
        """Plots Precision-Recall curve ONLY when ground truth is verified."""
        if pr_res is None or pr_res.status != "AVAILABLE":
            return None
        recall = getattr(pr_res, "recall", None) or getattr(pr_res, "tpr", None)
        precision = getattr(pr_res, "precision", None) or getattr(pr_res, "fpr", None)
        if recall is None or precision is None or len(recall) == 0:
            return None

        fig, ax = plt.subplots(figsize=(6, 6))
        auc_str = f"{pr_res.auc:.3f}" if pr_res.auc is not None else "N/A"
        ax.plot(recall, precision, color="#10B981", lw=2, label=f"PR Curve (AUC = {auc_str})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall (PR) Curve")
        ax.legend(loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / "pr_curve.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return str(save_path)

    def plot_all_available(
        self,
        latency: Optional[EndToEndLatencyResult] = None,
        ablation: Optional[AblationStudyResult] = None,
        scenarios: Optional[Sequence[ScenarioResult]] = None,
        threshold: Optional[ThresholdSensitivityResult] = None,
        confusion: Optional[ConfusionMatrixResult] = None,
        roc: Optional[CurveResult] = None,
        pr: Optional[CurveResult] = None,
    ) -> Dict[str, str]:
        """Generates all applicable figures and returns dictionary of generated file paths."""
        paths: Dict[str, str] = {}

        p_lat = self.plot_latency_by_module(latency)
        if p_lat:
            paths["latency"] = p_lat

        p_abl = self.plot_ablation_comparison(ablation)
        if p_abl:
            paths["ablation"] = p_abl

        p_scen = self.plot_scenario_scores(scenarios or [])
        if p_scen:
            paths["scenarios"] = p_scen

        p_thresh = self.plot_threshold_sensitivity(threshold)
        if p_thresh:
            paths["threshold"] = p_thresh

        p_cm = self.plot_confusion_matrix(confusion)
        if p_cm:
            paths["confusion"] = p_cm

        p_roc = self.plot_roc_curve(roc)
        if p_roc:
            paths["roc"] = p_roc

        return paths

    plot_latency = plot_latency_by_module
    plot_ablation = plot_ablation_comparison
