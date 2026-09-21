"""
Evaluation & Benchmarking Framework - Phase 12.

Provides an academic-grade evaluation suite measuring detection performance,
multimodal fusion, alert behavior, end-to-end latency, sensor robustness,
hardware resource consumption, and database persistence.
Strictly prohibits metric fabrication when ground truth is unconfigured.
"""

from src.evaluation.ablation import (
    AblationConfigResult,
    AblationStudy,
    AblationStudyResult,
)
from src.evaluation.confusion import (
    ConfusionMatrixEvaluator,
    ConfusionMatrixResult,
)
from src.evaluation.datasets import (
    EvaluationDataset,
    EvaluationSample,
)
from src.evaluation.evaluator import SystemEvaluator
from src.evaluation.ground_truth import (
    EventCategory,
    EvaluationSource,
    GroundTruthAnnotation,
    GroundTruthState,
)
from src.evaluation.latency import (
    EndToEndLatencyResult,
    LatencyEvaluator,
    LatencyResult,
)
from src.evaluation.metrics import (
    ClassificationResult,
    MetricResult,
    calculate_binary_metrics,
    calculate_multiclass_metrics,
)
from src.evaluation.performance import (
    ModuleBenchmarkResult,
    PerformanceEvaluator,
    ResourceUsageResult,
)
from src.evaluation.reports import (
    EvaluationReport,
    EvaluationReportGenerator,
)
from src.evaluation.robustness import (
    RobustnessConditionResult,
    RobustnessEvaluationResult,
    RobustnessEvaluator,
)
from src.evaluation.roc_pr import (
    CurveResult,
    PRAnalysis,
    ROCAnalysis,
)
from src.evaluation.scenarios import (
    ScenarioEvaluator,
    ScenarioResult,
)
from src.evaluation.threshold_sensitivity import (
    ThresholdPoint,
    ThresholdSensitivityEvaluator,
    ThresholdSensitivityResult,
)
from src.evaluation.visualizations import EvaluationVisualizer

__all__ = [
    "EvaluationSource",
    "GroundTruthState",
    "EventCategory",
    "GroundTruthAnnotation",
    "EvaluationSample",
    "EvaluationDataset",
    "MetricResult",
    "ClassificationResult",
    "calculate_binary_metrics",
    "calculate_multiclass_metrics",
    "ConfusionMatrixResult",
    "ConfusionMatrixEvaluator",
    "CurveResult",
    "ROCAnalysis",
    "PRAnalysis",
    "ThresholdPoint",
    "ThresholdSensitivityResult",
    "ThresholdSensitivityEvaluator",
    "LatencyResult",
    "EndToEndLatencyResult",
    "LatencyEvaluator",
    "ModuleBenchmarkResult",
    "ResourceUsageResult",
    "PerformanceEvaluator",
    "RobustnessConditionResult",
    "RobustnessEvaluationResult",
    "RobustnessEvaluator",
    "AblationConfigResult",
    "AblationStudyResult",
    "AblationStudy",
    "ScenarioResult",
    "ScenarioEvaluator",
    "EvaluationReport",
    "EvaluationReportGenerator",
    "EvaluationVisualizer",
    "SystemEvaluator",
]
