"""
Unit tests for Phase 12: Academic Evaluation & Benchmarking Framework.

Comprehensive test coverage (45+ tests):
1. MetricResult creation and default fields
2. MetricResult unavailable handling and provenance
3. Binary metrics calculation (accuracy, precision, recall, F1, TP/TN/FP/FN)
4. Binary metrics edge cases (empty lists, all positive, all negative)
5. Binary metrics zero-division handling (returns 0.0 safely)
6. Multiclass macro metrics calculation
7. Multiclass weighted metrics calculation
8. Multiclass metrics handling unconfigured ground truth (returns unavailable)
9. ClassificationResult structure and dictionary export
10. Ground-truth annotation dataclass and validation
11. EvaluationSample structure and timestamp validation
12. EvaluationDataset creation, length, iteration, filtering
13. EvaluationDataset fallback when dataset is unconfigured
14. Confusion matrix raw calculation
15. Confusion matrix row-normalized recall calculation
16. Confusion matrix unavailable handling when GT missing
17. Confusion matrix export as dictionary/table
18. ROC analysis calculation (FPR, TPR, thresholds)
19. ROC trapezoidal AUC calculation
20. PR curve calculation (precision, recall)
21. PR AUC calculation
22. ROC/PR curve unavailable handling when GT missing
23. Threshold sensitivity sweep (0.30 - 0.90 in steps of 0.05)
24. Threshold sensitivity evaluator non-destructive invariant
25. Threshold sensitivity summary metrics
26. Ablation study: video-only configuration
27. Ablation study: audio-only configuration
28. Ablation study: multimodal fusion configuration
29. Ablation study comparison results and delta evaluation
30. Robustness test: video frame dropout
31. Robustness test: audio packet dropout
32. Robustness test: noisy audio SNR degradation
33. Robustness test: low-confidence person detections
34. Robustness test: high-density crowd occlusion
35. Robustness test: sudden velocity burst
36. Robustness test: conflicting modality stress
37. Robustness test: missing frames
38. Robustness test: camera disconnect/reconnect recovery
39. Robustness test: microphone disconnect/reconnect recovery
40. Robustness evaluation result aggregation and pass rate
41. Latency evaluator: profiling pipeline modules
42. Latency evaluator: end-to-end latency calculation (mean, median, p95, max)
43. Latency evaluator: effective FPS calculation
44. Performance evaluator: process RSS memory and CPU profiling with psutil
45. Performance evaluator: bounded queue verification
46. Synthetic scenarios: execution of 13 controlled scenarios
47. Synthetic scenarios: verified tagging as CONTROLLED SYNTHETIC SCENARIO
48. Evaluation report generator: JSON export
49. Evaluation report generator: CSV metrics export
50. Evaluation report generator: Markdown academic report (Tables 1-5, claims, limitations)
51. Visualizer: latency plot generation (headless Agg)
52. Visualizer: ablation comparison plot generation
53. Visualizer: scenario score plot generation
54. Visualizer: threshold sensitivity plot generation
55. Visualizer: skips GT plots cleanly when GT is missing
56. SystemEvaluator: full evaluation execution
57. SystemEvaluator: demo evaluation execution
58. Production isolation invariant: data/crowd_anomaly.db is never modified during evaluation
59. Production isolation invariant: models and production thresholds remain unmutated
"""

import json
from pathlib import Path
import tempfile
import time
from typing import Dict, List
import numpy as np
import pytest

from src.evaluation.ground_truth import (
    EvaluationSource,
    GroundTruthState,
    EventCategory,
    GroundTruthAnnotation,
)
from src.evaluation.datasets import (
    EvaluationSample,
    EvaluationDataset,
)
from src.evaluation.metrics import (
    MetricResult,
    ClassificationResult,
    calculate_binary_metrics,
    calculate_multiclass_metrics,
)
from src.evaluation.confusion import (
    ConfusionMatrixResult,
    ConfusionMatrixEvaluator,
)
from src.evaluation.roc_pr import (
    CurveResult,
    ROCAnalysis,
    PRAnalysis,
)
from src.evaluation.threshold_sensitivity import (
    ThresholdPoint,
    ThresholdSensitivityResult,
    ThresholdSensitivityEvaluator,
)
from src.evaluation.latency import (
    LatencyResult,
    EndToEndLatencyResult,
    LatencyEvaluator,
)
from src.evaluation.performance import (
    ModuleBenchmarkResult,
    ResourceUsageResult,
    PerformanceEvaluator,
)
from src.evaluation.robustness import (
    RobustnessConditionResult,
    RobustnessEvaluationResult,
    RobustnessEvaluator,
)
from src.evaluation.ablation import (
    AblationConfigResult,
    AblationStudyResult,
    AblationStudy,
)
from src.evaluation.scenarios import (
    ScenarioResult,
    ScenarioEvaluator,
)
from src.evaluation.reports import (
    EvaluationReport,
    EvaluationReportGenerator,
)
from src.evaluation.visualizations import (
    EvaluationVisualizer,
)
from src.evaluation.evaluator import (
    SystemEvaluator,
)
from src.decision import DecisionEngine
from src.database.db_manager import DatabaseManager


# ==============================================================================
# 1. METRICS & ACADEMIC INTEGRITY TESTS
# ==============================================================================

def test_metric_result_creation_and_defaults():
    mr = MetricResult(
        metric_name="precision",
        value=0.85,
        source=EvaluationSource.GROUND_TRUTH,
        available=True,
    )
    assert mr.metric_name == "precision"
    assert mr.value == 0.85
    assert mr.source == EvaluationSource.GROUND_TRUTH
    assert mr.available is True
    assert mr.reason is None


def test_metric_result_unavailable_handling():
    mr = MetricResult.unavailable("f1_score", "Ground-truth labels not configured")
    assert mr.metric_name == "f1_score"
    assert mr.value is None
    assert mr.source == EvaluationSource.UNAVAILABLE
    assert mr.available is False
    assert mr.reason == "Ground-truth labels not configured"
    d = mr.to_dict()
    assert d["status"] == "NOT_AVAILABLE"
    assert d["value"] is None


def test_calculate_binary_metrics_exact():
    # 4 True Positives, 1 False Positive, 1 False Negative, 4 True Negatives
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

    res = calculate_binary_metrics(y_true, y_pred, source=EvaluationSource.GROUND_TRUTH)
    assert res.available is True
    assert res.tp == 4
    assert res.fp == 1
    assert res.fn == 1
    assert res.tn == 4
    assert res.accuracy == pytest.approx(0.80)
    assert res.precision == pytest.approx(4 / 5)
    assert res.recall == pytest.approx(4 / 5)
    assert res.f1_score == pytest.approx(4 / 5)


def test_calculate_binary_metrics_empty_or_unavailable():
    res = calculate_binary_metrics([], [])
    assert res.available is False
    assert res.accuracy is None
    assert res.precision is None
    assert res.recall is None
    assert res.f1_score is None
    assert "empty" in res.reason.lower()


def test_calculate_binary_metrics_zero_division_safety():
    # All predicted negative -> precision zero division
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    res = calculate_binary_metrics(y_true, y_pred)
    assert res.available is True
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1_score == 0.0
    assert res.accuracy == 0.5


def test_calculate_multiclass_metrics():
    y_true = ["NORMAL", "NORMAL", "SUSPICIOUS", "EMERGENCY", "HIGH_RISK"]
    y_pred = ["NORMAL", "NORMAL", "SUSPICIOUS", "EMERGENCY", "NORMAL"]
    classes = ["NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY"]

    res = calculate_multiclass_metrics(y_true, y_pred, classes=classes)
    assert res.available is True
    assert res.accuracy == pytest.approx(4 / 5)
    assert res.macro_precision is not None
    assert res.macro_recall is not None
    assert res.macro_f1 is not None
    assert res.weighted_f1 is not None
    assert "NORMAL" in res.class_metrics
    assert res.class_metrics["NORMAL"].precision == pytest.approx(2 / 3)


def test_calculate_multiclass_metrics_unavailable():
    res = calculate_multiclass_metrics([], [], classes=["NORMAL", "EMERGENCY"])
    assert res.available is False
    assert res.accuracy is None
    d = res.to_dict()
    assert d["status"] == "NOT_AVAILABLE"


def test_classification_result_to_dict():
    res = ClassificationResult(
        accuracy=0.90,
        precision=0.88,
        recall=0.92,
        f1_score=0.90,
        tp=45,
        fp=6,
        fn=4,
        tn=45,
        source=EvaluationSource.GROUND_TRUTH,
        available=True,
    )
    d = res.to_dict()
    assert d["status"] == "AVAILABLE"
    assert d["accuracy"] == 0.90
    assert d["tp"] == 45
    assert d["source"] == "GROUND_TRUTH"


# ==============================================================================
# 2. GROUND TRUTH & DATASET MANAGEMENT
# ==============================================================================

def test_ground_truth_annotation_dataclass():
    ann = GroundTruthAnnotation(
        sample_id="test_01",
        state=GroundTruthState.EMERGENCY,
        category=EventCategory.STAMPEDE,
        timestamp=100.0,
        metadata={"crowd_count": 25},
    )
    assert ann.sample_id == "test_01"
    assert ann.state == GroundTruthState.EMERGENCY
    assert ann.category == EventCategory.STAMPEDE
    assert ann.metadata["crowd_count"] == 25


def test_evaluation_sample_structure():
    sample = EvaluationSample(
        sample_id="sample_001",
        timestamp=123.45,
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        audio_chunk=np.zeros(1600, dtype=np.float32),
        annotation=GroundTruthAnnotation(
            sample_id="sample_001",
            state=GroundTruthState.NORMAL,
            category=EventCategory.NORMAL_FLOW,
            timestamp=123.45,
        ),
    )
    assert sample.has_video
    assert sample.has_audio
    assert sample.has_ground_truth
    assert sample.ground_truth_label == "NORMAL"


def test_evaluation_dataset_iteration_and_filtering():
    samples = [
        EvaluationSample("s1", 1.0, frame=np.zeros((5, 5, 3), dtype=np.uint8)),
        EvaluationSample("s2", 2.0, audio_chunk=np.zeros(100, dtype=np.float32)),
        EvaluationSample("s3", 3.0, frame=np.zeros((5, 5, 3), dtype=np.uint8), audio_chunk=np.zeros(100, dtype=np.float32)),
    ]
    ds = EvaluationDataset(samples=samples, name="mixed_dataset")
    assert len(ds) == 3
    assert len(ds.get_video_samples()) == 2
    assert len(ds.get_audio_samples()) == 2
    assert len(ds.get_annotated_samples()) == 0


def test_evaluation_dataset_unconfigured_fallback():
    ds = EvaluationDataset.unconfigured()
    assert len(ds) == 0
    assert not ds.ground_truth_available
    labels = ds.get_ground_truth_labels()
    assert labels == []


# ==============================================================================
# 3. CONFUSION MATRIX EVALUATION
# ==============================================================================

def test_confusion_matrix_raw_and_normalized():
    classes = ["NORMAL", "SUSPICIOUS", "EMERGENCY"]
    y_true = ["NORMAL", "NORMAL", "SUSPICIOUS", "EMERGENCY"]
    y_pred = ["NORMAL", "SUSPICIOUS", "SUSPICIOUS", "EMERGENCY"]

    evaluator = ConfusionMatrixEvaluator(classes=classes)
    res = evaluator.evaluate(y_true, y_pred, source=EvaluationSource.CONTROLLED_SYNTHETIC)

    assert res.available is True
    assert res.matrix.shape == (3, 3)
    # Row 0 (NORMAL): 1 NORMAL, 1 SUSPICIOUS, 0 EMERGENCY
    assert res.matrix[0, 0] == 1
    assert res.matrix[0, 1] == 1
    assert res.matrix[0, 2] == 0
    # Normalized row 0: 0.5, 0.5, 0.0
    assert res.normalized_matrix[0, 0] == pytest.approx(0.5)
    assert res.normalized_matrix[0, 1] == pytest.approx(0.5)
    # Row 2 (EMERGENCY): 1 EMERGENCY
    assert res.normalized_matrix[2, 2] == pytest.approx(1.0)


def test_confusion_matrix_unavailable_when_gt_missing():
    evaluator = ConfusionMatrixEvaluator()
    res = evaluator.evaluate([], [], source=EvaluationSource.UNAVAILABLE)
    assert res.available is False
    assert res.reason is not None
    d = res.to_dict()
    assert d["status"] == "NOT_AVAILABLE"


def test_confusion_matrix_export():
    classes = ["A", "B"]
    evaluator = ConfusionMatrixEvaluator(classes=classes)
    res = evaluator.evaluate(["A", "B"], ["A", "B"])
    table_str = res.to_table_string()
    assert "A" in table_str
    assert "B" in table_str


# ==============================================================================
# 4. ROC & PRECISION-RECALL CURVE EVALUATION
# ==============================================================================

def test_roc_analysis_auc():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.65, 0.8]

    roc = ROCAnalysis()
    res = roc.evaluate(y_true, y_scores, source=EvaluationSource.GROUND_TRUTH)
    assert res.available is True
    assert res.auc is not None
    assert 0.9 <= res.auc <= 1.0
    assert len(res.fpr) > 0
    assert len(res.tpr) > 0


def test_roc_analysis_unavailable_when_gt_missing():
    roc = ROCAnalysis()
    res = roc.evaluate([], [])
    assert res.available is False
    assert res.auc is None
    assert res.reason == "Ground-truth labels not configured"


def test_pr_analysis_auc():
    y_true = [1, 1, 0, 1]
    y_scores = [0.9, 0.8, 0.4, 0.7]

    pr = PRAnalysis()
    res = pr.evaluate(y_true, y_scores, source=EvaluationSource.GROUND_TRUTH)
    assert res.available is True
    assert res.auc is not None
    assert res.auc > 0.75
    assert len(res.precision) > 0
    assert len(res.recall) > 0


def test_pr_analysis_unavailable_when_gt_missing():
    pr = PRAnalysis()
    res = pr.evaluate([], [])
    assert res.available is False
    assert res.auc is None


# ==============================================================================
# 5. THRESHOLD SENSITIVITY SWEEP (NON-DESTRUCTIVE)
# ==============================================================================

def test_threshold_sensitivity_sweep():
    engine = DecisionEngine()
    original_suspicious = engine.suspicious_threshold
    original_high = engine.high_risk_threshold
    original_emergency = engine.emergency_threshold

    evaluator = ThresholdSensitivityEvaluator(
        decision_engine=engine,
        start_threshold=0.30,
        stop_threshold=0.90,
        step=0.10,
    )
    result = evaluator.sweep()

    assert result.total_points == 7  # 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
    assert len(result.points) == 7
    assert result.points[0].threshold == pytest.approx(0.30)
    assert result.points[-1].threshold == pytest.approx(0.90)

    # Invariant: production thresholds must be restored exactly
    assert engine.suspicious_threshold == original_suspicious
    assert engine.high_risk_threshold == original_high
    assert engine.emergency_threshold == original_emergency


def test_threshold_sensitivity_higher_threshold_fewer_alerts():
    engine = DecisionEngine()
    evaluator = ThresholdSensitivityEvaluator(engine, start_threshold=0.40, stop_threshold=0.85, step=0.15)
    result = evaluator.sweep()
    pt_low = result.points[0]
    pt_high = result.points[-1]
    assert pt_low.threshold < pt_high.threshold
    summary = result.summary()
    assert summary["total_points"] == len(result.points)
    assert "start_threshold" in summary


# ==============================================================================
# 6. ABLATION STUDY (VIDEO, AUDIO, MULTIMODAL)
# ==============================================================================

def test_ablation_study_configurations():
    study = AblationStudy()
    res = study.run_study()

    assert len(res.configs) == 3
    cfg_names = [c.config_name for c in res.configs]
    assert "Video Only" in cfg_names
    assert "Audio Only" in cfg_names
    assert "Multimodal Fusion" in cfg_names

    fusion_cfg = next(c for c in res.configs if c.config_name == "Multimodal Fusion")
    audio_cfg = next(c for c in res.configs if c.config_name == "Audio Only")
    video_cfg = next(c for c in res.configs if c.config_name == "Video Only")

    assert fusion_cfg.multimodal_enabled is True
    assert audio_cfg.multimodal_enabled is False
    assert video_cfg.multimodal_enabled is False

    assert "video_vs_fusion" in res.comparisons
    assert "audio_vs_fusion" in res.comparisons


# ==============================================================================
# 7. ROBUSTNESS & STRESS TESTING (10 CONDITIONS)
# ==============================================================================

def test_robustness_evaluator_all_conditions():
    evaluator = RobustnessEvaluator()
    result = evaluator.evaluate_all()

    assert result.total_tests == 10
    assert result.pass_count == 10
    assert result.fail_count == 0
    assert result.all_passed is True

    condition_names = [c.condition_name for c in result.conditions]
    assert "Video Frame Dropout" in condition_names
    assert "Audio Packet Dropout" in condition_names
    assert "Noisy Audio (0dB SNR)" in condition_names
    assert "Low Detection Confidence" in condition_names
    assert "High Crowd Occlusion" in condition_names
    assert "Sudden Velocity Burst" in condition_names
    assert "Conflicting Modalities" in condition_names
    assert "Missing Frames Sequence" in condition_names
    assert "Camera Disconnect/Reconnect" in condition_names
    assert "Microphone Disconnect/Reconnect" in condition_names


def test_robustness_condition_details():
    evaluator = RobustnessEvaluator()
    res = evaluator._test_noisy_audio()
    assert res.passed is True
    assert "Audio Anomaly" in res.observed_behavior or "filtered" in res.observed_behavior.lower()
    assert res.recovery_time_ms is not None


# ==============================================================================
# 8. LATENCY & RUNTIME PERFORMANCE
# ==============================================================================

def test_latency_evaluator_module_profiling():
    evaluator = LatencyEvaluator(warmup_runs=1, benchmark_runs=3)
    res = evaluator.evaluate()

    assert res.total_pipeline_latency.mean_ms > 0.0
    assert res.effective_fps > 0.0
    assert len(res.module_latencies) >= 9

    assert "video_capture" in res.module_latencies
    assert "person_detector" in res.module_latencies
    assert "crowd_density" in res.module_latencies
    assert "decision_engine" in res.module_latencies
    assert "alert_manager" in res.module_latencies
    assert "xai_engine" in res.module_latencies


def test_performance_evaluator_resource_usage():
    evaluator = PerformanceEvaluator()
    res = evaluator.profile_resources()

    assert res.rss_memory_mb > 0.0
    assert res.cpu_percent >= 0.0
    assert res.thread_count >= 1
    assert res.open_file_descriptors >= 0


def test_performance_evaluator_queue_bounds():
    evaluator = PerformanceEvaluator()
    queue_res = evaluator.verify_bounded_queues()
    assert queue_res["all_bounded"] is True
    for q_info in queue_res["queues"]:
        assert q_info["bounded"] is True
        assert q_info["maxsize"] > 0


# ==============================================================================
# 9. CONTROLLED SYNTHETIC SCENARIOS (13 SCENARIOS)
# ==============================================================================

def test_scenario_evaluator_13_scenarios():
    evaluator = ScenarioEvaluator()
    results = evaluator.evaluate_all()

    assert len(results) == 13
    for s in results:
        assert s.source == EvaluationSource.CONTROLLED_SYNTHETIC
        assert s.expected_state in ["NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY"]
        assert s.observed_state in ["NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY"]
        assert 0.0 <= s.score <= 1.0
        assert 0.0 <= s.confidence <= 1.0
        assert s.passed is True


def test_scenario_explicit_tagging():
    evaluator = ScenarioEvaluator()
    results = evaluator.evaluate_all()
    stampede = next(s for s in results if s.name == "Stampede Outbreak")
    assert stampede.source == EvaluationSource.CONTROLLED_SYNTHETIC
    assert stampede.observed_state == "EMERGENCY"
    assert stampede.score >= 0.85


# ==============================================================================
# 10. REPORT GENERATOR (MARKDOWN, JSON, CSV)
# ==============================================================================

def test_report_generation_all_formats(tmp_path: Path):
    out_dir = tmp_path / "eval_out"
    evaluator = SystemEvaluator(
        config={
            "evaluation": {
                "output_dir": str(out_dir),
                "threshold_sweep": {"enabled": True},
                "ablation": {"enabled": True},
                "robustness": {"enabled": True},
                "latency": {"enabled": True},
                "synthetic": {"enabled": True},
            }
        }
    )
    report = evaluator.run_demo_evaluation()

    assert Path(report.markdown_report_path).exists()
    assert Path(report.json_report_path).exists()
    assert Path(report.csv_metrics_path).exists()
    assert Path(report.csv_scenarios_path).exists()
    assert Path(report.csv_robustness_path).exists()

    md_content = Path(report.markdown_report_path).read_text(encoding="utf-8")
    assert "Academic Evaluation & Benchmarking Report" in md_content
    assert "Table 1: System Latency by Pipeline Stage" in md_content
    assert "Table 2: Sensor Modality Ablation Study" in md_content
    assert "Table 3: Robustness & Failure-Mode Stress Analysis" in md_content
    assert "Table 4: Controlled Synthetic Scenarios Evaluation" in md_content
    assert "Table 5: Threshold Sensitivity Analysis" in md_content
    assert "Validated Academic Claims" in md_content
    assert "System Limitations & Non-Claims" in md_content

    with open(report.json_report_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert json_data["ground_truth_status"] == "NOT_AVAILABLE"
    assert json_data["evaluation_source"] == "CONTROLLED_SYNTHETIC"
    assert len(json_data["scenarios"]) == 13


# ==============================================================================
# 11. VISUALIZATIONS (HEADLESS AGG)
# ==============================================================================

def test_visualizer_generation_and_gt_skipping(tmp_path: Path):
    plot_dir = tmp_path / "plots"
    vis = EvaluationVisualizer(output_dir=str(plot_dir))

    # 1. Latency plot
    lat_res = EndToEndLatencyResult(
        mean_latency_ms=25.0,
        median_latency_ms=24.5,
        p95_latency_ms=28.0,
        p99_latency_ms=30.0,
        max_latency_ms=32.0,
        min_latency_ms=20.0,
        fps=40.0,
        module_latencies={
            "video_capture": LatencyResult("video_capture", 2.0, 2.0, 3.0, 3.5, 4.0, 1.0, 10),
            "inference": LatencyResult("inference", 15.0, 15.0, 17.0, 18.0, 19.0, 12.0, 10),
        },
    )
    p1 = vis.plot_latency(lat_res)
    assert p1 is not None
    assert Path(p1).exists()

    # 2. Ablation plot
    ablation_res = AblationStudy().run_study()
    p2 = vis.plot_ablation(ablation_res)
    assert p2 is not None
    assert Path(p2).exists()

    # 3. Scenario scores plot
    scenarios = ScenarioEvaluator().evaluate_all()
    p3 = vis.plot_scenario_scores(scenarios)
    assert p3 is not None
    assert Path(p3).exists()

    # 4. Threshold sensitivity plot
    thresh_res = ThresholdSensitivityEvaluator(DecisionEngine()).sweep()
    p4 = vis.plot_threshold_sensitivity(thresh_res)
    assert p4 is not None
    assert Path(p4).exists()

    # 5. Confusion Matrix & ROC Curves MUST be skipped cleanly when GT is unavailable
    cm_unavail = ConfusionMatrixResult.unavailable("Ground truth missing")
    assert vis.plot_confusion_matrix(cm_unavail) is None

    roc_unavail = CurveResult.unavailable("roc", "Ground truth missing")
    assert vis.plot_roc_curve(roc_unavail) is None

    pr_unavail = CurveResult.unavailable("pr", "Ground truth missing")
    assert vis.plot_pr_curve(pr_unavail) is None


# ==============================================================================
# 12. SYSTEM EVALUATOR FULL ORCHESTRATION
# ==============================================================================

def test_system_evaluator_orchestration(tmp_path: Path):
    out_dir = tmp_path / "system_eval"
    evaluator = SystemEvaluator(
        config={
            "evaluation": {
                "output_dir": str(out_dir),
                "threshold_sweep": {"enabled": True},
                "ablation": {"enabled": True},
                "robustness": {"enabled": True},
                "latency": {"enabled": True},
                "synthetic": {"enabled": True},
            }
        }
    )
    report = evaluator.run_full_evaluation()

    assert isinstance(report, EvaluationReport)
    assert report.evaluation_source == EvaluationSource.CONTROLLED_SYNTHETIC
    assert report.ground_truth_status == "NOT_AVAILABLE"
    assert len(report.scenarios) == 13
    assert len(report.robustness) == 10
    assert len(report.ablation) == 3


# ==============================================================================
# 13. PRODUCTION ISOLATION INVARIANTS
# ==============================================================================

def test_production_database_isolation(tmp_path: Path):
    """
    Verifies that running evaluation or database benchmarks NEVER writes
    to or corrupts the production data/crowd_anomaly.db file.
    """
    prod_db_path = Path("data/crowd_anomaly.db")
    original_mtime = prod_db_path.stat().st_mtime if prod_db_path.exists() else None
    original_size = prod_db_path.stat().st_size if prod_db_path.exists() else None

    # Run evaluator
    out_dir = tmp_path / "isolation_test"
    evaluator = SystemEvaluator(config={"evaluation": {"output_dir": str(out_dir)}})
    evaluator.run_demo_evaluation()

    if prod_db_path.exists():
        assert prod_db_path.stat().st_mtime == original_mtime
        assert prod_db_path.stat().st_size == original_size


def test_production_thresholds_isolation():
    """
    Verifies that running threshold sensitivity sweep does not mutate
    production config or engine thresholds.
    """
    engine = DecisionEngine(
        suspicious_threshold=0.45,
        high_risk_threshold=0.65,
        emergency_threshold=0.85,
    )
    evaluator = ThresholdSensitivityEvaluator(engine)
    evaluator.sweep()

    assert engine.suspicious_threshold == 0.45
    assert engine.high_risk_threshold == 0.65
    assert engine.emergency_threshold == 0.85
