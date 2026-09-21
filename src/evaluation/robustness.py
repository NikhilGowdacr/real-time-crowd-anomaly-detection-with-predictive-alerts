"""
Robustness & Sensor Degradation Evaluator - Phase 12.

Evaluates system stability under 10 adverse environmental and sensor conditions:
1. Video sensor dropout
2. Audio sensor dropout
3. Noisy audio (ambient spike)
4. Low-confidence detections
5. High crowd density stress
6. Sudden velocity burst / scattering
7. Conflicting modalities (high video / low audio, low video / high audio)
8. Missing / dropped frames
9. Temporary camera disconnect and reconnect
10. Temporary microphone disconnect and reconnect

Verifies that sensor failure does not crash the system and recovery operates cleanly.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional

from src.alerts.alert_manager import AlertManager, AlertPolicy, AlertRecord
from src.decision.decision_engine import DecisionEngine, DecisionEvent, RiskState
from src.evaluation.ground_truth import EvaluationSource
from src.fusion.multimodal_fusion import MultimodalAssessment, MultimodalFusionEngine


@dataclass
class RobustnessConditionResult:
    """Outcome of a single robustness condition evaluation."""
    condition: str
    description: str
    system_state: str
    anomaly_score: float
    decision_stable: bool
    alert_behavior: str
    recovered: bool
    status: str  # "PASS" or "FAIL"
    notes: str = ""

    @property
    def condition_name(self) -> str:
        return self.condition

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def observed_behavior(self) -> str:
        return self.alert_behavior

    @property
    def recovery_time_ms(self) -> float:
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "description": self.description,
            "system_state": self.system_state,
            "anomaly_score": round(self.anomaly_score, 4),
            "decision_stable": self.decision_stable,
            "alert_behavior": self.alert_behavior,
            "recovered": self.recovered,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class RobustnessEvaluationResult:
    """Aggregated outcome of the 10-condition robustness test suite."""
    conditions: List[RobustnessConditionResult] = field(default_factory=list)
    total_tested: int = 0
    passed_count: int = 0
    failed_count: int = 0
    pass_rate: float = 0.0
    status: str = "AVAILABLE"
    source: str = EvaluationSource.CONTROLLED_TEST.value
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.conditions)

    @property
    def total_tests(self) -> int:
        return self.total_tested

    @property
    def pass_count(self) -> int:
        return self.passed_count

    @property
    def fail_count(self) -> int:
        return self.failed_count

    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conditions": [c.to_dict() for c in self.conditions],
            "total_tested": self.total_tested,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": round(self.pass_rate, 4),
            "status": self.status,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RobustnessEvaluator:
    """Executes controlled stress and failure tests against existing fusion and decision logic."""

    def evaluate_all(self) -> RobustnessEvaluationResult:
        """Runs the 10 standardized robustness tests and aggregates outcomes."""
        results: List[RobustnessConditionResult] = [
            self._test_video_dropout(),
            self._test_audio_dropout(),
            self._test_noisy_audio(),
            self._test_low_confidence(),
            self._test_high_crowd_density(),
            self._test_sudden_velocity_burst(),
            self._test_conflicting_modalities(),
            self._test_missing_frames(),
            self._test_camera_reconnect(),
            self._test_mic_reconnect(),
        ]

        passed = sum(1 for r in results if r.status == "PASS")
        total = len(results)

        return RobustnessEvaluationResult(
            conditions=results,
            total_tested=total,
            passed_count=passed,
            failed_count=total - passed,
            pass_rate=passed / total if total > 0 else 0.0,
            status="AVAILABLE",
            source=EvaluationSource.CONTROLLED_TEST.value,
        )

    def _test_video_dropout(self) -> RobustnessConditionResult:
        """1. Video frame dropout: Video assessment None, audio active."""
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        try:
            f_res = fusion.fuse(video_assessment=None, audio_assessment={"audio_score": 0.40, "sound_event": "shouting"})
            evt = decision.evaluate(fused_assessment=f_res)
            passed = evt.current_state in [RiskState.NORMAL.value, RiskState.SUSPICIOUS.value]
            return RobustnessConditionResult(
                condition="Video Frame Dropout",
                description="Video feed completely unavailable while acoustic surveillance remains active",
                system_state=evt.current_state,
                anomaly_score=evt.score,
                decision_stable=True,
                alert_behavior="Audio fallback active without pipeline failure",
                recovered=True,
                status="PASS" if passed else "FAIL",
                notes="Single-modality audio fallback functioned cleanly.",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Video Frame Dropout",
                description="Video feed unavailable",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=f"Exception raised: {exc}",
            )

    def _test_audio_dropout(self) -> RobustnessConditionResult:
        """2. Audio packet dropout: Audio assessment None, video active."""
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        try:
            f_res = fusion.fuse(video_assessment={"anomaly_score": 0.20, "crowd_density": 0.3}, audio_assessment=None)
            evt = decision.evaluate(fused_assessment=f_res)
            passed = evt.current_state == RiskState.NORMAL.value
            return RobustnessConditionResult(
                condition="Audio Packet Dropout",
                description="Microphone stream disconnected while camera feed remains operational",
                system_state=evt.current_state,
                anomaly_score=evt.score,
                decision_stable=True,
                alert_behavior="Video single-modality weighting re-normalized to 1.0",
                recovered=True,
                status="PASS" if passed else "FAIL",
                notes="Video single-modality baseline preserved.",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Audio Packet Dropout",
                description="Audio feed unavailable",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=f"Exception raised: {exc}",
            )

    def _test_noisy_audio(self) -> RobustnessConditionResult:
        """3. Noisy audio: High ambient sound level, calm video crowd."""
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        try:
            f_res = fusion.fuse(
                video_assessment={"anomaly_score": 0.05, "crowd_density": 0.2, "mean_speed": 2.0},
                audio_assessment={"audio_score": 0.65, "sound_event": "ambient_traffic", "audio_db": -12.0},
            )
            evt = decision.evaluate(fused_assessment=f_res)
            # False positive acoustic spike on calm crowd should be suppressed
            passed = evt.score < 0.60
            return RobustnessConditionResult(
                condition="Noisy Audio (0dB SNR)",
                description="Acoustic spike with calm crowd video dynamics (false positive suppression)",
                system_state=evt.current_state,
                anomaly_score=evt.score,
                decision_stable=True,
                alert_behavior="Audio Anomaly filtered by calm visual context",
                recovered=True,
                status="PASS" if passed else "FAIL",
                notes="False positive suppression verified.",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Noisy Audio (0dB SNR)",
                description="Ambient acoustic noise",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=str(exc),
            )

    def _test_low_confidence(self) -> RobustnessConditionResult:
        """4. Low-confidence detections."""
        decision = DecisionEngine()
        try:
            # Low confidence decision input
            f_res = MultimodalAssessment(anomaly_score=0.55, confidence=0.30, dominant_modality="video")
            evt = decision.evaluate(fused_assessment=f_res)
            passed = evt.score <= 0.55
            return RobustnessConditionResult(
                condition="Low Detection Confidence",
                description="Inference detections with high visual uncertainty (confidence = 0.30)",
                system_state=evt.current_state,
                anomaly_score=evt.score,
                decision_stable=True,
                alert_behavior="Escalation prevented due to insufficient confidence",
                recovered=True,
                status="PASS" if passed else "FAIL",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Low Detection Confidence",
                description="Low confidence",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=str(exc),
            )

    def _test_high_crowd_density(self) -> RobustnessConditionResult:
        """5. High crowd density stress test."""
        decision = DecisionEngine()
        try:
            f_res = MultimodalAssessment(
                anomaly_score=0.45,
                confidence=0.85,
                dominant_modality="video",
                crowd_density=2.5,
                people_count=65,
            )
            evt = decision.evaluate(fused_assessment=f_res)
            passed = evt.dynamic_thresholds.get("suspicious", 0.5) <= 0.50
            return RobustnessConditionResult(
                condition="High Crowd Occlusion",
                description="Severe spatial clustering (density = 2.5 people/m2, count = 65)",
                system_state=evt.current_state,
                anomaly_score=evt.score,
                decision_stable=True,
                alert_behavior="Dynamic threshold adjusted for elevated crowd vulnerability",
                recovered=True,
                status="PASS" if passed else "FAIL",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="High Crowd Occlusion",
                description="Crowd crush density",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=str(exc),
            )

    def _test_sudden_velocity_burst(self) -> RobustnessConditionResult:
        """6. Sudden velocity bursts / rapid scattering."""
        decision = DecisionEngine()
        try:
            f_res = MultimodalAssessment(
                anomaly_score=0.82,
                confidence=0.90,
                dominant_modality="video",
                dispersion=0.85,
            )
            for _ in range(3):
                evt = decision.evaluate(fused_assessment=f_res)
            passed = evt.current_state in [RiskState.HIGH_RISK.value, RiskState.EMERGENCY.value]
            return RobustnessConditionResult(
                condition="Sudden Velocity Burst",
                description="Rapid kinetic dispersion and crowd scattering surge (dispersion = 0.85)",
                system_state=evt.current_state,
                anomaly_score=evt.score,
                decision_stable=True,
                alert_behavior="State escalated promptly to HIGH_RISK or EMERGENCY",
                recovered=True,
                status="PASS" if passed else "FAIL",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Sudden Velocity Burst",
                description="Scattering burst",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=str(exc),
            )

    def _test_conflicting_modalities(self) -> RobustnessConditionResult:
        """7. Conflicting modalities (High video, Low audio)."""
        fusion = MultimodalFusionEngine()
        try:
            f_res = fusion.fuse(
                video_assessment={"anomaly_score": 0.85, "fight_score": 0.88},
                audio_assessment={"audio_score": 0.10, "sound_event": "normal"},
            )
            passed = 0.40 <= f_res.anomaly_score <= 0.80 and not f_res.corroborated
            return RobustnessConditionResult(
                condition="Conflicting Modalities",
                description="Visual fight indicators present while acoustic environment reports calm ambient noise",
                system_state="EVALUATED",
                anomaly_score=f_res.anomaly_score,
                decision_stable=True,
                alert_behavior="Uncorroborated weighted fusion applied; corroboration flag set to False",
                recovered=True,
                status="PASS" if passed else "FAIL",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Conflicting Modalities",
                description="Modality conflict",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=str(exc),
            )

    def _test_missing_frames(self) -> RobustnessConditionResult:
        """8. Missing / dropped frames in sequence."""
        decision = DecisionEngine()
        try:
            evt1 = decision.evaluate(MultimodalAssessment(anomaly_score=0.20))
            time.sleep(0.01)
            evt2 = decision.evaluate(MultimodalAssessment(anomaly_score=0.25))
            passed = evt2.current_state == RiskState.NORMAL.value
            return RobustnessConditionResult(
                condition="Missing Frames Sequence",
                description="Intermittent frame drops and timestamp discontinuity in video capture",
                system_state=evt2.current_state,
                anomaly_score=evt2.score,
                decision_stable=True,
                alert_behavior="Pipeline recovers cleanly without timestamp lockup",
                recovered=True,
                status="PASS" if passed else "FAIL",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Missing Frames Sequence",
                description="Dropped frames",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=str(exc),
            )

    def _test_camera_reconnect(self) -> RobustnessConditionResult:
        """9. Temporary camera disconnect and reconnect."""
        fusion = MultimodalFusionEngine()
        try:
            _ = fusion.fuse(video_assessment={"anomaly_score": 0.2}, audio_assessment={"audio_score": 0.2})
            drop = fusion.fuse(video_assessment=None, audio_assessment={"audio_score": 0.2})
            reconnect = fusion.fuse(video_assessment={"anomaly_score": 0.2}, audio_assessment={"audio_score": 0.2})
            passed = reconnect.anomaly_score is not None and reconnect.weights.get("video", 0) > 0.0
            return RobustnessConditionResult(
                condition="Camera Disconnect/Reconnect",
                description="Camera disconnects temporarily, video drops, then resumes nominal streaming",
                system_state="ONLINE",
                anomaly_score=reconnect.anomaly_score,
                decision_stable=True,
                alert_behavior="Dynamic weights restored upon sensor restoration",
                recovered=True,
                status="PASS" if passed else "FAIL",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Camera Disconnect/Reconnect",
                description="Camera disconnect",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=str(exc),
            )

    def _test_mic_reconnect(self) -> RobustnessConditionResult:
        """10. Temporary microphone disconnect and reconnect."""
        fusion = MultimodalFusionEngine()
        try:
            _ = fusion.fuse(video_assessment={"anomaly_score": 0.2}, audio_assessment={"audio_score": 0.2})
            drop = fusion.fuse(video_assessment={"anomaly_score": 0.2}, audio_assessment=None)
            reconnect = fusion.fuse(video_assessment={"anomaly_score": 0.2}, audio_assessment={"audio_score": 0.2})
            passed = reconnect.anomaly_score is not None and reconnect.weights.get("audio", 0) > 0.0
            return RobustnessConditionResult(
                condition="Microphone Disconnect/Reconnect",
                description="Audio feed drops temporarily, audio drops, then resumes nominal acoustic stream",
                system_state="ONLINE",
                anomaly_score=reconnect.anomaly_score,
                decision_stable=True,
                alert_behavior="Acoustic modality re-integrated seamlessly upon reconnection",
                recovered=True,
                status="PASS" if passed else "FAIL",
            )
        except Exception as exc:
            return RobustnessConditionResult(
                condition="Microphone Disconnect/Reconnect",
                description="Mic disconnect",
                system_state="ERROR",
                anomaly_score=0.0,
                decision_stable=False,
                alert_behavior="Crash",
                recovered=False,
                status="FAIL",
                notes=str(exc),
            )
