"""
Deterministic Scenario Evaluator - Phase 12.

Executes 13 controlled synthetic surveillance scenarios spanning nominal,
transitional, critical emergency, and failure-recovery conditions.
Explicitly tags every scenario result as 'CONTROLLED SYNTHETIC SCENARIO'.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional

from src.alerts.alert_manager import AlertManager, AlertPolicy
from src.decision.decision_engine import DecisionEngine, RiskState
from src.evaluation.ground_truth import EvaluationSource
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.fusion.types import MultimodalAssessment


@dataclass
class ScenarioResult:
    """Evaluated operational behavior for a single controlled scenario."""
    scenario_id: str
    name: str
    source: Any = EvaluationSource.CONTROLLED_SYNTHETIC
    expected_behavior: str = ""
    observed_state: str = ""
    score: float = 0.0
    confidence: float = 0.0
    alert_severity: str = "NONE"
    latency_ms: float = 0.0
    status: str = "PASS"  # "PASS" or "FAIL"
    corroborated: bool = False
    notes: str = ""
    expected_state: str = ""

    def __post_init__(self) -> None:
        if not self.expected_state:
            for state in ["EMERGENCY", "HIGH_RISK", "SUSPICIOUS", "NORMAL"]:
                if state in self.expected_behavior:
                    self.expected_state = state
                    break
            if not self.expected_state:
                self.expected_state = self.observed_state if self.observed_state != "RECOVERY" else "NORMAL"

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "source": self.source.value if hasattr(self.source, "value") else str(self.source),
            "expected_behavior": self.expected_behavior,
            "expected_state": self.expected_state,
            "observed_state": self.observed_state,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "alert_severity": self.alert_severity,
            "latency_ms": round(self.latency_ms, 3),
            "status": self.status,
            "corroborated": self.corroborated,
            "notes": self.notes,
        }


class ScenarioEvaluator:
    """Runs the 13 mandated controlled synthetic surveillance benchmark scenarios."""

    def evaluate_all(self) -> List[ScenarioResult]:
        """Executes all 13 controlled scenarios."""
        return [
            self._scen_01_normal_crowd(),
            self._scen_02_moderate_crowding(),
            self._scen_03_rapid_movement(),
            self._scen_04_suspicious_behavior(),
            self._scen_05_physical_struggle(),
            self._scen_06_audio_distress(),
            self._scen_07_video_only_anomaly(),
            self._scen_08_audio_only_anomaly(),
            self._scen_09_multimodal_corroboration(),
            self._scen_10_conflicting_modalities(),
            self._scen_11_critical_emergency(),
            self._scen_12_sensor_dropout(),
            self._scen_13_recovery(),
        ]

    def _scen_01_normal_crowd(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.08, "crowd_density": 0.2, "mean_speed": 2.5},
            audio_assessment={"audio_score": 0.05, "sound_event": "normal"},
        )
        evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state == RiskState.NORMAL.value
        return ScenarioResult(
            scenario_id="SCEN-01",
            name="Normal Crowd",
            expected_behavior="Maintain NORMAL state without alert dispatch",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="NONE",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
            notes="Nominal baseline pedestrian dynamics.",
        )

    def _scen_02_moderate_crowding(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.35, "crowd_density": 1.1, "mean_speed": 3.0},
            audio_assessment={"audio_score": 0.15, "sound_event": "normal"},
        )
        evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state in [RiskState.NORMAL.value, RiskState.SUSPICIOUS.value]
        return ScenarioResult(
            scenario_id="SCEN-02",
            name="Moderate Crowding",
            expected_behavior="Remain within NORMAL or escalate smoothly to SUSPICIOUS",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="WARNING" if evt.current_state == "SUSPICIOUS" else "NONE",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
        )

    def _scen_03_rapid_movement(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.70, "dispersion": 0.70, "mean_speed": 16.0},
            audio_assessment={"audio_score": 0.20, "sound_event": "normal"},
        )
        for _ in range(3):
            evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state == RiskState.SUSPICIOUS.value
        return ScenarioResult(
            scenario_id="SCEN-03",
            name="Rapid Movement",
            expected_behavior="Escalate to SUSPICIOUS on crowd dispersal",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="WARNING",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
        )

    def _scen_04_suspicious_behavior(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.62, "crowd_density": 1.4, "mean_speed": 12.0},
            audio_assessment={"audio_score": 0.45, "sound_event": "shouting"},
        )
        for _ in range(3):
            evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state == RiskState.SUSPICIOUS.value
        return ScenarioResult(
            scenario_id="SCEN-04",
            name="Suspicious Behavior",
            expected_behavior="Sustain SUSPICIOUS state with elevated confidence",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="WARNING",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
        )

    def _scen_05_physical_struggle(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.78, "fight_score": 0.85, "mean_speed": 15.0},
            audio_assessment={"audio_score": 0.55, "sound_event": "shouting"},
        )
        for _ in range(3):
            evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state in [RiskState.HIGH_RISK.value, RiskState.EMERGENCY.value]
        return ScenarioResult(
            scenario_id="SCEN-05",
            name="Physical Struggle",
            expected_behavior="Escalate to HIGH_RISK or EMERGENCY alert",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="HIGH" if evt.current_state == "HIGH_RISK" else "CRITICAL",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
        )

    def _scen_06_audio_distress(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.25, "crowd_density": 0.5},
            audio_assessment={"audio_score": 0.88, "sound_event": "scream", "audio_db": -8.0},
        )
        for _ in range(3):
            evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state in [RiskState.SUSPICIOUS.value, RiskState.HIGH_RISK.value]
        return ScenarioResult(
            scenario_id="SCEN-06",
            name="Audio Distress",
            expected_behavior="Detect acoustic scream distress without visual fight",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="WARNING" if evt.current_state == "SUSPICIOUS" else "HIGH",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
        )

    def _scen_07_video_only_anomaly(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.82, "fight_score": 0.80},
            audio_assessment=None,
        )
        for _ in range(3):
            evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state in [RiskState.HIGH_RISK.value, RiskState.EMERGENCY.value]
        return ScenarioResult(
            scenario_id="SCEN-07",
            name="Video-only Anomaly",
            expected_behavior="Escalate safely on unimodal video stream",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="HIGH",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=False,
        )

    def _scen_08_audio_only_anomaly(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment=None,
            audio_assessment={"audio_score": 0.72, "sound_event": "shouting"},
        )
        for _ in range(3):
            evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state in [RiskState.SUSPICIOUS.value, RiskState.HIGH_RISK.value]
        return ScenarioResult(
            scenario_id="SCEN-08",
            name="Audio-only Anomaly",
            expected_behavior="Escalate safely on unimodal acoustic feed",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="WARNING",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=False,
        )

    def _scen_09_multimodal_corroboration(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.88, "fight_score": 0.85, "dispersion": 0.80},
            audio_assessment={"audio_score": 0.90, "sound_event": "scream"},
        )
        for _ in range(3):
            evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state in [RiskState.HIGH_RISK.value, RiskState.EMERGENCY.value] and f_res.corroborated
        return ScenarioResult(
            scenario_id="SCEN-09",
            name="Multimodal Corroboration",
            expected_behavior="Corroborate cross-modal cues and accelerate escalation",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="CRITICAL",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
        )

    def _scen_10_conflicting_modalities(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.05, "crowd_density": 0.2, "mean_speed": 1.5},
            audio_assessment={"audio_score": 0.75, "sound_event": "traffic_noise", "audio_db": -10.0},
        )
        evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Calm crowd dampens false positive audio spike
        passed = evt.score < 0.65 and evt.current_state in [RiskState.NORMAL.value, RiskState.SUSPICIOUS.value]
        return ScenarioResult(
            scenario_id="SCEN-10",
            name="Conflicting Modalities",
            expected_behavior="Suppress isolated audio spike due to calm crowd context",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="NONE" if evt.current_state == "NORMAL" else "WARNING",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
        )

    def _scen_11_critical_emergency(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        f_res = fusion.fuse(
            video_assessment={"anomaly_score": 0.95, "weapon_score": 0.95, "dispersion": 0.90},
            audio_assessment={"audio_score": 0.92, "sound_event": "scream"},
        )
        evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state == RiskState.EMERGENCY.value and evt.score >= 0.85
        return ScenarioResult(
            scenario_id="SCEN-11",
            name="Stampede Outbreak",
            expected_behavior="Immediate emergency escalation with CRITICAL alert",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="CRITICAL",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=f_res.corroborated,
        )

    def _scen_12_sensor_dropout(self) -> ScenarioResult:
        fusion = MultimodalFusionEngine()
        decision = DecisionEngine()
        t0 = time.perf_counter()
        # Drop video, then audio
        _ = fusion.fuse(video_assessment=None, audio_assessment={"audio_score": 0.2})
        f_res = fusion.fuse(video_assessment=None, audio_assessment=None)
        evt = decision.evaluate(fused_assessment=f_res)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt.current_state in [RiskState.NORMAL.value, RiskState.RECOVERY.value]
        return ScenarioResult(
            scenario_id="SCEN-12",
            name="Sensor Dropout",
            expected_behavior="Maintain safe default baseline without system crash",
            observed_state=evt.current_state,
            score=evt.score,
            confidence=evt.confidence,
            alert_severity="NONE",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=False,
        )

    def _scen_13_recovery(self) -> ScenarioResult:
        decision = DecisionEngine()
        t0 = time.perf_counter()
        # Trigger emergency first
        decision.evaluate(MultimodalAssessment(anomaly_score=0.95, critical_evidence=True))
        # Drop score below de-escalation threshold to trigger RECOVERY
        evt_rec = decision.evaluate(MultimodalAssessment(anomaly_score=0.20))
        dt_ms = (time.perf_counter() - t0) * 1000.0

        passed = evt_rec.current_state in [RiskState.RECOVERY.value, RiskState.NORMAL.value]
        return ScenarioResult(
            scenario_id="SCEN-13",
            name="Recovery",
            expected_behavior="Transition through RECOVERY stabilization before clearing to NORMAL",
            expected_state="NORMAL",
            observed_state=evt_rec.current_state,
            score=evt_rec.score,
            confidence=evt_rec.confidence,
            alert_severity="INFO" if evt_rec.current_state == "RECOVERY" else "NONE",
            latency_ms=dt_ms,
            status="PASS" if passed else "FAIL",
            corroborated=False,
        )
