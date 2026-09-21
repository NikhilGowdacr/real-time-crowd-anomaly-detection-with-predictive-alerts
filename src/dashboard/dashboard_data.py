"""
==============================================================================
Dashboard Data Models and Data Adapter (Phase 10).
Provides the DashboardSnapshot dataclass and DashboardDataProvider adapter,
transforming sensory, decision, alert, and XAI telemetry into a unified,
safe, read-only representation for the surveillance monitoring console.
==============================================================================
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.alerts.alert_manager import AlertManager
from src.alerts.alert_record import AlertRecord
from src.audio.audio_anomaly import AudioEventRecord
from src.behavior.behavior_anomaly import BehaviorAssessment
from src.behavior.behavior_features import CrowdBehaviorMetrics
from src.decision.decision_record import DecisionEvent
from src.fusion.types import MultimodalAssessment
from src.video.crowd_density import DensityResult
from src.xai.evidence_record import EvidenceRecord


@dataclass
class DashboardSnapshot:
    """
    Unified immutable state snapshot of the surveillance system.
    Combines sensory detection, crowd dynamics, audio telemetry, multimodal fusion,
    decision states, alert lifecycle, and XAI evidence for real-time visualization.
    """
    timestamp: float = field(default_factory=time.time)
    fps: float = 0.0
    latency_ms: float = 0.0

    # Risk & Decision Telemetry (Phase 7)
    risk_state: str = "NORMAL"                            # "NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY", "RECOVERY"
    anomaly_score: float = 0.0                            # Composite multimodal anomaly score [0.0 - 1.0]
    decision_confidence: float = 1.0                       # Statistical confidence [0.0 - 1.0]
    alert_severity: str = "NONE"                          # "NONE", "INFO", "WARNING", "HIGH", "CRITICAL"
    dominant_modality: str = "none"                       # "video", "audio", "bimodal", "none"
    critical_evidence: bool = False                       # True if severe override triggered
    corroborated: bool = False                            # True if multi-sensor synergy confirmed
    confirmed: bool = True                                # True if persistence quota met

    # Crowd Vision & Dynamics (Phases 2-4)
    people_count: int = 0
    crowd_density: float = 0.0                            # People per square meter
    density_level: str = "LOW"                            # "LOW", "MODERATE", "HIGH", "CRITICAL"
    average_speed: float = 0.0                            # Mean velocity in px/frame
    movement_variance: float = 0.0
    dispersion_rate: float = 0.0                          # Radial scattering rate in px/frame
    directional_divergence: float = 0.0                   # Angular entropy
    abnormal_track_count: int = 0

    # Multimodal & Audio Telemetry (Phases 5-6)
    video_score: float = 0.0
    audio_score: float = 0.0
    fusion_score: float = 0.0
    synergy_detected: bool = False
    synergy_type: Optional[str] = None
    false_positive_suppressed: bool = False
    audio_event: str = "normal"
    audio_rms: float = 0.0
    audio_centroid: float = 0.0
    audio_rolloff: float = 0.0
    audio_model_mode: str = "Spectral Baseline"           # Transparent acoustic provenance
    audio_status: str = "ONLINE"                          # "ONLINE", "OFFLINE"
    video_status: str = "ONLINE"                          # "ONLINE", "OFFLINE"

    # Subsystem Health Statuses (Phases 1-9)
    detector_status: str = "READY"
    tracker_status: str = "READY"
    fusion_status: str = "READY"
    decision_status: str = "READY"
    alert_status: str = "READY"
    xai_status: str = "READY"
    dashboard_status: str = "ONLINE"

    # Operational Alerts (Phase 8)
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    recent_alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Explainable AI (XAI) Telemetry (Phase 9)
    top_evidence: List[str] = field(default_factory=list)
    evidence_contributions: Dict[str, float] = field(default_factory=dict)
    ranked_contributions: List[Tuple[str, float]] = field(default_factory=list)
    alert_rationale: Optional[str] = None
    short_explanation: str = ""
    model_explanation_available: bool = False

    # Live Annotated Video Frame
    frame: Optional[np.ndarray] = None

    # Flexible Aliases / Convenience Properties
    crowd_count: Optional[int] = None
    active_alert: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    pipeline_latency_ms: Optional[float] = None
    video_frame: Optional[np.ndarray] = None
    dominant_audio_event: Optional[str] = None
    audio_db: Optional[float] = None
    xai_explanation: Optional[str] = None
    operator_action: Optional[str] = None
    top_contributors: Optional[Dict[str, float]] = None
    weights: Optional[Dict[str, float]] = None
    system_health: Optional[Dict[str, bool]] = None
    movement_speed: Optional[float] = None

    def __post_init__(self) -> None:
        """Harmonize convenience aliases and primary fields."""
        # 1. People / Crowd count
        if self.crowd_count is not None:
            self.people_count = int(self.crowd_count)
        else:
            self.crowd_count = int(self.people_count)

        # 2. Confidence
        if self.confidence is not None:
            self.decision_confidence = float(self.confidence)
        else:
            self.confidence = float(self.decision_confidence)

        # 3. Latency
        if self.pipeline_latency_ms is not None:
            self.latency_ms = float(self.pipeline_latency_ms)
        else:
            self.pipeline_latency_ms = float(self.latency_ms)

        # 4. Active Alert
        if self.active_alert is not None:
            if not self.active_alerts:
                self.active_alerts = [dict(self.active_alert)]
            if not self.alert_severity or self.alert_severity == "NONE":
                self.alert_severity = self.active_alert.get("severity", "WARNING")
        else:
            self.active_alert = dict(self.active_alerts[0]) if self.active_alerts else None

        # 5. Video frame
        if self.video_frame is not None:
            self.frame = self.video_frame
        else:
            self.video_frame = self.frame

        # 6. Audio Event
        if self.dominant_audio_event is not None:
            self.audio_event = self.dominant_audio_event
        else:
            self.dominant_audio_event = self.audio_event

        # 7. Audio dB
        if self.audio_db is None:
            if self.audio_rms > 0.0001:
                self.audio_db = float(round(20.0 * np.log10(self.audio_rms), 1))
            else:
                self.audio_db = -40.0

        # 8. Speed
        if self.movement_speed is not None:
            self.average_speed = float(self.movement_speed)
        else:
            self.movement_speed = float(self.average_speed)

        # 9. XAI
        if self.xai_explanation is not None:
            self.short_explanation = self.xai_explanation
        else:
            self.xai_explanation = self.short_explanation

        if self.top_contributors is not None:
            self.evidence_contributions = dict(self.top_contributors)
        else:
            self.top_contributors = dict(self.evidence_contributions)

        if self.operator_action is None:
            if self.alert_rationale:
                self.operator_action = self.alert_rationale
            elif self.risk_state == "NORMAL":
                self.operator_action = "Continue standard monitoring."
            elif self.risk_state == "SUSPICIOUS":
                self.operator_action = "Monitor area closely and observe crowd movements."
            elif self.risk_state == "HIGH_RISK":
                self.operator_action = "Dispatch ground security officers for tactical assessment."
            elif self.risk_state == "EMERGENCY":
                self.operator_action = "Initiate emergency broadcast, notify rapid response units immediately."
            elif self.risk_state == "RECOVERY":
                self.operator_action = "Verify crowd return to nominal flow and record post-incident logs."
            else:
                self.operator_action = "Monitor system status."

        # 10. Weights
        if self.weights is None:
            self.weights = {"video": 0.6, "audio": 0.4}

        # 11. System Health
        if self.system_health is None:
            self.system_health = {
                "video_capture": self.video_status == "ONLINE",
                "person_detection": self.detector_status == "READY",
                "crowd_tracking": self.tracker_status == "READY",
                "behavior_analysis": True,
                "audio_analysis": self.audio_status == "ONLINE",
                "multimodal_fusion": self.fusion_status == "READY",
                "decision_engine": self.decision_status == "READY",
                "alert_manager": self.alert_status == "READY",
                "xai_engine": self.xai_status == "READY",
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot into a JSON-serializable dictionary (excluding raw frame)."""
        return {
            "timestamp": round(self.timestamp, 4),
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 2),
            "pipeline_latency_ms": round(self.latency_ms, 2),
            "risk_state": self.risk_state,
            "anomaly_score": round(self.anomaly_score, 4),
            "confidence": round(self.decision_confidence, 4),
            "decision_confidence": round(self.decision_confidence, 4),
            "alert_severity": self.alert_severity,
            "dominant_modality": self.dominant_modality,
            "critical_evidence": self.critical_evidence,
            "corroborated": self.corroborated,
            "confirmed": self.confirmed,
            "crowd_count": self.people_count,
            "crowd_density": round(self.crowd_density, 3),
            "active_alert": self.active_alert,
            "crowd": {
                "people_count": self.people_count,
                "crowd_density": round(self.crowd_density, 3),
                "density_level": self.density_level,
                "average_speed": round(self.average_speed, 2),
                "movement_variance": round(self.movement_variance, 2),
                "dispersion_rate": round(self.dispersion_rate, 2),
                "directional_divergence": round(self.directional_divergence, 2),
                "abnormal_track_count": self.abnormal_track_count,
            },
            "audio": {
                "audio_score": round(self.audio_score, 4),
                "audio_event": self.audio_event,
                "audio_rms": round(self.audio_rms, 3),
                "audio_centroid": round(self.audio_centroid, 1),
                "audio_rolloff": round(self.audio_rolloff, 1),
                "audio_model_mode": self.audio_model_mode,
                "audio_status": self.audio_status,
            },
            "fusion": {
                "video_score": round(self.video_score, 4),
                "audio_score": round(self.audio_score, 4),
                "fusion_score": round(self.fusion_score, 4),
                "synergy_detected": self.synergy_detected,
                "synergy_type": self.synergy_type,
                "false_positive_suppressed": self.false_positive_suppressed,
            },
            "system_health": self.system_health,
            "active_alerts_count": len(self.active_alerts),
            "recent_alerts_count": len(self.recent_alerts),
            "xai": {
                "top_evidence": self.top_evidence,
                "evidence_contributions": {k: round(v, 4) for k, v in self.evidence_contributions.items()},
                "ranked_contributions": [(k, round(v, 4)) for k, v in self.ranked_contributions],
                "alert_rationale": self.alert_rationale,
                "short_explanation": self.short_explanation,
                "model_explanation_available": self.model_explanation_available,
            },
            "has_frame": self.frame is not None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize snapshot dictionary to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)


class DashboardDataProvider:
    """
    Adapter transforming raw outputs from surveillance pipeline components
    (Phases 2-9) into a clean, unified DashboardSnapshot.
    
    Guarantees:
    - Read-Only: Never modifies underlying decision states, scores, or alert severities.
    - Fault-Tolerant: Gracefully handles None or missing sensor streams.
    - Honest: Transparently marks offline sensors and spectral baseline modes.
    - High Throughput: Transformation executes in < 2 ms per snapshot.
    """

    def __init__(self) -> None:
        self.last_snapshot: Optional[DashboardSnapshot] = None

    def create_snapshot(
        self,
        density: Optional[DensityResult] = None,
        kinematics: Optional[CrowdBehaviorMetrics] = None,
        behavior_assessment: Optional[BehaviorAssessment] = None,
        audio_record: Optional[AudioEventRecord] = None,
        multimodal_assessment: Optional[MultimodalAssessment] = None,
        decision_event: Optional[DecisionEvent] = None,
        alert_manager: Optional[AlertManager] = None,
        current_alert: Optional[AlertRecord] = None,
        evidence_record: Optional[EvidenceRecord] = None,
        frame: Optional[np.ndarray] = None,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        current_time: Optional[float] = None,
        video_connected: bool = True,
        audio_connected: bool = True,
    ) -> DashboardSnapshot:
        """
        Constructs a DashboardSnapshot from provided component outputs.
        """
        ts = current_time if current_time is not None else time.time()

        # 1. Video & Crowd Metrics
        people_count = 0
        crowd_density = 0.0
        density_level = "LOW"
        if density is not None:
            people_count = int(getattr(density, "person_count", 0))
            crowd_density = float(getattr(density, "density_per_m2", 0.0))
            density_level = str(getattr(density, "density_level", "LOW"))

        average_speed = 0.0
        movement_variance = 0.0
        dispersion_rate = 0.0
        directional_divergence = 0.0
        abnormal_track_count = 0
        if kinematics is not None:
            average_speed = float(getattr(kinematics, "mean_speed", 0.0))
            movement_variance = float(getattr(kinematics, "speed_variance", 0.0))
            dispersion_rate = float(getattr(kinematics, "dispersion_rate", 0.0))
            directional_divergence = float(getattr(kinematics, "direction_entropy", 0.0))
            abnormal_track_count = int(getattr(kinematics, "rapid_track_count", 0))

        # 2. Audio Telemetry
        audio_score = 0.0
        audio_event = "normal"
        audio_model_mode = "Spectral Baseline"
        audio_rms = 0.0
        audio_centroid = 0.0
        audio_rolloff = 0.0
        audio_status = "ONLINE" if audio_connected else "OFFLINE"

        if audio_record is not None:
            audio_score = float(getattr(audio_record, "audio_anomaly_score", 0.0))
            audio_event = str(getattr(audio_record, "audio_event", "normal"))
            # Extract raw audio features if available in details
            if hasattr(audio_record, "details") and isinstance(audio_record.details, dict):
                audio_rms = float(audio_record.details.get("rms_energy", 0.0))
                audio_centroid = float(audio_record.details.get("spectral_centroid", 0.0))
                audio_rolloff = float(audio_record.details.get("spectral_rolloff", 0.0))
                if audio_record.details.get("model_path"):
                    audio_model_mode = "Trained Classifier"
        elif not audio_connected:
            audio_status = "OFFLINE"

        video_status = "ONLINE" if (video_connected and (frame is not None or density is not None)) else "OFFLINE"

        # 3. Multimodal Fusion Telemetry
        video_score = 0.0
        fusion_score = 0.0
        synergy_detected = False
        synergy_type = None
        false_positive_suppressed = False

        if multimodal_assessment is not None:
            video_score = float(getattr(multimodal_assessment, "video_score", 0.0))
            if audio_connected and audio_score == 0.0:
                audio_score = float(getattr(multimodal_assessment, "audio_score", 0.0))
            fusion_score = float(getattr(multimodal_assessment, "multimodal_score", 0.0))
            if getattr(multimodal_assessment, "synergy", None):
                synergy_detected = bool(multimodal_assessment.synergy.synergy_detected)
                synergy_type = multimodal_assessment.synergy.synergy_type
            false_positive_suppressed = bool(getattr(multimodal_assessment, "false_positive_suppressed", False))

        # 4. Decision Engine Telemetry (Phase 7)
        risk_state = "NORMAL"
        anomaly_score = fusion_score
        decision_confidence = 1.0
        dominant_modality = "none"
        critical_evidence = False
        corroborated = synergy_detected
        confirmed = True

        if decision_event is not None:
            risk_state = str(decision_event.current_state)
            anomaly_score = float(decision_event.score)
            decision_confidence = float(decision_event.confidence)
            dominant_modality = str(decision_event.dominant_modality)
            critical_evidence = bool(decision_event.critical_evidence)
            corroborated = bool(decision_event.corroborated)
            confirmed = bool(decision_event.confirmed)
            if video_score == 0.0:
                video_score = float(decision_event.video_score)
            if audio_score == 0.0:
                audio_score = float(decision_event.audio_score)

        # 5. Alert Telemetry (Phase 8)
        active_alerts_list: List[Dict[str, Any]] = []
        recent_alerts_list: List[Dict[str, Any]] = []
        alert_severity = "NONE"

        if alert_manager is not None:
            active_objs = alert_manager.get_active_alerts()
            active_alerts_list = [a.to_dict() for a in active_objs]
            recent_objs = alert_manager.get_history(limit=25)
            recent_alerts_list = [a.to_dict() for a in recent_objs]

        if current_alert is not None:
            alert_severity = current_alert.severity.value
            if not any(a.get("alert_id") == current_alert.alert_id for a in active_alerts_list):
                active_alerts_list.insert(0, current_alert.to_dict())
        elif active_alerts_list:
            alert_severity = active_alerts_list[0].get("severity", "NONE")

        # 6. Explainable AI Telemetry (Phase 9)
        top_evidence: List[str] = []
        evidence_contributions: Dict[str, float] = {}
        ranked_contributions: List[Tuple[str, float]] = []
        alert_rationale = None
        short_explanation = ""
        model_explanation_available = False

        if evidence_record is not None:
            top_evidence = list(evidence_record.top_evidence)
            evidence_contributions = dict(evidence_record.evidence_contributions)
            ranked_contributions = list(evidence_record.ranked_contributions)
            alert_rationale = evidence_record.alert_rationale
            short_explanation = evidence_record.short_explanation
            model_explanation_available = bool(evidence_record.model_explanation_available)
        else:
            # Fallback formulation if XAI record is not provided
            if risk_state == "NORMAL":
                top_evidence = ["All visual kinematics, spatial density, and acoustic energy levels reside within nominal limits."]
                short_explanation = "NORMAL: Nominal baseline conditions."
            else:
                top_evidence = [f"Elevated {dominant_modality.upper()} activity observed (Score: {anomaly_score:.2f})."]
                short_explanation = f"{risk_state}: Anomaly score {anomaly_score:.2f}."

        snap = DashboardSnapshot(
            timestamp=ts,
            fps=fps,
            latency_ms=latency_ms,
            risk_state=risk_state,
            anomaly_score=anomaly_score,
            decision_confidence=decision_confidence,
            alert_severity=alert_severity,
            dominant_modality=dominant_modality,
            critical_evidence=critical_evidence,
            corroborated=corroborated,
            confirmed=confirmed,
            people_count=people_count,
            crowd_density=crowd_density,
            density_level=density_level,
            average_speed=average_speed,
            movement_variance=movement_variance,
            dispersion_rate=dispersion_rate,
            directional_divergence=directional_divergence,
            abnormal_track_count=abnormal_track_count,
            video_score=video_score,
            audio_score=audio_score,
            fusion_score=fusion_score,
            synergy_detected=synergy_detected,
            synergy_type=synergy_type,
            false_positive_suppressed=false_positive_suppressed,
            audio_event=audio_event,
            audio_rms=audio_rms,
            audio_centroid=audio_centroid,
            audio_rolloff=audio_rolloff,
            audio_model_mode=audio_model_mode,
            audio_status=audio_status,
            video_status=video_status,
            detector_status="READY" if video_connected else "OFFLINE",
            tracker_status="READY" if video_connected else "OFFLINE",
            fusion_status="READY",
            decision_status="READY",
            alert_status="READY",
            xai_status="READY",
            dashboard_status="ONLINE",
            active_alerts=active_alerts_list,
            recent_alerts=recent_alerts_list,
            top_evidence=top_evidence,
            evidence_contributions=evidence_contributions,
            ranked_contributions=ranked_contributions,
            alert_rationale=alert_rationale,
            short_explanation=short_explanation,
            model_explanation_available=model_explanation_available,
            frame=frame,
        )
        self.last_snapshot = snap
        return snap

    def create_snapshot_from_subsystems(
        self,
        decision_event: Optional[DecisionEvent] = None,
        multimodal_assessment: Optional[MultimodalAssessment] = None,
        behavior_assessment: Optional[BehaviorAssessment] = None,
        audio_record: Optional[AudioEventRecord] = None,
        density_result: Optional[DensityResult] = None,
        density: Optional[DensityResult] = None,
        kinematics: Optional[CrowdBehaviorMetrics] = None,
        alert_record: Optional[AlertRecord] = None,
        current_alert: Optional[AlertRecord] = None,
        alert_manager: Optional[AlertManager] = None,
        evidence_record: Optional[EvidenceRecord] = None,
        frame: Optional[np.ndarray] = None,
        detection_result: Optional[Any] = None,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        current_time: Optional[float] = None,
        video_connected: Optional[bool] = None,
        audio_connected: Optional[bool] = None,
    ) -> DashboardSnapshot:
        """Convenience method matching subsystem parameter naming."""
        eff_density = density_result or density
        eff_alert = alert_record or current_alert

        # Auto-detect sensor connection if not explicitly stated
        eff_video_conn = video_connected if video_connected is not None else (frame is not None or eff_density is not None or detection_result is not None)
        eff_audio_conn = audio_connected if audio_connected is not None else (audio_record is not None)

        return self.create_snapshot(
            density=eff_density,
            kinematics=kinematics,
            behavior_assessment=behavior_assessment,
            audio_record=audio_record,
            multimodal_assessment=multimodal_assessment,
            decision_event=decision_event,
            alert_manager=alert_manager,
            current_alert=eff_alert,
            evidence_record=evidence_record,
            frame=frame,
            fps=fps,
            latency_ms=latency_ms,
            current_time=current_time,
            video_connected=eff_video_conn,
            audio_connected=eff_audio_conn,
        )

    def create_demo_snapshot(self, scenario: str = "NORMAL", step: int = 0) -> DashboardSnapshot:
        """
        Creates a realistic, deterministic simulated DashboardSnapshot for Demo Mode.
        Scenarios: "NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY", "RECOVERY".
        """
        ts = time.time()
        scen = scenario.upper()

        # Simulated synthetic frame
        img = np.full((360, 640, 3), 25, dtype=np.uint8)

        if scen == "NORMAL":
            snap = DashboardSnapshot(
                timestamp=ts,
                fps=30.0,
                latency_ms=18.5,
                risk_state="NORMAL",
                anomaly_score=0.08,
                decision_confidence=0.92,
                alert_severity="NONE",
                dominant_modality="none",
                critical_evidence=False,
                corroborated=False,
                confirmed=True,
                people_count=18,
                crowd_density=0.45,
                density_level="LOW",
                average_speed=2.4,
                movement_variance=0.8,
                dispersion_rate=1.1,
                directional_divergence=0.8,
                abnormal_track_count=0,
                video_score=0.08,
                audio_score=0.06,
                fusion_score=0.07,
                audio_event="normal",
                audio_rms=0.02,
                audio_centroid=1200.0,
                audio_rolloff=2400.0,
                audio_model_mode="Spectral Baseline",
                top_evidence=["All visual kinematics, spatial density, and acoustic energy levels reside within nominal limits."],
                evidence_contributions={"crowd_density": 0.0, "rapid_movement": 0.0, "acoustic_anomaly": 0.0},
                short_explanation="NORMAL: Calm baseline conditions.",
                model_explanation_available=False,
                frame=img,
            )

        elif scen == "SUSPICIOUS":
            snap = DashboardSnapshot(
                timestamp=ts,
                fps=29.5,
                latency_ms=21.0,
                risk_state="SUSPICIOUS",
                anomaly_score=0.52,
                decision_confidence=0.78,
                alert_severity="WARNING",
                dominant_modality="audio",
                critical_evidence=False,
                corroborated=False,
                confirmed=True,
                people_count=24,
                crowd_density=1.2,
                density_level="MODERATE",
                average_speed=9.8,
                movement_variance=4.2,
                dispersion_rate=4.5,
                directional_divergence=1.6,
                abnormal_track_count=3,
                video_score=0.45,
                audio_score=0.62,
                fusion_score=0.52,
                audio_event="shouting",
                audio_rms=0.15,
                audio_centroid=2800.0,
                audio_rolloff=4900.0,
                audio_model_mode="Spectral Baseline",
                active_alerts=[{
                    "alert_id": "ALT-DEMO-001",
                    "severity": "WARNING",
                    "risk_state": "SUSPICIOUS",
                    "score": 0.52,
                    "confidence": 0.78,
                    "reason": "Elevated movement velocity accompanied by acoustic shouting.",
                    "status": "ACTIVE",
                    "timestamp": ts - 5.0,
                }],
                top_evidence=[
                    "Elevated crowd movement velocity (mean speed: 9.8 px/f, score: 0.45).",
                    "Spectral baseline evidence: elevated sound energy with acoustic signature 'SHOUTING' (score: 0.62).",
                ],
                evidence_contributions={"acoustic_anomaly": 0.55, "rapid_movement": 0.30, "crowd_density": 0.15},
                ranked_contributions=[("acoustic_anomaly", 0.55), ("rapid_movement", 0.30), ("crowd_density", 0.15)],
                short_explanation="SUSPICIOUS: Driven primarily by acoustic anomaly (55%) and rapid movement (30%).",
                alert_rationale="ALERT RATIONALE [WARNING | ALT-DEMO-001]: Surveillance threshold exceeded driven primarily by acoustic anomaly.",
                frame=img,
            )

        elif scen == "HIGH_RISK":
            snap = DashboardSnapshot(
                timestamp=ts,
                fps=28.0,
                latency_ms=23.5,
                risk_state="HIGH_RISK",
                anomaly_score=0.76,
                decision_confidence=0.85,
                alert_severity="HIGH",
                dominant_modality="video",
                critical_evidence=False,
                corroborated=False,
                confirmed=True,
                people_count=35,
                crowd_density=2.2,
                density_level="HIGH",
                average_speed=12.4,
                movement_variance=8.6,
                dispersion_rate=8.2,
                directional_divergence=2.3,
                abnormal_track_count=8,
                video_score=0.80,
                audio_score=0.55,
                fusion_score=0.76,
                audio_event="shouting",
                audio_rms=0.22,
                audio_centroid=3100.0,
                audio_rolloff=5200.0,
                audio_model_mode="Spectral Baseline",
                active_alerts=[{
                    "alert_id": "ALT-DEMO-002",
                    "severity": "HIGH",
                    "risk_state": "HIGH_RISK",
                    "score": 0.76,
                    "confidence": 0.85,
                    "reason": "Physical struggle dynamics observed with rapid crowd scattering.",
                    "status": "ACTIVE",
                    "timestamp": ts - 8.0,
                }],
                top_evidence=[
                    "Physical altercation dynamics observed (struggle score: 0.78).",
                    "Radial crowd dispersion detected (dispersion rate: 8.20 px/f) indicating rapid scattering outward.",
                    "High spatial crowd density (2.20 people/m^2, level: HIGH).",
                ],
                evidence_contributions={"physical_struggle": 0.48, "rapid_movement": 0.28, "acoustic_anomaly": 0.24},
                ranked_contributions=[("physical_struggle", 0.48), ("rapid_movement", 0.28), ("acoustic_anomaly", 0.24)],
                short_explanation="HIGH_RISK: Driven primarily by physical struggle (48%) and rapid movement (28%).",
                alert_rationale="ALERT RATIONALE [HIGH | ALT-DEMO-002]: Violent combat interaction confirmed with crowd scattering.",
                frame=img,
            )

        elif scen == "EMERGENCY":
            snap = DashboardSnapshot(
                timestamp=ts,
                fps=27.5,
                latency_ms=25.0,
                risk_state="EMERGENCY",
                anomaly_score=0.94,
                decision_confidence=0.96,
                alert_severity="CRITICAL",
                dominant_modality="bimodal",
                critical_evidence=True,
                corroborated=True,
                confirmed=True,
                people_count=42,
                crowd_density=2.9,
                density_level="CRITICAL",
                average_speed=18.5,
                movement_variance=14.2,
                dispersion_rate=14.8,
                directional_divergence=2.7,
                abnormal_track_count=16,
                video_score=0.92,
                audio_score=0.95,
                fusion_score=0.94,
                synergy_detected=True,
                synergy_type="WEAPON_AND_DISTRESS",
                audio_event="scream",
                audio_rms=0.48,
                audio_centroid=4200.0,
                audio_rolloff=7000.0,
                audio_model_mode="Spectral Baseline",
                active_alerts=[{
                    "alert_id": "ALT-DEMO-003",
                    "severity": "CRITICAL",
                    "risk_state": "EMERGENCY",
                    "score": 0.94,
                    "confidence": 0.96,
                    "reason": "Firearm detected with corroborating acoustic scream signature.",
                    "status": "ACTIVE",
                    "timestamp": ts - 2.0,
                }],
                top_evidence=[
                    "Weapon presence detected (firearm, score: 0.95) indicating severe physical hazard.",
                    "Cross-modal synergy confirmed (WEAPON_AND_DISTRESS): acoustic cues and visual kinematics mutually reinforce threat severity.",
                    "Critical threat rule triggered: immediate state escalation enforced without delay.",
                ],
                evidence_contributions={"weapon_presence": 0.38, "acoustic_anomaly": 0.28, "rapid_movement": 0.22, "cross_modal_corroboration": 0.12},
                ranked_contributions=[("weapon_presence", 0.38), ("acoustic_anomaly", 0.28), ("rapid_movement", 0.22), ("cross_modal_corroboration", 0.12)],
                short_explanation="EMERGENCY: Driven primarily by weapon presence (38%) and acoustic distress with cross-modal corroboration [CRITICAL OVERRIDE].",
                alert_rationale="ALERT RATIONALE [CRITICAL | ALT-DEMO-003]: Immediate critical safety override enforced due to confirmed weapon and scream synergy.",
                frame=img,
            )

        elif scen == "RECOVERY":
            snap = DashboardSnapshot(
                timestamp=ts,
                fps=30.0,
                latency_ms=19.0,
                risk_state="RECOVERY",
                anomaly_score=0.22,
                decision_confidence=0.88,
                alert_severity="INFO",
                dominant_modality="none",
                critical_evidence=False,
                corroborated=False,
                confirmed=True,
                people_count=15,
                crowd_density=0.40,
                density_level="LOW",
                average_speed=3.1,
                movement_variance=1.1,
                dispersion_rate=1.4,
                directional_divergence=0.9,
                abnormal_track_count=0,
                video_score=0.20,
                audio_score=0.15,
                fusion_score=0.22,
                audio_event="normal",
                audio_rms=0.03,
                audio_centroid=1400.0,
                audio_rolloff=2600.0,
                audio_model_mode="Spectral Baseline",
                top_evidence=["Stabilization window active: verifying post-incident crowd normalization across 5 consecutive confirmation cycles."],
                evidence_contributions={"crowd_density": 0.0, "rapid_movement": 0.0},
                short_explanation="RECOVERY: Anomaly score decreased below boundaries; post-incident stabilization active.",
                frame=img,
            )
        else:
            return self.create_demo_snapshot(scenario="NORMAL")

        self.last_snapshot = snap
        return snap
