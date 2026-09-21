"""
==============================================================================
Explainable AI (XAI) Explanation Engine (Phase 9).
Master explanation engine providing transparent, auditable diagnostic evidence,
deterministic factor attributions, and human-readable justifications for
DecisionEvent and AlertRecord instances.
==============================================================================
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from src.alerts.alert_record import AlertRecord
from src.decision.decision_record import DecisionEvent
from src.fusion.types import MultimodalAssessment
from src.xai.evidence_extractor import EvidenceExtractor
from src.xai.evidence_record import EvidenceRecord
from src.xai.explanation_formatter import ExplanationFormatter
from src.xai.feature_attribution import FeatureAttributor
from src.xai.gradcam_interface import GradCAMInterface
from src.utils.logger import setup_logger

logger = setup_logger("xai_engine")


class ExplanationEngine:
    """
    Unified Explainable AI (XAI) engine for the surveillance system.
    
    Translates raw and fused multimodal telemetry into structured EvidenceRecords
    with transparent causal justifications, deterministic feature contributions,
    and visual HUD bullet points.
    
    Invariants:
    - READ-ONLY: Never alters decision states, anomaly scores, or alert severities.
    - ACADEMIC INTEGRITY: Never synthesizes fake Grad-CAM or pseudo-SHAP heatmaps.
    - HIGH THROUGHPUT: Execution time is typically < 1 ms per assessment.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        gradcam_interface: Optional[GradCAMInterface] = None,
        feature_attributor: Optional[FeatureAttributor] = None,
        evidence_extractor: Optional[EvidenceExtractor] = None,
    ):
        """
        Initialize the ExplanationEngine.

        Args:
            config: Optional system configuration dictionary.
            gradcam_interface: Optional GradCAMInterface instance.
            feature_attributor: Optional FeatureAttributor instance.
            evidence_extractor: Optional EvidenceExtractor instance.
        """
        self.config = config or {}
        xai_cfg = self.config.get("xai", {})

        self.gradcam = gradcam_interface or GradCAMInterface()
        self.attributor = feature_attributor or FeatureAttributor()
        self.extractor = evidence_extractor or EvidenceExtractor()
        self.formatter = ExplanationFormatter()

        self.enabled = xai_cfg.get("enabled", True)
        self.max_hud_lines = xai_cfg.get("hud", {}).get("max_evidence_items", 4)
        self.max_attributions = xai_cfg.get("explanations", {}).get("max_items", 5)

    def explain_decision(
        self,
        decision_event: Optional[DecisionEvent] = None,
        multimodal_assessment: Optional[MultimodalAssessment] = None,
        alert_record: Optional[AlertRecord] = None,
        current_time: Optional[float] = None,
    ) -> EvidenceRecord:
        """
        Generates a comprehensive EvidenceRecord explaining the provided surveillance decision.

        Args:
            decision_event: Emitted DecisionEvent from Phase 7 DecisionEngine.
            multimodal_assessment: MultimodalAssessment from Phase 6 FusionEngine.
            alert_record: Optional emitted AlertRecord from Phase 8 AlertManager.
            current_time: Optional timestamp override.

        Returns:
            EvidenceRecord dataclass containing structured diagnostic explanations.
        """
        ts = current_time or (
            decision_event.timestamp
            if decision_event is not None
            else (alert_record.timestamp if alert_record is not None else time.time())
        )

        # 1. Extract factual observations and metrics
        extracted = self.extractor.extract_evidence(
            decision_event=decision_event,
            assessment=multimodal_assessment,
            alert_record=alert_record,
        )

        metrics = extracted["metrics"]
        statements = extracted["statements"]
        audio_details = extracted["audio_details"]

        # Determine core telemetry values safely
        if decision_event is not None:
            risk_state = str(decision_event.current_state)
            anomaly_score = float(decision_event.score)
            confidence = float(decision_event.confidence)
            dominant_mod = str(decision_event.dominant_modality)
            corroborated = bool(decision_event.corroborated)
            critical = bool(decision_event.critical_evidence)
        elif multimodal_assessment is not None:
            risk_state = str(multimodal_assessment.status)
            anomaly_score = float(multimodal_assessment.multimodal_score)
            confidence = float(multimodal_assessment.multimodal_confidence)
            dominant_mod = str(multimodal_assessment.dominant_modality)
            corroborated = bool(
                multimodal_assessment.synergy.synergy_detected
                if multimodal_assessment.synergy
                else False
            )
            critical = False
        elif alert_record is not None:
            risk_state = str(alert_record.risk_state)
            anomaly_score = float(alert_record.score)
            confidence = float(alert_record.confidence)
            dominant_mod = str(alert_record.dominant_modality)
            corroborated = bool(alert_record.corroborated)
            critical = bool(alert_record.critical_evidence)
        else:
            # All inputs None
            risk_state = "NORMAL"
            anomaly_score = 0.0
            confidence = 1.0
            dominant_mod = "none"
            corroborated = False
            critical = False

        # 2. Compute deterministic feature contributions
        contributions = self.attributor.compute_contributions(
            density_score=metrics.get("crowd_density", 0.0) / 3.0,  # Scaled to ~[0-1]
            movement_score=metrics.get("movement_score", 0.0),
            behavior_score=metrics.get("behavior_score", 0.0),
            audio_score=metrics.get("audio_score", 0.0),
            fight_score=metrics.get("fight_score", 0.0),
            weapon_score=metrics.get("weapon_score", 0.0),
            corroborated=corroborated,
        )

        ranked_tuples = self.attributor.rank_contributions(
            contributions=contributions,
            top_k=self.max_attributions,
        )

        # 3. Model Explainability Status (Academic Transparency)
        model_available = self.gradcam.is_available()
        model_reason = (
            "Grad-CAM active"
            if model_available
            else "No trained explainable model configured (baseline mode)"
        )

        # 4. Construct initial EvidenceRecord
        record = EvidenceRecord(
            timestamp=ts,
            risk_state=risk_state,
            anomaly_score=anomaly_score,
            decision_confidence=confidence,
            dominant_modality=dominant_mod,
            crowd_density=metrics.get("crowd_density", 0.0),
            movement_score=metrics.get("movement_score", 0.0),
            behavior_score=metrics.get("behavior_score", 0.0),
            audio_score=metrics.get("audio_score", 0.0),
            fight_score=metrics.get("fight_score", 0.0),
            weapon_score=metrics.get("weapon_score", 0.0),
            corroborated=corroborated,
            critical_evidence=critical,
            top_evidence=statements,
            evidence_contributions=contributions,
            ranked_contributions=ranked_tuples,
            visual_evidence_available=True,
            model_explanation_available=model_available,
            model_explanation_reason=model_reason,
            audio_details=audio_details,
            details={
                "decision_event_present": decision_event is not None,
                "assessment_present": multimodal_assessment is not None,
                "alert_record_present": alert_record is not None,
            },
        )

        # 5. Format textual representations
        record.short_explanation = self.formatter.format_short(record)
        record.explanation = self.formatter.format_detailed(record)
        record.hud_explanation = self.formatter.format_hud(record, max_lines=self.max_hud_lines)

        if alert_record is not None:
            record.alert_rationale = self.formatter.format_alert_rationale(alert_record, record)

        return record

    def explain_alert(
        self,
        alert_record: AlertRecord,
        decision_event: Optional[DecisionEvent] = None,
        multimodal_assessment: Optional[MultimodalAssessment] = None,
    ) -> EvidenceRecord:
        """
        Specialized explanation method tailored for operational AlertRecords.

        Args:
            alert_record: The emitted AlertRecord to justify.
            decision_event: Optional source DecisionEvent.
            multimodal_assessment: Optional source MultimodalAssessment.

        Returns:
            EvidenceRecord containing alert rationale and causal attributions.
        """
        return self.explain_decision(
            decision_event=decision_event,
            multimodal_assessment=multimodal_assessment,
            alert_record=alert_record,
        )

    def explain(
        self,
        decision_event: Optional[DecisionEvent] = None,
        multimodal_assessment: Optional[MultimodalAssessment] = None,
        alert_record: Optional[AlertRecord] = None,
        current_time: Optional[float] = None,
        *args,
        **kwargs,
    ) -> EvidenceRecord:
        """Universal explain method dispatching to decision or alert explanations."""
        if decision_event is not None and isinstance(decision_event, AlertRecord):
            return self.explain_alert(
                alert_record=decision_event,
                decision_event=multimodal_assessment if isinstance(multimodal_assessment, DecisionEvent) else None,
            )
        return self.explain_decision(
            decision_event=decision_event,
            multimodal_assessment=multimodal_assessment,
            alert_record=alert_record,
            current_time=current_time,
            *args,
            **kwargs,
        )

    generate_explanation = explain

