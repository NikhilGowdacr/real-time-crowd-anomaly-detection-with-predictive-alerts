"""
==============================================================================
Evidence Extractor (Phase 9).
Extracts objective, transparent factual evidence from DecisionEvent,
MultimodalAssessment, and AlertRecord telemetry across visual, acoustic,
kinematic, and multi-sensor correlation domains.
==============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple

from src.alerts.alert_record import AlertRecord
from src.decision.decision_record import DecisionEvent
from src.fusion.types import MultimodalAssessment


class EvidenceExtractor:
    """
    Translates raw and fused surveillance telemetry into objective,
    non-defamatory, auditable factual evidence statements.
    
    Adheres strictly to academic honesty standards:
    - Never presents spectral baseline heuristics as deep CNN inferences.
    - Explicitly highlights uncorroborated noise suppression or missing sensor streams.
    - Formulates transparent causal statements for all elevated metrics.
    """

    def __init__(
        self,
        density_threshold: float = 1.5,
        speed_threshold: float = 8.0,
        dispersion_threshold: float = 5.0,
        entropy_threshold: float = 1.8,
        anomaly_threshold: float = 0.35,
    ):
        self.density_threshold = float(density_threshold)
        self.speed_threshold = float(speed_threshold)
        self.dispersion_threshold = float(dispersion_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.anomaly_threshold = float(anomaly_threshold)

    def extract_evidence(
        self,
        decision_event: Optional[DecisionEvent] = None,
        assessment: Optional[MultimodalAssessment] = None,
        alert_record: Optional[AlertRecord] = None,
    ) -> Dict[str, Any]:
        """
        Extracts structured factual evidence statements and underlying metrics
        from available decision, fusion, and alert records.

        Args:
            decision_event: Emitted DecisionEvent from Phase 7 DecisionEngine.
            assessment: MultimodalAssessment from Phase 6 FusionEngine.
            alert_record: Optional emitted AlertRecord from Phase 8 AlertManager.

        Returns:
            Dict containing:
            - "statements": List[str] of factual evidence observations
            - "metrics": Dict of normalized metric scores
            - "audio_details": Dict of acoustic model provenance and classification
            - "dominant_factors": List of primary reasons
        """
        statements: List[str] = []
        metrics: Dict[str, float] = {
            "crowd_density": 0.0,
            "movement_score": 0.0,
            "behavior_score": 0.0,
            "video_score": 0.0,
            "audio_score": 0.0,
            "fight_score": 0.0,
            "weapon_score": 0.0,
            "corroborated": False,
            "critical_evidence": False,
        }
        audio_details: Dict[str, Any] = {
            "mode": "spectral_baseline",
            "model_path": None,
            "event_label": "normal",
            "confidence": 0.0,
            "confirmed": False,
        }

        # -----------------------------------------------------------------
        # 1. Gather Telemetry Sources
        # -----------------------------------------------------------------
        fused_score = 0.0
        risk_state = "NORMAL"
        dominant_mod = "none"
        corroborated = False
        critical = False

        if decision_event is not None:
            fused_score = float(decision_event.score)
            risk_state = str(decision_event.current_state)
            dominant_mod = str(decision_event.dominant_modality)
            corroborated = bool(decision_event.corroborated)
            critical = bool(decision_event.critical_evidence)
            metrics["audio_score"] = float(decision_event.audio_score)
            metrics["video_score"] = float(decision_event.video_score)
            metrics["weapon_score"] = float(decision_event.weapon_score)
            metrics["fight_score"] = float(decision_event.fight_score)
            metrics["corroborated"] = corroborated
            metrics["critical_evidence"] = critical
        elif assessment is not None:
            fused_score = float(assessment.multimodal_score)
            risk_state = str(assessment.status)
            dominant_mod = str(assessment.dominant_modality)
            corroborated = bool(assessment.synergy.synergy_detected) if assessment.synergy else False
            metrics["audio_score"] = float(assessment.audio_score)
            metrics["video_score"] = float(assessment.video_score)
            metrics["corroborated"] = corroborated
        elif alert_record is not None:
            fused_score = float(alert_record.score)
            risk_state = str(alert_record.risk_state)
            dominant_mod = str(alert_record.dominant_modality)
            corroborated = bool(alert_record.corroborated)
            critical = bool(alert_record.critical_evidence)
            metrics["audio_score"] = float(alert_record.audio_score)
            metrics["video_score"] = float(alert_record.video_score)
            metrics["weapon_score"] = float(alert_record.weapon_score)
            metrics["fight_score"] = float(alert_record.fight_score)
            metrics["corroborated"] = corroborated
            metrics["critical_evidence"] = critical

        # -----------------------------------------------------------------
        # 2. Extract Detailed Evidence from Breakdown Dicts
        # -----------------------------------------------------------------
        ev_summary = {}
        if decision_event is not None and decision_event.evidence_summary:
            ev_summary = decision_event.evidence_summary
        elif assessment is not None and assessment.evidence_breakdown:
            ev_summary = assessment.evidence_breakdown
        elif alert_record is not None and alert_record.source_event:
            ev_summary = alert_record.source_event.get("evidence_summary", {})

        vid_ev = ev_summary.get("video", {})
        aud_ev = ev_summary.get("audio", {})

        # Video Sub-metrics
        density_val = float(vid_ev.get("density_m2", 0.0))
        density_score = float(vid_ev.get("density_score", 0.0))
        density_level = str(vid_ev.get("density_level", "LOW"))
        metrics["crowd_density"] = density_val

        speed_val = float(vid_ev.get("mean_speed", 0.0))
        dispersion_val = float(vid_ev.get("dispersion_rate", 0.0))
        entropy_val = float(vid_ev.get("direction_entropy", 0.0))
        movement_score = float(vid_ev.get("movement_score", 0.0))
        metrics["movement_score"] = movement_score

        behavior_score = float(vid_ev.get("behavior_score", 0.0))
        behavior_class = str(vid_ev.get("behavior_class", "normal"))
        metrics["behavior_score"] = behavior_score

        fight_score = float(vid_ev.get("fight_score", metrics["fight_score"]))
        metrics["fight_score"] = fight_score

        weapon_score = float(vid_ev.get("weapon_score", metrics["weapon_score"]))
        metrics["weapon_score"] = weapon_score
        detected_weapons = vid_ev.get("detected_weapons", [])

        # Audio Sub-metrics
        audio_event = str(aud_ev.get("audio_event", "normal")).lower()
        audio_conf = float(aud_ev.get("audio_confidence", 0.0))
        audio_confirmed = bool(aud_ev.get("is_confirmed", False))
        audio_active = bool(aud_ev.get("is_active", True)) if aud_ev else True
        video_active = bool(vid_ev.get("is_active", True)) if vid_ev else True

        audio_details["event_label"] = audio_event
        audio_details["confidence"] = audio_conf
        audio_details["confirmed"] = audio_confirmed
        audio_details["mode"] = "trained_classifier" if aud_ev.get("model_path") else "spectral_baseline"

        # -----------------------------------------------------------------
        # 3. Formulate Domain Evidence Statements
        # -----------------------------------------------------------------

        # A. Dangerous Weapon Evidence
        if weapon_score >= 0.40 or detected_weapons:
            weap_desc = ", ".join(str(w) for w in detected_weapons) if detected_weapons else "potential weapon/dangerous object"
            statements.append(
                f"Weapon presence detected ({weap_desc}, score: {weapon_score:.2f}) indicating severe physical hazard."
            )

        # B. Physical Altercation / Fight Evidence
        if fight_score >= 0.40:
            statements.append(
                f"Physical altercation dynamics observed (struggle score: {fight_score:.2f}) involving aggressive rapid interactions."
            )

        # C. Rapid Movement & Kinematics
        if movement_score >= 0.40 or speed_val >= self.speed_threshold:
            statements.append(
                f"Elevated crowd movement velocity (mean speed: {speed_val:.1f} px/f, score: {movement_score:.2f}) exceeding normal pedestrian pace."
            )

        # D. Crowd Dispersion & Directional Divergence
        if dispersion_val >= self.dispersion_threshold:
            statements.append(
                f"Radial crowd dispersion detected (dispersion rate: {dispersion_val:.2f} px/f) indicating rapid scattering outward."
            )
        if entropy_val >= self.entropy_threshold:
            statements.append(
                f"High angular directional divergence (directional entropy: {entropy_val:.2f}) indicating disordered, non-uniform trajectory flow."
            )

        # E. Behavioral Pattern Analysis
        if behavior_score >= 0.40 and behavior_class not in ("normal", "stable"):
            confirmed_tag = " [confirmed across time]" if vid_ev.get("behavior_confirmed") else " [unconfirmed]"
            statements.append(
                f"Atypical crowd behavioral pattern classified as '{behavior_class}' (behavior score: {behavior_score:.2f}){confirmed_tag}."
            )

        # F. Crowd Density
        if density_score >= 0.50 or density_val >= self.density_threshold or density_level in ("HIGH", "CRITICAL"):
            statements.append(
                f"High spatial crowd density ({density_val:.2f} people/m^2, level: {density_level}) increasing congestion risk."
            )

        # G. Acoustic Evidence (Honest Provenance)
        if metrics["audio_score"] >= 0.35 or audio_event not in ("normal", "speech"):
            conf_str = f"conf: {audio_conf:.2f}"
            prov_str = "Spectral baseline evidence" if audio_details["mode"] == "spectral_baseline" else "Acoustic classifier"
            statements.append(
                f"{prov_str}: elevated sound energy with acoustic signature '{audio_event.upper()}' (score: {metrics['audio_score']:.2f}, {conf_str})."
            )

        # H. Multi-Sensor Corroboration / Synergy
        if corroborated:
            syn_type = ev_summary.get("synergy_type", "CROSS_MODAL_SYNERGY")
            statements.append(
                f"Cross-modal synergy confirmed ({syn_type}): acoustic cues and visual kinematics mutually reinforce threat severity."
            )
        elif video_active and not audio_active:
            statements.append(
                "Single modality reliance: acoustic stream is inactive/offline; assessment is based exclusively on visual video stream."
            )
        elif not video_active and audio_active:
            statements.append(
                "Single modality reliance: visual video stream is inactive/offline; assessment is based exclusively on acoustic microphone stream."
            )
        elif assessment is not None and assessment.false_positive_suppressed:
            supp_r = assessment.suppression_reason or "Isolated transient uncorroborated by complementary sensor"
            statements.append(
                f"Noise suppression active: {supp_r} (dampened single-channel spike to prevent false alarm)."
            )
        elif (metrics["video_score"] >= 0.50 and metrics["audio_score"] < 0.20) or (metrics["audio_score"] >= 0.50 and metrics["video_score"] < 0.20):
            statements.append(
                "Divergent sensor cues: visual and acoustic signals diverge, moderating joint decision confidence."
            )

        # I. Critical Rule Override
        if critical:
            statements.append(
                "Critical threat rule triggered: immediate state escalation enforced without waiting for multi-frame confirmation."
            )

        # Fallback for completely nominal condition
        if not statements:
            statements.append("All visual kinematics, spatial density, and acoustic energy levels reside within nominal baseline limits.")

        return {
            "statements": statements,
            "metrics": metrics,
            "audio_details": audio_details,
            "fused_score": fused_score,
            "risk_state": risk_state,
            "dominant_modality": dominant_mod,
            "corroborated": corroborated,
            "critical_evidence": critical,
        }
