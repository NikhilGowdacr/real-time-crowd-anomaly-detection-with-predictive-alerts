"""
==============================================================================
Explanation Formatter (Phase 9).
Formats XAI evidence records into operator-friendly textual representations:
- Concise single-line ticker summaries (format_short)
- In-depth multi-sentence diagnostic audit reports (format_detailed)
- Compact HUD bullet points for surveillance monitors (format_hud)
- Formal alert justifications for operations centers (format_alert_rationale)
==============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple

from src.alerts.alert_record import AlertRecord
from src.xai.evidence_record import EvidenceRecord


class ExplanationFormatter:
    """
    Renders structured EvidenceRecord objects into clean, professional,
    operator-facing text formats with clear factor attribution.
    """

    @staticmethod
    def format_short(record: EvidenceRecord) -> str:
        """
        Produces a concise 1-sentence operator summary.

        Args:
            record: EvidenceRecord instance.

        Returns:
            Single-line string summary.
        """
        state = str(record.risk_state)
        if state == "NORMAL" or record.anomaly_score < 0.20:
            return f"NORMAL: Crowd kinematics, spatial density, and acoustic energy remain within nominal limits (score: {record.anomaly_score:.2f})."

        top_factors = record.ranked_contributions[:2]
        if top_factors:
            factors_desc = " and ".join(
                f"{name.replace('_', ' ')} ({int(w * 100)}%)" for name, w in top_factors
            )
            corrob_str = " with cross-modal corroboration" if record.corroborated else ""
            crit_str = " [CRITICAL OVERRIDE]" if record.critical_evidence else ""
            return f"{state}: Driven primarily by {factors_desc}{corrob_str}{crit_str} (score: {record.anomaly_score:.2f}, conf: {record.decision_confidence:.2f})."

        # Fallback if no contributions are ranked
        return f"{state}: Elevated anomaly score ({record.anomaly_score:.2f}) observed across {record.dominant_modality} modality."

    @staticmethod
    def format_detailed(record: EvidenceRecord) -> str:
        """
        Produces a comprehensive multi-sentence diagnostic audit report.

        Args:
            record: EvidenceRecord instance.

        Returns:
            Formatted diagnostic text block.
        """
        lines = [
            f"=== SURVEILLANCE XAI DIAGNOSTIC REPORT ===",
            f"Timestamp: {record.timestamp:.3f} | Evaluated Risk State: {record.risk_state}",
            f"Anomaly Score: {record.anomaly_score:.4f} | Decision Confidence: {record.decision_confidence:.4f}",
            f"Dominant Modality: {record.dominant_modality.upper()} | Corroborated: {'YES' if record.corroborated else 'NO'}",
            "",
            "1. Observable Factor Contributions (Normalized):",
        ]

        if not record.ranked_contributions:
            lines.append("   - All factor contributions at nominal baseline (0.00%).")
        else:
            for factor, weight in record.ranked_contributions:
                lines.append(f"   - {factor.replace('_', ' ').title():28s}: {weight * 100:5.1f}% contribution")

        lines.append("")
        lines.append("2. Factual Evidence Statements:")
        for stmt in record.top_evidence:
            lines.append(f"   * {stmt}")

        lines.append("")
        lines.append("3. System Model Transparency:")
        saliency_msg = "Grad-CAM feature activation maps active" if record.model_explanation_available else f"Model explanation unavailable ({record.model_explanation_reason})"
        lines.append(f"   * Visual Saliency: {saliency_msg}")
        aud_mode = record.audio_details.get("mode", "spectral_baseline")
        lines.append(f"   * Acoustic Pipeline: {aud_mode} (event: {record.audio_details.get('event_label', 'none')})")

        return "\n".join(lines)

    @staticmethod
    def format_hud(record: EvidenceRecord, max_lines: int = 4) -> List[str]:
        """
        Extracts compact, high-priority bullet points formatted for HUD overlay display.

        Args:
            record: EvidenceRecord instance.
            max_lines: Maximum bullet points to return.

        Returns:
            List of clean text lines.
        """
        hud_lines: List[str] = []

        # 1. State and Score header
        corrob_tag = " [CORROBORATED]" if record.corroborated else ""
        hud_lines.append(f"State: {record.risk_state} ({record.anomaly_score:.2f}){corrob_tag}")

        # 2. Top contributing factor
        if record.ranked_contributions:
            top_f, top_w = record.ranked_contributions[0]
            hud_lines.append(f"Primary: {top_f.replace('_', ' ').title()} ({int(top_w * 100)}%)")

        # 3. Add key factual statements
        for stmt in record.top_evidence:
            if len(hud_lines) >= max_lines:
                break
            clean_stmt = stmt[:44] + ("..." if len(stmt) > 44 else "")
            hud_lines.append(f"* {clean_stmt}")

        return hud_lines[:max_lines]

    @staticmethod
    def format_alert_rationale(
        alert_record: Optional[AlertRecord],
        record: EvidenceRecord,
    ) -> str:
        """
        Generates a formal alert justification explaining WHY an operational alert was emitted.

        Args:
            alert_record: AlertRecord instance (or None).
            record: EvidenceRecord instance.

        Returns:
            Formatted alert rationale string.
        """
        sev = alert_record.severity.value if alert_record else record.risk_state
        alert_id = alert_record.alert_id if alert_record else "LOCAL"

        if record.critical_evidence:
            return (
                f"ALERT RATIONALE [{sev} | {alert_id}]: Immediate critical safety override enforced "
                f"due to severe weapon or violent struggle cues (score: {record.anomaly_score:.2f}, conf: {record.decision_confidence:.2f})."
            )

        if record.corroborated:
            return (
                f"ALERT RATIONALE [{sev} | {alert_id}]: Elevated risk confirmed by synchronized "
                f"cross-modal audio-visual evidence (fused score: {record.anomaly_score:.2f}, conf: {record.decision_confidence:.2f})."
            )

        top_str = ""
        if record.ranked_contributions:
            top_name, top_weight = record.ranked_contributions[0]
            top_str = f" driven primarily by {top_name.replace('_', ' ')} ({int(top_weight * 100)}%)"

        return (
            f"ALERT RATIONALE [{sev} | {alert_id}]: Surveillance threshold exceeded{top_str} "
            f"monitored on {record.dominant_modality.upper()} channel (score: {record.anomaly_score:.2f})."
        )
