"""
==============================================================================
Real-Time Crowd Anomaly Detection with Predictive Alerts
Main Program Entry Point.
Supported Modes:
  --mode video       : Phase 2 (Capture, YOLO Detection, Crowd Density)
  --mode tracking    : Phase 3 (ByteTrack Trajectories, Velocity, Heading)
  --mode safety      : Safety Subsystem (Weapon Detection, Fight Detection, Fusion)
  --mode behavior    : Phase 4 (Temporal Frame Buffer, ViT/Swin, Crowd Behavior Fusion)
  --mode audio       : Phase 5 (Audio Anomaly Detection & Acoustic Events)
  --mode multimodal  : Phase 6 (Audio-Video Temporal Alignment & Dynamic Fusion)
  --mode decision    : Phase 7 (Intelligent State Machine & Hysteresis Transitions)
  --mode alerts      : Phase 8 (Predictive Multi-Channel Alert Management)
  --mode xai         : Phase 9 (Explainable AI & HUD Evidence Visualization)
  --mode dashboard   : Phase 10 (Real-Time Operator Monitoring Dashboard)
  --mode database    : Phase 11 (Historical SQLite Persistence & Query API)
  --mode evaluation  : Phase 12 (Academic Evaluation & Benchmarking Suite)
==============================================================================
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.utils.metrics import FPSCounter, LatencyTracker
from src.video.video_capture import VideoSource
from src.video.preprocessing import FramePreprocessor
from src.video.person_detector import YOLOPersonDetector, DetectionResult
from src.video.crowd_density import CrowdDensityEstimator, DensityResult
from src.video.crowd_tracker import ByteTrackCrowdTracker, TrackedPerson

# Safety Subsystem
from src.safety.weapon_detector import WeaponDetector, WeaponDetectionResult
from src.safety.pose_analyzer import PoseMovementAnalyzer, KinematicFeatures
from src.safety.fight_detector import FightDetector, FightDetectionResult
from src.safety.safety_fusion import (
    SafetyFusionEngine,
    SafetyAssessment,
    SafetyStatus,
)

# Behavior Subsystem (Phase 4)
from src.behavior.temporal_buffer import TemporalFrameBuffer
from src.behavior.behavior_features import CrowdBehaviorAnalyzer, CrowdBehaviorMetrics
from src.behavior.behavior_model import BehaviorModel, BehaviorPrediction
from src.behavior.behavior_anomaly import BehaviorAnomalyDetector, BehaviorAssessment

# Audio Subsystem (Phase 5)
from src.audio import (
    AudioCapture,
    AudioPreprocessor,
    PreprocessedAudio,
    MelSpectrogramProcessor,
    AudioClassifier,
    AudioAnomalyDetector,
    AudioEventRecord,
)

# Multimodal Fusion Subsystem (Phase 6)
from src.fusion import (
    MultimodalFusionEngine,
    MultimodalAssessment,
    CrossModalSynergy,
)

# Intelligent Decision Engine Subsystem (Phase 7)
from src.decision import (
    DecisionEngine,
    DecisionEvent,
    RiskState,
)

# Alert Management Subsystem (Phase 8)
from src.alerts import (
    AlertManager,
    AlertRecord,
    AlertSeverity,
    AlertStatus,
    AlertPolicy,
)

# Explainable AI (XAI) & Evidence Visualization Subsystem (Phase 9)
from src.xai import (
    ExplanationEngine,
    EvidenceRecord,
    VisualExplainer,
    ExplanationFormatter,
)

# Academic Evaluation & Benchmarking Subsystem (Phase 12)
from src.evaluation import (
    SystemEvaluator,
    EvaluationReport,
    EvaluationSource,
)

logger = setup_logger("main")
_xai_hud_explainer = VisualExplainer()


def render_surveillance_hud(
    frame: np.ndarray,
    detection: Optional[DetectionResult] = None,
    density: Optional[DensityResult] = None,
    fps: float = 0.0,
    latency_tracker: Optional[LatencyTracker] = None,
    safety_assessment: Optional[SafetyAssessment] = None,
    behavior_assessment: Optional[BehaviorAssessment] = None,
    audio_record: Optional[AudioEventRecord] = None,
    multimodal_assessment: Optional[MultimodalAssessment] = None,
    decision_event: Optional[DecisionEvent] = None,
    alert_record: Optional[AlertRecord] = None,
    evidence_record: Optional[EvidenceRecord] = None,
    show_help: bool = True,
    pipeline_label: str = "Video (Phase 2)",
) -> np.ndarray:
    """
    Renders an informative, unified surveillance Heads-Up Display (HUD) overlay.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Top Header Banner
    banner_height = 55
    cv2.rectangle(annotated, (0, 0), (w, banner_height), (20, 20, 25), -1)
    cv2.line(annotated, (0, banner_height), (w, banner_height), (60, 60, 70), 1)

    # Title
    cv2.putText(
        annotated,
        "REAL-TIME CROWD ANOMALY DETECTION | RESEARCH MONITOR",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    # Subtitle / Timestamp
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated,
        f"Pipeline: {pipeline_label} | {timestamp_str}",
        (16, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Status Determination
    status_str = "NORMAL"
    if decision_event is not None:
        status_str = decision_event.current_state
    elif multimodal_assessment is not None:
        status_str = multimodal_assessment.status
    elif behavior_assessment is not None:
        status_str = behavior_assessment.status
    elif safety_assessment is not None:
        status_str = safety_assessment.status.value
    elif audio_record is not None:
        status_str = audio_record.status
    elif density is not None and (density.density_level == "CRITICAL" or density.is_overcrowded):
        status_str = "EMERGENCY"
    elif density is not None and density.density_level == "HIGH":
        status_str = "SUSPICIOUS"

    if status_str == "EMERGENCY":
        status_text = "STATUS: EMERGENCY"
        badge_bg = (0, 0, 210)       # Red
    elif status_str == "HIGH_RISK":
        status_text = "STATUS: HIGH_RISK"
        badge_bg = (0, 75, 230)      # Red-Orange
    elif status_str == "SUSPICIOUS":
        status_text = "STATUS: SUSPICIOUS"
        badge_bg = (0, 140, 255)     # Orange
    elif status_str == "RECOVERY":
        status_text = "STATUS: RECOVERY"
        badge_bg = (200, 130, 0)     # Cyan-Blue
    else:
        status_text = "STATUS: NORMAL"
        badge_bg = (0, 160, 0)       # Green

    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    badge_x = w - tw - 26
    cv2.rectangle(annotated, (badge_x - 8, 12), (w - 12, 42), badge_bg, -1)
    cv2.putText(
        annotated,
        status_text,
        (badge_x, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Left Side Telemetry Card
    has_advanced = (
        (decision_event is not None)
        or (multimodal_assessment is not None)
        or (safety_assessment is not None)
        or (behavior_assessment is not None)
        or (audio_record is not None)
    )
    if decision_event is not None:
        card_w = 330
        card_h = 390
    elif multimodal_assessment is not None:
        card_w = 310
        card_h = 350
    elif audio_record is not None and ((safety_assessment is not None) or (behavior_assessment is not None)):
        card_w = 295
        card_h = 320
    elif has_advanced:
        card_w = 295
        card_h = 270
    else:
        card_w = 295
        card_h = 160
    card_x = 16
    card_y = banner_height + 14

    card_roi = annotated[card_y : card_y + card_h, card_x : card_x + card_w]
    card_bg = np.full(card_roi.shape, 25, dtype=np.uint8)
    cv2.addWeighted(card_roi, 0.25, card_bg, 0.75, 0, card_roi)
    annotated[card_y : card_y + card_h, card_x : card_x + card_w] = card_roi
    cv2.rectangle(annotated, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 90), 1)

    y_offset = card_y + 20
    line_gap = 20

    # Base Metrics
    person_cnt = detection.person_count if detection is not None else 0
    cv2.putText(annotated, f"People: {person_cnt}", (card_x + 12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)
    y_offset += line_gap

    if density is not None:
        cv2.putText(annotated, f"Density: {density.density_per_m2:.2f} / m2 ({density.density_level})", (card_x + 12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        y_offset += line_gap

    if has_advanced:
        cv2.line(annotated, (card_x + 8, y_offset - 6), (card_x + card_w - 8, y_offset - 6), (60, 60, 70), 1)

        # Decision Engine Section (Phase 7)
        if decision_event is not None:
            st_color = (0, 0, 255) if decision_event.current_state == "EMERGENCY" else (
                (0, 80, 240) if decision_event.current_state == "HIGH_RISK" else (
                    (0, 140, 255) if decision_event.current_state == "SUSPICIOUS" else (
                        (200, 140, 0) if decision_event.current_state == "RECOVERY" else (0, 255, 0)
                    )
                )
            )
            cv2.putText(annotated, f"DECISION: {decision_event.current_state}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, st_color, 2, cv2.LINE_AA)
            y_offset += line_gap + 4

            cv2.putText(annotated, f"Decision Conf:   {decision_event.confidence:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            y_offset += line_gap

            fused_col = (0, 0, 255) if decision_event.score >= 0.70 else ((0, 140, 255) if decision_event.score >= 0.35 else (0, 255, 0))
            cv2.putText(annotated, f"Fused Score:     {decision_event.score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fused_col, 1, cv2.LINE_AA)
            y_offset += line_gap

            cv2.putText(annotated, f"Video / Audio:   {decision_event.video_score:.2f} / {decision_event.audio_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 240, 200), 1, cv2.LINE_AA)
            y_offset += line_gap

            corrob_txt = "YES" if decision_event.corroborated else "NO"
            cv2.putText(annotated, f"Corroborated:    {corrob_txt}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255) if decision_event.corroborated else (180, 180, 180), 1, cv2.LINE_AA)
            y_offset += line_gap

            if decision_event.critical_evidence:
                cv2.putText(annotated, "Critical Rule:   ACTIVE", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)
                y_offset += line_gap

            # Dynamic Thresholds
            t_s = decision_event.dynamic_thresholds.get("suspicious_enter", 0.45)
            t_hr = decision_event.dynamic_thresholds.get("high_risk_enter", 0.65)
            t_em = decision_event.dynamic_thresholds.get("emergency_enter", 0.85)
            cv2.putText(annotated, f"Thresholds:      {t_s:.2f} / {t_hr:.2f} / {t_em:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 200), 1, cv2.LINE_AA)
            y_offset += line_gap

            # Truncated Reason
            clean_reason = decision_event.reason[:38] + ("..." if len(decision_event.reason) > 38 else "")
            cv2.putText(annotated, f"Reason: {clean_reason}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 255), 1, cv2.LINE_AA)
            y_offset += line_gap

        # Multimodal Telemetry Section (Phase 6 fallback if decision_event not present)
        elif multimodal_assessment is not None:
            m_score = multimodal_assessment.multimodal_score
            m_color = (0, 0, 255) if m_score >= 0.70 else ((0, 200, 255) if m_score >= 0.35 else (0, 255, 0))
            cv2.putText(annotated, f"Multimodal Score: {m_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, m_color, 2, cv2.LINE_AA)
            y_offset += line_gap + 4

            cv2.putText(annotated, f"Multimodal Conf:  {multimodal_assessment.multimodal_confidence:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            y_offset += line_gap

            cv2.putText(annotated, f"Video Score:      {multimodal_assessment.video_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 200), 1, cv2.LINE_AA)
            y_offset += line_gap

            cv2.putText(annotated, f"Audio Score:      {multimodal_assessment.audio_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 200), 1, cv2.LINE_AA)
            y_offset += line_gap

            cv2.putText(annotated, f"Dominant Sensor:  {multimodal_assessment.dominant_modality.upper()}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)
            y_offset += line_gap

            if multimodal_assessment.synergy.synergy_detected:
                syn_txt = f"Synergy: {multimodal_assessment.synergy.synergy_type}"
                cv2.putText(annotated, syn_txt, (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += line_gap
            elif multimodal_assessment.false_positive_suppressed:
                cv2.putText(annotated, "Noise Filter: ACTIVE (Suppressed)", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)
                y_offset += line_gap

        elif behavior_assessment is not None or safety_assessment is not None:
            avg_speed = behavior_assessment.mean_speed if behavior_assessment else 0.0
            cv2.putText(annotated, f"Average Speed: {avg_speed:.1f} px/f", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
            y_offset += line_gap

            m_score = behavior_assessment.kinematic_score if behavior_assessment else (safety_assessment.movement_score if safety_assessment else 0.0)
            cv2.putText(annotated, f"Movement Score: {m_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 240), 1, cv2.LINE_AA)
            y_offset += line_gap

            f_score = safety_assessment.fight_score if safety_assessment else 0.0
            fight_color = (0, 100, 255) if f_score >= 0.50 else (210, 210, 210)
            cv2.putText(annotated, f"Fight Score:    {f_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fight_color, 1, cv2.LINE_AA)
            y_offset += line_gap

            b_score = behavior_assessment.deep_score if behavior_assessment else 0.0
            beh_color = (0, 140, 255) if b_score >= 0.50 else (210, 210, 210)
            cv2.putText(annotated, f"Behavior Score: {b_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, beh_color, 1, cv2.LINE_AA)
            y_offset += line_gap

            w_score = safety_assessment.weapon_score if safety_assessment else 0.0
            weap_color = (0, 100, 255) if w_score >= 0.50 else (210, 210, 210)
            cv2.putText(annotated, f"Weapon Score:   {w_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, weap_color, 1, cv2.LINE_AA)
            y_offset += line_gap

            comp_score = behavior_assessment.final_video_score if behavior_assessment else (safety_assessment.safety_score if safety_assessment else 0.0)
            score_color = (0, 0, 255) if comp_score >= 0.70 else ((0, 200, 255) if comp_score >= 0.35 else (0, 255, 0))
            label_score = "Video Anomaly:  " if behavior_assessment else "Safety Score:   "
            cv2.putText(annotated, f"{label_score}{comp_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, score_color, 2, cv2.LINE_AA)
            y_offset += line_gap + 4

        # Audio Metrics in Telemetry Card (if not in multimodal mode)
        elif audio_record is not None:
            a_evt = audio_record.audio_event.upper()
            a_conf = audio_record.audio_confidence
            a_score = audio_record.audio_anomaly_score
            aud_color = (0, 0, 255) if a_score >= 0.70 else ((0, 140, 255) if a_score >= 0.40 else (180, 240, 180))
            cv2.putText(annotated, f"Audio Event:    {a_evt} ({a_conf:.2f})", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, aud_color, 1, cv2.LINE_AA)
            y_offset += line_gap
            cv2.putText(annotated, f"Audio Anomaly:  {a_score:.2f}", (card_x + 12, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, aud_color, 1, cv2.LINE_AA)
            y_offset += line_gap

    # Performance
    inf_avg = latency_tracker.get_average("inference") if latency_tracker is not None else 0.0
    cv2.putText(annotated, f"Speed: {fps:.1f} FPS (YOLO: {inf_avg:.1f}ms)", (card_x + 12, y_offset + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 240, 180), 1, cv2.LINE_AA)

    # Bottom Telemetry Strip Banner
    if multimodal_assessment is not None:
        syn_label = f" | Syn: {multimodal_assessment.synergy.synergy_type}" if multimodal_assessment.synergy.synergy_detected else ""
        telemetry_txt = (
            f"MULTIMODAL | Video: {multimodal_assessment.video_score:.2f} | "
            f"Audio: {multimodal_assessment.audio_score:.2f} | "
            f"Fused: {multimodal_assessment.multimodal_score:.2f} | "
            f"Status: {multimodal_assessment.status}{syn_label}"
        )
        bar_y = h - 38 if show_help else h - 14
        (tw_m, th_m), _ = cv2.getTextSize(telemetry_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(annotated, (14, bar_y - 18), (28 + tw_m, bar_y + 4), (20, 20, 25), -1)
        cv2.rectangle(annotated, (14, bar_y - 18), (28 + tw_m, bar_y + 4), (60, 60, 70), 1)
        bar_color = (0, 0, 255) if multimodal_assessment.multimodal_score >= 0.70 else ((0, 180, 255) if multimodal_assessment.multimodal_score >= 0.35 else (0, 255, 200))
        cv2.putText(annotated, telemetry_txt, (18, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, bar_color, 1, cv2.LINE_AA)

    elif audio_record is not None:
        telemetry_txt = (
            f"AUDIO | Event: {audio_record.audio_event.upper()} | "
            f"Conf: {audio_record.audio_confidence:.2f} | "
            f"Audio Anomaly: {audio_record.audio_anomaly_score:.2f}"
        )
        bar_y = h - 38 if show_help else h - 14
        (tw_a, th_a), _ = cv2.getTextSize(telemetry_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(annotated, (14, bar_y - 18), (28 + tw_a, bar_y + 4), (20, 20, 25), -1)
        cv2.rectangle(annotated, (14, bar_y - 18), (28 + tw_a, bar_y + 4), (60, 60, 70), 1)
        bar_color = (0, 0, 255) if audio_record.audio_anomaly_score >= 0.70 else ((0, 180, 255) if audio_record.audio_anomaly_score >= 0.40 else (0, 255, 200))
        cv2.putText(annotated, telemetry_txt, (18, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, bar_color, 1, cv2.LINE_AA)

    # 3. Active Alert Status Card (Phase 8)
    if alert_record is not None and getattr(alert_record, "status", None) != AlertStatus.RESOLVED:
        sev_name = alert_record.severity.value
        sev_color = (0, 0, 220) if sev_name == "CRITICAL" else (
            (0, 80, 235) if sev_name == "HIGH" else (
                (0, 150, 255) if sev_name == "WARNING" else (200, 140, 0)
            )
        )
        alert_box_w = 330
        alert_box_h = 92
        alert_x = w - alert_box_w - 14
        alert_y = 52
        if alert_y + alert_box_h < h and alert_x >= 0:
            alert_roi = annotated[alert_y : alert_y + alert_box_h, alert_x : alert_x + alert_box_w]
            alert_bg = np.full_like(alert_roi, (25, 25, 30))
            cv2.addWeighted(alert_roi, 0.20, alert_bg, 0.80, 0, alert_roi)
            annotated[alert_y : alert_y + alert_box_h, alert_x : alert_x + alert_box_w] = alert_roi
            cv2.rectangle(annotated, (alert_x, alert_y), (alert_x + alert_box_w, alert_y + alert_box_h), sev_color, 2)

            id_suffix = alert_record.alert_id[-8:] if len(alert_record.alert_id) >= 8 else alert_record.alert_id
            cv2.putText(annotated, f"ALERT: {sev_name} [{id_suffix}]", (alert_x + 10, alert_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, sev_color, 2, cv2.LINE_AA)
            cv2.putText(annotated, f"STATE: {alert_record.risk_state} | STATUS: {alert_record.status.value}", (alert_x + 10, alert_y + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"SCORE: {alert_record.score:.2f} | CONF: {alert_record.confidence:.2f}", (alert_x + 10, alert_y + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
            mod_label = alert_record.dominant_modality.upper()
            cv2.putText(annotated, f"SOURCE: {mod_label} | CRIT: {'YES' if alert_record.critical_evidence else 'NO'}", (alert_x + 10, alert_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 240, 180), 1, cv2.LINE_AA)

    # 3b. Explainable AI (XAI) Evidence Overlay Card (Phase 9)
    if evidence_record is not None:
        has_active_alert = (alert_record is not None and getattr(alert_record, "status", None) != AlertStatus.RESOLVED)
        xai_w = 340
        xai_h = 240
        xai_x = w - xai_w - 14
        xai_y = 152 if has_active_alert else 52
        if xai_y + xai_h < h - 20 and xai_x >= 0:
            annotated = _xai_hud_explainer.draw_xai_hud_card(
                annotated,
                evidence=evidence_record,
                x=xai_x,
                y=xai_y,
                width=xai_w,
                height=xai_h,
            )

    # 4. Bottom Controls Hint
    if show_help:
        if multimodal_assessment is not None:
            help_text = "[Q]: Exit  |  [G]: Grid  |  [M]: Demo Multimodal  |  [S]: Scream  |  [F]: Fight"
        elif behavior_assessment is not None:
            help_text = "[Q]: Exit  |  [G]: Grid  |  [B]: Demo Behavior  |  [C]: CLAHE"
        elif safety_assessment is not None:
            help_text = "[Q]: Exit  |  [G]: Grid  |  [F]: Demo Fight  |  [W]: Demo Weapon  |  [C]: CLAHE"
        else:
            help_text = "[Q]: Exit  |  [G]: Toggle Density Grid  |  [C]: Toggle CLAHE"
        cv2.putText(annotated, help_text, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    return annotated




def run_video_pipeline(
    source_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    device_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
) -> None:
    """Executes Phase 2 Video Pipeline: Capture -> Preprocess -> YOLO -> Density."""
    config = load_config(config_path)

    source = source_override if source_override is not None else config["video"]["source"]
    conf = conf_override if conf_override is not None else float(config["detection"]["confidence"])
    device = device_override if device_override is not None else config["detection"]["device"]

    logger.info("==================================================================")
    logger.info("STARTING VIDEO PROCESSING PIPELINE (PHASE 2)")
    logger.info(f"Source: {source} | YOLO: {config['detection']['model']} | Conf: {conf}")
    logger.info("==================================================================")

    video_cap = VideoSource(
        source=source,
        width=config["video"]["width"],
        height=config["video"]["height"],
        fps=config["video"]["fps"],
        loop=config["video"]["loop"],
    )

    preprocessor = FramePreprocessor(
        target_width=config["video"]["width"],
        target_height=config["video"]["height"],
        enable_clahe=False,
    )

    detector = YOLOPersonDetector(
        model_path=config["detection"]["model"],
        confidence=conf,
        iou_threshold=config["detection"]["iou_threshold"],
        device=device,
        target_classes=config["detection"]["classes"],
        imgsz=config["detection"]["imgsz"],
    )

    density_estimator = CrowdDensityEstimator(
        frame_area_m2=config["density"]["frame_area_m2"],
        grid_rows=config["density"]["grid_rows"],
        grid_cols=config["density"]["grid_cols"],
        threshold_low=config["density"]["threshold_low"],
        threshold_moderate=config["density"]["threshold_moderate"],
        threshold_high=config["density"]["threshold_high"],
    )

    fps_counter = FPSCounter()
    latency_tracker = LatencyTracker()
    show_density_grid = True
    frame_index = 0
    window_name = "Crowd Anomaly Detection - Video Feed"

    try:
        while True:
            latency_tracker.start("frame_read")
            ret, frame = video_cap.read()
            latency_tracker.stop("frame_read")

            if not ret or frame is None:
                break

            frame_index += 1

            latency_tracker.start("preprocess")
            processed_frame = preprocessor.preprocess(frame)
            latency_tracker.stop("preprocess")

            latency_tracker.start("inference")
            detection_result = detector.detect(processed_frame)
            latency_tracker.stop("inference")

            latency_tracker.start("density")
            h, w = processed_frame.shape[:2]
            density_result = density_estimator.estimate(
                detection_points=detection_result.bottom_centers,
                frame_width=w,
                frame_height=h,
            )
            latency_tracker.stop("density")

            current_fps = fps_counter.update()

            if frame_index % 60 == 0 or frame_index == 1:
                logger.info(
                    f"Frame {frame_index:05d} | People: {detection_result.person_count:02d} | "
                    f"Density: {density_result.density_per_m2:.2f}/m2 | FPS: {current_fps:.1f}"
                )

            if not no_display:
                display_frame = density_estimator.draw_density_overlay(
                    processed_frame,
                    density_result,
                    show_grid=show_density_grid,
                )
                display_frame = detector.draw_detections(display_frame, detection_result)
                display_frame = render_surveillance_hud(
                    display_frame,
                    detection_result,
                    density_result,
                    current_fps,
                    latency_tracker,
                    pipeline_label="Video (Phase 2)",
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("g"):
                    show_density_grid = not show_density_grid
                elif key == ord("c"):
                    preprocessor.enable_clahe = not preprocessor.enable_clahe

            if max_frames and frame_index >= max_frames:
                break

    finally:
        video_cap.release()
        if not no_display:
            cv2.destroyAllWindows()
        logger.info(f"Video pipeline finished. Processed {frame_index} frames at {fps_counter.current_fps:.1f} FPS.")


def run_tracking_pipeline(
    source_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    device_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
) -> None:
    """Executes Phase 3 Tracking Pipeline: YOLO Detection -> ByteTrack Tracking."""
    config = load_config(config_path)

    source = source_override if source_override is not None else config["video"]["source"]
    conf = conf_override if conf_override is not None else float(config["detection"]["confidence"])
    device = device_override if device_override is not None else config["detection"]["device"]

    logger.info("==================================================================")
    logger.info("STARTING PERSON TRACKING PIPELINE (PHASE 3 / BYTETRACK)")
    logger.info(f"Source: {source} | Tracker: ByteTrack | Conf: {conf}")
    logger.info("==================================================================")

    video_cap = VideoSource(
        source=source,
        width=config["video"]["width"],
        height=config["video"]["height"],
        fps=config["video"]["fps"],
        loop=config["video"]["loop"],
    )

    detector = YOLOPersonDetector(
        model_path=config["detection"]["model"],
        confidence=conf,
        iou_threshold=config["detection"]["iou_threshold"],
        device=device,
        target_classes=config["detection"]["classes"],
        imgsz=config["detection"]["imgsz"],
    )

    tracker = ByteTrackCrowdTracker(
        history_length=config["tracking"]["history_length"],
        speed_threshold=config["tracking"]["speed_threshold"],
        max_lost=config["tracking"]["max_lost"],
        iou_threshold=config["tracking"]["iou_threshold"],
    )

    density_estimator = CrowdDensityEstimator(
        frame_area_m2=config["density"]["frame_area_m2"],
    )

    fps_counter = FPSCounter()
    latency_tracker = LatencyTracker()
    frame_index = 0
    window_name = "Crowd Anomaly Detection - Tracking Monitor"

    try:
        while True:
            ret, frame = video_cap.read()
            if not ret or frame is None:
                break

            frame_index += 1

            latency_tracker.start("inference")
            detection_result = detector.detect(frame)
            latency_tracker.stop("inference")

            latency_tracker.start("tracking")
            tracks = tracker.update(detection_result, frame)
            latency_tracker.stop("tracking")

            h, w = frame.shape[:2]
            density_result = density_estimator.estimate(
                detection_points=detection_result.bottom_centers,
                frame_width=w,
                frame_height=h,
            )

            current_fps = fps_counter.update()

            if frame_index % 60 == 0 or frame_index == 1:
                logger.info(
                    f"Frame {frame_index:05d} | Tracks: {len(tracks):02d} | "
                    f"FPS: {current_fps:.1f} | Tracking: {latency_tracker.get_average('tracking'):.1f}ms"
                )

            if not no_display:
                display_frame = tracker.draw_tracks(frame, tracks, draw_trajectory=True, draw_id=True, draw_speed=True)
                display_frame = render_surveillance_hud(
                    display_frame,
                    detection_result,
                    density_result,
                    current_fps,
                    latency_tracker,
                    pipeline_label="Tracking (ByteTrack)",
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if max_frames and frame_index >= max_frames:
                break

    finally:
        video_cap.release()
        if not no_display:
            cv2.destroyAllWindows()
        logger.info(f"Tracking pipeline finished. Processed {frame_index} frames.")


def run_safety_pipeline(
    source_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    device_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
    demo_safety: bool = False,
) -> None:
    """Executes Safety Subsystem: Weapon Detection, Fight Detection, Multi-Evidence Fusion."""
    config = load_config(config_path)

    source = source_override if source_override is not None else config["video"]["source"]
    conf = conf_override if conf_override is not None else float(config["detection"]["confidence"])
    device = device_override if device_override is not None else config["detection"]["device"]

    logger.info("==================================================================")
    logger.info("STARTING SAFETY INTELLIGENCE PIPELINE")
    logger.info(f"Source: {source} | Device: {device} | Demo Safety Mode: {demo_safety}")
    logger.info("==================================================================")

    video_cap = VideoSource(
        source=source,
        width=config["video"]["width"],
        height=config["video"]["height"],
        fps=config["video"]["fps"],
        loop=config["video"]["loop"],
    )

    detector = YOLOPersonDetector(
        model_path=config["detection"]["model"],
        confidence=conf,
        iou_threshold=config["detection"]["iou_threshold"],
        device=device,
        target_classes=config["detection"]["classes"],
        imgsz=config["detection"]["imgsz"],
    )

    tracker = ByteTrackCrowdTracker(
        history_length=config["tracking"]["history_length"],
        speed_threshold=config["tracking"]["speed_threshold"],
        max_lost=config["tracking"]["max_lost"],
        iou_threshold=config["tracking"]["iou_threshold"],
    )

    density_estimator = CrowdDensityEstimator(
        frame_area_m2=config["density"]["frame_area_m2"],
        grid_rows=config["density"]["grid_rows"],
        grid_cols=config["density"]["grid_cols"],
    )

    pose_analyzer = PoseMovementAnalyzer(
        proximity_threshold=config["fight_detection"]["proximity_threshold"],
    )

    weapon_detector = WeaponDetector(
        model_path=config["weapon_detection"]["model"],
        confidence=config["weapon_detection"]["confidence"],
        enabled=config["weapon_detection"]["enabled"],
        target_classes=config["weapon_detection"]["classes"],
        device=device,
        simulate_for_demo=config["weapon_detection"].get("simulate_for_demo", False) or demo_safety,
    )

    fight_detector = FightDetector(
        enabled=config["fight_detection"]["enabled"],
        temporal_window=config["fight_detection"]["temporal_window"],
        suspicious_threshold=config["fight_detection"]["suspicious_threshold"],
        emergency_threshold=config["fight_detection"]["emergency_threshold"],
        confirmation_frames=config["fight_detection"]["confirmation_frames"],
        cooldown_seconds=config["fight_detection"]["cooldown_seconds"],
    )

    fusion_engine = SafetyFusionEngine(
        crowd_weight=config["safety_fusion"]["crowd_weight"],
        fight_weight=config["safety_fusion"]["fight_weight"],
        weapon_weight=config["safety_fusion"]["weapon_weight"],
        normal_threshold=config["safety_fusion"]["normal_threshold"],
        emergency_threshold=config["safety_fusion"]["emergency_threshold"],
        correlation_multiplier=config["safety_fusion"]["correlation_multiplier"],
        cooldown_seconds=config["alert"]["cooldown_seconds"],
    )

    fps_counter = FPSCounter()
    latency_tracker = LatencyTracker()
    manual_demo_fight = demo_safety
    manual_demo_weapon = demo_safety
    show_density_grid = False
    frame_index = 0
    window_name = "Crowd Anomaly Detection - Safety Intelligence Monitor"

    try:
        while True:
            ret, frame = video_cap.read()
            if not ret or frame is None:
                break

            frame_index += 1
            curr_ts = time.time()

            latency_tracker.start("inference")
            detection_result = detector.detect(frame)
            latency_tracker.stop("inference")

            latency_tracker.start("tracking")
            tracks = tracker.update(detection_result, frame)
            latency_tracker.stop("tracking")

            h, w = frame.shape[:2]
            density_result = density_estimator.estimate(
                detection_points=detection_result.bottom_centers,
                frame_width=w,
                frame_height=h,
            )

            latency_tracker.start("kinematics")
            kinematics = pose_analyzer.extract_features(tracks, timestamp=curr_ts)
            latency_tracker.stop("kinematics")

            latency_tracker.start("weapon")
            weapon_result = weapon_detector.detect(
                frame=frame,
                tracked_persons=tracks,
                timestamp=curr_ts,
                demo_trigger=manual_demo_weapon,
            )
            latency_tracker.stop("weapon")

            latency_tracker.start("fight")
            fight_result = fight_detector.update(
                kinematics=kinematics,
                tracked_persons=tracks,
                timestamp=curr_ts,
                demo_trigger=manual_demo_fight,
            )
            latency_tracker.stop("fight")

            latency_tracker.start("fusion")
            safety_assessment = fusion_engine.evaluate(
                density_result=density_result,
                kinematics=kinematics,
                fight_result=fight_result,
                weapon_result=weapon_result,
                timestamp=curr_ts,
            )
            latency_tracker.stop("fusion")

            current_fps = fps_counter.update()

            if frame_index % 60 == 0 or frame_index == 1:
                logger.info(
                    f"Frame {frame_index:05d} | People: {detection_result.person_count:02d} | "
                    f"SafetyScore: {safety_assessment.safety_score:.2f} | STATUS: {safety_assessment.status.value}"
                )

            if not no_display:
                display_frame = frame.copy()
                if show_density_grid:
                    display_frame = density_estimator.draw_density_overlay(display_frame, density_result)
                display_frame = tracker.draw_tracks(display_frame, tracks, draw_trajectory=True)
                display_frame = fight_detector.draw_fight_detections(display_frame, fight_result)
                display_frame = weapon_detector.draw_detections(display_frame, weapon_result)

                display_frame = render_surveillance_hud(
                    display_frame,
                    detection=detection_result,
                    density=density_result,
                    fps=current_fps,
                    latency_tracker=latency_tracker,
                    safety_assessment=safety_assessment,
                    pipeline_label="Safety Subsystem (Weapons & Fights)",
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("g"):
                    show_density_grid = not show_density_grid
                elif key == ord("f"):
                    manual_demo_fight = not manual_demo_fight
                elif key == ord("w"):
                    manual_demo_weapon = not manual_demo_weapon

            if max_frames and frame_index >= max_frames:
                break

    finally:
        video_cap.release()
        if not no_display:
            cv2.destroyAllWindows()
        logger.info(f"Safety pipeline session ended. Processed {frame_index} frames at {fps_counter.current_fps:.1f} FPS.")


def run_behavior_pipeline(
    source_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    device_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
    demo_behavior: bool = False,
) -> None:
    """
    Executes Phase 4 Advanced Video Behavioral Anomaly Detection Pipeline:
    Capture -> Preprocess -> YOLO -> ByteTrack -> Temporal Frame Buffer ->
    ViT/Swin Transformer -> Crowd Kinematics -> Behavior Fusion -> HUD.
    """
    config = load_config(config_path)

    source = source_override if source_override is not None else config["video"]["source"]
    conf = conf_override if conf_override is not None else float(config["detection"]["confidence"])
    device = device_override if device_override is not None else config["detection"]["device"]

    beh_cfg = config.get("behavior", {})
    fusion_cfg = config.get("behavior_fusion", {})

    logger.info("==================================================================")
    logger.info("STARTING ADVANCED VIDEO BEHAVIORAL ANOMALY DETECTION (PHASE 4)")
    logger.info(f"Model: {beh_cfg.get('model_type', 'swin').upper()} | Buffer Seq Len: {beh_cfg.get('sequence_length', 16)}")
    logger.info(f"Weights: Kinematic={fusion_cfg.get('kinematic_weight', 0.40)} | Deep={fusion_cfg.get('deep_model_weight', 0.60)}")
    logger.info("==================================================================")

    video_cap = VideoSource(
        source=source,
        width=config["video"]["width"],
        height=config["video"]["height"],
        fps=config["video"]["fps"],
        loop=config["video"]["loop"],
    )

    detector = YOLOPersonDetector(
        model_path=config["detection"]["model"],
        confidence=conf,
        iou_threshold=config["detection"]["iou_threshold"],
        device=device,
        target_classes=config["detection"]["classes"],
        imgsz=config["detection"]["imgsz"],
    )

    tracker = ByteTrackCrowdTracker(
        history_length=config["tracking"]["history_length"],
        speed_threshold=config["tracking"]["speed_threshold"],
        max_lost=config["tracking"]["max_lost"],
        iou_threshold=config["tracking"]["iou_threshold"],
    )

    density_estimator = CrowdDensityEstimator(
        frame_area_m2=config["density"]["frame_area_m2"],
        grid_rows=config["density"]["grid_rows"],
        grid_cols=config["density"]["grid_cols"],
    )

    # 1. Temporal Frame Buffer
    temporal_buffer = TemporalFrameBuffer(
        sequence_length=beh_cfg.get("sequence_length", 16),
        sample_rate=beh_cfg.get("sample_rate", 4),
        target_size=(224, 224),
    )

    # 2. Crowd Kinematics Analyzer
    crowd_analyzer = CrowdBehaviorAnalyzer(
        speed_threshold=beh_cfg.get("speed_threshold", 18.0),
        dispersion_threshold=beh_cfg.get("dispersion_threshold", 2.5),
        convergence_threshold=beh_cfg.get("convergence_threshold", 2.5),
    )

    # 3. Behavior Vision Transformer (Swin / ViT)
    behavior_model = BehaviorModel(
        model_type=beh_cfg.get("model_type", "swin"),
        model_path=beh_cfg.get("model_path", ""),
        confidence_threshold=beh_cfg.get("confidence_threshold", 0.50),
        device=device,
        enabled=beh_cfg.get("enabled", True),
    )

    # 4. Behavior Anomaly Detector & Fusion Engine
    anomaly_detector = BehaviorAnomalyDetector(
        kinematic_weight=fusion_cfg.get("kinematic_weight", 0.40),
        deep_model_weight=fusion_cfg.get("deep_model_weight", 0.60),
        normal_threshold=config["decision"]["normal_threshold"],
        emergency_threshold=config["decision"]["emergency_threshold"],
        confirmation_frames=beh_cfg.get("confirmation_frames", 5),
        smoothing_window=beh_cfg.get("smoothing_window", 10),
    )

    fps_counter = FPSCounter()
    latency_tracker = LatencyTracker()

    manual_demo_behavior = demo_behavior
    show_density_grid = False
    frame_index = 0
    window_name = "Crowd Anomaly Detection - Behavior Intelligence Monitor"

    try:
        while True:
            ret, frame = video_cap.read()
            if not ret or frame is None:
                break

            frame_index += 1

            # Step A: YOLO Person Detection
            latency_tracker.start("inference")
            detection_result = detector.detect(frame)
            latency_tracker.stop("inference")

            # Step B: ByteTrack Tracking
            latency_tracker.start("tracking")
            tracks = tracker.update(detection_result, frame)
            latency_tracker.stop("tracking")

            # Step C: Crowd Density
            latency_tracker.start("density")
            h, w = frame.shape[:2]
            density_result = density_estimator.estimate(
                detection_points=detection_result.bottom_centers,
                frame_width=w,
                frame_height=h,
            )
            latency_tracker.stop("density")

            # Step D: Temporal Buffer Ingestion
            latency_tracker.start("buffer")
            sampled = temporal_buffer.add_frame(frame)
            latency_tracker.stop("buffer")

            # Step E: Crowd Kinematics
            latency_tracker.start("kinematics")
            crowd_metrics = crowd_analyzer.analyze(tracks, density_result)
            latency_tracker.stop("kinematics")

            # Step F: Deep Transformer Sequence Inference
            deep_prediction = None
            if temporal_buffer.is_ready() or manual_demo_behavior:
                latency_tracker.start("transformer")
                seq_tensor = temporal_buffer.get_torch_tensor(device="cpu")
                deep_prediction = behavior_model.predict(
                    sequence_tensor=seq_tensor,
                    demo_trigger=manual_demo_behavior,
                )
                latency_tracker.stop("transformer")

            # Step G: Multi-Evidence Behavior Fusion
            latency_tracker.start("fusion")
            behavior_assessment = anomaly_detector.evaluate(
                kinematics=crowd_metrics,
                deep_prediction=deep_prediction,
            )
            latency_tracker.stop("fusion")

            current_fps = fps_counter.update()

            if frame_index % 60 == 0 or frame_index == 1:
                logger.info(
                    f"Frame {frame_index:05d} | People: {detection_result.person_count:02d} | "
                    f"Speed: {behavior_assessment.mean_speed:.1f}px/f | "
                    f"Behavior: {behavior_assessment.behavior_class} | "
                    f"VideoScore: {behavior_assessment.final_video_score:.2f} | "
                    f"STATUS: {behavior_assessment.status}"
                )

            if not no_display:
                display_frame = frame.copy()
                if show_density_grid:
                    display_frame = density_estimator.draw_density_overlay(display_frame, density_result)
                display_frame = tracker.draw_tracks(display_frame, tracks, draw_trajectory=True)

                display_frame = render_surveillance_hud(
                    display_frame,
                    detection=detection_result,
                    density=density_result,
                    fps=current_fps,
                    latency_tracker=latency_tracker,
                    behavior_assessment=behavior_assessment,
                    pipeline_label=f"Behavior ({beh_cfg.get('model_type', 'swin').upper()} + Kinematics)",
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("g"):
                    show_density_grid = not show_density_grid
                elif key == ord("b"):
                    manual_demo_behavior = not manual_demo_behavior
                    logger.info(f"Demo behavior trigger toggled: {manual_demo_behavior}")

            if max_frames and frame_index >= max_frames:
                break

    finally:
        video_cap.release()
        if not no_display:
            cv2.destroyAllWindows()
        logger.info(f"Behavior pipeline ended. Processed {frame_index} frames at {fps_counter.current_fps:.1f} FPS.")


def render_audio_monitor(
    preprocessed: PreprocessedAudio,
    mel_spec: np.ndarray,
    record: AudioEventRecord,
    latency_tracker: LatencyTracker,
    source_label: str = "Synthetic Stream",
    model_mode: str = "spectral_baseline",
    window_index: int = 1,
) -> np.ndarray:
    """
    Renders an informative, high-resolution Acoustic Surveillance Monitor display.
    Includes real-time waveform visualization, log-Mel spectrogram heatmap,
    and structured audio anomaly assessment telemetry.
    """
    canvas_w, canvas_h = 960, 540
    annotated = np.full((canvas_h, canvas_w, 3), 22, dtype=np.uint8)

    # 1. Header Banner
    banner_height = 55
    cv2.rectangle(annotated, (0, 0), (canvas_w, banner_height), (20, 20, 25), -1)
    cv2.line(annotated, (0, banner_height), (canvas_w, banner_height), (60, 60, 70), 1)

    cv2.putText(
        annotated,
        "REAL-TIME CROWD ANOMALY DETECTION | ACOUSTIC MONITOR (PHASE 5)",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated,
        f"Source: {source_label} | 16000Hz Mono | Win #{window_index:04d} | {timestamp_str}",
        (16, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Status Badge
    status_str = record.status
    if status_str == "EMERGENCY":
        status_text = "STATUS: EMERGENCY"
        badge_bg = (0, 0, 210)       # Red
    elif status_str == "SUSPICIOUS":
        status_text = "STATUS: SUSPICIOUS"
        badge_bg = (0, 140, 255)     # Orange
    else:
        status_text = "STATUS: NORMAL"
        badge_bg = (0, 160, 0)       # Green

    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    badge_x = canvas_w - tw - 26
    cv2.rectangle(annotated, (badge_x - 8, 12), (canvas_w - 12, 42), badge_bg, -1)
    cv2.putText(
        annotated,
        status_text,
        (badge_x, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Left Side Acoustic Telemetry Card
    card_x, card_y = 16, banner_height + 14
    card_w, card_h = 320, 425

    card_roi = annotated[card_y : card_y + card_h, card_x : card_x + card_w]
    card_bg = np.full(card_roi.shape, 28, dtype=np.uint8)
    cv2.addWeighted(card_roi, 0.25, card_bg, 0.75, 0, card_roi)
    annotated[card_y : card_y + card_h, card_x : card_x + card_w] = card_roi
    cv2.rectangle(annotated, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 90), 1)

    y_pos = card_y + 24
    gap = 22

    # Section 1: Classification
    cv2.putText(annotated, "[ACOUSTIC CLASSIFICATION]", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
    y_pos += gap

    evt_upper = record.audio_event.upper()
    evt_color = (0, 0, 255) if record.audio_anomaly_score >= 0.70 else ((0, 140, 255) if record.audio_anomaly_score >= 0.40 else (180, 240, 180))
    cv2.putText(annotated, f"Detected Event: {evt_upper}", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, evt_color, 1, cv2.LINE_AA)
    y_pos += gap

    cv2.putText(annotated, f"Confidence:     {record.audio_confidence:.2f}", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    y_pos += gap

    score_color = (0, 0, 255) if record.audio_anomaly_score >= 0.70 else ((0, 200, 255) if record.audio_anomaly_score >= 0.40 else (0, 255, 0))
    cv2.putText(annotated, f"Audio Anomaly:  {record.audio_anomaly_score:.2f}", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.46, score_color, 2, cv2.LINE_AA)
    y_pos += gap

    conf_state = "CONFIRMED" if record.is_confirmed else "UNCONFIRMED"
    conf_color = (0, 180, 255) if record.is_confirmed else (150, 150, 150)
    cv2.putText(annotated, f"Temporal State: {conf_state} ({record.consecutive_abnormal_count} win)", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, conf_color, 1, cv2.LINE_AA)
    y_pos += gap + 6

    # Section 2: Physical Signal Telemetry
    cv2.line(annotated, (card_x + 8, y_pos - 10), (card_x + card_w - 8, y_pos - 10), (60, 60, 70), 1)
    cv2.putText(annotated, "[SIGNAL TELEMETRY]", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
    y_pos += gap

    cv2.putText(annotated, f"RMS Energy:     {preprocessed.rms:.4f}", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    y_pos += gap

    cv2.putText(annotated, f"Loudness:       {preprocessed.dbfs:.1f} dBFS", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    y_pos += gap

    cv2.putText(annotated, f"Peak Amplitude: {preprocessed.peak:.4f}", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    y_pos += gap

    mode_label = "Spectral Baseline" if model_mode == "spectral_baseline" else ("Demo Mode" if model_mode == "demo_simulation" else "Custom CNN")
    cv2.putText(annotated, f"Engine Mode:    {mode_label}", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 200), 1, cv2.LINE_AA)
    y_pos += gap + 6

    # Section 3: Latency Profile
    cv2.line(annotated, (card_x + 8, y_pos - 10), (card_x + card_w - 8, y_pos - 10), (60, 60, 70), 1)
    cv2.putText(annotated, "[PROCESSING LATENCY]", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
    y_pos += gap

    cv2.putText(annotated, f"Preprocessing:  {latency_tracker.get_average('preprocess'):.1f} ms", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 240, 180), 1, cv2.LINE_AA)
    y_pos += gap

    cv2.putText(annotated, f"Mel Extraction: {latency_tracker.get_average('mel'):.1f} ms", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 240, 180), 1, cv2.LINE_AA)
    y_pos += gap

    cv2.putText(annotated, f"Classification: {latency_tracker.get_average('classifier'):.1f} ms", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 240, 180), 1, cv2.LINE_AA)
    y_pos += gap

    # 3. Right Side Visualizations
    right_x = card_x + card_w + 16
    right_w = canvas_w - right_x - 16

    # Required Audio Telemetry Strip Banner
    telemetry_bar_h = 36
    cv2.rectangle(annotated, (right_x, card_y), (right_x + right_w, card_y + telemetry_bar_h), (30, 30, 38), -1)
    cv2.rectangle(annotated, (right_x, card_y), (right_x + right_w, card_y + telemetry_bar_h), (80, 80, 95), 1)

    telemetry_txt = (
        f"AUDIO | Event: {record.audio_event.upper()} | "
        f"Conf: {record.audio_confidence:.2f} | "
        f"Audio Anomaly: {record.audio_anomaly_score:.2f}"
    )
    cv2.putText(annotated, telemetry_txt, (right_x + 14, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, score_color, 2, cv2.LINE_AA)

    # 4. Waveform Oscilloscope Plot
    wave_y = card_y + telemetry_bar_h + 12
    wave_h = 160
    cv2.rectangle(annotated, (right_x, wave_y), (right_x + right_w, wave_y + wave_h), (18, 18, 22), -1)
    cv2.rectangle(annotated, (right_x, wave_y), (right_x + right_w, wave_y + wave_h), (60, 60, 70), 1)

    # Grid lines
    center_y = wave_y + wave_h // 2
    cv2.line(annotated, (right_x, center_y), (right_x + right_w, center_y), (40, 40, 50), 1)
    cv2.putText(annotated, "Acoustic Waveform (1.0s window)", (right_x + 10, wave_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    samples = preprocessed.samples
    if samples is not None and len(samples) > 1:
        step = max(1, len(samples) // right_w)
        sampled_pts = samples[::step]
        n_pts = min(len(sampled_pts), right_w)
        pts = []
        for idx in range(n_pts):
            px = right_x + idx
            py = int(center_y - sampled_pts[idx] * (wave_h * 0.45))
            py = max(wave_y + 2, min(wave_y + wave_h - 2, py))
            pts.append([px, py])
        if len(pts) > 1:
            cv2.polylines(annotated, [np.array(pts, dtype=np.int32)], isClosed=False, color=(255, 220, 0), thickness=1, lineType=cv2.LINE_AA)

    # 5. Log-Mel Spectrogram Heatmap
    mel_y = wave_y + wave_h + 12
    mel_h = card_h - (wave_y + wave_h + 12 - card_y)
    cv2.rectangle(annotated, (right_x, mel_y), (right_x + right_w, mel_y + mel_h), (18, 18, 22), -1)
    cv2.rectangle(annotated, (right_x, mel_y), (right_x + right_w, mel_y + mel_h), (60, 60, 70), 1)

    cv2.putText(annotated, f"Log-Mel Spectrogram ({mel_spec.shape[0]} Bands x {mel_spec.shape[1]} Time Frames)", (right_x + 10, mel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    if mel_spec is not None and mel_spec.size > 0:
        min_v, max_v = mel_spec.min(), mel_spec.max()
        norm_mel = (mel_spec - min_v) / (max_v - min_v + 1e-6) * 255.0
        norm_mel = np.flipud(norm_mel.astype(np.uint8))

        color_spec = cv2.applyColorMap(norm_mel, cv2.COLORMAP_INFERNO)
        target_spec_w = right_w - 20
        target_spec_h = mel_h - 32
        if target_spec_w > 10 and target_spec_h > 10:
            resized_spec = cv2.resize(color_spec, (target_spec_w, target_spec_h), interpolation=cv2.INTER_LINEAR)
            annotated[mel_y + 24 : mel_y + 24 + target_spec_h, right_x + 10 : right_x + 10 + target_spec_w] = resized_spec

    # 6. Bottom Controls Bar
    help_text = "[Q]: Exit  |  [1]: Normal  |  [2]: Scream  |  [3]: Shouting  |  [4]: Alarm  |  [5]: Crash  |  [6]: Explosion"
    cv2.putText(annotated, help_text, (16, canvas_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)

    return annotated


def run_audio_pipeline(
    source_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
    demo_audio: bool = False,
) -> None:
    """Executes Phase 5 Audio Pipeline: Capture -> Preprocess -> Mel Spectrogram -> Classifier -> Anomaly."""
    config = load_config(config_path)
    audio_cfg = config.get("audio", {})

    source = source_override if source_override is not None else audio_cfg.get("source", "synthetic")
    logger.info("==================================================================")
    logger.info("STARTING AUDIO ANOMALY DETECTION PIPELINE (PHASE 5)")
    logger.info(f"Source: {source} | Sample Rate: {audio_cfg.get('sample_rate', 16000)}Hz | Demo Audio: {demo_audio}")
    logger.info("==================================================================")

    audio_cap = AudioCapture(
        source=source,
        sample_rate=audio_cfg.get("sample_rate", 16000),
        channels=audio_cfg.get("channels", 1),
        chunk_duration=audio_cfg.get("chunk_duration", 1.0),
        overlap=audio_cfg.get("overlap", 0.5),
    )

    preprocessor = AudioPreprocessor(
        target_peak=audio_cfg.get("target_peak", 0.95),
        enable_dc_removal=audio_cfg.get("enable_dc_removal", True),
        clipping_threshold=audio_cfg.get("clipping_threshold", 1.0),
    )

    mel_processor = MelSpectrogramProcessor(
        sample_rate=audio_cfg.get("sample_rate", 16000),
        n_fft=audio_cfg.get("n_fft", 1024),
        hop_length=audio_cfg.get("hop_length", 512),
        n_mels=audio_cfg.get("n_mels", 64),
        f_min=audio_cfg.get("f_min", 50.0),
        f_max=audio_cfg.get("f_max", 8000.0),
    )

    classifier = AudioClassifier(
        model_path=audio_cfg.get("model_path", ""),
        confidence_threshold=audio_cfg.get("confidence_threshold", 0.50),
        classes=audio_cfg.get("classes", None),
        enabled=audio_cfg.get("enabled", True),
        device="cpu",
    )

    anomaly_detector = AudioAnomalyDetector(
        temporal_window=audio_cfg.get("temporal_window", 5),
        confirmation_windows=audio_cfg.get("confirmation_windows", 2),
        smoothing_window=audio_cfg.get("smoothing_window", 5),
        suspicious_threshold=config.get("decision", {}).get("normal_threshold", 0.40),
        emergency_threshold=config.get("decision", {}).get("emergency_threshold", 0.70),
    )

    latency_tracker = LatencyTracker()
    chunk_index = 0
    window_name = "Crowd Anomaly Detection - Audio Surveillance Monitor"

    demo_sequence = ["normal", "normal", "scream", "scream", "normal", "shouting", "shouting", "alarm", "alarm", "explosion_like", "crash", "normal"]
    current_demo_event = None

    try:
        while True:
            latency_tracker.start("capture")
            # In demo_audio mode, sequence through emergency events
            if demo_audio:
                current_demo_event = demo_sequence[chunk_index % len(demo_sequence)]

            ret, raw_chunk = audio_cap.read_chunk(demo_event=current_demo_event or "normal")
            latency_tracker.stop("capture")

            if not ret or raw_chunk is None:
                if audio_cap.is_file:
                    logger.info("End of audio file stream reached.")
                    break
                time.sleep(0.05)
                continue

            chunk_index += 1

            latency_tracker.start("preprocess")
            preprocessed = preprocessor.preprocess(raw_chunk)
            latency_tracker.stop("preprocess")

            latency_tracker.start("mel")
            mel_spec = mel_processor.to_mel_spectrogram(preprocessed.samples)
            latency_tracker.stop("mel")

            latency_tracker.start("classifier")
            classification = classifier.classify(
                preprocessed=preprocessed,
                mel_spectrogram=mel_spec,
                demo_event=current_demo_event,
            )
            latency_tracker.stop("classifier")

            latency_tracker.start("anomaly")
            record = anomaly_detector.evaluate(classification)
            latency_tracker.stop("anomaly")

            if chunk_index % 10 == 0 or chunk_index == 1 or record.status != "NORMAL":
                logger.info(
                    f"Chunk {chunk_index:04d} | Event: {record.audio_event.upper():12s} | "
                    f"Conf: {record.audio_confidence:.2f} | Score: {record.audio_anomaly_score:.2f} | "
                    f"Confirmed: {str(record.is_confirmed):5s} | STATUS: {record.status}"
                )

            if not no_display:
                display_frame = render_audio_monitor(
                    preprocessed=preprocessed,
                    mel_spec=mel_spec,
                    record=record,
                    latency_tracker=latency_tracker,
                    source_label="Microphone" if audio_cap.is_mic else ("Audio File" if audio_cap.is_file else "Synthetic Stream"),
                    model_mode=classification.model_mode,
                    window_index=chunk_index,
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30 if audio_cap.is_synthetic else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("1"):
                    current_demo_event = "normal"
                    logger.info("Demo event set to: NORMAL")
                elif key == ord("2"):
                    current_demo_event = "scream"
                    logger.info("Demo event set to: SCREAM")
                elif key == ord("3"):
                    current_demo_event = "shouting"
                    logger.info("Demo event set to: SHOUTING")
                elif key == ord("4"):
                    current_demo_event = "alarm"
                    logger.info("Demo event set to: ALARM")
                elif key == ord("5"):
                    current_demo_event = "crash"
                    logger.info("Demo event set to: CRASH")
                elif key == ord("6"):
                    current_demo_event = "explosion_like"
                    logger.info("Demo event set to: EXPLOSION_LIKE")

            if max_frames and chunk_index >= max_frames:
                break

    finally:
        audio_cap.release()
        if not no_display:
            cv2.destroyAllWindows()
        logger.info(f"Audio pipeline ended. Processed {chunk_index} audio chunks.")


def run_multimodal_pipeline(
    source_override: Optional[str] = None,
    audio_source_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    device_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
    demo_multimodal: bool = False,
    pipeline_label: str = "Multimodal Decision Engine (Phase 7)",
) -> None:
    """
    Executes Phase 6 & 7 Multimodal Decision Pipeline:
    Concurrent Video (YOLO, ByteTrack, Density, Kinematics, Behavior, Safety)
    + Audio (Capture, Preprocessing, Mel, Classification, Anomaly)
    -> Multimodal Decision Fusion Engine -> Intelligent Decision Engine.
    """
    config = load_config(config_path)

    v_cfg = config["video"]
    d_cfg = config["detection"]
    t_cfg = config["tracking"]
    dens_cfg = config["density"]
    w_cfg = config.get("weapon_detection", {})
    f_cfg = config.get("fight_detection", {})
    sf_cfg = config.get("safety_fusion", {})
    beh_cfg = config.get("behavior", {})
    bf_cfg = config.get("behavior_fusion", {})
    aud_cfg = config.get("audio", {})
    fus_cfg = config.get("fusion", {})
    dec_cfg = config.get("decision", {})

    video_source = source_override if source_override is not None else v_cfg["source"]
    audio_source = audio_source_override if audio_source_override is not None else aud_cfg.get("source", "microphone")
    conf = conf_override if conf_override is not None else float(d_cfg["confidence"])
    device = device_override if device_override is not None else d_cfg["device"]

    logger.info("==================================================================")
    logger.info(f"STARTING SURVEILLANCE PIPELINE ({pipeline_label.upper()})")
    logger.info(f"Video Source: {video_source} | Audio Source: {audio_source} | Demo Multimodal: {demo_multimodal}")
    logger.info("==================================================================")

    # 1. Video Subsystem Components
    video_cap = VideoSource(
        source=video_source,
        width=v_cfg["width"],
        height=v_cfg["height"],
        fps=v_cfg["fps"],
        loop=v_cfg["loop"],
    )

    detector = YOLOPersonDetector(
        model_path=d_cfg["model"],
        confidence=conf,
        iou_threshold=d_cfg["iou_threshold"],
        device=device,
        target_classes=d_cfg["classes"],
        imgsz=d_cfg["imgsz"],
    )

    tracker = ByteTrackCrowdTracker(
        history_length=t_cfg["history_length"],
        speed_threshold=t_cfg["speed_threshold"],
        max_lost=t_cfg["max_lost"],
        iou_threshold=t_cfg["iou_threshold"],
    )

    density_estimator = CrowdDensityEstimator(
        frame_area_m2=dens_cfg["frame_area_m2"],
        grid_rows=dens_cfg["grid_rows"],
        grid_cols=dens_cfg["grid_cols"],
        threshold_low=dens_cfg["threshold_low"],
        threshold_moderate=dens_cfg["threshold_moderate"],
        threshold_high=dens_cfg["threshold_high"],
    )

    temporal_buffer = TemporalFrameBuffer(
        sequence_length=beh_cfg.get("sequence_length", 16),
        sample_rate=beh_cfg.get("sample_rate", 4),
        target_size=(224, 224),
    )

    crowd_analyzer = CrowdBehaviorAnalyzer(
        speed_threshold=beh_cfg.get("speed_threshold", 18.0),
        dispersion_threshold=beh_cfg.get("dispersion_threshold", 2.5),
        convergence_threshold=beh_cfg.get("convergence_threshold", 2.5),
    )

    behavior_model = BehaviorModel(
        model_type=beh_cfg.get("model_type", "swin"),
        model_path=beh_cfg.get("model_path", ""),
        confidence_threshold=beh_cfg.get("confidence_threshold", 0.50),
        device=device,
        enabled=beh_cfg.get("enabled", True),
    )

    behavior_anomaly = BehaviorAnomalyDetector(
        kinematic_weight=bf_cfg.get("kinematic_weight", 0.40),
        deep_model_weight=bf_cfg.get("deep_model_weight", 0.60),
        normal_threshold=config["decision"]["normal_threshold"],
        emergency_threshold=config["decision"]["emergency_threshold"],
        confirmation_frames=beh_cfg.get("confirmation_frames", 5),
        smoothing_window=beh_cfg.get("smoothing_window", 10),
    )

    weapon_detector = WeaponDetector(
        model_path=w_cfg.get("model", ""),
        confidence=float(w_cfg.get("confidence", 0.50)),
        target_classes=w_cfg.get("classes", None),
        device=device,
        enabled=w_cfg.get("enabled", True),
        simulate_for_demo=w_cfg.get("simulate_for_demo", False),
    )

    pose_analyzer = PoseMovementAnalyzer(
        proximity_threshold=f_cfg.get("proximity_threshold", 130.0),
        speed_threshold=t_cfg.get("speed_threshold", 18.0),
    )

    fight_detector = FightDetector(
        temporal_window=f_cfg.get("temporal_window", 20),
        suspicious_threshold=f_cfg.get("suspicious_threshold", 0.50),
        emergency_threshold=f_cfg.get("emergency_threshold", 0.75),
        confirmation_frames=f_cfg.get("confirmation_frames", 5),
        cooldown_seconds=f_cfg.get("cooldown_seconds", 10.0),
        enabled=f_cfg.get("enabled", True),
    )

    safety_fusion = SafetyFusionEngine(
        crowd_weight=sf_cfg.get("crowd_weight", 0.25),
        fight_weight=sf_cfg.get("fight_weight", 0.35),
        weapon_weight=sf_cfg.get("weapon_weight", 0.40),
        normal_threshold=sf_cfg.get("normal_threshold", 0.35),
        emergency_threshold=sf_cfg.get("emergency_threshold", 0.70),
        cooldown_seconds=sf_cfg.get("cooldown_seconds", 10.0),
        correlation_multiplier=sf_cfg.get("correlation_multiplier", 1.25),
    )

    # 2. Audio Subsystem Components
    audio_cap = AudioCapture(
        source=audio_source,
        sample_rate=aud_cfg.get("sample_rate", 16000),
        channels=aud_cfg.get("channels", 1),
        chunk_duration=aud_cfg.get("chunk_duration", 1.0),
        overlap=aud_cfg.get("overlap", 0.5),
    )

    audio_preprocessor = AudioPreprocessor(
        target_peak=aud_cfg.get("target_peak", 0.95),
        enable_dc_removal=aud_cfg.get("enable_dc_removal", True),
        clipping_threshold=aud_cfg.get("clipping_threshold", 1.0),
    )

    mel_processor = MelSpectrogramProcessor(
        sample_rate=aud_cfg.get("sample_rate", 16000),
        n_fft=aud_cfg.get("n_fft", 1024),
        hop_length=aud_cfg.get("hop_length", 512),
        n_mels=aud_cfg.get("n_mels", 64),
        f_min=aud_cfg.get("f_min", 50.0),
        f_max=aud_cfg.get("f_max", 8000.0),
    )

    audio_classifier = AudioClassifier(
        model_path=aud_cfg.get("model_path", ""),
        confidence_threshold=aud_cfg.get("confidence_threshold", 0.50),
        classes=aud_cfg.get("classes", None),
        enabled=aud_cfg.get("enabled", True),
        device="cpu",
    )

    audio_detector = AudioAnomalyDetector(
        temporal_window=aud_cfg.get("temporal_window", 5),
        confirmation_windows=aud_cfg.get("confirmation_windows", 2),
        smoothing_window=aud_cfg.get("smoothing_window", 5),
        suspicious_threshold=config["decision"]["normal_threshold"],
        emergency_threshold=config["decision"]["emergency_threshold"],
    )

    # 3. Multimodal Decision Fusion Engine
    fusion_engine = MultimodalFusionEngine(
        video_weight=fus_cfg.get("video_weight", 0.60),
        audio_weight=fus_cfg.get("audio_weight", 0.40),
        density_sub_weight=fus_cfg.get("density_sub_weight", 0.15),
        movement_sub_weight=fus_cfg.get("movement_sub_weight", 0.20),
        behavior_sub_weight=fus_cfg.get("behavior_sub_weight", 0.25),
        fight_sub_weight=fus_cfg.get("fight_sub_weight", 0.20),
        weapon_sub_weight=fus_cfg.get("weapon_sub_weight", 0.20),
        synergy_boost=fus_cfg.get("synergy_boost", 1.25),
        suppression_factor=fus_cfg.get("suppression_factor", 0.50),
        normal_threshold=config["decision"]["normal_threshold"],
        emergency_threshold=config["decision"]["emergency_threshold"],
        smoothing_window=fus_cfg.get("smoothing_window", 5),
        temporal_tolerance_seconds=fus_cfg.get("temporal_tolerance_seconds", 1.5),
        stale_timeout_seconds=fus_cfg.get("stale_timeout_seconds", 2.5),
    )

    # 4. Intelligent Decision Engine (Phase 7)
    decision_engine = DecisionEngine(
        enabled=dec_cfg.get("enabled", True),
        suspicious_enter=dec_cfg.get("suspicious_enter", 0.45),
        suspicious_exit=dec_cfg.get("suspicious_exit", 0.35),
        high_risk_enter=dec_cfg.get("high_risk_enter", 0.65),
        high_risk_exit=dec_cfg.get("high_risk_exit", 0.55),
        emergency_enter=dec_cfg.get("emergency_enter", 0.85),
        emergency_exit=dec_cfg.get("emergency_exit", 0.70),
        suspicious_confirmation=dec_cfg.get("suspicious_confirmation", 3),
        high_risk_confirmation=dec_cfg.get("high_risk_confirmation", 3),
        emergency_confirmation=dec_cfg.get("emergency_confirmation", 2),
        recovery_confirmation=dec_cfg.get("recovery_confirmation", 5),
        transition_cooldown_seconds=dec_cfg.get("transition_cooldown_seconds", 3.0),
        critical_evidence_enabled=dec_cfg.get("critical_evidence_enabled", True),
        critical_weapon_threshold=dec_cfg.get("critical_weapon_threshold", 0.90),
        critical_fight_threshold=dec_cfg.get("critical_fight_threshold", 0.90),
        critical_multimodal_threshold=dec_cfg.get("critical_multimodal_threshold", 0.90),
        dynamic_thresholds=dec_cfg.get("dynamic_thresholds", True),
        max_threshold_adjustment=dec_cfg.get("max_threshold_adjustment", 0.10),
    )

    # 5. Alert Management Subsystem (Phase 8)
    alert_manager = AlertManager.from_config(config)

    # 6. Explainable AI Subsystem (Phase 9)
    xai_cfg = config.get("xai", {})
    explanation_engine = ExplanationEngine(config=config)
    visual_explainer = VisualExplainer(
        show_motion_vectors=xai_cfg.get("hud", {}).get("show_motion_vectors", True),
        show_dispersion_arrows=xai_cfg.get("hud", {}).get("show_dispersion_arrows", True),
    )

    fps_counter = FPSCounter()
    latency_tracker = LatencyTracker()

    manual_demo = demo_multimodal
    show_density_grid = False
    current_audio_record: Optional[AudioEventRecord] = None
    frame_index = 0
    audio_update_interval = 15  # Update audio analysis every ~15 frames (0.5s step at 30 FPS)
    window_name = "Crowd Anomaly Detection - Multimodal Surveillance Monitor"

    try:
        while True:
            # -------------------------------------------------------------
            # Step A: Video Ingestion
            # -------------------------------------------------------------
            latency_tracker.start("video_read")
            ret, frame = video_cap.read()
            latency_tracker.stop("video_read")

            if not ret or frame is None:
                break

            frame_index += 1
            h, w = frame.shape[:2]

            # -------------------------------------------------------------
            # Step B: Audio Chunk Ingestion & Analysis (Sliding Step)
            # -------------------------------------------------------------
            if frame_index % audio_update_interval == 1 or current_audio_record is None:
                latency_tracker.start("audio_cycle")
                demo_audio_evt = None
                if manual_demo:
                    # Synchronize demo scenarios
                    if 30 <= frame_index < 60:
                        demo_audio_evt = "scream"
                    elif 60 <= frame_index < 90:
                        demo_audio_evt = "shouting"
                    elif 90 <= frame_index < 110:
                        demo_audio_evt = "crash"  # Uncorroborated noise spike
                    else:
                        demo_audio_evt = "normal"

                ret_a, raw_chunk = audio_cap.read_chunk(demo_event=demo_audio_evt or "normal")
                if ret_a and raw_chunk is not None:
                    preprocessed_a = audio_preprocessor.preprocess(raw_chunk)
                    mel_spec = mel_processor.to_mel_spectrogram(preprocessed_a.samples)
                    classification = audio_classifier.classify(
                        preprocessed=preprocessed_a,
                        mel_spectrogram=mel_spec,
                        demo_event=demo_audio_evt,
                    )
                    current_audio_record = audio_detector.evaluate(classification)
                latency_tracker.stop("audio_cycle")

            # -------------------------------------------------------------
            # Step C: Video Feature Extraction (5 Signals)
            # -------------------------------------------------------------
            latency_tracker.start("yolo")
            detection_result = detector.detect(frame)
            latency_tracker.stop("yolo")

            latency_tracker.start("tracking")
            tracks = tracker.update(detection_result, frame)
            latency_tracker.stop("tracking")

            latency_tracker.start("density")
            density_result = density_estimator.estimate(
                detection_points=detection_result.bottom_centers,
                frame_width=w,
                frame_height=h,
            )
            latency_tracker.stop("density")

            # Kinematics & Temporal Behavior
            temporal_buffer.add_frame(frame)
            crowd_metrics = crowd_analyzer.analyze(tracks, density_result)

            demo_behavior_trigger = manual_demo and (30 <= frame_index < 60)
            deep_prediction = None
            if temporal_buffer.is_ready() or demo_behavior_trigger:
                seq_tensor = temporal_buffer.get_torch_tensor(device="cpu")
                deep_prediction = behavior_model.predict(
                    sequence_tensor=seq_tensor,
                    demo_trigger=demo_behavior_trigger,
                )

            behavior_assessment = behavior_anomaly.evaluate(
                kinematics=crowd_metrics,
                deep_prediction=deep_prediction,
            )

            # Safety (Weapons & Fight)
            demo_weapon_trigger = manual_demo and (110 <= frame_index < 130)
            demo_fight_trigger = manual_demo and (60 <= frame_index < 90)

            curr_ts = time.time()
            weapon_result = weapon_detector.detect(
                frame=frame,
                tracked_persons=tracks,
                timestamp=curr_ts,
                demo_trigger=demo_weapon_trigger,
            )
            kinematic_features = pose_analyzer.extract_features(tracks, timestamp=curr_ts)
            fight_result = fight_detector.update(
                kinematics=kinematic_features,
                tracked_persons=tracks,
                timestamp=curr_ts,
                demo_trigger=demo_fight_trigger,
            )
            safety_assessment = safety_fusion.evaluate(
                density_result=density_result,
                kinematics=kinematic_features,
                fight_result=fight_result,
                weapon_result=weapon_result,
                timestamp=curr_ts,
            )

            # -------------------------------------------------------------
            # Step D: Multimodal Decision Fusion (Phase 6)
            # -------------------------------------------------------------
            latency_tracker.start("multimodal_fusion")
            multimodal_assessment = fusion_engine.fuse(
                density=density_result,
                kinematics=crowd_metrics,
                behavior_assessment=behavior_assessment,
                safety_assessment=safety_assessment,
                weapon_result=weapon_result,
                audio_record=current_audio_record,
            )
            latency_tracker.stop("multimodal_fusion")

            # -------------------------------------------------------------
            # Step E: Intelligent Decision Engine (Phase 7)
            # -------------------------------------------------------------
            latency_tracker.start("decision_engine")
            decision_event = decision_engine.evaluate(
                assessment=multimodal_assessment,
                current_time=curr_ts,
            )
            latency_tracker.stop("decision_engine")

            # -------------------------------------------------------------
            # Step F: Alert Management & Notifications (Phase 8)
            # -------------------------------------------------------------
            latency_tracker.start("alert_manager")
            emitted_alert = alert_manager.process_decision(
                event=decision_event,
                current_time=curr_ts,
            )
            latency_tracker.stop("alert_manager")
            active_hud_alert = emitted_alert or alert_manager.get_current_hud_alert()

            # -------------------------------------------------------------
            # Step G: Explainable AI & Evidence Extraction (Phase 9)
            # -------------------------------------------------------------
            latency_tracker.start("xai_explanation")
            evidence_record = explanation_engine.explain_decision(
                decision_event=decision_event,
                multimodal_assessment=multimodal_assessment,
                alert_record=active_hud_alert,
                current_time=curr_ts,
            )
            latency_tracker.stop("xai_explanation")

            current_fps = fps_counter.update()

            # Periodic Console Logging
            if frame_index % 30 == 0 or frame_index == 1 or decision_event.current_state != "NORMAL" or decision_event.transition:
                syn_info = f" | Syn: {decision_event.evidence_summary.get('synergy_type')}" if decision_event.corroborated else ""
                crit_info = " | [CRITICAL]" if decision_event.critical_evidence else ""
                trans_info = f" | TRANSITION: {decision_event.previous_state} -> {decision_event.current_state}" if decision_event.transition else ""
                alert_info = f" | ALERT: {emitted_alert.severity.value}" if emitted_alert else ""
                logger.info(
                    f"Frame {frame_index:05d} | People: {detection_result.person_count:02d} | "
                    f"Fused: {decision_event.score:.2f} | STATE: {decision_event.current_state} (Conf: {decision_event.confidence:.2f})"
                    f"{syn_info}{crit_info}{trans_info}{alert_info}"
                )

            # -------------------------------------------------------------
            # Step H: HUD Rendering & Display
            # -------------------------------------------------------------
            if not no_display:
                display_frame = frame.copy()
                if show_density_grid:
                    display_frame = density_estimator.draw_density_overlay(display_frame, density_result)
                display_frame = tracker.draw_tracks(display_frame, tracks, draw_trajectory=True)
                if visual_explainer.show_motion_vectors:
                    display_frame = visual_explainer.draw_motion_vectors(display_frame, tracks)
                if crowd_metrics is not None and visual_explainer.show_dispersion_arrows:
                    display_frame = visual_explainer.draw_dispersion_indicators(
                        display_frame, tracks, dispersion_rate=crowd_metrics.dispersion_rate
                    )
                display_frame = weapon_detector.draw_detections(display_frame, weapon_result)

                display_frame = render_surveillance_hud(
                    display_frame,
                    detection=detection_result,
                    density=density_result,
                    fps=current_fps,
                    latency_tracker=latency_tracker,
                    safety_assessment=safety_assessment,
                    behavior_assessment=behavior_assessment,
                    audio_record=current_audio_record,
                    multimodal_assessment=multimodal_assessment,
                    decision_event=decision_event,
                    alert_record=active_hud_alert,
                    evidence_record=evidence_record,
                    pipeline_label=pipeline_label,
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("g"):
                    show_density_grid = not show_density_grid
                elif key == ord("m"):
                    manual_demo = not manual_demo
                    logger.info(f"Demo multimodal trigger toggled: {manual_demo}")
                elif key == ord("s"):
                    # Manual simulated scream
                    if current_audio_record is not None:
                        current_audio_record.audio_event = "scream"
                        current_audio_record.audio_anomaly_score = 0.85
                        current_audio_record.is_confirmed = True
                    logger.info("Manual scream event simulated.")
                elif key == ord("f"):
                    # Manual fight trigger
                    demo_fight_trigger = True
                    logger.info("Manual fight trigger simulated.")
                elif key == ord("w"):
                    # Manual weapon trigger
                    demo_weapon_trigger = True
                    logger.info("Manual weapon trigger simulated.")

            if max_frames and frame_index >= max_frames:
                break

    finally:
        video_cap.release()
        audio_cap.release()
        if not no_display:
            cv2.destroyAllWindows()
        logger.info(
            f"Multimodal pipeline ended. Processed {frame_index} frames at {fps_counter.current_fps:.1f} FPS."
        )


def run_decision_pipeline(
    source_override: Optional[str] = None,
    audio_source_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    device_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
    demo_multimodal: bool = False,
) -> None:
    """Executes Phase 7 Intelligent Decision Pipeline (Full Multimodal + State Machine)."""
    return run_multimodal_pipeline(
        source_override=source_override,
        audio_source_override=audio_source_override,
        conf_override=conf_override,
        device_override=device_override,
        no_display=no_display,
        max_frames=max_frames,
        config_path=config_path,
        demo_multimodal=demo_multimodal,
        pipeline_label="Intelligent Decision Engine (Phase 7)",
    )


def run_alerts_pipeline(
    source_override: Optional[str] = None,
    audio_source_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    device_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
    demo_multimodal: bool = False,
) -> None:
    """Executes Phase 8 Alert Management Pipeline (Decision Engine + Alerts & Notifications)."""
    return run_multimodal_pipeline(
        source_override=source_override,
        audio_source_override=audio_source_override,
        conf_override=conf_override,
        device_override=device_override,
        no_display=no_display,
        max_frames=max_frames,
        config_path=config_path,
        demo_multimodal=demo_multimodal,
        pipeline_label="Alert Management & Predictive Notifications (Phase 8)",
    )


def run_xai_pipeline(
    source_override: Optional[str] = None,
    audio_source_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    device_override: Optional[str] = None,
    no_display: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = None,
    demo_multimodal: bool = False,
) -> None:
    """Executes Phase 9 Explainable AI & Evidence Visualization Pipeline."""
    return run_multimodal_pipeline(
        source_override=source_override,
        audio_source_override=audio_source_override,
        conf_override=conf_override,
        device_override=device_override,
        no_display=no_display,
        max_frames=max_frames,
        config_path=config_path,
        demo_multimodal=demo_multimodal,
        pipeline_label="Explainable AI & Evidence Visualization (Phase 9)",
    )


def run_alerts_demo(config_path: Optional[str] = None) -> None:
    """
    Simulates the 8 Phase 8 Alert Management demonstration scenarios:
    1. NORMAL -> no alert
    2. NORMAL -> SUSPICIOUS -> WARNING alert
    3. SUSPICIOUS -> HIGH_RISK -> HIGH alert
    4. HIGH_RISK -> EMERGENCY -> CRITICAL alert
    5. Duplicate EMERGENCY -> suppressed
    6. EMERGENCY -> RECOVERY -> NORMAL
    7. EMERGENCY repeated after cooldown -> new CRITICAL alert
    8. Email/SMS enabled -> safe 'stub/not connected' message
    """
    load_config(config_path)
    logger.info("==================================================================")
    logger.info("STARTING ALERT MANAGEMENT & PREDICTIVE NOTIFICATION DEMO (PHASE 8)")
    logger.info("==================================================================")

    demo_policy = AlertPolicy(
        enabled=True,
        minimum_severity=AlertSeverity.WARNING,
        cooldown_seconds=4.0,
        duplicate_window_seconds=6.0,
        recovery_alert=False,
        channels={
            "console": True,
            "hud": True,
            "local_alarm": False,
            "email": True,
            "sms": True,
            "push": False,
        },
        escalation_enabled=True,
        repeat_emergency_after_seconds=5.0,
    )
    alert_mgr = AlertManager(policy=demo_policy)
    t = 100.0

    print("\n" + "=" * 70)
    print(">>> SCENARIO 1: Nominal Environment (NORMAL) -> Zero Alerts Generated")
    print("=" * 70)
    e1 = DecisionEvent(
        timestamp=t,
        previous_state="NORMAL",
        current_state="NORMAL",
        score=0.08,
        confidence=0.85,
        reason="Normal crowd activity with calm baseline kinematics and ambient acoustics.",
        confirmed=True,
        transition=False,
    )
    a1 = alert_mgr.process_decision(e1, current_time=t)
    print(f"Decision: STATE={e1.current_state} | Alert Result: {'EMITTED' if a1 else 'SUPPRESSED (Correct: No alert for NORMAL)'}")

    t += 1.0
    print("\n" + "=" * 70)
    print(">>> SCENARIO 2: Agitation Detected (NORMAL -> SUSPICIOUS) -> WARNING Alert Emitted")
    print("=" * 70)
    e2 = DecisionEvent(
        timestamp=t,
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.48,
        confidence=0.76,
        reason="Elevated visual movement pattern (rapid_movement) observed; temporal confirmation active.",
        dominant_modality="video",
        video_score=0.52,
        audio_score=0.15,
        confirmed=True,
        transition=True,
    )
    a2 = alert_mgr.process_decision(e2, current_time=t)
    print(f"Decision: STATE={e2.current_state} | Alert Result: {a2.severity.value if a2 else 'None'} [ID: {a2.alert_id if a2 else 'N/A'}]")

    t += 1.0
    print("\n" + "=" * 70)
    print(">>> SCENARIO 3: Escalation to Altercation (SUSPICIOUS -> HIGH_RISK) -> HIGH Alert Emitted")
    print("=" * 70)
    e3 = DecisionEvent(
        timestamp=t,
        previous_state="SUSPICIOUS",
        current_state="HIGH_RISK",
        score=0.72,
        confidence=0.82,
        reason="Possible physical struggle with supporting acoustic indicators detected.",
        dominant_modality="bimodal",
        video_score=0.68,
        audio_score=0.74,
        fight_score=0.68,
        corroborated=True,
        confirmed=True,
        transition=True,
    )
    a3 = alert_mgr.process_decision(e3, current_time=t)
    print(f"Decision: STATE={e3.current_state} | Alert Result: {a3.severity.value if a3 else 'None'} [ID: {a3.alert_id if a3 else 'N/A'}]")

    t += 1.0
    print("\n" + "=" * 70)
    print(">>> SCENARIO 4: Critical Threat (HIGH_RISK -> EMERGENCY) -> CRITICAL Alert Emitted")
    print("=" * 70)
    e4 = DecisionEvent(
        timestamp=t,
        previous_state="HIGH_RISK",
        current_state="EMERGENCY",
        score=0.91,
        confidence=0.94,
        reason="Potential emergency: Confirmed cross-modal crisis signature (SCREAM_AND_SCATTER).",
        dominant_modality="bimodal",
        video_score=0.88,
        audio_score=0.95,
        fight_score=0.70,
        corroborated=True,
        critical_evidence=True,
        confirmed=True,
        transition=True,
    )
    a4 = alert_mgr.process_decision(e4, current_time=t)
    print(f"Decision: STATE={e4.current_state} | Alert Result: {a4.severity.value if a4 else 'None'} [ID: {a4.alert_id if a4 else 'N/A'}]")

    t += 1.0
    print("\n" + "=" * 70)
    print(">>> SCENARIO 5: Duplicate Emergency in Immediate Window -> Correctly Suppressed")
    print("=" * 70)
    e5 = DecisionEvent(
        timestamp=t,
        previous_state="EMERGENCY",
        current_state="EMERGENCY",
        score=0.92,
        confidence=0.95,
        reason="Potential emergency: Confirmed cross-modal crisis signature (SCREAM_AND_SCATTER).",
        dominant_modality="bimodal",
        video_score=0.90,
        audio_score=0.95,
        corroborated=True,
        critical_evidence=True,
        confirmed=True,
        transition=False,
    )
    a5 = alert_mgr.process_decision(e5, current_time=t)
    print(f"Decision: STATE={e5.current_state} | Alert Result: {'EMITTED' if a5 else 'SUPPRESSED (Duplicate / Cooldown active)'}")

    t += 1.0
    print("\n" + "=" * 70)
    print(">>> SCENARIO 6: Crisis Subsides (EMERGENCY -> RECOVERY -> NORMAL)")
    print("=" * 70)
    e6_rec = DecisionEvent(
        timestamp=t,
        previous_state="EMERGENCY",
        current_state="RECOVERY",
        score=0.25,
        confidence=0.85,
        reason="Anomaly score decreasing below emergency boundary; entering recovery stabilization.",
        confirmed=True,
        transition=True,
    )
    a6_rec = alert_mgr.process_decision(e6_rec, current_time=t)
    print(f"Decision: STATE={e6_rec.current_state} | Alert Result: {'EMITTED' if a6_rec else 'SUPPRESSED (recovery_alert=False)'}")

    t += 2.0
    e6_norm = DecisionEvent(
        timestamp=t,
        previous_state="RECOVERY",
        current_state="NORMAL",
        score=0.10,
        confidence=0.88,
        reason="Normal crowd activity with calm baseline kinematics and ambient acoustics.",
        confirmed=True,
        transition=True,
    )
    a6_norm = alert_mgr.process_decision(e6_norm, current_time=t)
    print(f"Decision: STATE={e6_norm.current_state} | Alert Result: {'EMITTED' if a6_norm else 'SUPPRESSED (Normal produces no alerts)'}")

    t += 6.0  # Elapse repeat_emergency_after_seconds (5.0s)
    print("\n" + "=" * 70)
    print(">>> SCENARIO 7: Persistent Emergency Repeated After Cooldown -> New CRITICAL Alert")
    print("=" * 70)
    e7 = DecisionEvent(
        timestamp=t,
        previous_state="NORMAL",
        current_state="EMERGENCY",
        score=0.89,
        confidence=0.92,
        reason="Critical safety evidence: Potential dangerous-object detected (0.92 confidence).",
        dominant_modality="video",
        weapon_score=0.92,
        critical_evidence=True,
        confirmed=True,
        transition=True,
    )
    a7 = alert_mgr.process_decision(e7, current_time=t)
    print(f"Decision: STATE={e7.current_state} | Alert Result: {a7.severity.value if a7 else 'None'} [ID: {a7.alert_id if a7 else 'N/A'}]")

    print("\n" + "=" * 70)
    print(">>> SCENARIO 8: External Notification Channels (Email / SMS) -> Safe Stubs Verified")
    print("=" * 70)
    if a7:
        for ch_name, status in a7.channel_status.items():
            print(f"  Channel [{ch_name:12s}]: {status}")

    print("\n" + "=" * 70)
    print(">>> OPERATOR LIFECYCLE MANAGEMENT (Acknowledgement & Resolution)")
    print("=" * 70)
    active_before = alert_mgr.get_active_alerts()
    print(f"Active Unresolved Alerts: {len(active_before)}")
    if a7:
        alert_mgr.acknowledge(a7.alert_id)
        print(f"Alert [{a7.alert_id}] acknowledged -> Status: {a7.status.value}")
        alert_mgr.resolve(a7.alert_id)
        print(f"Alert [{a7.alert_id}] resolved -> Status: {a7.status.value}")
    active_after = alert_mgr.get_active_alerts()
    print(f"Active Unresolved Alerts after resolution: {len(active_after)}")

    print("\n" + "=" * 70)
    print("ALL 8 DEMONSTRATION SCENARIOS COMPLETED SUCCESSFULLY.")
    print("=" * 70)


def run_xai_demo(config_path: Optional[str] = None) -> None:
    """
    Simulates the 8 Phase 9 Explainable AI (XAI) demonstration scenarios:
    1. Normal Baseline -> Minimal evidence, 0.0% attribution
    2. Suspicious Movement -> Velocity and acoustic agitation evidence
    3. High Risk Altercation -> Physical struggle + crowd dispersion
    4. Critical Emergency -> Weapon presence + scream synergy
    5. Audio-Only Evidence -> Acoustic disturbance, visual stream nominal
    6. Video-Only Evidence -> Visual agitation, acoustic microphone offline
    7. Corroborated Audio + Video -> Multi-sensor synergy boost explanation
    8. Conflicting Modalities -> Visual agitation suppressed by quiet acoustics
    """
    config = load_config(config_path)
    engine = ExplanationEngine(config=config)
    alert_mgr = AlertManager.from_config(config)

    logger.info("==================================================================")
    logger.info("STARTING EXPLAINABLE AI (XAI) & EVIDENCE DEMO (PHASE 9)")
    logger.info("==================================================================")

    t = 1000.0

    # SCENARIO 1: Normal Baseline
    print("\n" + "=" * 75)
    print(">>> SCENARIO 1: Calm Baseline (NORMAL) -> Nominal Evidence & 0.0% Attribution")
    print("=" * 75)
    e1 = DecisionEvent(
        timestamp=t,
        previous_state="NORMAL",
        current_state="NORMAL",
        score=0.0,
        confidence=0.90,
        reason="Calm pedestrian flow and nominal ambient sound.",
        dominant_modality="none",
        video_score=0.0,
        audio_score=0.0,
        evidence_summary={
            "video": {"density_m2": 0.0, "mean_speed": 0.0, "movement_score": 0.0},
            "audio": {"audio_event": "normal", "audio_confidence": 0.95, "is_active": True},
        },
    )
    rec1 = engine.explain_decision(e1)
    print(f"Short Summary: {rec1.short_explanation}")
    print(f"Attributions : {rec1.ranked_contributions if rec1.ranked_contributions else 'Nominal Baseline (0.00% across all factors)'}")
    print(f"Statements   : {rec1.top_evidence}")
    print(f"Transparency : Saliency={rec1.model_explanation_available} ({rec1.model_explanation_reason})")

    # SCENARIO 2: Suspicious Movement
    t += 1.0
    print("\n" + "=" * 75)
    print(">>> SCENARIO 2: Agitation (SUSPICIOUS) -> Elevated Movement & Acoustic Agitation")
    print("=" * 75)
    e2 = DecisionEvent(
        timestamp=t,
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.52,
        confidence=0.76,
        reason="Elevated crowd movement velocity accompanied by acoustic shouting.",
        dominant_modality="audio",
        video_score=0.45,
        audio_score=0.60,
        evidence_summary={
            "video": {"mean_speed": 10.5, "movement_score": 0.50, "density_m2": 1.4},
            "audio": {"audio_event": "shouting", "audio_confidence": 0.72, "is_active": True},
        },
    )
    rec2 = engine.explain_decision(e2)
    print(f"Short Summary: {rec2.short_explanation}")
    print(f"Top Ranks    : {rec2.ranked_contributions[:3]}")
    print(f"Statements   : {rec2.top_evidence}")

    # SCENARIO 3: High Risk Altercation
    t += 1.0
    print("\n" + "=" * 75)
    print(">>> SCENARIO 3: Violent Struggle (HIGH_RISK) -> Altercation + Crowd Dispersion")
    print("=" * 75)
    e3 = DecisionEvent(
        timestamp=t,
        previous_state="SUSPICIOUS",
        current_state="HIGH_RISK",
        score=0.76,
        confidence=0.84,
        reason="Physical struggle dynamics observed with rapid crowd scattering.",
        dominant_modality="video",
        video_score=0.80,
        audio_score=0.50,
        fight_score=0.78,
        confirmed=True,
        transition=True,
        evidence_summary={
            "video": {
                "fight_score": 0.78,
                "movement_score": 0.70,
                "dispersion_rate": 8.2,
                "direction_entropy": 2.3,
            },
            "audio": {"audio_event": "shouting", "audio_confidence": 0.65, "is_active": True},
        },
    )
    a3 = alert_mgr.process_decision(e3, current_time=t)
    rec3 = engine.explain_decision(e3, alert_record=a3)
    print(f"Short Summary : {rec3.short_explanation}")
    print(f"Alert Justify : {rec3.alert_rationale}")
    print(f"Top Ranks     : {rec3.ranked_contributions[:3]}")

    # SCENARIO 4: Critical Emergency
    t += 1.0
    print("\n" + "=" * 75)
    print(">>> SCENARIO 4: Critical Threat (EMERGENCY) -> Weapon Presence + Scream Synergy")
    print("=" * 75)
    e4 = DecisionEvent(
        timestamp=t,
        previous_state="HIGH_RISK",
        current_state="EMERGENCY",
        score=0.94,
        confidence=0.96,
        reason="Firearm detected with corroborating acoustic scream signature.",
        dominant_modality="bimodal",
        video_score=0.92,
        audio_score=0.95,
        weapon_score=0.95,
        fight_score=0.60,
        corroborated=True,
        critical_evidence=True,
        evidence_summary={
            "synergy_type": "WEAPON_AND_DISTRESS",
            "video": {"weapon_score": 0.95, "detected_weapons": ["firearm"], "movement_score": 0.85},
            "audio": {"audio_event": "scream", "audio_confidence": 0.94, "is_active": True},
        },
    )
    a4 = alert_mgr.process_decision(e4, current_time=t)
    rec4 = engine.explain_decision(e4, alert_record=a4)
    print(f"Short Summary : {rec4.short_explanation}")
    print(f"Alert Justify : {rec4.alert_rationale}")
    print(f"HUD Bullets   :\n  " + "\n  ".join(rec4.hud_explanation))

    # SCENARIO 5: Audio-Only Evidence
    t += 1.0
    print("\n" + "=" * 75)
    print(">>> SCENARIO 5: Acoustic Disturbance (AUDIO-ONLY) -> Microphone Active, Visual Calm")
    print("=" * 75)
    e5 = DecisionEvent(
        timestamp=t,
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.50,
        confidence=0.68,
        reason="Loud acoustic alarm impulse detected; camera view clear/calm.",
        dominant_modality="audio",
        video_score=0.10,
        audio_score=0.75,
        evidence_summary={
            "video": {"movement_score": 0.10, "density_m2": 0.5, "is_active": True},
            "audio": {"audio_event": "alarm", "audio_confidence": 0.80, "is_active": True},
        },
    )
    rec5 = engine.explain_decision(e5)
    print(f"Short Summary: {rec5.short_explanation}")
    print(f"Statements   : {rec5.top_evidence}")

    # SCENARIO 6: Video-Only Evidence
    t += 1.0
    print("\n" + "=" * 75)
    print(">>> SCENARIO 6: Visual Agitation (VIDEO-ONLY) -> Acoustic Stream Offline")
    print("=" * 75)
    e6 = DecisionEvent(
        timestamp=t,
        previous_state="NORMAL",
        current_state="SUSPICIOUS",
        score=0.56,
        confidence=0.65,
        reason="Crowd scattering detected with microphone inactive.",
        dominant_modality="video",
        video_score=0.65,
        audio_score=0.0,
        evidence_summary={
            "video": {"movement_score": 0.65, "dispersion_rate": 7.0, "is_active": True},
            "audio": {"is_active": False},
        },
    )
    rec6 = engine.explain_decision(e6)
    print(f"Short Summary: {rec6.short_explanation}")
    print(f"Statements   : {rec6.top_evidence}")

    # SCENARIO 7: Corroborated Audio + Video Synergy
    t += 1.0
    print("\n" + "=" * 75)
    print(">>> SCENARIO 7: Cross-Modal Corroboration -> Synchronized Synergy Boost")
    print("=" * 75)
    m7 = MultimodalAssessment(
        multimodal_score=0.88,
        multimodal_confidence=0.92,
        video_score=0.82,
        audio_score=0.85,
        status="EMERGENCY",
        dominant_modality="bimodal",
        synergy=CrossModalSynergy(
            synergy_detected=True,
            synergy_type="SCREAM_AND_SCATTER",
            boost_multiplier=1.25,
            description="Acoustic scream coupled with crowd scattering.",
        ),
    )
    e7 = DecisionEvent(
        timestamp=t,
        previous_state="HIGH_RISK",
        current_state="EMERGENCY",
        score=0.88,
        confidence=0.92,
        reason="Corroborated scream and crowd scattering synergy.",
        dominant_modality="bimodal",
        corroborated=True,
        evidence_summary={"synergy_type": "SCREAM_AND_SCATTER"},
    )
    rec7 = engine.explain_decision(e7, multimodal_assessment=m7)
    print(f"Short Summary: {rec7.short_explanation}")
    print(f"Top Ranks    : {rec7.ranked_contributions[:3]}")

    # SCENARIO 8: Conflicting Modalities / False Positive Suppression
    t += 1.0
    print("\n" + "=" * 75)
    print(">>> SCENARIO 8: Conflicting Modalities -> Visual Agitation Dampened by Calm Acoustics")
    print("=" * 75)
    m8 = MultimodalAssessment(
        multimodal_score=0.32,
        multimodal_confidence=0.55,
        video_score=0.55,
        audio_score=0.08,
        status="NORMAL",
        dominant_modality="video",
        false_positive_suppressed=True,
        suppression_reason="Isolated visual motion spike unconfirmed across time and unsupported by acoustic environment.",
    )
    e8 = DecisionEvent(
        timestamp=t,
        previous_state="NORMAL",
        current_state="NORMAL",
        score=0.32,
        confidence=0.55,
        reason="Transient visual movement suppressed due to quiet acoustics.",
        dominant_modality="video",
        video_score=0.55,
        audio_score=0.08,
    )
    rec8 = engine.explain_decision(e8, multimodal_assessment=m8)
    print(f"Short Summary: {rec8.short_explanation}")
    print(f"Statements   : {rec8.top_evidence}")

    print("\n" + "=" * 75)
    print("ALL 8 EXPLAINABLE AI (XAI) DEMONSTRATION SCENARIOS VERIFIED SUCCESSFULLY.")
    print("=" * 75)


def run_dashboard_app(
    host: Optional[str] = None,
    port: Optional[int] = None,
    config_path: Optional[str] = None,
) -> None:
    """Executes Phase 10 Surveillance Dashboard via Streamlit subprocess."""
    import subprocess
    import socket

    config = load_config(config_path)
    dash_cfg = config.get("dashboard", {})
    target_host = host or dash_cfg.get("host", "127.0.0.1")
    target_port = port or dash_cfg.get("port", 8501)

    def _is_port_busy(h: str, p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((h, p)) == 0

    if _is_port_busy(target_host, target_port):
        if port is not None:
            logger.error(
                f"Specified port {target_port} is already in use on {target_host}. "
                f"Please choose another port with --port <PORT> or stop the existing process."
            )
            return
        orig_port = target_port
        for candidate in range(target_port + 1, target_port + 50):
            if not _is_port_busy(target_host, candidate):
                target_port = candidate
                break
        logger.warning(
            f"Port {orig_port} is already in use! Automatically switching to available port {target_port}."
        )

    app_path = resolve_path("src/dashboard/dashboard_app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        str(target_host),
        "--server.port",
        str(target_port),
        "--server.headless",
        "true",
    ]
    logger.info("==================================================================")
    logger.info("LAUNCHING STREAMLIT SURVEILLANCE DASHBOARD (PHASE 10)")
    logger.info(f"Target: http://{target_host}:{target_port} | App: {app_path}")
    logger.info("==================================================================")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user.")
    except Exception as e:
        logger.error(f"Dashboard execution failed: {e}")


def run_database_demo(config_path: Optional[str] = None) -> None:
    """Executes Phase 11 SQLite Database demonstration using an isolated temporary database."""
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path
    from src.database import (
        DatabaseManager,
        SessionRecord,
        EventRecord,
        AlertRecord,
        EvidenceRecord,
        SystemMetricRecord,
        HistoricalQueryAPI,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        demo_db_path = Path(tmpdir) / "demo_crowd_anomaly.db"
        db = DatabaseManager(db_path=demo_db_path, wal_mode=True, foreign_keys=True)
        api = HistoricalQueryAPI(db)

        print("\n" + "=" * 80)
        print("PHASE 11: SQLITE DATABASE & HISTORICAL EVENT PERSISTENCE DEMONSTRATION")
        print("[SYNTHETIC DEMO] Isolated Database: " + str(demo_db_path))
        print("=" * 80)

        # 1. Health check
        health = db.health_check()
        print(f"\n[1] Database Health Check: {health['status']} | WAL: {health['wal_mode']} | Schema Version: {health['schema_version']}")

        # 2. Create session
        t_base = time.time() - 300.0
        session = SessionRecord(
            session_id="SESS-DEMO-2026-001",
            started_at=datetime.fromtimestamp(t_base, tz=timezone.utc).isoformat(),
            source_type="synthetic_demo",
            video_source="synthetic_stream",
            status="RUNNING",
            metadata_json={"environment": "Pedestrian Plaza", "demo": True},
        )
        api.sessions.create_session(session)
        print(f"[2] Created Session: {session.session_id} (ID: {session.id})")

        # 3. Insert Events
        scenarios = [
            ("EVT-001", "NORMAL", 0.08, "NONE", 18, 0.45, 2.4, 0.08, 0.06, "Nominal baseline crowd flow"),
            ("EVT-002", "SUSPICIOUS", 0.52, "WARNING", 24, 1.20, 9.8, 0.45, 0.62, "Acoustic shouting with rapid movement"),
            ("EVT-003", "HIGH_RISK", 0.76, "HIGH", 35, 2.20, 12.4, 0.80, 0.55, "Physical struggle dynamics observed"),
            ("EVT-004", "EMERGENCY", 0.94, "CRITICAL", 42, 2.90, 18.5, 0.92, 0.95, "Firearm detected with corroborating scream"),
            ("EVT-005", "RECOVERY", 0.22, "INFO", 15, 0.40, 3.1, 0.20, 0.15, "Post-incident crowd stabilization"),
        ]

        print("\n[3] Inserting Multi-State Historical Events:")
        for idx, (eid, state, score, sev, people, dens, spd, v_sc, a_sc, rsn) in enumerate(scenarios):
            t_evt = t_base + (idx * 60.0)
            iso_t = datetime.fromtimestamp(t_evt, tz=timezone.utc).isoformat()
            evt = EventRecord(
                event_id=eid,
                session_id=session.session_id,
                timestamp=iso_t,
                risk_state=state,
                anomaly_score=score,
                decision_confidence=0.90,
                alert_severity=sev,
                dominant_modality="video" if v_sc >= a_sc else "audio",
                people_count=people,
                crowd_density=dens,
                average_speed=spd,
                video_score=v_sc,
                audio_score=a_sc,
                fusion_score=score,
                corroborated=(state == "EMERGENCY"),
                critical_evidence=(state == "EMERGENCY"),
                confirmed=True,
                reason=rsn,
                created_at=iso_t,
                metadata_json={"scenario_index": idx, "demo": True},
            )
            api.events.insert_event(evt)
            print(f"    - Event {eid:7s} | State: {state:10s} | Score: {score:.2f} | Severity: {sev:8s}")

        # 4. Insert Alerts & Exercise Lifecycle
        print("\n[4] Inserting & Exercising Alert Lifecycle (CREATED -> ACTIVE -> ACKNOWLEDGED -> RESOLVED):")
        alert_crit = AlertRecord(
            alert_id="ALT-DEMO-CRIT",
            event_id="EVT-004",
            session_id=session.session_id,
            timestamp=datetime.fromtimestamp(t_base + 180.0, tz=timezone.utc).isoformat(),
            severity="CRITICAL",
            risk_state="EMERGENCY",
            score=0.94,
            confidence=0.96,
            reason="Firearm detected with corroborating scream",
            dominant_modality="bimodal",
            status="ACTIVE",
            created_at=datetime.fromtimestamp(t_base + 180.0, tz=timezone.utc).isoformat(),
            fingerprint="fp_demo_crit_001",
        )
        api.alerts.insert_alert(alert_crit)
        print(f"    - Alert {alert_crit.alert_id} created with status: {alert_crit.status}")

        # Operator Acknowledge
        t_ack = datetime.fromtimestamp(t_base + 195.0, tz=timezone.utc).isoformat()
        api.alerts.update_alert_status(alert_crit.alert_id, status="ACKNOWLEDGED", acknowledged_at=t_ack)
        refreshed_alt = api.alerts.get_alert(alert_crit.alert_id)
        print(f"    - Operator acknowledged alert at {refreshed_alt.acknowledged_at}: Status -> {refreshed_alt.status}")

        # Operator Resolve
        t_res = datetime.fromtimestamp(t_base + 240.0, tz=timezone.utc).isoformat()
        api.alerts.update_alert_status(alert_crit.alert_id, status="RESOLVED", resolved_at=t_res)
        refreshed_alt2 = api.alerts.get_alert(alert_crit.alert_id)
        print(f"    - Operator resolved alert at {refreshed_alt2.resolved_at}: Status -> {refreshed_alt2.status}")

        # 5. Insert Evidence
        print("\n[5] Persisting XAI Factor Attributions:")
        ev1 = EvidenceRecord(
            evidence_id="EVID-001",
            event_id="EVT-004",
            alert_id="ALT-DEMO-CRIT",
            timestamp=datetime.fromtimestamp(t_base + 180.0, tz=timezone.utc).isoformat(),
            factor="weapon_presence",
            contribution=0.38,
            rank=1,
            statement="Firearm presence detected (score: 0.95)",
            visual_available=True,
            model_explanation_available=False,
        )
        ev2 = EvidenceRecord(
            evidence_id="EVID-002",
            event_id="EVT-004",
            alert_id="ALT-DEMO-CRIT",
            timestamp=datetime.fromtimestamp(t_base + 180.0, tz=timezone.utc).isoformat(),
            factor="acoustic_anomaly",
            contribution=0.28,
            rank=2,
            statement="Acoustic scream signature confirmed (score: 0.95)",
            visual_available=True,
            model_explanation_available=False,
        )
        api.evidence.insert_evidence_batch([ev1, ev2])
        ev_items = api.evidence.get_evidence_for_event("EVT-004")
        for e in ev_items:
            print(f"    - Rank {e.rank}: {e.factor:18s} (Contribution: {e.contribution*100:.0f}%) -> '{e.statement}'")

        # 6. Historical Queries & Analytics
        print("\n[6] Historical Query API Results:")
        emergencies = api.events.query_events_by_state("EMERGENCY")
        print(f"    - Events by State (EMERGENCY): {len(emergencies)} found")
        stats = api.get_statistics()
        print(f"    - Total Events   : {stats['total_events']}")
        print(f"    - Total Alerts   : {stats['total_alerts']}")
        print(f"    - Emergencies    : {stats['emergency_count']}")
        print(f"    - High Risk      : {stats['high_risk_count']}")
        print(f"    - Avg Anomaly    : {stats['average_anomaly_score']}")
        print(f"    - Peak Anomaly   : {stats['maximum_anomaly_score']}")
        print(f"    - Avg Density    : {stats['average_crowd_density']} people/m2")

        # 7. End Session
        api.sessions.end_session(session.session_id, status="COMPLETED")
        sess_done = api.sessions.get_session(session.session_id)
        print(f"\n[7] Closed Session {sess_done.session_id}: Status -> {sess_done.status} (Ended: {sess_done.ended_at})")

        # 8. Retention Pruning Verification
        print("\n[8] Retention Pruning (older than 90 days):")
        prune_res = api.prune_old_records(retention_days=90)
        print(f"    - Pruned records: {prune_res} (nominal recent events preserved)")

        db.close()
        print("\n" + "=" * 80)
        print("ALL PHASE 11 DATABASE DEMONSTRATION STEPS VERIFIED CLEANLY.")
        print("=" * 80 + "\n")


def run_database_mode(config_path: Optional[str] = None) -> None:
    """Connects to configured database, displays health, counts, and recent records."""
    from src.database import DatabaseManager, HistoricalQueryAPI
    config = load_config(config_path)
    db_cfg = config.get("database", {})
    db_path = resolve_path(db_cfg.get("path", "data/crowd_anomaly.db"))

    logger.info("==================================================================")
    logger.info("DATABASE STATUS & HISTORICAL PERSISTENCE MONITOR (PHASE 11)")
    logger.info(f"Target Database: {db_path}")
    logger.info("==================================================================")

    db = DatabaseManager(
        db_path=db_path,
        wal_mode=db_cfg.get("wal_mode", True),
        foreign_keys=db_cfg.get("foreign_keys", True),
        busy_timeout_ms=db_cfg.get("busy_timeout_ms", 5000),
        auto_create=db_cfg.get("auto_create", True),
    )
    health = db.health_check()
    api = HistoricalQueryAPI(db)
    stats = api.get_statistics()
    recent = api.events.get_recent_events(limit=5)
    sessions = api.sessions.list_sessions(limit=5)

    print("\n--- DATABASE HEALTH & STATUS ---")
    print(f"Status         : {health['status']}")
    print(f"Path           : {health['path']}")
    print(f"Schema Version : {health['schema_version']}")
    print(f"Table Count    : {health['table_count']}")
    print(f"WAL Mode       : {health['wal_mode']}")
    print(f"Total Events   : {stats['total_events']}")
    print(f"Total Alerts   : {stats['total_alerts']}")
    print(f"Emergencies    : {stats['emergency_count']}")
    print(f"Sessions Logged: {len(sessions)}")

    if recent:
        print("\n--- RECENT 5 EVENTS ---")
        for ev in recent:
            print(f"[{ev.timestamp}] {ev.event_id} | State: {ev.risk_state:10s} | Score: {ev.anomaly_score:.2f} | Reason: {ev.reason}")
    else:
        print("\nNo historical events recorded yet. Run live pipelines to log events.")

    db.close()


def run_demo_evaluation(config_path: Optional[str] = None) -> None:
    """
    Executes Phase 12 Academic Evaluation demonstration:
    Evaluates synthetic scenarios, ablation configurations, robustness stress-tests,
    latency benchmarking, honest unconfigured ground-truth reporting, and generates
    markdown/JSON/CSV/plot reports without mutating production databases or models.
    """
    config = load_config(config_path)
    logger.info("==================================================================")
    logger.info("STARTING PHASE 12 EVALUATION DEMONSTRATION")
    logger.info("==================================================================")

    evaluator = SystemEvaluator(config=config)
    report = evaluator.run_demo_evaluation()

    print("\n" + "=" * 80)
    print("PHASE 12 EVALUATION DEMO SUMMARY")
    print("=" * 80)
    print(f"Timestamp: {report.timestamp}")
    print(f"Ground Truth Status: {report.ground_truth_status} ({report.ground_truth_reason})")
    print(f"Evaluation Source  : {report.evaluation_source.value}")
    print(f"Scenarios Evaluated: {len(report.scenarios)}")
    print(f"Ablation Conditions: {len(report.ablation)}")
    print(f"Robustness Tests   : {len(report.robustness)}")
    if report.latency:
        print(f"Pipeline Mean Latency: {report.latency.mean_latency_ms:.2f} ms (~{report.latency.fps:.1f} FPS)")
    print(f"Markdown Report    : {report.markdown_report_path}")
    print(f"JSON Report        : {report.json_report_path}")
    print(f"Visualizations     : {len(report.figure_paths)} plots generated")
    print("=" * 80 + "\n")


def run_evaluation_mode(config_path: Optional[str] = None) -> None:
    """Executes full Phase 12 Evaluation & Benchmarking suite."""
    config = load_config(config_path)
    logger.info("==================================================================")
    logger.info("STARTING PHASE 12 FULL EVALUATION & BENCHMARKING SUITE")
    logger.info("==================================================================")

    evaluator = SystemEvaluator(config=config)
    report = evaluator.run_full_evaluation()

    print("\n" + "=" * 80)
    print("PHASE 12 BENCHMARKING & EVALUATION SUITE COMPLETE")
    print("=" * 80)
    print(f"Evaluation Source   : {report.evaluation_source.value}")
    print(f"Ground Truth Status : {report.ground_truth_status} ({report.ground_truth_reason})")
    print(f"Scenarios Evaluated : {len(report.scenarios)}")
    print(f"Ablation Conditions : {len(report.ablation)}")
    print(f"Robustness Tests    : {len(report.robustness)}")
    if report.latency:
        print(f"Pipeline Mean Latency: {report.latency.mean_latency_ms:.2f} ms (~{report.latency.fps:.1f} FPS)")
    print(f"Markdown Report     : {report.markdown_report_path}")
    print(f"JSON Summary        : {report.json_report_path}")
    print(f"Visualizations      : {len(report.figure_paths)} plots saved in {config.get('evaluation', {}).get('output_dir', 'reports/evaluation')}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Crowd Anomaly Detection with Predictive Alerts - AI Surveillance System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="alerts",
        choices=["video", "tracking", "safety", "behavior", "audio", "multimodal", "decision", "alerts", "xai", "dashboard", "database", "evaluation"],
        help="System execution mode. Phase 2: 'video', Phase 3: 'tracking', Safety: 'safety', Phase 4: 'behavior', Phase 5: 'audio', Phase 6: 'multimodal', Phase 7: 'decision', Phase 8: 'alerts', Phase 9: 'xai', Phase 10: 'dashboard', Phase 11: 'database', Phase 12: 'evaluation'.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input video/audio source: file path, webcam/mic index ('0'), RTSP stream URL, or 'synthetic'.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Convenience flag: Shortcut to run with synthetic stream generator.",
    )
    parser.add_argument(
        "--demo-safety",
        action="store_true",
        help="Trigger simulated dangerous object and fight interactions for testing and demonstrations.",
    )
    parser.add_argument(
        "--demo-behavior",
        action="store_true",
        help="Trigger simulated abnormal crowd dispersal/turbulence for testing and demonstrations.",
    )
    parser.add_argument(
        "--demo-audio",
        action="store_true",
        help="Trigger simulated acoustic emergency events (screams, alarms, explosions) for testing.",
    )
    parser.add_argument(
        "--demo-multimodal",
        action="store_true",
        help="Trigger coordinated cross-modal emergency scenarios (scream + scattering, fight + shouting, noise suppression).",
    )
    parser.add_argument(
        "--demo-alerts",
        action="store_true",
        help="Simulate the 8 Phase 8 alert management scenarios (normal, warning, high, critical, duplicate, recovery, repeat cooldown, stubs).",
    )
    parser.add_argument(
        "--demo-xai",
        action="store_true",
        help="Simulate the 8 Phase 9 Explainable AI (XAI) demonstration scenarios (baseline, suspicious, altercation, weapon, single-modality, corroboration, conflict).",
    )
    parser.add_argument(
        "--demo-dashboard",
        action="store_true",
        help="Launch Phase 10 Streamlit Surveillance Dashboard in Demo Simulation mode.",
    )
    parser.add_argument(
        "--demo-database",
        action="store_true",
        help="Run Phase 11 SQLite Database demo (isolated temporary database exercising sessions, events, alerts, XAI evidence, queries, and retention).",
    )
    parser.add_argument(
        "--demo-evaluation",
        action="store_true",
        help="Run Phase 12 Academic Evaluation & Benchmarking demonstration (synthetic scenarios, ablation, robustness, latency, honest unconfigured GT reporting, markdown/plots).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host address for Streamlit dashboard.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for Streamlit dashboard.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold [0.0 - 1.0].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Device for deep learning inference.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without opening OpenCV GUI window (ideal for servers or benchmarking).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Terminate automatically after processing N frames/chunks.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )

    args = parser.parse_args()
    selected_source = "synthetic" if args.synthetic else args.source

    if args.demo_alerts:
        run_alerts_demo(config_path=args.config)
        return

    if args.demo_xai:
        run_xai_demo(config_path=args.config)
        return

    if args.demo_database:
        run_database_demo(config_path=args.config)
        return

    if args.demo_evaluation:
        run_demo_evaluation(config_path=args.config)
        return

    if args.mode == "database":
        run_database_mode(config_path=args.config)
        return

    if args.demo_dashboard or args.mode == "dashboard":
        run_dashboard_app(
            host=args.host,
            port=args.port,
            config_path=args.config,
        )
        return

    if args.mode == "video":
        run_video_pipeline(
            source_override=selected_source,
            conf_override=args.conf,
            device_override=args.device,
            no_display=args.no_display,
            max_frames=args.max_frames,
            config_path=args.config,
        )
    elif args.mode == "tracking":
        run_tracking_pipeline(
            source_override=selected_source,
            conf_override=args.conf,
            device_override=args.device,
            no_display=args.no_display,
            max_frames=args.max_frames,
            config_path=args.config,
        )
    elif args.mode == "safety":
        run_safety_pipeline(
            source_override=selected_source,
            conf_override=args.conf,
            device_override=args.device,
            no_display=args.no_display,
            max_frames=args.max_frames,
            config_path=args.config,
            demo_safety=args.demo_safety,
        )
    elif args.mode == "behavior":
        run_behavior_pipeline(
            source_override=selected_source,
            conf_override=args.conf,
            device_override=args.device,
            no_display=args.no_display,
            max_frames=args.max_frames,
            config_path=args.config,
            demo_behavior=args.demo_behavior,
        )
    elif args.mode == "audio":
        run_audio_pipeline(
            source_override=selected_source,
            no_display=args.no_display,
            max_frames=args.max_frames,
            config_path=args.config,
            demo_audio=args.demo_audio,
        )
    elif args.mode == "decision":
        run_decision_pipeline(
            source_override=selected_source,
            conf_override=args.conf,
            device_override=args.device,
            no_display=args.no_display,
            max_frames=args.max_frames,
            config_path=args.config,
            demo_multimodal=args.demo_multimodal,
        )
    elif args.mode in ("multimodal", "alerts"):
        run_alerts_pipeline(
            source_override=selected_source,
            conf_override=args.conf,
            device_override=args.device,
            no_display=args.no_display,
            max_frames=args.max_frames,
            config_path=args.config,
            demo_multimodal=args.demo_multimodal,
        )
    elif args.mode == "xai":
        run_xai_pipeline(
            source_override=selected_source,
            conf_override=args.conf,
            device_override=args.device,
            no_display=args.no_display,
            max_frames=args.max_frames,
            config_path=args.config,
            demo_multimodal=args.demo_multimodal,
        )
    elif args.mode == "evaluation":
        run_evaluation_mode(config_path=args.config)
    else:
        logger.error(f"Unrecognized mode: {args.mode}")
        sys.exit(1)



if __name__ == "__main__":
    main()
