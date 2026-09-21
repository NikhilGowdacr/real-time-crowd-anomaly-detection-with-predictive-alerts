"""
==============================================================================
Visual Explanation & Evidence Overlay Module (Phase 9).
Renders observable visual evidence on OpenCV surveillance frames:
- Kinematic motion trajectory vectors with velocity-based color coding
- Radial crowd dispersion indicator arrows during panic dispersal
- Structured XAI HUD evidence explanation card
==============================================================================
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.xai.evidence_record import EvidenceRecord
from src.video.crowd_tracker import TrackedPerson


class VisualExplainer:
    """
    Renders observable physical evidence directly onto surveillance frames
    to provide visual explainability for human surveillance operators.
    """

    def __init__(
        self,
        show_motion_vectors: bool = True,
        show_dispersion_arrows: bool = True,
        vector_scale: float = 3.0,
        min_speed_threshold: float = 1.0,
        rapid_speed_threshold: float = 8.0,
        brisk_speed_threshold: float = 4.0,
    ):
        self.show_motion_vectors = show_motion_vectors
        self.show_dispersion_arrows = show_dispersion_arrows
        self.vector_scale = float(vector_scale)
        self.min_speed_threshold = float(min_speed_threshold)
        self.rapid_speed_threshold = float(rapid_speed_threshold)
        self.brisk_speed_threshold = float(brisk_speed_threshold)

    def draw_motion_vectors(
        self,
        frame: np.ndarray,
        tracks: List[TrackedPerson],
        vector_scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draws velocity heading arrows for active tracked persons.
        Arrow color indicates velocity magnitude:
        - Green: Normal pedestrian walking (< 4 px/f)
        - Yellow/Orange: Brisk walking / agitated pace (4 - 8 px/f)
        - Red: Running / rapid panic movement (>= 8 px/f)

        Args:
            frame: Input BGR video frame.
            tracks: List of TrackedPerson objects from ByteTrack.
            vector_scale: Optional override for arrow length multiplier.

        Returns:
            Annotated BGR frame copy.
        """
        if frame is None:
            return frame

        annotated = frame.copy()
        if not tracks or not self.show_motion_vectors:
            return annotated

        scale = vector_scale if vector_scale is not None else self.vector_scale

        for track in tracks:
            speed = float(getattr(track, "speed", 0.0))
            if speed < self.min_speed_threshold:
                continue

            # Obtain track center
            bbox = getattr(track, "current_bbox", None)
            if bbox is not None and len(bbox) == 4:
                cx = int((bbox[0] + bbox[2]) / 2.0)
                cy = int((bbox[1] + bbox[3]) / 2.0)
            elif getattr(track, "history", None) and len(track.history) > 0:
                cx, cy = int(track.history[-1][0]), int(track.history[-1][1])
            else:
                continue

            # Determine displacement vector
            vx, vy = getattr(track, "velocity", (0.0, 0.0))
            if abs(vx) < 1e-3 and abs(vy) < 1e-3:
                angle_deg = getattr(track, "direction_angle", 0.0)
                rad = math.radians(angle_deg)
                vx = speed * math.cos(rad)
                vy = speed * math.sin(rad)

            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)

            # Clamp coordinates to frame dimensions
            h, w = annotated.shape[:2]
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))

            # Select color based on velocity threshold
            if speed >= self.rapid_speed_threshold:
                color = (0, 0, 255)       # Red: rapid movement
                thickness = 2
            elif speed >= self.brisk_speed_threshold:
                color = (0, 200, 255)     # Yellow-Orange: brisk movement
                thickness = 2
            else:
                color = (0, 255, 0)       # Green: normal walk
                thickness = 1

            cv2.arrowedLine(
                annotated,
                (cx, cy),
                (end_x, end_y),
                color,
                thickness=thickness,
                tipLength=0.35,
                line_type=cv2.LINE_AA,
            )

        return annotated

    def draw_dispersion_indicators(
        self,
        frame: np.ndarray,
        tracks: List[TrackedPerson],
        dispersion_rate: float,
        min_dispersion: float = 4.5,
    ) -> np.ndarray:
        """
        Draws radial outward dispersion arrows from crowd center of mass
        when crowd dispersion / scattering rate is elevated.

        Args:
            frame: Input BGR video frame.
            tracks: List of TrackedPerson objects.
            dispersion_rate: Estimated dispersion rate (pixels/frame).
            min_dispersion: Threshold to trigger dispersion visualization.

        Returns:
            Annotated BGR frame copy.
        """
        if frame is None:
            return frame

        annotated = frame.copy()
        if (
            not self.show_dispersion_arrows
            or dispersion_rate < min_dispersion
            or len(tracks) < 3
        ):
            return annotated

        # Calculate crowd centroid
        centers: List[Tuple[float, float]] = []
        for track in tracks:
            bbox = getattr(track, "current_bbox", None)
            if bbox is not None and len(bbox) == 4:
                centers.append(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))

        if len(centers) < 3:
            return annotated

        mean_cx = sum(c[0] for c in centers) / len(centers)
        mean_cy = sum(c[1] for c in centers) / len(centers)
        c_pt = (int(mean_cx), int(mean_cy))

        # Draw central dispersion hub circle
        cv2.circle(annotated, c_pt, 8, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, c_pt, 16, (0, 140, 255), 1, cv2.LINE_AA)

        # Draw 8 radial outward dispersion arrows
        arrow_len = min(60, int(15 + dispersion_rate * 5))
        for angle_deg in range(0, 360, 45):
            rad = math.radians(angle_deg)
            start_x = int(mean_cx + 20 * math.cos(rad))
            start_y = int(mean_cy + 20 * math.sin(rad))
            end_x = int(mean_cx + (20 + arrow_len) * math.cos(rad))
            end_y = int(mean_cy + (20 + arrow_len) * math.sin(rad))

            cv2.arrowedLine(
                annotated,
                (start_x, start_y),
                (end_x, end_y),
                (0, 140, 255),
                thickness=2,
                tipLength=0.3,
                line_type=cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            f"SCATTERING DETECTED ({dispersion_rate:.1f} px/f)",
            (c_pt[0] - 80, c_pt[1] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated

    def draw_xai_hud_card(
        self,
        frame: np.ndarray,
        evidence: EvidenceRecord,
        x: int = 16,
        y: int = 70,
        width: int = 340,
        height: int = 240,
    ) -> np.ndarray:
        """
        Renders an XAI Diagnostic Evidence card on the video surveillance monitor.

        Args:
            frame: Input BGR frame.
            evidence: EvidenceRecord containing explanations and attributions.
            x, y: Top-left coordinate for HUD card.
            width, height: Dimensions of card.

        Returns:
            Annotated BGR frame copy.
        """
        if frame is None or evidence is None:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Clamp box bounds
        card_w = min(width, w - x - 10)
        card_h = min(height, h - y - 10)
        if card_w < 100 or card_h < 80:
            return annotated

        # Draw semi-transparent card background
        card_roi = annotated[y : y + card_h, x : x + card_w]
        card_bg = np.full(card_roi.shape, (22, 22, 28), dtype=np.uint8)
        cv2.addWeighted(card_roi, 0.20, card_bg, 0.80, 0, card_roi)
        annotated[y : y + card_h, x : x + card_w] = card_roi
        cv2.rectangle(annotated, (x, y), (x + card_w, y + card_h), (80, 80, 100), 1)

        # Card Title
        cv2.putText(
            annotated,
            "EXPLAINABLE AI (XAI) EVIDENCE",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(annotated, (x + 8, y + 28), (x + card_w - 8, y + 28), (60, 60, 80), 1)

        y_offset = y + 46
        line_gap = 18

        # 1. State and Score breakdown
        state_str = str(evidence.risk_state)
        st_color = (0, 0, 255) if state_str == "EMERGENCY" else (
            (0, 100, 255) if state_str == "HIGH_RISK" else (
                (0, 180, 255) if state_str == "SUSPICIOUS" else (0, 255, 0)
            )
        )
        cv2.putText(
            annotated,
            f"State: {state_str} | Anomaly: {evidence.anomaly_score:.2f} | Conf: {evidence.decision_confidence:.2f}",
            (x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            st_color,
            1,
            cv2.LINE_AA,
        )
        y_offset += line_gap

        # 2. Top Evidence Contributions (Attributions)
        cv2.putText(
            annotated,
            "Top Evidence Contributions:",
            (x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        y_offset += line_gap

        # Draw up to 3 top factor contribution bars
        ranked = evidence.ranked_contributions[:3]
        if not ranked:
            cv2.putText(
                annotated,
                "- Nominal Baseline (All contributions 0%)",
                (x + 16, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
            y_offset += line_gap
        else:
            for factor_name, weight in ranked:
                if y_offset > y + card_h - 40:
                    break
                factor_display = factor_name.replace("_", " ").title()[:18]
                pct_str = f"{int(weight * 100)}%"
                cv2.putText(
                    annotated,
                    f"- {factor_display}: {pct_str}",
                    (x + 16, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                # Mini bar
                bar_x = x + 165
                bar_w = int(max(2, min(90, weight * 90)))
                bar_color = (0, 0, 255) if weight > 0.4 else ((0, 180, 255) if weight > 0.2 else (0, 255, 120))
                cv2.rectangle(annotated, (bar_x, y_offset - 10), (bar_x + bar_w, y_offset - 2), bar_color, -1)
                cv2.rectangle(annotated, (bar_x, y_offset - 10), (bar_x + 90, y_offset - 2), (80, 80, 80), 1)
                y_offset += line_gap

        # 3. Primary Evidence Summary Statement
        if y_offset <= y + card_h - 22:
            cv2.line(annotated, (x + 8, y_offset - 6), (x + card_w - 8, y_offset - 6), (50, 50, 70), 1)
            summary_txt = evidence.short_explanation[:50] + ("..." if len(evidence.short_explanation) > 50 else "")
            cv2.putText(
                annotated,
                summary_txt,
                (x + 10, y_offset + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (200, 220, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += line_gap

        # 4. Model Explainability Status (Academic Transparency)
        if y_offset <= y + card_h - 6:
            saliency_tag = "Grad-CAM: ACTIVE" if evidence.model_explanation_available else "Model Saliency: UNAVAILABLE (Baseline Heuristics)"
            cv2.putText(
                annotated,
                saliency_tag,
                (x + 10, y + card_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (140, 140, 160),
                1,
                cv2.LINE_AA,
            )

        return annotated
