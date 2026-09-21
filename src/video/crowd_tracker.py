"""
Crowd Tracker Module.
Provides multi-object tracking with unique persistent IDs, trajectory history,
velocity, acceleration, and heading calculations for behavioral surveillance.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.video.person_detector import DetectionResult
from src.utils.logger import setup_logger

logger = setup_logger("crowd_tracker")


@dataclass
class TrackedPerson:
    """
    Temporal state of an individually tracked person across video frames.
    """
    track_id: int
    current_bbox: np.ndarray                    # [x1, y1, x2, y2]
    history: List[Tuple[float, float]] = field(default_factory=list)  # [(cx, cy), ...]
    speed: float = 0.0                          # Current velocity magnitude (pixels/frame)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
    acceleration: float = 0.0                   # Rate of change of speed
    direction_angle: float = 0.0                # Movement heading in degrees [0, 360)
    confidence: float = 1.0
    age: int = 1                                # Total frames tracked
    time_since_update: int = 0                  # Frames since last detection match

    def update_position(
        self,
        bbox: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        history_length: int = 30,
    ) -> None:
        """Updates track kinematics with new detection observation."""
        self.current_bbox = bbox
        self.confidence = confidence
        self.time_since_update = 0
        self.age += 1

        prev_speed = self.speed
        if self.history:
            prev_cx, prev_cy = self.history[-1]
            vx = center[0] - prev_cx
            vy = center[1] - prev_cy
            inst_speed = math.hypot(vx, vy)

            # Smooth speed and velocity
            self.speed = 0.7 * inst_speed + 0.3 * prev_speed
            self.velocity = (vx, vy)
            self.acceleration = abs(self.speed - prev_speed)

            if inst_speed > 0.5:
                angle_rad = math.atan2(vy, vx)
                self.direction_angle = (math.degrees(angle_rad) + 360.0) % 360.0
        else:
            self.speed = 0.0
            self.velocity = (0.0, 0.0)
            self.acceleration = 0.0

        self.history.append(center)
        if len(self.history) > history_length:
            self.history.pop(0)

    def mark_missed(self) -> None:
        """Called when no detection was matched in the current frame."""
        self.time_since_update += 1
        # Dampen speed on missing observation
        self.speed *= 0.8
        self.acceleration = 0.0


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise Intersection-over-Union (IoU) matrix between two sets of boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1_a, y1_a, x2_a, y2_a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    x1_b, y1_b, x2_b, y2_b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    iou_mat = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    for i in range(len(boxes_a)):
        xx1 = np.maximum(x1_a[i], x1_b)
        yy1 = np.maximum(y1_a[i], y1_b)
        xx2 = np.minimum(x2_a[i], x2_b)
        yy2 = np.minimum(y2_a[i], y2_b)

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = area_a[i] + area_b - inter
        iou_mat[i] = np.where(union > 0, inter / union, 0.0)

    return iou_mat


class ByteTrackCrowdTracker:
    """
    Robust Multi-Object Tracker utilizing IoU association and centroid proximity.
    Maintains persistent track identities and movement metrics across frames.
    """

    def __init__(
        self,
        history_length: int = 30,
        speed_threshold: float = 25.0,
        max_lost: int = 15,
        iou_threshold: float = 0.25,
    ):
        self.history_length = history_length
        self.speed_threshold = speed_threshold
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

        self.next_track_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}
        logger.info(f"ByteTrack Crowd Tracker initialized (history_len={history_length}, iou_thresh={iou_threshold}).")

    def update(self, detection: DetectionResult, frame: Optional[np.ndarray] = None) -> List[TrackedPerson]:
        """
        Associates incoming detections with existing tracks or initializes new tracks.

        Args:
            detection: DetectionResult containing current bounding boxes and centers.
            frame: Optional input frame (for visual feature tracking if needed).

        Returns:
            List of currently active TrackedPerson instances.
        """
        det_count = detection.person_count
        det_boxes = detection.boxes
        det_centers = detection.centers
        det_confs = detection.confidences

        existing_track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        if existing_track_ids and det_count > 0:
            track_boxes = np.array([self.tracks[tid].current_bbox for tid in existing_track_ids])
            iou_matrix = compute_iou_matrix(det_boxes, track_boxes)

            # Greedy bipartite matching
            num_dets, num_tracks = iou_matrix.shape
            for _ in range(min(num_dets, num_tracks)):
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]

                if max_iou < self.iou_threshold:
                    break

                det_i, track_j = max_idx
                tid = existing_track_ids[track_j]

                # Update matched track
                center = (float(det_centers[det_i, 0]), float(det_centers[det_i, 1]))
                self.tracks[tid].update_position(
                    bbox=det_boxes[det_i],
                    center=center,
                    confidence=float(det_confs[det_i]),
                    history_length=self.history_length,
                )
                matched_dets.add(det_i)
                matched_tracks.add(tid)

                # Suppress rows and columns
                iou_matrix[det_i, :] = -1.0
                iou_matrix[:, track_j] = -1.0

        # Create new tracks for unmatched detections
        for det_i in range(det_count):
            if det_i not in matched_dets:
                center = (float(det_centers[det_i, 0]), float(det_centers[det_i, 1]))
                new_track = TrackedPerson(
                    track_id=self.next_track_id,
                    current_bbox=det_boxes[det_i],
                    history=[center],
                    speed=0.0,
                    velocity=(0.0, 0.0),
                    acceleration=0.0,
                    direction_angle=0.0,
                    confidence=float(det_confs[det_i]),
                    age=1,
                    time_since_update=0,
                )
                self.tracks[self.next_track_id] = new_track
                matched_tracks.add(self.next_track_id)
                self.next_track_id += 1

        # Age unmatched tracks and remove stale ones
        stale_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.mark_missed()
                if track.time_since_update > self.max_lost:
                    stale_ids.append(tid)

        for tid in stale_ids:
            del self.tracks[tid]

        # Return active tracks visible in recent frames
        return [t for t in self.tracks.values() if t.time_since_update == 0]

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[TrackedPerson],
        draw_trajectory: bool = True,
        draw_id: bool = True,
        draw_speed: bool = True,
    ) -> np.ndarray:
        """
        Renders track IDs, bounding boxes, speed tags, and motion trajectory trails.
        """
        annotated = frame.copy()

        # Seed consistent colors based on track ID
        for track in tracks:
            np.random.seed(track.track_id * 37)
            color = (
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
            )

            x1, y1, x2, y2 = track.current_bbox.astype(int)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw trajectory path
            if draw_trajectory and len(track.history) > 1:
                pts = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)

            # Draw tag
            label_parts = []
            if draw_id:
                label_parts.append(f"ID:{track.track_id}")
            if draw_speed and track.speed > 0.5:
                label_parts.append(f"{track.speed:.1f}px/f")

            if label_parts:
                label = " | ".join(label_parts)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
                y_label = max(y1 - 6, th + 4)
                cv2.rectangle(annotated, (x1, y_label - th - 2), (x1 + tw + 4, y_label + 2), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y_label), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
