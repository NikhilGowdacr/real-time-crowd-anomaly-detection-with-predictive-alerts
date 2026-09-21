"""
Pose and Movement Kinematics Analysis Module.
Extracts spatiotemporal motion features, pairwise interaction distances,
relative velocities, and reciprocal movement dynamics from tracked pedestrians.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.video.crowd_tracker import TrackedPerson
from src.utils.logger import setup_logger

logger = setup_logger("pose_analyzer")


@dataclass
class InteractingPair:
    """
    Kinematic metrics between two interacting tracked persons.
    """
    track_id_1: int
    track_id_2: int
    distance: float                           # Centroid distance in pixels
    relative_speed: float                     # Magnitude difference between velocities
    convergence_rate: float                   # Rate at which distance is decreasing (pixels/f)
    reciprocal_motion_score: float            # Oscillating opposite-direction motion indicator [0.0 - 1.0]
    center_point: Tuple[float, float]         # Midpoint between interaction (cx, cy)
    union_bbox: np.ndarray                    # [x1, y1, x2, y2] enclosing both persons


@dataclass
class KinematicFeatures:
    """
    Aggregated movement metrics across all active tracks in a frame.
    """
    active_track_count: int
    mean_speed: float
    max_individual_speed: float
    max_individual_accel: float
    interacting_pairs: List[InteractingPair] = field(default_factory=list)
    abnormal_movement_score: float = 0.0      # [0.0 - 1.0]
    timestamp: float = 0.0


class PoseMovementAnalyzer:
    """
    Analyzes multi-pedestrian kinematic interactions:
    - Pairwise Euclidean distance matrix
    - Spatial convergence (rapidly closing distance)
    - High-frequency reciprocal velocity reversals (signature of violent physical struggles)
    - Acceleration outliers
    """

    def __init__(
        self,
        proximity_threshold: float = 130.0,
        accel_threshold: float = 10.0,
        speed_threshold: float = 18.0,
    ):
        self.proximity_threshold = proximity_threshold
        self.accel_threshold = accel_threshold
        self.speed_threshold = speed_threshold

        # Track-pair distance history for convergence rate calculation: (id1, id2) -> [d_prev, ...]
        self._pair_distance_history: Dict[Tuple[int, int], List[float]] = {}

    def extract_features(
        self,
        tracked_persons: List[TrackedPerson],
        timestamp: Optional[float] = None,
    ) -> KinematicFeatures:
        """
        Extracts kinematic interaction features across active tracks.

        Args:
            tracked_persons: List of active TrackedPerson instances.
            timestamp: Frame epoch timestamp.

        Returns:
            KinematicFeatures dataclass.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        track_count = len(tracked_persons)

        if track_count == 0:
            return KinematicFeatures(
                active_track_count=0,
                mean_speed=0.0,
                max_individual_speed=0.0,
                max_individual_accel=0.0,
                interacting_pairs=[],
                abnormal_movement_score=0.0,
                timestamp=curr_time,
            )

        speeds = [t.speed for t in tracked_persons]
        accels = [t.acceleration for t in tracked_persons]

        mean_speed = float(np.mean(speeds))
        max_speed = float(np.max(speeds))
        max_accel = float(np.max(accels))

        # 1. Pairwise Interaction Analysis
        interacting_pairs: List[InteractingPair] = []
        active_pair_keys = set()

        for i in range(track_count):
            p1 = tracked_persons[i]
            cx1, cy1 = p1.history[-1] if p1.history else (0.0, 0.0)

            for j in range(i + 1, track_count):
                p2 = tracked_persons[j]
                cx2, cy2 = p2.history[-1] if p2.history else (0.0, 0.0)

                dist = math.hypot(cx1 - cx2, cy1 - cy2)
                pair_key = (min(p1.track_id, p2.track_id), max(p1.track_id, p2.track_id))
                active_pair_keys.add(pair_key)

                # Track pairwise distance history
                if pair_key not in self._pair_distance_history:
                    self._pair_distance_history[pair_key] = []
                hist = self._pair_distance_history[pair_key]
                hist.append(dist)
                if len(hist) > 10:
                    hist.pop(0)

                # Convergence rate: positive if distance decreased
                convergence_rate = 0.0
                if len(hist) >= 2:
                    convergence_rate = max(0.0, hist[-2] - hist[-1])

                # Reciprocal motion analysis:
                # Interacting people moving towards each other or fighting show opposite velocity vectors
                reciprocal_score = 0.0
                vx1, vy1 = p1.velocity
                vx2, vy2 = p2.velocity
                dot_product = (vx1 * vx2) + (vy1 * vy2)

                # If dot product is negative, headings oppose each other
                if dot_product < -5.0 and dist < self.proximity_threshold:
                    relative_vel_mag = math.hypot(vx1 - vx2, vy1 - vy2)
                    reciprocal_score = min(1.0, relative_vel_mag / 25.0)

                # Check proximity interaction
                if dist <= self.proximity_threshold:
                    # Enclosing union bounding box
                    b1 = p1.current_bbox
                    b2 = p2.current_bbox
                    union_bbox = np.array([
                        min(b1[0], b2[0]),
                        min(b1[1], b2[1]),
                        max(b1[2], b2[2]),
                        max(b1[3], b2[3]),
                    ], dtype=np.float32)

                    midpoint = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
                    rel_speed = abs(p1.speed - p2.speed)

                    interacting_pairs.append(
                        InteractingPair(
                            track_id_1=p1.track_id,
                            track_id_2=p2.track_id,
                            distance=dist,
                            relative_speed=rel_speed,
                            convergence_rate=convergence_rate,
                            reciprocal_motion_score=reciprocal_score,
                            center_point=midpoint,
                            union_bbox=union_bbox,
                        )
                    )

        # Clean stale pair histories
        self._pair_distance_history = {
            k: v for k, v in self._pair_distance_history.items() if k in active_pair_keys
        }

        # 2. Compute Abnormal Movement Score [0.0 - 1.0]
        # Combines speed outliers, acceleration spikes, and reciprocal interactions
        speed_score = min(1.0, max_speed / (self.speed_threshold * 1.5))
        accel_score = min(1.0, max_accel / (self.accel_threshold * 1.5))
        pair_score = max([p.reciprocal_motion_score for p in interacting_pairs], default=0.0)

        abnormal_movement_score = round(
            0.35 * speed_score + 0.35 * accel_score + 0.30 * pair_score,
            3,
        )

        return KinematicFeatures(
            active_track_count=track_count,
            mean_speed=round(mean_speed, 2),
            max_individual_speed=round(max_speed, 2),
            max_individual_accel=round(max_accel, 2),
            interacting_pairs=interacting_pairs,
            abnormal_movement_score=abnormal_movement_score,
            timestamp=curr_time,
        )
