"""
Crowd Behavior Feature Extraction Module.
Calculates macroscopic crowd dynamics: velocity variance, directional entropy,
crowd dispersion (scattering), crowd convergence (gathering), and rapid movement ratios.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.video.crowd_tracker import TrackedPerson
from src.video.crowd_density import DensityResult
from src.utils.logger import setup_logger

logger = setup_logger("behavior_features")


@dataclass
class CrowdBehaviorMetrics:
    """
    Comprehensive crowd-level kinematic and behavioral metrics for a video frame.
    """
    active_people: int
    mean_velocity: float                      # Average speed across all tracked pedestrians
    velocity_variance: float                  # Speed variance
    direction_variance: float                 # Circular variance of headings [0.0 - 1.0]
    rapid_mover_ratio: float                  # Fraction of crowd moving above speed threshold [0.0 - 1.0]
    direction_change_ratio: float             # Fraction of crowd exhibiting sudden heading shifts
    dispersion_rate: float                    # Positive when crowd is scattering outward (pixels/f)
    convergence_rate: float                   # Positive when crowd is gathering inward (pixels/f)
    kinematic_score: float                    # Normalized kinematic anomaly score [0.0 - 1.0]
    candidate_class: str                      # "normal", "rapid_movement", "scattering", "crowd_gathering", "abnormal_behavior"
    description: str


class CrowdBehaviorAnalyzer:
    """
    Extracts high-level behavioral dynamics across all tracked individuals:
    - Identifies panic scattering vs. laminar pedestrian transit
    - Detects crowd convergence (mobbing / gathering)
    - Quantifies directional chaos via circular variance
    """

    def __init__(
        self,
        speed_threshold: float = 18.0,
        dispersion_threshold: float = 2.5,
        convergence_threshold: float = 2.5,
    ):
        self.speed_threshold = speed_threshold
        self.dispersion_threshold = dispersion_threshold
        self.convergence_threshold = convergence_threshold

        # Historical pairwise mean distance for dispersion rate calculation
        self._prev_mean_distance: Optional[float] = None

    def analyze(
        self,
        tracked_persons: List[TrackedPerson],
        density_result: Optional[DensityResult] = None,
    ) -> CrowdBehaviorMetrics:
        """
        Analyzes crowd-level kinematics and classifies behavior.

        Args:
            tracked_persons: List of active TrackedPerson instances.
            density_result: Optional crowd density information.

        Returns:
            CrowdBehaviorMetrics dataclass.
        """
        n_people = len(tracked_persons)

        if n_people == 0:
            return CrowdBehaviorMetrics(
                active_people=0,
                mean_velocity=0.0,
                velocity_variance=0.0,
                direction_variance=0.0,
                rapid_mover_ratio=0.0,
                direction_change_ratio=0.0,
                dispersion_rate=0.0,
                convergence_rate=0.0,
                kinematic_score=0.0,
                candidate_class="normal",
                description="No active individuals tracked.",
            )

        speeds = np.array([t.speed for t in tracked_persons], dtype=np.float32)
        mean_vel = float(np.mean(speeds))
        vel_var = float(np.var(speeds))

        # 1. Directional Circular Variance
        # When all individuals head the same direction (laminar flow), circular variance ~ 0.
        # When individuals run in conflicting/opposing directions (panic/chaos), variance ~ 1.
        angles_rad = [math.radians(t.direction_angle) for t in tracked_persons if t.speed > 1.0]
        if len(angles_rad) >= 2:
            sin_sum = sum(math.sin(a) for a in angles_rad)
            cos_sum = sum(math.cos(a) for a in angles_rad)
            r = math.hypot(cos_sum, sin_sum) / len(angles_rad)
            dir_var = max(0.0, min(1.0, 1.0 - r))
        else:
            dir_var = 0.0

        # 2. Rapid Mover Ratio
        rapid_count = sum(1 for s in speeds if s >= self.speed_threshold)
        rapid_ratio = rapid_count / n_people

        # 3. Direction Change Ratio (acceleration outliers)
        accels = [t.acceleration for t in tracked_persons]
        accel_count = sum(1 for a in accels if a >= 6.0)
        direction_change_ratio = accel_count / n_people

        # 4. Crowd Dispersion and Convergence Rates
        # Average pairwise distance between all centroids
        centroids = [t.history[-1] for t in tracked_persons if t.history]
        dispersion_rate = 0.0
        convergence_rate = 0.0

        if len(centroids) >= 2:
            pair_dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
                    pair_dists.append(dist)

            current_mean_dist = float(np.mean(pair_dists))

            if self._prev_mean_distance is not None:
                delta_dist = current_mean_dist - self._prev_mean_distance
                if delta_dist > 0:
                    dispersion_rate = delta_dist
                else:
                    convergence_rate = abs(delta_dist)

            self._prev_mean_distance = current_mean_dist
        else:
            self._prev_mean_distance = None

        # 5. Compute Kinematic Behavior Score [0.0 - 1.0]
        speed_norm = min(1.0, mean_vel / 22.0)
        rapid_norm = rapid_ratio
        dir_norm = dir_var
        disp_norm = min(1.0, dispersion_rate / 6.0)
        conv_norm = min(1.0, convergence_rate / 6.0)

        kinematic_score = round(
            0.25 * speed_norm +
            0.25 * rapid_norm +
            0.20 * dir_norm +
            0.15 * disp_norm +
            0.15 * conv_norm,
            3,
        )

        # 6. Classify Behavioral Phenotype
        if dispersion_rate >= self.dispersion_threshold and rapid_ratio >= 0.35:
            candidate_class = "scattering"
            description = f"Rapid crowd dispersion detected ({dispersion_rate:.1f}px/f expansion, {int(rapid_ratio * 100)}% rapid movers)."
        elif convergence_rate >= self.convergence_threshold and n_people >= 3:
            candidate_class = "crowd_gathering"
            description = f"Sudden crowd convergence detected ({convergence_rate:.1f}px/f contraction towards centroid)."
        elif rapid_ratio >= 0.50:
            candidate_class = "rapid_movement"
            description = f"High proportion of rapid crowd motion ({int(rapid_ratio * 100)}% exceeding threshold)."
        elif dir_var >= 0.70 and mean_vel >= 10.0:
            candidate_class = "abnormal_behavior"
            description = f"Disorganized directional movement with high heading variance ({dir_var:.2f})."
        elif kinematic_score >= 0.50:
            candidate_class = "abnormal_behavior"
            description = f"Elevated kinematic score ({kinematic_score:.2f}) from combined motion factors."
        else:
            candidate_class = "normal"
            description = "Standard crowd motion within nominal velocity thresholds."

        return CrowdBehaviorMetrics(
            active_people=n_people,
            mean_velocity=round(mean_vel, 2),
            velocity_variance=round(vel_var, 2),
            direction_variance=round(dir_var, 3),
            rapid_mover_ratio=round(rapid_ratio, 2),
            direction_change_ratio=round(direction_change_ratio, 2),
            dispersion_rate=round(dispersion_rate, 2),
            convergence_rate=round(convergence_rate, 2),
            kinematic_score=kinematic_score,
            candidate_class=candidate_class,
            description=description,
        )
