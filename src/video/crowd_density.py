"""
Crowd Density Estimation Module.
Calculates global crowd density (persons/m^2), localized spatial grid densities,
identifies clustering hotspots, and generates visual density heatmaps.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("crowd_density")


@dataclass
class DensityResult:
    """
    Encapsulates crowd density metrics for a single frame.
    """
    person_count: int
    density_per_m2: float               # Global density: count / frame_area_m2
    density_score: float                # Normalized anomaly indicator [0.0 - 1.0]
    density_level: str                  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    grid_counts: np.ndarray             # (grid_rows, grid_cols) integer counts
    grid_densities: np.ndarray          # (grid_rows, grid_cols) density per cell
    hotspot_cells: List[Tuple[int, int]]# List of (row, col) indices exceeding high threshold
    is_overcrowded: bool                # Flag indicating overcrowding emergency condition


class CrowdDensityEstimator:
    """
    Computes global and spatial-grid crowd density from person detection centroids.
    Splits the monitored camera plane into an M x N spatial grid to detect local bottlenecks.
    """

    def __init__(
        self,
        frame_area_m2: float = 50.0,
        grid_rows: int = 3,
        grid_cols: int = 3,
        threshold_low: float = 0.3,
        threshold_moderate: float = 0.8,
        threshold_high: float = 1.5,
    ):
        self.frame_area_m2 = max(1.0, float(frame_area_m2))
        self.grid_rows = max(1, int(grid_rows))
        self.grid_cols = max(1, int(grid_cols))
        self.threshold_low = float(threshold_low)
        self.threshold_moderate = float(threshold_moderate)
        self.threshold_high = float(threshold_high)

        # Area of a single grid cell in square meters
        self.cell_area_m2 = self.frame_area_m2 / (self.grid_rows * self.grid_cols)

    def estimate(self, detection_points: np.ndarray, frame_width: int, frame_height: int) -> DensityResult:
        """
        Calculates density metrics from detected person ground contact or center points.

        Args:
            detection_points: (N, 2) array of coordinates (x, y). Ground feet points are preferred.
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.

        Returns:
            DensityResult dataclass.
        """
        person_count = len(detection_points)
        density_per_m2 = person_count / self.frame_area_m2

        # Compute normalized density score [0.0, 1.0]
        # At threshold_high (e.g. 1.5 persons/m^2), score reaches 1.0
        density_score = min(1.0, density_per_m2 / self.threshold_high)

        # Determine qualitative category
        if density_per_m2 < self.threshold_low:
            density_level = "LOW"
        elif density_per_m2 < self.threshold_moderate:
            density_level = "MODERATE"
        elif density_per_m2 < self.threshold_high:
            density_level = "HIGH"
        else:
            density_level = "CRITICAL"

        # Compute spatial grid distribution
        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

        if person_count > 0 and frame_width > 0 and frame_height > 0:
            cell_w = frame_width / self.grid_cols
            cell_h = frame_height / self.grid_rows

            for pt in detection_points:
                px, py = pt[0], pt[1]
                col_idx = int(np.clip(px // cell_w, 0, self.grid_cols - 1))
                row_idx = int(np.clip(py // cell_h, 0, self.grid_rows - 1))
                grid_counts[row_idx, col_idx] += 1

        grid_densities = grid_counts / self.cell_area_m2

        # Identify cells exceeding high threshold
        hotspot_cells = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if grid_densities[r, c] >= self.threshold_high:
                    hotspot_cells.append((r, c))

        is_overcrowded = (density_level == "CRITICAL") or (len(hotspot_cells) > 0)

        return DensityResult(
            person_count=person_count,
            density_per_m2=density_per_m2,
            density_score=density_score,
            density_level=density_level,
            grid_counts=grid_counts,
            grid_densities=grid_densities,
            hotspot_cells=hotspot_cells,
            is_overcrowded=is_overcrowded,
        )

    def draw_density_overlay(
        self,
        frame: np.ndarray,
        density_result: DensityResult,
        show_grid: bool = True,
        alpha: float = 0.30,
    ) -> np.ndarray:
        """
        Draws semi-transparent spatial density heatmap and grid cell boundaries onto the frame.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cell_w = w / self.grid_cols
        cell_h = h / self.grid_rows

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)

                cell_density = density_result.grid_densities[r, c]
                cell_count = density_result.grid_counts[r, c]

                # Choose cell color based on density level
                if cell_density >= self.threshold_high:
                    color = (0, 0, 220)      # Red (Critical Hotspot)
                elif cell_density >= self.threshold_moderate:
                    color = (0, 140, 255)    # Orange (High)
                elif cell_density >= self.threshold_low:
                    color = (0, 215, 255)    # Yellow (Moderate)
                else:
                    color = (0, 180, 0)      # Green (Low)

                # Fill cell on overlay
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

                if show_grid:
                    # Cell boundary
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 220, 220), 1)

                    # Text label: count & density
                    text = f"{cell_count}p ({cell_density:.1f}/m2)"
                    cv2.putText(
                        overlay,
                        text,
                        (x1 + 10, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # Blend overlay with original frame
        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
