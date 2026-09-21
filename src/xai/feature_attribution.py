"""
==============================================================================
Feature Attribution and Evidence Contribution Engine (Phase 9).
Calculates deterministic, normalized relative contributions across observable
sensory factors without fabricating pseudo-SHAP values or unverified saliency.
==============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple


class FeatureAttributor:
    """
    Computes deterministic evidence contributions for surveillance anomaly events.
    
    Attributes relative weights across 7 observable factors:
    1. Crowd Density (spatial packing)
    2. Rapid Movement (kinematic velocity / panic acceleration)
    3. Behavioral Anomaly (dispersion, directional divergence)
    4. Acoustic Anomaly (sound energy, distress cues)
    5. Physical Struggle (fight dynamics)
    6. Weapon Presence (dangerous handheld objects)
    7. Cross-Modal Corroboration (multi-sensor synergy amplification)
    
    All evidence contributions are normalized to [0.0 - 1.0] and sum to 1.0
    (or 0.0 under nominal baseline conditions).
    """

    DEFAULT_BASE_WEIGHTS = {
        "crowd_density": 0.15,
        "rapid_movement": 0.20,
        "behavioral_anomaly": 0.25,
        "acoustic_anomaly": 0.30,
        "physical_struggle": 0.35,
        "weapon_presence": 0.40,
        "cross_modal_corroboration": 0.25,
    }

    # Canonical priority for deterministic tie-breaking
    PRIORITY_ORDER = [
        "weapon_presence",
        "physical_struggle",
        "cross_modal_corroboration",
        "behavioral_anomaly",
        "rapid_movement",
        "acoustic_anomaly",
        "crowd_density",
    ]

    def __init__(self, base_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the FeatureAttributor with optional custom base factor weights.
        """
        self.weights = dict(self.DEFAULT_BASE_WEIGHTS)
        if base_weights:
            self.weights.update(base_weights)

    def compute_contributions(
        self,
        density_score: float = 0.0,
        movement_score: float = 0.0,
        behavior_score: float = 0.0,
        audio_score: float = 0.0,
        fight_score: float = 0.0,
        weapon_score: float = 0.0,
        corroborated: bool = False,
    ) -> Dict[str, float]:
        """
        Computes normalized relative evidence contributions for observable factors.

        Args:
            density_score: Scaled crowd density factor [0.0 - 1.0]
            movement_score: Normalized crowd movement/speed score [0.0 - 1.0]
            behavior_score: Temporal behavioral anomaly score [0.0 - 1.0]
            audio_score: Acoustic anomaly score [0.0 - 1.0]
            fight_score: Altercation detection score [0.0 - 1.0]
            weapon_score: Dangerous object detection score [0.0 - 1.0]
            corroborated: True if audio-visual synergy was confirmed

        Returns:
            Dict mapping factor name to normalized contribution score in [0.0 - 1.0].
            Sum of contributions is 1.0 (or 0.0 if all factor inputs are 0.0).
        """
        corrob_val = 1.0 if corroborated else 0.0

        raw_scores = {
            "crowd_density": max(0.0, float(density_score)),
            "rapid_movement": max(0.0, float(movement_score)),
            "behavioral_anomaly": max(0.0, float(behavior_score)),
            "acoustic_anomaly": max(0.0, float(audio_score)),
            "physical_struggle": max(0.0, float(fight_score)),
            "weapon_presence": max(0.0, float(weapon_score)),
            "cross_modal_corroboration": corrob_val,
        }

        # Calculate weighted contributions
        weighted = {
            factor: raw_scores[factor] * self.weights.get(factor, 0.20)
            for factor in raw_scores
        }

        total_weight = sum(weighted.values())

        if total_weight <= 1e-6:
            # Nominal calm baseline: all contributions zero
            return {k: 0.0 for k in weighted}

        # Normalize to sum to 1.0
        normalized = {
            factor: round(val / total_weight, 4)
            for factor, val in weighted.items()
        }

        # Ensure exact sum = 1.0 rounding consistency if positive
        current_sum = sum(normalized.values())
        if current_sum > 0 and abs(current_sum - 1.0) > 1e-5:
            # Adjust the max contributing factor slightly to guarantee exact 1.0
            max_key = max(normalized, key=lambda k: normalized[k])
            normalized[max_key] = round(normalized[max_key] + (1.0 - current_sum), 4)

        return normalized

    def rank_contributions(
        self,
        contributions: Dict[str, float],
        top_k: int = 5,
        min_threshold: float = 0.001,
    ) -> List[Tuple[str, float]]:
        """
        Ranks contributing factors in descending order of relative evidence contribution.
        Deterministic tie-breaking uses predefined domain priority order.

        Args:
            contributions: Dict of factor -> contribution weight.
            top_k: Maximum number of top factors to return.
            min_threshold: Minimum contribution weight to include.

        Returns:
            List of (factor, weight) tuples ordered from most to least significant.
        """
        priority_map = {name: idx for idx, name in enumerate(self.PRIORITY_ORDER)}

        # Sort key: (-weight, priority_index, name)
        sorted_items = sorted(
            contributions.items(),
            key=lambda item: (
                -item[1],
                priority_map.get(item[0], 999),
                item[0],
            ),
        )

        filtered = [item for item in sorted_items if item[1] >= min_threshold]
        return filtered[:top_k]
