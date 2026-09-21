"""
==============================================================================
Alert Deduplication and Event Fingerprinting (Phase 8).
Prevents identical surveillance events from emitting repeated alerts within
a rolling duplicate window while distinguishing materially different evidence.
==============================================================================
"""

import hashlib
import time
from typing import Any, Dict, Optional, Union

from src.alerts.alert_record import AlertSeverity
from src.decision.decision_record import DecisionEvent
from src.utils.logger import setup_logger

logger = setup_logger("alert_deduplication")


class AlertDeduplicator:
    """
    Computes deterministic signatures from decision event features and suppresses
    duplicate alerts occurring within a configurable temporal window.
    """

    def __init__(self, duplicate_window_seconds: float = 15.0):
        self.duplicate_window_seconds = max(1.0, float(duplicate_window_seconds))
        # Maps event_signature -> last_seen_epoch_timestamp
        self._recent_signatures: Dict[str, float] = {}

        logger.info(
            f"AlertDeduplicator initialized: duplicate_window={self.duplicate_window_seconds}s."
        )

    def generate_signature(
        self,
        event: Union[DecisionEvent, Dict[str, Any]],
        severity: AlertSeverity,
    ) -> str:
        """
        Generates a deterministic SHA-256 fingerprint for a surveillance event.
        Combines operational risk state, dominant sensory modality, normalized reason category,
        critical evidence flag, and severity tier.

        NOTE: Explicitly excludes exact timestamps so temporal proximity within the
        duplicate window can be evaluated independently.
        """
        if isinstance(event, DecisionEvent):
            risk_state = str(event.current_state).strip().upper()
            dominant_mod = str(event.dominant_modality).strip().lower()
            reason = str(event.reason).strip().lower()
            critical = bool(event.critical_evidence)
            has_weapon = bool(event.weapon_score >= 0.50)
            has_fight = bool(event.fight_score >= 0.50)
            corroborated = bool(event.corroborated)
        else:
            risk_state = str(event.get("current_state", "")).strip().upper()
            dominant_mod = str(event.get("dominant_modality", "none")).strip().lower()
            reason = str(event.get("reason", "")).strip().lower()
            critical = bool(event.get("critical_evidence", False))
            has_weapon = bool(float(event.get("weapon_score", 0.0)) >= 0.50)
            has_fight = bool(float(event.get("fight_score", 0.0)) >= 0.50)
            corroborated = bool(event.get("corroborated", False))

        # Extract normalized reason keyword to cluster semantically identical reasons
        reason_cat = "general"
        if "weapon" in reason or "dangerous-object" in reason or has_weapon:
            reason_cat = "weapon"
        elif "altercation" in reason or "struggle" in reason or "fight" in reason or has_fight:
            reason_cat = "fight"
        elif "scattering" in reason or "dispersion" in reason:
            reason_cat = "scattering"
        elif "acoustic" in reason or "shouting" in reason or "scream" in reason:
            reason_cat = "acoustic"
        elif "density" in reason or "overcrowding" in reason:
            reason_cat = "density"

        raw_key = (
            f"state={risk_state}|"
            f"sev={severity.value}|"
            f"mod={dominant_mod}|"
            f"cat={reason_cat}|"
            f"crit={critical}|"
            f"corrob={corroborated}"
        )

        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def is_duplicate(
        self,
        signature: str,
        current_time: Optional[float] = None,
    ) -> bool:
        """
        Checks whether the signature was already processed within the duplicate window.

        Args:
            signature: Deterministic event signature.
            current_time: Epoch timestamp.

        Returns:
            True if duplicate should be suppressed, False otherwise.
        """
        curr_t = time.time() if current_time is None else float(current_time)

        # Self-prune expired signatures older than 2x the duplicate window
        self._prune_stale_signatures(curr_t)

        last_t = self._recent_signatures.get(signature)
        if last_t is not None:
            time_delta = curr_t - last_t
            if time_delta < self.duplicate_window_seconds:
                logger.debug(
                    f"Duplicate alert suppressed for signature [{signature}] "
                    f"({time_delta:.2f}s < {self.duplicate_window_seconds}s)."
                )
                return True

        return False

    def record_signature(
        self,
        signature: str,
        current_time: Optional[float] = None,
    ) -> None:
        """Records an active signature timestamp."""
        curr_t = time.time() if current_time is None else float(current_time)
        self._recent_signatures[signature] = curr_t

    def _prune_stale_signatures(self, current_time: float) -> None:
        """Removes expired signatures to maintain bounded memory footprint."""
        expiration_horizon = self.duplicate_window_seconds * 2.0
        stale_keys = [
            k for k, t in self._recent_signatures.items()
            if (current_time - t) > expiration_horizon
        ]
        for k in stale_keys:
            del self._recent_signatures[k]

    def reset(self) -> None:
        """Clears all cached duplicate signatures."""
        self._recent_signatures.clear()
