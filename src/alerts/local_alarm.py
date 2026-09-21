"""
==============================================================================
Local Audio Alarm Management (Phase 8).
Dispatches local audio alarms/sirens non-blockingly for high-severity alerts.
Engineered with fail-safe error isolation to guarantee surveillance continuity.
==============================================================================
"""

import os
import sys
import threading
from typing import Optional

from src.alerts.alert_record import AlertSeverity
from src.utils.logger import setup_logger

logger = setup_logger("local_alarm")


class LocalAlarmManager:
    """
    Manages optional local audible alarm dispatching (buzzers, beeps, or WAV playback).
    Executes non-blockingly via daemon threads with fail-safe error trapping.
    """

    def __init__(
        self,
        enabled: bool = False,
        emergency_sound_path: str = "",
        warning_sound_path: str = "",
    ):
        self.enabled = bool(enabled)
        self.emergency_sound_path = str(emergency_sound_path)
        self.warning_sound_path = str(warning_sound_path)
        self._is_playing = False
        self._lock = threading.Lock()

        # Check audio backend availability
        self._winsound_available = False
        if sys.platform == "win32":
            try:
                import winsound
                self._winsound_available = True
            except ImportError:
                self._winsound_available = False

        status_str = "ENABLED" if self.enabled else "DISABLED (default)"
        logger.info(
            f"LocalAlarmManager initialized: status={status_str}, "
            f"winsound={self._winsound_available}."
        )

    def trigger(self, severity: AlertSeverity) -> str:
        """
        Triggers an audible alarm appropriate for the alert severity level.

        Returns:
            Status string: "MUTED", "ALARM_TRIGGERED", or "DEVICE_UNAVAILABLE".
        """
        if not self.enabled:
            return "MUTED"

        # Only HIGH and CRITICAL trigger audible alarms
        if severity not in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
            return "MUTED"

        # Dispatch non-blocking playback thread
        t = threading.Thread(
            target=self._play_alarm_worker,
            args=(severity,),
            daemon=True,
            name=f"AlarmWorker-{severity.value}",
        )
        t.start()
        return "ALARM_TRIGGERED"

    def _play_alarm_worker(self, severity: AlertSeverity) -> None:
        """Background worker executing the sound playback safely."""
        with self._lock:
            try:
                sound_path = self.emergency_sound_path if severity == AlertSeverity.CRITICAL else self.warning_sound_path

                # Check if custom WAV file exists
                if sound_path and os.path.exists(sound_path) and self._winsound_available:
                    import winsound
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    logger.info(f"Local audio alarm playing from file: '{sound_path}'.")
                    return

                # Fallback: System tone generation on Windows
                if self._winsound_available:
                    import winsound
                    if severity == AlertSeverity.CRITICAL:
                        # High-pitch rapid pulse for EMERGENCY
                        for _ in range(3):
                            winsound.Beep(1200, 150)
                            winsound.Beep(800, 100)
                    elif severity == AlertSeverity.HIGH:
                        # Moderate tone for HIGH_RISK
                        winsound.Beep(750, 250)
                    logger.info(f"Local audio tone dispatched for {severity.value}.")
                else:
                    logger.debug(f"Local audio alarm requested for {severity.value} (simulated; no audio device).")

            except Exception as e:
                # Fail-safe: audio failure MUST never terminate surveillance pipeline
                logger.warning(f"Local alarm playback encountered safe non-fatal error: {e}")

    def stop(self) -> None:
        """Stops any active WAV playback if backend supports it."""
        if self._winsound_available:
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as e:
                logger.debug(f"Local alarm stop non-fatal notice: {e}")
