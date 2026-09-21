"""
Performance and Timing Metrics Utilities.
Measures real-time FPS, frame latency, and module execution times.
"""

import time
from collections import deque
from typing import Dict, Optional


class FPSCounter:
    """
    Rolling window FPS calculator for smooth display and benchmarking.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.fps = 0.0

    def update(self) -> float:
        """Call on each processed frame to compute current FPS."""
        now = time.perf_counter()
        self.timestamps.append(now)

        if len(self.timestamps) > 1:
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration > 0:
                self.fps = (len(self.timestamps) - 1) / duration
            else:
                self.fps = 0.0
        return self.fps

    @property
    def current_fps(self) -> float:
        return self.fps


class LatencyTracker:
    """
    Measures latency of individual pipeline stages (ms).
    """

    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self._latencies: Dict[str, deque] = {}

    def start(self, stage: str) -> None:
        """Begin timing a stage."""
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """
        End timing a stage and return elapsed time in milliseconds.
        """
        start = self._start_times.pop(stage, None)
        if start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if stage not in self._latencies:
            self._latencies[stage] = deque(maxlen=60)
        self._latencies[stage].append(elapsed_ms)
        return elapsed_ms

    def get_average(self, stage: str) -> float:
        """Return rolling average latency in milliseconds for a stage."""
        window = self._latencies.get(stage)
        if not window:
            return 0.0
        return sum(window) / len(window)

    def summary(self) -> Dict[str, float]:
        """Return dictionary of average latencies for all tracked stages."""
        return {stage: self.get_average(stage) for stage in self._latencies}
