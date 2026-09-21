"""
Dashboard Metrics Engine - Phase 10.

Computes operational metrics (FPS, pipeline latency, alert statistics,
peak crowd densities, rolling averages) for surveillance dashboard telemetry.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class DashboardMetrics:
    """
    Computes summary telemetry and real-time operational statistics
    from bounded historical data.
    """

    @staticmethod
    def calculate_fps(timestamps: List[float], window: int = 30) -> float:
        """
        Calculate rolling frames-per-second based on timestamp deltas.

        Returns 0.0 if fewer than 2 timestamps are available.
        """
        if not timestamps or len(timestamps) < 2:
            return 0.0

        recent = timestamps[-window:] if len(timestamps) > window else timestamps
        delta_t = recent[-1] - recent[0]
        if delta_t <= 0.0001:
            return 0.0

        return float(round((len(recent) - 1) / delta_t, 2))

    @staticmethod
    def calculate_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
        """
        Compute mean, min, max, and p95 latency statistics.
        """
        if not latencies_ms:
            return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}

        arr = np.array(latencies_ms, dtype=np.float64)
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
        }

    @staticmethod
    def calculate_score_stats(scores: List[float]) -> Dict[str, float]:
        """
        Compute rolling average, peak, and minimum anomaly scores.
        """
        if not scores:
            return {"current": 0.0, "mean": 0.0, "peak": 0.0, "min": 0.0}

        arr = np.array(scores, dtype=np.float64)
        return {
            "current": round(float(arr[-1]), 3),
            "mean": round(float(np.mean(arr)), 3),
            "peak": round(float(np.max(arr)), 3),
            "min": round(float(np.min(arr)), 3),
        }

    @classmethod
    def compute_summary(cls, state: Any) -> Dict[str, Any]:
        """
        Compile full telemetry metrics summary from a DashboardState instance.
        """
        chart_data = state.get_chart_data()
        timestamps = chart_data.get("timestamps", [])
        scores = chart_data.get("anomaly_scores", [])
        latencies = chart_data.get("latencies_ms", [])
        counts = chart_data.get("crowd_counts", [])
        densities = chart_data.get("crowd_densities", [])

        fps = cls.calculate_fps(timestamps)
        latency_stats = cls.calculate_latency_stats(latencies)
        score_stats = cls.calculate_score_stats(scores)

        peak_crowd_count = max(counts) if counts else 0
        peak_crowd_density = max(densities) if densities else 0.0

        total_alerts = len(state.recent_alerts)
        ack_alerts = len(state.acknowledged_alert_ids)
        res_alerts = len(state.resolved_alert_ids)
        active_count = 1 if state.active_alert is not None else 0

        return {
            "fps": fps,
            "latency": latency_stats,
            "anomaly": score_stats,
            "peak_crowd_count": peak_crowd_count,
            "peak_crowd_density": round(peak_crowd_density, 3),
            "total_alerts": total_alerts,
            "active_alerts": active_count,
            "acknowledged_alerts": ack_alerts,
            "resolved_alerts": res_alerts,
            "unresolved_alerts": max(0, total_alerts - res_alerts),
            "history_frames": len(timestamps),
        }
