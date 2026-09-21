"""
Real-Time Crowd Anomaly Detection - Phase 10: Monitoring Dashboard & Operator Interface.

Exposes:
    - DashboardSnapshot: Unified telemetry record from Phases 2-9
    - DashboardDataProvider: Adapter converting subsystem outputs into snapshots
    - DashboardState: In-memory bounded rolling state & alert lifecycle
    - DashboardMetrics: Telemetry metrics engine (FPS, latency, rolling statistics)
    - DashboardTheme: Surveillance dark theme visual tokens and badges
"""

from src.dashboard.dashboard_data import DashboardDataProvider, DashboardSnapshot
from src.dashboard.dashboard_metrics import DashboardMetrics
from src.dashboard.dashboard_state import DashboardState
from src.dashboard.dashboard_theme import DashboardTheme

__all__ = [
    "DashboardSnapshot",
    "DashboardDataProvider",
    "DashboardState",
    "DashboardMetrics",
    "DashboardTheme",
]
