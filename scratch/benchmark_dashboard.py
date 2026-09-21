"""
Phase 10 Benchmark: DashboardDataProvider Throughput & Memory Bounds.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.dashboard_data import DashboardDataProvider
from src.dashboard.dashboard_state import DashboardState
from src.dashboard.dashboard_metrics import DashboardMetrics

def run_benchmark():
    provider = DashboardDataProvider()
    state = DashboardState(max_history=120)

    print("=" * 60)
    print("PHASE 10 PERFORMANCE BENCHMARK: DASHBOARD DATA & STATE")
    print("=" * 60)

    # 1. Benchmark Snapshot Generation
    num_iterations = 2000
    start_t = time.perf_counter()
    scenarios = ["NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY", "RECOVERY"]

    for i in range(num_iterations):
        scen = scenarios[i % len(scenarios)]
        snap = provider.create_demo_snapshot(scenario=scen, step=i)
        state.add_snapshot(snap)

    total_time = time.perf_counter() - start_t
    avg_latency_ms = (total_time / num_iterations) * 1000.0
    throughput_hz = num_iterations / total_time

    print(f"Iterations Processed      : {num_iterations}")
    print(f"Total Execution Time      : {total_time:.4f} s")
    print(f"Per-Snapshot Latency      : {avg_latency_ms:.4f} ms (Target: < 2.0 ms)")
    print(f"Transformation Throughput : {throughput_hz:.1f} snapshots/sec")
    print(f"Bounded History Length    : {len(state.history_timestamps)} (Max: {state.max_history})")
    print(f"Recent Alerts Logged      : {len(state.recent_alerts)} (Max: {state.max_recent_alerts})")

    # 2. Benchmark Metrics Computation
    start_m = time.perf_counter()
    for _ in range(500):
        summary = DashboardMetrics.compute_summary(state)
    m_time = time.perf_counter() - start_m
    m_latency_ms = (m_time / 500) * 1000.0

    print(f"Metrics Engine Latency    : {m_latency_ms:.4f} ms per evaluation")
    print("=" * 60)

    assert avg_latency_ms < 2.0, f"Latency {avg_latency_ms} ms exceeds 2.0 ms threshold!"
    assert len(state.history_timestamps) <= 120, "History queue exceeded maxlen!"
    print("ALL PERFORMANCE BENCHMARKS PASSED.")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
