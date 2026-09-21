"""
Phase 11 Benchmark: SQLite Database Persistence Latency, Throughput & Scaling.
"""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager
from src.database.models import (
    AlertRecord,
    EventRecord,
    EvidenceRecord,
    SessionRecord,
    SystemMetricRecord,
)
from src.database.repositories import HistoricalQueryAPI


def run_benchmark():
    print("=" * 70)
    print("PHASE 11 PERFORMANCE BENCHMARK: SQLITE DATABASE PERSISTENCE LAYER")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "benchmark_crowd.db")
        db = DatabaseManager(
            db_path=db_path,
            wal_mode=True,
            foreign_keys=True,
            busy_timeout_ms=5000,
            auto_create=True,
        )
        api = HistoricalQueryAPI(db)

        # 1. Create Session
        sess = SessionRecord(
            session_id="SESS-BENCH-001",
            started_at=datetime.now(timezone.utc).isoformat(),
            source_type="synthetic",
            video_source="benchmark_stream",
            status="ACTIVE",
        )
        api.sessions.create_session(sess)

        # 2. Benchmark Event Insert Latency (Individual Inserts)
        num_events = 1000
        t_base = time.time()
        start_ins = time.perf_counter()

        for i in range(num_events):
            t_iso = datetime.fromtimestamp(t_base + i * 0.1, tz=timezone.utc).isoformat()
            state = "NORMAL" if i % 10 != 0 else ("HIGH_RISK" if i % 50 != 0 else "EMERGENCY")
            evt = EventRecord(
                event_id=f"EVT-B-{i:05d}",
                session_id=sess.session_id,
                timestamp=t_iso,
                risk_state=state,
                anomaly_score=0.1 + (i % 10) * 0.08,
                decision_confidence=0.92,
                alert_severity="INFO" if state == "NORMAL" else "CRITICAL",
                dominant_modality="video",
                people_count=10 + (i % 20),
                crowd_density=0.2 + (i % 10) * 0.05,
                average_speed=4.0 + (i % 5) * 2.0,
                video_score=0.15,
                audio_score=0.08,
                fusion_score=0.12,
                corroborated=(state == "EMERGENCY"),
                reason="Benchmark event evaluation",
            )
            api.events.insert_event(evt)

        total_ins_time = time.perf_counter() - start_ins
        avg_ins_latency_ms = (total_ins_time / num_events) * 1000.0
        ins_throughput = num_events / total_ins_time

        print(f"Events Inserted           : {num_events}")
        print(f"Total Insert Time         : {total_ins_time:.4f} s")
        print(f"Mean Event Insert Latency : {avg_ins_latency_ms:.4f} ms (Target: < 5.0 ms in WAL mode)")
        print(f"Event Insert Throughput   : {ins_throughput:.1f} events/sec")

        # 3. Benchmark Alert Insertion and Status Update Latency
        num_alerts = 100
        start_alt = time.perf_counter()
        for i in range(num_alerts):
            aid = f"ALT-B-{i:04d}"
            eid = f"EVT-B-{(i * 10):05d}"
            alt = AlertRecord(
                alert_id=aid,
                event_id=eid,
                session_id=sess.session_id,
                severity="WARNING" if i % 2 == 0 else "CRITICAL",
                risk_state="HIGH_RISK" if i % 2 == 0 else "EMERGENCY",
                score=0.85,
                status="ACTIVE",
            )
            api.alerts.insert_alert(alt)
            # Acknowledge
            api.alerts.update_alert_status(aid, status="ACKNOWLEDGED")
            # Resolve
            api.alerts.update_alert_status(aid, status="RESOLVED")

        total_alt_time = time.perf_counter() - start_alt
        avg_alt_lifecycle_ms = (total_alt_time / num_alerts) * 1000.0

        print(f"Alerts Processed (Full LC): {num_alerts}")
        print(f"Alert Lifecycle Latency   : {avg_alt_lifecycle_ms:.4f} ms per (Insert + Ack + Resolve)")

        # 4. Benchmark Evidence Batch Insert
        evidence_items = []
        for i in range(500):
            evidence_items.append(
                EvidenceRecord(
                    evidence_id=f"EVI-B-{i:05d}",
                    event_id="EVT-B-00050",
                    alert_id="ALT-B-0005",
                    factor="weapon_presence",
                    contribution=0.35,
                    rank=1,
                    statement="Synthetic firearm detection corroboration",
                )
            )
        start_evi = time.perf_counter()
        api.evidence.insert_evidence_batch(evidence_items)
        evi_time = time.perf_counter() - start_evi
        print(f"Evidence Batch (500 items): {evi_time*1000.0:.2f} ms ({500/evi_time:.1f} items/sec)")

        # 5. Benchmark Query Latencies
        # Recent events query
        start_q1 = time.perf_counter()
        for _ in range(200):
            _ = api.events.get_recent_events(limit=50)
        q1_time = (time.perf_counter() - start_q1) / 200 * 1000.0
        print(f"Recent Events Query (50)  : {q1_time:.4f} ms per query (Target: < 2.0 ms)")

        # Query by state
        start_q2 = time.perf_counter()
        for _ in range(200):
            _ = api.events.query_events_by_state("EMERGENCY", limit=50)
        q2_time = (time.perf_counter() - start_q2) / 200 * 1000.0
        print(f"Query by State (Indexed)  : {q2_time:.4f} ms per query (Target: < 2.0 ms)")

        # 6. Benchmark Summary Statistics Aggregation
        start_stats = time.perf_counter()
        for _ in range(100):
            stats = api.get_statistics()
        stats_time = (time.perf_counter() - start_stats) / 100 * 1000.0
        print(f"Statistics Aggregation    : {stats_time:.4f} ms per computation")
        print(f"   -> Events: {stats['total_events']}, Emergencies: {stats['emergency_count']}, Avg Score: {stats['average_anomaly_score']}")

        # 7. Database File Size Footprint
        db_size_bytes = os.path.getsize(db_path)
        wal_file = db_path + "-wal"
        wal_size_bytes = os.path.getsize(wal_file) if os.path.exists(wal_file) else 0
        total_kb = (db_size_bytes + wal_size_bytes) / 1024.0

        print(f"Database On-Disk Footprint: {total_kb:.2f} KB (DB: {db_size_bytes/1024.0:.1f} KB, WAL: {wal_size_bytes/1024.0:.1f} KB)")
        print("=" * 70)

        # Assertions
        assert avg_ins_latency_ms < 15.0, f"Insert latency {avg_ins_latency_ms} ms exceeds 15.0 ms threshold!"
        assert q1_time < 5.0, f"Query latency {q1_time} ms exceeds 5.0 ms threshold!"
        assert stats["total_events"] == num_events, "Total events mismatch!"
        print("ALL PHASE 11 DATABASE PERFORMANCE BENCHMARKS PASSED.")
        print("=" * 70)

        db.close()


if __name__ == "__main__":
    run_benchmark()
