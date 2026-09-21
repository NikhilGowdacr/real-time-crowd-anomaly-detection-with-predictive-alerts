"""
Surveillance Monitoring Dashboard - Streamlit Application Entrypoint - Phase 10.

Real-Time Crowd Anomaly Detection with Predictive Alerts.
Provides an interactive operator terminal supporting live pipeline streaming,
operator alert lifecycle actions, and deterministic 5-scenario demo mode.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import streamlit as st

from src.dashboard.dashboard_components import (
    render_active_alert_card,
    render_alert_history_table,
    render_audio_panel,
    render_crowd_metrics,
    render_database_panel,
    render_fusion_panel,
    render_header,
    render_realtime_charts,
    render_risk_banner,
    render_subsystem_matrix,
    render_video_panel,
    render_xai_panel,
)
from src.dashboard.dashboard_data import DashboardDataProvider, DashboardSnapshot
from src.dashboard.dashboard_metrics import DashboardMetrics
from src.dashboard.dashboard_state import DashboardState
from src.utils.config import load_config, resolve_path
from src.video.person_detector import YOLOPersonDetector
from src.video.crowd_density import CrowdDensityEstimator
from src.video.crowd_tracker import ByteTrackCrowdTracker
from src.safety.pose_analyzer import PoseMovementAnalyzer
from src.safety.fight_detector import FightDetector
from src.safety.weapon_detector import WeaponDetector
from src.safety.safety_fusion import SafetyFusionEngine
import cv2
import tempfile
import numpy as np
from src.database import (
    AlertRecord as DBAlertRecord,
    DatabaseManager,
    EventRecord as DBEventRecord,
    EvidenceRecord as DBEvidenceRecord,
    HistoricalQueryAPI,
    SessionRecord as DBSessionRecord,
)

# Page configuration
st.set_page_config(
    page_title="Crowd Anomaly Surveillance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_dashboard_state() -> DashboardState:
    """Retrieve or initialize DashboardState in Streamlit session state."""
    if "dashboard_state" not in st.session_state:
        st.session_state["dashboard_state"] = DashboardState(max_history=120)
    return st.session_state["dashboard_state"]


def get_data_provider() -> DashboardDataProvider:
    """Retrieve or initialize DashboardDataProvider in Streamlit session state."""
    if "data_provider" not in st.session_state:
        st.session_state["data_provider"] = DashboardDataProvider()
    return st.session_state["data_provider"]


def _seed_demo_database(api: HistoricalQueryAPI) -> None:
    """Populate temporary in-memory database with representative demo records."""
    t_now = datetime.now(timezone.utc)
    sess = DBSessionRecord(
        session_id="SESS-DEMO-SIM",
        started_at=(t_now - timedelta(minutes=15)).isoformat(),
        source_type="synthetic",
        video_source="synthetic_stream",
        audio_enabled=True,
        status="ACTIVE",
        metadata_json={"operator": "console", "scenario": "multi-scenario-evaluation"},
    )
    api.sessions.create_session(sess)

    scenarios = [
        ("EVT-DEMO-001", "NORMAL", 0.12, 0.95, "INFO", 14, 0.28, 4.2, 0.10, 0.05, "Normal baseline pedestrian movement"),
        ("EVT-DEMO-002", "SUSPICIOUS", 0.45, 0.88, "WARNING", 18, 0.42, 8.5, 0.48, 0.20, "Localized clustering with elevated speed variance"),
        ("EVT-DEMO-003", "HIGH_RISK", 0.72, 0.90, "HIGH", 24, 0.65, 14.1, 0.75, 0.55, "Physical altercation detected with rapid crowd dispersal"),
        ("EVT-DEMO-004", "EMERGENCY", 0.94, 0.96, "CRITICAL", 28, 0.82, 22.4, 0.96, 0.88, "Firearm detected with corroborating panic acoustic signature"),
        ("EVT-DEMO-005", "RECOVERY", 0.35, 0.85, "INFO", 12, 0.25, 5.0, 0.32, 0.15, "Perimeter secured; crowd returning to baseline dynamics"),
    ]

    for idx, (eid, state, score, conf, sev, cnt, dens, spd, v_sc, a_sc, rsn) in enumerate(scenarios):
        t_evt = (t_now - timedelta(minutes=14 - idx * 3)).isoformat()
        evt = DBEventRecord(
            event_id=eid,
            session_id="SESS-DEMO-SIM",
            timestamp=t_evt,
            risk_state=state,
            anomaly_score=score,
            decision_confidence=conf,
            alert_severity=sev,
            dominant_modality="video" if v_sc >= a_sc else "audio",
            people_count=cnt,
            crowd_density=dens,
            average_speed=spd,
            video_score=v_sc,
            audio_score=a_sc,
            fusion_score=score,
            corroborated=(state == "EMERGENCY"),
            critical_evidence=(state == "EMERGENCY"),
            confirmed=True,
            reason=rsn,
            created_at=t_evt,
            metadata_json={"scenario_index": idx, "demo": True},
        )
        api.events.insert_event(evt)

    # Sample alert
    alt = DBAlertRecord(
        alert_id="ALT-DEMO-CRIT",
        event_id="EVT-DEMO-004",
        session_id="SESS-DEMO-SIM",
        timestamp=(t_now - timedelta(minutes=5)).isoformat(),
        severity="CRITICAL",
        risk_state="EMERGENCY",
        score=0.94,
        confidence=0.96,
        reason="Firearm detected with corroborating panic acoustic signature",
        dominant_modality="bimodal",
        status="ACTIVE",
        created_at=(t_now - timedelta(minutes=5)).isoformat(),
        fingerprint="fp_demo_crit_001",
    )
    api.alerts.insert_alert(alt)

    # Sample evidence
    ev1 = DBEvidenceRecord(
        evidence_id="EVID-DEMO-001",
        event_id="EVT-DEMO-004",
        alert_id="ALT-DEMO-CRIT",
        timestamp=(t_now - timedelta(minutes=5)).isoformat(),
        factor="weapon_presence",
        contribution=0.45,
        rank=1,
        statement="Firearm presence detected (confidence: 0.95)",
        visual_available=True,
    )
    ev2 = DBEvidenceRecord(
        evidence_id="EVID-DEMO-002",
        event_id="EVT-DEMO-004",
        alert_id="ALT-DEMO-CRIT",
        timestamp=(t_now - timedelta(minutes=5)).isoformat(),
        factor="acoustic_scream",
        contribution=0.35,
        rank=2,
        statement="Acoustic scream signature corroborated (score: 0.88)",
        visual_available=True,
    )
    api.evidence.insert_evidence_batch([ev1, ev2])


def get_database_context(op_mode: str) -> Tuple[Optional[HistoricalQueryAPI], Optional[Dict[str, Any]]]:
    """
    Retrieve or initialize DatabaseManager and HistoricalQueryAPI.
    In Demo Simulation mode, if on-disk DB is absent, creates an isolated in-memory DB
    seeded with demo events so data/crowd_anomaly.db is never polluted.
    """
    key = f"db_api_{op_mode}"
    if key not in st.session_state:
        try:
            config = load_config()
            db_cfg = config.get("database", {})
            if not db_cfg.get("enabled", True):
                st.session_state[key] = (None, {"status": "OFFLINE", "reason": "Database persistence disabled in config"})
                return st.session_state[key]

            db_path = resolve_path(db_cfg.get("path", "data/crowd_anomaly.db"))
            if op_mode == "Demo Simulation":
                if not Path(db_path).exists():
                    db = DatabaseManager(db_path=":memory:", wal_mode=False)
                    api = HistoricalQueryAPI(db)
                    _seed_demo_database(api)
                    health = db.health_check()
                    st.session_state[key] = (api, health)
                    return st.session_state[key]

            db = DatabaseManager(
                db_path=db_path,
                wal_mode=db_cfg.get("wal_mode", True),
                foreign_keys=db_cfg.get("foreign_keys", True),
                busy_timeout_ms=db_cfg.get("busy_timeout_ms", 5000),
                auto_create=db_cfg.get("auto_create", True),
            )
            health = db.health_check()
            api = HistoricalQueryAPI(db)
            st.session_state[key] = (api, health)
        except Exception as exc:
            st.session_state[key] = (None, {"status": "OFFLINE", "error": str(exc)})
    return st.session_state[key]


@st.cache_resource
def get_video_analyzers(conf: float = 0.40):
    detector = YOLOPersonDetector(model_path="models/detection/yolov8n.pt", confidence=conf)
    tracker = ByteTrackCrowdTracker(history_length=30, speed_threshold=12.0)
    density = CrowdDensityEstimator(frame_area_m2=50.0)
    pose = PoseMovementAnalyzer(proximity_threshold=140.0, speed_threshold=10.0)
    fight = FightDetector(temporal_window=15, suspicious_threshold=0.35, emergency_threshold=0.60, confirmation_frames=2)
    weapon = WeaponDetector(model_path="models/detection/weapon_model.pt", confidence=conf)
    fusion = SafetyFusionEngine()
    return detector, tracker, density, pose, fight, weapon, fusion


def main() -> None:
    """Main dashboard application flow."""
    state = get_dashboard_state()
    provider = get_data_provider()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.title("🎛️ Operator Console")
        st.markdown("---")

        # Mode selection
        op_mode = st.radio(
            "Operating Mode",
            ["Demo Simulation", "Video File Analysis", "Live Pipeline"],
            index=0,
            help="Select Demo Simulation, Video File Analysis, or Live Pipeline.",
        )

        selected_scenario = "NORMAL"
        auto_cycle = False
        video_target_path = "data/videos/video.mp4"
        conf_val = 0.40

        if op_mode == "Demo Simulation":
            st.markdown("#### Scenario Simulation")
            selected_scenario = st.selectbox(
                "Trigger Scenario",
                ["NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY", "RECOVERY"],
                index=0,
                help="Inject deterministic telemetry snapshot corresponding to specified risk state.",
            )
            auto_cycle = st.checkbox("Auto-cycle scenarios", value=False, help="Automatically rotate scenarios every few seconds.")
        elif op_mode == "Video File Analysis":
            st.markdown("#### 🎬 Video Source")
            vid_source_type = st.selectbox(
                "Video Source",
                ["Project Video (data/videos/video.mp4)", "Custom File Path", "Upload Video File"],
                index=0,
            )
            if vid_source_type == "Upload Video File":
                uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
                if uploaded_file is not None:
                    upload_dir = Path("scratch/uploads")
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    target_file = upload_dir / uploaded_file.name
                    if not target_file.exists() or target_file.stat().st_size != uploaded_file.size:
                        target_file.write_bytes(uploaded_file.getvalue())
                    video_target_path = str(target_file.resolve())
                else:
                    video_target_path = str(resolve_path("data/videos/video.mp4"))
            elif vid_source_type == "Custom File Path":
                raw_path = st.text_input("Enter Path", value="data/videos/video.mp4")
                video_target_path = str(resolve_path(raw_path))
            else:
                video_target_path = str(resolve_path("data/videos/video.mp4"))

            conf_val = st.slider("Detection Confidence", min_value=0.20, max_value=0.85, value=0.40, step=0.05)

        st.markdown("---")
        st.markdown("#### Stream Controls")
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            pause_label = "▶ Resume" if state.paused else "⏸ Pause"
            if st.button(pause_label, use_container_width=True):
                state.paused = not state.paused
                st.rerun()

        with col_ctrl2:
            if st.button("🗑 Reset", use_container_width=True, help="Clear rolling telemetry buffer."):
                state.clear_history()
                st.rerun()

        refresh_rate = st.slider("Refresh Interval (s)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

        # Operational Summary Metrics
        st.markdown("---")
        st.markdown("#### System Health & Telemetry")
        summary = DashboardMetrics.compute_summary(state)
        st.write(f"**FPS:** {summary['fps']:.1f}")
        st.write(f"**Avg Latency:** {summary['latency']['mean_ms']:.1f} ms")
        st.write(f"**Active Alerts:** {summary['active_alerts']}")
        st.write(f"**Acknowledged:** {summary['acknowledged_alerts']}")
        st.write(f"**Resolved:** {summary['resolved_alerts']}")
        st.write(f"**History Depth:** {summary['history_frames']}/{state.max_history}")

        # Database Health Status
        db_api, db_health = get_database_context(op_mode)
        db_status = "🟢 ONLINE" if (db_health and db_health.get("status") == "ONLINE") else "⚪ OFFLINE"
        st.write(f"**Database:** {db_status}")

    # --- TELEMETRY INGESTION ---
    # Auto-cycling logic
    if auto_cycle and not state.paused:
        scenarios = ["NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY", "RECOVERY"]
        cycle_idx = int(time.time() // 4) % len(scenarios)
        selected_scenario = scenarios[cycle_idx]

    # Generate or obtain current snapshot
    if op_mode == "Demo Simulation":
        snapshot = provider.create_demo_snapshot(scenario=selected_scenario)
    elif op_mode == "Video File Analysis":
        # Check if we already have a previous frame or load initial frame
        cap_init = cv2.VideoCapture(str(video_target_path))
        if cap_init.isOpened():
            ret, frame = cap_init.read()
            cap_init.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                if w > 800:
                    scale = 800.0 / w
                    frame = cv2.resize(frame, (800, int(h * scale)))
                    h, w = frame.shape[:2]
                analyzers = get_video_analyzers(conf=conf_val)
                detector, tracker, density_est, pose_analyzer, fight_detector, weapon_detector, fusion_engine = analyzers
                det = detector.detect(frame)
                tracks = tracker.update(det, frame)
                dens = density_est.estimate(
                    detection_points=det.bottom_centers,
                    frame_width=w,
                    frame_height=h,
                )
                curr_ts = time.time()
                kinematics = pose_analyzer.extract_features(tracks, timestamp=curr_ts)
                fight_res = fight_detector.update(
                    kinematics=kinematics,
                    tracked_persons=tracks,
                    timestamp=curr_ts,
                )
                weapon_res = weapon_detector.detect(
                    frame=frame,
                    tracked_persons=tracks,
                    timestamp=curr_ts,
                )
                safety_res = fusion_engine.evaluate(
                    density_result=dens,
                    kinematics=kinematics,
                    fight_result=fight_res,
                    weapon_result=weapon_res,
                    timestamp=curr_ts,
                )
                annotated = frame.copy()
                annotated = tracker.draw_tracks(annotated, tracks, draw_trajectory=True)
                annotated = fight_detector.draw_fight_detections(annotated, fight_res)
                annotated = weapon_detector.draw_detections(annotated, weapon_res)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                snapshot = DashboardSnapshot(
                    timestamp=curr_ts,
                    video_frame=annotated_rgb,
                    frame=annotated_rgb,
                    people_count=len(tracks),
                    crowd_count=len(tracks),
                    crowd_density=dens.density_per_m2,
                    density_level=dens.density_level,
                    anomaly_score=safety_res.safety_score,
                    video_score=safety_res.safety_score,
                    risk_state=safety_res.status.value,
                    decision_confidence=0.92,
                    dominant_modality="video",
                    movement_speed=kinematics.mean_speed,
                )
            else:
                snapshot = provider.create_demo_snapshot(scenario="NORMAL")
        else:
            snapshot = provider.create_demo_snapshot(scenario="NORMAL")
    else:
        # Live pipeline mode fallback to provider snapshot if no pipeline attached
        snapshot = provider.create_demo_snapshot(scenario="NORMAL")

    # Add snapshot to rolling state
    state.add_snapshot(snapshot)

    # --- MAIN INTERFACE RENDER ---
    # Header & Subsystems
    render_header(snapshot, state)
    render_subsystem_matrix(snapshot, state)

    # Top Level Tabs separating Live Stream from Historical Persistence
    tab_live, tab_history = st.tabs([
        "🔴 Live Surveillance Stream",
        "🗄️ Historical Database & Persistence",
    ])

    with tab_live:
        render_risk_banner(snapshot)

        # Upper Grid: Video & Crowd | Audio & Fusion
        row1_left, row1_right = st.columns(2)
        with row1_left:
            if op_mode == "Video File Analysis":
                st.markdown("#### 📹 Video Analysis Player")
                col_vp1, col_vp2 = st.columns([1, 1])
                with col_vp1:
                    play_video = st.button("▶ Play Live AI Detection Stream", key="btn_play_stream", use_container_width=True)
                with col_vp2:
                    show_native = st.checkbox("Show Browser Native Player", value=False)

                if show_native:
                    st.video(str(video_target_path))

                video_box = st.empty()
                metrics_box = st.empty()

                if play_video:
                    cap = cv2.VideoCapture(str(video_target_path))
                    if not cap.isOpened():
                        st.error(f"Could not open video file: {video_target_path}")
                    else:
                        analyzers = get_video_analyzers(conf=conf_val)
                        detector, tracker, density_est, pose_analyzer, fight_detector, weapon_detector, fusion_engine = analyzers
                        frame_num = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                break
                            frame_num += 1
                            if frame_num % 2 != 0:
                                continue
                            h, w = frame.shape[:2]
                            if w > 800:
                                scale = 800.0 / w
                                frame = cv2.resize(frame, (800, int(h * scale)))
                                h, w = frame.shape[:2]

                            curr_ts = time.time()
                            det = detector.detect(frame)
                            tracks = tracker.update(det, frame)
                            dens = density_est.estimate(
                                detection_points=det.bottom_centers,
                                frame_width=w,
                                frame_height=h,
                            )
                            kinematics = pose_analyzer.extract_features(tracks, timestamp=curr_ts)
                            fight_res = fight_detector.update(
                                kinematics=kinematics,
                                tracked_persons=tracks,
                                timestamp=curr_ts,
                            )
                            weapon_res = weapon_detector.detect(
                                frame=frame,
                                tracked_persons=tracks,
                                timestamp=curr_ts,
                            )
                            safety_res = fusion_engine.evaluate(
                                density_result=dens,
                                kinematics=kinematics,
                                fight_result=fight_res,
                                weapon_result=weapon_res,
                                timestamp=curr_ts,
                            )

                            annotated = frame.copy()
                            annotated = tracker.draw_tracks(annotated, tracks, draw_trajectory=True)
                            annotated = fight_detector.draw_fight_detections(annotated, fight_res)
                            annotated = weapon_detector.draw_detections(annotated, weapon_res)
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                            risk_state = safety_res.status.value
                            status_label = "🚨 EMERGENCY: VIOLENT ATTACK" if fight_res.is_fight else (
                                "⚠️ SUSPICIOUS ALTERCATION" if fight_res.is_suspicious else (
                                    f"NORMAL ({len(tracks)} tracked)"
                                )
                            )

                            video_box.image(
                                annotated_rgb,
                                caption=f"Frame #{frame_num} | People: {len(tracks)} | Safety: {safety_res.safety_score:.2f} | {status_label}",
                                use_container_width=True,
                            )

                            with metrics_box.container():
                                st.markdown("#### 👥 Real-Time Threat & Crowd Dynamics")
                                c1, c2, c3 = st.columns(3)
                                c1.metric(label="Tracked Persons", value=str(len(tracks)))
                                c2.metric(label="Fight / Attack Score", value=f"{fight_res.fight_score:.2f}")
                                c3.metric(label="Threat Status", value=risk_state)
                                if fight_res.is_fight:
                                    st.error(f"🚨 VIOLENT ALTERCATION DETECTED! (Confidence: {fight_res.fight_score*100:.0f}%) | Implicated Tracks: {fight_res.involved_track_ids}")
                                elif fight_res.is_suspicious:
                                    st.warning(f"⚠️ Rapid convergence / abnormal proximity detected (Score: {fight_res.fight_score:.2f})")

                            snap = DashboardSnapshot(
                                timestamp=curr_ts,
                                video_frame=annotated_rgb,
                                frame=annotated_rgb,
                                people_count=len(tracks),
                                crowd_count=len(tracks),
                                crowd_density=dens.density_per_m2,
                                density_level=dens.density_level,
                                anomaly_score=safety_res.safety_score,
                                video_score=safety_res.safety_score,
                                risk_state=risk_state,
                                decision_confidence=0.92,
                                dominant_modality="video",
                                movement_speed=kinematics.mean_speed,
                            )
                            state.add_snapshot(snap)
                            time.sleep(0.04)
                        cap.release()
                else:
                    render_video_panel(snapshot)
                    st.markdown("---")
                    render_crowd_metrics(snapshot)
            else:
                render_video_panel(snapshot)
                st.markdown("---")
                render_crowd_metrics(snapshot)

        with row1_right:
            render_audio_panel(snapshot)
            st.markdown("---")
            render_fusion_panel(snapshot)

        st.markdown("---")

        # Lower Grid: Alerts & Audit Log | XAI Evidence
        row2_left, row2_right = st.columns(2)
        with row2_left:
            st.markdown("#### 🚨 Alert Management (Phase 8)")
            render_active_alert_card(snapshot, state)
            st.markdown("---")
            render_alert_history_table(state)

        with row2_right:
            render_xai_panel(snapshot)

        st.markdown("---")

        # Bottom Full-Width: Real-time Telemetry Charts
        render_realtime_charts(state)

    with tab_history:
        render_database_panel(db_api, db_health)

    # Auto-rerun loop if not paused and not in video file analysis
    if not state.paused and op_mode != "Video File Analysis":
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
