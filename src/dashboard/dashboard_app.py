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

        if op_mode == "Video File Analysis":
            st.markdown("### 📹 Surveillance Command Monitor")

            # High-visibility player mode switcher
            mode_choice = st.radio(
                "Player View",
                ["🔴 Live AI Surveillance Stream (Real-Time Vision Pipeline)", "🎬 Native Browser Player (60 FPS Native)"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if "Live AI Surveillance Stream" in mode_choice:
                # Stream Controls & Performance Profile
                c_btn, c_stop, c_perf, c_scale = st.columns([2, 1, 3, 2])
                with c_btn:
                    play_video = st.button("▶ Start Live AI Stream", key="btn_play_stream", type="primary", use_container_width=True)
                with c_stop:
                    stop_video = st.button("⏹ Stop", key="btn_stop_stream", use_container_width=True)
                with c_perf:
                    speed_mode = st.selectbox(
                        "Performance Profile",
                        ["⚡ Ultra-Smooth (25-30 FPS)", "🎯 High Precision (15-20 FPS)"],
                        index=0,
                    )
                with c_scale:
                    theater_view = st.checkbox("🖥️ Theater HD View", value=True)

                if stop_video:
                    st.session_state["stop_requested"] = True
                elif play_video:
                    st.session_state["stop_requested"] = False

                # Full-width video container
                video_box = st.empty()
                metrics_box = st.empty()

                if play_video and not st.session_state.get("stop_requested", False):
                    cap = cv2.VideoCapture(str(video_target_path))
                    if not cap.isOpened():
                        st.error(f"Could not open video file: {video_target_path}")
                    else:
                        fps_val = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        stride = 2 if "Ultra-Smooth" in speed_mode else 1
                        keyframe_stride = 3 if "Ultra-Smooth" in speed_mode else 1
                        target_delay = (1.0 / max(fps_val, 15.0)) * stride
                        target_w = 640

                        analyzers = get_video_analyzers(conf=conf_val)
                        detector, tracker, density_est, pose_analyzer, fight_detector, weapon_detector, fusion_engine = analyzers
                        frame_num = 0
                        last_det = None
                        prev_fight = False
                        fps_timer = time.perf_counter()
                        fps_counter = 0
                        live_fps = fps_val

                        while cap.isOpened() and not st.session_state.get("stop_requested", False):
                            t_start = time.perf_counter()
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                break
                            frame_num += 1
                            if frame_num % stride != 0:
                                continue

                            # Downscale frame for lightning-fast YOLO inference
                            h, w = frame.shape[:2]
                            if w > target_w:
                                scale = target_w / float(w)
                                infer_frame = cv2.resize(frame, (target_w, int(h * scale)))
                            else:
                                infer_frame = frame
                            ih, iw = infer_frame.shape[:2]

                            curr_ts = time.time()

                            # Keyframe detection: Run YOLO every keyframe_stride frames
                            if frame_num % keyframe_stride == 0 or last_det is None:
                                last_det = detector.detect(infer_frame)

                            tracks = tracker.update(last_det, infer_frame)
                            dens = density_est.estimate(
                                detection_points=last_det.bottom_centers,
                                frame_width=iw,
                                frame_height=ih,
                            )
                            kinematics = pose_analyzer.extract_features(tracks, timestamp=curr_ts)
                            fight_res = fight_detector.update(
                                kinematics=kinematics,
                                tracked_persons=tracks,
                                timestamp=curr_ts,
                            )
                            weapon_res = weapon_detector.detect(
                                frame=infer_frame,
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

                            annotated = infer_frame.copy()
                            annotated = tracker.draw_tracks(annotated, tracks, draw_trajectory=True)
                            annotated = fight_detector.draw_fight_detections(annotated, fight_res)
                            annotated = weapon_detector.draw_detections(annotated, weapon_res)

                            # Compute live FPS
                            fps_counter += 1
                            if time.perf_counter() - fps_timer >= 0.5:
                                live_fps = fps_counter / (time.perf_counter() - fps_timer)
                                fps_counter = 0
                                fps_timer = time.perf_counter()

                            # Draw Sleek High-Tech Surveillance HUD directly on frame
                            ah, aw = annotated.shape[:2]
                            cv2.rectangle(annotated, (0, 0), (aw, 36), (15, 20, 28), -1)

                            if fight_res.is_fight:
                                status_hud = "EMERGENCY: VIOLENT ATTACK"
                                hud_color = (0, 0, 255)
                            elif fight_res.is_suspicious:
                                status_hud = "WARNING: SUSPICIOUS ALTERCATION"
                                hud_color = (0, 165, 255)
                            else:
                                status_hud = "MONITORING: NORMAL"
                                hud_color = (0, 255, 128)

                            cv2.putText(annotated, "CAM-01 [LIVE]", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                            cv2.putText(annotated, f"| {status_hud}", (150, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
                            cv2.putText(annotated, f"FPS: {live_fps:.1f} | TRACKS: {len(tracks)}", (max(10, aw - 220), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1)

                            if fight_res.is_fight:
                                cv2.rectangle(annotated, (0, ah - 34), (aw, ah), (0, 0, 200), -1)
                                cv2.putText(annotated, f"CRITICAL: VIOLENT ATTACK DETECTED (CONF: {fight_res.fight_score*100:.0f}%)",
                                            (max(10, aw // 2 - 250), ah - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

                            # Scale up for theater display if enabled (at least 1024px width for crisp visibility)
                            if theater_view and aw < 1024:
                                scale_up = 1024.0 / float(aw)
                                disp_frame = cv2.resize(annotated, (1024, int(ah * scale_up)), interpolation=cv2.INTER_LINEAR)
                            else:
                                disp_frame = annotated

                            annotated_rgb = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
                            risk_state = safety_res.status.value

                            # Display large full-width image
                            video_box.image(
                                annotated_rgb,
                                caption=f"Frame #{frame_num} | Live Stream Rate: {live_fps:.1f} FPS | Active Tracks: {len(tracks)} | Risk: {risk_state}",
                                use_container_width=True,
                            )

                            # Throttle metrics DOM update every 6 frames or on alert change to prevent browser DOM thrashing
                            if frame_num % 6 == 0 or fight_res.is_fight != prev_fight:
                                prev_fight = fight_res.is_fight
                                with metrics_box.container():
                                    st.markdown("#### 👥 Real-Time Threat & Crowd Dynamics")
                                    m1, m2, m3, m4 = st.columns(4)
                                    m1.metric(label="Tracked Persons", value=str(len(tracks)))
                                    m2.metric(label="Fight / Attack Score", value=f"{fight_res.fight_score:.2f}")
                                    m3.metric(label="Crowd Density", value=f"{dens.density_per_m2:.2f}/m²")
                                    m4.metric(label="Threat Status", value=risk_state)

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

                            # Dynamic millisecond pacing
                            t_proc = time.perf_counter() - t_start
                            remainder = target_delay - t_proc
                            if remainder > 0.003:
                                time.sleep(remainder)

                        cap.release()
                        st.success("✅ Surveillance Stream Finished")
                else:
                    # Initial / Idle preview in Theater view
                    if snapshot and snapshot.video_frame is not None:
                        disp_init = snapshot.video_frame
                        if theater_view and disp_init.shape[1] < 1024:
                            scale_init = 1024.0 / disp_init.shape[1]
                            disp_init = cv2.resize(disp_init, (1024, int(disp_init.shape[0] * scale_init)), interpolation=cv2.INTER_LINEAR)
                        video_box.image(disp_init, caption="Surveillance Feed Standby | Click 'Start Live AI Stream' to begin live detection", use_container_width=True)
                    else:
                        render_video_panel(snapshot)
                    st.markdown("---")
                    render_crowd_metrics(snapshot)

            else:
                # 60 FPS Native Browser Player Mode
                st.markdown("#### 🎬 Native Hardware-Accelerated Video Player (60 FPS)")
                st.caption("Native browser player provides instant full 60 FPS playback with timeline scrubbing, pause/resume, and fullscreen expansion.")
                st.video(str(video_target_path))
                st.markdown("---")
                render_crowd_metrics(snapshot)

        else:
            # Default Two-Column Grid for Demo Simulation & Live Pipeline
            row1_left, row1_right = st.columns(2)
            with row1_left:
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
