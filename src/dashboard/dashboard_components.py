"""
Streamlit UI Components for Surveillance Dashboard - Phase 10.

Implements modular UI renderers for system telemetry, live streams,
alert workflows, explainability panels, and time-series charts.
"""

from datetime import datetime
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.dashboard.dashboard_data import DashboardSnapshot
from src.dashboard.dashboard_state import DashboardState
from src.dashboard.dashboard_theme import DashboardTheme


def render_header(snapshot: Optional[DashboardSnapshot], state: DashboardState) -> None:
    """Render top terminal header with title, current clock, and operational status."""
    st.markdown(DashboardTheme.CUSTOM_CSS, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🛡️ REAL-TIME CROWD ANOMALY MONITORING")
        st.caption("AI-Powered Multimodal Surveillance & Predictive Alert System | Phase 10 Interface")
    with col2:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_text = "⏸ PAUSED" if state.paused else "🟢 LIVE"
        st.markdown(
            f'<div style="text-align: right; font-family: monospace; font-size: 13px; color: #94A3B8;">'
            f'STATUS: <b style="color: #F8FAFC;">{status_text}</b><br>'
            f'SYSTEM TIME: <b style="color: #06B6D4;">{now_str}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_subsystem_matrix(snapshot: Optional[DashboardSnapshot], state: DashboardState) -> None:
    """Render 9-subsystem health grid."""
    health = snapshot.system_health if snapshot and snapshot.system_health else state.subsystem_status
    pills = [DashboardTheme.get_health_badge_html(name, status) for name, status in health.items()]
    st.markdown(
        f'<div style="background: #1A1E29; border: 1px solid #2D3748; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px;">'
        f'{" ".join(pills)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_risk_banner(snapshot: Optional[DashboardSnapshot]) -> None:
    """Render high-visibility dynamic risk state banner."""
    if not snapshot:
        st.info("Awaiting telemetry stream...")
        return

    badge_html = DashboardTheme.get_state_badge_html(snapshot.risk_state, snapshot.confidence)
    st.markdown(badge_html, unsafe_allow_html=True)


def render_active_alert_card(
    snapshot: Optional[DashboardSnapshot],
    state: DashboardState,
    on_ack: Optional[Any] = None,
    on_resolve: Optional[Any] = None,
) -> None:
    """Render active emergency/warning alert card with operator action buttons."""
    alert = state.active_alert or (snapshot.active_alert if snapshot else None)
    if not alert:
        st.markdown(
            '<div style="background: #1A1E29; border: 1px solid #2D3748; border-radius: 6px; padding: 12px; text-align: center; color: #10B981; font-size: 13px;">'
            '✔ No Active Emergency Alerts. System Operating within Normal Parameters.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    severity = alert.get("severity", "WARNING").upper()
    alert_id = str(alert.get("alert_id", "alert_001"))
    state_str = alert.get("state", "EMERGENCY")
    msg = alert.get("message", "High risk activity detected")
    reason = alert.get("reason", "Anomaly threshold exceeded")
    is_ack = alert_id in state.acknowledged_alert_ids
    is_res = alert_id in state.resolved_alert_ids

    card_class = "alert-card" if severity == "CRITICAL" else "alert-card-warning"
    border_color = "#EF4444" if severity == "CRITICAL" else "#F59E0B"

    st.markdown(
        f'<div class="{card_class}">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
        f'<span class="alert-title">🚨 [{severity}] {state_str} ALERT</span>'
        f'{DashboardTheme.get_severity_badge_html(severity)}'
        f'</div>'
        f'<div style="color: #F8FAFC; font-size: 14px; margin-bottom: 6px;"><b>Event:</b> {msg}</div>'
        f'<div style="color: #94A3B8; font-size: 12px; margin-bottom: 10px;"><b>Trigger:</b> {reason} | <b>ID:</b> {alert_id}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Operator Action Buttons
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        ack_disabled = is_ack or is_res
        btn_label = "✔ Acknowledged" if is_ack else "Acknowledge"
        if st.button(btn_label, key=f"ack_{alert_id}", disabled=ack_disabled, use_container_width=True):
            state.acknowledge_alert(alert_id, operator_id="operator_1")
            st.rerun()

    with col_b:
        res_disabled = is_res
        btn_label_res = "✔ Resolved" if is_res else "Resolve"
        if st.button(btn_label_res, key=f"res_{alert_id}", disabled=res_disabled, use_container_width=True):
            state.resolve_alert(alert_id, operator_id="operator_1", reason="Operator cleared alert")
            st.rerun()

    with col_c:
        if is_res:
            st.caption("Status: Resolved by operator")
        elif is_ack:
            st.caption("Status: Acknowledged - Investigation underway")
        else:
            st.caption("Action Required: Immediate operator review")


def render_video_panel(snapshot: Optional[DashboardSnapshot]) -> None:
    """Render surveillance video stream with detections or honest offline fallback."""
    st.markdown("#### 📹 Live Video Stream")
    if snapshot and snapshot.video_frame is not None:
        st.image(snapshot.video_frame, caption="Live Video Feed (OpenCV)", use_container_width=True)
    else:
        st.markdown(
            '<div style="background: #11141C; border: 1px dashed #4B5563; border-radius: 8px; height: 260px; '
            'display: flex; flex-direction: column; justify-content: center; align-items: center; color: #64748B;">'
            '<span style="font-size: 32px; margin-bottom: 8px;">📷</span>'
            '<b>Camera Feed Offline / Simulation Mode</b>'
            '<span style="font-size: 11px; margin-top: 4px;">Connect RTSP stream or upload video file</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Video score progress bar
    v_score = snapshot.video_score if snapshot else 0.0
    st.progress(min(max(v_score, 0.0), 1.0), text=f"Video Anomaly Score: {v_score:.3f}")


def render_crowd_metrics(snapshot: Optional[DashboardSnapshot]) -> None:
    """Render crowd counting and density metrics."""
    st.markdown("#### 👥 Crowd Dynamics")
    c1, c2, c3 = st.columns(3)
    count = snapshot.crowd_count if snapshot else 0
    density = snapshot.crowd_density if snapshot else 0.0
    speed = snapshot.movement_speed if snapshot else 0.0

    with c1:
        st.metric(label="Person Count", value=str(count))
    with c2:
        st.metric(label="Crowd Density", value=f"{density:.2f}")
    with c3:
        st.metric(label="Mean Speed", value=f"{speed:.1f} px/f")


def render_audio_panel(snapshot: Optional[DashboardSnapshot]) -> None:
    """Render audio stream telemetry and classification mode."""
    st.markdown("#### 🎙️ Acoustic Surveillance")
    a_score = snapshot.audio_score if snapshot else 0.0
    dominant_sound = snapshot.dominant_audio_event if snapshot else "Normal"
    db_level = snapshot.audio_db if snapshot else -40.0

    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Audio Event", value=dominant_sound.title())
    with c2:
        st.metric(label="Sound Level", value=f"{db_level:.1f} dB")

    # Honest classifier provenance
    st.caption("Acoustic Engine: **Spectral Baseline (Phase 5)**")
    st.progress(min(max(a_score, 0.0), 1.0), text=f"Audio Anomaly Score: {a_score:.3f}")


def render_fusion_panel(snapshot: Optional[DashboardSnapshot]) -> None:
    """Render multimodal fusion telemetry and dynamic weights."""
    st.markdown("#### 🔀 Multimodal Fusion & Decision Engine")
    score = snapshot.anomaly_score if snapshot else 0.0
    dom_mod = snapshot.dominant_modality if snapshot else "video"
    corrob = snapshot.corroborated if snapshot else False
    w_vid = snapshot.weights.get("video", 0.6) if snapshot else 0.6
    w_aud = snapshot.weights.get("audio", 0.4) if snapshot else 0.4

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="Fused Anomaly Score", value=f"{score:.3f}")
    with c2:
        st.metric(label="Dominant Modality", value=dom_mod.upper())
    with c3:
        st.metric(label="Cross-Modal Corroborated", value="YES" if corrob else "NO")

    st.caption(f"Dynamic Weights: Video ({w_vid:.2f}) | Audio ({w_aud:.2f})")
    st.progress(min(max(score, 0.0), 1.0), text=f"Composite Risk Level: {score:.3f}")


def render_xai_panel(snapshot: Optional[DashboardSnapshot]) -> None:
    """Render Explainable AI natural language summary and feature contributions."""
    st.markdown("#### 💡 Explainable AI (XAI) Evidence")
    if not snapshot:
        st.caption("No XAI telemetry available.")
        return

    expl_text = snapshot.xai_explanation or "System operating within normal baseline activity."
    action_text = snapshot.operator_action or "Continue standard monitoring."

    st.markdown(
        f'<div style="background: #1A1E29; border-left: 4px solid #06B6D4; padding: 12px; border-radius: 4px; margin-bottom: 10px;">'
        f'<b style="color: #06B6D4;">Evidence Explanation:</b><br>'
        f'<span style="color: #F8FAFC; font-size: 13px;">{expl_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="background: #1A1E29; border-left: 4px solid #10B981; padding: 12px; border-radius: 4px; margin-bottom: 12px;">'
        f'<b style="color: #10B981;">Recommended Action:</b><br>'
        f'<span style="color: #F8FAFC; font-size: 13px;">{action_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if snapshot.top_contributors:
        st.markdown("<b>Contributing Modality Weights:</b>", unsafe_allow_html=True)
        for factor, weight in snapshot.top_contributors.items():
            label = factor.replace("_", " ").title()
            val = float(weight) if isinstance(weight, (int, float)) else 0.0
            st.progress(min(max(val, 0.0), 1.0), text=f"{label}: {val:.2f}")


def render_realtime_charts(state: DashboardState) -> None:
    """Render real-time rolling telemetry charts."""
    st.markdown("#### 📈 Real-Time Anomaly Telemetry")
    chart_data = state.get_chart_data()
    timestamps = chart_data.get("timestamps", [])

    if len(timestamps) < 2:
        st.info("Gathering historical data for time-series charts...")
        return

    # Create pandas DataFrame for synchronized charts
    df = pd.DataFrame({
        "Anomaly Score": chart_data.get("anomaly_scores", []),
        "Video Score": chart_data.get("video_scores", []),
        "Audio Score": chart_data.get("audio_scores", []),
    })

    st.line_chart(df, height=220)


def render_alert_history_table(state: DashboardState) -> None:
    """Render bounded history table of recent alert records."""
    st.markdown("#### 📋 Recent Alert Audit Log")
    if not state.recent_alerts:
        st.caption("No alerts logged in current session.")
        return

    rows = []
    for a in state.recent_alerts[:15]:
        aid = a.get("alert_id", "N/A")
        state_str = a.get("state", "N/A")
        sev = a.get("severity", "INFO")
        reason = a.get("reason", a.get("message", "N/A"))
        status = "RESOLVED" if aid in state.resolved_alert_ids else ("ACK" if aid in state.acknowledged_alert_ids else "ACTIVE")

        ts = a.get("timestamp", time.time())
        ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if isinstance(ts, (int, float)) else str(ts)

        rows.append({
            "Time": ts_str,
            "ID": aid,
            "State": state_str,
            "Severity": sev,
            "Status": status,
            "Description": reason,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_database_panel(
    historical_api: Optional[Any] = None,
    db_health: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render historical database analytics, connection status,
    summary statistics, and searchable historical logs - Phase 11.
    """
    st.markdown("### 🗄️ Historical Database & Event Persistence (Phase 11)")
    st.caption("Persistent SQLite storage for monitoring sessions, decision events, alerts, and XAI evidence.")

    # 1. Health & Connection Status
    if not historical_api or not db_health or db_health.get("status") != "ONLINE":
        status_text = db_health.get("status", "OFFLINE") if db_health else "OFFLINE"
        st.markdown(
            f'<div style="background: #1A1E29; border: 1px solid #EF4444; border-radius: 6px; padding: 14px; margin-bottom: 16px;">'
            f'<span style="color: #EF4444; font-weight: bold; font-size: 15px;">DATABASE STATUS: {status_text}</span><br>'
            f'<span style="color: #94A3B8; font-size: 13px;">'
            f'Historical persistence layer is offline or unreachable. Real-time monitoring continues unaffected with bounded in-memory buffer.'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # Online status badges
    db_path = db_health.get("path", "data/crowd_anomaly.db")
    schema_ver = db_health.get("schema_version", 1)
    wal = "WAL ENABLED" if db_health.get("wal_mode") else "STANDARD"
    fk = "FK ON" if db_health.get("foreign_keys") else "FK OFF"

    st.markdown(
        f'<div style="background: #1A1E29; border: 1px solid #10B981; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
        f'<div>'
        f'<span style="background: #064E3B; color: #34D399; font-weight: bold; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px;">🟢 ONLINE</span>'
        f'<span style="background: #1E293B; color: #94A3B8; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px;">SCHEMA: v{schema_ver}</span>'
        f'<span style="background: #1E293B; color: #94A3B8; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px;">{wal}</span>'
        f'<span style="background: #1E293B; color: #94A3B8; padding: 3px 8px; border-radius: 4px; font-size: 12px;">{fk}</span>'
        f'</div>'
        f'<div style="color: #64748B; font-family: monospace; font-size: 12px;">DB: {db_path}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 2. Historical Summary Statistics
    try:
        stats = historical_api.get_statistics()
    except Exception as exc:
        st.warning(f"Could not compute historical statistics: {exc}")
        stats = {}

    st.markdown("#### 📊 Persistent Telemetry Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(label="Total Events", value=str(stats.get("total_events", 0)))
    with c2:
        st.metric(label="Total Alerts", value=str(stats.get("total_alerts", 0)))
    with c3:
        st.metric(label="Emergencies", value=str(stats.get("emergency_count", 0)))
    with c4:
        st.metric(label="High Risk", value=str(stats.get("high_risk_count", 0)))
    with c5:
        st.metric(label="Avg Anomaly", value=f"{stats.get('average_anomaly_score', 0.0):.3f}")
    with c6:
        st.metric(label="Peak Anomaly", value=f"{stats.get('maximum_anomaly_score', 0.0):.3f}")

    st.markdown("---")

    # 3. Query & Historical Data Views
    st.markdown("#### 🔍 Historical Record Explorer")

    q_col1, q_col2 = st.columns([2, 1])
    with q_col1:
        state_filter = st.selectbox(
            "Filter by Risk State",
            ["ALL", "NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY", "RECOVERY"],
            index=0,
            key="hist_state_filter",
        )
    with q_col2:
        limit_val = st.selectbox(
            "Display Limit",
            [10, 25, 50, 100],
            index=1,
            key="hist_limit_val",
        )

    # View subtabs for Events, Alerts, Sessions
    ev_tab, alt_tab, sess_tab = st.tabs(["📋 Historical Events", "🚨 Historical Alerts", "🎬 Sessions"])

    with ev_tab:
        try:
            if state_filter == "ALL":
                events = historical_api.events.get_recent_events(limit=limit_val)
            else:
                events = historical_api.events.query_events_by_state(state_filter, limit=limit_val)

            if events:
                rows = []
                for e in events:
                    rows.append({
                        "Time (UTC)": e.timestamp,
                        "Event ID": e.event_id,
                        "State": e.risk_state,
                        "Score": f"{e.anomaly_score:.2f}",
                        "Modality": e.dominant_modality,
                        "People": e.people_count if e.people_count is not None else "-",
                        "Density": f"{e.crowd_density:.2f}" if e.crowd_density is not None else "-",
                        "Corroborated": "YES" if e.corroborated else "NO",
                        "Reason": e.reason,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No historical events match the query criteria.")
        except Exception as exc:
            st.error(f"Error querying historical events: {exc}")

    with alt_tab:
        try:
            alerts = historical_api.alerts.get_recent_alerts(limit=limit_val)
            if alerts:
                rows = []
                for a in alerts:
                    rows.append({
                        "Time (UTC)": a.timestamp,
                        "Alert ID": a.alert_id,
                        "Severity": a.severity,
                        "State": a.risk_state,
                        "Status": a.status,
                        "Score": f"{a.score:.2f}",
                        "Ack At": a.acknowledged_at or "-",
                        "Resolved At": a.resolved_at or "-",
                        "Reason": a.reason,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No historical alerts recorded.")
        except Exception as exc:
            st.error(f"Error querying historical alerts: {exc}")

    with sess_tab:
        try:
            sessions = historical_api.sessions.list_sessions(limit=limit_val)
            if sessions:
                rows = []
                for s in sessions:
                    rows.append({
                        "Session ID": s.session_id,
                        "Started At": s.started_at,
                        "Ended At": s.ended_at or "ACTIVE",
                        "Status": s.status,
                        "Source Type": s.source_type,
                        "Video Source": s.video_source,
                        "Audio": "ON" if s.audio_enabled else "OFF",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No sessions recorded.")
        except Exception as exc:
            st.error(f"Error querying historical sessions: {exc}")

