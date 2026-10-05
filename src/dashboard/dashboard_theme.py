"""
Surveillance Terminal Theme & Visual Tokens - Phase 10.

Provides dark surveillance terminal palette, risk state badges,
severity colors, and custom CSS styling for the Streamlit dashboard.
"""

from typing import Dict

# Primary Surveillance Color Palette
BG_COLOR = "#0E1117"
SURFACE_COLOR = "#1A1E29"
SURFACE_BORDER = "#2D3748"
TEXT_PRIMARY = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
ACCENT_CYAN = "#06B6D4"

# Risk State Color Tokens
STATE_COLORS: Dict[str, str] = {
    "NORMAL": "#10B981",     # Emerald Green
    "SUSPICIOUS": "#F59E0B", # Amber
    "HIGH_RISK": "#EA580C",  # Deep Orange
    "EMERGENCY": "#EF4444",  # Crimson
    "RECOVERY": "#06B6D4",   # Cyan
    "UNKNOWN": "#6B7280",    # Slate
}

# Alert Severity Color Tokens
SEVERITY_COLORS: Dict[str, str] = {
    "INFO": "#3B82F6",       # Blue
    "WARNING": "#F59E0B",    # Amber
    "CRITICAL": "#EF4444",   # Red
}

CUSTOM_CSS = """
<style>
/* Surveillance Terminal Custom Stylesheet */
.main {
    background-color: #0E1117;
    color: #F8FAFC;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}
.stApp {
    background-color: #0E1117;
}
.metric-card {
    background: #1A1E29;
    border: 1px solid #2D3748;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.5px;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94A3B8;
}
.health-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px 4px 2px 0;
}
.health-online {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.health-offline {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.state-banner {
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
    border: 2px solid;
    text-transform: uppercase;
}
.alert-card {
    background: #20171B;
    border: 2px solid #EF4444;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 14px;
}
.alert-card-warning {
    background: #241D14;
    border: 2px solid #F59E0B;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 14px;
}
.alert-title {
    font-size: 18px;
    font-weight: 700;
    color: #F8FAFC;
}
.evidence-pill {
    background: #2D3748;
    color: #E2E8F0;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    margin-right: 6px;
    display: inline-block;
}

/* Surveillance Monitor - Full-Width Theater Display */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 98% !important;
}
[data-testid="stImage"] {
    width: 100% !important;
    display: flex;
    justify-content: center;
    background: #080B10;
    border-radius: 8px;
}
[data-testid="stImage"] > img {
    width: 100% !important;
    min-height: 460px;
    max-height: 80vh;
    object-fit: contain;
    border-radius: 8px;
    border: 2px solid #06B6D4;
    background-color: #07090E;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7), 0 0 20px rgba(6, 182, 212, 0.2);
}
[data-testid="stVideo"] video {
    width: 100% !important;
    min-height: 460px;
    max-height: 80vh;
    border-radius: 8px;
    border: 2px solid #06B6D4;
    background-color: #07090E;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7), 0 0 20px rgba(6, 182, 212, 0.2);
}
</style>
"""


class DashboardTheme:
    """Surveillance UI theme and visual tokens."""

    CUSTOM_CSS: str = CUSTOM_CSS
    STATE_COLORS: Dict[str, str] = STATE_COLORS
    SEVERITY_COLORS: Dict[str, str] = SEVERITY_COLORS

    @staticmethod
    def get_state_color(state: str) -> str:
        """Return hex color corresponding to risk state."""
        return STATE_COLORS.get(state.upper(), "#6B7280")

    @staticmethod
    def get_severity_color(severity: str) -> str:
        """Return hex color corresponding to alert severity."""
        return SEVERITY_COLORS.get(severity.upper(), "#94A3B8")

    @staticmethod
    def get_state_badge_html(state: str, confidence: float = 1.0) -> str:
        """Generate HTML badge for a given risk state."""
        color = DashboardTheme.get_state_color(state)
        conf_str = f" ({confidence * 100:.0f}%)" if confidence < 1.0 else ""
        return (
            f'<div class="state-banner" style="background: {color}22; border-color: {color}; color: {color};">'
            f'SYSTEM STATE: {state.upper()}{conf_str}'
            f'</div>'
        )

    @staticmethod
    def get_health_badge_html(subsystem: str, online: bool) -> str:
        """Generate HTML health pill for a subsystem."""
        css_class = "health-online" if online else "health-offline"
        status_txt = "ONLINE" if online else "OFFLINE"
        label = subsystem.replace("_", " ").title()
        return f'<span class="health-pill {css_class}">● {label}: {status_txt}</span>'

    @staticmethod
    def get_severity_badge_html(severity: str) -> str:
        """Generate HTML badge for an alert severity."""
        color = DashboardTheme.get_severity_color(severity)
        return (
            f'<span style="background: {color}25; color: {color}; border: 1px solid {color}55; '
            f'padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">'
            f'{severity.upper()}</span>'
        )
