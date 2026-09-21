"""
Configuration Loader and Validator.
Loads settings from configs/config.yaml and resolves project-relative paths.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

_GLOBAL_CONFIG: Optional[Dict[str, Any]] = None


def get_default_config() -> Dict[str, Any]:
    """Provides fallback configuration dictionary if YAML is missing."""
    return {
        "video": {
            "source": "synthetic",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "loop": True,
            "auto_generate_synthetic": True,
        },
        "detection": {
            "model": "yolov8n.pt",
            "weights_dir": "models/detection",
            "confidence": 0.40,
            "iou_threshold": 0.45,
            "device": "auto",
            "classes": [0],
            "imgsz": 640,
        },
        "density": {
            "frame_area_m2": 50.0,
            "grid_rows": 3,
            "grid_cols": 3,
            "threshold_low": 0.3,
            "threshold_moderate": 0.8,
            "threshold_high": 1.5,
        },
        "tracking": {
            "tracker_type": "bytetrack",
            "tracker_config": "models/tracking/bytetrack.yaml",
            "history_length": 30,
            "speed_threshold": 25.0,
            "direction_bins": 8,
            "max_lost": 15,
            "iou_threshold": 0.25,
        },
        "weapon_detection": {
            "enabled": True,
            "model": "models/detection/weapon_model.pt",
            "confidence": 0.50,
            "classes": ["firearm", "knife", "dangerous_object"],
            "simulate_for_demo": False,
        },
        "fight_detection": {
            "enabled": True,
            "temporal_window": 20,
            "suspicious_threshold": 0.50,
            "emergency_threshold": 0.75,
            "confirmation_frames": 5,
            "cooldown_seconds": 10.0,
            "proximity_threshold": 130.0,
        },
        "safety_fusion": {
            "crowd_weight": 0.25,
            "fight_weight": 0.35,
            "weapon_weight": 0.40,
            "normal_threshold": 0.35,
            "emergency_threshold": 0.70,
            "correlation_multiplier": 1.25,
        },
        "alert": {
            "confirmation_frames": 5,
            "cooldown_seconds": 10.0,
        },
        "behavior": {
            "enabled": True,
            "model_type": "swin",
            "model_path": "",
            "sequence_length": 16,
            "sample_rate": 4,
            "confidence_threshold": 0.50,
            "confirmation_frames": 5,
            "smoothing_window": 10,
            "speed_threshold": 18.0,
            "dispersion_threshold": 2.5,
            "convergence_threshold": 2.5,
        },
        "behavior_fusion": {
            "kinematic_weight": 0.40,
            "deep_model_weight": 0.60,
        },
        "audio": {
            "enabled": True,
            "source": "microphone",
            "sample_rate": 16000,
            "channels": 1,
            "chunk_duration": 1.0,
            "overlap": 0.5,
            "n_fft": 1024,
            "hop_length": 512,
            "n_mels": 64,
            "model_path": "",
            "confidence_threshold": 0.50,
            "temporal_window": 5,
            "confirmation_windows": 2,
            "smoothing_window": 5,
            "suspicious_threshold": 0.40,
            "emergency_threshold": 0.70,
        },
        "fusion": {
            "video_weight": 0.60,
            "audio_weight": 0.40,
            "density_sub_weight": 0.15,
            "movement_sub_weight": 0.20,
            "behavior_sub_weight": 0.25,
            "fight_sub_weight": 0.20,
            "weapon_sub_weight": 0.20,
            "synergy_boost": 1.25,
            "suppression_factor": 0.50,
            "temporal_tolerance_seconds": 1.5,
            "stale_timeout_seconds": 2.5,
            "smoothing_window": 5,
        },
        "decision": {
            "enabled": True,
            "normal_threshold": 0.35,
            "emergency_threshold": 0.70,
            "cooldown_seconds": 5.0,
            "suspicious_enter": 0.45,
            "suspicious_exit": 0.35,
            "high_risk_enter": 0.65,
            "high_risk_exit": 0.55,
            "emergency_enter": 0.85,
            "emergency_exit": 0.70,
            "suspicious_confirmation": 3,
            "high_risk_confirmation": 3,
            "emergency_confirmation": 2,
            "recovery_confirmation": 5,
            "transition_cooldown_seconds": 3.0,
            "critical_evidence_enabled": True,
            "critical_weapon_threshold": 0.90,
            "critical_fight_threshold": 0.90,
            "critical_multimodal_threshold": 0.90,
            "dynamic_thresholds": True,
            "max_threshold_adjustment": 0.10,
        },
        "alerts": {
            "enabled": True,
            "minimum_severity": "WARNING",
            "cooldown_seconds": 10.0,
            "duplicate_window_seconds": 15.0,
            "recovery_alert": False,
            "channels": {
                "console": True,
                "hud": True,
                "local_alarm": False,
                "email": False,
                "sms": False,
                "push": False,
            },
            "escalation": {
                "enabled": True,
                "repeat_emergency_after_seconds": 30.0,
            },
            # Legacy channel switches preserved
            "enable_console": True,
            "enable_visual": True,
            "enable_sound": False,
            "enable_sms": False,
            "enable_email": False,
        },
        "xai": {
            "enabled": True,
            "hud": {
                "enabled": True,
                "max_evidence_items": 4,
                "show_confidence": True,
                "show_scores": True,
                "show_motion_vectors": True,
                "show_dispersion_arrows": True,
            },
            "gradcam": {
                "enabled": False,
            },
            "explanations": {
                "max_items": 5,
                "min_contribution_threshold": 0.01,
            },
        },
        "dashboard": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8501,
            "max_history": 120,
            "refresh_interval_seconds": 2.0,
            "theme": "dark",
            "demo_mode_default": True,
        },
        "database": {
            "enabled": True,
            "path": "data/crowd_anomaly.db",
            "auto_create": True,
            "wal_mode": True,
            "foreign_keys": True,
            "busy_timeout_ms": 5000,
            "retention_days": 90,
            "persist_events": True,
            "persist_alerts": True,
            "persist_evidence": True,
            "persist_system_metrics": False,
            "metrics_persist_interval_seconds": 1.0,
        },
        "evaluation": {
            "enabled": True,
            "dataset": {
                "type": "none",
                "path": None,
                "ground_truth_available": False,
            },
            "output_dir": "reports/evaluation",
            "synthetic": {
                "enabled": True,
            },
            "threshold_sweep": {
                "enabled": True,
            },
            "ablation": {
                "enabled": True,
            },
            "robustness": {
                "enabled": True,
            },
            "latency": {
                "enabled": True,
            },
            "database_benchmark": {
                "enabled": True,
            },
        },
        "logging": {
            "level": "INFO",
            "log_to_file": True,
            "log_file": "logs/crowd_anomaly.log",
        },
    }


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and merge with default fallback.

    Args:
        config_path: Path to YAML configuration file. Defaults to configs/config.yaml.

    Returns:
        Dict containing merged configuration settings.
    """
    global _GLOBAL_CONFIG

    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = get_default_config()

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
                if isinstance(user_config, dict):
                    # Recursive shallow-to-medium merge
                    for section, values in user_config.items():
                        if isinstance(values, dict) and section in config:
                            config[section].update(values)
                        else:
                            config[section] = values
        except Exception as e:
            print(f"[WARNING] Failed to parse config file {target_path}: {e}. Using defaults.")
    else:
        print(f"[INFO] Config file {target_path} not found. Utilizing default configuration.")

    _GLOBAL_CONFIG = config
    return config


def get_config() -> Dict[str, Any]:
    """Returns currently loaded configuration or loads default if not yet initialized."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = load_config()
    return _GLOBAL_CONFIG


def resolve_path(relative_or_absolute_path: str | Path) -> Path:
    """
    Resolves relative paths against PROJECT_ROOT.
    Leaves absolute paths unchanged.
    """
    p = Path(relative_or_absolute_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
