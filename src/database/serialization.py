"""
Data Serialization and Type Conversion Helpers - Phase 11.

Handles conversion between SQLite storage types (TEXT, INTEGER, REAL, NULL)
and Python data structures (datetime, enums, dataclasses, booleans, nested JSON).
"""

from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, Optional, Union


def to_iso8601(val: Optional[Union[float, int, str, datetime]]) -> Optional[str]:
    """
    Convert timestamp representation (epoch seconds, datetime, or ISO string)
    to a standardized UTC ISO-8601 formatted string.
    """
    if val is None:
        return None

    if isinstance(val, datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.isoformat()

    if isinstance(val, (int, float)):
        # Epoch seconds
        dt = datetime.fromtimestamp(float(val), tz=timezone.utc)
        return dt.isoformat()

    if isinstance(val, str):
        # Validate or return raw string if already formatted
        val_clean = val.strip()
        if not val_clean:
            return None
        return val_clean

    return str(val)


def from_iso8601(iso_str: Optional[str]) -> Optional[float]:
    """
    Parse an ISO-8601 formatted string into epoch seconds (float).
    Returns None if the string is empty or None.
    """
    if not iso_str:
        return None

    try:
        # Handle trailing Z
        clean_str = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        return dt.timestamp()
    except Exception:
        return None


def serialize_enum(enum_val: Any) -> Optional[str]:
    """
    Extract string representation from an Enum or return the string directly.
    """
    if enum_val is None:
        return None
    if isinstance(enum_val, Enum):
        return str(enum_val.value)
    return str(enum_val)


def serialize_bool(bool_val: Optional[Union[bool, int]]) -> Optional[int]:
    """
    Convert boolean to SQLite INTEGER (0 or 1), preserving None as NULL.
    """
    if bool_val is None:
        return None
    return 1 if bool(bool_val) else 0


def deserialize_bool(int_val: Optional[Union[int, bool]]) -> Optional[bool]:
    """
    Convert SQLite INTEGER (0 or 1) back to Python boolean, preserving None.
    """
    if int_val is None:
        return None
    return bool(int_val)


def to_json_str(data: Any) -> Optional[str]:
    """
    Serialize dictionary, list, or primitive to a JSON string.
    Returns None if data is None.
    """
    if data is None:
        return None

    def _default(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        return str(obj)

    try:
        return json.dumps(data, default=_default, ensure_ascii=False)
    except Exception:
        return "{}"


def from_json_str(json_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse a JSON string into a Python dictionary.
    Returns empty dictionary if None or malformed.
    """
    if not json_str:
        return {}

    try:
        res = json.loads(json_str)
        if isinstance(res, dict):
            return res
        return {"data": res}
    except Exception:
        return {}
