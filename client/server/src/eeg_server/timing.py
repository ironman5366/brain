"""Helpers for aligning browser event markers to EEG timestamps."""

from collections.abc import Mapping, Sequence


def marker_epoch_seconds(marker: object, offset_seconds: float | None = None) -> float | None:
    """Return a marker timestamp as Unix seconds when possible.

    Preferred source is a client-provided epoch timestamp in milliseconds.
    Falls back to performance.now() + a derived server offset for older sessions.
    """
    client_time_ms = _field(marker, "client_time_ms")
    if client_time_ms is not None:
        return float(client_time_ms) / 1000.0

    timestamp = _field(marker, "timestamp")
    if timestamp is None or offset_seconds is None:
        return None

    return offset_seconds + float(timestamp) / 1000.0


def estimate_marker_offset_seconds(markers: Sequence[object]) -> float | None:
    """Estimate performance.now() -> Unix epoch offset for legacy markers."""
    for marker in markers:
        client_time_ms = _field(marker, "client_time_ms")
        if client_time_ms is not None:
            return 0.0

        timestamp = _field(marker, "timestamp")
        server_timestamp = _field(marker, "server_timestamp")
        if timestamp is None or server_timestamp is None:
            continue
        return float(server_timestamp) - float(timestamp) / 1000.0
    return None


def _field(marker: object, name: str) -> object | None:
    if isinstance(marker, Mapping):
        return marker.get(name)
    return getattr(marker, name, None)
