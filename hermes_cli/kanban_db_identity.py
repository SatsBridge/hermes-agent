"""Claim identity and liveness boundaries for the Kanban database."""

from __future__ import annotations

import os
import socket


def _effective_uid() -> int:
    """Translate this UID through the matching uid_map extent when available.

    Nested namespace maps are parent-relative, not proof of a universal host UID.
    """
    uid = os.getuid() if hasattr(os, "getuid") else 0
    try:
        with open("/proc/self/uid_map", encoding="utf-8") as fh:
            for line in fh:
                try:
                    inner, outer, count = (int(x) for x in line.split()[:3])
                except (ValueError, TypeError):
                    continue
                if count > 0 and inner <= uid < inner + count:
                    return outer + (uid - inner)
    except OSError:
        pass
    return uid


def _claimer_id() -> str:
    """Return ``host:uid:pid`` so claim locality is scoped to one mapped UID."""
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{_effective_uid()}:{os.getpid()}"


def _host_prefix() -> str:
    """Prefix shared by claims this mapped UID may probe and signal."""
    return _claimer_id().rsplit(":", 1)[0] + ":"


def _claim_is_live(trow) -> bool:
    """Protect live local workers; keep foreign/legacy claims without probing."""
    from hermes_cli import kanban_db as _kb

    lock = trow["claim_lock"]
    if trow["status"] != "running" or not lock or not trow["worker_pid"]:
        return False
    if not lock.startswith(_kb._host_prefix()):
        return True
    return _kb._worker_alive(trow["worker_pid"], trow["worker_started_at"])
