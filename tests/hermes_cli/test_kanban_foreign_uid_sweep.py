"""Claim locality and per-attempt crash-grace regressions."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _foreign_uid_lock() -> str:
    host, uid, _pid = kb._claimer_id().rsplit(":", 2)
    return f"{host}:{int(uid) + 1}:4242"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("legacy", [False, True], ids=["foreign-uid", "legacy-host-pid"])
def test_nonlocal_claim_is_not_crash_swept_or_probed(conn, monkeypatch, legacy):
    tid = kb.create_task(conn, title="foreign", assignee="w")
    host, _uid, _pid = kb._claimer_id().rsplit(":", 2)
    lock = f"{host}:4242" if legacy else _foreign_uid_lock()
    kb.claim_task(conn, tid, claimer=lock)
    conn.execute("UPDATE tasks SET worker_pid=4242 WHERE id=?", (tid,))

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("nonlocal worker must not be liveness-probed")

    monkeypatch.setattr(kbd, "_worker_alive", unexpected_probe)
    assert kbd.detect_crashed_workers(conn) == []
    row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == lock


@pytest.mark.platforms("posix")
def test_same_uid_dead_worker_is_crash_swept(conn):
    tid = kb.create_task(conn, title="local", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    kbd._set_worker_pid(conn, tid, _dead_pid())

    assert kbd.detect_crashed_workers(conn) == [tid]
    row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] != "running"
    assert row["claim_lock"] is None


def test_foreign_orphan_is_not_liveness_probed(conn, monkeypatch):
    tid = kb.create_task(conn, title="foreign orphan", assignee="w")
    lock = _foreign_uid_lock()
    kb.claim_task(conn, tid, claimer=lock)
    conn.execute("UPDATE tasks SET worker_pid=4242, claim_expires=NULL WHERE id=?", (tid,))

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("foreign orphan must not be liveness-probed")

    monkeypatch.setattr(kbd, "_worker_alive", unexpected_probe)
    assert kbd.reconcile_orphaned_running(conn) == []
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0] == "running"


def test_foreign_worker_termination_does_not_probe_or_signal(monkeypatch):
    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("foreign worker must not be probed")

    monkeypatch.setattr(kb, "_pid_alive", unexpected_probe)
    signals = []
    result = kbd._terminate_reclaimed_worker(
        4242, _foreign_uid_lock(), signal_fn=lambda *args: signals.append(args),
    )
    assert result["host_local"] is False
    assert result["termination_attempted"] is False
    assert signals == []


def test_foreign_expired_claim_still_uses_ttl_without_signalling(conn, monkeypatch):
    tid = kb.create_task(conn, title="foreign ttl", assignee="w")
    lock = _foreign_uid_lock()
    kb.claim_task(conn, tid, claimer=lock)
    conn.execute("UPDATE tasks SET worker_pid=4242, claim_expires=1 WHERE id=?", (tid,))

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("foreign worker must not be liveness-probed")

    monkeypatch.setattr(kbd, "_worker_alive", unexpected_probe)
    signals = []
    assert kb.release_stale_claims(conn, signal_fn=lambda *args: signals.append(args)) == 1
    row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert signals == []


def test_foreign_claim_protects_manual_completion_without_pid_probe(monkeypatch):
    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("foreign worker must not be liveness-probed")

    monkeypatch.setattr(kb, "_worker_alive", unexpected_probe)
    assert kb._claim_is_live({
        "status": "running",
        "claim_lock": _foreign_uid_lock(),
        "worker_pid": 4242,
        "worker_started_at": None,
    })


@pytest.mark.platforms("posix")
def test_reclaim_respawn_gets_fresh_crash_grace(conn, monkeypatch):
    tid = kb.create_task(conn, title="grace", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    conn.execute("UPDATE tasks SET started_at=1 WHERE id=?", (tid,))
    conn.execute("UPDATE task_runs SET started_at=1 WHERE task_id=?", (tid,))
    kbd._set_worker_pid(conn, tid, _dead_pid())
    assert kbd.detect_crashed_workers(conn) == [tid]

    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "3600")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    task_start = conn.execute("SELECT started_at FROM tasks WHERE id=?", (tid,)).fetchone()[0]
    run_start = conn.execute(
        "SELECT started_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()[0]
    assert task_start == 1
    assert abs(time.time() - run_start) < 60
    kbd._set_worker_pid(conn, tid, _dead_pid())

    assert kbd.detect_crashed_workers(conn) == []
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0] == "running"

    conn.execute(
        "UPDATE task_runs SET started_at=1 WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
        (tid,),
    )
    assert kbd.detect_crashed_workers(conn) == [tid]


def test_effective_uid_matches_the_extent_containing_uid():
    with patch("os.getuid", return_value=1000, create=True), patch(
        "builtins.open", mock_open(read_data="0 100000 1\n1000 200000 1\n"),
    ):
        assert kb._effective_uid() == 200000


def test_effective_uid_translates_offset_inside_extent():
    with patch("os.getuid", return_value=1005, create=True), patch(
        "builtins.open", mock_open(read_data="0 500000 100\n1000 700000 100\n"),
    ):
        assert kb._effective_uid() == 700005


def test_effective_uid_preserves_identity_map_and_unreadable_fallback():
    with patch("os.getuid", return_value=1000, create=True), patch(
        "builtins.open", mock_open(read_data="0 0 4294967295\n"),
    ):
        assert kb._effective_uid() == 1000
    with patch("os.getuid", return_value=1000, create=True), patch("builtins.open", side_effect=OSError):
        assert kb._effective_uid() == 1000
