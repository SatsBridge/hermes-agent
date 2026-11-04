"""Dispatcher singleton locking for shared-board identities."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    return home


def _boot():
    runner = SimpleNamespace(_kanban_dispatcher_lock_handle=None)
    return runner, GatewayKanbanWatchersMixin._kanban_dispatcher_boot(runner)


def _write_identity(home, value):
    (home / "config.yaml").write_text(f"kanban:\n  agent_profile_name: {value}\n", encoding="utf-8")


def test_same_identity_serializes_gateway_dispatchers(kanban_home):
    _write_identity(kanban_home, "sage")
    owner, boot = _boot()
    assert boot is not None
    contender, contender_boot = _boot()
    assert contender_boot is None
    assert contender._kanban_dispatcher_lock_handle is None
    lock_dir = Path(boot[1].kanban_home()) / "kanban"
    assert (lock_dir / ".dispatcher.sage.lock").exists()
    GatewayKanbanWatchersMixin._release_kanban_dispatcher_lock(owner)


def test_distinct_identities_use_distinct_locks(kanban_home):
    _write_identity(kanban_home, "sage")
    first, first_boot = _boot()
    assert first_boot is not None
    _write_identity(kanban_home, "researcher")
    second, second_boot = _boot()
    assert second_boot is not None
    lock_dir = Path(first_boot[1].kanban_home()) / "kanban"
    assert (lock_dir / ".dispatcher.sage.lock").exists()
    assert (lock_dir / ".dispatcher.researcher.lock").exists()
    GatewayKanbanWatchersMixin._release_kanban_dispatcher_lock(first)
    GatewayKanbanWatchersMixin._release_kanban_dispatcher_lock(second)


@pytest.mark.parametrize("identity", [None, "bad/name"], ids=["unset", "invalid"])
def test_unset_or_invalid_identity_keeps_legacy_global_lock(kanban_home, identity):
    if identity is not None:
        _write_identity(kanban_home, identity)
    owner, boot = _boot()
    assert boot is not None
    _contender, contender_boot = _boot()
    assert contender_boot is None
    lock_dir = Path(boot[1].kanban_home()) / "kanban"
    assert (lock_dir / ".dispatcher.lock").exists()
    assert not list(lock_dir.glob(".dispatcher.*.lock"))
    GatewayKanbanWatchersMixin._release_kanban_dispatcher_lock(owner)


def test_malformed_config_keeps_legacy_global_lock(kanban_home, caplog):
    (kanban_home / "config.yaml").write_text("kanban: [broken\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="hermes_cli.kanban_db"):
        owner, boot = _boot()
    assert boot is not None
    lock_dir = Path(boot[1].kanban_home()) / "kanban"
    assert (lock_dir / ".dispatcher.lock").exists()
    assert any("agent_profile_name unavailable" in record.getMessage() for record in caplog.records)
    GatewayKanbanWatchersMixin._release_kanban_dispatcher_lock(owner)
