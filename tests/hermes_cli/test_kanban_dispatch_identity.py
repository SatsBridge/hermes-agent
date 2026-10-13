"""Shared-board identity aliases must stay within their declaring Hermes home."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.profiles import normalize_profile_name


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return root


@pytest.fixture(autouse=True)
def _normal_memory_pressure(monkeypatch):
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)


def _task(assignee="sage"):
    return kb.Task(
        id="t_identity", title="identity alias", body=None, assignee=assignee,
        status="running", priority=0, created_by="test", created_at=1,
        started_at=None, completed_at=None, workspace_kind="dir", workspace_path=None,
        claim_lock="lock", claim_expires=None, tenant=None, current_run_id=1,
    )


def _spawn(monkeypatch, task, workspace):
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kbd._default_spawn(task, str(workspace))
    return captured


def test_identity_claims_allowlisted_missing_profile(kanban_home):
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  agent_profile_name: sage\n  dispatch_profiles: [sage]\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="own card", assignee="sage")
        assert kbd.has_spawnable_ready(conn)
        result = kbd.dispatch_once(conn, dry_run=True)
    assert [task_id for task_id, _assignee, _workspace in result.spawned] == [tid]
    assert not result.skipped_nonspawnable


def test_identity_alias_works_without_allowlist(kanban_home):
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  agent_profile_name: sage\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="own card", assignee="sage")
        assert kbd.has_spawnable_ready(conn)
        result = kbd.dispatch_once(conn, dry_run=True)
    assert [task_id for task_id, _assignee, _workspace in result.spawned] == [tid]
    assert not result.skipped_nonspawnable


def test_allowlist_remains_authoritative_for_identity(kanban_home):
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  agent_profile_name: sage\n  dispatch_profiles: [researcher]\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="foreign card", assignee="sage")
        assert not kbd.has_spawnable_ready(conn)
        result = kbd.dispatch_once(conn, dry_run=True)
    assert result.spawned == []
    assert result.skipped_nonspawnable == [tid]


def test_path_like_identity_is_rejected_before_claim_and_spawn(kanban_home, caplog):
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  agent_profile_name: bad/name\n  dispatch_profiles: [bad/name]\n", encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="hermes_cli.kanban_db"):
        assert kbd._dispatch_identity(normalize_profile_name) is None
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="invalid identity", assignee="bad/name")
        assert not kbd.has_spawnable_ready(conn)
        result = kbd.dispatch_once(conn, dry_run=True)
    assert result.spawned == []
    assert result.skipped_nonspawnable == [tid]
    assert any("agent_profile_name" in record.getMessage() and "invalid" in record.getMessage()
               for record in caplog.records)


def test_identity_config_read_failure_disables_alias(kanban_home, monkeypatch, caplog):
    from hermes_cli import config_effective

    def fail_read(**_kwargs):
        raise OSError("config unavailable")

    monkeypatch.setattr(config_effective, "load_user_config_effective", fail_read)
    with caplog.at_level("WARNING", logger="hermes_cli.kanban_db"):
        assert kbd._dispatch_identity(normalize_profile_name) is None
    assert any("agent_profile_name unavailable" in record.getMessage() for record in caplog.records)


def test_malformed_config_disables_alias_and_claims(kanban_home, caplog):
    (kanban_home / "config.yaml").write_text("kanban: [broken\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="hermes_cli.kanban_db"):
        assert kbd._dispatch_identity(normalize_profile_name) is None
        assert kbd._dispatch_profile_allowlist(normalize_profile_name) == frozenset()
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="closed config", assignee="default")
            assert not kbd.has_spawnable_ready(conn)
            result = kbd.dispatch_once(conn, dry_run=True)
    assert result.spawned == []
    assert result.skipped_nonspawnable == [tid]
    assert any("agent_profile_name unavailable" in record.getMessage() for record in caplog.records)
    assert any("dispatch_profiles unavailable" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("named_home", [False, True], ids=["root-home", "named-home"])
def test_alias_worker_preserves_declaring_home_and_real_cli_preparse(
    tmp_path, monkeypatch, kanban_home, named_home,
):
    owner = kanban_home / "profiles" / "owner" if named_home else kanban_home
    if named_home:
        owner.mkdir(parents=True)
    (owner / "config.yaml").write_text(
        "kanban:\n  agent_profile_name: sage\n", encoding="utf-8",
    )
    (kanban_home / "active_profile").write_text("other", encoding="utf-8")
    other = kanban_home / "profiles" / "other"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(owner))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = _spawn(monkeypatch, _task(), workspace)
    cmd = captured["cmd"]
    assert "-p" not in cmd
    assert captured["env"]["HERMES_HOME"] == str(owner)
    assert captured["env"]["HERMES_PROFILE"] == "sage"

    # Exercise the actual CLI pre-parser on the exact child argv: neither the
    # root's active_profile nor the alias may move a worker out of its home.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_identity")
    monkeypatch.setattr(sys, "argv", cmd)
    from hermes_cli.main import _apply_profile_override

    _apply_profile_override()
    assert os.environ["HERMES_HOME"] == str(owner)


def test_existing_profile_directory_takes_priority(kanban_home, monkeypatch, tmp_path):
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  agent_profile_name: sage\n", encoding="utf-8",
    )
    profile = kanban_home / "profiles" / "sage"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    captured = _spawn(monkeypatch, _task(), workspace)
    # The worker is the owner client (no ``-p``): the real profile is selected through its home.
    assert captured["cmd"][-2:] == ["-m", "hermes_cli.kanban_worker_client"]
    assert captured["env"]["HERMES_HOME"] == str(profile)
    assert captured["env"]["HERMES_PROFILE"] == "sage"
