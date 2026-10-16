"""Fast slash commands must report and mutate the live session authority."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.compute_host import ComputeHost
from tui_gateway.host_supervisor import HostSupervisor


class _StaleWorker:
    def run(self, command: str) -> str:
        return "Priority Processing: fast"


def _agent(*, tier="priority", model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1"):
    return SimpleNamespace(
        model=model, provider=provider, base_url=base_url, api_key=None,
        service_tier=tier, request_overrides={"service_tier": tier} if tier else {},
        _service_tier_session_pinned=False, _framework_baked_tier_keys=(),
    )


def _slash(monkeypatch, session, command: str) -> dict:
    monkeypatch.setitem(server._sessions, "sid-fast", session)
    return server._methods["slash.exec"]("rid", {"session_id": "sid-fast", "command": command})


def _host_ack(monkeypatch, session, command: str) -> dict:
    with monkeypatch.context() as scoped:
        scoped.setattr(server, "_sessions", {"sid-fast": session})
        scoped.setattr(server, "_history_to_messages", lambda *_args, **_kwargs: [])
        scoped.setattr(server, "_session_info", lambda *_args, **_kwargs: {})
        return _run_host_control(session, command)


def _run_host_control(session, command: str) -> dict:
    host = ComputeHost.__new__(ComputeHost)
    replies = []
    host._guarded = lambda frame, _error_type, body, **_kwargs: body(
        server, frame["sid"], frame["request_id"])
    host._reply = lambda kind, sid, request_id, **payload: replies.append({
        "type": kind, "sid": sid, "request_id": request_id, **payload,
    })
    supervisor = HostSupervisor.__new__(HostSupervisor)
    supervisor.start = lambda: None
    supervisor._send_frame = lambda _frame: None
    supervisor._await_reply = lambda frame, _request_id, _timeout: (
        host._handle_control(frame), replies.pop())[-1]
    return supervisor.control("sid-fast", route_name="slash.fast", payload={"command": command})


@pytest.mark.parametrize("command", ["/fast", "/fast status"])
def test_slash_readback_forms_use_live_tier(monkeypatch, command):
    session = {"agent": _agent(tier="flex"), "slash_worker": _StaleWorker()}

    response = _slash(monkeypatch, session, command)

    assert "flex" in response["result"]["output"].lower()


def test_slash_readback_uses_per_model_precedence(monkeypatch):
    agent = _agent(tier="priority", model="openai/gpt-5.4")
    session = {"agent": agent, "slash_worker": _StaleWorker()}
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {
        "service_tier": "priority", "service_tier_overrides": {"gpt-5.4": "flex"},
    }})

    response = _slash(monkeypatch, session, "/fast status")

    assert response["result"]["output"] == "Fast mode: flex."


def test_setting_then_status_reads_updated_live_tier(monkeypatch):
    session = {"agent": _agent(), "slash_worker": _StaleWorker()}

    changed = _slash(monkeypatch, session, "/fast flex --session")
    status = _slash(monkeypatch, session, "/fast status")

    assert changed["result"]["output"] == "Fast mode set to flex."
    assert status["result"]["output"] == "Fast mode: flex."
    assert session["agent"].service_tier == "flex"


def test_global_save_failure_is_reported_as_session_only(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    original = "agent:\n  service_tier: priority\n"
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session = {"agent": _agent(tier="priority"), "slash_worker": _StaleWorker(),
               "profile_home": str(tmp_path)}

    def fail_write(*_args):
        raise OSError("read-only config")

    monkeypatch.setattr(server, "_save_cfg", fail_write)
    response = _slash(monkeypatch, session, "/fast flex --global")

    assert "session only" in response["result"]["output"].lower()
    assert session["agent"].service_tier == "flex"
    assert session["create_service_tier_override"] == "flex"
    assert config_path.read_text(encoding="utf-8") == original


def test_supervisor_classifies_fast_control_route():
    supervisor = HostSupervisor.__new__(HostSupervisor)
    sent = []
    supervisor.start = lambda: None
    supervisor._send_frame = sent.append

    result = supervisor.control("sid-fast", route_name="slash.fast", payload={"command": "/fast normal"}, wait=False)

    assert result["status"] == "sent"
    assert sent[0]["route_name"] == "slash.fast"


@pytest.mark.parametrize("mode", ["flex", "auto", "cold", "normal"])
def test_compute_host_fast_control_returns_accepted_result(monkeypatch, mode):
    agent = _agent(tier="priority", provider="openrouter", base_url="https://openrouter.ai/api/v1")
    session = {"agent": agent, "session_key": "key", "history": [], "history_lock": threading.RLock(),
               "_compute_host_active": True}
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_write_config_key", lambda *_args: None)
    monkeypatch.setattr(server, "_send_compute_host_control",
                        lambda _sid, **kwargs: _host_ack(monkeypatch, session, kwargs["command"]))

    ack = _slash(monkeypatch, session, f"/fast {mode} --session")["result"]

    assert mode in ack["output"].lower()
    assert agent.service_tier == {"normal": None}.get(mode, mode)


@pytest.mark.parametrize("mode", ["flex", "auto", "cold"])
def test_compute_host_fast_control_refusal_does_not_mutate(monkeypatch, mode):
    agent = _agent(tier="priority", model="Qwen3.8-27B-UD-Q4_K_M", provider="llamacpp",
                   base_url="http://127.0.0.1:18434/v1")
    session = {"agent": agent, "session_key": "key", "history": [], "history_lock": threading.RLock(),
               "create_service_tier_override": "priority", "_compute_host_active": True}
    writes = []
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: writes.append("persist"))
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_send_compute_host_control",
                        lambda _sid, **kwargs: _host_ack(monkeypatch, session, kwargs["command"]))

    ack = _slash(monkeypatch, session, f"/fast {mode} --session")["result"]

    assert "not available" in ack["output"].lower()
    assert agent.service_tier == "priority"
    assert session["create_service_tier_override"] == "priority"
    assert not writes


def test_compute_host_fast_global_persists_config_and_reports_scope(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("agent:\n  service_tier: priority\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent(tier="priority", provider="openrouter", base_url="https://openrouter.ai/api/v1")
    session = {"agent": agent, "session_key": "key", "history": [], "history_lock": threading.RLock(),
               "create_service_tier_override": "priority", "_compute_host_active": True,
               "profile_home": str(tmp_path)}
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_send_compute_host_control",
                        lambda _sid, **kwargs: _host_ack(monkeypatch, session, kwargs["command"]))

    ack = _slash(monkeypatch, session, "/fast flex --global")["result"]

    import yaml
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {"agent": {"service_tier": "flex"}}
    assert "global" in ack["output"].lower()
    assert "create_service_tier_override" not in session
    assert agent.service_tier == "flex"


@pytest.mark.parametrize("command, mode, pin", [
    ("/fast normal --session", "normal", ""),
    ("/fast auto --session", "auto", "auto"),
    ("/fast cold --session", "cold", "cold"),
    ("/fast flex --session", "flex", "flex"),
])
def test_compute_host_fast_pin_crosses_parent_boundary(monkeypatch, command, mode, pin):
    child = {"agent": _agent(tier="priority", provider="openrouter",
                             base_url="https://openrouter.ai/api/v1"),
             "session_key": "key", "history": [], "history_lock": threading.RLock(),
             "create_service_tier_override": "priority"}
    parent = {"agent": None, "session_key": "key", "history": [],
              "history_lock": threading.RLock(), "create_service_tier_override": "priority",
              "_compute_host_active": True}
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_write_config_key", lambda *_args: None)
    monkeypatch.setattr(server, "_send_compute_host_control",
                        lambda _sid, **kwargs: _host_ack(monkeypatch, child, kwargs["command"]))

    response = _slash(monkeypatch, parent, command)["result"]["output"]

    assert f"Fast mode set to {mode}." in response
    assert parent.get("create_service_tier_override") == pin
    assert server._compute_host_turn_frame("rid", "sid-fast", parent, "next turn")[
        "service_tier_override"] == pin


def test_compute_host_fast_global_unpin_crosses_parent_and_restart_boundary(monkeypatch):
    child = {"agent": _agent(tier="priority", provider="openrouter",
                             base_url="https://openrouter.ai/api/v1"),
             "session_key": "key", "history": [], "history_lock": threading.RLock(),
             "create_service_tier_override": "priority"}
    parent = {"agent": None, "session_key": "key", "history": [],
              "history_lock": threading.RLock(), "create_service_tier_override": "priority",
              "_compute_host_active": True}
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_write_config_key", lambda *_args: None)
    monkeypatch.setattr(server, "_send_compute_host_control",
                        lambda _sid, **kwargs: _host_ack(monkeypatch, child, kwargs["command"]))

    response = _slash(monkeypatch, parent, "/fast normal --global")["result"]["output"]

    assert "globally" in response
    assert "create_service_tier_override" not in parent
    frame = server._compute_host_turn_frame("rid", "sid-fast", parent, "next turn")
    assert frame["service_tier_override"] is None


def test_compute_host_fast_failed_global_save_mirrors_session_pin(monkeypatch):
    child = {"agent": _agent(tier="priority", provider="openrouter",
                             base_url="https://openrouter.ai/api/v1"),
             "session_key": "key", "history": [], "history_lock": threading.RLock(),
             "create_service_tier_override": "priority"}
    parent = {"agent": None, "session_key": "key", "history": [],
              "history_lock": threading.RLock(), "create_service_tier_override": "priority",
              "_compute_host_active": True}
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    def fail_save(*_args):
        raise OSError("readonly")

    monkeypatch.setattr(server, "_write_config_key", fail_save)
    monkeypatch.setattr(server, "_send_compute_host_control",
                        lambda _sid, **kwargs: _host_ack(monkeypatch, child, kwargs["command"]))

    response = _slash(monkeypatch, parent, "/fast flex --global")["result"]["output"]

    assert "session only" in response
    assert parent["create_service_tier_override"] == "flex"
    assert server._compute_host_turn_frame("rid", "sid-fast", parent, "next turn")[
        "service_tier_override"] == "flex"


def test_compute_host_fast_refusal_preserves_parent_pin(monkeypatch):
    child = {"agent": _agent(tier="priority", model="Qwen3.8-27B-UD-Q4_K_M", provider="llamacpp",
                             base_url="http://127.0.0.1:18434/v1"),
             "session_key": "key", "history": [], "history_lock": threading.RLock(),
             "create_service_tier_override": "priority"}
    parent = {"agent": None, "session_key": "key", "history": [],
              "history_lock": threading.RLock(), "create_service_tier_override": "priority",
              "_compute_host_active": True}
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: pytest.fail("refusal persisted"))
    monkeypatch.setattr(server, "_send_compute_host_control",
                        lambda _sid, **kwargs: _host_ack(monkeypatch, child, kwargs["command"]))

    response = _slash(monkeypatch, parent, "/fast flex --session")["result"]["output"]

    assert "not available" in response.lower()
    assert parent["create_service_tier_override"] == "priority"


@pytest.mark.parametrize("command, expected, pin", [
    ("/fast normal --session", "normal", ""), ("/fast status", "fast", None)])
def test_lazy_compute_host_fast_works_before_first_turn(monkeypatch, command, expected, pin):
    parent = {"agent": None, "agent_ready": threading.Event(), "session_key": "lazy-key",
              "history": [], "history_lock": threading.RLock()}
    forwarded = []
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *_args: (
        "gpt-5.4", {"provider": "openai", "base_url": "https://api.openai.com/v1", "api_key": None}))
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"service_tier": "priority"}})
    monkeypatch.setattr(server, "_send_compute_host_control", lambda *_args, **_kwargs: forwarded.append(True))

    response = _slash(monkeypatch, parent, command)["result"]["output"]

    assert expected.lower() in response.lower()
    assert not forwarded
    if pin is not None:
        assert parent["create_service_tier_override"] == pin
        assert "normal" in _slash(monkeypatch, parent, "/fast status")["result"]["output"].lower()


def test_lazy_fast_pin_is_used_when_compute_host_builds_first_agent(monkeypatch):
    parent = {"agent": None, "agent_ready": threading.Event(), "session_key": "lazy-key",
              "history": [], "history_lock": threading.RLock(), "model_override": None,
              "create_service_tier_override": ""}
    frame = server._compute_host_turn_frame("rid", "sid-fast", parent, "first turn")
    captured = {}

    class _Server:
        _sessions = {}

        @staticmethod
        def _make_agent(*_args, **kwargs):
            captured.update(kwargs)
            return object()

        @staticmethod
        def _transfer_db_to_agent(*_args):
            return False

        @staticmethod
        def _init_session(sid, key, agent, history, **_kwargs):
            _Server._sessions[sid] = {"agent": agent, "session_key": key, "history": history,
                                      "history_lock": threading.RLock()}

    host = ComputeHost.__new__(ComputeHost)
    host._transport = object()
    monkeypatch.setattr("tui_gateway.transport.bind_transport", lambda *_args: None)
    monkeypatch.setattr("tui_gateway.transport.reset_transport", lambda *_args: None)

    host._build_server_session(_Server, frame, "sid-fast")

    assert captured["service_tier_override"] == ""
