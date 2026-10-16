"""The TUI/desktop agent build path must SEND the configured service tier.

``_make_agent`` passed ``service_tier=`` to ``AIAgent`` but never
``request_overrides=``. Only the runtime ``/fast`` toggle
(``config.set key=fast``) ever populated ``agent.request_overrides``, and
``agent/transports/chat_completions.py`` emits ``service_tier`` solely from
those overrides.

So a tier set in config.yaml was accepted, displayed in session info, and
then silently dropped on the wire for every TUI/desktop session — for
``priority`` as much as for ``flex``. These tests pin the emitted kwargs, not
just the config resolution.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hermes_cli.config import DEFAULT_CONFIG, atomic_config_write

# Imported for real (NOT via patch.dict(sys.modules)): a sys.modules patch
# would evict every module first-imported inside its window on exit,
# leaving later tests with split-brain duplicates of tui_gateway.entry /
# hermes_cli.mcp_startup whose module globals nobody else reads.
import run_agent

from tui_gateway import server


def _build(tier: str | None, model: str = "gpt-5.4", provider: str = "openai",
           base_url: str = "https://api.openai.com/v1", runtime_request_overrides=None,
           service_tier_override: str | None = None):
    """Call _make_agent with the runtime stubbed out, returning AIAgent kwargs.

    The default runtime is a first-party OpenAI route. OpenRouter supports its own
    service tiers for any catalog model; other aggregators remain gated.
    """
    captured = {}

    class _Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            captured["agent"] = self

    resolution = MagicMock()
    resolution.used_fallback = False
    resolution.selected_model = model
    resolution.runtime = {
        "provider": provider,
        "base_url": base_url,
        "api_key": "***",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "request_overrides": runtime_request_overrides,
    }

    with (
        patch.object(run_agent, "AIAgent", _Agent),
        # _make_agent waits on MCP discovery; running the real wait would
        # initialize the process-wide discovery state under this test run's
        # HERMES_HOME and pollute later profile-scoped discovery tests.
        patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"),
        patch.object(server, "_load_service_tier", return_value=tier),
        patch.object(server, "_resolve_startup_runtime", return_value=(model, provider)),
        patch.object(server, "_resolve_runtime_with_fallback", return_value=resolution),
        patch.object(server, "_resolve_model", return_value=model),
        patch.object(server, "_agent_cbs", return_value={}),
        patch.object(server, "_get_db", return_value=None),
        patch.object(server, "_load_reasoning_config", return_value=None),
        patch.object(server, "_load_provider_routing", return_value={}),
        patch.object(server, "_load_fallback_model", return_value=None),
        patch.object(server, "_load_enabled_toolsets", return_value=None),
    ):
        server._make_agent("sid-1", "key-1", service_tier_override=service_tier_override)

    return captured


def test_flex_from_config_reaches_request_overrides():
    kwargs = _build("flex")

    assert kwargs["service_tier"] == "flex"
    assert kwargs["request_overrides"] == {"service_tier": "flex"}


def test_session_tier_pin_provenance_is_applied_when_agent_is_built():
    pinned = _build("priority", service_tier_override="")
    assert pinned["agent"]._service_tier_session_pinned is True
    assert not pinned["agent"]._framework_baked_tier_keys

    configured = _build("priority")
    assert configured["agent"]._service_tier_session_pinned is False
    assert configured["agent"]._framework_baked_tier_keys == frozenset({"service_tier"})


def test_background_agent_uses_a_frozen_service_tier_snapshot(monkeypatch):
    from tui_gateway import server

    for name, value in {
        "_load_cfg": dict, "_load_service_tier": lambda: "priority", "_resolve_model": lambda: "gpt-5.4",
        "_cfg_max_turns": lambda *_args: 25, "_load_enabled_toolsets": lambda *_args: ["file"],
        "_load_disabled_toolsets": list, "_load_reasoning_config": lambda *_args: None,
        "_get_db": lambda: None,
    }.items():
        monkeypatch.setattr(server, name, value)
    parent = SimpleNamespace(
        _service_tier_session_pinned=True, service_tier="flex",
        _framework_baked_tier_keys=frozenset({"service_tier"}),
        request_overrides={"service_tier": "flex", "extra_body": {"keep": 1}},
        _fallback_chain=[],
    )

    snap = server._background_tier_snapshot(parent)
    kwargs = server._background_agent_kwargs(parent, "bg-1", snap)
    side_agent = SimpleNamespace(**kwargs)
    server._apply_background_tier_provenance(side_agent, parent, snap)

    assert kwargs["service_tier"] == "flex"
    assert kwargs["request_overrides"] == {"service_tier": "flex", "extra_body": {"keep": 1}}
    assert side_agent._service_tier_session_pinned is True
    assert side_agent._framework_baked_tier_keys == frozenset({"service_tier"})


def test_resume_preserves_legacy_unmarked_service_tier(monkeypatch):
    from hermes_state import SessionDB
    from tui_gateway import server

    monkeypatch.setattr(server, "_row_follows_profile", lambda _row: False)
    monkeypatch.setattr(
        SessionDB, "session_gateway_runtime",
        staticmethod(lambda _row: {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}),
    )
    row = {
        "id": "legacy-session", "title": "Legacy conversation", "model": "gpt-5.4",
        "billing_provider": "openrouter",
        "model_config": '{"service_tier":"flex","base_url":"https://openrouter.ai/api/v1"}',
    }

    assert server._stored_session_runtime_overrides(row)["service_tier_override"] == "flex"


def test_tier_merges_with_custom_provider_request_body():
    """#127015 passes the resolved entry's request body through _make_agent; the tier
    merges ON TOP of it (the two-dicts merge that PR's notes call for): the entry's
    extra_body passes through untouched and the tier adds only its own top-level keys."""
    kwargs = _build("flex", runtime_request_overrides={"extra_body": {"user": "proxy-user"}})

    assert kwargs["request_overrides"] == {
        "extra_body": {"user": "proxy-user"},
        "service_tier": "flex",
    }


def test_custom_provider_request_body_survives_without_tier():
    """No tier configured must not disturb #127015's pass-through."""
    kwargs = _build(None, runtime_request_overrides={"extra_body": {"user": "proxy-user"}})

    assert kwargs["request_overrides"] == {"extra_body": {"user": "proxy-user"}}


def test_priority_from_config_reaches_request_overrides():
    """Pre-existing gap: even priority never reached the wire from config."""
    kwargs = _build("priority")

    assert kwargs["request_overrides"] == {"service_tier": "priority"}


def test_session_tier_override_beats_profile_config_at_build():
    kwargs = _build("priority", service_tier_override="flex")

    assert kwargs["service_tier"] == "flex"
    assert kwargs["request_overrides"] == {"service_tier": "flex"}


def test_explicit_session_normal_drops_profile_tier_at_build():
    kwargs = _build("flex", service_tier_override="")

    assert kwargs["service_tier"] == ""
    assert not kwargs.get("request_overrides")


def test_tui_build_does_not_pin_denied_codex_ultrafast(monkeypatch):
    import agent.codex_service_tiers as codex_tiers
    from hermes_cli import models

    monkeypatch.setattr(models, "resolve_fast_mode_overrides", lambda *a, **k: {"service_tier": "ultrafast"})
    monkeypatch.setattr(codex_tiers, "ultrafast_offered", lambda *a, **k: False)
    kwargs = _build("ultrafast", model="gpt-6-astra", provider="openai-codex",
                    base_url="https://chatgpt.com/backend-api/codex")

    assert kwargs["service_tier"] == "ultrafast"
    assert not kwargs.get("request_overrides")


def test_no_tier_sends_no_overrides():
    kwargs = _build(None)

    assert not kwargs.get("request_overrides")


def test_anthropic_flex_sends_nothing():
    """Anthropic has no flex equivalent — must not fall back to speed=fast."""
    kwargs = _build("flex", model="claude-opus-4-8",
                    provider="anthropic", base_url="https://api.anthropic.com")

    assert not kwargs.get("request_overrides")


def test_openrouter_route_accepts_catalog_wide_flex():
    """OpenRouter accepts service tiers for any catalog model."""
    kwargs = _build("flex", model="gpt-4.1",
                    provider="openrouter", base_url="https://openrouter.ai/api/v1")

    assert kwargs["request_overrides"] == {"service_tier": "flex"}


def test_anthropic_priority_sends_speed_not_service_tier():
    """Anthropic Fast Mode uses ``speed``; the resolver must pick per provider."""
    kwargs = _build("priority", model="claude-opus-4-8",
                    provider="anthropic", base_url="https://api.anthropic.com")

    assert kwargs["request_overrides"] == {"speed": "fast"}


def test_ineligible_model_sends_no_overrides():
    """A tier set for a model with no tier support must not invent one."""
    kwargs = _build("priority", model="gpt-5.3-codex")

    assert not kwargs.get("request_overrides")


@pytest.mark.parametrize("window_tier", ["auto", "cold"])
def test_window_tiers_are_not_pinned_at_build_time(window_tier):
    """auto/cold are bounded windows applied per request by ``agent.fast_mode`` —
    pinning them at build time would bill a fixed tier the user never chose
    (caught in review on #101524)."""
    kwargs = _build(window_tier)

    assert kwargs["service_tier"] == window_tier
    assert not kwargs.get("request_overrides")


def _mirror(agent, arg: str):
    """Run the typed-slash `/fast <arg>` mirror against a live agent."""
    session = {"agent": agent}
    with patch.object(server, "_emit"), patch.object(server, "_session_info", return_value={}):
        server._mirror_slash_side_effects("sid-1", session, f"/fast {arg}")
    return agent


def test_typed_fast_off_clears_stale_tier_overrides():
    """`/fast off` must stop sending the tier, not just relabel the session.

    A session built with a configured tier now carries request_overrides. If
    the typed-slash mirror only clears `service_tier`, the tier keeps going
    out on the wire while session info reports normal.
    """
    agent = SimpleNamespace(
        model="gpt-5.4",
        service_tier="flex",
        request_overrides={"service_tier": "flex"},
    )

    _mirror(agent, "off")

    assert agent.service_tier is None
    assert not agent.request_overrides


def test_typed_fast_on_sends_priority():
    agent = SimpleNamespace(
        model="gpt-5.4",
        service_tier=None,
        request_overrides={},
    )

    _mirror(agent, "on")

    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"service_tier": "priority"}


def test_typed_fast_on_uses_provider_appropriate_key():
    """Anthropic uses ``speed``, not ``service_tier`` — and must not carry both:
    popping both keys before re-resolving makes a mismatched pair unrepresentable."""
    agent = SimpleNamespace(
        model="claude-opus-4-8",
        service_tier=None,
        request_overrides={"service_tier": "priority"},
    )

    _mirror(agent, "on")

    assert agent.request_overrides == {"speed": "fast"}


def test_typed_fast_flex_sets_static_tier_and_preserves_extra_body():
    agent = SimpleNamespace(
        model="gpt-5.4",
        provider="openai",
        base_url="https://api.openai.com/v1",
        service_tier="priority",
        request_overrides={"extra_body": {"tenant": "custom"}, "service_tier": "priority"},
    )

    _mirror(agent, "flex")

    assert agent.service_tier == "flex"
    assert agent.request_overrides == {"extra_body": {"tenant": "custom"}, "service_tier": "flex"}


def _slash_exec(monkeypatch, *, agent, command, output):
    from tui_gateway import server

    class _Worker:
        def run(self, _command):
            return output

    session = {
        "agent": agent,
        "session_key": "fast-session-key",
        "slash_worker": _Worker(),
        "history": [],
        "profile_home": None,
        "create_service_tier_override": "priority",
    }
    server._sessions["fast-session"] = session
    monkeypatch.setattr(server, "_session_home_scope", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(server, "_profile_skill_command", lambda *_args: False)
    monkeypatch.setattr(server, "_plugin_command_handler", lambda *_args: None)
    monkeypatch.setattr(server, "_live_slash_command_output", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    persisted = []
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda s: persisted.append(dict(s)))
    try:
        response = server.handle_request({
            "id": "slash-fast",
            "method": "slash.exec",
            "params": {"session_id": "fast-session", "command": command},
        })
    finally:
        server._sessions.pop("fast-session", None)
    return response, session, persisted


@pytest.mark.parametrize("word", ["ultrafast", "ultra"])
def test_slash_exec_entitled_ultrafast_mirrors_exact_tier(monkeypatch, word):
    import agent.codex_service_tiers as codex_tiers
    from hermes_cli import models
    from tui_gateway import server

    agent = SimpleNamespace(
        model="gpt-6-astra", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", api_key="token",
        service_tier="priority", request_overrides={"service_tier": "priority", "other": "keep"},
    )
    monkeypatch.setattr(models, "resolve_fast_mode_overrides", lambda *a, **k: {"service_tier": "ultrafast"})
    monkeypatch.setattr(codex_tiers, "ultrafast_offered", lambda *a, **k: True)

    response, session, persisted = _slash_exec(monkeypatch, agent=agent, command=f"/fast {word}", output="set")

    assert response["result"]["output"] == "set"
    assert agent.service_tier == "ultrafast"
    assert agent.request_overrides == {"other": "keep", "service_tier": "ultrafast"}
    assert session["create_service_tier_override"] == "ultrafast"
    assert persisted


@pytest.mark.parametrize(
    ("command", "model", "route", "ultrafast_offered"),
    [
        ("/fast ultra", "gpt-6-astra", "openai-codex", False),
        ("/fast flex", "unsupported-model", "openai", True),
    ],
)
def test_slash_exec_refusal_keeps_existing_tier_state(monkeypatch, command, model, route, ultrafast_offered):
    import agent.codex_service_tiers as codex_tiers
    from hermes_cli import models

    agent = SimpleNamespace(
        model=model, provider=route, base_url="https://openrouter.ai/api/v1" if route == "openrouter" else None,
        api_key="token", service_tier="priority",
        request_overrides={"service_tier": "priority", "other": "keep"},
    )
    session_pin = "priority"
    # The resolver's real gate determines unsupported model/route cases; the ultrafast row
    # reaches the entitlement check and is refused there.
    if command.endswith("ultra"):
        monkeypatch.setattr(models, "resolve_fast_mode_overrides", lambda *a, **k: {"service_tier": "ultrafast"})
        monkeypatch.setattr(codex_tiers, "ultrafast_offered", lambda *a, **k: ultrafast_offered)
    response, session, persisted = _slash_exec(
        monkeypatch, agent=agent, command=command, output="not available for this model/account/route")

    assert "not available" in response["result"]["output"]
    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"service_tier": "priority", "other": "keep"}
    assert session["create_service_tier_override"] == session_pin
    assert not persisted


@pytest.mark.parametrize(
    ("command", "output"),
    [
        ("/fast flex garbage", "Unknown argument: flex garbage"),
        ("/fast on --bogus", "Unknown argument: on --bogus"),
    ],
)
def test_slash_exec_rejected_fast_arguments_leave_state_unchanged(monkeypatch, command, output):
    agent = SimpleNamespace(
        model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1",
        service_tier="priority", request_overrides={"service_tier": "priority", "other": "keep"},
    )

    response, session, persisted = _slash_exec(monkeypatch, agent=agent, command=command, output=output)

    assert response["result"]["output"] == output
    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"service_tier": "priority", "other": "keep"}
    assert session["create_service_tier_override"] == "priority"
    assert not persisted


def _write_fast_test_config(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "agent:\n  service_tier: priority\n  service_tier_overrides:\n    openai/gpt-5: flex\n",
        encoding="utf-8",
    )


def test_slash_fast_normal_pins_wire_and_resume_over_per_model_tier(monkeypatch, tmp_path):
    from agent import fast_mode

    _write_fast_test_config(monkeypatch, tmp_path)
    agent = SimpleNamespace(
        model="openai/gpt-5", provider="openrouter", base_url="https://openrouter.ai/api/v1",
        api_key="token", service_tier="flex", request_overrides={"service_tier": "flex"},
        _service_tier_session_pinned=False,
    )

    response, _session, persisted = _slash_exec(
        monkeypatch, agent=agent, command="/fast normal --session", output="(._.) /fast set to NORMAL")

    assert response["result"]["output"].startswith("(._.)")
    assert agent._service_tier_session_pinned is True
    assert fast_mode.effective_request_overrides(agent) == {}
    runtime = server._runtime_model_config(agent)
    assert runtime["service_tier"] == "normal"
    assert runtime["service_tier_session_pinned"] is True
    import json
    resumed = server._stored_session_runtime_overrides({"model_config": json.dumps(runtime)})
    assert resumed["service_tier_override"] == ""
    restored = _build(
        "priority", model="openai/gpt-5", provider="openrouter", base_url="https://openrouter.ai/api/v1",
        service_tier_override=resumed["service_tier_override"])
    restored_agent = SimpleNamespace(
        model=restored["model"], provider=restored["provider"], base_url=restored["base_url"],
        api_mode=restored["api_mode"], service_tier=restored["service_tier"],
        request_overrides=restored["request_overrides"] or {}, _service_tier_session_pinned=True,
        _framework_baked_tier_keys=restored["agent"]._framework_baked_tier_keys,
    )
    assert fast_mode.effective_request_overrides(restored_agent) == {}
    assert persisted


def test_slash_fast_global_unpins_live_agent_and_uses_per_model_resolution(monkeypatch, tmp_path):
    from agent import fast_mode

    _write_fast_test_config(monkeypatch, tmp_path)
    agent = SimpleNamespace(
        model="openai/gpt-5", provider="openrouter", base_url="https://openrouter.ai/api/v1",
        api_key="token", service_tier="priority", request_overrides={"service_tier": "priority"},
        _service_tier_session_pinned=True,
    )

    _, session, persisted = _slash_exec(
        monkeypatch, agent=agent, command="/fast fast --global", output="fast enabled")

    from hermes_cli.config import get_config_path, load_config_readonly

    assert load_config_readonly()["agent"]["service_tier"] == "priority"
    assert get_config_path().read_text(encoding="utf-8").startswith("agent:")
    assert "create_service_tier_override" not in session
    assert agent._service_tier_session_pinned is False
    assert fast_mode.effective_request_overrides(agent)["service_tier"] == "flex"
    assert "service_tier_session_pinned" not in server._runtime_model_config(agent)
    assert persisted


@pytest.mark.parametrize("command,model,provider,base_url,output", [
    ("/fast flex --global", "gpt-5.4", "openai", "https://proxy.example/v1",
     "(._.) /fast is only available for supported models/routes."),
    ("/fast ultra --global", "gpt-6-astra", "openai-codex", "https://chatgpt.com/backend-api/codex",
     "(._.) Ultrafast is not available for this account."),
])
def test_slash_fast_real_refusal_text_does_not_mutate_global_or_session(
        monkeypatch, command, model, provider, base_url, output):
    if command.startswith("/fast ultra"):
        import agent.codex_service_tiers as codex_tiers
        from hermes_cli import models
        monkeypatch.setattr(models, "resolve_service_tier_overrides", lambda *a, **k: {"service_tier": "ultrafast"})
        monkeypatch.setattr(codex_tiers, "ultrafast_offered", lambda *a, **k: False)
    writes = []
    monkeypatch.setattr(server, "_write_config_key", lambda *args: writes.append(args))
    agent = SimpleNamespace(
        model=model, provider=provider, base_url=base_url, api_key="token",
        service_tier="priority", request_overrides={"service_tier": "priority", "extra_body": {"keep": 1}},
        _service_tier_session_pinned=True,
    )

    response, session, persisted = _slash_exec(
        monkeypatch, agent=agent, command=command, output=output)

    assert response["result"]["output"] == output
    assert writes == []
    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"service_tier": "priority", "extra_body": {"keep": 1}}
    assert agent._service_tier_session_pinned is True
    assert session["create_service_tier_override"] == "priority"
    assert not persisted


def test_config_set_flex_is_session_scoped_and_preserves_custom_body(monkeypatch):
    from tui_gateway import server

    agent = SimpleNamespace(
        model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1",
        service_tier="priority", request_overrides={"extra_body": {"tenant": "custom"}, "speed": "stale"},
    )
    session = {"agent": agent, "session_key": "fast-session-key"}
    server._sessions["fast-session"] = session
    writes = []
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_write_config_key", lambda *args: writes.append(args))
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})

    try:
        response = server.handle_request({
            "id": "config-fast-flex",
            "method": "config.set",
            "params": {"session_id": "fast-session", "key": "fast", "value": "flex"},
        })
    finally:
        server._sessions.pop("fast-session", None)

    assert response["result"]["value"] == "flex"
    assert agent.service_tier == "flex"
    assert agent.request_overrides == {"extra_body": {"tenant": "custom"}, "service_tier": "flex"}
    assert session["create_service_tier_override"] == "flex"
    assert not writes


@pytest.mark.parametrize(
    ("provider", "accepted"),
    [
        ("openrouter", True),
    ],
)
def test_config_set_lazy_session_validates_its_selected_route(monkeypatch, tmp_path, provider, accepted):
    from tui_gateway import server

    home = _configure_profile_route(monkeypatch, tmp_path, provider, None)
    session = {
        "agent": None, "session_key": "fast-session-key",
        "profile_home": str(home),
        "model_override": {"model": "gpt-5.4", "provider": provider},
        "create_service_tier_override": "priority",
    }
    server._sessions["fast-session"] = session
    writes = []
    monkeypatch.setattr(server, "_write_config_key", lambda *args: writes.append(args))

    try:
        response = server.handle_request({
            "id": "config-fast-flex-lazy",
            "method": "config.set",
            "params": {"session_id": "fast-session", "key": "fast", "value": "flex"},
        })
    finally:
        server._sessions.pop("fast-session", None)

    if accepted:
        assert response.get("result", {}).get("value") == "flex", response
        assert session["create_service_tier_override"] == "flex"
    else:
        assert response["error"]["code"] == 4002
        assert session["create_service_tier_override"] == "priority"
    assert not writes


def _configure_profile_route(monkeypatch, tmp_path, provider, base_url):
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    for name in ("HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER", "HERMES_INFERENCE_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (home / ".env").write_text(
        "OPENAI_API_KEY=test-openai-key\nOPENROUTER_API_KEY=test-openrouter-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    model = {"provider": provider, "default": "gpt-5.4"}
    if base_url:
        model["base_url"] = base_url
    atomic_config_write(home / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "model": model,
        "agent": {"service_tier": "priority"},
    })
    from hermes_cli import runtime_provider
    monkeypatch.setattr(runtime_provider, "load_config", lambda: {
        "model": model,
        "agent": {"service_tier": "priority"},
    })
    return home


@pytest.mark.parametrize(
    ("provider", "base_url", "override", "accepted"),
    [
        ("openai-api", None, None, True),
        ("openrouter", None, None, True),
        ("openrouter", None, {"model": "gpt-5.4"}, True),
        ("openai-api", "https://proxy.example/v1", None, False),
    ],
)
def test_config_set_lazy_flex_uses_profile_default_and_partial_override_route(
    monkeypatch, tmp_path, provider, base_url, override, accepted
):
    from tui_gateway import server

    home = _configure_profile_route(monkeypatch, tmp_path, provider, base_url)
    if provider == "openrouter":
        # Startup selection has no provider hint without a launch seed; config.set must still
        # validate against the effective provider resolved from this profile's config.
        assert server._resolve_startup_runtime() == ("gpt-5.4", None)
    session = {"agent": None, "session_key": "fast-session-key", "profile_home": str(home),
               "create_service_tier_override": "priority"}
    if override is not None:
        session["model_override"] = override
    resolved_model, resolved_runtime = server._resolve_agent_model_runtime(override, None)
    assert resolved_model == "gpt-5.4"
    assert resolved_runtime.get("provider") == provider
    if base_url:
        assert resolved_runtime.get("base_url") == base_url
    server._sessions["fast-session"] = session
    writes = []
    monkeypatch.setattr(server, "_write_config_key", lambda *args: writes.append(args))

    try:
        response = server.handle_request({
            "id": "config-fast-flex-lazy-default",
            "method": "config.set",
            "params": {"session_id": "fast-session", "key": "fast", "value": "flex"},
        })
    finally:
        server._sessions.pop("fast-session", None)

    if accepted:
        assert response.get("result", {}).get("value") == "flex", response
        assert session["create_service_tier_override"] == "flex"
    else:
        assert response["error"]["code"] == 4002
        assert session["create_service_tier_override"] == "priority"
    assert not writes


@pytest.mark.parametrize(
    ("provider", "base_url", "accepted"),
    [
        ("openai-api", None, True),
        ("openrouter", None, True),
        ("openai-api", "https://proxy.example/v1", False),
    ],
)
def test_config_set_global_flex_uses_profile_route(monkeypatch, tmp_path, provider, base_url, accepted):
    from tui_gateway import server

    _configure_profile_route(monkeypatch, tmp_path, provider, base_url)
    writes = []
    monkeypatch.setattr(server, "_write_config_key", lambda *args: writes.append(args))

    response = server.handle_request({
        "id": "config-fast-flex-global",
        "method": "config.set",
        "params": {"key": "fast", "value": "flex"},
    })

    if accepted:
        assert response["result"]["value"] == "flex"
        assert writes == [("agent.service_tier", "flex")]
    else:
        assert response["error"]["code"] == 4002
        assert not writes


def test_config_set_unsupported_flex_does_not_mutate_session(monkeypatch):
    from tui_gateway import server

    agent = SimpleNamespace(
        model="claude-opus-4-8", provider="anthropic", base_url="https://api.anthropic.com",
        service_tier="priority", request_overrides={"speed": "fast", "extra_body": {"keep": True}},
    )
    session = {"agent": agent, "session_key": "fast-session-key", "create_service_tier_override": "priority"}
    server._sessions["fast-session"] = session
    monkeypatch.setattr(server, "_write_config_key", lambda *_args: None)
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})

    try:
        response = server.handle_request({
            "id": "config-fast-flex",
            "method": "config.set",
            "params": {"session_id": "fast-session", "key": "fast", "value": "flex"},
        })
    finally:
        server._sessions.pop("fast-session", None)

    assert response["error"]["code"] == 4002
    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"speed": "fast", "extra_body": {"keep": True}}
    assert session["create_service_tier_override"] == "priority"


def test_resolver_crash_fails_open_to_no_overrides():
    """A resolver bug must not break agent construction: fail open to an
    unpinned build rather than a failed one (coverage idea from #101524)."""
    with patch("hermes_cli.models.resolve_fast_mode_overrides", side_effect=RuntimeError("boom")):
        kwargs = _build("priority")

    assert kwargs["service_tier"] == "priority"
    assert not kwargs.get("request_overrides")
