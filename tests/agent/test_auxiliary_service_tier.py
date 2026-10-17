"""Service-tier source precedence and auxiliary request mapping contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize(("agent_cfg", "slot", "expected"), [
    ({"service_tier": "priority"}, {}, "priority"),
    ({"service_tier": "priority", "service_tier_overrides": {"gpt-5.4": "flex"}}, {}, "flex"),
    ({"service_tier": "priority"}, {"service_tier": "flex"}, "flex"),
    ({"service_tier": "priority"}, {"service_tier": "priority", "extra_body": {"service_tier": "flex"}}, "flex"),
    ({"service_tier": "priority"}, {"service_tier": "normal"}, None),
    ({"service_tier": "priority"}, {"service_tier": "default"}, None),
    ({"service_tier": "priority"}, {"service_tier": "off"}, None),
    ({"service_tier": "priority"}, {"service_tier": ""}, "priority"),
    ({"service_tier": "priority"}, {"service_tier": "auto"}, None),
    ({"service_tier": "priority"}, {"service_tier": "cold"}, None),
    ({"service_tier": "priority"}, {"service_tier": "invalid"}, "priority"),
    ({"service_tier": "invalid"}, {}, None),
    ({"service_tier": "auto"}, {}, None),
    ({"service_tier": "cold"}, {}, None),
    ({"service_tier": "priority", "service_tier_overrides": {"gpt-5.4": "normal"}}, {}, None),
    ({"service_tier": "priority", "service_tier_overrides": {"gpt-5.4": "invalid"}}, {}, "priority"),
])
def test_aux_service_tier_sources_resolve_for_actual_openai_route(agent_cfg, slot, expected):
    from agent.auxiliary_client import _build_call_kwargs, _get_task_extra_body

    config = {"agent": agent_cfg, "auxiliary": {"compression": slot}}
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        body = _get_task_extra_body("compression")
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-5.4", messages=[{"role": "user", "content": "hi"}],
            extra_body=body, base_url="https://api.openai.com/v1", task="compression",
        )
    assert kwargs.get("service_tier") == expected
    assert "speed" not in kwargs
    assert "service_tier" not in (kwargs.get("extra_body") or {})


def test_agent_tier_applies_without_auxiliary_task_name(monkeypatch):
    from agent.auxiliary_client import _build_call_kwargs

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"agent": {"service_tier": "priority"}})
    kwargs = _build_call_kwargs(
        provider="openai", model="gpt-5.4", messages=[{"role": "user", "content": "hi"}],
        base_url="https://api.openai.com/v1",
    )
    assert kwargs["service_tier"] == "priority"


@pytest.mark.parametrize("empty_tier", [None, ""])
def test_empty_extra_body_tier_is_omitted_without_mutating_input(monkeypatch, empty_tier):
    from agent.auxiliary_client import _build_call_kwargs

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    body = {"service_tier": empty_tier, "keep": True}
    kwargs = _build_call_kwargs(
        provider="openai", model="gpt-5.4", messages=[{"role": "user", "content": "hi"}],
        extra_body=body, base_url="https://api.openai.com/v1", task="compression",
    )
    assert kwargs["extra_body"] == {"keep": True}
    assert body == {"service_tier": empty_tier, "keep": True}


def test_fallback_request_resolves_tier_for_its_actual_model(monkeypatch):
    from agent import auxiliary_client as aux

    cfg = {
        "agent": {"service_tier": "priority", "service_tier_overrides": {"openai/gpt-5": "flex"}},
        "auxiliary": {"compression": {}},
    }
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    monkeypatch.setattr(aux, "_replan_synchronous_cache_sections", lambda messages, tools, **_k: (messages, tools or []))

    def fallback_kwargs(model):
        return aux._fallback_request_kwargs(
            aux._FallbackDestination("openrouter", "https://openrouter.ai/api/v1", None, model),
            task="compression", messages=[{"role": "user", "content": "hi"}], tools=None,
            temperature=None, max_tokens=None, effective_timeout=30,
            effective_extra_body={}, reasoning_config=None, fallback_entry={}, task_config={},
            apply_fast_lane=False,
        )

    assert fallback_kwargs("openai/gpt-5")["service_tier"] == "flex"
    assert fallback_kwargs("meta-llama/llama-3.1-8b-instruct")["service_tier"] == "priority"


def test_aux_service_tier_uses_real_profile_config_and_plugin_inheritance(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from agent.auxiliary_client import _build_call_kwargs

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "agent": {"service_tier": "priority"},
        "auxiliary": {"inherited_base": {"service_tier": "flex"}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_auxiliary_tasks",
        lambda: [{"key": "inherited_task", "defaults": {}, "inherit_from": "inherited_base"}],
    )

    kwargs = _build_call_kwargs(
        provider="openai", model="gpt-5.4", messages=[{"role": "user", "content": "hi"}],
        base_url="https://api.openai.com/v1", task="inherited_task",
    )
    assert kwargs["service_tier"] == "flex"


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [("openai", "gpt-5.4", "https://api.openai.com/v1"),
     ("openrouter", "openai/gpt-5", "https://openrouter.ai/api/v1")],
)
def test_aux_service_tier_reaches_sync_async_and_stream_request_kwargs(monkeypatch, provider, model, base_url):
    from agent import auxiliary_client as aux

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"agent": {"service_tier": "priority"}})
    client = MagicMock()
    client.base_url = base_url
    monkeypatch.setattr(
        aux, "_resolve_task_provider_model",
        lambda *_a: (provider, model, client.base_url, "key", None),
    )
    monkeypatch.setattr(
        aux, "_resolve_call_client",
        lambda *_a, **_k: aux._ResolvedAuxRoute(client, model, provider, provider),
    )

    def prepare(async_mode):
        return aux._prepare_aux_request(
            "title_generation", provider=provider, model=model, base_url=client.base_url,
            api_key="key", main_runtime={}, messages=[{"role": "user", "content": "hi"}],
            temperature=None, max_tokens=None, tools=None, timeout=None, extra_body=None,
            reasoning_config=None, extra_headers=None, api_mode=None, route_info=None,
            async_mode=async_mode,
        )

    sync_request = prepare(False)
    assert sync_request.kwargs["service_tier"] == "priority"
    assert prepare(True).kwargs["service_tier"] == "priority"

    sent = {}
    monkeypatch.setattr(aux, "_plan_aux_call", lambda *_a, **_k: (sync_request, {}, {}))
    monkeypatch.setattr(
        aux, "_relay_sync_stream", lambda _client, kwargs, **_k: (sent.update(kwargs), iter(()))[1],
    )
    aux._call_llm_impl(task="title_generation", messages=[], stream=True)
    assert sent["service_tier"] == "priority"
    assert sent["stream"] is True


def test_anthropic_slot_tier_survives_framework_binding():
    from agent.fast_mode import effective_request_overrides
    from hermes_cli.models_service_tiers import apply_aux_service_tier_overrides, bind_aux_slot_service_tier

    slot = {"service_tier": "priority"}
    with patch("hermes_cli.config.load_config_readonly", return_value={"agent": {"service_tier": "flex"}}):
        mapped = apply_aux_service_tier_overrides(
            {}, slot, model="claude-opus-5", provider="anthropic",
            base_url="https://api.anthropic.com", task="review",
        )
    assert mapped["speed"] == "fast"
    agent = SimpleNamespace(
        request_overrides=mapped, model="claude-opus-5", provider="anthropic",
        base_url="https://api.anthropic.com", service_tier=None,
    )
    bind_aux_slot_service_tier(agent, slot)
    assert agent.service_tier == "priority"
    assert effective_request_overrides(agent)["speed"] == "fast"


def test_aux_slot_logical_tier_survives_route_changes(monkeypatch):
    from agent.fast_mode import effective_request_overrides
    from hermes_cli.models_service_tiers import apply_aux_service_tier_overrides, bind_aux_slot_service_tier

    slot = {"service_tier": "flex"}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"agent": {"service_tier": "priority"}})
    mapped = apply_aux_service_tier_overrides(
        {}, slot, model="claude-opus-5", provider="anthropic",
        base_url="https://api.anthropic.com", task="review",
    )
    agent = SimpleNamespace(
        request_overrides=mapped, model="claude-opus-5", provider="anthropic",
        base_url="https://api.anthropic.com", service_tier=None,
    )
    bind_aux_slot_service_tier(agent, slot)

    for model, provider, base_url, expected in (
        ("gpt-5.4", "openai", "https://api.openai.com/v1", "flex"),
        ("claude-opus-5", "anthropic", "https://api.anthropic.com", None),
        ("gpt-5.4", "openai", "https://api.openai.com/v1", "flex"),
    ):
        agent.model, agent.provider, agent.base_url = model, provider, base_url
        assert effective_request_overrides(agent).get("service_tier") == expected


def test_aux_global_bounded_mode_does_not_reopen_window_in_agent_turn(monkeypatch):
    from agent.fast_mode import begin_turn, effective_request_overrides
    from hermes_cli.models_service_tiers import bind_aux_slot_service_tier

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"agent": {"service_tier": "auto"}})
    agent = SimpleNamespace(
        request_overrides={}, model="gpt-5.4", provider="openai",
        base_url="https://api.openai.com/v1", service_tier="auto",
        _block_service_tier_escalation=True,
    )
    bind_aux_slot_service_tier(agent, {})
    begin_turn(agent, [])
    assert agent._service_tier_session_pinned is False
    assert agent.service_tier is None
    assert "service_tier" not in effective_request_overrides(agent)


def test_aux_global_and_model_tiers_follow_current_aux_model(monkeypatch):
    from agent.fast_mode import begin_turn, effective_request_overrides
    from hermes_cli.models_service_tiers import bind_aux_slot_service_tier

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "agent": {
            "service_tier": "priority",
            "service_tier_overrides": {
                "gpt-5.4": "flex", "gpt-5.5": "normal", "gpt-5.6": "auto",
            },
        },
    })
    agent = SimpleNamespace(
        request_overrides={}, model="gpt-5.4", provider="openai",
        base_url="https://api.openai.com/v1", service_tier="priority",
    )
    bind_aux_slot_service_tier(agent, {})
    begin_turn(agent, [])
    assert effective_request_overrides(agent)["service_tier"] == "flex"
    for model, expected in (("gpt-5.5", None), ("gpt-5.6", None), ("gpt-5.7", "priority")):
        agent.model = model
        assert effective_request_overrides(agent).get("service_tier") == expected


@pytest.mark.parametrize("bounded_tier", ["auto", "cold"])
@pytest.mark.parametrize("review", [{}, {"provider": "openai", "model": "gpt-5.4"}])
def test_review_bounded_tier_is_suppressed_in_final_wire_kwargs(monkeypatch, bounded_tier, review):
    from agent.fast_mode import begin_turn
    from hermes_cli.models_service_tiers import bind_aux_slot_service_tier
    from run_agent import AIAgent

    from agent.review_engine import _load_review_credentials_cfg

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "agent": {"service_tier": bounded_tier}, "auxiliary": {"review": review},
    })
    credentials_cfg = _load_review_credentials_cfg()
    slot = (credentials_cfg or {}).get("_aux_slot", {})
    agent = AIAgent(
        api_key="fixture-only", base_url="https://api.openai.com/v1",
        provider="openai", api_mode="chat_completions",
        model=(credentials_cfg or {}).get("model") or "gpt-5.4",
        service_tier=bounded_tier, quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False, enabled_toolsets=["file"],
    )
    try:
        bind_aux_slot_service_tier(agent, slot)
        begin_turn(agent, [])
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "review"}])
        assert "service_tier" not in kwargs
        assert "speed" not in kwargs
    finally:
        agent.close()


def test_aux_explicit_normal_binding_stops_global_fallback(monkeypatch):
    from agent.fast_mode import effective_request_overrides
    from hermes_cli.models_service_tiers import apply_aux_service_tier_overrides, bind_aux_slot_service_tier

    cfg = {"agent": {"service_tier": "priority"}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    slot = {"service_tier": "normal"}
    overrides = apply_aux_service_tier_overrides(
        {}, slot, model="gpt-5.4", provider="openai",
        base_url="https://api.openai.com/v1", task="review",
    )
    agent = SimpleNamespace(
        request_overrides=overrides, model="gpt-5.4", provider="openai",
        base_url="https://api.openai.com/v1", service_tier="priority",
    )
    bind_aux_slot_service_tier(agent, slot)
    assert agent._service_tier_session_pinned is True
    assert "service_tier" not in effective_request_overrides(agent)
