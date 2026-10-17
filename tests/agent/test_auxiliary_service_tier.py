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
