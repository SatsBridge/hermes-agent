"""Interactive CLI handler for static and bounded fast modes."""

from __future__ import annotations

from hermes_cli.cli_agent_setup_mixin import _retire_agent
from hermes_cli.cli_commands_session_tools import _accent_line, _command_arg, _cp, _dim_line, _probe, _save, _t


_FAST_TIERS = {
    "fast": ("priority", "fast"), "on": ("priority", "fast"), "normal": (None, "normal"),
    "off": (None, "normal"), "auto": ("auto", "auto"), "cold": ("cold", "cold"),
    "ultrafast": ("ultrafast", "ultrafast"), "ultra": ("ultrafast", "ultrafast"),
    "flex": ("flex", "flex"),
}


def handle_fast_command(cli, cmd: str):
    """Handle /fast, session-scoped unless --global persists the config value."""
    from hermes_cli.cli_commands_mixin import _scope_outcome, _split_scope_flags

    if not cli._fast_command_available():
        return _cp(f"  {_t('fast.not_supported')}")
    model = getattr(getattr(cli, "agent", None), "model", None) or getattr(cli, "model", None)
    anthropic = _probe("hermes_cli.models", "_is_anthropic_fast_model", None, model)
    feature_name = _t("fast.feature_generic" if anthropic is None
                      else "fast.feature_anthropic" if anthropic else "fast.feature_openai")
    raw = _command_arg(cmd)
    usage = _dim_line(_t("fast.usage"))
    if not raw or raw.lower() == "status":
        from agent.fast_mode import service_tier_word
        status = service_tier_word(cli.service_tier)
        return _cp(_accent_line(_t("fast.status", feature=feature_name, status=status)), usage)
    arg, explicit_global = _split_scope_flags(raw)
    if arg not in _FAST_TIERS:
        return _cp(_dim_line(_t("shared.unknown_argument", arg=arg)), usage)
    if _FAST_TIERS[arg][0] == "ultrafast":
        if not _probe("agent.codex_service_tiers", "ultrafast_offered_for", False, cli, model):
            return _cp(_dim_line(_t("fast.ultrafast_not_supported", model=model or "?")), usage)
        feature_name = _t("fast.feature_ultrafast")
    elif _FAST_TIERS[arg][0] == "flex":
        from hermes_cli.models import resolve_fast_mode_overrides
        if resolve_fast_mode_overrides(
            model, provider=getattr(cli, "provider", None), base_url=getattr(cli, "base_url", None),
            tier="flex") is None:
            return _cp(_dim_line(_t("fast.not_supported")), usage)
        feature_name = _t("fast.feature_flex")
    cli.service_tier, saved_value = _FAST_TIERS[arg]
    _retire_agent(cli)
    saved = explicit_global and _save("agent.service_tier", saved_value)
    outcome = _scope_outcome(explicit_global, saved)
    _cp(_accent_line(_t("fast.set_to", feature=feature_name, value=saved_value.upper(), scope=outcome)))
