"""Model and route eligibility for provider service tiers."""

from __future__ import annotations

import re
from urllib.parse import urlparse


def is_google_service_tier_model(model_id: str | None) -> bool:
    """Gemini 2.5+ accepts top-level service_tier; aggregator IDs and non-Gemini models do not."""
    base = str(model_id or "").strip().lower()
    base = base.split("/", 1)[1] if "/" in base else base
    base = base.split(":", 1)[0]
    match = re.match(r"gemini-(\d+)(?:\.(\d+))?", base)
    return bool(match) and (int(match.group(1)), int(match.group(2) or 0)) >= (2, 5)


def fast_mode_route_supported(model_id: str | None, provider: str | None, base_url: str | None) -> bool:
    """Only first-party providers and hosts receive their published tier parameters."""
    from agent.model_metadata import is_grok_46_family
    from hermes_cli.models import _is_anthropic_fast_model, normalize_provider

    if _is_anthropic_fast_model(model_id):
        allowed = {"anthropic": "api.anthropic.com"}
    elif is_grok_46_family(str(model_id or "")):
        allowed = {"xai": "api.x.ai"}
    elif is_google_service_tier_model(model_id):
        allowed = {"gemini": "generativelanguage.googleapis.com"}
    else:
        allowed = {"openai": "api.openai.com", "openai-api": "api.openai.com", "openai-codex": "chatgpt.com"}
    if provider and normalize_provider(provider) not in allowed:
        return False
    host = (urlparse(str(base_url or "")).hostname or "").lower()
    return not host or host in allowed.values()
