"""Auxiliary service-tier resolution and binding."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)
_aux_service_tier_ignored: set[str] = set()


def _warn_aux_service_tier_ignored(
    task: Optional[str], raw: Any, provider: Optional[str], base_url: Optional[str],
) -> None:
    """Warn once per auxiliary task when its tier is unsupported on the resolved route."""
    key = str(task or "").strip() or str(provider or base_url or "aux")
    if key in _aux_service_tier_ignored:
        return
    _aux_service_tier_ignored.add(key)
    logger.warning(
        "auxiliary%s.service_tier=%r is unsupported on %s; omitting the tier",
        f".{task}" if task else "", raw, provider or base_url or "this route",
    )


def apply_aux_service_tier_overrides(
    overrides: dict[str, Any] | None,
    slot: dict[str, Any] | None,
    *,
    model: Optional[str],
    provider: Optional[str],
    base_url: Optional[str],
    task: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve auxiliary tier sources against the actual route and map supported wire fields.

    Precedence: explicit request/slot ``extra_body.service_tier`` > slot
    ``service_tier`` > per-model agent tier > global agent tier. Explicit
    disabled values stop fallback. Auto/cold are bounded main-turn windows,
    so auxiliary calls do not inherit them.
    """
    merged = dict(overrides or {})
    extra_src = merged.get("extra_body")
    had_dict_extra = isinstance(extra_src, dict)
    extra = dict(extra_src) if had_dict_extra else {}
    extra_tier = extra.get("service_tier") if had_dict_extra else None
    slot_extra = slot.get("extra_body") if isinstance(slot, dict) else None
    slot_extra_tier = slot_extra.get("service_tier") if isinstance(slot_extra, dict) else None
    slot_tier = slot.get("service_tier") if isinstance(slot, dict) else None
    if extra_tier not in (None, ""):
        raw = extra_tier
    elif slot_extra_tier not in (None, ""):
        raw = slot_extra_tier
    else:
        raw = slot_tier

    from hermes_constants import (
        SERVICE_TIER_BOUNDED_VALUES, SERVICE_TIER_DISABLED_VALUES,
        parse_service_tier, resolve_service_tier_source,
    )

    configured = False
    tier = None
    if raw not in (None, ""):
        normalized = str(raw).strip().lower()
        tier = parse_service_tier(raw)
        configured = tier is not None or normalized in SERVICE_TIER_DISABLED_VALUES
        if not configured:
            logger.warning("Invalid auxiliary%s service_tier %r — falling back to agent tier",
                           f".{task}" if task else "", raw)
            if had_dict_extra:
                extra.pop("service_tier", None)
                if extra:
                    merged["extra_body"] = extra
                else:
                    merged.pop("extra_body", None)
    if not configured:
        try:
            from hermes_cli.config import load_config_readonly
            agent_cfg = load_config_readonly().get("agent") or {}
            tier, configured = resolve_service_tier_source(agent_cfg, model)
        except Exception:
            logger.debug("Could not resolve agent service tier for auxiliary request", exc_info=True)
    if not configured:
        return merged

    if had_dict_extra:
        extra.pop("service_tier", None)
        if extra:
            merged["extra_body"] = extra
        else:
            merged.pop("extra_body", None)
    merged.pop("service_tier", None)
    merged.pop("speed", None)
    if tier in SERVICE_TIER_BOUNDED_VALUES:
        return merged
    from hermes_cli.models import resolve_service_tier_overrides
    mapped = resolve_service_tier_overrides(model, tier, provider=provider, base_url=base_url) if tier else None
    if tier and not mapped:
        _warn_aux_service_tier_ignored(task, tier, provider, base_url)
    if mapped:
        merged.update(mapped)
    return merged


def aux_slot_configures_service_tier(slot: dict[str, Any] | None) -> bool:
    """True when the auxiliary slot names a tier at top-level or in ``extra_body``."""
    if not isinstance(slot, dict):
        return False
    extra = slot.get("extra_body")
    extra_tier = extra.get("service_tier") if isinstance(extra, dict) else None
    return extra_tier not in (None, "") or slot.get("service_tier") not in (None, "")


def bind_aux_slot_service_tier(agent: Any, slot: dict[str, Any] | None) -> None:
    """Pin an already-mapped auxiliary slot tier so the agent fallback cannot replace it."""
    if not aux_slot_configures_service_tier(slot):
        return
    from hermes_constants import parse_service_tier
    from agent.fast_mode import TIER_WIRE_KEYS, set_framework_baked_tier_keys

    overrides = getattr(agent, "request_overrides", None)
    mapped = {
        key: overrides[key]
        for key in TIER_WIRE_KEYS
        if isinstance(overrides, dict) and key in overrides
    }
    agent.service_tier = parse_service_tier(mapped.get("service_tier"))
    if agent.service_tier is None and mapped.get("speed") == "fast":
        agent.service_tier = "priority"
    agent._service_tier_session_pinned = True
    set_framework_baked_tier_keys(agent, mapped or None)
