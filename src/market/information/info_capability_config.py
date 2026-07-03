"""Scenario-level configuration for per-agent private information signals.

This module turns a scenario's ``AGENT_PARAMS['info_capabilities']`` block into
concrete :class:`InfoCapability` objects per agent, so scenarios can give
individual agents noisy / delayed / depth-truncated / disabled views of the
market's information signals (most commonly the FUNDAMENTAL signal, to build
genuine asymmetric-information designs).

Config schema (all keys optional)::

    'info_capabilities': {
        # Baseline applied to every agent (indexed by GLOBAL agent index when
        # a field is a list, so a list here spreads across all agents in order).
        'default': {
            'fundamental': {'noise_level': 0.0},
        },
        # Per agent-TYPE overrides. When a field is a list it is spread across
        # the agents of that type in composition order (the "who picks off whom"
        # design), cycling if the list is shorter than the count.
        'by_type': {
            'value': {'fundamental': {'noise_level': [0.0, 0.05, 0.1, 0.2]}},
        },
        # Per GLOBAL agent index overrides (0-based, composition order). Highest
        # precedence. Fields must be scalars here (the index already selects one
        # agent). Keys may be int or str.
        'by_index': {
            3: {'fundamental': {'noise_level': 0.5, 'accuracy': 0.8}},
        },
        # Prompt disclosure toggles (see build_own_quality_text /
        # summarize_others_quality).
        'disclose_signal_quality': False,   # tell each agent its OWN signal quality
        'disclose_others_quality': False,   # tell each agent the DISTRIBUTION of all agents' quality
    }

Signal keys are :class:`InformationType` values (``'fundamental'``,
``'dividend'``, ``'order_book'``, ``'insider'``, ...). Capability fields are the
fields of :class:`InfoCapability`: ``enabled`` (bool), ``noise_level`` (float),
``delay`` (int), ``depth`` (int), ``accuracy`` (float).

Field support (see ``InformationService._modify_signal``):
    - ``enabled=False``  : hides the signal value from the agent (all types).
    - ``noise_level``    : Gaussian noise, std = noise_level * |value|, on
                           FUNDAMENTAL-category signals (fundamental, dividend).
    - ``depth``          : truncates order-book (MARKET-category) levels.
    - ``accuracy``       : scales the signal's reported ``reliability``.
    - ``delay``          : serves the signal VALUE from ``delay`` rounds ago
                           (from ``signal_history``) instead of the current
                           round; falls back to the current value for the first
                           ``delay`` rounds (no history yet). Only the value is
                           staled — the current round's structural metadata
                           (``round``, ``periods_remaining``, ``redemption_value``)
                           is preserved so time-to-redemption stays correct.
                           Delay is applied BEFORE noise/accuracy, so the agent's
                           own noise is realized on the stale value. Metadata
                           records ``original_round``, ``current_round``,
                           ``delay`` and ``is_stale``. Example:
                           ``{'fundamental': {'delay': 1}}``.

Precedence (later wins, merged per signal-type per field):
``default`` -> ``by_type[agent_type]`` -> ``by_index[global_index]``.
"""

from typing import Any, Dict, List, Optional

from .information_types import InformationType, InfoCapability

# Capability fields that a scenario may set, mapped to a coercion function so we
# fail loudly on obviously-wrong config rather than silently passing garbage
# into numpy later.
_CAPABILITY_FIELDS = {
    'enabled': bool,
    'noise_level': float,
    'delay': int,
    'depth': int,
    'accuracy': float,
}

_TOP_LEVEL_KEYS = {
    'default', 'by_type', 'by_index',
    'disclose_signal_quality', 'disclose_others_quality',
}


def _resolve_signal_type(key: str) -> InformationType:
    """Map a config signal key (e.g. ``'fundamental'``) to an InformationType."""
    if isinstance(key, InformationType):
        return key
    try:
        return InformationType(key)
    except ValueError:
        valid = ", ".join(sorted(t.value for t in InformationType))
        raise ValueError(
            f"Unknown info_capabilities signal key '{key}'. "
            f"Expected one of: {valid}."
        )


def _pick(value: Any, index: int) -> Any:
    """If ``value`` is a list, select an element by ``index`` (cycling); else return it.

    A list of capability values is spread across agents in order. Cycling with
    modulo means a short list repeats, and a single-element list applies to all.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return value[index % len(value)]
    return value


def _coerce_field(field: str, value: Any) -> Any:
    """Validate/coerce a capability field value, raising on unknown fields."""
    if field not in _CAPABILITY_FIELDS:
        valid = ", ".join(sorted(_CAPABILITY_FIELDS))
        raise ValueError(
            f"Unknown info capability field '{field}'. Expected one of: {valid}."
        )
    if value is None:
        return None
    caster = _CAPABILITY_FIELDS[field]
    # bool is a subclass of int; guard so 0/1 don't silently become booleans.
    if caster is bool:
        if not isinstance(value, bool):
            raise ValueError(f"info capability field 'enabled' must be a bool, got {value!r}")
        return value
    try:
        return caster(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"info capability field '{field}' expected {caster.__name__}, got {value!r}"
        )


def _merge_layer(
    accum: Dict[InformationType, Dict[str, Any]],
    layer: Optional[Dict[str, Any]],
    index: int,
) -> None:
    """Merge one config layer (default / by_type / by_index) into ``accum`` in place.

    ``index`` selects list elements for spread fields. Scalar layers should pass
    ``index=0``.
    """
    if not layer:
        return
    for signal_key, fields in layer.items():
        info_type = _resolve_signal_type(signal_key)
        if not isinstance(fields, dict):
            raise ValueError(
                f"info_capabilities entry for '{signal_key}' must be a dict of "
                f"capability fields, got {fields!r}"
            )
        target = accum.setdefault(info_type, {})
        for field, raw in fields.items():
            picked = _pick(raw, index)
            if picked is None and isinstance(raw, (list, tuple)):
                # Empty list -> nothing to apply for this field.
                continue
            target[field] = _coerce_field(field, picked)


def validate_config(config: Dict[str, Any]) -> None:
    """Raise ValueError if the top-level config contains unexpected keys."""
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError(f"info_capabilities must be a dict, got {type(config).__name__}")
    unknown = set(config) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown info_capabilities key(s): {sorted(unknown)}. "
            f"Expected a subset of: {sorted(_TOP_LEVEL_KEYS)}."
        )


def resolve_agent_info_capabilities(
    config: Dict[str, Any],
    agent_type: str,
    type_index: int,
    global_index: int,
) -> Dict[InformationType, InfoCapability]:
    """Resolve the InfoCapability map for one agent.

    Args:
        config: The ``AGENT_PARAMS['info_capabilities']`` dict.
        agent_type: The agent's type string (e.g. ``'value'``).
        type_index: 0-based position of this agent among agents of the same type
            (used to spread ``by_type`` list fields).
        global_index: 0-based position among all agents in composition order
            (used to spread ``default`` list fields and to look up ``by_index``).

    Returns:
        Dict mapping InformationType to InfoCapability. Empty if the config does
        not touch this agent.
    """
    if not config:
        return {}

    accum: Dict[InformationType, Dict[str, Any]] = {}

    # 1. Global default (list fields spread by GLOBAL index).
    _merge_layer(accum, config.get('default'), global_index)

    # 2. Per-type overrides (list fields spread by TYPE-local index).
    by_type = config.get('by_type') or {}
    _merge_layer(accum, by_type.get(agent_type), type_index)

    # 3. Per-index overrides (scalars; key may be int or str).
    by_index = config.get('by_index') or {}
    index_layer = by_index.get(global_index)
    if index_layer is None:
        index_layer = by_index.get(str(global_index))
    _merge_layer(accum, index_layer, 0)

    return {
        info_type: InfoCapability(**fields)
        for info_type, fields in accum.items()
    }


def build_own_quality_text(info_type: InformationType, capability: InfoCapability) -> str:
    """Human-readable disclosure of an agent's own signal quality for the prompt."""
    parts: List[str] = []
    if capability.noise_level and capability.noise_level > 0:
        parts.append(
            f"it is a PRIVATE, NOISY estimate with Gaussian noise of "
            f"standard deviation ~{capability.noise_level * 100:.1f}% of the true value"
        )
    else:
        parts.append("it is observed without added noise")
    if capability.delay and capability.delay > 0:
        parts.append(f"reported with a delay of {capability.delay} round(s)")
    if capability.accuracy is not None and capability.accuracy < 1.0:
        parts.append(f"overall reliability {capability.accuracy:.2f}")
    return (
        f"[Signal quality] Your {info_type.value} signal is private: "
        + "; ".join(parts)
        + ". Other traders may see different, independently-drawn estimates."
    )


def summarize_others_quality(noise_levels: List[float], num_agents: int) -> str:
    """Common-knowledge disclosure of the DISTRIBUTION of fundamental-signal noise.

    Args:
        noise_levels: Fundamental-signal noise level for each agent (0.0 for
            agents with no noise configured).
        num_agents: Total number of agents.
    """
    if not noise_levels:
        return ""
    lo = min(noise_levels) * 100
    hi = max(noise_levels) * 100
    mean = (sum(noise_levels) / len(noise_levels)) * 100
    n_noisy = sum(1 for n in noise_levels if n > 0)
    return (
        f"[Market-wide signal quality] Across the {num_agents} traders, private "
        f"fundamental-signal noise ranges from {lo:.1f}% to {hi:.1f}% of value "
        f"(mean {mean:.1f}%); {n_noisy} of {num_agents} traders receive a noisy signal. "
        f"You do not know which specific traders are better or worse informed."
    )
