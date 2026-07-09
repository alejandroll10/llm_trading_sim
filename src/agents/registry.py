"""Unified agent-type registry.

Single source of truth for resolving an agent-type string (as used in a
scenario's agent_composition) to its implementation:

- Deterministic (rule-based) agents live in DETERMINISTIC_AGENTS
  (src/agents/deterministic/deterministic_registry.py):
  type string -> BaseAgent subclass.
- LLM personalities live in AGENT_TYPES (src/agents/agent_types.py):
  type string -> AgentType (name + prompts).

Importing this module also mirrors every deterministic type into AGENT_TYPES
with a placeholder AgentType (via setdefault, so hand-written entries win).
Adding a deterministic agent therefore requires only two edits — the class
file and a DETERMINISTIC_AGENTS entry — instead of a third manual placeholder
in agent_types.py, which historically drifted out of sync.
"""
from agents.agent_types import AGENT_TYPES, AgentType
from agents.deterministic.deterministic_registry import DETERMINISTIC_AGENTS


def _mirror_deterministic_types() -> None:
    for det_type, det_cls in DETERMINISTIC_AGENTS.items():
        AGENT_TYPES.setdefault(det_type, AgentType(
            name=det_cls.__name__,
            system_prompt="Deterministic agent - no prompt needed",
            user_prompt_template="",
            type_id=det_type,
        ))


_mirror_deterministic_types()


def known_agent_types() -> set:
    """All valid agent_composition type strings (LLM + deterministic)."""
    return set(AGENT_TYPES) | set(DETERMINISTIC_AGENTS)


def is_deterministic(agent_type: str) -> bool:
    """True if the type string names a rule-based (non-LLM) agent."""
    return agent_type in DETERMINISTIC_AGENTS


def deterministic_agent_class(agent_type: str):
    """Return the BaseAgent subclass for a deterministic agent type."""
    if agent_type not in DETERMINISTIC_AGENTS:
        raise ValueError(
            f"Unknown deterministic agent type '{agent_type}'. "
            f"Known deterministic types: {sorted(DETERMINISTIC_AGENTS)}")
    return DETERMINISTIC_AGENTS[agent_type]


def validate_agent_composition(agent_composition):
    """Validate AGENT_PARAMS['agent_composition'] before any agent is built.

    A typo'd agent type would otherwise surface as a bare KeyError deep inside
    LLMAgent construction. Raises ValueError with the list of known types so a
    bad scenario or sweep variant fails loudly at simulation construction.
    """
    if not isinstance(agent_composition, dict) or not agent_composition:
        raise ValueError(
            "AGENT_PARAMS['agent_composition'] must be a non-empty dict "
            "mapping agent type -> count.")
    known_types = known_agent_types()
    unknown = sorted(set(agent_composition) - known_types)
    if unknown:
        raise ValueError(
            f"agent_composition references unknown agent type(s) {unknown}. "
            f"Known types: {sorted(known_types)}")
    for agent_type, count in agent_composition.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"agent_composition['{agent_type}'] must be a non-negative "
                f"integer, got {count!r}")


def validate_system_prompt_overrides(overrides: dict):
    """Validate SYSTEM_PROMPT_OVERRIDES (prompt-family sweeps, issue #102).

    Each key must name a known, non-deterministic agent type and map to a
    non-empty replacement system prompt. Raises ValueError so a typo'd pack
    fails at simulation construction, not silently mid-sweep.
    """
    for override_type, override_prompt in (overrides or {}).items():
        if override_type not in AGENT_TYPES:
            raise ValueError(
                f"SYSTEM_PROMPT_OVERRIDES references unknown agent type '{override_type}'. "
                f"Known types: {sorted(AGENT_TYPES.keys())}")
        if override_type in DETERMINISTIC_AGENTS:
            raise ValueError(
                f"SYSTEM_PROMPT_OVERRIDES targets deterministic agent type "
                f"'{override_type}', which does not use a prompt.")
        if not isinstance(override_prompt, str) or not override_prompt.strip():
            raise ValueError(
                f"SYSTEM_PROMPT_OVERRIDES['{override_type}'] must be a non-empty string.")
