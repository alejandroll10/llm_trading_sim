"""Agent-type registry.

LLM persona prompts are data, not code: each persona lives in
src/agents/prompts/<type_id>.md with a small front-matter header:

    ---
    name: Display Name
    ---
    <system prompt text>

Adding or editing an LLM personality is a no-code change — drop or edit a
.md file and it is loaded into AGENT_TYPES at import (mirroring the
config-file scenario path in src/scenarios/configs/). Prompt text drives
paper results, so tests/test_persona_prompts.py pins a sha256 of every
persona prompt; update the hash there when a prompt change is intentional.

AGENT_TYPES stays the runtime registry: agents.registry mirrors
deterministic types into it on import (setdefault, hand-written entries
win), and SYSTEM_PROMPT_OVERRIDES validation checks membership here.
"""
from pathlib import Path

from pydantic import BaseModel

from .LLMs.llm_prompt_templates import STANDARD_USER_TEMPLATE


class AgentType(BaseModel):
    name: str
    system_prompt: str
    user_prompt_template: str
    type_id: str = ""


def resolve_agent_type(requested: str) -> str:
    """Resolve a requested agent type to a registered AGENT_TYPES key.

    Matching is exact. Prefix matching was removed on purpose: it silently
    resolved ambiguous requests (e.g. 'default' -> 'default'/'default_gpt'/
    'default_llama') by dict insertion order. Unknown types raise ValueError
    with near-matches as a hint.
    """
    if requested in AGENT_TYPES:
        return requested
    near_matches = sorted(t for t in AGENT_TYPES if t.startswith(requested))
    hint = f" Did you mean one of {near_matches}?" if near_matches else ""
    raise ValueError(
        f"Unknown agent type '{requested}'.{hint} "
        f"Known types: {sorted(AGENT_TYPES)}")


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _parse_persona_file(path: Path) -> tuple[str, str]:
    """Parse a persona .md file into (display name, system prompt).

    The prompt is everything after the closing '---' line, minus exactly one
    trailing newline — so files end with a POSIX newline without the newline
    becoming part of the prompt. Prompt text is otherwise byte-exact,
    including internal indentation and trailing spaces.
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(
            f"{path}: persona file must start with a '---' front-matter block")
    try:
        end = text.index("\n---\n", len("---\n") - 1)
    except ValueError:
        raise ValueError(
            f"{path}: unterminated front matter (missing closing '---' line)") from None
    front_matter = text[len("---\n"):end]
    body = text[end + len("\n---\n"):]
    if body.endswith("\n"):
        body = body[:-1]

    meta = {}
    for line in front_matter.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(
                f"{path}: bad front-matter line {line!r} (expected 'key: value')")
        meta[key.strip()] = value.strip()
    unknown_keys = set(meta) - {"name"}
    if unknown_keys:
        raise ValueError(
            f"{path}: unknown front-matter key(s) {sorted(unknown_keys)}")
    if "name" not in meta or not meta["name"]:
        raise ValueError(f"{path}: front matter must set a non-empty 'name'")
    return meta["name"], body


def _load_personas() -> dict[str, AgentType]:
    personas = {}
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        type_id = path.stem
        name, system_prompt = _parse_persona_file(path)
        personas[type_id] = AgentType(
            name=name,
            system_prompt=system_prompt,
            user_prompt_template=STANDARD_USER_TEMPLATE,
            type_id=type_id,
        )
    if not personas:
        raise RuntimeError(
            f"No persona prompt files found in {PROMPTS_DIR} — "
            "the src/agents/prompts/ directory is missing or empty.")
    return personas


AGENT_TYPES: dict[str, AgentType] = _load_personas()

# Hand-written placeholders for deterministic (rule-based) agents keep their
# historical display names; agents.registry mirrors any DETERMINISTIC_AGENTS
# entry missing here with the class name via setdefault.
_DETERMINISTIC_PLACEHOLDER_NAMES = {
    "gap_trader": "Gap Trader",
    "mean_reversion": "Mean Reversion Trader",
    "buy_trader": "Always Buy Trader",
    "sell_trader": "Always Sell Trader",
    "momentum_trader": "Momentum Trader",
    "market_maker_buy": "Market Maker Buy",
    "market_maker_sell": "Market Maker Sell",
    "hold_trader": "Always Hold Trader",
    "short_sell_trader": "Short Sell Trader",
    "buy_to_close_trader": "Buy to Close Trader",
    "deterministic_market_maker": "Deterministic Market Maker",
    "squeeze_buyer": "Squeeze Buyer",
}
for _type_id, _name in _DETERMINISTIC_PLACEHOLDER_NAMES.items():
    if _type_id in AGENT_TYPES:
        raise ValueError(
            f"prompts/{_type_id}.md collides with deterministic agent type "
            f"'{_type_id}' — persona type_ids and deterministic types must differ.")
    AGENT_TYPES[_type_id] = AgentType(
        name=_name,
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id=_type_id,
    )
del _type_id, _name
