"""
Static, dependency-free price table for LLM token usage (issue #104).

Prices are USD per 1,000,000 tokens as (input, output). There are no live pricing
lookups and no network dependency -- update the numbers here when provider prices
change. Values are estimates, not invoices; verify against your provider dashboard.

Backend-aware pricing
---------------------
The dollar cost of an OPEN-WEIGHT model (gpt-oss-*, llama-*) depends on WHO serves
it, not just its name. The exact same "gpt-oss-120b" is:
  - free at the margin when self-hosted (local vLLM) or on the UF Hypergator API,
  - billed per token on a paid host like DeepInfra.
So open-model rates are grouped by backend and the active backend is inferred from
DEFAULT_LLM_BASE_URL at cost-computation time (deepinfra vs. self-hosted). First-
party OpenAI models (gpt-4o, o3, ...) are billed by OpenAI regardless of base_url,
so they sit in a single backend-independent table.

Model-name matching is exact first, then longest-prefix, then longest-substring,
over both the raw name and its provider-stripped, lowercased form -- so
"openai/gpt-oss-120b-Turbo" resolves to "gpt-oss-120b" (DeepInfra prices the Turbo
endpoint at the same per-token rate as the base model) and "gpt-4o-2024-08-06"
resolves to "gpt-4o". Only specific versioned open-model keys are listed (no bare
"gpt-oss"/"llama" catch-all), so an unrecognized variant is reported as unpriced
rather than silently costed at $0.

Sources (verified 2026-07, DeepInfra standard tier): gpt-oss-120b $0.037/$0.17,
gpt-oss-20b $0.03/$0.14 (deepinfra.com model pages); llama-3.3-70b $0.10/$0.32,
llama-3.1-70b $0.23/$0.40 (deepinfra.com/pricing). DeepInfra's Priority tier is
~1.5x; adjust if you use it.
"""

from typing import Dict, Optional, Tuple

# First-party OpenAI models: billed by OpenAI no matter which base_url is set.
OPENAI_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
}

# Open-weight models, priced per serving backend. Only specific versioned keys --
# no bare "gpt-oss"/"llama" catch-all -- so unknown variants surface as unpriced.
# hold_llm short-circuits before any API call, so it is free on every backend.
_SELF_HOSTED_OPEN = {
    "gpt-oss-120b": (0.0, 0.0),
    "gpt-oss-20b": (0.0, 0.0),
    "llama-3.3-70b": (0.0, 0.0),
    "llama-3.1-70b": (0.0, 0.0),
    "llama-3.1-8b": (0.0, 0.0),
    "gemma-4-31b": (0.0, 0.0),
    "gemma-4-26b": (0.0, 0.0),
    "deepseek-v4-flash": (0.0, 0.0),
    "nemotron-3-nano-omni-30b": (0.0, 0.0),
    "hold_llm": (0.0, 0.0),
}
_DEEPINFRA_OPEN = {
    "gpt-oss-120b": (0.037, 0.17),
    "gpt-oss-20b": (0.03, 0.14),
    "llama-3.3-70b": (0.10, 0.32),
    "llama-3.1-70b": (0.23, 0.40),
    "llama-3.1-8b": (0.03, 0.05),
    # Gemma 4 family (DeepInfra): 31B dense and 26B-A4B MoE, both ~$0.07/$0.34.
    # 26B is from DeepInfra's own blog; 31B matches aggregators but its DeepInfra
    # model page didn't resolve to confirm -- verify on the dashboard if used.
    "gemma-4-31b": (0.07, 0.34),
    "gemma-4-26b": (0.07, 0.34),
    # DeepSeek V4 Flash: cached input is far cheaper ($0.018/1M) but we bill all
    # input at the full rate since usage.prompt_tokens_details cache hits aren't
    # tracked here -- a small overestimate when long prompt prefixes are cached.
    "deepseek-v4-flash": (0.09, 0.18),
    # NVIDIA Nemotron 3 Nano Omni 30B-A3B (Reasoning), DeepInfra standard tier.
    # A reasoning model: expect high output-token counts, so the $0.80 output rate
    # dominates. Key matches the -A3B-Reasoning variant by longest-prefix.
    "nemotron-3-nano-omni-30b": (0.20, 0.80),
    "hold_llm": (0.0, 0.0),
}
OPEN_MODEL_PRICING: Dict[str, Dict[str, Tuple[float, float]]] = {
    "self_hosted": _SELF_HOSTED_OPEN,
    "deepinfra": _DEEPINFRA_OPEN,
}


def active_backend() -> str:
    """Infer the serving backend from the configured LLM base URL.

    Returns "deepinfra" when DEFAULT_LLM_BASE_URL points at DeepInfra, else
    "self_hosted" (UF Hypergator API, local vLLM, or unset). Resolved lazily so
    importing this module never forces scenarios.base to load, and so a run picks
    up whatever backend is configured when the cost is computed.
    """
    try:
        from scenarios.base import DEFAULT_LLM_BASE_URL
    except Exception:
        return "self_hosted"
    url = (DEFAULT_LLM_BASE_URL or "").lower()
    if "deepinfra" in url:
        return "deepinfra"
    return "self_hosted"


def _name_variants(model: str):
    """The raw lowercased name plus, if present, its provider-stripped form
    ("openai/gpt-oss-120b-Turbo" -> "gpt-oss-120b-turbo")."""
    lowered = model.lower()
    variants = [lowered]
    if "/" in lowered:
        variants.append(lowered.rsplit("/", 1)[1])
    return variants


def _lookup(variants, table: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Exact -> longest-prefix -> longest-substring match of any variant in table."""
    for name in variants:
        if name in table:
            return table[name]
    for name in variants:
        pref = [k for k in table if name.startswith(k)]
        if pref:
            return table[max(pref, key=len)]
    for name in variants:
        sub = [k for k in table if k in name]
        if sub:
            return table[max(sub, key=len)]
    return None


def price_for(model: str, backend: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Return (input_per_1M, output_per_1M) for `model`, or None if unknown.

    OpenAI first-party models match the backend-independent table; open-weight
    models match the active backend's table (auto-detected when `backend` is None).
    None means no table entry -- callers treat dollar cost as 0 but should flag the
    model as unpriced so the omission is visible.
    """
    if not model:
        return None
    variants = _name_variants(model)
    hit = _lookup(variants, OPENAI_PRICING)
    if hit is not None:
        return hit
    backend = backend or active_backend()
    return _lookup(variants, OPEN_MODEL_PRICING.get(backend, _SELF_HOSTED_OPEN))


def is_known_model(model: str, backend: Optional[str] = None) -> bool:
    """True if the model has a price-table entry for the (active) backend."""
    return price_for(model, backend) is not None


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int,
                 backend: Optional[str] = None) -> float:
    """Dollar cost of a call. Unknown models cost 0.0 (flag them via is_known_model).

    Backend defaults to the configured one, so on a DeepInfra deployment gpt-oss/
    llama calls carry their real per-token cost, while on UF/local they stay $0.
    """
    price = price_for(model, backend)
    if price is None:
        return 0.0
    in_rate, out_rate = price
    return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate
