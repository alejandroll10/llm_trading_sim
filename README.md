# LLM Trading Simulation

This repository contains the source code for a trading simulation environment powered by Large Language Models (LLMs).

## Description

This project simulates a financial market where agents, powered by LLMs, make trading decisions.

## Paper

This repository accompanies the research paper:

**"Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations"**

*Author:* Alejandro Lopez-Lira (University of Florida - Department of Finance, Insurance and Real Estate)

*Date:* April 14, 2025

### Abstract

This paper presents a realistic simulated stock market where large language models (LLMs) act as heterogeneous competing trading agents. The open-source framework incorporates a persistent order book with market and limit orders, partial fills, dividends, and equilibrium clearing alongside agents with varied strategies, information sets, and endowments. Agents submit standardized decisions using structured outputs and function calls while expressing their reasoning in natural language. Three findings emerge: First, LLMs demonstrate consistent strategy adherence and can function as value investors, momentum traders, or market makers per their instructions. Second, market dynamics exhibit features of real financial markets, including price discovery, bubbles, underreaction, and strategic liquidity provision. Third, the framework enables analysis of LLMs' responses to varying market conditions, similar to partial dependence plots in machine-learning interpretability. The framework allows simulating financial theories without closed-form solutions, creating experimental designs that would be costly with human participants, and establishing how prompts can generate correlated behaviors affecting market stability.

### Citation

```bibtex
@article{lopez2025llm,
  title={Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations},
  author={Lopez-Lira, Alejandro},
  year={2025},
  month={April},
  day={14},
  url={https://ssrn.com/abstract=5217340},
  doi={10.2139/ssrn.5217340}
}
```

*Available at SSRN:* https://ssrn.com/abstract=5217340

*Keywords:* LLM Agents, Agent-Based Markets, Experimental Finance, AI Trading, Multi-Agent Systems

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alejandroll10/llm_trading_sim.git
    cd llm_trading_sim
    ```

2.  **Create a virtual environment:**
    It is highly recommended to use a virtual environment. For example, with conda:
    ```bash
    conda create -n llm_trading python=3.11
    conda activate llm_trading
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    For development (running the test suite, call-graph utilities), install the
    package in editable mode with the dev extras instead:
    ```bash
    pip install -e .[dev]
    ```

4.  **Configure LLM Provider:**

    **Step 1 - Add API Key:**

    Create a `.env` file in the project root:
    ```bash
    OPENAI_API_KEY="sk-..."
    ```

    - **UF Hypergator:** Get virtual key from https://api.ai.it.ufl.edu/ui/
    - **OpenAI:** Get API key from https://platform.openai.com/api-keys
    - **DeepInfra:** Set `DEEPINFRA_TOKEN` instead of `OPENAI_API_KEY`

    **Step 2 - Select Endpoint & Model:**

    Copy the example config and edit it (the copy is gitignored, so your local
    settings never end up in a commit):
    ```bash
    cp src/llm_config.example.py src/llm_config.py
    ```

    **Option A - UF Hypergator (Free for UF users):**
    ```python
    LLM_BASE_URL = "https://api.ai.it.ufl.edu/v1"
    LLM_MODEL = "gpt-oss-120b"
    ```

    **Option B - OpenAI (Paid service):**
    ```python
    LLM_BASE_URL = None  # None = use OpenAI's default endpoint
    LLM_MODEL = "gpt-4o-2024-11-20"
    ```

    Any OpenAI-compatible endpoint works (DeepInfra, local vLLM, ...); see
    `src/llm_config.example.py` for more options. If `src/llm_config.py` does
    not exist, the UF Hypergator defaults are used.

    **Verified Working Models:**
    - UF Hypergator: `llama-3.1-70b-instruct` ✅, `llama-3.3-70b-instruct` ✅, `gpt-oss-20b` ✅, `gpt-oss-120b` ✅
    - OpenAI: `gpt-4o` ✅, `gpt-4o-2024-11-20` ✅

    **Note:** Smaller models like `llama-3.1-8b-instruct` fail structured output validation. Use 70B+ models for reliable results.

## Usage

To run the simulation, you can execute the `run_base_sim.py` script from the `src/` directory.

1.  **List Available Scenarios:**
    To see a list of all available scenarios and their descriptions, run:
    ```bash
    python3 src/run_base_sim.py --list
    ```

2.  **Run a Specific Scenario:**
    To run a specific scenario, provide its name as a command-line argument. For example, to run the `price_discovery_above_fundamental` scenario:
    ```bash
    python3 src/run_base_sim.py price_discovery_above_fundamental
    ```

    If you run the script without any arguments, it will display a list of available scenario names.

    Simulation results, including plots and data, will be saved in the `logs/` directory.

3.  **Run a Robustness Sweep:**
    `src/run_sweep.py` runs one scenario across a grid of seeds × temperatures × models × prompt/param variants, with a resumable manifest and a pre-launch cost estimate. `src/aggregate_sweep.py` collects the per-cell CSVs into tidy panels with `cell_id`, `seed`, `temperature`, `model`, `variant`, and `prompt_family` metadata columns (`prompt_family` is the clustering key for inference — runs sharing a prompt family are not independent draws).
    ```bash
    # Sweep seeds x temperatures, then aggregate
    python3 src/run_sweep.py simple_mixed_traders --seeds 42 7 13 --temperatures 0.0 0.5 --yes
    python3 src/aggregate_sweep.py logs/sweeps/<sweep_name>
    ```

    **Prompt-family variant packs** (`sweeps/variants/`): checked-in JSON packs for robustness across prompt wordings, consumed via `--variants-file`. Each pack has one unmodified control plus variants that set `SYSTEM_PROMPT_OVERRIDES` (agent type → replacement system prompt) and optionally `FUNDAMENTAL_INFO_MODE`:
    - `paraphrases_<persona>.json` — 4 semantically equivalent paraphrases each for `default`, `value`, `momentum`, `market_maker`, `optimistic`, `profit_maximizer`
    - `persona_families.json` — index-matched paraphrases applied to all six personas at once (for mixed compositions)
    - `framing_cook.json` — trader vs. financial-advisor framing × explicit vs. masked economic context (advisor rewrites exist only for the six workhorse personas, so use compositions drawn from those)
    - `objective_framing_a3.json` — the same value strategy under different stated objectives (maximize wealth / follow the strategy even at a loss / maximize risk-adjusted return / no objective)

    ```bash
    python3 src/run_sweep.py prompt_variant_smoke --variants-file sweeps/variants/paraphrases_value.json --dry-run
    ```
    The packs are generated artifacts — edit `scripts/generate_prompt_variant_packs.py` (the single source of truth) and rerun it; `tests/test_prompt_variants.py` fails if the JSON drifts from the generator.

## Simulation Lifecycle

The simulation operates in discrete rounds. The following steps occur in each round:

1.  **Update Market State:** The simulation updates the market context, including the fundamental price and any potential dividend payments for the upcoming round.
2.  **Collect Agent Decisions:** Each agent analyzes the current market state and their own internal state to decide whether to place a buy, sell, or hold order.
3.  **Match Orders:** The matching engine resolves the collected orders, executing trades and determining the new market price.
4.  **Record Round Data:** All data from the round, including trades, prices, and agent decisions, is recorded.
5.  **Pay Dividends/Interest:** Any scheduled dividends or interest payments are distributed to the agents.

This lifecycle is orchestrated by the `execute_round` method in `src/base_sim.py`.

**Round lifecycle hooks:** custom behavior can attach to the lifecycle without
editing the orchestrator. Register callables on a simulation instance before
calling `run()`; they are invoked as `hook(sim, round_number)`:

```python
sim = BaseSimulation(...)
sim.register_before_round(lambda sim, r: ...)  # runs before each round
sim.register_after_round(lambda sim, r: ...)   # runs after each round's updates
sim.run()
```

## Key Features

### Core Trading Mechanics
- **Order Book:** Persistent order book with market and limit orders
- **Price Discovery:** Dynamic price formation through order matching
- **Partial Fills:** Orders can be partially executed
- **Dividends:** Configurable dividend payments
- **Multi-Stock Support:** Trade multiple securities simultaneously

### Advanced Features

#### Dividend Regime Shifts (Time-Varying Dividend Parameters)
The dividend process can change mid-run at scheduled round boundaries, enabling out-of-distribution robustness experiments: do agents' valuations update toward the new regime, and does the price re-converge?

**Configuration:**
```python
"DIVIDEND_PARAMS": {
    'base_dividend': 1.4,
    'dividend_probability': 0.5,
    'dividend_variation': 1.0,
    # From round 5 (0-indexed) onward, base_dividend drops to 1.0.
    # Each entry overrides the base params (entries do not stack).
    'regime_schedule': [
        {'round': 5, 'base_dividend': 1.0},
    ],
    # Optional contrast cell: announce the shift to agents (default: silent)
    'announce_regime_shifts': False,
}
```

**Conventions:**
- The fundamental value follows the piecewise no-arbitrage path `FV_t = (E[d_t] + FV_{t+1}) / (1+r)`, computed per round and recorded in `market_data.csv` (`fundamental_price` reflects the active regime).
- Redemption (finite horizon) equals the terminal-regime fundamental `E[d_last]/r`, so the fundamental is constant within the terminal regime segment.
- Shifts are unannounced by default: use `FUNDAMENTAL_INFO_MODE = "realizations_only"` so agents can only infer the change from realized dividends. Info modes that reveal model parameters truthfully show the active regime. Setting `announce_regime_shifts: true` adds a notice (without revealing the new parameters) to agents' dividend info from the shift round onward.
- The simulation verifier checks every round that the recorded fundamental matches the scheduled path.
- Not yet supported in multi-stock mode.

**Example Scenario:** `test_regime_shift` - deterministic agents with an unannounced shift at round 5 (E[d] 1.4 → 1.0, fundamental 21.73 → 20.00)

#### Leverage Trading (Margin Trading)
Agents can borrow cash to amplify their long positions, enabling research on leveraged trading strategies and risk management.

**Configuration:**
```python
scenario = {
    "leverage_enabled": True,
    "AGENT_PARAMS": {
        'deterministic_params': {
            'momentum_trader': {
                'leverage_ratio': 2.0,        # Allow 2x leverage
                'initial_margin': 0.5,        # 50% down payment required
                'maintenance_margin': 0.25,   # 25% minimum margin (liquidation threshold)
            }
        }
    },
    "leverage_interest_rate": 0.05,  # 5% annual interest on borrowed cash
    "cash_lending_pool": 1000000,    # Optional: limit available lending pool
}
```

**Features:**
- Automatic borrowing when placing orders beyond available cash
- Margin calls with forced liquidation when positions fall below maintenance margin
- Per-round interest charges on borrowed cash
- Full visibility for LLM agents (leverage metrics included in observations)

**Example Scenarios:**
- `test_leverage` - Deterministic agents with 2x leverage
- `test_leverage_llm` - LLM agents using leverage strategically

#### Short Selling
Agents can borrow shares to sell short, enabling research on bearish strategies and market dynamics.

**Configuration:**
```python
scenario = {
    "AGENT_PARAMS": {
        'allow_short_selling': True,
        'margin_requirement': 0.5,  # 50% margin for shorts
    },
    "LENDABLE_SHARES": 10000,  # Total shares available to borrow
}
```

**Features:**
- Borrow shares from lending pool to sell short
- Margin calls when short positions become underwater
- Per-round borrowing fees
- Works with both single and multi-stock scenarios

### Agent Types
The simulation supports multiple agent types with different trading strategies:

**LLM Agent Types** (prompt files in `src/agents/prompts/`, one `.md` per persona):
- **default:** Balanced, analytical trader
- **speculator:** Risk-seeking, momentum-focused
- **optimistic:** Bullish bias, sees upside potential
- **pessimistic:** Bearish bias, focuses on risks
- **short_seller:** Actively shorts overvalued assets
- **leverage_trader:** Uses maximum leverage for amplified returns
- **long_short:** Pairs trading - long undervalued, short overvalued simultaneously

**Deterministic Agent Types** (defined in `src/agents/deterministic/`):
- **buy_trader / sell_trader:** Simple directional traders
- **margin_buyer:** Uses leverage to buy aggressively
- **multi_stock_buy_agent / multi_stock_sell_agent:** Multi-stock traders

### Memory and Social Features
Agents can be configured with memory and social messaging capabilities:

```python
"MEMORY_ENABLED": True,   # Agents can write notes_to_self between rounds
"SOCIAL_ENABLED": True,   # Agents can post messages to a shared feed
```

When enabled, agents receive their previous notes and can read messages from other agents, enabling more sophisticated multi-round strategies and emergent social dynamics.

## Testing

Run the unit test suite (no API key needed; also runs in CI on every push/PR):

```bash
pip install pytest   # or: pip install -e .[dev]
pytest tests/ -q
```

Run the health check script to verify all features work correctly end-to-end
(this one runs real simulations):

```bash
# Quick test (single-stock scenarios only, ~5 minutes)
python scripts/health_check.py --quick

# Full test (all 8 systematic scenarios, ~15 minutes)
python scripts/health_check.py

# Verbose output
python scripts/health_check.py --verbose
```

The health check verifies:
- ✅ Trading execution (trades happen)
- ✅ Short selling (borrowed_shares > 0)
- ✅ Leverage (borrowed_cash > 0)
- ✅ Multi-stock mode

### Systematic Test Scenarios

| Scenario | Leverage | Short Selling | Multi-Stock |
|----------|----------|---------------|-------------|
| `single_basic` | ❌ | ❌ | ❌ |
| `single_short` | ❌ | ✅ | ❌ |
| `single_leverage` | ✅ | ❌ | ❌ |
| `single_leverage_short` | ✅ | ✅ | ❌ |
| `multi_basic` | ❌ | ❌ | ✅ |
| `multi_short` | ❌ | ✅ | ✅ |
| `multi_leverage` | ✅ | ❌ | ✅ |
| `multi_leverage_short` | ✅ | ✅ | ✅ |

## Adding New Scenarios

Scenarios live in the `src/scenarios/` package and are **auto-discovered** —
no registration edits are needed. Scenario names must be unique across all
modules and config files (duplicates raise at import time). There are two
ways to add one:

**Option A - Config file (no Python needed):** drop a `.yaml` (or `.json`)
file into `src/scenarios/configs/`:

```yaml
# src/scenarios/configs/my_custom_scenario.yaml
name: my_custom_scenario
description: "A custom scenario for testing a new agent type."
parameters:            # deep-merged over DEFAULT_PARAMS
  NUM_ROUNDS: 5
  AGENT_PARAMS:
    agent_composition:
      value: 1
      market_maker: 1
```

See `src/scenarios/configs/example_yaml_scenario.yaml` for a working example
and `src/scenarios/config_loader.py` for the full format (including several
scenarios per file).

**Option B - Python module:** create (or extend) a module in `src/scenarios/`
that defines a module-level `SCENARIOS` dict. Build parameters with
`merge_params`, which deep-merges your overrides onto `DEFAULT_PARAMS` — you
only state what differs, and nested defaults are inherited instead of
restated:

```python
# src/scenarios/my_scenarios.py
from .base import SimulationScenario, DEFAULT_PARAMS, merge_params

SCENARIOS = {
    "my_custom_scenario": SimulationScenario(
        name="my_custom_scenario",
        description="A custom scenario for testing a new agent type.",
        parameters=merge_params(DEFAULT_PARAMS, {
            "NUM_ROUNDS": 5,
            "AGENT_PARAMS": {
                "agent_composition": {
                    "value": 1,
                    "market_maker": 1,
                },
            },
        }),
    ),
}
```

Notes on `merge_params`: nested dicts merge recursively, but
`agent_composition` and `STOCKS` replace wholesale (an override is a complete
specification, so default agent types don't leak in). Every type named in
`agent_composition` is validated against the agent registry at simulation
construction, so typos fail immediately with the list of known types.

After adding your scenario, run it by name as described above:
```bash
python3 src/run_base_sim.py my_custom_scenario
```

## Adding New Agent Types

**LLM personality (no code):** drop a prompt file at
`src/agents/prompts/<type_id>.md` — the filename is the type string used in
`agent_composition`:

```markdown
---
name: My Persona
---
You are a trader who ...
```

Everything after the closing `---` line is the system prompt, byte-exact
(minus the file's final newline). All persona files are loaded into the
`AGENT_TYPES` registry at import. Existing persona prompts are pinned by
sha256 in `tests/test_persona_prompts.py` because prompt text drives paper
results — if you deliberately edit a prompt, update its hash there in the
same commit (new personas just need an entry added).

**Deterministic (rule-based) agent (two edits):**
1. Create a class under `src/agents/deterministic/` subclassing `BaseAgent`
   and implementing `make_decision()` (see `buy_agent.py` for a template).
2. Register it in `DETERMINISTIC_AGENTS` in
   `src/agents/deterministic/deterministic_registry.py`.

The unified registry (`src/agents/registry.py`) mirrors deterministic types
into `AGENT_TYPES` automatically — no third placeholder edit is needed.
