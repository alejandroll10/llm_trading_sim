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

4.  **Configure LLM Provider:**

    **Step 1 - Add API Key:**

    Create a `.env` file in the project root:
    ```bash
    OPENAI_API_KEY="sk-..."
    ```

    - **UF Hypergator:** Get virtual key from https://api.ai.it.ufl.edu/ui/
    - **OpenAI:** Get API key from https://platform.openai.com/api-keys

    **Step 2 - Select Model:**

    Edit lines 8-9 in `src/scenarios/base.py`:

    **Option A - UF Hypergator (Free for UF users):**
    ```python
    DEFAULT_LLM_BASE_URL = "https://api.ai.it.ufl.edu/v1"
    DEFAULT_LLM_MODEL = "llama-3.1-70b-instruct"
    ```

    **Option B - OpenAI (Paid service):**
    ```python
    DEFAULT_LLM_BASE_URL = None  # None = use OpenAI's default endpoint
    DEFAULT_LLM_MODEL = "gpt-4o-2024-11-20"
    ```

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

## Simulation Lifecycle

The simulation operates in discrete rounds. The following steps occur in each round:

1.  **Update Market State:** The simulation updates the market context, including the fundamental price and any potential dividend payments for the upcoming round.
2.  **Collect Agent Decisions:** Each agent analyzes the current market state and their own internal state to decide whether to place a buy, sell, or hold order.
3.  **Match Orders:** The matching engine resolves the collected orders, executing trades and determining the new market price.
4.  **Record Round Data:** All data from the round, including trades, prices, and agent decisions, is recorded.
5.  **Pay Dividends/Interest:** Any scheduled dividends or interest payments are distributed to the agents.

This lifecycle is orchestrated by the `execute_round` method in `src/base_sim.py`.

## Key Features

### Core Trading Mechanics
- **Order Book:** Persistent order book with market and limit orders
- **Price Discovery:** Dynamic price formation through order matching
- **Partial Fills:** Orders can be partially executed
- **Dividends:** Configurable dividend payments
- **Multi-Stock Support:** Trade multiple securities simultaneously

### Advanced Features

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

**LLM Agent Types** (defined in `src/agents/agent_types.py`):
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

Run the health check script to verify all features work correctly:

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

You can define custom scenarios by adding new `SimulationScenario` objects to the `SCENARIOS` dictionary in `src/scenarios.py`.

Each scenario requires:
- A unique `name`.
- A `description`.
- A `parameters` dictionary, which can override the `DEFAULT_PARAMS`.

Here is an example of a new scenario definition:
```python
# src/scenarios.py

"my_custom_scenario": SimulationScenario(
    name="my_custom_scenario",
    description="A custom scenario for testing a new agent type.",
    parameters={
        **DEFAULT_PARAMS,
        "NUM_ROUNDS": 5,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'agent_composition': {
                'my_new_agent': 1,
                'market_maker': 1,
            },
        }
    }
),
```
After adding your new scenario, you can run it by providing its name as a command-line argument as described above. 
