# Leverage Trading Documentation

## Overview

Leverage trading (margin trading) allows agents to borrow cash to amplify their long positions beyond their available capital. This feature enables research on:
- Risk management with borrowed capital
- Margin call dynamics and forced liquidations
- LLM agent behavior under leverage constraints
- Market stability with leveraged participants
- Strategy performance amplification

## Quick Start

### Basic Configuration

```python
from scenarios import SimulationScenario, DEFAULT_PARAMS

leverage_scenario = SimulationScenario(
    name="my_leverage_test",
    description="Test leverage trading with 2x leverage",
    parameters={
        **DEFAULT_PARAMS,
        "leverage_enabled": True,
        "NUM_ROUNDS": 50,
        "AGENT_PARAMS": {
            'allow_short_selling': False,
            'agent_composition': {
                'momentum_trader': 2,
            },
            'deterministic_params': {
                'momentum_trader': {
                    'leverage_ratio': 2.0,        # Max 2x leverage
                    'initial_margin': 0.5,        # 50% down payment required
                    'maintenance_margin': 0.25,   # Liquidate below 25% margin
                }
            }
        },
        "leverage_interest_rate": 0.05,  # 5% annual interest rate
    }
)
```

### Running Leverage Scenarios

```bash
# Run deterministic leverage test
python src/run_base_sim.py test_leverage

# Run LLM leverage test
python src/run_base_sim.py test_leverage_llm
```

## How It Works

### 1. Borrowing Mechanism

When an agent places a buy order that exceeds their available cash:

```
Agent wants to buy: 100 shares @ $100 = $10,000
Agent has cash: $6,000
Agent needs to borrow: $4,000
```

**If leverage enabled (e.g., 2x):**
1. Calculate available borrowing power based on current equity and positions
2. Automatically borrow required cash from lending pool
3. Execute the order
4. Track borrowed cash and charge interest

**If leverage disabled:**
- Order is rejected with "Insufficient cash" error

### 2. Margin Requirements

**Initial Margin:** Minimum equity required to open a leveraged position
- Example: 50% initial margin means you must put up $5,000 to buy $10,000 worth of stock

**Maintenance Margin:** Minimum equity required to keep a position open
- Example: 25% maintenance margin means your equity must stay above 25% of position value
- If equity falls below this, a margin call is triggered

**Leverage Ratio:** Maximum position value relative to equity
- 1.0 = No leverage (100% cash only)
- 2.0 = 2x leverage (can control $2 of stock per $1 of equity)
- 3.0 = 3x leverage (can control $3 of stock per $1 of equity)

### 3. Margin Calls

When an agent's margin ratio falls below the maintenance margin:

```
Equity: $3,000
Position Value: $15,000
Margin Ratio: $3,000 / $15,000 = 20%

If maintenance_margin = 25%:
→ MARGIN CALL TRIGGERED
```

**Liquidation Process:**
1. Calculate target position value to restore initial margin (more conservative)
2. Liquidate positions proportionally across all stocks
3. Use proceeds to repay borrowed cash
4. Log the forced liquidation

**Example:**
```
Before margin call:
- Equity: $3,000
- Position value: $15,000
- Borrowed cash: $8,000
- Margin ratio: 20% (below 25% threshold)

After liquidation:
- Sell $9,000 worth of stock
- Repay $8,000 borrowed cash
- Remaining equity: $3,000
- New position value: $6,000
- New margin ratio: 50% (restored to initial_margin)
```

### 4. Interest Charges

Interest is charged per round on all borrowed cash:

```
Annual interest rate: 5%
Rounds per year: 252 (trading days)
Per-round rate: 5% / 252 = 0.0198% per round

If borrowed cash = $10,000:
Interest per round = $10,000 * 0.0198% = $1.98
```

Interest is:
- Deducted from agent's cash
- Tracked in `leverage_interest_paid`
- Recorded as a payment in agent's transaction history

## Architecture

### Core Components

#### 1. CashLendingRepository
**Location:** `src/agents/agent_manager/services/cash_lending_repository.py`

Manages the pool of cash available for lending:

```python
class CashLendingRepository:
    def __init__(self, total_lendable_cash: float):
        self.total_lendable_cash = total_lendable_cash
        self.available_cash = total_lendable_cash
        self.borrowed: Dict[str, float] = {}

    def allocate_cash(self, agent_id: str, amount: float) -> float:
        """Allocate cash to agent. Returns actual amount allocated."""

    def release_cash(self, agent_id: str, amount: float):
        """Return borrowed cash to pool."""
```

**Key Features:**
- Tracks total lending pool
- Manages per-agent borrowing
- Supports partial borrows if pool limited
- Thread-safe operations

#### 2. BaseAgent Leverage Methods
**Location:** `src/agents/base_agent.py`

Key methods added to agents:

```python
# State tracking
self.borrowed_cash: float = 0.0
self.leverage_ratio: float = 2.0
self.maintenance_margin: float = 0.25
self.initial_margin: float = 0.5
self.leverage_interest_paid: float = 0.0
self.cash_lending_repo: CashLendingRepository = None

# Helper methods
def get_equity(self, prices: Dict[str, float]) -> float:
    """Calculate equity = assets - liabilities"""

def get_gross_position_value(self, prices: Dict[str, float]) -> float:
    """Get total market value of all long positions"""

def get_leverage_margin_ratio(self, prices: Dict[str, float]) -> float:
    """Calculate current margin ratio (equity / position_value)"""

def get_available_borrowing_power(self, prices: Dict[str, float]) -> float:
    """Calculate additional cash that can be borrowed"""

def is_under_leverage_margin(self, prices: Dict[str, float]) -> bool:
    """Check if below maintenance margin"""
```

#### 3. Auto-Borrowing in commit_cash()
**Location:** `src/agents/base_agent.py:367-477`

When committing cash for an order:

```python
def commit_cash(self, amount: float, debug: bool = False):
    # Check if we need to borrow
    if amount > self.cash:
        if self.leverage_ratio <= 1.0:
            raise ValueError("Insufficient cash")

        # Calculate shortage and borrowing power
        shortage = amount - self.cash
        borrowing_power = self.get_available_borrowing_power(self.last_prices)

        if shortage > borrowing_power:
            raise ValueError("Insufficient buying power")

        # Borrow the shortage
        actual_borrowed = self.cash_lending_repo.allocate_cash(
            self.agent_id, shortage
        )
        self.borrowed_cash += actual_borrowed
        self.cash += actual_borrowed

    # Commit the cash
    self.cash -= amount
    self.committed_cash += amount
```

#### 4. Margin Call Handler
**Location:** `src/agents/base_agent.py:1449-1560`

Forced liquidation when under-margined:

```python
def handle_leverage_margin_call(
    self,
    prices: Dict[str, float],
    round_number: int
):
    """Force sell positions when leverage margin requirements violated."""

    if self.borrowed_cash <= 0:
        return

    if not self.is_under_leverage_margin(prices):
        return

    # Calculate liquidation needed
    equity = self.get_equity(prices)
    gross_position_value = self.get_gross_position_value(prices)
    target_position_value = equity / self.initial_margin
    value_to_liquidate = gross_position_value - target_position_value

    # Liquidate proportionally across stocks
    for stock_id, price in prices.items():
        position_shares = self.positions.get(stock_id, 0)
        proportion = (position_shares * price) / gross_position_value
        shares_to_sell = (value_to_liquidate * proportion) / price

        # Execute forced sale
        self.positions[stock_id] -= shares_to_sell
        proceeds = shares_to_sell * price
        self.cash += proceeds

    # Use proceeds to repay borrowed cash
    repayment = min(total_proceeds, self.borrowed_cash)
    self.cash -= repayment
    self.borrowed_cash -= repayment
    self.cash_lending_repo.release_cash(self.agent_id, repayment)
```

#### 5. LeverageInterestService
**Location:** `src/market/state/services/leverage_interest_service.py`

Charges per-round interest on borrowed cash:

```python
class LeverageInterestService:
    def __init__(self, annual_interest_rate: float = 0.05):
        self.annual_interest_rate = annual_interest_rate

    def charge_interest(
        self,
        agents: List[BaseAgent],
        rounds_per_year: int = 252
    ) -> Dict[str, float]:
        """Charge interest on borrowed cash for all agents."""

        per_round_rate = self.annual_interest_rate / rounds_per_year

        for agent in agents:
            if agent.borrowed_cash > 0:
                interest = agent.borrowed_cash * per_round_rate
                agent.cash -= interest
                agent.leverage_interest_paid += interest
```

### Integration Points

#### Initialization (base_sim.py)

```python
# In __init__
if scenario.get('leverage_enabled', False):
    self.cash_lending_repo = CashLendingRepository(
        total_lendable_cash=scenario.get('cash_lending_pool', float('inf'))
    )
    self.leverage_interest_service = LeverageInterestService(
        annual_interest_rate=scenario.get('leverage_interest_rate', 0.05)
    )

    # Assign to agents during creation
    for agent in agents:
        agent.cash_lending_repo = self.cash_lending_repo
```

#### Per-Round Processing (base_sim.py:execute_round)

```python
# After processing interest on cash reserves
if self.leverage_enabled:
    self.leverage_interest_service.charge_interest(
        self.agent_repository.get_all_agents(),
        rounds_per_year=252
    )
```

#### Wealth Update (base_agent.py:update_wealth)

```python
# Subtract borrowed cash from wealth calculation
self.wealth = self.total_cash + share_value - self.borrowed_cash

# Check leverage margin requirements
if self.borrowed_cash > 0:
    self.handle_leverage_margin_call(prices, round_number)
```

## Configuration Reference

### Scenario-Level Parameters

```python
{
    # Enable leverage feature
    "leverage_enabled": True,  # Default: False

    # Annual interest rate on borrowed cash (5% = 0.05)
    "leverage_interest_rate": 0.05,  # Default: 0.05

    # Total cash available for lending (None = unlimited)
    "cash_lending_pool": 1000000,  # Default: float('inf')

    # Allow partial borrows if pool exhausted
    "allow_partial_borrows": False,  # Default: False
}
```

### Agent-Level Parameters

```python
{
    "AGENT_PARAMS": {
        'deterministic_params': {
            'momentum_trader': {
                # Maximum leverage ratio (2.0 = 2x leverage)
                'leverage_ratio': 2.0,  # Default: 1.0 (no leverage)

                # Initial margin requirement (0.5 = 50%)
                'initial_margin': 0.5,  # Default: 0.5

                # Maintenance margin (liquidation threshold)
                'maintenance_margin': 0.25,  # Default: 0.25
            }
        }
    }
}
```

### Per-Agent Override

You can set different leverage ratios for individual agents:

```python
"agents": [
    {
        "agent_type": "momentum_trader",
        "leverage_ratio": 3.0,  # Aggressive leverage
        "initial_margin": 0.33,
        "maintenance_margin": 0.20
    },
    {
        "agent_type": "momentum_trader",
        "leverage_ratio": 1.5,  # Conservative leverage
        "initial_margin": 0.67,
        "maintenance_margin": 0.50
    },
    {
        "agent_type": "value_investor",
        "leverage_ratio": 1.0,  # No leverage
    }
]
```

## LLM Agent Integration

### Leverage Information Available to LLMs

LLM agents receive comprehensive leverage metrics in their observations:

```python
{
    "leverage_info": {
        "leverage_enabled": True,
        "borrowed_cash": 5000.0,
        "equity": 12000.0,
        "gross_position_value": 17000.0,
        "leverage_margin_ratio": 0.706,  # 70.6% margin
        "maintenance_margin": 0.25,
        "available_borrowing_power": 9000.0,
        "max_leverage_ratio": 2.0,
        "leverage_interest_paid": 124.5,
        "is_under_margined": False
    }
}
```

### LLM Prompt Enhancement

The leverage information is automatically included in LLM prompts:

```
Current Financial Position:
- Cash: $7,000
- Borrowed Cash: $5,000
- Equity: $12,000
- Leverage Margin Ratio: 70.6%
- Maintenance Margin: 25% (liquidation if below)
- Available Borrowing Power: $9,000
- Interest Paid (total): $124.50

Risk Status: ✓ Healthy margin (well above maintenance requirement)
```

This allows LLMs to:
- Understand their leverage exposure
- Make informed decisions about position sizing
- Manage risk proactively
- Avoid margin calls

## Research Applications

### Example Research Questions

1. **Risk Management:**
   - Do LLM agents manage leverage risk effectively?
   - How do agents respond to approaching margin calls?
   - Can LLMs learn optimal leverage ratios over time?

2. **Market Dynamics:**
   - Do leveraged agents amplify market volatility?
   - How do margin calls affect price discovery?
   - Do liquidation cascades occur with multiple leveraged agents?

3. **Strategy Performance:**
   - Does leverage improve risk-adjusted returns?
   - How does leverage affect strategy stability?
   - What leverage ratios optimize Sharpe ratios?

4. **Behavioral Analysis:**
   - Do LLM agents exhibit overconfidence with leverage?
   - How do agents react to forced liquidations?
   - Can agents learn from margin call experiences?

### Example Experimental Designs

#### Experiment 1: Optimal Leverage Ratio

```python
leverage_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]

for ratio in leverage_ratios:
    scenario = create_scenario(
        leverage_ratio=ratio,
        num_agents=10,
        num_rounds=100
    )
    results = run_simulation(scenario)
    analyze_sharpe_ratio(results)
```

#### Experiment 2: Margin Call Contagion

```python
# Test if margin calls trigger cascading liquidations
scenario = {
    "agents": [
        {"leverage_ratio": 3.0} for _ in range(20)  # All highly leveraged
    ],
    "price_shock": -20%  # Sudden price drop
}
analyze_liquidation_cascades(run_simulation(scenario))
```

#### Experiment 3: LLM Risk Management

```python
# Compare LLM vs deterministic agents with leverage
llm_scenario = create_scenario(
    agent_composition={'llm_trader': 5},
    leverage_ratio=2.0
)

deterministic_scenario = create_scenario(
    agent_composition={'momentum_trader': 5},
    leverage_ratio=2.0
)

compare_margin_call_frequency(
    run_simulation(llm_scenario),
    run_simulation(deterministic_scenario)
)
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
python tests/test_leverage_basic.py
```

**Test Coverage:**
- Cash lending repository operations
- Leverage helper method calculations
- Margin call triggering logic
- Interest service accuracy
- Borrowing power enforcement
- Repository exhaustion handling
- Multi-stock leverage calculations
- Bankruptcy scenarios

### Integration Tests

Test end-to-end leverage scenarios:

```bash
# Test leverage with deterministic agents
python src/run_base_sim.py test_leverage

# Test leverage with LLM agents
python src/run_base_sim.py test_leverage_llm
```

### Custom Testing

Create your own test scenarios:

```python
from scenarios import SimulationScenario, DEFAULT_PARAMS

test_scenario = SimulationScenario(
    name="my_leverage_test",
    description="Custom leverage test",
    parameters={
        **DEFAULT_PARAMS,
        "leverage_enabled": True,
        "NUM_ROUNDS": 20,
        # ... your configuration
    }
)

# Run and analyze
python src/run_base_sim.py my_leverage_test
```

## Troubleshooting

### Common Issues

**1. Orders rejected with "Insufficient cash"**
- **Cause:** Leverage not enabled or leverage_ratio = 1.0
- **Fix:** Set `leverage_enabled: True` and `leverage_ratio > 1.0` in scenario

**2. Immediate margin calls**
- **Cause:** Leverage ratio too high or margin requirements too strict
- **Fix:** Lower leverage_ratio or decrease maintenance_margin

**3. "Cannot compute borrowing power without current prices"**
- **Cause:** Price data not available during validation (should be fixed in issue #42)
- **Fix:** Ensure you're using the latest version with the #42 fix

**4. Lending pool exhausted**
- **Cause:** `cash_lending_pool` set too low
- **Fix:** Increase pool size or set to `float('inf')` for unlimited

### Validation

Check leverage is working correctly:

```python
# After simulation, check agent states
for agent in simulation.agents:
    print(f"Agent {agent.agent_id}:")
    print(f"  Borrowed cash: ${agent.borrowed_cash:.2f}")
    print(f"  Interest paid: ${agent.leverage_interest_paid:.2f}")
    print(f"  Margin ratio: {agent.get_leverage_margin_ratio(prices):.2%}")
```

## Future Enhancements

Potential additions (not currently implemented):

- **Portfolio Margin:** Cross-margining across multiple stocks
- **Variable Margin Rates:** Risk-based margin requirements per agent
- **Grace Periods:** Delay before forced liquidation
- **Leverage Metrics Plotting:** Visualize borrowing and margin over time
- **Smart Liquidation:** Intelligent order of position closures

See GitHub issues for planned enhancements.

## References

### Key Files

- **Core Implementation:**
  - `src/agents/base_agent.py` - Leverage methods and margin calls
  - `src/agents/agent_manager/services/cash_lending_repository.py`
  - `src/market/state/services/leverage_interest_service.py`

- **Integration:**
  - `src/base_sim.py` - Simulation orchestration
  - `src/market/orders/order_state_manager.py` - Order validation

- **Configuration:**
  - `src/scenarios/test_scenarios.py` - Example scenarios
  - `src/scenarios/base.py` - Default parameters

- **Testing:**
  - `tests/test_leverage_basic.py` - Unit tests

### Related Issues

- **#37:** Add Leverage Trading Capability (main implementation)
- **#42:** Fix leverage validation with prices (required for LLM agents)
- **#38:** Multi-Stock Margin Calls (short selling integration)

### Academic References

For theoretical background on margin trading and leverage:
- Regulation T (Federal Reserve) - Margin requirements
- Risk Management in Financial Markets
- Portfolio Theory with Leverage

## Support

For questions or issues:
1. Check this documentation
2. Review test cases in `tests/test_leverage_basic.py`
3. Examine example scenarios: `test_leverage`, `test_leverage_llm`
4. Open a GitHub issue with detailed description

## License

Same as parent project.
