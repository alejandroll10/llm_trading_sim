# Defensive Validation Summary

## Issue
Silent `.get(stock_id, 0)` calls on position dictionaries can mask stock_id mismatch bugs, causing normal sell orders to be treated as short sells.

## Fixes Applied

### 1. **Early Validation (STRICTEST) ✅**
**Location**: `src/agents/agent_manager/services/agent_decision_service.py:113-124`

**What it does**: Validates stock_id exists in agent positions BEFORE creating sell order.

**Result**: Fails fast with clear error message if stock_id mismatch detected.

```python
if side == 'sell':
    agent = self.agent_repository.get_agent(agent_id)
    if stock_id_val not in agent.positions:
        raise ValueError(
            f"Agent {agent_id} cannot sell stock_id='{stock_id_val}' - "
            f"not found in agent positions. Available stocks: {available_stocks}. "
            f"This indicates a stock_id mismatch bug."
        )
```

### 2. **Defensive Logging in Resource Manager ✅**
**Location**: `src/services/agent_resource_manager.py:46-60`

**What it does**: Logs warning if stock_id not found, but continues (allows legitimate short sells from empty position).

**Result**: Non-fatal warning that helps debug stock_id mismatches without breaking simulations.

```python
if stock_id not in agent.positions:
    logger.warning(
        f"Agent {agent.agent_id} committing shares for stock_id='{stock_id}' "
        f"which is not in positions {list(agent.positions.keys())}. "
        f"Treating as short sell from zero position. "
        f"If this is unexpected, it may indicate a stock_id mismatch bug."
    )
    current_shares = 0
else:
    current_shares = agent.positions[stock_id]
```

### 3. **Defensive Logging in Base Agent ✅**
**Location**: `src/agents/base_agent.py:515-524`

**What it does**: Similar defensive logging in agent's commit_shares method.

**Result**: Provides visibility into potential mismatches at the agent level.

## Other `.get()` Locations (Safe Defaults)

The following locations use `.get(stock_id, 0)` but are **SAFE** because:
- They're read-only queries (reporting, calculations)
- Defaulting to 0 is the correct behavior for missing stocks
- They don't affect trade execution logic

### Data Recording (SAFE)
- `src/market/data_recorder.py:273-275` - Recording positions for visualization
- Default of 0 is correct: agent has no position in that stock

### Dividend Calculations (SAFE)
- `src/services/dividend_calculator.py:55-57` - Calculating dividends
- Default of 0 is correct: no shares = no dividends

### Margin Calculations (SAFE)
- `src/agents/services/margin_service.py:100` - Portfolio value calculations
- Default of 0 is correct: missing stock contributes 0 to portfolio

### Release Operations (SAFE)
- `src/services/agent_resource_manager.py:183,186,227,228` - Release tracking
- Default of 0 is safe for tracking borrowed/released amounts

### Position Queries (SAFE)
- `src/agents/base_agent.py:592,598,645,665,674,678,706,723-724` - Position lookups
- Mostly in calculations where 0 is the correct default

### Deterministic Agents (SAFE)
- `src/agents/deterministic/*.py` - Simple trading logic
- Default of 0 is correct behavior

## Prevention Strategy

### Primary Prevention (Schema-level) ✅ IMPLEMENTED
- Single-stock mode: `stock_id` field excluded from LLM schema
- Multi-stock mode: `stock_id` validated against available stocks
- See: `src/agents/LLMs/services/schema_features.py`

### Secondary Prevention (Validation-level) ✅ IMPLEMENTED
- Early validation in order creation (fails fast)
- Defensive logging in critical paths (warns but continues)

### Tertiary Prevention (Monitoring)
- Warning logs help identify any remaining edge cases
- Can be monitored in production to catch regressions

## Testing

Run simulations and check logs for warnings:
```bash
python3 src/run_base_sim.py simple_mixed_traders
grep "stock_id mismatch" logs/simple_mixed_traders/*/agents.log
```

Expected result: No warnings (stock_id handled correctly by schema)

## Summary

✅ **Fixed at source**: Schema prevents LLM from generating wrong stock_id
✅ **Fail fast**: Early validation catches mismatches before execution
✅ **Defensive**: Logging helps debug any edge cases
✅ **Safe defaults**: Read-only queries correctly default to 0

The system is now **robust** against stock_id mismatch bugs!
