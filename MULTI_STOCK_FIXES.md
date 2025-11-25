# Multi-Stock Bug Fixes

## Summary
Fixed two critical bugs preventing multi-stock scenarios from running and improved the DEFAULT_STOCK architecture.

## Fix #1: Logger Initialization (FATAL)

**Problem**: Multi-stock matching engines tried to get stock-specific loggers that don't exist, causing:
```
AttributeError: 'NoneType' object has no attribute 'warning'
```

**Root Cause**: `base_sim.py:364-365` tried to create loggers named `matching_TECH_OVERVALUED`, `trades_TECH_OVERVALUED`, etc. but LoggingService only creates generic loggers like `order_book` and `market`.

**Solution**: Use generic loggers for all stocks in multi-stock mode.

**File Modified**: `src/base_sim.py:364-365`
```python
# Before:
logger=LoggingService.get_logger(f'matching_{stock_id}'),
trades_logger=LoggingService.get_logger(f'trades_{stock_id}'),

# After:
logger=LoggingService.get_logger('order_book'),  # Generic logger
trades_logger=LoggingService.get_logger('market'),  # Generic logger
```

## Fix #2: DEFAULT_STOCK Architecture (DESIGN)

**Problem**: DEFAULT_STOCK accumulator caused confusion and validation warnings in multi-stock mode:
```
INVARIANT VIOLATION: DEFAULT_STOCK accumulator mismatch.
DEFAULT_STOCK=0, sum of other stocks=1000.
```

**Root Cause**: DEFAULT_STOCK was a backwards-compatibility mechanism for single-stock code, but in multi-stock mode it became an inconsistent "accumulator" causing double-counting and confusion.

**Solution**: Clean separation - DEFAULT_STOCK only exists in single-stock mode.

### Architecture Decision

**Single-Stock Mode**:
```python
agent.positions = {"DEFAULT_STOCK": 1000}
agent.shares  # Returns 1000 (positions["DEFAULT_STOCK"])
```

**Multi-Stock Mode**:
```python
agent.positions = {"TECH": 500, "PHARMA": 300}  # No DEFAULT_STOCK
agent.shares  # Returns 800 (sum of all stocks)
```

### Files Modified

#### 1. `src/agents/base_agent.py:138-160` - Smarter `shares` property
```python
@property
def shares(self) -> int:
    """Get shares for DEFAULT_STOCK (single-stock) or total shares (multi-stock)"""
    if "DEFAULT_STOCK" in self.positions:
        return self.positions["DEFAULT_STOCK"]  # Single-stock
    else:
        return sum(self.positions.values())  # Multi-stock

@shares.setter
def shares(self, value: int):
    """Set shares - raises error in multi-stock mode"""
    if "DEFAULT_STOCK" not in self.positions:
        raise ValueError("Cannot use agent.shares setter in multi-stock mode.")
    self._update_position("DEFAULT_STOCK", value)
```

####  2. `src/agents/base_agent.py:194-204` - New helper method
```python
def _update_position(self, stock_id: str, new_value: int):
    """Update a stock position."""
    self.positions[stock_id] = new_value
```

#### 3. `src/agents/base_agent.py:735-737` - Skip DEFAULT_STOCK in totals
```python
@property
def total_shares(self):
    total = 0
    for stock_id in self.positions.keys():
        # Skip DEFAULT_STOCK (only exists in single-stock mode)
        if stock_id != "DEFAULT_STOCK":
            total += self.positions[stock_id] + self.committed_positions.get(stock_id, 0)
    return total
```

#### 4. `src/base_sim.py:533-540` - Don't create DEFAULT_STOCK in multi-stock
```python
if 'initial_positions' in agent_params:
    agent.positions = agent_params['initial_positions'].copy()
    # NOTE: Do NOT add DEFAULT_STOCK in multi-stock mode
    agent.committed_positions = {stock_id: 0 for stock_id in agent.positions.keys()}
    agent.borrowed_positions = {stock_id: 0 for stock_id in agent.positions.keys()}
    agent.initial_shares = sum(agent_params['initial_positions'].values())
```

#### 5. Updated all position updates to use `_update_position()`:
- `src/agents/base_agent.py:641, 704, 708` - Share commit/release
- `src/services/agent_resource_manager.py:248, 254, 286` - Resource management
- `src/agents/services/margin_service.py:259, 475` - Margin calls
- `src/market/state/services/dividend_service.py:170` - Redemption

## Benefits

### Cleaner Architecture
- DEFAULT_STOCK only exists in single-stock mode
- No confusing accumulator that needs to be skipped everywhere
- Clear separation between single-stock and multi-stock code paths

### Backwards Compatible
- `agent.shares` still works in single-stock mode (returns DEFAULT_STOCK)
- `agent.shares` now returns meaningful total in multi-stock mode
- Setting `agent.shares` in multi-stock raises clear error with guidance

### No More Warnings
- Eliminated DEFAULT_STOCK mismatch warnings
- No double-counting in share conservation checks
- Cleaner invariant validation

## Testing
- ✅ Multi-stock scenarios work (gptoss_multistock_memory_test - Successfully completed)
- ✅ No DEFAULT_STOCK warnings in multi-stock mode
- ✅ Memory system works across multiple stocks
- ✅ Share conservation maintained
- ✅ Final redemption correctly redeems all shares to 0

**Note**: Single-stock scenarios with heavy trading currently have a pre-existing cash commitment tracking bug unrelated to these fixes.

## Related Issues
- Built on top of #67 and #68 (stock_id modular design)
- Respects the feature toggle architecture from #63
