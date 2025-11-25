# Multi-Stock Margin Call Implementation

## Status: IMPLEMENTED ✅

Implementation completed on 2024-XX-XX. This document describes the multi-stock margin call support.

## Current State

### Single-Stock Implementation (Complete)
- Margin checking after price changes in `match_engine.py:_check_and_process_margin_calls()`
- Creates forced buy orders when `borrowed_shares > max_borrowable_shares`
- Orders processed through normal flow: register → validate → commit → match
- Share returns handled automatically via `_process_margin_call_share_returns()`

### Multi-Stock Architecture (Existing)
- Separate matching engine per stock: `self.matching_engines[stock_id]`
- Separate borrowing pool per stock: `self.borrowing_repositories[stock_id]`
- Per-stock positions: `agent.borrowed_positions[stock_id]`
- Orders carry `stock_id` throughout lifecycle

## Key Challenge

In multi-stock mode, each `MatchingEngine` only knows about its own stock. For margin checking:
- **Cash**: Shared across all stocks (one pool)
- **Borrowed positions**: Per-stock (`agent.borrowed_positions[stock_id]`)
- **Max borrowable**: Depends on cash AND stock-specific price

## Implementation Plan

### Phase 1: Pass Stock ID to Matching Engine

**File**: `src/market/engine/match_engine.py`

Currently the matching engine has `is_multi_stock` flag but doesn't know WHICH stock it manages. We need to add a `stock_id` parameter.

```python
# match_engine.py __init__ changes
def __init__(self, ..., stock_id: str = "DEFAULT_STOCK"):
    self.stock_id = stock_id  # Add this
```

**File**: `src/base_sim.py`

Pass stock_id when creating matching engines:

```python
# base_sim.py - already creates engines in a loop
for stock_id in self.contexts.keys():
    self.matching_engines[stock_id] = MatchingEngine(
        ...,
        stock_id=stock_id,  # Add this parameter
    )
```

### Phase 2: Multi-Stock Margin Checking

**File**: `src/market/engine/match_engine.py`

Replace the TODO in `_check_and_process_margin_calls()`:

```python
else:
    # Multi-stock case: check margin for THIS stock only
    for agent_id in self.agent_repository.get_all_agent_ids():
        agent = self.agent_repository.get_agent(agent_id)

        # Get borrowed shares for THIS SPECIFIC stock
        borrowed_for_stock = agent.borrowed_positions.get(self.stock_id, 0)

        # Skip agents without short positions for this stock
        if borrowed_for_stock <= 0:
            continue

        # Check margin requirement at current price for this stock
        max_borrowable = agent.margin_service.get_max_borrowable_shares(current_price)

        if borrowed_for_stock > max_borrowable:
            # Margin violation for this stock!
            excess = borrowed_for_stock - max_borrowable

            LoggingService.get_logger('market').warning(
                f"[MARGIN_VIOLATION] Agent {agent_id} on {self.stock_id}: "
                f"borrowed {borrowed_for_stock:.2f}, "
                f"max allowed {max_borrowable:.2f}, "
                f"excess {excess:.2f} shares"
            )

            # Create forced buy order for this specific stock
            order = self._create_margin_call_order(
                agent=agent,
                quantity=excess,
                price=current_price,
                round_number=round_number,
                stock_id=self.stock_id
            )
            margin_orders.append(order)
```

### Phase 3: Verify Share Return Logic

**File**: `src/market/engine/match_engine.py`

The `_process_margin_call_share_returns()` method already handles per-stock:
- Uses `trade.stock_id` to identify which stock
- Updates `agent.borrowed_positions[stock_id]`
- Returns shares to correct borrowing repo via `_get_borrowing_repo(stock_id)`

**Verify**: The existing implementation should work, but we need to ensure the borrowing repository lookup works correctly:

```python
# Verify this line uses correct repo lookup
borrowing_repo = self.agent_repository._get_borrowing_repo(stock_id)
```

### Phase 4: Test Scenarios

**File**: `src/scenarios/multi_stock.py`

Add a multi-stock short squeeze scenario:

```python
'multi_stock_short_squeeze': {
    'description': 'Test margin calls across multiple stocks',
    'is_multi_stock': True,
    'stock_configs': {
        'STOCK_A': {
            'INITIAL_PRICE': 100,
            'FUNDAMENTAL_PRICE': 100,
            'LENDABLE_SHARES': 1000,
        },
        'STOCK_B': {
            'INITIAL_PRICE': 200,
            'FUNDAMENTAL_PRICE': 200,
            'LENDABLE_SHARES': 500,
        },
    },
    'agent_composition': {
        'multi_stock_short_seller': 1,  # Shorts both stocks
        'squeeze_buyer_a': 1,           # Squeezes STOCK_A
        'squeeze_buyer_b': 1,           # Squeezes STOCK_B
        'deterministic_market_maker': 2,
    },
    'market_config': {
        'MARGIN_REQUIREMENT': 0.5,
        'ENABLE_INTRA_ROUND_MARGIN_CHECKING': True,
    },
}
```

### Phase 5: Verification Updates

**File**: `src/verification/simulation_verifier.py`

Add margin-specific verification for multi-stock:

```python
def verify_margin_consistency_multi_stock(self):
    """Verify margin constraints are satisfied per-stock after margin calls."""
    if not self.is_multi_stock:
        return True

    for stock_id, context in self.contexts.items():
        current_price = context.current_price

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            borrowed = agent.borrowed_positions.get(stock_id, 0)

            if borrowed > 0:
                max_borrowable = agent.margin_service.get_max_borrowable_shares(current_price)

                if borrowed > max_borrowable * 1.01:  # 1% tolerance for rounding
                    self.logger.error(
                        f"Margin violation persists: {agent_id} on {stock_id}"
                    )
                    return False

    return True
```

## Files to Modify

| File | Change |
|------|--------|
| `src/market/engine/match_engine.py` | Add `stock_id` param, implement multi-stock margin check |
| `src/base_sim.py` | Pass `stock_id` to matching engine constructor |
| `src/scenarios/multi_stock.py` | Add test scenario |
| `src/verification/simulation_verifier.py` | Add margin verification for multi-stock |
| `tests/test_multi_stock_margin_calls.py` | Add integration tests |

## Edge Cases to Handle

1. **Agent shorts multiple stocks**: Each stock checked independently by its own matching engine
2. **Cash exhaustion**: If margin call on STOCK_A uses all cash, STOCK_B margin call may fail - need to handle partial deleveraging
3. **Order priority**: If multiple stocks have margin calls same round, which processes first?
4. **Cross-stock effects**: Buying to cover STOCK_A doesn't affect STOCK_B borrowed position

## Risk Considerations

1. **Double margin calls**: If agent is over-margined on both stocks, they get two separate margin calls - could over-correct if both execute same round
2. **Cascading effects**: Margin call liquidation on one stock could affect cash available for other stocks
3. **Timing**: Order of stock processing matters when cash is shared

## Recommended Order Processing

For multi-stock margin calls, process stocks in alphabetical order (deterministic) to ensure reproducible results:

```python
for stock_id in sorted(self.contexts.keys()):
    # Process margin calls for this stock
```

## Implementation Timeline

1. **Phase 1**: Add stock_id parameter (simple plumbing)
2. **Phase 2**: Implement multi-stock margin checking (core logic)
3. **Phase 3**: Verify share returns work correctly
4. **Phase 4**: Add test scenarios
5. **Phase 5**: Add verification

## Notes

- The existing `_process_margin_call_share_returns()` already uses `trade.stock_id` so should work for multi-stock
- The `_create_margin_call_order()` already takes `stock_id` parameter
- Main work is in `_check_and_process_margin_calls()` replacing the TODO block
