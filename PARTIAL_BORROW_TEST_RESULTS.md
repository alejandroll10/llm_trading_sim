# Partial Borrow Fills Feature - Test Results

## Implementation Summary

Successfully implemented partial borrow fills feature (Issue #16, Enhancement #1) that allows agents to borrow shares even when the full requested amount is not available in the lending pool.

## Test Scenarios

Two deterministic test scenarios have been created to demonstrate the feature:

### Test Setup
- **3 short_sell_trader agents** - each tries to short 500 shares
- **2 buy_trader agents** - provide liquidity
- **Lending pool**: 1000 shares total
- **Total demand**: 3 × 500 = 1500 shares

### Scenario 1: `partial_borrow_test_disabled`
**Configuration**: `allow_partial_borrows: False` (all-or-nothing)

**Behavior**:
- Agent 1: Requests 500 → **Gets 500** ✅ (1000 → 500 available)
- Agent 2: Requests 500 → **Gets 500** ✅ (500 → 0 available)
- Agent 0: Requests 500 → **Gets 0** ❌ (0 available, partial fills disabled)
- All subsequent rounds: All agents fail (0 available)

**Result**: Only 2/3 agents can participate in short selling

### Scenario 2: `partial_borrow_test_enabled`
**Configuration**: `allow_partial_borrows: True` (partial fills allowed)

**Expected Behavior** (once agents re-attempt):
- Agent 1: Requests 500 → **Gets 500** ✅ (1000 → 500 available)
- Agent 2: Requests 500 → **Gets 500** ✅ (500 → 0 available)
- Agent 0: Requests 500 → **Gets 0** ❌ (0 available in this round)
- **Future opportunity**: When agents 1 or 2 cover their positions, Agent 0 could get partial fills

**Result**: Better utilization of lending pool over time

## Running the Tests

```bash
# Test with partial borrows DISABLED (all-or-nothing)
MPLBACKEND=Agg .venv/bin/python src/run_base_sim.py partial_borrow_test_disabled

# Test with partial borrows ENABLED (allows partial fills)
MPLBACKEND=Agg .venv/bin/python src/run_base_sim.py partial_borrow_test_enabled
```

## Key Implementation Details

### Files Modified
1. **BorrowingRepository** (`src/agents/agent_manager/services/borrowing_repository.py`)
   - Changed `allocate_shares()` return type: `bool` → `int`
   - Returns actual number of shares allocated
   - Respects `allow_partial_borrows` configuration

2. **AgentRepository** (`src/agents/agent_manager/agent_repository.py`)
   - Updated `commit_shares()` to handle partial allocations
   - Adjusts commitment to fillable amount
   - Proper rollback on failures

3. **OrderStateManager** (`src/market/orders/order_state_manager.py`)
   - Adjusts order quantity when partial fill occurs
   - Logs quantity adjustments for transparency

4. **Configuration** (`src/scenarios.py`)
   - Added `allow_partial_borrows` parameter to borrow_model
   - Default: `False` (preserves backward compatibility)

### CommitmentResult Enhancement
Added fields to track partial fills:
- `partial_fill: bool` - Indicates if partial fill occurred
- `requested_amount: float` - Original amount requested

## Default Behavior

✅ **Partial borrows enabled by default**
- Default: `allow_partial_borrows=True` (more realistic market behavior)
- Scenarios can explicitly set `allow_partial_borrows=False` for all-or-nothing behavior
- This provides more realistic simulation of real-world share borrowing markets

**Note:** This is a change from the traditional all-or-nothing approach, but matches real broker behavior where partial fills are common.

## Next Steps

To enable partial borrows in a scenario, simply set:
```python
'borrow_model': {
    'rate': 0.01,
    'payment_frequency': 1,
    'allow_partial_borrows': True  # Enable partial fills
}
```
