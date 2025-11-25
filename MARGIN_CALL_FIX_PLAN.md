# Detailed Implementation Plan: Margin Call Fix

## 🎯 Progress Tracker

- ✅ **Phase 0: Planning** - Architecture designed, plan documented
- ✅ **Phase 1: Modular Configuration** - Feature flag infrastructure added
- ✅ **Phase 2: Single-Stock Detection** - Detection logic implemented and tested
- ✅ **Phase 3: Order Creation** - Order creation helper implemented and integrated
- ✅ **Phase 4: Infrastructure Verification** - Margin checking confirmed working correctly
- ✅ **Phase 5: Share Return Mechanism** - Automatic share returns implemented
- ⚠️ **Phase 6: Order Execution Integration** - Margin call orders created but need state machine adjustments
- ⏳ **Phase 7: Multi-Stock Support** - Pending
- ⏳ **Phase 8: Final Testing & Validation** - Pending

### Latest Update
**Date**: 2025-11-24
**Status**: Phase 6 In Progress - Margin violations successfully triggered, order execution needs integration

**Phase 4 (Infrastructure Verification)**:
- Created deterministic test scenario (`margin_violation_test` in comprehensive_tests.py)
- Created short squeeze scenario (`short_squeeze_test`) with `SqueezeBuyerAgent` that triggers price spikes
- Verified `[MARGIN_CHECK]` messages appear every round (confirmed infrastructure running)
- Tested with margin_requirement ranging from 1.5 to 3.0
- **SUCCESS**: Triggered actual margin violation with 214% price spike ($50 → $157)
  - Agent borrowed: 413 shares
  - Max allowed: 304.19 shares
  - **Violation detected: 108.81 excess shares**
- Confirmed detection logic correctly calculates max_borrowable_shares

**Phase 5 (Share Return Mechanism)**:
- Implemented `_process_margin_call_share_returns()` in match_engine.py:248-286
- Automatically returns borrowed shares to lending pool after margin call trades execute
- When agent buys shares via margin call, shares immediately cover borrowed position
- Logs: `[MARGIN_CALL_RETURN]` messages track share returns
- Integrated into margin call flow (line 127)

**Phase 6 (Order Execution Integration)** - ⚠️ IN PROGRESS:
- ✅ Margin call orders created successfully
- ✅ Orders registered in order repository (match_engine.py:115-116)
- ⚠️ **Remaining**: Order state machine needs adjustments for forced orders
  - Issue: Forced orders bypass normal validation (no cash commitment upfront)
  - Need: Special state transition path for system-forced orders
  - Files affected: `market_handler.py`, order state manager
- Added validation skip for margin call orders (market_handler.py:96-98)
- Next: Complete state machine integration or use alternative execution path

**Effective Leverage Analysis**:
- `margin_requirement: 1.5` (150%) = effective leverage **0.67x**
- Agent with $38k can borrow 506 shares @ $50
- Short selling GIVES cash (~$25k proceeds), providing collateral buffer
- Requires extreme price spikes (>70%) to trigger violations

**Infrastructure Status**: ✅ **DETECTION COMPLETE**, ⚠️ **EXECUTION IN PROGRESS**
- ✅ Margin checks run every round
- ✅ Violations detected correctly
- ✅ Forced orders created with correct quantity
- ✅ Share return mechanism ready
- ⚠️ Order execution needs state machine integration

**Next**: Complete Phase 6 order execution integration, then multi-stock support (Phase 7)

---

## Overview
Convert margin calls from direct state manipulation to real market orders that execute during the matching phase.

## Architecture

```
Round N Flow:
1. Update market → current_price
2. Collect decisions → orders from agents
3. Match orders:
   a. Process liquidations (existing)
   b. Process limit orders (existing)
   c. Process market orders (existing)
   d. Calculate new price from trades
   e. ✨ CHECK MARGINS at new price (NEW)
   f. Create forced margin call orders (NEW)
   g. Process margin call orders (NEW)
   h. Recalculate final price (NEW)
4. Update wealth (no manipulation needed!)
5. Record data
6. End of round
7. Verify (should pass!)
```

---

## ✅ Completed: Phase 1 - Modular Configuration

### What Was Done

**1. Added Feature Flag to BaseSimulation**
- File: `src/base_sim.py:71`
- Parameter: `enable_intra_round_margin_checking: bool = False`
- Defaults to `False` for backwards compatibility
- Stored in instance: `self.enable_intra_round_margin_checking`

**2. Passed Flag to MatchingEngine**
- Single-stock: `src/base_sim.py:390`
- Multi-stock: `src/base_sim.py:374` (for each stock)
- MatchingEngine now accepts and stores the flag

**3. Added Skeleton Method**
- File: `src/market/engine/match_engine.py:118-146`
- Method: `_check_and_create_margin_call_orders()`
- Returns empty list when feature disabled
- Includes logging and TODO for implementation

**4. Integrated Into match_orders Flow**
- File: `src/market/engine/match_engine.py:106-132`
- Calls margin check AFTER price calculation, BEFORE wealth update
- Processes returned orders if any
- Recalculates price after margin trades

### How to Use
```python
sim = BaseSimulation(
    # ... standard parameters ...
    enable_intra_round_margin_checking=True  # Enable feature
)
```

### Testing
- Feature disabled by default - no impact on existing code
- Feature enabled - method called but returns empty list (safe)

---

## Implementation Steps

### Step 1: Add Margin Check Method to MatchingEngine ✅ DONE (skeleton)

**File:** `src/market/engine/match_engine.py`

**Add new method:**
```python
def _check_and_create_margin_call_orders(self, current_price: float, round_number: int) -> List[Order]:
    """Check all agents for margin violations and create forced orders.
    
    Args:
        current_price: Price after regular trading
        round_number: Current round number
        
    Returns:
        List of forced market orders to resolve margin violations
    """
    margin_orders = []
    
    if self.is_multi_stock:
        # Multi-stock: need all current prices
        # Get prices from all contexts
        prices = {stock_id: ctx.current_price 
                 for stock_id, ctx in self.contexts.items()}
        
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            
            # Check if agent has any short positions
            if agent.total_borrowed_shares <= 0:
                continue
                
            # Get margin status
            margin_status = agent.margin_service.get_portfolio_margin_status(prices)
            
            if margin_status['is_margin_violated']:
                # Create forced orders proportionally across stocks
                orders = self._create_multi_stock_margin_orders(
                    agent, prices, margin_status, round_number
                )
                margin_orders.extend(orders)
    else:
        # Single stock: simpler check
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            
            # Check if agent has short position
            if agent.borrowed_shares <= 0:
                continue
            
            # Check margin requirement
            max_borrowable = agent.margin_service.get_max_borrowable_shares(current_price)
            
            if agent.borrowed_shares > max_borrowable:
                # Create forced buy order
                excess = agent.borrowed_shares - max_borrowable
                order = self._create_margin_call_order(
                    agent, excess, current_price, round_number
                )
                margin_orders.append(order)
    
    return margin_orders
```

**Add helper methods:**
```python
def _create_margin_call_order(self, agent, quantity: float, 
                              price: float, round_number: int) -> Order:
    """Create a forced market buy order for margin call."""
    from market.orders.order import Order
    
    order = Order(
        agent_id=agent.agent_id,
        order_type='market',
        side='buy',
        quantity=quantity,
        round_placed=round_number,
        stock_id=agent.stock_id if hasattr(agent, 'stock_id') else "DEFAULT_STOCK",
        is_margin_call=True  # Flag this as forced
    )
    
    LoggingService.get_logger('market').warning(
        f"[MARGIN_CALL] Creating forced buy order for agent {agent.agent_id}: "
        f"{quantity} shares @ ${price:.2f}"
    )
    
    return order

def _create_multi_stock_margin_orders(self, agent, prices: dict, 
                                      margin_status: dict, round_number: int) -> List[Order]:
    """Create proportional margin call orders across multiple stocks."""
    orders = []
    total_borrowed_value = margin_status['borrowed_value']
    excess_value = margin_status['excess_borrowed_value']
    
    for stock_id, price in prices.items():
        if stock_id == "DEFAULT_STOCK":
            continue
            
        borrowed_shares = agent.borrowed_positions.get(stock_id, 0)
        if borrowed_shares <= 0:
            continue
        
        # Calculate proportion to cover
        stock_value = borrowed_shares * price
        proportion = stock_value / total_borrowed_value if total_borrowed_value > 0 else 0
        shares_to_cover = min(
            (excess_value * proportion) / price if price > 0 else 0,
            borrowed_shares
        )
        
        if shares_to_cover > 0:
            order = Order(
                agent_id=agent.agent_id,
                order_type='market',
                side='buy',
                quantity=shares_to_cover,
                round_placed=round_number,
                stock_id=stock_id,
                is_margin_call=True
            )
            orders.append(order)
    
    return orders
```

**Modify `match_orders` method (around line 103):**
```python
def match_orders(self, new_orders: List[Order], current_price: float, round_number: int) -> MarketResult:
    """Match orders for a trading round with prioritized processing"""
    # ... existing code for liquidations, limits, markets ...
    
    # Line 102-103: Calculate new price
    LoggingService.log_trades(trades, round_number)
    new_price = self.trade_processing_service.calculate_new_price(trades, current_price)
    
    # ✨ NEW: Check for margin violations and create forced orders
    margin_orders = self._check_and_create_margin_call_orders(new_price, round_number)
    
    if margin_orders:
        LoggingService.get_logger('market').warning(
            f"[MARGIN_CALL] Processing {len(margin_orders)} forced margin call orders"
        )
        
        # Process margin call orders through normal matching
        margin_trades, _ = self.market_order_handler.process_orders(
            margin_orders, new_price, round_number
        )
        
        # Add to trades list
        trades.extend(
            self.trade_processing_service.process_market_order_results(
                margin_trades, [], []
            )
        )
        
        # Recalculate price after margin call trades
        new_price = self.trade_processing_service.calculate_new_price(trades, new_price)
        
        LoggingService.get_logger('market').info(
            f"[MARGIN_CALL] Processed {len(margin_trades)} margin call trades, "
            f"new price: ${new_price:.2f}"
        )
    
    # Line 107-109: Update wealth and log
    if not self.is_multi_stock:
        self.agent_repository.update_all_wealth(new_price)
    LoggingService.log_market_state(self.order_book, round_number, "End of Round State")
    
    return MarketResult(...)
```

---

### Step 2: Handle Share Return After Trade Fills

**Challenge:** When margin call order fills, we need to return shares to lending pool.

**Option A: Hook in Trade Execution**

**File:** `src/market/engine/services/trade_processing_service.py` or trade execution

Add check after trade execution:
```python
def process_market_order_results(self, market_trades, ...):
    """Process results from market order matching"""
    for trade in market_trades:
        # ... existing trade processing ...
        
        # Check if this was a margin call order
        buyer_order = self.order_repository.get_order(trade.buy_order_id)
        if buyer_order and buyer_order.is_margin_call:
            # This buy covered a short position - return shares to pool
            self._handle_short_covering(trade)
    
    return trades

def _handle_short_covering(self, trade: Trade):
    """Handle returning shares to pool when short position is covered."""
    agent = self.agent_repository.get_agent(trade.buyer_id)
    stock_id = trade.stock_id
    quantity = trade.quantity
    
    # Reduce borrowed position
    current_borrowed = agent.borrowed_positions.get(stock_id, 0)
    shares_to_return = min(quantity, current_borrowed)
    
    if shares_to_return > 0:
        agent.borrowed_positions[stock_id] = current_borrowed - shares_to_return
        
        # Return to lending pool
        borrowing_repo = self._get_borrowing_repo(stock_id)
        borrowing_repo.release_shares(agent.agent_id, shares_to_return)
        
        LoggingService.get_logger('market').info(
            f"[MARGIN_CALL] Agent {agent.agent_id} covered {shares_to_return} "
            f"shares of {stock_id}, returned to pool"
        )
```

**Option B: Add to Position Update Service**

Modify `update_shares_with_covering` to detect margin call context.

**Recommendation: Option A** - cleaner, handles it immediately after trade.

---

### Step 3: Disable Direct Manipulation in margin_service.py

**File:** `src/agents/services/margin_service.py`

**Modify `handle_margin_call` method (line 153):**
```python
def handle_margin_call(self, current_price: float, round_number: int):
    """Check for margin violations.
    
    NOTE: This method now only CHECKS for violations.
    Actual forced orders are created by the matching engine.
    Direct state manipulation is DEPRECATED.
    """
    if self.agent.borrowed_shares <= 0:
        return
    
    max_borrowable = self.get_max_borrowable_shares(current_price)
    if self.agent.borrowed_shares > max_borrowable:
        excess = self.agent.borrowed_shares - max_borrowable
        
        # Just log the violation - matching engine will handle it
        LoggingService.log_margin_call(
            round_number=round_number,
            agent_id=self.agent.agent_id,
            agent_type=self.agent.agent_type.name,
            borrowed_shares=self.agent.borrowed_shares,
            max_borrowable=max_borrowable,
            action="VIOLATION_DETECTED",
            excess_shares=excess,
            price=current_price
        )
        
        LoggingService.get_logger('agents').warning(
            f"[MARGIN_VIOLATION] Agent {self.agent.agent_id} exceeded margin: "
            f"borrowed {self.agent.borrowed_shares}, max {max_borrowable:.2f}"
        )
        
        # NO MORE DIRECT MANIPULATION!
        # The matching engine will create forced orders
```

**Similarly for `handle_multi_stock_margin_call`** - convert to detection only.

---

### Step 4: Handle Edge Cases

#### 4.1 No Liquidity Scenario

If margin call order can't fill (no sellers in book):

**Option A:** Create synthetic seller
```python
if len(margin_trades) < len(margin_orders):
    # Some orders didn't fill
    for order in margin_orders:
        if order.state != OrderState.FILLED:
            # Create synthetic trade at current price
            synthetic_trade = self._create_synthetic_trade(order, new_price)
            trades.append(synthetic_trade)
```

**Option B:** Let it partially fill, retry next round
- Orders stay in book
- Agent goes further underwater
- More realistic but riskier

**Recommendation:** Start with Option A (synthetic seller) for stability.

#### 4.2 Cascading Margin Calls

If margin call trades cause OTHER agents to violate margin:

**Solution:** Single pass per round
- Only check margins once after regular trading
- Don't recursively check after margin trades
- Cascades will be caught next round

#### 4.3 Agent Bankruptcy

If agent can't cover even with all available cash:

**Solution:** Mark agent as bankrupt
- Liquidate all positions
- Return all borrowed shares
- Set cash to 0
- Remove from future trading

---

### Step 5: Testing Strategy

**Test 1: Single Agent, Simple Violation**
- Agent shorts 10k shares at $100
- Price rises to $150
- Margin violated → forced buy
- Verify: trade executes, shares returned, cash conserved

**Test 2: Multiple Agents**
- 2 agents short
- Both violate margin
- Verify: both get forced orders, no interference

**Test 3: No Liquidity**
- Agent violates margin
- Order book empty
- Verify: synthetic seller creates trade OR order waits

**Test 4: Multi-Stock**
- Agent shorts multiple stocks
- One stock triggers margin violation
- Verify: proportional covering across stocks

**Test 5: single_leverage_short (Full Scenario)**
- Run existing test scenario
- Verify: cash conservation passes, share conservation passes

---

## Files to Modify

1. **src/market/engine/match_engine.py**
   - Add `_check_and_create_margin_call_orders()`
   - Add `_create_margin_call_order()`
   - Add `_create_multi_stock_margin_orders()`
   - Modify `match_orders()` to call margin check

2. **src/market/engine/services/trade_processing_service.py**
   - Add `_handle_short_covering()` 
   - Modify `process_market_order_results()` to check for margin call orders

3. **src/agents/services/margin_service.py**
   - Modify `handle_margin_call()` to only detect, not manipulate
   - Modify `handle_multi_stock_margin_call()` similarly

4. **src/market/orders/order.py**
   - Already has `is_margin_call` flag ✓

---

## Success Criteria

1. ✅ Margin calls execute as real trades
2. ✅ Cash transfers to actual sellers (zero-sum)
3. ✅ Cash conservation verification passes
4. ✅ Share conservation verification passes
5. ✅ Shares returned to lending pool automatically
6. ✅ Multi-stock proportional covering works
7. ✅ No direct state manipulation in margin_service

---

## Risk Mitigation

**Risk 1:** Margin call orders fail validation
- **Mitigation:** Mark as `is_margin_call=True`, skip normal validation

**Risk 2:** Infinite loop of margin calls
- **Mitigation:** Single pass per round, no recursion

**Risk 3:** Breaking existing tests
- **Mitigation:** Keep old code as fallback initially, gradual migration

---

## Implementation Order

1. ✅ Add `is_margin_call` flag to Order (DONE)
2. Add margin check method to MatchingEngine
3. Integrate into `match_orders()` flow
4. Add share return mechanism
5. Disable direct manipulation
6. Test single-stock scenario
7. Add multi-stock support
8. Test comprehensive scenarios

Ready to implement?
