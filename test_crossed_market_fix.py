"""
Test for crossed market fix (Issue #88).

This test verifies that aggressive limit orders (converted from unfilled market orders)
are properly matched against the book before being added, preventing crossed markets.

Bug scenario:
1. Book has Ask at $51.50
2. Market buy can't fill due to insufficient cash commitment
3. Becomes aggressive limit at $51.50 * 1.10 = $56.65
4. OLD BUG: Added directly to book → crossed market (bid $56.65 > ask $51.50)
5. FIX: Should match against the ask first
"""

import sys
sys.path.insert(0, 'src')

from market.orders.order import Order, OrderState
from market.orders.order_book import OrderBook
from market.engine.services.trade_processing_service import TradeProcessingService
from market.orders.order_repository import OrderRepository
from market.orders.order_state_manager import OrderStateManager
# TradeExecutionService not needed - using MockTradeExecutionService instead
from market.state.sim_context import SimulationContext
from agents.agent_manager.services.commitment_services import CommitmentCalculator
from services.logging_service import LoggingService

# Initialize logging for tests
LoggingService.initialize(run_id='test_crossed_market/test')


class MockAgentRepository:
    """Agent repository mock for testing."""
    def __init__(self):
        self.agents = {}

    def get_agent(self, agent_id):
        return self.agents.get(agent_id)

    def sync_agent_orders(self, agent_id, orders=None):
        pass


class MockTradeExecutionService:
    """Mock trade execution service that simply updates order quantities.

    This focuses on testing the crossing detection logic without full trade execution.
    """
    def __init__(self, order_repository):
        self.order_repository = order_repository

    def handle_trade_execution(self, trade):
        """Execute a trade by updating order quantities"""
        # Look up orders from repository (like the real service does)
        buy_order = self.order_repository.get_order(trade.buyer_order_id)
        sell_order = self.order_repository.get_order(trade.seller_order_id)
        # Update remaining quantities on both orders
        buy_order.remaining_quantity -= trade.quantity
        sell_order.remaining_quantity -= trade.quantity
        return True


class MockAgent:
    """Minimal agent for testing"""
    def __init__(self, agent_id, cash=100000, shares=1000):
        self.id = agent_id
        self.cash = cash
        self.positions = {'DEFAULT_STOCK': shares}
        self.committed_cash = {}
        self.committed_positions = {}
        self.borrowed_cash = {}
        self.borrowed_positions = {}


def test_aggressive_limit_crossing():
    """Test that aggressive limits that cross are matched before adding to book"""
    print("=" * 60)
    print("TEST: Aggressive limit crossing fix")
    print("=" * 60)

    # Setup
    order_repository = OrderRepository()
    agent_repository = MockAgentRepository()
    context = SimulationContext(
        num_rounds=10,
        initial_price=50.0,
        fundamental_price=50.0,
        redemption_value=50.0,
        transaction_cost=0.0,
        round_number=1
    )
    order_book = OrderBook(context=context, order_repository=order_repository)

    # Create agents
    agent_a = MockAgent('agent_a', cash=100000, shares=1000)
    agent_b = MockAgent('agent_b', cash=100000, shares=1000)
    agent_repository.agents = {'agent_a': agent_a, 'agent_b': agent_b}

    # Create minimal services
    commitment_calculator = CommitmentCalculator(agent_repository)

    order_state_manager = OrderStateManager(
        order_repository=order_repository,
        agent_repository=agent_repository,
        order_book=order_book,
        logger=None,
        commitment_calculator=commitment_calculator
    )

    # Use mock trade execution service
    trade_execution_service = MockTradeExecutionService(order_repository)

    trade_processing_service = TradeProcessingService(
        agent_manager=None,
        order_state_manager=order_state_manager,
        order_book=order_book,
        order_repository=order_repository,
        agent_repository=agent_repository,
        context=context,
        trade_execution_service=trade_execution_service
    )

    # Step 1: Add a sell order to the book at $51.50
    print("\n1. Adding sell order to book: 100 shares @ $51.50")
    sell_order = Order(
        agent_id='agent_a',
        stock_id='DEFAULT_STOCK',
        order_type='limit',
        side='sell',
        quantity=100,
        price=51.50,
        round_placed=1
    )
    order_repository.create_order(sell_order)
    order_repository.transition_state(sell_order.order_id, OrderState.VALIDATED)
    order_repository.transition_state(sell_order.order_id, OrderState.COMMITTED)
    order_state_manager.transition_non_crossing_limit(sell_order)
    order_book.add_limit_order(sell_order)
    order_state_manager.transition_to_active(sell_order)

    best_ask = order_book.get_best_ask()
    print(f"   Book state: Best Ask = ${best_ask}")
    assert best_ask == 51.50, f"Expected ask $51.50, got ${best_ask}"

    # Step 2: Create an aggressive buy limit at $56.65 (110% of $51.50)
    # This simulates a market order that couldn't fill and was converted
    print("\n2. Creating aggressive buy limit: 100 shares @ $56.65")
    aggressive_buy = Order(
        agent_id='agent_b',
        stock_id='DEFAULT_STOCK',
        order_type='limit',
        side='buy',
        quantity=100,
        price=56.65,  # 110% of best ask
        round_placed=1
    )
    order_repository.create_order(aggressive_buy)
    order_repository.transition_state(aggressive_buy.order_id, OrderState.VALIDATED)
    order_repository.transition_state(aggressive_buy.order_id, OrderState.COMMITTED)

    # Step 3: Process through the fixed process_aggressive_limits
    print("\n3. Processing aggressive limit through fix...")
    processed, trades = trade_processing_service.process_aggressive_limits([aggressive_buy])

    # Step 4: Verify results
    print("\n4. Verifying results:")

    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()

    print(f"   Trades executed: {len(trades)}")
    print(f"   Book state: Best Bid = {best_bid}, Best Ask = {best_ask}")

    # Check for crossed market
    if best_bid is not None and best_ask is not None:
        if best_bid > best_ask:
            print(f"\n   FAIL: Crossed market detected! Bid ${best_bid} > Ask ${best_ask}")
            return False
        else:
            print(f"   PASS: No crossed market (bid {best_bid} <= ask {best_ask})")
    else:
        print(f"   PASS: Book is not crossed (bid={best_bid}, ask={best_ask})")

    # Verify trade occurred
    if len(trades) > 0:
        print(f"   PASS: Trade executed at ${trades[0].price} for {trades[0].quantity} shares")
    else:
        print("   WARNING: No trades executed")

    # Verify sell order was filled
    if sell_order.remaining_quantity == 0:
        print(f"   PASS: Sell order fully filled")
    else:
        print(f"   INFO: Sell order has {sell_order.remaining_quantity} remaining")

    return True


def test_non_crossing_aggressive_limit():
    """Test that aggressive limits that don't cross are added to book normally"""
    print("\n" + "=" * 60)
    print("TEST: Non-crossing aggressive limit")
    print("=" * 60)

    # Setup
    order_repository = OrderRepository()
    agent_repository = MockAgentRepository()
    context = SimulationContext(
        num_rounds=10,
        initial_price=50.0,
        fundamental_price=50.0,
        redemption_value=50.0,
        transaction_cost=0.0,
        round_number=1
    )
    order_book = OrderBook(context=context, order_repository=order_repository)

    agent_a = MockAgent('agent_a', cash=100000, shares=1000)
    agent_repository.agents = {'agent_a': agent_a}

    commitment_calculator = CommitmentCalculator(agent_repository)

    order_state_manager = OrderStateManager(
        order_repository=order_repository,
        agent_repository=agent_repository,
        order_book=order_book,
        logger=None,
        commitment_calculator=commitment_calculator
    )

    # Use mock trade execution service
    trade_execution_service = MockTradeExecutionService(order_repository)

    trade_processing_service = TradeProcessingService(
        agent_manager=None,
        order_state_manager=order_state_manager,
        order_book=order_book,
        order_repository=order_repository,
        agent_repository=agent_repository,
        context=context,
        trade_execution_service=trade_execution_service
    )

    # Step 1: Create aggressive buy with empty book (no crossing possible)
    print("\n1. Creating aggressive buy limit with empty book: 100 shares @ $55.00")
    aggressive_buy = Order(
        agent_id='agent_a',
        stock_id='DEFAULT_STOCK',
        order_type='limit',
        side='buy',
        quantity=100,
        price=55.00,
        round_placed=1
    )
    order_repository.create_order(aggressive_buy)
    order_repository.transition_state(aggressive_buy.order_id, OrderState.VALIDATED)
    order_repository.transition_state(aggressive_buy.order_id, OrderState.COMMITTED)

    # Step 2: Process
    print("\n2. Processing aggressive limit...")
    processed, trades = trade_processing_service.process_aggressive_limits([aggressive_buy])

    # Step 3: Verify
    print("\n3. Verifying results:")
    best_bid = order_book.get_best_bid()
    print(f"   Trades executed: {len(trades)}")
    print(f"   Best Bid: ${best_bid}")

    if best_bid == 55.00:
        print("   PASS: Order added to book correctly")
        return True
    else:
        print(f"   FAIL: Expected bid $55.00, got ${best_bid}")
        return False


def test_partial_fill_aggressive_limit():
    """Test aggressive limit that partially fills then adds remainder to book"""
    print("\n" + "=" * 60)
    print("TEST: Partial fill aggressive limit")
    print("=" * 60)

    # Setup
    order_repository = OrderRepository()
    agent_repository = MockAgentRepository()
    context = SimulationContext(
        num_rounds=10,
        initial_price=50.0,
        fundamental_price=50.0,
        redemption_value=50.0,
        transaction_cost=0.0,
        round_number=1
    )
    order_book = OrderBook(context=context, order_repository=order_repository)

    agent_a = MockAgent('agent_a', cash=100000, shares=1000)
    agent_b = MockAgent('agent_b', cash=100000, shares=1000)
    agent_repository.agents = {'agent_a': agent_a, 'agent_b': agent_b}

    commitment_calculator = CommitmentCalculator(agent_repository)

    order_state_manager = OrderStateManager(
        order_repository=order_repository,
        agent_repository=agent_repository,
        order_book=order_book,
        logger=None,
        commitment_calculator=commitment_calculator
    )

    # Use mock trade execution service
    trade_execution_service = MockTradeExecutionService(order_repository)

    trade_processing_service = TradeProcessingService(
        agent_manager=None,
        order_state_manager=order_state_manager,
        order_book=order_book,
        order_repository=order_repository,
        agent_repository=agent_repository,
        context=context,
        trade_execution_service=trade_execution_service
    )

    # Step 1: Add small sell order to book
    print("\n1. Adding sell order to book: 50 shares @ $51.50")
    sell_order = Order(
        agent_id='agent_a',
        stock_id='DEFAULT_STOCK',
        order_type='limit',
        side='sell',
        quantity=50,
        price=51.50,
        round_placed=1
    )
    order_repository.create_order(sell_order)
    order_repository.transition_state(sell_order.order_id, OrderState.VALIDATED)
    order_repository.transition_state(sell_order.order_id, OrderState.COMMITTED)
    order_state_manager.transition_non_crossing_limit(sell_order)
    order_book.add_limit_order(sell_order)
    order_state_manager.transition_to_active(sell_order)

    # Step 2: Create larger aggressive buy
    print("\n2. Creating aggressive buy: 100 shares @ $56.65")
    aggressive_buy = Order(
        agent_id='agent_b',
        stock_id='DEFAULT_STOCK',
        order_type='limit',
        side='buy',
        quantity=100,
        price=56.65,
        round_placed=1
    )
    order_repository.create_order(aggressive_buy)
    order_repository.transition_state(aggressive_buy.order_id, OrderState.VALIDATED)
    order_repository.transition_state(aggressive_buy.order_id, OrderState.COMMITTED)

    # Step 3: Process
    print("\n3. Processing aggressive limit...")
    processed, trades = trade_processing_service.process_aggressive_limits([aggressive_buy])

    # Step 4: Verify
    print("\n4. Verifying results:")
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()

    print(f"   Trades executed: {len(trades)}")
    if trades:
        print(f"   Trade: {trades[0].quantity} shares @ ${trades[0].price}")
    print(f"   Best Bid: ${best_bid}, Best Ask: {best_ask}")
    print(f"   Buy order remaining: {aggressive_buy.remaining_quantity}")

    # Verify no crossed market
    if best_bid is not None and best_ask is not None and best_bid > best_ask:
        print(f"   FAIL: Crossed market!")
        return False

    # Verify partial fill (50 shares traded)
    if len(trades) == 1 and trades[0].quantity == 50:
        print("   PASS: Correct partial fill")
    else:
        print(f"   FAIL: Expected 1 trade of 50 shares")
        return False

    # Verify remainder added to book (50 shares at $56.65)
    if best_bid == 56.65 and aggressive_buy.remaining_quantity == 50:
        print("   PASS: Remainder correctly added to book")
        return True
    else:
        print(f"   FAIL: Remainder not correctly added")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CROSSED MARKET FIX TESTS (Issue #88)")
    print("=" * 60)

    results = []

    try:
        results.append(("Crossing aggressive limit", test_aggressive_limit_crossing()))
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Crossing aggressive limit", False))

    try:
        results.append(("Non-crossing aggressive limit", test_non_crossing_aggressive_limit()))
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Non-crossing aggressive limit", False))

    try:
        results.append(("Partial fill aggressive limit", test_partial_fill_aggressive_limit()))
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Partial fill aggressive limit", False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)
