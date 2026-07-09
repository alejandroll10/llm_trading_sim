"""Tests for the crossed market fix (Issue #88).

Aggressive limit orders (converted from unfilled market orders) must be
matched against the book before being added, preventing crossed markets.

Bug scenario:
1. Book has Ask at $51.50
2. Market buy can't fill due to insufficient cash commitment
3. Becomes aggressive limit at $51.50 * 1.10 = $56.65
4. OLD BUG: Added directly to book -> crossed market (bid $56.65 > ask $51.50)
5. FIX: Should match against the ask first
"""
import sys
from pathlib import Path

# Support standalone execution (`python tests/test_crossed_market.py`): under
# pytest, conftest.py has already installed the stub and this is a no-op.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _logging_stub
_logging_stub.install()

from market.orders.order import Order, OrderState
from market.orders.order_book import OrderBook
from market.engine.services.trade_processing_service import TradeProcessingService
from market.orders.order_repository import OrderRepository
from market.orders.order_state_manager import OrderStateManager
from market.state.sim_context import SimulationContext
from agents.agent_manager.services.commitment_services import CommitmentCalculator


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

    This focuses on testing the crossing detection logic without full trade
    execution.
    """
    def __init__(self, order_repository):
        self.order_repository = order_repository

    def handle_trade_execution(self, trade):
        """Execute a trade by updating order quantities"""
        buy_order = self.order_repository.get_order(trade.buyer_order_id)
        sell_order = self.order_repository.get_order(trade.seller_order_id)
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


def make_trading_env(agent_ids):
    """Build the minimal order-processing stack shared by all tests here."""
    order_repository = OrderRepository()
    agent_repository = MockAgentRepository()
    agent_repository.agents = {
        agent_id: MockAgent(agent_id) for agent_id in agent_ids
    }
    context = SimulationContext(
        num_rounds=10,
        initial_price=50.0,
        fundamental_price=50.0,
        redemption_value=50.0,
        transaction_cost=0.0,
        round_number=1
    )
    order_book = OrderBook(context=context, order_repository=order_repository)
    order_state_manager = OrderStateManager(
        order_repository=order_repository,
        agent_repository=agent_repository,
        order_book=order_book,
        logger=None,
        commitment_calculator=CommitmentCalculator(agent_repository)
    )
    trade_processing_service = TradeProcessingService(
        agent_manager=None,
        order_state_manager=order_state_manager,
        order_book=order_book,
        order_repository=order_repository,
        agent_repository=agent_repository,
        context=context,
        trade_execution_service=MockTradeExecutionService(order_repository)
    )
    return order_repository, order_book, order_state_manager, trade_processing_service


def add_resting_sell(order_repository, order_book, order_state_manager,
                     agent_id, quantity, price):
    """Place a committed sell limit into the book."""
    sell_order = Order(
        agent_id=agent_id,
        stock_id='DEFAULT_STOCK',
        order_type='limit',
        side='sell',
        quantity=quantity,
        price=price,
        round_placed=1
    )
    order_repository.create_order(sell_order)
    order_repository.transition_state(sell_order.order_id, OrderState.VALIDATED)
    order_repository.transition_state(sell_order.order_id, OrderState.COMMITTED)
    order_state_manager.transition_non_crossing_limit(sell_order)
    order_book.add_limit_order(sell_order)
    order_state_manager.transition_to_active(sell_order)
    return sell_order


def make_committed_buy(order_repository, agent_id, quantity, price):
    """Create a committed aggressive buy limit (not yet in the book)."""
    buy_order = Order(
        agent_id=agent_id,
        stock_id='DEFAULT_STOCK',
        order_type='limit',
        side='buy',
        quantity=quantity,
        price=price,
        round_placed=1
    )
    order_repository.create_order(buy_order)
    order_repository.transition_state(buy_order.order_id, OrderState.VALIDATED)
    order_repository.transition_state(buy_order.order_id, OrderState.COMMITTED)
    return buy_order


def assert_not_crossed(order_book):
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    if best_bid is not None and best_ask is not None:
        assert best_bid <= best_ask, (
            f"Crossed market: bid ${best_bid} > ask ${best_ask}")


def test_aggressive_limit_crossing():
    """Aggressive limits that cross are matched before adding to book"""
    order_repository, order_book, order_state_manager, trade_processing_service = \
        make_trading_env(['agent_a', 'agent_b'])

    # Sell order in the book at $51.50
    sell_order = add_resting_sell(
        order_repository, order_book, order_state_manager,
        'agent_a', quantity=100, price=51.50)
    assert order_book.get_best_ask() == 51.50

    # Aggressive buy limit at $56.65 (simulates a converted market order)
    aggressive_buy = make_committed_buy(
        order_repository, 'agent_b', quantity=100, price=56.65)

    processed, trades = trade_processing_service.process_aggressive_limits(
        [aggressive_buy])

    assert_not_crossed(order_book)
    assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"
    assert trades[0].quantity == 100
    assert sell_order.remaining_quantity == 0, "Sell order should be fully filled"


def test_non_crossing_aggressive_limit():
    """Aggressive limits that don't cross are added to book normally"""
    order_repository, order_book, order_state_manager, trade_processing_service = \
        make_trading_env(['agent_a'])

    # Aggressive buy with an empty book: no crossing possible
    aggressive_buy = make_committed_buy(
        order_repository, 'agent_a', quantity=100, price=55.00)

    processed, trades = trade_processing_service.process_aggressive_limits(
        [aggressive_buy])

    assert len(trades) == 0
    assert order_book.get_best_bid() == 55.00, "Order should rest in the book"


def test_partial_fill_aggressive_limit():
    """Aggressive limit partially fills then adds remainder to book"""
    order_repository, order_book, order_state_manager, trade_processing_service = \
        make_trading_env(['agent_a', 'agent_b'])

    # Small sell order in the book
    add_resting_sell(
        order_repository, order_book, order_state_manager,
        'agent_a', quantity=50, price=51.50)

    # Larger aggressive buy
    aggressive_buy = make_committed_buy(
        order_repository, 'agent_b', quantity=100, price=56.65)

    processed, trades = trade_processing_service.process_aggressive_limits(
        [aggressive_buy])

    assert_not_crossed(order_book)
    assert len(trades) == 1 and trades[0].quantity == 50, \
        "Expected a single partial fill of 50 shares"
    assert order_book.get_best_bid() == 56.65, "Remainder should rest in the book"
    assert aggressive_buy.remaining_quantity == 50


if __name__ == '__main__':
    test_aggressive_limit_crossing()
    test_non_crossing_aggressive_limit()
    test_partial_fill_aggressive_limit()
    print("ALL TESTS PASSED!")
