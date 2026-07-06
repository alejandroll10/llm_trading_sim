"""Regression test for issue #88/#101: order-book heap invariant after removal.

`OrderBookModifiers._remove_order_from_book` and `remove_agent_orders` rebuild the
buy/sell heaps with a filtered list comprehension. A comprehension does NOT
preserve the min-heap invariant when a non-leaf element is removed, so `heap[0]`
could stop being the true best price. That made `get_best_ask()/get_best_bid()`
(and every crossing check that reads them) return a phantom top-of-book, which
produced a crossed market (best bid > best ask surviving to end-of-round).

The fix re-heapifies after each filtered rebuild. These tests assert that the
reported best bid/ask always equals the true min-ask / max-bid of the remaining
orders after removals.
"""
import sys
import types
import logging
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Stub logging service so OrderBook construction/logging doesn't need full init.
_stub = types.ModuleType("services.logging_service")
class _LS:
    @staticmethod
    def get_logger(name): return logging.getLogger(name)
    @staticmethod
    def log_order_state(*a, **k): pass
    @staticmethod
    def log_market_state(*a, **k): pass
_stub.LoggingService = _LS
sys.modules.setdefault("services.logging_service", _stub)
sys.modules["services.logging_service"].LoggingService = _LS

from market.orders.order import Order, OrderState
from market.orders.order_book import OrderBook
from market.orders.order_entry import OrderEntry


class _StubRepo:
    """Order repository stub: transition_state is a no-op for these tests."""
    def transition_state(self, order_id, state, notes=None):
        pass


class _Ctx:
    round_number = 1
    current_price = 25.0
    def __init__(self):
        self.public_info = {"trade_history": [], "order_book_state": {}}


def _make_book():
    book = OrderBook(context=_Ctx(), order_repository=_StubRepo())
    book.logger = logging.getLogger("test")
    # The public-view/quote-history update needs a full sim context and is
    # irrelevant to the heap invariant under test; no-op it.
    book._update_public_view = lambda *a, **k: None
    return book


def _add_sell(book, agent_id, price, qty=100):
    o = Order(agent_id=agent_id, stock_id="DEFAULT_STOCK", order_type="limit",
              side="sell", quantity=qty, price=price, round_placed=1)
    o.state = OrderState.ACTIVE
    book.sell_orders.append(OrderEntry.create_sell(o))
    import heapq
    heapq.heapify(book.sell_orders)  # valid heap to start
    return o


def _add_buy(book, agent_id, price, qty=100):
    o = Order(agent_id=agent_id, stock_id="DEFAULT_STOCK", order_type="limit",
              side="buy", quantity=qty, price=price, round_placed=1)
    o.state = OrderState.ACTIVE
    book.buy_orders.append(OrderEntry.create_buy(o))
    import heapq
    heapq.heapify(book.buy_orders)
    return o


def _true_best_ask(book):
    return min(e.display_price for e in book.sell_orders) if book.sell_orders else None


def _true_best_bid(book):
    return max(e.display_price for e in book.buy_orders) if book.buy_orders else None


def test_remove_order_preserves_best_ask():
    """Removing the resting best ask must expose the true next-best ask."""
    book = _make_book()
    # Prices chosen so a naive comprehension-removal of the root leaves heap[0] wrong.
    orders = [_add_sell(book, "a", p) for p in [20.8, 21.0, 20.9, 30.0, 21.1]]
    assert book.get_best_ask() == 20.8

    # Remove the current best (root of the heap).
    book.remove_order(orders[0])  # the 20.8 sell

    assert book.get_best_ask() == _true_best_ask(book) == 20.9, (
        f"best ask {book.get_best_ask()} != true min {_true_best_ask(book)} "
        "-> heap invariant broken after removal")


def test_remove_middle_orders_keep_best_ask_correct():
    """Repeated removals never leave a phantom best ask (the #88/#101 crash)."""
    book = _make_book()
    prices = [30.0, 21.1, 21.3, 21.4, 25.0, 22.2]
    orders = {p: _add_sell(book, "mm", p) for p in prices}
    # Remove in an order that repeatedly deletes near-root elements.
    for p in [21.1, 21.3, 30.0, 21.4]:
        book.remove_order(orders[p])
        assert book.get_best_ask() == _true_best_ask(book), (
            f"after removing {p}: best ask {book.get_best_ask()} != "
            f"true min {_true_best_ask(book)}")


def test_remove_buy_order_preserves_best_bid():
    """Buy-side branch of _remove_order_from_book must also re-heapify.

    Buy entries store negated prices, so the 'best bid' is the max true price =
    min heap key. Removing the current best bid must expose the true next-best.
    """
    book = _make_book()
    # true best bid is the max price (21.9); pick prices so removing it from a
    # naive filtered heap would leave heap[0] (min negated key) non-extreme.
    orders = {p: _add_buy(book, "b", p) for p in [21.9, 21.0, 21.5, 10.0, 21.4]}
    assert book.get_best_bid() == 21.9

    book.remove_order(orders[21.9])  # remove the best bid (a heap root on negated keys)

    assert book.get_best_bid() == _true_best_bid(book) == 21.5, (
        f"best bid {book.get_best_bid()} != true max {_true_best_bid(book)} "
        "-> buy-side heap invariant broken after removal")


def test_remove_agent_orders_preserves_both_sides():
    """remove_agent_orders (used by replace_decision) must re-heapify both sides."""
    book = _make_book()
    # Agent mm places bracketing quotes plus others resting.
    _add_sell(book, "mm", 30.0)
    _add_sell(book, "other", 21.1)   # true best ask, different agent -> must survive
    _add_sell(book, "mm", 21.5)
    _add_buy(book, "mm", 19.0)
    _add_buy(book, "other", 21.0)    # true best bid, must survive
    _add_buy(book, "mm", 18.0)

    book.remove_agent_orders("mm")   # cancel all of mm's orders (a Replace)

    assert book.get_best_ask() == _true_best_ask(book) == 21.1
    assert book.get_best_bid() == _true_best_bid(book) == 21.0
    # And crucially: still not crossed after the re-heapify.
    assert book.get_best_bid() <= book.get_best_ask()


def test_crossing_check_correct_after_cancel():
    """End-to-end of the #101 bug->symptom boundary.

    Reproduces the crash shape: resting ask 21.1 buried under a 30.0 order that a
    market maker then cancels (replace). The ENGINE decides whether an incoming
    buy @ 21.4 crosses by calling LimitOrderBookService._would_cross_market, which
    reads peek_best_sell() == sell_orders[0]. With the heap corrupted (no fix),
    sell_orders[0] is the phantom 30.0 -> the buy is mislabeled NON-crossing and
    would rest on the book on top of the 21.1 ask (the crossed market). With the
    fix, peek_best_sell() is the true 21.1 -> the buy is correctly flagged crossing
    and the engine routes it to matching instead of resting it.
    """
    from market.orders.handlers.services.order_book_service import LimitOrderBookService

    book = _make_book()
    # A valid min-heap array where removing the ROOT (the market maker's best-ask
    # quote at 20.0) leaves, without re-heapify, a phantom root of 30.0 while the
    # TRUE next-best ask is 21.1. This mirrors the crash: a cancel/replace removed
    # the best quote and left a far-too-high phantom best ask.
    prices = [20.0, 30.0, 21.1, 40.0, 50.0, 21.2]  # valid min-heap on price
    sells = []
    for p in prices:
        o = Order(agent_id="mm" if p == 20.0 else "s", stock_id="DEFAULT_STOCK",
                  order_type="limit", side="sell", quantity=100, price=p, round_placed=1)
        o.state = OrderState.ACTIVE
        sells.append(o)
    book.sell_orders = [OrderEntry.create_sell(o) for o in sells]  # exact heap-array order
    assert book.get_best_ask() == 20.0

    book.remove_order(sells[0])      # market maker cancels its best quote (a replace)

    # get_best_ask and peek_best_sell must both reflect the TRUE resting ask 21.1,
    # not the phantom 30.0 that a non-heapified removal would leave at index 0.
    assert book.get_best_ask() == 21.1
    assert book.peek_best_sell().price == 21.1

    # The engine's actual crossing check must classify a buy @ 21.4 as crossing.
    svc = LimitOrderBookService(order_book=book, order_state_manager=None)
    incoming_buy = Order(agent_id="b", stock_id="DEFAULT_STOCK", order_type="limit",
                         side="buy", quantity=100, price=21.4, round_placed=1)
    assert svc._would_cross_market(incoming_buy) is True, (
        "buy @ 21.4 must be detected as crossing resting ask @ 21.1; a phantom "
        "best ask from the un-heapified removal would return False here and let "
        "the buy rest on the book -> crossed market (#88/#101)")
