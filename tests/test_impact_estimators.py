"""
Tests for the phase-2 impact / lambda estimators (issue #111).

The load-bearing piece is the coupling between how the prompt renders the order
book (MarketStateFormatter._format_order_book) and how the estimator parses it
back out of data/rendered_prompts.jsonl. A silent drift there does not raise --
it just makes every lambda_book NaN and quietly deletes the primary coherence
benchmark from the results. So the parser is tested against the REAL formatter
output rather than a hand-written fixture.

Also covers the arithmetic that the estimates rest on: the ladder slope, the
walk-the-book cost (including the limit-price cap and depth exhaustion), and
end-to-end recovery of a known lambda from a synthetic flow panel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _logging_stub
_logging_stub.install()

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.LLMs.services.formatting_services import MarketStateFormatter
from market.information.information_types import InformationSignal, InformationType
from analysis.impact_estimators import (
    build_flow_frame, ladder_lambda, lambda_realized_tables, parse_book_block,
    walk_book,
)


ASK_LEVELS = [(28.0, 1000.0), (29.0, 2000.0), (30.5, 400.0)]
BID_LEVELS = [(27.0, 800.0), (26.5, 1500.0)]


def render_prompt(best_bid=27.0, best_ask=28.0, buy_levels=None, sell_levels=None):
    """Build a user prompt whose market-depth block comes from the real formatter."""
    signal = InformationSignal(
        type=InformationType.ORDER_BOOK,
        value={
            'buy_levels': buy_levels if buy_levels is not None else [
                {'price': 27.0, 'quantity': 500}, {'price': 27.0, 'quantity': 300},
                {'price': 26.5, 'quantity': 1500},
            ],
            'sell_levels': sell_levels if sell_levels is not None else [
                {'price': 28.0, 'quantity': 1000}, {'price': 29.0, 'quantity': 2000},
                {'price': 30.5, 'quantity': 400},
            ],
        },
        reliability=1.0,
        metadata={'best_bid': best_bid, 'best_ask': best_ask, 'depth_levels': 5},
    )
    book_display = MarketStateFormatter._format_order_book(signal)
    own_orders = MarketStateFormatter._format_outstanding_orders(
        {'buy': [{'quantity': 999, 'price': 1.0, 'order_type': 'limit'}], 'sell': []}
    )
    return f"\nMarket State:\n- Last Price: $27.50\n\nMarket Depth:\n{book_display}\n{own_orders}\n"


class TestBookParsing:
    def test_round_trips_the_real_formatter_output(self):
        book = parse_book_block(render_prompt())
        assert book is not None
        assert book["best_bid"] == 27.0
        assert book["best_ask"] == 28.0
        # Best first: asks ascending, bids descending; same-price levels merged.
        assert book["asks"] == ASK_LEVELS
        assert book["bids"] == BID_LEVELS

    def test_excludes_the_agents_own_outstanding_orders(self):
        # The own-orders block repeats the "Buy Orders:" header, so a parser that
        # does not stop at it would swallow the agent's own 999-share line.
        book = parse_book_block(render_prompt())
        assert all(qty != 999 for _, qty in book["bids"] + book["asks"])

    def test_empty_book_parses_to_empty_ladders(self):
        prompt = render_prompt(best_bid=None, best_ask=None,
                               buy_levels=[], sell_levels=[])
        book = parse_book_block(prompt)
        assert book["asks"] == [] and book["bids"] == []
        assert np.isnan(book["best_bid"]) and np.isnan(book["best_ask"])

    def test_prompt_without_a_depth_block_returns_none(self):
        assert parse_book_block("no market depth section here") is None
        assert parse_book_block(None) is None

    def test_best_touch_falls_back_to_the_ladder(self):
        # metadata carries no best bid/ask (the formatter then prints no such
        # line), but the ladders still pin the touch.
        book = parse_book_block(render_prompt(best_bid=None, best_ask=None))
        assert book["best_ask"] == 28.0
        assert book["best_bid"] == 27.0


class TestLadderLambda:
    def test_matches_the_depth_weighted_least_squares_slope(self):
        lam, n_levels, depth = ladder_lambda(ASK_LEVELS)
        p = np.array([28.0, 29.0, 30.5])
        q = np.array([1000.0, 2000.0, 400.0])
        mid = np.cumsum(q) - q / 2
        mbar, pbar = np.average(mid, weights=q), np.average(p, weights=q)
        expected = (np.average((mid - mbar) * (p - pbar), weights=q)
                    / np.average((mid - mbar) ** 2, weights=q))
        assert lam == pytest.approx(abs(expected))
        assert (n_levels, depth) == (3, 3400.0)

    def test_is_positive_on_both_sides(self):
        # The bid ladder slopes down in price; lambda is a cost, never negative.
        assert ladder_lambda(BID_LEVELS)[0] > 0

    def test_undefined_below_two_levels(self):
        # NaN, not 0: a flat/absent slope would send q* = edge/lambda to infinity
        # and score every order as infinitely under-sized.
        assert np.isnan(ladder_lambda([(28.0, 500.0)])[0])
        assert np.isnan(ladder_lambda([])[0])

    def test_undefined_when_all_depth_sits_at_one_price(self):
        assert np.isnan(ladder_lambda([(28.0, 500.0), (28.0, 700.0)])[0])


class TestWalkBook:
    def test_average_price_walks_levels_best_first(self):
        avg, filled, exhausted = walk_book(ASK_LEVELS, 1500)
        assert avg == pytest.approx((1000 * 28.0 + 500 * 29.0) / 1500)
        assert filled == 1500 and not exhausted

    def test_flags_exhausted_visible_depth(self):
        avg, filled, exhausted = walk_book(ASK_LEVELS, 5000)
        assert filled == 3400 and exhausted

    def test_limit_price_caps_a_marketable_buy(self):
        # A buy limit at $28.50 lifts the $28 level and stops; it cannot reach $29.
        avg, filled, exhausted = walk_book(ASK_LEVELS, 1500, price_cap=28.5, is_buy=True)
        assert avg == pytest.approx(28.0)
        assert filled == 1000 and exhausted

    def test_limit_price_caps_a_marketable_sell(self):
        avg, filled, exhausted = walk_book(BID_LEVELS, 2000, price_cap=26.9, is_buy=False)
        assert avg == pytest.approx(27.0)
        assert filled == 800 and exhausted

    def test_non_positive_quantity_is_unscored(self):
        assert np.isnan(walk_book(ASK_LEVELS, 0)[0])
        assert np.isnan(walk_book([], 100)[0])


class TestLambdaRealized:
    """A price path built as p_{t+1} = p_t + LAMBDA * net_flow_t must come back
    out of the pooled regression exactly."""

    LAMBDA = 0.002

    def _panels(self):
        rng = np.random.default_rng(0)
        market, orders = [], []
        for cell in ["c1", "c2"]:
            price = 28.0
            for rnd in range(1, 13):
                market.append(dict(cell_id=cell, round=rnd, price=price,
                                   fundamental_price=28.0, best_bid=price - 0.1,
                                   best_ask=price + 0.1))
                if rnd == 12:
                    continue
                flow = float(rng.integers(-4000, 4000))
                # order_data rows are labeled round_number + 1, i.e. the market
                # row whose price move they caused.
                orders.append(dict(cell_id=cell, round=rnd + 1, prompt_family="pf",
                                   model="m", temperature=0.7, seed=1, variant="v",
                                   decision="buy" if flow > 0 else "sell",
                                   quantity=abs(flow), price_limit=price,
                                   order_type="market"))
                price += self.LAMBDA * flow
        return pd.DataFrame(market), pd.DataFrame(orders)

    def test_recovers_the_true_slope(self):
        market, orders = self._panels()
        flow = build_flow_frame(orders, market)
        pooled, per_cell = lambda_realized_tables(flow, "cell_id")
        row = pooled[(pooled["group"] == "pooled")
                     & (pooled["regression"] == "price_change~net_flow")].iloc[0]
        assert row["lambda_hat"] == pytest.approx(self.LAMBDA, rel=1e-9)
        assert per_cell["lambda_hat_realized"].dropna().tolist() == \
            pytest.approx([self.LAMBDA, self.LAMBDA], rel=1e-9)

    def test_drops_the_first_round_which_has_no_prior_price(self):
        market, orders = self._panels()
        flow = build_flow_frame(orders, market)
        # Market rows start at 1, so round-1 flow has no measurable price change.
        assert flow["round"].min() == 2

    def test_missing_order_panel_is_handled(self):
        market, _ = self._panels()
        assert build_flow_frame(None, market) is None
        assert build_flow_frame(pd.DataFrame(), market) is None
        # An all-empty order panel aggregates to metadata columns only.
        assert build_flow_frame(pd.DataFrame(columns=["cell_id"]), market) is None
