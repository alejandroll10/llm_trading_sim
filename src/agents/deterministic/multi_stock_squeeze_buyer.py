"""Multi-stock squeeze buyer agent for testing margin calls across stocks"""

from agents.agents_api import OrderDetails, OrderType, TradeDecision
from agents.deterministic.multi_stock_base import MultiStockAgent


class MultiStockSqueezeBuyer(MultiStockAgent):
    """Buys aggressively starting at specific round to trigger price spike and margin calls.

    Like the single-stock SqueezeBuyerAgent, this activates at a specific round and
    aggressively buys to push prices up and trigger margin violations on short sellers.
    """

    VALUATION_REASONING = "Aggressive buying to trigger short squeeze"

    def __init__(self, activation_round: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_round = activation_round

    def make_decision(self, market_state: dict, history: list, round_number: int) -> TradeDecision:
        # Only activate starting at the specified round
        if round_number < self.activation_round:
            return self.neutral_decision(
                orders=[],
                reasoning=f"Waiting until round {self.activation_round} to activate squeeze",
                valuation_reasoning="Waiting for activation round",
            )
        return super().make_decision(market_state, history, round_number)

    def decide_orders(self, stocks_data: dict, round_number: int):
        orders = []
        num_stocks = len(stocks_data)

        for stock_id, stock_state in stocks_data.items():
            price = stock_state['price']

            # Calculate max shares we can buy - use ALL available cash aggressively
            # Split cash evenly across stocks
            available_cash = self.cash / num_stocks
            max_shares = int(available_cash / (price * 1.10))  # 10% buffer for price limit

            if max_shares <= 0:
                continue

            # Buy aggressively with limit order above current price
            orders.append(OrderDetails(
                stock_id=stock_id,
                decision="Buy",
                quantity=max_shares,
                order_type=OrderType.LIMIT,
                price_limit=price * 1.50  # Willing to pay 50% above current - very aggressive!
            ))

        reasoning = (
            f"SQUEEZE ACTIVATED! Round {round_number}: "
            f"Placed {len(orders)} aggressive buy orders")
        return orders, reasoning
