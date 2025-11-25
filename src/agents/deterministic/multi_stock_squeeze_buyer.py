"""Multi-stock squeeze buyer agent for testing margin calls across stocks"""

from agents.base_agent import BaseAgent
from typing import List


class MultiStockSqueezeBuyer(BaseAgent):
    """Buys aggressively starting at specific round to trigger price spike and margin calls.

    Like the single-stock SqueezeBuyerAgent, this activates at a specific round and
    aggressively buys to push prices up and trigger margin violations on short sellers.
    """

    def __init__(self, activation_round: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_round = activation_round

    def make_decision(self, market_state: dict, history: list, round_number: int):
        from agents.agents_api import TradeDecision, OrderDetails, OrderType

        orders = []

        # Only activate starting at the specified round
        if round_number < self.activation_round:
            return TradeDecision(
                valuation_reasoning="Waiting for activation round",
                valuation=0.0,
                price_target_reasoning="N/A",
                price_target=0.0,
                orders=[],
                reasoning=f"Waiting until round {self.activation_round} to activate squeeze",
                replace_decision="Replace"
            )

        if market_state.get('is_multi_stock'):
            stocks_data = market_state['stocks']
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

        return TradeDecision(
            valuation_reasoning="Aggressive buying to trigger short squeeze",
            valuation=0.0,
            price_target_reasoning="N/A",
            price_target=0.0,
            orders=orders,
            reasoning=f"SQUEEZE ACTIVATED! Round {round_number}: Placed {len(orders)} aggressive buy orders",
            replace_decision="Replace"
        )
