from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails


class MarginBuyAgent(BaseAgent):
    """Deterministic agent that always buys using leverage.

    Unlike BuyTrader which only uses available cash, this agent uses its full
    buying power (cash + borrowing capacity) to buy shares. This ensures leverage
    is actually used when testing leverage functionality.
    """

    def __init__(self,
                 buy_proportion: float = 0.5,  # Use 50% of total buying power
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buy_proportion = buy_proportion

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        price = market_state['price']
        prices = {"DEFAULT_STOCK": price}

        # Calculate total buying power: available cash + borrowing capacity
        available_cash = self.available_cash
        borrowing_power = self.get_available_borrowing_power(prices)
        total_buying_power = available_cash + borrowing_power

        # Calculate quantity to buy using total buying power
        max_shares = int(total_buying_power / price)
        total_quantity = int(max_shares * self.buy_proportion)

        if total_quantity == 0:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning=f"No buying power: cash=${available_cash:.2f}, borrow=${borrowing_power:.2f}"
            )

        # Single aggressive buy order at 1% above market
        orders = [
            OrderDetails(
                decision="Buy",
                quantity=total_quantity,
                order_type=OrderType.LIMIT,
                price_limit=price * 1.01
            )
        ]

        return TradeDecision(
            orders=orders,
            replace_decision="Replace",
            reasoning=f"Buying {total_quantity} shares using leverage. Cash: ${available_cash:.2f}, Borrowing: ${borrowing_power:.2f}",
            valuation=100.0,
            valuation_reasoning="Margin buyer always buys",
            price_prediction_reasoning="Target 10% above current price",
            price_prediction_t=price,
            price_prediction_t1=price * 1.1,
            price_prediction_t2=price * 1.1,
        )
