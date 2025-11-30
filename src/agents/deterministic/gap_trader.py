from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class ProportionalGapTrader(BaseAgent):
    """Trades proportionally to the gap between price and fundamental value"""
    
    def __init__(self, 
                 threshold: float = 0.05,  # Minimum 5% gap to trade
                 max_proportion: float = 0.5,  # Maximum 50% of holdings/cash to trade
                 scaling_factor: float = 2.0,  # How aggressively to scale with gap size
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.max_proportion = max_proportion
        self.scaling_factor = scaling_factor

    def calculate_trade_proportion(self, gap_percentage: float) -> float:
        """Calculate what proportion of holdings/cash to trade based on gap size"""
        if abs(gap_percentage) < self.threshold:
            return 0.0
        
        # Scale linearly with gap size, capped at max_proportion
        proportion = min(
            self.max_proportion,
            abs(gap_percentage) * self.scaling_factor
        )
        return proportion

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        price = market_state['price']
        fundamental = market_state.get('fundamental_price', price)

        # Calculate gap as percentage
        gap_percentage = (fundamental - price) / price

        # Calculate proportion to trade
        proportion = self.calculate_trade_proportion(gap_percentage)

        if proportion == 0:
            decision = TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning="Gap between price and fundamental too small",
                valuation=fundamental,
                valuation_reasoning="Fundamental estimate provided",
                price_prediction_reasoning="No trade executed",
                price_prediction_t=price,
                price_prediction_t1=price,
                price_prediction_t2=price,
            )
            self.broadcast_message(round_number, {
                'valuation': decision.valuation,
                'price_prediction_t1': decision.price_prediction_t1,
                'reasoning': decision.reasoning,
            })
            return decision

        # Decide whether to buy or sell
        if gap_percentage > 0:  # Undervalued -> Buy
            # Use total buying power (cash + borrowing power) if leverage enabled
            available_cash = self.available_cash
            if self.leverage_ratio > 1.0:
                prices = {"DEFAULT_STOCK": price}  # For margin calculations
                borrowing_power = self.get_available_borrowing_power(prices)
                total_buying_power = available_cash + borrowing_power
            else:
                total_buying_power = available_cash
            max_shares = int(total_buying_power / price)
            quantity = int(max_shares * proportion)

            if quantity == 0:
                decision = TradeDecision(
                    orders=[],
                    replace_decision="Add",
                    reasoning="Insufficient cash for minimum trade",
                    valuation=fundamental,
                    valuation_reasoning="Fundamental estimate provided",
                    price_prediction_reasoning="Cannot participate",
                    price_prediction_t=price,
                    price_prediction_t1=price,
                    price_prediction_t2=price,
                )
                self.broadcast_message(round_number, {
                    'valuation': decision.valuation,
                    'price_prediction_t1': decision.price_prediction_t1,
                    'reasoning': decision.reasoning,
                })
                return decision

            order = OrderDetails(
                decision="Buy",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price_limit=price * 1.01,
            )
            target_price = price * (1 + min(abs(gap_percentage), 0.05))
            decision = TradeDecision(
                orders=[order],
                replace_decision="Replace",
                reasoning=f"Price ${price:.2f} below fundamental ${fundamental:.2f} by {gap_percentage:.1%}",
                valuation=fundamental,
                valuation_reasoning="Fundamental estimate provided",
                price_prediction_reasoning="Expect move toward fundamental",
                price_prediction_t=price,
                price_prediction_t1=target_price,
                price_prediction_t2=target_price,
            )
            self.broadcast_message(round_number, {
                'valuation': decision.valuation,
                'price_prediction_t1': decision.price_prediction_t1,
                'reasoning': decision.reasoning,
            })
            return decision

        # Overvalued -> Sell
        available_shares = self.available_shares
        quantity = int(available_shares * proportion)

        if quantity == 0:
            decision = TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning="Insufficient shares for minimum trade",
                valuation=fundamental,
                valuation_reasoning="Fundamental estimate provided",
                price_prediction_reasoning="Cannot participate",
                price_prediction_t=price,
                price_prediction_t1=price,
                price_prediction_t2=price,
            )
            self.broadcast_message(round_number, {
                'valuation': decision.valuation,
                'price_prediction_t1': decision.price_prediction_t1,
                'reasoning': decision.reasoning,
            })
            return decision

        order = OrderDetails(
            decision="Sell",
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price_limit=price * 0.99,
        )
        target_price = price * (1 - min(abs(gap_percentage), 0.05))
        decision = TradeDecision(
            orders=[order],
            replace_decision="Replace",
            reasoning=f"Price ${price:.2f} above fundamental ${fundamental:.2f} by {abs(gap_percentage):.1%}",
            valuation=fundamental,
            valuation_reasoning="Fundamental estimate provided",
            price_prediction_reasoning="Expect move toward fundamental",
            price_prediction_t=price,
            price_prediction_t1=target_price,
            price_prediction_t2=target_price,
        )
        self.broadcast_message(round_number, {
            'valuation': decision.valuation,
            'price_prediction_t1': decision.price_prediction_t1,
            'reasoning': decision.reasoning,
        })
        return decision
