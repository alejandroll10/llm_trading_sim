"""Component manager for updating and formatting market state components"""
from datetime import datetime
from typing import Optional
from services.logging_service import LoggingService


class ComponentManager:
    """Manages market component updates and state formatting"""

    def __init__(self, context, order_book, dividend_service=None,
                 interest_service=None, borrow_service=None,
                 hide_fundamental_price=False):
        """Initialize component manager

        Args:
            context: Simulation context
            order_book: Order book instance
            dividend_service: Optional dividend service
            interest_service: Optional interest service
            borrow_service: Optional borrow service
            hide_fundamental_price: Whether to hide fundamental price
        """
        self.context = context
        self.order_book = order_book
        self.dividend_service = dividend_service
        self.interest_service = interest_service
        self.borrow_service = borrow_service
        self.hide_fundamental_price = hide_fundamental_price

    def update_components(self, round_number: int):
        """Update all market components in the correct order

        Args:
            round_number: Current round number
        """
        # 1. Update order book and depth
        self.update_market_depth()

        # 2. Update dividend state if exists
        if self.dividend_service:
            self.dividend_service.update(round_number)
        else:
            LoggingService.get_logger('simulation').error("No dividend service found")
            raise RuntimeError("No dividend service found")

        # 3. Update borrow state if exists
        if self.borrow_service:
            self.borrow_service.update(round_number)

    def update_market_depth(self):
        """Update market depth information from order book"""
        if self.order_book is None:
            raise RuntimeError("Order book not registered")

        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        midpoint = self.order_book.get_midpoint()

        LoggingService.log_order_state(
            f"Updating market depth - "
            f"Best bid: {best_bid if best_bid else None}, "
            f"Best ask: {best_ask if best_ask else None}, "
            f"Midpoint: {midpoint if midpoint else None}"
        )

        # Store historical quote
        public_info = self.context.get_public_info()

        # Ensure historical_quotes exists
        if 'historical_quotes' not in public_info['order_book_state']:
            public_info['order_book_state']['historical_quotes'] = []

        # Store historical quote
        public_info['order_book_state']['historical_quotes'].append({
            'round': self.context.round_number,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'midpoint': midpoint,
            'timestamp': datetime.now().isoformat()
        })

        # Update order book state
        public_info['order_book_state'].update({
            'best_bid': best_bid,
            'best_ask': best_ask,
            'midpoint': midpoint,
            'aggregated_levels': self.order_book.get_aggregated_levels()
        })

        # Update context with new public info
        self.context.public_info.update(public_info)

    def get_current_market_state(self, round_number: int, last_volume: float) -> dict:
        """Get current market state snapshot

        Args:
            round_number: Current round number
            last_volume: Volume from last round

        Returns:
            Dictionary containing current market state
        """
        public_info = self.context.get_public_info()

        # Get dividend state safely
        dividend_state = self.format_dividend_state()
        last_paid_dividend = (
            dividend_state['last_paid_dividend']
            if dividend_state and 'last_paid_dividend' in dividend_state
            else 0.0
        )

        return {
            'price': self.context.current_price,
            'fundamental_price': self.context.fundamental_price,
            'market_depth': self.order_book.get_aggregated_levels(),
            'best_bid': public_info['order_book_state']['best_bid'],
            'best_ask': public_info['order_book_state']['best_ask'],
            'midpoint': public_info['order_book_state']['midpoint'],
            'last_trade_price': public_info['last_trade'],
            'volume': last_volume,
            'round_number': round_number + 1,
            'num_rounds': self.context._num_rounds,
            'periods_remaining': (self.context._num_rounds - round_number
                                 if not self.context.infinite_rounds else "Infinite"),
            'dividend_state': dividend_state,
            'last_paid_dividend': last_paid_dividend,
            'infinite_rounds': self.context.infinite_rounds
        }

    def format_observable_state(self) -> dict:
        """Format complete observable market state

        Returns:
            Dictionary with all formatted state components
        """
        public_info = self.context.get_public_info()

        return {
            'market': self.format_market_state(public_info),
            'fundamental': self.format_fundamental_state(),
            'dividend': self.format_dividend_state(),
            'interest': self.format_interest_state(self.interest_service),
            'borrow': self.format_borrow_state(),
            'metadata': self.format_metadata(public_info)
        }

    def format_market_state(self, public_info: dict) -> dict:
        """Format market component state

        Args:
            public_info: Public information from context

        Returns:
            Formatted market state
        """
        return {
            'price': self.context.current_price,
            'order_book': self.order_book.get_aggregated_levels(),
            'best_bid': public_info['order_book_state']['best_bid'],
            'best_ask': public_info['order_book_state']['best_ask'],
            'midpoint': public_info['order_book_state']['midpoint'],
            'last_trade_price': public_info['last_trade']['price'],
            'volume': public_info['last_trade']['volume'],
            'trade_history': public_info['trade_history'][-5:]
        }

    def format_fundamental_state(self) -> dict:
        """Format fundamental component state

        Returns:
            Formatted fundamental state
        """
        return {
            'price': (self.context.fundamental_price
                     if not self.hide_fundamental_price else None),
            'periods_remaining': (self.context._num_rounds - self.context.round_number
                                 if not self.context.infinite_rounds else "Infinite"),
            'redemption_value': (self.context.redemption_value
                                if not self.context.infinite_rounds else None)
        }

    def format_dividend_state(self) -> Optional[dict]:
        """Format dividend component state

        Returns:
            Formatted dividend state, or None if no dividend service
        """
        if not self.dividend_service:
            return None
        return self.dividend_service.get_state()

    def format_interest_state(self, interest_service) -> dict:
        """Format interest state information

        Args:
            interest_service: Interest service instance

        Returns:
            Formatted interest state
        """
        if not interest_service:
            return {}

        return {
            'rate': interest_service.get_current_rate(),
            'compound_frequency': interest_service.interest_model['compound_frequency'],
            'last_payment': (interest_service.interest_history[-1]
                            if interest_service.interest_history else None),
            'next_payment_round': interest_service.next_payment_round,
            'destination': interest_service.interest_model['destination']
        }

    def format_borrow_state(self) -> dict:
        """Format borrow fee state information

        Returns:
            Formatted borrow state
        """
        if not self.borrow_service:
            return {}

        return {
            'rate': self.borrow_service.get_current_rate(),
            'payment_frequency': self.borrow_service.borrow_model.get('payment_frequency', 1),
            'last_payment': (self.borrow_service.borrow_history[-1]
                            if self.borrow_service.borrow_history else None),
            'next_payment_round': self.borrow_service.next_payment_round
        }

    def format_metadata(self, public_info: dict) -> dict:
        """Format metadata component

        Args:
            public_info: Public information from context

        Returns:
            Formatted metadata
        """
        return {
            'round': self.context.round_number,
            'last_trade': public_info['last_trade']
        }

    @property
    def dividend_model(self):
        """Access dividend model through dividend service

        Returns:
            Dividend model or None
        """
        if self.dividend_service is None:
            return None
        # Use get_state() to access model info
        return self.dividend_service.get_state()['model']
