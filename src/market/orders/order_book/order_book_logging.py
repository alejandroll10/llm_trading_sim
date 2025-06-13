from services.logging_service import LoggingService

class OrderBookLogging:
    """
    No need to redefine __init__ as it will use OrderBook's
    All methods can access self.buy_orders, self.sell_orders, and self.context
    """

    def _log_removals(self, buy_entries, sell_entries):
        """Log details of orders being removed from the book"""
        LoggingService.log_order_state("\n=== Removing Orders ===")
        
        if buy_entries:
            LoggingService.log_order_state("Buy Orders:")
            for entry in buy_entries:
                LoggingService.log_order_state(
                    f"  Agent {entry.agent_id}: {entry.quantity} @ "
                    f"${abs(entry.display_price):.2f} (ID: {entry.order.order_id})"
                )
        
        if sell_entries:
            LoggingService.log_order_state("Sell Orders:")
            for entry in sell_entries:
                LoggingService.log_order_state(
                    f"  Agent {entry.agent_id}: {entry.quantity} @ "
                    f"${entry.display_price:.2f} (ID: {entry.order.order_id})"
                )

    def log_order_book_state(self, message="Current Order Book State"):
        """Log the current state of the order book"""
        LoggingService.log_order_state(f"\n{message}")
        
        # Detailed View
        LoggingService.log_order_state(
            f"=== Detailed Order Book Round {self.context.public_info['round_number'] + 1} ==="
        )
        self._log_detailed_view()
        
        # Aggregated View
        LoggingService.log_order_state(
            f"\n=== Aggregated Order Book Round {self.context.public_info['round_number'] + 1} ==="
        )
        self._log_aggregated_view()

    def _log_detailed_view(self):
        """Log detailed view of order book"""
        if self.buy_orders:
            LoggingService.log_order_state("Buy Orders:")
            for entry in sorted(self.buy_orders, key=lambda x: (-x.display_price, x.timestamp)):
                LoggingService.log_order_state(
                    f"  Agent {entry.agent_id}: {entry.quantity} @ ${entry.display_price:.2f}"
                )
        else:
            LoggingService.log_order_state("No buy orders")
        
        if self.sell_orders:
            LoggingService.log_order_state("Sell Orders:")
            for entry in sorted(self.sell_orders, key=lambda x: (x.display_price, x.timestamp)):
                LoggingService.log_order_state(
                    f"  Agent {entry.agent_id}: {entry.quantity} @ ${entry.display_price:.2f}"
                )
        else:
            LoggingService.log_order_state("No sell orders")

    def _log_aggregated_view(self):
        """Log aggregated view of order book"""
        # Aggregate buy orders
        if self.buy_orders:
            buy_aggregated = {}
            for entry in self.buy_orders:
                price = entry.display_price
                buy_aggregated[price] = buy_aggregated.get(price, 0) + entry.quantity
            
            LoggingService.log_order_state("Buy Side:")
            for price in sorted(buy_aggregated.keys(), reverse=True):
                LoggingService.log_order_state(f"  {buy_aggregated[price]} @ ${price:.2f}")
        else:
            LoggingService.log_order_state("No buy orders")
        
        # Aggregate sell orders
        if self.sell_orders:
            sell_aggregated = {}
            for entry in self.sell_orders:
                price = entry.display_price
                sell_aggregated[price] = sell_aggregated.get(price, 0) + entry.order.quantity
            
            LoggingService.log_order_state("Sell Side:")
            for price in sorted(sell_aggregated.keys()):
                LoggingService.log_order_state(f"  {sell_aggregated[price]} @ ${price:.2f}")
        else:
            LoggingService.log_order_state("No sell orders")

    def print_raw_book(self, logger=None) -> str:
        """Print raw state of order book for debugging
        Args:
            logger: Optional logger object. If None, prints to terminal instead
        Returns:
            str: String representation of the order book
        """
        output = []
        output.append("\n=== RAW ORDER BOOK STATE ===")
        
        output.append("RAW BUY ORDERS:")
        for entry in self.buy_orders:
            output.append(f"  Price: {entry.price}, Time: {entry.timestamp}, "
                       f"Order: [Agent {entry.agent_id}, "
                       f"Qty: {entry.quantity}, "
                       f"Price: {entry.order.price}]")
        
        output.append("RAW SELL ORDERS:")
        for entry in self.sell_orders:
            output.append(f"  Price: {entry.price}, Time: {entry.timestamp}, "
                       f"Order: [Agent {entry.agent_id}, "
                       f"Qty: {entry.quantity}, "
                       f"Price: {entry.order.price}]")

        result = "\n".join(output)
        
        if logger:
            for line in output:
                logger.info(line)
        else:
            print(result)
            
        return result
   