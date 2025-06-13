from .base_order_book import OrderBook
from .order_book_queries import OrderBookQueries
from .order_book_modifiers import OrderBookModifiers
from .order_book_logging import OrderBookLogging

class CompleteOrderBook(OrderBook, OrderBookQueries, OrderBookModifiers, OrderBookLogging):
    pass
        
# Make this the default export
OrderBook = CompleteOrderBook