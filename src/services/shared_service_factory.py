from typing import Optional, Dict, Union
from agents.agent_manager.services.commitment_services import CommitmentCalculator
from agents.agent_manager.services.position_services import PositionCalculator
from services.messaging_service import MessagingService

class SharedServiceFactory:
    """Factory for services shared across multiple components"""

    _commitment_calculator: Optional[CommitmentCalculator] = None
    _position_calculator: Optional[PositionCalculator] = None
    _order_book = None
    _order_books: Optional[Dict] = None  # For multi-stock support
    _is_multi_stock: bool = False

    @classmethod
    def initialize(cls, order_book=None, order_books: Optional[Dict] = None) -> None:
        """Initialize the factory with required dependencies

        Args:
            order_book: Single order book for single-stock mode
            order_books: Dict of {stock_id: OrderBook} for multi-stock mode
        """
        if order_books is not None:
            # Multi-stock mode
            cls._order_books = order_books
            cls._order_book = list(order_books.values())[0]  # Backwards compatibility
            cls._is_multi_stock = True
        else:
            # Single-stock mode
            cls._order_book = order_book
            cls._order_books = None
            cls._is_multi_stock = False

    @classmethod
    def get_commitment_calculator(cls) -> CommitmentCalculator:
        """Get or create CommitmentCalculator singleton"""
        if cls._commitment_calculator is None:
            if cls._order_book is None:
                raise RuntimeError("SharedServiceFactory not initialized with order_book")
            # Pass both single and multi-stock order books to calculator
            cls._commitment_calculator = CommitmentCalculator(
                order_book=cls._order_book,
                order_books=cls._order_books
            )
        return cls._commitment_calculator
    
    @classmethod
    def get_position_calculator(cls) -> PositionCalculator:
        """Get or create PositionCalculator singleton"""
        if cls._position_calculator is None:
            cls._position_calculator = PositionCalculator()
        return cls._position_calculator
    
    @classmethod
    def get_order_book_for_stock(cls, stock_id: str):
        """Get the order book for a specific stock (multi-stock support)

        Args:
            stock_id: The stock identifier

        Returns:
            The order book for the specified stock, or the single order book if not multi-stock
        """
        if cls._order_books is not None and stock_id in cls._order_books:
            return cls._order_books[stock_id]
        # Fallback to single order book
        return cls._order_book

    @classmethod
    def reset(cls) -> None:
        """Reset all shared services"""
        cls._commitment_calculator = None
        cls._position_calculator = None
        cls._order_book = None
        cls._order_books = None
        cls._is_multi_stock = False
        MessagingService.reset()
