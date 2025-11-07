from typing import Optional
from agents.agent_manager.services.commitment_services import CommitmentCalculator
from agents.agent_manager.services.position_services import PositionCalculator
from services.messaging_service import MessagingService

class SharedServiceFactory:
    """Factory for services shared across multiple components"""
    
    _commitment_calculator: Optional[CommitmentCalculator] = None
    _position_calculator: Optional[PositionCalculator] = None
    _order_book = None
    
    @classmethod
    def initialize(cls, order_book) -> None:
        """Initialize the factory with required dependencies"""
        cls._order_book = order_book
    
    @classmethod
    def get_commitment_calculator(cls) -> CommitmentCalculator:
        """Get or create CommitmentCalculator singleton"""
        if cls._commitment_calculator is None:
            if cls._order_book is None:
                raise RuntimeError("SharedServiceFactory not initialized with order_book")
            cls._commitment_calculator = CommitmentCalculator(cls._order_book)
        return cls._commitment_calculator
    
    @classmethod
    def get_position_calculator(cls) -> PositionCalculator:
        """Get or create PositionCalculator singleton"""
        if cls._position_calculator is None:
            cls._position_calculator = PositionCalculator()
        return cls._position_calculator
    
    @classmethod
    def reset(cls) -> None:
        """Reset all shared services"""
        cls._commitment_calculator = None
        cls._position_calculator = None
        cls._order_book = None
        MessagingService.reset()
