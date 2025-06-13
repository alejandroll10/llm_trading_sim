from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

class InformationType(Enum):
    PRICE = "price"
    VOLUME = "volume"
    ORDER_BOOK = "order_book"
    ROUND = "round"
    FUNDAMENTAL = "fundamental"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    INSIDER = "insider"

class SignalCategory(Enum):
    PUBLIC = "public"      # Always visible, perfect info (price, volume, round)
    MARKET = "market"      # Market data that could be depth-limited (order book)
    FUNDAMENTAL = "fundamental"  # Can be noisy (fundamental value, dividends)
    RESTRICTED = "restricted"    # Limited access (insider info, if implemented)

@dataclass
class InfoCapability:
    """Defines how an agent receives/processes a type of information"""
    enabled: bool = True
    noise_level: float = 0.0    
    delay: int = 0              
    depth: Optional[int] = None 
    accuracy: float = 1.0       

@dataclass
class InformationSignal:
    """Base class for all information signals"""
    type: InformationType
    value: Any
    reliability: float  
    duration: int = 1  
    metadata: Dict[str, Any] = field(default_factory=dict)

class InformationProvider(Protocol):
    """Protocol for information providers"""
    def generate_signal(self, round_number: int) -> InformationSignal:
        pass

SIGNAL_CATEGORIES = {
    InformationType.PRICE: SignalCategory.PUBLIC,
    InformationType.VOLUME: SignalCategory.PUBLIC,
    InformationType.ROUND: SignalCategory.PUBLIC,
    InformationType.ORDER_BOOK: SignalCategory.MARKET,
    InformationType.FUNDAMENTAL: SignalCategory.FUNDAMENTAL,
    InformationType.DIVIDEND: SignalCategory.FUNDAMENTAL,
    InformationType.INTEREST: SignalCategory.PUBLIC,
    InformationType.INSIDER: SignalCategory.RESTRICTED
}

DEFAULT_CAPABILITIES = {
    SignalCategory.PUBLIC: InfoCapability(
        enabled=True,
        noise_level=0.0,
        accuracy=1.0
    ),
    SignalCategory.MARKET: InfoCapability(
        enabled=True,
        depth=None,  # Full depth by default
        accuracy=1.0
    ),
    SignalCategory.FUNDAMENTAL: InfoCapability(
        enabled=True,
        noise_level=0.0,
        accuracy=1.0
    ),
    SignalCategory.RESTRICTED: InfoCapability(
        enabled=False  # Disabled by default
    )
}