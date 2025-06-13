from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PaymentDestination(Enum):
    MAIN_ACCOUNT = "main"
    DIVIDEND_ACCOUNT = "dividend"

    @classmethod
    def from_string(cls, value: Optional[str]) -> 'PaymentDestination':
        """Safely convert string to PaymentDestination enum"""
        if value is None:
            return cls.DIVIDEND_ACCOUNT  # default
            
        # Normalize input
        value = value.lower().strip()
        
        # Handle various input formats
        if value in ('main', 'main_account', 'mainaccount'):
            return cls.MAIN_ACCOUNT
        elif value in ('dividend', 'dividend_account', 'dividendaccount'):
            return cls.DIVIDEND_ACCOUNT
        else:
            valid_values = ", ".join(f"'{v.value}'" for v in cls)
            raise ValueError(
                f"Invalid payment destination: '{value}'. "
                f"Valid values are: {valid_values}"
            )

@dataclass
class PaymentCalculation:
    agent_id: str
    pre_cash: float
    payment_amount: float
    payment_type: str
    destination: PaymentDestination
    details: str