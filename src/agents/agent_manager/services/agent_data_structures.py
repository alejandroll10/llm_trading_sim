from dataclasses import dataclass
from typing import Dict
from market.information.base_information_services import InformationType, InfoCapability

@dataclass
class AgentDecisionContext:
    agent_id: str
    agent_type: str
    cash: float
    shares: int
    committed_cash: float
    committed_shares: int

@dataclass
class AgentCommitmentState:
    agent_id: str
    available_cash: float
    available_shares: int
    total_cash: float
    total_shares: int
    committed_cash: float
    committed_shares: int

@dataclass
class CommitmentResult:
    success: bool
    message: str
    committed_amount: float = 0

@dataclass
class AgentStateSnapshot:
    agent_id: str
    agent_type: str
    cash: float
    dividend_cash: float
    shares: int
    committed_cash: float
    committed_shares: int
    total_shares: int
    borrowed_shares: int
    net_shares: int
    wealth: float
    orders_by_state: dict
    trade_summary: dict

@dataclass
class PositionUpdate:
    agent_id: str
    cash_change: float
    shares_change: int
    new_cash: float
    new_shares: int

@dataclass
class AgentInfoProfile:
    """Information profile configuration"""
    capabilities: Dict[InformationType, InfoCapability]
    profile_id: str
    description: str = ""

@dataclass
class AgentDecisionState:
    agent_id: str
    last_replace_decision: str
    last_decision: dict 