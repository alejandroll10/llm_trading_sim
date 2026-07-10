"""Per-stock market bundle and the collection BaseSimulation iterates.

A StockMarket owns everything specific to one traded stock: its context,
order book, borrowing pool, dividend service, market state manager,
matching engine, and (optionally) a per-round fundamental-value path.
MarketCollection maps stock_id -> StockMarket so the simulation's round
phases can iterate stocks uniformly; a single-stock run is simply the
N=1 case keyed by DEFAULT_STOCK_ID.

The scenario's mode flag (is_multi_stock) lives on BaseSimulation, not
here, because the agent-facing market_state shape, the recorded CSV
formats, and a few service constructors follow the scenario mode rather
than the stock count: a one-stock multi-stock config must keep
multi-stock semantics.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

DEFAULT_STOCK_ID = "DEFAULT_STOCK"


@dataclass
class StockMarket:
    """Everything owned by a single traded stock.

    Built in stages during BaseSimulation construction: context and
    borrowing repository first, then order book, dividend service,
    market state manager, and matching engine as their dependencies
    become available.
    """
    stock_id: str
    config: dict  # normalized per-stock scenario config
    context: object = None
    borrowing_repository: object = None
    order_book: object = None
    dividend_service: object = None
    market_state_manager: object = None
    matching_engine: object = None
    # Per-round fundamental values for dividend regime schedules;
    # None for stationary scenarios.
    fundamental_path: Optional[List[float]] = None
    # Style category for style-level dividend shocks (multi-stock only).
    style: Optional[str] = None


class MarketCollection:
    """Ordered mapping of stock_id -> StockMarket."""

    def __init__(self):
        self._markets: Dict[str, StockMarket] = {}

    def add(self, market: StockMarket) -> StockMarket:
        self._markets[market.stock_id] = market
        return market

    @property
    def primary(self) -> StockMarket:
        """The first stock (the only one in single-stock mode)."""
        return next(iter(self._markets.values()))

    def __getitem__(self, stock_id: str) -> StockMarket:
        return self._markets[stock_id]

    def __contains__(self, stock_id: str) -> bool:
        return stock_id in self._markets

    def __len__(self) -> int:
        return len(self._markets)

    def __iter__(self) -> Iterator[StockMarket]:
        return iter(self._markets.values())

    def keys(self):
        return self._markets.keys()

    def values(self):
        return self._markets.values()

    def items(self):
        return self._markets.items()

    # Dict views over per-stock components, for services that take
    # {stock_id: component} mappings.
    def contexts(self) -> Dict[str, object]:
        return {sid: m.context for sid, m in self._markets.items()}

    def order_books(self) -> Dict[str, object]:
        return {sid: m.order_book for sid, m in self._markets.items()}

    def borrowing_repositories(self) -> Dict[str, object]:
        return {sid: m.borrowing_repository for sid, m in self._markets.items()}

    def market_state_managers(self) -> Dict[str, object]:
        return {sid: m.market_state_manager for sid, m in self._markets.items()}

    def dividend_services(self) -> Dict[str, object]:
        """Only stocks that actually pay dividends (matches the legacy
        self.dividend_services dict, which omitted stocks without one)."""
        return {
            sid: m.dividend_service
            for sid, m in self._markets.items()
            if m.dividend_service is not None
        }

    def prices(self) -> Dict[str, float]:
        return {sid: m.context.current_price for sid, m in self._markets.items()}
