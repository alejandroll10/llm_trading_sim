from .gap_trader import ProportionalGapTrader
from .mean_reversion_trader import MeanReversionTrader
from .buy_agent import BuyTrader
from .sell_agent import SellTrader
from .momentum_trader import MomentumTrader
from .market_maker_buy import MarketMakerBuy
from .market_maker_sell import MarketMakerSell
from .deterministic_market_maker import DeterministicMarketMaker
from .hold_agent import HoldTrader
from .short_sell_agent import ShortSellTrader
from .buy_to_close_agent import BuyToCloseTrader

DETERMINISTIC_AGENTS = {
    "gap_trader": ProportionalGapTrader,
    "mean_reversion": MeanReversionTrader,
    "buy_trader": BuyTrader,
    "sell_trader": SellTrader,
    "momentum_trader": MomentumTrader,
    "market_maker_buy": MarketMakerBuy,
    "market_maker_sell": MarketMakerSell,
    "deterministic_market_maker": DeterministicMarketMaker,
    "hold_trader": HoldTrader,
    "short_sell_trader": ShortSellTrader,
    "buy_to_close_trader": BuyToCloseTrader
}
