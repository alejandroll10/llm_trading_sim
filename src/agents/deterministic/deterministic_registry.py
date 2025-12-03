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
from .squeeze_buyer_agent import SqueezeBuyerAgent
from .margin_buy_agent import MarginBuyAgent
from .market_buy_agent import MarketBuyAgent
from .multi_stock_value_trader import MultiStockValueTrader
from .multi_stock_buy_agent import MultiStockBuyAgent
from .multi_stock_sell_agent import MultiStockSellAgent
from .multi_stock_short_seller import MultiStockShortSeller
from .multi_stock_squeeze_buyer import MultiStockSqueezeBuyer
from .multi_stock_market_maker import MultiStockMarketMaker

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
    "short_seller": ShortSellTrader,  # Alias for consistency with agent_types.py
    "buy_to_close_trader": BuyToCloseTrader,
    "squeeze_buyer": SqueezeBuyerAgent,
    "margin_buyer": MarginBuyAgent,
    "market_buyer": MarketBuyAgent,  # For testing crossed market fix
    "multi_stock_test": MultiStockValueTrader,
    "multi_stock_value": MultiStockValueTrader,
    "multi_stock_buy": MultiStockBuyAgent,
    "multi_stock_sell": MultiStockSellAgent,
    "multi_stock_short_seller": MultiStockShortSeller,
    "multi_stock_squeeze_buyer": MultiStockSqueezeBuyer,
    "multi_stock_market_maker": MultiStockMarketMaker
}
