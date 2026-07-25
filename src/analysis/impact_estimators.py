"""
Phase-2 belief-action estimators: the lambda machinery (issue #111).

Phase 1 (``belief_action_estimators.py``) covered the assumption-free layers --
price coherence, direction coherence, forecast skill, belief-message
divergence. This module adds everything that needs a price-impact model:

  1. lambda_book -- the walk-the-book impact slope computed from the order-book
     snapshot each agent ACTUALLY SAW, parsed out of the exact per-round prompt
     logged to ``data/rendered_prompts.jsonl``. Because the info-capability
     layer can truncate book depth per agent, the prompt is the only ground
     truth for the agent's information set; reconstructing it from
     market_data.csv + config would be wrong for depth-limited agents.
  2. lambda_hat_realized -- equilibrium impact: a pooled regression of the
     round's price change on that round's net submitted order flow (all
     agents, from ``order_data.csv``), on marketable flow only (the
     theoretically right regressor -- resting limits away from the touch
     cannot move the price), and on returns rather than dollar changes for
     comparability across cells with different price levels.
  3. lambda_perceived -- inferred from order shading, via three channels:
       (a) size shading: q = edge / lambda  =>  slope of q on edge is
           1/lambda. Confounded with risk aversion (a mean-variance agent
           trades edge / (lambda + gamma sigma^2)), so this is reported as a
           JOINT estimate and is an upper bound on true perceived impact.
       (b) placement distance: the price concession a limit order demands
           relative to the agent's OWN same-response next-price forecast
           (price_prediction_t), regressed on order size. Using the agent's own
           forecast is what separates impact from patience ("the price will
           come to me" vs "I will move the price").
       (c) marketable share vs size: the fraction of a decision's volume that
           crosses the visible book, regressed on decision size -- agents who
           perceive impact go passive when they trade big.
  4. Size coherence c = q_submitted / q*, with q* = edge / lambda -- reported
     against lambda_book (the agent's own information set: the primary
     benchmark), against lambda_hat_realized (robustness), and in a fit-free
     form c_walk = 2 (avg_exec(q) - p_touch) / edge_at_touch for orders that
     actually consume the book, where calibrating the linear model at the
     order's own size cancels lambda out entirely. Reported separately for
     passive limits, marketable limits and market orders.
  5. The lambda gaps, which are estimands rather than nuisances:
       lambda_perceived - lambda_book  : does the agent use the book it is shown?
       lambda_book - lambda_hat_realized: is the visible book informative at all,
                                          given simultaneous submission, shuffled
                                          market-order processing and Replace churn?

Inputs (all under ``<sweep_root>/aggregated/``, from ``aggregate_sweep.py``):

    structured_decisions_panel.csv   per-order LLM decisions + stated beliefs
    market_data_panel.csv            per-round price / touch
    order_data_panel.csv             ALL agents' submitted orders (flow)

plus, read directly from each cell's run directory via ``manifest.json``:

    <run_dir>/data/rendered_prompts.jsonl   the exact prompts, for lambda_book

The parsed book snapshots are cached to
``<sweep_root>/aggregated/book_snapshots_panel.csv`` (re-parse with
``--rebuild-books``). Runs recorded before the phase-1 commit have no
rendered_prompts.jsonl; lambda_book is then simply absent and every table that
depends on it is skipped with a note -- the lambda_hat_realized arm still runs.

Round-timing convention (same as phase 1): ``structured_decisions.csv`` and
``rendered_prompts.jsonl`` log the raw decision round r (0-indexed), while
``market_data.csv`` and ``order_data.csv`` log ``round_number + 1``. So market
row k is the state AFTER round k-1's matching: the state a round-r
decision-maker saw is market row r, the outcome of round r is market row r+1,
and order-data row k holds the orders submitted in decision round k-1.

Limitations, stated rather than modelled:
  * c is unconditional -- no fill-probability weighting -- so passive c
    overstates intended exposure. Passive and marketable are reported apart so
    the reader can discount accordingly.
  * lambda_perceived from size shading is joint with risk aversion.
  * lambda_book uses a linear (weighted least-squares) fit through the visible
    ladder, which is the linear impact model that makes q* = edge/lambda the
    optimum. The exact non-linear walk-the-book cost is reported alongside it.

Usage
-----
    python src/analysis/impact_estimators.py logs/sweeps/<sweep_name>
    python src/analysis/impact_estimators.py logs/sweeps/<sweep_name> --rebuild-books
    python src/analysis/impact_estimators.py logs/sweeps/<sweep_name> --cluster prompt_family
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make src/ importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.a1_estimators import ols_cluster_robust  # noqa: E402
from analysis.belief_action_estimators import (  # noqa: E402
    CELL_COLS, SUMMARY_GROUP, market_per_round, normalize_order_type,
)
from aggregate_sweep import cell_prompt_family  # noqa: E402


# Prompt landmarks. BASE_MARKET_TEMPLATE emits "Market Depth:" then the book,
# then MarketStateFormatter._format_outstanding_orders' "Your Outstanding
# Orders:" -- which repeats the "Buy Orders:" / "Sell Orders:" headers for the
# agent's OWN resting orders, so the block must be cut there.
BOOK_START = "Market Depth:"
BOOK_END = "Your Outstanding Orders:"
_LEVEL_RE = re.compile(r"^-\s*([\d,]+)\s+shares\s*@\s*\$([\d,.]+)\s*$")
_BEST_BID_RE = re.compile(r"^Best Bid:\s*\$([\d,.]+)\s*$")
_BEST_ASK_RE = re.compile(r"^Best Ask:\s*\$([\d,.]+)\s*$")

# c is a ratio of submitted to optimal size; these cut the distribution into
# "traded far too small / about right / far too big" buckets for the tables.
C_UNDER = 0.5
C_OVER = 2.0


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


# --------------------------------------------------------------------------- #
# Book snapshots: parse the exact prompt each agent saw
# --------------------------------------------------------------------------- #
def parse_book_block(user_prompt: str):
    """Extract the order book an agent was shown from its rendered user prompt.

    Returns a dict with ``bids`` and ``asks`` as lists of (price, quantity)
    ordered BEST FIRST (bids descending, asks ascending), plus the stated best
    bid / best ask. Returns None when the prompt has no market-depth block.

    The displayed book is already consolidated by price level and printed
    highest-price-first on both sides (see
    MarketStateFormatter._format_order_book), so the sell side is reversed here
    to put the best (lowest) ask first.
    """
    if not isinstance(user_prompt, str):
        return None
    start = user_prompt.find(BOOK_START)
    if start < 0:
        return None
    end = user_prompt.find(BOOK_END, start)
    block = user_prompt[start:end if end > 0 else None]

    best_bid = best_ask = np.nan
    sells, buys = [], []
    section = None
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _BEST_BID_RE.match(line)
        if m:
            best_bid = _to_float(m.group(1))
            continue
        m = _BEST_ASK_RE.match(line)
        if m:
            best_ask = _to_float(m.group(1))
            continue
        low = line.lower()
        if low.startswith("sell orders"):
            section = "sell"
            continue
        if low.startswith("buy orders"):
            section = "buy"
            continue
        if low.startswith("no sell orders") or low.startswith("no buy orders"):
            section = None
            continue
        m = _LEVEL_RE.match(line)
        if m and section is not None:
            qty, price = _to_float(m.group(1)), _to_float(m.group(2))
            (sells if section == "sell" else buys).append((price, qty))

    asks = sorted(sells, key=lambda t: t[0])          # best (lowest) ask first
    bids = sorted(buys, key=lambda t: -t[0])          # best (highest) bid first
    if np.isnan(best_ask) and asks:
        best_ask = asks[0][0]
    if np.isnan(best_bid) and bids:
        best_bid = bids[0][0]
    return {"bids": bids, "asks": asks, "best_bid": best_bid, "best_ask": best_ask}


def ladder_lambda(levels):
    """Linear impact slope (per share) fitted through one side of the book.

    ``levels`` are (price, quantity) best-first. Each level is placed at the
    MIDPOINT of the cumulative-depth interval it occupies and the fit is
    weighted by that level's size, so the slope lambda is the one for which
    walking q shares costs about ``p_best + lambda * q / 2`` -- exactly the
    linear model under which q* = edge / lambda is optimal.

    Returns (lambda, n_levels, total_depth). lambda is NaN whenever the ladder
    carries no slope information -- fewer than two levels, no depth, or a flat
    fit. Undefined rather than zero: zero impact sends q* = edge/lambda to
    infinity and would score every order as infinitely under-sized.
    """
    if len(levels) < 2:
        return np.nan, len(levels), float(sum(q for _, q in levels))
    p = np.array([lv[0] for lv in levels], dtype=float)
    q = np.array([lv[1] for lv in levels], dtype=float)
    depth = float(q.sum())
    if depth <= 0:
        return np.nan, len(levels), depth
    cum = np.cumsum(q)
    mid = cum - q / 2.0
    w = q
    mbar = np.average(mid, weights=w)
    pbar = np.average(p, weights=w)
    var = np.average((mid - mbar) ** 2, weights=w)
    if var <= 0:
        return np.nan, len(levels), depth
    cov = np.average((mid - mbar) * (p - pbar), weights=w)
    slope = abs(cov / var)
    return (slope if slope > 0 else np.nan), len(levels), depth


def walk_book(levels, quantity, price_cap=None, is_buy=True):
    """Average execution price for consuming ``quantity`` shares best-first.

    ``price_cap`` is the order's own limit price, beyond which it cannot
    execute: a marketable buy limit only lifts asks at or below its limit, a
    marketable sell limit only hits bids at or above it. Market orders pass
    None.

    Returns (avg_price, filled_quantity, exhausted) where ``exhausted`` is True
    when the executable depth ran out before the order was filled -- avg_price
    then covers only the filled part and understates the full cost, so those
    rows are flagged rather than dropped.
    """
    if quantity is None or not np.isfinite(quantity) or quantity <= 0 or not levels:
        return np.nan, 0.0, False
    remaining = float(quantity)
    cost = 0.0
    filled = 0.0
    capped = price_cap is not None and np.isfinite(price_cap)
    for price, qty in levels:
        price = float(price)
        if capped and ((is_buy and price > price_cap) or
                       (not is_buy and price < price_cap)):
            break
        take = min(remaining, float(qty))
        cost += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled <= 0:
        return np.nan, 0.0, True
    return cost / filled, filled, remaining > 0


def build_book_snapshots(sweep_root: Path) -> pd.DataFrame:
    """Parse every cell's rendered_prompts.jsonl into a tidy per-decision frame.

    One row per (cell_id, agent_id, round) with the seen touch, both fitted
    ladder slopes, level counts and visible depth. Cells recorded before
    rendered-prompt logging landed contribute nothing (and are counted).
    """
    manifest_path = sweep_root / "manifest.json"
    if not manifest_path.exists():
        print(f"[warn] no manifest.json in {sweep_root}; lambda_book unavailable.")
        return pd.DataFrame()
    with open(manifest_path) as f:
        manifest = json.load(f)
    cells = [c for c in manifest.get("cells", {}).values()
             if c.get("status") == "done" and c.get("run_dir")]

    rows = []
    n_missing = 0
    for cell in cells:
        path = Path(cell["run_dir"]) / "data" / "rendered_prompts.jsonl"
        if not path.exists():
            n_missing += 1
            continue
        prompts = pd.read_json(path, lines=True)
        for rec in prompts.to_dict("records"):
            book = parse_book_block(rec.get("user_prompt"))
            if book is None:
                continue
            lam_ask, n_ask, depth_ask = ladder_lambda(book["asks"])
            lam_bid, n_bid, depth_bid = ladder_lambda(book["bids"])
            rows.append({
                "cell_id": cell["cell_id"],
                "seed": cell.get("seed"),
                "temperature": cell.get("temperature"),
                "model": cell.get("model"),
                "variant": cell.get("variant"),
                "prompt_family": cell_prompt_family(cell),
                "agent_id": rec.get("agent_id"),
                "round": rec.get("round"),
                "book_best_bid": book["best_bid"],
                "book_best_ask": book["best_ask"],
                # lambda_book is side-specific: a buy walks the ask ladder.
                "lambda_book_buy": lam_ask,
                "lambda_book_sell": lam_bid,
                "ask_levels": n_ask,
                "bid_levels": n_bid,
                "ask_depth": depth_ask,
                "bid_depth": depth_bid,
                "asks_json": json.dumps(book["asks"]),
                "bids_json": json.dumps(book["bids"]),
            })
    if n_missing:
        print(f"[warn] {n_missing} of {len(cells)} cells have no "
              f"data/rendered_prompts.jsonl (recorded before the phase-1 "
              f"logging change); they contribute no lambda_book.")
    return pd.DataFrame(rows)


def load_book_snapshots(sweep_root: Path, rebuild: bool) -> pd.DataFrame:
    """Load the cached book-snapshot panel, parsing the prompts if needed."""
    cache = sweep_root / "aggregated" / "book_snapshots_panel.csv"
    if cache.exists() and not rebuild:
        return pd.read_csv(cache)
    books = build_book_snapshots(sweep_root)
    if len(books):
        cache.parent.mkdir(parents=True, exist_ok=True)
        books.to_csv(cache, index=False)
        print(f"Parsed {len(books):,} book snapshots -> {cache}")
    return books


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def load_panels(sweep_root: Path):
    """Load the decisions, market and order-flow panels."""
    agg = sweep_root / "aggregated"
    dec_path = agg / "structured_decisions_panel.csv"
    if not dec_path.exists():
        sys.exit(f"Error: {dec_path} not found. Run aggregate_sweep.py on the sweep first.")
    decisions = pd.read_csv(dec_path)
    mkt_path = agg / "market_data_panel.csv"
    market = pd.read_csv(mkt_path) if mkt_path.exists() else None
    ord_path = agg / "order_data_panel.csv"
    orders = pd.read_csv(ord_path) if ord_path.exists() else None
    if orders is None:
        print("[warn] order_data_panel.csv not found -- lambda_hat_realized needs "
              "market-wide flow. Re-run: python src/aggregate_sweep.py <sweep> "
              "(order_data.csv is aggregated by default).")
    return decisions, market, orders


def build_decision_frame(decisions: pd.DataFrame, market: pd.DataFrame,
                         books: pd.DataFrame) -> pd.DataFrame:
    """Per-order frame with the agent's beliefs, the book it saw, and lambda_book.

    The seen touch comes from the rendered prompt when available (it reflects
    per-agent depth truncation) and falls back to market row r, which is the
    pre-round public state for decision round r.
    """
    df = decisions.copy()
    df["otype"] = normalize_order_type(df["order_type"])
    df["side"] = df["decision"].astype(str).str.lower().str.strip()
    for col in ["quantity", "price", "valuation", "price_prediction_t"]:
        df[col] = (pd.to_numeric(df[col], errors="coerce")
                   if col in df.columns else np.nan)

    if market is not None and len(market):
        seen = market_per_round(market).rename(columns={
            "obs_price": "seen_price", "best_bid": "mkt_best_bid",
            "best_ask": "mkt_best_ask"})
        df = df.merge(seen[["cell_id", "round", "seen_price",
                            "mkt_best_bid", "mkt_best_ask"]],
                      on=["cell_id", "round"], how="left")
    else:
        for col in ["seen_price", "mkt_best_bid", "mkt_best_ask"]:
            df[col] = np.nan

    if len(books):
        cols = ["cell_id", "agent_id", "round", "book_best_bid", "book_best_ask",
                "lambda_book_buy", "lambda_book_sell", "ask_levels", "bid_levels",
                "ask_depth", "bid_depth", "asks_json", "bids_json"]
        # agent_id round-trips through CSV and JSONL, which do not agree on its
        # dtype when ids are numeric-looking strings; match on text.
        b = books[cols].copy()
        b["agent_id"] = b["agent_id"].astype(str)
        b["round"] = pd.to_numeric(b["round"], errors="coerce")
        df["_agent_key"] = df["agent_id"].astype(str)
        df = df.merge(b.rename(columns={"agent_id": "_agent_key"}),
                      on=["cell_id", "_agent_key", "round"], how="left")
        df = df.drop(columns="_agent_key")
    else:
        for col in ["book_best_bid", "book_best_ask", "lambda_book_buy",
                    "lambda_book_sell", "ask_levels", "bid_levels",
                    "ask_depth", "bid_depth", "asks_json", "bids_json"]:
            df[col] = np.nan

    # Prefer the prompt's touch (per-agent view) over the public market panel.
    df["seen_best_bid"] = df["book_best_bid"].fillna(df["mkt_best_bid"])
    df["seen_best_ask"] = df["book_best_ask"].fillna(df["mkt_best_ask"])

    df["has_valuation"] = df["valuation"].notna() & (df["valuation"] > 0)

    # Marketability against the touch the agent saw (same rule as phase 1).
    df["marketability"] = "unknown"
    df.loc[df["otype"] == "market", "marketability"] = "market"
    is_limit = df["otype"] == "limit"
    buy_m = is_limit & (df["side"] == "buy") & df["seen_best_ask"].notna() \
        & (df["seen_best_ask"] > 0) & (df["price"] >= df["seen_best_ask"])
    sell_m = is_limit & (df["side"] == "sell") & df["seen_best_bid"].notna() \
        & (df["seen_best_bid"] > 0) & (df["price"] <= df["seen_best_bid"])
    buy_p = is_limit & (df["side"] == "buy") & df["seen_best_ask"].notna() \
        & (df["seen_best_ask"] > 0) & (df["price"] < df["seen_best_ask"])
    sell_p = is_limit & (df["side"] == "sell") & df["seen_best_bid"].notna() \
        & (df["seen_best_bid"] > 0) & (df["price"] > df["seen_best_bid"])
    df.loc[buy_m | sell_m, "marketability"] = "marketable_limit"
    df.loc[buy_p | sell_p, "marketability"] = "passive_limit"
    # A limit order facing a book the agent demonstrably saw with nothing on the
    # far side cannot cross: that is passive, not unclassifiable. Only rows with
    # no book snapshot at all stay "unknown".
    has_snapshot = df["ask_levels"].notna() | df["bid_levels"].notna()
    empty_far_side = ((df["side"] == "buy") & df["seen_best_ask"].isna()) | \
                     ((df["side"] == "sell") & df["seen_best_bid"].isna())
    df.loc[is_limit & has_snapshot & empty_far_side, "marketability"] = "passive_limit"

    # Per-share edge, in the agent's own units of value. Limit orders execute at
    # their own price under price-time priority, so V - P_limit is the exact
    # edge conditional on fill (a lower bound for crossing limits, which may
    # execute better). Market orders are scored at the touch they saw.
    is_buy = df["side"] == "buy"
    limit_ref = df["price"]
    market_ref = np.where(is_buy, df["seen_best_ask"], df["seen_best_bid"])
    ref = np.where(df["otype"] == "limit", limit_ref, market_ref)
    df["exec_ref_price"] = pd.to_numeric(pd.Series(ref, index=df.index), errors="coerce")
    df["edge"] = np.where(is_buy,
                          df["valuation"] - df["exec_ref_price"],
                          df["exec_ref_price"] - df["valuation"])
    df.loc[~df["has_valuation"] | df["exec_ref_price"].isna(), "edge"] = np.nan

    # The same edge measured at the touch instead of at the agent's own limit.
    # The linear impact model is anchored at the touch (cost = p_touch +
    # lambda q / 2), so this is the edge that belongs in the walk-based c; for
    # market orders the two coincide, and for a crossing limit this is the
    # larger of the two -- which is exactly why the P_limit version is only a
    # lower bound there.
    touch = pd.to_numeric(
        pd.Series(np.where(is_buy, df["seen_best_ask"], df["seen_best_bid"]),
                  index=df.index), errors="coerce")
    df["touch_price"] = touch
    df["edge_at_touch"] = np.where(is_buy, df["valuation"] - touch,
                                   touch - df["valuation"])
    df.loc[~df["has_valuation"] | touch.isna(), "edge_at_touch"] = np.nan
    # The side-matched book slope: a buy walks the asks, a sell walks the bids.
    # This holds for passive limits too -- the ask ladder is still the price of
    # acquiring shares now, which is the alternative a resting buy declines --
    # but the linear-impact story is weakest there, which is why passive orders
    # are always reported as their own row rather than pooled.
    df["lambda_book"] = np.where(is_buy, df["lambda_book_buy"], df["lambda_book_sell"])
    df["is_trade"] = df["otype"].isin(["limit", "market"]) & df["quantity"].gt(0)
    return df


# --------------------------------------------------------------------------- #
# Estimator: lambda_hat_realized (equilibrium impact from submitted flow)
# --------------------------------------------------------------------------- #
def build_flow_frame(orders: pd.DataFrame, market: pd.DataFrame):
    """Per (cell, round) net submitted order flow and the price change it met.

    ``order_data.csv`` labels rows with ``round_number + 1``, so the orders on
    row k were submitted in decision round k-1 and met the price move from
    market row k-1 to market row k. Market rows start at 1 (round 0's
    pre-trade price is the scenario's INITIAL_PRICE and is not in the panel),
    so the first round's flow has no measurable price change and is dropped.
    """
    if orders is None or not len(orders) or market is None or not len(market):
        return None
    required = {"cell_id", "round", "decision", "quantity", "order_type"}
    missing = required - set(orders.columns)
    if missing:
        print(f"[warn] order_data_panel.csv lacks {sorted(missing)} "
              f"(every cell's order_data.csv was empty?) -- skipping "
              f"lambda_hat_realized.")
        return None
    o = orders.copy()
    o["side"] = o["decision"].astype(str).str.lower().str.strip()
    o["otype"] = normalize_order_type(o["order_type"])
    for col in ["quantity", "price_limit"]:
        o[col] = pd.to_numeric(o[col], errors="coerce") if col in o.columns else np.nan
    o = o[o["quantity"].notna() & (o["quantity"] > 0)
          & o["side"].isin(["buy", "sell"])].copy()
    if o.empty:
        return None

    per_round = market_per_round(market)
    # The book state the flow was submitted into is the previous market row.
    prev = per_round.rename(columns={
        "round": "prev_round", "obs_price": "prev_price",
        "best_bid": "prev_best_bid", "best_ask": "prev_best_ask"})
    prev["round"] = prev["prev_round"] + 1
    o = o.merge(prev[["cell_id", "round", "prev_price",
                      "prev_best_bid", "prev_best_ask"]],
                on=["cell_id", "round"], how="left")

    is_buy = o["side"] == "buy"
    o["signed_qty"] = np.where(is_buy, o["quantity"], -o["quantity"])
    crossing = ((o["otype"] == "market")
                | (is_buy & o["prev_best_ask"].notna()
                   & (o["price_limit"] >= o["prev_best_ask"]))
                | (~is_buy & o["prev_best_bid"].notna()
                   & (o["price_limit"] <= o["prev_best_bid"])))
    o["marketable_signed_qty"] = np.where(crossing, o["signed_qty"], 0.0)

    flow = (o.groupby(["cell_id", "round"], as_index=False)
             .agg(net_flow=("signed_qty", "sum"),
                  marketable_net_flow=("marketable_signed_qty", "sum"),
                  gross_qty=("quantity", "sum"),
                  n_orders=("quantity", "size"),
                  prev_price=("prev_price", "first")))

    flow = flow.merge(per_round[["cell_id", "round", "obs_price"]],
                      on=["cell_id", "round"], how="left")
    flow = flow[flow["prev_price"].notna() & flow["obs_price"].notna()].copy()
    if flow.empty:
        return None
    flow["price_change"] = flow["obs_price"] - flow["prev_price"]
    flow["return"] = flow["price_change"] / flow["prev_price"]

    # Re-attach cell metadata (one row per cell in the order panel).
    meta_cols = ["prompt_family", "model", "temperature", "seed", "variant"]
    present = [c for c in meta_cols if c in orders.columns]
    meta = orders[["cell_id"] + present].drop_duplicates("cell_id")
    flow = flow.merge(meta, on="cell_id", how="left")
    for col in meta_cols:  # keep the grouping keys addressable even if absent
        if col not in flow.columns:
            flow[col] = np.nan
    return flow


def _flow_regression(sub: pd.DataFrame, xcol: str, cluster_col: str, label: str,
                     group_label: str, ycol: str = "price_change"):
    """Regress the price response on net flow; the slope IS lambda_hat_realized."""
    sub = sub[sub[xcol].notna() & sub[ycol].notna()]
    row = {"regression": label, "group": group_label, "n_obs": len(sub),
           "n_clusters": sub[cluster_col].nunique() if len(sub) else 0,
           "lambda_hat": np.nan, "se": np.nan, "t": np.nan, "intercept": np.nan}
    if len(sub) < 3 or sub[xcol].nunique() < 2:
        return row
    X = np.column_stack([np.ones(len(sub)), sub[xcol].to_numpy(dtype=float)])
    beta, se, n, G = ols_cluster_robust(sub[ycol].to_numpy(dtype=float), X,
                                        sub[cluster_col].to_numpy())
    if G < 2:
        # Cluster-robust SEs are degenerate with one cluster: keep the point
        # estimate, blank the inference so it is not read as a test.
        se = np.array([np.nan, np.nan])
    row.update(n_obs=n, n_clusters=G, lambda_hat=beta[1], se=se[1],
               t=beta[1] / se[1] if se[1] and se[1] > 0 else np.nan,
               intercept=beta[0])
    return row


def lambda_realized_tables(flow: pd.DataFrame, cluster_col: str):
    """Pooled + per-group lambda_hat_realized, on total and marketable flow.

    The marketable-flow spec is the theoretically right one -- resting limits
    away from the touch cannot move the price -- and the total-flow spec is
    kept as the naive benchmark. The ``return~`` spec is the scale-free
    version: pooling cells whose price levels differ by a factor of several
    makes a $-per-share slope a weighted average of incomparable regimes, so
    the return slope is what should be compared across scenarios.
    """
    rows = []
    specs = [("price_change~net_flow", "net_flow", "price_change"),
             ("price_change~marketable_net_flow", "marketable_net_flow", "price_change"),
             ("return~marketable_net_flow", "marketable_net_flow", "return")]
    for label, xcol, ycol in specs:
        rows.append(_flow_regression(flow, xcol, cluster_col, label, "pooled", ycol))
        for keys, sub in flow.groupby(["prompt_family", "model"], dropna=False):
            rows.append(_flow_regression(sub, xcol, cluster_col, label,
                                         f"{keys[0]} / {keys[1]}", ycol))
    pooled = pd.DataFrame(rows)

    # Per-cell slopes give the cross-cell distribution (a1 convention). Both
    # regressors are kept; lambda_hat_realized -- the one c_realized is scored
    # against -- prefers marketable flow and falls back to total flow only when
    # the cell has too little marketable variation to fit a slope.
    def _slope(sub, xcol):
        s = sub[sub[xcol].notna() & sub["price_change"].notna()]
        if len(s) < 3 or s[xcol].nunique() < 2:
            return np.nan, len(s)
        return np.polyfit(s[xcol].to_numpy(dtype=float),
                          s["price_change"].to_numpy(dtype=float), 1)[0], len(s)

    per_cell = []
    for cell_id, sub in flow.groupby("cell_id"):
        lam_m, n_rounds = _slope(sub, "marketable_net_flow")
        lam_t, _ = _slope(sub, "net_flow")
        per_cell.append({"cell_id": cell_id,
                         "prompt_family": sub["prompt_family"].iloc[0],
                         "model": sub["model"].iloc[0],
                         "temperature": sub["temperature"].iloc[0],
                         "seed": sub["seed"].iloc[0],
                         "n_rounds": n_rounds,
                         "lambda_realized_marketable": lam_m,
                         "lambda_realized_total": lam_t,
                         "lambda_hat_realized": lam_m if np.isfinite(lam_m) else lam_t})
    return pooled, pd.DataFrame(per_cell)


# --------------------------------------------------------------------------- #
# Estimator: lambda_perceived (three channels)
# --------------------------------------------------------------------------- #
def _cluster_ols(sub, ycol, xcol, cluster_col, label, group_label):
    sub = sub[sub[ycol].notna() & sub[xcol].notna()]
    row = {"channel": label, "group": group_label, "n_obs": len(sub),
           "n_clusters": sub[cluster_col].nunique() if len(sub) else 0,
           "slope": np.nan, "se": np.nan, "t": np.nan, "intercept": np.nan}
    if len(sub) < 4 or sub[xcol].nunique() < 2:
        return row
    X = np.column_stack([np.ones(len(sub)), sub[xcol].to_numpy(dtype=float)])
    beta, se, n, G = ols_cluster_robust(sub[ycol].to_numpy(dtype=float), X,
                                        sub[cluster_col].to_numpy())
    if G < 2:
        se = np.array([np.nan, np.nan])
    row.update(n_obs=n, n_clusters=G, slope=beta[1], se=se[1],
               t=beta[1] / se[1] if se[1] and se[1] > 0 else np.nan,
               intercept=beta[0])
    return row


def size_shading_table(df: pd.DataFrame, cluster_col: str):
    """Channel (a): q = edge / lambda  =>  lambda_perceived = 1 / slope(q on edge).

    Restricted to orders with a positive edge (the model has nothing to say
    about trading against your own stated valuation -- that is the phase-1
    direction-coherence layer). The implied lambda is JOINT with risk aversion:
    a mean-variance agent trades edge / (lambda + gamma sigma^2), so the number
    below is an upper bound on perceived impact. Order sizes are additionally
    censored from above by max_order_size and by the agent's cash / share
    budget, which attenuates the slope and pushes the implied lambda up again
    -- both distortions point the same way, so read it as a ceiling.
    """
    sub = df[df["is_trade"] & df["edge"].notna() & (df["edge"] > 0)
             & df["quantity"].notna() & (df["quantity"] > 0)].copy()
    if sub.empty:
        return None
    rows = [_cluster_ols(sub, "quantity", "edge", cluster_col,
                         "size_shading(q~edge)", "pooled")]
    for key, g in sub.groupby("agent_type", dropna=False):
        rows.append(_cluster_ols(g, "quantity", "edge", cluster_col,
                                 "size_shading(q~edge)", f"agent_type={key}"))
    for key, g in sub.groupby("marketability", dropna=False):
        rows.append(_cluster_ols(g, "quantity", "edge", cluster_col,
                                 "size_shading(q~edge)", f"marketability={key}"))
    out = pd.DataFrame(rows)
    # lambda = 1/slope; a non-positive slope means "trades bigger when the edge
    # is smaller", which no impact model rationalizes -> undefined, not negative.
    out["lambda_perceived_joint"] = np.where(out["slope"] > 0, 1.0 / out["slope"], np.nan)
    return out


def placement_distance_table(df: pd.DataFrame, cluster_col: str):
    """Channel (b): price concession demanded, vs order size.

    Concession is measured against the agent's OWN same-response forecast of
    the next price (price_prediction_t): a buy limit posted far below where the
    agent itself expects the price to be is either patience or perceived
    impact, and conditioning on the agent's own forecast is what removes the
    patience story. A positive slope on size = "I demand more concession when I
    trade big" = perceived impact.
    """
    lim = df[(df["otype"] == "limit") & df["price"].notna() & (df["price"] > 0)
             & df["price_prediction_t"].notna() & (df["price_prediction_t"] > 0)
             & df["quantity"].notna() & (df["quantity"] > 0)].copy()
    if lim.empty:
        return None, None, None
    is_buy = lim["side"] == "buy"
    lim["concession"] = np.where(is_buy,
                                 lim["price_prediction_t"] - lim["price"],
                                 lim["price"] - lim["price_prediction_t"])
    lim["concession_pct"] = lim["concession"] / lim["price_prediction_t"]
    lim["log10_qty"] = np.log10(lim["quantity"])

    rows = [_cluster_ols(lim, "concession_pct", "log10_qty", cluster_col,
                         "placement_distance(concession%~log10 q)", "pooled")]
    for key, g in lim.groupby("agent_type", dropna=False):
        rows.append(_cluster_ols(g, "concession_pct", "log10_qty", cluster_col,
                                 "placement_distance(concession%~log10 q)",
                                 f"agent_type={key}"))
    per_cell = (lim.groupby(["agent_type", "marketability"] + CELL_COLS,
                            as_index=False)
                   .agg(n_limit_orders=("concession_pct", "size"),
                        mean_concession=("concession", "mean"),
                        mean_concession_pct=("concession_pct", "mean")))
    return pd.DataFrame(rows), per_cell, lim


def marketable_share_table(df: pd.DataFrame, cluster_col: str):
    """Channel (c): does the agent go passive when it trades big?

    Aggregated to the decision (all orders in one response), because the
    marketable share is a property of the schedule, not of a single line.
    Orders whose marketability could not be classified (no record of the book
    the agent saw) are excluded from both numerator and denominator rather than
    silently counted as passive, which would bias the share down.
    """
    trades = df[df["is_trade"] & (df["marketability"] != "unknown")].copy()
    n_unknown = int((df["is_trade"] & (df["marketability"] == "unknown")).sum())
    if n_unknown:
        print(f"[warn] {n_unknown} orders have no book context and are excluded "
              f"from the marketable-share channel.")
    if trades.empty:
        return None, None
    trades["marketable_qty"] = np.where(
        trades["marketability"].isin(["market", "marketable_limit"]),
        trades["quantity"], 0.0)
    grp = ["cell_id", "agent_id", "round", "agent_type"] + \
          [c for c in ["prompt_family", "model", "temperature", "seed"]
           if c in trades.columns]
    dec = (trades.groupby(grp, as_index=False)
                 .agg(total_qty=("quantity", "sum"),
                      marketable_qty=("marketable_qty", "sum"),
                      n_orders=("quantity", "size")))
    dec = dec[dec["total_qty"] > 0].copy()
    if dec.empty:
        return None, None
    dec["marketable_share"] = dec["marketable_qty"] / dec["total_qty"]
    dec["log10_qty"] = np.log10(dec["total_qty"])
    rows = [_cluster_ols(dec, "marketable_share", "log10_qty", cluster_col,
                         "marketable_share(~log10 decision size)", "pooled")]
    for key, g in dec.groupby("agent_type", dropna=False):
        rows.append(_cluster_ols(g, "marketable_share", "log10_qty", cluster_col,
                                 "marketable_share(~log10 decision size)",
                                 f"agent_type={key}"))
    return pd.DataFrame(rows), dec


# --------------------------------------------------------------------------- #
# Estimator: size coherence c = q / q*
# --------------------------------------------------------------------------- #
def size_coherence_tables(df: pd.DataFrame, lambda_realized_by_cell: pd.DataFrame,
                          walk: pd.DataFrame = None):
    """c = q_submitted / q* with q* = edge / lambda, against three benchmarks.

    ``c_book`` judges the order against the linear impact slope implied by the
    book the agent itself was shown (the primary benchmark: coherence is judged
    against the agent's information set). ``c_walk`` is the fit-free version of
    the same idea for orders that actually consume the book (see
    ``walk_cost_table``). ``c_realized`` uses that cell's equilibrium impact as
    a robustness check. All three are unconditional -- no fill-probability
    weighting -- so passive c overstates intended exposure.
    """
    sub = df[df["is_trade"] & df["edge"].notna() & (df["edge"] > 0)
             & df["quantity"].notna() & (df["quantity"] > 0)].copy()
    if sub.empty:
        return None, None, None

    if walk is not None and len(walk):
        sub["c_walk"] = walk.set_index("row_idx")["c_walk"].reindex(sub.index)
    else:
        sub["c_walk"] = np.nan

    if lambda_realized_by_cell is not None and len(lambda_realized_by_cell):
        sub = sub.merge(
            lambda_realized_by_cell[["cell_id", "lambda_hat_realized"]],
            on="cell_id", how="left")
    else:
        sub["lambda_hat_realized"] = np.nan

    for name, lam_col in [("book", "lambda_book"), ("realized", "lambda_hat_realized")]:
        lam = pd.to_numeric(sub[lam_col], errors="coerce")
        q_star = np.where(lam > 0, sub["edge"] / lam, np.nan)
        sub[f"q_star_{name}"] = q_star
        sub[f"c_{name}"] = np.where(
            np.isfinite(q_star) & (q_star > 0), sub["quantity"] / q_star, np.nan)

    if not sub[["c_book", "c_walk", "c_realized"]].notna().any().any():
        return sub, None, None

    def _rate(series, predicate):
        # Undefined (NaN), not 0, when a group has no scoreable order: a rate
        # over an empty set would dilute the cross-cell mean.
        vals = series.dropna()
        return predicate(vals).mean() if len(vals) else np.nan

    grp = ["agent_type", "marketability"] + CELL_COLS
    per_cell = (sub.groupby(grp, as_index=False)
                   .agg(n_orders=("quantity", "size"),
                        n_scored_book=("c_book", "count"),
                        n_scored_walk=("c_walk", "count"),
                        n_scored_realized=("c_realized", "count"),
                        median_c_book=("c_book", "median"),
                        mean_c_book=("c_book", "mean"),
                        median_c_walk=("c_walk", "median"),
                        median_c_realized=("c_realized", "median"),
                        under_rate_book=("c_book",
                                         lambda s: _rate(s, lambda v: v < C_UNDER)),
                        over_rate_book=("c_book",
                                        lambda s: _rate(s, lambda v: v > C_OVER))))
    summary = (per_cell.groupby(SUMMARY_GROUP + ["marketability"], as_index=False)
                       .agg(n_cells=("median_c_book", "size"),
                            n_orders=("n_orders", "sum"),
                            n_scored_book=("n_scored_book", "sum"),
                            n_scored_walk=("n_scored_walk", "sum"),
                            median_c_book_mean=("median_c_book", "mean"),
                            median_c_book_std=("median_c_book", "std"),
                            median_c_walk_mean=("median_c_walk", "mean"),
                            median_c_realized_mean=("median_c_realized", "mean"),
                            under_rate_book_mean=("under_rate_book", "mean"),
                            over_rate_book_mean=("over_rate_book", "mean")))
    return sub, per_cell, summary


def walk_cost_table(df: pd.DataFrame):
    """Exact (non-linear) walk-the-book cost of each marketable order.

    Two things come out of this. First, an audit of the linear lambda_book fit:
    ``implied_lambda_walk`` inverts the same linear model
    (avg cost = p_touch + lambda q / 2) from the exact walk, so the two can be
    compared directly. Second, ``c_walk`` -- the model-light coherence ratio.
    Calibrating the linear model at the order's own size gives
    lambda(q) = 2 (avg_exec(q) - p_touch) / q, so

        c = q / q* = q lambda(q) / edge = 2 (avg_exec(q) - p_touch) / edge

    which drops lambda entirely: it is just the order's own realized average
    impact measured against the edge the agent claims to be harvesting. c_walk
    is exact for a linear ladder and needs no fit; it is undefined once the
    executable depth is exhausted (the walk then understates the cost).

    The edge here is ``edge_at_touch``, not the P_limit-based ``edge`` used by
    c_book: this model is anchored at the touch, and for a crossing limit the
    P_limit edge is only a lower bound. The two coincide for market orders.
    """
    sub = df[df["is_trade"] & df["marketability"].isin(["market", "marketable_limit"])
             & df["asks_json"].notna() & df["bids_json"].notna()].copy()
    if sub.empty:
        return None
    recs = []
    for idx, rec in zip(sub.index, sub.to_dict("records")):
        is_buy = rec["side"] == "buy"
        try:
            levels = json.loads(rec["asks_json"] if is_buy else rec["bids_json"])
        except (TypeError, ValueError):
            continue
        # A marketable limit cannot execute through its own limit price.
        cap = rec["price"] if rec["marketability"] == "marketable_limit" else None
        avg, filled, exhausted = walk_book([tuple(lv) for lv in levels],
                                           rec["quantity"], price_cap=cap,
                                           is_buy=is_buy)
        touch = rec["seen_best_ask"] if is_buy else rec["seen_best_bid"]
        if not np.isfinite(avg) or not np.isfinite(touch) or filled <= 0:
            continue
        impact = (avg - touch) if is_buy else (touch - avg)
        edge = rec.get("edge_at_touch")
        c_walk = (2.0 * impact / edge
                  if (edge is not None and np.isfinite(edge) and edge > 0
                      and not exhausted) else np.nan)
        recs.append({
            "row_idx": idx,
            "cell_id": rec["cell_id"], "agent_id": rec["agent_id"],
            "round": rec["round"], "agent_type": rec["agent_type"],
            "prompt_family": rec.get("prompt_family"), "model": rec.get("model"),
            "side": rec["side"], "marketability": rec["marketability"],
            "quantity": rec["quantity"], "edge_at_touch": edge,
            "edge_at_limit": rec.get("edge"),
            "filled_visible": filled, "depth_exhausted": exhausted,
            "avg_exec_price": avg, "touch": touch, "walk_impact": impact,
            "implied_lambda_walk": 2.0 * impact / filled if filled > 0 else np.nan,
            "lambda_book": rec["lambda_book"], "c_walk": c_walk,
        })
    return pd.DataFrame(recs) if recs else None


def lambda_gaps_table(books: pd.DataFrame, lambda_realized_by_cell: pd.DataFrame,
                      shading: pd.DataFrame):
    """The two headline gaps, per cell where both sides are defined.

    lambda_book here is a property of the BOOK, not of any one order, so it is
    aggregated over the raw snapshots (both ladders pooled) rather than over
    the side-matched per-order column -- which would weight the buy/sell
    ladders by how often agents happened to trade each side.
    """
    if books is None or not len(books):
        return None
    long = pd.concat([
        books[["cell_id", "prompt_family", "model", "lambda_book_buy"]]
        .rename(columns={"lambda_book_buy": "lambda_book"}),
        books[["cell_id", "prompt_family", "model", "lambda_book_sell"]]
        .rename(columns={"lambda_book_sell": "lambda_book"}),
    ], ignore_index=True)
    long = long[long["lambda_book"].notna()]
    if long.empty:
        return None
    book = (long.groupby(["cell_id", "prompt_family", "model"], as_index=False)
                .agg(lambda_book_mean=("lambda_book", "mean"),
                     lambda_book_median=("lambda_book", "median"),
                     n_ladders=("lambda_book", "size")))
    if lambda_realized_by_cell is not None and len(lambda_realized_by_cell):
        book = book.merge(lambda_realized_by_cell[["cell_id", "lambda_hat_realized"]],
                          on="cell_id", how="left")
    else:
        book["lambda_hat_realized"] = np.nan
    if shading is not None and len(shading):
        pooled = shading[shading["group"] == "pooled"]
        lam_p = pooled["lambda_perceived_joint"].iloc[0] if len(pooled) else np.nan
    else:
        lam_p = np.nan
    book["lambda_perceived_joint"] = lam_p
    book["gap_perceived_minus_book"] = book["lambda_perceived_joint"] - book["lambda_book_median"]
    book["gap_book_minus_realized"] = book["lambda_book_median"] - book["lambda_hat_realized"]
    return book


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_lambda_comparison(gaps: pd.DataFrame, out_path: Path):
    """Cross-cell spread of the two per-cell lambdas, with the pooled
    lambda_perceived drawn as a line -- it is one pooled regression, not a
    distribution, and boxing a repeated constant would imply otherwise."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    series = [("lambda_book_median", "lambda_book\n(seen book)"),
              ("lambda_hat_realized", "lambda_realized\n(flow)")]
    data, labels = [], []
    for col, label in series:
        vals = (pd.to_numeric(gaps[col], errors="coerce").dropna().to_numpy()
                if col in gaps.columns else np.array([]))
        if len(vals):
            data.append(vals)
            labels.append(label)
    if not data:
        ax.set_axis_off()
        ax.set_title("No lambda estimates available")
    else:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        lam_p = pd.to_numeric(gaps.get("lambda_perceived_joint"),
                              errors="coerce").dropna()
        if len(lam_p):
            ax.axhline(lam_p.iloc[0], color="crimson", ls="--", lw=1.2,
                       label="lambda_perceived (pooled shading, joint)")
            ax.legend(fontsize=8)
        # The three lambdas routinely differ by orders of magnitude -- that gap
        # is the result -- so a linear axis flattens two of them into the floor.
        if all((v > 0).all() for v in data):
            ax.set_yscale("log")
        ax.set_ylabel("lambda ($ per share, log scale)")
        ax.set_title("Price-impact coefficients: seen book vs realized vs perceived")
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_coherence_ratio(scored: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    groups, data = [], []
    for label in ["market", "marketable_limit", "passive_limit"]:
        vals = scored.loc[scored["marketability"] == label, "c_book"].dropna()
        vals = vals[vals > 0]
        if len(vals):
            groups.append(label)
            data.append(np.log10(vals.to_numpy()))
    if not data:
        ax.set_axis_off()
        ax.set_title("No scoreable coherence ratios (lambda_book missing)")
    else:
        ax.boxplot(data, tick_labels=groups, showfliers=False)
        ax.axhline(0, color="crimson", ls="--", lw=1, label="c = 1 (coherent size)")
        ax.set_ylabel("log10 c  =  log10 (q submitted / q*)")
        ax.set_title("Size coherence against the agent's own seen book")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_concession_vs_size(per_order: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.scatter(per_order["log10_qty"], per_order["concession_pct"],
               s=10, alpha=0.35, color="#4878a8")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("log10 order size (shares)")
    ax.set_ylabel("price concession vs own next-price forecast (%)")
    ax.set_title("Perceived impact: concession demanded rises with size?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_root", help="Sweep dir containing aggregated/ panels.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir. Default: <sweep_root>/aggregated/impact_estimates/")
    ap.add_argument("--cluster", default="cell_id",
                    help="Column to cluster regression SEs on (default cell_id = "
                         "independent run).")
    ap.add_argument("--rebuild-books", action="store_true",
                    help="Re-parse rendered_prompts.jsonl instead of using the "
                         "cached book_snapshots_panel.csv.")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = (Path(args.out_dir) if args.out_dir
               else sweep_root / "aggregated" / "impact_estimates")
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions, market, orders = load_panels(sweep_root)
    books = load_book_snapshots(sweep_root, args.rebuild_books)
    df = build_decision_frame(decisions, market, books)
    cluster_col = args.cluster if args.cluster in df.columns else "cell_id"
    if cluster_col != args.cluster:
        print(f"[warn] cluster column '{args.cluster}' not found; using 'cell_id'.")

    n_books = int(df["lambda_book"].notna().sum())
    print(f"Impact estimators over {len(df):,} order rows across "
          f"{df['cell_id'].nunique()} cells (clustered on '{cluster_col}'); "
          f"{n_books:,} rows carry a lambda_book from the seen order book.\n")

    # 1. lambda_hat_realized
    flow = build_flow_frame(orders, market)
    lam_realized_cell = None
    if flow is not None:
        pooled, lam_realized_cell = lambda_realized_tables(flow, cluster_col)
        flow.to_csv(out_dir / "order_flow_by_round.csv", index=False)
        pooled.to_csv(out_dir / "lambda_realized_regression.csv", index=False)
        lam_realized_cell.to_csv(out_dir / "lambda_realized_by_cell.csv", index=False)
        print("== lambda_hat_realized (price change on net submitted flow) ==")
        with pd.option_context("display.width", 170, "display.max_rows", None):
            print(pooled[pooled["group"] == "pooled"]
                  [["regression", "n_obs", "n_clusters", "lambda_hat", "se", "t"]]
                  .to_string(index=False))
    else:
        print("== lambda_hat_realized: no usable order flow "
              "(missing order_data_panel.csv, no orders, or single-round cells) ==")

    # 2. lambda_book (+ the exact walk-the-book costs it is audited against).
    # The walk is computable whenever a snapshot exists, even where the ladder
    # is too thin to fit a slope, so it is written independently of n_books.
    walk = walk_cost_table(df)
    if walk is not None:
        walk.to_csv(out_dir / "walk_the_book_costs.csv", index=False)
    if n_books:
        book_cols = ["cell_id", "agent_id", "round", "agent_type", "side",
                     "marketability", "lambda_book", "lambda_book_buy",
                     "lambda_book_sell", "ask_levels", "bid_levels",
                     "ask_depth", "bid_depth"]
        df[book_cols].to_csv(out_dir / "lambda_book_by_decision.csv", index=False)
        summ = (df[df["lambda_book"].notna()]
                .groupby(SUMMARY_GROUP, as_index=False)
                .agg(n=("lambda_book", "size"),
                     lambda_book_mean=("lambda_book", "mean"),
                     lambda_book_median=("lambda_book", "median"),
                     lambda_book_std=("lambda_book", "std")))
        summ.to_csv(out_dir / "lambda_book_summary.csv", index=False)
        print("\n== lambda_book (walk-the-book slope of the seen ladder) ==")
        print(f"   median over the side the order takes: "
              f"{df.loc[df['is_trade'], 'lambda_book'].median():.6f} $/share")
        if walk is not None:
            n_exh = int(walk["depth_exhausted"].sum())
            print(f"   exact walk-the-book cost computed for {len(walk):,} "
                  f"marketable orders ({n_exh} exhausted the executable depth)")
            both = walk[walk["implied_lambda_walk"].notna() & walk["lambda_book"].notna()]
            if len(both):
                inside = both["walk_impact"] <= 0
                print(f"   linear fit vs exact walk: median lambda_book="
                      f"{both['lambda_book'].median():.6f} vs median "
                      f"implied_lambda_walk={both['implied_lambda_walk'].median():.6f}")
                print(f"   ({inside.mean():.0%} of these orders fit inside the top "
                      f"level, so their exact impact is exactly zero; over the "
                      f"{int((~inside).sum())} that walked past it the median "
                      f"implied_lambda_walk is "
                      f"{both.loc[~inside, 'implied_lambda_walk'].median():.6f})")
    elif len(books):
        print(f"\n== lambda_book: {len(books):,} book snapshots parsed, but no "
              "ladder carried two priced levels on the side an order took -- an "
              "empty or single-level book has no slope, so lambda_book is "
              "undefined rather than zero. ==")
    else:
        print("\n== lambda_book: no parsed book snapshots. Runs recorded before "
              "rendered-prompt logging cannot supply the agent's seen book; "
              "re-run the sweep to populate it. ==")

    # 3. lambda_perceived
    shading = size_shading_table(df, cluster_col)
    if shading is not None:
        shading.to_csv(out_dir / "lambda_perceived_size_shading.csv", index=False)
        print("\n== lambda_perceived, channel (a): size shading q ~ edge ==")
        print("   (joint with risk aversion -- an upper bound on perceived impact)")
        with pd.option_context("display.width", 170):
            print(shading[shading["group"] == "pooled"]
                  [["n_obs", "n_clusters", "slope", "se", "t",
                    "lambda_perceived_joint"]].to_string(index=False))
    else:
        print("\n== lambda_perceived (size shading): no positive-edge orders to score ==")

    placement, placement_cell, placement_orders = placement_distance_table(df, cluster_col)
    if placement is not None:
        placement.to_csv(out_dir / "lambda_perceived_placement.csv", index=False)
        placement_cell.to_csv(out_dir / "placement_distance_by_cell.csv", index=False)
        print("\n== lambda_perceived, channel (b): concession vs own forecast, by size ==")
        with pd.option_context("display.width", 170):
            print(placement[placement["group"] == "pooled"]
                  [["n_obs", "n_clusters", "slope", "se", "t"]].to_string(index=False))

    mshare, mshare_dec = marketable_share_table(df, cluster_col)
    if mshare is not None:
        mshare.to_csv(out_dir / "lambda_perceived_marketable_share.csv", index=False)
        mshare_dec.to_csv(out_dir / "marketable_share_by_decision.csv", index=False)
        print("\n== lambda_perceived, channel (c): marketable share vs decision size ==")
        print("   (negative slope = goes passive when trading big = perceives impact)")
        with pd.option_context("display.width", 170):
            print(mshare[mshare["group"] == "pooled"]
                  [["n_obs", "n_clusters", "slope", "se", "t"]].to_string(index=False))

    # 4. Size coherence c
    scored, coh_cell, coh_summary = size_coherence_tables(df, lam_realized_cell, walk)
    if coh_cell is not None:
        coh_cell.to_csv(out_dir / "size_coherence_by_cell.csv", index=False)
        coh_summary.to_csv(out_dir / "size_coherence_summary.csv", index=False)
        print("\n== Size coherence c = q / q*, q* = edge / lambda ==")
        for label in ["market", "marketable_limit", "passive_limit"]:
            vals = scored.loc[scored["marketability"] == label, "c_book"].dropna()
            if len(vals):
                print(f"   {label:17s} n={len(vals):5d}  median c_book="
                      f"{vals.median():8.3f}  share c<{C_UNDER}="
                      f"{(vals < C_UNDER).mean():.1%}  share c>{C_OVER}="
                      f"{(vals > C_OVER).mean():.1%}")
        c_walk = scored["c_walk"].dropna()
        if len(c_walk):
            print(f"   fit-free (exact walk): n={len(c_walk)}, median c_walk="
                  f"{c_walk.median():.3f}")
        c_real = scored["c_realized"].dropna()
        if len(c_real):
            print(f"   vs lambda_realized: n={len(c_real)}, median c="
                  f"{c_real.median():.3f}")
    elif scored is not None:
        print("\n== Size coherence: edges available but no usable lambda "
              "(no seen book and no realized-flow slope) ==")
    else:
        print("\n== Size coherence: no positive-edge orders to score ==")

    # 5. Gaps + figures
    gaps = lambda_gaps_table(books, lam_realized_cell, shading)
    if gaps is not None:
        gaps.to_csv(out_dir / "lambda_gaps.csv", index=False)
        print("\n== lambda gaps ==")
        for col, name in [("gap_perceived_minus_book",
                           "perceived - book (does the agent use the book it is shown?)"),
                          ("gap_book_minus_realized",
                           "book - realized (is the visible book informative?)")]:
            vals = gaps[col].dropna()
            if len(vals):
                print(f"   {name}: mean {vals.mean():+.6f} over {len(vals)} cells")
        plot_lambda_comparison(gaps, out_dir / "lambda_comparison.png")
    if scored is not None and scored["c_book"].notna().any():
        plot_coherence_ratio(scored, out_dir / "coherence_ratio.png")
    if placement_orders is not None and len(placement_orders):
        plot_concession_vs_size(placement_orders, out_dir / "concession_vs_size.png")

    print(f"\nTables + figures written to {out_dir}")


if __name__ == "__main__":
    main()
