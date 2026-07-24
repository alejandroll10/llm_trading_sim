"""
Phase-1 belief-action estimators (issue #111).

Consumes the aggregated sweep panels produced by ``aggregate_sweep.py``:

    <sweep_root>/aggregated/structured_decisions_panel.csv
    <sweep_root>/aggregated/market_data_panel.csv

and produces the assumption-free / lambda-free belief-action estimators, each
reported as a distribution across the seed x temperature grid with the
clustering columns (``prompt_family``, ``model``) preserved, following the
a1_estimators.py conventions:

  1. Price coherence (limit orders only, assumption-free): the violation
     margin max(0, P_limit - V)/V for buys and max(0, V - P_limit)/V for
     sells, where V is the stated valuation from the same TradeDecision.
     Bidding above (asking below) your own stated valuation is incoherent
     under any impact model. Reported separately for passive vs marketable
     limit orders (classified against the prior round's best bid/ask).
  2. Direction coherence: market orders judged against the agent's own
     valuation and the prior round's touch (a market buy with V below the
     ask pays more than the agent's own stated value); plus the
     self-crossing flag (simultaneous buy and sell limits that cross each
     other within one decision are internally incoherent). Sell limits
     above V by a bull are liquidity provision, NOT incoherence -- they are
     never flagged here; the symmetric limit-order violations live entirely
     in the price-coherence layer.
  3. Population-forecast skill: multi-horizon stated forecasts
     (price_prediction_t / _t1 / _t2) scored against the realized price
     path, net of the agent's own stated valuation (Mincer-Zarnowitz-style
     regression realized ~ forecast + valuation with cluster-robust SEs);
     internal consistency via (a) accuracy by horizon and (b) the revision
     between overlapping forecasts of the same target round.
  4. Belief-message divergence: lexicon sentiment of the outgoing message
     (services.sentiment_scorer) against the sign/size of the agent's own
     stated edge (V - p)/p -- do agents talk their book, stay quiet, or
     talk against it?

Deliberately excluded from phase 1 (see issue #111): the size-coherence
ratio c = q/q*, perceived price impact, and every lambda (book / realized /
perceived) -- those need an impact model and land in phase 2.

Round-timing convention: structured_decisions.csv logs the raw decision
round r (0-indexed) while market_data.csv logs round_number + 1, i.e. its
row labeled k is the market state AFTER round k-1's matching. Therefore the
state a round-r decision-maker saw is market row r, and the realized price
for forecast horizon k in {0, 1, 2} is market row r + 1 + k. Round-0
decisions have no recorded pre-round state and are excluded from the
book-dependent pieces.

Usage
-----
    python src/analysis/belief_action_estimators.py logs/sweeps/<sweep_name>
    python src/analysis/belief_action_estimators.py logs/sweeps/<sweep_name> --out-dir /some/where
    python src/analysis/belief_action_estimators.py logs/sweeps/<sweep_name> --cluster prompt_family
"""

import argparse
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
from services.sentiment_scorer import score_sentiment  # noqa: E402


# Sentiment cutoffs mirror services.sentiment_scorer.sentiment_label.
SENTIMENT_CUTOFF = 0.25
# Grouping used for all distribution summaries (a1 convention + agent_type,
# since coherence is a per-persona property).
SUMMARY_GROUP = ["agent_type", "prompt_family", "model"]
CELL_COLS = ["prompt_family", "model", "temperature", "seed", "cell_id"]
HORIZONS = [("t", 0), ("t1", 1), ("t2", 2)]


# --------------------------------------------------------------------------- #
# Loading & shaping
# --------------------------------------------------------------------------- #
def load_panels(sweep_root: Path):
    """Load the aggregated decisions and market panels; return (decisions, market)."""
    agg = sweep_root / "aggregated"
    dec_path = agg / "structured_decisions_panel.csv"
    mkt_path = agg / "market_data_panel.csv"
    if not dec_path.exists():
        sys.exit(f"Error: {dec_path} not found. Run aggregate_sweep.py on the sweep first.")
    decisions = pd.read_csv(dec_path)
    market = pd.read_csv(mkt_path) if mkt_path.exists() else None
    return decisions, market


def normalize_order_type(s: pd.Series) -> pd.Series:
    """Map order_type spellings ('OrderType.LIMIT', 'limit', 'none') to
    {'limit', 'market', 'none'}."""
    return (s.astype(str).str.lower()
             .str.replace("ordertype.", "", regex=False)
             .str.strip())


def market_per_round(market: pd.DataFrame) -> pd.DataFrame:
    """Collapse the market panel to one row per (cell_id, round).

    Single stock expected; if multiple stock_ids exist, average within a
    round (a1 convention -- order rows carry no stock_id, so per-stock
    matching is not possible from this panel).
    """
    m = market.copy()
    for col in ["price", "fundamental_price", "best_bid", "best_ask"]:
        m[col] = pd.to_numeric(m.get(col), errors="coerce")
    return (m.groupby(["cell_id", "round"], as_index=False)
             .agg(obs_price=("price", "mean"),
                  true_fv=("fundamental_price", "mean"),
                  best_bid=("best_bid", "mean"),
                  best_ask=("best_ask", "mean")))


def build_analysis_frame(decisions: pd.DataFrame, market: pd.DataFrame):
    """Normalize the decisions panel and attach same-round and prior-round
    market context.

    Returns a per-order frame with columns: side ('buy'/'sell'/'hold'),
    otype ('limit'/'market'/'none'), quantity, price (= P_limit for limit
    rows), valuation, price_prediction_t/_t1/_t2, seen_price / seen_best_bid
    / seen_best_ask / true_fv (market row r = the state the round-r
    decision-maker saw; see module docstring for the +1 offset),
    marketability, plus the preserved cell metadata.
    """
    df = decisions.copy()
    df["otype"] = normalize_order_type(df["order_type"])
    df["side"] = df["decision"].astype(str).str.lower().str.strip()

    numeric = ["quantity", "price", "valuation",
               "price_prediction_t", "price_prediction_t1", "price_prediction_t2"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if market is not None and len(market):
        # Market row r is the state after round r-1's matching = what the
        # round-r decision-maker saw, so this same-key merge attaches the
        # agent's information set, not the round's outcome.
        seen = market_per_round(market).rename(columns={
            "obs_price": "seen_price", "best_bid": "seen_best_bid",
            "best_ask": "seen_best_ask"})
        df = df.merge(seen, on=["cell_id", "round"], how="left")
    else:
        for col in ["seen_price", "true_fv", "seen_best_bid", "seen_best_ask"]:
            df[col] = np.nan

    # Passive vs marketable, judged against the touch the agent saw.
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

    # A usable stated valuation is required by every phase-1 estimator except
    # raw forecast accuracy; keep a flag instead of dropping rows.
    df["has_valuation"] = df["valuation"].notna() & (df["valuation"] > 0)
    n_bad = int((~df["has_valuation"]).sum())
    if n_bad:
        print(f"[warn] {n_bad} of {len(df)} rows lack a positive stated valuation "
              f"(fallback holds / parse failures); excluded from coherence estimators.")
    return df


def beliefs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, agent, round) decision -- belief fields repeat
    across the order rows of a multi-order decision."""
    return df.drop_duplicates(subset=["cell_id", "agent_id", "round"]).copy()


def summarize_cells(per_cell: pd.DataFrame, value_cols) -> pd.DataFrame:
    """Distribution across cells (seeds x temperatures) of per-cell means,
    grouped by agent_type x prompt_family x model."""
    aggs = {}
    for col in value_cols:
        aggs[f"{col}_mean"] = (col, "mean")
        aggs[f"{col}_std"] = (col, "std")
        aggs[f"{col}_median"] = (col, "median")
        aggs[f"{col}_p90"] = (col, lambda s: s.quantile(0.90))
    aggs["n_cells"] = (value_cols[0], "size")
    return per_cell.groupby(SUMMARY_GROUP, as_index=False).agg(**aggs)


# --------------------------------------------------------------------------- #
# Estimator 1: price coherence (limit orders, assumption-free)
# --------------------------------------------------------------------------- #
def price_coherence_tables(df: pd.DataFrame):
    """Violation margin of limit prices against the same-response valuation.

    Buy limit above own V (sell limit below own V) is incoherent under any
    impact model. Margin is relative: max(0, P - V)/V for buys,
    max(0, V - P)/V for sells.
    """
    lim = df[(df["otype"] == "limit") & df["has_valuation"]
             & df["price"].notna() & (df["price"] > 0)].copy()
    if lim.empty:
        return None, None

    buy = lim["side"] == "buy"
    lim["violation_margin"] = np.where(
        buy,
        np.maximum(0.0, lim["price"] - lim["valuation"]) / lim["valuation"],
        np.maximum(0.0, lim["valuation"] - lim["price"]) / lim["valuation"])
    lim["violates"] = lim["violation_margin"] > 0

    grp = ["agent_type", "marketability"] + CELL_COLS
    per_cell = (lim.groupby(grp, as_index=False)
                   .agg(n_limit_orders=("violates", "size"),
                        violation_rate=("violates", "mean"),
                        mean_margin=("violation_margin", "mean"),
                        mean_margin_if_violating=(
                            "violation_margin",
                            # NaN, not 0: a cell with no violations has no
                            # conditional margin, and 0 would dilute the
                            # cross-cell mean of this conditional statistic.
                            lambda s: s[s > 0].mean() if (s > 0).any() else np.nan)))

    aggs = dict(n_cells=("violation_rate", "size"),
                n_orders=("n_limit_orders", "sum"),
                violation_rate_mean=("violation_rate", "mean"),
                violation_rate_std=("violation_rate", "std"),
                violation_rate_median=("violation_rate", "median"),
                margin_mean=("mean_margin", "mean"),
                margin_if_violating_mean=("mean_margin_if_violating", "mean"))
    summary = (per_cell.groupby(SUMMARY_GROUP + ["marketability"], as_index=False)
                       .agg(**aggs))
    return per_cell, summary


# --------------------------------------------------------------------------- #
# Estimator 2: direction coherence (market orders + self-crossing)
# --------------------------------------------------------------------------- #
def direction_coherence_tables(df: pd.DataFrame):
    """Market orders against own valuation and the touch the agent saw, plus
    the within-decision self-crossing flag.

    A market buy is direction-incoherent when the agent's own V is below the
    price it expects to pay (the best ask it saw; fallback last price); a
    market sell symmetric against the best bid. Limit orders are never
    flagged here: their violations are the price-coherence layer, and a
    passive sell above V is liquidity provision.
    """
    mkt = df[(df["otype"] == "market") & df["has_valuation"]].copy()
    if not mkt.empty:
        ref = np.where(mkt["side"] == "buy",
                       mkt["seen_best_ask"].fillna(mkt["seen_price"]),
                       mkt["seen_best_bid"].fillna(mkt["seen_price"]))
        mkt["ref_price"] = ref
        mkt = mkt[mkt["ref_price"].notna() & (mkt["ref_price"] > 0)]
    if not mkt.empty:
        buy = mkt["side"] == "buy"
        mkt["incoherence_margin"] = np.where(
            buy,
            np.maximum(0.0, mkt["ref_price"] - mkt["valuation"]) / mkt["valuation"],
            np.maximum(0.0, mkt["valuation"] - mkt["ref_price"]) / mkt["valuation"])
        mkt["incoherent"] = mkt["incoherence_margin"] > 0

        grp = ["agent_type"] + CELL_COLS
        per_cell_mkt = (mkt.groupby(grp, as_index=False)
                           .agg(n_market_orders=("incoherent", "size"),
                                incoherence_rate=("incoherent", "mean"),
                                mean_incoherence_margin=("incoherence_margin", "mean")))
        summary_mkt = summarize_cells(per_cell_mkt,
                                      ["incoherence_rate", "mean_incoherence_margin"])
    else:
        per_cell_mkt, summary_mkt = None, None

    # Self-crossing: within one (cell, agent, round) decision, a buy limit at
    # or above a sell limit from the same response is internally incoherent
    # regardless of beliefs.
    lim = df[(df["otype"] == "limit") & df["price"].notna()]
    cross = None
    if not lim.empty:
        pivot = (lim.groupby(["cell_id", "agent_id", "round"] + SUMMARY_GROUP[1:]
                             + ["agent_type", "temperature", "seed"])
                    .apply(lambda g: pd.Series({
                        "max_buy_limit": g.loc[g["side"] == "buy", "price"].max(),
                        "min_sell_limit": g.loc[g["side"] == "sell", "price"].min()}),
                        include_groups=False)
                    .reset_index())
        pivot["self_crossing"] = (pivot["max_buy_limit"].notna()
                                  & pivot["min_sell_limit"].notna()
                                  & (pivot["max_buy_limit"] >= pivot["min_sell_limit"]))
        grp = ["agent_type"] + CELL_COLS
        cross = (pivot.groupby([c for c in grp if c in pivot.columns], as_index=False)
                      .agg(n_decisions=("self_crossing", "size"),
                           self_crossing_rate=("self_crossing", "mean")))
    return per_cell_mkt, summary_mkt, cross


# --------------------------------------------------------------------------- #
# Estimator 3: population-forecast skill (multi-horizon)
# --------------------------------------------------------------------------- #
def forecast_frames(bel: pd.DataFrame, market: pd.DataFrame):
    """Long frame: one row per (decision, horizon) with forecast + realized."""
    if market is None or not len(market):
        return None
    per_round = market_per_round(market)[["cell_id", "round", "obs_price"]]

    frames = []
    for label, k in HORIZONS:
        col = f"price_prediction_{label}"
        sub = bel[bel[col].notna() & (bel[col] > 0)].copy()
        if sub.empty:
            continue
        sub["horizon"] = label
        # Market row r+1 is the outcome of decision round r ('THIS round').
        sub["target_round"] = sub["round"] + 1 + k
        realized = per_round.rename(columns={"round": "target_round",
                                             "obs_price": "realized"})
        sub = sub.merge(realized, on=["cell_id", "target_round"], how="inner")
        sub["forecast"] = sub[col]
        frames.append(sub[["cell_id", "agent_id", "agent_type", "round",
                           "target_round", "horizon", "forecast", "realized",
                           "valuation", "seen_price",
                           "prompt_family", "model", "temperature", "seed"]])
    if not frames:
        return None
    long = pd.concat(frames, ignore_index=True)
    long["error"] = long["forecast"] - long["realized"]
    long["abs_error"] = long["error"].abs()
    long["ape"] = long["abs_error"] / long["realized"]
    return long


def forecast_skill_tables(long: pd.DataFrame, cluster_col: str):
    """Accuracy by horizon + net-of-valuation Mincer-Zarnowitz regression.

    The regression realized ~ forecast + valuation asks whether the forecast
    predicts the realized price beyond what the agent's own stated valuation
    already implies -- forecast slope > 0 with valuation held fixed is skill
    that is not just a restatement of the value estimate.
    """
    grp = ["horizon", "agent_type"] + CELL_COLS
    per_cell = (long.groupby(grp, as_index=False)
                    .agg(n_forecasts=("abs_error", "size"),
                         mae=("abs_error", "mean"),
                         mape=("ape", "mean"),
                         bias=("error", "mean")))
    summary = (per_cell.groupby(["horizon"] + SUMMARY_GROUP, as_index=False)
                       .agg(n_cells=("mae", "size"),
                            mae_mean=("mae", "mean"),
                            mae_std=("mae", "std"),
                            mape_mean=("mape", "mean"),
                            bias_mean=("bias", "mean")))

    # Naive no-change benchmark: forecast = the price the agent last saw.
    naive = long[long["seen_price"].notna()].copy()
    if len(naive):
        naive["naive_abs_error"] = (naive["seen_price"] - naive["realized"]).abs()
        bench = (naive.groupby("horizon", as_index=False)
                      .agg(n=("naive_abs_error", "size"),
                           model_mae=("abs_error", "mean"),
                           naive_mae=("naive_abs_error", "mean")))
        # Naive MAE of 0 (price never moved) makes skill undefined, not 0.
        bench["skill_vs_naive"] = np.where(
            bench["naive_mae"] > 0,
            1.0 - bench["model_mae"] / bench["naive_mae"], np.nan)
    else:
        bench = None

    rows = []
    for horizon in [h for h, _ in HORIZONS]:
        sub = long[(long["horizon"] == horizon) & long["valuation"].notna()
                   & (long["valuation"] > 0)]
        if len(sub) < 4 or sub["forecast"].nunique() < 2:
            rows.append({"horizon": horizon, "n_obs": len(sub), "n_clusters": 0,
                         "forecast_slope": np.nan, "forecast_se": np.nan,
                         "forecast_t": np.nan, "valuation_slope": np.nan,
                         "valuation_se": np.nan})
            continue
        X = np.column_stack([np.ones(len(sub)),
                             sub["forecast"].to_numpy(dtype=float),
                             sub["valuation"].to_numpy(dtype=float)])
        beta, se, n, G = ols_cluster_robust(sub["realized"].to_numpy(), X,
                                            sub[cluster_col].to_numpy())
        if G < 2:
            print(f"[warn] forecast-skill horizon={horizon}: only {G} cluster on "
                  f"'{cluster_col}'; SE/t suppressed.")
            se = np.full(3, np.nan)
        rows.append({"horizon": horizon, "n_obs": n, "n_clusters": G,
                     "forecast_slope": beta[1], "forecast_se": se[1],
                     "forecast_t": beta[1] / se[1] if se[1] > 0 else np.nan,
                     "valuation_slope": beta[2], "valuation_se": se[2]})
    mz = pd.DataFrame(rows)
    return per_cell, summary, bench, mz


def forecast_revision_table(bel: pd.DataFrame):
    """Internal consistency: revision between overlapping forecasts of the
    same target round.

    The t1 forecast made at round r-1 and the t forecast made at round r both
    target round r; large mean revisions with no accuracy gain indicate
    unstable stated beliefs rather than information arrival.
    """
    keys = ["cell_id", "agent_id"]
    cur = bel[["cell_id", "agent_id", "agent_type", "round",
               "price_prediction_t", "price_prediction_t1", "price_prediction_t2"]].copy()
    prev = cur[keys + ["round", "price_prediction_t1", "price_prediction_t2"]].copy()
    prev["round"] = prev["round"] + 1
    prev = prev.rename(columns={"price_prediction_t1": "lag1_t1",
                                "price_prediction_t2": "lag1_t2"})
    merged = cur.merge(prev, on=keys + ["round"], how="inner")

    out = []
    for cols, label in [(("price_prediction_t", "lag1_t1"), "t_vs_lag_t1"),
                        (("price_prediction_t1", "lag1_t2"), "t1_vs_lag_t2")]:
        a, b = cols
        sub = merged[merged[a].notna() & merged[b].notna()
                     & (merged[a] > 0) & (merged[b] > 0)].copy()
        if sub.empty:
            continue
        sub["revision"] = sub[a] - sub[b]
        sub["abs_revision_pct"] = (sub["revision"].abs() / sub[b])
        g = (sub.groupby("agent_type", as_index=False)
                .agg(pair=("revision", lambda s: label),
                     n=("revision", "size"),
                     mean_revision=("revision", "mean"),
                     mean_abs_revision_pct=("abs_revision_pct", "mean")))
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else None


# --------------------------------------------------------------------------- #
# Estimator 4: belief-message divergence
# --------------------------------------------------------------------------- #
def divergence_tables(bel: pd.DataFrame):
    """Sentiment of the outgoing message vs the sign of the agent's stated
    edge (V - p)/p, using the price the agent saw when writing the message.

    'Misaligned' = bullish message (sentiment >= +0.25) while the agent's own
    valuation is below the price, or bearish while above -- i.e. talking
    against one's own book.
    """
    if "post_message" not in bel.columns:
        return None, None
    msg = bel[bel["post_message"].notna()
              & (bel["post_message"].astype(str).str.strip() != "")
              & bel["has_valuation"]
              & bel["seen_price"].notna() & (bel["seen_price"] > 0)].copy()
    if msg.empty:
        return None, None
    msg["ref_price"] = msg["seen_price"]

    msg["sentiment"] = msg["post_message"].astype(str).map(score_sentiment)
    msg["stated_edge"] = (msg["valuation"] - msg["ref_price"]) / msg["ref_price"]
    bullish = msg["sentiment"] >= SENTIMENT_CUTOFF
    bearish = msg["sentiment"] <= -SENTIMENT_CUTOFF
    msg["misaligned"] = (bullish & (msg["stated_edge"] < 0)) | \
                        (bearish & (msg["stated_edge"] > 0))
    msg["aligned"] = (bullish & (msg["stated_edge"] > 0)) | \
                     (bearish & (msg["stated_edge"] < 0))

    grp = ["agent_type"] + CELL_COLS
    per_cell = (msg.groupby(grp, as_index=False)
                   .agg(n_messages=("misaligned", "size"),
                        misaligned_rate=("misaligned", "mean"),
                        aligned_rate=("aligned", "mean"),
                        neutral_rate=("sentiment",
                                      lambda s: (s.abs() < SENTIMENT_CUTOFF).mean()),
                        sentiment_edge_corr=("sentiment", lambda s: np.nan)))
    # Correlation needs both columns; recompute per cell properly.
    corr = (msg.groupby(grp)
               .apply(lambda g: g["sentiment"].corr(g["stated_edge"])
                      if g["sentiment"].nunique() > 1 and g["stated_edge"].nunique() > 1
                      else np.nan, include_groups=False)
               .rename("sentiment_edge_corr").reset_index())
    per_cell = per_cell.drop(columns="sentiment_edge_corr").merge(corr, on=grp)

    summary = (per_cell.groupby(SUMMARY_GROUP, as_index=False)
                       .agg(n_cells=("misaligned_rate", "size"),
                            n_messages=("n_messages", "sum"),
                            misaligned_rate_mean=("misaligned_rate", "mean"),
                            misaligned_rate_std=("misaligned_rate", "std"),
                            aligned_rate_mean=("aligned_rate", "mean"),
                            neutral_rate_mean=("neutral_rate", "mean"),
                            sentiment_edge_corr_mean=("sentiment_edge_corr", "mean")))
    return per_cell, summary


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_coherence_rates(price_cell, mkt_cell, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [(price_cell, "violation_rate", "limit price vs own valuation"),
              (mkt_cell, "incoherence_rate", "market-order direction vs own valuation")]
    for ax, (cell, col, title) in zip(axes, panels):
        if cell is None or cell.empty:
            ax.set_axis_off()
            ax.set_title(f"{title}\n(no data)", fontsize=9)
            continue
        agg = (cell.groupby("agent_type")[col]
                   .agg(["mean", "std"]).sort_values("mean").reset_index())
        ax.barh(agg["agent_type"], agg["mean"],
                xerr=agg["std"].fillna(0), color="#4878a8", alpha=0.85)
        ax.set_xlabel("incoherence rate (mean across cells)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(left=0)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Belief-action coherence violations by agent type", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_forecast_mae(per_cell: pd.DataFrame, bench, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    order = [h for h, _ in HORIZONS if h in set(per_cell["horizon"])]
    data = [per_cell[per_cell["horizon"] == h]["mae"].dropna().to_numpy()
            for h in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    if bench is not None:
        for i, h in enumerate(order):
            row = bench[bench["horizon"] == h]
            if len(row):
                ax.plot([i + 0.75, i + 1.25],
                        [row["naive_mae"].iloc[0]] * 2,
                        color="crimson", ls="--", lw=1.2,
                        label="naive (no-change)" if i == 0 else None)
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("MAE of stated price forecast (per cell)")
    ax.set_title("Population-forecast skill by horizon")
    if bench is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sentiment_vs_edge(per_msg_frame: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.scatter(per_msg_frame["stated_edge"], per_msg_frame["sentiment"],
               s=10, alpha=0.35, color="#4878a8")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.axhspan(SENTIMENT_CUTOFF, ax.get_ylim()[1], xmin=0, xmax=0.5,
               color="crimson", alpha=0.06)
    ax.set_xlabel("stated edge (V - p) / p")
    ax.set_ylabel("message sentiment")
    ax.set_title("Belief-message divergence: private edge vs public message")
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
                    help="Output dir for tables/figures. "
                         "Default: <sweep_root>/aggregated/belief_action_estimates/")
    ap.add_argument("--cluster", default="cell_id",
                    help="Column to cluster regression SEs on (default cell_id = "
                         "independent run). prompt_family / model are preserved "
                         "as columns regardless.")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = (Path(args.out_dir) if args.out_dir
               else sweep_root / "aggregated" / "belief_action_estimates")
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions, market = load_panels(sweep_root)
    df = build_analysis_frame(decisions, market)
    bel = beliefs_frame(df)
    cluster_col = args.cluster if args.cluster in df.columns else "cell_id"
    if cluster_col != args.cluster:
        print(f"[warn] cluster column '{args.cluster}' not found; using 'cell_id'.")

    print(f"Belief-action estimators over {len(df):,} order rows / "
          f"{len(bel):,} decisions across {df['cell_id'].nunique()} cells "
          f"(clustered on '{cluster_col}').\n")

    # 1. Price coherence
    price_cell, price_summary = price_coherence_tables(df)
    if price_cell is not None:
        price_cell.to_csv(out_dir / "price_coherence_by_cell.csv", index=False)
        price_summary.to_csv(out_dir / "price_coherence_summary.csv", index=False)
        overall = price_cell["violation_rate"].mean()
        print("== Price coherence (limit price vs own stated valuation) ==")
        print(f"   mean per-cell violation rate: {overall:.1%} "
              f"({int(price_cell['n_limit_orders'].sum())} limit orders)")
        show = price_summary[["agent_type", "marketability", "n_orders",
                              "violation_rate_mean", "margin_if_violating_mean"]]
        with pd.option_context("display.width", 160, "display.max_rows", None):
            print(show.to_string(index=False))
    else:
        print("== Price coherence: no scoreable limit orders in panel ==")

    # 2. Direction coherence
    mkt_cell, mkt_summary, cross = direction_coherence_tables(df)
    if mkt_cell is not None:
        mkt_cell.to_csv(out_dir / "direction_coherence_by_cell.csv", index=False)
        mkt_summary.to_csv(out_dir / "direction_coherence_summary.csv", index=False)
        print("\n== Direction coherence (market orders vs own valuation at the touch) ==")
        print(f"   mean per-cell incoherence rate: "
              f"{mkt_cell['incoherence_rate'].mean():.1%} "
              f"({int(mkt_cell['n_market_orders'].sum())} market orders)")
    else:
        print("\n== Direction coherence: no scoreable market orders (book/price "
              "context missing or round-0 only) ==")
    if cross is not None:
        cross.to_csv(out_dir / "self_crossing_by_cell.csv", index=False)
        rate = cross["self_crossing_rate"].mean()
        print(f"   self-crossing decisions (buy limit >= own sell limit): "
              f"mean per-cell rate {rate:.1%}")

    # 3. Population-forecast skill
    long = forecast_frames(bel, market)
    if long is not None:
        fc_cell, fc_summary, bench, mz = forecast_skill_tables(long, cluster_col)
        fc_cell.to_csv(out_dir / "forecast_skill_by_cell.csv", index=False)
        fc_summary.to_csv(out_dir / "forecast_skill_summary.csv", index=False)
        mz.to_csv(out_dir / "forecast_mz_regression.csv", index=False)
        print("\n== Population-forecast skill by horizon ==")
        by_h = (fc_cell.groupby("horizon")["mae"].mean()
                       .reindex([h for h, _ in HORIZONS]).dropna())
        print("   per-cell MAE by horizon: "
              + ", ".join(f"{h}={v:.3f}" for h, v in by_h.items()))
        if bench is not None:
            bench.to_csv(out_dir / "forecast_naive_benchmark.csv", index=False)
            for _, r in bench.iterrows():
                print(f"   horizon {r['horizon']}: skill vs naive no-change = "
                      f"{r['skill_vs_naive']:+.1%}")
        print("   Mincer-Zarnowitz (realized ~ forecast + valuation):")
        with pd.option_context("display.width", 160):
            print(mz.to_string(index=False))
        rev = forecast_revision_table(bel)
        if rev is not None:
            rev.to_csv(out_dir / "forecast_revisions.csv", index=False)
        plot_forecast_mae(fc_cell, bench, out_dir / "forecast_mae_by_horizon.png")
    else:
        print("\n== Forecast skill: no usable multi-horizon forecasts "
              "(pre-#111 schema or no market panel) ==")

    # 4. Belief-message divergence
    div_cell, div_summary = divergence_tables(bel)
    if div_cell is not None:
        div_cell.to_csv(out_dir / "belief_message_divergence_by_cell.csv", index=False)
        div_summary.to_csv(out_dir / "belief_message_divergence_summary.csv", index=False)
        print("\n== Belief-message divergence ==")
        print(f"   messages talking against own book: "
              f"{div_cell['misaligned_rate'].mean():.1%} of messages "
              f"(aligned: {div_cell['aligned_rate'].mean():.1%})")
        msg_frame = bel[bel["post_message"].notna()
                        & (bel["post_message"].astype(str).str.strip() != "")
                        & bel["has_valuation"]
                        & bel["seen_price"].notna() & (bel["seen_price"] > 0)].copy()
        if len(msg_frame):
            msg_frame["sentiment"] = msg_frame["post_message"].astype(str).map(score_sentiment)
            msg_frame["stated_edge"] = ((msg_frame["valuation"] - msg_frame["seen_price"])
                                        / msg_frame["seen_price"])
            plot_sentiment_vs_edge(msg_frame, out_dir / "sentiment_vs_edge.png")
    else:
        print("\n== Belief-message divergence: no messages in panel "
              "(SOCIAL feature off?) ==")

    plot_coherence_rates(price_cell, mkt_cell, out_dir / "coherence_rates.png")
    print(f"\nTables + figures written to {out_dir}")


if __name__ == "__main__":
    main()
