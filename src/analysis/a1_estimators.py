"""
Estimators for experiment A1: information ladder x price anchor (issue #103).

Consumes the aggregated sweep panels produced by ``aggregate_sweep.py``:

    <sweep_root>/aggregated/structured_decisions_panel.csv
    <sweep_root>/aggregated/market_data_panel.csv

and produces the three A1 estimators, each reported as a distribution across the
seed x temperature grid with the clustering columns (``prompt_family``, ``model``)
preserved:

  1. Valuation error vs the true fundamental value, by info mode x price anchor.
     The FULL row is the arithmetic control; the FULL - REALIZATIONS_ONLY gap is
     the "inference vs arithmetic" headline: does the agent infer value, or just execute supplied arithmetic?
  2. Anchoring regression: stated valuation on the contemporaneous observed price,
     coefficient by info mode (0 => tracks fundamentals, 1 => pure price anchoring),
     with cluster-robust standard errors. A second table regresses valuation on the
     exogenous initial-price anchor.
  3. Learning curve: mean absolute valuation error by round, showing convergence as
     dividend realizations accumulate (headline for REALIZATIONS_ONLY; all modes
     reported for contrast).

The cell key (the sweep ``variant`` column) encodes ``<mode>__<anchor>`` where
anchor in {0p5x, 1x, 2x}. True FV is read per round from the market panel's
``fundamental_price`` column (falls back to --fv, default 28.0, if absent).

Usage
-----
    python src/analysis/a1_estimators.py logs/sweeps/<sweep_name>
    python src/analysis/a1_estimators.py logs/sweeps/<sweep_name> --out-dir /some/where
    python src/analysis/a1_estimators.py logs/sweeps/<sweep_name> --cluster prompt_family
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Anchor label -> (multiple of FV). Mirrors sweeps/variants/a1_info_ladder.json.
ANCHOR_MULT = {"0p5x": 0.5, "1x": 1.0, "2x": 2.0}
# Canonical mode order for tables/plots (the information ladder, most -> least info).
MODE_ORDER = ["full", "process_only", "realizations_only", "average", "none"]
DEFAULT_FV = 28.0


# --------------------------------------------------------------------------- #
# Loading & shaping
# --------------------------------------------------------------------------- #
def parse_variant(variant: str):
    """Split a sweep variant name '<mode>__<anchor>' into (mode, anchor_label)."""
    if not isinstance(variant, str) or "__" not in variant:
        return (None, None)
    mode, _, anchor = variant.rpartition("__")
    return (mode, anchor)


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


def build_analysis_frame(decisions: pd.DataFrame, market: pd.DataFrame, fv_fallback: float):
    """Merge decisions with per-round market price/FV and derive A1 quantities.

    Returns a tidy per-decision frame with columns: mode, anchor, anchor_mult,
    initial_price, valuation, obs_price, true_fv, error, abs_error, pct_error, plus
    the preserved cell metadata (seed, temperature, model, prompt_family, cell_id,
    round, agent_id).
    """
    df = decisions.copy()

    # Parse the experimental axes out of the variant name.
    parsed = df["variant"].apply(parse_variant)
    df["mode"] = parsed.apply(lambda t: t[0])
    df["anchor"] = parsed.apply(lambda t: t[1])
    df["anchor_mult"] = df["anchor"].map(ANCHOR_MULT)

    # Rows whose variant does not parse into <mode>__<anchor> (e.g. a stray
    # 'baseline' cell) are dropped by the downstream dropna=True groupbys; surface
    # that loudly rather than letting them vanish silently.
    bad = df[df["mode"].isna() | df["anchor_mult"].isna()]
    if len(bad):
        print(f"[warn] {len(bad)} rows across {bad['variant'].nunique()} variant(s) "
              f"did not parse as <mode>__<anchor> and are excluded: "
              f"{sorted(bad['variant'].dropna().unique())[:5]}")

    # Numeric coercion; the stated valuation is the object of study.
    df["valuation"] = pd.to_numeric(df["valuation"], errors="coerce")

    # Attach contemporaneous observed price and true FV per (cell, round) from market.
    if market is not None and len(market):
        m = market.copy()
        m["price"] = pd.to_numeric(m["price"], errors="coerce")
        m["fundamental_price"] = pd.to_numeric(m.get("fundamental_price"), errors="coerce")
        # Single stock expected; if multiple stock_ids exist, average within a round.
        per_round = (m.groupby(["cell_id", "round"], as_index=False)
                       .agg(obs_price=("price", "mean"),
                            true_fv=("fundamental_price", "mean")))
        df = df.merge(per_round, on=["cell_id", "round"], how="left")
    else:
        df["obs_price"] = np.nan
        df["true_fv"] = np.nan

    # Fall back to the anchor-derived FV / constant FV where the market panel is silent.
    df["true_fv"] = df["true_fv"].fillna(fv_fallback)
    df["initial_price"] = df["anchor_mult"] * df["true_fv"]

    # Drop rows without a usable stated valuation (parse failures / fallback holds
    # from failed API calls come through as NaN or non-positive valuations).
    df = df[df["valuation"].notna() & (df["valuation"] > 0)].copy()

    df["error"] = df["valuation"] - df["true_fv"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["error"] / df["true_fv"]
    return df


def _ordered_modes(df: pd.DataFrame):
    present = [m for m in MODE_ORDER if m in set(df["mode"])]
    # Append any unexpected modes so nothing is silently dropped.
    present += [m for m in sorted(set(df["mode"].dropna())) if m not in present]
    return present


# --------------------------------------------------------------------------- #
# Estimator 1: valuation error by mode x anchor (distributions across the grid)
# --------------------------------------------------------------------------- #
def valuation_error_tables(df: pd.DataFrame):
    """Per-cell means + a mode x anchor distribution summary across the seed/temp grid."""
    # Per-cell (one independent run) means: collapse agents/rounds within a cell.
    cell_cols = ["mode", "anchor", "anchor_mult", "initial_price",
                 "prompt_family", "model", "temperature", "seed", "cell_id"]
    per_cell = (df.groupby(cell_cols, as_index=False)
                  .agg(mean_valuation=("valuation", "mean"),
                       mean_error=("error", "mean"),
                       mean_abs_error=("abs_error", "mean"),
                       mean_pct_error=("pct_error", "mean"),
                       n_decisions=("valuation", "size")))

    # Distribution across cells (seeds x temperatures) within each mode x anchor,
    # preserving prompt_family / model as clustering/grouping columns.
    grp = ["mode", "anchor", "anchor_mult", "prompt_family", "model"]
    summary = (per_cell.groupby(grp, as_index=False)
                       .agg(n_cells=("mean_error", "size"),
                            error_mean=("mean_error", "mean"),
                            error_std=("mean_error", "std"),
                            error_p10=("mean_error", lambda s: s.quantile(0.10)),
                            error_median=("mean_error", "median"),
                            error_p90=("mean_error", lambda s: s.quantile(0.90)),
                            abs_error_mean=("mean_abs_error", "mean"),
                            abs_error_median=("mean_abs_error", "median"),
                            pct_error_mean=("mean_pct_error", "mean")))
    # Order rows by the information ladder then anchor.
    mode_rank = {m: i for i, m in enumerate(_ordered_modes(df))}
    summary["_mrank"] = summary["mode"].map(mode_rank)
    summary = summary.sort_values(["_mrank", "anchor_mult"]).drop(columns="_mrank")
    return per_cell, summary


# --------------------------------------------------------------------------- #
# Estimator 2: anchoring regression (cluster-robust), coefficient by mode
# --------------------------------------------------------------------------- #
def ols_cluster_robust(y, X, clusters):
    """OLS with cluster-robust (CR0 + small-sample-adjusted) standard errors.

    y: (n,) outcome; X: (n, k) design incl. intercept; clusters: (n,) group labels.
    Returns (beta, se, n_obs, n_clusters).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    G = len(uniq)
    meat = np.zeros((k, k))
    for g in uniq:
        idx = clusters == g
        Xg = X[idx]
        ug = resid[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    # Standard small-sample correction (Stata-style).
    if G > 1 and n > k:
        adj = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    else:
        adj = 1.0
    V = XtX_inv @ meat @ XtX_inv * adj
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return beta, se, n, G


def _regress_by_mode(df: pd.DataFrame, xcol: str, cluster_col: str, label: str):
    """Regress valuation on `xcol` within each mode; return a tidy coefficient table."""
    rows = []
    for mode in _ordered_modes(df):
        sub = df[(df["mode"] == mode) & df[xcol].notna() & df["valuation"].notna()]
        if len(sub) < 3 or sub[xcol].nunique() < 2:
            rows.append({"regression": label, "mode": mode, "n_obs": len(sub),
                         "n_clusters": sub[cluster_col].nunique(),
                         "slope": np.nan, "slope_se": np.nan, "slope_t": np.nan,
                         "intercept": np.nan})
            continue
        X = np.column_stack([np.ones(len(sub)), sub[xcol].to_numpy(dtype=float)])
        beta, se, n, G = ols_cluster_robust(sub["valuation"].to_numpy(), X,
                                            sub[cluster_col].to_numpy())
        # Cluster-robust SEs are degenerate with a single cluster (e.g. a 1-seed
        # smoke): report the point estimate but blank the SE/t so it is not read
        # as inference. The full A1 grid has >=5 seeds per mode, so G>=5 there.
        if G < 2:
            print(f"[warn] {label} mode={mode}: only {G} cluster on '{cluster_col}'; "
                  f"slope SE/t suppressed (need >=2 clusters for inference).")
            se = np.array([np.nan, np.nan])
        slope_t = beta[1] / se[1] if se[1] > 0 else np.nan
        rows.append({"regression": label, "mode": mode, "n_obs": n, "n_clusters": G,
                     "slope": beta[1], "slope_se": se[1], "slope_t": slope_t,
                     "intercept": beta[0]})
    return pd.DataFrame(rows)


def anchoring_tables(df: pd.DataFrame, cluster_col: str):
    """Two anchoring regressions by mode: on observed price and on the initial anchor."""
    tables = []
    if df["obs_price"].notna().any():
        tables.append(_regress_by_mode(df, "obs_price", cluster_col,
                                       "valuation~observed_price"))
    tables.append(_regress_by_mode(df, "initial_price", cluster_col,
                                   "valuation~initial_anchor"))
    return pd.concat(tables, ignore_index=True)


# --------------------------------------------------------------------------- #
# Estimator 3: learning curve (valuation convergence by round)
# --------------------------------------------------------------------------- #
def learning_curve_table(df: pd.DataFrame):
    """Mean absolute valuation error by mode x round, with cross-cell dispersion."""
    # Per (cell, round) mean first (so each independent run weighs equally), then
    # summarize across cells by round.
    per_cell_round = (df.groupby(["mode", "round", "cell_id"], as_index=False)
                        .agg(cell_abs_error=("abs_error", "mean"),
                             cell_valuation=("valuation", "mean")))
    curve = (per_cell_round.groupby(["mode", "round"], as_index=False)
                           .agg(n_cells=("cell_abs_error", "size"),
                                abs_error_mean=("cell_abs_error", "mean"),
                                abs_error_median=("cell_abs_error", "median"),
                                abs_error_std=("cell_abs_error", "std"),
                                valuation_dispersion=("cell_valuation", "std")))
    mode_rank = {m: i for i, m in enumerate(_ordered_modes(df))}
    curve["_mrank"] = curve["mode"].map(mode_rank)
    curve = curve.sort_values(["_mrank", "round"]).drop(columns="_mrank")
    return curve


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_valuation_by_mode_anchor(df: pd.DataFrame, fv: float, out_path: Path):
    modes = _ordered_modes(df)
    anchors = [a for a in ANCHOR_MULT if a in set(df["anchor"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(3.0 * len(modes), 4.2), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        data = [df[(df["mode"] == mode) & (df["anchor"] == a)]["valuation"].dropna().to_numpy()
                for a in anchors]
        data = [d if len(d) else np.array([np.nan]) for d in data]
        # matplotlib renamed labels -> tick_labels in 3.9; requirements pin 3.10.
        ax.boxplot(data, tick_labels=anchors, showfliers=False)
        ax.axhline(fv, color="crimson", ls="--", lw=1, label=f"true FV={fv:g}")
        ax.set_title(mode, fontsize=9)
        ax.set_xlabel("price anchor")
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("stated valuation")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("A1: stated valuation by info mode x price anchor", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_learning_curve(curve: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in curve["mode"].drop_duplicates():
        sub = curve[curve["mode"] == mode]
        ax.plot(sub["round"], sub["abs_error_mean"], marker="o", ms=3, label=mode)
    ax.set_xlabel("round")
    ax.set_ylabel("mean |valuation - true FV|")
    ax.set_title("A1: valuation convergence as realizations accumulate")
    ax.legend(fontsize=8, title="info mode")
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
                    help="Output dir for tables/figures. Default: <sweep_root>/aggregated/a1_estimates/")
    ap.add_argument("--fv", type=float, default=DEFAULT_FV,
                    help=f"Fallback true fundamental value if market panel lacks it (default {DEFAULT_FV}).")
    ap.add_argument("--cluster", default="cell_id",
                    help="Column to cluster regression SEs on (default cell_id = independent run). "
                         "prompt_family / model are preserved as columns regardless.")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_root / "aggregated" / "a1_estimates"
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions, market = load_panels(sweep_root)
    df = build_analysis_frame(decisions, market, args.fv)
    if df.empty:
        sys.exit("Error: no usable (positive, parsed) valuations after filtering. "
                 "Check that the sweep produced real decisions (not fallback holds).")

    cluster_col = args.cluster if args.cluster in df.columns else "cell_id"
    if cluster_col != args.cluster:
        print(f"[warn] cluster column '{args.cluster}' not found; using 'cell_id'.")

    fv_used = float(df["true_fv"].median())

    # Estimator 1
    per_cell, err_summary = valuation_error_tables(df)
    per_cell.to_csv(out_dir / "valuation_error_by_cell.csv", index=False)
    err_summary.to_csv(out_dir / "valuation_error_summary.csv", index=False)

    # Estimator 2
    anchoring = anchoring_tables(df, cluster_col)
    anchoring.to_csv(out_dir / "anchoring_regression.csv", index=False)

    # Estimator 3
    curve = learning_curve_table(df)
    curve.to_csv(out_dir / "learning_curve.csv", index=False)

    # Figures
    plot_valuation_by_mode_anchor(df, fv_used, out_dir / "valuation_by_mode_anchor.png")
    plot_learning_curve(curve, out_dir / "learning_curve.png")

    # Console summary
    n_cells = df["cell_id"].nunique()
    print(f"A1 estimators over {len(df):,} decisions across {n_cells} cells "
          f"(true FV = {fv_used:g}, clustered on '{cluster_col}').\n")

    print("== Valuation error vs FV, by mode x anchor (mean across seeds/temps) ==")
    show = err_summary[["mode", "anchor", "n_cells", "error_mean", "error_std",
                        "abs_error_mean", "pct_error_mean"]]
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(show.to_string(index=False))

    print("\n== Anchoring regression (valuation on price), slope by mode ==")
    print("   slope 0 => tracks fundamentals; slope 1 => pure price anchoring")
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(anchoring[["regression", "mode", "n_obs", "n_clusters",
                         "slope", "slope_se", "slope_t"]].to_string(index=False))

    print("\n== FULL vs REALIZATIONS_ONLY headline (inference vs arithmetic) ==")
    head = (err_summary[err_summary["mode"].isin(["full", "realizations_only"])]
            .groupby("mode")["abs_error_mean"].mean())
    if {"full", "realizations_only"}.issubset(set(head.index)):
        gap = head["realizations_only"] - head["full"]
        print(f"   mean |error|: full={head['full']:.3f}, "
              f"realizations_only={head['realizations_only']:.3f}, gap={gap:.3f}")
    else:
        print("   (need both FULL and REALIZATIONS_ONLY cells present)")

    print(f"\nTables + figures written to {out_dir}")


if __name__ == "__main__":
    main()
