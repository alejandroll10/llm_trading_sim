"""Systematic / style-level dividend shock generation.

Extracted from BaseSimulation: pure draw logic with no simulation state.
Uses the module-level `random` generator (seeded per run in run_base_sim) so
extraction preserves reproducibility of seeded runs.
"""
import random


def generate_dividend_shocks(shock_config: dict, enabled: bool, logger=None) -> dict:
    """Generate systematic and style-level dividend shocks for one round.

    Shocks are drawn from normal distributions with volatilities configured in
    shock_config:
    - Systematic shock: affects all stocks (drawn once per round)
    - Style shocks: affect stocks within the same style category (one per style)

    Returns:
        dict with 'systematic' (float) and 'styles' (dict of style -> shock)
    """
    if not enabled:
        return {'systematic': 0.0, 'styles': {}}

    # Draw systematic shock (affects all stocks)
    systematic_volatility = shock_config.get('systematic_volatility', 0.0)
    systematic_shock = random.gauss(0, systematic_volatility) if systematic_volatility > 0 else 0.0

    # Draw style-level shocks (affects stocks in same style)
    style_shocks = {}
    style_config = shock_config.get('styles', {})
    for style, config in style_config.items():
        vol = config.get('volatility', 0.0) if isinstance(config, dict) else config
        style_shocks[style] = random.gauss(0, vol) if vol > 0 else 0.0

    shocks = {
        'systematic': systematic_shock,
        'styles': style_shocks
    }

    if logger:
        logger.info(
            f"Generated dividend shocks: systematic={systematic_shock:.4f}, "
            f"styles={style_shocks}"
        )

    return shocks
