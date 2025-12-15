#!/usr/bin/env python3
"""
Inject actual LLM prompts and responses into paper LaTeX files.

This script pulls real data from simulation logs and generates LaTeX
snippets showing what LLMs actually see and produce.

Usage:
    python scripts/inject_paper_examples.py                    # Generate all examples
    python scripts/inject_paper_examples.py --preview          # Print without writing
    python scripts/inject_paper_examples.py --scenario social  # Specific scenario
"""

import sys
import json
import argparse
import textwrap
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

LOGS_DIR = Path(__file__).parent.parent / "logs" / "paper_final_results"
PAPER_DIR = Path(__file__).parent.parent.parent / "paper"

# Map scenario short names to directory patterns
SCENARIOS = {
    "social": "paper_social_manipulation_infinite",
    "bubble_short": "paper_bubble_with_shorts",
    "bubble_no_short": "paper_bubble_without_shorts",
    "correlated": "paper_correlated_crash",
    "price_above": "paper_price_discovery_above",
    "price_below": "paper_price_discovery_below",
}


def find_latest_run(scenario_pattern: str) -> Path:
    """Find the latest run directory for a scenario."""
    matches = sorted(LOGS_DIR.glob(f"{scenario_pattern}_*"))
    if not matches:
        raise FileNotFoundError(f"No runs found for {scenario_pattern}")
    return matches[-1]


def load_scenario_data(run_dir: Path) -> dict:
    """Load all relevant data from a scenario run."""
    data = {}

    # Load structured decisions (LLM responses)
    decisions_path = run_dir / "structured_decisions.csv"
    if decisions_path.exists():
        data['decisions'] = pd.read_csv(decisions_path)

    # Load market data
    market_path = run_dir / "data" / "market_data.csv"
    if market_path.exists():
        data['market'] = pd.read_csv(market_path)

    # Load agent data
    agent_path = run_dir / "data" / "agent_data.csv"
    if agent_path.exists():
        data['agents'] = pd.read_csv(agent_path)

    # Load parameters
    params_path = run_dir / "parameters.json"
    if params_path.exists():
        with open(params_path) as f:
            data['params'] = json.load(f)

    # Load metadata
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data['metadata'] = json.load(f)

    return data


def get_agent_prompt(agent_type: str) -> str:
    """Get system prompt for an agent type from agent_types.py."""
    from agents.agent_types import AGENT_TYPES

    # AGENT_TYPES is a dict mapping type_id -> AgentType
    if agent_type in AGENT_TYPES:
        return AGENT_TYPES[agent_type].system_prompt
    return f"You are a {agent_type} trader."


def format_market_state(data: dict, round_num: int, agent_id: int, condensed: bool = False) -> str:
    """Reconstruct the market state portion of the user prompt."""
    market = data['market']
    round_data = market[market['round'] == round_num].iloc[0]

    # Get agent position for this round
    agents = data['agents']
    agent_data = agents[(agents['round'] == round_num) & (agents['agent_id'] == agent_id)].iloc[0]

    params = data.get('params', {})
    dividend_params = params.get('DIVIDEND_PARAMS', {})
    interest_model = params.get('INTEREST_MODEL', {})

    lines = []

    # Market State
    lines.append("Market State:")
    lines.append(f"- Last Price: ${round_data['price']:.2f}")

    num_rounds = params.get('NUM_ROUNDS', 'Infinite')
    if params.get('INFINITE_ROUNDS', False):
        num_rounds = 'Infinite'
    lines.append(f"- Round Number: {round_num}/{num_rounds}")

    # Fundamental value display
    fund_mode = params.get('FUNDAMENTAL_INFO_MODE', 'full')
    if fund_mode == 'full':
        lines.append(f"- Fundamental Value: ${round_data['fundamental_price']:.2f}")
    else:
        lines.append("- Fundamental Value: Unavailable")

    lines.append(f"- Last Trading Volume: {round_data.get('total_volume', 0):.0f}")

    if not condensed:
        # Market Depth
        lines.append("")
        lines.append("Market Depth:")
        best_bid = round_data.get('best_bid', None)
        best_ask = round_data.get('best_ask', None)
        lines.append(f"- Best Bid: ${best_bid:.2f}" if best_bid and not pd.isna(best_bid) else "- Best Bid: None")
        lines.append(f"- Best Ask: ${best_ask:.2f}" if best_ask and not pd.isna(best_ask) else "- Best Ask: None")

    # Position Info
    lines.append("")
    lines.append("Your Position:")

    # Check if short selling is allowed
    agent_params = params.get('AGENT_PARAMS', {})
    allow_short = agent_params.get('allow_short_selling', False)
    short_note = "" if allow_short else " (Short selling not allowed)"

    lines.append(f"- Available Shares: {agent_data['available_shares']:.0f} shares{short_note}")
    lines.append(f"- Cash: ${agent_data['cash']:.2f}")

    if not condensed:
        lines.append(f"- Dividend Cash: ${agent_data.get('dividend_cash', 0):.2f}")

        # Dividend Information
        lines.append("")
        lines.append("Dividend Information:")
        base_div = dividend_params.get('base_dividend', 1.40)
        variation = dividend_params.get('dividend_variation', 1.0)
        prob = dividend_params.get('dividend_probability', 0.5)

        lines.append(f"- Expected Dividend: ${base_div:.2f}")
        lines.append(f"- Base Dividend: ${base_div:.2f}")
        lines.append(f"- Variation: +/- ${variation:.2f}")
        lines.append(f"- Max: ${base_div + variation:.2f} ({prob*100:.0f}% prob)")
        lines.append(f"- Min: ${base_div - variation:.2f} ({(1-prob)*100:.0f}% prob)")

        # Interest Rate
        lines.append("")
        lines.append("Interest Rate:")
        rate = interest_model.get('rate', 0.05)
        lines.append(f"- Rate: {rate*100:.1f}% per round")

        # Redemption
        lines.append("")
        if params.get('INFINITE_ROUNDS', False):
            lines.append("Redemption: Infinite horizon (no redemption)")
        else:
            lines.append(f"Redemption: At round {num_rounds}")

    return "\n".join(lines)


def format_trading_options(allow_short: bool = False) -> str:
    """Format the trading options section."""
    short_note = "Short selling is allowed" if allow_short else "Short selling is NOT allowed"

    return f"""Trading Options:
- Market order: order_type='market'
- Limit order: order_type='limit', specify price_limit
- Minimum tick size: $0.01
- {short_note}"""


def clean_text(text: str) -> str:
    """Clean unicode characters for LaTeX verbatim."""
    if not isinstance(text, str):
        return str(text) if text is not None else ''
    return (text
            .replace('\u2011', '-')   # non-breaking hyphen
            .replace('\u2013', '-')   # en-dash
            .replace('\u2014', '--')  # em-dash
            .replace(';', ','))       # semicolons to commas for readability


def format_response_wrapped(row: pd.Series, width: int = 70) -> str:
    """Format a structured decision as pretty-printed text for verbatim blocks.

    Manually formats to ensure lines don't exceed width for LaTeX verbatim.
    """
    valuation_reasoning = clean_text(row.get('valuation_reasoning', ''))
    reasoning = clean_text(row.get('reasoning', ''))

    # Wrap long text fields
    val_reason_wrapped = textwrap.fill(valuation_reasoning, width=width-4,
                                        initial_indent='    ', subsequent_indent='    ')
    reasoning_wrapped = textwrap.fill(reasoning, width=width-4,
                                       initial_indent='    ', subsequent_indent='    ')

    # Build order info
    order_lines = []
    if row.get('decision') not in ['Hold', 'none', None]:
        order_type = str(row.get('order_type', 'none')).replace('OrderType.', '').lower()
        order_lines.append(f'    decision: {row.get("decision", "Hold")}')
        order_lines.append(f'    quantity: {int(row.get("quantity", 0))}')
        order_lines.append(f'    order_type: {order_type}')
        if row.get('price') is not None:
            order_lines.append(f'    price_limit: {row.get("price")}')

    lines = [
        "valuation_reasoning:",
        val_reason_wrapped,
        f"valuation: {row.get('valuation', 0)}",
        f"price_prediction_t: {row.get('price_prediction_t', 0)}",
        f"price_prediction_t1: {row.get('price_prediction_t1', 0)}",
        f"price_prediction_t2: {row.get('price_prediction_t2', 0)}",
        "order:" if order_lines else "order: none",
    ]
    if order_lines:
        lines.extend(order_lines)
    lines.extend([
        "reasoning:",
        reasoning_wrapped,
    ])

    return '\n'.join(lines)


def format_response(row: pd.Series, wrap_width: int = 65) -> str:
    """Format a structured decision as JSON response."""
    response = {
        "valuation_reasoning": clean_text(row.get('valuation_reasoning', '')),
        "valuation": row.get('valuation', 0),
        "price_prediction_t": row.get('price_prediction_t', 0),
        "price_prediction_t1": row.get('price_prediction_t1', 0),
        "price_prediction_t2": row.get('price_prediction_t2', 0),
        "orders": [{
            "decision": row.get('decision', 'Hold'),
            "quantity": int(row.get('quantity', 0)),
            "order_type": str(row.get('order_type', 'none')).replace('OrderType.', '').lower(),
            "price_limit": row.get('price', None)
        }] if row.get('decision') not in ['Hold', 'none', None] else [],
        "replace_decision": "Add" if row.get('decision') not in ['Hold', 'none', None] else "Cancel",
        "reasoning": clean_text(row.get('reasoning', ''))
    }

    # Clean up None values
    if response['orders'] and response['orders'][0]['price_limit'] is None:
        del response['orders'][0]['price_limit']

    return json.dumps(response, indent=2, ensure_ascii=False)


def generate_example(scenario: str, round_num: int = 1, agent_type: str = None,
                     condensed: bool = False) -> dict:
    """Generate a complete prompt/response example for a scenario."""
    pattern = SCENARIOS.get(scenario, scenario)
    run_dir = find_latest_run(pattern)
    data = load_scenario_data(run_dir)

    if 'decisions' not in data:
        raise ValueError(f"No structured_decisions.csv found in {run_dir}")

    decisions = data['decisions']

    # Filter to requested round
    round_decisions = decisions[decisions['round'] == round_num]

    # Filter by agent type if specified
    if agent_type:
        round_decisions = round_decisions[
            round_decisions['agent_type'].str.lower().str.contains(agent_type.lower())
        ]

    if round_decisions.empty:
        raise ValueError(f"No decisions found for round {round_num}, agent_type={agent_type}")

    # Pick first matching decision
    row = round_decisions.iloc[0]

    # Get system prompt
    agent_type_id = row.get('agent_type_id', row.get('agent_type', 'default'))
    system_prompt = get_agent_prompt(agent_type_id)

    # Build user prompt (market state)
    user_prompt = format_market_state(data, round_num, row['agent_id'], condensed=condensed)

    # Format response
    response = format_response(row)

    return {
        'scenario': scenario,
        'run_dir': str(run_dir),
        'round': round_num,
        'agent_type': row.get('agent_type', 'Unknown'),
        'agent_id': row['agent_id'],
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'response': response,
        'raw_row': row.to_dict()
    }


def to_latex_verbatim(text: str, small: bool = True) -> str:
    """Convert text to LaTeX verbatim block."""
    # Verbatim blocks don't need escaping - they're literal
    # Just clean up any unicode dashes to regular dashes
    text = text.replace('\u2011', '-')  # non-breaking hyphen
    text = text.replace('\u2013', '-')  # en-dash
    text = text.replace('\u2014', '--') # em-dash

    prefix = r"\begin{small}" + "\n" if small else ""
    suffix = "\n" + r"\end{small}" if small else ""

    return f"{prefix}\\begin{{verbatim}}\n{text}\n\\end{{verbatim}}{suffix}"


def generate_appendix_example(scenario: str = "social", round_num: int = 1,
                              agent_type: str = "value") -> str:
    """Generate the complete appendix example in LaTeX format."""
    example = generate_example(scenario, round_num, agent_type)

    latex = []
    latex.append(f"% Auto-generated from {example['run_dir']}")
    latex.append(f"% Scenario: {example['scenario']}, Round: {example['round']}, Agent: {example['agent_type']}")
    latex.append(f"% Generated: {datetime.now().isoformat()}")
    latex.append("")

    latex.append(r"\paragraph{System Prompt}")
    latex.append(to_latex_verbatim(example['system_prompt']))
    latex.append("")

    latex.append(r"\paragraph{User Prompt (Market State)}")
    latex.append(to_latex_verbatim(example['user_prompt']))
    latex.append("")

    latex.append(r"\paragraph{LLM Response}")
    latex.append(to_latex_verbatim(example['response']))

    return "\n".join(latex)


def generate_main_text_figure(scenario: str = "social", round_num: int = 1,
                               agent_type: str = "value") -> str:
    """Generate two figures for main text: prompt and response."""
    # Use full (non-condensed) example to get all info
    example = generate_example(scenario, round_num, agent_type, condensed=False)

    # Parse the response to get key fields
    response_data = json.loads(example['response'])

    latex = []
    latex.append(f"% Auto-generated from {example['run_dir']}")
    latex.append(f"% Generated: {datetime.now().isoformat()}")
    latex.append("")

    # === FIGURE 1: PROMPT ===
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\begin{tcolorbox}[colback=gray!5, colframe=gray!50, width=\textwidth, boxrule=0.5pt]")
    latex.append("")

    # System prompt (full - these are typically short)
    system_prompt_clean = example['system_prompt'].replace('\n', ' ').strip()
    system_prompt_clean = ' '.join(system_prompt_clean.split())
    latex.append(r"\textbf{System Prompt:} \texttt{" + system_prompt_clean + "}")
    latex.append(r"\vspace{0.5em}")
    latex.append(r"\hrule")
    latex.append(r"\vspace{0.5em}")
    latex.append("")

    # User prompt (full market state)
    latex.append(r"\textbf{User Prompt:}")
    latex.append(r"\begin{small}")
    latex.append(r"\begin{verbatim}")
    latex.append(example['user_prompt'])
    latex.append(r"\end{verbatim}")
    latex.append(r"\end{small}")
    latex.append("")

    latex.append(r"\end{tcolorbox}")
    latex.append(r"\caption{Example prompt to an LLM trading agent. The system prompt establishes the agent's strategy. The user prompt provides market state, position information, dividend parameters, and interest rate---all information needed for fundamental valuation.}")
    latex.append(r"\label{fig:llm_prompt}")
    latex.append(r"\end{figure}")
    latex.append("")

    # === FIGURE 2: RESPONSE ===
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\begin{tcolorbox}[colback=gray!5, colframe=gray!50, width=\textwidth, boxrule=0.5pt]")
    latex.append("")

    latex.append(r"\textbf{LLM Response:}")
    latex.append(r"\begin{small}")
    latex.append(r"\begin{verbatim}")
    # Use wrapped format for readable line lengths
    raw_row = pd.Series(example['raw_row'])
    latex.append(format_response_wrapped(raw_row, width=70))
    latex.append(r"\end{verbatim}")
    latex.append(r"\end{small}")
    latex.append("")

    latex.append(r"\end{tcolorbox}")
    latex.append(r"\caption{LLM agent response to the prompt in Figure~\ref{fig:llm_prompt}. The agent calculates fundamental value as \$28.00 using the dividend discount model (\$1.40/0.05), recognizes the current price of \$31.50 is overvalued, and places a limit sell order. The structured output format ensures machine-readable decisions while the reasoning fields reveal the agent's decision process.}")
    latex.append(r"\label{fig:llm_response}")
    latex.append(r"\end{figure}")

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="Inject paper examples from logs")
    parser.add_argument("--preview", action="store_true", help="Print without writing")
    parser.add_argument("--scenario", default="social", choices=list(SCENARIOS.keys()),
                       help="Scenario to use for examples")
    parser.add_argument("--round", type=int, default=1, help="Round number for example")
    parser.add_argument("--agent", default="value", help="Agent type filter")
    parser.add_argument("--list", action="store_true", help="List available scenarios/runs")
    parser.add_argument("--type", choices=["appendix", "main", "both"], default="both",
                       help="Type of output: appendix (full), main (condensed figure), or both")

    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for short, pattern in SCENARIOS.items():
            try:
                run_dir = find_latest_run(pattern)
                print(f"  {short}: {run_dir.name}")
            except FileNotFoundError:
                print(f"  {short}: NOT FOUND")
        return

    try:
        outputs = []

        if args.type in ["appendix", "both"]:
            appendix_latex = generate_appendix_example(args.scenario, args.round, args.agent)
            outputs.append(("appendix", appendix_latex, "generated_example_appendix.tex"))

        if args.type in ["main", "both"]:
            main_latex = generate_main_text_figure(args.scenario, args.round, args.agent)
            outputs.append(("main", main_latex, "generated_example_main.tex"))

        for name, latex, filename in outputs:
            if args.preview:
                print(f"\n{'='*60}")
                print(f"{name.upper()} VERSION")
                print('='*60)
                print(latex)
            else:
                output_path = PAPER_DIR / "sections" / filename
                output_path.write_text(latex)
                print(f"Written {name} to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
