"""
Multi-Model Competition Scenarios

Scenarios where different LLM models compete against each other in the same simulation.
Useful for comparing model performance, reasoning quality, and trading strategies.
"""

from scenarios.base import SimulationScenario, DEFAULT_PARAMS

# Base values from DEFAULT_PARAMS
BASE_NUM_ROUNDS = DEFAULT_PARAMS["NUM_ROUNDS"]
BASE_INITIAL_CASH = DEFAULT_PARAMS["AGENT_PARAMS"]["initial_cash"]
BASE_INITIAL_SHARES = DEFAULT_PARAMS["AGENT_PARAMS"]["initial_shares"]
BASE_MAX_ORDER_SIZE = DEFAULT_PARAMS["AGENT_PARAMS"]["max_order_size"]

MULTI_MODEL_SCENARIOS = {
    "llama_vs_gpt_oss": SimulationScenario(
        name="llama_vs_gpt_oss",
        description="Llama 3.3 70B vs GPT-OSS 120B: Head-to-head competition with value investors",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 4,  # 2 Llama + 2 GPT-OSS default traders
                },
                'type_specific_params': {
                    # Split the 4 'default' agents: first 2 use Llama, last 2 use GPT-OSS
                    # This is achieved by making them different agent types
                }
            }
        }
    ),

    "three_model_battle": SimulationScenario(
        name="three_model_battle",
        description="Llama 3.3 vs Llama 3.1 vs GPT-OSS 120B: Three models competing",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 15,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 2,         # Llama 3.3 70B value investors
                    'momentum': 2,      # Llama 3.1 70B momentum traders
                    'default': 2,       # GPT-OSS 120B default traders
                    'market_maker': 1,  # Uses global default model
                },
                'type_specific_params': {
                    'value': {
                        'model': 'llama-3.3-70b-instruct',
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'momentum': {
                        'model': 'llama-3.1-70b-instruct',
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'default': {
                        'model': 'gpt-oss-120b',
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'market_maker': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(3.0 * BASE_INITIAL_SHARES),
                    }
                }
            }
        }
    ),

    "reasoning_vs_fast": SimulationScenario(
        name="reasoning_vs_fast",
        description="GPT-OSS 120B (reasoning) vs GPT-OSS 20B (faster): Quality vs Speed",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 3,         # GPT-OSS 120B (large reasoning model)
                    'momentum': 3,      # GPT-OSS 20B (smaller, faster)
                    'market_maker': 1,  # Neutral liquidity provider
                },
                'type_specific_params': {
                    'value': {
                        'model': 'gpt-oss-120b',  # Large reasoning model
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'momentum': {
                        'model': 'gpt-oss-20b',  # Smaller, faster model
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'market_maker': {
                        'initial_cash': 2.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(2.0 * BASE_INITIAL_SHARES),
                    }
                }
            }
        }
    ),

    "model_diversity_test": SimulationScenario(
        name="model_diversity_test",
        description="All verified models in one simulation: Ultimate model diversity test",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 20,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 1,          # Llama 3.3 70B
                    'momentum': 1,       # Llama 3.1 70B
                    'optimistic': 1,     # GPT-OSS 120B
                    'pessimistic': 1,    # GPT-OSS 20B
                    'market_maker': 1,   # Default model (from base.py)
                },
                'type_specific_params': {
                    'value': {
                        'model': 'llama-3.3-70b-instruct',
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'momentum': {
                        'model': 'llama-3.1-70b-instruct',
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'optimistic': {
                        'model': 'gpt-oss-120b',
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'pessimistic': {
                        'model': 'gpt-oss-20b',
                        'initial_cash': BASE_INITIAL_CASH,
                        'initial_shares': BASE_INITIAL_SHARES,
                    },
                    'market_maker': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(3.0 * BASE_INITIAL_SHARES),
                    }
                }
            }
        }
    ),
}
