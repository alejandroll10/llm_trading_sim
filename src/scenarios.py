from typing import Dict, Any
from calculate_fundamental import calculate_fundamental_price, calibrate_redemption_value

class SimulationScenario:
    """
    Represents a specific simulation scenario with a defined set of parameters.

    This class encapsulates the configuration for a simulation run, including
    its name, description, and all the necessary parameters to initialize
    and run the simulation. It also automatically calculates and validates
    the fundamental value and redemption value based on the provided
    dividend and interest rate parameters.

    Attributes:
        name (str): The unique name of the scenario.
        description (str): A brief description of what the scenario is testing.
        parameters (Dict[str, Any]): A dictionary of parameters for the simulation.
    """
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        
        # Calculate and validate fundamental prices
        self._calculate_fundamental_values()
    
    def _calculate_fundamental_values(self):
        """Calculate and enforce the constant fundamental value principle where:
        fundamental_value = redemption_value = expected_dividend/interest_rate
        """
        params = self.parameters
        
        # Extract required parameters
        num_rounds = params["NUM_ROUNDS"]
        is_infinite = params.get("INFINITE_ROUNDS", False)
        
        # Get dividend parameters
        dividend_params = params.get("DIVIDEND_PARAMS", {})
        base_dividend = dividend_params.get("base_dividend", 1.4)
        dividend_probability = dividend_params.get("dividend_probability", 0.5)
        dividend_variation = dividend_params.get("dividend_variation", 0.0)
        
        # Calculate expected dividend based on the dividend model - use the same formula as DividendService

        expected_dividend = dividend_probability * (base_dividend + dividend_variation) + \
                            (1 - dividend_probability) * (base_dividend - dividend_variation)
        
        # Get interest rate
        interest_model = params.get("INTEREST_MODEL", {})
        interest_rate = interest_model.get("rate", 0.05)
        
        # Calculate the constant fundamental value
        constant_fundamental = expected_dividend / interest_rate
        
        # Update the parameters
        params["FUNDAMENTAL_PRICE"] = constant_fundamental
        
        # For infinite horizon, no redemption value is needed
        if is_infinite:
            if "REDEMPTION_VALUE" in params:
                del params["REDEMPTION_VALUE"]
        # For finite horizon, set redemption value equal to fundamental value
        else:
            params["REDEMPTION_VALUE"] = constant_fundamental
            
        # Verify that with these parameters, the fundamental value is constant
        # across all periods (for debugging purposes)
        if not is_infinite:
            # Get the calculated fundamental with these parameters
            test_fundamental = calculate_fundamental_price(
                num_rounds, expected_dividend, interest_rate, constant_fundamental
            )
            
            # The difference should be very small (floating point precision)
            difference = abs(test_fundamental - constant_fundamental)
            assert difference < 1e-10, f"Fundamental value not constant: {test_fundamental} != {constant_fundamental}"

FUNDAMENTAL_WITH_DEFAULT_PARAMS = 28.0
BASE_NUM_ROUNDS = 15
BASE_INITIAL_CASH = 1000000.0
BASE_INITIAL_SHARES = 10000
BASE_INITIAL_PRICE = FUNDAMENTAL_WITH_DEFAULT_PARAMS
BASE_MAX_ORDER_SIZE = 1000
BASE_POSITION_LIMIT = 100000000
# Default parameters that can be overridden by specific scenarios
DEFAULT_PARAMS = {
    # Core simulation parameters
    "RANDOM_SEED": 42,
    "NUM_ROUNDS": BASE_NUM_ROUNDS,
    "INFINITE_ROUNDS": False,
    "HIDE_FUNDAMENTAL_PRICE": True,
    
    # Market parameters
    "INITIAL_PRICE": FUNDAMENTAL_WITH_DEFAULT_PARAMS,
    "TRANSACTION_COST": 0.0,
    "LENDABLE_SHARES": 0,

    # Agent parameters
    "MODEL_OPEN_AI": "gpt-4o-2024-11-20",
    "AGENT_PARAMS": {
        'allow_short_selling': False,
        'margin_requirement': 0.5,
        'borrow_model': {
            'rate': 0.01,
            'payment_frequency': 1
        },
        'position_limit': BASE_POSITION_LIMIT,
        'initial_cash': BASE_INITIAL_CASH,
        'initial_shares': BASE_INITIAL_SHARES,
        'max_order_size': BASE_MAX_ORDER_SIZE,
        'agent_composition': {
            'value': 2,
            'momentum': 2,
            'market_maker': 2
        },
        'deterministic_params': {
            'gap_trader': {
                'threshold': 0.05,
                'max_proportion': 0.5,
                'scaling_factor': 2.0
            }
        }
    },
    
    # Interest model parameters
    "INTEREST_MODEL": {
        'rate': 0.05,
        'compound_frequency': 1,
        'destination': 'dividend'
    },
    
    # Dividend parameters
    "DIVIDEND_PARAMS": {
        'type': 'stochastic',
        'base_dividend': 1.4,
        'dividend_frequency': 1,
        'dividend_growth': 0.0,
        'dividend_probability': 0.5,
        'dividend_variation': 1.0,
        'destination': 'dividend'
    }
}


SCENARIOS = {
    "price_discovery_above_fundamental": SimulationScenario(
        name="price_discovery_above_fundamental",
        description="Testing price discovery with initial mispricing",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": 20, #BASE_NUM_ROUNDS,
            "INITIAL_PRICE": round(FUNDAMENTAL_WITH_DEFAULT_PARAMS*1.25, 2),  # Start above fundamental
            # Fundamental will be calculated as E(d)/r = 1.4*0.5/0.05 = 14
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,  # Default initial cash
                'initial_shares': BASE_INITIAL_SHARES,    # Default initial shares
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2,        # Default investors
                    'optimistic': 2,      # Optimistic traders 
                    'market_maker': 2,   # Market makers for liquidity
                    'speculator': 2       # Speculators
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'speculator': {
                        # 'initial_cash': 10.0 * BASE_INITIAL_CASH,  # 1.5x default
                        # 'initial_shares': int(10.0 * BASE_INITIAL_SHARES)      # Half default
                    },
                    'default': {
                        # 'initial_cash': 1.0 * BASE_INITIAL_CASH,  # Default
                        # 'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # Default
                    },
                    'optimistic': {
                        # 'initial_cash': 1.2 * BASE_INITIAL_CASH,  # 1.2x default
                        # 'initial_shares': int(1.2 * BASE_INITIAL_SHARES)     # 1.2x default
                    }
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "price_discovery_below_fundamental": SimulationScenario(
        name="price_discovery_below_fundamental",
        description="Testing price discovery with initial mispricing",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": 20, #BASE_NUM_ROUNDS,
            "INITIAL_PRICE": round(FUNDAMENTAL_WITH_DEFAULT_PARAMS*0.75, 2),  # Start below fundamental
            # Fundamental will be calculated as E(d)/r = 1.4*0.5/0.05 = 14
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2,        # Default investors
                    'optimistic': 2,      # Optimistic traders 
                    'market_maker': 2,   # Market makers for liquidity
                    'speculator': 2       # Speculators
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'speculator': {
                        # 'initial_cash': 10.0 * BASE_INITIAL_CASH,  # 1.5x default
                        # 'initial_shares': int(10.0 * BASE_INITIAL_SHARES)      # Half default
                    }
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),

    "price_discovery_infinite_above_fundamental": SimulationScenario(
        name="price_discovery_infinite_above_fundamental",
        description="Testing price discovery with initial mispricing and infinite rounds",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": BASE_NUM_ROUNDS,
            "INFINITE_ROUNDS": True,
            "INITIAL_PRICE": 2*FUNDAMENTAL_WITH_DEFAULT_PARAMS,  # Start above fundamental
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2,        # Default investors
                    'optimistic': 2,      # Optimistic traders 
                    'market_maker': 2,   # Market makers for liquidity
                    'speculator': 2       # Speculators
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "price_discovery_infinite_below_fundamental": SimulationScenario(
        name="price_discovery_infinite_below_fundamental",
        description="Testing price discovery with initial mispricing and infinite rounds",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": BASE_NUM_ROUNDS,
            "INFINITE_ROUNDS": True,
            "INITIAL_PRICE": round(FUNDAMENTAL_WITH_DEFAULT_PARAMS/2, 2),  # Start below fundamental
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2,        # Default investors
                    'optimistic': 2,      # Optimistic traders 
                    'market_maker': 2,   # Market makers for liquidity
                    'speculator': 2       # Speculators
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "liquidity_crisis": SimulationScenario(
        name="liquidity_crisis",
        description="Testing market resilience under limited liquidity",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 75,
            "INITIAL_PRICE": 28.0,
            # Fundamental calculated as E(d)/r
            "INITIAL_CASH": 0.5 * BASE_INITIAL_CASH,  # Lower initial cash
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': 50000000,  # Lower position limit
                'initial_cash': 0.5 * BASE_INITIAL_CASH,    # Lower initial cash
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'momentum': 3,           # Trend followers
                    'market_maker_sell': 1,  # Liquidity provider with sell bias
                    'contrarian': 2          # Potential stabilizing force
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                },
                'type_specific_params': {
                    'market_maker_sell': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x scenario default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'momentum': {
                        'initial_cash': 0.3 * BASE_INITIAL_CASH,   # Lower than default
                        'initial_shares': int(0.8 * BASE_INITIAL_SHARES)      # Lower than default
                    },
                    'contrarian': {
                        'initial_cash': 0.7 * BASE_INITIAL_CASH,   # Higher than momentum
                        'initial_shares': int(1.5 * BASE_INITIAL_SHARES)     # Higher than momentum
                    }
                }
            }
        }
    ),

    "market_stress": SimulationScenario(
        name="market_stress",
        description="Testing market behavior with opposing views",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 100,
            "INITIAL_PRICE": 28.0,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'position_limit': 100000000,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'optimistic': 2,    # Believes price should be higher
                    'pessimistic': 2,   # Believes price should be lower
                    'market_maker': 2,  # Liquidity providers
                    'value': 2          # Rational anchor
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 2.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(2.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'optimistic': {
                        'initial_cash': 1.5 * BASE_INITIAL_CASH,  # Higher resources to push prices up
                        'initial_shares': int(0.5 * BASE_INITIAL_SHARES)      # Fewer shares (wants to buy more)
                    },
                    'pessimistic': {
                        'initial_cash': 0.5 * BASE_INITIAL_CASH,   # Less cash (wants to sell)
                        'initial_shares': int(1.5 * BASE_INITIAL_SHARES)     # More shares to sell
                    },
                    'value': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,  # Balanced
                        'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # Balanced
                    }
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),

    "test_hidden_fundamental": SimulationScenario(
        name="test_hidden_fundamental",
        description="Testing agent behavior when fundamental value is hidden",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": 3,  # Short test with just 3 rounds
            "HIDE_FUNDAMENTAL_PRICE": True,  # Hide fundamental value
            "INITIAL_PRICE": 28.0,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': 100000000,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 1,        # Value investor (relies on fundamentals)
                    'momentum': 1,     # Momentum trader (relies on price trends)
                    'market_maker': 1  # Market maker (provides liquidity)
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),

    "divergent_beliefs_above_fundamental": SimulationScenario(
        name="divergent_beliefs_above_fundamental",
        description="Testing market with agents having different beliefs about fundamental value",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": BASE_NUM_ROUNDS,
            "HIDE_FUNDAMENTAL_PRICE": True,  # Hide actual fundamental
            "INITIAL_PRICE": 2*FUNDAMENTAL_WITH_DEFAULT_PARAMS,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': int(0.5 * BASE_INITIAL_SHARES),
                    'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'optimistic': 2,    # Believes fundamental is much higher
                    'pessimistic': 2,   # Believes fundamental is much lower
                    'market_maker': 2,  # Neutral liquidity providers
                    'momentum': 2,      # Doesn't care about fundamentals
                    'default': 2        # Default investors
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'speculator': {
                        'initial_cash': 10.0 * BASE_INITIAL_CASH,  # 1.5x default
                        'initial_shares': int(10.0 * BASE_INITIAL_SHARES)      # Half default
                    },
                    'default': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,  # Default
                        'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # Default
                    },
                    'optimistic': {
                        'initial_cash': 1.2 * BASE_INITIAL_CASH,  # 1.2x default
                        'initial_shares': int(1.2 * BASE_INITIAL_SHARES)     # 1.2x default
                    }
                }
            }
        }
    ),
    "divergent_beliefs_below_fundamental": SimulationScenario(
        name="divergent_beliefs_below_fundamental",
        description="Testing market with agents having different beliefs about fundamental value",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": BASE_NUM_ROUNDS,
            "HIDE_FUNDAMENTAL_PRICE": True,  # Hide actual fundamental
            "INITIAL_PRICE": round(FUNDAMENTAL_WITH_DEFAULT_PARAMS/2, 2),
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': int(0.5 * BASE_INITIAL_SHARES),
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'optimistic': 2,    # Believes fundamental is much higher
                    'pessimistic': 2,   # Believes fundamental is much lower
                    'market_maker': 2,  # Neutral liquidity providers
                    'momentum': 2,      # Doesn't care about fundamentals
                    'default': 2        # Default investors
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'optimistic': {
                        'initial_cash': 10.0 * BASE_INITIAL_CASH,  # More cash to buy
                        'initial_shares': int(10.0 * BASE_INITIAL_SHARES)      # Fewer shares
                    },
                    'pessimistic': {
                        'initial_cash': 0.5 * BASE_INITIAL_CASH,   # Less cash
                        'initial_shares': int(10.0 * BASE_INITIAL_SHARES)     # More shares to sell
                    },
                    'value': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,  # Balanced
                        'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # Balanced
                    }
                }
            }
        }
    ),
    "test_default_trader": SimulationScenario(
        name="test_default_trader",
        description="Testing behavior of default traders in a minimal setting",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 2,  # Short test with just 2 rounds
            "INITIAL_PRICE": 28.0,
            "HIDE_FUNDAMENTAL_PRICE": True,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2  # Only 2 default traders
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "test_imbalanced_agents": SimulationScenario(
        name="test_imbalanced_agents",
        description="Testing market with highly imbalanced agent resources",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,  # Short test with just 3 rounds
            "INITIAL_PRICE": 28.0,
            "HIDE_FUNDAMENTAL_PRICE": True,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': 0.01 * BASE_INITIAL_CASH,    # Default initial cash (low)
                'initial_shares': 0.01 * BASE_INITIAL_SHARES,      # Default initial shares (low)
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 1,             # Just one value investor
                    'optimistic': 1       # Just one optimistic investor
                },
                'type_specific_params': {   # Extreme imbalance
                    'optimistic': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,  # 100x more cash
                        'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # 100x more shares
                    }
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "simple_mixed_traders": SimulationScenario(
        name="simple_mixed_traders",
        description="Simple scenario with one market maker, one optimistic trader, and one pessimistic trader",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,  # Short scenario with 10 rounds
            "INITIAL_PRICE": FUNDAMENTAL_WITH_DEFAULT_PARAMS,
            "HIDE_FUNDAMENTAL_PRICE": True,
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'market_maker': 1,   # One market maker for liquidity
                    'optimistic': 1,     # One optimistic trader
                    'pessimistic': 1     # One pessimistic trader
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 5.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(5.0 * BASE_INITIAL_SHARES)
                    },
                    'optimistic': {
                        'initial_cash': 1.5 * BASE_INITIAL_CASH,  # More cash to buy
                        'initial_shares': int(0.8 * BASE_INITIAL_SHARES)  # Fewer shares initially
                    },
                    'pessimistic': {
                        'initial_cash': 0.8 * BASE_INITIAL_CASH,  # Less cash
                        'initial_shares': int(1.5 * BASE_INITIAL_SHARES)  # More shares to sell
                    }
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "short_selling_benchmark": SimulationScenario(
        name="short_selling_benchmark",
        description="Benchmark scenario with short selling enabled and borrow costs",
        parameters={
            **DEFAULT_PARAMS,
            "LENDABLE_SHARES": 20000,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
            }
        }
    ),
    "cascade_test": SimulationScenario(
        name="cascade_test",
        description="Subset of optimistic agents broadcast messages to test information cascades",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 20,
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.01,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'optimistic': 2,
                    'value': 4,
                    'market_maker': 2
                }
            }
        }
    ),
    "deterministic_only": SimulationScenario(
        name="deterministic_only",
        description="Scenario with only deterministic agents to test messaging integration",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.01,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'gap_trader': 1,
                    'momentum_trader': 1,
                    'hold_trader': 1
                }
            }
        }
    ),
    "test_short_seller": SimulationScenario(
        name="test_short_seller",
        description="Test scenario with new short_seller agent and overvalued price",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "INITIAL_PRICE": 35.0,  # 25% above fundamental to encourage shorting
            "LENDABLE_SHARES": 30000,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'agent_composition': {
                    'short_seller': 2,      # New aggressive short sellers
                    'optimistic': 1,        # Provides buy-side liquidity
                    'market_maker': 1       # Provides liquidity
                },
                'type_specific_params': {
                    'short_seller': {
                        'initial_cash': 1.5 * BASE_INITIAL_CASH,
                        'initial_shares': int(0.5 * BASE_INITIAL_SHARES)  # Only 5000 shares
                    },
                    'optimistic': {
                        'initial_cash': 2.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(0.2 * BASE_INITIAL_SHARES)
                    },
                    'market_maker': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(2.0 * BASE_INITIAL_SHARES)
                    }
                }
            }
        }
    ),
    "aggressive_short_selling": SimulationScenario(
        name="aggressive_short_selling",
        description="Force aggressive short selling with extreme overpricing and low dividends",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 45.0,  # 60% above fundamental ($28)!
            "LENDABLE_SHARES": 50000,  # Plenty of shares to borrow
            "DIVIDEND_PARAMS": {
                'type': 'stochastic',
                'base_dividend': 0.2,  # Very low dividend (was 1.4)
                'dividend_frequency': 1,
                'dividend_growth': 0.0,
                'dividend_probability': 0.5,
                'dividend_variation': 0.1,  # Very low variation
                'destination': 'dividend'
            },
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.01,  # Lower borrow cost (1% vs 2%)
                    'payment_frequency': 1
                },
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'agent_composition': {
                    'short_seller': 3,      # 3 aggressive short sellers
                    'optimistic': 2,        # 2 optimistic buyers for liquidity
                },
                'type_specific_params': {
                    'short_seller': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,  # Lots of cash for margin
                        'initial_shares': 0  # ZERO shares - MUST borrow to short!
                    },
                    'optimistic': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,  # Lots of cash to buy
                        'initial_shares': int(0.1 * BASE_INITIAL_SHARES)  # Very few shares
                    }
                }
            }
        }
    ),
    "social_manipulation": SimulationScenario(
        name="social_manipulation",
        description="Test market manipulation via social media: influencers vs herd followers",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 15,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 2,        # 2 influencers trying to manipulate
                    'herd_follower': 4,     # 4 herd followers susceptible to influence
                    'value': 2,             # 2 value investors as control
                    'contrarian': 1,        # 1 contrarian to create conflict
                }
            }
        }
    ),
    "echo_chamber": SimulationScenario(
        name="echo_chamber",
        description="Echo chamber effect: mostly herd followers reinforcing each other",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 1,        # 1 influencer
                    'herd_follower': 7,     # 7 herd followers (majority)
                    'value': 1,             # 1 rational agent
                }
            }
        }
    ),
}

def get_scenario(scenario_name: str) -> SimulationScenario:
    """Get a scenario by name"""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available scenarios: {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario_name]

def list_scenarios() -> Dict[str, str]:
    """List all available scenarios and their descriptions"""
    return {name: scenario.description for name, scenario in SCENARIOS.items()}
