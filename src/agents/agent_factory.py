"""Agent construction, extracted from BaseSimulation.

Builds the agent population declared in AGENT_PARAMS['agent_composition'].
Resolution rule: a type string registered in DETERMINISTIC_AGENTS gets its
rule-based class; anything else is treated as an LLM personality and must be
a key of AGENT_TYPES (validated up front by validate_agent_composition).
"""
from agents.LLMs.llm_agent import LLMAgent
from agents.registry import DETERMINISTIC_AGENTS, validate_agent_composition
from market.information.info_capability_config import (
    resolve_agent_info_capabilities,
    validate_config as validate_info_capabilities_config,
)
from services.logging_service import LoggingService


class AgentFactory:
    """Builds agents for one simulation run from its LLM/market settings."""

    def __init__(self, *, initial_price, model_open_ai, llm_temperature,
                 llm_seed, fundamental_info_mode, system_prompt_overrides,
                 logger):
        self.initial_price = initial_price
        self.model_open_ai = model_open_ai
        self.llm_temperature = llm_temperature
        self.llm_seed = llm_seed
        self.fundamental_info_mode = fundamental_info_mode
        self.system_prompt_overrides = system_prompt_overrides or {}
        self.logger = logger

    def create_agent(self, agent_id: int, agent_type: str, agent_params: dict):
        """Factory method to create appropriate agent type with explicit parameters"""
        # Get type-specific parameters if they exist, otherwise use defaults
        type_specific_params = agent_params.get('type_specific_params', {}).get(agent_type, {})

        # Handle both single-stock (initial_shares) and multi-stock (initial_positions)
        if 'initial_positions' in agent_params:
            # Multi-stock: check type_specific_params first, then fall back to agent_params
            positions = type_specific_params.get('initial_positions', agent_params['initial_positions'])
            initial_shares_value = sum(positions.values())
        else:
            # Single-stock: use initial_shares
            initial_shares_value = type_specific_params.get('initial_shares', agent_params['initial_shares'])

        # Get leverage parameters with centralized defaults
        leverage_params = agent_params.get('leverage_params', {})

        base_params = {
            'agent_id': agent_id,
            'initial_cash': type_specific_params.get('initial_cash', agent_params['initial_cash']),
            'initial_shares': initial_shares_value,
            'position_limit': type_specific_params.get('position_limit', agent_params['position_limit']),
            'allow_short_selling': type_specific_params.get('allow_short_selling', agent_params['allow_short_selling']),
            # Margin parameters (for short selling)
            'margin_requirement': type_specific_params.get('margin_requirement', agent_params.get('margin_requirement', 0.5)),
            'margin_base': type_specific_params.get('margin_base', agent_params.get('margin_base', 'cash')),
            'logger': LoggingService.get_logger('decisions'),
            'info_signals_logger': LoggingService.get_logger('info_signals'),
            'initial_price': self.initial_price,
            # Leverage parameters (for leveraged long positions)
            'leverage_ratio': type_specific_params.get('leverage_ratio', leverage_params.get('max_leverage_ratio', 1.0)),
            'initial_margin': type_specific_params.get('initial_margin', leverage_params.get('initial_margin', 0.5)),
            'maintenance_margin': type_specific_params.get('maintenance_margin', leverage_params.get('maintenance_margin', 0.25)),
        }

        # Check if it's a deterministic agent
        if agent_type in DETERMINISTIC_AGENTS:
            return DETERMINISTIC_AGENTS[agent_type](**base_params)

        # Set model name for hold_llm agent, or use type-specific model override
        model = "hold_llm" if agent_type == "hold_llm" else type_specific_params.get('model', self.model_open_ai)

        # LLM sampling params: per-agent-type override (parallel to per-type 'model'),
        # falling back to the simulation-wide defaults.
        llm_temperature = type_specific_params.get('temperature', self.llm_temperature)
        llm_seed = type_specific_params.get('seed', self.llm_seed)

        # Extract enabled features from agent_params
        from agents.LLMs.services.schema_features import FeatureRegistry
        enabled_features = FeatureRegistry.extract_features_from_config(agent_params)

        # Create LLM agent with appropriate model and feature configuration
        return LLMAgent(
            **base_params,
            agent_type=agent_type,
            model_open_ai=model,
            enabled_features=enabled_features,
            fundamental_info_mode=self.fundamental_info_mode,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
            system_prompt_override=self.system_prompt_overrides.get(agent_type)
        )

    def initialize_agents(self, agent_params: dict):
        """Initialize agents based on explicitly provided parameters"""
        agents = []
        agent_id = 0

        # Extract required parameters
        agent_composition = agent_params['agent_composition']  # Use the provided composition directly
        validate_agent_composition(agent_composition)
        self.logger.warning(f"Agent composition: {agent_composition}")

        # Scenario-level per-agent private-signal config (optional). Validated once
        # up front so bad config fails loudly at construction, not mid-simulation.
        info_capabilities_config = agent_params.get('info_capabilities')
        validate_info_capabilities_config(info_capabilities_config)
        type_counters: dict = {}  # per-type index, for spreading heterogeneous signals

        for agent_type, count in agent_composition.items():
            for _ in range(count):
                agent = self.create_agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    agent_params=agent_params
                )

                # Apply per-agent information capabilities (noise/delay/depth/etc.)
                # resolved from the scenario config. type_index spreads by-type
                # lists in composition order; global agent_id drives by_index/default.
                if info_capabilities_config:
                    type_index = type_counters.get(agent_type, 0)
                    capabilities = resolve_agent_info_capabilities(
                        info_capabilities_config,
                        agent_type=agent_type,
                        type_index=type_index,
                        global_index=agent_id,
                    )
                    for info_type, capability in capabilities.items():
                        agent.set_info_capability(info_type, capability)
                type_counters[agent_type] = type_counters.get(agent_type, 0) + 1

                # For multi-stock scenarios, set positions dict
                if 'initial_positions' in agent_params:
                    # Check type_specific_params first, then fall back to agent_params
                    type_specific = agent_params.get('type_specific_params', {}).get(agent_type, {})
                    positions = type_specific.get('initial_positions', agent_params['initial_positions'])
                    agent.positions = positions.copy()
                    # NOTE: Do NOT add DEFAULT_STOCK in multi-stock mode - only actual stocks exist
                    # Reset committed and borrowed positions for all stocks
                    agent.committed_positions = {stock_id: 0 for stock_id in agent.positions.keys()}
                    agent.borrowed_positions = {stock_id: 0 for stock_id in agent.positions.keys()}
                    # Update initial_shares to be the sum across all stocks for verification
                    agent.initial_shares = sum(positions.values())

                agents.append(agent)
                agent_id += 1

        return agents
