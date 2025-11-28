"""
Modular Prompt Builder for Feature Toggle System

This module builds agent prompts dynamically based on enabled features,
ensuring agents only receive instructions for capabilities they have access to.

Architecture:
    Enabled Features → Prompt Builder → Modular Instructions → Agent Prompt
"""

from typing import Set
from .schema_features import Feature


class PromptBuilder:
    """
    Builds agent prompts dynamically based on enabled features.

    This class maintains instruction templates for each feature and combines
    them into a complete strategic instructions section based on which
    features are enabled.
    """

    # Feature-specific instruction templates
    FEATURE_INSTRUCTIONS = {
        Feature.MEMORY: """
MEMORY SYSTEM:
Use 'notes_to_self' ONLY for genuine insights - what you LEARNED vs what you EXPECTED.
Compare your last reasoning to what actually happened:
- Did your prediction come true? Why or why not?
- What surprised you? What would you do differently?
- Any NEW hypothesis to test next round?

DON'T write: "Monitor price" or generic reminders (you'll see all data next round)
DO write: "Expected price drop but it rose - others more bullish than I thought" """,

        Feature.SOCIAL: """
MESSAGING:
You can optionally post a message visible to all agents next round using the 'post_message' field.
Before posting, explain your intent in 'message_reasoning' - what effect do you want your message to have?

Strategic Considerations:
- Messages can influence other agents' beliefs and decisions
- You may share information to shape market sentiment
- You may withhold information for competitive advantage
- You may signal confidence, uncertainty, or specific views to move prices
- Consider: What do you want other agents to believe?
- Be explicit about your messaging strategy in 'message_reasoning'""",

        Feature.SELF_MODIFY: """
STRATEGY EVOLUTION:
You can modify your trading strategy over time using 'prompt_modification'.
If you want to change how you approach trading decisions, propose a modification.

When to modify your strategy:
- Your current approach isn't working as expected
- You've discovered a new pattern or insight worth incorporating
- Market conditions have changed significantly
- You want to try a different trading style

How to propose modifications:
1. First explain WHY in 'modification_reasoning' (what you learned, what's not working)
2. Then write the new strategy text in 'prompt_modification'
3. Be specific and actionable - vague modifications won't help

Example:
- modification_reasoning: "My value investing approach keeps missing momentum - prices trend longer than expected"
- prompt_modification: "Consider momentum signals alongside fundamental value. When price has moved >5% in 3 rounds, weight trend-following more heavily."

Your modifications will be APPENDED to your base strategy, so you don't need to repeat everything.
Only modify when you have genuine insight - unnecessary changes add noise.""",
    }

    @staticmethod
    def build_instructions(enabled_features: Set[Feature]) -> str:
        """
        Build strategic instructions section based on enabled features.

        Args:
            enabled_features: Set of Feature enums that are enabled

        Returns:
            Combined instruction string with only enabled feature instructions

        Example:
            >>> features = {Feature.MEMORY}
            >>> instructions = PromptBuilder.build_instructions(features)
            >>> # Returns only memory instructions, not social instructions
        """
        if not enabled_features:
            return ""  # No additional instructions if no features enabled

        instruction_parts = []

        # Add instructions for each enabled feature
        for feature in enabled_features:
            if feature in PromptBuilder.FEATURE_INSTRUCTIONS:
                instruction_parts.append(PromptBuilder.FEATURE_INSTRUCTIONS[feature])

        # Combine all instructions with blank lines between sections
        return "\n\n".join(instruction_parts)

    @staticmethod
    def build_memory_section(memory_notes: list, display_limit: int = 10) -> str:
        """
        Build the memory log section for agents with memory enabled.

        Args:
            memory_notes: List of (round_number, note) tuples
            display_limit: Maximum number of recent notes to display

        Returns:
            Formatted memory section string, or empty string if memory disabled

        Example:
            >>> notes = [(1, "Price rising"), (2, "High volatility")]
            >>> section = PromptBuilder.build_memory_section(notes)
        """
        if not memory_notes:
            return "\n\n=== YOUR MEMORY LOG ===\n(No notes yet - you can start writing notes to track your strategy)\n"

        recent_notes = memory_notes[-display_limit:]
        total_notes = len(memory_notes)

        # Create a well-formatted memory display
        memory_lines = ["\n\n=== YOUR MEMORY LOG ==="]
        memory_lines.append(f"(Showing last {len(recent_notes)} of {total_notes} total notes)")
        memory_lines.append("")

        for r, note in recent_notes:
            # Format each note with clear round marker and indentation
            memory_lines.append(f"[Round {r}]")
            # Indent multi-line notes for better readability
            note_lines = note.strip().split('\n')
            for line in note_lines:
                memory_lines.append(f"  {line}")
            memory_lines.append("")  # Blank line between notes

        return "\n".join(memory_lines)

    @staticmethod
    def build_social_section(last_messages: list, enabled_features: Set[Feature]) -> str:
        """
        Build the social media feed section for agents.

        Args:
            last_messages: List of message dicts with 'agent_id' and 'message' keys
            enabled_features: Set of enabled features (to check if social is enabled)

        Returns:
            Formatted social feed section with strategic instructions if enabled

        Example:
            >>> messages = [{'agent_id': 'A1', 'message': 'Bullish!'}]
            >>> features = {Feature.SOCIAL}
            >>> section = PromptBuilder.build_social_section(messages, features)
        """
        # Only include social instructions if feature is enabled
        if Feature.SOCIAL not in enabled_features:
            # If social is disabled, don't show the feed at all
            return ""

        strategic_instructions = PromptBuilder.FEATURE_INSTRUCTIONS[Feature.SOCIAL]

        if last_messages:
            formatted_messages = "\n".join(
                f"- Agent {m['agent_id']}: {m['message']}"
                for m in last_messages
            )
            return f"\n\nSocial Feed (previous round):\n{formatted_messages}\n{strategic_instructions}"
        else:
            return f"\n\nSocial Feed: No messages yet.\n{strategic_instructions}"

    @staticmethod
    def build_last_reasoning_section(last_reasoning: dict) -> str:
        """
        Build section showing agent's reasoning from last round.

        This provides continuity by showing the agent WHY they made
        their previous decision, so they can build on their strategy.

        Args:
            last_reasoning: Dict with 'round', 'reasoning', 'valuation_reasoning',
                          'price_target_reasoning' keys

        Returns:
            Formatted string showing last round's reasoning, or empty if none
        """
        if not last_reasoning or not last_reasoning.get('reasoning'):
            return ""

        round_num = last_reasoning.get('round', '?')
        reasoning = last_reasoning.get('reasoning', '')
        valuation_reasoning = last_reasoning.get('valuation_reasoning', '')
        price_target_reasoning = last_reasoning.get('price_target_reasoning', '')

        lines = ["\n\n=== YOUR LAST ROUND REASONING ==="]
        lines.append(f"(Round {round_num})")
        lines.append("")

        if valuation_reasoning:
            lines.append(f"Valuation: {valuation_reasoning}")
        if price_target_reasoning:
            lines.append(f"Price Target: {price_target_reasoning}")
        if reasoning:
            lines.append(f"Decision: {reasoning}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def build_self_modify_section(prompt_history: list, current_prompt: str) -> str:
        """
        Build section showing agent their prompt evolution history.

        This helps agents understand how their strategy has evolved and
        make more informed decisions about further modifications.

        Args:
            prompt_history: List of (round_number, prompt) tuples
            current_prompt: The current (possibly modified) system prompt

        Returns:
            Formatted string showing prompt evolution, or empty if no modifications
        """
        if not prompt_history or len(prompt_history) <= 1:
            return "\n\n=== YOUR STRATEGY STATUS ===\nNo modifications yet. You may propose strategy changes if needed.\n"

        num_modifications = len(prompt_history) - 1
        original_prompt = prompt_history[0][1]

        lines = ["\n\n=== YOUR STRATEGY EVOLUTION ==="]
        lines.append(f"Total modifications: {num_modifications}")
        lines.append("")
        lines.append(f"Original strategy: {original_prompt[:150]}{'...' if len(original_prompt) > 150 else ''}")
        lines.append("")

        # Show recent modifications (last 3)
        recent_mods = prompt_history[-3:] if len(prompt_history) > 3 else prompt_history[1:]
        if recent_mods:
            lines.append("Recent modifications:")
            for round_num, prompt in recent_mods:
                if round_num == 0:
                    continue  # Skip original
                # Extract just the modification part (after the last [Strategy Update])
                if "[Strategy Update" in prompt:
                    mod_start = prompt.rfind("[Strategy Update")
                    mod_text = prompt[mod_start:mod_start + 200]
                    lines.append(f"  {mod_text}{'...' if len(prompt[mod_start:]) > 200 else ''}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def get_all_instructions() -> str:
        """
        Get instructions for all features (backward compatibility mode).

        Returns:
            Combined instructions for all available features
        """
        from .schema_features import FeatureRegistry
        return PromptBuilder.build_instructions(FeatureRegistry.get_all_features())
