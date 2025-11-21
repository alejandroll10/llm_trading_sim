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
Use 'notes_to_self' to record your strategic thinking and learnings.
What are you trying? What did you learn from previous rounds?
Don't repeat trades or prices.""",

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
    def get_all_instructions() -> str:
        """
        Get instructions for all features (backward compatibility mode).

        Returns:
            Combined instructions for all available features
        """
        from .schema_features import FeatureRegistry
        return PromptBuilder.build_instructions(FeatureRegistry.get_all_features())
