"""CSV logging functionality with verification."""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class CSVLogger:
    """Handles CSV logging with optional verification."""

    @staticmethod
    def log_validation_error(
        logger: logging.Logger,
        csv_file_path: Path,
        round_number: int,
        agent_id: str,
        agent_type: str,
        error_type: str,
        details: str,
        attempted_action: str,
        debug: bool = False
    ) -> None:
        """
        Log validation error with verification.

        Args:
            logger: Logger instance to use
            csv_file_path: Path to CSV file
            round_number: Round number
            agent_id: Agent ID
            agent_type: Agent type
            error_type: Error type
            details: Error details
            attempted_action: Action that was attempted
            debug: If True, also log human-readable format to console

        Raises:
            RuntimeError: If CSV file not found or write verification fails
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # Create CSV formatted string
        csv_message = f"{timestamp},{round_number},{agent_id},{agent_type},{error_type},{details},{attempted_action}"

        # Get number of lines before writing
        with open(csv_file_path, 'r') as f:
            pre_lines = sum(1 for _ in f)

        # Log CSV format to file
        logger.info(csv_message)

        # Force flush the handlers
        for handler in logger.handlers:
            handler.flush()

        # Get number of lines after writing
        with open(csv_file_path, 'r') as f:
            post_lines = sum(1 for _ in f)

        # Verify the file exists and has more lines
        if not os.path.exists(csv_file_path):
            raise RuntimeError(
                f"Validation errors CSV file not found at {csv_file_path}. "
                "Logger initialization may have failed."
            )

        if post_lines <= pre_lines:  # Must have strictly more lines
            raise RuntimeError(
                f"CRITICAL: Validation error not written to CSV file. "
                f"Lines before: {pre_lines}, after: {post_lines}. "
                f"Message was: {csv_message}\n"
                f"Handlers: {logger.handlers}\n"
                f"File path: {csv_file_path}"
            )

        if debug:
            # Log human readable format to console
            logger.warning(
                f"Validation Error - Agent {agent_id} ({agent_type}) - {error_type}: {details} "
                f"[Attempted: {attempted_action}]"
            )

    @staticmethod
    def log_margin_call(
        logger: logging.Logger,
        round_number: int,
        agent_id: str,
        agent_type: str,
        borrowed_shares: float,
        max_borrowable: float,
        action: str,
        excess_shares: float,
        price: float
    ) -> None:
        """
        Log margin call event.

        Args:
            logger: Logger instance to use
            round_number: Round number
            agent_id: Agent ID
            agent_type: Agent type
            borrowed_shares: Number of borrowed shares
            max_borrowable: Maximum borrowable shares
            action: Action taken
            excess_shares: Excess shares
            price: Current price
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        csv_message = (
            f"{timestamp},{round_number},{agent_id},{agent_type},{borrowed_shares},"
            f"{max_borrowable},{action},{excess_shares},{price}"
        )
        logger.info(csv_message)
