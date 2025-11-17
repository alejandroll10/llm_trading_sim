"""Factory for creating and configuring loggers with consistent patterns."""
import logging
from pathlib import Path
from typing import Optional


class LoggerFactory:
    """Factory for creating loggers with standardized configurations."""

    @staticmethod
    def create_logger(
        name: str,
        run_dir: Path,
        latest_dir: Optional[Path],
        filename: str,
        console_handler: logging.Handler,
        formatter: Optional[logging.Formatter] = None,
        use_dual_output: bool = True
    ) -> logging.Logger:
        """
        Create a logger with file and console handlers.

        Args:
            name: Logger name
            run_dir: Run-specific directory for logs
            latest_dir: Latest directory for logs (optional)
            filename: Log filename
            console_handler: Console handler for warnings/errors
            formatter: Custom formatter (default: timestamped)
            use_dual_output: If True, write to both run_dir and latest_dir

        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Create file handler for run directory
        run_handler = logging.FileHandler(run_dir / filename)
        run_handler.setLevel(logging.INFO)
        run_handler.setFormatter(formatter)
        logger.addHandler(run_handler)

        # Create file handler for latest directory if dual output enabled
        if use_dual_output and latest_dir:
            latest_handler = logging.FileHandler(latest_dir / filename)
            latest_handler.setLevel(logging.INFO)
            latest_handler.setFormatter(formatter)
            logger.addHandler(latest_handler)

        # Add console handler
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def create_csv_logger(
        name: str,
        run_dir: Path,
        latest_dir: Optional[Path],
        filename: str,
        console_handler: logging.Handler,
        use_dual_output: bool = True
    ) -> logging.Logger:
        """
        Create a CSV logger with plain formatting (no timestamps).

        Args:
            name: Logger name
            run_dir: Run-specific directory for logs
            latest_dir: Latest directory for logs (optional)
            filename: CSV filename
            console_handler: Console handler for warnings/errors
            use_dual_output: If True, write to both run_dir and latest_dir

        Returns:
            Configured CSV logger
        """
        # CSV loggers use plain formatter (no timestamp prefix)
        csv_formatter = logging.Formatter('%(message)s')

        return LoggerFactory.create_logger(
            name=name,
            run_dir=run_dir,
            latest_dir=latest_dir,
            filename=filename,
            console_handler=console_handler,
            formatter=csv_formatter,
            use_dual_output=use_dual_output
        )

    @staticmethod
    def create_simple_logger(
        name: str,
        run_dir: Path,
        filename: str,
        console_handler: logging.Handler,
        formatter: Optional[logging.Formatter] = None
    ) -> logging.Logger:
        """
        Create a simple logger with only run directory output.

        Args:
            name: Logger name
            run_dir: Run-specific directory for logs
            filename: Log filename
            console_handler: Console handler for warnings/errors
            formatter: Custom formatter (default: timestamped)

        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - %(message)s')

        # Single file handler for run directory only
        file_handler = logging.FileHandler(run_dir / filename, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
