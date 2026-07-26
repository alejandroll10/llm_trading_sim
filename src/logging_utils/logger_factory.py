"""Factory for creating and configuring loggers with consistent patterns."""
import logging
from pathlib import Path
from typing import Optional


class LoggerFactory:
    """Factory for creating loggers with standardized configurations."""

    @staticmethod
    def reset_handlers(logger: logging.Logger) -> None:
        """Detach and close every handler currently attached to `logger`.

        Loggers are process-global and keyed by name, so the same logger object
        comes back on the second run in a process. Without this, the previous
        run's handlers stay attached and every line is written into both runs'
        files -- in a sweep (run_sweep.py runs all cells in one process) cell k's
        files end up holding the rows of cells k+1..N (issue #120).

        Closing matters as much as removing: 15 loggers x N cells otherwise leaks
        a file descriptor per logger per cell.
        """
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                # A handler whose stream was closed underneath us (pytest capture,
                # an already-closed file) must not abort the next run's setup.
                pass

    @staticmethod
    def create_logger(
        name: str,
        run_dir: Path,
        latest_dir: Optional[Path],
        filename: str,
        console_handler: logging.Handler,
        formatter: Optional[logging.Formatter] = None,
        use_dual_output: bool = True,
        latest_mode: str = 'w'
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
            latest_mode: File mode for the latest_dir handler. 'w' (default) makes
                latest_sim/<file> hold only the latest run, as its name promises;
                pass 'a' when the file's header was already written before this
                call (the CSV loggers) so it is not truncated away.

        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        # Drop the previous run's handlers before attaching this run's (#120).
        LoggerFactory.reset_handlers(logger)
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
            latest_handler = logging.FileHandler(latest_dir / filename, mode=latest_mode)
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
            use_dual_output=use_dual_output,
            # CSVHeaderManager writes the header before this call, so the handler
            # must append rather than truncate it away.
            latest_mode='a'
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
        # Drop the previous run's handlers before attaching this run's (#120).
        LoggerFactory.reset_handlers(logger)
        logger.setLevel(logging.INFO)

        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - %(message)s')

        # Single file handler for run directory only
        file_handler = logging.FileHandler(run_dir / filename, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
