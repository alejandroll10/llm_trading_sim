"""Manager for CSV file headers and initialization."""
from pathlib import Path
from typing import List


class CSVHeaders:
    """Constants for CSV file headers."""

    VALIDATION_ERRORS = "timestamp,round_number,agent_id,agent_type,error_type,details,attempted_action"

    MARGIN_CALLS = (
        "timestamp,round_number,agent_id,agent_type,borrowed_shares,"
        "max_borrowable,action,excess_shares,price"
    )

    STRUCTURED_DECISIONS = (
        "timestamp,round,agent_id,agent_type,agent_type_id,decision,order_type,"
        "quantity,price,reasoning,valuation,price_prediction_t,price_prediction_t1,price_prediction_t2,"
        "valuation_reasoning,price_prediction_reasoning,notes_to_self,post_message,message_reasoning,"
        "valuation_confidence,prediction_confidence,others_avg_valuation,others_avg_valuation_reasoning"
    )


class CSVHeaderManager:
    """Manages initialization of CSV files with headers."""

    @staticmethod
    def initialize_csv_file(file_path: Path, header: str, truncate: bool = False) -> None:
        """
        Initialize a CSV file with a header if it doesn't exist or is empty.

        Args:
            file_path: Path to the CSV file
            header: Header string to write
            truncate: If True, rewrite the header even when the file already has
                rows, discarding them. Used for the shared logs/latest_sim/
                copies, which otherwise accumulate every run ever made.
        """
        if truncate or not file_path.exists() or file_path.stat().st_size == 0:
            with open(file_path, 'w') as f:
                f.write(f"{header}\n")

    @staticmethod
    def initialize_csv_files(file_paths: List[Path], header: str, truncate: bool = False) -> None:
        """
        Initialize multiple CSV files with the same header.

        Args:
            file_paths: List of paths to CSV files
            header: Header string to write to all files
            truncate: Passed through to initialize_csv_file.
        """
        for path in file_paths:
            CSVHeaderManager.initialize_csv_file(path, header, truncate=truncate)
