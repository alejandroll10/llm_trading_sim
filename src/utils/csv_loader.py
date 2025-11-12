"""Utility functions for loading CSV files with consistent error handling."""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


def load_csv(
    file_path: Union[str, Path],
    file_description: str = "CSV file",
    silent: bool = False
) -> Optional[pd.DataFrame]:
    """
    Load a CSV file with consistent error handling and validation.

    Args:
        file_path: Path to the CSV file
        file_description: Human-readable description of the file for error messages
        silent: If True, suppress informational messages (errors still printed)

    Returns:
        pd.DataFrame if successful, None if file doesn't exist or is empty

    Examples:
        >>> df = load_csv('data/agents.csv', 'agent data')
        >>> if df is not None:
        >>>     # Process the data
        >>>     pass
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        if not silent:
            print(f"  {file_description.capitalize()} file not found: {file_path}")
        return None

    try:
        # Load the CSV
        df = pd.read_csv(file_path)

        # Check if dataframe is empty
        if df.empty:
            if not silent:
                print(f"  {file_description.capitalize()} file exists but is empty")
            return None

        return df

    except pd.errors.EmptyDataError:
        print(f"  Error: {file_description} file is empty or malformed: {file_path}")
        return None
    except pd.errors.ParserError as e:
        print(f"  Error parsing {file_description} file: {str(e)}")
        return None
    except Exception as e:
        print(f"  Unexpected error loading {file_description} file: {str(e)}")
        return None
