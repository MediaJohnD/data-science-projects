import pandas as pd
from typing import List, Optional


def load_csv(path: str, expected_columns: Optional[List[str]] = None):
    """Load a CSV file and optionally validate required columns.

    Parameters
    ----------
    path: str
        Path to the CSV file. S3 paths are not supported and will raise
        ``FileNotFoundError``.
    expected_columns: list[str] | None
        Columns that must be present in the loaded dataframe.

    Returns
    -------
    pandas.DataFrame
        The loaded dataframe if validation succeeds.

    Raises
    ------
    FileNotFoundError
        If the path is an S3 path or the file does not exist.
    ValueError
        If any of the expected columns are missing.
    """

    if path.startswith("s3://"):
        # In this simplified example we do not support S3 downloads.
        raise FileNotFoundError(f"Bad S3 path: {path}")

    df = pd.read_csv(path)

    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    return df
