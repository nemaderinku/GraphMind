import pandas as pd
from pathlib import Path

class Ingestor:
    """Load data from a CSV file."""

    def load_csv(self, path: str) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {path} not found")
        return pd.read_csv(file_path)
