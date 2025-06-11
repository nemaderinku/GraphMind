import pandas as pd

class Cleaner:
    """Basic data cleaning operations."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.drop_duplicates(inplace=True)
        cleaned.fillna(method='ffill', inplace=True)
        return cleaned
