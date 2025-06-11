import pandas as pd

class Analyst:
    """Find basic insights from the data."""

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.describe()
