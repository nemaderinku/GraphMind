import plotly.express as px
import pandas as pd

class ChartMaster:
    """Create basic charts using Plotly."""

    def generate_scatter(self, df: pd.DataFrame, x: str, y: str):
        fig = px.scatter(df, x=x, y=y)
        return fig.to_dict()
