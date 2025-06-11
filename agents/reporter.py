import pandas as pd

class Reporter:
    """Generate a simple textual report from stats."""

    def summarize(self, stats: pd.DataFrame) -> str:
        summary_lines = [
            "Data Summary:",
            stats.to_string(),
        ]
        return "\n".join(summary_lines)
