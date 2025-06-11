import sys
import webbrowser
from agents.ingestor import Ingestor
from agents.cleaner import Cleaner
from agents.analyst import Analyst
from agents.chartmaster import ChartMaster
from agents.reporter import Reporter


def main(csv_path: str):
    ingestor = Ingestor()
    cleaner = Cleaner()
    analyst = Analyst()
    chartmaster = ChartMaster()
    reporter = Reporter()

    df = ingestor.load_csv(csv_path)
    cleaned = cleaner.clean(df)
    stats = analyst.describe(cleaned)
    chart_html = chartmaster.generate_scatter(
        cleaned, cleaned.columns[0], cleaned.columns[1]
    )
    with open("chart.html", "w", encoding="utf-8") as f:
        f.write(chart_html)
    webbrowser.open("chart.html")
    report = reporter.summarize(stats)

    print(report)
    print("\nChart saved to chart.html")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
