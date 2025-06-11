# GraphMind

GraphMind is a simple prototype of a multi-agent analyst for your data graph. It
allows you to ingest a CSV file, clean it, generate basic statistics and a chart
rendered with D3.js, then output an executive style summary.

## Requirements

- Python 3.8+
- `pip install pandas`

## Running locally

1. Install the requirements:

   ```bash
   pip install pandas
   ```

2. Execute the pipeline with a CSV file:

   ```bash
   python main.py path/to/your/data.csv
   ```

The script will generate `chart.html` using D3.js and open it in your default
browser. This runs entirely on your local machine and should work on most
laptops, including a Lenovo P52. No GPU is required.
