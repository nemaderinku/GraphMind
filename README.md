# GraphMind

GraphMind is a simple prototype of a multi-agent analyst for your data graph. It
allows you to ingest a CSV file, clean it, generate basic statistics and a chart,
then output an executive style summary. The generated chart uses D3.js so you can
hover over points to inspect their values.

## Requirements

- Python 3.8+
- `pip install pandas plotly`

## Running locally

1. Install the requirements:

   ```bash
   pip install pandas plotly
   ```

2. Execute the pipeline with a CSV file and an interactive D3 chart will open in your browser:

   ```bash
   python main.py path/to/your/data.csv
   ```

This runs entirely on your local machine and should work on most laptops,
including a Lenovo P52. No GPU is required.
