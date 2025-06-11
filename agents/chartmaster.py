import json
import plotly.express as px
from pathlib import Path
import pandas as pd
import webbrowser


class ChartMaster:
    """Create basic charts using Plotly."""

    def generate_scatter(self, df: pd.DataFrame, x: str, y: str):
        fig = px.scatter(df, x=x, y=y)
        return fig.to_dict()

    def generate_d3_html(self, df: pd.DataFrame, x: str, y: str, output: str = "scatter.html") -> Path:
        """Generate a simple D3 scatter plot saved to an HTML file."""
        data = df[[x, y]].to_dict(orient="records")
        html = f"""<!DOCTYPE html>
<meta charset='utf-8'>
<script src='https://d3js.org/d3.v7.min.js'></script>
<style>
  .tooltip {{
    position: absolute;
    text-align: center;
    padding: 4px;
    background: lightsteelblue;
    border: 1px solid #ccc;
    pointer-events: none;
    opacity: 0;
  }}
</style>
<body>
<div id='chart'></div>
<div id='tooltip' class='tooltip'></div>
<script>
const data = {json.dumps(data)};

const width = 600;
const height = 400;
const margin = {{top:20,right:20,bottom:30,left:40}};

const x = d3.scaleLinear()
  .domain(d3.extent(data, d => d['{x}'])).nice()
  .range([margin.left, width - margin.right]);

const y = d3.scaleLinear()
  .domain(d3.extent(data, d => d['{y}'])).nice()
  .range([height - margin.bottom, margin.top]);

const svg = d3.select('#chart')
  .append('svg')
  .attr('width', width)
  .attr('height', height);

svg.append('g')
  .attr('transform', `translate(0,${height - margin.bottom})`)
  .call(d3.axisBottom(x));

svg.append('g')
  .attr('transform', `translate(${margin.left},0)`)
  .call(d3.axisLeft(y));

const tooltip = d3.select('#tooltip');

svg.append('g')
  .selectAll('circle')
  .data(data)
  .join('circle')
    .attr('cx', d => x(d['{x}']))
    .attr('cy', d => y(d['{y}']))
    .attr('r', 4)
    .attr('fill', '#636efa')
    .on('mouseover', (event,d) => {{
      tooltip.style('opacity',1)
             .style('left',(event.pageX+5)+'px')
             .style('top',(event.pageY-28)+'px')
             .html('{x}='+d['{x}']+'<br>{y}='+d['{y}']);
    }})
    .on('mouseout', () => tooltip.style('opacity',0));
</script>
</body>
"""
        out_path = Path(output)
        out_path.write_text(html)
        return out_path

    def open_d3_html(self, path: Path):
        """Open the generated HTML file in the default web browser."""
        webbrowser.open(f"file://{path.resolve()}")
