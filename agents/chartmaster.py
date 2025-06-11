import json
import pandas as pd

class ChartMaster:
    """Create basic charts using D3.js."""

    def generate_scatter(self, df: pd.DataFrame, x: str, y: str) -> str:
        """Return an HTML page with an embedded D3 scatter plot."""
        data = json.dumps(df[[x, y]].to_dict(orient="records"))
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Scatter Plot</title>
    <script src='https://d3js.org/d3.v7.min.js'></script>
</head>
<body>
    <svg width='800' height='600'></svg>
    <script>
        const data = {data};
        const svg = d3.select('svg');
        const margin = {{top: 20, right: 30, bottom: 30, left: 40}};
        const width = +svg.attr('width') - margin.left - margin.right;
        const height = +svg.attr('height') - margin.top - margin.bottom;
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const xScale = d3.scalePoint()
            .domain(data.map(d => d['{x}']))
            .range([0, width]);

        const yScale = d3.scalePoint()
            .domain(data.map(d => d['{y}']))
            .range([height, 0]);

        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale));

        g.append('g')
            .call(d3.axisLeft(yScale));

        g.selectAll('circle')
            .data(data)
            .enter().append('circle')
            .attr('cx', d => xScale(d['{x}']))
            .attr('cy', d => yScale(d['{y}']))
            .attr('r', 5)
            .attr('fill', '#636efa');
    </script>
</body>
</html>"""
        return html
