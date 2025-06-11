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
        const g = svg.append('g')
            .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

        const xScale = d3.scalePoint()
            .domain(data.map(d => d['{x}']))
            .range([0, width]);

        const yScale = d3.scalePoint()
            .domain(data.map(d => d['{y}']))
            .range([height, 0]);

        g.append('g')
            .attr('transform', `translate(0,${{height}})`)
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

    def generate_line(self, df: pd.DataFrame, x: str, y: str) -> str:
        """Return an HTML page with an embedded D3 line chart."""
        data = json.dumps(df[[x, y]].to_dict(orient="records"))
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Line Chart</title>
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
        const g = svg.append('g')
            .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

        const xScale = d3.scalePoint()
            .domain(data.map(d => d['{x}']))
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d['{y}']))
            .range([height, 0]);

        g.append('g')
            .attr('transform', `translate(0,${{height}})`)
            .call(d3.axisBottom(xScale));

        g.append('g')
            .call(d3.axisLeft(yScale));

        const line = d3.line()
            .x(d => xScale(d['{x}']))
            .y(d => yScale(d['{y}']));

        g.append('path')
            .datum(data)
            .attr('fill', 'none')
            .attr('stroke', '#636efa')
            .attr('stroke-width', 2)
            .attr('d', line);
    </script>
</body>
</html>"""
        return html

    def generate_histogram(self, df: pd.DataFrame, column: str) -> str:
        """Return an HTML page with an embedded D3 histogram."""
        data = json.dumps(df[[column]].to_dict(orient="records"))
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Histogram</title>
    <script src='https://d3js.org/d3.v7.min.js'></script>
</head>
<body>
    <svg width='800' height='600'></svg>
    <script>
        const rawData = {data}.map(d => d['{column}']);
        const svg = d3.select('svg');
        const margin = {{top: 20, right: 30, bottom: 30, left: 40}};
        const width = +svg.attr('width') - margin.left - margin.right;
        const height = +svg.attr('height') - margin.top - margin.bottom;
        const g = svg.append('g')
            .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

        const x = d3.scaleLinear()
            .domain(d3.extent(rawData))
            .range([0, width]);

        const histogram = d3.histogram()
            .domain(x.domain())
            .thresholds(x.ticks(20));

        const bins = histogram(rawData);

        const y = d3.scaleLinear()
            .domain([0, d3.max(bins, d => d.length)])
            .range([height, 0]);

        g.append('g')
            .attr('transform', `translate(0,${{height}})`)
            .call(d3.axisBottom(x));

        g.append('g')
            .call(d3.axisLeft(y));

        g.selectAll('rect')
            .data(bins)
            .enter().append('rect')
            .attr('x', 1)
            .attr('transform', d => `translate(${{x(d.x0)}},${{y(d.length)}})`)
            .attr('width', d => Math.max(0, x(d.x1) - x(d.x0) - 1))
            .attr('height', d => height - y(d.length))
            .attr('fill', '#636efa');
    </script>
</body>
</html>"""
        return html

    def generate_bar(self, df: pd.DataFrame, category: str, value: str) -> str:
        """Return an HTML page with an embedded D3 bar chart."""
        data = json.dumps(df[[category, value]].to_dict(orient="records"))
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Bar Chart</title>
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
        const g = svg.append('g')
            .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

        const xScale = d3.scaleBand()
            .domain(data.map(d => d['{category}']))
            .range([0, width])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d['{value}'])])
            .range([height, 0]);

        g.append('g')
            .attr('transform', `translate(0,${{height}})`)
            .call(d3.axisBottom(xScale));

        g.append('g')
            .call(d3.axisLeft(yScale));

        g.selectAll('rect')
            .data(data)
            .enter().append('rect')
            .attr('x', d => xScale(d['{category}']))
            .attr('y', d => yScale(d['{value}']))
            .attr('width', xScale.bandwidth())
            .attr('height', d => height - yScale(d['{value}']))
            .attr('fill', '#636efa');
    </script>
</body>
</html>"""
        return html

    def determine_charts(self, df: pd.DataFrame, max_charts: int = 5):
        """Return a list of chart specifications based on available data."""
        if max_charts < 1:
            raise ValueError("max_charts must be at least 1")

        charts = []
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        if len(numeric_cols) >= 2:
            charts.append(("scatter", numeric_cols[0], numeric_cols[1]))

        if len(numeric_cols) >= 1:
            charts.append(("histogram", numeric_cols[0]))

        if len(numeric_cols) >= 2:
            charts.append(("line", numeric_cols[0], numeric_cols[1]))

        if categorical_cols and numeric_cols:
            charts.append(("bar", categorical_cols[0], numeric_cols[0]))

        # Truncate and enforce minimum/maximum
        charts = charts[:max_charts]
        if not charts:
            raise ValueError("No suitable charts could be determined")
        return charts

    def generate_charts(self, df: pd.DataFrame, max_charts: int = 5):
        """Generate up to `max_charts` HTML chart files and return their paths."""
        chart_specs = self.determine_charts(df, max_charts)
        files = []
        for idx, spec in enumerate(chart_specs, start=1):
            ctype, *cols = spec
            if ctype == "scatter":
                html = self.generate_scatter(df, cols[0], cols[1])
            elif ctype == "line":
                html = self.generate_line(df, cols[0], cols[1])
            elif ctype == "histogram":
                html = self.generate_histogram(df, cols[0])
            elif ctype == "bar":
                html = self.generate_bar(df, cols[0], cols[1])
            else:
                continue
            file_name = f"chart_{idx}.html"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(html)
            files.append(file_name)
        return files
