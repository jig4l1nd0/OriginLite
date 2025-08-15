from typing import Optional, Dict
import pandas as pd
import plotly.express as px

def make_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: Optional[str],
    y: Optional[str],
    color: Optional[str] = None,
    size: Optional[str] = None,
    symbol: Optional[str] = None,
    facet_col: Optional[str] = None,
    bins: int = 30,
    z: Optional[str] = None,
    agg: Optional[str] = None,
    color_discrete_map: Optional[Dict[str, str]] = None,
    error_y: Optional[str] = None,
    error_x: Optional[str] = None,
):
    common = dict(color=color, facet_col=facet_col)
    if color_discrete_map:
        # plotly express expects this keyword exactly
        common["color_discrete_map"] = color_discrete_map
    if chart_type == "scatter":
        kwargs = {}
        if error_y and error_y in df.columns:
            kwargs['error_y'] = error_y
        if error_x and error_x in df.columns:
            kwargs['error_x'] = error_x
        return px.scatter(
            df, x=x, y=y, size=size, symbol=symbol, **common, **kwargs
        )
    if chart_type == "line":
        kwargs = {}
        if error_y and error_y in df.columns:
            kwargs['error_y'] = error_y
        if error_x and error_x in df.columns:
            kwargs['error_x'] = error_x
        return px.line(
            df, x=x, y=y, **common, **kwargs
        )
    if chart_type == "bar":
        kwargs = {}
        if error_y and error_y in df.columns:
            kwargs['error_y'] = error_y
        if error_x and error_x in df.columns:
            kwargs['error_x'] = error_x
        return px.bar(
            df, x=x, y=y, **common, **kwargs
        )
    if chart_type == "hist":
        return px.histogram(df, x=x or y, nbins=bins, **common)
    if chart_type == "box":
        return px.box(df, x=x, y=y, **common)
    if chart_type == "violin":
        return px.violin(df, x=x, y=y, box=True, points="outliers", **common)
    if chart_type == "heatmap":
        if z and x and y:
            # Heatmap ignores color grouping; color_discrete_map not applicable
            return px.density_heatmap(df, x=x, y=y, z=z)
        raise ValueError("Heatmap requires x, y, and z columns.")
    raise ValueError(f"Unknown chart type: {chart_type}")
