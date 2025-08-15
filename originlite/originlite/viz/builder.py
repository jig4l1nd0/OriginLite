"""High-level visualization builder for multi-panel layouts."""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def combine_subplots(
    figs: List[go.Figure],
    orientation: str = "vertical",
    x_titles: Optional[List[str]] = None,
    y_titles: Optional[List[str]] = None,
    shared_x: bool = False,
    shared_y: bool = False,
    y_ranges: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    vertical_spacing: float = 0.07,
    horizontal_spacing: float = 0.07,
):
    """Combine a list of figures into subplots.

    orientation: 'vertical' (default) or 'horizontal'
    x_titles, y_titles: optional per-subplot axis titles (fallback to existing)
    """
    if not figs:
        return go.Figure()
    n = len(figs)
    if orientation == "horizontal":
        sp = make_subplots(
            rows=1,
            cols=n,
            shared_yaxes=shared_y,
            horizontal_spacing=horizontal_spacing,
        )
        for i, f in enumerate(figs, start=1):
            for tr in f.data:
                sp.add_trace(tr, row=1, col=i)
            # Axis titles via update API so they render properly
            if y_titles and i <= len(y_titles) and y_titles[i - 1]:
                sp.update_yaxes(title_text=y_titles[i - 1], row=1, col=i)
            if x_titles and i <= len(x_titles) and x_titles[i - 1]:
                sp.update_xaxes(title_text=x_titles[i - 1], row=1, col=i)
            if y_ranges and i <= len(y_ranges) and y_ranges[i - 1]:
                yr = y_ranges[i - 1]
                if yr:
                    sp.update_yaxes(range=list(yr), row=1, col=i)
        sp.update_layout(width=400 * n, height=400)
    else:  # vertical
        sp = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=shared_x,
            vertical_spacing=vertical_spacing,
        )
        for i, f in enumerate(figs, start=1):
            for tr in f.data:
                sp.add_trace(tr, row=i, col=1)
            if y_titles and i <= len(y_titles) and y_titles[i - 1]:
                sp.update_yaxes(title_text=y_titles[i - 1], row=i, col=1)
            if x_titles and i <= len(x_titles) and x_titles[i - 1]:
                sp.update_xaxes(title_text=x_titles[i - 1], row=i, col=1)
            if y_ranges and i <= len(y_ranges) and y_ranges[i - 1]:
                yr = y_ranges[i - 1]
                if yr:
                    sp.update_yaxes(range=list(yr), row=i, col=1)
        sp.update_layout(height=300 * n)
    return sp
