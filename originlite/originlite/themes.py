import plotly.io as pio

THEMES = {
    "Plotly (Light)": "plotly",
    "Plotly (Dark)": "plotly_dark",
    "GGPlot2": "ggplot2",
    "Seaborn": "seaborn",
    "Simple White": "simple_white",
    "Presentation": "presentation",
}

def set_theme(name: str):
    tmpl = THEMES.get(name, "plotly")
    pio.templates.default = tmpl
    return tmpl
