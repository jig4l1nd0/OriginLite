"""Central constants & enumerations."""

PALETTES = {
    "Plotly": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ],
    "Viridis": [
        "#440154", "#482777", "#3e4989", "#31688e", "#26828e",
        "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"
    ],
    "Plasma": [
        "#0d0887", "#41049d", "#6a00a8", "#8f0da4", "#b12a90",
        "#cc4778", "#e16462", "#f2844b", "#fca636", "#fcce25"
    ],
    "Greys": [
        "#111111", "#333333", "#555555", "#777777", "#999999",
        "#bbbbbb", "#dddddd", "#f0f0f0"
    ],
}

FIT_MODELS = [
    "None",
    "Linear",
    "Poly 2",
    "Poly 3",
    "Poly 4",
    "Poly 5",
    "Exponential",
    "Power law",
    "Langmuir",
    "Freundlich",
    "Redlich-Peterson",
    "Custom",
]

__all__ = ["PALETTES", "FIT_MODELS"]
