# OriginLite — a lightweight OriginPro alternative (MVP)

This is an open-source starter that covers a growing **data visualization & analysis** workflow:
- Import CSV/XLSX
- Interactive charts (line, scatter, bar, histogram, box, violin, heatmap)
- Multi-trace & multi-subplot plotting (multi Y selection)
- Mapping: color / size / symbol / facet
- Fits: linear, polynomial (deg 2–5), exponential, power-law, custom expression (generic non-linear)
- Confidence band (simple stderr) overlay
- Signal ops: moving average, Savitzky–Golay, baseline (ASLS), peak detection
- Data transforms: sort, filter (query), formulas (numexpr)
- Operation log (JSON) for reproducibility
- Themes, export PNG/SVG/HTML
- Templates & project bundle (.olite: data + config)

## Quickstart

```bash
# 1) Create a venv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Files

- `app.py` — Streamlit UI entrypoint
- `originlite/` — library modules
- `sample_data/` — try with `iris.csv` or `tips.csv`

## Roadmap
- Residual + leverage plots for fits; better CI bands
- Advanced peak analysis (width integration, deconvolution)
- Column formula editor with validation & caching
- Operation pipeline replay on reload
- Plugin discovery (entry points)
- Large data optimization (Polars / Arrow streaming)
- Advanced subplot layout editor (drag & drop)
- UI for baseline parameter tuning with preview overlay
- Batch processing (apply template to folder of files)

Licensed MIT — use this as a foundation for your own tool.
