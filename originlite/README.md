# OriginLite — a lightweight OriginPro alternative (MVP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

OriginLite is an open-source Streamlit app + modular library for quick **data visualization & analysis** without the bulk of full scientific suites. It aims for fast exploratory work, reproducibility, and extensibility.

## ✨ Current Feature Set

Data & Project
- Import: CSV / (plain text numeric) — drag & drop
- Project bundle: save / load `.olite` (data + config + operation log)
- Operation log & replay groundwork (JSON trace of transforms)

Visualization
- Interactive Plotly charts: line, scatter, bar, histogram, box, violin, heatmap
- Multi-trace, multi-Y, multi-subplot layout builder
- Aesthetic mappings: color, size, symbol, facet
- Per-plot style & theming controls (axes, legend, fonts)
- Per-plot logo overlay section (add/remove image, independent per tab)
- Selective plot tab removal (not just last tab)
- Export: PNG / SVG / standalone HTML

Analysis & Fitting
- Curve / isotherm models: linear, polynomial (deg 2–5), exponential, power-law
- Adsorption / isotherm fits: Langmuir, Freundlich, Redlich–Peterson
- Custom expression (generic non-linear) — uses `eval` (will be sandboxed later)
- Auto-fit executes immediately upon model selection (no extra click)
- Legend shows model name + numeric equation (multi-line)
- Parameter list & R² displayed contextually (only active model)
- Simple 95% band (stderr-based) overlay (placeholder for future CI improvements)

Signal Processing
- Moving average
- Savitzky–Golay smoothing
- Baseline correction (ASLS)
- Peak detection

Data Transforms
- Sort & filter (query)
- Column formula evaluation via `numexpr`

Extensibility
- Early plugin registry scaffold (for future custom models / operations)

## 🚀 Quickstart

```bash
# 1) Clone
git clone https://github.com/jig4l1nd0/OriginLite.git
cd OriginLite/originlite

# 2) (Optional) Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the app
streamlit run app.py
```

Open the URL Streamlit prints (default: http://localhost:8501).

## 🧪 Running Tests

Tests live under `tests/`.

```bash
pytest -q
```

All core data model, fitting, signal, and security guards are covered by the current suite.

## 📂 Key Files

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit entrypoint orchestrating per-plot workflow |
| `originlite/analysis/fits.py` | Fitting routines & `FitResult` dataclass |
| `originlite/core/data_model.py` | In-memory columnar model & operations log |
| `originlite/ui/sections.py` | Modular sidebar / panel UI sections |
| `originlite/viz/builder.py` | Plotly figure assembly utilities |
| `sample_data/iris.csv` | Example dataset |

## 🛡️ Security Notes
- The custom expression fit currently relies on Python `eval` (restricted context) — do **not** run untrusted formulas; a safer parser is on the roadmap.
- Future: sandboxed parser + allowlisted functions.

## 🗺️ Roadmap (Short / Mid Term)
Near-term
- Parameter uncertainties (CI per parameter) & goodness-of-fit metrics (AIC/BIC)
- Residual / leverage plots
- Improved confidence & prediction bands (bootstrap / covariance propagation)
- Caching of repeated fits when inputs unchanged
- Secure expression parser (AST transform / sympy / numexpr hybrid)

Mid-term
- Advanced peak deconvolution & batch peak stats
- Drag & drop subplot layout designer
- Operation pipeline replay & shareable templates
- Plugin discovery (entry points) & third-party model packs
- Large dataset backend switch (Arrow / Polars streaming)
- Batch processing (apply template to folder) with report export

## 🤝 Contributing
PRs and issues welcome! Suggested first contributions:
- Add statistical diagnostics for fits
- Replace `eval` in custom model with secure parser
- Improve baseline & peak parameter UI
- Add Polars optional dependency path

Workflow:
1. Fork & branch: `feat/your-feature`
2. Run tests: `pytest -q`
3. Ensure formatting / lint (if you add tooling) & open PR

## 📜 License

MIT — see [`LICENSE`](../LICENSE). You are free to use, modify, and distribute with attribution.

## 📸 (Optional Screenshot)
Add a screenshot/gif of a multi-plot layout with a fit & logo once you have one; drop it in `docs/` and reference here.

---
If this saves you time, consider a ⭐ on the repository.
