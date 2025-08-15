"""UI modular sections extracted from app.py to simplify maintenance."""
from __future__ import annotations
import streamlit as st
from originlite.data import list_numeric_columns, list_categorical_columns
from originlite.core import operations as ops
from originlite.analysis.signal import detect_peaks
from originlite.core.formula import FormulaError, validate_formula
from originlite.constants import FIT_MODELS
from originlite.analysis.fits import (
    fit_linear,
    fit_poly,
    fit_exponential,
    fit_powerlaw,
    fit_langmuir,
    fit_freundlich,
    fit_redlich_peterson,
    fit_generic,
)
import pandas as pd


def sidebar_transform_section(dm, container=st.sidebar):
    """Prepare data: transformations and signal processing.

    container: st container (e.g., sidebar, column) where widgets render.
    """
    with container.expander("Prepare", expanded=False):
        # --- Multi-column sort ---
        sort_cols = st.multiselect(
            "Sort columns (order applied left→right)",
            list(dm.df.columns),
            key="sort_cols_multi",
        )
        sort_orders = {}
        for col in sort_cols:
            sort_orders[col] = st.checkbox(
                f"Ascending: {col}", True, key=f"sort_asc_{col}"
            )
        if sort_cols:
            if st.button("Apply sort", key="apply_multi_sort"):
                try:
                    ascending_list = [sort_orders[c] for c in sort_cols]
                    dm.df = dm.df.sort_values(
                        by=list(sort_cols), ascending=ascending_list
                    )
                    dm.log(
                        "sort",
                        columns=list(sort_cols),
                        ascending=ascending_list,
                    )
                except Exception as e:
                    st.warning(f"Sort error: {e}")
        st.markdown("---")
        # --- Multi-column filter ---
        filt_cols = st.multiselect(
            "Filter columns (all conditions ANDed)",
            list(dm.df.columns),
            key="filter_cols_multi",
        )
        filter_exprs = {}
        for col in filt_cols:
            filter_exprs[col] = st.text_input(
                f"Expr for {col} (e.g., > 5, == 'A')",
                value="",
                key=f"filter_expr_{col}",
            )
        if filt_cols:
            if st.button("Apply filters", key="apply_multi_filters"):
                clauses = []
                for c in filt_cols:
                    expr = filter_exprs.get(c, "").strip()
                    if expr:
                        clauses.append(f"`{c}` {expr}")
                if clauses:
                    query_str = " and ".join(clauses)
                    try:
                        dm.df = dm.df.query(query_str)
                        dm.log(
                            "filter",
                            columns=filt_cols,
                            expressions={
                                c: filter_exprs[c] for c in filt_cols
                            },
                            query=query_str,
                        )
                    except Exception as e:
                        st.warning(f"Filter error: {e}")
        st.markdown("---")
        # (Savitzky-Golay smoothing removed per user request)
        # --- Baseline multi ---
        default_y_cols = st.session_state.get('default_peak_cols', [])
        base_cols = st.multiselect(
            "Baseline (ASLS) columns",
            list(dm.df.columns),
            default=[c for c in default_y_cols if c in dm.df.columns],
            help="Apply ASLS (creates *_baseline and *_corr columns).",
            key="baseline_cols_mod",
        )
        if base_cols:
            lam = st.number_input(
                "λ (smoothness)", 1e2, 1e7, 1e5, format="%.0f", key="asls_lam"
            )
            p_asls = st.slider("p (asymmetry)", 0.001, 0.1, 0.01, key="asls_p")
            if st.button("Apply baseline", key="baseline_btn_mod"):
                for bc in base_cols:
                    try:
                        dm.df = ops.op_baseline_asls(
                            dm.df, column=bc, lam=lam, p=p_asls
                        )
                        dm.log("baseline_asls", column=bc, lam=lam, p=p_asls)
                    except Exception as e:
                        st.warning(f"Baseline error {bc}: {e}")
        st.markdown("---")
        # (Peak detection moved to Analysis section)
        # --- Formula ---
        st.markdown("**Add formula column**")
        formula = st.text_input(
            "Formula (numexpr)", "y * 2", key="formula_expr_mod"
        )
        out_col = st.text_input(
            "Output column", "calc_col", key="formula_out_mod"
        )
        if (
            st.button("Apply formula", key="apply_formula_mod")
            and formula
            and out_col
        ):
            try:
                validate_formula(formula, dm.df.columns)
                # Ensure unique name if already exists
                base_name = out_col
                idx_suffix = 1
                while out_col in dm.df.columns:
                    out_col = f"{base_name}_{idx_suffix}"
                    idx_suffix += 1
                # Use operation wrapper for consistent logging
                dm.apply_operation(
                    'formula', formula=formula, out_col=out_col
                )
                # Verify effective creation (formula op may fail silently)
                if out_col not in dm.df.columns:
                    st.warning(
                        "Could not create column (invalid formula or "
                        "incompatible types)."
                    )
                    return
                # Store list of calculated columns for management
                calc_cols = st.session_state.get(
                    'formula_columns', []
                )
                if out_col not in calc_cols:
                    calc_cols.append(out_col)
                    st.session_state['formula_columns'] = calc_cols
                st.success(
                    f"Column '{out_col}' created. You can use it in "
                    "other formulas."
                )
            except FormulaError as fe:
                st.warning(str(fe))
        
    # Calculated columns management (delete) - moved outside expander
    calc_cols = st.session_state.get('formula_columns', [])
    if calc_cols:
        st.markdown("**Calculated columns**")
        st.caption(
            "List of columns created by formulas this session."
        )
        to_delete = st.multiselect(
            "Delete columns", calc_cols, key="formula_del_cols"
        )
        if to_delete and st.button(
            "Delete selected", key="delete_formula_cols"
        ):
            for c in to_delete:
                if c in dm.df.columns:
                    try:
                        dm.df.drop(columns=[c], inplace=True)
                    except Exception:
                        pass
            st.session_state['formula_columns'] = [
                c for c in calc_cols if c not in to_delete
            ]
            st.success("Columns deleted.")
    
    # st.markdown("**Operation log**")
    # st.json(dm.operations)


def section_chart_sidebar(df):
    num_cols = list_numeric_columns(df)
    cat_cols = list_categorical_columns(df)
    st.sidebar.header("Chart")
    chart_type = st.sidebar.selectbox(
        "Type",
        ["scatter", "line", "bar", "hist", "box", "violin", "heatmap"],
        index=0,
    )
    x = st.sidebar.selectbox("X", [None] + list(df.columns))
    y = st.sidebar.selectbox("Y (single)", [None] + list(df.columns))
    multi_y = st.sidebar.multiselect(
        "Y (multi subplots)", [c for c in df.columns if c != x]
    )
    color = st.sidebar.selectbox("Color/group", [None] + list(df.columns))
    size = st.sidebar.selectbox("Size", [None] + num_cols)
    symbol = st.sidebar.selectbox("Symbol", [None] + cat_cols)
    facet = st.sidebar.selectbox("Facet (columns)", [None] + cat_cols)
    z = st.sidebar.selectbox("Z (heatmap)", [None] + num_cols)
    bins = st.sidebar.slider("Bins (hist)", 5, 200, 30)
    return dict(
        chart_type=chart_type,
        x=x,
        y=y,
        multi_y=multi_y,
        color=color,
        size=size,
        symbol=symbol,
        facet=facet,
        z=z,
        bins=bins,
    )


__all__ = ["sidebar_transform_section", "section_chart_sidebar"]


# --- Sidebar modular sections ---

def sidebar_chart_section(
    df: pd.DataFrame, container=st.sidebar, key_prefix: str = ""
):
    """Chart mapping controls inside an expander.

    Returns dict with keys:
    chart_type,x,y,multi_y,color,size,symbol,facet,z,bins
    """
    num_cols = list_numeric_columns(df)
    cat_cols = list_categorical_columns(df)

    def k(s: str) -> str:
        """Namespace a widget/session key with the provided prefix."""
        return f"{key_prefix}{s}" if key_prefix else s

    with container.expander("Chart", expanded=True):
        # Use new key name to avoid stale session state issues
        if (
            'chart_type_sel' in st.session_state
            and 'chart_type_sel_v2' not in st.session_state
        ):
            st.session_state['chart_type_sel_v2'] = (
                st.session_state['chart_type_sel']
            )
        # Pre-resolve repeated keys (helps when we need raw key names later)
        key_chart_type = k("chart_type_sel_v2")
        key_x = k("x_sel")
        key_y = k("y_sel")
        key_restore = k("auto_restore_xy")
        key_last_x = k("last_x_sel")
        key_last_y = k("last_y_sel")
        key_multi_y = k("multi_y_sel")
        chart_type = st.selectbox(
            "Type",
            ["scatter", "line", "bar", "hist", "box", "violin", "heatmap"],
            index=0,
            key=key_chart_type,
        )
        x = st.selectbox("X", [None] + list(df.columns), key=key_x)
        y = st.selectbox("Y (single)", [None] + list(df.columns), key=key_y)
        restore_toggle = st.checkbox(
            "Auto-restore cleared X/Y (legacy)", False, key=key_restore
        )
        if restore_toggle:
            if key_last_x not in st.session_state and x:
                st.session_state[key_last_x] = x
            if key_last_y not in st.session_state and y:
                st.session_state[key_last_y] = y
            if x is not None:
                st.session_state[key_last_x] = x
            if y is not None:
                st.session_state[key_last_y] = y
            if x is None and y is None:
                lx = st.session_state.get(key_last_x)
                ly = st.session_state.get(key_last_y)
                if lx in df.columns:
                    x = lx
                    st.session_state[key_x] = lx
                if ly in df.columns:
                    y = ly
                    st.session_state[key_y] = ly
        multi_y = st.multiselect(
            "Y (multi)", [c for c in df.columns if c != x], key=key_multi_y
        )
        # Error bars selectors
        if x:
            x_err = st.selectbox(
                "X error (common)",
                [None] + [c for c in df.columns if c != x],
                key=k("x_err_sel"),
                help="Column with uncertainty in X (absolute error).",
            )
        else:
            x_err = None
        if y or multi_y:
            y_err = st.selectbox(
                "Y error (common)",
                [None] + [c for c in df.columns if c not in ()],
                key=k("y_err_sel"),
                help=(
                    "Column with uncertainty in Y (absolute error). "
                    "Applied to all series for multi-Y overlay/subplots."
                ),
            )
        else:
            y_err = None
        per_trace_errors = {}
        if multi_y:
            with st.expander("Per-trace Y errors", expanded=False):
                st.caption(
                    "Define specific error columns per series. Overrides "
                    "common Y error. Leave None to use the common one."
                )
                for yi in multi_y:
                    per_err = st.selectbox(
                        f"Y error for {yi}",
                        [None] + [c for c in df.columns if c != yi],
                        key=k(f"err_{yi}"),
                    )
                    if per_err:
                        per_trace_errors[yi] = per_err
        overlay_multi = False
        subplot_orientation = None
        if multi_y:
            # Only offer overlay for supported chart types
            overlay_supported = chart_type in {"line", "scatter", "bar"}
            if overlay_supported:
                overlay_multi = st.checkbox(
                    "Overlay multi-Y in a single chart (scatter/line/bar)",
                    value=st.session_state.get("overlay_multi", False),
                    key=k("overlay_multi"),
                    help=(
                        "Show selected Y columns as traces in one chart. "
                        "Uncheck to create a subplot per Y."
                    ),
                )
            else:
                # Reset if previously enabled on another chart type
                st.session_state["overlay_multi"] = False
                st.caption(
                    "Multi-Y overlay not available here; using subplots."
                )
            if not overlay_multi:
                subplot_orientation = st.selectbox(
                    "Subplots orientation",
                    ["vertical", "horizontal"],
                    index=0,
                    key=k("subplot_orientation"),
                )
        # Conditional mappings by type
        if chart_type != "heatmap":
            color = st.selectbox(
                "Color/group", [None] + list(df.columns), key=k("color_sel")
            )
        else:
            color = None
        if chart_type == "scatter":
            size = st.selectbox("Size", [None] + num_cols, key=k("size_sel"))
            symbol = st.selectbox(
                "Symbol", [None] + cat_cols, key=k("symbol_sel")
            )
        else:
            size = None
            symbol = None
        if chart_type != "heatmap":
            facet = st.selectbox(
                "Facet (columns)", [None] + cat_cols, key=k("facet_sel")
            )
        else:
            facet = None
        if chart_type == "heatmap":
            z = st.selectbox("Z (heatmap)", [None] + num_cols, key=k("z_sel"))
        else:
            z = None
        if chart_type == "hist":
            bins = st.slider("Bins (hist)", 5, 200, 30, key=k("bins_slider"))
        else:
            bins = 30  # default / unused
    return dict(
        chart_type=chart_type,
        x=x,
        y=y,
        y_err=y_err,
        x_err=x_err,
        per_trace_errors=per_trace_errors,
        multi_y=multi_y,
        overlay_multi=overlay_multi,
        subplot_orientation=subplot_orientation,
        color=color,
        size=size,
        symbol=symbol,
        facet=facet,
        z=z,
        bins=bins,
    )


def sidebar_subplots_edit_section(
    multi_y: list[str],
    orientation: str,
    container=st.sidebar,
    key_prefix: str = "",
):
    """Advanced per-subplot editing (visibility, titles, ranges, shared axes).

    Returns dict with keys:
    configs: {col: {visible,title,ymin,ymax}}
        shared_x: bool
        shared_y: bool
        uniform_y: bool
    """
    if not multi_y:
        return dict(
            configs={}, shared_x=False, shared_y=False, uniform_y=False
        )
    
    def k(s: str) -> str:
        return f"{key_prefix}{s}" if key_prefix else s
    with container.expander("Subplots edit", expanded=False):
        st.caption("Configure each subplot independently.")
        shared_x = st.checkbox(
            "Share X axis",
            value=(orientation == 'vertical'),
            key=k("subplots_shared_x"),
        )
        shared_y = st.checkbox(
            "Share Y axis", False, key=k("subplots_shared_y")
        )
        uniform_y = st.checkbox(
            "Force uniform Y range (auto from data if mins/maxs blank)",
            False,
            key=k("subplots_uniform_y"),
        )
        configs = {}
        for yi in multi_y:
            with st.container():
                cols = st.columns([3, 2, 2, 1])
                with cols[0]:
                    title = st.text_input(
                        f"Title {yi}", yi, key=k(f"subplot_title_{yi}")
                    )
                with cols[1]:
                    ymin = st.text_input(
                        f"Ymin {yi}", "", key=k(f"subplot_ymin_{yi}")
                    )
                with cols[2]:
                    ymax = st.text_input(
                        f"Ymax {yi}", "", key=k(f"subplot_ymax_{yi}")
                    )
                with cols[3]:
                    visible = st.checkbox(
                        "Show", True, key=k(f"subplot_visible_{yi}")
                    )
                # Parse numeric input

                def _parse(v):
                    v = v.strip()
                    if not v:
                        return None
                    try:
                        return float(v)
                    except Exception:
                        return None
                configs[yi] = dict(
                    title=title,
                    ymin=_parse(ymin),
                    ymax=_parse(ymax),
                    visible=visible,
                )
    return dict(
        configs=configs,
        shared_x=shared_x,
        shared_y=shared_y,
        uniform_y=uniform_y,
    )


def sidebar_fit_section(container=st.sidebar):
    """Fit model selection controls.
    Returns dict with keys: fit_kind, custom_expr, custom_params
    """
    with container.expander("Fit", expanded=False):
        fit_kind = st.selectbox(
            "Model", FIT_MODELS, index=0, key="fit_kind_sel"
        )
        custom_expr = None
        custom_params = None
        if fit_kind == "Custom":
            custom_expr = st.text_input(
                "y = f(x, params)",
                "a * exp(b * x) + c",
                key="custom_expr",
            )
            custom_params = st.text_input(
                "Params (comma order)", "a,b,c", key="custom_params"
            )
    return dict(
        fit_kind=fit_kind, custom_expr=custom_expr, custom_params=custom_params
    )


def sidebar_analysis_section(
    multi_y_active: bool,
    df: pd.DataFrame | None = None,
    x: str | None = None,
    y: str | None = None,
    container=st.sidebar,
    key_prefix: str = "",
):
    """Analysis: fitting and peak annotation options."""
    def k(s: str) -> str:
        return f"{key_prefix}{s}" if key_prefix else s
    with container.expander("Analysis", expanded=False):
        if multi_y_active:
            st.info("Fitting disabled when multiple Y are active.")
            fit_kind = "None"
            custom_expr = None
            custom_params = None
        else:
            fit_kind = st.selectbox(
                "Fit model", FIT_MODELS, index=0, key=k("fit_kind_sel")
            )
            custom_expr = None
            custom_params = None
            if fit_kind == "Custom":
                custom_expr = st.text_input(
                    "y = f(x, params)",
                    "a * exp(b * x) + c",
                    key=k("custom_expr"),
                )
                custom_params = st.text_input(
                    "Params (comma order)", "a,b,c", key=k("custom_params")
                )
            # Early auto-fit so params show immediately after selection
            if (
                fit_kind != 'None'
                and df is not None
                and x
                and y
                and x in df.columns
                and y in df.columns
            ):
                try:
                    xvals = pd.to_numeric(df[x], errors='coerce')
                    yvals = pd.to_numeric(df[y], errors='coerce')
                    mask = xvals.notna() & yvals.notna()
                    xvals = xvals[mask]
                    yvals = yvals[mask]
                    fit_obj = None
                    if fit_kind == 'Linear':
                        fit_obj = fit_linear(xvals, yvals)
                    elif fit_kind.startswith('Poly'):
                        deg = int(fit_kind.split()[1])
                        fit_obj = fit_poly(xvals, yvals, deg=deg)
                    elif fit_kind == 'Exponential':
                        fit_obj = fit_exponential(xvals, yvals)
                    elif fit_kind == 'Power law':
                        fit_obj = fit_powerlaw(xvals, yvals)
                    elif fit_kind == 'Langmuir':
                        fit_obj = fit_langmuir(xvals, yvals)
                    elif fit_kind == 'Freundlich':
                        fit_obj = fit_freundlich(xvals, yvals)
                    elif fit_kind == 'Redlich-Peterson':
                        fit_obj = fit_redlich_peterson(xvals, yvals)
                    elif (
                        fit_kind == 'Custom'
                        and custom_expr
                        and custom_params
                    ):
                        fit_obj = fit_generic(
                            xvals,
                            yvals,
                            expr=custom_expr,
                            param_names=custom_params,
                        )
                    if fit_obj is not None:
                        st.session_state[f"{key_prefix}fit_result"] = fit_obj
                except Exception:
                    pass
            # Display last fit parameters for this plot (if available)
            fit_res_key = (
                f"{key_prefix}fit_result" if key_prefix else "fit_result"
            )
            fit_res = st.session_state.get(fit_res_key)
            # Show params only if last fit matches current selection
            if (
                fit_res is not None
                and getattr(fit_res, 'params', None)
                and (
                    (fit_kind == 'Linear' and fit_res.model == 'linear')
                    or (
                        fit_kind.startswith('Poly')
                        and fit_res.model.startswith('poly_')
                        and fit_res.model.endswith(
                            fit_kind.split()[1]
                        )
                    )
                    or (
                        fit_kind == 'Exponential'
                        and fit_res.model == 'exponential'
                    )
                    or (
                        fit_kind == 'Power law'
                        and fit_res.model == 'powerlaw'
                    )
                    or (fit_kind == 'Langmuir' and fit_res.model == 'langmuir')
                    or (
                        fit_kind == 'Freundlich'
                        and fit_res.model == 'freundlich'
                    )
                    or (
                        fit_kind == 'Redlich-Peterson'
                        and fit_res.model == 'redlich-peterson'
                    )
                    or (fit_kind == 'Custom' and fit_res.model == 'custom')
                )
            ):
                eq = getattr(fit_res, 'equation', '') or ''
                params_str = ", ".join(
                    f"{k}={v:.4g}" for k, v in fit_res.params.items()
                )
                r2_val = getattr(fit_res, 'r2', None)
                r2_txt = f" | R²={r2_val:.4f}" if r2_val is not None else ""
                st.caption(
                    f"Fit: {fit_res.model}{r2_txt}\n{eq}\n{params_str}"
                )
                # Detailed param listing for linear & polynomial models
                if (
                    fit_res.model == 'linear'
                    or fit_res.model.startswith('poly_')
                ):
                    # Render a compact bullet list for clarity
                    lines = []
                    for name, val in fit_res.params.items():
                        lines.append(f"- **{name}** = {val:.6g}")
                    st.markdown("\n".join(lines))
        st.markdown("---")
        # Peak detection controls (moved from Prepare)
        dm = st.session_state.get('data_model')
        label_peaks = False  # Default value
        if dm is not None:
            default_peak_cols = st.session_state.get(
                k('default_peak_cols'),
                st.session_state.get('default_peak_cols', []),
            )
            # Restrict selectable peak columns strictly to selected Y series
            candidate_peak_cols = [
                c for c in default_peak_cols if c in dm.df.columns
            ]
            peak_cols = st.multiselect(
                "Peak detect columns",
                candidate_peak_cols,
                default=candidate_peak_cols,
                key=k("peak_detect_cols"),
                help="Only current Y selection is available.",
            )
            prom = st.number_input(
                "Prominence",
                0.0,
                1e9,
                0.1,
                step=0.0001,
                format="%.4f",
                key=k("prominence"),
            )
            label_peaks = st.checkbox(
                "Label peaks on chart", False, key=k("label_peaks")
            )
            if peak_cols and st.button(
                "Detect peaks", key=k("detect_peaks_mod")
            ):
                results = {}
                for col in peak_cols:
                    try:
                        results[col] = detect_peaks(
                            dm.df[col], prominence=prom
                        )
                    except Exception:
                        continue
                st.session_state[k('peaks_multi')] = results
            peaks_key = k('peaks_multi')
            if peaks_key in st.session_state:
                lines = []
                total = 0
                for col, data in st.session_state[peaks_key].items():
                    n = len(data.get('indices', []))
                    total += n
                    lines.append(f"{col}: {n}")
                if lines:
                    st.caption(
                        "Peaks (" + str(total) + ") | " + ", ".join(lines)
                    )
        st.markdown("---")
    return dict(
        fit_kind=fit_kind,
        custom_expr=custom_expr,
        custom_params=custom_params,
        label_peaks=label_peaks,
    )


def sidebar_format_section(x_val, y_val, container=st.sidebar):
    """Formatting / legend / labels controls.
    Returns dict with keys:
    x_title,y_title,tick_format_x,tick_format_y,tick_angle_x,
    tick_angle_y,nticks_x,nticks_y,legend_show,
    legend_orientation,legend_position,lx,ly,label_peaks
    """
    with container.expander("Format", expanded=False):
        chart_title = st.text_input("Chart title", "", key="chart_title")
        title_position = st.selectbox(
            "Title position",
            ["top-center", "top-left", "top-right"],
            index=0,
            key="title_position",
            help="Posición horizontal del título (arriba).",
        )
        x_title = st.text_input(
            "X axis title", value=x_val or "", key="x_title"
        )
        y_title = st.text_input(
            "Y axis title", value=y_val or "", key="y_title"
        )
        # Presets para formatos d3: https://github.com/d3/d3-format
        format_presets = [
            "",  # auto
            ".2f",
            ".3f",
            ".1e",
            ".2g",
            ".0%",
            ".2%",
            "$.2f",
            ".2~s",
            "Custom",
        ]
        # X format
        cur_x = st.session_state.get("tick_fmt_x", "")
        default_x_choice = (
            cur_x
            if cur_x in format_presets[:-1]
            else ("Custom" if cur_x else "")
        )
        x_choice = st.selectbox(
            "X tick format (d3)",
            format_presets,
            index=format_presets.index(default_x_choice),
            key="tick_fmt_x_choice",
            help=(
                "Formato numérico d3. 'Custom' permite especificar "
                "manualmente."
            ),
        )
        if x_choice == "Custom":
            tick_format_x = st.text_input(
                "Custom X format",
                cur_x if cur_x not in format_presets else "",
                key="tick_fmt_x",
            )
        else:
            tick_format_x = x_choice  # puede ser '' (auto)
            st.session_state["tick_fmt_x"] = tick_format_x
        # Y format
        cur_y = st.session_state.get("tick_fmt_y", "")
        default_y_choice = (
            cur_y
            if cur_y in format_presets[:-1]
            else ("Custom" if cur_y else "")
        )
        y_choice = st.selectbox(
            "Y tick format (d3)",
            format_presets,
            index=format_presets.index(default_y_choice),
            key="tick_fmt_y_choice",
            help=(
                "Formato numérico d3. 'Custom' permite especificar "
                "manualmente."
            ),
        )
        if y_choice == "Custom":
            tick_format_y = st.text_input(
                "Custom Y format",
                cur_y if cur_y not in format_presets else "",
                key="tick_fmt_y",
            )
        else:
            tick_format_y = y_choice
            st.session_state["tick_fmt_y"] = tick_format_y
        tick_angle_x = st.slider(
            "X tick angle", -90, 90, 0, key="tick_angle_x"
        )
        tick_angle_y = st.slider(
            "Y tick angle", -90, 90, 0, key="tick_angle_y"
        )
        nticks_x = st.number_input(
            "X nticks (0=auto)", 0, 100, 0, key="nticks_x"
        )
        nticks_y = st.number_input(
            "Y nticks (0=auto)", 0, 100, 0, key="nticks_y"
        )
        force_x_ticks = st.checkbox(
            "Force X ticks spacing",
            False,
            key="force_x_ticks",
            help=(
                "Calcula dtick uniforme a partir de nticks (si >1)."
            ),
        )
        force_y_ticks = st.checkbox(
            "Force Y ticks spacing",
            False,
            key="force_y_ticks",
            help=(
                "Calcula dtick uniforme a partir de nticks (si >1)."
            ),
        )
        legend_show = st.checkbox("Show legend", True, key="legend_show")
        legend_orientation = st.selectbox(
            "Legend orientation", ["v", "h"], index=0, key="legend_orientation"
        )
        # Legend position (inside this expander, not at sidebar root)
        legend_position = st.selectbox(
            "Legend position",
            ["top-right", "top-left", "bottom-right", "bottom-left"],
            index=0,
            key="legend_position",
        )
        legend_pos_map = {
            "top-right": (1, 1),
            "top-left": (0, 1),
            "bottom-right": (1, 0),
            "bottom-left": (0, 0),
        }
        (lx, ly) = legend_pos_map[legend_position]
    return dict(
        chart_title=chart_title,
        title_position=title_position,
        x_title=x_title,
        y_title=y_title,
        tick_format_x=tick_format_x,
        tick_format_y=tick_format_y,
        tick_angle_x=tick_angle_x,
        tick_angle_y=tick_angle_y,
        nticks_x=nticks_x,
        nticks_y=nticks_y,
        legend_show=legend_show,
        legend_orientation=legend_orientation,
        legend_position=legend_position,
        lx=lx,
        ly=ly,
        force_x_ticks=force_x_ticks,
        force_y_ticks=force_y_ticks,
    )


def sidebar_style_group(
    df: pd.DataFrame,
    color_col,
    fig,
    chart_type,
    container=st.sidebar,
    key_prefix: str = "",
):
    """Encapsulated Style section (single expander) including:
    - Format (axes, legend, ticks, title)
    - Figure (background, font, grid, size)
    - Categorical color map editor
    - Per-trace styling

    Returns (fmt_cfg, fig_style_cfg, style_cfg, custom_color_map)
    preserving previous signature for compatibility.
    """
    # Use per-plot style configuration (namespaced) to avoid plots
    # overwriting each other's trace style settings.
    global_key = "style_cfg"
    namespaced_key = f"{key_prefix}style_cfg" if key_prefix else global_key
    if namespaced_key not in st.session_state:
        # Initialize from global (legacy) if present to allow duplication
        base_cfg = st.session_state.get(global_key, {}) if key_prefix else {}
        st.session_state[namespaced_key] = base_cfg.copy()
    style_cfg = st.session_state[namespaced_key]

    def k(s: str) -> str:
        return f"{key_prefix}{s}" if key_prefix else s

    with container.expander("Style", expanded=False):
        # Inject scroll box CSS (fixed height to avoid huge page growth)
        st.markdown(
            """
            <style>
            .style-scroll-box {
                max-height:480px;
                overflow-y:auto;
                padding-right:0.5rem;
                border:1px solid #ddd;
                border-radius:4px;
                background: var(--background-color, #ffffff10);
            }
            .style-scroll-box::-webkit-scrollbar { width:8px; }
            .style-scroll-box::-webkit-scrollbar-track {
                background:transparent;
            }
            .style-scroll-box::-webkit-scrollbar-thumb {
                background:#bbb;
                border-radius:4px;
            }
            .style-scroll-box::-webkit-scrollbar-thumb:hover {
                background:#999;
            }
            </style>
            <div class="style-scroll-box">
            """,
            unsafe_allow_html=True,
        )
        # --- FORMAT SECTION (inline formerly sidebar_format_section) ---
        st.subheader("Format")
        chart_title = st.text_input("Chart title", "", key=k("chart_title"))
        title_position = st.selectbox(
            "Title position",
            ["top-center", "top-left", "top-right"],
            index=0,
            key=k("title_position"),
        )
        x_title = st.text_input("X axis title", value="", key=k("x_title"))
        y_title = st.text_input("Y axis title", value="", key=k("y_title"))
        format_presets = [
            "", ".2f", ".3f", ".1e", ".2g",
            ".0%", ".2%", "$.2f", ".2~s", "Custom",
        ]
        cur_x = st.session_state.get("tick_fmt_x", "")
        default_x_choice = (
            cur_x if cur_x in format_presets[:-1]
            else ("Custom" if cur_x else "")
        )
        x_choice = st.selectbox(
            "X tick format (d3)",
            format_presets,
            index=format_presets.index(default_x_choice),
            key=k("tick_fmt_x_choice"),
        )
        if x_choice == "Custom":
            tick_format_x = st.text_input(
                "Custom X format",
                cur_x if cur_x not in format_presets else "",
                key=k("tick_fmt_x"),
            )
        else:
            tick_format_x = x_choice
            st.session_state["tick_fmt_x"] = tick_format_x
        cur_y = st.session_state.get("tick_fmt_y", "")
        default_y_choice = (
            cur_y if cur_y in format_presets[:-1]
            else ("Custom" if cur_y else "")
        )
        y_choice = st.selectbox(
            "Y tick format (d3)",
            format_presets,
            index=format_presets.index(default_y_choice),
            key=k("tick_fmt_y_choice"),
        )
        if y_choice == "Custom":
            tick_format_y = st.text_input(
                "Custom Y format",
                cur_y if cur_y not in format_presets else "",
                key=k("tick_fmt_y"),
            )
        else:
            tick_format_y = y_choice
            st.session_state["tick_fmt_y"] = tick_format_y
        tick_angle_x = st.slider(
            "X tick angle", -90, 90, 0, key=k("tick_angle_x")
        )
        tick_angle_y = st.slider(
            "Y tick angle", -90, 90, 0, key=k("tick_angle_y")
        )
        nticks_x = st.number_input(
            "X nticks (0=auto)", 0, 100, 0, key=k("nticks_x")
        )
        nticks_y = st.number_input(
            "Y nticks (0=auto)", 0, 100, 0, key=k("nticks_y")
        )
        force_x_ticks = st.checkbox(
            "Force X ticks spacing", False, key=k("force_x_ticks")
        )
        force_y_ticks = st.checkbox(
            "Force Y ticks spacing", False, key=k("force_y_ticks")
        )
        legend_show = st.checkbox("Show legend", True, key=k("legend_show"))
        legend_orientation = st.selectbox(
            "Legend orientation",
            ["v", "h"],
            index=0,
            key=k("legend_orientation"),
        )
        legend_position = st.selectbox(
            "Legend position",
            ["top-right", "top-left", "bottom-right", "bottom-left"],
            index=0,
            key=k("legend_position"),
        )
        legend_pos_map = {
            "top-right": (1, 1),
            "top-left": (0, 1),
            "bottom-right": (1, 0),
            "bottom-left": (0, 0),
        }
        lx, ly = legend_pos_map[legend_position]

        fmt_cfg = dict(
            chart_title=chart_title,
            title_position=title_position,
            x_title=x_title,
            y_title=y_title,
            tick_format_x=tick_format_x,
            tick_format_y=tick_format_y,
            tick_angle_x=tick_angle_x,
            tick_angle_y=tick_angle_y,
            nticks_x=nticks_x,
            nticks_y=nticks_y,
            legend_show=legend_show,
            legend_orientation=legend_orientation,
            legend_position=legend_position,
            lx=lx,
            ly=ly,
            force_x_ticks=force_x_ticks,
            force_y_ticks=force_y_ticks,
        )

        st.markdown("---")
        # --- FIGURE STYLE (inline) ---
        st.subheader("Figure")
        override_theme = st.checkbox(
            "Override theme in figure", True, key=k("fig_override_theme")
        )
        fig_plot_bg = st.color_picker(
            "Plot background", value="#ffffff", key=k("plot_bg_col")
        )
        fig_paper_bg = st.color_picker(
            "Outer background", value="#ffffff", key=k("paper_bg_col")
        )
        fig_font_family = st.selectbox(
            "Font family",
            [
                "Sans-Serif",
                "Arial",
                "Helvetica",
                "Times New Roman",
                "Courier New",
                "Monospace",
            ],
            index=0,
            key=k("fig_font_family"),
        )
        fig_font_size = st.number_input(
            "Font size", 6, 40, 12, key=k("fig_font_size")
        )
        grid_x = st.checkbox("Show X grid", True, key=k("fig_grid_x"))
        grid_y = st.checkbox("Show Y grid", True, key=k("fig_grid_y"))
        grid_color = st.color_picker(
            "Grid color", value="#e0e0e0", key=k("fig_grid_color")
        )
        grid_width = st.slider("Grid width", 0, 5, 1, key=k("fig_grid_width"))
        custom_size = st.checkbox(
            "Custom size (px)", False, key=k("fig_custom_size")
        )
        if custom_size:
            fig_width = st.slider(
                "Width (px)", 300, 3000, 900, step=10, key=k("fig_width")
            )
            fig_height = st.slider(
                "Height (px)", 200, 2000, 600, step=10, key=k("fig_height")
            )
        else:
            fig_width = None
            fig_height = None

        fig_style_cfg = dict(
            override_theme=override_theme,
            fig_plot_bg=fig_plot_bg,
            fig_paper_bg=fig_paper_bg,
            fig_font_family=fig_font_family,
            fig_font_size=fig_font_size,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_color=grid_color,
            grid_width=grid_width,
            custom_size=custom_size,
            fig_width=fig_width,
            fig_height=fig_height,
        )
        # Removed color map editor per user request (keeping return slot)
        custom_color_map = st.session_state.get("custom_color_map", {})

        st.markdown("---")
        # --- PER-TRACE STYLE (inline) ---
        st.subheader("Per-trace style")
        if fig and getattr(fig, "data", None):
            for i, tr in enumerate(fig.data):
                name = tr.name or f"trace_{i}"
                orig_name = name
                cfg = style_cfg.get(name, {})
                with st.container():
                    disp_name = st.text_input(
                        f"Trace name {i+1}",
                        value=cfg.get("display_name", name),
                        key=k(f"disp_{orig_name}"),
                    )
                    st.markdown(f"**{disp_name}**")
                    new_color = st.color_picker(
                        f"Color {name}",
                        value=cfg.get(
                            "color",
                            getattr(
                                getattr(tr, 'line', None), 'color', '#1f77b4'
                            ) or "#1f77b4",
                        ),
                        key=k(f"col_{name}"),
                    )
                    if chart_type in ["line", "scatter"]:
                        if chart_type == "scatter":
                            st.selectbox(
                                f"Mode {name}",
                                ["markers", "lines", "lines+markers"],
                                index=[
                                    "markers", "lines", "lines+markers"
                                ].index(cfg.get("mode", "markers")),
                                key=k(f"mode_{name}"),
                            )
                        else:
                            st.checkbox(
                                f"Show markers {name}",
                                value=cfg.get("show_markers", False),
                                key=k(f"mk_{name}"),
                            )
                        line_w = st.slider(
                            f"Line width {name}", 1, 12,
                            cfg.get("line_width", 2), key=k(f"lw_{name}")
                        )
                        line_dash = st.selectbox(
                            f"Line style {name}",
                            ["solid", "dash", "dot", "dashdot"],
                            index=[
                                "solid", "dash", "dot", "dashdot"
                            ].index(cfg.get("line_dash", "solid")),
                            key=k(f"ld_{name}"),
                        )
                        show_markers = (
                            chart_type == "scatter"
                            and "markers"
                            in (
                                st.session_state.get(
                                    k(f"mode_{name}"),
                                    cfg.get("mode", "markers"),
                                )
                            )
                        ) or (
                            chart_type == "line"
                            and st.session_state.get(
                                k(f"mk_{name}"), cfg.get("show_markers", False)
                            )
                        )
                        if show_markers:
                            marker_size = st.slider(
                                f"Marker size {name}", 3, 30,
                                cfg.get("marker_size", 8), key=k(f"ms_{name}")
                            )
                            marker_symbol = st.selectbox(
                                f"Symbol {name}",
                                [
                                    "circle", "square", "diamond", "cross",
                                    "x", "triangle-up"
                                ],
                                index=[
                                    "circle", "square", "diamond", "cross",
                                    "x", "triangle-up"
                                ].index(cfg.get("marker_symbol", "circle")),
                                key=k(f"sym_{name}"),
                            )
                            marker_opacity = st.slider(
                                f"Marker opacity {name}", 0.1, 1.0,
                                cfg.get("marker_opacity", 0.9),
                                key=k(f"mo_{name}"),
                            )
                        else:
                            marker_size = cfg.get("marker_size", 8)
                            marker_symbol = cfg.get("marker_symbol", "circle")
                            marker_opacity = cfg.get("marker_opacity", 0.9)
                    else:
                        line_w = cfg.get("line_width", 2)
                        line_dash = cfg.get("line_dash", "solid")
                        marker_size = cfg.get("marker_size", 8)
                        marker_symbol = cfg.get("marker_symbol", "circle")
                        marker_opacity = cfg.get("marker_opacity", 0.9)
                    new_cfg = {
                        "color": new_color,
                        "line_width": line_w,
                        "line_dash": line_dash,
                        "marker_size": marker_size,
                        "marker_symbol": marker_symbol,
                        "marker_opacity": marker_opacity,
                    }
                    if chart_type == "scatter":
                        new_cfg["mode"] = st.session_state.get(
                            k(f"mode_{name}"), cfg.get("mode", "markers")
                        )
                    if chart_type == "line":
                        new_cfg["show_markers"] = st.session_state.get(
                            k(f"mk_{name}"), cfg.get("show_markers", False)
                        )
                    new_cfg["display_name"] = st.session_state.get(
                        k(f"disp_{orig_name}"), cfg.get("display_name", name)
                    )
                    if new_cfg != cfg:
                        style_cfg[name] = new_cfg
                    if (
                        new_cfg["display_name"]
                        and new_cfg["display_name"] != orig_name
                    ):
                        disp_name = new_cfg["display_name"]
                        if disp_name not in style_cfg:
                            style_cfg[disp_name] = style_cfg.pop(orig_name)
                        else:
                            style_cfg[orig_name][
                                "display_name"
                            ] = disp_name
                        tr.name = disp_name
            if st.button("Reset styles", key=k("reset_styles_btn")):
                style_cfg.clear()
        else:
            st.caption("Generate the figure first.")

    # Close scroll box wrapper
    st.markdown("</div>", unsafe_allow_html=True)

    # custom_color_map already obtained; keep return signature

    # Persist style_cfg
    st.session_state["style_cfg"] = style_cfg
    # Persist the possibly mutated style_cfg back to the session (already ref)
    st.session_state[namespaced_key] = style_cfg
    return fmt_cfg, fig_style_cfg, style_cfg, custom_color_map


def sidebar_figure_style_section(container=st.sidebar):
    """Figure-level style overrides.
    Returns dict with keys:
    override_theme, fig_plot_bg, fig_paper_bg, fig_font_family,
    fig_font_size, grid_x, grid_y, grid_color, grid_width
    """
    with container.expander("Figure style", expanded=False):
        override_theme = st.checkbox(
            "Override theme in figure", True, key="fig_override_theme"
        )
        fig_plot_bg = st.color_picker(
            "Plot background", value="#ffffff", key="plot_bg_col"
        )
        fig_paper_bg = st.color_picker(
            "Outer background", value="#ffffff", key="paper_bg_col"
        )
        fig_font_family = st.selectbox(
            "Font family",
            [
                "Sans-Serif",
                "Arial",
                "Helvetica",
                "Times New Roman",
                "Courier New",
                "Monospace",
            ],
            index=0,
            key="fig_font_family",
        )
        fig_font_size = st.number_input(
            "Font size", 6, 40, 12, key="fig_font_size"
        )
        grid_x = st.checkbox("Show X grid", True, key="fig_grid_x")
        grid_y = st.checkbox("Show Y grid", True, key="fig_grid_y")
        grid_color = st.color_picker(
            "Grid color", value="#e0e0e0", key="fig_grid_color"
        )
        grid_width = st.slider("Grid width", 0, 5, 1, key="fig_grid_width")
        st.markdown("---")
        custom_size = st.checkbox(
            "Custom size (px)", False, key="fig_custom_size"
        )
        if custom_size:
            fig_width = st.slider(
                "Width (px)", 300, 3000, 900, step=10, key="fig_width"
            )
            fig_height = st.slider(
                "Height (px)", 200, 2000, 600, step=10, key="fig_height"
            )
        else:
            fig_width = None
            fig_height = None
    return dict(
        override_theme=override_theme,
        fig_plot_bg=fig_plot_bg,
        fig_paper_bg=fig_paper_bg,
        fig_font_family=fig_font_family,
        fig_font_size=fig_font_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_color=grid_color,
        grid_width=grid_width,
        custom_size=custom_size,
        fig_width=fig_width,
        fig_height=fig_height,
    )


def sidebar_custom_color_map_section(
    df: pd.DataFrame, color, container=st.sidebar
):
    """Categorical color map editor.

    Shown if the color column is categorical or has few distinct values.
    """
    custom_color_map = st.session_state.get("custom_color_map", {})
    if color and color in df.columns:
        col_series = df[color]
        nunique = col_series.nunique(dropna=True)
        if (not pd.api.types.is_numeric_dtype(col_series)) or nunique <= 20:
            with container.expander("Custom color map", expanded=False):
                cats = list(col_series.dropna().unique())[:50]
                updated = False
                for cat in cats:
                    default_col = custom_color_map.get(color, {}).get(
                        str(cat), "#1f77b4"
                    )
                    pick = st.color_picker(str(cat), value=default_col)
                    if color not in custom_color_map:
                        custom_color_map[color] = {}
                    if custom_color_map[color].get(str(cat)) != pick:
                        custom_color_map[color][str(cat)] = pick
                        updated = True
                if st.button("Reset color map"):
                    if color in custom_color_map:
                        del custom_color_map[color]
                        updated = True
                if updated:
                    st.session_state["custom_color_map"] = custom_color_map
    return custom_color_map


def sidebar_per_trace_style_section(
    fig, chart_type, style_cfg, container=st.sidebar
):
    """Per trace style editing (used only in per_trace mode)."""
    with container.expander("Per-trace style", expanded=False):
        if fig and getattr(fig, "data", None):
            for i, tr in enumerate(fig.data):
                name = tr.name or f"trace_{i}"
                orig_name = name
                cfg = style_cfg.get(name, {})
                with st.container():
                    # Editable name
                    disp_name = st.text_input(
                        f"Trace name {i+1}",
                        value=cfg.get("display_name", name),
                        key=f"disp_{orig_name}",
                    )
                    st.markdown(f"**{disp_name}**")
                    new_color = st.color_picker(
                        f"Color {name}",
                        value=cfg.get(
                            "color",
                            getattr(
                                getattr(tr, 'line', None), 'color', '#1f77b4'
                            )
                            or "#1f77b4",
                        ),
                        key=f"col_{name}",
                    )
                    if chart_type in ["line", "scatter"]:
                        if chart_type == "scatter":
                            st.selectbox(
                                f"Mode {name}",
                                ["markers", "lines", "lines+markers"],
                                index=[
                                    "markers",
                                    "lines",
                                    "lines+markers",
                                ].index(cfg.get("mode", "markers")),
                                key=f"mode_{name}",
                            )
                        else:
                            st.checkbox(
                                f"Show markers {name}",
                                value=cfg.get("show_markers", False),
                                key=f"mk_{name}",
                            )
                        line_w = st.slider(
                            f"Line width {name}",
                            1,
                            12,
                            cfg.get("line_width", 2),
                            key=f"lw_{name}",
                        )
                        line_dash = st.selectbox(
                            f"Line style {name}",
                            ["solid", "dash", "dot", "dashdot"],
                            index=[
                                "solid",
                                "dash",
                                "dot",
                                "dashdot",
                            ].index(cfg.get("line_dash", "solid")),
                            key=f"ld_{name}",
                        )
                        show_markers = (
                            chart_type == "scatter"
                            and "markers"
                            in (
                                st.session_state.get(
                                    f"mode_{name}", cfg.get("mode", "markers")
                                )
                            )
                        ) or (
                            chart_type == "line"
                            and st.session_state.get(
                                f"mk_{name}", cfg.get("show_markers", False)
                            )
                        )
                        if show_markers:
                            marker_size = st.slider(
                                f"Marker size {name}",
                                3,
                                30,
                                cfg.get("marker_size", 8),
                                key=f"ms_{name}",
                            )
                            marker_symbol = st.selectbox(
                                f"Symbol {name}",
                                [
                                    "circle",
                                    "square",
                                    "diamond",
                                    "cross",
                                    "x",
                                    "triangle-up",
                                ],
                                index=[
                                    "circle",
                                    "square",
                                    "diamond",
                                    "cross",
                                    "x",
                                    "triangle-up",
                                ].index(cfg.get("marker_symbol", "circle")),
                                key=f"sym_{name}",
                            )
                            marker_opacity = st.slider(
                                f"Marker opacity {name}",
                                0.1,
                                1.0,
                                cfg.get("marker_opacity", 0.9),
                                key=f"mo_{name}",
                            )
                        else:
                            marker_size = cfg.get("marker_size", 8)
                            marker_symbol = cfg.get("marker_symbol", "circle")
                            marker_opacity = cfg.get("marker_opacity", 0.9)
                    else:
                        line_w = cfg.get("line_width", 2)
                        line_dash = cfg.get("line_dash", "solid")
                        marker_size = cfg.get("marker_size", 8)
                        marker_symbol = cfg.get("marker_symbol", "circle")
                        marker_opacity = cfg.get("marker_opacity", 0.9)
                    new_cfg = {
                        "color": new_color,
                        "line_width": line_w,
                        "line_dash": line_dash,
                        "marker_size": marker_size,
                        "marker_symbol": marker_symbol,
                        "marker_opacity": marker_opacity,
                    }
                    if chart_type == "scatter":
                        new_cfg["mode"] = st.session_state.get(
                            f"mode_{name}", cfg.get("mode", "markers")
                        )
                    if chart_type == "line":
                        new_cfg["show_markers"] = st.session_state.get(
                            f"mk_{name}", cfg.get("show_markers", False)
                        )
                    new_cfg["display_name"] = disp_name
                    if new_cfg != cfg:
                        style_cfg[name] = new_cfg
                    # Rename key and trace if name changed
                    if disp_name and disp_name != orig_name:
                        if disp_name not in style_cfg:
                            style_cfg[disp_name] = style_cfg.pop(orig_name)
                        else:
                            # Collision: keep original but update display_name
                            style_cfg[orig_name]["display_name"] = disp_name
                        tr.name = disp_name
            if st.button("Reset styles"):
                style_cfg.clear()
        else:
            st.caption("Generate the figure first.")
    return style_cfg


__all__ += [
    "sidebar_transform_section",
    "sidebar_chart_section",
    "sidebar_fit_section",
    "sidebar_format_section",
    "sidebar_style_group",
    "sidebar_analysis_section",
    "sidebar_figure_style_section",
    "sidebar_custom_color_map_section",
    "sidebar_per_trace_style_section",
]
