import numpy as np
import pandas as pd
import streamlit as st
from originlite.charts import make_chart
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
from originlite.ui.helpers import load_data_sidebar
from originlite.themes import THEMES, set_theme
from originlite.viz.builder import combine_subplots
from originlite.ui.sections import (
    sidebar_transform_section,
    sidebar_chart_section,
    sidebar_analysis_section,
    sidebar_style_group,
    sidebar_subplots_edit_section,
)
from originlite.ui.sections_export import export_sections
import base64

st.set_page_config(page_title="OriginLite", layout="wide")

# Clear caches & initialize only once per user session (survives code reloads)
if 'startup_initialized' not in st.session_state:
    with st.spinner('Initializing...'):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
    # Keys to clear (exclude x_sel / y_sel so choices persist)
        _startup_keys = [
            'data_upload', 'load_mode', 'proj_select', 'data_model',
            'data_model_source', 'formula_columns', 'loaded_template',
            'chart_type_sel', 'chart_type_sel_v2', 'last_x_sel', 'last_y_sel',
            'multi_y_sel', 'overlay_multi', 'subplot_orientation', 'color_sel',
            'size_sel', 'symbol_sel', 'facet_sel', 'z_sel', 'bins_slider',
            'baseline_cols_mod', 'peak_detect_cols', 'formula_expr_mod',
            'formula_out_mod', 'default_peak_cols', 'style_cfg',
            'custom_color_map', 'fig_override_theme', 'plot_bg_col',
            'paper_bg_col', 'fig_font_family',
            'fig_font_size', 'fig_grid_x', 'fig_grid_y', 'fig_grid_color',
            'fig_grid_width', 'fig_custom_size', 'fig_width', 'fig_height',
            'label_peaks',
        ]
        for _k in _startup_keys:
            st.session_state.pop(_k, None)
        if 'style_cfg' not in st.session_state:
            st.session_state['style_cfg'] = {}
        st.session_state['startup_initialized'] = True
        st.sidebar.info("Startup: initialized (session persistent)")

"""Sidebar brand header with fixed logo (user cannot change)."""
FIXED_LOGO_URL = (
    "https://www.kindpng.com/picc/m/294-2945675_please-help-with-"
    "installing-the-software-origin-origin.png"
)
brand_ct = st.sidebar.container()
with brand_ct:
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;margin-bottom:0.75rem;'>
          <img src='{FIXED_LOGO_URL}' alt='Logo'
              style='height:36px;max-height:36px;margin-right:0.5rem;' />
            <h1 style='font-size:1.6rem;margin:0;'>OriginLite</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
# Theme selector hidden: using the first theme by default.
theme = list(THEMES.keys())[0]
set_theme(theme)

# Stable placeholder container for Export UI (pre-created to avoid flicker)
export_sidebar_ct = st.sidebar.container()

# (Removed sidebar data preview controls; rows selector moved to main area)

# Global logo section removed – now configured per plot.

dm = load_data_sidebar()
df = dm.df

# Manage multiple plot contexts
if 'plot_contexts' not in st.session_state:
    # Each context: {id, prefix}
    st.session_state['plot_contexts'] = [
        {'id': 1, 'prefix': 'p1_'},
    ]
    st.session_state['next_plot_id'] = 2

# UI to add/remove plot contexts (top area)
with st.expander("Plot managers", expanded=False):
    cols_m = st.columns([1, 1, 6])
    with cols_m[0]:
        if st.button("Add plot", key="add_plot_btn"):
            pid = st.session_state['next_plot_id']
            st.session_state['plot_contexts'].append(
                {'id': pid, 'prefix': f"p{pid}_"}
            )
            st.session_state['next_plot_id'] += 1
    with cols_m[1]:
        if len(st.session_state['plot_contexts']) > 1:
            # Select specific plot to remove
            removable_ids = [
                c['id'] for c in st.session_state['plot_contexts']
            ]
            rem_id = st.selectbox(
                "Plot to remove",
                removable_ids,
                key="remove_plot_select",
            )
            if st.button("Remove selected", key="rem_plot_btn"):
                # Prevent removing if it would leave zero plots
                new_list = [
                    c for c in st.session_state['plot_contexts']
                    if c['id'] != rem_id
                ]
                # Identify removed (should be exactly one)
                removed = [
                    c for c in st.session_state['plot_contexts']
                    if c['id'] == rem_id
                ]
                if removed and new_list:
                    pref = removed[0]['prefix']
                    st.session_state['plot_contexts'] = new_list
                    # Clean keys for that prefix
                    for k in list(st.session_state.keys()):
                        if k.startswith(pref):
                            st.session_state.pop(k, None)
                elif not new_list:
                    st.warning("Cannot remove the last remaining plot.")

# Build tab labels dynamically
plot_labels = [f"Plot {c['id']}" for c in st.session_state['plot_contexts']]
data_tab, *plot_tabs = st.tabs(["Data", *plot_labels])

# Collect per-plot configs (reinitialized once per rerun)
per_plot_exports = []

# Auto-select defaults for x / y if missing and numeric columns exist
# (Do this BEFORE chart_cfg is read to ensure proper initialization)
if len(df.columns) > 0:
    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    if numeric_cols:
        # Handle legacy (unprefixed) single-plot state for backward compat
        x_current = st.session_state.get('x_sel')
        y_current = st.session_state.get('y_sel')
        multi_y_current = st.session_state.get('multi_y_sel', [])
        if not multi_y_current and (x_current is None or y_current is None):
            changed = False
            if x_current is None:
                st.session_state['x_sel'] = numeric_cols[0]
                changed = True
            if y_current is None:
                y_candidates = [
                    c for c in numeric_cols
                    if c != st.session_state.get('x_sel')
                ] or numeric_cols
                if y_candidates:
                    st.session_state['y_sel'] = y_candidates[0]
                    changed = True
            if changed:
                x_auto = st.session_state.get('x_sel')
                y_auto = st.session_state.get('y_sel')
                st.info(
                    f"Auto-selected X='{x_auto}' Y='{y_auto}' "
                    f"(first numeric cols). Change in Chart section."
                )
        # New: per-plot (prefixed) auto-selection for each plot context
        for ctx in st.session_state['plot_contexts']:
            pref = ctx['prefix']
            px_key = f"{pref}x_sel"
            py_key = f"{pref}y_sel"
            pmulti_key = f"{pref}multi_y_sel"
            x_cur = st.session_state.get(px_key)
            y_cur = st.session_state.get(py_key)
            multi_cur = st.session_state.get(pmulti_key, [])
            if not multi_cur and (x_cur is None or y_cur is None):
                changed = False
                if x_cur is None:
                    st.session_state[px_key] = numeric_cols[0]
                    changed = True
                if y_cur is None:
                    y_candidates = [
                        c for c in numeric_cols
                        if c != st.session_state.get(px_key)
                    ] or numeric_cols
                    if y_candidates:
                        st.session_state[py_key] = y_candidates[0]
                        changed = True
                if changed:
                    x_auto = st.session_state.get(px_key)
                    y_auto = st.session_state.get(py_key)
                    st.info(
                        f"[{pref.rstrip('_')}] Auto-selected X='{x_auto}' "
                        f"Y='{y_auto}' (first numeric cols)."
                    )

per_plot_exports = []  # Collect per-plot configs for export
for (ctx, tab) in zip(st.session_state['plot_contexts'], plot_tabs):
    prefix = ctx['prefix']
    with tab:
        col_cfg, col_style = st.columns([1, 1])
        with col_cfg:
            chart_cfg = sidebar_chart_section(
                df, container=col_cfg, key_prefix=prefix
            )
            analysis_cfg = sidebar_analysis_section(
                bool(chart_cfg['multi_y']),
                df,
                chart_cfg.get('x'),
                chart_cfg.get('y'),
                container=col_cfg,
                key_prefix=prefix,
            )
            show_debug = st.checkbox(
                "Show debug info",
                value=False,
                key=f"{prefix}show_debug_main",
            )
            if show_debug:
                with st.expander("Debug", expanded=True):
                    _ct = (
                        st.session_state.get(f'{prefix}chart_type_sel_v2')
                        or st.session_state.get(f'{prefix}chart_type_sel')
                    )
                    _xv = st.session_state.get(f'{prefix}x_sel')
                    _yv = st.session_state.get(f'{prefix}y_sel')
                    _mv = st.session_state.get(f'{prefix}multi_y_sel')
                    st.caption(f"type={_ct} | x={_xv}")
                    st.caption(f"y={_yv} | multi={_mv}")
        style_column_container = col_style

        # Per-plot logo configuration (after analysis, before build)
        with col_style.expander("Logo", expanded=False):
            lp_key = f"{prefix}logo_position"
            lu_key = f"{prefix}logo_upload"
            lw_key = f"{prefix}logo_width_pct"
            lo_key = f"{prefix}logo_opacity"
            lb_key = f"{prefix}logo_bytes"
            lm_key = f"{prefix}logo_mime"
            logo_file = st.file_uploader(
                "Upload logo image",
                type=["png", "jpg", "jpeg", "gif"],
                key=lu_key,
                help="Optional logo to display on this figure only.",
            )
            logo_position = st.selectbox(
                "Position",
                [
                    "None",
                    "Top-left",
                    "Top-right",
                    "Bottom-left",
                    "Bottom-right",
                ],
                index=0,
                key=lp_key,
            )
            logo_width_pct = st.slider(
                "Width (% of figure)", 5, 40, 15, key=lw_key
            )
            logo_opacity = st.slider(
                "Opacity", 0.1, 1.0, 1.0, 0.05, key=lo_key
            )
            if logo_file is not None:
                st.image(
                    logo_file, caption="Logo preview", use_column_width=False
                )
                st.session_state[lb_key] = logo_file.getvalue()
                st.session_state[lm_key] = logo_file.type or 'image/png'
            else:
                if st.button(
                    "Clear stored logo", key=f"{prefix}clear_logo_btn"
                ):
                    st.session_state.pop(lb_key, None)
                    st.session_state.pop(lm_key, None)

        # Unpack chart config per plot
        ctype = chart_cfg['chart_type']
        x = chart_cfg['x']
        y = chart_cfg['y']
        y_err = chart_cfg.get('y_err')
        x_err = chart_cfg.get('x_err')
        per_trace_err = chart_cfg.get('per_trace_errors', {})
        multi_y = chart_cfg['multi_y']
        # Normalize: if user picked a single Y and also a multi_y list
        # ensure the single Y is included for plotting consistency.
        if y and multi_y and y not in multi_y:
            multi_y = [y] + list(multi_y)
        overlay_multi = chart_cfg.get('overlay_multi')
        sub_orient = chart_cfg.get('subplot_orientation')
        color = chart_cfg['color']
        size = chart_cfg['size']
        symbol = chart_cfg['symbol']
        facet = chart_cfg['facet']
        z = chart_cfg['z']
        bins = chart_cfg['bins']
        fit_kind = analysis_cfg['fit_kind']
        custom_expr = analysis_cfg['custom_expr']
        custom_params = analysis_cfg['custom_params']
        label_peaks = analysis_cfg['label_peaks']

        # Subplot configuration
        is_subplots_mode = bool(
            multi_y and not overlay_multi and len(multi_y) > 1
        )
        subplot_cfg = {}
        if is_subplots_mode:
            subplot_cfg = sidebar_subplots_edit_section(
                multi_y,
                sub_orient or 'vertical',
                container=col_cfg,
                key_prefix=prefix,
            )
        # Peak defaults (namespaced)
        st.session_state[f'{prefix}default_peak_cols'] = (
            multi_y if multi_y else ([y] if y else [])
        )

        # Build figure
        fig_local = None
        # Wrap full build in try to avoid breaking rest of UI
        try:
            if multi_y:
                if overlay_multi and ctype in {"line", "scatter", "bar"}:
                    base_y = multi_y[0]
                    fig_local = make_chart(
                        df,
                        ctype,
                        x,
                        base_y,
                        error_y=per_trace_err.get(base_y, y_err),
                        error_x=x_err,
                        color=None if color in multi_y else color,
                        size=size,
                        symbol=symbol,
                        facet_col=facet,
                        bins=bins,
                        z=z,
                    )
                    for yi in multi_y[1:]:
                        try:
                            tfig = make_chart(
                                df,
                                ctype,
                                x,
                                yi,
                                error_y=per_trace_err.get(yi, y_err),
                                error_x=x_err,
                                color=(
                                    color
                                    if (color and color not in multi_y)
                                    else None
                                ),
                                size=size,
                                symbol=symbol,
                                facet_col=facet,
                                bins=bins,
                                z=z,
                            )
                            for tr in tfig.data:
                                if not getattr(tr, 'name', None):
                                    tr.name = yi
                                fig_local.add_trace(tr)
                        except Exception as e:  # noqa: BLE001
                            st.warning(f"Overlay error {yi}: {e}")
                else:
                    figs_local = []
                    for yi in multi_y:
                        try:
                            figs_local.append(
                                make_chart(
                                    df,
                                    ctype,
                                    x,
                                    yi,
                                    error_y=per_trace_err.get(yi, y_err),
                                    error_x=x_err,
                                    color=color,
                                    size=size,
                                    symbol=symbol,
                                    facet_col=facet,
                                    bins=bins,
                                    z=z,
                                )
                            )
                        except Exception as e:  # noqa: BLE001
                            st.warning(f"Chart error {yi}: {e}")
                    if figs_local:
                        try:
                            orientation_use = sub_orient or 'vertical'
                            fig_local = combine_subplots(
                                figs_local, orientation=orientation_use
                            )
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Subplot build error: {e}")
                            fig_local = figs_local[0]
                    else:
                        st.info("No valid subplots produced.")
            else:
                if x and y:
                    fig_local = make_chart(
                        df,
                        ctype,
                        x,
                        y,
                        error_y=y_err,
                        error_x=x_err,
                        color=color,
                        size=size,
                        symbol=symbol,
                        facet_col=facet,
                        bins=bins,
                        z=z,
                    )
        except Exception as e:  # noqa: BLE001
            st.error(f"Build error: {e}")
            fig_local = None

        # Fallback: figure might not build (silent failure upstream).
        if fig_local is None and x and y and not multi_y:
            try:
                fig_local = make_chart(
                    df,
                    ctype,
                    x,
                    y,
                    error_y=y_err,
                    error_x=x_err,
                    color=color,
                    size=size,
                    symbol=symbol,
                    facet_col=facet,
                    bins=bins,
                    z=z,
                )
                st.info(
                    "Recovered chart via fallback build (single Y). "
                    "If this message persists, report steps to reproduce."
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Fallback build failed: {e}")

    # Style section per plot (applies formatting and overlays
    # including logo)
        if fig_local is not None:
            (
                fmt_cfg,
                fig_style_cfg,
                style_cfg,
                custom_color_map,
            ) = sidebar_style_group(
                df, color, fig_local, ctype,
                container=style_column_container, key_prefix=prefix
            )
            # Extract formatting configs
            chart_title = fmt_cfg.get('chart_title')
            title_position = fmt_cfg.get('title_position', 'top-center')
            x_title = fmt_cfg.get('x_title') or (x or '')
            y_title = fmt_cfg.get('y_title') or (y or '')
            tick_format_x = fmt_cfg.get('tick_format_x') or None
            tick_format_y = fmt_cfg.get('tick_format_y') or None
            tick_angle_x = fmt_cfg.get('tick_angle_x', 0)
            tick_angle_y = fmt_cfg.get('tick_angle_y', 0)
            nticks_x = fmt_cfg.get('nticks_x', 0)
            nticks_y = fmt_cfg.get('nticks_y', 0)
            legend_show = fmt_cfg.get('legend_show', True)
            legend_orientation = fmt_cfg.get('legend_orientation', 'v')
            legend_position = fmt_cfg.get('legend_position', 'top-right')
            lx = fmt_cfg.get('lx', 1)
            ly = fmt_cfg.get('ly', 1)
            force_x_ticks = fmt_cfg.get('force_x_ticks')
            force_y_ticks = fmt_cfg.get('force_y_ticks')

            override_theme = fig_style_cfg.get('override_theme', True)
            fig_plot_bg = fig_style_cfg.get('fig_plot_bg', '#ffffff')
            fig_paper_bg = fig_style_cfg.get('fig_paper_bg', '#ffffff')
            fig_font_family = fig_style_cfg.get(
                'fig_font_family', 'Sans-Serif'
            )
            fig_font_size = fig_style_cfg.get('fig_font_size', 12)
            grid_x = fig_style_cfg.get('grid_x', True)
            grid_y = fig_style_cfg.get('grid_y', True)
            grid_color = fig_style_cfg.get('grid_color', '#e0e0e0')
            grid_width = fig_style_cfg.get('grid_width', 1)
            custom_size = fig_style_cfg.get('custom_size')
            fig_width = fig_style_cfg.get('fig_width')
            fig_height = fig_style_cfg.get('fig_height')

            # Legend orientation mapping
            legend_orientation_val = 'h' if legend_orientation == 'h' else 'v'
            # Title position mapping
            pos_map = {
                'top-left': (0.0, 'left'),
                'top-center': (0.5, 'center'),
                'top-right': (1.0, 'right'),
            }
            x_pos, x_anchor = pos_map.get(title_position, (0.5, 'center'))
            title_cfg = (
                dict(text=chart_title, x=x_pos, xanchor=x_anchor)
                if chart_title
                else None
            )
            fig_local.update_layout(
                title=title_cfg,
                showlegend=legend_show,
                legend=dict(
                    orientation=legend_orientation_val,
                    x=lx,
                    y=ly,
                    xanchor='right' if lx == 1 else 'left',
                    yanchor='bottom' if ly == 0 else 'top',
                ),
            )
            # dtick calculations
            dtick_x = None
            if (
                force_x_ticks
                and nticks_x
                and nticks_x > 1
                and x
                and x in df.columns
            ):
                try:
                    xv = pd.to_numeric(df[x], errors='coerce')
                    rng = xv.max() - xv.min()
                    if pd.notna(rng) and rng > 0:
                        dtick_x = rng / (nticks_x - 1)
                except Exception:
                    pass
            dtick_y = None
            if (
                force_y_ticks
                and nticks_y
                and nticks_y > 1
                and y
                and y in df.columns
                and (not multi_y or len(multi_y) == 1)
            ):
                try:
                    yv = pd.to_numeric(df[y], errors='coerce')
                    rngy = yv.max() - yv.min()
                    if pd.notna(rngy) and rngy > 0:
                        dtick_y = rngy / (nticks_y - 1)
                except Exception:
                    pass

            xaxis_common = dict(
                tickformat=tick_format_x,
                tickangle=tick_angle_x,
                nticks=None
                if (nticks_x == 0 or dtick_x is not None)
                else nticks_x,
                dtick=dtick_x,
                tickmode='linear' if dtick_x is not None else None,
                showgrid=grid_x if override_theme else None,
                gridcolor=grid_color if override_theme and grid_x else None,
                gridwidth=grid_width if override_theme and grid_x else None,
            )
            if not is_subplots_mode:
                xaxis_common['title_text'] = x_title
            fig_local.update_xaxes(**xaxis_common)

            yaxis_common = dict(
                tickformat=tick_format_y,
                tickangle=tick_angle_y,
                nticks=None
                if (nticks_y == 0 or dtick_y is not None)
                else nticks_y,
                dtick=dtick_y,
                tickmode='linear' if dtick_y is not None else None,
                showgrid=grid_y if override_theme else None,
                gridcolor=grid_color if override_theme and grid_y else None,
                gridwidth=grid_width if override_theme and grid_y else None,
            )
            if not is_subplots_mode:
                yaxis_common['title_text'] = y_title
            fig_local.update_yaxes(**yaxis_common)

            # --- Per-plot fitting (was legacy global) ---
            fit_result = None
            if (
                fit_kind != 'None'
                and x
                and y
                and (not multi_y or len(multi_y) == 1)
                and x in df.columns
                and y in df.columns
            ):
                try:
                    xvals = pd.to_numeric(df[x], errors='coerce')
                    yvals = pd.to_numeric(df[y], errors='coerce')
                    mask = xvals.notna() & yvals.notna()
                    xvals = xvals[mask]
                    yvals = yvals[mask]
                    fit = None
                    if fit_kind == 'Linear':
                        fit = fit_linear(xvals, yvals)
                    elif fit_kind.startswith('Poly'):
                        deg = int(fit_kind.split()[1])
                        fit = fit_poly(xvals, yvals, deg=deg)
                    elif fit_kind == 'Exponential':
                        fit = fit_exponential(xvals, yvals)
                    elif fit_kind == 'Power law':
                        fit = fit_powerlaw(xvals, yvals)
                    elif fit_kind == 'Langmuir':
                        fit = fit_langmuir(xvals, yvals)
                    elif fit_kind == 'Freundlich':
                        fit = fit_freundlich(xvals, yvals)
                    elif fit_kind == 'Redlich-Peterson':
                        fit = fit_redlich_peterson(xvals, yvals)
                    elif (
                        fit_kind == 'Custom'
                        and custom_expr
                        and custom_params
                    ):
                        fit = fit_generic(
                            xvals,
                            yvals,
                            expr=custom_expr,
                            param_names=custom_params,
                        )
                    if fit is not None:
                        order = np.argsort(xvals.values)
                        xs = xvals.values[order]
                        ys = fit.y_fit.values[order]
                        label_eq = fit.equation or fit.model
                        # Legend: first line model name, second line equation
                        # with numeric values
                        fig_local.add_scatter(
                            x=xs,
                            y=ys,
                            mode='lines',
                            name=f"{fit.model} fit<br>{label_eq}",
                        )
                        band = 1.96 * fit.stderr
                        fig_local.add_scatter(
                            x=xs,
                            y=ys + band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip',
                            name='upper',
                        )
                        fig_local.add_scatter(
                            x=xs,
                            y=ys - band,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(0,0,0,0.1)',
                            showlegend=False,
                            hoverinfo='skip',
                            name='lower',
                        )
                        st.session_state[f'{prefix}fit_result'] = fit
                    else:
                        st.warning(
                            f"Fit '{fit_kind}' failed for this plot.",
                            icon='⚠️',
                        )
                except Exception as _e:  # noqa: BLE001
                    st.warning(f"Fit error: {_e}")

            # Figure style overrides
            if override_theme:
                fig_local.update_layout(
                    plot_bgcolor=fig_plot_bg,
                    paper_bgcolor=fig_paper_bg,
                    font=dict(family=fig_font_family, size=fig_font_size),
                )
            if custom_size and (fig_width or fig_height):
                fig_local.update_layout(
                    width=fig_width if fig_width else None,
                    height=fig_height if fig_height else None,
                    autosize=False,
                )

            # Apply per-trace style overrides if present
            if fig_local.data and style_cfg:
                for i, tr in enumerate(fig_local.data):
                    name = tr.name or f"trace_{i}"
                    cfg = style_cfg.get(name)
                    if not cfg:
                        continue
                    if hasattr(tr, 'marker'):
                        if 'marker_size' in cfg and hasattr(tr.marker, 'size'):
                            try:
                                tr.marker.size = cfg['marker_size']
                            except Exception:
                                pass
                        if 'color' in cfg and hasattr(tr.marker, 'color'):
                            try:
                                tr.marker.color = cfg['color']
                            except Exception:
                                pass
                        if 'marker_symbol' in cfg and hasattr(
                            tr.marker, 'symbol'
                        ):
                            try:
                                tr.marker.symbol = cfg['marker_symbol']
                            except Exception:
                                pass
                        if 'marker_opacity' in cfg and hasattr(
                            tr.marker, 'opacity'
                        ):
                            try:
                                tr.marker.opacity = cfg['marker_opacity']
                            except Exception:
                                pass
                    if hasattr(tr, 'line'):
                        if 'line_width' in cfg and hasattr(tr.line, 'width'):
                            try:
                                tr.line.width = cfg['line_width']
                            except Exception:
                                pass
                        if 'line_dash' in cfg and hasattr(tr.line, 'dash'):
                            try:
                                tr.line.dash = cfg['line_dash']
                            except Exception:
                                pass
                        if 'line_color' in cfg and hasattr(tr.line, 'color'):
                            try:
                                tr.line.color = cfg['line_color']
                            except Exception:
                                pass
            # Apply per-plot logo overlay (if configured earlier)
            lb_key = f"{prefix}logo_bytes"
            lm_key = f"{prefix}logo_mime"
            lp_key = f"{prefix}logo_position"
            lw_key = f"{prefix}logo_width_pct"
            lo_key = f"{prefix}logo_opacity"
            logo_bytes = st.session_state.get(lb_key)
            logo_position = st.session_state.get(lp_key, 'None')
            logo_width_pct = st.session_state.get(lw_key, 15)
            logo_opacity = st.session_state.get(lo_key, 1.0)
            if logo_bytes and logo_position and logo_position != 'None':
                try:
                    mime = st.session_state.get(lm_key, 'image/png')
                    b64 = base64.b64encode(logo_bytes).decode('utf-8')
                    src = f"data:{mime};base64,{b64}"
                    pos_map = {
                        'Top-left': dict(
                            x=0, y=1, xanchor='left', yanchor='top'
                        ),
                        'Top-right': dict(
                            x=1, y=1, xanchor='right', yanchor='top'
                        ),
                        'Bottom-left': dict(
                            x=0, y=0, xanchor='left', yanchor='bottom'
                        ),
                        'Bottom-right': dict(
                            x=1, y=0, xanchor='right', yanchor='bottom'
                        ),
                    }
                    p = pos_map.get(logo_position)
                    if p:
                        width_frac = max(
                            0.01, min(0.95, logo_width_pct / 100.0)
                        )
                        img_dict = dict(
                            source=src,
                            xref='paper',
                            yref='paper',
                            x=p['x'],
                            y=p['y'],
                            sizex=width_frac,
                            sizey=width_frac,
                            xanchor=p['xanchor'],
                            yanchor=p['yanchor'],
                            sizing='contain',
                            opacity=logo_opacity,
                            layer='above',
                        )
                        existing_images = list(
                            getattr(fig_local.layout, 'images', []) or []
                        )
                        filtered = [
                            im for im in existing_images
                            if not (
                                hasattr(im, 'source')
                                and isinstance(im.source, str)
                                and im.source.startswith('data:image')
                            )
                        ]
                        filtered.append(img_dict)
                        fig_local.update_layout(images=filtered)
                except Exception as _e:  # noqa: BLE001
                    st.warning(f"Logo overlay error: {_e}")
            # Minimal layout application (title + basic legend show)
            if fmt_cfg.get('chart_title'):
                pos_map = {
                    'top-left': (0.0, 'left'),
                    'top-center': (0.5, 'center'),
                    'top-right': (1.0, 'right'),
                }
                xp, xa = pos_map.get(
                    fmt_cfg.get('title_position', 'top-center'),
                    (0.5, 'center'),
                )
                fig_local.update_layout(
                    title=dict(text=fmt_cfg['chart_title'], x=xp, xanchor=xa)
                )
            st.plotly_chart(
                fig_local,
                use_container_width=not fig_style_cfg.get('custom_size'),
                key=f"{prefix}plot_fig",
            )
        else:
            if x and y and not multi_y:
                st.warning(
                    "Figure build returned None despite valid X/Y. "
                    "(Enable 'Show debug info' to inspect state.)"
                )
            else:
                st.info("Select chart mappings to render a figure.")

        per_plot_exports.append(
            dict(
                prefix=prefix,
                chart_type=ctype,
                x=x,
                y=y,
                multi_y=multi_y,
                color=color,
                size=size,
                symbol=symbol,
                facet=facet,
                z=z,
                bins=bins,
                y_err=y_err,
                x_err=x_err,
                per_trace_errors=per_trace_err,
                fit_kind=fit_kind,
            )
        )

    # Track previous chart types per prefix to allow targeted clearing
for ctx in st.session_state['plot_contexts']:
    pref = ctx['prefix']
    prev_key = f"{pref}_prev_chart_type"
    cur_type = (
        st.session_state.get(f'{pref}chart_type_sel_v2')
        or st.session_state.get(f'{pref}chart_type_sel')
    )
    prev_val = st.session_state.get(prev_key)
    if prev_val is not None and prev_val != cur_type:
        for k in ['_cached_fig']:
            st.session_state.pop(f"{pref}{k}", None)
    st.session_state[prev_key] = cur_type

# Export pane uses last plot's variables (backwards compatible)
"""Legacy equation->legend block removed (handled per-plot during fit)."""

with data_tab:
    st.markdown("### Data preview")
    # Move transform section into Data tab
    sidebar_transform_section(dm, container=data_tab)
    # Rows selector
    cols_prev = st.columns([2, 2, 8])
    with cols_prev[0]:
        st.number_input(
            "Rows to show",
            min_value=1,
            max_value=5000,
            value=st.session_state.get('preview_rows', 5),
            key='preview_rows',
            help="Select how many rows to display in data preview tables.",
        )
    prev_n = st.session_state.get('preview_rows', 5)
    st.dataframe(df.head(prev_n))
    if len(df) > 0:
        st.markdown("### Relevant statistics:")
        st.dataframe(df.describe())
    # DataFrame download (moved from export section per request)
    csv_data = dm.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download DataFrame (CSV)",
        data=csv_data,
        file_name="data.csv",
        mime="text/csv",
        help=(
            "Download the current DataFrame with all columns including "
            "calculated ones."
        ),
    )

# Shared legacy rendering loop removed (each plot rendered in its tab)

if per_plot_exports:
    last_exp = per_plot_exports[-1]
    # Minimal figure style snapshot from last plot (keys guarded by .get)
    figure_style_cfg = {
        "override_theme": True,
        "plot_bg": '#ffffff',
        "paper_bg": '#ffffff',
        "font_family": 'Sans-Serif',
        "font_size": 12,
        "grid_x": True,
        "grid_y": True,
        "grid_color": '#e0e0e0',
        "grid_width": 1,
        "custom_size": False,
        "fig_width": None,
        "fig_height": None,
    }
    with export_sidebar_ct:
        export_sections(
            dm,
            theme,
            last_exp['chart_type'],
            last_exp['x'],
            last_exp['y'],
            last_exp['y_err'],
            last_exp['x_err'],
            last_exp['per_trace_errors'],
            last_exp['multi_y'],
            last_exp['color'],
            last_exp['size'],
            last_exp['symbol'],
            last_exp['facet'],
            last_exp['z'],
            last_exp['bins'],
            last_exp['fit_kind'],
            '',  # chart_title (per-plot handled already)
            '',  # x_title
            '',  # y_title
            '',  # tick_format_x
            '',  # tick_format_y
            0,   # tick_angle_x
            0,   # tick_angle_y
            True,  # legend_show
            'v',   # legend_orientation
            'top-right',  # legend_position
            False,  # label_peaks placeholder
            {},  # style_cfg placeholder
            {},  # custom_color_map placeholder
            figure_style_cfg,
            container=export_sidebar_ct,
        )
# Consolidated reset tools (apply to all plot contexts)
with st.expander("Reset & Cache (all plots)", expanded=False):
    st.caption("Reset all plotting/styling or clear session & caches.")
    if st.button("Reset ALL settings", key="global_reset_btn"):
        format_keys = [
            "chart_title", "title_position", "x_title", "y_title",
            "tick_fmt_x_choice", "tick_fmt_x", "tick_fmt_y_choice",
            "tick_fmt_y", "tick_angle_x", "tick_angle_y",
            "nticks_x", "nticks_y", "force_x_ticks", "force_y_ticks",
            "legend_show", "legend_orientation", "legend_position",
            "label_peaks",
        ]
        style_keys = [
            "style_mode", "palette_name", "apply_palette", "style_cfg",
            "custom_color_map",
        ]
        fig_style_keys = [
            "fig_override_theme", "plot_bg_col", "paper_bg_col",
            "fig_font_family", "fig_font_size", "fig_grid_x", "fig_grid_y",
            "fig_grid_color", "fig_grid_width", "fig_custom_size", "fig_width",
            "fig_height",
        ]
        chart_map_keys = [
            "chart_type_sel", "x_sel", "y_sel", "last_x_sel", "last_y_sel",
            "multi_y_sel", "overlay_multi", "subplot_orientation", "color_sel",
            "size_sel", "symbol_sel", "facet_sel", "z_sel", "bins_slider",
        ]
        transform_keys = [
            "filtcol_mod", "macol_mod", "baseline_cols_mod",
            "peak_detect_cols", "formula_expr_mod", "formula_out_mod",
        ]
        misc_keys = ["peaks_multi", "default_peak_cols"]
        # Include prefixed versions
        prefixes = [c['prefix'] for c in st.session_state['plot_contexts']]

        def prefixed(keys):
            out = []
            for p in prefixes:
                out.extend([f"{p}{k}" for k in keys])
            return out

        for k in (
            format_keys
            + style_keys
            + fig_style_keys
            + chart_map_keys
            + transform_keys
            + misc_keys
        ):
            st.session_state.pop(k, None)
        for k in prefixed(
            format_keys
            + style_keys
            + fig_style_keys
            + chart_map_keys
            + transform_keys
            + misc_keys
        ):
            st.session_state.pop(k, None)
        for _k in list(st.session_state.keys()):
            if _k.startswith('err_'):
                st.session_state.pop(_k, None)
        # Reset global style configs and any per-plot ones
        for _k in list(st.session_state.keys()):
            if _k.endswith('style_cfg'):
                st.session_state.pop(_k, None)
        st.session_state['style_cfg'] = {}
        st.success("Settings reset. Reselect data if needed.")
    if st.button("Hard reset (clear cache)", key="hard_reset_btn"):
        keys_to_clear = [
            "chart_title", "title_position", "x_title", "y_title",
            "tick_fmt_x_choice", "tick_fmt_x", "tick_fmt_y_choice",
            "tick_fmt_y", "tick_angle_x", "tick_angle_y", "nticks_x",
            "nticks_y", "force_x_ticks", "force_y_ticks", "legend_show",
            "legend_orientation", "legend_position", "label_peaks",
            "style_mode", "palette_name", "apply_palette", "style_cfg",
            "custom_color_map", "fig_override_theme", "plot_bg_col",
            "paper_bg_col", "fig_font_family", "fig_font_size",
            "fig_grid_x", "fig_grid_y", "fig_grid_color",
            "fig_grid_width", "fig_custom_size", "fig_width",
            "fig_height", "chart_type_sel", "x_sel", "y_sel",
            "last_x_sel", "last_y_sel", "multi_y_sel", "overlay_multi",
            "subplot_orientation", "color_sel", "size_sel", "symbol_sel",
            "facet_sel", "z_sel", "bins_slider", "filtcol_mod",
            "macol_mod", "baseline_cols_mod", "peak_detect_cols",
            "formula_expr_mod", "formula_out_mod", "peaks_multi",
            "default_peak_cols", "logo_bytes", "logo_mime",
            "logo_position", "logo_width_pct", "logo_opacity",
            "preview_rows", "subplots_shared_x", "subplots_shared_y",
            "subplots_uniform_y",
        ]
        for p in [c['prefix'] for c in st.session_state['plot_contexts']]:
            keys_to_clear.extend([f"{p}{k}" for k in keys_to_clear])
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        for _k in list(st.session_state.keys()):
            if _k.startswith('err_'):
                st.session_state.pop(_k, None)
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        for _k in list(st.session_state.keys()):
            if _k.endswith('style_cfg'):
                st.session_state.pop(_k, None)
        st.session_state['style_cfg'] = {}
        st.success(
            "Hard reset done. Please reload the page if data seems stale."
        )
