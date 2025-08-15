"""Export / template / project related sidebar & body sections."""
from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
from originlite.project.serializer import save_project, load_project
from originlite.ui.helpers import build_template_dict


def export_sections(dm, theme, chart_type, x, y, multi_y, color, size, symbol,
                    facet, z, bins, fit_kind, x_title, y_title, tick_format_x,
                    tick_format_y, tick_angle_x, tick_angle_y, legend_show,
                    legend_orientation, legend_position, label_peaks,
                    palette_name, apply_palette, style_cfg,
                    custom_color_map, figure_style_cfg):
    """Render export/template/project controls.

    Returns compiled template dict (tmpl).
    """
    tmpl = build_template_dict({
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "multi_y": multi_y,
        "color": color,
        "size": size,
        "symbol": symbol,
        "facet": facet,
        "z": z,
        "bins": bins,
        "fit_kind": fit_kind,
        "theme": theme,
        "format": {
            "x_title": x_title,
            "y_title": y_title,
            "tick_format_x": tick_format_x,
            "tick_format_y": tick_format_y,
            "tick_angle_x": tick_angle_x,
            "tick_angle_y": tick_angle_y,
            "legend_show": legend_show,
            "legend_orientation": legend_orientation,
            "legend_position": legend_position,
            "label_peaks": label_peaks,
        },
        "style": {
            "palette": palette_name,
            "apply_palette": apply_palette,
            "overrides": style_cfg,
        },
        "custom_color_map": custom_color_map,
        "figure_style": figure_style_cfg,
        "data_model_operations": dm.operations,
    })
    st.subheader("Export")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Template JSON",
            data=json.dumps(tmpl, indent=2),
            file_name="template.json",
        )
    with col2:
        up_t = st.file_uploader(
            "Load template (.json)", type=["json"], key="tmpl"
        )
        if up_t:
            cfg = json.loads(up_t.read())
            st.session_state['loaded_template'] = cfg
            st.success("Template loaded")
            if st.button("Apply template"):
                for key in [
                    "chart_type", "x", "y", "color", "size",
                    "symbol", "facet", "z"
                ]:
                    if key in cfg:
                        st.session_state[key] = cfg[key]
                fs = cfg.get('figure_style', {})
                for sk, sv in fs.items():
                    st.session_state[f'fig_{sk}'] = sv
                style_loaded = cfg.get('style', {}).get('overrides', {})
                st.session_state['style_cfg'] = style_loaded
                if 'custom_color_map' in cfg:
                    st.session_state['custom_color_map'] = cfg[
                        'custom_color_map'
                    ]
                st.experimental_rerun()
    with col3:
        if st.button("Download Project (.olite)"):
            proj_path = save_project("current.olite", dm, tmpl)
            with open(proj_path, 'rb') as f:
                st.download_button(
                    "Save .olite",
                    data=f.read(),
                    file_name="current.olite",
                )
        up_proj = st.file_uploader(
            "Load project (.olite)", type=["olite"], key="proj"
        )
        if up_proj:
            tmp_path = Path("_uploaded.olite")
            with open(tmp_path, 'wb') as fw:
                fw.write(up_proj.getbuffer())
            dm_loaded, chart_cfg, extra = load_project(tmp_path)
            st.success(f"Project loaded with {len(dm_loaded.df)} rows.")
            st.session_state['loaded_template'] = chart_cfg
            if 'data_model_operations' in chart_cfg:
                dm.operations = chart_cfg['data_model_operations']
            if st.button("Apply project chart config"):
                fs = chart_cfg.get('figure_style', {})
                for sk, sv in fs.items():
                    st.session_state[f'fig_{sk}'] = sv
                style_loaded = chart_cfg.get('style', {}).get('overrides', {})
                st.session_state['style_cfg'] = style_loaded
                if 'custom_color_map' in chart_cfg:
                    st.session_state['custom_color_map'] = chart_cfg[
                        'custom_color_map'
                    ]
                st.experimental_rerun()
    return tmpl


__all__ = ["export_sections"]
