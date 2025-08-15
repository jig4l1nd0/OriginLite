"""Helper utilities for the Streamlit UI (data loading, legend, templates)."""

import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Tuple
import hashlib

from originlite.utils import df_from_upload
from originlite.project.serializer import list_projects, load_project
from originlite.core.data_model import DataModel


def load_data_sidebar() -> DataModel:
    """Render data/project loader in the sidebar and return a DataModel.

    Features:
        - Mode switching (Upload / Sample / Project) resets dependent session
            state.
        - Upload fingerprint uses MD5(content) so replacing a file with same
            name triggers reload.
    - Reload button forces re-parse even if fingerprint unchanged.
    - Hard reset clears all data-related session state keys.
    - Always returns a DataModel (empty placeholder if nothing loaded) so the
      rest of the app can still render without guard clauses needing st.stop().
    """
    with st.sidebar.expander("Load data / Project", expanded=True):
        prev_mode = st.session_state.get("_prev_load_mode")
        
        mode = st.radio(
            "Source",
            ["Upload", "Sample", "Project"],
            index=0,  # Always start with Upload as default
            key="load_mode_radio",  # Changed key to avoid conflicts
            help=(
                "Upload a file, use a sample dataset, or open a saved "
                ".olite project."
            ),
        )
        
        if mode != prev_mode:
            # Clear state that depends on the existing data model
            for k in [
                "data_model",
                "data_model_source",
                "formula_columns",
                "peaks_multi",
                "default_peak_cols",
            ]:
                st.session_state.pop(k, None)
        st.session_state["_prev_load_mode"] = mode

        file_obj = None  # UploadedFile or Path
        if mode == "Upload":
            # Use a more stable key for the file uploader
            file_obj = st.file_uploader(
                "Upload CSV/XLSX/TXT/XRDML",
                type=["csv", "xlsx", "xls", "txt", "xrdml"],
                key="file_uploader_widget",  # Changed key name
                help=(
                    "csv, xlsx/xls, txt (2 numeric cols), xrdml (XRD pattern)"
                ),
            )
            
            # Store uploaded file in session state for persistence
            if file_obj is not None:
                st.session_state["uploaded_file"] = file_obj
                st.sidebar.success(f"File uploaded: {file_obj.name}")
            elif "uploaded_file" in st.session_state:
                # Try to use previously uploaded file
                file_obj = st.session_state["uploaded_file"]
                st.sidebar.info(f"Using cached file: {file_obj.name}")
                # Add a button to clear the cached file
                if st.sidebar.button("Clear cached file", key="clear_upload"):
                    st.session_state.pop("uploaded_file", None)
                    file_obj = None
            
            if file_obj is not None:
                st.sidebar.success(
                    f"File uploaded: {file_obj.name} ({file_obj.size} bytes)"
                )
            else:
                pass
                
        elif mode == "Sample":
            sample = st.selectbox(
                "Sample dataset", [None, "iris", "tips"], key="sample_choice"
            )
            if sample == "iris":
                file_obj = Path("sample_data/iris.csv")
            elif sample == "tips":
                file_obj = Path("sample_data/tips.csv")
        else:  # Project mode
            proj_dir = Path(".")
            projects = list_projects(proj_dir)
            sel = st.selectbox(
                "Project (.olite)",
                [None] + [p.name for p in projects],
                key="proj_select",
            )
            if sel:
                ppath = proj_dir / sel
                dm_loaded, chart_cfg, extra = load_project(ppath)
                meta = extra.get("meta", {})
                st.success(
                    f"Loaded project '{meta.get('name', sel)}' ("
                    f"{len(dm_loaded.df)} rows)"
                )
                st.session_state["loaded_template"] = chart_cfg
                st.session_state["data_model"] = dm_loaded
                st.session_state["data_model_source"] = f"project:{ppath}"
                if st.button("Apply project config", key="apply_proj_cfg"):
                    fs = chart_cfg.get("figure_style", {})
                    for sk, sv in fs.items():
                        st.session_state[f"fig_{sk}"] = sv
                    style_loaded = chart_cfg.get("style", {}).get(
                        "overrides", {}
                    )
                    st.session_state["style_cfg"] = style_loaded
                    if "custom_color_map" in chart_cfg:
                        st.session_state["custom_color_map"] = chart_cfg[
                            "custom_color_map"
                        ]
                    for k in [
                        "chart_type",
                        "x",
                        "y",
                        "color",
                        "size",
                        "symbol",
                        "facet",
                        "z",
                    ]:
                        if k in chart_cfg:
                            st.session_state[k] = chart_cfg[k]
                return st.session_state["data_model"]

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            force_reload = st.button("Reload", key="force_reload_btn")
        with col2:
            # Renamed key to avoid conflict with global hard reset in app.py
            hard_reset = st.button(
                "Data hard reset", key="hard_reset_data_btn",
                help="Clear only loaded data & related session state."
            )
        if hard_reset:
            for k in [
                "data_model",
                "data_model_source",
                "formula_columns",
                "peaks_multi",
                "default_peak_cols",
            ]:
                st.session_state.pop(k, None)
            st.info("State cleared. Choose a data source again.")
            placeholder = DataModel(pd.DataFrame())
            st.session_state["data_model"] = placeholder
            st.session_state["data_model_source"] = "empty"
            return placeholder

    # Outside expander: fallback or reuse
    
    # TEMPORARILY DISABLED - always process new uploads
    # if file_obj is None and "data_model" in st.session_state:
    #     return st.session_state["data_model"]
    
    if file_obj is None:
        # Only return existing if we have one AND no file was selected
        if "data_model" in st.session_state:
            return st.session_state["data_model"]
        placeholder = DataModel(pd.DataFrame())
        st.session_state["data_model"] = placeholder
        st.session_state["data_model_source"] = "empty"
        return placeholder

    # Load & fingerprint
    try:
        # Check if it's an uploaded file (has both name and getvalue)
        if (
            hasattr(file_obj, "name")
            and hasattr(file_obj, "getvalue")
        ):  # Uploaded file
            try:
                raw = (
                    file_obj.getvalue()
                    if hasattr(file_obj, "getvalue")
                    else None
                )
                if raw is not None:
                    digest = hashlib.md5(raw).hexdigest()
                    source_id = f"upload:{file_obj.name}:{len(raw)}:{digest}"
                else:
                    size = getattr(file_obj, "size", "na")
                    source_id = f"upload:{file_obj.name}:{size}"
                
                # Check if we already have this data loaded
                if (
                    not force_reload
                    and st.session_state.get("data_model_source") == source_id
                    and "data_model" in st.session_state
                ):
                    st.sidebar.caption(f"Using cached data: {source_id[-24:]}")
                    return st.session_state["data_model"]
                
                # Load the data
                st.sidebar.info("Processing uploaded file...")
                st.sidebar.text(f"File type: {type(file_obj)}")
                st.sidebar.text(f"File name: {file_obj.name}")
                st.sidebar.text(f"File size: {file_obj.size} bytes")
                
                df = df_from_upload(file_obj)
                
                if df is None:
                    st.sidebar.error("df_from_upload returned None")
                elif df.empty:
                    st.sidebar.warning(
                        "df_from_upload returned empty DataFrame"
                    )
                else:
                    st.sidebar.info(f"df_from_upload succeeded: {df.shape}")
                    st.sidebar.text(f"Columns: {list(df.columns)}")
                    st.sidebar.text(
                        f"First few values: {df.head(2).values.tolist()}"
                    )
                
            except Exception as e:
                st.sidebar.error(f"Error processing upload: {str(e)}")
                st.sidebar.text(f"Exception type: {type(e).__name__}")
                # Try basic pandas fallback
                try:
                    st.sidebar.info("Trying pandas fallback...")
                    if file_obj.name.lower().endswith('.csv'):
                        df = pd.read_csv(file_obj)
                    elif file_obj.name.lower().endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_obj)
                    else:
                        df = None
                    if df is not None:
                        st.sidebar.success("Pandas fallback successful!")
                except Exception as fallback_e:
                    st.sidebar.error(
                        f"Fallback also failed: {str(fallback_e)}"
                    )
                    df = None
                
        else:  # Sample path
            source_id = f"sample:{file_obj}"
            if (
                not force_reload
                and st.session_state.get("data_model_source") == source_id
                and "data_model" in st.session_state
            ):
                return st.session_state["data_model"]
            try:
                st.sidebar.info("Loading sample data...")
                df = pd.read_csv(file_obj)
                st.sidebar.success(f"Sample data loaded: {df.shape}")
            except Exception as e:
                st.sidebar.error(f"Could not read sample file: {str(e)}")
                df = None

        if df is None or df.empty:
            st.sidebar.warning("Empty or unreadable file.")
            placeholder = DataModel(pd.DataFrame())
            st.session_state["data_model"] = placeholder
            st.session_state["data_model_source"] = "empty"
            return placeholder

        # Successfully loaded data
        dm_new = DataModel(df)
        st.session_state["data_model"] = dm_new
        st.session_state["data_model_source"] = source_id
        st.session_state.pop("formula_columns", None)
        
        # Show success message and data info
        st.sidebar.success(
            f"Data loaded: {len(df)} rows, {len(df.columns)} columns"
        )
        st.sidebar.caption(f"Data source: {source_id[-24:]}")
        
        return dm_new
        
    except Exception as e:
        st.sidebar.error(f"Unexpected error loading data: {str(e)}")
        placeholder = DataModel(pd.DataFrame())
        st.session_state["data_model"] = placeholder
        st.session_state["data_model_source"] = "error"
        return placeholder


def legend_position_selector() -> Tuple[str, Tuple[int, int]]:
    """Sidebar selector for legend position returning (label, (x,y))."""
    legend_position = st.sidebar.selectbox(
        "Legend position",
        ["top-right", "top-left", "bottom-right", "bottom-left"],
        index=0,
    )
    legend_pos_map = {
        "top-right": (1, 1),
        "top-left": (0, 1),
        "bottom-right": (1, 0),
        "bottom-left": (0, 0),
    }
    return legend_position, legend_pos_map[legend_position]


def build_template_dict(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return template metadata unchanged (placeholder for future logic)."""
    return meta
