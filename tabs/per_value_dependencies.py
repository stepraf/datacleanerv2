from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Optional
import json

import pandas as pd
import numpy as np
import streamlit as st

from utils.per_value_dependencies import build_probability_tree


def _get_active_dataframe() -> Optional[pd.DataFrame]:
    """Get the active dataframe from session state."""
    df = st.session_state.get("processed_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def _stringify_tree(tree: Dict[str, Any]) -> str:
    """Convert tree to string representation."""
    out = StringIO()

    def _rec_print(node: Dict[str, Any], prefix: str = "", is_last: bool = True) -> None:
        if "tree" not in node:
            return
        tree_nodes = node.get("tree", {})
        for col_idx, (col_name, col_values) in enumerate(tree_nodes.items()):
            is_last_col = (col_idx == len(tree_nodes) - 1)
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
            out.write(f"{prefix}{connector}{col_name}\n")
            sorted_values = sorted(col_values.items(), key=lambda x: x[1]["probability"], reverse=True)
            for val_idx, (val, info) in enumerate(sorted_values):
                is_last_val = (val_idx == len(sorted_values) - 1)
                val_str = str(val) if val is not None else "<NA>"
                prob_str = f"{info['probability']:.2%}"
                count_str = f"(n={info['count']})"
                if is_last_col:
                    val_connector = new_prefix + ("└── " if is_last_val else "├── ")
                    next_prefix = new_prefix + ("    " if is_last_val else "│   ")
                else:
                    val_connector = prefix + ("│   └── " if is_last_val else "│   ├── ")
                    next_prefix = prefix + ("│       " if is_last_val else "│   │   ")
                out.write(f"{val_connector}{val_str} [{prob_str}] {count_str}\n")
                children = info.get("children", {})
                for child_col_idx, (child_col, child_values) in enumerate(children.items()):
                    is_last_child = (child_col_idx == len(children) - 1)
                    _rec_print({"tree": {child_col: child_values}}, prefix=next_prefix, is_last=is_last_val and is_last_child)

    # Header
    root = tree.get("root", {})
    out.write("=" * 100 + "\n")
    out.write("MULTI-LEVEL PROBABILITY TREE (Column Relationships)\n")
    out.write("=" * 100 + "\n\n")
    out.write(f"Root: {root.get('column')} = {root.get('value')}\n")
    out.write(
        f"  Matching Rows: {root.get('matching_rows', 0):,} / {root.get('total_rows', 0):,} ("
        f"{root.get('prior_probability', 0.0):.4%})\n\n"
    )

    _rec_print(tree, prefix="", is_last=True)
    out.write("\n" + "=" * 100 + "\n")
    return out.getvalue()


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if pd.isna(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {_make_json_serializable(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        # Try to convert to native Python type
        try:
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
        except (ValueError, AttributeError):
            pass
        return obj


def _render_tree_expandable(tree: Dict[str, Any]) -> None:
    """Render tree as expandable sections."""
    root = tree.get("root", {})
    st.write(f"Root: `{root.get('column')}` = {root.get('value')}")
    st.caption(
        f"Matching Rows: {root.get('matching_rows', 0):,} / {root.get('total_rows', 0):,} "
        f"({root.get('prior_probability', 0.0):.2%})"
    )

    def _render_level(node: Dict[str, Any], depth: int = 0) -> None:
        for col_name, col_values in node.get("tree", {}).items():
            with st.expander(f"Level {depth+1}: {col_name}", expanded=(depth == 0)):
                # Sort values by probability
                sorted_values = sorted(col_values.items(), key=lambda x: x[1]["probability"], reverse=True)
                for val, info in sorted_values:
                    val_str = "NA" if (val is None) else str(val)
                    st.markdown(f"- **{val_str}**: {info['probability']:.2%} (n={info['count']})")
                    children = info.get("children", {})
                    if children:
                        _render_level({"tree": children}, depth + 1)

    _render_level(tree, 0)


def render():
    """Render the per value dependencies tab."""
    st.subheader("Per value dependencies")

    df = _get_active_dataframe()
    if df is None:
        st.info("Upload and prepare data first. No `processed_df` found in session state.")
        return

    # Parameter controls
    with st.expander("Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_depth = st.number_input("Max depth", min_value=1, max_value=100, value=20, step=1)
        with col2:
            min_probability = st.number_input("Min probability", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
        with col3:
            max_values_per_column = st.number_input("Max values per column", min_value=1, max_value=50, value=10, step=1)
        with col4:
            remove_fully_determined = st.checkbox("Remove 100% correlated columns", value=False)

    # Select column and value
    columns = list(df.columns)
    if not columns:
        st.warning("No columns available in the dataframe.")
        return
    
    effective_col = st.selectbox("Column", options=columns, key="perval_effective_col")
    
    # Show NaN as "NA" but keep original mapping
    col_series = df[effective_col]
    unique_vals = pd.unique(col_series)
    display_vals = ["NA" if (pd.isna(v) or v is None) else v for v in unique_vals]
    # Build mapping display->actual
    to_actual: Dict[str, Any] = {}
    used_display: set[str] = set()
    for orig, disp in zip(unique_vals, display_vals):
        key = str(disp)
        if key in used_display:
            key = f"{key} ({repr(orig)})"
        used_display.add(key)
        to_actual[key] = orig
    
    selected_disp = st.selectbox("Value", options=list(to_actual.keys()), key="perval_effective_val")
    effective_val = to_actual[selected_disp]

    st.write(f"Selected: column `{effective_col}` value = {('NA' if (pd.isna(effective_val) or effective_val is None) else effective_val)}")

    # Build tree from in-memory df
    with st.spinner("Computing dependencies..."):
        tree = build_probability_tree(
            df,
            column_name=effective_col,
            value=effective_val,
            max_depth=int(max_depth),
            min_probability=float(min_probability),
            max_values_per_column=int(max_values_per_column),
            remove_fully_determined=bool(remove_fully_determined),
        )

    # Render tree as plain text
    text_output = _stringify_tree(tree)
    st.code(text_output, language="text")

    # Export buttons
    json_output = _make_json_serializable(tree)
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            label="Download tree (text)",
            data=text_output,
            file_name="per_value_dependencies.txt",
            mime="text/plain",
        )
    with col_b:
        st.download_button(
            label="Download tree (JSON)",
            data=json.dumps(json_output, indent=2),
            file_name="per_value_dependencies.json",
            mime="application/json",
        )

    # Build trees for top 15 most frequent values
    with st.expander("Trees for top 15 most frequent values", expanded=False):
        col_series = df[effective_col]
        value_counts = col_series.value_counts(dropna=False)
        top_values = value_counts.head(15)
        
        st.caption(f"Building trees for top {len(top_values)} most frequent values in column `{effective_col}`")
        
        for idx, (value, count) in enumerate(top_values.items(), 1):
            percentage = (count / len(df)) * 100
            val_display = "NA" if (pd.isna(value) or value is None) else str(value)
            
            with st.expander(f"{idx}. {val_display} ({count:,} rows, {percentage:.2f}%)", expanded=False):
                with st.spinner(f"Computing tree for {val_display}..."):
                    top_tree = build_probability_tree(
                        df,
                        column_name=effective_col,
                        value=value,
                        max_depth=int(max_depth),
                        min_probability=float(min_probability),
                        max_values_per_column=int(max_values_per_column),
                        remove_fully_determined=bool(remove_fully_determined),
                    )
                    top_text_output = _stringify_tree(top_tree)
                    st.code(top_text_output, language="text")

