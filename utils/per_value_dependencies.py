"""
Per-value dependencies: simple, UI-friendly API for building a probability tree.

Exported API:
  - build_probability_tree(df, column_name, value, *, max_depth, min_probability, max_values_per_column)
  - build_multilevel_probability_tree: backwards-compatible alias
"""

from typing import Any, Dict, Optional, List

import pandas as pd


SENTINEL = "<NA>"


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.where(~series.isna(), SENTINEL)


def _grouping_score(value_counts: pd.Series, total_count: int) -> float:
    if total_count == 0 or len(value_counts) == 0:
        return 0.0
    max_prob = value_counts.iloc[0] / total_count
    unique_count = len(value_counts)
    # Simple penalty for high cardinality
    penalty = 1.0 + 0.10 * max(0, unique_count - 2)
    return float(max_prob / penalty)


def _select_best_column(current_df: pd.DataFrame, used_columns: List[str], root_column: str, *, remove_fully_determined: bool = False) -> Optional[str]:
    if current_df.empty:
        return None
    candidates = [c for c in current_df.columns if c not in used_columns and c != root_column]
    if not candidates:
        return None
    best_col = None
    best_score = -1.0
    total = len(current_df)
    for col in candidates:
        series = _normalize_series(current_df[col])
        vc = series.value_counts()
        # Optionally skip fully determined columns (only one unique value in this subset)
        if remove_fully_determined and len(vc) == 1:
            continue
        # Skip degenerate columns (all unique or all same after normalization)
        if len(vc) == 0:
            continue
        score = _grouping_score(vc, total)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def build_probability_tree(
    df: pd.DataFrame,
    *,
    column_name: str,
    value: Any,
    max_depth: int = 20,
    min_probability: float = 0.0,
    max_values_per_column: int = 5,
    remove_fully_determined: bool = False,
) -> Dict[str, Any]:
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    total_rows = len(df)

    norm_root = _normalize_series(df[column_name])
    search_value = SENTINEL if pd.isna(value) else value
    matching_rows = df[norm_root == search_value]

    prior_probability = (len(matching_rows) / total_rows) if total_rows else 0.0

    def build_node(current_df: pd.DataFrame, used_columns: List[str], depth: int) -> Optional[Dict[str, Any]]:
        if depth >= max_depth or current_df.empty:
            return None
        next_col = _select_best_column(current_df, used_columns, root_column=column_name, remove_fully_determined=remove_fully_determined)
        if next_col is None:
            return None
        series = _normalize_series(current_df[next_col])
        vc = series.value_counts()
        total = len(current_df)
        node_values: Dict[Any, Dict[str, Any]] = {}
        for val, count in vc.head(max_values_per_column).items():
            prob = count / total if total else 0.0
            if prob < min_probability:
                continue
            val_key = None if val == SENTINEL else val
            child_df = current_df[series == val]
            children = build_node(child_df, used_columns + [next_col], depth + 1)
            node_values[val_key] = {
                "probability": prob,
                "count": int(count),
                "children": (children if children else {}),
            }
        return {next_col: node_values} if node_values else None

    tree_children = build_node(matching_rows, used_columns=[], depth=0)
    return {
        "root": {
            "column": column_name,
            "value": (value if not pd.isna(value) else SENTINEL),
            "total_rows": total_rows,
            "matching_rows": int(len(matching_rows)),
            "prior_probability": prior_probability,
        },
        "tree": (tree_children if tree_children else {}),
    }


# Backwards-compatible alias
def build_multilevel_probability_tree(
    df: pd.DataFrame,
    value: Any,
    column_name: str,
    max_depth: int = 20,
    min_probability: float = 0.0,
    max_values_per_column: int = 5,
    remove_fully_determined: bool = False,
) -> Dict[str, Any]:
    return build_probability_tree(
        df,
        column_name=column_name,
        value=value,
        max_depth=max_depth,
        min_probability=min_probability,
        max_values_per_column=max_values_per_column,
        remove_fully_determined=remove_fully_determined,
    )

