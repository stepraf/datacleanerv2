"""
Helper module for calculating 1:N relationship violation ratios between two columns.

A 1:N relationship means:
- Parent column (col1): each unique value can correspond to zero or more values in child column
- Child column (col2): each unique value must correspond to exactly ONE value in parent column

Violation ratio = 1 - (percentage of rows satisfying the constraint / 100)
"""

import pandas as pd
import numpy as np


def calculate_one_to_many_violation_ratio(pair_df, col1, col2):
    """
    Calculate the violation ratio for a 1:N relationship between two columns.
    
    Args:
        pair_df: DataFrame containing at least col1 and col2 columns
        col1: Name of the parent column (the "one" side)
        col2: Name of the child column (the "many" side)
    
    Returns:
        tuple: (violation_ratio, percentage, valid_rows, total_rows, valid_mask)
        - violation_ratio: float between 0 and 1 (0 = perfect, 1 = all violations)
        - percentage: float between 0 and 100 (percentage of valid rows)
        - valid_rows: int (number of rows satisfying the constraint)
        - total_rows: int (total number of rows)
        - valid_mask: pandas Series of booleans indicating which rows are valid
    """
    if len(pair_df) == 0:
        return 1.0, 0.0, 0, 0, pd.Series([], dtype=bool)
    
    total_rows = len(pair_df)
    
    # Check if columns exist
    if col1 not in pair_df.columns or col2 not in pair_df.columns:
        raise ValueError(f"Columns {col1} or {col2} not found in dataframe")
    
    # Calculate how many unique col1 values each col2 value maps to
    # For 1:N: each col2 value should map to exactly one col1 value
    # Note: groupby excludes NA values by default, but we need to handle them explicitly
    col2_to_col1_counts = pair_df.groupby(col2)[col1].nunique()
    
    # Handle NA values in col2: if col2 is NA, we can't determine the relationship
    # Rows with NA in col2 should be considered invalid (violations)
    na_in_col2 = pair_df[col2].isna()
    
    # Quick check: if all col2 values map to exactly one col1 value, it's perfect
    # Note: groupby excludes NA values, so col2_to_col1_counts won't have entries for NA col2 values
    # We need to check that: (1) all non-NA col2 values map to exactly one col1, and (2) there are no NA values in col2
    if len(col2_to_col1_counts) > 0 and (col2_to_col1_counts == 1).all() and not na_in_col2.any():
        valid_mask = pd.Series([True] * total_rows, index=pair_df.index)
        percentage = 100.0
        violation_ratio = 0.0
        return violation_ratio, percentage, total_rows, total_rows, valid_mask
    
    # For each row, check if its col2 value maps to exactly one col1 value
    # Map the col2 values to their unique count
    # Rows with NA in col2 will have NaN in the map result, which we'll treat as invalid
    col2_unique_counts = pair_df[col2].map(col2_to_col1_counts)
    
    # A row is valid if: 
    # 1. The col2 value in that row maps to exactly one col1 value (count == 1)
    # 2. The col2 value is not NA (we already handle this via na_in_col2)
    valid_mask = (col2_unique_counts == 1) & ~na_in_col2
    
    valid_rows = valid_mask.sum()
    percentage = (valid_rows / total_rows) * 100 if total_rows > 0 else 0.0
    
    # Violation ratio = 1 - (percentage / 100)
    violation_ratio = 1.0 - (percentage / 100.0) if percentage > 0 else 1.0
    
    return violation_ratio, percentage, valid_rows, total_rows, valid_mask

