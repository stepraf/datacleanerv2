"""
Helper module for calculating and applying corrections to fix 1:N relationship violations.

The correction algorithm identifies violations where a child column value maps to multiple
parent column values, and determines the correct parent value (most frequent) to fix the violation.
"""

import pandas as pd
from helpers.one_to_many import calculate_one_to_many_violation_ratio


def prepare_pair_df(df, col1, col2, drop_na):
    """
    Extract and optionally filter a column pair dataframe.
    
    Args:
        df: Source DataFrame
        col1: First column name
        col2: Second column name
        drop_na: If True, drop rows where either column has NA
    
    Returns:
        DataFrame with the two columns, or None if empty
    """
    pair_df = df[[col1, col2]].copy()
    if drop_na:
        pair_df = pair_df.dropna(subset=[col1, col2])
    return pair_df if len(pair_df) > 0 else None


def calculate_corrections_for_pair(df, parent_col, child_col, drop_na):
    """
    Calculate corrections for a column pair to fix 1:N relationship violations.
    Corrects parent column values to maintain 1:N relationship (parent → child).
    
    Algorithm:
    1. Identify violations where a child value maps to multiple parent values
    2. For each violation, determine the correct parent value (most frequent)
    3. Return a correction plan with all necessary changes
    
    Args:
        df: DataFrame containing the columns
        parent_col: The parent column (the "one" side)
        child_col: The child column (the "many" side)
        drop_na: Whether to drop NA values before analysis
    
    Returns:
        dict with correction plan:
        {
            'column_to_correct': parent_col,
            'reference_column': child_col,
            'corrections': list of correction dicts,
            'total_changes': int,
            'pair_df': DataFrame with original data (may be filtered if drop_na=True)
        }
        Returns None if no corrections needed
    """
    pair_df = prepare_pair_df(df, parent_col, child_col, drop_na)
    if pair_df is None:
        return None
    
    # Calculate violations for parent -> child direction
    # In a 1:N relationship, each child value should map to exactly one parent value
    _, _, _, _, valid_mask = calculate_one_to_many_violation_ratio(pair_df, parent_col, child_col)
    violations = pair_df[~valid_mask]
    
    if len(violations) == 0:
        return None
    
    corrections = []
    total_changes = 0
    
    # Group violations by child value
    for child_val, group in violations.groupby(child_col):
        if pd.isna(child_val):
            continue
        
        # Count frequency of each parent value for this child value
        parent_counts = group[parent_col].value_counts()
        if len(parent_counts) <= 1:
            continue
        
        # Most frequent parent value is the correct one
        correct_parent_val = parent_counts.index[0]
        rows_to_correct = group[group[parent_col] != correct_parent_val]
        num_changes = len(rows_to_correct)
        
        if num_changes > 0:
            corrections.append({
                'violating_value': child_val,
                'current_values': parent_counts.to_dict(),
                'correct_value': correct_parent_val,
                'num_changes': num_changes,
                'rows_to_correct': rows_to_correct.index.tolist()
            })
            total_changes += num_changes
    
    if total_changes == 0:
        return None
    
    return {
        'column_to_correct': parent_col,
        'reference_column': child_col,
        'corrections': corrections,
        'total_changes': total_changes,
        'pair_df': pair_df,
        'direction': f'{parent_col} → {child_col}'
    }


def _calculate_dynamic_confidence_threshold(sample_size):
    """
    Calculate dynamic confidence threshold based on sample size.
    
    Larger samples require lower thresholds (more reliable), 
    smaller samples require higher thresholds (less reliable).
    
    Args:
        sample_size: Number of occurrences for a child value
    
    Returns:
        float: Confidence threshold as a percentage (0-100)
    """
    if sample_size >= 100:
        return 60.0  # Large samples: 60% threshold
    elif sample_size >= 50:
        return 65.0  # Medium-large: 65% threshold
    elif sample_size >= 20:
        return 70.0  # Medium: 70% threshold
    elif sample_size >= 10:
        return 75.0  # Small-medium: 75% threshold
    elif sample_size >= 5:
        return 80.0  # Small: 80% threshold
    else:
        return 85.0  # Very small: 85% threshold


def is_correction_confident(correction, min_count_threshold, ratio_threshold=3.0):
    """
    Determine if a correction is confident enough to apply automatically.
    
    Criteria:
    1. Sample size >= min_count_threshold
    2. Most frequent value meets dynamic confidence threshold based on sample size
    3. Most frequent value is at least ratio_threshold times more common than second most frequent
    
    Args:
        correction: Correction dict with 'current_values' and 'num_changes'
        min_count_threshold: Minimum number of occurrences required
        ratio_threshold: Minimum ratio between most frequent and second most frequent (default 3.0)
    
    Returns:
        bool: True if correction is confident, False otherwise
    """
    current_values = correction['current_values']
    
    if len(current_values) == 0:
        return False
    
    # Get sorted values by frequency (descending)
    sorted_values = sorted(current_values.items(), key=lambda x: x[1], reverse=True)
    most_frequent_count = sorted_values[0][1]
    total_count = sum(current_values.values())
    
    # Criterion 1: Minimum count threshold
    if total_count < min_count_threshold:
        return False
    
    # Criterion 2: Dynamic confidence threshold
    confidence_threshold = _calculate_dynamic_confidence_threshold(total_count)
    most_frequent_percentage = (most_frequent_count / total_count) * 100
    
    if most_frequent_percentage < confidence_threshold:
        return False
    
    # Criterion 3: Ratio threshold (only if there are at least 2 values)
    if len(sorted_values) >= 2:
        second_frequent_count = sorted_values[1][1]
        if second_frequent_count > 0:
            ratio = most_frequent_count / second_frequent_count
            if ratio < ratio_threshold:
                return False
    
    return True


def filter_high_confidence_corrections(correction_plan, min_count_threshold, ratio_threshold=3.0):
    """
    Filter corrections to only include high-confidence ones.
    
    Args:
        correction_plan: Full correction plan dict
        min_count_threshold: Minimum number of occurrences required
        ratio_threshold: Minimum ratio between most frequent and second most frequent (default 3.0)
    
    Returns:
        dict: Filtered correction plan with:
        - 'confident_corrections': list of confident corrections
        - 'non_confident_corrections': list of non-confident corrections
        - 'confident_total_changes': total changes for confident corrections
        - 'non_confident_total_changes': total changes for non-confident corrections
        - All original fields from correction_plan
    """
    if not correction_plan or not correction_plan.get('corrections'):
        return {
            **correction_plan,
            'confident_corrections': [],
            'non_confident_corrections': [],
            'confident_total_changes': 0,
            'non_confident_total_changes': 0
        }
    
    confident_corrections = []
    non_confident_corrections = []
    confident_total_changes = 0
    non_confident_total_changes = 0
    
    for correction in correction_plan['corrections']:
        if is_correction_confident(correction, min_count_threshold, ratio_threshold):
            confident_corrections.append(correction)
            confident_total_changes += correction['num_changes']
        else:
            non_confident_corrections.append(correction)
            non_confident_total_changes += correction['num_changes']
    
    return {
        **correction_plan,
        'confident_corrections': confident_corrections,
        'non_confident_corrections': non_confident_corrections,
        'confident_total_changes': confident_total_changes,
        'non_confident_total_changes': non_confident_total_changes
    }

