import streamlit as st
import pandas as pd
import hashlib
from helpers.one_to_many import calculate_one_to_many_violation_ratio


# ============================================================================
# Helper Functions
# ============================================================================

def _has_data():
    """Check if processed data is available."""
    return ('processed_df' in st.session_state and 
            st.session_state.processed_df is not None and 
            len(st.session_state.processed_df) > 0)


def _get_available_columns():
    """Get list of available columns excluding initial_id."""
    if not _has_data():
        return []
    return [col for col in st.session_state.processed_df.columns if col != 'initial_id']


def _columns_exist(col1, col2):
    """Check if both columns exist in processed_df."""
    if not _has_data():
        return False, False
    df_cols = list(st.session_state.processed_df.columns)
    return col1 in df_cols, col2 in df_cols


def _add_message(message):
    """Add a message to the shared messages log."""
    if 'shared_messages' not in st.session_state:
        st.session_state.shared_messages = []
    st.session_state.shared_messages.append(message)


def _filter_valid_results(results):
    """Filter results to only include pairs where both columns still exist."""
    valid_results = []
    for result in results:
        col1_exists, col2_exists = _columns_exist(result['column1'], result['column2'])
        if col1_exists and col2_exists:
            valid_results.append(result)
    return valid_results


def _remove_results_involving_columns(columns_to_remove):
    """Remove results that involve any of the specified columns."""
    if 'one_to_many_results' not in st.session_state:
        return
    
    if not st.session_state.one_to_many_results:
        return
    
    st.session_state.one_to_many_results = [
        result for result in st.session_state.one_to_many_results
        if result['column1'] not in columns_to_remove and 
           result['column2'] not in columns_to_remove
    ]


# ============================================================================
# 1:N Relationship Analysis
# ============================================================================

def _calculate_one_to_many_percentage(pair_df, col1, col2):
    """
    Calculate the percentage of rows that have a 1:N relationship.
    
    Returns: (percentage, valid_rows, valid_mask)
    A 1:N relationship means:
    - col1 is the parent ("one" side): each unique value in col1 can correspond to zero or more values in col2
    - col2 is the child ("many" side): each unique value in col2 must correspond to exactly ONE value in col1
    
    Note: 1:1 is a subset of 1:N where N=1. So if col2->col1 is 1:1, it's a valid 1:N relationship.
    """
    # Use the helper function for consistency
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(pair_df, col1, col2)
    
    return percentage, valid_rows, valid_mask


def _analyze_column_pair(df_original, df_clean, col1, col2, threshold, drop_na):
    """Analyze a single column pair for 1:N relationship."""
    # Extract the pair columns from the cleaned dataframe
    pair_df = df_clean[[col1, col2]].copy()
    
    # If drop_na is True, drop rows where either col1 or col2 has NA for this specific pair
    if drop_na:
        pair_df = pair_df.dropna(subset=[col1, col2])
    
    if len(pair_df) == 0:
        return None
    
    percentage, valid_rows, valid_mask = _calculate_one_to_many_percentage(pair_df, col1, col2)
    
    # Calculate violation ratio using the helper function for consistency
    violation_ratio, _, _, _, _ = calculate_one_to_many_violation_ratio(pair_df, col1, col2)
    
    if percentage >= threshold * 100:
        # Calculate unique values count for each column (from cleaned data)
        unique_col1 = pair_df[col1].nunique()
        unique_col2 = pair_df[col2].nunique()
        
        # Calculate NA counts from original dataframe
        na_col1 = df_original[col1].isna().sum()
        na_col2 = df_original[col2].isna().sum()
        
        # Separate matching and non-matching rows (limit to 1000 rows each to avoid memory issues)
        matching_df = pair_df[valid_mask].head(1000).reset_index(drop=True)
        non_matching_df = pair_df[~valid_mask].head(1000).reset_index(drop=True)
        
        return {
            'column1': col1,
            'column2': col2,
            'percentage': percentage,
            'violation_ratio': violation_ratio,
            'valid_rows': int(valid_rows),
            'total_rows': len(pair_df),
            'unique_col1': unique_col1,
            'unique_col2': unique_col2,
            'na_col1': int(na_col1),
            'na_col2': int(na_col2),
            'drop_na': drop_na,
            'matching_df': matching_df,
            'non_matching_df': non_matching_df,
            'matching_count': int(valid_rows),
            'non_matching_count': len(pair_df) - int(valid_rows)
        }
    return None


def analyze_one_to_many(df, threshold, drop_na=True):
    """Analyze 1:N relationships between all column pairs."""
    st.subheader("Analysis Results")
    
    columns = _get_available_columns()
    results = []
    
    # Use the original dataframe - NA handling will be done per-pair in _analyze_column_pair
    df_clean = df
    
    with st.spinner("Analyzing column pairs..."):
        total_pairs = len(columns) * (len(columns) - 1) // 2
        progress_bar = st.progress(0) if total_pairs > 10 else None
        status_text = st.empty() if total_pairs > 10 else None
        
        pair_count = 0
        
        # Check all pairs of columns (check both directions: col1->col2 and col2->col1)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                pair_count += 1
                
                # Update progress
                if progress_bar:
                    progress_bar.progress(pair_count / total_pairs)
                if status_text:
                    status_text.text(f"Analyzing pair {pair_count}/{total_pairs}: {col1} → {col2}")
                
                # Analyze this pair (col1 -> col2: 1:N means col1 is "one", col2 is "many")
                result = _analyze_column_pair(df, df_clean, col1, col2, threshold, drop_na)
                if result:
                    results.append(result)
                
                # Also check reverse direction (col2 -> col1: 1:N means col2 is "one", col1 is "many")
                result_reverse = _analyze_column_pair(df, df_clean, col2, col1, threshold, drop_na)
                if result_reverse:
                    results.append(result_reverse)
        
        # Clear progress indicators
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
    
    # Sort by percentage descending
    results.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Store results in session state
    st.session_state.one_to_many_results = results
    
    # Don't display results here - let render() handle it to avoid duplicate keys


# ============================================================================
# Action Handlers
# ============================================================================

def _handle_keep_column(col_to_keep, col_to_drop):
    """Keep one column and drop another, removing affected results."""
    if col_to_drop not in st.session_state.processed_df.columns:
        return
    
    st.session_state.processed_df = st.session_state.processed_df.drop(columns=[col_to_drop])
    _remove_results_involving_columns([col_to_drop])
    _add_message(f"🗑️ **Dropped column {col_to_drop}** (kept **{col_to_keep}**) - from 1:N analysis")
    st.rerun()


def _handle_concatenate(col1, col2):
    """Concatenate two columns, drop originals, and remove affected results."""
    if col1 not in st.session_state.processed_df.columns or col2 not in st.session_state.processed_df.columns:
        return
    
    # Create concatenated column
    new_col_name = f"{col1} | {col2}"
    st.session_state.processed_df[new_col_name] = (
        st.session_state.processed_df[col1].astype(str) + 
        " | " + 
        st.session_state.processed_df[col2].astype(str)
    )
    
    # Drop original columns
    st.session_state.processed_df = st.session_state.processed_df.drop(columns=[col1, col2])
    
    # Remove results involving the dropped columns
    _remove_results_involving_columns([col1, col2])
    
    _add_message(f"➕ **Created new column {new_col_name}** (concatenated **{col1}** and **{col2}**, then dropped original columns) - from 1:N analysis")
    st.rerun()


def _handle_remove_both(col1, col2):
    """Remove both columns and remove affected results."""
    if col1 not in st.session_state.processed_df.columns and col2 not in st.session_state.processed_df.columns:
        return
    
    # Drop both columns (if they exist)
    columns_to_drop = [c for c in [col1, col2] if c in st.session_state.processed_df.columns]
    if columns_to_drop:
        st.session_state.processed_df = st.session_state.processed_df.drop(columns=columns_to_drop)
        _remove_results_involving_columns([col1, col2])
        _add_message(f"🗑️ **Dropped columns {col1} and {col2}** - from 1:N analysis")
        st.rerun()


# ============================================================================
# Display Functions
# ============================================================================

def _create_button_key(col1, col2, idx, action):
    """Create a unique key for a button."""
    # Include idx, action, and both column names to ensure uniqueness
    # Sort column names for consistent hashing
    sorted_cols = tuple(sorted([col1, col2]))
    pair_hash = hashlib.md5(f"{sorted_cols[0]}_{sorted_cols[1]}_{idx}_{action}".encode()).hexdigest()[:12]
    return f"{action}_{idx}_{pair_hash}_1N"


def display_results(results, threshold):
    """Display the 1:N relationship results with action buttons."""
    threshold_percent = threshold * 100
    
    if not results:
        st.info(f"No column pairs found with ≥{threshold_percent:.0f}% 1:N relationship.")
        return
    
    st.success(f"Found {len(results)} column pair(s) with ≥{threshold_percent:.0f}% 1:N relationship:")
    
    for idx, result in enumerate(results):
        col1 = result['column1']
        col2 = result['column2']
        percentage = result['percentage']
        valid_rows = result['valid_rows']
        total_rows = result['total_rows']
        drop_na = result.get('drop_na', False)
        unique_col1 = result.get('unique_col1', 0)
        unique_col2 = result.get('unique_col2', 0)
        
        col1_exists, col2_exists = _columns_exist(col1, col2)
        
        with st.container():
            info_col, button_col = st.columns([3, 1])
            
            with info_col:
                st.write(f"**{col1} → {col2}**")
                violation_ratio = result.get('violation_ratio', 0)
                info_line = f"- 1:N Relationship: {percentage:.2f}% | Valid rows: {valid_rows:,} / {total_rows:,} | Violation ratio: {violation_ratio:.4f}"
                st.write(info_line)
                
                # Show unique values and NA counts combined
                na_col1 = result.get('na_col1', 0)
                na_col2 = result.get('na_col2', 0)
                if drop_na:
                    stats_line = f"- {col1}: {unique_col1:,} unique, {na_col1:,} NA | {col2}: {unique_col2:,} unique, {na_col2:,} NA"
                else:
                    stats_line = f"- {col1}: {na_col1:,} NA | {col2}: {na_col2:,} NA"
                st.write(stats_line)
                
                # Show matching and non-matching tables in expandable sections
                matching_df = result.get('matching_df')
                non_matching_df = result.get('non_matching_df')
                matching_count = result.get('matching_count', 0)
                non_matching_count = result.get('non_matching_count', 0)
                
                if matching_df is not None and len(matching_df) > 0:
                    # Group matching values by column pairs and count occurrences
                    matching_grouped = matching_df.groupby([col1, col2]).size().reset_index(name='Count')
                    matching_grouped = matching_grouped.sort_values('Count', ascending=False)
                    
                    with st.expander(f"📊 Matching values ({matching_count:,} rows, {len(matching_grouped):,} unique pairs)"):
                        st.dataframe(matching_grouped, use_container_width=True, hide_index=True)
                        if matching_count > 1000:
                            st.caption(f"Showing grouped results from first 1,000 of {matching_count:,} matching rows")
                
                if non_matching_df is not None and len(non_matching_df) > 0:
                    # Group non-matching values by column pairs and count occurrences
                    non_matching_grouped = non_matching_df.groupby([col1, col2]).size().reset_index(name='Count')
                    non_matching_grouped = non_matching_grouped.sort_values('Count', ascending=False)
                    
                    with st.expander(f"⚠️ Non-matching values ({non_matching_count:,} rows, {len(non_matching_grouped):,} unique pairs)"):
                        st.dataframe(non_matching_grouped, use_container_width=True, hide_index=True)
                        if non_matching_count > 1000:
                            st.caption(f"Showing grouped results from first 1,000 of {non_matching_count:,} non-matching rows")
            
            with button_col:
                if col1_exists and col2_exists:
                    if st.button(f"Keep only {col1}", key=_create_button_key(col1, col2, idx, "keep1"), use_container_width=True):
                        _handle_keep_column(col1, col2)
                    
                    if st.button(f"Keep only {col2}", key=_create_button_key(col1, col2, idx, "keep2"), use_container_width=True):
                        _handle_keep_column(col2, col1)
                    
                    if st.button("Concatenate", key=_create_button_key(col1, col2, idx, "concat"), use_container_width=True):
                        _handle_concatenate(col1, col2)
                    
                    if st.button("Remove both", key=_create_button_key(col1, col2, idx, "remove_both"), use_container_width=True):
                        _handle_remove_both(col1, col2)
                else:
                    # Show which columns were removed
                    if not col1_exists and not col2_exists:
                        st.info("Both removed")
                    elif not col1_exists:
                        st.info(f"{col1} removed")
                    else:
                        st.info(f"{col2} removed")
            
            st.divider()


# ============================================================================
# Main Render Function
# ============================================================================

def render():
    """Render the 1:N Relationship Analysis tab."""
    st.header("1:N Relationship Analysis")
    st.write("Analyze columns to find 1:N (one-to-many) relationships between pairs of columns.")
    
    # Check if data is available
    if not _has_data():
        st.info("No data loaded. Please import a CSV file in the 'Import Data' tab.")
        return
    
    available_columns = _get_available_columns()
    
    if len(available_columns) < 2:
        st.warning("Need at least 2 columns to analyze 1:N relationships.")
        return
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider(
            "Minimum percentage for 1:N relationship",
            min_value=0,
            max_value=100,
            value=95,
            step=5,
            help="Only show column pairs that have at least this percentage of values in a 1:N relationship",
            key="one_to_many_threshold"
        )
    
    with col2:
        drop_na = st.checkbox(
            "Drop NA values",
            value=True,
            help="If checked, rows with NA/null values in either column will be excluded from analysis",
            key="one_to_many_drop_na"
        )
    
    # Analyze button
    if st.button("Analyze 1:N Relationships", type="primary", key="one_to_many_analyze"):
        analyze_one_to_many(st.session_state.processed_df, threshold / 100, drop_na)
    
    # Show previous results if available
    if 'one_to_many_results' in st.session_state and st.session_state.one_to_many_results:
        # Filter and update results to only include pairs where both columns still exist
        valid_results = _filter_valid_results(st.session_state.one_to_many_results)
        
        if len(valid_results) != len(st.session_state.one_to_many_results):
            st.session_state.one_to_many_results = valid_results
        
        # Display results if any are valid
        if valid_results:
            display_results(valid_results, threshold / 100)

