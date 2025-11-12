import streamlit as st
import pandas as pd
import hashlib


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
    if 'one_to_one_results' not in st.session_state:
        return
    
    if not st.session_state.one_to_one_results:
        return
    
    st.session_state.one_to_one_results = [
        result for result in st.session_state.one_to_one_results
        if result['column1'] not in columns_to_remove and 
           result['column2'] not in columns_to_remove
    ]


# ============================================================================
# 1:1 Relationship Analysis
# ============================================================================

def _calculate_one_to_one_percentage(pair_df, col1, col2):
    """
    Calculate the percentage of rows that have a 1:1 relationship.
    
    Returns: (percentage, valid_rows, valid_mask)
    A 1:1 relationship means:
    - Each unique value in col1 maps to exactly one unique value in col2
    - Each unique value in col2 maps to exactly one unique value in col1
    """
    total_rows = len(pair_df)
    
    # Quick check: if all groups are perfect 1:1, return 100%
    col1_counts = pair_df.groupby(col1)[col2].nunique()
    col2_counts = pair_df.groupby(col2)[col1].nunique()
    
    if (col1_counts == 1).all() and (col2_counts == 1).all():
        valid_mask = pd.Series([True] * total_rows, index=pair_df.index)
        return 100.0, total_rows, valid_mask
    
    # Estimate minimum percentage for early exit optimization
    col1_violations = (col1_counts > 1).sum()
    col2_violations = (col2_counts > 1).sum()
    
    if len(col1_counts) > 0:
        min_pct_col1 = ((len(col1_counts) - col1_violations) / len(col1_counts)) * 100
    else:
        min_pct_col1 = 0
    
    if len(col2_counts) > 0:
        min_pct_col2 = ((len(col2_counts) - col2_violations) / len(col2_counts)) * 100
    else:
        min_pct_col2 = 0
    
    min_estimated_percentage = min(min_pct_col1, min_pct_col2)
    
    # Detailed row-by-row check for accurate percentage
    col1_valid = pair_df.groupby(col1)[col2].transform('nunique') == 1
    col2_valid = pair_df.groupby(col2)[col1].transform('nunique') == 1
    valid_mask = col1_valid & col2_valid
    valid_rows = valid_mask.sum()
    percentage = (valid_rows / total_rows) * 100 if total_rows > 0 else 0
    
    return percentage, valid_rows, valid_mask


def _analyze_column_pair(df_original, df_clean, col1, col2, threshold, drop_na):
    """Analyze a single column pair for 1:1 relationship."""
    pair_df = df_clean[[col1, col2]].copy()
    
    if len(pair_df) == 0:
        return None
    
    percentage, valid_rows, valid_mask = _calculate_one_to_one_percentage(pair_df, col1, col2)
    
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


def analyze_one_to_one(df, threshold, drop_na=True):
    """Analyze 1:1 relationships between all column pairs."""
    st.subheader("Analysis Results")
    
    columns = _get_available_columns()
    results = []
    
    # Pre-filter dataframe if dropping NA
    df_clean = df.dropna(subset=columns) if drop_na else df
    
    with st.spinner("Analyzing column pairs..."):
        total_pairs = len(columns) * (len(columns) - 1) // 2
        progress_bar = st.progress(0) if total_pairs > 10 else None
        status_text = st.empty() if total_pairs > 10 else None
        
        pair_count = 0
        
        # Check all pairs of columns (only once per pair, since 1:1 is symmetric)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                pair_count += 1
                
                # Update progress
                if progress_bar:
                    progress_bar.progress(pair_count / total_pairs)
                if status_text:
                    status_text.text(f"Analyzing pair {pair_count}/{total_pairs}: {col1} ↔ {col2}")
                
                # Analyze this pair
                result = _analyze_column_pair(df, df_clean, col1, col2, threshold, drop_na)
                if result:
                    results.append(result)
        
        # Clear progress indicators
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
    
    # Sort by percentage descending
    results.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Store results in session state
    st.session_state.one_to_one_results = results
    
    # Display results
    display_results(results, threshold)


# ============================================================================
# Action Handlers
# ============================================================================

def _handle_keep_column(col_to_keep, col_to_drop):
    """Keep one column and drop another, removing affected results."""
    if col_to_drop not in st.session_state.processed_df.columns:
        return
    
    st.session_state.processed_df = st.session_state.processed_df.drop(columns=[col_to_drop])
    _remove_results_involving_columns([col_to_drop])
    _add_message(f"🗑️ **Dropped column {col_to_drop}** (kept **{col_to_keep}**) - from 1:1 analysis")
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
    
    _add_message(f"➕ **Created new column {new_col_name}** (concatenated **{col1}** and **{col2}**, then dropped original columns) - from 1:1 analysis")
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
        _add_message(f"🗑️ **Dropped columns {col1} and {col2}** - from 1:1 analysis")
        st.rerun()


# ============================================================================
# Display Functions
# ============================================================================

def _create_button_key(col1, col2, idx, action):
    """Create a unique key for a button."""
    pair_hash = hashlib.md5(f"{col1}_{col2}".encode()).hexdigest()[:8]
    return f"{action}_{idx}_{pair_hash}"


def display_results(results, threshold):
    """Display the 1:1 relationship results with action buttons."""
    threshold_percent = threshold * 100
    
    if not results:
        st.info(f"No column pairs found with ≥{threshold_percent:.0f}% 1:1 relationship.")
        return
    
    st.success(f"Found {len(results)} column pair(s) with ≥{threshold_percent:.0f}% 1:1 relationship:")
    
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
                st.write(f"**{col1} ↔ {col2}**")
                info_line = f"- 1:1 Relationship: {percentage:.2f}% | Valid rows: {valid_rows:,} / {total_rows:,}"
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
    """Render the 1:1 Relationship Analysis tab."""
    st.header("1:1 Relationship Analysis")
    st.write("Analyze columns to find 1:1 relationships between pairs of columns.")
    
    # Check if data is available
    if not _has_data():
        st.info("No data loaded. Please import a CSV file in the 'Import Data' tab.")
        return
    
    available_columns = _get_available_columns()
    
    if len(available_columns) < 2:
        st.warning("Need at least 2 columns to analyze 1:1 relationships.")
        return
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider(
            "Minimum percentage for 1:1 relationship",
            min_value=0,
            max_value=100,
            value=95,
            step=5,
            help="Only show column pairs that have at least this percentage of values in a 1:1 relationship"
        )
    
    with col2:
        drop_na = st.checkbox(
            "Drop NA values",
            value=True,
            help="If checked, rows with NA/null values in either column will be excluded from analysis"
        )
    
    # Analyze button
    if st.button("Analyze 1:1 Relationships", type="primary"):
        analyze_one_to_one(st.session_state.processed_df, threshold / 100, drop_na)
    
    # Show previous results if available
    if 'one_to_one_results' in st.session_state and st.session_state.one_to_one_results:
        # Filter and update results to only include pairs where both columns still exist
        valid_results = _filter_valid_results(st.session_state.one_to_one_results)
        
        if len(valid_results) != len(st.session_state.one_to_one_results):
            st.session_state.one_to_one_results = valid_results
        
        # Display results if any are valid
        if valid_results:
            display_results(valid_results, threshold / 100)
