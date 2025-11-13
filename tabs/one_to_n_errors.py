import streamlit as st
import pandas as pd
import hashlib
from helpers.one_to_many import calculate_one_to_many_violation_ratio
from helpers.one_to_many_corrections import calculate_corrections_for_pair, prepare_pair_df, filter_high_confidence_corrections


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
    df_cols = set(st.session_state.processed_df.columns)
    return col1 in df_cols, col2 in df_cols


def _both_columns_exist(col1, col2):
    """Check if both columns exist in processed_df."""
    col1_exists, col2_exists = _columns_exist(col1, col2)
    return col1_exists and col2_exists


def _filter_valid_results(results):
    """Filter results to only include pairs where both columns still exist."""
    return [r for r in results if _both_columns_exist(r['column1'], r['column2'])]


def _add_message(message):
    """Add a message to the shared messages log."""
    if 'shared_messages' not in st.session_state:
        st.session_state.shared_messages = []
    st.session_state.shared_messages.append(message)




# ============================================================================
# 1:N Relationship Analysis
# ============================================================================

def _analyze_column_pair(df_original, df_clean, col1, col2, threshold, drop_na):
    """Analyze a single column pair for 1:N relationship."""
    pair_df = prepare_pair_df(df_clean, col1, col2, drop_na)
    if pair_df is None:
        return None
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(pair_df, col1, col2)
    
    if percentage < threshold * 100:
        return None
    
    non_matching_df = pair_df[~valid_mask].head(1000).reset_index(drop=True)
    
    return {
        'column1': col1,
        'column2': col2,
        'percentage': percentage,
        'violation_ratio': violation_ratio,
        'valid_rows': int(valid_rows),
        'total_rows': total_rows,
        'unique_col1': pair_df[col1].nunique(),
        'unique_col2': pair_df[col2].nunique(),
        'na_col1': int(df_original[col1].isna().sum()),
        'na_col2': int(df_original[col2].isna().sum()),
        'drop_na': drop_na,
        'non_matching_df': non_matching_df,
        'non_matching_count': len(pair_df) - int(valid_rows)
    }


def _update_progress(progress_bar, status_text, current_pair, total_pairs, col1, col2):
    """Update progress bar and status text."""
    if progress_bar:
        progress_bar.progress(current_pair / total_pairs)
    if status_text:
        status_text.text(f"Analyzing pair {current_pair//2}/{total_pairs//2}: {col1} ↔ {col2}")


def analyze_one_to_n_errors(df, threshold, drop_na=True):
    """Analyze 1:N relationships between all column pairs and identify errors."""
    st.subheader("Analysis Results")
    
    columns = _get_available_columns()
    results = []
    
    with st.spinner("Analyzing column pairs..."):
        total_pairs = len(columns) * (len(columns) - 1)
        show_progress = total_pairs > 10
        progress_bar = st.progress(0) if show_progress else None
        status_text = st.empty() if show_progress else None
        
        pair_count = 0
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                pair_count += 2  # Both directions
                
                _update_progress(progress_bar, status_text, pair_count, total_pairs, col1, col2)
                
                # Check both directions
                for parent, child in [(col1, col2), (col2, col1)]:
                    result = _analyze_column_pair(df, df, parent, child, threshold, drop_na)
                    if result:
                        results.append(result)
        
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
    
    results.sort(key=lambda x: x['violation_ratio'], reverse=True)
    st.session_state.one_to_n_errors_results = results


# ============================================================================
# Auto-Correction Functions
# ============================================================================


def _get_confident_correction_keys(correction_plan):
    """Extract set of confident correction keys (violating_value, correct_value)."""
    if 'confident_corrections' not in correction_plan:
        return set()
    return {(c['violating_value'], c['correct_value']) 
            for c in correction_plan['confident_corrections']}


def _build_preview_row(ref_col_val, current_val, new_value, count, is_confident, 
                       ref_col_name, col_to_correct_name, new_value_name):
    """Build a single row for the preview table."""
    return {
        ref_col_name: ref_col_val,
        col_to_correct_name: current_val,
        new_value_name: new_value,
        'number of values': count,
        '_is_confident': is_confident,
        '_violating_value': ref_col_val
    }


def _generate_preview_data(correction_plan):
    """Generate preview data for display as a pivot table with confidence indicators."""
    if not correction_plan or not correction_plan.get('corrections'):
        return None
    
    col_to_correct = correction_plan['column_to_correct']
    ref_col = correction_plan['reference_column']
    
    # Prepare column names
    ref_col_name = f"Column not to be corrected ({ref_col})"
    col_to_correct_name = f"Column to be corrected ({col_to_correct})"
    new_value_name = f"New value ({col_to_correct})"
    
    # Get confident correction keys
    confident_keys = _get_confident_correction_keys(correction_plan)
    
    # Build pivot table data
    pivot_data = []
    for correction in correction_plan['corrections']:
        ref_col_val = correction['violating_value']
        new_value = correction['correct_value']
        is_confident = (ref_col_val, new_value) in confident_keys
        
        for current_val, count in correction['current_values'].items():
            row = _build_preview_row(
                ref_col_val, current_val, new_value, count, is_confident,
                ref_col_name, col_to_correct_name, new_value_name
            )
            pivot_data.append(row)
    
    if not pivot_data:
        return None
    
    # Create and sort DataFrame
    pivot_table = pd.DataFrame(pivot_data)
    pivot_table = pivot_table.sort_values([ref_col_name, col_to_correct_name])
    
    return {
        'summary': pivot_table,
        'total_changes': correction_plan['total_changes'],
        'col_to_correct_name': col_to_correct_name,
        'new_value_name': new_value_name
    }


def _get_corrections_to_apply(plan, use_confident_only):
    """Get the list of corrections to apply from a plan."""
    if use_confident_only:
        return plan.get('confident_corrections', [])
    return plan.get('corrections', [])


def _apply_correction_plan(plan, use_confident_only=False):
    """Apply a single correction plan to processed_df."""
    corrections_to_apply = _get_corrections_to_apply(plan, use_confident_only)
    
    if not corrections_to_apply:
        return 0
    
    col_to_correct = plan['column_to_correct']
    total_corrected = 0
    
    for correction in corrections_to_apply:
        correct_val = correction['correct_value']
        for idx in correction['rows_to_correct']:
            if idx in st.session_state.processed_df.index:
                st.session_state.processed_df.loc[idx, col_to_correct] = correct_val
                total_corrected += 1
    
    return total_corrected


def _apply_corrections(correction_plans, use_confident_only=False, reanalyze=False, threshold=None, drop_na=None):
    """Apply corrections to processed_df."""
    if not correction_plans:
        return 0
    
    total_corrected = 0
    corrected_pairs = []
    
    for plan in correction_plans:
        num_corrected = _apply_correction_plan(plan, use_confident_only=use_confident_only)
        if num_corrected > 0:
            total_corrected += num_corrected
            corrected_pairs.append(plan['direction'])
    
    if total_corrected > 0:
        pairs_str = ', '.join(corrected_pairs)
        correction_type = "confident" if use_confident_only else "all"
        _add_message(f"🔧 **Auto-corrected {total_corrected:,} {correction_type} value(s)** in {len(corrected_pairs)} column pair(s): {pairs_str} - from 1:N Errors analysis")
        
        if reanalyze and threshold is not None:
            analyze_one_to_n_errors(st.session_state.processed_df, threshold, drop_na)
    
    return total_corrected


def calculate_corrections_for_all_pairs(df, results, drop_na, min_count_threshold=5):
    """Calculate corrections for all column pairs with errors and filter by confidence."""
    correction_plans = []
    RATIO_THRESHOLD = 3.0
    
    for result in results:
        parent_col, child_col = result['column1'], result['column2']
        
        if not _both_columns_exist(parent_col, child_col):
            continue
        
        plan = calculate_corrections_for_pair(df, parent_col, child_col, drop_na)
        if plan:
            plan = filter_high_confidence_corrections(plan, min_count_threshold, ratio_threshold=RATIO_THRESHOLD)
            plan['result_col1'] = parent_col
            plan['result_col2'] = child_col
            correction_plans.append(plan)
    
    return correction_plans


# ============================================================================
# Display Functions
# ============================================================================

def _get_row_background_color(is_confident):
    """Get background color for a row based on confidence."""
    if is_confident:
        return ['background-color: #d4edda']  # Light green for confident
    return ['background-color: #fff3cd']  # Light orange for non-confident


def _style_preview_table(df, col_to_correct_name, new_value_name):
    """Apply styling to highlight rows: green for confident corrections, orange for non-confident."""
    # Store confidence info before dropping helper columns
    if '_is_confident' in df.columns:
        confidence_series = df['_is_confident'].copy()
    else:
        confidence_series = pd.Series([False] * len(df), index=df.index)
    
    # Remove helper columns before styling
    display_df = df.drop(columns=['_is_confident', '_violating_value'], errors='ignore')
    
    def highlight_corrections(row):
        """Return background color based on correction confidence."""
        # Only highlight rows that will be corrected (current != new)
        if row[col_to_correct_name] != row[new_value_name]:
            is_confident = confidence_series.loc[row.name] if row.name in confidence_series.index else False
            return _get_row_background_color(is_confident) * len(row)
        return [''] * len(row)
    
    return display_df.style.apply(highlight_corrections, axis=1)


def _display_correction_preview(correction_plan):
    """Display correction preview for a single pair."""
    preview_data = _generate_preview_data(correction_plan)
    if not preview_data:
        return
    
    confident_changes = correction_plan.get('confident_total_changes', 0)
    non_confident_changes = correction_plan.get('non_confident_total_changes', 0)
    total_changes = preview_data['total_changes']
    
    with st.expander(f"🔧 Correction Preview ({total_changes:,} changes)"):
        st.write(f"**Will correct: {correction_plan['column_to_correct']}** (based on {correction_plan['reference_column']})")
        st.write(f"**Direction: {correction_plan['direction']}**")
        
        if confident_changes > 0 or non_confident_changes > 0:
            st.write(f"**Confident corrections**: {confident_changes:,} | **Non-confident corrections**: {non_confident_changes:,}")
        
        st.subheader("Summary Table")
        styled_summary = _style_preview_table(
            preview_data['summary'],
            preview_data['col_to_correct_name'],
            preview_data['new_value_name']
        )
        st.dataframe(styled_summary, use_container_width=True, hide_index=True)


def _display_error_table(result):
    """Display error table for a column pair."""
    col1, col2 = result['column1'], result['column2']
    non_matching_df = result.get('non_matching_df')
    non_matching_count = result.get('non_matching_count', 0)
    
    if non_matching_df is None or len(non_matching_df) == 0:
        st.info("No errors found for this column pair.")
        return
    
    non_matching_grouped = (non_matching_df.groupby([col1, col2])
                           .size()
                           .reset_index(name='Count')
                           .sort_values('Count', ascending=False))
    
    unique_pairs_count = len(non_matching_grouped)
    with st.expander(f"⚠️ Non-matching values (errors) ({non_matching_count:,} rows, {unique_pairs_count:,} unique pairs)"):
        st.dataframe(non_matching_grouped, use_container_width=True, hide_index=True)
        if non_matching_count > 1000:
            st.caption(f"Showing grouped results from first 1,000 of {non_matching_count:,} non-matching rows")


def _generate_button_key(col1, col2, idx, suffix):
    """Generate a unique button key."""
    pair_hash = hashlib.md5(f"{col1}_{col2}_{idx}_{suffix}".encode()).hexdigest()[:12]
    return f"auto_correct_{suffix}_{idx}_{pair_hash}"


def _create_correction_button(col1, col2, idx, label, changes_count, use_confident_only, 
                               correction_plan, threshold, drop_na):
    """Create a correction button and handle its click."""
    button_key = _generate_button_key(col1, col2, idx, "confident" if use_confident_only else "all")
    
    if st.button(label, key=button_key, use_container_width=True, type="secondary"):
        correction_type = "confident" if use_confident_only else "all"
        with st.spinner(f"Applying {correction_type} corrections for {col1} → {col2}..."):
            _apply_corrections([correction_plan], use_confident_only=use_confident_only, 
                              reanalyze=True, threshold=threshold, drop_na=drop_na)
            st.success(f"✅ Corrected {changes_count:,} {correction_type} value(s) for {col1} → {col2}")
            st.rerun()


def _display_correction_buttons(col1, col2, idx, correction_plan, threshold, drop_na):
    """Display correction buttons for a column pair."""
    if not correction_plan or correction_plan['total_changes'] == 0:
        return
    
    button_col1, button_col2 = st.columns(2)
    confident_changes = correction_plan.get('confident_total_changes', 0)
    
    with button_col1:
        if confident_changes > 0:
            _create_correction_button(
                col1, col2, idx, f"✅ Confident ({confident_changes:,})",
                confident_changes, True, correction_plan, threshold, drop_na
            )
    
    with button_col2:
        _create_correction_button(
            col1, col2, idx, f"🔧 All ({correction_plan['total_changes']:,})",
            correction_plan['total_changes'], False, correction_plan, threshold, drop_na
        )


def _display_result_statistics(result, col1, col2):
    """Display statistics for a result."""
    violation_ratio = result.get('violation_ratio', 0)
    st.write(f"- 1:N Relationship: {result['percentage']:.2f}% | "
            f"Valid rows: {result['valid_rows']:,} / {result['total_rows']:,} | "
            f"Violation ratio: {violation_ratio:.4f}")
    
    # Column stats
    if result.get('drop_na', False):
        st.write(f"- {col1}: {result['unique_col1']:,} unique, {result['na_col1']:,} NA | "
                f"{col2}: {result['unique_col2']:,} unique, {result['na_col2']:,} NA")
    else:
        st.write(f"- {col1}: {result['na_col1']:,} NA | {col2}: {result['na_col2']:,} NA")


def _display_column_warnings(col1, col2):
    """Display warnings if columns were removed."""
    col1_exists, col2_exists = _columns_exist(col1, col2)
    if not col1_exists or not col2_exists:
        removed = [col for col, exists in [(col1, col1_exists), (col2, col2_exists)] if not exists]
        st.warning(f"⚠️ Column(s) removed: {', '.join(removed)}")


def _find_matching_correction_plan(correction_plans, col1, col2):
    """Find the correction plan matching a column pair."""
    return next((p for p in correction_plans 
                if p['result_col1'] == col1 and p['result_col2'] == col2), None)


def display_results(results, threshold, drop_na, min_count_threshold=5):
    """Display the 1:N relationship errors (non-matching values only)."""
    threshold_percent = threshold * 100
    
    if not results:
        st.info(f"No column pairs found with ≥{threshold_percent:.0f}% 1:N relationship.")
        return
    
    st.success(f"Found {len(results)} column pair(s) with ≥{threshold_percent:.0f}% 1:N relationship:")
    
    correction_plans = calculate_corrections_for_all_pairs(
        st.session_state.processed_df, results, drop_na, min_count_threshold
    )
    
    # Show auto-correct all button (only confident corrections)
    if correction_plans:
        total_confident_changes = sum(plan.get('confident_total_changes', 0) 
                                     for plan in correction_plans)
        
        if total_confident_changes > 0:
            if st.button(f"🔧 Auto-correct confident errors ({total_confident_changes:,} changes)", 
                        type="primary", key="auto_correct_all"):
                with st.spinner("Applying confident corrections..."):
                    _apply_corrections(correction_plans, use_confident_only=True, 
                                     reanalyze=True, threshold=threshold, drop_na=drop_na)
                    st.success(f"✅ Corrected {total_confident_changes:,} confident value(s) "
                             f"across {len(correction_plans)} column pair(s)")
                    st.rerun()
        
        st.info("💡 **Color coding**: 🟢 Green = Confident corrections | "
               "🟠 Orange = Non-confident corrections")
    
    # Display each result
    for idx, result in enumerate(results):
        col1, col2 = result['column1'], result['column2']
        correction_plan = _find_matching_correction_plan(correction_plans, col1, col2)
        
        with st.container():
            # Header with correction buttons
            header_col1, header_col2 = st.columns([3, 2])
            with header_col1:
                st.write(f"**{col1} → {col2}**")
            with header_col2:
                _display_correction_buttons(col1, col2, idx, correction_plan, threshold, drop_na)
            
            # Statistics
            _display_result_statistics(result, col1, col2)
            
            # Correction preview
            if correction_plan and correction_plan['total_changes'] > 0:
                _display_correction_preview(correction_plan)
            
            # Error table
            _display_error_table(result)
            
            # Column removal warnings
            _display_column_warnings(col1, col2)
            
            st.divider()


# ============================================================================
# Main Render Function
# ============================================================================

def render():
    """Render the 1:N Errors tab."""
    st.header("1:N Errors")
    st.write("Analyze columns to find and display errors (non-matching values) in 1:N relationships.")
    
    if not _has_data():
        st.info("No data loaded. Please import a CSV file in the 'Import Data' tab.")
        return
    
    available_columns = _get_available_columns()
    if len(available_columns) < 2:
        st.warning("Need at least 2 columns to analyze 1:N errors.")
        return
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        threshold = st.slider(
            "Minimum percentage for 1:N relationship",
            min_value=0, max_value=100, value=95, step=1,
            help="Only show column pairs that have at least this percentage of values in a 1:N relationship",
            key="one_to_n_errors_threshold"
        )
    with col2:
        drop_na = st.checkbox(
            "Drop NA values", value=True,
            help="If checked, rows with NA/null values in either column will be excluded from analysis",
            key="one_to_n_errors_drop_na"
        )
    with col3:
        min_count_threshold = st.slider(
            "Minimum count threshold",
            min_value=1, max_value=50, value=5, step=1,
            help="Minimum number of occurrences required for confident corrections",
            key="one_to_n_errors_min_count"
        )
    
    # Analyze button
    if st.button("Analyze 1:N Errors", type="primary", key="one_to_n_errors_analyze"):
        analyze_one_to_n_errors(st.session_state.processed_df, threshold / 100, drop_na)
    
    # Show previous results
    if 'one_to_n_errors_results' in st.session_state and st.session_state.one_to_n_errors_results:
        valid_results = _filter_valid_results(st.session_state.one_to_n_errors_results)
        if len(valid_results) != len(st.session_state.one_to_n_errors_results):
            st.session_state.one_to_n_errors_results = valid_results
        
        if valid_results:
            display_results(valid_results, threshold / 100, drop_na, min_count_threshold)
