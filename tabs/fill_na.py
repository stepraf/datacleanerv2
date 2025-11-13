import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import hashlib

# Maximum number of rows to display in dataframes
MAX_DISPLAY_ROWS = 1000


# ============================================================================
# Helper Functions
# ============================================================================

def _has_processed_data():
    """Check if processed data is available."""
    return ('processed_df' in st.session_state and 
            st.session_state.processed_df is not None and 
            len(st.session_state.processed_df) > 0)


def _get_available_columns():
    """Get list of available columns excluding initial_id."""
    if not _has_processed_data():
        return []
    return [col for col in st.session_state.processed_df.columns if col != 'initial_id']


def _is_numeric(series):
    """Check if a series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series):
    """Check if a series is categorical (object type with limited unique values)."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        # Consider categorical if unique values are less than 50% of total non-null values
        non_null_count = series.notna().sum()
        if non_null_count > 0:
            unique_count = series.nunique()
            return unique_count < (non_null_count * 0.5) or unique_count < 100
    return False


def _add_message(message):
    """Add a message to the shared messages log."""
    if 'shared_messages' not in st.session_state:
        st.session_state.shared_messages = []
    st.session_state.shared_messages.append(message)


# ============================================================================
# Mutual Information Analysis
# ============================================================================

@st.cache_data(show_spinner=False)
def _calculate_mutual_information_cached(df_hash, target_col, feature_cols, df_subset):
    """
    Cached version of mutual information calculation.
    Uses dataframe hash to detect changes.
    """
    # Reconstruct the dataframe subset for calculation
    df = pd.DataFrame(df_subset)
    
    # Prepare target column
    target_series = df[target_col].copy()
    
    # Remove rows where target is NA for MI calculation
    valid_mask = target_series.notna()
    if valid_mask.sum() == 0:
        return {}
    
    target_clean = target_series[valid_mask]
    df_clean = df.loc[valid_mask, feature_cols].copy()
    
    # Determine if target is numeric or categorical
    target_is_numeric = _is_numeric(target_clean)
    
    mi_scores = {}
    
    for feature_col in feature_cols:
        if feature_col == target_col:
            continue
            
        feature_series = df_clean[feature_col].copy()
        
        # Skip if feature has no valid values
        if feature_series.notna().sum() == 0:
            continue
        
        # Handle missing values in feature by dropping them
        feature_valid_mask = feature_series.notna()
        if feature_valid_mask.sum() == 0:
            continue
        
        feature_clean = feature_series[feature_valid_mask]
        target_for_feature = target_clean[feature_valid_mask]
        
        # Prepare feature for sklearn
        if _is_numeric(feature_clean):
            # Numeric feature: use as is
            X = feature_clean.values.reshape(-1, 1)
        else:
            # Categorical feature: encode
            le = LabelEncoder()
            try:
                X = le.fit_transform(feature_clean.astype(str)).reshape(-1, 1)
            except:
                continue
        
        # Calculate mutual information
        try:
            if target_is_numeric:
                mi = mutual_info_regression(X, target_for_feature.values, random_state=42)[0]
            else:
                # Encode target if categorical
                le_target = LabelEncoder()
                y = le_target.fit_transform(target_for_feature.astype(str))
                mi = mutual_info_classif(X, y, random_state=42)[0]
            
            mi_scores[feature_col] = mi
        except Exception as e:
            # Skip if calculation fails
            continue
    
    return mi_scores


def _calculate_mutual_information(df, target_col, feature_cols):
    """
    Calculate mutual information between target column and feature columns.
    Auto-detects numeric vs categorical and uses appropriate method.
    Uses caching when possible.
    """
    # Create a hash of the relevant dataframe subset for caching
    # Only hash the columns we need to avoid unnecessary computation
    relevant_cols = [target_col] + [col for col in feature_cols if col != target_col]
    df_subset = df[relevant_cols].copy()
    
    # Create a hash of the dataframe for cache key
    # Use a combination of shape and data hash
    try:
        # Try using pandas hash function if available
        hash_values = pd.util.hash_pandas_object(df_subset).values
        df_hash = hashlib.md5(hash_values.tobytes()).hexdigest()
    except (AttributeError, TypeError):
        # Fallback: use shape and sample of data
        sample_data = str(df_subset.shape) + str(df_subset.head(100).values.tobytes())
        df_hash = hashlib.md5(sample_data.encode()).hexdigest()
    
    # Convert dataframe to dict for caching (more efficient than passing full dataframe)
    df_dict = df_subset.to_dict('list')
    
    # Use cached version if available
    return _calculate_mutual_information_cached(df_hash, target_col, feature_cols, df_dict)


# ============================================================================
# Group-based Conditional Probability Calculation
# ============================================================================

@st.cache_data(show_spinner=False)
def _calculate_conditional_probabilities_cached(df_hash, target_col, group_cols, df_subset):
    """
    Cached version of conditional probability calculation.
    """
    # Reconstruct the dataframe subset for calculation
    df = pd.DataFrame(df_subset)
    
    # Create a copy for analysis
    analysis_df = df[[target_col] + group_cols].copy()
    
    # Remove rows where any grouping column is NA
    analysis_df = analysis_df.dropna(subset=group_cols)
    
    if len(analysis_df) == 0:
        return {}
    
    # Group by the grouping columns
    grouped = analysis_df.groupby(group_cols)
    
    conditional_probs = {}
    
    for group_key, group_df in grouped:
        # Ensure group_key is a tuple and convert to Python native types for consistent matching
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        
        # Convert pandas scalar types to Python native types for consistent matching
        # This ensures tuple keys match when we look them up later
        normalized_key = tuple()
        for val in group_key:
            # Convert pandas scalars to Python native types
            if pd.isna(val):
                normalized_key += (val,)  # Keep NaN as is
            elif hasattr(val, 'item'):
                try:
                    normalized_key += (val.item(),)
                except (ValueError, AttributeError):
                    normalized_key += (val,)
            else:
                normalized_key += (val,)
        
        target_values = group_df[target_col].dropna()
        
        if len(target_values) == 0:
            continue
        
        # Calculate probability distribution
        if _is_numeric(target_values):
            # For numeric: calculate mean, median, mode
            mean_val = target_values.mean()
            median_val = target_values.median()
            mode_val = target_values.mode()
            mode_val = mode_val.iloc[0] if len(mode_val) > 0 else None
            
            conditional_probs[normalized_key] = {
                'type': 'numeric',
                'mean': mean_val,
                'median': median_val,
                'mode': mode_val,
                'count': len(target_values),
                'std': target_values.std() if len(target_values) > 1 else 0
            }
        else:
            # For categorical: calculate probability distribution
            value_counts = target_values.value_counts()
            total = len(target_values)
            probs = (value_counts / total).to_dict()
            
            conditional_probs[normalized_key] = {
                'type': 'categorical',
                'probabilities': probs,
                'most_likely': value_counts.index[0] if len(value_counts) > 0 else None,
                'count': total
            }
    
    return conditional_probs


def _calculate_conditional_probabilities(df, target_col, group_cols):
    """
    Calculate conditional probabilities for target column given group columns.
    Returns a dictionary mapping group combinations to probability distributions.
    Uses caching when possible.
    """
    # Create a hash of the relevant dataframe subset for caching
    relevant_cols = [target_col] + group_cols
    df_subset = df[relevant_cols].copy()
    
    # Create a hash of the dataframe for cache key
    try:
        # Try using pandas hash function if available
        hash_values = pd.util.hash_pandas_object(df_subset).values
        df_hash = hashlib.md5(hash_values.tobytes()).hexdigest()
    except (AttributeError, TypeError):
        # Fallback: use shape and sample of data
        sample_data = str(df_subset.shape) + str(df_subset.head(100).values.tobytes())
        df_hash = hashlib.md5(sample_data.encode()).hexdigest()
    
    # Convert dataframe to dict for caching
    df_dict = df_subset.to_dict('list')
    
    # Use cached version if available
    return _calculate_conditional_probabilities_cached(df_hash, target_col, group_cols, df_dict)


def _calculate_confidence_score(conditional_prob, mi_score, sample_size, max_mi):
    """
    Calculate confidence score based on conditional probability, mutual information, and sample size.
    
    The confidence score is a weighted combination of three factors that indicate how reliable
    a predicted value is:
    1. Conditional Probability (40% weight): How likely the predicted value is given the group
    2. Normalized Mutual Information (30% weight): How strong the dependency relationship is
    3. Normalized Sample Size (30% weight): How much data we have to base the prediction on
    
    Args:
        conditional_prob: The conditional probability of the predicted value (0 to 1)
            - For categorical: probability of the most likely value in the group
            - For numeric: inverse of normalized std (1 / (1 + std)), higher when values are similar
        mi_score: Mutual information score for the feature (raw value, not normalized)
        sample_size: Number of samples in the group used for prediction
        max_mi: Maximum MI score across all features (used for normalization)
    
    Returns:
        Confidence score between 0 and 1, where:
        - 1.0 = very high confidence (high prob, strong dependency, large sample)
        - 0.0 = very low confidence (low prob, weak dependency, small sample)
    """
    # Step 1: Normalize Mutual Information (0 to 1 scale)
    # Mutual information scores can vary widely, so we normalize by dividing by the maximum
    # This tells us: "How strong is this dependency relative to the strongest dependency?"
    # Example: If max_mi = 0.5 and mi_score = 0.25, normalized_mi = 0.5 (moderate dependency)
    normalized_mi = mi_score / max_mi if max_mi > 0 else 0
    
    # Step 2: Normalize Sample Size (0 to 1 scale) using logarithmic scaling
    # Sample sizes can range from 1 to thousands, so we use log10 to compress the scale.
    # The formula: log10(sample_size) / log10(100) means:
    # - sample_size = 1   → normalized ≈ 0.0  (very low confidence)
    # - sample_size = 10  → normalized ≈ 0.5  (moderate confidence)
    # - sample_size = 100 → normalized = 1.0  (high confidence, capped)
    # - sample_size > 100 → normalized = 1.0  (capped at 1.0)
    # This reflects that confidence increases quickly from 1-100 samples, but levels off after 100.
    normalized_sample = min(1.0, np.log10(max(sample_size, 1)) / np.log10(100))
    
    # Step 3: Weighted Combination
    # We combine the three normalized components using weighted averaging:
    # - 40% weight on conditional_prob: Most important - directly measures prediction likelihood
    # - 30% weight on normalized_mi: Important - measures strength of dependency relationship
    # - 30% weight on normalized_sample: Important - measures reliability based on data quantity
    #
    # Example calculation:
    #   conditional_prob = 0.8 (80% probability)
    #   normalized_mi = 0.6 (60% of max dependency strength)
    #   normalized_sample = 0.7 (sample size between 10-100)
    #   confidence = 0.4*0.8 + 0.3*0.6 + 0.3*0.7 = 0.32 + 0.18 + 0.21 = 0.71 (71% confidence)
    #
    # The weights (40/30/30) prioritize conditional probability because it's the most direct
    # measure of prediction quality, while still giving significant weight to dependency
    # strength and sample size to ensure reliability.
    confidence = (
        0.4 * conditional_prob +      # 40%: How likely is this value given the group?
        0.3 * normalized_mi +          # 30%: How strong is the dependency relationship?
        0.3 * normalized_sample        # 30%: How much data supports this prediction?
    )
    
    # Ensure confidence is bounded between 0 and 1
    return min(1.0, max(0.0, confidence))


# ============================================================================
# Fill NA Values
# ============================================================================

def _fill_na_values(df, target_col, group_cols, mi_scores, confidence_threshold):
    """
    Fill NA values in target column using group-based conditional probabilities.
    
    Returns:
        tuple: (filled_df, fill_results)
        fill_results: list of dicts with fill information
    """
    filled_df = df.copy()
    fill_results = []
    
    # Calculate conditional probabilities
    conditional_probs = _calculate_conditional_probabilities(filled_df, target_col, group_cols)
    
    if not conditional_probs:
        return filled_df, fill_results
    
    # Find rows with NA in target column
    na_mask = filled_df[target_col].isna()
    na_indices = filled_df[na_mask].index
    
    # Get max MI for normalization
    max_mi = max(mi_scores.values()) if mi_scores else 1.0
    
    # Calculate average MI for group columns
    avg_mi = np.mean([mi_scores.get(col, 0) for col in group_cols]) if group_cols else 0
    
    target_is_numeric = _is_numeric(filled_df[target_col])
    
    for idx in na_indices:
        row = filled_df.loc[idx]
        
        # Skip if any group column has NA (we need all group columns to be non-NA to use them for prediction)
        if row[group_cols].isna().any():
            continue
        
        # Get group key for this row and normalize to match groupby keys
        # (We already checked that group columns don't have NA above)
        group_values = tuple()
        for col in group_cols:
            val = row[col]
            # Convert pandas scalars to Python native types to match groupby keys
            if hasattr(val, 'item'):
                try:
                    group_values += (val.item(),)
                except (ValueError, AttributeError):
                    group_values += (val,)
            else:
                group_values += (val,)
        
        # Check if we have conditional probability for this group
        if group_values in conditional_probs:
            prob_info = conditional_probs[group_values]
            sample_size = prob_info['count']
            
            if target_is_numeric:
                # Use median for numeric (more robust than mean)
                predicted_value = prob_info['median']
                # For numeric, use a proxy probability based on how close values are
                # (lower std = higher confidence)
                std = prob_info.get('std', 1.0)
                conditional_prob = 1.0 / (1.0 + std) if std > 0 else 1.0
            else:
                # Use most likely value for categorical
                predicted_value = prob_info['most_likely']
                conditional_prob = prob_info['probabilities'].get(predicted_value, 0.0)
            
            # Calculate confidence score
            confidence = _calculate_confidence_score(
                conditional_prob,
                avg_mi,
                sample_size,
                max_mi
            )
            
            # Only fill if confidence meets threshold
            if confidence >= confidence_threshold:
                filled_df.loc[idx, target_col] = predicted_value
                
                fill_results.append({
                    'index': idx,
                    'predicted_value': predicted_value,
                    'confidence': confidence,
                    'sample_size': sample_size,
                    'group_values': dict(zip(group_cols, group_values))
                })
    
    return filled_df, fill_results


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_and_fill_na(target_col, top_k=3, confidence_threshold=0.5):
    """
    Analyze dependencies and prepare to fill NA values for a single column.
    
    Returns:
        dict with analysis results and fill information
    """
    df = st.session_state.processed_df.copy()
    
    # Get available feature columns (excluding target and initial_id)
    feature_cols = [col for col in _get_available_columns() if col != target_col]
    
    if len(feature_cols) == 0:
        return None
    
    # Calculate mutual information
    with st.spinner(f"Calculating mutual information for '{target_col}'..."):
        mi_scores = _calculate_mutual_information(df, target_col, feature_cols)
    
    if not mi_scores:
        return None
    
    # Get top-k columns by mutual information
    sorted_cols = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_cols = [col for col, _ in sorted_cols[:top_k]]
    
    # Calculate conditional probabilities and prepare fill
    with st.spinner(f"Calculating conditional probabilities for '{target_col}'..."):
        conditional_probs = _calculate_conditional_probabilities(df, target_col, top_k_cols)
    
    # Count NA values that can be filled
    na_mask = df[target_col].isna()
    na_df = df[na_mask]
    
    fillable_count = 0
    max_mi = max(mi_scores.values()) if mi_scores else 1.0
    
    for idx in na_df.index:
        row = na_df.loc[idx]
        # Only check if group columns don't have NA (we need all group columns to be non-NA)
        if not row[top_k_cols].isna().any():
            # Normalize group values to match groupby keys
            group_values = tuple()
            for col in top_k_cols:
                val = row[col]
                if hasattr(val, 'item'):
                    try:
                        group_values += (val.item(),)
                    except (ValueError, AttributeError):
                        group_values += (val,)
                else:
                    group_values += (val,)
            if group_values in conditional_probs:
                fillable_count += 1
    
    return {
        'target_col': target_col,
        'mi_scores': mi_scores,
        'top_k_cols': top_k_cols,
        'conditional_probs': conditional_probs,
        'na_count': na_mask.sum(),
        'fillable_count': fillable_count,
        'max_mi': max_mi
    }


def analyze_all_columns(top_k=3, confidence_threshold=0.5):
    """
    Analyze dependencies and prepare to fill NA values for all columns with NA values.
    
    Returns:
        dict mapping column names to analysis results
    """
    df = st.session_state.processed_df.copy()
    available_columns = _get_available_columns()
    
    # Find columns with NA values
    columns_with_na = [col for col in available_columns if df[col].isna().sum() > 0]
    
    if not columns_with_na:
        return {}
    
    all_results = {}
    total_cols = len(columns_with_na)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, target_col in enumerate(columns_with_na):
        status_text.text(f"Analyzing column {idx + 1}/{total_cols}: {target_col}")
        progress_bar.progress((idx + 1) / total_cols)
        
        # Get available feature columns (excluding target and initial_id)
        feature_cols = [col for col in available_columns if col != target_col]
        
        if len(feature_cols) == 0:
            continue
        
        # Calculate mutual information
        mi_scores = _calculate_mutual_information(df, target_col, feature_cols)
        
        if not mi_scores:
            continue
        
        # Get top-k columns by mutual information
        sorted_cols = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_cols = [col for col, _ in sorted_cols[:top_k]]
        
        # Calculate conditional probabilities
        conditional_probs = _calculate_conditional_probabilities(df, target_col, top_k_cols)
        
        # Count NA values that can be filled
        na_mask = df[target_col].isna()
        na_df = df[na_mask]
        
        fillable_count = 0
        max_mi = max(mi_scores.values()) if mi_scores else 1.0
        
        for row_idx in na_df.index:
            row = na_df.loc[row_idx]
            # Only check if group columns don't have NA (we need all group columns to be non-NA)
            if not row[top_k_cols].isna().any():
                # Normalize group values to match groupby keys
                group_values = tuple()
                for col in top_k_cols:
                    val = row[col]
                    if hasattr(val, 'item'):
                        try:
                            group_values += (val.item(),)
                        except (ValueError, AttributeError):
                            group_values += (val,)
                    else:
                        group_values += (val,)
                if group_values in conditional_probs:
                    fillable_count += 1
        
        all_results[target_col] = {
            'target_col': target_col,
            'mi_scores': mi_scores,
            'top_k_cols': top_k_cols,
            'conditional_probs': conditional_probs,
            'na_count': na_mask.sum(),
            'fillable_count': fillable_count,
            'max_mi': max_mi
        }
    
    progress_bar.empty()
    status_text.empty()
    
    return all_results


def apply_fill_na(analysis_result, confidence_threshold):
    """Apply the fill NA operation to processed_df."""
    df = st.session_state.processed_df.copy()
    target_col = analysis_result['target_col']
    group_cols = analysis_result['top_k_cols']
    mi_scores = analysis_result['mi_scores']
    
    filled_df, fill_results = _fill_na_values(
        df, target_col, group_cols, mi_scores, confidence_threshold
    )
    
    # Update processed_df
    st.session_state.processed_df = filled_df
    
    # Add message
    filled_count = len(fill_results)
    _add_message(
        f"🔧 **Filled {filled_count} NA value(s) in column {target_col}** "
        f"(using dependencies: {', '.join(group_cols)})"
    )
    
    return fill_results


def apply_fill_na_all(all_results, confidence_threshold):
    """Apply the fill NA operation to processed_df for all columns."""
    df = st.session_state.processed_df.copy()
    total_filled = 0
    columns_filled = []
    
    for target_col, analysis_result in all_results.items():
        group_cols = analysis_result['top_k_cols']
        mi_scores = analysis_result['mi_scores']
        
        filled_df, fill_results = _fill_na_values(
            df, target_col, group_cols, mi_scores, confidence_threshold
        )
        
        df = filled_df
        filled_count = len(fill_results)
        if filled_count > 0:
            total_filled += filled_count
            columns_filled.append((target_col, filled_count, ', '.join(group_cols)))
    
    # Update processed_df
    st.session_state.processed_df = df
    
    # Add messages
    if columns_filled:
        for col, count, deps in columns_filled:
            _add_message(
                f"🔧 **Filled {count} NA value(s) in column {col}** "
                f"(using dependencies: {deps})"
            )
    
    return total_filled


# ============================================================================
# Display Functions
# ============================================================================

def display_analysis_results(analysis_result, confidence_threshold):
    """Display analysis results and preview fill operations."""
    target_col = analysis_result['target_col']
    mi_scores = analysis_result['mi_scores']
    top_k_cols = analysis_result['top_k_cols']
    na_count = analysis_result['na_count']
    fillable_count = analysis_result['fillable_count']
    
    st.subheader("Dependency Analysis Results")
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NA Values", na_count)
    with col2:
        st.metric("Fillable Values", fillable_count)
    with col3:
        st.metric("Dependencies Found", len(top_k_cols))
    
    # Mutual Information Scores
    st.write("**Mutual Information Scores (top dependencies):**")
    mi_df = pd.DataFrame([
        {'Column': col, 'Mutual Information': f"{score:.4f}"}
        for col, score in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    ])
    st.dataframe(mi_df, use_container_width=True, hide_index=True)
    
    # Top K columns used for grouping
    st.write(f"**Top {len(top_k_cols)} columns used for grouping:** {', '.join(top_k_cols)}")
    
    # Fill results
    st.subheader("Fill Results")
    
    # Check if we have cached fill results for this confidence threshold
    cache_key = f"fill_results_{target_col}_{confidence_threshold}"
    if cache_key in st.session_state:
        fill_results = st.session_state[cache_key]
    else:
        # Calculate fill results
        df = st.session_state.processed_df.copy()
        mi_scores_dict = analysis_result['mi_scores']
        
        filled_df, fill_results = _fill_na_values(
            df, target_col, top_k_cols, mi_scores_dict, confidence_threshold
        )
        # Cache the results
        st.session_state[cache_key] = fill_results
    
    if fill_results:
        # Limit the number of results displayed to avoid browser overload
        display_results = fill_results[:MAX_DISPLAY_ROWS]
        total_results = len(fill_results)
        
        # Create dataframe with limited results
        # Format group values as a more readable string
        display_data = []
        for result in display_results:
            # Format group values as "col1=val1, col2=val2, ..." with line breaks for readability
            group_str_parts = []
            for k, v in result['group_values'].items():
                # Truncate long values for better display
                val_str = str(v)
                if len(val_str) > 30:
                    val_str = val_str[:27] + "..."
                group_str_parts.append(f"{k}={val_str}")
            group_str = " | ".join(group_str_parts)  # Use | separator for better readability
            
            display_data.append({
                'Row Index': result['index'],
                'Predicted Value': result['predicted_value'],
                'Confidence': f"{result['confidence']:.2%}",
                'Sample Size': result['sample_size'],
                'Group Values': group_str
            })
        
        results_df = pd.DataFrame(display_data)
        
        # Display results with better formatting
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        if total_results > MAX_DISPLAY_ROWS:
            st.caption(f"Showing first {MAX_DISPLAY_ROWS} of {total_results} fillable values")
        else:
            st.caption(f"Showing all {total_results} fillable values")
        
        # Statistics (calculate on all results, not just displayed ones)
        confidences = [r['confidence'] for r in fill_results]
        if confidences:
            st.write(f"**Confidence Statistics:**")
            st.write(f"- Mean: {np.mean(confidences):.2%}")
            st.write(f"- Median: {np.median(confidences):.2%}")
            st.write(f"- Min: {np.min(confidences):.2%}")
            st.write(f"- Max: {np.max(confidences):.2%}")
        
        return fill_results
    else:
        st.info("No values meet the confidence threshold. Try lowering the threshold.")
        return []


def display_all_columns_results(all_results, confidence_threshold):
    """Display analysis results for all columns."""
    if not all_results:
        st.info("No columns with NA values found or no dependencies could be identified.")
        return {}
    
    st.subheader("All Columns Analysis Summary")
    
    # Create summary dataframe
    summary_data = []
    all_fill_results = {}
    
    for target_col, analysis_result in all_results.items():
        # Extract top_k_cols from analysis_result (needed for summary)
        top_k_cols = analysis_result['top_k_cols']
        
        # Check if we have cached fill results for this confidence threshold
        cache_key = f"fill_results_{target_col}_{confidence_threshold}"
        if cache_key in st.session_state:
            fill_results = st.session_state[cache_key]
        else:
            # Calculate actual fill results for this column
            df = st.session_state.processed_df.copy()
            mi_scores_dict = analysis_result['mi_scores']
            
            filled_df, fill_results = _fill_na_values(
                df, target_col, top_k_cols, mi_scores_dict, confidence_threshold
            )
            # Cache the results
            st.session_state[cache_key] = fill_results
        
        all_fill_results[target_col] = fill_results
        
        summary_data.append({
            'Column': target_col,
            'NA Count': analysis_result['na_count'],
            'Fillable Count': len(fill_results),
            'Dependencies': ', '.join(top_k_cols[:3]) + ('...' if len(top_k_cols) > 3 else ''),
            'Top MI Score': f"{max(analysis_result['mi_scores'].values()):.4f}" if analysis_result['mi_scores'] else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Show detailed results for each column
    st.subheader("Detailed Results by Column")
    
    for target_col, analysis_result in all_results.items():
        fill_results = all_fill_results[target_col]
        
        if fill_results:
            # Limit displayed results to avoid browser overload
            display_results = fill_results[:MAX_DISPLAY_ROWS]
            total_results = len(fill_results)
            
            with st.expander(f"📊 {target_col} - {total_results} fillable value(s)"):
                # Create detailed dataframe with limited results
                display_data = []
                for result in display_results:
                    group_str_parts = []
                    for k, v in result['group_values'].items():
                        val_str = str(v)
                        if len(val_str) > 30:
                            val_str = val_str[:27] + "..."
                        group_str_parts.append(f"{k}={val_str}")
                    group_str = " | ".join(group_str_parts)
                    
                    display_data.append({
                        'Row Index': result['index'],
                        'Predicted Value': result['predicted_value'],
                        'Confidence': f"{result['confidence']:.2%}",
                        'Sample Size': result['sample_size'],
                        'Group Values': group_str
                    })
                
                results_df = pd.DataFrame(display_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                if total_results > MAX_DISPLAY_ROWS:
                    st.caption(f"Showing first {MAX_DISPLAY_ROWS} of {total_results} results")
                
                # Statistics (calculate on all results)
                confidences = [r['confidence'] for r in fill_results]
                if confidences:
                    st.write(f"**Confidence Statistics:**")
                    st.write(f"- Mean: {np.mean(confidences):.2%}")
                    st.write(f"- Median: {np.median(confidences):.2%}")
                    st.write(f"- Min: {np.min(confidences):.2%}")
                    st.write(f"- Max: {np.max(confidences):.2%}")
    
    return all_fill_results


# ============================================================================
# Main Render Function
# ============================================================================

def render():
    """Render the Fill NA tab."""
    st.header("Fill NA")
    st.write("Fill missing values by finding dependencies via mutual information and computing group-based conditional probabilities.")
    
    # Check if data is available
    if not _has_processed_data():
        st.info("No processed data available. Please import and process data in the 'Import Data' tab.")
        return
    
    available_columns = _get_available_columns()
    
    if len(available_columns) == 0:
        st.warning("No columns available for analysis.")
        return
    
    # Mode selection
    mode = st.radio(
        "Analysis Mode",
        options=["Single Column", "All Columns"],
        help="Select whether to analyze a single column or all columns with NA values",
        key="fill_na_mode",
        horizontal=True
    )
    
    # Options (shared for both modes)
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider(
            "Number of dependencies (top-k)",
            min_value=1,
            max_value=min(10, len(available_columns) - 1),
            value=3,
            step=1,
            help="Number of top columns (by mutual information) to use for grouping",
            key="fill_na_top_k"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score required to fill a value",
            key="fill_na_confidence_threshold"
        )
    
    if mode == "Single Column":
        # Single column mode
        selected_column = st.selectbox(
            "Select column to fill NA values",
            options=available_columns,
            help="Select the column that contains NA values you want to fill",
            key="fill_na_column"
        )
        
        if selected_column is None:
            return
        
        # Check if column has NA values
        df = st.session_state.processed_df
        na_count = df[selected_column].isna().sum()
        
        if na_count == 0:
            st.success(f"✅ Column '{selected_column}' has no NA values.")
            return
        
        st.info(f"Column '{selected_column}' has {na_count} NA value(s).")
        
        # Analyze button
        if st.button("Analyze Dependencies", type="primary", key="fill_na_analyze"):
            analysis_result = analyze_and_fill_na(selected_column, top_k, confidence_threshold)
            if analysis_result:
                st.session_state.fill_na_analysis = analysis_result
            else:
                st.error("Could not analyze dependencies. Make sure there are other columns with valid data.")
        
        # Show analysis results if available (only show if we're in single column mode)
        if 'fill_na_analysis' in st.session_state and mode == "Single Column":
            analysis_result = st.session_state.fill_na_analysis
            
            # Update confidence threshold if changed
            current_threshold = st.session_state.get('fill_na_confidence_threshold', 0.5)
            
            # Clear cache if threshold changed
            cache_key = f"fill_results_{analysis_result['target_col']}_{current_threshold}"
            if cache_key not in st.session_state:
                # Clear old cache entries for this column
                keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"fill_results_{analysis_result['target_col']}_")]
                for k in keys_to_remove:
                    del st.session_state[k]
            
            # Display results
            fill_results = display_analysis_results(analysis_result, current_threshold)
            
            # Apply button
            if fill_results:
                st.divider()
                if st.button("Apply Fill NA", type="primary", key="fill_na_apply"):
                    apply_fill_na(analysis_result, current_threshold)
                    st.success(f"✅ Filled {len(fill_results)} NA value(s) in column '{selected_column}'")
                    # Clear analysis and cache to force re-analysis if needed
                    if 'fill_na_analysis' in st.session_state:
                        del st.session_state.fill_na_analysis
                    # Clear fill results cache for this column
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"fill_results_{selected_column}_")]
                    for k in keys_to_remove:
                        del st.session_state[k]
                    st.rerun()
    
    else:
        # All columns mode
        df = st.session_state.processed_df
        columns_with_na = [col for col in available_columns if df[col].isna().sum() > 0]
        
        if not columns_with_na:
            st.success("✅ No columns with NA values found.")
            return
        
        st.info(f"Found {len(columns_with_na)} column(s) with NA values: {', '.join(columns_with_na)}")
        
        # Analyze button
        if st.button("Analyze All Columns", type="primary", key="fill_na_analyze_all"):
            all_results = analyze_all_columns(top_k, confidence_threshold)
            if all_results:
                st.session_state.fill_na_all_analysis = all_results
            else:
                st.error("Could not analyze dependencies for any columns.")
        
        # Show analysis results if available (only show if we're in all columns mode)
        if 'fill_na_all_analysis' in st.session_state and mode == "All Columns":
            all_results = st.session_state.fill_na_all_analysis
            
            # Update confidence threshold if changed
            current_threshold = st.session_state.get('fill_na_confidence_threshold', 0.5)
            
            # Clear cache if threshold changed (check if any column needs cache refresh)
            for target_col in all_results.keys():
                cache_key = f"fill_results_{target_col}_{current_threshold}"
                if cache_key not in st.session_state:
                    # Clear old cache entries for this column
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"fill_results_{target_col}_")]
                    for k in keys_to_remove:
                        del st.session_state[k]
            
            # Display results
            all_fill_results = display_all_columns_results(all_results, current_threshold)
            
            # Apply buttons
            total_fillable = sum(len(results) for results in all_fill_results.values())
            if total_fillable > 0:
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Apply Fill NA for All Columns", type="primary", key="fill_na_apply_all"):
                        total_filled = apply_fill_na_all(all_results, current_threshold)
                        st.success(f"✅ Filled {total_filled} NA value(s) across all columns")
                        # Clear analysis and cache to force re-analysis if needed
                        if 'fill_na_all_analysis' in st.session_state:
                            del st.session_state.fill_na_all_analysis
                        # Clear all fill results cache
                        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("fill_results_")]
                        for k in keys_to_remove:
                            del st.session_state[k]
                        st.rerun()
                
                with col2:
                    st.write("Or apply fills for individual columns using the buttons below:")
                
                # Individual column apply buttons
                for target_col, analysis_result in all_results.items():
                    fill_results = all_fill_results.get(target_col, [])
                    if fill_results:
                        if st.button(f"Apply Fill NA for {target_col}", key=f"fill_na_apply_{target_col}"):
                            apply_fill_na(analysis_result, current_threshold)
                            st.success(f"✅ Filled {len(fill_results)} NA value(s) in column '{target_col}'")
                            # Clear cache for this column since data has changed
                            keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"fill_results_{target_col}_")]
                            for k in keys_to_remove:
                                del st.session_state[k]
                            # Update the stored results
                            if 'fill_na_all_analysis' in st.session_state:
                                # Recalculate for this column
                                df = st.session_state.processed_df.copy()
                                mi_scores_dict = analysis_result['mi_scores']
                                top_k_cols = analysis_result['top_k_cols']
                                _, updated_fill_results = _fill_na_values(
                                    df, target_col, top_k_cols, mi_scores_dict, current_threshold
                                )
                                all_results[target_col]['fillable_count'] = len(updated_fill_results)
                            st.rerun()

