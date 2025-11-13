import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict, Counter


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
    """Check if a series is categorical."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        non_null_count = series.notna().sum()
        if non_null_count > 0:
            unique_count = series.nunique()
            return unique_count < (non_null_count * 0.5) or unique_count < 100
    return False


# ============================================================================
# Data Preprocessing
# ============================================================================

def _discretize_numeric_column(series, n_bins=5, method='quantile'):
    """
    Discretize a numeric column into bins.
    
    Args:
        series: Numeric pandas Series
        n_bins: Number of bins
        method: 'quantile' or 'uniform'
    
    Returns:
        Series with binned values as strings
    """
    # Create result series
    result_series = pd.Series(index=series.index, dtype=object)
    
    # Handle NaN values - they will remain NaN
    non_null_mask = series.notna()
    non_null_series = series[non_null_mask]
    
    if len(non_null_series) == 0:
        return result_series
    
    # Create bins
    try:
        if method == 'quantile':
            binned_result = pd.qcut(non_null_series, q=n_bins, duplicates='drop', retbins=True)
        else:  # uniform
            binned_result = pd.cut(non_null_series, bins=n_bins, duplicates='drop', retbins=True)
        
        binned = binned_result[0]
        bin_edges = binned_result[1]
        
        # Create labels from bin intervals
        # binned is a Categorical, we can use its categories
        for cat in binned.cat.categories:
            # Extract numeric values from interval string
            # Format is like "(0.123, 0.456]" or "[0.123, 0.456]"
            interval_str = str(cat)
            result_series.loc[non_null_mask & (binned == cat)] = interval_str
        
    except (ValueError, TypeError) as e:
        # If binning fails, use simple string conversion
        result_series.loc[non_null_mask] = non_null_series.astype(str)
    
    return result_series


def _prepare_transactions(df, selected_columns, treat_na_as_category=True, n_bins=5):
    """
    Convert dataframe to transactions format.
    Each row becomes a transaction with column-value pairs as items.
    
    Args:
        df: Input dataframe
        selected_columns: List of columns to include
        treat_na_as_category: If True, treat NA as separate category; if False, skip NA values
        n_bins: Number of bins for numeric columns
    
    Returns:
        List of transactions, where each transaction is a set of items (column=value pairs)
    """
    transactions = []
    work_df = df[selected_columns].copy()
    
    # Process each column
    processed_df = pd.DataFrame(index=work_df.index)
    
    for col in selected_columns:
        series = work_df[col]
        
        if _is_numeric(series):
            # Discretize numeric columns
            binned_series = _discretize_numeric_column(series, n_bins=n_bins)
            processed_df[col] = binned_series
        else:
            # Keep categorical as-is
            processed_df[col] = series
    
    # Convert to transactions
    for idx, row in processed_df.iterrows():
        transaction = set()
        
        for col in selected_columns:
            value = row[col]
            
            if pd.isna(value):
                if treat_na_as_category:
                    transaction.add(f"{col}=NA")
            else:
                transaction.add(f"{col}={value}")
        
        if transaction:  # Only add non-empty transactions
            transactions.append(transaction)
    
    return transactions


# ============================================================================
# Apriori Algorithm
# ============================================================================

def _get_frequent_itemsets_1(transactions, min_support):
    """
    Get frequent 1-itemsets.
    
    Args:
        transactions: List of transactions (sets of items)
        min_support: Minimum support threshold (0 to 1)
    
    Returns:
        Dictionary mapping itemset (frozenset) to support count
    """
    item_counts = Counter()
    total_transactions = len(transactions)
    
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    
    # Filter by minimum support
    min_support_count = min_support * total_transactions
    frequent_1 = {}
    
    for item, count in item_counts.items():
        support = count / total_transactions
        if support >= min_support:
            frequent_1[frozenset([item])] = {
                'support': support,
                'count': count
            }
    
    return frequent_1


def _generate_candidates(frequent_k_minus_1, k):
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets.
    
    Args:
        frequent_k_minus_1: Dictionary of frequent (k-1)-itemsets
        k: Size of itemsets to generate
    
    Returns:
        Set of candidate k-itemsets (frozensets)
    """
    candidates = set()
    itemsets = list(frequent_k_minus_1.keys())
    
    # Generate candidates by joining itemsets that share (k-2) items
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            itemset1 = itemsets[i]
            itemset2 = itemsets[j]
            
            # Check if they share (k-2) items
            union = itemset1 | itemset2
            if len(union) == k:
                candidates.add(union)
    
    # Prune: remove candidates that have infrequent subsets
    pruned_candidates = set()
    for candidate in candidates:
        # Check all (k-1)-subsets
        all_subsets_frequent = True
        for subset in combinations(candidate, k - 1):
            subset_frozenset = frozenset(subset)
            if subset_frozenset not in frequent_k_minus_1:
                all_subsets_frequent = False
                break
        
        if all_subsets_frequent:
            pruned_candidates.add(candidate)
    
    return pruned_candidates


def _count_support(transactions, candidates):
    """
    Count support for candidate itemsets.
    
    Args:
        transactions: List of transactions
        candidates: Set of candidate itemsets (frozensets)
    
    Returns:
        Dictionary mapping itemset to support count
    """
    support_counts = defaultdict(int)
    total_transactions = len(transactions)
    
    for transaction in transactions:
        transaction_set = set(transaction)
        for candidate in candidates:
            if candidate.issubset(transaction_set):
                support_counts[candidate] += 1
    
    # Convert counts to support
    support_dict = {}
    for itemset, count in support_counts.items():
        support_dict[itemset] = {
            'support': count / total_transactions,
            'count': count
        }
    
    return support_dict


def apriori_algorithm(transactions, min_support, max_itemset_size=None):
    """
    Run Apriori algorithm to find frequent itemsets.
    
    Args:
        transactions: List of transactions (sets of items)
        min_support: Minimum support threshold (0 to 1)
        max_itemset_size: Maximum size of itemsets to find (None = no limit)
    
    Returns:
        Dictionary mapping itemset size to dictionary of frequent itemsets
    """
    all_frequent = {}
    
    # Get frequent 1-itemsets
    frequent_1 = _get_frequent_itemsets_1(transactions, min_support)
    if not frequent_1:
        return all_frequent
    
    all_frequent[1] = frequent_1
    k = 2
    
    # Iteratively find larger frequent itemsets
    while True:
        if max_itemset_size and k > max_itemset_size:
            break
        
        # Generate candidates
        candidates = _generate_candidates(all_frequent[k - 1], k)
        
        if not candidates:
            break
        
        # Count support
        support_dict = _count_support(transactions, candidates)
        
        # Filter by minimum support
        frequent_k = {
            itemset: support_info
            for itemset, support_info in support_dict.items()
            if support_info['support'] >= min_support
        }
        
        if not frequent_k:
            break
        
        all_frequent[k] = frequent_k
        k += 1
    
    return all_frequent


# ============================================================================
# Association Rules Generation
# ============================================================================

def _generate_association_rules(frequent_itemsets, min_confidence):
    """
    Generate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets: Dictionary mapping itemset size to frequent itemsets
        min_confidence: Minimum confidence threshold (0 to 1)
    
    Returns:
        List of rules, each as a dict with 'antecedent', 'consequent', 'support', 'confidence', 'lift'
    """
    rules = []
    
    # Process itemsets of size 2 or more
    for k in range(2, len(frequent_itemsets) + 1):
        if k not in frequent_itemsets:
            continue
        
        for itemset, itemset_info in frequent_itemsets[k].items():
            itemset_list = list(itemset)
            
            # Generate all possible rules from this itemset
            # For each non-empty proper subset as antecedent
            for r in range(1, k):
                for antecedent in combinations(itemset_list, r):
                    antecedent_set = frozenset(antecedent)
                    consequent_set = itemset - antecedent_set
                    
                    # Find support of antecedent
                    antecedent_support_info = None
                    for size in frequent_itemsets:
                        if antecedent_set in frequent_itemsets[size]:
                            antecedent_support_info = frequent_itemsets[size][antecedent_set]
                            break
                    
                    if antecedent_support_info is None:
                        continue
                    
                    # Calculate confidence
                    itemset_support = itemset_info['support']
                    antecedent_support = antecedent_support_info['support']
                    
                    if antecedent_support > 0:
                        confidence = itemset_support / antecedent_support
                        
                        if confidence >= min_confidence:
                            # Calculate lift
                            # Find support of consequent
                            consequent_support_info = None
                            for size in frequent_itemsets:
                                if consequent_set in frequent_itemsets[size]:
                                    consequent_support_info = frequent_itemsets[size][consequent_set]
                                    break
                            
                            consequent_support = consequent_support_info['support'] if consequent_support_info else 0
                            
                            if consequent_support > 0:
                                lift = confidence / consequent_support
                            else:
                                lift = float('inf')
                            
                            rules.append({
                                'antecedent': antecedent_set,
                                'consequent': consequent_set,
                                'support': itemset_support,
                                'confidence': confidence,
                                'lift': lift
                            })
    
    return rules


# ============================================================================
# Analysis Function
# ============================================================================

def run_apriori_analysis(selected_columns, min_support, min_confidence, 
                         max_itemset_size, treat_na_as_category, n_bins=5):
    """
    Run complete Apriori analysis.
    
    Args:
        selected_columns: List of columns to analyze
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        max_itemset_size: Maximum itemset size
        treat_na_as_category: Whether to treat NA as separate category
        n_bins: Number of bins for numeric columns
    
    Returns:
        Dictionary with analysis results
    """
    df = st.session_state.processed_df.copy()
    
    # Prepare transactions
    with st.spinner("Preparing transactions..."):
        transactions = _prepare_transactions(
            df, selected_columns, treat_na_as_category, n_bins
        )
    
    if len(transactions) == 0:
        return None
    
    # Run Apriori algorithm
    with st.spinner("Finding frequent itemsets..."):
        frequent_itemsets = apriori_algorithm(
            transactions, min_support, max_itemset_size
        )
    
    # Generate association rules
    rules = []
    if min_confidence > 0:
        with st.spinner("Generating association rules..."):
            rules = _generate_association_rules(frequent_itemsets, min_confidence)
    
    return {
        'transactions': transactions,
        'frequent_itemsets': frequent_itemsets,
        'rules': rules,
        'total_transactions': len(transactions)
    }


# ============================================================================
# Display Functions
# ============================================================================

def _format_itemset(itemset):
    """Format itemset for display."""
    items = sorted(list(itemset))
    return "{" + ", ".join(items) + "}"


def display_frequent_itemsets(frequent_itemsets, total_transactions):
    """Display frequent itemsets."""
    if not frequent_itemsets:
        st.info("No frequent itemsets found.")
        return
    
    st.subheader("Frequent Itemsets")
    
    # Summary
    total_itemsets = sum(len(itemsets) for itemsets in frequent_itemsets.values())
    max_size = max(frequent_itemsets.keys())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Frequent Itemsets", total_itemsets)
    with col2:
        st.metric("Maximum Itemset Size", max_size)
    with col3:
        st.metric("Total Transactions", total_transactions)
    
    # Display by size
    for size in sorted(frequent_itemsets.keys()):
        itemsets = frequent_itemsets[size]
        
        with st.expander(f"Size {size} Itemsets ({len(itemsets)} itemsets)"):
            # Prepare data for display
            display_data = []
            for itemset, info in sorted(itemsets.items(), 
                                       key=lambda x: x[1]['support'], 
                                       reverse=True):
                display_data.append({
                    'Itemset': _format_itemset(itemset),
                    'Support': f"{info['support']:.4f}",
                    'Support %': f"{info['support'] * 100:.2f}%",
                    'Count': info['count']
                })
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True, hide_index=True)


def display_association_rules(rules):
    """Display association rules."""
    if not rules:
        st.info("No association rules found.")
        return
    
    st.subheader("Association Rules")
    
    # Summary
    st.metric("Total Rules", len(rules))
    
    # Sort by confidence (descending)
    sorted_rules = sorted(rules, key=lambda x: x['confidence'], reverse=True)
    
    # Prepare data for display
    display_data = []
    for rule in sorted_rules:
        antecedent_str = _format_itemset(rule['antecedent'])
        consequent_str = _format_itemset(rule['consequent'])
        
        lift_str = f"{rule['lift']:.4f}" if rule['lift'] != float('inf') else "∞"
        
        display_data.append({
            'Antecedent': antecedent_str,
            'Consequent': consequent_str,
            'Support': f"{rule['support']:.4f}",
            'Support %': f"{rule['support'] * 100:.2f}%",
            'Confidence': f"{rule['confidence']:.4f}",
            'Confidence %': f"{rule['confidence'] * 100:.2f}%",
            'Lift': lift_str
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Statistics
    if rules:
        confidences = [r['confidence'] for r in rules]
        lifts = [r['lift'] for r in rules if r['lift'] != float('inf')]
        
        st.write("**Rule Statistics:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confidence:**")
            st.write(f"- Mean: {np.mean(confidences):.4f}")
            st.write(f"- Median: {np.median(confidences):.4f}")
            st.write(f"- Min: {np.min(confidences):.4f}")
            st.write(f"- Max: {np.max(confidences):.4f}")
        
        with col2:
            if lifts:
                st.write("**Lift:**")
                st.write(f"- Mean: {np.mean(lifts):.4f}")
                st.write(f"- Median: {np.median(lifts):.4f}")
                st.write(f"- Min: {np.min(lifts):.4f}")
                st.write(f"- Max: {np.max(lifts):.4f}")


def display_analysis_results(results):
    """Display complete analysis results."""
    if results is None:
        st.error("Analysis failed. Please check your data and column selection.")
        return
    
    # Display frequent itemsets
    display_frequent_itemsets(
        results['frequent_itemsets'],
        results['total_transactions']
    )
    
    # Display association rules
    if results['rules']:
        st.divider()
        display_association_rules(results['rules'])


# ============================================================================
# Main Render Function
# ============================================================================

def render():
    """Render the Apriori tab."""
    st.header("Apriori Algorithm")
    st.write("Find frequent itemsets and generate association rules using the Apriori algorithm. "
             "Each row is treated as a transaction with column-value pairs as items.")
    
    # Check if data is available
    if not _has_processed_data():
        st.info("No processed data available. Please import and process data in the 'Import Data' tab.")
        return
    
    available_columns = _get_available_columns()
    
    if len(available_columns) == 0:
        st.warning("No columns available for analysis.")
        return
    
    # Column selection
    st.subheader("Column Selection")
    selected_columns = st.multiselect(
        "Select columns to analyze",
        options=available_columns,
        default=available_columns,
        help="Select columns to include in the Apriori analysis. "
             "Numeric columns will be discretized into bins."
    )
    
    if not selected_columns:
        st.warning("Please select at least one column.")
        return
    
    # Parameters
    st.subheader("Algorithm Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider(
            "Minimum Support",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Minimum support threshold (proportion of transactions containing itemset)"
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence threshold for association rules"
        )
    
    with col3:
        max_itemset_size = st.number_input(
            "Maximum Itemset Size",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Maximum size of itemsets to find"
        )
    
    # Data preprocessing options
    st.subheader("Data Preprocessing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        treat_na_as_category = st.checkbox(
            "Treat NA as separate category",
            value=True,
            help="If enabled, NA values will be treated as a separate category. "
                 "If disabled, rows with NA values will be skipped for those columns."
        )
    
    with col2:
        n_bins = st.number_input(
            "Number of bins for numeric columns",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Number of bins to use when discretizing numeric columns"
        )
    
    # Run analysis button
    if st.button("Run Apriori Analysis", type="primary"):
        results = run_apriori_analysis(
            selected_columns,
            min_support,
            min_confidence,
            max_itemset_size,
            treat_na_as_category,
            n_bins
        )
        
        if results:
            st.session_state.apriori_results = results
        else:
            st.error("Analysis failed. Please check your data and column selection.")
    
    # Display results if available
    if 'apriori_results' in st.session_state:
        st.divider()
        display_analysis_results(st.session_state.apriori_results)

