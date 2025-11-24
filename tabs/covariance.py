import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
import prince
from io import BytesIO


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


def _classify_columns(df, columns):
    """Classify columns as numeric or categorical."""
    numeric_cols = []
    categorical_cols = []
    
    for col in columns:
        series = df[col]
        if _is_numeric(series):
            numeric_cols.append(col)
        elif _is_categorical(series):
            categorical_cols.append(col)
        # Skip columns that are neither clearly numeric nor categorical
    
    return numeric_cols, categorical_cols


# ============================================================================
# Data Preprocessing
# ============================================================================

def _prepare_data(df, selected_columns, drop_na=False):
    """
    Prepare data for covariance analysis.
    
    Args:
        df: Input dataframe
        selected_columns: List of columns to include
        drop_na: Whether to drop rows with NA values
    
    Returns:
        tuple: (prepared_df, numeric_cols, categorical_cols, na_info)
    """
    # Select only the columns we need
    work_df = df[selected_columns].copy()
    
    # Check for NA values
    na_info = {
        'has_na': work_df.isna().any().any(),
        'na_counts': work_df.isna().sum().to_dict(),
        'total_rows_before': len(work_df)
    }
    
    # Drop NA if requested
    if drop_na:
        work_df = work_df.dropna()
        na_info['total_rows_after'] = len(work_df)
        na_info['rows_dropped'] = na_info['total_rows_before'] - na_info['total_rows_after']
    else:
        na_info['total_rows_after'] = len(work_df)
        na_info['rows_dropped'] = 0
    
    # Classify columns
    numeric_cols, categorical_cols = _classify_columns(work_df, selected_columns)
    
    return work_df, numeric_cols, categorical_cols, na_info


# ============================================================================
# MCA and Standardization
# ============================================================================

def _apply_mca(df, categorical_cols, n_components=None):
    """
    Apply Multiple Correspondence Analysis (MCA) for scaling categorical columns.
    
    The approach:
    1. Apply MCA to understand relationships between all categorical variables
    2. Use MCA row coordinates to create individual representations for each column
    3. For each column, use the row coordinates weighted by how much that column's 
       categories contribute to each MCA component
    4. This preserves individual column identities while using MCA insights
    
    Args:
        df: Dataframe with categorical columns
        categorical_cols: List of categorical column names
        n_components: Number of components to keep (None = auto)
    
    Returns:
        tuple: (mca_coords_df, mca_model, column_mapping)
        mca_coords_df: DataFrame with individual column representations, columns named after original categorical columns
        mca_model: The fitted MCA model
        column_mapping: Dict mapping original column names to their representations
    """
    if not categorical_cols:
        return None, None, None
    
    # Prepare data for MCA (convert to string and handle NaN)
    mca_data = df[categorical_cols].copy()
    mca_data = mca_data.fillna('_MISSING_')  # MCA can handle this
    
    num_rows = len(mca_data)
    
    # For large datasets (>10k rows), use sampling for MCA to reduce memory usage
    # MCA creates a large indicator matrix internally, so sampling helps significantly
    MCA_SAMPLE_SIZE = 10000
    use_sampling = num_rows > MCA_SAMPLE_SIZE
    
    if use_sampling:
        # Sample data for MCA computation
        sample_size = min(MCA_SAMPLE_SIZE, num_rows)
        mca_sample = mca_data.sample(n=sample_size, random_state=42)
    else:
        mca_sample = mca_data
    
    # Determine number of components (use min of n_components or available)
    if n_components is None:
        # Use a reasonable default: min of 10 or number of categories - 1
        max_components = sum([mca_sample[col].nunique() for col in categorical_cols]) - len(categorical_cols)
        n_components = min(10, max(1, max_components))
    
    # Apply MCA to understand relationships between categories (on sample)
    mca = prince.MCA(n_components=n_components, random_state=42)
    mca.fit(mca_sample)  # Fit on sample
    
    # Get column coordinates - these show where each category level is positioned
    try:
        column_coords = mca.column_coordinates(mca_sample)
    except:
        column_coords = None
    
    num_cat_cols = len(categorical_cols)
    num_rows = len(mca_data)
    
    # Create individual numeric representation for each categorical column
    # Use frequency encoding as base (preserves individual column identity)
    # Then refine using MCA column coordinates if available
    column_representations = np.zeros((num_rows, num_cat_cols))
    
    for col_idx, col in enumerate(categorical_cols):
        # Use frequency encoding: encode each category by its frequency in the dataset
        # This preserves the column's individual distribution
        value_counts = mca_data[col].value_counts()
        total = len(mca_data[col])
        frequencies = value_counts / total
        
        # Vectorized mapping: use pandas map() instead of row-by-row iteration
        column_representations[:, col_idx] = mca_data[col].map(frequencies).fillna(0.0).values
    
    # Now refine using MCA column coordinates if available
    # This adds MCA insights while preserving individual column identities
    if column_coords is not None:
        for col_idx, col in enumerate(categorical_cols):
            # Get unique categories for this column
            unique_categories = mca_data[col].unique()
            
            # For each category, get its MCA coordinate
            # Use the first principal component as a scaling factor
            category_scales = {}
            for cat in unique_categories:
                col_coord_key = f"{col}_{cat}"
                if col_coord_key in column_coords.index:
                    coord_values = column_coords.loc[col_coord_key].values
                    if len(coord_values) > 0:
                        # Use first component as a scale factor
                        category_scales[cat] = coord_values[0]
            
            # If we have scales, apply them to refine the frequency encoding
            if category_scales:
                # Normalize scales to have mean 0 and std 1 (relative to this column)
                scale_values = np.array([category_scales.get(str(cat), 0.0) for cat in unique_categories])
                if scale_values.std() > 1e-10:
                    scale_values = (scale_values - scale_values.mean()) / scale_values.std()
                    scale_dict = {str(cat): scale_values[i] for i, cat in enumerate(unique_categories)}
                    
                    # Vectorized combination: use pandas map() instead of row-by-row iteration
                    freq_vals = column_representations[:, col_idx]
                    mca_scales = mca_data[col].map(scale_dict).fillna(0.0).values
                    # Weighted combination: 70% frequency, 30% MCA scale
                    column_representations[:, col_idx] = 0.7 * freq_vals + 0.3 * (mca_scales + 1) / 2
    
    # Create dataframe with original categorical column names
    mca_coords_df = pd.DataFrame(
        column_representations,
        index=mca_data.index,
        columns=categorical_cols  # Use original column names
    )
    
    # Create column mapping info (for reference)
    column_representations_dict = {col: f"MCA_frequency_scaled_{col}" for col in categorical_cols}
    
    return mca_coords_df, mca, column_representations_dict


def _standardize_numeric(df, numeric_cols):
    """
    Standardize numeric columns using z-score normalization.
    
    Args:
        df: Dataframe with numeric columns
        numeric_cols: List of numeric column names
    
    Returns:
        tuple: (standardized_df, scaler)
    """
    if not numeric_cols:
        return None, None
    
    numeric_data = df[numeric_cols].copy()
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_data)
    standardized_df = pd.DataFrame(
        standardized_data,
        index=numeric_data.index,
        columns=numeric_cols
    )
    
    return standardized_df, scaler


def _combine_features(numeric_df, mca_coords, numeric_cols, categorical_cols):
    """
    Combine standardized numeric features and MCA-transformed categorical features.
    Then standardize all features together to ensure proper covariance calculation.
    
    Args:
        numeric_df: Standardized numeric dataframe
        mca_coords: MCA coordinates dataframe (with original categorical column names)
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
    
    Returns:
        Combined and standardized feature matrix as dataframe
    """
    feature_parts = []
    feature_names = []
    
    # Add numeric columns (already standardized)
    if numeric_df is not None and len(numeric_cols) > 0:
        feature_parts.append(numeric_df)
        feature_names.extend(numeric_cols)
    
    # Add MCA-transformed categorical columns (not yet standardized)
    if mca_coords is not None:
        feature_parts.append(mca_coords)
        feature_names.extend(mca_coords.columns.tolist())
    
    if not feature_parts:
        return None
    
    # Combine all features
    combined_df = pd.concat(feature_parts, axis=1)
    combined_df.columns = feature_names
    
    # Standardize ALL features together (both numeric and categorical)
    # This ensures they're on the same scale and covariance is calculated correctly
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    standardized_values = scaler.fit_transform(combined_df)
    combined_df = pd.DataFrame(
        standardized_values,
        index=combined_df.index,
        columns=combined_df.columns
    )
    
    return combined_df


# ============================================================================
# Covariance and Clustering
# ============================================================================

def _calculate_covariance_matrix(feature_df):
    """
    Calculate covariance matrix from feature dataframe.
    
    Args:
        feature_df: Combined feature dataframe
    
    Returns:
        Covariance matrix as dataframe
    """
    cov_matrix = feature_df.cov()
    return cov_matrix


def _perform_hierarchical_clustering(cov_matrix, method='ward', metric='euclidean'):
    """
    Perform hierarchical clustering on columns based on covariance matrix.
    
    Args:
        cov_matrix: Covariance matrix dataframe
        method: Linkage method ('ward', 'complete', 'average', 'single')
        metric: Distance metric (only used if method != 'ward')
    
    Returns:
        tuple: (linkage_matrix, column_order)
    """
    # Convert covariance to distance matrix
    # Use 1 - normalized covariance as distance (higher covariance = lower distance)
    # Normalize covariance matrix to [0, 1] range
    cov_values = cov_matrix.values
    cov_min = cov_values.min()
    cov_max = cov_values.max()
    
    if cov_max - cov_min > 1e-10:
        normalized_cov = (cov_values - cov_min) / (cov_max - cov_min)
        # Convert to distance: 1 - normalized_cov (higher covariance = lower distance)
        distance_matrix = 1 - normalized_cov
    else:
        # If all values are the same, use zeros
        distance_matrix = np.zeros_like(cov_values)
    
    # Make sure distance matrix is symmetric and has zeros on diagonal
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed distance matrix for linkage
    condensed_distances = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    if method == 'ward':
        linkage_matrix = linkage(condensed_distances, method='ward')
    else:
        linkage_matrix = linkage(condensed_distances, method=method, metric=metric)
    
    # Get column order from dendrogram
    from scipy.cluster.hierarchy import leaves_list
    column_order = cov_matrix.columns[leaves_list(linkage_matrix)].tolist()
    
    return linkage_matrix, column_order


# ============================================================================
# Visualization
# ============================================================================

def _create_covariance_visualization(cov_matrix, linkage_matrix, column_order):
    """
    Create visualization with covariance heatmap and dendrogram.
    
    Args:
        cov_matrix: Covariance matrix dataframe
        linkage_matrix: Linkage matrix from hierarchical clustering
        column_order: Column order from clustering
    
    Returns:
        BytesIO object containing the image
    """
    # Calculate figure size based on number of columns
    num_cols = len(cov_matrix.columns)
    # Adjust figure size: minimum 14x12, scale up for more columns
    fig_width = max(14, num_cols * 0.8)
    fig_height = max(12, num_cols * 0.7)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid layout: dendrogram on top, heatmap below
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.5)
    
    # Top: dendrogram
    ax_dendro = fig.add_subplot(gs[0, 0])
    
    # Bottom: heatmap
    ax_heatmap = fig.add_subplot(gs[1, 0])
    
    # Plot dendrogram with proper labels
    dendrogram(linkage_matrix, ax=ax_dendro, orientation='top', 
               labels=cov_matrix.columns.tolist(), leaf_rotation=45, leaf_font_size=8)
    ax_dendro.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold', pad=10)
    ax_dendro.set_xlabel('')
    ax_dendro.set_ylabel('Distance', fontsize=10)
    
    # Reorder covariance matrix according to clustering
    reordered_cov = cov_matrix.loc[column_order, column_order]
    
    # Plot heatmap with explicit column names
    sns.heatmap(reordered_cov, annot=False, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax_heatmap, 
                xticklabels=reordered_cov.columns.tolist(), 
                yticklabels=reordered_cov.index.tolist())
    ax_heatmap.set_title('Covariance Matrix (Reordered by Clustering)', 
                        fontsize=12, fontweight='bold', pad=20)
    ax_heatmap.set_xlabel('Features', fontsize=10)
    ax_heatmap.set_ylabel('Features', fontsize=10)
    
    # Rotate and adjust labels for better readability
    # X-axis labels: rotate 45 degrees, align right
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    # Y-axis labels: keep horizontal, align right
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, ha='right', fontsize=9)
    
    # Ensure all labels are visible
    plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax_heatmap.get_yticklabels(), rotation=0, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # Convert to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ============================================================================
# Main Analysis Function
# ============================================================================

def calculate_covariance_analysis(selected_columns, drop_na=False, 
                                   mca_components=None, linkage_method='ward'):
    """
    Perform complete covariance analysis with MCA and hierarchical clustering.
    
    Args:
        selected_columns: List of columns to analyze
        drop_na: Whether to drop rows with NA values
        mca_components: Number of MCA components (None = auto)
        linkage_method: Linkage method for clustering
    
    Returns:
        dict with analysis results
    """
    # Use view instead of copy when possible to save memory
    df = st.session_state.processed_df
    
    # Prepare data
    work_df, numeric_cols, categorical_cols, na_info = _prepare_data(
        df, selected_columns, drop_na
    )
    
    if len(work_df) == 0:
        return None
    
    # Check if we have any columns to work with
    if not numeric_cols and not categorical_cols:
        return None
    
    # Store minimal data in results to reduce memory usage
    results = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'na_info': na_info,
        'num_rows': len(work_df),  # Store row count instead of full dataframe
    }
    
    # Apply MCA to categorical columns
    mca_coords = None
    mca_model = None
    mca_column_mapping = None
    if categorical_cols:
        with st.spinner("Applying Multiple Correspondence Analysis (MCA) to categorical columns..."):
            mca_coords, mca_model, mca_column_mapping = _apply_mca(work_df, categorical_cols, mca_components)
            if mca_coords is not None:
                # Don't store large MCA objects in results to save memory
                results['mca_column_mapping'] = mca_column_mapping
                results['mca_explained_inertia'] = mca_model.explained_inertia_ if hasattr(mca_model, 'explained_inertia_') else None
                # mca_coords and mca_model are only needed temporarily
    
    # Prepare numeric columns (will be standardized together with categorical in _combine_features)
    numeric_df = None
    if numeric_cols:
        # Just get the numeric data, standardization happens in _combine_features
        # Use view when possible, only copy if needed
        numeric_df = work_df[numeric_cols]
        # Don't store in results to save memory
    
    # Combine features (this will standardize everything together)
    with st.spinner("Combining and standardizing features..."):
        feature_df = _combine_features(numeric_df, mca_coords, numeric_cols, categorical_cols)
        if feature_df is None:
            return None
        results['feature_df'] = feature_df
    
    # Calculate covariance matrix
    with st.spinner("Calculating covariance matrix..."):
        cov_matrix = _calculate_covariance_matrix(feature_df)
        results['cov_matrix'] = cov_matrix
    
    # Perform hierarchical clustering
    with st.spinner("Performing hierarchical clustering..."):
        linkage_matrix, column_order = _perform_hierarchical_clustering(
            cov_matrix, method=linkage_method
        )
        results['linkage_matrix'] = linkage_matrix
        results['column_order'] = column_order
    
    return results


# ============================================================================
# Display Functions
# ============================================================================

def display_analysis_results(results):
    """Display analysis results and visualizations."""
    if results is None:
        st.error("Analysis failed. Please check your data and column selection.")
        return
    
    # Display summary information
    st.subheader("Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Numeric Columns", len(results['numeric_cols']))
    
    with col2:
        st.metric("Categorical Columns", len(results['categorical_cols']))
    
    with col3:
        total_features = len(results['feature_df'].columns)
        st.metric("Total Features", total_features)
    
    with col4:
        st.metric("Rows Used", results.get('num_rows', len(results['feature_df'])))
    
    # Display NA information
    na_info = results['na_info']
    if na_info['has_na'] and not na_info['rows_dropped']:
        st.warning(f"⚠️ **Warning:** Data contains missing values. "
                  f"NA counts: {dict(na_info['na_counts'])}. "
                  f"Consider enabling 'Drop NA values' for more accurate results.")
    elif na_info['rows_dropped'] > 0:
        st.info(f"ℹ️ Dropped {na_info['rows_dropped']} row(s) with missing values.")
    
    # Display column classifications
    if results['numeric_cols']:
        st.write("**Numeric Columns:**", ", ".join(results['numeric_cols']))
    
    if results['categorical_cols']:
        st.write("**Categorical Columns:**", ", ".join(results['categorical_cols']))
    
    # Display MCA information if applicable
    if 'mca_explained_inertia' in results and results['mca_explained_inertia'] is not None:
        st.subheader("MCA Information")
        if results.get('mca_explained_inertia') is not None:
            explained = results['mca_explained_inertia']
            st.write(f"MCA Components: {len(explained)}")
            total_explained = sum(explained[:min(10, len(explained))]) * 100
            st.write(f"Explained Inertia (first 10 components): {total_explained:.2f}%")
    
    # Display covariance matrix
    st.subheader("Covariance Matrix")
    cov_matrix = results['cov_matrix']
    
    # Ensure index and columns have proper names for display
    cov_matrix_display = cov_matrix.copy()
    cov_matrix_display.index.name = 'Features'
    cov_matrix_display.columns.name = 'Features'
    
    # Show matrix as dataframe with proper formatting
    st.dataframe(cov_matrix_display, use_container_width=True)
    
    # Display statistics
    st.write("**Covariance Statistics:**")
    cov_values = cov_matrix.values
    st.write(f"- Min: {cov_values.min():.4f}")
    st.write(f"- Max: {cov_values.max():.4f}")
    st.write(f"- Mean: {cov_values.mean():.4f}")
    st.write(f"- Std: {cov_values.std():.4f}")
    
    # Create and display visualization
    st.subheader("Covariance Heatmap with Dendrogram")
    
    try:
        img_buf = _create_covariance_visualization(
            cov_matrix,
            results['linkage_matrix'],
            results['column_order']
        )
        
        st.image(img_buf, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# ============================================================================
# Main Render Function
# ============================================================================

def render():
    """Render the Covariance tab."""
    st.header("Covariance Analysis")
    st.write("Calculate covariance matrix with dendrograms using MCA (Multiple Correspondence Analysis) "
             "for categorical columns and numeric scaling for numeric columns.")
    
    # Check if data is available
    if not _has_processed_data():
        st.info("No processed data available. Please import and process data in the 'Import Data' tab.")
        return
    
    available_columns = _get_available_columns()
    
    if len(available_columns) == 0:
        st.warning("No columns available for analysis.")
        return
    
    # Column selection (all selected by default)
    st.subheader("Column Selection")
    selected_columns = st.multiselect(
        "Select columns to analyze",
        options=available_columns,
        default=available_columns,  # All selected by default
        help="Select columns to include in the covariance analysis. "
             "Categorical columns will be processed with MCA, numeric columns will be standardized."
    )
    
    if not selected_columns:
        st.warning("Please select at least one column.")
        return
    
    # Options
    st.subheader("Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        drop_na = st.checkbox(
            "Drop NA values",
            value=False,
            help="If enabled, rows with any missing values will be dropped before analysis."
        )
    
    with col2:
        linkage_method = st.selectbox(
            "Linkage method",
            options=['ward', 'complete', 'average', 'single'],
            index=0,
            help="Method for hierarchical clustering. 'ward' minimizes variance, "
                 "'complete' uses maximum distances, 'average' uses mean distances, "
                 "'single' uses minimum distances."
        )
    
    # MCA components (optional)
    mca_components = st.number_input(
        "MCA components (leave 0 for auto)",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
        help="Number of MCA components to use. Set to 0 for automatic selection."
    )
    
    if mca_components == 0:
        mca_components = None
    
    # Analyze button
    if st.button("Calculate Covariance Matrix", type="primary"):
        # Perform analysis
        results = calculate_covariance_analysis(
            selected_columns,
            drop_na=drop_na,
            mca_components=mca_components,
            linkage_method=linkage_method
        )
        
        if results:
            st.session_state.covariance_results = results
        else:
            st.error("Analysis failed. Please check your data and column selection.")
    
    # Display results if available
    if 'covariance_results' in st.session_state:
        st.divider()
        display_analysis_results(st.session_state.covariance_results)

