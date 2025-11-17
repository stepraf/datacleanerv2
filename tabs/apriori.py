import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from io import BytesIO
import base64

# Increase PIL/Pillow image size limit to allow very large graphs
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 2000000000  # 2 billion pixels
except ImportError:
    pass


# ============================================================================
# Constants
# ============================================================================

# Graph size thresholds
SMALL_GRAPH_NODES = 20
MEDIUM_GRAPH_NODES = 50
LARGE_GRAPH_NODES = 100
VERY_LARGE_GRAPH_NODES = 200
HUGE_GRAPH_NODES = 500

# Figure size configuration
BASE_FIG_WIDTH = 14
BASE_FIG_HEIGHT = 10
MIN_FIG_WIDTH = 20
MIN_FIG_HEIGHT = 14
MAX_FIG_WIDTH = 60
MAX_FIG_HEIGHT = 50

# Node size configuration
BASE_NODE_SIZE = 3000
MIN_NODE_SIZE_SMALL = 2000
MIN_NODE_SIZE_MEDIUM = 1800
MIN_NODE_SIZE_LARGE = 1500

# Font size configuration
BASE_NODE_FONT_SIZE = 10
MIN_NODE_FONT_SIZE = 8

# Edge configuration
BASE_ARROW_SIZE = 20
BASE_EDGE_WIDTH = 2.0
BASE_NODE_LINEWIDTH = 2

# DPI configuration
DPI_SMALL = 150
DPI_MEDIUM = 200
DPI_LARGE = 300
DPI_VERY_LARGE = 500
DPI_HUGE = 750
DPI_MAXIMUM = 1000
MAX_PIXELS = 250000000  # 250 million pixels

# Spring layout parameters
SPRING_LAYOUT_PARAMS = {
    SMALL_GRAPH_NODES: {'k': 4.5, 'iterations': 120},
    MEDIUM_GRAPH_NODES: {'k': 4.0, 'iterations': 100},
    LARGE_GRAPH_NODES: {'k': 3.5, 'iterations': 80},
    VERY_LARGE_GRAPH_NODES: {'k': 3.0, 'iterations': 70},
    HUGE_GRAPH_NODES: {'k': 2.5, 'iterations': 60},
}

# Edge density thresholds for layout adjustments
DENSE_EDGE_DENSITY = 0.3
MODERATE_EDGE_DENSITY = 0.15
LOW_EDGE_DENSITY = 0.05
WEIGHTED_LAYOUT_THRESHOLD = 0.2


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
    result_series = pd.Series(index=series.index, dtype=object)
    non_null_mask = series.notna()
    non_null_series = series[non_null_mask]
    
    if len(non_null_series) == 0:
        return result_series
    
    try:
        if method == 'quantile':
            binned_result = pd.qcut(non_null_series, q=n_bins, duplicates='drop', retbins=True)
        else:
            binned_result = pd.cut(non_null_series, bins=n_bins, duplicates='drop', retbins=True)
        
        binned = binned_result[0]
        
        for cat in binned.cat.categories:
            interval_str = str(cat)
            result_series.loc[non_null_mask & (binned == cat)] = interval_str
        
    except (ValueError, TypeError):
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
    processed_df = pd.DataFrame(index=work_df.index)
    
    # Process each column
    for col in selected_columns:
        series = work_df[col]
        if _is_numeric(series):
            processed_df[col] = _discretize_numeric_column(series, n_bins=n_bins)
        else:
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
        
        if transaction:
            transactions.append(transaction)
    
    return transactions


# ============================================================================
# Apriori Algorithm
# ============================================================================

def _get_frequent_itemsets_1(transactions, min_support):
    """Get frequent 1-itemsets."""
    item_counts = Counter()
    total_transactions = len(transactions)
    
    for transaction in transactions:
        item_counts.update(transaction)
    
    min_support_count = min_support * total_transactions
    inv_total = 1.0 / total_transactions
    frequent_1 = {}
    
    for item, count in item_counts.items():
        if count >= min_support_count:
            frequent_1[frozenset([item])] = {
                'support': count * inv_total,
                'count': count
            }
    
    return frequent_1


def _generate_candidates(frequent_k_minus_1, k):
    """Generate candidate k-itemsets from frequent (k-1)-itemsets."""
    candidates = set()
    itemsets = list(frequent_k_minus_1.keys())
    frequent_set = set(frequent_k_minus_1.keys())
    
    # Generate candidates by joining itemsets that share (k-2) items
    for i in range(len(itemsets)):
        itemset1 = itemsets[i]
        for j in range(i + 1, len(itemsets)):
            itemset2 = itemsets[j]
            union = itemset1 | itemset2
            if len(union) == k:
                candidates.add(union)
    
    # Prune: remove candidates that have infrequent subsets
    pruned_candidates = set()
    for candidate in candidates:
        all_subsets_frequent = True
        for subset in combinations(candidate, k - 1):
            if frozenset(subset) not in frequent_set:
                all_subsets_frequent = False
                break
        
        if all_subsets_frequent:
            pruned_candidates.add(candidate)
    
    return pruned_candidates


def _count_support(transactions, candidates, min_support_count=0):
    """Count support for candidate itemsets."""
    if not candidates:
        return {}
    
    support_counts = defaultdict(int)
    total_transactions = len(transactions)
    
    # Pre-convert transactions to sets
    transaction_sets = [t if isinstance(t, set) else set(t) for t in transactions]
    
    # Count support
    for trans_set in transaction_sets:
        for candidate in candidates:
            if candidate.issubset(trans_set):
                support_counts[candidate] += 1
    
    # Convert counts to support and filter
    support_dict = {}
    inv_total = 1.0 / total_transactions
    for itemset, count in support_counts.items():
        if count >= min_support_count:
            support_dict[itemset] = {
                'support': count * inv_total,
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
    frequent_1 = _get_frequent_itemsets_1(transactions, min_support)
    
    if not frequent_1:
        return all_frequent
    
    all_frequent[1] = frequent_1
    k = 2
    total_transactions = len(transactions)
    min_support_count = min_support * total_transactions
    
    while True:
        if max_itemset_size and k > max_itemset_size:
            break
        
        candidates = _generate_candidates(all_frequent[k - 1], k)
        if not candidates:
            break
        
        frequent_k = _count_support(transactions, candidates, min_support_count)
        if not frequent_k:
            break
        
        all_frequent[k] = frequent_k
        k += 1
    
    return all_frequent


# ============================================================================
# Association Rules Generation
# ============================================================================

def _generate_association_rules(frequent_itemsets, min_confidence):
    """Generate association rules from frequent itemsets."""
    rules = []
    
    # Create lookup dictionary for O(1) access
    itemset_lookup = {}
    for size_dict in frequent_itemsets.values():
        for itemset, info in size_dict.items():
            itemset_lookup[itemset] = info
    
    # Process itemsets of size 2 or more
    for k in range(2, len(frequent_itemsets) + 1):
        if k not in frequent_itemsets:
            continue
        
        for itemset, itemset_info in frequent_itemsets[k].items():
            itemset_list = list(itemset)
            itemset_support = itemset_info['support']
            
            # Generate all possible rules
            for r in range(1, k):
                for antecedent in combinations(itemset_list, r):
                    antecedent_set = frozenset(antecedent)
                    consequent_set = itemset - antecedent_set
                    
                    antecedent_support_info = itemset_lookup.get(antecedent_set)
                    if not antecedent_support_info:
                        continue
                    
                    antecedent_support = antecedent_support_info['support']
                    if antecedent_support <= 0:
                        continue
                    
                    confidence = itemset_support / antecedent_support
                    
                    if confidence >= min_confidence:
                        consequent_support_info = itemset_lookup.get(consequent_set)
                        consequent_support = consequent_support_info['support'] if consequent_support_info else 0
                        
                        lift = float('inf') if consequent_support <= 0 else confidence / consequent_support
                        
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
    """Run complete Apriori analysis."""
    df = st.session_state.processed_df.copy()
    
    with st.spinner("Preparing transactions..."):
        transactions = _prepare_transactions(
            df, selected_columns, treat_na_as_category, n_bins
        )
    
    if len(transactions) == 0:
        return None
    
    with st.spinner("Finding frequent itemsets..."):
        frequent_itemsets = apriori_algorithm(
            transactions, min_support, max_itemset_size
        )
    
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


# ============================================================================
# Graph Layout Helper Functions
# ============================================================================

def _get_graphviz_spacing(num_nodes):
    """Get Graphviz spacing parameters based on graph size."""
    if num_nodes <= SMALL_GRAPH_NODES:
        return '1.0', '1.5'
    elif num_nodes <= MEDIUM_GRAPH_NODES:
        return '2.0', '2.5'
    elif num_nodes <= LARGE_GRAPH_NODES:
        return '3.0', '3.5'
    else:
        return '4.0', '4.5'


def _calculate_figure_size(num_nodes):
    """Calculate figure dimensions based on number of nodes."""
    if num_nodes <= SMALL_GRAPH_NODES:
        return BASE_FIG_WIDTH, BASE_FIG_HEIGHT
    
    scale_factors = {
        MEDIUM_GRAPH_NODES: (0.6, 2.0),
        LARGE_GRAPH_NODES: (0.5, 3.0),
    }
    
    for threshold, (exponent, multiplier) in scale_factors.items():
        if num_nodes <= threshold:
            scale_factor = (num_nodes / SMALL_GRAPH_NODES) ** exponent
            width = BASE_FIG_WIDTH * scale_factor * multiplier
            height = BASE_FIG_HEIGHT * scale_factor * multiplier
            break
    else:
        scale_factor = (num_nodes / SMALL_GRAPH_NODES) ** 0.45
        width = BASE_FIG_WIDTH * scale_factor * 4.0
        height = BASE_FIG_HEIGHT * scale_factor * 4.0
    
    width = max(MIN_FIG_WIDTH, min(width, MAX_FIG_WIDTH))
    height = max(MIN_FIG_HEIGHT, min(height, MAX_FIG_HEIGHT))
    return width, height


def _calculate_node_size(num_nodes):
    """Calculate node size based on number of nodes."""
    if num_nodes <= SMALL_GRAPH_NODES:
        return BASE_NODE_SIZE
    
    size_configs = {
        MEDIUM_GRAPH_NODES: (0.3, MIN_NODE_SIZE_SMALL),
        LARGE_GRAPH_NODES: (0.25, MIN_NODE_SIZE_MEDIUM),
    }
    
    for threshold, (exponent, min_size) in size_configs.items():
        if num_nodes <= threshold:
            return max(BASE_NODE_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** exponent, min_size)
    
    return max(BASE_NODE_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.2, MIN_NODE_SIZE_LARGE)


def _calculate_font_size(num_nodes):
    """Calculate font size for nodes based on number of nodes."""
    if num_nodes <= SMALL_GRAPH_NODES:
        return BASE_NODE_FONT_SIZE
    
    font_configs = {
        MEDIUM_GRAPH_NODES: (0.15, BASE_NODE_FONT_SIZE),
        LARGE_GRAPH_NODES: (0.1, 9),
    }
    
    for threshold, (exponent, min_size) in font_configs.items():
        if num_nodes <= threshold:
            return max(BASE_NODE_FONT_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** exponent, min_size)
    
    return max(BASE_NODE_FONT_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.05, MIN_NODE_FONT_SIZE)


def _calculate_edge_sizes(num_nodes):
    """Calculate arrow size, edge width, and node linewidth."""
    if num_nodes <= SMALL_GRAPH_NODES:
        return BASE_ARROW_SIZE, BASE_EDGE_WIDTH, BASE_NODE_LINEWIDTH
    
    if num_nodes <= MEDIUM_GRAPH_NODES:
        scale = (SMALL_GRAPH_NODES / num_nodes) ** 0.3
        return (
            max(BASE_ARROW_SIZE * scale, 12),
            max(BASE_EDGE_WIDTH * scale, 1.0),
            max(BASE_NODE_LINEWIDTH * scale, 1)
        )
    else:
        scale = (SMALL_GRAPH_NODES / num_nodes) ** 0.4
        return (
            max(BASE_ARROW_SIZE * scale, 10),
            max(BASE_EDGE_WIDTH * scale, 0.8),
            max(BASE_NODE_LINEWIDTH * scale, 0.8)
        )


def _calculate_dpi(num_nodes, fig_width, fig_height):
    """Calculate DPI based on graph size and figure dimensions."""
    dpi_map = {
        SMALL_GRAPH_NODES: DPI_SMALL,
        MEDIUM_GRAPH_NODES: DPI_MEDIUM,
        LARGE_GRAPH_NODES: DPI_LARGE,
        VERY_LARGE_GRAPH_NODES: DPI_VERY_LARGE,
        HUGE_GRAPH_NODES: DPI_HUGE,
    }
    
    desired_dpi = DPI_MAXIMUM
    for threshold, dpi in dpi_map.items():
        if num_nodes <= threshold:
            desired_dpi = dpi
            break
    
    max_safe_dpi = int((MAX_PIXELS / (fig_width * fig_height)) ** 0.5)
    dpi = min(desired_dpi, max_safe_dpi)
    
    min_dpi = DPI_SMALL if num_nodes <= MEDIUM_GRAPH_NODES else (DPI_MEDIUM if num_nodes <= LARGE_GRAPH_NODES else 250)
    return max(dpi, min_dpi)


def _get_graphviz_layout(G):
    """Get graph layout using Graphviz dot algorithm."""
    try:
        A = nx.nx_agraph.to_agraph(G)
        nodesep, ranksep = _get_graphviz_spacing(len(G.nodes()))
        
        A.graph_attr['nodesep'] = nodesep
        A.graph_attr['ranksep'] = ranksep
        A.graph_attr['dpi'] = '75'
        A.layout(prog='dot')
        
        pos = {}
        for node in G.nodes():
            n = A.get_node(node)
            if 'pos' in n.attr:
                coords = n.attr['pos'].split(',')
                pos[node] = (float(coords[0]), float(coords[1]))
        
        if pos:
            return pos, "Graphviz dot (minimizes crossings, increased spacing)", None
        else:
            raise Exception("Failed to extract positions")
            
    except Exception:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            return pos, "Graphviz dot (minimizes crossings)", None
        except Exception:
            try:
                pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
                return pos, "Graphviz dot via pydot (minimizes crossings)", None
            except Exception as e2:
                return None, None, f"Graphviz not available: {str(e2)}"


def _get_spring_layout_params(num_nodes, edge_density):
    """Get spring layout parameters based on graph size and density."""
    # Find appropriate base parameters
    params = {'k': 2.5, 'iterations': 60}
    for threshold, config in SPRING_LAYOUT_PARAMS.items():
        if num_nodes <= threshold:
            params = config
            break
    
    # Adjust k based on edge density
    if edge_density > DENSE_EDGE_DENSITY:
        params['k'] *= 1.5
    elif edge_density > MODERATE_EDGE_DENSITY:
        params['k'] *= 1.3
    elif edge_density > LOW_EDGE_DENSITY:
        params['k'] *= 1.15
    
    params['weight'] = 'weight' if edge_density < WEIGHTED_LAYOUT_THRESHOLD else None
    return params


def _build_rule_graph(rules, edge_metric):
    """Build NetworkX graph from association rules."""
    G = nx.DiGraph()
    all_items = set()
    edge_data = {}
    
    def get_weight(rule):
        if edge_metric == 'confidence':
            return rule['confidence']
        return rule['lift'] if rule['lift'] != float('inf') else 0
    
    for rule in rules:
        antecedent_items = list(rule['antecedent'])
        consequent_items = list(rule['consequent'])
        ant_size = len(antecedent_items)
        cons_size = len(consequent_items)
        rule_weight = get_weight(rule)
        rule_type = f"{ant_size}→{cons_size}"
        
        all_items.update(antecedent_items)
        all_items.update(consequent_items)
        
        for ant_item in antecedent_items:
            for cons_item in consequent_items:
                edge_key = (ant_item, cons_item)
                if edge_key in edge_data:
                    edge_data[edge_key]['weight'] = max(edge_data[edge_key]['weight'], rule_weight)
                else:
                    edge_data[edge_key] = {
                        'weight': rule_weight,
                        'ant_size': ant_size,
                        'cons_size': cons_size,
                        'rule_type': rule_type
                    }
    
    G.add_nodes_from(all_items)
    for (ant_item, cons_item), data in edge_data.items():
        G.add_edge(ant_item, cons_item, **data)
    
    return G


def _prepare_edge_visualization(G):
    """Prepare edge attributes for visualization."""
    edges = list(G.edges())
    if not edges:
        return None, None, None, None, None
    
    weights = []
    edge_rule_types = []
    for u, v in edges:
        edge_attrs = G[u][v]
        weights.append(edge_attrs.get('weight', 0))
        edge_rule_types.append(edge_attrs.get('rule_type', '?→?'))
    
    weights_arr = np.array(weights)
    if weights_arr.max() == 0:
        return None, None, None, None, None
    
    rule_type_counts = Counter(edge_rule_types)
    
    # Normalize weights
    min_weight = weights_arr.min()
    max_weight = weights_arr.max()
    weight_range = max_weight - min_weight
    
    if weight_range > 0:
        normalized_weights = ((weights_arr - min_weight) / weight_range * 3 + 0.5).tolist()
        normalized_colors_by_weight = ((weights_arr - min_weight) / weight_range).tolist()
    else:
        normalized_weights = [1.0] * len(edges)
        normalized_colors_by_weight = [0.5] * len(edges)
    
    # Create color map for rule types
    unique_rule_types = sorted(set(edge_rule_types))
    if unique_rule_types:
        color_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_rule_types)))
        rule_type_colors = dict(zip(unique_rule_types, color_palette))
        edge_colors_by_type = [rule_type_colors.get(rt, 'gray') for rt in edge_rule_types]
    else:
        rule_type_colors = {}
        edge_colors_by_type = ['gray'] * len(edges)
    
    return (
        normalized_weights,
        normalized_colors_by_weight,
        edge_colors_by_type,
        rule_type_colors,
        rule_type_counts
    )


def _get_edge_curvature(edge_density):
    """Calculate edge curvature based on graph density."""
    if edge_density > DENSE_EDGE_DENSITY:
        return 0.3
    elif edge_density > MODERATE_EDGE_DENSITY:
        return 0.2
    elif edge_density > LOW_EDGE_DENSITY:
        return 0.15
    else:
        return 0.1


def _draw_graph(G, pos, edge_widths, edge_colors, use_colormap, edge_alpha,
                arrow_size, connection_style, node_size, font_size, node_linewidth,
                min_weight, max_weight, edge_metric):
    """Draw the network graph."""
    fig, ax = plt.subplots(figsize=_calculate_figure_size(len(G.nodes())))
    
    # Draw edges
    if use_colormap:
        edges_drawn = nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,
            alpha=edge_alpha,
            arrows=True,
            arrowsize=arrow_size,
            arrowstyle='->',
            connectionstyle=connection_style
        )
    else:
        edges_drawn = nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=edge_alpha,
            arrows=True,
            arrowsize=arrow_size,
            arrowstyle='->',
            connectionstyle=connection_style
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color='lightblue',
        node_size=node_size,
        alpha=0.9,
        edgecolors='black',
        linewidths=node_linewidth
    )
    
    # Draw labels
    for node, (x, y) in pos.items():
        ax.text(x, y, node, 
                fontsize=font_size, 
                fontweight='bold',
                color='black',
                ha='center',
                va='center',
                rotation=10,
                rotation_mode='anchor')
    
    # Add colorbar or legend
    if use_colormap and edges_drawn is not None:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, 
            norm=plt.Normalize(vmin=min_weight, vmax=max_weight)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'{edge_metric.capitalize()}', rotation=270, labelpad=20)
    
    return fig, ax, edges_drawn


def _save_and_display_graph(fig, num_nodes, fig_width, fig_height):
    """Save graph to buffer and return image data."""
    dpi = _calculate_dpi(num_nodes, fig_width, fig_height)
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', dpi=dpi, bbox_inches='tight')
    img_buf.seek(0)
    plt.close()
    
    # Get image dimensions
    img_buf.seek(0)
    try:
        pil_img = Image.open(img_buf)
        img_width, img_height = pil_img.size
    except (NameError, AttributeError):
        from PIL import Image as PILImage
        img_buf.seek(0)
        pil_img = PILImage.open(img_buf)
        img_width, img_height = pil_img.size
    
    # Convert to base64
    img_buf.seek(0)
    img_data = img_buf.read()
    img_base64 = base64.b64encode(img_data).decode()
    
    return img_buf, img_base64, img_width, img_height


def display_rule_network_graph(rules, max_rules=100, edge_metric='confidence', color_by='rule_type'):
    """Display association rules as a directed network graph."""
    if not rules:
        st.info("No rules to visualize.")
        return
    
    sorted_rules = rules[:max_rules]
    if not sorted_rules:
        st.info("No rules to visualize after filtering.")
        return
    
    # Build graph
    G = _build_rule_graph(sorted_rules, edge_metric)
    
    if len(G.nodes()) == 0:
        st.info("No nodes to visualize.")
        return
    
    # Prepare edge visualization
    result = _prepare_edge_visualization(G)
    if result[0] is None:
        st.warning("No valid edge weights to visualize.")
        return
    
    normalized_weights, normalized_colors_by_weight, edge_colors_by_type, rule_type_colors, rule_type_counts = result
    
    # Calculate graph metrics
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    edge_density = num_edges / max(1, num_nodes * (num_nodes - 1))
    
    # Calculate scaling factors
    fig_width, fig_height = _calculate_figure_size(num_nodes)
    node_size = _calculate_node_size(num_nodes)
    font_size = _calculate_font_size(num_nodes)
    arrow_size, base_edge_width, node_linewidth = _calculate_edge_sizes(num_nodes)
    
    # Get layout
    pos, layout_method, layout_error = _get_graphviz_layout(G)
    
    if pos is None:
        params = _get_spring_layout_params(num_nodes, edge_density)
        pos = nx.spring_layout(
            G, 
            k=params['k'], 
            iterations=params['iterations'], 
            seed=42,
            weight=params['weight'],
            pos=None,
            fixed=None,
            threshold=1e-6
        )
        layout_method = f"Spring layout (k={params['k']:.1f}, {params['iterations']} iterations)"
    
    # Prepare edge visualization
    if color_by == 'rule_type':
        edge_colors_to_use = edge_colors_by_type
        use_colormap = False
    else:
        edge_colors_to_use = normalized_colors_by_weight
        use_colormap = True
    
    edge_curvature = _get_edge_curvature(edge_density)
    connection_style = f'arc3,rad={edge_curvature}'
    edge_alpha = 0.5 if edge_density > 0.2 else 0.6
    edge_widths = [w * base_edge_width for w in normalized_weights]
    
    # Get weight range for colorbar
    weights_arr = np.array([G[u][v].get('weight', 0) for u, v in G.edges()])
    min_weight = weights_arr.min()
    max_weight = weights_arr.max()
    
    # Draw graph
    fig, ax, edges_drawn = _draw_graph(
        G, pos, edge_widths, edge_colors_to_use, use_colormap, edge_alpha,
        arrow_size, connection_style, node_size, font_size, node_linewidth,
        min_weight, max_weight, edge_metric
    )
    
    # Add legend if needed
    if not use_colormap and rule_type_colors:
        unique_rule_types = sorted(rule_type_counts.keys())
        legend_elements = [
            Patch(facecolor=rule_type_colors[rt], label=f'{rt} ({rule_type_counts[rt]} edges)') 
            for rt in unique_rule_types
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=8)
    
    # Set title
    if color_by == 'rule_type':
        title_color_info = f"Edge thickness represents {edge_metric.capitalize()}, color represents rule type (antecedent→consequent size)"
    else:
        title_color_info = f"Edge thickness and color represent {edge_metric.capitalize()}"
    
    ax.set_title(
        f'Association Rules Network Graph\n'
        f'({len(sorted_rules)} rules, {num_nodes} items, {num_edges} edges)\n'
        f'{title_color_info}',
        fontsize=12,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')
    plt.tight_layout()
    
    # Save and display
    img_buf, img_base64, img_width, img_height = _save_and_display_graph(fig, num_nodes, fig_width, fig_height)
    
    st.markdown(
        f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;" />',
        unsafe_allow_html=True
    )
    
    st.info(
        f"📐 **Image Resolution:** {img_width} × {img_height} pixels ({img_width*img_height/1_000_000:.1f} MP). "
        f"💡 **Tip:** Use your browser's zoom (Ctrl/Cmd + Mouse Wheel) to zoom in for better readability. "
        f"Right-click and 'Open Image in New Tab' to view at full resolution."
    )
    
    img_buf.seek(0)
    st.download_button(
        label="Download Full Resolution Graph Image",
        data=img_buf,
        file_name="apriori_network_graph.png",
        mime="image/png"
    )
    
    # Display info
    rule_type_summary = ", ".join([f"{rt}: {count}" for rt, count in sorted(rule_type_counts.items())])
    
    if color_by == 'rule_type':
        info_text = (
            f"**Graph Info:**\n"
            f"- Layout: {layout_method}\n"
            f"- Nodes represent items\n"
            f"- Edges represent rules (antecedent → consequent)\n"
            f"- Edge thickness represents {edge_metric.capitalize()}\n"
            f"- Edge color represents rule type (antecedent→consequent size)\n"
            f"- Edge curvature adjusted for density (curvature={edge_curvature:.2f}) to minimize overlap\n"
            f"- Rule type distribution: {rule_type_summary}\n"
            f"- Showing {len(sorted_rules)} rules (using same filters and sort order as table)"
        )
    else:
        info_text = (
            f"**Graph Info:**\n"
            f"- Layout: {layout_method}\n"
            f"- Nodes represent items\n"
            f"- Edges represent rules (antecedent → consequent)\n"
            f"- Edge thickness and color represent {edge_metric.capitalize()}\n"
            f"- Edge curvature adjusted for density (curvature={edge_curvature:.2f}) to minimize overlap\n"
            f"- Rule type distribution: {rule_type_summary}\n"
            f"- Showing {len(sorted_rules)} rules (using same filters and sort order as table)"
        )
    
    st.info(info_text)


# ============================================================================
# Association Rules Display
# ============================================================================

def _calculate_rule_statistics(rules):
    """Calculate statistics from rules for filter ranges."""
    supports = []
    confidences = []
    lifts = []
    antecedent_sizes = []
    consequent_sizes = []
    
    for r in rules:
        supports.append(r['support'])
        confidences.append(r['confidence'])
        lift_val = r['lift']
        if lift_val != float('inf'):
            lifts.append(lift_val)
        antecedent_sizes.append(len(r['antecedent']))
        consequent_sizes.append(len(r['consequent']))
    
    return supports, confidences, lifts, antecedent_sizes, consequent_sizes


def _create_filter_ui(supports, confidences, lifts, antecedent_sizes, consequent_sizes):
    """Create filter UI components."""
    with st.expander("🔍 Filter and Sort Rules", expanded=True):
        st.write("**Filter by Metrics:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_support = st.slider("Min Support", 0.0, 1.0, 0.0, 0.01, key="filter_min_support")
            max_support = st.slider("Max Support", 0.0, 1.0, 1.0, 0.01, key="filter_max_support")
        
        with col2:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.01, key="filter_min_confidence")
            max_confidence = st.slider("Max Confidence", 0.0, 1.0, 1.0, 0.01, key="filter_max_confidence")
        
        with col3:
            if lifts:
                min_lift_val = max(0.0, min(lifts))
                max_lift_val = min(100.0, max(lifts))
            else:
                min_lift_val, max_lift_val = 0.0, 10.0
            
            min_lift = st.slider("Min Lift", 0.0, max_lift_val if max_lift_val > 0 else 10.0, 0.0, 0.1, key="filter_min_lift")
            max_lift = st.slider("Max Lift", 0.0, max_lift_val if max_lift_val > 0 else 10.0, max_lift_val if max_lift_val > 0 else 10.0, 0.1, key="filter_max_lift")
        
        st.write("**Filter by Itemset Size:**")
        col1, col2 = st.columns(2)
        
        with col1:
            max_ant_size = max(antecedent_sizes) if antecedent_sizes else 10
            min_antecedent_size = st.slider("Min Antecedent Size", 1, max_ant_size, 1, 1, key="filter_min_antecedent_size")
            max_antecedent_size = st.slider("Max Antecedent Size", 1, max_ant_size, max_ant_size, 1, key="filter_max_antecedent_size")
        
        with col2:
            max_cons_size = max(consequent_sizes) if consequent_sizes else 10
            min_consequent_size = st.slider("Min Consequent Size", 1, max_cons_size, 1, 1, key="filter_min_consequent_size")
            max_consequent_size = st.slider("Max Consequent Size", 1, max_cons_size, max_cons_size, 1, key="filter_max_consequent_size")
        
        st.write("**Search Items:**")
        search_term = st.text_input(
            "Search for items (comma-separated)",
            value="",
            help="Enter items separated by commas to filter rules containing those items",
            key="filter_search_items"
        )
        
        st.write("**Sort Options:**")
        col1, col2 = st.columns(2)
        
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                options=["Confidence", "Support", "Lift", "Antecedent Size", "Consequent Size"],
                index=0,
                key="sort_by_metric"
            )
        
        with col2:
            sort_order = st.selectbox(
                "Sort order",
                options=["Descending", "Ascending"],
                index=0,
                key="sort_order"
            )
    
    return {
        'min_support': min_support,
        'max_support': max_support,
        'min_confidence': min_confidence,
        'max_confidence': max_confidence,
        'min_lift': min_lift,
        'max_lift': max_lift,
        'min_antecedent_size': min_antecedent_size,
        'max_antecedent_size': max_antecedent_size,
        'min_consequent_size': min_consequent_size,
        'max_consequent_size': max_consequent_size,
        'search_term': search_term,
        'sort_by': sort_by,
        'sort_order': sort_order
    }


def _filter_rules(rules, filters):
    """Filter rules based on filter criteria."""
    filtered_rules = []
    search_items_lower = None
    
    if filters['search_term']:
        search_items = [s.strip() for s in filters['search_term'].split(',') if s.strip()]
        if search_items:
            search_items_lower = [item.lower() for item in search_items]
    
    for r in rules:
        # Numeric filters
        if not (filters['min_support'] <= r['support'] <= filters['max_support']):
            continue
        if not (filters['min_confidence'] <= r['confidence'] <= filters['max_confidence']):
            continue
        
        # Size filters
        ant_size = len(r['antecedent'])
        cons_size = len(r['consequent'])
        if not (filters['min_antecedent_size'] <= ant_size <= filters['max_antecedent_size']):
            continue
        if not (filters['min_consequent_size'] <= cons_size <= filters['max_consequent_size']):
            continue
        
        # Lift filter
        lift_val = r['lift']
        if lift_val != float('inf') and not (filters['min_lift'] <= lift_val <= filters['max_lift']):
            continue
        
        # Search filter
        if search_items_lower:
            rule_items_lower = set(str(item).lower() for item in r['antecedent'])
            rule_items_lower.update(str(item).lower() for item in r['consequent'])
            if not any(search_item in rule_items_lower for search_item in search_items_lower):
                continue
        
        filtered_rules.append(r)
    
    return filtered_rules


def _sort_rules(filtered_rules, sort_by, sort_order):
    """Sort rules based on sort criteria."""
    sort_key_map = {
        "Confidence": lambda x: x['confidence'],
        "Support": lambda x: x['support'],
        "Lift": lambda x: x['lift'] if x['lift'] != float('inf') else 0,
        "Antecedent Size": lambda x: len(x['antecedent']),
        "Consequent Size": lambda x: len(x['consequent'])
    }
    
    reverse_order = (sort_order == "Descending")
    filtered_rules.sort(key=sort_key_map[sort_by], reverse=reverse_order)
    return filtered_rules


def _prepare_rule_display_data(filtered_rules):
    """Prepare data for rule display table."""
    display_data = []
    for rule in filtered_rules:
        lift_str = f"{rule['lift']:.4f}" if rule['lift'] != float('inf') else "∞"
        display_data.append({
            'Antecedent': _format_itemset(rule['antecedent']),
            'Consequent': _format_itemset(rule['consequent']),
            'Support': f"{rule['support']:.4f}",
            'Confidence': f"{rule['confidence']:.4f}",
            'Lift': lift_str,
            'Antecedent Size': len(rule['antecedent']),
            'Consequent Size': len(rule['consequent'])
        })
    return display_data


def _display_rule_statistics(filtered_rules):
    """Display statistics for filtered rules."""
    if not filtered_rules:
        return
    
    filtered_confidences = [r['confidence'] for r in filtered_rules]
    filtered_lifts = [r['lift'] for r in filtered_rules if r['lift'] != float('inf')]
    
    conf_arr = np.array(filtered_confidences)
    
    st.write("**Rule Statistics (Filtered):**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Confidence:**")
        st.write(f"- Mean: {conf_arr.mean():.4f}")
        st.write(f"- Median: {np.median(conf_arr):.4f}")
        st.write(f"- Min: {conf_arr.min():.4f}")
        st.write(f"- Max: {conf_arr.max():.4f}")
    
    with col2:
        st.write("**Lift:**")
        if filtered_lifts:
            lift_arr = np.array(filtered_lifts)
            st.write(f"- Mean: {lift_arr.mean():.4f}")
            st.write(f"- Median: {np.median(lift_arr):.4f}")
            st.write(f"- Min: {lift_arr.min():.4f}")
            st.write(f"- Max: {lift_arr.max():.4f}")
        else:
            st.write("- No valid lift values in filtered rules")


def _create_graph_settings_ui(filtered_rules):
    """Create graph settings UI."""
    with st.expander("⚙️ Graph Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            max_rules_graph = st.number_input(
                "Maximum rules to visualize",
                min_value=10,
                max_value=20000,
                value=min(100, len(filtered_rules)),
                step=10,
                help="Limiting the number of rules helps avoid overcrowding the graph"
            )
        
        with col2:
            edge_metric = st.selectbox(
                "Edge visualization metric",
                options=["confidence", "lift"],
                index=0,
                help="Choose whether edge thickness represents confidence or lift"
            )
        
        color_by = st.selectbox(
            "Edge color coding",
            options=["rule_type", "weight"],
            index=0,
            format_func=lambda x, em=edge_metric: "Rule Type (antecedent→consequent size)" if x == "rule_type" else f"{em.capitalize()} Value",
            help="Color edges by rule type (shows different antecedent/consequent sizes) or by metric value"
        )
    
    return max_rules_graph, edge_metric, color_by


def display_association_rules(rules):
    """Display association rules with interactive filtering and exploration."""
    if not rules:
        st.info("No association rules found.")
        return
    
    st.subheader("Association Rules")
    total_rules = len(rules)
    st.metric("Total Rules", total_rules)
    
    # Calculate statistics
    supports, confidences, lifts, antecedent_sizes, consequent_sizes = _calculate_rule_statistics(rules)
    
    # Create filter UI
    filters = _create_filter_ui(supports, confidences, lifts, antecedent_sizes, consequent_sizes)
    
    # Filter and sort rules
    filtered_rules = _filter_rules(rules, filters)
    filtered_rules = _sort_rules(filtered_rules, filters['sort_by'], filters['sort_order'])
    
    # Display results
    st.info(f"**Showing {len(filtered_rules)} of {total_rules} rules**")
    
    display_data = _prepare_rule_display_data(filtered_rules)
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Display statistics
    _display_rule_statistics(filtered_rules)
    
    # Network graph visualization
    if filtered_rules:
        st.divider()
        st.subheader("Network Graph Visualization")
        st.caption(f"Graph displays {len(filtered_rules)} filtered rule(s) (out of {total_rules} total)")
        
        max_rules_graph, edge_metric, color_by = _create_graph_settings_ui(filtered_rules)
        rules_to_visualize = filtered_rules[:max_rules_graph]
        display_rule_network_graph(rules_to_visualize, max_rules=len(rules_to_visualize), edge_metric=edge_metric, color_by=color_by)
    elif total_rules > 0:
        st.divider()
        st.subheader("Network Graph Visualization")
        st.warning("No rules match the current filters. Adjust filters to see the graph.")


def display_analysis_results(results):
    """Display complete analysis results."""
    if results is None:
        st.error("Analysis failed. Please check your data and column selection.")
        return
    
    display_frequent_itemsets(
        results['frequent_itemsets'],
        results['total_transactions']
    )
    
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
    
    # Explanation section
    st.divider()
    st.subheader("Understanding the Metrics")
    
    with st.expander("What do Support, Confidence, and Lift mean?"):
        st.markdown("""
        **Support**
        - Support measures how frequently an itemset (a set of items) appears in the dataset.
        - It is calculated as: Support = (Number of transactions containing itemset) / (Total number of transactions)
        - Range: 0 to 1 (or 0% to 100%)
        - Example: If 30 out of 100 transactions contain both "bread" and "butter", the support is 0.30 (30%).
        - Higher support indicates that the itemset is more common in the dataset.
        
        **Confidence**
        - Confidence measures how often items in the consequent appear in transactions that contain the antecedent.
        - It is calculated as: Confidence = Support(antecedent ∪ consequent) / Support(antecedent)
        - Range: 0 to 1 (or 0% to 100%)
        - Example: For the rule "bread → butter", if 30 transactions contain both bread and butter, and 50 transactions contain bread, 
          then confidence = 30/50 = 0.60 (60%). This means 60% of transactions with bread also contain butter.
        - Higher confidence indicates a stronger association between the antecedent and consequent.
        
        **Lift**
        - Lift measures how much more likely the consequent is to appear when the antecedent is present, compared to its baseline probability.
        - It is calculated as: Lift = Confidence / Support(consequent) = Support(antecedent ∪ consequent) / (Support(antecedent) × Support(consequent))
        - Range: 0 to ∞ (infinity)
        - Interpretation:
          - Lift = 1: The antecedent and consequent are independent (no association)
          - Lift > 1: Positive association (the antecedent makes the consequent more likely)
          - Lift < 1: Negative association (the antecedent makes the consequent less likely)
        - Example: If bread appears in 50% of transactions and butter appears in 40% of transactions, but they appear together in 30% of transactions,
          then Lift = 0.30 / (0.50 × 0.40) = 1.5. This means bread and butter appear together 1.5 times more often than expected by chance.
        - Higher lift indicates a stronger positive association between items.
        """)
