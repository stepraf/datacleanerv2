import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
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
BASE_EDGE_FONT_SIZE = 14
MIN_NODE_FONT_SIZE = 8
MIN_EDGE_FONT_SIZE = 12

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

# Color thresholds
VIOLATION_RATIO_RED_THRESHOLD = 0.05  # 5%
NA_PERCENTAGE_ORANGE_THRESHOLD = 0.15  # 15%
NA_PERCENTAGE_RED_THRESHOLD = 0.50  # 50%

# Label configuration
LABEL_MAX_CHARS = 16
MAX_EDGE_LABELS = 100

# Colors
COLOR_GREEN = '#00FF00'
COLOR_RED = '#FF0000'
COLOR_DARK_GREEN = '#006400'
COLOR_DARK_ORANGE = '#8B4513'
COLOR_DARK_RED = '#8B0000'


# ============================================================================
# Helper Functions
# ============================================================================

def _has_data():
    """Check if processed data is available."""
    return ('processed_df' in st.session_state and 
            st.session_state.processed_df is not None and 
            len(st.session_state.processed_df) > 0)


def _columns_exist(col1, col2):
    """Check if both columns exist in processed_df."""
    return (col1 in st.session_state.processed_df.columns and 
            col2 in st.session_state.processed_df.columns)


def _get_one_to_many_relationships():
    """Extract 1:N relationships from session state."""
    if 'one_to_many_results' not in st.session_state:
        return []
    
    relationships = []
    for result in st.session_state.one_to_many_results:
        col1 = result.get('column1')
        col2 = result.get('column2')
        if col1 and col2 and _columns_exist(col1, col2):
            relationships.append((col1, col2))
    
    return relationships


def _get_violation_ratios():
    """Extract violation ratios for 1:N relationships from session state."""
    if 'one_to_many_results' not in st.session_state:
        return {}
    
    violation_ratios = {}
    for result in st.session_state.one_to_many_results:
        col1 = result.get('column1')
        col2 = result.get('column2')
        violation_ratio = result.get('violation_ratio', 0.0)
        
        if col1 and col2 and _columns_exist(col1, col2):
            violation_ratios[(col1, col2)] = violation_ratio
    
    return violation_ratios


def _get_na_percentages():
    """Calculate NA percentages for all columns in processed_df."""
    if 'processed_df' not in st.session_state or st.session_state.processed_df is None:
        return {}
    
    df = st.session_state.processed_df
    na_percentages = {}
    
    total_rows = len(df)
    if total_rows == 0:
        return na_percentages
    
    for col in df.columns:
        na_count = df[col].isna().sum()
        na_percentage = na_count / total_rows
        na_percentages[col] = na_percentage
    
    return na_percentages


def _remove_transitive_edges(relationships):
    """
    Remove transitive edges from a list of directed relationships.
    
    For example, if we have A→B, B→C, and A→C, we remove A→C
    because it's implied by A→B→C.
    
    Handles 1:1 relationships (bidirectional edges) by keeping only one direction
    to avoid cycles.
    
    Args:
        relationships: List of tuples (source, target)
    
    Returns:
        List of tuples with transitive edges removed
    """
    if not relationships:
        return []
    
    # Build a directed graph
    G = nx.DiGraph()
    G.add_edges_from(relationships)
    
    # Separate bidirectional edges (1:1 relationships) from directional edges
    # A bidirectional edge is when both (A→B) and (B→A) exist
    bidirectional_pairs = set()
    directional_edges = []
    
    for u, v in relationships:
        if G.has_edge(v, u):
            # This is a bidirectional edge (1:1 relationship)
            # Store as sorted tuple to avoid duplicates
            pair = tuple(sorted([u, v]))
            bidirectional_pairs.add(pair)
        else:
            # This is a directional edge (1:N relationship)
            directional_edges.append((u, v))
    
    # For bidirectional edges, keep only one direction (alphabetically first → second)
    # This breaks the cycle and allows transitive reduction to work
    bidirectional_edges_kept = []
    for u, v in sorted(bidirectional_pairs):
        # Keep the edge in alphabetical order for consistency
        bidirectional_edges_kept.append((u, v))
    
    # Now work with only the directional edges for transitive reduction
    if not directional_edges:
        # Only bidirectional edges, return them (one direction each)
        return bidirectional_edges_kept
    
    # Build graph with only directional edges
    G_directional = nx.DiGraph()
    G_directional.add_edges_from(directional_edges)
    
    # Check if directional graph is acyclic (required for transitive reduction)
    if not nx.is_directed_acyclic_graph(G_directional):
        # If there are cycles in directional edges, we can't do transitive reduction
        # Return all edges (bidirectional + directional)
        return bidirectional_edges_kept + directional_edges
    
    # Use NetworkX's transitive reduction on directional edges only
    try:
        G_reduced = nx.transitive_reduction(G_directional)
        reduced_directional_edges = list(G_reduced.edges())
        
        # Verify the reduction is correct: check that reachability is preserved
        # For each original directional edge, if it was removed, verify there's still a path
        original_directional_set = set(directional_edges)
        reduced_directional_set = set(reduced_directional_edges)
        removed_directional = original_directional_set - reduced_directional_set
        
        # Verify each removed edge is indeed transitive
        for u, v in removed_directional:
            if not nx.has_path(G_reduced, u, v):
                # This shouldn't happen - if NetworkX removed it, there should be a path
                # But let's be safe and keep it if there's no path
                reduced_directional_edges.append((u, v))
                G_reduced.add_edge(u, v)
        
        # Combine bidirectional edges (kept as-is) with reduced directional edges
        return bidirectional_edges_kept + reduced_directional_edges
    except Exception as e:
        # Fallback: manual transitive reduction on directional edges only
        reduced_directional_edges = _manual_transitive_reduction(directional_edges)
        return bidirectional_edges_kept + reduced_directional_edges


def _manual_transitive_reduction(relationships):
    """
    Manually remove transitive edges.
    
    For each edge (u, v), check if there's a path from u to v
    that doesn't use the direct edge (u, v).
    
    This is a more robust implementation that handles edge cases.
    """
    if not relationships:
        return []
    
    # Build the graph
    G = nx.DiGraph()
    G.add_edges_from(relationships)
    
    # Ensure graph is acyclic
    if not nx.is_directed_acyclic_graph(G):
        return relationships
    
    reduced_edges = []
    
    # Process edges in a stable order
    for u, v in relationships:
        # Skip if edge doesn't exist (shouldn't happen, but be safe)
        if not G.has_edge(u, v):
            continue
        
        # Temporarily remove the edge
        G.remove_edge(u, v)
        
        # Check if there's still a path from u to v
        # If no path exists, this edge is necessary (not transitive)
        if not nx.has_path(G, u, v):
            reduced_edges.append((u, v))
        
        # Restore the edge for next iteration
        G.add_edge(u, v)
    
    # Verify the reduction preserves reachability
    # Build the reduced graph
    G_reduced = nx.DiGraph()
    G_reduced.add_edges_from(reduced_edges)
    
    # Check that all original reachability is preserved
    # For each original edge, verify that if it was removed, there's still a path
    original_edges_set = set(relationships)
    reduced_edges_set = set(reduced_edges)
    removed_edges = original_edges_set - reduced_edges_set
    
    # Verify each removed edge is indeed transitive (has a path in reduced graph)
    for u, v in removed_edges:
        if not nx.has_path(G_reduced, u, v):
            # This edge was incorrectly removed - there's no path without it
            # Add it back to preserve reachability
            reduced_edges.append((u, v))
            G_reduced.add_edge(u, v)
    
    return reduced_edges



def _interpolate_color(start_rgb, end_rgb, ratio):
    """Interpolate between two RGB colors."""
    red = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
    green = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
    blue = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
    return f'#{red:02X}{green:02X}{blue:02X}'


def _na_percentage_to_color(na_percentage):
    """
    Convert NA percentage to node color.
    Dark green for 0% NA values.
    Shades of dark orange for 0-15% NA values.
    Dark red for 50%+ NA values.
    
    Args:
        na_percentage: float between 0.0 (no NA) and 1.0 (all NA)
    
    Returns:
        str: Hex color string
    """
    na_percentage = max(0.0, min(1.0, na_percentage))
    
    if na_percentage == 0.0:
        return COLOR_DARK_GREEN
    elif na_percentage >= NA_PERCENTAGE_RED_THRESHOLD:
        return COLOR_DARK_RED
    elif na_percentage <= NA_PERCENTAGE_ORANGE_THRESHOLD:
        # Interpolate from dark green to dark orange (0-15%)
        normalized_ratio = na_percentage / NA_PERCENTAGE_ORANGE_THRESHOLD
        return _interpolate_color((0x00, 0x64, 0x00), (0x8B, 0x45, 0x13), normalized_ratio)
    else:
        # Interpolate from dark orange to dark red (15-50%)
        normalized_ratio = (na_percentage - NA_PERCENTAGE_ORANGE_THRESHOLD) / (
            NA_PERCENTAGE_RED_THRESHOLD - NA_PERCENTAGE_ORANGE_THRESHOLD)
        return _interpolate_color((0x8B, 0x45, 0x13), (0x8B, 0x00, 0x00), normalized_ratio)


def _calculate_figure_size(num_nodes):
    """Calculate figure dimensions based on number of nodes."""
    if num_nodes <= SMALL_GRAPH_NODES:
        return BASE_FIG_WIDTH, BASE_FIG_HEIGHT
    elif num_nodes <= MEDIUM_GRAPH_NODES:
        scale_factor = (num_nodes / SMALL_GRAPH_NODES) ** 0.6
        width = BASE_FIG_WIDTH * scale_factor * 2.0
        height = BASE_FIG_HEIGHT * scale_factor * 2.0
    elif num_nodes <= LARGE_GRAPH_NODES:
        scale_factor = (num_nodes / SMALL_GRAPH_NODES) ** 0.5
        width = BASE_FIG_WIDTH * scale_factor * 3.0
        height = BASE_FIG_HEIGHT * scale_factor * 3.0
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
    elif num_nodes <= MEDIUM_GRAPH_NODES:
        return max(BASE_NODE_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.3, MIN_NODE_SIZE_SMALL)
    elif num_nodes <= LARGE_GRAPH_NODES:
        return max(BASE_NODE_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.25, MIN_NODE_SIZE_MEDIUM)
    else:
        return max(BASE_NODE_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.2, MIN_NODE_SIZE_LARGE)


def _calculate_font_sizes(num_nodes):
    """Calculate font sizes for nodes and edges based on number of nodes."""
    if num_nodes <= SMALL_GRAPH_NODES:
        return BASE_NODE_FONT_SIZE, BASE_EDGE_FONT_SIZE
    elif num_nodes <= MEDIUM_GRAPH_NODES:
        node_size = max(BASE_NODE_FONT_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.15, BASE_NODE_FONT_SIZE)
        edge_size = max(BASE_EDGE_FONT_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.15, BASE_EDGE_FONT_SIZE)
    elif num_nodes <= LARGE_GRAPH_NODES:
        node_size = max(BASE_NODE_FONT_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.1, 9)
        edge_size = max(BASE_EDGE_FONT_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.1, 13)
    else:
        node_size = max(BASE_NODE_FONT_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.05, MIN_NODE_FONT_SIZE)
        edge_size = max(BASE_EDGE_FONT_SIZE * (SMALL_GRAPH_NODES / num_nodes) ** 0.05, MIN_EDGE_FONT_SIZE)
    return node_size, edge_size


def _calculate_edge_sizes(num_nodes):
    """Calculate arrow size, edge width, and node linewidth based on number of nodes."""
    if num_nodes <= SMALL_GRAPH_NODES:
        return BASE_ARROW_SIZE, BASE_EDGE_WIDTH, BASE_NODE_LINEWIDTH
    elif num_nodes <= MEDIUM_GRAPH_NODES:
        scale = (SMALL_GRAPH_NODES / num_nodes) ** 0.3
        arrow = max(BASE_ARROW_SIZE * scale, 12)
        width = max(BASE_EDGE_WIDTH * scale, 1.0)
        linewidth = max(BASE_NODE_LINEWIDTH * scale, 1)
    else:
        scale = (SMALL_GRAPH_NODES / num_nodes) ** 0.4
        arrow = max(BASE_ARROW_SIZE * scale, 10)
        width = max(BASE_EDGE_WIDTH * scale, 0.8)
        linewidth = max(BASE_NODE_LINEWIDTH * scale, 0.8)
    return arrow, width, linewidth


def _wrap_label(text, max_chars=LABEL_MAX_CHARS):
    """Wrap text at max_chars per line, breaking at word boundaries when possible."""
    if len(text) <= max_chars:
        return text
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        if len(test_line) <= max_chars:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                while len(word) > max_chars:
                    lines.append(word[:max_chars])
                    word = word[max_chars:]
                current_line = [word] if word else []
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)


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
        
        if not pos:
            raise Exception("Failed to extract positions")
        
        return pos, "Graphviz dot (minimizes crossings, increased spacing)", None
    except Exception:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            return pos, "Graphviz dot (minimizes crossings)", None
        except Exception as e:
            try:
                pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
                return pos, "Graphviz dot via pydot (minimizes crossings)", None
            except Exception as e2:
                raise Exception(f"Graphviz dot layout is required but not available. Error: {str(e2)}")


def _calculate_dpi(num_nodes, fig_width, fig_height):
    """Calculate DPI based on graph size and figure dimensions."""
    if num_nodes <= SMALL_GRAPH_NODES:
        desired_dpi = DPI_SMALL
    elif num_nodes <= MEDIUM_GRAPH_NODES:
        desired_dpi = DPI_MEDIUM
    elif num_nodes <= LARGE_GRAPH_NODES:
        desired_dpi = DPI_LARGE
    elif num_nodes <= VERY_LARGE_GRAPH_NODES:
        desired_dpi = DPI_VERY_LARGE
    elif num_nodes <= HUGE_GRAPH_NODES:
        desired_dpi = DPI_HUGE
    else:
        desired_dpi = DPI_MAXIMUM
    
    max_safe_dpi = int((MAX_PIXELS / (fig_width * fig_height)) ** 0.5)
    dpi = min(desired_dpi, max_safe_dpi)
    
    if num_nodes <= MEDIUM_GRAPH_NODES:
        min_dpi = DPI_SMALL
    elif num_nodes <= LARGE_GRAPH_NODES:
        min_dpi = DPI_MEDIUM
    else:
        min_dpi = 250
    
    return max(dpi, min_dpi)


def _violation_ratio_to_color(violation_ratio):
    """
    Convert violation ratio to color.
    Green for 0% violation ratio.
    Shades transitioning to red for 0-5% violation ratio.
    Red for 5%+ violation ratio.
    
    Args:
        violation_ratio: float between 0.0 (perfect) and 1.0 (all violations)
    
    Returns:
        str: Hex color string
    """
    violation_ratio = max(0.0, min(1.0, violation_ratio))
    
    if violation_ratio == 0.0:
        return COLOR_GREEN
    elif violation_ratio >= VIOLATION_RATIO_RED_THRESHOLD:
        return COLOR_RED
    else:
        # Interpolate from green to red (0-5%)
        normalized_ratio = violation_ratio / VIOLATION_RATIO_RED_THRESHOLD
        return _interpolate_color((0x00, 0xFF, 0x00), (0xFF, 0x00, 0x00), normalized_ratio)


def _create_graph_visualization(relationships, violation_ratios=None, na_percentages=None):
    """
    Create a visualization of the relationship graph.
    Uses Graphviz 'dot' layout to minimize edge crossings.
    Requires Graphviz to be installed.
    
    Args:
        relationships: List of tuples (source, target)
        violation_ratios: Dictionary mapping (source, target) tuples to violation ratios (0.0-1.0)
        na_percentages: Dictionary mapping column names to NA percentages (0.0-1.0)
    
    Returns:
        tuple: (BytesIO object containing the image, layout_method string, layout_error string or None)
    
    Raises:
        Exception: If Graphviz is not available
    """
    if not relationships:
        return None, None, None
    
    # Build the graph
    G = nx.DiGraph()
    G.add_edges_from(relationships)
    
    if len(G.nodes()) == 0:
        return None, None, None
    
    pos, layout_method, layout_error = _get_graphviz_layout(G)
    
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    
    # Calculate all scaling factors
    fig_width, fig_height = _calculate_figure_size(num_nodes)
    node_size = _calculate_node_size(num_nodes)
    node_font_size, edge_font_size = _calculate_font_sizes(num_nodes)
    arrow_size, edge_width, node_linewidth = _calculate_edge_sizes(num_nodes)
    
    # Create figure with scaled size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Prepare node colors based on NA percentages
    if na_percentages:
        node_colors = []
        for node in G.nodes():
            na_percentage = na_percentages.get(node, 0.0)
            color = _na_percentage_to_color(na_percentage)
            node_colors.append(color)
    else:
        # Default to white if no NA percentages provided
        node_colors = 'white'
    
    # Draw nodes with scaled size
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_size, alpha=0.9, edgecolors='black', linewidths=node_linewidth)
    
    # Prepare edge colors based on violation ratios
    if violation_ratios:
        edge_colors = []
        for edge in G.edges():
            # Get violation ratio for this edge
            # NetworkX edges are tuples, so edge should match directly
            violation_ratio = violation_ratios.get(edge, 0.0)
            color = _violation_ratio_to_color(violation_ratio)
            edge_colors.append(color)
    else:
        # Default to gray if no violation ratios provided
        edge_colors = 'gray'
    
    # Draw edges with arrows (straight edges for hierarchical layout)
    # Use edge-specific colors if violation ratios are available
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, 
                          arrows=True, arrowsize=arrow_size, arrowstyle='->', 
                          width=edge_width, alpha=0.7)
    
    # Draw edge labels with subset symbol (only if not too many edges)
    if num_edges <= MAX_EDGE_LABELS:
        edge_labels = {(u, v): '⊂' for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=edge_font_size, 
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Create wrapped labels dictionary
    wrapped_labels = {node: _wrap_label(node) for node in G.nodes()}
    
    label_positions = {}
    for node, (x, y) in pos.items():
        # Slight vertical offset to improve readability
        label_positions[node] = (x, y)
    
    # Draw labels with improved spacing and larger padding for readability
    # Increase padding for larger graphs to make text more readable
    label_padding = 0.3 if num_nodes <= 50 else (0.4 if num_nodes <= 100 else 0.5)
    
    nx.draw_networkx_labels(G, label_positions, labels=wrapped_labels, ax=ax, 
                           font_size=node_font_size, font_weight='bold', 
                           font_family='sans-serif',
                           bbox=dict(boxstyle='round,pad=' + str(label_padding), 
                                   facecolor='white', edgecolor='none', alpha=0.9))
    
    ax.axis('off')
    plt.tight_layout()
    
    dpi = _calculate_dpi(num_nodes, fig_width, fig_height)
    
    # Convert to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf, layout_method, layout_error


def _get_graph_statistics(relationships):
    """Get statistics about the graph."""
    if not relationships:
        return {}
    
    G = nx.DiGraph()
    G.add_edges_from(relationships)
    
    stats = {
        'total_nodes': len(G.nodes()),
        'total_edges': len(G.edges()),
        'is_acyclic': nx.is_directed_acyclic_graph(G),
        'has_cycles': not nx.is_directed_acyclic_graph(G),
    }
    
    # Find root nodes (nodes with no incoming edges)
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    stats['root_nodes'] = root_nodes
    stats['num_root_nodes'] = len(root_nodes)
    
    # Find leaf nodes (nodes with no outgoing edges)
    leaf_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    stats['leaf_nodes'] = leaf_nodes
    stats['num_leaf_nodes'] = len(leaf_nodes)
    
    return stats


# ============================================================================
# Main Render Function
# ============================================================================

def render():
    """Render the Relation Graph tab."""
    st.header("Relation Graph")
    st.write("Visualize hierarchical relationships between columns based on 1:N dependencies.")
    
    # Check if data is available
    if not _has_data():
        st.info("No data loaded. Please import a CSV file in the 'Import Data' tab.")
        return
    
    # Check if 1:N analysis has been performed
    relationships = _get_one_to_many_relationships()
    violation_ratios = _get_violation_ratios()
    na_percentages = _get_na_percentages()
    
    if not relationships:
        st.info("No 1:N relationships found. Please run the 1:N analysis in the '1:N' tab first.")
        return
    
    st.success(f"Found {len(relationships)} relationship(s) from 1:N analysis.")
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        remove_transitive = st.checkbox(
            "Remove transitive edges",
            value=True,
            help="Remove edges that are implied by other paths (e.g., if A→B→C exists, remove A→C)",
            key="relation_graph_remove_transitive"
        )
    
    with col2:
        show_statistics = st.checkbox(
            "Show graph statistics",
            value=True,
            help="Display statistics about the graph structure",
            key="relation_graph_show_stats"
        )
    
    # Process relationships
    if remove_transitive:
        # Check for bidirectional edges (1:1 relationships)
        G_check = nx.DiGraph()
        G_check.add_edges_from(relationships)
        bidirectional_count = 0
        for u, v in relationships:
            if G_check.has_edge(v, u):
                bidirectional_count += 1
        
        # Count unique bidirectional pairs (divide by 2 since each pair has 2 edges)
        unique_bidirectional_pairs = bidirectional_count // 2
        
        if bidirectional_count > 0:
            st.info(f"📊 Found {unique_bidirectional_pairs} bidirectional relationship(s) (1:1). These will be shown as single edges to avoid cycles.")
        
        # Process relationships (handles bidirectional edges internally)
        processed_relationships = _remove_transitive_edges(relationships)
        removed_count = len(relationships) - len(processed_relationships)
        
        # Filter violation ratios to only include processed relationships
        # Default to 0.0 if violation ratio not found (perfect relationship)
        processed_violation_ratios = {
            edge: violation_ratios.get(edge, 0.0) 
            for edge in processed_relationships
        }
        
        if removed_count > 0:
            st.info(f"After removing transitive edges: {len(processed_relationships)} direct relationship(s) (removed {removed_count} transitive edge(s)).")
        else:
            st.info(f"After removing transitive edges: {len(processed_relationships)} direct relationship(s) (no transitive edges found).")
        
        # Show what was removed
        if removed_count > 0:
            G_original = nx.DiGraph()
            G_original.add_edges_from(relationships)
            G_reduced = nx.DiGraph()
            G_reduced.add_edges_from(processed_relationships)
            removed_edges = set(relationships) - set(processed_relationships)
            
            # Identify bidirectional pairs (1:1 relationships)
            bidirectional_removed = set()
            for u, v in removed_edges:
                if (v, u) in removed_edges or (v, u) in processed_relationships:
                    bidirectional_removed.add((u, v))
            
            if removed_edges:
                with st.expander("🔍 Removed Transitive Edges"):
                    st.write("The following edges were removed:")
                    for u, v in sorted(removed_edges):
                        # Check if this is a bidirectional edge (1:1 relationship)
                        if (u, v) in bidirectional_removed:
                            # This is part of a bidirectional pair - removed to break cycle
                            reverse_edge = (v, u)
                            if reverse_edge in processed_relationships:
                                st.write(f"- **{u} → {v}** (removed to break bidirectional cycle, kept: {v} → {u})")
                            else:
                                st.write(f"- **{u} → {v}** (removed to break bidirectional cycle)")
                        else:
                            # Check if there's a path that implies this edge
                            G_temp = G_reduced.copy()
                            if nx.has_path(G_temp, u, v):
                                try:
                                    path = nx.shortest_path(G_temp, u, v)
                                    path_str = " → ".join(path)
                                    st.write(f"- **{u} → {v}** (implied by: {path_str})")
                                except Exception:
                                    st.write(f"- **{u} → {v}** (implied by another path)")
                            else:
                                # This shouldn't happen, but if it does, it might be a bidirectional edge
                                # or an edge case in transitive reduction
                                st.write(f"- **{u} → {v}** (removed during transitive reduction)")
    else:
        processed_relationships = relationships
        processed_violation_ratios = violation_ratios
    
    # Show statistics if requested
    if show_statistics and processed_relationships:
        stats = _get_graph_statistics(processed_relationships)
        
        # Build graph for detailed analysis
        G = nx.DiGraph()
        G.add_edges_from(processed_relationships)
        
        # Find nodes with multiple edges
        nodes_multiple_incoming = [n for n in G.nodes() if G.in_degree(n) > 1]
        nodes_multiple_outgoing = [n for n in G.nodes() if G.out_degree(n) > 1]
        
        with st.expander("📊 Graph Statistics"):
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Total Nodes", stats['total_nodes'])
                st.metric("Total Edges", stats['total_edges'])
            
            with col_stat2:
                st.metric("Root Nodes", stats['num_root_nodes'])
                st.metric("Leaf Nodes", stats['num_leaf_nodes'])
            
            with col_stat3:
                acyclic_status = "✓ Acyclic" if stats['is_acyclic'] else "✗ Has Cycles"
                st.write(f"**Graph Type:** {acyclic_status}")
                
                if stats['root_nodes']:
                    st.write("**Root Nodes:**")
                    for node in stats['root_nodes']:
                        st.write(f"- {node}")
        
        # Show detailed edge information
        if nodes_multiple_incoming or nodes_multiple_outgoing:
            with st.expander("🔍 Edge Analysis - Why Some Nodes Have Multiple Edges"):
                if nodes_multiple_incoming:
                    st.write("**Nodes with Multiple Incoming Edges:**")
                    st.write("These nodes are subsets of multiple parent columns independently.")
                    for node in nodes_multiple_incoming:
                        predecessors = list(G.predecessors(node))
                        st.write(f"- **{node}** has {len(predecessors)} parent(s): {', '.join(predecessors)}")
                        for pred in predecessors:
                            st.write(f"  - {pred} → {node}")
                
                if nodes_multiple_outgoing:
                    st.write("**Nodes with Multiple Outgoing Edges:**")
                    st.write("These nodes have multiple child columns that are subsets of them.")
                    for node in nodes_multiple_outgoing:
                        successors = list(G.successors(node))
                        st.write(f"- **{node}** has {len(successors)} child(ren): {', '.join(successors)}")
                        for succ in successors:
                            st.write(f"  - {node} → {succ}")
                
                if not nodes_multiple_incoming and not nodes_multiple_outgoing:
                    st.info("All nodes have at most one incoming and one outgoing edge. This is a simple chain structure.")
        
    # Create and display visualization
    if processed_relationships:
        st.subheader("Relationship Graph")
        
        # Note about Graphviz requirement
        with st.expander("ℹ️ About Graph Layout"):
            st.write("""
            The graph uses Graphviz 'dot' layout to minimize edge crossings.
            
            **Required**: Graphviz must be installed:
            - **Linux**: `sudo apt-get install graphviz` or `sudo yum install graphviz`
            - **macOS**: `brew install graphviz`
            - **Windows**: Download from https://graphviz.org/download/
            
            Then install Python bindings: `pip install pygraphviz` or `pip install pydot`
            """)
        
        # Show graph size info
        G_size = nx.DiGraph()
        G_size.add_edges_from(processed_relationships)
        num_nodes = len(G_size.nodes())
        num_edges = len(G_size.edges())
        
        if num_nodes > 20:
            st.info(f"📊 Large graph detected ({num_nodes} nodes, {num_edges} edges). Graph size and element sizes have been automatically scaled for better visibility.")
        
        # Show node color legend
        if na_percentages:
            st.info("🎨 **Node Colors:** Dark green = 0% NA values, Dark orange shades = 0-15% NA values, Dark red = 50%+ NA values")
        
        # Show edge color legend
        if processed_violation_ratios:
            non_zero_processed = {k: v for k, v in processed_violation_ratios.items() if v > 0.0}
            if non_zero_processed:
                st.info(f"🎨 **Edge Colors:** Green = 0% violation ratio (perfect 1:N relationship), Yellow/Orange shades = 0-5% violations, Red = 5%+ violation ratio. Found {len(non_zero_processed)} edge(s) with violations.")
            else:
                st.info("🎨 **Edge Colors:** Green = 0% violation ratio (perfect 1:N relationship), Yellow/Orange shades = 0-5% violations, Red = 5%+ violation ratio. (All edges have 0% violation ratio)")
        
        with st.spinner("Generating graph visualization..."):
            try:
                result = _create_graph_visualization(processed_relationships, processed_violation_ratios, na_percentages)
                
                if result and result[0]:
                    img_buf, layout_method, layout_error = result
                    
                    # Show layout method info
                    if layout_method:
                        st.success(f"✓ Using {layout_method}")
                    
                    # Convert image to base64 for HTML display (bypasses Streamlit's image size limits)
                    img_buf.seek(0)  # Reset buffer position
                    img_data = img_buf.read()
                    img_base64 = base64.b64encode(img_data).decode()
                    
                    # Get image dimensions for display info
                    img_buf.seek(0)  # Reset again for PIL
                    try:
                        # Use the Image import from the top of the file if available
                        pil_img = Image.open(img_buf)
                        img_width, img_height = pil_img.size
                    except (NameError, AttributeError):
                        # Fallback if Image is not available
                        from PIL import Image as PILImage
                        img_buf.seek(0)
                        pil_img = PILImage.open(img_buf)
                        img_width, img_height = pil_img.size
                    
                    # Display image using HTML to bypass Streamlit's resolution limits
                    # This allows the browser to display the full resolution image
                    st.markdown(
                        f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;" />',
                        unsafe_allow_html=True
                    )
                    
                    # Show image info
                    st.info(f"📐 **Image Resolution:** {img_width} × {img_height} pixels ({img_width*img_height/1_000_000:.1f} MP). "
                           f"💡 **Tip:** Use your browser's zoom (Ctrl/Cmd + Mouse Wheel) to zoom in for better readability. "
                           f"Right-click and 'Open Image in New Tab' to view at full resolution.")
                    
                    # Download button (reset buffer position again)
                    img_buf.seek(0)
                    st.download_button(
                        label="Download Full Resolution Graph Image",
                        data=img_buf,
                        file_name="relation_graph.png",
                        mime="image/png"
                    )
                else:
                    st.error("Failed to generate graph visualization.")
            except Exception as e:
                st.error(f"❌ Graphviz is required but not available. Error: {str(e)}")
                st.info("""
                **To fix this:**
                1. Install Graphviz on your system:
                   - Linux: `sudo apt-get install graphviz` or `sudo yum install graphviz`
                   - macOS: `brew install graphviz`
                   - Windows: Download from https://graphviz.org/download/
                
                2. Install Python bindings:
                   - `pip install pygraphviz` or `pip install pydot`
                
                3. Restart the application
                """)
    else:
        st.warning("No relationships to visualize after processing.")

