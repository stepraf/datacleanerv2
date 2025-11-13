import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Increase PIL/Pillow image size limit to allow very large graphs
# Default limit is ~178 million pixels, we'll increase it significantly
try:
    from PIL import Image
    # Set to 2 billion pixels (much higher than default)
    Image.MAX_IMAGE_PIXELS = 2000000000
except ImportError:
    pass


# ============================================================================
# Helper Functions
# ============================================================================

def _has_data():
    """Check if processed data is available."""
    return ('processed_df' in st.session_state and 
            st.session_state.processed_df is not None and 
            len(st.session_state.processed_df) > 0)


def _get_one_to_many_relationships():
    """Extract 1:N relationships from session state."""
    if 'one_to_many_results' not in st.session_state:
        return []
    
    relationships = []
    for result in st.session_state.one_to_many_results:
        col1 = result.get('column1')
        col2 = result.get('column2')
        if col1 and col2:
            # Check if both columns still exist
            if (col1 in st.session_state.processed_df.columns and 
                col2 in st.session_state.processed_df.columns):
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
        
        if col1 and col2:
            # Check if both columns still exist
            if (col1 in st.session_state.processed_df.columns and 
                col2 in st.session_state.processed_df.columns):
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



def _na_percentage_to_color(na_percentage):
    """
    Convert NA percentage to node color.
    Dark green for 0% NA values.
    Shades of dark orange for 0-15% NA values.
    Dark red for 50%+ NA values.
    
    Args:
        na_percentage: float between 0.0 (no NA) and 1.0 (all NA)
    
    Returns:
        str: Hex color string (e.g., '#006400' for dark green, '#8B4513' for dark orange, '#8B0000' for dark red)
    """
    # Clamp NA percentage to [0, 1]
    na_percentage = max(0.0, min(1.0, na_percentage))
    
    if na_percentage == 0.0:
        # Perfect: dark green
        return '#006400'  # Dark green
    elif na_percentage >= 0.50:
        # 50% or more NA: dark red
        return '#8B0000'  # Dark red
    elif na_percentage <= 0.15:
        # 0-15% NA: interpolate from dark green to dark orange
        # Map na_percentage from [0, 0.15] to [0, 1] for color interpolation
        normalized_ratio = na_percentage / 0.15
        
        # Dark green: #006400 = RGB(0, 100, 0)
        # Dark orange: #8B4513 = RGB(139, 69, 19)
        # Interpolate between these colors
        red_start, green_start, blue_start = 0x00, 0x64, 0x00  # Dark green RGB
        red_end, green_end, blue_end = 0x8B, 0x45, 0x13  # Dark orange RGB
        
        red = int(red_start + (red_end - red_start) * normalized_ratio)
        green = int(green_start + (green_end - green_start) * normalized_ratio)
        blue = int(blue_start + (blue_end - blue_start) * normalized_ratio)
        
        return f'#{red:02X}{green:02X}{blue:02X}'
    else:
        # 15-50% NA: interpolate from dark orange to dark red
        # Map na_percentage from [0.15, 0.50] to [0, 1] for color interpolation
        normalized_ratio = (na_percentage - 0.15) / (0.50 - 0.15)
        
        # Dark orange: #8B4513 = RGB(139, 69, 19)
        # Dark red: #8B0000 = RGB(139, 0, 0)
        red_start, green_start, blue_start = 0x8B, 0x45, 0x13  # Dark orange RGB
        red_end, green_end, blue_end = 0x8B, 0x00, 0x00  # Dark red RGB
        
        red = int(red_start + (red_end - red_start) * normalized_ratio)
        green = int(green_start + (green_end - green_start) * normalized_ratio)
        blue = int(blue_start + (blue_end - blue_start) * normalized_ratio)
        
        return f'#{red:02X}{green:02X}{blue:02X}'


def _violation_ratio_to_color(violation_ratio):
    """
    Convert violation ratio to color.
    Green for 0% violation ratio.
    Shades transitioning to red for 0-5% violation ratio.
    Red (#FF0000) for 5% violation ratio and above.
    
    Args:
        violation_ratio: float between 0.0 (perfect) and 1.0 (all violations)
    
    Returns:
        str: Hex color string (e.g., '#00FF00' for green, '#FF0000' for red)
    """
    # Clamp violation ratio to [0, 1]
    violation_ratio = max(0.0, min(1.0, violation_ratio))
    
    if violation_ratio == 0.0:
        # Perfect: green
        return '#00FF00'  # Green
    elif violation_ratio >= 0.05:
        # 5% or more violation: full red
        return '#FF0000'  # Red
    else:
        # Interpolate from green to red between 0% and 5%
        # Map violation_ratio from [0, 0.05] to [0, 1] for color interpolation
        normalized_ratio = violation_ratio / 0.05
        
        # Interpolate from green to red
        # Green component decreases as violation increases
        # Red component increases as violation increases
        green = int((1.0 - normalized_ratio) * 255)
        red = int(normalized_ratio * 255)
        blue = 0
        # Format as hex color string
        return f'#{red:02X}{green:02X}{blue:02X}'


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
    
    # Use Graphviz dot layout only (best for minimizing crossings)
    # dot uses a hierarchical layout algorithm specifically designed to minimize crossings
    pos = None
    layout_method = None
    layout_error = None
    
    try:
        # Use Graphviz with increased spacing for better readability
        # Try with increased node separation based on graph size
        try:
            # Use AGraph with custom attributes for better spacing
            A = nx.nx_agraph.to_agraph(G)
            
            # Scale spacing based on number of nodes - MUCH more spacing for large graphs
            num_nodes_temp = len(G.nodes())
            if num_nodes_temp <= 20:
                nodesep, ranksep = '1.0', '1.5'
            elif num_nodes_temp <= 50:
                nodesep, ranksep = '2.0', '2.5'  # Increased spacing
            elif num_nodes_temp <= 100:
                nodesep, ranksep = '3.0', '3.5'  # Much more spacing
            else:
                nodesep, ranksep = '4.0', '4.5'  # Very large spacing for large graphs
            
            # Increase node separation - more space between nodes
            A.graph_attr['nodesep'] = nodesep  # Horizontal spacing between nodes
            A.graph_attr['ranksep'] = ranksep  # Vertical spacing between ranks
            A.graph_attr['dpi'] = '75'  # Resolution for Graphviz layout
            
            # Layout with dot
            A.layout(prog='dot')
            
            # Extract positions from Graphviz layout
            pos = {}
            for node in G.nodes():
                n = A.get_node(node)
                if 'pos' in n.attr:
                    coords = n.attr['pos'].split(',')
                    pos[node] = (float(coords[0]), float(coords[1]))
            
            if not pos:
                raise Exception("Failed to extract positions")
            
            layout_method = "Graphviz dot (minimizes crossings, increased spacing)"
        except Exception as layout_err:
            # Fallback to standard layout if custom attributes fail
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            layout_method = "Graphviz dot (minimizes crossings)"
    except Exception as e:
        layout_error = str(e)
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
            layout_method = "Graphviz dot via pydot (minimizes crossings)"
            layout_error = None
        except Exception as e2:
            layout_error = f"Graphviz not available: {str(e2)}"
            # No fallback - raise error
            raise Exception(f"Graphviz dot layout is required but not available. Error: {str(e2)}")
    
    # Calculate scaling factors based on number of nodes
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    
    # Scale figure size based on number of nodes - MUCH larger for big graphs
    # Base size: 14x10 for small graphs (< 20 nodes)
    # Scale aggressively for larger graphs to ensure readability
    if num_nodes <= 20:
        fig_width, fig_height = 14, 10
    elif num_nodes <= 50:
        # Scale more aggressively
        scale_factor = (num_nodes / 20) ** 0.6
        fig_width = 14 * scale_factor * 2.0  # Much larger multiplier
        fig_height = 10 * scale_factor * 2.0
    elif num_nodes <= 100:
        # Even more aggressive scaling
        scale_factor = (num_nodes / 20) ** 0.5
        fig_width = 14 * scale_factor * 3.0
        fig_height = 10 * scale_factor * 3.0
    else:
        # For very large graphs, scale even more
        scale_factor = (num_nodes / 20) ** 0.45
        fig_width = 14 * scale_factor * 4.0  # Can get very large
        fig_height = 10 * scale_factor * 4.0
    
    # Ensure minimum readable sizes
    fig_width = max(fig_width, 20)  # At least 20 inches wide
    fig_height = max(fig_height, 14)  # At least 14 inches tall
    
    # Cap figure size - now with higher pixel limit, we can allow larger figures
    # Maximum figure size: 60x50 inches for very large graphs (with high DPI support)
    fig_width = min(fig_width, 60)
    fig_height = min(fig_height, 50)
    
    # Scale node size - prioritize readability over fitting everything
    # Keep nodes MUCH larger for large graphs to ensure readability
    if num_nodes <= 20:
        node_size = 3000
    elif num_nodes <= 50:
        node_size = max(3000 * (20 / num_nodes) ** 0.3, 2000)  # Much larger minimum
    elif num_nodes <= 100:
        node_size = max(3000 * (20 / num_nodes) ** 0.25, 1800)  # Keep very large
    else:
        node_size = max(3000 * (20 / num_nodes) ** 0.2, 1500)  # Still very readable
    
    # Scale font sizes - AGGRESSIVELY prioritize readability
    # Keep fonts MUCH larger, especially for large graphs
    if num_nodes <= 20:
        node_font_size = 10
        edge_font_size = 14
    elif num_nodes <= 50:
        node_font_size = max(10 * (20 / num_nodes) ** 0.15, 10)  # Don't shrink much
        edge_font_size = max(14 * (20 / num_nodes) ** 0.15, 14)
    elif num_nodes <= 100:
        node_font_size = max(10 * (20 / num_nodes) ** 0.1, 9)  # Keep large
        edge_font_size = max(14 * (20 / num_nodes) ** 0.1, 13)
    else:
        node_font_size = max(10 * (20 / num_nodes) ** 0.05, 8)  # Minimum but still readable
        edge_font_size = max(14 * (20 / num_nodes) ** 0.05, 12)
    
    # Scale arrow and edge sizes
    if num_nodes <= 20:
        arrow_size = 20
        edge_width = 2.0
        node_linewidth = 2
    elif num_nodes <= 50:
        arrow_size = max(20 * (20 / num_nodes) ** 0.3, 12)
        edge_width = max(2.0 * (20 / num_nodes) ** 0.3, 1.0)
        node_linewidth = max(2 * (20 / num_nodes) ** 0.3, 1)
    else:
        arrow_size = max(20 * (20 / num_nodes) ** 0.4, 10)
        edge_width = max(2.0 * (20 / num_nodes) ** 0.4, 0.8)
        node_linewidth = max(2 * (20 / num_nodes) ** 0.4, 0.8)
    
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
        missing_edges = []
        found_edges = []
        for edge in G.edges():
            # Get violation ratio for this edge
            # NetworkX edges are tuples, so edge should match directly
            violation_ratio = violation_ratios.get(edge, None)
            
            # If not found, the edge might not be in violation_ratios
            if violation_ratio is None:
                violation_ratio = 0.0
                missing_edges.append(edge)
            else:
                found_edges.append((edge, violation_ratio))
            
            color = _violation_ratio_to_color(violation_ratio)
            edge_colors.append(color)
            
            # Debug: Log first few edges with non-zero violation ratios
            if violation_ratio > 0.0 and len([e for e in found_edges if e[1] > 0.0]) <= 3:
                print(f"Debug: Edge {edge} has violation_ratio={violation_ratio:.4f}, color={color}")
        
        # Debug: Print some info about edge matching (only if there are issues)
        if missing_edges and len(missing_edges) < len(G.edges()):
            # Only show debug if some edges are missing but not all
            print(f"Debug: {len(missing_edges)} edges not found in violation_ratios out of {len(G.edges())} total edges")
            if found_edges:
                print(f"Debug: Sample found edges with ratios: {found_edges[:3]}")
    else:
        # Default to gray if no violation ratios provided
        edge_colors = 'gray'
    
    # Draw edges with arrows (straight edges for hierarchical layout)
    # Use edge-specific colors if violation ratios are available
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, 
                          arrows=True, arrowsize=arrow_size, arrowstyle='->', 
                          width=edge_width, alpha=0.7)
    
    # Draw edge labels with subset symbol (only if not too many edges)
    # For very dense graphs, skip edge labels to reduce clutter
    if num_edges <= 100:
        edge_labels = {(u, v): '⊂' for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=edge_font_size, 
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Draw node labels with scaled font size
    # Wrap labels at 16 characters per line to prevent overlap and improve readability
    # Adjust label positioning to avoid overlap, especially for nodes on same level
    def wrap_label(text, max_chars=16):
        """Wrap text at max_chars per line, breaking at word boundaries when possible."""
        if len(text) <= max_chars:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Calculate length if we add this word to current line
            test_line = ' '.join(current_line + [word])
            
            if len(test_line) <= max_chars:
                # Word fits on current line
                current_line.append(word)
            else:
                # Word doesn't fit, start a new line
                if current_line:
                    # Save current line and start new one
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word itself is longer than max_chars, break it
                    while len(word) > max_chars:
                        lines.append(word[:max_chars])
                        word = word[max_chars:]
                    current_line = [word] if word else []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    # Create wrapped labels dictionary
    wrapped_labels = {node: wrap_label(node, max_chars=16) for node in G.nodes()}
    
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
    
    # Adjust DPI based on graph size for better quality
    # Larger graphs need higher DPI to maintain readability
    # PIL/Pillow limit has been increased to 2 billion pixels
    
    # Calculate desired DPI based on graph size
    if num_nodes <= 20:
        desired_dpi = 150
    elif num_nodes <= 50:
        desired_dpi = 200
    elif num_nodes <= 100:
        desired_dpi = 300
    elif num_nodes <= 200:
        desired_dpi = 500
    elif num_nodes <= 500:
        desired_dpi = 750
    else:
        desired_dpi = 1000  # Maximum DPI for extremely large graphs
    
    # Calculate maximum safe DPI based on figure size
    # We've increased PIL's limit to 2 billion pixels, but still use a reasonable cap
    # Formula: pixels = (width_inches * dpi) * (height_inches * dpi) = width * height * dpi^2
    # So: dpi^2 = max_pixels / (width * height)
    # Use a reasonable limit (250 million pixels) to allow high-DPI images without being excessive
    max_pixels = 250000000  # 250 million pixels - reasonable limit
    max_safe_dpi = int((max_pixels / (fig_width * fig_height)) ** 0.5)
    
    # Use the smaller of desired DPI and safe DPI
    dpi = min(desired_dpi, max_safe_dpi)
    
    # Ensure minimum DPI for quality
    if num_nodes <= 50:
        min_dpi = 150
    elif num_nodes <= 100:
        min_dpi = 200
    else:
        min_dpi = 250  # Minimum for very large graphs
    
    dpi = max(dpi, min_dpi)
    
    # Store layout info for debugging (will be returned or can be displayed)
    # The layout_method and layout_error are available if needed for display
    
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
    
    # Debug: Show violation ratios if available
    if violation_ratios:
        non_zero_ratios = {k: v for k, v in violation_ratios.items() if v > 0.0}
        if non_zero_ratios:
            st.write(f"🔍 Debug: Found {len(non_zero_ratios)} relationship(s) with violations: {non_zero_ratios}")
        else:
            st.write(f"🔍 Debug: All {len(violation_ratios)} relationships have 0% violation ratio")
    else:
        st.write("🔍 Debug: No violation ratios found in session state")
    
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
        processed_violation_ratios = {}
        missing_ratios = []
        for edge in processed_relationships:
            # Try to get violation ratio - edges should be tuples
            violation_ratio = violation_ratios.get(edge, None)
            if violation_ratio is None:
                # Edge not found in violation_ratios - default to 0.0
                violation_ratio = 0.0
                missing_ratios.append(edge)
            processed_violation_ratios[edge] = violation_ratio
        
        # Debug: Show if any edges are missing violation ratios
        if missing_ratios and violation_ratios:
            st.write(f"⚠️ Debug: {len(missing_ratios)} edge(s) not found in violation_ratios: {missing_ratios[:3]}")
            st.write(f"Available violation_ratio keys: {list(violation_ratios.keys())[:3]}")
        
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
            
            if removed_edges:
                with st.expander("🔍 Removed Transitive Edges"):
                    st.write("The following edges were removed because they are implied by other paths:")
                    for u, v in sorted(removed_edges):
                        # Find the path that implies this edge
                        G_temp = G_reduced.copy()
                        if nx.has_path(G_temp, u, v):
                            try:
                                path = nx.shortest_path(G_temp, u, v)
                                path_str = " → ".join(path)
                                st.write(f"- **{u} → {v}** (implied by: {path_str})")
                            except Exception as e:
                                st.write(f"- **{u} → {v}** (implied by another path)")
                        else:
                            st.warning(f"- **{u} → {v}** ⚠️ (WARNING: No path found in reduced graph - this edge may have been incorrectly removed)")
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
        
        # Show all edges for debugging
        with st.expander("📋 All Relationships"):
            for u, v in sorted(processed_relationships):
                st.write(f"- {u} → {v}")
    
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

