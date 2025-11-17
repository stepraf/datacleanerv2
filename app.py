import streamlit as st
from tabs import import_data, one_to_one, one_to_many, one_to_n_errors, relation_graph, export_data, fill_na, covariance, apriori, ai_simplification, ai_advanced
from components import sidebar


def _initialize_session_state():
    """Initialize all session state variables."""
    if 'initial_df' not in st.session_state:
        st.session_state.initial_df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'shared_messages' not in st.session_state:
        st.session_state.shared_messages = []
    if 'manually_dropped_columns' not in st.session_state:
        st.session_state.manually_dropped_columns = []
    if 'removed_duplicates_count' not in st.session_state:
        st.session_state.removed_duplicates_count = 0


# Page configuration
st.set_page_config(
    page_title="Data Cleaner",
    page_icon="🧹",
    layout="wide"
)

# Initialize session state
_initialize_session_state()

# Render sidebar
sidebar.render()

# Main title
st.title("Data")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(["📥 Import Data", "1:1", "1:N", "1:N Errors", "Relation Graph", "Fill NA", "Covariance", "Apriori", "AI Simplification", "AI Advanced", "📤 Export Data"])

with tab1:
    import_data.render()

with tab2:
    one_to_one.render()

with tab3:
    one_to_many.render()

with tab4:
    one_to_n_errors.render()

with tab5:
    relation_graph.render()

with tab6:
    fill_na.render()

with tab7:
    covariance.render()

with tab8:
    apriori.render()

with tab9:
    ai_simplification.render()

with tab10:
    ai_advanced.render()

with tab11:
    export_data.render()

