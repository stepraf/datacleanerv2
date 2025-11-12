import streamlit as st


def _has_initial_data():
    """Check if initial data is available."""
    return ('initial_df' in st.session_state and 
            st.session_state.initial_df is not None)


def render():
    """Render the Data View tab."""
    st.header("Data View")
    
    if _has_initial_data():
        df = st.session_state.initial_df
        st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data loaded. Please import a CSV file in the 'Import Data' tab.")


