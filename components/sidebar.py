import streamlit as st


def render():
    """Render the retractable sidebar with all persistent messages."""
    with st.sidebar:
        st.header("📋 Activity Log")
        
        # Collapsible section
        with st.expander("View all actions", expanded=True):
            if st.session_state.shared_messages:
                for msg in st.session_state.shared_messages:
                    st.write(msg)
            else:
                st.info("No actions taken yet.")
            
            # Clear button
            if st.session_state.shared_messages:
                if st.button("Clear log", type="secondary", use_container_width=True):
                    st.session_state.shared_messages = []
                    st.rerun()

