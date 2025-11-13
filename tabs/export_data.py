import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime


def _has_processed_data():
    """Check if processed data is available."""
    return ('processed_df' in st.session_state and 
            st.session_state.processed_df is not None and 
            len(st.session_state.processed_df) > 0)


def render():
    """Render the Export Data tab."""
    st.header("Export Data")
    st.write("Export your processed data to a CSV file.")
    
    # Check if data is available
    if not _has_processed_data():
        st.info("No processed data available. Please import and process data in the 'Import Data' tab.")
        return
    
    # Display data summary
    df = st.session_state.processed_df
    st.subheader("Data Summary")
    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Display preview
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_index = st.checkbox(
            "Include index",
            value=False,
            help="Include the DataFrame index as a column in the exported CSV"
        )
    
    with col2:
        encoding = st.selectbox(
            "Encoding",
            options=['utf-8', 'latin-1', 'iso-8859-1', 'cp1252'],
            index=0,
            help="Character encoding for the CSV file"
        )
    
    # Export button
    if st.button("Export to CSV", type="primary"):
        try:
            # Convert DataFrame to CSV string
            csv_buffer = StringIO()
            df.to_csv(
                csv_buffer,
                index=include_index
            )
            csv_string = csv_buffer.getvalue()
            
            # Convert to bytes with selected encoding
            csv_bytes = csv_string.encode(encoding)
            
            # Generate filename with date and time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"processed_data_{timestamp}.csv"
            
            # Create download button
            st.download_button(
                label="Download CSV file",
                data=csv_bytes,
                file_name=file_name,
                mime="text/csv",
                type="primary"
            )
            
            st.success("✅ CSV file ready for download!")
            
        except Exception as e:
            st.error(f"❌ Error exporting data: {str(e)}")

