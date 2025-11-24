import streamlit as st
import pandas as pd
import time
import random


def _load_csv_file(uploaded_file):
    """Load CSV file with multiple encoding attempts."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)  # Reset file pointer
            return pd.read_csv(uploaded_file, encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # If it's not an encoding error, re-raise it
            if 'codec' not in str(e).lower() and 'decode' not in str(e).lower():
                raise
    
    raise ValueError("Could not decode the file with any of the attempted encodings. Please ensure the file is a valid CSV.")


def _load_xlsx_file(uploaded_file):
    """Load XLSX or XLS file."""
    try:
        uploaded_file.seek(0)  # Reset file pointer
        file_name = uploaded_file.name.lower()
        
        # Use openpyxl for .xlsx files, auto-detect for .xls files
        if file_name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl'), None
        else:  # .xls files - let pandas auto-detect the engine
            return pd.read_excel(uploaded_file), None
    except Exception as e:
        error_msg = str(e)
        if 'openpyxl' in error_msg.lower() or 'no engine' in error_msg.lower():
            raise ValueError(f"Could not load the Excel file. Please ensure openpyxl is installed (pip install openpyxl). For .xls files, xlrd may also be needed. Error: {error_msg}")
        raise ValueError(f"Could not load the Excel file: {error_msg}. Please ensure the file is a valid Excel file.")


def _load_file(uploaded_file):
    """Load file (CSV or XLSX) based on file extension."""
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        return _load_xlsx_file(uploaded_file)
    elif file_name.endswith('.csv'):
        return _load_csv_file(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type. Please upload a CSV or XLSX file.")


def _add_message(message):
    """Add a message to the shared messages log."""
    if 'shared_messages' not in st.session_state:
        st.session_state.shared_messages = []
    st.session_state.shared_messages.append(message)


def _remove_duplicate_rows(df):
    """Remove rows that have identical values in all columns."""
    rows_before = len(df)
    df_cleaned = df.drop_duplicates()
    rows_after = len(df_cleaned)
    duplicates_removed = rows_before - rows_after
    return df_cleaned, duplicates_removed


def _remove_trailing_leading_spaces(df):
    """Remove trailing and leading spaces from string columns."""
    leading_spaces_count = 0
    trailing_spaces_count = 0
    na_replacements_count = 0
    
    string_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    for col in string_columns:
        # Count values with leading and trailing spaces before stripping
        mask_notna = df[col].notna()
        if mask_notna.any():
            # Check if non-NA values have leading spaces (start with whitespace)
            has_leading_spaces = df[col][mask_notna].astype(str).str.match(r'^\s+.+', na=False)
            leading_spaces_count += has_leading_spaces.sum()
            # Check if non-NA values have trailing spaces (end with whitespace)
            has_trailing_spaces = df[col][mask_notna].astype(str).str.match(r'.+\s+$', na=False)
            trailing_spaces_count += has_trailing_spaces.sum()
        
        # Remove leading and trailing spaces
        df[col] = df[col].str.strip()
        
        # Count values that will be replaced by NA (space-only or empty)
        before_na_replace = df[col].copy()
        # Replace values that contain spaces only (whitespace-only) with NA
        df[col] = df[col].replace(r'^\s+$', pd.NA, regex=True)
        # Also replace empty strings with NA
        df[col] = df[col].replace('', pd.NA)
        # Count how many values were replaced by NA
        na_replacements_count += (before_na_replace.notna() & df[col].isna()).sum()
    
    return df, leading_spaces_count, trailing_spaces_count, na_replacements_count


def _remove_useless_columns(df):
    """Remove columns that are >98% empty or >98% same value (same logic as import_data)."""
    removed_columns = []
    total_rows = len(df)
    threshold = 0.98
    columns_to_check = [col for col in df.columns if col != 'initial_id']
    
    for col in columns_to_check:
        # Check if column is >98% empty
        null_percentage = df[col].isna().sum() / total_rows if total_rows > 0 else 0
        if null_percentage > threshold:
            removed_columns.append({
                'column': col,
                'reason': f'{null_percentage*100:.2f}% empty'
            })
            continue
        
        # Check if column has the same value >98% of the time
        value_counts = df[col].value_counts(normalize=True, dropna=True)
        if len(value_counts) > 0:
            most_common_percentage = value_counts.iloc[0]
            if most_common_percentage > threshold:
                removed_columns.append({
                    'column': col,
                    'reason': f'{most_common_percentage*100:.2f}% same value ({value_counts.index[0]})'
                })
    
    # Remove identified columns
    if removed_columns:
        cols_to_remove = [col_info['column'] for col_info in removed_columns]
        df = df.drop(columns=cols_to_remove)
    
    return df, removed_columns


def render():
    """Render the File merge tab."""
    st.header("File merge")
    st.write("Upload multiple CSV or XLSX files to merge them together.")
    
    # Initialize uploader key with timestamp for unique reset
    if 'file_merge_uploader_key' not in st.session_state:
        st.session_state.file_merge_uploader_key = f"file_merge_uploader_{time.time()}_{random.randint(1000, 9999)}"
    
    # Multiple file uploader (use unique key that changes on reset)
    uploaded_files = st.file_uploader(
        "Upload files to merge",
        type=['csv', 'xlsx', 'xls'],
        help="Select one or more CSV or Excel files to merge",
        key=st.session_state.file_merge_uploader_key,
        accept_multiple_files=True
    )
    
    # Load and store uploaded files
    if uploaded_files and len(uploaded_files) > 0:
        # Initialize session state for file data if needed
        if 'file_merge_data' not in st.session_state:
            st.session_state.file_merge_data = {}
        
        # Load files
        file_data = {}
        load_errors = []
        
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.file_merge_data:
                try:
                    df, encoding = _load_file(uploaded_file)
                    st.session_state.file_merge_data[file_id] = {
                        'name': uploaded_file.name,
                        'df': df,
                        'shape': df.shape
                    }
                except Exception as e:
                    load_errors.append(f"{uploaded_file.name}: {str(e)}")
        
        # Show loading errors if any
        if load_errors:
            for error in load_errors:
                st.error(f"❌ {error}")
        
        # Display summary of uploaded files
        if st.session_state.file_merge_data:
            st.subheader("Uploaded Files Summary")
            summary_data = []
            for file_id, file_info in st.session_state.file_merge_data.items():
                summary_data.append({
                    'File Name': file_info['name'],
                    'Rows': file_info['shape'][0],
                    'Columns': file_info['shape'][1]
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Merge options
            st.divider()
            st.subheader("Merge Options")
            
            remove_duplicates = st.checkbox(
                "Remove duplicate rows",
                help="Remove rows that have identical values in all columns",
                key="file_merge_remove_duplicates"
            )
            
            remove_spaces = st.checkbox(
                "Remove trailing and leading spaces",
                help="Remove leading and trailing whitespace from string values",
                key="file_merge_remove_spaces"
            )
            
            remove_useless_cols = st.checkbox(
                "Remove useless columns",
                help="Remove columns that are >98% empty or >98% same value",
                key="file_merge_remove_useless_cols"
            )
            
            # Clear data button
            if st.button("Clear File Merge Data", type="secondary", key="file_merge_clear_button"):
                # Clear all file merge related data
                if 'file_merge_data' in st.session_state:
                    del st.session_state.file_merge_data
                if 'initial_df' in st.session_state:
                    del st.session_state.initial_df
                if 'processed_df' in st.session_state:
                    del st.session_state.processed_df
                if 'shared_messages' in st.session_state:
                    del st.session_state.shared_messages
                # Generate new unique key for uploader (forces complete reset)
                old_key = st.session_state.file_merge_uploader_key
                st.session_state.file_merge_uploader_key = f"file_merge_uploader_{time.time()}_{random.randint(1000, 9999)}"
                # Clear old uploader widget state
                if old_key in st.session_state:
                    del st.session_state[old_key]
                # Clear any other uploader-related keys
                keys_to_remove = [k for k in list(st.session_state.keys()) if 'file_merge_uploader' in k and k != st.session_state.file_merge_uploader_key]
                for key in keys_to_remove:
                    try:
                        del st.session_state[key]
                    except:
                        pass
                st.success("🧹 Cleared file merge data from memory.")
                st.rerun()
            
            # Merge button
            if st.button("Merge Files", type="primary", key="file_merge_button"):
                with st.spinner("Merging files..."):
                    try:
                        # Concatenate all dataframes vertically
                        dataframes = [file_info['df'] for file_info in st.session_state.file_merge_data.values()]
                        merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
                        
                        # Apply cleaning options to merged result
                        messages = []
                        
                        # Remove duplicate rows
                        if remove_duplicates:
                            merged_df, duplicates_removed = _remove_duplicate_rows(merged_df)
                            if duplicates_removed > 0:
                                messages.append(f"🗑️ **Removed {duplicates_removed} duplicate row(s)**")
                        
                        # Remove trailing and leading spaces
                        if remove_spaces:
                            merged_df, leading_count, trailing_count, na_count = _remove_trailing_leading_spaces(merged_df)
                            if leading_count > 0:
                                messages.append(f"🧹 **Removed leading spaces from {leading_count} value(s)**")
                            if trailing_count > 0:
                                messages.append(f"🧹 **Removed trailing spaces from {trailing_count} value(s)**")
                            if na_count > 0:
                                messages.append(f"🔄 **Replaced {na_count} space-only value(s) with NA**")
                        
                        # Remove useless columns
                        if remove_useless_cols:
                            merged_df, removed_columns = _remove_useless_columns(merged_df)
                            if removed_columns:
                                col_details = [f"{col_info['column']} ({col_info['reason']})" for col_info in removed_columns]
                                messages.append(f"🗑️ **Removed {len(removed_columns)} useless column(s)**: {', '.join(col_details)}")
                        
                        # Add initial_id column
                        merged_df['initial_id'] = range(1, len(merged_df) + 1)
                        
                        # Store merged dataframe
                        st.session_state.initial_df = merged_df
                        st.session_state.processed_df = merged_df
                        
                        # Add messages
                        for message in messages:
                            _add_message(message)
                        
                        # Add merge message
                        file_names = [info['name'] for info in st.session_state.file_merge_data.values()]
                        _add_message(f"🔗 **Merged {len(file_names)} file(s)**: {', '.join(file_names)}")
                        
                        st.success(f"✅ Files merged successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error merging files: {str(e)}")
            
            # Preview merged file (before applying options)
            st.divider()
            st.subheader("Merged File Preview")
            
            try:
                # Concatenate all dataframes for preview
                dataframes = [file_info['df'] for file_info in st.session_state.file_merge_data.values()]
                preview_df = pd.concat(dataframes, axis=0, ignore_index=True)
                
                st.write(f"**Merged file size:** {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")
                st.dataframe(preview_df.head(20), use_container_width=True)
                if len(preview_df) > 20:
                    st.caption(f"Showing first 20 rows of {len(preview_df)} total rows")

                # Download merged data (applies to latest merged result if available)
                if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
                    csv_data = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download merged data as CSV",
                        data=csv_data,
                        file_name="merged_data.csv",
                        mime="text/csv",
                        key="file_merge_download_button"
                    )
                else:
                    st.caption("Merge the files first to enable CSV download.")
            except Exception as e:
                st.error(f"❌ Error creating preview: {str(e)}")
    else:
        st.info("👆 Please upload one or more CSV or XLSX files to continue")
