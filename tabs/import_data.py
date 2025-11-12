import streamlit as st
import pandas as pd


def _has_processed_data():
    """Check if processed data is available."""
    return ('processed_df' in st.session_state and 
            st.session_state.processed_df is not None and 
            len(st.session_state.processed_df) > 0)


def _add_message(message):
    """Add a message to the shared messages log."""
    if 'shared_messages' not in st.session_state:
        st.session_state.shared_messages = []
    st.session_state.shared_messages.append(message)


def _initialize_messages():
    """Initialize or clear shared messages."""
    if 'shared_messages' not in st.session_state:
        st.session_state.shared_messages = []
    else:
        # Clear previous messages when uploading a new file
        st.session_state.shared_messages = []


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


def _clean_dataframe(df):
    """Remove empty columns and rows from dataframe."""
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    removed_columns = []
    total_rows = len(df)
    threshold = 0.98
    total_columns = len(df.columns)
    
    # Step 1: Analyze columns for removal (0-40% of progress)
    status_text.text("Analyzing columns...")
    columns_to_check = [col for col in df.columns if col != 'initial_id']
    
    for idx, col in enumerate(columns_to_check):
        # Update progress: 0-40% for column analysis
        if len(columns_to_check) > 0:
            progress = (idx + 1) / len(columns_to_check) * 0.4
            progress_bar.progress(progress)
        
        # Check if column is >98% empty
        null_percentage = df[col].isna().sum() / total_rows
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
    
    # Step 2: Remove identified columns (40-50% of progress)
    status_text.text("Removing empty columns...")
    progress_bar.progress(0.45)
    if removed_columns:
        cols_to_remove = [col_info['column'] for col_info in removed_columns]
        df = df.drop(columns=cols_to_remove)
    progress_bar.progress(0.5)
    
    # Step 3: Remove empty rows (50-60% of progress)
    status_text.text("Removing empty rows...")
    progress_bar.progress(0.55)
    non_id_columns = [col for col in df.columns if col != 'initial_id']
    if len(non_id_columns) > 0:
        empty_rows_mask = df[non_id_columns].isna().all(axis=1)
        empty_rows_count = empty_rows_mask.sum()
        df = df[~empty_rows_mask]
    else:
        empty_rows_count = 0
    progress_bar.progress(0.6)
    
    # Step 4: Remove trailing spaces and replace space-only values (60-100% of progress)
    status_text.text("Cleaning string values...")
    trailing_spaces_count = 0
    na_replacements_count = 0
    
    string_columns = [col for col in df.columns if df[col].dtype == 'object']
    for idx, col in enumerate(string_columns):
        # Update progress: 60-100% for string cleaning
        if len(string_columns) > 0:
            progress = 0.6 + (idx + 1) / len(string_columns) * 0.4
            progress_bar.progress(progress)
        
        # Count values with trailing spaces before stripping
        # Use regex to check specifically for trailing whitespace
        mask_notna = df[col].notna()
        if mask_notna.any():
            # Check if non-NA values have trailing spaces (end with whitespace)
            has_trailing_spaces = df[col][mask_notna].astype(str).str.match(r'.+\s+$', na=False)
            trailing_spaces_count += has_trailing_spaces.sum()
        
        # Remove trailing spaces
        df[col] = df[col].str.rstrip()
        
        # Count values that will be replaced by NA (space-only or empty)
        before_na_replace = df[col].copy()
        # Replace values that contain spaces only (whitespace-only) with NA
        df[col] = df[col].replace(r'^\s+$', pd.NA, regex=True)
        # Also replace empty strings with NA
        df[col] = df[col].replace('', pd.NA)
        # Count how many values were replaced by NA
        na_replacements_count += (before_na_replace.notna() & df[col].isna()).sum()
    
    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text("Cleaning complete!")
    
    return df, removed_columns, empty_rows_count, trailing_spaces_count, na_replacements_count


def _handle_file_upload(uploaded_file):
    """Handle file upload (CSV or XLSX) and processing."""
    try:
        # Load file (CSV or XLSX) based on file extension
        initial_df, encoding_used = _load_file(uploaded_file)
        
        # Add initial_id column
        initial_df['initial_id'] = range(1, len(initial_df) + 1)
        
        # Store initial dataframe
        st.session_state.initial_df = initial_df
        
        # Clean and process dataframe
        processed_df, removed_columns, empty_rows_count, trailing_spaces_count, na_replacements_count = _clean_dataframe(initial_df.copy())
        
        # Store processed dataframe and metadata
        st.session_state.processed_df = processed_df
        st.session_state.removed_columns = removed_columns
        st.session_state.removed_empty_rows_count = empty_rows_count
        st.session_state.trailing_spaces_count = trailing_spaces_count
        st.session_state.na_replacements_count = na_replacements_count
        
        # Track the file by its unique identifier (name + size)
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        st.session_state.last_uploaded_file_id = file_id
        
        # Reset state for new file
        st.session_state.manually_dropped_columns = []
        st.session_state.removed_duplicates_count = 0
        
        # Initialize/clear messages
        _initialize_messages()
        
        # Add messages for auto-removals
        messages_added = False
        if removed_columns:
            col_details = [f"{col_info['column']} ({col_info['reason']})" for col_info in removed_columns]
            _add_message(f"🗑️ **Auto-removed {len(removed_columns)} column(s)**: {', '.join(col_details)}")
            messages_added = True
        
        if empty_rows_count > 0:
            _add_message(f"🗑️ **Removed {empty_rows_count} empty row(s)**")
            messages_added = True
        
        # Add messages for trailing spaces and NA replacements
        if trailing_spaces_count > 0:
            _add_message(f"🧹 **Removed trailing spaces from {trailing_spaces_count} value(s)**")
            messages_added = True
        
        if na_replacements_count > 0:
            _add_message(f"🔄 **Replaced {na_replacements_count} space-only value(s) with NA**")
            messages_added = True
        
        st.success(f"✅ File uploaded successfully!")
        st.write(f"**Shape:** {initial_df.shape[0]} rows × {initial_df.shape[1]} columns")
        
        # Trigger rerun to update UI (show buttons, update sidebar)
        st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")


def _handle_remove_duplicates():
    """Handle removing duplicate rows."""
    if not _has_processed_data():
        return
    
    rows_before = len(st.session_state.processed_df)
    st.session_state.processed_df = st.session_state.processed_df.drop_duplicates()
    rows_after = len(st.session_state.processed_df)
    duplicates_removed = rows_before - rows_after
    
    st.session_state.removed_duplicates_count = duplicates_removed
    
    if duplicates_removed > 0:
        _add_message(f"🗑️ **Removed {duplicates_removed} duplicate row(s)**")
    
    st.rerun()


def _handle_drop_columns(selected_columns):
    """Handle dropping selected columns."""
    if not selected_columns:
        return
    
    # Drop selected columns from processed_df
    st.session_state.processed_df = st.session_state.processed_df.drop(columns=selected_columns)
    
    # Add to manually dropped columns list
    if 'manually_dropped_columns' not in st.session_state:
        st.session_state.manually_dropped_columns = []
    st.session_state.manually_dropped_columns.extend(selected_columns)
    
    # Add message
    _add_message(f"🗑️ **Manually dropped {len(selected_columns)} column(s)**: {', '.join(selected_columns)}")
    st.rerun()


def render():
    """Render the Import Data tab."""
    st.header("Import Data")
    st.write("Upload a CSV or XLSX file to get started with data cleaning.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=['csv', 'xlsx', 'xls'],
        help="Select a CSV or Excel file to upload and load into the application",
        key="file_uploader"
    )
    
    # Handle file upload - process only when file actually changes
    # Streamlit widgets with keys store their value in session_state
    # We can detect changes by comparing current file with what we've processed
    if uploaded_file is not None:
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Process if this file hasn't been processed yet (new upload or reupload)
        # On rerun after st.rerun(), last_uploaded_file_id will match, so we skip
        if 'last_uploaded_file_id' not in st.session_state or st.session_state.last_uploaded_file_id != current_file_id:
            _handle_file_upload(uploaded_file)
    else:
        # Uploader cleared - reset tracking to allow reuploading same file
        if 'last_uploaded_file_id' in st.session_state:
            st.session_state.last_uploaded_file_id = None
    
    # Data manipulation controls (only show when data is loaded)
    if _has_processed_data():
        # Remove duplicates button
        if st.button("Remove duplicates", type="primary"):
            _handle_remove_duplicates()
        
        # Column selection dropdown for manual removal
        available_columns = [
            col for col in st.session_state.processed_df.columns 
            if col not in st.session_state.get('manually_dropped_columns', [])
        ]
        
        if available_columns:
            selected_columns = st.multiselect(
                "Select columns to drop",
                options=available_columns,
                help="Select one or more columns to remove from the dataset"
            )
            
            if selected_columns:
                if st.button("Drop selected columns", type="secondary"):
                    _handle_drop_columns(selected_columns)
    
    # Display preview
    if 'initial_df' in st.session_state and st.session_state.initial_df is not None:
        st.subheader("Preview")
        st.dataframe(st.session_state.initial_df.head(10), use_container_width=True)
    elif uploaded_file is None:
        st.info("👆 Please upload a CSV or XLSX file to continue")

