import streamlit as st
import pandas as pd
from openai import AzureOpenAI
import json
import re
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_NAME,
        AZURE_OPENAI_API_VERSION
    )
except ImportError:
    # Set defaults if config file doesn't exist
    AZURE_OPENAI_API_KEY = None
    AZURE_OPENAI_ENDPOINT = None
    AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-5-nano"
    AZURE_OPENAI_API_VERSION = "2025-01-01-preview"


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


def _is_categorical(series):
    """Check if a series is categorical (object type with limited unique values)."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        # Consider categorical if unique values are less than 50% of total non-null values
        non_null_count = series.notna().sum()
        if non_null_count > 0:
            unique_count = series.nunique()
            return unique_count < (non_null_count * 0.5) or unique_count < 100
    return False


def _get_unique_values(series):
    """Get unique values from a series, excluding null/empty values."""
    unique_vals = series.dropna().unique()
    # Filter out empty strings
    unique_vals = [val for val in unique_vals if str(val).strip() != '']
    return sorted(unique_vals)


def _batch_values(values, batch_size=100):
    """Split values into batches of specified size, ordered alphabetically.
    Batches with less than 10 items are merged with the next batch if possible."""
    sorted_values = sorted(values)
    batches = []
    for i in range(0, len(sorted_values), batch_size):
        batches.append(sorted_values[i:i + batch_size])
    
    # Merge small batches (< 10 items) with the next batch if they don't exceed batch_size
    # Use a more aggressive merging strategy: keep merging small batches forward
    merged_batches = []
    i = 0
    while i < len(batches):
        current_batch = batches[i]
        
        # If current batch has less than 10 items, try to merge with following batches
        if len(current_batch) < 10:
            # Try to merge with next batches until we reach batch_size or run out of batches
            merged_batch = current_batch.copy()
            j = i + 1
            
            while j < len(batches) and len(merged_batch) + len(batches[j]) <= batch_size:
                merged_batch.extend(batches[j])
                j += 1
                # Stop if we've merged enough (at least 10 items or reached a reasonable size)
                if len(merged_batch) >= 10:
                    break
            
            merged_batches.append(merged_batch)
            i = j  # Skip all batches we merged
        else:
            # Batch is already >= 10 items, add as-is
            merged_batches.append(current_batch)
            i += 1
    
    return merged_batches


def _create_azure_client():
    """Create Azure OpenAI client."""
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )


def _call_azure_openai(values_batch, column_name, data_type, status_callback=None):
    """Call Azure OpenAI API to simplify values."""
    try:
        if status_callback:
            status_callback(f"📡 Connecting to Azure OpenAI endpoint: {AZURE_OPENAI_ENDPOINT}")
        
        client = _create_azure_client()
        
        if status_callback:
            status_callback(f"📝 Preparing prompt for {len(values_batch)} values...")
        
        # Create prompt
        values_list = "\n".join([f"- {val}" for val in values_batch])
        prompt = f"""You are analyzing a categorical column named "{column_name}" with data type "{data_type}".

The column contains the following unique values:
{values_list}

Your task is to identify similar values that should be merged together. For example:
- "Company inc." and "Company" should be merged
- "New York" and "NY" might be merged if they refer to the same thing
- "USA" and "United States" should be merged

Please provide a JSON mapping ONLY for values that should be merged/simplified:
- Keys are the original values that should be changed
- Values are the simplified/merged values they should become
- DO NOT include values that should remain unchanged

Return ONLY a valid JSON object with this structure:
{{
  "original_value_to_change": "simplified_value",
  "another_value_to_change": "simplified_value",
  ...
}}

Important rules:
1. ONLY include values that should be merged or simplified - exclude unchanged values
2. Similar values should map to the same simplified value (the canonical form)
3. Choose the most representative/canonical form as the simplified value
4. Preserve the original case and formatting unless merging requires change
5. Return valid JSON only, no additional text or explanation
6. If no values need merging, return an empty JSON object: {{}}"""

        if status_callback:
            status_callback(f"🚀 Sending API request to model '{AZURE_OPENAI_DEPLOYMENT_NAME}'...")
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a data cleaning assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=20000
        )
        
        if status_callback:
            status_callback(f"✅ Received response from API (tokens used: {response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 'N/A'})")
            status_callback(f"📥 Processing response content...")
        
        # Get response content
        if not response.choices or len(response.choices) == 0:
            raise ValueError("No response choices returned from API")
        
        # Debug: Check response structure
        choice = response.choices[0]
        finish_reason = getattr(choice, 'finish_reason', None)
        
        if status_callback:
            status_callback(f"🔍 Response finish_reason: {finish_reason}")
            if finish_reason == 'length':
                status_callback(f"⚠️ Warning: Response was truncated (hit token limit)")
            elif finish_reason == 'content_filter':
                status_callback(f"⚠️ Warning: Response was filtered by content filter")
            elif finish_reason == 'stop':
                status_callback(f"✅ Response completed normally")
        
        # Try different ways to get content
        if hasattr(choice.message, 'content'):
            content = choice.message.content
        elif hasattr(choice, 'message') and hasattr(choice.message, 'text'):
            content = choice.message.text
        else:
            # Try to get content directly
            content = str(choice.message) if choice.message else None
        
        if content is None:
            if status_callback:
                status_callback(f"❌ Error: Content is None")
                status_callback(f"📄 Finish reason: {finish_reason}")
                status_callback(f"📄 Message object type: {type(choice.message)}")
            raise ValueError(f"Response content is None. Finish reason: {finish_reason}")
        
        content = str(content).strip()
        
        if not content:
            if status_callback:
                status_callback(f"❌ Error: Empty response content")
                status_callback(f"📄 Finish reason: {finish_reason}")
                if finish_reason == 'length':
                    status_callback(f"💡 Response was truncated - this batch may be too large")
            # If it was truncated, raise error to retry with smaller batch
            if finish_reason == 'length':
                raise ValueError("Response was truncated due to token limit. The batch size will be automatically reduced.")
            return {}
        
        if status_callback:
            status_callback(f"📄 Raw response length: {len(content)} characters")
            # Show first 200 chars for debugging
            preview = content[:200] + "..." if len(content) > 200 else content
            status_callback(f"📄 Response preview: {preview}")
        
        # Extract JSON from response (in case there's extra text)
        # Try to find complete JSON objects first
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            extracted_content = json_match.group(0)
            # Try to parse the extracted JSON
            try:
                json.loads(extracted_content)
                content = extracted_content
                if status_callback:
                    status_callback(f"🔍 Extracted complete JSON from response")
            except json.JSONDecodeError:
                # If extracted JSON is incomplete, try to fix it
                if status_callback:
                    status_callback(f"⚠️ Extracted JSON appears incomplete, attempting to fix...")
                # Try to find the last complete key-value pair
                # Remove trailing incomplete parts
                content = extracted_content
                # Try to close any unclosed braces
                open_braces = content.count('{')
                close_braces = content.count('}')
                if open_braces > close_braces:
                    content = content.rstrip().rstrip(',') + '\n' + '}' * (open_braces - close_braces)
                    if status_callback:
                        status_callback(f"🔧 Attempted to fix incomplete JSON by closing braces")
        else:
            if status_callback:
                status_callback(f"⚠️ Warning: No JSON object found in response, trying to parse entire content")
        
        # Try to parse JSON
        try:
            mapping = json.loads(content)
        except json.JSONDecodeError as json_err:
            if status_callback:
                status_callback(f"❌ JSON parsing error: {str(json_err)}")
                status_callback(f"📄 Content that failed to parse (first 500 chars): {content[:500]}")
                status_callback(f"📄 Content length: {len(content)} characters")
            # If it was truncated, provide helpful error
            if finish_reason == 'length':
                raise ValueError("Response was truncated and JSON is incomplete. Reducing batch size...")
            raise ValueError(f"Failed to parse JSON response: {str(json_err)}\nContent preview: {content[:500]}")
        
        if status_callback:
            merged_count = len(mapping)
            status_callback(f"✨ Parsed {merged_count} value(s) to merge from response")
        
        return mapping
        
    except Exception as e:
        if status_callback:
            status_callback(f"❌ Error: {str(e)}")
        st.error(f"Error calling Azure OpenAI API: {str(e)}")
        return None


def _merge_mappings(mappings_list):
    """Merge multiple mapping dictionaries into one."""
    merged = {}
    for mapping in mappings_list:
        if mapping:
            merged.update(mapping)
    return merged


def _generate_simplification_mapping(unique_values, column_name, data_type, test_mode=False):
    """Generate simplification mapping for all unique values."""
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Processing Status")
        progress_bar = st.progress(0)
        status_container = st.empty()
        log_container = st.empty()
        
        # Initialize log
        log_messages = []
        
        def update_status(message):
            """Update status and add to log."""
            log_messages.append(message)
            # Show last 10 messages in a code block for better readability
            recent_logs = "\n".join([f"{msg}" for msg in log_messages[-10:]])
            log_container.code(recent_logs, language=None)
        
        # Check if we need to batch (reduced to 100 to avoid token limits)
        if len(unique_values) > 100:
            batches = _batch_values(unique_values, batch_size=100)
            update_status(f"📊 Total: {len(unique_values)} unique values")
            update_status(f"📦 Splitting into {len(batches)} batches of ~100 values each")
            
            # Apply test mode limit if enabled
            if test_mode:
                max_batches_to_process = 5
                batches_to_process = batches[:max_batches_to_process]
                if len(batches) > max_batches_to_process:
                    update_status(f"🧪 TESTING MODE: Processing only first {max_batches_to_process} batches (out of {len(batches)} total)")
            else:
                batches_to_process = batches
            
            all_mappings = []
            
            for i, batch in enumerate(batches_to_process):
                batch_num = i + 1
                update_status(f"\n{'='*50}")
                update_status(f"📦 Processing Batch {batch_num}/{len(batches_to_process)} ({len(batch)} values)")
                update_status(f"{'='*50}")
                
                # Update progress bar
                progress_bar.progress(i / len(batches_to_process))
                
                # Show batch range
                if len(batch) > 0:
                    batch_start = batch[0][:30] + "..." if len(batch[0]) > 30 else batch[0]
                    batch_end = batch[-1][:30] + "..." if len(batch[-1]) > 30 else batch[-1]
                    update_status(f"Range: '{batch_start}' → '{batch_end}'")
                
                # Call API with status updates
                mapping = _call_azure_openai(batch, column_name, data_type, status_callback=update_status)
                
                if mapping:
                    merged_count = len(mapping)
                    all_mappings.append(mapping)
                    update_status(f"✅ Batch {batch_num} complete: {merged_count} value(s) identified for merging")
                else:
                    update_status(f"⚠️ Batch {batch_num} failed: No mapping returned")
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(batches_to_process))
            
            progress_bar.progress(1.0)
            update_status(f"\n{'='*50}")
            update_status(f"🔄 Merging all batch results...")
            
            if not all_mappings:
                update_status("❌ No mappings generated from any batch")
                progress_bar.empty()
                status_container.empty()
                log_container.empty()
                return None
            
            final_mapping = _merge_mappings(all_mappings)
            total_merged = len(final_mapping)
            update_status(f"✅ All batches processed successfully!")
            update_status(f"📊 Total values to merge: {total_merged}")
            
            progress_bar.empty()
            status_container.empty()
            
        else:
            update_status(f"📊 Processing {len(unique_values)} unique values in a single batch")
            progress_bar.progress(0.3)
            
            mapping = _call_azure_openai(unique_values, column_name, data_type, status_callback=update_status)
            
            progress_bar.progress(1.0)
            
            if mapping:
                merged_count = len(mapping)
                update_status(f"✅ Processing complete: {merged_count} value(s) identified for merging")
            else:
                update_status("❌ No mapping generated")
            
            progress_bar.empty()
            status_container.empty()
            
            return mapping
        
        return final_mapping


def _create_mapping_dataframe(mapping):
    """Create a DataFrame from the mapping for display and editing."""
    if not mapping:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {"Original Value": orig, "Simplified Value": simplified}
        for orig, simplified in mapping.items()
    ])
    
    return df.sort_values("Original Value").reset_index(drop=True)


def _apply_mapping_to_column(df, column_name, mapping_df):
    """Apply the mapping to create a new simplified column."""
    # Convert mapping dataframe to dictionary
    mapping_dict = dict(zip(mapping_df["Original Value"], mapping_df["Simplified Value"]))
    
    # Create new column
    new_column_name = f"{column_name}_simplified"
    df[new_column_name] = df[column_name].map(mapping_dict).fillna(df[column_name])
    
    return df, new_column_name


def _create_preview(df, column_name, new_column_name):
    """Create a preview showing original vs simplified values."""
    preview_df = df[[column_name, new_column_name]].copy()
    preview_df = preview_df.drop_duplicates().sort_values(column_name)
    return preview_df


# ============================================================================
# Main Render Function
# ============================================================================

def render():
    """Render the AI Simplification tab."""
    st.header("AI Simplification")
    
    if not _has_processed_data():
        st.warning("⚠️ Please import data first using the 'Import Data' tab.")
        return
    
    # Check Azure configuration
    if AZURE_OPENAI_API_KEY is None or AZURE_OPENAI_ENDPOINT is None:
        st.error("⚠️ Please configure Azure OpenAI credentials.")
        st.info("Create a `config.py` file from `config.example.py` and fill in your Azure OpenAI credentials.")
        return
    
    if AZURE_OPENAI_API_KEY == "YOUR_API_KEY_HERE" or AZURE_OPENAI_ENDPOINT == "YOUR_ENDPOINT_HERE":
        st.error("⚠️ Please configure Azure OpenAI credentials in config.py (API key and endpoint).")
        st.info("Edit `config.py` to set your Azure OpenAI credentials.")
        return
    
    df = st.session_state.processed_df
    
    # Column selection
    available_columns = _get_available_columns()
    categorical_columns = [
        col for col in available_columns 
        if _is_categorical(df[col])
    ]
    
    if not categorical_columns:
        st.warning("⚠️ No categorical columns found. Please ensure your dataset contains categorical/text columns.")
        return
    
    st.subheader("Select Column to Simplify")
    selected_column = st.selectbox(
        "Choose a categorical column:",
        options=categorical_columns,
        help="Select a column with categorical/text data that you want to simplify"
    )
    
    if selected_column:
        # Get unique values
        unique_values = _get_unique_values(df[selected_column])
        
        if not unique_values:
            st.warning(f"⚠️ Column '{selected_column}' has no valid values to simplify.")
            return
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Unique Values", len(unique_values))
        with col_info2:
            st.metric("Total Rows", len(df))
        with col_info3:
            data_type = str(df[selected_column].dtype)
            st.metric("Data Type", data_type)
        
        # Test mode checkbox
        test_mode = st.checkbox(
            "🧪 Test Mode (Process only first 5 batches)",
            value=False,
            help="Enable to process only the first 5 batches for testing purposes",
            key="test_mode_simplification"
        )
        
        # Generate simplification mapping
        if st.button("🤖 Generate Simplification Mapping", type="primary"):
            with st.spinner("Analyzing values with AI..."):
                mapping = _generate_simplification_mapping(
                    unique_values, 
                    selected_column, 
                    data_type,
                    test_mode=test_mode
                )
                
                if mapping:
                    st.session_state[f"simplification_mapping_{selected_column}"] = mapping
                    st.success("✅ Simplification mapping generated!")
                else:
                    st.error("Failed to generate mapping. Please try again.")
        
        # Display and edit mapping if available
        mapping_key = f"simplification_mapping_{selected_column}"
        if mapping_key in st.session_state:
            st.subheader("Review and Edit Mapping")
            st.info("Review the mapping below. You can edit the 'Simplified Value' column to make manual adjustments.")
            
            # Create mapping dataframe
            mapping_df = _create_mapping_dataframe(st.session_state[mapping_key])
            
            # Display editable dataframe
            edited_mapping_df = st.data_editor(
                mapping_df,
                use_container_width=True,
                num_rows="dynamic",
                key=f"mapping_editor_{selected_column}",
                column_config={
                    "Original Value": st.column_config.TextColumn(
                        "Original Value",
                        disabled=True,
                        width="medium"
                    ),
                    "Simplified Value": st.column_config.TextColumn(
                        "Simplified Value",
                        width="medium"
                    )
                }
            )
            
            # Store edited mapping
            if not edited_mapping_df.empty:
                edited_mapping = dict(zip(
                    edited_mapping_df["Original Value"],
                    edited_mapping_df["Simplified Value"]
                ))
                st.session_state[mapping_key] = edited_mapping
                
                # Show preview
                st.subheader("Preview")
                preview_df, new_column_name = _apply_mapping_to_column(
                    df.copy(),
                    selected_column,
                    edited_mapping_df
                )
                preview = _create_preview(preview_df, selected_column, new_column_name)
                
                st.dataframe(preview, use_container_width=True, hide_index=True)
                
                # Show statistics
                original_unique = len(df[selected_column].dropna().unique())
                simplified_unique = len(preview_df[new_column_name].dropna().unique())
                reduction = original_unique - simplified_unique
                reduction_pct = (reduction / original_unique * 100) if original_unique > 0 else 0
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Original Unique Values", original_unique)
                with col_stat2:
                    st.metric("Simplified Unique Values", simplified_unique)
                with col_stat3:
                    st.metric("Reduction", f"{reduction} ({reduction_pct:.1f}%)")
                
                # Apply button
                if st.button("✅ Apply Simplification", type="primary"):
                    # Apply to processed_df
                    st.session_state.processed_df, new_col_name = _apply_mapping_to_column(
                        st.session_state.processed_df.copy(),
                        selected_column,
                        edited_mapping_df
                    )
                    
                    st.success(f"✅ Created new column '{new_col_name}' with simplified values!")
                    st.info(f"💡 The original column '{selected_column}' remains unchanged. The simplified values are in '{new_col_name}'.")
                    
                    # Clear the mapping from session state to allow new operations
                    del st.session_state[mapping_key]
                    st.rerun()
