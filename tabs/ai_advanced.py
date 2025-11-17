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

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = "Extract product information from the following rows: [rows]. Return a simple product name for each row."

# Maximum rows per batch
MAX_ROWS_PER_BATCH = 30


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


def _create_azure_client():
    """Create Azure OpenAI client."""
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )


def _format_row_for_prompt(row, columns, include_group_value=False, group_value=None):
    """Format a row for the prompt."""
    row_data = {}
    for col in columns:
        value = row[col]
        if pd.notna(value) and str(value).strip() != '':
            row_data[col] = str(value).strip()
    
    # Include group value if requested and not already in columns
    if include_group_value and group_value is not None and group_value not in row_data.values():
        row_data['_group'] = str(group_value)
    
    return row_data


def _format_rows_batch_for_prompt(df_batch, columns, group_value=None, include_group_in_columns=False):
    """Format a batch of rows for the prompt."""
    rows_list = []
    for idx, row in df_batch.iterrows():
        row_data = _format_row_for_prompt(row, columns, include_group_in_columns, group_value)
        if row_data:  # Only include if there's at least one non-NA value
            row_data['_row_id'] = int(idx)
            rows_list.append(row_data)
    
    return rows_list


def _batch_rows_by_group(df, group_column, max_rows_per_batch=MAX_ROWS_PER_BATCH):
    """Batch rows by group value, splitting large groups if needed.
    Batches with less than 10 rows are merged with the next batch if possible."""
    batches = []
    
    # Group by the grouping column
    for group_value, group_df in df.groupby(group_column):
        group_rows = group_df.copy()
        
        # Handle null/empty group values
        if pd.isna(group_value) or str(group_value).strip() == '':
            # Process rows individually, filtering out rows where all selected columns are NA
            # (We'll filter these out when formatting for prompt)
            # Split into batches like normal groups
            num_rows = len(group_rows)
            if num_rows <= max_rows_per_batch:
                batches.append({
                    'group_value': None,
                    'rows': group_rows
                })
            else:
                # Split into multiple batches
                for i in range(0, num_rows, max_rows_per_batch):
                    batch_rows = group_rows.iloc[i:i + max_rows_per_batch]
                    batches.append({
                        'group_value': None,
                        'rows': batch_rows
                    })
            continue
        
        # Split large groups into smaller batches
        num_rows = len(group_rows)
        if num_rows <= max_rows_per_batch:
            batches.append({
                'group_value': group_value,
                'rows': group_rows
            })
        else:
            # Split into multiple batches
            for i in range(0, num_rows, max_rows_per_batch):
                batch_rows = group_rows.iloc[i:i + max_rows_per_batch]
                batches.append({
                    'group_value': group_value,
                    'rows': batch_rows
                })
    
    # Merge small batches (< 10 rows) with the next batch if they don't exceed max_rows_per_batch
    merged_batches = []
    i = 0
    while i < len(batches):
        current_batch = batches[i]
        current_rows = current_batch['rows']
        
        # If current batch has less than 10 rows, try to merge with following batches
        if len(current_rows) < 10:
            # Try to merge with next batches until we reach max_rows_per_batch or run out of batches
            merged_rows = [current_rows]
            merged_group_values = [current_batch['group_value']]
            j = i + 1
            
            while j < len(batches):
                next_batch = batches[j]
                next_rows = next_batch['rows']
                # Calculate total rows: sum of all DataFrames in merged_rows + next_rows
                total_rows = sum(len(df) for df in merged_rows) + len(next_rows)
                
                # Check if merging would exceed max_rows_per_batch
                if total_rows <= max_rows_per_batch:
                    merged_rows.append(next_rows)
                    merged_group_values.append(next_batch['group_value'])
                    j += 1
                    # Stop if we've merged enough (at least 10 rows or reached a reasonable size)
                    if total_rows >= 10:
                        break
                else:
                    break
            
            # Combine all merged rows into a single DataFrame
            if len(merged_rows) > 1:
                combined_rows = pd.concat(merged_rows, ignore_index=True)
            else:
                combined_rows = merged_rows[0]
            
            # Use the first group_value, or None if all are None
            combined_group_value = merged_group_values[0] if merged_group_values[0] is not None else None
            
            merged_batches.append({
                'group_value': combined_group_value,
                'rows': combined_rows
            })
            i = j  # Skip all batches we merged
        else:
            # Batch is already >= 10 rows, add as-is
            merged_batches.append(current_batch)
            i += 1
    
    return merged_batches


def _call_azure_openai(rows_data, prompt_template, status_callback=None):
    """Call Azure OpenAI API to extract information."""
    try:
        if status_callback:
            status_callback(f"📡 Connecting to Azure OpenAI endpoint: {AZURE_OPENAI_ENDPOINT}")
        
        client = _create_azure_client()
        
        if status_callback:
            status_callback(f"📝 Preparing prompt for {len(rows_data)} rows...")
        
        # Format rows for prompt
        rows_text = "\n".join([f"Row {row['_row_id']}: {json.dumps({k: v for k, v in row.items() if k != '_row_id'}, ensure_ascii=False)}" for row in rows_data])
        
        # Replace [rows] placeholder in prompt template
        prompt = prompt_template.replace("[rows]", rows_text)
        
        # Add JSON format instruction
        prompt += "\n\nReturn ONLY a valid JSON array with this structure:\n[\n  {\"row_id\": 1, \"product_name\": \"Product Name 1\"},\n  {\"row_id\": 2, \"product_name\": \"Product Name 2\"},\n  ...\n]\n\nImportant: Return valid JSON only, no additional text or explanation."
        
        if status_callback:
            status_callback(f"🚀 Sending API request to model '{AZURE_OPENAI_DEPLOYMENT_NAME}'...")
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Return only valid JSON arrays."},
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
        
        choice = response.choices[0]
        finish_reason = getattr(choice, 'finish_reason', None)
        
        if status_callback:
            status_callback(f"🔍 Response finish_reason: {finish_reason}")
            if finish_reason == 'length':
                status_callback(f"⚠️ Warning: Response was truncated (hit token limit)")
        
        # Get content
        if hasattr(choice.message, 'content'):
            content = choice.message.content
        else:
            content = str(choice.message) if choice.message else None
        
        if content is None:
            raise ValueError(f"Response content is None. Finish reason: {finish_reason}")
        
        content = str(content).strip()
        
        if not content:
            if finish_reason == 'length':
                raise ValueError("Response was truncated due to token limit.")
            return []
        
        if status_callback:
            status_callback(f"📄 Raw response length: {len(content)} characters")
            preview = content[:200] + "..." if len(content) > 200 else content
            status_callback(f"📄 Response preview: {preview}")
        
        # Extract JSON array from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            extracted_content = json_match.group(0)
            try:
                json.loads(extracted_content)
                content = extracted_content
                if status_callback:
                    status_callback(f"🔍 Extracted complete JSON array from response")
            except json.JSONDecodeError:
                content = extracted_content
                if status_callback:
                    status_callback(f"⚠️ Extracted JSON appears incomplete, attempting to fix...")
        else:
            if status_callback:
                status_callback(f"⚠️ Warning: No JSON array found in response, trying to parse entire content")
        
        # Parse JSON
        try:
            results = json.loads(content)
            if not isinstance(results, list):
                raise ValueError("Response is not a JSON array")
            
            if status_callback:
                status_callback(f"✨ Parsed {len(results)} result(s) from response")
                status_callback(f"\n📋 Extracted Values:")
                # Display returned values
                for result in results[:10]:  # Show first 10 results
                    row_id = result.get('row_id', 'N/A')
                    product_name = result.get('product_name', 'N/A')
                    # Truncate long product names for display
                    display_name = product_name[:50] + "..." if len(str(product_name)) > 50 else product_name
                    status_callback(f"  • Row {row_id}: {display_name}")
                if len(results) > 10:
                    status_callback(f"  ... and {len(results) - 10} more result(s)")
            
            return results
            
        except json.JSONDecodeError as json_err:
            if status_callback:
                status_callback(f"❌ JSON parsing error: {str(json_err)}")
                status_callback(f"📄 Content that failed to parse (first 500 chars): {content[:500]}")
            raise ValueError(f"Failed to parse JSON response: {str(json_err)}\nContent preview: {content[:500]}")
        
    except Exception as e:
        if status_callback:
            status_callback(f"❌ Error: {str(e)}")
        raise


def _extract_information_for_batches(batches, columns, prompt_template, include_group_in_columns, test_mode=False):
    """Extract information for all batches."""
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Processing Status")
        progress_bar = st.progress(0)
        log_container = st.empty()
        
        log_messages = []
        
        def update_status(message):
            """Update status and add to log."""
            log_messages.append(message)
            # Show more messages to accommodate detailed value displays
            recent_logs = "\n".join([f"{msg}" for msg in log_messages[-20:]])
            log_container.code(recent_logs, language=None)
        
        all_results = []
        total_batches = len(batches)
        
        # Apply test mode limit if enabled
        if test_mode:
            max_batches_to_process = 5
            batches_to_process = batches[:max_batches_to_process]
            if total_batches > max_batches_to_process:
                update_status(f"🧪 TESTING MODE: Processing only first {max_batches_to_process} batches (out of {total_batches} total)")
        else:
            batches_to_process = batches
        
        update_status(f"📊 Total batches to process: {len(batches_to_process)}")
        
        for i, batch in enumerate(batches_to_process):
            batch_num = i + 1
            group_value = batch['group_value']
            batch_rows = batch['rows']
            
            update_status(f"\n{'='*50}")
            update_status(f"📦 Processing Batch {batch_num}/{len(batches_to_process)}")
            update_status(f"{'='*50}")
            
            if group_value is not None:
                update_status(f"📌 Group value: {group_value}")
            else:
                update_status(f"📌 Group value: (null/empty)")
            
            update_status(f"📊 Rows in batch: {len(batch_rows)}")
            
            progress_bar.progress(i / len(batches_to_process))
            
            # Format rows for prompt
            rows_data = _format_rows_batch_for_prompt(
                batch_rows, 
                columns, 
                group_value, 
                include_group_in_columns
            )
            
            if not rows_data:
                update_status(f"⚠️ Batch {batch_num}: No valid rows (all NA), skipping")
                continue
            
            update_status(f"📝 Sending {len(rows_data)} rows to API...")
            
            try:
                results = _call_azure_openai(rows_data, prompt_template, status_callback=update_status)
                
                if results:
                    all_results.extend(results)
                    update_status(f"✅ Batch {batch_num} complete: {len(results)} result(s)")
                    # Show summary of returned values for this batch
                    update_status(f"📊 Batch {batch_num} Summary:")
                    for result in results[:5]:  # Show first 5 results from this batch
                        row_id = result.get('row_id', 'N/A')
                        product_name = result.get('product_name', 'N/A')
                        display_name = product_name[:40] + "..." if len(str(product_name)) > 40 else product_name
                        update_status(f"  → Row {row_id}: {display_name}")
                    if len(results) > 5:
                        update_status(f"  ... and {len(results) - 5} more in this batch")
                else:
                    update_status(f"⚠️ Batch {batch_num}: No results returned")
                    raise ValueError(f"Batch {batch_num} returned no results")
                    
            except Exception as e:
                update_status(f"❌ Batch {batch_num} failed: {str(e)}")
                progress_bar.empty()
                log_container.empty()
                raise ValueError(f"Failed to process batch {batch_num}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(batches_to_process))
        
        progress_bar.progress(1.0)
        update_status(f"\n{'='*50}")
        update_status(f"✅ All batches processed successfully!")
        update_status(f"📊 Total results: {len(all_results)}")
        
        progress_bar.empty()
        log_container.empty()
        
        return all_results


def _create_results_dataframe(results, original_df):
    """Create a DataFrame from extraction results."""
    if not results:
        return pd.DataFrame()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Ensure row_id and product_name columns exist
    if 'row_id' not in results_df.columns:
        raise ValueError("Results missing 'row_id' column")
    if 'product_name' not in results_df.columns:
        raise ValueError("Results missing 'product_name' column")
    
    # Merge with original dataframe to show original values
    merged_df = original_df.copy()
    merged_df = merged_df.merge(
        results_df[['row_id', 'product_name']],
        left_index=True,
        right_on='row_id',
        how='left'
    )
    
    return merged_df, results_df


# ============================================================================
# Main Render Function
# ============================================================================

def render():
    """Render the AI Advanced tab."""
    st.header("AI Advanced - Information Extraction")
    
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
    available_columns = _get_available_columns()
    
    if not available_columns:
        st.warning("⚠️ No columns available for extraction.")
        return
    
    st.subheader("Configuration")
    
    # Column selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select columns to extract information from:**")
        selected_columns = st.multiselect(
            "Columns:",
            options=available_columns,
            help="Select one or more columns that contain the information to extract"
        )
    
    with col2:
        st.write("**Select grouping column:**")
        grouping_column = st.selectbox(
            "Group by:",
            options=available_columns,
            help="Rows with the same value in this column will be processed together"
        )
    
    if not selected_columns:
        st.info("👆 Please select at least one column to extract information from.")
        return
    
    if not grouping_column:
        st.info("👆 Please select a grouping column.")
        return
    
    if grouping_column in selected_columns:
        include_group_in_prompt = True
        st.info(f"ℹ️ Grouping column '{grouping_column}' is also selected, so it will be included in the row information.")
    else:
        include_group_in_prompt = False
    
    # Prompt template editor
    st.subheader("Prompt Template")
    st.write("Edit the prompt template below. Use `[rows]` as a placeholder for the rows data.")
    
    prompt_template = st.text_area(
        "Prompt Template:",
        value=DEFAULT_PROMPT_TEMPLATE,
        height=150,
        help="The prompt sent to the AI model. [rows] will be replaced with the actual row data."
    )
    
    if '[rows]' not in prompt_template:
        st.warning("⚠️ Warning: Prompt template should contain '[rows]' placeholder.")
    
    # Show statistics
    st.subheader("Dataset Information")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("Total Rows", len(df))
    with col_info2:
        unique_groups = df[grouping_column].nunique()
        st.metric("Unique Groups", unique_groups)
    with col_info3:
        avg_rows_per_group = len(df) / unique_groups if unique_groups > 0 else 0
        st.metric("Avg Rows/Group", f"{avg_rows_per_group:.1f}")
    
    # Test mode checkbox
    test_mode = st.checkbox(
        "🧪 Test Mode (Process only first 5 batches)",
        value=False,
        help="Enable to process only the first 5 batches for testing purposes",
        key="test_mode_advanced"
    )
    
    # Generate extraction
    if st.button("🤖 Generate Extraction", type="primary"):
        # Prepare batches
        batches = _batch_rows_by_group(df, grouping_column, MAX_ROWS_PER_BATCH)
        
        st.info(f"📦 Prepared {len(batches)} batch(es) for processing (max {MAX_ROWS_PER_BATCH} rows per batch)")
        
        # Extract information
        try:
            results = _extract_information_for_batches(
                batches,
                selected_columns,
                prompt_template,
                include_group_in_prompt,
                test_mode=test_mode
            )
            
            if results:
                # Store results in session state
                st.session_state[f"extraction_results_{grouping_column}"] = results
                st.success(f"✅ Extraction complete! Generated {len(results)} result(s).")
            else:
                st.error("❌ No results generated.")
                
        except Exception as e:
            st.error(f"❌ Error during extraction: {str(e)}")
            return
    
    # Display and edit results if available
    results_key = f"extraction_results_{grouping_column}"
    if results_key in st.session_state:
        st.subheader("Review and Edit Results")
        st.info("Review the extraction results below. You can edit the 'product_name' column to make manual adjustments.")
        
        results = st.session_state[results_key]
        
        # Create results DataFrame
        try:
            merged_df, results_df = _create_results_dataframe(results, df)
            
            # Create editable table
            edited_results_df = st.data_editor(
                results_df[['row_id', 'product_name']].sort_values('row_id'),
                use_container_width=True,
                num_rows="dynamic",
                key=f"results_editor_{grouping_column}",
                column_config={
                    "row_id": st.column_config.NumberColumn(
                        "Row ID",
                        disabled=True,
                        width="small"
                    ),
                    "product_name": st.column_config.TextColumn(
                        "Product Name",
                        width="large"
                    )
                }
            )
            
            # Store edited results
            if not edited_results_df.empty:
                edited_results = edited_results_df.to_dict('records')
                st.session_state[results_key] = edited_results
                
                # Show preview
                st.subheader("Preview")
                preview_df = df.copy()
                preview_df = preview_df.merge(
                    edited_results_df[['row_id', 'product_name']],
                    left_index=True,
                    right_on='row_id',
                    how='left'
                )
                
                # Show sample of original vs extracted
                preview_cols = selected_columns + ['product_name']
                if 'product_name' in preview_df.columns:
                    preview_sample = preview_df[preview_cols].head(20)
                    st.dataframe(preview_sample, use_container_width=True, hide_index=True)
                    
                    # Statistics
                    extracted_count = preview_df['product_name'].notna().sum()
                    total_count = len(preview_df)
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Rows with Extraction", extracted_count)
                    with col_stat2:
                        st.metric("Total Rows", total_count)
                    
                    # Apply button
                    if st.button("✅ Apply Extraction", type="primary"):
                        # Create new column
                        new_column_name = f"{grouping_column}_extracted_info"
                        
                        # Merge results into processed_df
                        results_dict = dict(zip(edited_results_df['row_id'], edited_results_df['product_name']))
                        st.session_state.processed_df[new_column_name] = st.session_state.processed_df.index.to_series().map(results_dict)
                        
                        st.success(f"✅ Created new column '{new_column_name}' with extracted information!")
                        st.info(f"💡 The extracted information is now available in '{new_column_name}' column.")
                        
                        # Clear results from session state
                        del st.session_state[results_key]
                        st.rerun()
                        
        except Exception as e:
            st.error(f"❌ Error displaying results: {str(e)}")
            st.exception(e)

