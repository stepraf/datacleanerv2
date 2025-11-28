import streamlit as st
import pandas as pd
import json
import sys
import os
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION
    )
except ImportError:
    AZURE_OPENAI_API_KEY = None
    AZURE_OPENAI_ENDPOINT = None
    AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-5-nano"
    AZURE_OPENAI_API_VERSION = "2025-01-01-preview"


from prompt_templates import DEFAULT_PROMPT_TEMPLATE


MAX_ROWS_PER_BATCH = 50
MIN_BATCH_SIZE = 35
MAX_COMPLETION_TOKENS = 20000
INTERMEDIATE_RESULTS_DIR = "intermediate_extraction_results"


class ProductRow(BaseModel):
    row_id: int = Field(description="The row ID from the input data")
    product_name: str = Field(description="The extracted product name or information")
    manufacturer: str = Field(description="The manufacturer or vendor name of the product")


class ExtractionOutput(BaseModel):
    results: List[ProductRow] = Field(description="List of extracted product information, one per input row")


def _has_processed_data():
    df = st.session_state.get('processed_df')
    return df is not None and len(df) > 0


def _get_available_columns():
    if not _has_processed_data():
        return []
    return [col for col in st.session_state.processed_df.columns if col != 'initial_id']


def _is_valid_azure_config():
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        return False
    if AZURE_OPENAI_API_KEY == "YOUR_API_KEY_HERE" or AZURE_OPENAI_ENDPOINT == "YOUR_ENDPOINT_HERE":
        return False
    return True


def _ensure_results_directory():
    """Create intermediate results directory if it doesn't exist."""
    if not os.path.exists(INTERMEDIATE_RESULTS_DIR):
        os.makedirs(INTERMEDIATE_RESULTS_DIR)
    return INTERMEDIATE_RESULTS_DIR


def _save_intermediate_results(df, results, output_file_path, grouping_column):
    """Save intermediate results to CSV file with full dataframe merged with current results."""
    try:
        # Create a copy of the dataframe
        output_df = df.copy()
        
        # Convert results to DataFrame
        if results:
            results_df = pd.DataFrame(results)
            
            # Merge results with dataframe using row_id
            product_dict = dict(zip(results_df['row_id'], results_df['product_name']))
            manufacturer_dict = dict(zip(results_df['row_id'], results_df['manufacturer']))
            
            # Add columns to output dataframe
            product_column = f"{grouping_column}_extracted_info"
            manufacturer_column = f"{grouping_column}_manufacturer"
            
            output_df[product_column] = output_df.index.to_series().map(product_dict)
            output_df[manufacturer_column] = output_df.index.to_series().map(manufacturer_dict)
        
        # Save to CSV
        output_df.to_csv(output_file_path, index=False, encoding='utf-8')
        return True
    except Exception as e:
        return False


def _format_rows_for_prompt(df_batch, columns, group_value=None, include_group=False):
    """Format batch rows for prompt. Uses df_batch.index as row_id to maintain mapping."""
    rows_list = []
    for idx, row in df_batch.iterrows():
        row_data = {}
        for col in columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                row_data[col] = str(value).strip()
        if include_group and group_value and group_value not in row_data.values():
            row_data['_group'] = str(group_value)
        if row_data:
            row_data['_row_id'] = int(idx)
            rows_list.append(row_data)
    return rows_list


def _split_group_into_batches(group_df, group_value, max_rows):
    batches = []
    if len(group_df) <= max_rows:
        batches.append({'group_value': group_value, 'rows': group_df})
    else:
        for i in range(0, len(group_df), max_rows):
            batches.append({'group_value': group_value, 'rows': group_df.iloc[i:i + max_rows]})
    return batches


def _merge_small_batches(batches, min_size, max_size):
    """Merge small batches. CRITICAL: preserve index (ignore_index=False) for row_id mapping."""
    merged = []
    i = 0
    while i < len(batches):
        current = batches[i]
        if len(current['rows']) < min_size:
            merged_rows = [current['rows']]
            j = i + 1
            while j < len(batches):
                next_batch = batches[j]
                total_size = sum(len(df) for df in merged_rows) + len(next_batch['rows'])
                if total_size <= max_size:
                    merged_rows.append(next_batch['rows'])
                    j += 1
                    if total_size >= min_size:
                        break
                else:
                    break
            combined_rows = pd.concat(merged_rows, ignore_index=False) if len(merged_rows) > 1 else merged_rows[0]
            merged.append({'group_value': current['group_value'], 'rows': combined_rows})
            i = j
        else:
            merged.append(current)
            i += 1
    return merged


def _batch_rows_by_group(df, group_column, max_rows_per_batch=MAX_ROWS_PER_BATCH):
    batches = []
    for group_value, group_df in df.groupby(group_column):
        normalized_group = None if (pd.isna(group_value) or str(group_value).strip() == '') else group_value
        batches.extend(_split_group_into_batches(group_df, normalized_group, max_rows_per_batch))
    return _merge_small_batches(batches, MIN_BATCH_SIZE, max_rows_per_batch)


def _call_llm(rows_data, prompt_template, status_callback=None):
    """Call LLM with structured output."""
    def log(msg):
        if status_callback:
            status_callback(msg)
    
    rows_text = "\n".join([
        f"Row {row['_row_id']}: {json.dumps({k: v for k, v in row.items() if k != '_row_id'}, ensure_ascii=False)}"
        for row in rows_data
    ])
    prompt = prompt_template.replace("[rows]", rows_text)
    system_message = "You are a data extraction assistant. Extract the requested information from each row."
    
    log(f"📤 Prompt ({len(prompt)} chars):\n{system_message}\n\n{prompt[:500]}...")
    
    try:
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            max_completion_tokens=MAX_COMPLETION_TOKENS
        )
        structured_llm = llm.with_structured_output(ExtractionOutput)
        result = structured_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ])
        
        log(f"📥 Response: {len(result.results)} results")
        log(f"📥 Full response:\n{json.dumps([{'row_id': r.row_id, 'product_name': r.product_name, 'manufacturer': r.manufacturer} for r in result.results], indent=2, ensure_ascii=False)}")
        
        return [{"row_id": r.row_id, "product_name": r.product_name, "manufacturer": r.manufacturer} for r in result.results]
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        raise


def _process_batches(batches, columns, prompt_template, include_group, df, grouping_column, test_mode=False):
    """Process all batches and save intermediate results after each batch."""
    st.subheader("🔄 Processing Status")
    progress_bar = st.progress(0)
    log_expander = st.expander("📋 Logs", expanded=True)
    with log_expander:
        log_container = st.empty()
    
    log_messages = []
    def log(msg):
        log_messages.append(msg)
        log_container.code("\n".join(log_messages), language=None)
    
    # Create output directory and generate unique filename for this extraction run
    _ensure_results_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"extraction_{grouping_column}_{timestamp}.csv"
    output_file_path = os.path.join(INTERMEDIATE_RESULTS_DIR, output_filename)
    
    log(f"💾 Intermediate results will be saved to: {output_file_path}")
    
    batches_to_process = batches[:5] if test_mode else batches
    if test_mode and len(batches) > 5:
        log(f"🧪 TEST MODE: Processing first 5 of {len(batches)} batches")
    
    all_results = []
    try:
        for i, batch in enumerate(batches_to_process, 1):
            log(f"\n📦 Batch {i}/{len(batches_to_process)}: {len(batch['rows'])} rows")
            rows_data = _format_rows_for_prompt(batch['rows'], columns, batch['group_value'], include_group)
            if rows_data:
                results = _call_llm(rows_data, prompt_template, log)
                all_results.extend(results)
                
                # Save intermediate results after each batch
                if _save_intermediate_results(df, all_results, output_file_path, grouping_column):
                    log(f"💾 Saved intermediate results: {len(all_results)} results so far")
                else:
                    log(f"⚠️ Failed to save intermediate results")
            
            progress_bar.progress(i / len(batches_to_process))
        
        log(f"\n✅ Complete: {len(all_results)} results")
        log(f"💾 Final results saved to: {output_file_path}")
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        # Try to save partial results even on error
        if all_results:
            _save_intermediate_results(df, all_results, output_file_path, grouping_column)
            log(f"💾 Saved partial results before error: {output_file_path}")
        raise
    finally:
        progress_bar.empty()
    
    return all_results, output_file_path


def render():
    st.header("AI Advanced - Information Extraction")
    
    if not _has_processed_data():
        st.warning("⚠️ Please import data first.")
        return
    
    if not _is_valid_azure_config():
        st.error("⚠️ Please configure Azure OpenAI credentials in config.py")
        return
    
    df = st.session_state.processed_df
    available_columns = _get_available_columns()
    
    if not available_columns:
        st.warning("⚠️ No columns available.")
        return
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        selected_columns = st.multiselect("Columns to extract from:", options=available_columns)
    with col2:
        grouping_column = st.selectbox("Group by:", options=available_columns)
    
    if not selected_columns or not grouping_column:
        st.info("👆 Please select columns and grouping column.")
        return
    
    include_group = grouping_column in selected_columns
    prompt_template = st.text_area("Prompt Template:", value=DEFAULT_PROMPT_TEMPLATE, height=150)
    
    if '[rows]' not in prompt_template:
        st.warning("⚠️ Prompt template should contain '[rows]' placeholder.")
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Unique Groups", df[grouping_column].nunique())
    with col3:
        st.metric("Avg Rows/Group", f"{len(df) / df[grouping_column].nunique():.1f}")
    
    test_mode = st.checkbox("🧪 Test Mode (first 5 batches)", value=False)
    
    # Generate extraction
    if st.button("🤖 Generate Extraction", type="primary"):
        batches = _batch_rows_by_group(df, grouping_column, MAX_ROWS_PER_BATCH)
        st.info(f"📦 Prepared {len(batches)} batch(es)")
        
        try:
            results, output_file_path = _process_batches(
                batches, selected_columns, prompt_template, include_group, df, grouping_column, test_mode
            )
            if results:
                results_key = f"extraction_results_{grouping_column}"
                st.session_state[results_key] = results
                st.success(f"✅ Generated {len(results)} result(s)")
                st.info(f"💾 Intermediate results saved to: `{output_file_path}`")
            else:
                st.error("❌ No results generated.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    
    # Display results
    results_key = f"extraction_results_{grouping_column}"
    if results_key in st.session_state:
        results = st.session_state[results_key]
        results_df = pd.DataFrame(results)
        
        st.subheader("Review and Edit Results")
        edited_df = st.data_editor(
            results_df[['row_id', 'product_name', 'manufacturer']].sort_values('row_id'),
            use_container_width=True,
            num_rows="dynamic",
            key=f"results_editor_{grouping_column}",
            column_config={
                "row_id": st.column_config.NumberColumn("Row ID", disabled=True),
                "product_name": st.column_config.TextColumn("Product Name"),
                "manufacturer": st.column_config.TextColumn("Manufacturer")
            }
        )
        
        if not edited_df.empty:
            # Preview
            preview_df = df.merge(
                edited_df[['row_id', 'product_name', 'manufacturer']],
                left_index=True, right_on='row_id', how='left'
            )
            if 'product_name' in preview_df.columns:
                st.dataframe(preview_df[selected_columns + ['product_name', 'manufacturer']].head(20), use_container_width=True)
                st.metric("Rows with Extraction", preview_df['product_name'].notna().sum())
            
            # Apply
            if st.button("✅ Apply Extraction", type="primary"):
                # Create product_name column
                product_column = f"{grouping_column}_extracted_info"
                product_dict = dict(zip(edited_df['row_id'], edited_df['product_name']))
                st.session_state.processed_df[product_column] = st.session_state.processed_df.index.to_series().map(product_dict)
                
                # Create manufacturer column
                manufacturer_column = f"{grouping_column}_manufacturer"
                manufacturer_dict = dict(zip(edited_df['row_id'], edited_df['manufacturer']))
                st.session_state.processed_df[manufacturer_column] = st.session_state.processed_df.index.to_series().map(manufacturer_dict)
                
                st.success(f"✅ Created columns '{product_column}' and '{manufacturer_column}'")
                del st.session_state[results_key]
                st.rerun()
