"""
Simple script to extract product codes from CSV rows using Azure OpenAI with LangGraph.
Processes data in batches of 10 rows, extracts product codes, and saves results incrementally.
"""

# ============================================================================
# Configuration Parameters
# ============================================================================
MAX_COMPLETION_TOKENS = 20000
TEMPERATURE = 1
BATCH_SIZE = 10  # Number of rows per batch
INPUT_CSV_FILE = "product_desc.txt"  # Will be set via command line argument or default
OUTPUT_FILE_PREFIX = "PassWithBatchSize10ProductCategory"

# ============================================================================
# Imports
# ============================================================================
import sys
import os
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import (
    AZURE_OPENAI_API_VERSION, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_DEPLOYMENT_NAME
)


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================
class ProductCodeRow(BaseModel):
    row_id: int = Field(description="The row ID (1-10) from the batch")
    product_category: str = Field(description="The extracted product category for this row")


class ProductCategoryOutput(BaseModel):
    product_categories: List[ProductCodeRow] = Field(
        description="List of product categories, one per row ID in the batch"
    )


# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(log_file: str):
    """Configure logging to write to both stdout and a log file"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# LangGraph State
# ============================================================================
class ProcessingState(dict):
    """State for processing a batch of rows"""
    batch_rows: List[str]  # List of row strings with row IDs (e.g., "1 value1 value2 value3")
    batch_indices: List[int]  # Original dataframe indices for this batch
    product_categories_result: ProductCategoryOutput  # Structured output from LLM
    output_file: str  # Path to the output CSV file


# ============================================================================
# LangGraph Node
# ============================================================================
def process_batch_node(state: ProcessingState) -> ProcessingState:
    """Process a batch of rows and extract product categories using Azure OpenAI with structured output"""
    batch_rows = state["batch_rows"]
    batch_indices = state["batch_indices"]
    
    # Create prompt with all row strings
    rows_text = "\n".join(batch_rows)
    
    system_message = """You are a data extraction assistant. Your task is to extract product categories from each row of data.

(FACE) SMART WATCH (GPS UNIT,VENU 2 PLUS,W/O B,BLACK+SLATE,KOR/SEA,RECERTIFIED)-MODEL:A04125
The product category is: SMART WATCH

00-13060-00-VN#&AUDIO SIGNAL TRANSMITTER, MODEL EA-300-LYNK, VOLTAGE 5V DC, CURRENT 2A, 100% NEW#&VN
The product category is: AUDIO SIGNAL TRANSMITTER

1103052-VER2503#&VEHICLE ROUTER (WIRELESS) PN#1103052#&VN
The product category is: VEHICLE ROUTER

Extract the most relevant product category for each row. If no clear product category is found, return an empty string."""

    prompt = f"""Given the following rows of data, extract a product category for each row.

Each row is prefixed with its row ID (1-{BATCH_SIZE}). Extract the product category from the data in each row.

Input rows:
{rows_text}

Return a JSON object with product categories keyed by row ID. For each row, provide the extracted product category.
If a product category cannot be clearly identified, return an empty string for that row."""

    logger = logging.getLogger()
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing batch with {len(batch_rows)} rows")
    logger.info(f"{'='*80}")
    logger.info(f"\n📤 PROMPT SENT TO AZURE OPENAI:")
    logger.info(f"{'-'*80}")
    logger.info(f"System Message: {system_message}")
    logger.info(f"\nUser Message:\n{prompt}")
    logger.info(f"{'-'*80}\n")

    try:
        # Initialize LangChain AzureChatOpenAI
        llm_kwargs = {
            "azure_deployment": AZURE_OPENAI_DEPLOYMENT_NAME,
            "api_key": AZURE_OPENAI_API_KEY,
            "azure_endpoint": AZURE_OPENAI_ENDPOINT,
            "api_version": AZURE_OPENAI_API_VERSION,
            "max_completion_tokens": MAX_COMPLETION_TOKENS
        }
        if TEMPERATURE is not None:
            llm_kwargs["temperature"] = TEMPERATURE
        
        llm = AzureChatOpenAI(**llm_kwargs)
        
        # Use with_structured_output to get structured response
        structured_llm = llm.with_structured_output(ProductCategoryOutput)
        
        # Get structured response
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        result = structured_llm.invoke(messages)
        
        # Log the response
        logger = logging.getLogger()
        logger.info(f"📥 RESPONSE FROM AZURE OPENAI:")
        logger.info(f"{'-'*80}")
        for row in result.product_categories:
            logger.info(f"Row ID {row.row_id}: Product Category = '{row.product_category}'")
        logger.info(f"{'-'*80}\n")
        
        return {
            "product_categories_result": result,
            "batch_rows": batch_rows,
            "batch_indices": batch_indices,
            "output_file": state["output_file"]
        }
    
    except Exception as e:
        error_msg = f"Error processing batch: {str(e)}"
        logger = logging.getLogger()
        logger.error(f"\n❌ {error_msg}\n")
        # Return empty result on error
        return {
            "product_categories_result": ProductCategoryOutput(product_categories=[]),
            "batch_rows": batch_rows,
            "batch_indices": batch_indices,
            "output_file": state["output_file"]
        }


# ============================================================================
# LangGraph Builder
# ============================================================================
def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(ProcessingState)
    workflow.add_node("process_batch", process_batch_node)
    workflow.set_entry_point("process_batch")
    workflow.add_edge("process_batch", END)
    return workflow.compile()


# ============================================================================
# Helper Functions
# ============================================================================
def row_to_string(row: pd.Series, row_id: int) -> str:
    """Convert a dataframe row to a string with row ID prefix"""
    # Get all values, convert to string, filter out NaN/None
    values = [str(val) if pd.notna(val) else "" for val in row.values]
    # Join with spaces
    row_string = " ".join(values)
    # Add row ID prefix
    return f"{row_id} {row_string}"


def process_batches(df: pd.DataFrame, output_file: str, graph):
    """Process dataframe in batches and save results incrementally"""
    logger = logging.getLogger()
    total_rows = len(df)
    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    logger.info(f"📊 Total rows: {total_rows}")
    logger.info(f"📦 Number of batches: {num_batches}")
    logger.info(f"💾 Output file: {output_file}\n")
    
    # Initialize output dataframe with original data
    output_df = df.copy()
    
    # Initialize product_category column if it doesn't exist
    if "product_category" not in output_df.columns:
        output_df["product_category"] = ""
    
    # Process each batch
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        batch_indices = batch_df.index.tolist()
        
        logger.info(f"🔄 Processing batch {batch_num + 1}/{num_batches} (rows {start_idx + 1}-{end_idx})...")
        
        # Convert rows to strings with row IDs (1-10)
        batch_rows = []
        for i, (idx, row) in enumerate(batch_df.iterrows(), start=1):
            row_string = row_to_string(row, i)
            batch_rows.append(row_string)
        
        # Create initial state
        initial_state = ProcessingState(
            batch_rows=batch_rows,
            batch_indices=batch_indices,
            product_categories_result=ProductCategoryOutput(product_categories=[]),
            output_file=output_file
        )
        
        # Run the graph for this batch
        result = graph.invoke(initial_state)
        
        # Extract product codes and add to dataframe
        if result.get("product_categories_result") and result["product_categories_result"].product_categories:
            # Create a mapping of row_id -> product_category
            product_category_map = {
                pc.row_id: pc.product_category 
                for pc in result["product_categories_result"].product_categories
            }
            
            # Update the dataframe with product codes
            for i, original_idx in enumerate(batch_indices, start=1):
                if i in product_category_map:
                    output_df.at[original_idx, "product_category"] = product_category_map[i]
                    logger.info(f"  ✓ Row {i} (index {original_idx}): '{product_category_map[i]}'")
                else:
                    logger.warning(f"  ⚠ Row {i} (index {original_idx}): No product code found")
        
        # Save updated dataframe after each batch
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"  💾 Saved progress to {output_file}\n")
    
    return output_df


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main function to process CSV and extract product codes"""
    # Get input file from command line or use default
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = INPUT_CSV_FILE
        if input_file is None:
            print("❌ Error: Please provide an input file (CSV or TXT) as an argument")
            print("Usage: python simple_llm_pass.py <input_file>")
            print("  - CSV files: Standard CSV format with multiple columns")
            print("  - TXT files: Each line becomes a row, column name is the filename")
            sys.exit(1)
    
    # Generate timestamped output filename early (before logging setup)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_FILE_PREFIX}_{timestamp}.csv"
    log_file = output_file.replace('.csv', '.log')
    
    # Setup logging to both stdout and file
    logger = setup_logging(log_file)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Output CSV file: {output_file}")
    
    # Verify Azure OpenAI configuration
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        missing = []
        if not AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        logger.error(f"❌ Missing required configuration in config.py: {', '.join(missing)}")
        logger.error("\nPlease set the following in your config.py file:")
        logger.error("  AZURE_OPENAI_API_KEY=your_api_key")
        logger.error("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        logger.error("  AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name")
        logger.error("  AZURE_OPENAI_API_VERSION=2024-12-01-preview")
        sys.exit(1)
    
    # Load input file (CSV or TXT)
    file_ext = os.path.splitext(input_file)[1].lower()
    logger.info(f"📖 Loading file: {input_file} (extension: {file_ext})")
    
    try:
        if file_ext == '.txt':
            # For TXT files, each line is a single value row
            # Column name is the filename without extension
            filename_without_ext = os.path.splitext(os.path.basename(input_file))[0]
            logger.info(f"   Reading TXT file: each line will be a row, column name: '{filename_without_ext}'")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
            
            # Create dataframe with single column named after the file
            df = pd.DataFrame({filename_without_ext: lines})
            logger.info(f"✅ Loaded {len(df)} rows from TXT file")
            logger.info(f"   Column: '{filename_without_ext}'\n")
        else:
            # For CSV files, use pandas read_csv
            df = pd.read_csv(input_file)
            logger.info(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
            logger.info(f"   Columns: {', '.join(df.columns.tolist())}\n")
    except Exception as e:
        logger.error(f"❌ Error loading file: {str(e)}")
        sys.exit(1)
    
    if len(df) == 0:
        logger.error("❌ Error: Input file is empty")
        sys.exit(1)
    
    # Build the graph
    graph = build_graph()
    
    # Process batches
    try:
        output_df = process_batches(df, output_file, graph)
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Processing complete!")
        logger.info(f"   Total rows processed: {len(output_df)}")
        logger.info(f"   Rows with product codes: {len(output_df[output_df['product_category'].str.strip() != ''])}")
        logger.info(f"   Output saved to: {output_file}")
        logger.info(f"   Log file: {log_file}")
        logger.info(f"{'='*80}\n")
    except Exception as e:
        logger.error(f"\n❌ Error during processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

