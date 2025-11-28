"""
LangGraph script to process product naming conventions CSV, merge duplicate vendors,
and save to timestamped CSV file using Azure OpenAI with structured output.
"""

# ============================================================================
# Configuration Parameters
# ============================================================================
MAX_COMPLETION_TOKENS = 20000  # Maximum tokens for completion
TEMPERATURE = 1  # Set to None to use model default, or specify a value if supported
CSV_DELIMITER = ';'  # Delimiter used in CSV files
INPUT_CSV_FILE = "initial_product_naming_conventions.csv"  # Input CSV file path

# ============================================================================
# Imports
# ============================================================================
import csv as csv_module
import pandas as pd
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME


# Define Pydantic model for structured output
class VendorRow(BaseModel):
    """Single row in the output CSV"""
    vendor: str = Field(description="Vendor name")
    product_families: str = Field(description="Product Families / Model Series - combined and deduplicated")
    aliases: str = Field(description="Aliases / Naming Conventions - combined and deduplicated")
    product_examples: str = Field(description="Real product examples - specific model numbers/names that exist in the market, formatted as a numbered list")


class MergedVendorsOutput(BaseModel):
    """Output containing list of merged vendor rows"""
    merged_rows: List[VendorRow] = Field(description="List of merged vendor rows, one per unique vendor")


# Define the state for our graph
class ProcessingState(dict):
    """State for processing vendor batches"""
    batch_data: str  # CSV string of the batch
    batch_letter: str  # First letter of vendors in this batch
    merged_result: MergedVendorsOutput  # Structured output from LLM


def process_batch_node(state: ProcessingState) -> ProcessingState:
    """Process a batch of vendors and merge duplicates using Azure OpenAI with structured output"""
    batch_data = state["batch_data"]
    batch_letter = state["batch_letter"]
    
    # Create prompt for merging vendors
    system_message = """You are a data processing assistant specialized in vendor data normalization and product information. Your task is to merge rows with the same vendor name into a single row, 
while preserving all granular information from the Product Families and Aliases columns, and provide real product examples. 

CRITICAL: You must normalize vendor names by removing parenthetical additions and descriptive suffixes. For example:
- "Vendor (expanded)" → "Vendor"
- "Vendor CPE (FWA extension)" → "Vendor"  
- "Vendor Networks" and "Vendor (FWA-focused)" → both become "Vendor Networks"

All rows with the same base vendor name MUST be merged into ONE row. Remove only exact duplicates from Product Families and Aliases, but keep all unique information. Combine values with commas when merging.

For each merged vendor, provide real product examples - specific model numbers/names that actually exist in the market, based on the Product Families and Aliases information."""

    prompt = f"""Given the following CSV data (semicolon-separated), merge rows with the same vendor name into a single row per vendor.

CRITICAL MERGING RULES:
1. VENDOR NAME MATCHING: Merge rows that have the SAME BASE VENDOR NAME, even if they have different suffixes or parenthetical additions.
   Examples that MUST be merged:
   - "Arcadyan" and "Arcadyan (expanded)" → merge into "Arcadyan"
   - "Cambium Networks" and "Cambium (FWA-focused)" → merge into "Cambium Networks"
   - "Huawei", "Huawei CPE (FWA extension)", and "Huawei ONT (full detail)" → merge into "Huawei"
   - "Nokia" and "Nokia (expanded)" → merge into "Nokia"
   - "MitraStar (Unizyx / Zyxel Group)" and "MitraStar (Zyxel subsidiary)" → merge into "MitraStar"
   - "Compal Broadband" and "Compal / Arcadyan OEM Networks (global ISP supply)" → merge into "Compal Broadband" (if they refer to the same company)
   
2. VENDOR NAME NORMALIZATION: 
   - Remove parenthetical additions like "(expanded)", "(FWA-focused)", "(full detail)", etc.
   - Remove descriptive prefixes like "Huawei CPE", "Huawei ONT" → use just "Huawei"
   - Keep the most common/base name of the vendor
   - If unsure, use the shorter, simpler vendor name

3. MERGING REQUIREMENTS:
   - ALL rows with the same base vendor name MUST be merged into ONE single row
   - Do NOT create multiple rows for the same vendor
   - The output must have exactly ONE row per unique vendor

4. DATA PRESERVATION:
   - For Product Families / Model Series: Combine ALL unique values from all merged rows, remove only exact duplicates, preserve all granular information
   - For Aliases / Naming Conventions: Combine ALL unique values from all merged rows, remove only exact duplicates, preserve all granular information
   - Use commas to separate combined values

5. PRODUCT EXAMPLES (NEW COLUMN):
   - For each merged vendor, provide 5-15 real, specific product model examples that exist in the market
   - These should be actual model numbers/names that customers might encounter
   - Format as a numbered list (e.g., "1. Vendor Model-1234, 2. Vendor Model-5678, ...")
   - Base the examples on the Product Families and Aliases information provided
   - Include examples from different product families/series when available
   - Ensure all examples are real products that exist in the market

6. OUTPUT FORMAT:
   - Return the merged data as a list of vendor rows with 4 columns: Vendor, Product Families, Aliases, Product Examples
   - Each vendor should appear exactly ONCE in the output
   - Ensure no duplicate vendors in the final output
   - Format: Vendor;Product Families / Model Series;Aliases / Naming Conventions;Product Examples

Input data:
{batch_data}

IMPORTANT: Review the input data carefully. If you see multiple rows with variations of the same vendor name (e.g., "Vendor", "Vendor (expanded)", "Vendor CPE"), they MUST all be merged into a single row with the base vendor name. 

For each merged vendor, provide real product examples based on the Product Families and Aliases information. Return the merged data as a list of vendor rows with product examples, ensuring each vendor appears only once."""

    # Print the full prompt being sent
    print(f"\n{'='*80}")
    print(f"Processing batch for vendors starting with '{batch_letter.upper()}'")
    print(f"{'='*80}")
    print(f"\n📤 PROMPT SENT TO AZURE OPENAI:")
    print(f"{'-'*80}")
    print(f"System Message: {system_message}")
    print(f"\nUser Message:\n{prompt}")
    print(f"{'-'*80}\n")

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
        structured_llm = llm.with_structured_output(MergedVendorsOutput)
        
        # Get structured response
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        result = structured_llm.invoke(messages)
        
        # Print the full response
        print(f"📥 RESPONSE FROM AZURE OPENAI:")
        print(f"{'-'*80}")
        for row in result.merged_rows:
            print(f"Vendor: {row.vendor}")
            print(f"  Product Families: {row.product_families}")
            print(f"  Aliases: {row.aliases}")
            print(f"  Product Examples: {row.product_examples}")
        print(f"{'-'*80}\n")
        
        return {
            "merged_result": result,
            "batch_data": batch_data,
            "batch_letter": batch_letter
        }
    
    except Exception as e:
        error_msg = f"Error processing batch '{batch_letter}': {str(e)}"
        print(f"\n❌ {error_msg}\n")
        # Return empty result on error
        return {
            "merged_result": MergedVendorsOutput(merged_rows=[]),
            "batch_data": batch_data,
            "batch_letter": batch_letter
        }


def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(ProcessingState)
    workflow.add_node("process_batch", process_batch_node)
    workflow.set_entry_point("process_batch")
    workflow.add_edge("process_batch", END)
    return workflow.compile()


def load_and_prepare_data(csv_path: str):
    """Load CSV, sort by vendor, and group by first letter"""
    # Manual parsing to handle semicolons within fields (fields are not properly quoted)
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Skip header
        if not lines:
            raise ValueError("CSV file is empty")
        
        for line_num, line in enumerate(lines[1:], start=2):  # Start at 2 (after header)
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Split by semicolon
            parts = line.split(CSV_DELIMITER)
            
            if len(parts) >= 3:
                vendor = parts[0].strip()
                # If more than 3 parts, everything between first and last is Product Families
                if len(parts) > 3:
                    product_families = CSV_DELIMITER.join(parts[1:-1]).strip()
                    aliases = parts[-1].strip()
                else:
                    product_families = parts[1].strip()
                    aliases = parts[2].strip()
                
                rows.append({
                    'Vendor': vendor,
                    'Product Families / Model Series': product_families,
                    'Aliases / Naming Conventions': aliases
                })
            elif len(parts) == 2:
                # Handle rows with only 2 fields (missing aliases)
                vendor = parts[0].strip()
                product_families = parts[1].strip()
                rows.append({
                    'Vendor': vendor,
                    'Product Families / Model Series': product_families,
                    'Aliases / Naming Conventions': ""
                })
            elif len(parts) == 1 and parts[0].strip():
                # Handle rows with only vendor
                rows.append({
                    'Vendor': parts[0].strip(),
                    'Product Families / Model Series': "",
                    'Aliases / Naming Conventions': ""
                })
    
    # Create DataFrame from parsed rows
    df = pd.DataFrame(rows)
    
    # Remove empty rows
    df = df.dropna(subset=['Vendor'])
    df = df[df['Vendor'].str.strip() != '']
    
    # Sort alphabetically by vendor
    df = df.sort_values('Vendor', key=lambda x: x.str.upper())
    
    # Group by first letter of vendor
    batches = {}
    for _, row in df.iterrows():
        vendor = str(row['Vendor']).strip()
        if vendor:
            first_letter = vendor[0].upper()
            if first_letter not in batches:
                batches[first_letter] = []
            batches[first_letter].append(row)
    
    return df, batches


def create_batch_csv_string(batch_rows: List[pd.Series]) -> str:
    """Convert batch rows to CSV string format"""
    header = f"Vendor{CSV_DELIMITER}Product Families / Model Series{CSV_DELIMITER}Aliases / Naming Conventions"
    lines = [header]
    for row in batch_rows:
        vendor = str(row['Vendor']).strip()
        product_families = str(row.get('Product Families / Model Series', '')).strip()
        aliases = str(row.get('Aliases / Naming Conventions', '')).strip()
        lines.append(f"{vendor}{CSV_DELIMITER}{product_families}{CSV_DELIMITER}{aliases}")
    return "\n".join(lines)


def main():
    """Main function to process CSV, merge vendors, and save results"""
    csv_path = INPUT_CSV_FILE
    
    # Verify Azure OpenAI configuration from config.py
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        missing = []
        if not AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        print(f"❌ Missing required configuration in config.py: {', '.join(missing)}")
        print("\nPlease set the following in your config.py file:")
        print("  AZURE_OPENAI_API_KEY=your_api_key")
        print("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("  AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name")
        print("  AZURE_OPENAI_API_VERSION=2025-01-01-preview")
        return
    
    # Load and prepare data
    print(f"📖 Loading CSV file: {csv_path}")
    df, batches = load_and_prepare_data(csv_path)
    print(f"✅ Loaded {len(df)} rows, grouped into {len(batches)} batches by first letter\n")
    
    if not batches:
        print("❌ No data found in CSV file")
        return
    
    # Build the graph
    graph = build_graph()
    
    # Process each batch
    all_merged_rows = []
    for letter in sorted(batches.keys()):
        batch_rows = batches[letter]
        batch_csv = create_batch_csv_string(batch_rows)
        
        print(f"🔄 Processing batch '{letter}' ({len(batch_rows)} rows)...")
        
        initial_state = ProcessingState(
            batch_data=batch_csv,
            batch_letter=letter,
            merged_result=MergedVendorsOutput(merged_rows=[])
        )
        
        # Run the graph for this batch
        result = graph.invoke(initial_state)
        
        # Extract merged rows
        if result.get("merged_result") and result["merged_result"].merged_rows:
            all_merged_rows.extend(result["merged_result"].merged_rows)
            print(f"✅ Batch '{letter}' processed: {len(result['merged_result'].merged_rows)} merged vendors\n")
        else:
            print(f"⚠️  Batch '{letter}' returned no results\n")
    
    # Create output DataFrame
    output_data = []
    for row in all_merged_rows:
        output_data.append({
            'Vendor': row.vendor,
            'Product Families / Model Series': row.product_families,
            'Aliases / Naming Conventions': row.aliases,
            'Product Examples': row.product_examples
        })
    
    output_df = pd.DataFrame(output_data)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"merged_vendors_{timestamp}.csv"
    
    # Save to CSV with configured delimiter
    output_df.to_csv(output_file, sep=CSV_DELIMITER, index=False, encoding='utf-8')
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Processing complete!")
    print(f"   Original rows: {len(df)}")
    print(f"   Merged vendors: {len(output_df)}")
    print(f"   Output saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
