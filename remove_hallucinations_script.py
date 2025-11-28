"""
LangGraph script to process merged vendors CSV and remove hallucinations from product examples
and other data using Azure OpenAI with structured output.
"""

# ============================================================================
# Configuration Parameters
# ============================================================================
MAX_COMPLETION_TOKENS = 20000  # Maximum tokens for completion
TEMPERATURE = 1  # Set to None to use model default, or specify a value if supported
CSV_DELIMITER = ';'  # Delimiter used in CSV files
INPUT_CSV_FILE = "merged_vendors_20251124_123606.csv"  # Input CSV file path
OUTPUT_CSV_PREFIX = "verified_vendors"  # Prefix for output CSV file

# ============================================================================
# Imports
# ============================================================================
import csv as csv_module
import pandas as pd
from datetime import datetime
from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME


# Define Pydantic model for structured output
class VerifiedVendorRow(BaseModel):
    """Single verified row in the output CSV"""
    vendor: str = Field(description="Vendor name (verified)")
    product_families: str = Field(description="Product Families / Model Series (verified, no hallucinations)")
    aliases: str = Field(description="Aliases / Naming Conventions (verified, no hallucinations)")
    product_examples: str = Field(description="Product Examples (verified real products only, remove any hallucinations)")


class VerifiedOutput(BaseModel):
    """Output containing verified vendor row"""
    verified_row: VerifiedVendorRow = Field(description="Verified vendor row with hallucinations removed")


# Define the state for our graph
class ProcessingState(dict):
    """State for processing vendor rows"""
    row_data: dict  # Dictionary with vendor, product_families, aliases, product_examples
    row_number: int  # Row number being processed
    verified_result: VerifiedOutput  # Structured output from LLM


def process_row_node(state: ProcessingState) -> ProcessingState:
    """Process a single row and remove hallucinations using Azure OpenAI with structured output"""
    row_data = state["row_data"]
    row_number = state["row_number"]
    
    vendor = row_data.get("Vendor", "")
    product_families = row_data.get("Product Families / Model Series", "")
    aliases = row_data.get("Aliases / Naming Conventions", "")
    product_examples = row_data.get("Product Examples", "")
    
    # Create prompt for removing hallucinations
    system_message = """You are a fact-checking assistant specialized in networking equipment and CPE devices. 
Your task is to review vendor data and remove any hallucinations, incorrect information, or made-up product models.

CRITICAL: Only keep information that you are certain exists in the real world. Remove or correct:
- Made-up product model numbers
- Incorrect vendor names or variations
- Non-existent product families
- Fabricated aliases
- Any product examples that don't actually exist

Be conservative - if you're not certain something is real, remove it or mark it clearly."""
    
    prompt = f"""Review the following vendor data and remove any hallucinations, incorrect information, or made-up product details.

Your task:
1. Verify the vendor name is correct
2. Verify Product Families / Model Series contain only real, existing product families
3. Verify Aliases / Naming Conventions contain only real naming patterns
4. CRITICALLY REVIEW Product Examples - remove any made-up or incorrect product model numbers/names
5. Keep only information you are certain is accurate and real

Input data:
Vendor: {vendor}
Product Families / Model Series: {product_families}
Aliases / Naming Conventions: {aliases}
Product Examples: {product_examples}

Instructions:
- If a product example doesn't exist or you're unsure, remove it from the list
- Keep only verified, real product models
- Maintain the numbered list format for product examples
- Preserve all verified information
- Be conservative - when in doubt, remove rather than keep incorrect information

Return the verified data with all hallucinations removed."""

    # Print the full prompt being sent
    print(f"\n{'='*80}")
    print(f"Processing row {row_number}: {vendor}")
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
        structured_llm = llm.with_structured_output(VerifiedOutput)
        
        # Get structured response
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        result = structured_llm.invoke(messages)
        
        # Print the full response
        verified = result.verified_row
        print(f"📥 RESPONSE FROM AZURE OPENAI:")
        print(f"{'-'*80}")
        print(f"Vendor: {verified.vendor}")
        print(f"  Product Families: {verified.product_families}")
        print(f"  Aliases: {verified.aliases}")
        print(f"  Product Examples: {verified.product_examples}")
        print(f"{'-'*80}\n")
        
        return {
            "verified_result": result,
            "row_data": row_data,
            "row_number": row_number
        }
    
    except Exception as e:
        error_msg = f"Error processing row {row_number}: {str(e)}"
        print(f"\n❌ {error_msg}\n")
        # Return original data on error
        verified_row = VerifiedVendorRow(
            vendor=vendor,
            product_families=product_families,
            aliases=aliases,
            product_examples=product_examples
        )
        return {
            "verified_result": VerifiedOutput(verified_row=verified_row),
            "row_data": row_data,
            "row_number": row_number
        }


def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(ProcessingState)
    workflow.add_node("process_row", process_row_node)
    workflow.set_entry_point("process_row")
    workflow.add_edge("process_row", END)
    return workflow.compile()


def load_csv_data(csv_path: str):
    """Load CSV data into a list of dictionaries"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv_module.DictReader(f, delimiter=CSV_DELIMITER)
        for idx, row in enumerate(reader, start=2):  # Start at 2 (after header)
            if any(row.values()):  # Skip empty rows
                rows.append(row)
    return rows


def main():
    """Main function to process CSV, remove hallucinations, and save results"""
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
    
    # Load CSV data
    print(f"📖 Loading CSV file: {csv_path}")
    rows = load_csv_data(csv_path)
    print(f"✅ Loaded {len(rows)} rows to process\n")
    
    if not rows:
        print("❌ No rows found in CSV file")
        return
    
    # Build the graph
    graph = build_graph()
    
    # Process each row
    all_verified_rows = []
    for idx, row_data in enumerate(rows, start=1):
        print(f"🔄 Processing row {idx}/{len(rows)}...")
        
        initial_state = ProcessingState(
            row_data=row_data,
            row_number=idx,
            verified_result=VerifiedOutput(verified_row=VerifiedVendorRow(
                vendor="",
                product_families="",
                aliases="",
                product_examples=""
            ))
        )
        
        # Run the graph for this row
        result = graph.invoke(initial_state)
        
        # Extract verified row
        if result.get("verified_result") and result["verified_result"].verified_row:
            all_verified_rows.append(result["verified_result"].verified_row)
            print(f"✅ Row {idx} processed\n")
        else:
            print(f"⚠️  Row {idx} returned no result, using original data\n")
            # Fallback to original data
            verified_row = VerifiedVendorRow(
                vendor=row_data.get("Vendor", ""),
                product_families=row_data.get("Product Families / Model Series", ""),
                aliases=row_data.get("Aliases / Naming Conventions", ""),
                product_examples=row_data.get("Product Examples", "")
            )
            all_verified_rows.append(verified_row)
    
    # Create output DataFrame
    output_data = []
    for row in all_verified_rows:
        output_data.append({
            'Vendor': row.vendor,
            'Product Families / Model Series': row.product_families,
            'Aliases / Naming Conventions': row.aliases,
            'Product Examples': row.product_examples
        })
    
    output_df = pd.DataFrame(output_data)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_CSV_PREFIX}_{timestamp}.csv"
    
    # Save to CSV with configured delimiter
    output_df.to_csv(output_file, sep=CSV_DELIMITER, index=False, encoding='utf-8')
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Processing complete!")
    print(f"   Processed rows: {len(rows)}")
    print(f"   Verified rows: {len(all_verified_rows)}")
    print(f"   Output saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

