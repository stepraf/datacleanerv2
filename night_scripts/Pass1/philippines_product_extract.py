#!/usr/bin/env python3
"""
Standalone script for AI-powered product information extraction from CSV files.
This script performs the same extraction logic as the Streamlit app but runs as a standalone script.
"""

import pandas as pd
import json
import os
import logging
import sys
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================================================
# CONFIGURATION - Edit these values for your use case
# ============================================================================

# Input CSV file path
INPUT_CSV_FILE = "Philippines_merged_data.csv"

# Columns to extract information from
SELECTED_COLUMNS = ["BUYER_NAME", "EXPORTER_NAME", "PRODUCT_DESCRIPTION"]  # Add your column names here

# Column to group by
GROUPING_COLUMN = "BUYER_NAME"  # Change to your grouping column

# Prompt template (must contain [rows] placeholder)
PROMPT_TEMPLATE = """
You will receive a batch of product-description strings from a freight or logistics database.
Each description may or may not contain enough information to deduce a specific CPE product name.

Your task:

For each input description, analyze the text and determine whether it likely refers to a CPE device from known vendors and families 
examples include but are not limited to:

Vendor	Product Families / Model Series	Aliases / Naming Conventions	Product Examples
Arcadyan	VRV/VGV series gateways, GPON ONT series	VRV-xxxx, VGV-xxxx, Arcadyan ONT	Arcadyan VGV7519
Arris / Comcast (Xfinity OEM)	Arris TG/TS/CM DOCSIS gateways; Comcast Xfinity xFi gateways (XB6, XB7, XB8)	XB6/XB7/XB8, TGxxxx/TMxxxx, ISP OEM SKUs	Arris TG1682G, Arris TG2482A, Xfinity XB6, Xfinity XB7, Xfinity XB8
Askey	RAC2V series, 5G/LTE CPE	RAC2V, Askey 5G CPE	Askey RAC2V1K
AVM	FRITZ!Box (75xx, 66xx, 56xx, 55xx series), FRITZ!Repeater line, FRITZ!WLAN, FRITZ!Fon	FRITZ!Box <model>, FRITZ!Repeater <model>, FRITZ!WLAN, FRITZ!Fon	FRITZ!Box 7590 AX, FRITZ!Box 7530, FRITZ!Box 6690 Cable, FRITZ!Box 5590 Fiber, FRITZ!Repeater 3000 AX, FRITZ!Repeater 1200 AX, FRITZ!Fon C6
Cambium Networks	PMP 450 subscriber modules, ePMP subscriber CPE (Force 180/200/300), cnRanger LTE CPE, cnPilot routers, cnWave 60GHz CPE, ePMP Force CPE series, XV Wi-Fi, cnMatrix edge, 450/450b subscriber modules	PMP450, Force 180/200/300, cnPilot R, cnRanger CPE, Force 300/400, 450b, cnWave CPE, XV Wi-Fi	Cambium PMP 450b, Cambium PMP 450i, Cambium ePMP Force 180, Cambium ePMP Force 200, Cambium ePMP Force 300-16, Cambium ePMP Force 400C, Cambium cnPilot R190W, Cambium cnPilot R200, Cambium cnWave V5000, Cambium cnWave V3000, 1Cambium XV2-2 Wi-Fi AP, 1Cambium cnMatrix EX1028
Cradlepoint	IBR series (IBR600/IBR900), E300/E3000 enterprise gateways, R1900/R2100 5G, W-series 5G outdoor units	IBR600/IBR900/IBR1700, E300/E3000, R1900/R2100, W1850/W2005	Cradlepoint IBR600C, Cradlepoint IBR900, Cradlepoint IBR1700, Cradlepoint E300, Cradlepoint E3000, Cradlepoint R1900, Cradlepoint R2100, Cradlepoint W1850, Cradlepoint W2005
Ericsson	W-series fixed wireless terminals	W-series fixed wireless terminal	Ericsson W30 Terminal, Ericsson W35 Fixed Wireless Terminal
Huawei	EchoLife HG/HS/EG series gateways; HN series ONT; CPE Pro/Pro 2 (H112/H122); 5G CPE Win/Win2; B-series LTE CPE (B3xx/B5xx); EchoLife ONT (HG8xx/HG9xx/HS8xx/HS86xx); OptiXstar ONT (HN8xx/HN82xx/HN85xx); Wi-Fi 6 ONT (OptiXstar HN8546/HN8245Q/HN8145V)	EchoLife HG/HS/EG; HN ONT; CPE Pro; H112/H122; 5G CPE Win/Win2; B3xx/B5xx LTE CPE; B535/B818; HG8xx/HG9xx; HS8xx/HS86xx; HN82xx/HN85xx; OptiXstar <model>; Q/V family Wi-Fi 6 ONT	Huawei EchoLife HG8245H, Huawei EchoLife HG8546M, Huawei OptiXstar HN8245Q, Huawei OptiXstar HN8145V, Huawei CPE Pro H112-370, Huawei CPE Pro 2 H122-373, Huawei B525, Huawei B535, Huawei B818-263, Huawei 5G CPE Win
Inseego	FX2000, FX3100, Wavemaker FG series, MiFi mobile hotspot series	"FXxxxx", "Wavemaker FG", "MiFi <model>"	Inseego Wavemaker FX2000, Inseego Wavemaker FX3100, Inseego Wavemaker FG2000, Inseego MiFi 8000, Inseego MiFi 8800L, Inseego MiFi X PRO 5G
Kaon Media	KSTB series set‑top boxes	KSTB-xxxx	
Milesight	UR industrial routers (UR32, UR75), UG LoRaWAN gateways, cellular CPE	UR series, UG series, Milesight 5G CPE	Milesight UR32, Milesight UR75 5G, Milesight UG65 Edge Gateway, Milesight UG56
NetComm Wireless	NF/NL series fixed wireless CPE, 4G/5G ODU/IDU, NTC industrial routers	NFxx, NLxx, NTC-xxxx, IDU/ODU <model>	NetComm NF18ACV, NetComm NF18MESH, NetComm NL1901ACV, NetComm NTC-140W, NetComm NTC-40WV
Netgear	Nighthawk routers (R/AX series), Orbi mesh (RBK/RBR/RBS/RBE series), LAX/LM/LBR 4G/5G CPE, DOCSIS CM/CB/CAX series, Nighthawk M (mobile hotspot) series, LBR cellular gateway, Insight BR gateways	"Nighthawk R/AX <model>", "Orbi RBK/RBE <model>", "LAX20/LBR20", "CM/CB-xxxx", "Nighthawk <model>", "AX/BE", "Nighthawk M5/M6", "Orbi <model>", "RBK/RBR/RBS", "CAX/CMxxxx", "LBR-20", "Insight gateway"	Netgear R7000, Netgear R8000, Netgear RAXE500, Netgear Orbi RBK753, Netgear LAX20, Netgear LBR20, Netgear Nighthawk M5 (MR5200), Netgear Nighthawk M6 (MR6150), Netgear CM2000, Netgear CAX80, 1Netgear BR500
Nokia	ONT/ONU series (G-010/G-240), Beacon Wi-Fi mesh series, FastMile 4G/5G CPE/gateways, 7368 ISAM ONT family	G-010/G-240, XS-01x, FastMile 4G/5G, Beacon <model>, 7368 ONT	Nokia G-010G-P, Nokia G-240G-E, Nokia G-240W-F, Nokia XS-010X-Q, Nokia 7368 ISAM ONT G-240W-F, Nokia Beacon 1, Nokia Beacon 6, 1Nokia FastMile 5G Gateway 3, 1Nokia FastMile 4G Gateway
Peplink	Balance routers; MAX BR/MAX HD cellular routers; MAX Transit series; Surf SOHO series	Balance <model>; MAX BR/HD; Transit Duo; Surf SOHO	Peplink Balance 20X, Peplink Balance 310, Peplink MAX BR1 Mini, Peplink MAX HD2, Peplink MAX Transit Duo, Peplink Surf SOHO MK3, Peplink BR1 Pro 5G
Sagemcom	F@ST broadband gateways; PON ONT/ONU; 4G/5G CPE; Video Soundbox/STB series	F@ST <model>; Sagemcom ONT/ONU; Video Soundbox	Sagemcom F@ST 5688W, Sagemcom F@ST 3890, Sagemcom F@ST 5560, Sagemcom Video Soundbox
Technicolor / Vantiva	Home Gateways (CGA/CGM/CGI series), DOCSIS Gateways (TC and TG series), xPON ONTs (FG series), Android/Operator STB (UIW/USW and KM7 series)	TC<model>, CGA/CGM/CGI-xxxx, TG-xxxx, FG ONT series, UIW/USW STB, KM7 Android STB	Technicolor TC4400 DOCSIS 3.1, Vantiva TG789vac v2, Technicolor UIW4001 STB
ZTE	ZXHN home gateways (F6xx/H2xx series), MC/MU 4G/5G indoor CPE, MF 4G routers, F41xx/F62xx ONT series, ZXHN F6xx/F8xx ONTs, F660/F680/F689 series, ZXHN H series gateways, ZTE B-series IPTV STB	"ZXHN <series>", "MC/MU 5G CPE", "MF <model>", "F41xx/F62xx ONT", "ZTE B-series STB", "MC801A/MC8020/MC888", "MU500", "MF289", "F6xx/F8xx", "F660/F680/F689", "ZXHN Hxxx series"	ZTE MC801A 5G CPE, ZTE MC8020 5G CPE, ZTE MC888 5G CPE, ZTE MU500 5G hotspot, ZTE MF289 LTE CPE, ZTE ZXHN F660 ONT, ZTE ZXHN F680 ONT, ZTE ZXHN F689 ONT, ZTE ZXHN H298A gateway, ZTE ZXHN H267A gateway, 1ZTE ZXHN F620 GPON ONT, 1ZTE B760H IPTV STB, 1ZTE MF286 4G router


If the description contains enough information to identify a specific model with high probability, extract:
1. Product name in standardized form: "<Vendor> <Model>"
   Example: "Huawei EchoLife HG8546M", "Netgear Orbi RBK753", "Cradlepoint IBR600C", "ZTE ZXHN F660"
2. Manufacturer/Vendor name: The company that makes the product
   Example: "Huawei", "Netgear", "Cradlepoint", "ZTE"

If the product cannot be confidently inferred, output empty strings for both fields.

Do not output explanations, reasoning, or uncertainty scores—only the extracted product name and manufacturer, or blank values if not found.

Extract product information from the following rows: 

[rows]."""

# Azure OpenAI configuration (imported from config.py)
try:
    from config import (
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION
    )
except ImportError:
    print("ERROR: Could not import Azure OpenAI configuration from config.py")
    print("Please ensure config.py exists with valid credentials.")
    exit(1)

# Processing parameters
MAX_ROWS_PER_BATCH = 50
MIN_BATCH_SIZE = 35
MAX_COMPLETION_TOKENS = 20000
INTERMEDIATE_RESULTS_DIR = "extraction_results"
LOG_DIR = "extraction_results"

# Output file prefix (used for log files and CSV output files)
OUTPUT_FILE_PREFIX = "philippines"

# ============================================================================
# DATA MODELS
# ============================================================================

class ProductRow(BaseModel):
    row_id: int = Field(description="The row ID from the input data")
    product_name: str = Field(description="The extracted product name or information")
    manufacturer: str = Field(description="The manufacturer or vendor name of the product")


class ExtractionOutput(BaseModel):
    results: List[ProductRow] = Field(description="List of extracted product information, one per input row")


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir=LOG_DIR):
    """Set up logging to both console and file."""
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{OUTPUT_FILE_PREFIX}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger, log_file_path


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_results_directory():
    """Create intermediate results directory if it doesn't exist."""
    logger = logging.getLogger()
    if not os.path.exists(INTERMEDIATE_RESULTS_DIR):
        os.makedirs(INTERMEDIATE_RESULTS_DIR)
        logger.info(f"Created directory: {INTERMEDIATE_RESULTS_DIR}")
    return INTERMEDIATE_RESULTS_DIR


def save_intermediate_results(df, results, output_file_path, grouping_column):
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
        logger = logging.getLogger()
        logger.error(f"Error saving intermediate results: {e}")
        return False


def format_rows_for_prompt(df_batch, columns, group_value=None, include_group=False, grouping_column=None):
    """Format batch rows for prompt. Uses df_batch.index as row_id to maintain mapping."""
    rows_list = []
    for idx, row in df_batch.iterrows():
        row_data = {}
        for col in columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                row_data[col] = str(value).strip()
        # Note: _group is no longer added to the prompt
        if row_data:
            row_data['_row_id'] = int(idx)
            rows_list.append(row_data)
    return rows_list


def split_group_into_batches(group_df, group_value, max_rows):
    batches = []
    if len(group_df) <= max_rows:
        batches.append({'group_value': group_value, 'rows': group_df})
    else:
        for i in range(0, len(group_df), max_rows):
            batches.append({'group_value': group_value, 'rows': group_df.iloc[i:i + max_rows]})
    return batches


def merge_small_batches(batches, min_size, max_size):
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


def batch_rows_by_group(df, group_column, max_rows_per_batch=MAX_ROWS_PER_BATCH):
    batches = []
    for group_value, group_df in df.groupby(group_column):
        normalized_group = None if (pd.isna(group_value) or str(group_value).strip() == '') else group_value
        batches.extend(split_group_into_batches(group_df, normalized_group, max_rows_per_batch))
    return merge_small_batches(batches, MIN_BATCH_SIZE, max_rows_per_batch)


def call_llm(rows_data, prompt_template):
    """Call LLM with structured output."""
    logger = logging.getLogger()
    rows_text = "\n".join([
        f"Row {row['_row_id']}: {json.dumps({k: v for k, v in row.items() if k not in ['_row_id', '_group']}, ensure_ascii=False)}"
        for row in rows_data
    ])
    prompt = prompt_template.replace("[rows]", rows_text)
    system_message = "You are a data extraction assistant. Extract the requested information from each row."
    
    logger.info(f"  📤 Calling LLM with {len(rows_data)} rows...")
    
    # Log full prompt
    logger.info("  " + "=" * 66)
    logger.info("  FULL PROMPT TO LLM:")
    logger.info("  " + "=" * 66)
    logger.info("  System Message:")
    logger.info("  " + "-" * 66)
    logger.info(f"  {system_message}")
    logger.info("  " + "-" * 66)
    logger.info("  User Prompt:")
    logger.info("  " + "-" * 66)
    logger.info(f"  {prompt}")
    logger.info("  " + "=" * 66)
    
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
        
        # Log full response
        logger.info("  " + "=" * 66)
        logger.info("  FULL RESPONSE FROM LLM:")
        logger.info("  " + "=" * 66)
        logger.info(f"  Number of results: {len(result.results)}")
        logger.info("  " + "-" * 66)
        
        # Format and log each result
        response_data = [{"row_id": r.row_id, "product_name": r.product_name, "manufacturer": r.manufacturer} for r in result.results]
        logger.info(f"  {json.dumps(response_data, indent=2, ensure_ascii=False)}")
        logger.info("  " + "=" * 66)
        logger.info(f"  📥 Received {len(result.results)} results")
        
        return response_data
    except Exception as e:
        logger.error(f"  ❌ Error calling LLM: {str(e)}")
        raise


def process_batches(batches, columns, prompt_template, include_group, df, grouping_column):
    """Process all batches and save intermediate results after each batch."""
    logger = logging.getLogger()
    # Create output directory and generate unique filename for this extraction run
    ensure_results_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{OUTPUT_FILE_PREFIX}_{grouping_column}_{timestamp}.csv"
    output_file_path = os.path.join(INTERMEDIATE_RESULTS_DIR, output_filename)
    
    logger.info(f"💾 Intermediate results will be saved to: {output_file_path}")
    logger.info("")
    
    all_results = []
    try:
        for i, batch in enumerate(batches, 1):
            logger.info(f"📦 Processing batch {i}/{len(batches)}: {len(batch['rows'])} rows")
            rows_data = format_rows_for_prompt(batch['rows'], columns, batch['group_value'], include_group, grouping_column)
            if rows_data:
                results = call_llm(rows_data, prompt_template)
                all_results.extend(results)
                
                # Save intermediate results after each batch
                if save_intermediate_results(df, all_results, output_file_path, grouping_column):
                    logger.info(f"  💾 Saved intermediate results: {len(all_results)} results so far")
                else:
                    logger.warning(f"  ⚠️  Failed to save intermediate results")
            logger.info("")
        
        logger.info(f"✅ Complete: {len(all_results)} results extracted")
        logger.info(f"💾 Final results saved to: {output_file_path}")
    except Exception as e:
        logger.error(f"❌ Error during processing: {str(e)}")
        # Try to save partial results even on error
        if all_results:
            save_intermediate_results(df, all_results, output_file_path, grouping_column)
            logger.info(f"💾 Saved partial results before error: {output_file_path}")
        raise
    
    return all_results, output_file_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Set up logging first
    logger, log_file_path = setup_logging()
    
    logger.info("=" * 70)
    logger.info("AI Product Information Extraction Script")
    logger.info("=" * 70)
    logger.info(f"📝 Log file: {log_file_path}")
    logger.info("")
    
    # Validate configuration
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        logger.error("ERROR: Azure OpenAI credentials not configured in config.py")
        return
    
    if AZURE_OPENAI_API_KEY == "YOUR_API_KEY_HERE" or AZURE_OPENAI_ENDPOINT == "YOUR_ENDPOINT_HERE":
        logger.error("ERROR: Please configure Azure OpenAI credentials in config.py")
        return
    
    # Load CSV file
    if not os.path.exists(INPUT_CSV_FILE):
        logger.error(f"ERROR: Input CSV file not found: {INPUT_CSV_FILE}")
        return
    
    logger.info(f"📂 Loading CSV file: {INPUT_CSV_FILE}")
    df = pd.read_csv(INPUT_CSV_FILE)
    logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info("")
    
    # Validate columns
    missing_columns = [col for col in SELECTED_COLUMNS + [GROUPING_COLUMN] if col not in df.columns]
    if missing_columns:
        logger.error(f"ERROR: The following columns are not found in the CSV: {missing_columns}")
        logger.error(f"Available columns: {list(df.columns)}")
        return
    
    # Validate prompt template
    if '[rows]' not in PROMPT_TEMPLATE:
        logger.error("ERROR: Prompt template must contain '[rows]' placeholder")
        return
    
    # Display configuration
    logger.info("Configuration:")
    logger.info(f"  Selected columns: {SELECTED_COLUMNS}")
    logger.info(f"  Grouping column: {GROUPING_COLUMN}")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Unique groups: {df[GROUPING_COLUMN].nunique()}")
    logger.info(f"  Avg rows/group: {len(df) / df[GROUPING_COLUMN].nunique():.1f}")
    logger.info("")
    
    # Determine if group column should be included
    include_group = GROUPING_COLUMN in SELECTED_COLUMNS
    
    # Create batches
    logger.info("📦 Creating batches...")
    batches = batch_rows_by_group(df, GROUPING_COLUMN, MAX_ROWS_PER_BATCH)
    logger.info(f"✅ Prepared {len(batches)} batch(es)")
    logger.info("")
    
    # Process batches
    try:
        results, output_file_path = process_batches(
            batches, SELECTED_COLUMNS, PROMPT_TEMPLATE, include_group, df, GROUPING_COLUMN
        )
        
        if results:
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ Extraction completed successfully!")
            logger.info(f"   Total results: {len(results)}")
            logger.info(f"   Output file: {output_file_path}")
            logger.info(f"   Log file: {log_file_path}")
            logger.info("=" * 70)
        else:
            logger.error("❌ No results generated.")
    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error(f"❌ Extraction failed: {str(e)}")
        logger.error("=" * 70)
        return


if __name__ == "__main__":
    main()

