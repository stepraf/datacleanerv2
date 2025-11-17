#!/bin/bash
# Script to remove Azure secrets from git history
# Note: This script should read secrets from environment variables or config files
# DO NOT hardcode secrets in this file

# Read secrets from environment variables or use placeholders
SECRET_KEY="${AZURE_OPENAI_API_KEY:-PLACEHOLDER_KEY}"
SECRET_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-PLACEHOLDER_ENDPOINT}"

# Function to clean a file
clean_file() {
    local file="$1"
    if [ -f "$file" ]; then
        # Replace the secret API key line
        sed -i "s|AZURE_OPENAI_API_KEY = \".*\"|# Secret removed - import from config.py|g" "$file"
        # Replace the secret endpoint line  
        sed -i "s|AZURE_OPENAI_ENDPOINT = \".*\"|# Secret removed - import from config.py|g" "$file"
    fi
}

# Clean both files
clean_file "tabs/ai_simplification.py"
clean_file "tabs/ai_advanced.py"

