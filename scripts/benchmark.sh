#!/bin/bash

source cnn_venv/bin/activate

# ANSI escape codes for colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color


python3.10 src/DataProcessing.py
exit_code=$?

if [ $exit_code -ne 1 ]; then
    echo -e "${GREEN}Program finished successfully.${NC}"
    exit 0
fi

echo -e "${RED}Program exited with code 1.${NC}"
