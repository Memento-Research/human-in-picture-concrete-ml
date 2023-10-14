#!/bin/bash

source cnn_venv/bin/activate

# ANSI escape codes for colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

IMAGE_SIZE="64"
N_BITS="6"
P_ERROR="0.1"

# Remove .artifacts at the beginning
rm -rf .artifacts

python3.10 src/ConvolutionalNeuralNetwork.py "$IMAGE_SIZE" "$N_BITS" "$P_ERROR" "T" 1
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}Program finished successfully.${NC}"
    exit 0
fi

echo -e "${RED}Program exited with code ${exit_code}.${NC}"

echo
rm -rf .artifacts
