#!/bin/bash

source cnn_venv/bin/activate

# ANSI escape codes for colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# List of images size to pass to the program
image_sizes=( 32 64 96 )

for image_size in "${image_sizes[@]}"; do
    # Remove .artifacts at the beginning
    rm -rf .artifacts

    python3 src/ConvolutionalNeuralNetwork.py "$image_size"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Program finished successfully.${NC}"
        continue
    fi

    echo -e "${RED}Program exited with code 1.${NC}"
    rm -rf .artifacts

done
