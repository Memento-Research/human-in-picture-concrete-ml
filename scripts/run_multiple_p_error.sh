#!/bin/bash

# ANSI escape codes for colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# List of images size to pass to the program
image_sizes=( "96"   "96"  "96"   "96"  "96"   )
n_bits=(      "6"    "6"   "6"    "6"   "6"    )
p_errors=(    "0.05" "0.1" "0.15" "0.2" "0.25" )

for ((i=0; i<${#image_sizes[@]}; i++)); do
    for ((j=0; j<10; j++)); do
        echo -e "${GREEN}Running program with image size ${image_sizes[i]}, n_bits ${n_bits[i]}, p_error ${p_errors[i]}, using SIM${NC}"
        poetry run python3 src/ConvolutionalNeuralNetwork.py "${image_sizes[i]}" "${n_bits[i]}" "${p_errors[i]}" "use_sim" "${j}" "/p_error"
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}Program finished successfully.${NC}"
            continue
        fi

        echo -e "${RED}Program exited with code 1.${NC}"
        rm -rf .artifacts
    done
done
