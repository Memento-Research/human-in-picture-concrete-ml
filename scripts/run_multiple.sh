##!/bin/bash
#
## ANSI escape codes for colors
#RED='\033[0;31m'
#GREEN='\033[0;32m'
#NC='\033[0m' # No Color
#
## List of images size to pass to the program
#image_sizes=(   "32"    "64"    "96"    "128"   "64"  "64"  "64"   "64"   "64"   "64"   "64"   "96"      "96"    "96"    "96"  "96"  )
#n_bits=(        "6"     "6"     "6"     "7"     "4"   "6"   "5"    "7"    "8"    "9"    "10"   "6"       "6"     "6"     "6"   "6"   )
#p_errors=(      "0.1"   "0.1"   "0.1"   "0.2"   "0.1" "0.1" "0.1"  "0.1"  "0.1"  "0.1"  "0.1"  "0.05"    "0.1"   "0.15"  "0.2" "0.25" )
#times=(         1       1       1       1       10    10    10     10     10     10     10     10        10      10      10    10 )
#use_fhe=(       "T"     "T"     "T"     "T"     "F"   "F"   "F"    "F"    "F"    "F"    "F"    "F"       "F"     "F"     "F"   "F" )
#
#for ((i=0; i<${#image_sizes[@]}; i++)); do
#    for ((j=0; j<${times[i]}; j++)); do
#        echo -e "${GREEN}Running program with image size ${image_sizes[i]}, n_bits ${n_bits[i]}, p_error ${p_errors[i]}, use_fhe ${use_fhe[i]} for the ${j}th time.${NC}"
#        poetry run python3 src/ConvolutionalNeuralNetwork.py "${image_sizes[i]}" "${n_bits[i]}" "${p_errors[i]}" "${use_fhe[i]}" "${j}"
#        exit_code=$?
#
#        if [ $exit_code -eq 0 ]; then
#            echo -e "${GREEN}Program finished successfully.${NC}"
#            continue
#        fi
#
#        echo -e "${RED}Program exited with code 1.${NC}"
#        rm -rf .artifacts
#    done
#done
