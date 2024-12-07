#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Get the directories from the command-line arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run extract.py with the provided directories
echo "Running extract_common_align.py with input: $INPUT_DIR and output: $OUTPUT_DIR"
python ./mcpat/filterCSV/extract_common_align.py "$INPUT_DIR" "$OUTPUT_DIR"

echo "Extraction complete."
