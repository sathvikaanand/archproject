#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Get command-line arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find and process all XML files in subdirectories
find "$INPUT_DIR" -type f -name "*.xml" | while read -r INPUT_FILE; do
    # Extract the base filename without extension
    BASE_NAME=$(basename "$INPUT_FILE" .xml)
    OUTPUT_FILE="$OUTPUT_DIR/${BASE_NAME}_output.txt"
    
    echo "Processing: $INPUT_FILE"
    ./mcpat -infile "$INPUT_FILE" -print_level 2 > "$OUTPUT_FILE"
    
    echo "Saved output to: $OUTPUT_FILE"
done

echo "McPAT processing complete."

