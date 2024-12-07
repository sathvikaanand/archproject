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

# Process all .txt files in the input directory
for INPUT_FILE in "$INPUT_DIR"/*.txt; do
    if [ -f "$INPUT_FILE" ]; then
        echo "Processing: $INPUT_FILE"
        python ./models/scrapedata.py "$INPUT_FILE" "$OUTPUT_DIR"
        echo "Saved to: $OUTPUT_DIR"
    else
        echo "No .txt files found in $INPUT_DIR."
        break
    fi
done

echo "Data extraction complete."
