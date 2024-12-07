#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <parent_directory> <output_directory>"
    exit 1
fi

# Get the directories from the command-line arguments
PARENT_DIR="$1"
OUTPUT_DIR="$2"

# Check if the parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Directory '$PARENT_DIR' does not exist."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find and process all XML files in subdirectories
find "$PARENT_DIR" -type f -name "*.xml" | while read -r XML_FILE; do
    echo "Processing: $XML_FILE"
    python ./mcpat/parseXML/parse_xml_to_csv.py "$XML_FILE" "$OUTPUT_DIR"
    echo "Saved to: $OUTPUT_DIR"
done

echo "XML parsing complete."
