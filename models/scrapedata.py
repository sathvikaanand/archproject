import argparse
import os
import re
import csv

''' python3 
        /Users/sathv/archproject/models/processfolders.py 
        /Users/sathv/archproject/mcpat/processorresults 
        /Users/sathv/archproject/models/scrapedata.py 
        /Users/sathv/archproject/outputs  
'''
def parse_mcpat_file(file_path):
    result = []
    hierarchy = []
    
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith(" "): 
                line += ":"
            # Skip empty lines, separators, or warnings
            if not line.strip() or line.startswith("*") or "Warning" in line:
                continue
            
            # Match hierarchical components (e.g., "Processor:", "Core:", "Memory Controller:")
            if ":" in line and "=" not in line:
                block_name = line.split(":")[0].strip()
                # If it's a new top-level section (like L2, NOC), reset hierarchy
                if not line.startswith(" "):  # Top-level component (e.g., Core, Processor, L2)
                    hierarchy = [block_name]
                else:  # Subcomponent
                    hierarchy.append(block_name)
            elif "=" in line:
                 # Match key-value pairs (e.g., "Area = 127.712 mm^2")
                key, value = map(str.strip, line.split("=", 1))
                
                # Separate the value (number) and the unit using regex
                match = re.match(r"([0-9.]+)(\s*[a-zA-Z\^0-9]+)?", value)
                if match:
                    number = match.group(1)  # Extract numeric part
                    unit = match.group(2).strip() if match.group(2) else ''  # Extract unit part
                # else:
                #     number = None
                #     unit = value
                
                    path = " > ".join(hierarchy)
                    result.append((path, key, unit, number))
            elif not ":" in line and not "=" in line and hierarchy:
                # End of the current hierarchy when no colon or equals sign is found
                hierarchy.pop()
    
    return result

def write_csv(data, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Component", "Metric", "Unit", "Value"])
        for row in data:
            writer.writerow(row)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse an McPAT output file and save the data as a CSV.")
    parser.add_argument("input_file", type=str, help="Path to the input McPAT file.")
    args = parser.parse_args()

    # Automatically generate output file name based on input file
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_file = f"{base_name}_parsed.csv"

    # Parse the input file and write to the output CSV
    parsed_data = parse_mcpat_file(args.input_file)
    write_csv(parsed_data, output_file)

    print(f"Data has been successfully parsed and saved to {output_file}")

if __name__ == "__main__":
    main()
