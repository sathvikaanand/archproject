import argparse
import os
import xml.etree.ElementTree as ET
import pandas as pd
''' python3 /Users/sathv/archproject/models/processfolders.py 
            /Users/sathv/archproject/mcpat/ProcessorDescriptionFiles 
            /Users/sathv/archproject/mcpat/parseXML/parse_xml_to_csv.py 
            /Users/sathv/archproject/outputs/mcpatinput'''
def parse_xml_to_csv(xml_file, output_csv):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Prepare a list to store the extracted data
    data = []

    # Extract <param> and <stat> attributes
    for component in root.findall(".//component"):
        component_id = component.get("id", "N/A")
        component_name = component.get("name", "N/A")

        for param in component.findall("param"):
            name = param.get("name")
            value = param.get("value")
            #if value.isdigit():
            try: 
                data.append({"component_id": component_id, "component_name": component_name, "type": "param", "name": name, "value": float(value)})
            except ValueError:
                continue
                #data.append({"component_id": component_id, "component_name": component_name, "type": "param", "name": name, "value": None})
        
        for stat in component.findall("stat"):
            name = stat.get("name")
            value = stat.get("value")
            try: 
                data.append({"component_id": component_id, "component_name": component_name, "type": "stat", "name": name, "value": float(value)})
            except ValueError:
                continue
                #data.append({"component_id": component_id, "component_name": component_name, "type": "stat", "name": name, "value": None})
        
    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse an XML file and save the data as a CSV.")
    parser.add_argument("input_xml", type=str, help="Path to the input XML file.")
    args = parser.parse_args()

    # Automatically generate output file name based on input XML file
    base_name = os.path.splitext(os.path.basename(args.input_xml))[0]
    output_csv = f"{base_name}_xml_parsed.csv"

    # Parse the XML file and save to CSV
    parse_xml_to_csv(args.input_xml, output_csv)

if __name__ == "__main__":
    main()
