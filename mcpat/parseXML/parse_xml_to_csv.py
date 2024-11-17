import xml.etree.ElementTree as ET
import pandas as pd

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
            data.append({"component_id": component_id, "component_name": component_name, "type": "param", "name": name, "value": value})

        for stat in component.findall("stat"):
            name = stat.get("name")
            value = stat.get("value")
            data.append({"component_id": component_id, "component_name": component_name, "type": "stat", "name": name, "value": value})

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")

# Usage
if __name__ == "__main__":
    input_xml = "../ProcessorDescriptionFiles/Xeon.xml"  # Relative path to XML file
    output_csv = "xeon_inputs.csv"  # Output will be created in parseXML folder
    parse_xml_to_csv(input_xml, output_csv)

