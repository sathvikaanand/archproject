import os
from xml.etree import ElementTree as ET

def sweep_device_type(input_file, output_dir, device_values):
    """
    Sweep the device_type parameter in an XML file and generate new XML files.

    :param input_file: Path to the input XML file.
    :param output_dir: Directory to save the modified XML files.
    :param device_values: List of device_type values to iterate through.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Locate the device_type parameter in the XML
    param_found = False
    for param in root.iter('param'):
        if param.get('name') == 'device_type':
            param_found = True
            # Sweep the device_type values
            for value in device_values:
                param.set('value', str(value))
                output_file = os.path.join(output_dir, f"{os.path.basename(input_file).split('.')[0]}_device_{value}.xml")
                tree.write(output_file)
                print(f"Generated: {output_file}")
            break

    if not param_found:
        print("device_type parameter not found in the XML file.")

if __name__ == "__main__":
    # Configuration for each processor
    configs = [
        {"file": "ProcessorDescriptionFiles/ARM_A9_2GHz.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/ARM_A9_2GHz_withIOC.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/Alpha21364.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/Niagara1.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/Niagara1_sharing_DC.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/Niagara1_sharing_SBT.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/Niagara1_sharing_ST.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/Niagara2.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/Penryn.xml", "device_values": [0, 1, 2]},
        {"file": "ProcessorDescriptionFiles/Xeon.xml", "device_values": [0, 1, 2]},
    ]

    # Output directory
    output_base_dir = "ProcessorDescriptionFiles/ExtraXMLDeviceType"  # Change to your desired output directory

    # Sweep device_type for each processor
    for config in configs:
        input_file = config["file"]
        output_dir = os.path.join(output_base_dir, os.path.basename(input_file).split('.')[0])
        sweep_device_type(input_file, output_dir, config["device_values"])
