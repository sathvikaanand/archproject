import os
from xml.etree import ElementTree as ET

def sweep_clock_rate(input_file, output_dir, start, end, step):
    """
    Sweep the clock_rate parameter in an XML file and generates new XML files.

    :param input_file: Path to the input XML file.
    :param output_dir: Directory to save the modified XML files.
    :param start: Starting clock rate value.
    :param end: Ending clock rate value.
    :param step: Step to increment the clock rate.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Locate the clock_rate parameter in the XML
    param_found = False
    for param in root.iter('param'):
        if param.get('name') == 'clock_rate':
            param_found = True
            # Sweep the clock_rate values
            for value in range(start, end + step, step):
                param.set('value', str(value))
                output_file = os.path.join(output_dir, f"{os.path.basename(input_file).split('.')[0]}_clock_{value}.txt")
                tree.write(output_file)
                print(f"Generated: {output_file}")
            break

    if not param_found:
        print("clock_rate parameter not found in the XML file.")

if __name__ == "__main__":
    # Configuration for each processor
    configs = [
        {"file": "ProcessorDescriptionFiles/ARM_A9_2GHz.xml", "start": 1500, "end": 2500, "step": 100},
        {"file": "ProcessorDescriptionFiles/ARM_A9_2GHz_withIOC.xml", "start": 1500, "end": 2500, "step": 100},
        {"file": "ProcessorDescriptionFiles/Alpha21364.xml", "start": 800, "end": 1600, "step": 100},
        {"file": "ProcessorDescriptionFiles/Niagara1.xml", "start": 800, "end": 1600, "step": 100},
        {"file": "ProcessorDescriptionFiles/Niagara1_sharing_DC.xml", "start": 3000, "end": 4000, "step": 100},
        {"file": "ProcessorDescriptionFiles/Niagara1_sharing_SBT.xml", "start": 3000, "end": 4000, "step": 100},
        {"file": "ProcessorDescriptionFiles/Niagara1_sharing_ST.xml", "start": 3000, "end": 4000, "step": 100},
        {"file": "ProcessorDescriptionFiles/Niagara2.xml", "start": 1000, "end": 1800, "step": 100},
        {"file": "ProcessorDescriptionFiles/Penryn.xml", "start": 2200, "end": 3000, "step": 100},
        {"file": "ProcessorDescriptionFiles/Xeon.xml", "start": 3000, "end": 3800, "step": 100},
    ]

    # Output directory
    output_base_dir = "ProcessorDescriptionFiles/ExtraXMLClockRate"  # Change to your desired output directory

    # Sweep clock_rate for each processor
    for config in configs:
        input_file = config["file"]
        output_dir = os.path.join(output_base_dir, os.path.basename(input_file).split('.')[0])
        sweep_clock_rate(input_file, output_dir, config["start"], config["end"], config["step"])

