import os
import subprocess

# Define directories
base_input_dir = "ProcessorDescriptionFiles/ExtraXMLClockRate"  # Base directory containing processor folders
output_dir = "ExtraOutputFiles"  # Directory to save output files
mcpat_executable = "./mcpat"  # Path to the McPAT executable

# Processor to test (e.g., Xeon)
test_processor_name = "Xeon"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Recursively process XML files
for root, dirs, files in os.walk(base_input_dir):
    for xml_file in files:
        if xml_file.endswith(".xml"):
            # Check if the file is related to the specified processor
            if test_processor_name.lower() in root.lower() or test_processor_name.lower() in xml_file.lower():
                # Full path to the input file
                input_path = os.path.join(root, xml_file)

                # Create a mirrored directory structure in the output folder
                relative_path = os.path.relpath(root, base_input_dir)
                processor_output_dir = os.path.join(output_dir, relative_path)
                os.makedirs(processor_output_dir, exist_ok=True)

                # Output file path
                output_file = f"{os.path.splitext(xml_file)[0]}_output.txt"
                output_path = os.path.join(processor_output_dir, output_file)

                # Command to run McPAT
                command = f"{mcpat_executable} -infile {input_path} -print_level 5 > {output_path}"

                print(f"Running McPAT for: {input_path}")
                try:
                    # Run McPAT using subprocess
                    subprocess.run(command, shell=True, check=True)
                    print(f"Output saved to: {output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error while processing {xml_file}: {e}")

