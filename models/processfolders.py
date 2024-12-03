import os
import subprocess
import sys

def process_files(input_folder, script, output_folder):
    # Ensure the input folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    # Ensure the script file exists
    if not os.path.isfile(script):
        print(f"Error: Script '{script}' does not exist.")
        sys.exit(1)

    # Ensure the output folder exists, or create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)

        # Skip if it's not a file
        if not os.path.isfile(input_file):
            continue

        # Define the output file path
        output_file = os.path.join(output_folder, filename)

        # Run the script on the file
        try:
            with open(output_file, "w") as outfile:
                subprocess.run(
                    [sys.executable, script, input_file],
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    check=True
                )
            print(f"Processed '{filename}' -> '{output_file}'")
        except subprocess.CalledProcessError as e:
            print(f"Error processing '{filename}': {e.stderr.decode()}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_folder.py <input_folder> <script> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    script = sys.argv[2]
    output_folder = sys.argv[3]

    process_files(input_folder, script, output_folder)
