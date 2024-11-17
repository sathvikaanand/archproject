import pandas as pd
import re

# File path to the McPAT output
file_path = '/Users/sathv/archproject/mcpat/ExampleResults/T1_DC_64'

main_section_pattern = re.compile(r"^\*{20,}\n([A-Za-z0-9 /()]+):", re.MULTILINE)
metric_pattern = re.compile(r"^\s{2,}([A-Za-z /()]+) = ([\d.eE+-]+) (W|mm\^2)?$", re.MULTILINE)
subsection_pattern = re.compile(r"^\s{4,}([A-Za-z /()]+):", re.MULTILINE)

# Data storage
data = {
    "Section": [],
    "Subsection": [],
    "Metric": [],
    "Value": [],
    "Unit": []
}

# Load content of file
with open(file_path, 'r') as file:
    content = file.read()

# Find each main section and parse it separately
main_sections = list(main_section_pattern.finditer(content))
main_sections.append(None)  # Add None to handle the end of the file for last section

for i in range(len(main_sections) - 1):
    start_pos = main_sections[i].end()
    section_name = main_sections[i].group(1).strip()
    end_pos = main_sections[i + 1].start() if main_sections[i + 1] else len(content)
    section_content = content[start_pos:end_pos]

    # Capture top-level metrics in this main section
    for top_level_metric in metric_pattern.finditer(section_content):
        metric, value, unit = top_level_metric.groups()
        data["Section"].append(section_name)
        data["Subsection"].append("")  # Top-level
        data["Metric"].append(metric.strip())
        data["Value"].append(float(value))
        data["Unit"].append(unit.strip() if unit else "")

    # Capture subsections and their metrics
    subsections = list(subsection_pattern.finditer(section_content))
    subsections.append(None)  # Handle end of section for last subsection

    for j in range(len(subsections) - 1):
        subsection_start = subsections[j].end()
        subsection_name = subsections[j].group(1).strip()
        subsection_end = subsections[j + 1].start() if subsections[j + 1] else len(section_content)
        subsection_content = section_content[subsection_start:subsection_end]

        for metric_match in metric_pattern.finditer(subsection_content):
            metric, value, unit = metric_match.groups()
            data["Section"].append(section_name)
            data["Subsection"].append(subsection_name)
            data["Metric"].append(metric.strip())
            data["Value"].append(float(value))
            data["Unit"].append(unit.strip() if unit else "")

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
output_csv = 'mcpat_parsed.csv'
df.to_csv(output_csv, index=False)

print(f"Data extracted and saved to {output_csv}")
