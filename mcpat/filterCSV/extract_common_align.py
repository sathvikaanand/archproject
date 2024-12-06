import pandas as pd
import os
from pathlib import Path

def align_features(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Identify all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # Step 2: Find common features
    common_features = None
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(input_path)

        # Extract unique 'name' values from 'param' rows
        features = set(df[df['type'] == 'param']['name'])

        # Intersect features across all files
        if common_features is None:
            common_features = features
        else:
            common_features = common_features.intersection(features)

    print(f"Common features found: {len(common_features)}")

    # Step 3: Create aligned CSVs for each processor
    total_files_processed = 0
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        filename_without_ext = os.path.splitext(csv_file)[0]
        new_filename = f"{filename_without_ext}_aligned.csv"
        output_path = os.path.join(output_dir, new_filename)

        # Read the current file
        df = pd.read_csv(input_path)

        # Filter for common features
        filtered_df = df[df['name'].isin(common_features)]

        # Drop duplicates by 'name' to ensure unique entries
        filtered_df = filtered_df.drop_duplicates(subset='name', keep='first')

        # Align the data by creating a DataFrame with all common features
        aligned_df = pd.DataFrame({'name': list(common_features)})
        aligned_df = aligned_df.merge(filtered_df[['name', 'value']], on='name', how='left')

        # Save the aligned file
        aligned_df.to_csv(output_path, index=False)

        # Print stats
        total_params = len(df)
        aligned_params = len(filtered_df)
        print(f"\nProcessing: {csv_file}")
        print(f"Output as: {new_filename}")
        print(f"Original parameters: {total_params}")
        print(f"Parameters aligned: {aligned_params}")

        total_files_processed += 1

    print(f"\nTotal files processed: {total_files_processed}")
    print(f"Aligned files saved to: {output_dir}")

if __name__ == "__main__":
    # Set input and output directories
    input_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpatinput/"  # Replace with your input directory
    output_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpatinput_aligned/"  # Replace with your output directory
    align_features(input_dir, output_dir)
