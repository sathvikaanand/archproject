import pandas as pd
import os
from pathlib import Path

def filter_nonzero_params(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    total_files_processed = 0
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        filename_without_ext = os.path.splitext(csv_file)[0]
        new_filename = f"{filename_without_ext}_filtered.csv"
        output_path = os.path.join(output_dir, new_filename)
        
        #read csv
        df = pd.read_csv(input_path)
        
        filtered_df = df[df['value'] != 0.0]
    
        filtered_df.to_csv(output_path, index=False)
        
        # print stats
        total_params = len(df)
        filtered_params = len(filtered_df)
        removed_params = total_params - filtered_params
        
        print(f"\nProcessing: {csv_file}")
        print(f"Output as: {new_filename}")
        print(f"Original parameters: {total_params}")
        print(f"Parameters after filtering: {filtered_params}")
        print(f"Parameters removed: {removed_params}")
        
        total_files_processed += 1
    
    print(f"\nTotal files processed: {total_files_processed}")
    print(f"Filtered files saved to: {output_dir}")

if __name__ == "__main__":
    input_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpatinput/"
    output_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpatinputfiltered/"
    filter_nonzero_params(input_dir, output_dir)