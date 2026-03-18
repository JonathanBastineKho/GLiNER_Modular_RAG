import json
import os
import random

def combine_and_shuffle(splits, output_dir):
    """
    Reads the processed GLiNER files for both datasets, combines the 
    matching splits, shuffles the rows, and saves them to a new directory.
    """
    # Create the target directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a seed so the random shuffle is reproducible 
    random.seed(42) 
    
    for split in splits:
        combined_lines = []
        
        # Define the paths to the files we generated in the previous step
        anatem_path = f"data/AnatEM/{split}_processed.jsonl"
        ncbi_path = f"data/NCBI/{split}_processed.jsonl"
        
        # 1. Read AnatEM data
        if os.path.exists(anatem_path):
            with open(anatem_path, "r", encoding="utf-8") as f:
                combined_lines.extend(f.readlines())
        else:
            print(f"Missing {anatem_path}")
                
        # 2. Read NCBI data
        if os.path.exists(ncbi_path):
            with open(ncbi_path, "r", encoding="utf-8") as f:
                combined_lines.extend(f.readlines())
        else:
            print(f"Missing {ncbi_path}")
                
        # 3. Shuffle the combined data thoroughly
        random.shuffle(combined_lines)
        
        # 4. Write to the new combined file
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(combined_lines)
            
        print(f"Created {output_path} with {len(combined_lines)} mixed examples.")

if __name__ == "__main__":
    # The splits we want to merge
    splits_to_process = ["train", "validation", "test"]
    
    print("Merging and shuffling datasets...")
    combine_and_shuffle(
        splits=splits_to_process, 
        output_dir="data/combined_dataset"
    )