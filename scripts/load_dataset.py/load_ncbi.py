import json
import os

def process_to_gliner_format(example, entity_label):
    tokens = example["tokens"]
    tags = example["ner_tags"]
    
    ner_spans = []
    
    current_start_idx = None
    current_end_idx = None
    
    for i, tag in enumerate(tags):
        # Handle both string tags ("B-DISEASE") and integer tags (1)
        is_b_tag = (tag == 1) or (isinstance(tag, str) and tag.startswith("B-"))
        is_i_tag = (tag == 2) or (isinstance(tag, str) and tag.startswith("I-"))
        
        if is_b_tag:
            if current_start_idx is not None:
                ner_spans.append([current_start_idx, current_end_idx, entity_label])
            
            current_start_idx = i
            current_end_idx = i
            
        elif is_i_tag:
            if current_start_idx is not None:
                current_end_idx = i
            else:
                current_start_idx = i
                current_end_idx = i
                
        else: # It's an 'O' or 0
            if current_start_idx is not None:
                ner_spans.append([current_start_idx, current_end_idx, entity_label])
                current_start_idx = None
                current_end_idx = None
                
    # Catch any entity that ends on the very last word of the sentence
    if current_start_idx is not None:
        ner_spans.append([current_start_idx, current_end_idx, entity_label])
        
    return {
        "tokens": tokens,
        "ner": ner_spans
    }

def convert_all_splits(dataset_dir, dataset, splits, entity_label):
    """Reads the raw JSONL files for all splits, formats them, and saves them in the same directory."""
    for split in splits:
        input_path = os.path.join(dataset_dir, f"{dataset}_{split}_raw.jsonl")
        
        # Save to the exact same folder, but append _processed so we don't overwrite the raw data
        output_path = os.path.join(dataset_dir, f"{split}_processed.jsonl")
        
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue
            
        processed_count = 0
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                    
                example = json.loads(line)
                formatted_example = process_to_gliner_format(example, entity_label)
                
                outfile.write(json.dumps(formatted_example, ensure_ascii=False) + "\n")
                processed_count += 1
                
        print(f"Processed {processed_count} lines: {input_path} -> {output_path}")

if __name__ == "__main__":
    # Define the splits we want to process
    splits_to_process = ["train", "validation", "test"]
    
    print("Processing ncbi...")
    convert_all_splits(
        dataset_dir="data/ncbi/raw", 
        dataset="ncbi",
        splits=splits_to_process, 
        entity_label="disease"
    )