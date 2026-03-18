import json
from pathlib import Path

def parse_conll_file(filepath):
    """Parses a single .conll file and returns a list of formatted sentence dictionaries."""
    sentences = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        tokens = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "tags": tags})
                    tokens = []
                    tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1])
        
        if tokens:
            sentences.append({"tokens": tokens, "tags": tags})

    formatted_data = []
    
    for sentence in sentences:
        tokens = sentence["tokens"]
        tags = sentence["tags"]
        entities = []
        current_entity = None
        
        for i, tag in enumerate(tags):
            if tag == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                label = tag[2:].lower()
                current_entity = [i, i, label]
            elif tag.startswith('I-'):
                if current_entity and current_entity[2] == tag[2:].lower():
                    current_entity[1] = i
                else:
                    if current_entity:
                        entities.append(current_entity)
                    label = tag[2:].lower()
                    current_entity = [i, i, label]
        
        if current_entity:
            entities.append(current_entity)
            
        formatted_data.append({"tokens": tokens, "ner": entities})
        
    return formatted_data

def consolidate_splits(input_base_dir, output_base_dir):
    """Combines multiple .conll files per split into single .jsonl files."""
    base_in = Path(input_base_dir)
    base_out = Path(output_base_dir)
    
    base_out.mkdir(parents=True, exist_ok=True)
    
    # Updated mapping to output .jsonl extensions
    split_mapping = {
        'train': 'train_processed.jsonl',
        'devel': 'validation_processed.jsonl',
        'test': 'test_processed.jsonl'
    }
    
    for input_folder_name, output_filename in split_mapping.items():
        folder_path = base_in / input_folder_name
        output_filepath = base_out / output_filename
        
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Skipping: Directory not found -> {folder_path}")
            continue
            
        conll_files = list(folder_path.glob('*.conll'))
        num_files = len(conll_files)
        
        if num_files == 0:
            print(f"No .conll files found in {folder_path}")
            continue
            
        # Print the exact number of files found in the directory for cross-checking
        print(f"Found {num_files} .conll files in '{input_folder_name}/'. Processing...")
        
        all_formatted_data = []
        
        for conll_file in conll_files:
            file_data = parse_conll_file(conll_file)
            all_formatted_data.extend(file_data)
            
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for item in all_formatted_data:
                f.write(json.dumps(item) + '\n')
                
        print(f"-> Successfully saved {len(all_formatted_data)} combined sentences to {output_filepath}\n")

if __name__ == "__main__":
    # The folder where train/, devel/, and test/ are located
    input_directory = "./data/AnatEM/AnatEM/raw" 
    
    # The new target folder for the consolidated files
    output_directory = "./data/AnatEM" 
    
    consolidate_splits(input_directory, output_directory)