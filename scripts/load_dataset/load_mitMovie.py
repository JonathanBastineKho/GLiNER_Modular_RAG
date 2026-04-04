import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mitMovie")

# Load label mapping: id -> label_name (without B-/I- prefix)
with open(os.path.join(DATA_DIR, "label.jsonl")) as f:
    id_to_label = {}
    for label_name, label_id in json.loads(f.readline()).items():
        id_to_label[label_id] = label_name


def convert(tags, tokens):
    """Convert BIO tags to span-based NER format."""
    ner = []
    i = 0
    while i < len(tags):
        label = id_to_label.get(tags[i], "O")
        if label.startswith("B-"):
            entity_type = label[2:]
            start = i
            end = i
            # Consume following I- tags of the same type
            while end + 1 < len(tags):
                next_label = id_to_label.get(tags[end + 1], "O")
                if next_label == f"I-{entity_type}":
                    end += 1
                else:
                    break
            ner.append([start, end, entity_type])
            i = end + 1
        else:
            i += 1
    return ner


for split in ["train", "test", "valid"]:
    input_path = os.path.join(DATA_DIR, f"{split}.jsonl")
    output_path = os.path.join(DATA_DIR, f"{split}_processed.jsonl")
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line)
            ner = convert(record["tags"], record["tokens"])
            fout.write(json.dumps({"tokens": record["tokens"], "ner": ner}) + "\n")
    print(f"Processed {split} -> {output_path}")
