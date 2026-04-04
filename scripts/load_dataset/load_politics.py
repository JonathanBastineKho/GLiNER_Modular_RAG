import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "crossNER_data", "politics")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "crossNER_politics")


def parse_conll(filepath):
    sentences = []
    tokens = []
    tags = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
            else:
                parts = line.split("\t") if "\t" in line else line.split()
                tokens.append(parts[0])
                tags.append(parts[1])
        if tokens:
            sentences.append((tokens, tags))
    return sentences


def bio_to_spans(tokens, tags):
    ner = []
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            entity_type = tags[i][2:]
            start = i
            end = i
            while end + 1 < len(tags) and tags[end + 1] == f"I-{entity_type}":
                end += 1
            ner.append([start, end, entity_type])
            i = end + 1
        else:
            i += 1
    return ner


def convert_file(input_path, output_path):
    sentences = parse_conll(input_path)
    with open(output_path, "w") as fout:
        for tokens, tags in sentences:
            ner = bio_to_spans(tokens, tags)
            fout.write(json.dumps({"tokens": tokens, "ner": ner}) + "\n")
    print(f"Processed {input_path} -> {output_path} ({len(sentences)} sentences)")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split_in, split_out in [("train", "train"), ("test", "test"), ("validation", "valid")]:
        convert_file(
            os.path.join(DATA_DIR, f"{split_in}.txt"),
            os.path.join(OUTPUT_DIR, f"{split_out}_processed.jsonl"),
        )
