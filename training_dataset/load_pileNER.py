from datasets import load_dataset

dataset = load_dataset(
    "Universal-NER/Pile-NER-definition",
    split="train",
    streaming=True   # avoids dataset script
)

print(next(iter(dataset)))


import ast
def convert(example):
    text = example["conversations"][0]["value"].replace("Text:", "").strip()
    all_entities = []

    # Iterate through conversations to find entity extractions
    for i in range(len(example["conversations"])):
        # Look for a 'human' turn asking "What describes X in the text?"
        # followed by a 'gpt' turn providing a list of entities.
        if example["conversations"][i]["from"] == "human" and \
           example["conversations"][i]["value"].startswith("What describes") and \
           i + 1 < len(example["conversations"]) and \
           example["conversations"][i+1]["from"] == "gpt":

            question_value = example["conversations"][i]["value"]
            # Extract the label from the human's question (e.g., "medical condition")
            try:
                label_start_idx = question_value.find("What describes ") + len("What describes ")
                label_end_idx = question_value.find(" in the text?", label_start_idx)
                if label_start_idx != -1 and label_end_idx != -1:
                    label = question_value[label_start_idx:label_end_idx].strip()
                else:
                    continue # Could not parse the label, skip this conversation pair
            except Exception: # Catch any parsing errors
                continue

            labels_str = example["conversations"][i+1]["value"]
            try:
                # ast.literal_eval will parse the string representation of a list (e.g., '["entity1", "entity2"]')
                extracted_entity_texts = ast.literal_eval(labels_str)
                if isinstance(extracted_entity_texts, list):
                    for entity_text_val in extracted_entity_texts:
                        # Find all occurrences of the entity_text_val in the main text
                        current_start_index = 0
                        while True:
                            start = text.find(entity_text_val, current_start_index)
                            if start == -1:
                                break
                            end = start + len(entity_text_val)
                            all_entities.append({"start": start, "end": end, "label": label})
                            current_start_index = end # Continue searching from after this found entity
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails (e.g., it's not a list, or malformed), ignore this entry
                pass

    return {"text": text, "entities": all_entities}

data = []
for i, example in enumerate(dataset):
    if i > 2000:   # small subset first
        break
    data.append(convert(example))


import json

with open("json_pileNER.json1", "w") as f:
    for sample in data:
        json.dump(sample, f)
        f.write("\n")