import os
import requests

_URLS = {
    "conll2003": {
        "train": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/conll2003/train.txt",
        "validation": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/conll2003/dev.txt",
        "test": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/conll2003/test.txt",
    },
    "politics": {
        "train": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/politics/train.txt",
        "validation": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/politics/dev.txt",
        "test": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/politics/test.txt",
    },
    "science": {
        "train": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/science/train.txt",
        "validation": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/science/dev.txt",
        "test": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/science/test.txt",
    },
    "music": {
        "train": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/music/train.txt",
        "validation": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/music/dev.txt",
        "test": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/music/test.txt",
    },
    "literature": {
        "train": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/literature/train.txt",
        "validation": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/literature/dev.txt",
        "test": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/literature/test.txt",
    },
    "ai": {
        "train": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/ai/train.txt",
        "validation": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/ai/dev.txt",
        "test": "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/ai/test.txt",
    },
}

def download_file(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✓ Downloaded: {save_path}")
    except Exception as e:
        print(f"✗ Failed: {url} -> {e}")


def main(root_dir="crossNER_data"):
    os.makedirs(root_dir, exist_ok=True)

    for dataset_name, splits in _URLS.items():
        dataset_dir = os.path.join(root_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        for split_name, url in splits.items():
            filename = f"{split_name}.txt"
            save_path = os.path.join(dataset_dir, filename)

            download_file(url, save_path)


if __name__ == "__main__":
    main()