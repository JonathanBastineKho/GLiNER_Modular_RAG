import json
import pickle
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset


class GlinerRagDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]):
        """
        Args:
            pairs: list of (jsonl_path, pkl_path) tuples
        """
        self.samples = []
        self.rag_contexts = []

        for jsonl_path, pkl_path in pairs:
            samples = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    samples.append({
                        "tokenized_text": raw["tokens"],
                        "ner": raw.get("ner", []),
                    })

            with open(pkl_path, "rb") as f:
                rag_contexts = pickle.load(f)

            assert len(samples) == len(rag_contexts), (
                f"Mismatch in {jsonl_path}: {len(samples)} samples vs {len(rag_contexts)} RAG contexts"
            )

            self.samples.extend(samples)
            self.rag_contexts.extend(rag_contexts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.rag_contexts[idx]


class GlinerRagCollator:

    def __init__(self, gliner, global_labels: list, max_len: int = 512, 
                 prepare_labels: bool = True, neg_ratio: float = 0.5):
        self.tokenizer = gliner.data_processor.transformer_tokenizer
        self.max_len = max_len
        self.global_labels = global_labels
        self.neg_ratio = neg_ratio

        self.gliner_collator = gliner.data_collator_class(
            gliner.config,
            data_processor=gliner.data_processor,
            prepare_labels=prepare_labels,
        )

    def _sample_labels(self, positive_labels: set) -> list:
        """Mix positive labels with random negatives sampled from global_labels."""
        negatives_pool = [l for l in self.global_labels if l not in positive_labels]

        n_pos = len(positive_labels)
        # neg_ratio=0.5 means 50% of final label set are negatives
        n_neg = int(n_pos * self.neg_ratio / (1 - self.neg_ratio)) if self.neg_ratio < 1.0 else len(negatives_pool)
        n_neg = min(n_neg, len(negatives_pool))

        sampled_negs = random.sample(negatives_pool, n_neg) if n_neg > 0 else []
        combined = list(positive_labels) + sampled_negs
        random.shuffle(combined)
        return combined

    def __call__(self, batch: list) -> dict:
        samples, rag_contexts = zip(*batch)

        label_sets = [
            self._sample_labels({span[2] for span in s["ner"]})
            for s in samples
        ]

        batch_inputs = self.gliner_collator(list(samples), label_sets)

        encoded_ctx = self.tokenizer(
            list(rag_contexts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

        batch_inputs["context_ids"] = encoded_ctx["input_ids"]
        batch_inputs["context_attention_mask"] = encoded_ctx["attention_mask"].long()

        return batch_inputs