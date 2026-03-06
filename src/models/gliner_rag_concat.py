import torch
import torch.nn as nn
from gliner import GLiNER

class GLiNERRagConcat(nn.Module):
    
    VALID_KEYS = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask", "labels"}

    def __init__(self, model_name: str = "urchade/gliner_large-v1"):
        super().__init__()
        self.gliner = GLiNER.from_pretrained(model_name)
        self.tokenizer = self.gliner.data_processor.transformer_tokenizer
        self.collator = self.gliner.data_collator_class(
            self.gliner.config,
            data_processor=self.gliner.data_processor,
            prepare_labels=False,
            return_tokens=True,
            return_id_to_classes=True,
        )

        for param in self.gliner.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in self.VALID_KEYS}
        return self.gliner.model(**filtered)

    def predict_entities(self, text: str, labels: list, context: str = "", threshold: float = 0.5, flat_ner: bool = True):
        # Build base batch
        input_x = [{"tokenized_text": text.split(), "ner": None}]
        batch = self.collator(input_x, entity_types=labels)

        # Append context tokens if provided
        if context:
            original_len = batch["input_ids"].shape[1]
            max_context_len = self.gliner.config.max_len - original_len
            
            context_ids = self.tokenizer(
                context,
                add_special_tokens=False,
                truncation=True,
                max_length=max_context_len
            )["input_ids"]
            sep_id = self.tokenizer.sep_token_id
            suffix = torch.tensor([[*context_ids, sep_id]])

            batch["input_ids"] = torch.cat([batch["input_ids"][:, :-1], suffix], dim=1)
            batch["attention_mask"] = torch.cat([batch["attention_mask"][:, :-1], torch.ones(1, suffix.shape[1])], dim=1)

        # Forward pass
        output = self(**batch)

        entities = self.gliner.decoder.decode(
            batch["tokens"],
            batch["id_to_classes"],
            output.logits,
            flat_ner=flat_ner,
            threshold=threshold,
        )

        # Convert Span objects to dicts to match vanilla GLiNER interface
        words = text.split()
        return [
            {
                "text": " ".join(words[span.start:span.end + 1]),
                "label": span.entity_type,
                "score": span.score,
                "start": span.start,
                "end": span.end,
            }
            for span in entities[0]
    ]