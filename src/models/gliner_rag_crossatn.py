import torch
import torch.nn as nn
from gliner import GLiNER

from src.components import ContextCrossAttn

class GLiNERRagCrossAttn(nn.Module):

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

        hidden_dim = self.gliner.config.hidden_size
        self.context_cross_attn = ContextCrossAttn(hidden_dim=hidden_dim)
        self.context_encoder = self.gliner.model.token_rep_layer

        for param in self.gliner.parameters():
            param.requires_grad = False

    def _encode_context(self, context_ids: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.context_encoder(context_ids, context_mask)  # (B, L, D)
    
    def forward(self, context_ids=None, context_attention_mask=None, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in self.VALID_KEYS}

        prompts_embedding, prompts_embedding_mask, word_embeds, word_mask = (
            self.gliner.model.get_representations(
                filtered["input_ids"],
                filtered["attention_mask"],
                filtered["text_lengths"],
                filtered["words_mask"],
            )
        )

        # Inject context via cross-attention + gate
        if context_ids is not None:
            context_embeds = self._encode_context(context_ids, context_attention_mask)
            context_pad_mask = (context_attention_mask == 0) if context_attention_mask is not None else None
            word_embeds = self.context_cross_attn(
                word_embeds=word_embeds,
                context_embeds=context_embeds,
                context_mask=context_pad_mask,
            )

        # Fit lengths
        target_W = filtered["span_idx"].size(1) // self.gliner.config.max_width
        word_embeds, word_mask = self.gliner.model._fit_length(word_embeds, word_mask, target_W)

        target_C = prompts_embedding.size(1)
        prompts_embedding, prompts_embedding_mask = self.gliner.model._fit_length(
            prompts_embedding, prompts_embedding_mask, target_C
        )

        # Score spans against prompts
        span_idx = filtered["span_idx"] * filtered["span_mask"].unsqueeze(-1)
        span_rep = self.gliner.model.span_rep_layer(word_embeds, span_idx)

        prompts_embedding = self.gliner.model.prompt_rep_layer(prompts_embedding)
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)

        return scores
    
    def predict_entities(self, text: str, labels: list, context: str = "", threshold: float = 0.5, flat_ner: bool = True):
        input_x = [{"tokenized_text": text.split(), "ner": None}]
        batch = self.collator(input_x, entity_types=labels)

        context_ids = context_attn_mask = None
        if context:
            encoded = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=self.gliner.config.max_len,
            )
            context_ids = encoded["input_ids"]
            context_attn_mask = encoded["attention_mask"]

        logits = self.forward(
            context_ids=context_ids,
            context_attention_mask=context_attn_mask,
            **batch,
        )

        entities = self.gliner.decoder.decode(
            batch["tokens"],
            batch["id_to_classes"],
            logits,
            flat_ner=flat_ner,
            threshold=threshold,
        )

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