import json, random, torch
from pathlib import Path
from src import RAGRetriever
from src.models import GLiNERRagCrossAttn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = Path("training_dataset/json_pileNER.json1")
BATCH_SIZE, STEPS, MAX_SAMPLES = 8, 300, 2000

# Example raw JSON sample:
# {"text":"today is a sunny day","entities":[{"start":11,"end":16,"label":"weather"}]}
# Example GLiNER format after conversion:
# {"tokenized_text":["today","is","a","sunny","day"],"ner":[[3,3,"weather"]]}
# (char span 11..16 for "sunny" becomes token span 3..3)
def to_sample(r):
    text, words, p, spans = r["text"], r["text"].split(), 0, []

    # Find each word in the original text to find their character spans. 
    # 'sunny' becomes indexed in span(4) = (11,16)
    for w in words:
        s = text.find(w, p); 
        e = s + len(w); 
        spans.append((s, e)); p = e
    
    ner = []

    # Find in dict r the key "entities", i.e. searching for the label "weather"
    # To convert entities":[{"start":11,"end":16,"label":"weather"}]} -> ner = [[3,3,"weather"]]} 
    for ent in r.get("entities", []): 
        hit = [i for i, (a, b) in enumerate(spans) if b > ent["start"] and a < ent["end"]]
        if hit: ner.append([hit[0], hit[-1], ent["label"]])  #  ner = [[3,3,"weather"]]} 
    return {"tokenized_text": words, "ner": ner}

# Only pass tensors to device, keep other batch items (like id_to_classes dict) on CPU for collator decoding.
def to_dev(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


# Prepare data for training
raw = [json.loads(x) for x in DATA_PATH.open(encoding="utf-8") if x.strip()][:MAX_SAMPLES]
data = [to_sample(x) for x in raw]
# Clean up data by checking if it has at least one entity annotation, else skip sample.
data = [x for x in data if x["ner"]]

# To create a global label set for the taining run
label_set = set()
for sample in data:
    for span in sample["ner"]:
        label_set.add(span[2])
labels = sorted(label_set)

model = GLiNERRagCrossAttn("urchade/gliner_large-v1").to(DEVICE).train()

# Checks always that the gliner parameters are frozen and only the cross-attention is trained. Else, stop script.
assert not any(p.requires_grad for p in model.gliner.parameters())  

# Instatiating collator as the object of the class, data_collator_class
# Set prepare_labels=True for training so the collator builds label tensors
# while assembling sample dicts into a model-ready batch.
collator = model.gliner.data_collator_class(
    model.gliner.config, data_processor=model.gliner.data_processor, prepare_labels=True
)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.context_cross_attn.parameters(), lr=1e-4)

# Build RAG context once for every sample before training.
retriever = RAGRetriever(k=3)
queries = []
for sample in data:
    query_text = " ".join(sample["tokenized_text"])
    queries.append(query_text)

contexts = []
for query_text in queries:
    context_text = retriever.retrieve_context(query_text)
    contexts.append(context_text)

print("RAG retriever enabled.")


# Start training proper
for step in range(STEPS):
    idxs = random.sample(range(len(data)), min(BATCH_SIZE, len(data)))
    batch_items = [data[i] for i in idxs]

    # Collator takes a list of sample dicts and converts/pads them into a batched tensor dictionary.
    batch = to_dev(collator(batch_items, entity_types=labels))

    # Use retrieved context for cross-attention.
    ctx_text = [contexts[i] for i in idxs]
    ctx = model.tokenizer(
        ctx_text, return_tensors="pt", padding=True, truncation=True, max_length=model.gliner.config.max_len
    )

    # clear old gradients from previous step.
    optimizer.zero_grad(); 

    # forward pass: model(...) calls __call__ -> forward(...) internally.
    logits = model(
        context_ids=ctx["input_ids"].to(DEVICE),
        context_attention_mask=ctx["attention_mask"].to(DEVICE),
        **batch,
    )
    # compare predictions vs gold labels to get scalar training loss.
    loss = criterion(logits, batch["labels"].float())

    # backprop: compute gradients for trainable params (cross-attn here).
    loss.backward(); 
    
    # optimizer step: update parameters using computed gradients.
    optimizer.step()
    
    if step % 20 == 0: print(f"step={step} loss={loss.item():.4f}")

Path("models").mkdir(exist_ok=True)
torch.save(model.context_cross_attn.state_dict(), "models/cross_attn.pt")
print("saved models/cross_attn.pt (GLiNER frozen, cross-attn trained)")
