import json, random, torch
from pathlib import Path
from src import RAGRetriever
from src import GLiNERRagCrossAttn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_PATH = Path("data/combined_dataset/train.jsonl")
VAL_DATA_PATH = Path("data/combined_dataset/validation.jsonl")
BATCH_SIZE, EPOCHS, MAX_SAMPLES = 1, 32, 8


# Only pass tensors to device, keep other batch items (like id_to_classes dict) on CPU for collator decoding.
def to_dev(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}

def align_for_bce(logits: torch.Tensor, target: torch.Tensor):
    # BCEWithLogitsLoss requires logits and target to have exactly the same shape.
    if logits.shape == target.shape:
        return logits, target

    # Common GLiNER case: labels are [B, L*K, C] while logits are [B, L, K, C].
    if target.dim() == 3 and logits.dim() == 4:
        b, lk, c = target.shape
        _, l, k, c_logits = logits.shape
        if lk == l * k and c == c_logits:
            return logits, target.view(b, l, k, c)

    # Reverse case: logits flattened but labels are 4D.
    if target.dim() == 4 and logits.dim() == 3:
        b, l, k, c = target.shape
        b2, lk, c_logits = logits.shape
        if b == b2 and lk == l * k and c == c_logits:
            return logits.view(b, l, k, c), target

    raise RuntimeError(
        f"Label/logit shape mismatch for BCE: logits={tuple(logits.shape)}, target={tuple(target.shape)}"
    )


# Prepare data for training
# raw = [json.loads(x) for x in DATA_PATH.open(encoding="utf-8") if x.strip()][:MAX_SAMPLES]
# data = [to_sample(x) for x in raw]
# # Clean up data by checking if it has at least one entity annotation, else skip sample.
# data = [x for x in data if x["ner"]]

def load_dataset(filepath, max_samples=None):
    """Helper function to load and format JSONL data."""
    dataset = []
    if not filepath.exists():
        print(f"Warning: File not found -> {filepath}")
        return dataset
        
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            sample = json.loads(line)
            if "ner" in sample:
                sample["tokenized_text"] = sample.pop("tokens")
                dataset.append(sample)
    if max_samples:
        dataset = dataset[:max_samples]
    return dataset

# --- 1. Data Loading ---
print("Loading datasets...") 
# Should remove the MAX_SAMPLES for complete training
train_data = load_dataset(TRAIN_DATA_PATH, MAX_SAMPLES)
val_data = load_dataset(VAL_DATA_PATH, MAX_SAMPLES)
              
# To create a global label set for the taining run
label_set = set()
for sample in train_data:
    for span in sample["ner"]:
        label_set.add(span[2])
labels = sorted(label_set)
print(f"Target labels: {labels}")

# --- 2. Model Setup ---
model = GLiNERRagCrossAttn("urchade/gliner_large-v1").to(DEVICE)

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

# Build RAG context once for every sample before training to eliminate bottleneck of retrieving context every training step.
retriever = RAGRetriever(k=3)

train_contexts = []
for sample in train_data:
    query_text = " ".join(sample["tokenized_text"])
    retrieved_context = retriever.retrieve_context(query_text)
    train_contexts.append(retrieved_context)

val_contexts = []
for sample in val_data:
    query_text = " ".join(sample["tokenized_text"])
    retrieved_context = retriever.retrieve_context(query_text)
    val_contexts.append(retrieved_context)

print("RAG retriever enabled.")

# --- 4. Epoch-Based Train & Validate Loop ---
save_dir = Path("models")
save_dir.mkdir(exist_ok=True)
save_path = save_dir/"cross_attn_best.pt"

best_val_loss = float('inf') # Track the best loss

# Start training proper
for epoch in range(1, EPOCHS+1):
    
    # TRAINING PHASE
    model.train()
    train_loss = 0.0
    num_train_batches = 0
    
    # Create a list of all indices and shuffle it once per epoch
    all_idxs = list(range(len(train_data)))
    random.shuffle(all_idxs)
    
    # Iterate through the shuffled list in chunks of BATCH_SIZE
    for i in range(0, len(all_idxs), BATCH_SIZE):
        idxs = all_idxs[i:i+BATCH_SIZE]
        batch_items = [train_data[i] for i in idxs]

        # Collator takes a list of sample dicts and converts/pads them into a batched tensor dictionary, of which some form VALID_KEYS
        # Collator is aprt of __call__
        batch_entity_types = [labels for _ in batch_items]
        batch = to_dev(collator(batch_items, entity_types=batch_entity_types))

        # Use retrieved context for cross-attention.
        ctx_text = [train_contexts[i] for i in idxs]
        ctx = model.tokenizer(
            ctx_text, return_tensors="pt", padding=True, truncation=True, max_length=model.gliner.config.max_len
        )

        ctx_ids = ctx["input_ids"].to(DEVICE)
        ctx_mask = ctx["attention_mask"].to(DEVICE).long()

        if ctx_ids.dim() == 1:
            ctx_ids = ctx_ids.unsqueeze(0)
        if ctx_mask.dim() == 1:
            ctx_mask = ctx_mask.unsqueeze(0)


        batch["attention_mask"] = batch["attention_mask"].long()
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert ctx_ids.shape == ctx_mask.shape

        # clear old gradients from previous step.
        optimizer.zero_grad(); 

        # forward pass: model(...) calls __call__ -> forward(...) internally.
        logits = model(
            context_ids=ctx_ids,
            context_attention_mask=ctx_mask,
            **batch,
        )

        target = batch["labels"].float()
        logits, target = align_for_bce(logits, target)

        # print some status along the way
        # print(f"logits shape={tuple(logits.shape)} target shape={tuple(target.shape)}")

        loss = criterion(logits, target)

        # backprop: compute gradients for trainable params (cross-attn here).
        loss.backward(); 
        
        # optimizer step: update parameters using computed gradients.
        optimizer.step()

        train_loss += loss.item()
        num_train_batches += 1
           
    avg_train_loss = train_loss / num_train_batches
    
    # VALIDATION PHASE
    model.eval()
    val_loss = 0.0
    num_val_batches = 0
    val_idxs = list(range(len(val_data)))
    
    with torch.no_grad(): # Disable gradient math to save massive amounts of memory/time
        for i in range(0, len(val_idxs), BATCH_SIZE):
            idxs = val_idxs[i : i + BATCH_SIZE]
            batch_items = [val_data[idx] for idx in idxs]
            batch_entity_types = [labels for _ in batch_items]
            
            batch = to_dev(collator(batch_items, entity_types=batch_entity_types))

            ctx_text = [val_contexts[idx] for idx in idxs]
            ctx = model.tokenizer(ctx_text, return_tensors="pt", padding=True, truncation=True, max_length=model.gliner.config.max_len)
            
            ctx_ids = ctx["input_ids"].to(DEVICE)
            ctx_mask = ctx["attention_mask"].to(DEVICE).long()
            
            if ctx_ids.dim() == 1:
                ctx_ids = ctx_ids.unsqueeze(0)
            if ctx_mask.dim() == 1:
                ctx_mask = ctx_mask.unsqueeze(0)

            batch["attention_mask"] = batch["attention_mask"].long()

            logits = model(context_ids=ctx_ids, context_attention_mask=ctx_mask, **batch)
            target = batch["labels"].float()
            logits, target = align_for_bce(logits, target)
            
            loss = criterion(logits, target)
            val_loss += loss.item()
            num_val_batches += 1

    avg_val_loss = val_loss / num_val_batches
    
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    
    # CHECKPOINTING
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.context_cross_attn.state_dict(), save_path)
        
    print("-" * 40)
    
print(f"\nTraining complete. Best model weights saved to {save_path}")
