# -*- coding: utf-8 -*-


import os
import json, random, torch
import time
from pathlib import Path
from src import RAGRetriever
from src import GLiNERRagCrossAttn
import tqdm


PATH_TRAIN_DATA = Path("data/combined_dataset/train.jsonl")
PATH_VAL_DATA = Path("data/combined_dataset/validation.jsonl")
PATH_TEST_DATA = Path("data/combined_dataset/test.jsonl")
PATH_TRAIN_DATA_RAG = Path('data/combined_dataset/train_w_rag.pkl')
PATH_VAL_DATA_RAG = Path('data/combined_dataset/val_w_rag.pkl')
PATH_TEST_DATA_RAG = Path('data/combined_dataset/test_w_rag.pkl')

BATCH_SIZE, EPOCHS = 32, 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Only pass tensors to device, keep other batch items (like id_to_classes dict) on CPU for collator decoding.
def to_dev(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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
                # Ensure 'text' key is present, either from original or reconstructed
                if "text" not in sample and "tokens" in sample:
                    sample["text"] = " ".join(sample["tokens"])
                elif "text" not in sample and "tokenized_text" in sample: # Fallback if 'tokens' already renamed
                    sample["text"] = " ".join(sample["tokenized_text"])

                sample["tokenized_text"] = sample.pop("tokens") # Rename 'tokens' to 'tokenized_text'
                dataset.append(sample)
    if max_samples:
        dataset = dataset[:max_samples]
    return dataset


def fn_rag_context (data, retriever, num_print=5):
    contexts = []
    for i, sample in enumerate(data):
        query_text = " ".join(sample["tokenized_text"])
        retrieved_context = retriever.retrieve_context(query_text)
        contexts.append(retrieved_context)

        if i < num_print:
            print(f"{i}. {query_text}")
            print(f"{i}. {retrieved_context}")
            print("-" * 50)
    return contexts

def binary_accuracy(preds, y):
  """
  Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
  """
  #round predictions to the closest integer
  rounded_preds = torch.round(torch.sigmoid(preds))
  correct = (rounded_preds == y).float() #convert into float for division
  acc = correct.sum() / len(correct)
  return acc

def binary_prf(epoch_TP, epoch_FP, epoch_FN):
    precision = epoch_TP / (epoch_TP + epoch_FP + 1e-8)
    recall    = epoch_TP / (epoch_TP + epoch_FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def binary_prf_counts(logits, targets, threshold=0.5):
  probs = torch.sigmoid(logits)
  preds = (probs > threshold).float()

  TP = ((preds == 1) & (targets == 1)).sum().item()
  FP = ((preds == 1) & (targets == 0)).sum().item()
  FN = ((preds == 0) & (targets == 1)).sum().item()

  return TP, FP, FN

# --- 1. Data Loading ---
print("Loading datasets...")
# Should remove the MAX_SAMPLES for complete training
train_data = load_dataset(PATH_TRAIN_DATA)
val_data = load_dataset(PATH_VAL_DATA)
test_data = load_dataset(PATH_TEST_DATA)
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(val_data)}')
print(f'Number of testing examples: {len(test_data)}')

# To create a global label set for the taining run
# TPL: THIS IS A BIG WRONG WRONG. WE should not be aggregating labels. Every example has its own set of entity types (labels).
# labels = fn_aggre_labels(train_data)
# print(f"Target labels: {labels}")

# This cell is to be run once only! After rag-retrieval is done, it is saved to the .pkl script
# Subsequently, we just need to open the .pkl script to to see what has been retrieved by the rag, rather than running the rag every single launch of the script

#import pickle

# Build RAG context once for every sample before training to eliminate bottleneck of retrieving context every training step.
retriever = RAGRetriever(k=3)
print("RAG retriever enabled.")


print("RAG retriever done for train/val/test data.")


import pickle
with open(PATH_TRAIN_DATA_RAG, 'rb') as f:
  train_data_rag = pickle.load(f)

with open(PATH_VAL_DATA_RAG, 'rb') as f:
  val_data_rag = pickle.load(f)

with open(PATH_TEST_DATA_RAG, 'rb') as f:
  test_data_rag = pickle.load(f)

def fn_score_postproc(output_logits, input_targets):

  # 1. Reshape logits: [BATCH_SIZE, START_POS, WIDTH_SPAN, CLASS]: [B, 56, 12, 13] → [B, 672, 13]
  B, S, W, C = output_logits.shape
  logits = output_logits.view(B, S * W, C)  # [B, 672, 13]

  # 2. Targets already match flattened span layout
  # TPL: THIS IS A BIG WRONG WRONG
  targets = input_targets["labels"].float()  # [B, 672, 13]

  # 3. Apply span mask to every batch to remove invalid spans and hence invalid scores
  span_mask = input_targets["span_mask"].unsqueeze(-1)  # [B, 672, 1]
  logits = logits[span_mask.bool().expand_as(logits)] # model’s score: [B, 672, 1]
  targets = targets[span_mask.bool().expand_as(targets)] # target: 0 or 1, [B, 672, 13]

  return logits, targets

def fn_run_epoch(model: GLiNERRagCrossAttn, collator, data, data_rag, BATCH_SIZE, optimizer, DEVICE=DEVICE,train=True, threshold=0.5):
    epoch_loss, epoch_TP, epoch_FP, epoch_FN = 0, 0, 0, 0

    if train:
        model.train()
    else:
        model.eval()

    all_idxs = list(range(len(data)))
    # if train:
    #     random.shuffle(all_idxs)

    nbatches = len(all_idxs) // BATCH_SIZE
    if len(all_idxs) % BATCH_SIZE != 0:
        nbatches += 1

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for i in tqdm.trange(nbatches):

            # ===== Prepare batch =====
            idxs = all_idxs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            data_b = [data[j] for j in idxs]
            data_rag_b = [data_rag[j] for j in idxs]

            batch_inputs, batch_ctx_ids, batch_ctx_mask, span_target_label = fn_prepare_batch(model, collator, data_b, data_rag_b)

            # span_target_label = batch_inputs["labels"].float()  # [B, 672, 13]

            if train:
                optimizer.zero_grad()

            # ===== Forward =====
            logit_outputs = model(batch_ctx_ids, batch_ctx_mask, **batch_inputs)
            B, S, W, C = logit_outputs.shape
            reshaped_logits_output = logit_outputs.view(B, S * W, C)  # [B, 672, 13]
            # logits, targets = fn_score_postproc(output_logits=logit_outputs,input_targets=batch_inputs)

            # ===== Compute pos_weightage =====
            total_pos = (span_target_label == 1).sum().item()
            total_neg = (span_target_label == 0).sum().item()
            pos_weight_value = total_neg / max(total_pos, 1)
            print(f'pos weight value: {pos_weight_value}')
            criterion = torch.nn.BCEWithLogitsLoss(reduction="mean",pos_weight=torch.tensor([pos_weight_value], device=DEVICE))
            loss = criterion(reshaped_logits_output, span_target_label)

            # ===== Metrics =====
            TP, FP, FN = binary_prf_counts(reshaped_logits_output, span_target_label, threshold=threshold)

            # ===== Backprop (train only) =====
            if train:
                loss.backward()
                optimizer.step()

            # ===== Accumulate =====
            epoch_loss += loss.item()
            epoch_TP += TP
            epoch_FP += FP
            epoch_FN += FN

    # ===== Final metrics =====
    precision = epoch_TP / (epoch_TP + epoch_FP + 1e-8)
    recall    = epoch_TP / (epoch_TP + epoch_FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    epoch_loss = epoch_loss / nbatches

    return epoch_loss, precision, recall, f1


def fn_prepare_batch(model: GLiNERRagCrossAttn, collator, train_data_b, train_data_rag_b):
    """
    Prepare the input batch for the model, including tokenization, collating, and RAG context processing.
    as well as the output labels for backprop.
    """
    label_b = [[et[2] for et in x['ner']] for x in train_data_b]  # THIS IS MOST LIKELY WRONG
    print("label_b:", label_b)
    batch_inputs = to_dev(collator(train_data_b, label_b))


    # Use retrieved context for cross-attention.
    batch_ctx = model.tokenizer(
        train_data_rag_b, return_tensors="pt", padding=True, truncation=True, max_length=model.gliner.config.max_len
    )

    batch_ctx_ids = batch_ctx["input_ids"].to(DEVICE)
    batch_ctx_mask = batch_ctx["attention_mask"].to(DEVICE).long()

    if batch_ctx_ids.dim() == 1:
        batch_ctx_ids = batch_ctx_ids.unsqueeze(0)
    if batch_ctx_mask.dim() == 1:
        batch_ctx_mask = batch_ctx_mask.unsqueeze(0)

    batch_inputs["attention_mask"] = batch_inputs["attention_mask"].long()
    assert batch_inputs["input_ids"].shape == batch_inputs["attention_mask"].shape
    print(f"Batch input", batch_inputs.keys())
    # print("batch_inputs[labels]", batch_inputs["labels"], batch_inputs["labels"].shape)   # 32, 864, 3
    assert batch_ctx_ids.shape == batch_ctx_mask.shape


    # prepare output for backprop
    # TPL: FOR NOW I BELIEVE THAT ENTITY TYPES ARE MEANT TO BE PLACED IN THE 'ner' FIELD OF train_data_b, 
    # and that entity_types is optional and JoJo is wrong to use it. 

    raw_batch = model.gliner.data_processor.collate_raw_batch(train_data_b)

    print("raw_batch from collator:")
    print("raw_batch keys:", raw_batch.keys())
    print("raw_batch['span_label']:", raw_batch['span_label'])
    print("raw_batch['span_label'].shape:", raw_batch['span_label'].shape)
    # print(raw_batch['seq_length'])
    # print(raw_batch['span_idx'])
    # print("raw_batch['tokens']:", raw_batch['tokens'])
    # print("raw_batch['entities']:", raw_batch['entities'])
    # print("raw_batch['classes_to_id']:", raw_batch['classes_to_id'])
    # print(raw_batch['id_to_classes'])


    raw_batch2 = {"tokens": raw_batch["tokens"], "entities": raw_batch["entities"], "classes_to_id": raw_batch["classes_to_id"]}
    # print(type(model.gliner.data_processor))
    span_labels = model.gliner.data_processor.create_labels(raw_batch2)

    # print("span_labels", span_labels)
    print("span_labels.shape", span_labels.shape)  # 32, 864, 5

    return batch_inputs, batch_ctx_ids, batch_ctx_mask, span_labels

# Load GLiNER
model = GLiNERRagCrossAttn("urchade/gliner_large-v1").to(DEVICE)

# Checks always that the gliner parameters are frozen and only the cross-attention is trained. Else, stop script.
assert not any(p.requires_grad for p in model.gliner.parameters())

# Instatiating collator as the object of the class, data_collator_class
# Set prepare_labels=True for training so the collator builds label tensors
# while assembling sample dicts into a model-ready batch.
# collator = model.gliner.data_collator_class(
#     model.gliner.config, data_processor=model.gliner.data_processor, prepare_labels=True, return_tokens=True,
#             return_id_to_classes=True)

collator = model.collator

import time

criterion = torch.nn.BCEWithLogitsLoss( reduction='mean')
optimizer = torch.optim.AdamW(model.context_cross_attn.parameters(), lr=2e-4)

# --- 4. Epoch-Based Train & Validate Loop ---
save_dir = Path("models")
save_dir.mkdir(exist_ok=True)
#save_path = save_dir/"cross_attn_best.pt"

best_val_loss = float('inf')
threshold = 0.8


# Start training proper
for epoch in range(1, EPOCHS+1):

    start_time = time.perf_counter()
    train_loss, train_precision, train_recall, train_f1 = fn_run_epoch(model, collator, train_data, train_data_rag, BATCH_SIZE, optimizer, DEVICE=DEVICE,train=True, threshold=threshold)
    val_loss, val_precision, val_recall, val_f1         = fn_run_epoch(model, collator, val_data, val_data_rag, BATCH_SIZE, optimizer, DEVICE=DEVICE,train=False, threshold=threshold)
    end_time = time.perf_counter()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        #torch.save(model.context_cross_attn.state_dict(), save_path)

    print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f} | Train Precision: {train_precision:.3f} | Train Recall: {train_recall:.3f} | Train F1: {train_f1:.3f} | ')
    print(f'Val. Loss:  {val_loss:.3f}   | Val. Precision:  {val_precision:.3f} | Val. Recall: {val_recall:.3f}  | Val. F1: {val_f1:.3f} | ')

print(f"\nTraining complete. Best model weights saved to {save_path}")


# Load saved model
# --- 4. Epoch-Based Train & Validate Loop ---
# save_dir = Path("src/models")
# save_dir.mkdir(exist_ok=True)
# save_path = save_dir/"cross_attn_best.pt"
model.context_cross_attn.load_state_dict(torch.load(save_path))

best_val_f1 = float('inf')

# Sweep threshold to see which is the best for f1 score
for i in range(1,10):
  start_thresh = 0.995
  threshold = (start_thresh)+ (1-start_thresh)*i/10
  print(f'Current threshold: {threshold}')
  val_loss, val_precision, val_recall, val_f1 = fn_run_epoch(model, collator, val_data, val_data_rag, BATCH_SIZE, optimizer, DEVICE=DEVICE,train=False, threshold=threshold)
  print(f'Val. Loss:  {val_loss:.3f}   | Val. Precision:  {val_precision:.3f} | Val. Recall: {val_recall:.3f}  | Val. F1: {val_f1:.3f} | ')

  if val_f1 > best_val_f1:
    best_threshold = threshold

test_loss, test_precision, test_recall, test_f1 = fn_run_epoch(model, collator, test_data, test_data_rag, BATCH_SIZE, optimizer, DEVICE=DEVICE,train=False, threshold=threshold)

print(f'Epoch: 1 | Epoch Time: {epoch_mins}m {epoch_secs}s')
print(f'Test Loss: {test_loss:.3f} | Test Precision: {test_precision:.3f} | Test Recall: {test_recall:.3f} | Test F1: {test_f1:.3f} | ')

