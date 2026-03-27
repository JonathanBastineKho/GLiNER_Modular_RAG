import os

import json, random, torch
import time
from pathlib import Path
from src import RAGRetriever
from src import GLiNERRagCrossAttn
import tqdm
import pickle
import time

PATH_TRAIN_DATA = Path("data/combined_dataset/train.jsonl")
PATH_VAL_DATA = Path("data/combined_dataset/validation.jsonl")
PATH_TEST_DATA = Path("data/combined_dataset/test.jsonl")
PATH_TRAIN_DATA_RAG = Path('data/combined_dataset/train_w_rag.pkl')
PATH_VAL_DATA_RAG = Path('data/combined_dataset/val_w_rag.pkl')
PATH_TEST_DATA_RAG = Path('data/combined_dataset/test_w_rag.pkl')

BATCH_SIZE, EPOCHS = 64, 30
WARMUP_EPOCHS = 2
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

    # Common GLiNER case: global_labels are [B, L*K, C] while logits are [B, L, K, C].
    if target.dim() == 3 and logits.dim() == 4:
        b, lk, c = target.shape
        _, l, k, c_logits = logits.shape
        if lk == l * k and c == c_logits:
            return logits, target.view(b, l, k, c)

    # Reverse case: logits flattened but global_labels are 4D.
    if target.dim() == 4 and logits.dim() == 3:
        b, l, k, c = target.shape
        b2, lk, c_logits = logits.shape
        if b == b2 and lk == l * k and c == c_logits:
            return logits.view(b, l, k, c), target

    raise RuntimeError(
        f"Label/logit shape mismatch for BCE: logits={tuple(logits.shape)}, target={tuple(target.shape)}"
    )

# Prepare data for training
def load_dataset(filepath, max_samples=None):
    """Helper function to load and format JSONL data."""
    dataset = []
    if not filepath.exists():
        print(f"Warning: File not found -> {filepath}")
        return dataset
    i = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            sample = json.loads(line)
            sample["tokenized_text"] = sample.pop("tokens")
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

def fn_aggre_global_labels (data):
  label_set = set()
  for sample in data:
    for span in sample["ner"]:
        label_set.add(span[2])
  global_labels = sorted(label_set)
  return global_labels

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
global_labels = fn_aggre_global_labels(train_data)
print(f"Target global_labels: {global_labels}")

# This cell is to be run once only! After rag-retrieval is done, it is saved to the .pkl script
# Subsequently, we just need to open the .pkl script to to see what has been retrieved by the rag, rather than running the rag every single launch of the script

#import pickle

# Build RAG context once for every sample before training to eliminate bottleneck of retrieving context every training step.
# retriever = RAGRetriever(k=3)
print("RAG retriever enabled.")

# train_data_rag = []
# train_data_rag = fn_rag_context(train_data, retriever, num_print=5)

#train_data_rag = []
#train_data_rag = fn_rag_context(train_data, retriever, num_print=5)
# val_data_rag = []
# val_data_rag = fn_rag_context(val_data, retriever, num_print=5)

#val_data_rag = []
#val_data_rag = fn_rag_context(val_data, retriever, num_print=5)
print("RAG retriever done.")
# test_data_rag = []
# test_data_rag = fn_rag_context(test_data, retriever, num_print=5)

print("RAG retriever done for train/val/test data.")

# with open(PATH_TRAIN_DATA_RAG, 'wb') as f:
#     pickle.dump(train_data_rag, f)

# with open(PATH_VAL_DATA_RAG, 'wb') as f:
#     pickle.dump(val_data_rag, f)

# with open(PATH_TEST_DATA_RAG, 'wb') as f:
#     pickle.dump(test_data_rag, f)

with open(PATH_TRAIN_DATA_RAG, 'rb') as f:
  train_data_rag = pickle.load(f)

with open(PATH_VAL_DATA_RAG, 'rb') as f:
  val_data_rag = pickle.load(f)

with open(PATH_TEST_DATA_RAG, 'rb') as f:
  test_data_rag = pickle.load(f)

def fn_score_postproc (output_logits, input_targets):

  # 1. Reshape logits: [BATCH_SIZE, START_POS, WIDTH_SPAN, CLASS]: [B, 56, 12, 13] → [B, 672, 13]
  B, S, W, C = output_logits.shape
  logits = output_logits.view(B, S * W, C)  # [B, 672, 13]

  # 2. Targets already match flattened span layout
#   labels = input_targets["labels"].detach().cpu()

#   torch.set_printoptions(threshold=10_000_000, linewidth=200)  # avoid "..."
#   with open("labels_dump.txt", "a", encoding="utf-8") as f:
#     f.write(f"labels shape: {tuple(labels.shape)}\n")
#     f.write(str(labels))
#     f.write("\n\n")
#   torch.set_printoptions(profile="default")


#   print(f"logits: {logits}")
#   print(logits.shape)
 
#   print(f"input_targets['labels']: {input_targets['labels']}")
#   print(input_targets['labels'].shape)
#   print(input_targets['labels'])
  targets = input_targets["labels"].float()  # [B, 672, 13]
  
#   torch.set_printoptions(threshold=10_000_000, linewidth=200)  # avoid "..."
#   with open("labels_dump.txt", "a", encoding="utf-8") as f:
#     f.write(f"labels shape: {tuple(targets.shape)}\n")
#     f.write(str(targets))
#     f.write("\n\n")
#   torch.set_printoptions(profile="default")

#   print(f"targets: {targets}")
#   print(targets.shape)

  # 3. Apply span mask to every batch to remove invalid spans and hence invalid scores
  span_mask = input_targets["span_mask"].unsqueeze(-1)  # [B, 672, 1]
  logits = logits[span_mask.bool().expand_as(logits)] # model’s score: [B, 672, 1]
  targets = targets[span_mask.bool().expand_as(targets)] # target: 0 or 1, [B, 672, 13]

  return logits, targets

def fn_prepare_batch(model, collator, data_b, data_rag_b, train=True):
  # No Aggregate global labels since negative samples not working for the dataset now
  label_b = [list(set([et[2] for et in x.get('ner', [])])) for x in data_b]
            
  batch_inputs = to_dev(collator(data_b, label_b))

  # Use retrieved context for cross-attention.
  batch_ctx = model.tokenizer(
      data_rag_b, return_tensors="pt", padding=True, truncation=True, max_length=model.gliner.config.max_len
  )

  batch_ctx_ids = batch_ctx["input_ids"].to(DEVICE)
  batch_ctx_mask = batch_ctx["attention_mask"].to(DEVICE).long()

  if batch_ctx_ids.dim() == 1:
      batch_ctx_ids = batch_ctx_ids.unsqueeze(0)
  if batch_ctx_mask.dim() == 1:
      batch_ctx_mask = batch_ctx_mask.unsqueeze(0)

  batch_inputs["attention_mask"] = batch_inputs["attention_mask"].long()
  assert batch_inputs["input_ids"].shape == batch_inputs["attention_mask"].shape
  assert batch_ctx_ids.shape == batch_ctx_mask.shape

  return batch_inputs, batch_ctx_ids, batch_ctx_mask

def fn_run_epoch(model, collator, data, data_rag, BATCH_SIZE, optimizer, criterion, DEVICE=DEVICE,train=True, threshold=0.5, return_preds=False):
    epoch_loss, epoch_TP, epoch_FP, epoch_FN = 0, 0, 0, 0

    # --- LOCATION 1: Initialize confidence accumulators ---
    tp_conf_sum, fp_conf_sum = 0.0, 0.0
    tp_count, fp_count = 0, 0
    # Lists to store raw probabilities and targets for the offline sweep
    all_probs, all_targets = [], []
    
    if train:
        model.train()
    else:
        model.eval()

    all_idxs = list(range(len(data)))
    if train:
        random.shuffle(all_idxs)

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

            batch_inputs, batch_ctx_ids, batch_ctx_mask = fn_prepare_batch(model, collator, data_b, data_rag_b, train) 

            if train:
                optimizer.zero_grad()

            # ===== Forward =====
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit_outputs = model(batch_ctx_ids, batch_ctx_mask, **batch_inputs)
                logits, targets = fn_score_postproc(output_logits=logit_outputs,input_targets=batch_inputs)

            loss = criterion(logits, targets)

            # --- LOCATION 2: Calculate Confidence Per Batch ---
            probs = torch.sigmoid(logits).detach()
            preds = (probs > threshold).float()

            # Identify True Positives and False Positives
            tp_mask = (preds == 1) & (targets == 1)
            fp_mask = (preds == 1) & (targets == 0)
            
            # Accumulate confidence values
            tp_conf_sum += probs[tp_mask].sum().item()
            fp_conf_sum += probs[fp_mask].sum().item()
            tp_count += tp_mask.sum().item()
            fp_count += fp_mask.sum().item()
            
               # ===== Collect predictions OR calculate metrics =====
            if not train and return_preds:
                # Save to RAM for offline sweep
                all_probs.append(torch.sigmoid(logits).detach().cpu())
                all_targets.append(targets.detach().cpu())
            else:
                # ===== Metrics =====
                TP, FP, FN = binary_prf_counts(logits, targets, threshold=threshold)
                epoch_TP += TP
                epoch_FP += FP
                epoch_FN += FN

            # ===== Backprop (train only) =====
            if train:
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            # ===== Accumulate =====
            epoch_loss += loss.item()

    epoch_loss = epoch_loss / nbatches

     # --- LOCATION 3: Final Average Calculations ---
    avg_tp_conf = tp_conf_sum / (tp_count + 1e-8)
    avg_fp_conf = fp_conf_sum / (fp_count + 1e-8)
    
    # ===== Return raw predictions if sweeping =====
    if not train and return_preds:
        return epoch_loss, torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)
    
     # ===== Final metrics =====
    precision = epoch_TP / (epoch_TP + epoch_FP + 1e-8)
    recall    = epoch_TP / (epoch_TP + epoch_FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    
    return epoch_loss, precision, recall, f1, avg_tp_conf, avg_fp_conf, epoch_FP

# Load GLiNER
model = GLiNERRagCrossAttn("urchade/gliner_large-v1").to(DEVICE)

# Checks always that the gliner parameters are frozen and only the cross-attention is trained. Else, stop script.
assert not any(p.requires_grad for p in model.gliner.parameters())

# Instatiating collator as the object of the class, data_collator_class
# Set prepare_global_labels=True for training so the collator builds label tensors
# while assembling sample dicts into a model-ready batch.
collator = model.gliner.data_collator_class(
    model.gliner.config, data_processor=model.gliner.data_processor, prepare_labels=True
)

pos_weight = torch.tensor([1.0], device=DEVICE)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.context_cross_attn.parameters(), lr=1e-4, weight_decay=0.1)

nbatches = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE
total_steps = nbatches * EPOCHS
warmup_steps = nbatches * WARMUP_EPOCHS

warmup_sch = torch.optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=0.01, 
    end_factor=1.0, 
    total_iters=warmup_steps
)

main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=(total_steps - warmup_steps)
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, 
    schedulers=[warmup_sch, main_sch], 
    milestones=[warmup_steps]
)

# --- 4. Epoch-Based Train & Validate Loop ---
save_dir = Path("models")
save_dir.mkdir(exist_ok=True)
save_path = save_dir/"cross_attn_best.pt"

best_val_loss = float('inf')
best_val_f1_tracker = 0.0
# Start training proper
for epoch in range(1, EPOCHS+1):

    start_time = time.perf_counter()
    train_loss, train_precision, train_recall, train_f1, t_tp_c, t_fp_c, t_fp_count = fn_run_epoch(model, collator, train_data, train_data_rag, BATCH_SIZE, optimizer, criterion, DEVICE=DEVICE,train=True)
    val_loss, val_precision, val_recall, val_f1, v_tp_c, v_fp_c, v_fp_count = fn_run_epoch(model, collator, val_data, val_data_rag, BATCH_SIZE, optimizer, criterion, DEVICE=DEVICE,train=False)
    
    end_time = time.perf_counter()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if val_f1 > best_val_f1_tracker:
        best_val_f1_tracker = val_f1
        torch.save(model.context_cross_attn.state_dict(), save_path)

    print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.3f} | Train Precision: {train_precision:.3f} | Train Recall: {train_recall:.3f} | Train F1: {train_f1:.3f} | ')
    print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f} | Val. Loss:  {val_loss:.3f}   | Val. Precision:  {val_precision:.3f} | Val. Recall: {val_recall:.3f}  | Val. F1: {val_f1:.3f} | ')
    print(f'>>> [DEBUG] Val FP Count: {int(v_fp_count)} | TP Conf: {v_tp_c:.3f} | FP Conf: {v_fp_c:.3f}')
print(f"\nTraining complete. Best model weights saved to {save_path}")

# Load saved model
model.context_cross_attn.load_state_dict(torch.load(save_path))

print("\nRunning single validation pass for offline threshold sweep...")
# return_preds=True tells the function to give us the raw tensors, not the calculated F1
val_loss, val_probs, val_targets = fn_run_epoch(
    model, collator, val_data, val_data_rag, BATCH_SIZE, optimizer, criterion, DEVICE=DEVICE, train=False, return_preds=True
)

best_val_f1 = 0
best_threshold = 0.5
print("\n--- Sweeping Thresholds Offline ---")
# Sweep threshold to see which is the best for f1 score
for i in range(1, 10):
    threshold = i / 10.0
    
    # Instantly apply the threshold to the entire validation set in memory
    preds = (val_probs > threshold).float()

    # Calculate metrics instantly
    TP = ((preds == 1) & (val_targets == 1)).sum().item()
    FP = ((preds == 1) & (val_targets == 0)).sum().item()
    FN = ((preds == 0) & (val_targets == 1)).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f'Threshold: {threshold:.1f} | Val. Loss: {val_loss:.3f} | Val. Precision: {precision:.3f} | Val. Recall: {recall:.3f} | Val. F1: {f1:.3f}')

    if f1 > best_val_f1:
        best_val_f1 = f1
        best_threshold = threshold

print(f'\nBest Validation F1: {best_val_f1:.3f} at Threshold: {best_threshold:.1f}')

print("\n--- Running Final Test Set ---")
test_loss, test_precision, test_recall, test_f1, _, _, _ = fn_run_epoch(model, collator, test_data, test_data_rag, BATCH_SIZE, optimizer, criterion, DEVICE=DEVICE,train=False, threshold=best_threshold)

print(f'\n--- FINAL TEST RESULTS (Threshold {best_threshold:.1f}) ---')
print(f'Test Loss: {test_loss:.3f} | Test Precision: {test_precision:.3f} | Test Recall: {test_recall:.3f} | Test F1: {test_f1:.3f} | ')
