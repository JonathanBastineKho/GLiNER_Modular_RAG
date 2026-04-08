import argparse
import logging
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models import GLiNERRagConcat
from src.utils.dataset import GlinerRagDataset, GlinerRagCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def epoch_time(start: float, end: float):
    elapsed = end - start
    return int(elapsed // 60), int(elapsed % 60)


def compute_prf(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    TP = ((preds == 1) & (targets == 1)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def postprocess(logits: torch.Tensor, batch: dict):
    """Flatten [B, L, K, C] → masked 1D logits + targets."""
    B, L, K, C = logits.shape
    logits_flat  = logits.view(B, L * K, C)
    targets_flat = batch["labels"].float()
    span_mask    = batch["span_mask"].unsqueeze(-1)          # (B, L*K, 1)
    mask_expanded = span_mask.bool().expand_as(logits_flat)
    return logits_flat[mask_expanded], targets_flat[mask_expanded]


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, return_preds=False):
    model.eval()
    loss_accum, tp_accum, fp_accum, fn_accum = 0.0, 0, 0, 0
    all_probs, all_targets = [], []

    pbar = tqdm(loader, desc="           [val] ", leave=False)
    for batch in pbar:
        batch = to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            
            logits = model(**batch)
            flat_logits, flat_targets = postprocess(logits.logits, batch)
            loss = criterion(flat_logits, flat_targets)

        loss_accum += loss.item()

        if return_preds:
            all_probs.append(torch.sigmoid(flat_logits).cpu())
            all_targets.append(flat_targets.cpu())
        else:
            probs = torch.sigmoid(flat_logits)
            preds = (probs > threshold).float()
            tp_accum += ((preds == 1) & (flat_targets == 1)).sum().item()
            fp_accum += ((preds == 1) & (flat_targets == 0)).sum().item()
            fn_accum += ((preds == 0) & (flat_targets == 1)).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    if return_preds:
        return loss_accum / n, torch.cat(all_probs), torch.cat(all_targets)

    p = tp_accum / (tp_accum + fp_accum + 1e-8)
    r = tp_accum / (tp_accum + fn_accum + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    return loss_accum / n, p, r, f


def threshold_sweep(val_probs, val_targets):
    best_f1, best_thresh = 0.0, 0.5
    logger.info("--- Sweeping thresholds ---")
    for i in range(1, 10):
        t = i / 10.0
        preds = (val_probs > t).float()
        TP = ((preds == 1) & (val_targets == 1)).sum().item()
        FP = ((preds == 1) & (val_targets == 0)).sum().item()
        FN = ((preds == 0) & (val_targets == 1)).sum().item()
        p = TP / (TP + FP + 1e-8)
        r = TP / (TP + FN + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        logger.info(f"  t={t:.1f} | P={p:.3f} R={r:.3f} F1={f:.3f}")
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh, best_f1

def build_pairs(domain_dirs: list[str], split: str) -> list[tuple]:
    split_map = {
        "train": ("train.jsonl",      "train_w_rag.pkl"),
        "val":   ("validation.jsonl", "val_w_rag.pkl"),
        "test":  ("test.jsonl",       "test_w_rag.pkl"),
    }
    jsonl_name, pkl_name = split_map[split]
    return [(Path(d) / jsonl_name, Path(d) / pkl_name) for d in domain_dirs]


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")


    # --- Data ---
    val_datasets  = {d: GlinerRagDataset(build_pairs([d], "val"))  for d in args.val_domains}
    test_datasets = {d: GlinerRagDataset(build_pairs([d], "test")) for d in args.val_domains}


    # --- Model ---
    model = GLiNERRagConcat("urchade/gliner_large-v1").to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,}")

    # --- Collators ---
    global_labels = {d: sorted({span[2] for s in test_datasets[d].samples for span in s["ner"]}) for d in args.val_domains}
    for d in args.val_domains:
        logger.info(f"Domain '{d}' label set size: {len(global_labels[d])}")


    eval_collator = {d: GlinerRagCollator(
        model.gliner, global_labels=global_labels[d],
        max_len=model.gliner.config.max_len,
        prepare_labels=True, neg_ratio=0.0,      # no negatives at eval
    ) for d in args.val_domains}

    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=eval_collator[name], num_workers=args.num_workers, pin_memory=True)
        for name, ds in val_datasets.items()
    }
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=eval_collator[name], num_workers=args.num_workers, pin_memory=True)
        for name, ds in test_datasets.items()
    }
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")



    all_val_probs, all_val_targets = [], []
    for loader in val_loaders.values():
        _, val_probs, val_targets = evaluate(model, loader, criterion, device, return_preds=True)
        all_val_probs.append(val_probs)
        all_val_targets.append(val_targets)
    all_val_probs   = torch.cat(all_val_probs)
    all_val_targets = torch.cat(all_val_targets)

    best_thresh, best_val_f1 = threshold_sweep(all_val_probs, all_val_targets)
    logger.info(f"Best threshold: {best_thresh:.1f} | Val F1: {best_val_f1:.3f}")
    print({"best_threshold": best_thresh})


    # --- Final test per domain ---
    logger.info("\n--- Final test ---")
    test_f1s = {}
    for domain_name, loader in test_loaders.items():
        test_loss, test_p, test_r, test_f1 = evaluate(
            model, loader, criterion, device, threshold=best_thresh
        )
        test_f1s[domain_name] = test_f1
        logger.info(f"  test [{domain_name}] P {test_p:.3f} R {test_r:.3f} F1 {test_f1:.3f}")
        print({
            f"test/{domain_name}/loss": test_loss,
            f"test/{domain_name}/precision": test_p,
            f"test/{domain_name}/recall": test_r,
            f"test/{domain_name}/f1": test_f1,
        })

    avg_test_f1 = sum(test_f1s.values()) / len(test_f1s)
    logger.info(f"  test avg F1: {avg_test_f1:.3f}")
    print({"test/avg_f1": avg_test_f1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLiNER Concat Evaluation Script")
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--val_domains",   nargs="+", required=True,
                        help="Paths to val/test domain folders e.g. data/politics data/ai")
    main(parser.parse_args())