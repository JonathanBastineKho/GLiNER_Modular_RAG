import argparse
import logging
import random
import numpy as np
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models import GLiNERRagCrossAttn
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


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    loss_accum, tp_accum, fp_accum, fn_accum = 0.0, 0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3} [train]", leave=False)
    for batch in pbar:
        batch = to_device(batch, device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context_ids   = batch.pop("context_ids")
            context_mask  = batch.pop("context_attention_mask")
            logits = model(
                context_ids=context_ids,
                context_attention_mask=context_mask,
                **batch,
            )
            flat_logits, flat_targets = postprocess(logits, batch)
            loss = criterion(flat_logits, flat_targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        probs = torch.sigmoid(flat_logits).detach()
        preds = (probs > 0.5).float()
        tp_accum += ((preds == 1) & (flat_targets == 1)).sum().item()
        fp_accum += ((preds == 1) & (flat_targets == 0)).sum().item()
        fn_accum += ((preds == 0) & (flat_targets == 1)).sum().item()
        loss_accum += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    p = tp_accum / (tp_accum + fp_accum + 1e-8)
    r = tp_accum / (tp_accum + fn_accum + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    return loss_accum / n, p, r, f


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, return_preds=False):
    model.eval()
    loss_accum, tp_accum, fp_accum, fn_accum = 0.0, 0, 0, 0
    all_probs, all_targets = [], []

    pbar = tqdm(loader, desc="           [val] ", leave=False)
    for batch in pbar:
        batch = to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context_ids   = batch.pop("context_ids")
            context_mask  = batch.pop("context_attention_mask")
            logits = model(
                context_ids=context_ids,
                context_attention_mask=context_mask,
                **batch,
            )
            flat_logits, flat_targets = postprocess(logits, batch)
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


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # --- Data ---
    train_ds = GlinerRagDataset(args.data_dir, split="train")
    val_ds   = GlinerRagDataset(args.data_dir, split="val")
    test_ds  = GlinerRagDataset(args.data_dir, split="test")

    # --- Model ---
    model = GLiNERRagCrossAttn(args.model_name).to(device)
    assert not any(p.requires_grad for p in model.gliner.parameters()), \
        "GLiNER backbone should be frozen"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,}")
    wandb.config.update({"trainable_params": trainable})

    # --- Collators ---
    global_labels = sorted({span[2] for s in train_ds.samples for span in s["ner"]})
    logger.info(f"Global label set size: {len(global_labels)}")

    train_collator = GlinerRagCollator(
        model.gliner, global_labels=global_labels,
        max_len=model.gliner.config.max_len,
        prepare_labels=True, neg_ratio=args.neg_ratio,
    )
    eval_collator = GlinerRagCollator(
        model.gliner, global_labels=global_labels,
        max_len=model.gliner.config.max_len,
        prepare_labels=True, neg_ratio=0.0,      # no negatives at eval
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_collator, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=eval_collator, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=eval_collator, num_workers=args.num_workers, pin_memory=True,
    )

    # --- Optimiser + scheduler ---
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.context_cross_attn.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps)),
        ],
        milestones=[warmup_steps],
    )

    # --- Training loop ---
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "cross_attn_best.pt"

    best_val_f1 = 0.0
    import time

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_p, train_r, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        val_loss, val_p, val_r, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        elapsed_m, elapsed_s = epoch_time(t0, time.perf_counter())

        wandb.log({
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train/loss": train_loss, "train/precision": train_p,
            "train/recall": train_r,  "train/f1": train_f1,
            "val/loss":   val_loss,   "val/precision": val_p,
            "val/recall": val_r,      "val/f1": val_f1,
        })

        logger.info(
            f"Epoch {epoch:02} [{elapsed_m}m{elapsed_s}s] "
            f"| train loss {train_loss:.3f} P {train_p:.3f} R {train_r:.3f} F1 {train_f1:.3f} "
            f"| val loss {val_loss:.3f} P {val_p:.3f} R {val_r:.3f} F1 {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.context_cross_attn.state_dict(), save_path)
            logger.info(f"  ↑ New best val F1 {best_val_f1:.3f} — saved to {save_path}")
            wandb.config.update({"best_epoch": epoch, "best_val_f1": best_val_f1})

    # --- Threshold sweep on val ---
    logger.info("\nLoading best checkpoint for threshold sweep...")
    model.context_cross_attn.load_state_dict(torch.load(save_path))
    _, val_probs, val_targets = evaluate(
        model, val_loader, criterion, device, return_preds=True
    )
    best_thresh, best_val_f1 = threshold_sweep(val_probs, val_targets)
    logger.info(f"Best threshold: {best_thresh:.1f} | Val F1: {best_val_f1:.3f}")
    wandb.config.update({"best_threshold": best_thresh})

    # --- Final test ---
    logger.info("\n--- Final test ---")
    test_loss, test_p, test_r, test_f1 = evaluate(
        model, test_loader, criterion, device, threshold=best_thresh
    )
    logger.info(
        f"Test loss {test_loss:.3f} | P {test_p:.3f} | R {test_r:.3f} | F1 {test_f1:.3f}"
    )
    wandb.log({"test/loss": test_loss, "test/precision": test_p,
               "test/recall": test_r, "test/f1": test_f1})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLiNER RAG cross-attention training")
    parser.add_argument("--data_dir",      default="data/combined_dataset")
    parser.add_argument("--model_name",    default="urchade/gliner_large-v1")
    parser.add_argument("--save_dir",      default="models")
    parser.add_argument("--wandb_project", default="gliner_modular_rag")
    parser.add_argument("--run_name",      default=None)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--warmup_epochs", type=int,   default=2)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.1)
    parser.add_argument("--neg_ratio",     type=float, default=0.5)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--seed",          type=int,   default=42)
    main(parser.parse_args())