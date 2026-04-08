import argparse
import json
import logging
import random
import gliner
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


def score_batch_output(batch_pred_entities, batch_gt_entities):
    false_positive = 0
    false_negative = 0
    true_positive = 0


    for pred_entities, gt_entities in zip(batch_pred_entities, batch_gt_entities):
        # print(f"Pred: {pred_entities}")
        # print(f"GT: {gt_entities}")
        pred_set = set((e["text"], e["label"]) for e in pred_entities)
        gt_set   = set((gt["text"], gt["label"]) for gt in gt_entities)
        true_positive += len(pred_set & gt_set)
        false_positive += len(pred_set - gt_set)
        false_negative += len(gt_set - pred_set)

    return true_positive, false_positive, false_negative

@torch.no_grad()
def evaluate(model: gliner.GLiNER, dataset: list[str, tuple[int,int,str], list[dict[str,str]]], batchsize, device: torch.device, threshold=0.5):
    model.eval()
    tp_accum, fp_accum, fn_accum = 0, 0, 0,

    nbatch = (len(dataset) + batchsize - 1) // batchsize
    for batch_idx in tqdm(range(nbatch), desc="Evaluating", unit="batch"):
        batch_samples = dataset[batch_idx*batchsize:(batch_idx+1)*batchsize]

        batch_text, batch_ners, batch_gt_entities = zip(*batch_samples)

        labels = sorted({label for ners in batch_ners for (start, stop, label) in ners})

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            
            batch_pred_entities  = model.inference(
                batch_text, labels, threshold=threshold, return_class_probs=False
            )

            true_positive, false_positive, false_negative = score_batch_output(batch_pred_entities, batch_gt_entities)
            tp_accum += true_positive
            fp_accum += false_positive
            fn_accum += false_negative


    precision = tp_accum / (tp_accum + fp_accum + 1e-8)
    recall    = tp_accum / (tp_accum + fn_accum + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


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



def load_dataset(domaindir: str, split: str) -> list[tuple[str, list[tuple[int, int, str]]]]:
    jsonl_path = Path(domaindir) / f"{split}.jsonl"
    samples = []    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            text = " ".join(raw["tokens"])
            ners = raw.get("ner", [])
            entities = [{"text":" ".join(raw["tokens"][start:stop+1]), "label": label} for (start, stop, label) in ners]
            samples.append((text, ners, entities))
    return samples

def main(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")


    # --- Data ---
    # val_datasets  = {d: load_dataset(d, "validation")  for d in args.val_domains}
    test_datasets = {d: load_dataset(d, "test") for d in args.val_domains}


    # --- Model ---
    model = gliner.GLiNER.from_pretrained("urchade/gliner_large-v1", compile_torch_model=True)



    # all_val_probs, all_val_targets = [], []
    # for loader in val_loaders.values():
    #     _, val_probs, val_targets = evaluate(model, loader, criterion, device, return_preds=True)
    #     all_val_probs.append(val_probs)
    #     all_val_targets.append(val_targets)
    # all_val_probs   = torch.cat(all_val_probs)
    # all_val_targets = torch.cat(all_val_targets)

    # best_thresh, best_val_f1 = threshold_sweep(all_val_probs, all_val_targets)
    # logger.info(f"Best threshold: {best_thresh:.1f} | Val F1: {best_val_f1:.3f}")
    # print({"best_threshold": best_thresh})


    # --- Final test per domain ---
    logger.info("\n--- Final test ---")
    test_f1s = {}
    for domain_name, dataset in test_datasets.items():
        test_p, test_r, test_f1 = evaluate(
            model, dataset, args.batch_size, device,
        )
        test_f1s[domain_name] = test_f1
        logger.info(f"  test [{domain_name}] P {test_p:.3f} R {test_r:.3f} F1 {test_f1:.3f}")
        print({
            # f"test/{domain_name}/loss": test_loss,
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