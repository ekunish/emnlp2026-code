"""Adversarial shortcut-debiasing trainer for PII classification.

Architecture:
  [input text] -> RoBERTa encoder -> pooled representation
                    |-> main head: Sensitive / Non-sensitive (CE loss)
                    |-> adv head: has_1p / no-1p  (CE loss, through GRL at rate lambda)

Training objective:  L = L_main + lambda * L_adv
via GRL: encoder gradient from adv head is -lambda * grad, so encoder removes
1P information while adv head still tries to predict it.

Measured on: in-domain test + (optional) cross-dataset test.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel

FIRST_PERSON_RE = re.compile(r"\b(I|my|me|mine|myself)\b", re.IGNORECASE)


def read_jsonl_physical_lines(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return items


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grl(x: torch.Tensor, lam: float) -> torch.Tensor:
    return GRL.apply(x, lam)


class AdvModel(nn.Module):
    def __init__(self, backbone: str = "roberta-base", num_main: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.main = nn.Linear(hidden, num_main)
        self.adv = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, lam: float):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[
            :, 0
        ]
        hidden = self.drop(hidden)
        main_logits = self.main(hidden)
        adv_logits = self.adv(grl(hidden, lam))
        return main_logits, adv_logits


class JsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer, label2id: dict[str, int], max_len: int = 256):
        self.items = read_jsonl_physical_lines(path)
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        enc = self.tok(
            it["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        label = self.label2id.get(it["label"], 0)
        aux = int(bool(FIRST_PERSON_RE.search(it["text"])))
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(label, dtype=torch.long),
            "aux": torch.tensor(aux, dtype=torch.long),
        }


def run_eval(
    model,
    loader,
    device,
    lam: float = 0.0,
    return_logits: bool = False,
    desc: str | None = None,
) -> tuple[float, float] | tuple[float, float, list[list[float]]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    all_logits: list[list[float]] = []
    iterator = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False) if desc else loader
    with torch.no_grad():
        for batch in iterator:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            main_logits, _ = model(ids, att, lam)
            y_pred.extend(main_logits.argmax(dim=-1).cpu().tolist())
            y_true.extend(batch["label"].cpu().tolist())
            if return_logits:
                all_logits.extend(main_logits.detach().cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    if return_logits:
        return acc, f1, all_logits
    return acc, f1


def dataset_item(dataset: Dataset, idx: int) -> dict:
    if isinstance(dataset, Subset):
        return dataset_item(dataset.dataset, dataset.indices[idx])
    return dataset.items[idx]


def selected_items(dataset: Dataset, indices: list[int]) -> list[dict]:
    return [dataset_item(dataset, idx) for idx in indices]


def softmax_probs(logits: list[float]) -> np.ndarray:
    lg = np.array(logits)
    probs = np.exp(lg - lg.max())
    return probs / probs.sum()


def labels_from_logits(logits_list: list[list[float]]) -> list[int]:
    return [int(np.array(logits).argmax()) for logits in logits_list]


def labels_from_threshold(
    logits_list: list[list[float]],
    *,
    positive_id: int,
    negative_id: int,
    threshold: float,
) -> list[int]:
    labels: list[int] = []
    for logits in logits_list:
        probs = softmax_probs(logits)
        labels.append(positive_id if float(probs[positive_id]) >= threshold else negative_id)
    return labels


def metrics_from_pred_ids(
    items: list[dict],
    pred_ids: list[int],
    label2id: dict[str, int],
) -> tuple[float, float]:
    y_true = [label2id[item["label"]] for item in items]
    return accuracy_score(y_true, pred_ids), f1_score(y_true, pred_ids, average="macro")


def best_binary_threshold(
    items: list[dict],
    logits_list: list[list[float]],
    *,
    label2id: dict[str, int],
    positive_label: str = "Sensitive",
) -> dict:
    if len(label2id) != 2:
        raise ValueError("Threshold calibration is only implemented for binary classification.")
    if positive_label not in label2id:
        raise ValueError(f"Missing positive label for threshold calibration: {positive_label}")
    positive_id = label2id[positive_label]
    negative_ids = [idx for label, idx in label2id.items() if label != positive_label]
    if len(negative_ids) != 1:
        raise ValueError("Threshold calibration expects exactly one negative label.")
    negative_id = negative_ids[0]
    y_true = [label2id[item["label"]] for item in items]
    scores = [float(softmax_probs(logits)[positive_id]) for logits in logits_list]
    unique_scores = sorted(set(scores))
    thresholds = [-1e-12]
    thresholds.extend((left + right) / 2 for left, right in pairwise(unique_scores))
    thresholds.append(1.0 + 1e-12)

    best: dict | None = None
    for threshold in thresholds:
        pred_ids = [
            positive_id if score >= threshold else negative_id
            for score in scores
        ]
        acc = accuracy_score(y_true, pred_ids)
        macro_f1 = f1_score(y_true, pred_ids, average="macro")
        candidate = {
            "threshold": threshold,
            "positive_label": positive_label,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "valid_acc": acc,
            "valid_f1": macro_f1,
        }
        if best is None or (macro_f1, acc) > (best["valid_f1"], best["valid_acc"]):
            best = candidate
    if best is None:
        raise ValueError("Could not determine threshold calibration.")
    return best


def write_prediction_file(
    path: Path,
    items: list[dict],
    logits_list: list[list[float]],
    id2label: dict[int, str],
    pred_ids: list[int] | None = None,
    threshold: float | None = None,
) -> None:
    with path.open("w", encoding="utf-8") as pf:
        if pred_ids is None:
            pred_ids = labels_from_logits(logits_list)
        for item, logits, pred_id in zip(items, logits_list, pred_ids, strict=True):
            probs = softmax_probs(logits)
            margin = (
                float(abs(probs[1] - probs[0]))
                if len(probs) == 2
                else float(probs.max() - np.partition(probs, -2)[-2])
            )
            pf.write(
                json.dumps(
                    {
                        "text": item["text"],
                        "true_label": item["label"],
                        "pred_label": id2label[pred_id],
                        "correct": item["label"] == id2label[pred_id],
                        "logits": logits,
                        "probs": probs.tolist(),
                        "margin": margin,
                        "threshold": threshold,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def stratified_indices(dataset: Dataset, max_size: int, seed: int) -> list[int]:
    n_items = len(dataset)
    if max_size <= 0 or n_items <= max_size:
        return list(range(n_items))

    groups: dict[str, list[int]] = {}
    for idx in range(n_items):
        label = str(dataset_item(dataset, idx)["label"])
        groups.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    labels = sorted(groups)
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    assigned = 0
    for label in labels:
        rng.shuffle(groups[label])
        raw = max_size * len(groups[label]) / n_items
        quota = int(raw)
        if quota == 0 and max_size >= len(labels):
            quota = 1
        quota = min(quota, len(groups[label]))
        quotas[label] = quota
        fractions.append((raw - int(raw), label))
        assigned += quota

    for _, label in sorted(fractions, reverse=True):
        if assigned >= max_size:
            break
        if quotas[label] < len(groups[label]):
            quotas[label] += 1
            assigned += 1

    for _, label in sorted(fractions):
        if assigned <= max_size:
            break
        if quotas[label] > 1:
            quotas[label] -= 1
            assigned -= 1

    chosen: list[int] = []
    for label in labels:
        chosen.extend(groups[label][: quotas[label]])
    return sorted(chosen)


def make_eval_subset(dataset: Dataset, max_size: int, seed: int, name: str) -> tuple[Subset, dict]:
    indices = stratified_indices(dataset, max_size, seed)
    items = selected_items(dataset, indices)
    counts = Counter(str(item["label"]) for item in items)
    meta = {
        "name": name,
        "original_size": len(dataset),
        "evaluated_size": len(indices),
        "requested_max_size": max_size,
        "subsampled": len(indices) < len(dataset),
        "subsample_seed": seed,
        "label_counts": dict(sorted(counts.items())),
    }
    return Subset(dataset, indices), meta


def state_dict_to_cpu(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def split_train_valid(path: Path, ratio: float, seed: int, tokenizer, label2id):
    ds = JsonlDataset(path, tokenizer, label2id)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - ratio))
    return torch.utils.data.Subset(ds, idx[:cut]), torch.utils.data.Subset(ds, idx[cut:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument(
        "--valid-data",
        default=None,
        help="Optional explicit valid jsonl (e.g., official spedac1_valid). "
        "If omitted, split --valid-ratio from --train-data.",
    )
    ap.add_argument("--in-test", required=True)
    ap.add_argument("--cross-test", default=None)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--backbone", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--valid-ratio", type=float, default=0.2)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--eval-valid-size",
        type=int,
        default=2000,
        help="Maximum validation examples evaluated after each epoch. 0 = full valid.",
    )
    ap.add_argument(
        "--eval-test-size",
        type=int,
        default=0,
        help="Maximum in-test examples evaluated once at the best epoch. 0 = full test.",
    )
    ap.add_argument(
        "--eval-subsample-seed",
        type=int,
        default=0,
        help="Seed for deterministic label-stratified eval subsampling.",
    )
    ap.add_argument(
        "--save-best",
        action="store_true",
        help="Save best encoder+heads state_dict to out/ckpt_best.pt",
    )
    ap.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save in_test per-example predictions + logits to out/predictions.jsonl at best epoch",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (epochs without monitor improvement). "
        "0 = disabled (run all --epochs). Existing experiments used 0; "
        "set >0 only when the run is documented as patience-enabled.",
    )
    ap.add_argument(
        "--monitor",
        choices=["valid_acc", "valid_f1"],
        default="valid_acc",
        help="Validation metric used for best checkpoint selection and early stopping.",
    )
    ap.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help=(
            "For binary classification, choose the Sensitive probability threshold "
            "that maximizes validation Macro F1 at the best epoch and apply it to test."
        ),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    label_map = yaml.safe_load(Path(args.label_map).read_text())
    labels = list(label_map.keys()) if isinstance(label_map, dict) else [str(x) for x in label_map]
    label2id = {lbl: i for i, lbl in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if args.valid_data:
        train_ds = JsonlDataset(Path(args.train_data), tok, label2id, max_len=args.max_len)
        valid_ds = JsonlDataset(Path(args.valid_data), tok, label2id, max_len=args.max_len)
        print(f"using explicit valid: {args.valid_data}")
    else:
        train_ds, valid_ds = split_train_valid(
            Path(args.train_data), args.valid_ratio, args.seed, tok, label2id
        )
    in_ds = JsonlDataset(Path(args.in_test), tok, label2id, max_len=args.max_len)
    cross_ds = (
        JsonlDataset(Path(args.cross_test), tok, label2id, max_len=args.max_len)
        if args.cross_test
        else None
    )
    valid_eval_ds, valid_eval_meta = make_eval_subset(
        valid_ds, args.eval_valid_size, args.eval_subsample_seed, "valid"
    )
    in_eval_ds, in_eval_meta = make_eval_subset(
        in_ds, args.eval_test_size, args.eval_subsample_seed, "in_test"
    )
    cross_eval_ds = None
    cross_eval_meta = None
    if cross_ds is not None:
        cross_eval_ds, cross_eval_meta = make_eval_subset(
            cross_ds, args.eval_test_size, args.eval_subsample_seed, "cross_test"
        )
    print(f"valid eval: {valid_eval_meta}")
    print(f"in-test eval: {in_eval_meta}")
    if cross_eval_meta:
        print(f"cross-test eval: {cross_eval_meta}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_eval_ds, batch_size=args.batch_size, shuffle=False)
    in_loader = DataLoader(in_eval_ds, batch_size=args.batch_size, shuffle=False)
    cross_loader = (
        DataLoader(cross_eval_ds, batch_size=args.batch_size, shuffle=False)
        if cross_eval_ds
        else None
    )

    model = AdvModel(args.backbone, num_main=len(labels), dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_valid = -1.0
    best: dict = {}
    best_state: dict | None = None
    history: list[dict] = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = {"main": 0.0, "adv": 0.0, "total": 0.0, "n": 0}
        pbar = tqdm(train_loader, desc=f"ep{epoch}", dynamic_ncols=True, mininterval=2.0)
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            y_main = batch["label"].to(device)
            y_adv = batch["aux"].to(device)
            main_logits, adv_logits = model(ids, att, args.lam)
            loss_main = F.cross_entropy(main_logits, y_main)
            loss_adv = F.cross_entropy(adv_logits, y_adv)
            loss = loss_main + args.lam * loss_adv
            opt.zero_grad()
            loss.backward()
            opt.step()
            bsz = ids.size(0)
            tot["main"] += loss_main.item() * bsz
            tot["adv"] += loss_adv.item() * bsz
            tot["total"] += loss.item() * bsz
            tot["n"] += bsz

        v_acc, v_f1 = run_eval(model, valid_loader, device, desc=f"valid{epoch}")
        entry = {
            "epoch": epoch,
            "train_loss_main": tot["main"] / tot["n"],
            "train_loss_adv": tot["adv"] / tot["n"],
            "valid_acc": v_acc,
            "valid_f1": v_f1,
        }
        history.append(entry)
        print(
            f"[ep{epoch}] main={entry['train_loss_main']:.3f} adv={entry['train_loss_adv']:.3f} "
            f"valid={v_acc:.4f} valid_f1={v_f1:.4f}"
        )
        monitor_value = entry[args.monitor]
        if monitor_value > best_valid:
            best_valid = monitor_value
            no_improve = 0
            best_state = state_dict_to_cpu(model)
            best = {
                "epoch": epoch,
                "valid_acc": v_acc,
                "valid_f1": v_f1,
                "monitor": args.monitor,
                "monitor_value": monitor_value,
                "lambda": args.lam,
            }
            if args.save_best:
                torch.save(
                    {
                        "state_dict": best_state,
                        "label2id": label2id,
                        "backbone": args.backbone,
                        "max_len": args.max_len,
                        "epoch": epoch,
                    },
                    out_dir / "ckpt_best.pt",
                )
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(
                    f"[early-stop] no improvement in {args.patience} epochs "
                    f"(best at ep{best.get('epoch', '?')} {args.monitor}={best_valid:.4f}); "
                    f"stopping at ep{epoch}"
                )
                break

    if best_state is None:
        best_state = state_dict_to_cpu(model)
    model.load_state_dict(best_state)
    if args.save_predictions:
        valid_acc_best, valid_f1_best, valid_logits = run_eval(
            model, valid_loader, device, lam=0.0, return_logits=True, desc="valid-best"
        )
        in_acc, in_f1, best_logits = run_eval(
            model, in_loader, device, lam=0.0, return_logits=True, desc="test"
        )
    else:
        in_acc, in_f1 = run_eval(model, in_loader, device, lam=0.0, desc="test")
        best_logits = None
        valid_logits = None
        valid_acc_best = None
        valid_f1_best = None
    c_acc, c_f1 = (
        run_eval(model, cross_loader, device, lam=0.0, desc="cross")
        if cross_loader
        else (None, None)
    )
    threshold_config = None
    calibrated_valid_pred_ids = None
    calibrated_in_pred_ids = None
    if args.calibrate_threshold:
        if not args.save_predictions or best_logits is None or valid_logits is None:
            raise ValueError("--calibrate-threshold requires --save-predictions.")
        valid_items = selected_items(valid_ds, valid_eval_ds.indices)
        in_items = selected_items(in_ds, in_eval_ds.indices)
        threshold_config = best_binary_threshold(valid_items, valid_logits, label2id=label2id)
        threshold = float(threshold_config["threshold"])
        positive_id = int(threshold_config["positive_id"])
        negative_id = int(threshold_config["negative_id"])
        calibrated_valid_pred_ids = labels_from_threshold(
            valid_logits,
            positive_id=positive_id,
            negative_id=negative_id,
            threshold=threshold,
        )
        calibrated_in_pred_ids = labels_from_threshold(
            best_logits,
            positive_id=positive_id,
            negative_id=negative_id,
            threshold=threshold,
        )
        calibrated_valid_acc, calibrated_valid_f1 = metrics_from_pred_ids(
            valid_items,
            calibrated_valid_pred_ids,
            label2id,
        )
        calibrated_in_acc, calibrated_in_f1 = metrics_from_pred_ids(
            in_items,
            calibrated_in_pred_ids,
            label2id,
        )
        threshold_config.update(
            {
                "valid_acc": calibrated_valid_acc,
                "valid_f1": calibrated_valid_f1,
                "in_test_acc": calibrated_in_acc,
                "in_test_f1": calibrated_in_f1,
                "selection_metric": "valid_f1",
            }
        )
        best.update(
            {
                "argmax_valid_acc_best": valid_acc_best,
                "argmax_valid_f1_best": valid_f1_best,
                "argmax_in_test_acc": in_acc,
                "argmax_in_test_f1": in_f1,
                "calibrated_threshold": threshold_config,
            }
        )
        in_acc, in_f1 = calibrated_in_acc, calibrated_in_f1
    best.update(
        {
            "in_test_acc": in_acc,
            "in_test_f1": in_f1,
            "cross_test_acc": c_acc,
            "cross_test_f1": c_f1,
        }
    )
    if args.save_predictions and best_logits is not None and valid_logits is not None:
        id2label = {i: lbl for lbl, i in label2id.items()}
        valid_items = selected_items(valid_ds, valid_eval_ds.indices)
        in_items = selected_items(in_ds, in_eval_ds.indices)
        threshold = (
            float(threshold_config["threshold"])
            if threshold_config is not None
            else None
        )
        write_prediction_file(
            out_dir / "valid_predictions.jsonl",
            valid_items,
            valid_logits,
            id2label,
            pred_ids=calibrated_valid_pred_ids,
            threshold=threshold,
        )
        write_prediction_file(
            out_dir / "predictions.jsonl",
            in_items,
            best_logits,
            id2label,
            pred_ids=calibrated_in_pred_ids,
            threshold=threshold,
        )

    config = vars(args)
    config["eval_valid"] = valid_eval_meta
    config["eval_in_test"] = in_eval_meta
    config["eval_cross_test"] = cross_eval_meta
    result = {"best": best, "history": history, "config": config}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out_dir / "adv_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
