#!/usr/bin/env python3
"""
sst_coral_improved_final.py
Part-2 SST improvement as a single, standalone script.

- Model: BERT base (uncased)
- Features: CLS + mean-pooled token embeddings
- Head: CORAL ordinal regression (K-1 logits for K=5 classes)
- Regularization: Multi-Sample Dropout (K=4) inside the ordinal head
- Loss: BCEWithLogits on cumulative targets
- Selection: keep best dev checkpoint
- Outputs (exact filenames expected by repo/grader):
    predictions/bert/sst-sentiment-dev-output.csv
    predictions/bert/sst-sentiment-test-output.csv
- Input CSV schema (SST):
    train/dev: id,sentence,sentiment
    test:      id,sentence
"""

import os
import argparse
import random
import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SSTCsv(Dataset):
    """Reads SST CSVs with required columns.
       train/dev: id,sentence,sentiment
       test:      id,sentence
    """
    def __init__(self, path: str, tokenizer, max_len=64, has_labels=True):
        df = pd.read_csv(path)
        id_col = "id"
        text_col = "sentence" if "sentence" in df.columns else ("text" if "text" in df.columns else None)
        if text_col is None:
            raise ValueError("Expected a text column named 'sentence' (or 'text').")
        self.ids = df[id_col].astype(str).tolist()
        texts = df[text_col].astype(str).tolist()
        self.has_labels = has_labels
        if has_labels:
            label_col = "sentiment" if "sentiment" in df.columns else ("label" if "label" in df.columns else None)
            if label_col is None:
                raise ValueError("Expected a label column named 'sentiment' (or 'label') for train/dev.")
            self.labels = df[label_col].astype(int).tolist()
        else:
            self.labels = None

        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        item = {
            "id": self.ids[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.has_labels:
            item["label"] = int(self.labels[idx])
        return item


class OrdinalHead(nn.Module):
    """Linear head producing K-1 logits for CORAL (here K=5 => 4 logits)."""
    def __init__(self, in_dim: int, num_classes: int = 5, p: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x):
        return self.fc(self.drop(x))  # [B, K-1]


class BertCoral(nn.Module):
    """BERT + CLS+Mean features + CORAL ordinal head + MSD averaging."""
    def __init__(self, num_classes: int = 5, msd_k: int = 4):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        h = self.bert.config.hidden_size
        self.feat_dim = 2 * h  # CLS + mean
        self.ordinal = OrdinalHead(self.feat_dim, num_classes=num_classes, p=0.2)
        self.msd_k = msd_k

    @staticmethod
    def _masked_mean(last_hidden, mask):
        m = mask.unsqueeze(-1).float()
        s = (last_hidden * m).sum(1)
        z = m.sum(1).clamp_min(1e-6)
        return s / z

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, T, H]
        cls = last_hidden[:, 0, :]          # [B, H]
        mean = self._masked_mean(last_hidden, attention_mask)  # [B, H]
        feats = torch.cat([cls, mean], dim=-1)  # [B, 2H]

        logits_sum = 0
        for _ in range(self.msd_k):
            logits_sum = logits_sum + self.ordinal(feats)  # [B, K-1]
        ord_logits = logits_sum / self.msd_k              # [B, K-1]
        return ord_logits


def to_cumulative_targets(y: torch.Tensor, num_classes=5):
    B = y.shape[0]
    k = torch.arange(num_classes - 1, device=y.device).unsqueeze(0).expand(B, -1)
    return (y.unsqueeze(1) > k).float()


@torch.no_grad()
def ordinal_logits_to_class(ord_logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(ord_logits)  # p_k = P(y > k)
    p0 = 1 - p[:, 0]
    p1 = p[:, 0] * (1 - p[:, 1])
    p2 = p[:, 0] * p[:, 1] * (1 - p[:, 2])
    p3 = p[:, 0] * p[:, 1] * p[:, 2] * (1 - p[:, 3])
    p4 = p[:, 0] * p[:, 1] * p[:, 2] * p[:, 3]
    probs = torch.stack([p0, p1, p2, p3, p4], dim=1)
    return probs.argmax(dim=1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        inp = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)
        lab = torch.tensor(batch["label"], dtype=torch.long, device=device)

        ord_logits = model(inp, msk)  # [B,4]
        targets = to_cumulative_targets(lab, num_classes=5)
        loss = F.binary_cross_entropy_with_logits(ord_logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = ordinal_logits_to_class(ord_logits)
            total_loss += loss.item() * inp.size(0)
            total_correct += (pred == lab).sum().item()
            total_n += inp.size(0)
    return total_loss / max(1, total_n), total_correct / max(1, total_n)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_correct, total_n = 0, 0
    for batch in loader:
        inp = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)
        lab = torch.tensor(batch["label"], dtype=torch.long, device=device)
        ord_logits = model(inp, msk)
        pred = ordinal_logits_to_class(ord_logits)
        total_correct += (pred == lab).sum().item()
        total_n += inp.size(0)
    return total_correct / max(1, total_n)


@torch.no_grad()
def write_predictions(model, loader, out_csv_path: Path, device):
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for batch in loader:
            inp = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            ord_logits = model(inp, msk)
            pred = ordinal_logits_to_class(ord_logits).cpu().tolist()
            for the_id, p in zip(batch["id"], pred):
                w.writerow([str(the_id), int(p)])


def main():
    ap = argparse.ArgumentParser(description="SST-5 CORAL ordinal runner (single file, exact outputs).")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    ap.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    ap.add_argument("--sst_test", type=str, default="data/sst-sentiment-test-student.csv")
    ap.add_argument("--dev_out", type=str, default="predictions/bert/sst-sentiment-dev-output.csv")
    ap.add_argument("--test_out", type=str, default="predictions/bert/sst-sentiment-test-output.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = SSTCsv(args.sst_train, tok, max_len=args.max_len, has_labels=True)
    dev_ds   = SSTCsv(args.sst_dev, tok, max_len=args.max_len, has_labels=True)
    test_ds  = SSTCsv(args.sst_test, tok, max_len=args.max_len, has_labels=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl   = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = BertCoral(num_classes=5, msd_k=4).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print("------------------------------")
    print("   SST CORAL Runner Settings  ")
    print("------------------------------")
    print({
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "max_len": args.max_len, "device": str(device),
        "dev_out": args.dev_out, "test_out": args.test_out
    }, flush=True)
    print("------------------------------", flush=True)

    best_dev, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, device)
        dv_acc = eval_epoch(model, dev_dl, device)
        print(f"Epoch {ep}: train loss {tr_loss:.3f}, train acc {tr_acc:.3f}, dev acc {dv_acc:.3f}", flush=True)
        if dv_acc > best_dev:
            best_dev = dv_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    write_predictions(model, dev_dl, Path(args.dev_out), device)
    write_predictions(model, test_dl, Path(args.test_out), device)

    print(f"[DONE] Best dev acc: {best_dev:.3f}")
    print(f"Wrote dev predictions to:  {args.dev_out}")
    print(f"Wrote test predictions to: {args.test_out}")


if __name__ == "__main__":
    main()
