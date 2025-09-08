#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETPC augmentation script for BART paraphrase-type detection.

What it does:
- Reads ETPC train CSV from the given path.
- Creates label-preserving POSITIVES by back-translating S1/S2/both (EN→DE→EN).
- Creates hard NEGATIVES by pairing S1 with a different S2 from another sample.
- Converts paraphrase_type_ids to a 26-d binary vector (drops {12,19,20,23,27}).
- Writes an augmented CSV compatible with your bart_detection pipeline.

Usage:
    python etpc_augment.py \
      --input /user/shrinath.madde/u17468/DL_NLP/data/etpc-paraphrase-train.csv \
      --output /user/shrinath.madde/u17468/DL_NLP/data/etpc-paraphrase-train.augmented.csv \
      --bt_lang de --bt_frac_pos 0.6 --bt_frac_neg 0.5 --neg_frac 0.7 --seed 13

Notes:
- Requires: transformers, torch, pandas, tqdm (optional).
- If back-translation is slow, lower --bt_frac_* or increase --batch_size.
- Only numbers are masked by default (safe & lightweight). You can extend masking.
"""

import argparse
import ast
import random
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

from transformers import MarianMTModel, MarianTokenizer


# -----------------------
# Config & small helpers
# -----------------------

DROP_TYPES = {12, 19, 20, 23, 27}   # dropped from ETPC => 26 remaining labels

NUM_PAT = re.compile(r"\$?\b\d[\d,]*(\.\d+)?\b")  # mask numbers/currency-like tokens


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            return ast.literal_eval(obj)
        except Exception:
            # fallback: strip brackets like "[1, 2]"
            s = obj.strip().strip("[]")
            if not s:
                return []
            return [int(x.strip()) for x in s.split(",")]
    return []


def etpc_ids_to_binary(type_ids: List[int]) -> List[int]:
    """
    Map original ETPC type ids to a 26-d binary vector by dropping
    {12,19,20,23,27} and reindexing remaining ids to [0..25] in ascending order.
    """
    # Build (once) a mapping from kept original ids -> compact [0..25]
    # We compute lazily on first call and memoize
    if not hasattr(etpc_ids_to_binary, "_map"):
        kept = [i for i in range(0, 32) if i not in DROP_TYPES]
        etpc_ids_to_binary._kept = kept
        etpc_ids_to_binary._map = {orig: j for j, orig in enumerate(kept)}
        etpc_ids_to_binary._size = len(kept)  # 26

    vec = [0] * etpc_ids_to_binary._size
    for tid in type_ids:
        if tid in etpc_ids_to_binary._map:
            vec[etpc_ids_to_binary._map[tid]] = 1
    return vec


def mask_numbers(s: str) -> str:
    return NUM_PAT.sub("[NUM]", s)


# -----------------------
# Back-translation
# -----------------------

class BackTranslator:
    def __init__(self, mid_lang: str = "de", device: str = None):
        """
        mid_lang: hub language code ('de', 'fr', 'es', ...)
        """
        lang = mid_lang.lower()
        src2mid = f"Helsinki-NLP/opus-mt-en-{lang}"
        mid2src = f"Helsinki-NLP/opus-mt-{lang}-en"

        self.tok_en_mid = MarianTokenizer.from_pretrained(src2mid)
        self.mt_en_mid = MarianMTModel.from_pretrained(src2mid)
        self.tok_mid_en = MarianTokenizer.from_pretrained(mid2src)
        self.mt_mid_en = MarianMTModel.from_pretrained(mid2src)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mt_en_mid.to(self.device)
        self.mt_mid_en.to(self.device)

    @torch.inference_mode()
    def __call__(self, texts: List[str], batch_size: int = 16, max_length: int = 256) -> List[str]:
        outs = []
        rng = range(0, len(texts), batch_size)
        if TQDM:
            rng = tqdm(rng, desc=f"Back-translation ({self.device})", leave=False)
        for i in rng:
            batch = texts[i:i + batch_size]
            # EN -> MID
            enc = self.tok_en_mid(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            mid_ids = self.mt_en_mid.generate(**enc, max_length=max_length, num_beams=4)
            mid_txt = self.tok_en_mid.batch_decode(mid_ids, skip_special_tokens=True)

            # MID -> EN
            enc2 = self.tok_mid_en(mid_txt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            en_ids = self.mt_mid_en.generate(**enc2, max_length=max_length, num_beams=4)
            en_txt = self.tok_mid_en.batch_decode(en_ids, skip_special_tokens=True)

            outs.extend(en_txt)
        return outs


# -----------------------
# Augmentation logic
# -----------------------

def augment_positive_pair(s1: str, s2: str, label_vec: List[int], bt, do_bt_s1: bool, do_bt_s2: bool,
                          mask_nums: bool = True) -> List[Tuple[str, str, List[int]]]:
    cands = []
    s1_in = mask_numbers(s1) if mask_nums else s1
    s2_in = mask_numbers(s2) if mask_nums else s2

    # S1 only
    if do_bt_s1:
        s1_bt = bt([s1_in], batch_size=1)[0]
        cands.append((s1_bt, s2, label_vec))

    # S2 only
    if do_bt_s2:
        s2_bt = bt([s2_in], batch_size=1)[0]
        cands.append((s1, s2_bt, label_vec))

    # Both sides
    if do_bt_s1 and do_bt_s2:
        s1_bt, s2_bt = bt([s1_in, s2_in], batch_size=2)
        cands.append((s1_bt, s2_bt, label_vec))

    return cands


def augment_negative_mix(s1: str, s2_pool: List[str], label_size: int) -> Tuple[str, str, List[int]]:
    """
    Create a hard negative by pairing s1 with a random s2 from a different instance.
    Label is all-zeros (no paraphrase types).
    """
    # choose a different s2
    s2_neg = s2_pool[random.randrange(len(s2_pool))]
    return (s1, s2_neg, [0] * label_size)


def build_output_df(rows, label_size):
    recs = []
    for rid, s1, s2, label_vec, tag in rows:
        recs.append({
            "id": rid,
            "sentence1": s1,
            "sentence2": s2,
            "labels_binary": label_vec,
            "source": tag
        })
    return pd.DataFrame(recs)


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="Path to ETPC train CSV (e.g., /user/.../etpc-paraphrase-train.csv)")
    ap.add_argument("--output", type=str, required=True,
                    help="Where to write augmented CSV")
    ap.add_argument("--bt_lang", type=str, default="de", help="Pivot language for back-translation (de, fr, es, ...)")
    ap.add_argument("--bt_frac_pos", type=float, default=0.6, help="Chance to BT each positive side (S1/S2) independently")
    ap.add_argument("--bt_frac_neg", type=float, default=0.5, help="Chance to BT one side for negatives (makes them harder)")
    ap.add_argument("--neg_frac", type=float, default=0.7, help="Fraction of rows to create a negative mix for")
    ap.add_argument("--batch_size", type=int, default=16, help="BT batch size")
    ap.add_argument("--max_length", type=int, default=256, help="BT max length")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no_mask_numbers", action="store_true", help="Disable masking numbers during BT")
    args = ap.parse_args()

    set_seed(args.seed)

    # Load ETPC CSV
    df = pd.read_csv(args.input)

    # Expected columns: id, sentence1, sentence2, paraphrase_type_ids (list-like)
    needed = {"id", "sentence1", "sentence2"}
    if not needed.issubset(set(df.columns)):
        print(f"[ERROR] CSV must contain columns: {needed}", file=sys.stderr)
        sys.exit(1)

    # Parse type ids and convert to 26-d binary labels
    if "paraphrase_type_ids" in df.columns:
        raw_ids = df["paraphrase_type_ids"].apply(to_list)
    elif "paraphrase_type" in df.columns:
        raw_ids = df["paraphrase_type"].apply(to_list)
    else:
        print("[WARN] paraphrase_type_ids not found; assuming all rows have at least one type id []")
        raw_ids = pd.Series([[] for _ in range(len(df))])

    labels_bin = raw_ids.apply(etpc_ids_to_binary)
    label_size = len(labels_bin.iloc[0]) if len(labels_bin) > 0 else 26

    # Prepare back-translator
    print(f"[INFO] Initializing back-translator EN↔{args.bt_lang} ...")
    bt = BackTranslator(mid_lang=args.bt_lang)

    rows_out = []  # (id, s1, s2, label_vec, tag)

    # Original positives (keep them)
    for i, row in df.iterrows():
        rid = row["id"]
        s1 = str(row["sentence1"])
        s2 = str(row["sentence2"])
        lv = labels_bin.iloc[i]
        rows_out.append((f"{rid}", s1, s2, lv, "orig"))

    # Positive augmentations via BT
    it = range(len(df))
    if TQDM:
        it = tqdm(it, desc="Augmenting positives", leave=False)
    for i in it:
        row = df.iloc[i]
        rid = row["id"]
        s1 = str(row["sentence1"])
        s2 = str(row["sentence2"])
        lv = labels_bin.iloc[i]

        do_bt_s1 = random.random() < args.bt_frac_pos
        do_bt_s2 = random.random() < args.bt_frac_pos

        if not (do_bt_s1 or do_bt_s2):
            continue

        cands = augment_positive_pair(
            s1, s2, lv, bt,
            do_bt_s1=do_bt_s1,
            do_bt_s2=do_bt_s2,
            mask_nums=(not args.no_mask_numbers),
        )

        # add with provenance tags
        for k, (a_s1, a_s2, a_lv) in enumerate(cands):
            tag = f"pos_bt_{'s1' if do_bt_s1 else ''}{'s2' if do_bt_s2 else ''}".strip("_")
            rows_out.append((f"{rid}::{tag}::{k}", a_s1, a_s2, a_lv, tag))

    # Hard negatives by cross-mixing s2
    s2_pool = [str(x) for x in df["sentence2"].tolist()]
    num_negs = int(args.neg_frac * len(df))
    neg_indices = random.sample(range(len(df)), k=num_negs)
    it = neg_indices
    if TQDM:
        it = tqdm(neg_indices, desc="Creating hard negatives", leave=False)
    for i in it:
        row = df.iloc[i]
        rid = row["id"]
        s1 = str(row["sentence1"])

        n_s1, n_s2, n_lv = augment_negative_mix(s1, s2_pool, label_size)

        # Optional: make negatives harder by BT one side with some prob
        if random.random() < args.bt_frac_neg:
            side = random.choice(["s1", "s2"])
            if side == "s1":
                n_s1 = bt([mask_numbers(n_s1) if not args.no_mask_numbers else n_s1], batch_size=1)[0]
                tag = "neg_mix_bt_s1"
            else:
                n_s2 = bt([mask_numbers(n_s2) if not args.no_mask_numbers else n_s2], batch_size=1)[0]
                tag = "neg_mix_bt_s2"
        else:
            tag = "neg_mix"

        rows_out.append((f"{rid}::{tag}", n_s1, n_s2, n_lv, tag))

    # Build output DataFrame
    out_df = build_output_df(rows_out, label_size)

    # Reorder/rename columns to match your pipeline expectations
    # If your training expects 'labels' as a JSON-like list string, serialize it:
    out_df["labels"] = out_df["labels_binary"].apply(lambda x: str(list(map(int, x))))

    # Final columns
    final_cols = ["id", "sentence1", "sentence2", "labels", "source"]
    out_df = out_df[final_cols]

    out_df.to_csv(args.output, index=False)
    print(f"[DONE] Wrote augmented data: {args.output}")
    print(f"       Originals: {len(df)} | Total (with aug): {len(out_df)}")
    print(f"       Positives kept+augmented: ~{len(df)} + ? | Negatives added: ~{num_negs}")


if __name__ == "__main__":
    main()
