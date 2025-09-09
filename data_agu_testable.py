#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETPC augmentation script for BART paraphrase-type detection (testable version).

Adds test mode & safety:
- --test_n N : run on a small random subset (fast sanity check)
- --dry_run  : don't write output, print preview/stats only
- --bt_on    : enable back-translation explicitly (default OFF)
- Marian guard: if sentencepiece/Marian unavailable, BT is skipped safely
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

# Try to import Marian; if not available, we keep going without BT
HAS_MARIAN = True
try:
    from transformers import MarianMTModel, MarianTokenizer
except Exception:
    HAS_MARIAN = False


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
    def __init__(self, mid_lang: str = "de", device: str = None, num_beams: int = 2):
        """
        mid_lang: hub language code ('de', 'fr', 'es', ...)
        num_beams: beam size for generation (2 is a good speed/quality tradeoff)
        """
        if not HAS_MARIAN:
            raise RuntimeError("Marian not available (sentencepiece missing?).")

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
        self.num_beams = int(num_beams)
        self.mt_en_mid.to(self.device)
        self.mt_mid_en.to(self.device)

    @torch.inference_mode()
    def __call__(self, texts: List[str], batch_size: int = 16, max_length: int = 192) -> List[str]:
        outs = []
        rng = range(0, len(texts), batch_size)
        if TQDM:
            rng = tqdm(rng, desc=f"Back-translation ({self.device})", leave=False)
        for i in rng:
            batch = texts[i:i + batch_size]
            # EN -> MID
            enc = self.tok_en_mid(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            mid_ids = self.mt_en_mid.generate(**enc, max_length=max_length, num_beams=self.num_beams)
            mid_txt = self.tok_en_mid.batch_decode(mid_ids, skip_special_tokens=True)
            # MID -> EN
            enc2 = self.tok_mid_en(mid_txt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            en_ids = self.mt_mid_en.generate(**enc2, max_length=max_length, num_beams=self.num_beams)
            en_txt = self.tok_mid_en.batch_decode(en_ids, skip_special_tokens=True)
            outs.extend(en_txt)
        return outs


# -----------------------
# Augmentation logic
# -----------------------

def augment_positive_pair(s1: str, s2: str, label_vec: List[int], bt, do_bt_s1: bool, do_bt_s2: bool,
                          mask_nums: bool = True, batch_size: int = 1, max_length: int = 192) -> List[Tuple[str, str, List[int]]]:
    cands = []
    s1_in = mask_numbers(s1) if mask_nums else s1
    s2_in = mask_numbers(s2) if mask_nums else s2

    if bt is None:
        return cands

    if do_bt_s1:
        s1_bt = bt([s1_in], batch_size=batch_size, max_length=max_length)[0]
        cands.append((s1_bt, s2, label_vec))

    if do_bt_s2:
        s2_bt = bt([s2_in], batch_size=batch_size, max_length=max_length)[0]
        cands.append((s1, s2_bt, label_vec))

    if do_bt_s1 and do_bt_s2:
        s1_bt, s2_bt = bt([s1_in, s2_in], batch_size=2, max_length=max_length)
        cands.append((s1_bt, s2_bt, label_vec))

    return cands


def augment_negative_mix(s1: str, s2_pool: List[str], label_size: int) -> Tuple[str, str, List[int]]:
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
    ap.add_argument("--input", type=str, required=True, help="Path to ETPC train CSV")
    ap.add_argument("--output", type=str, required=True, help="Where to write augmented CSV")
    # Test / control
    ap.add_argument("--test_n", type=int, default=0, help="If >0, randomly sample N rows (fast test)")
    ap.add_argument("--dry_run", action="store_true", help="Run and show preview/stats but do not write output")
    ap.add_argument("--save_every", type=int, default=0, help="If >0, write temp CSV every K processed rows")
    # BT controls
    ap.add_argument("--bt_on", action="store_true", help="Enable back-translation (off by default)")
    ap.add_argument("--bt_lang", type=str, default="de", help="Pivot language for back-translation (de, fr, es, ...)")
    ap.add_argument("--bt_frac_pos", type=float, default=0.6, help="Prob to BT each positive side independently")
    ap.add_argument("--bt_frac_neg", type=float, default=0.5, help="Prob to BT one side for negatives")
    ap.add_argument("--bt_beams", type=int, default=2, help="Beam size for BT generation (1-2 = faster)")
    ap.add_argument("--bt_device", type=str, default=None, help="'cuda' or 'cpu'. Default auto.")
    ap.add_argument("--batch_size", type=int, default=16, help="BT batch size")
    ap.add_argument("--max_length", type=int, default=192, help="BT max length (shorter = faster)")
    # Other aug
    ap.add_argument("--neg_frac", type=float, default=0.7, help="Fraction of rows to create a negative mix for")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no_mask_numbers", action="store_true", help="Disable masking numbers during BT")
    args = ap.parse_args()

    set_seed(args.seed)

    # Load ETPC CSV
    df = pd.read_csv(args.input)

    # Expect columns
    needed = {"id", "sentence1", "sentence2"}
    if not needed.issubset(set(df.columns)):
        print(f"[ERROR] CSV must contain columns: {needed}", file=sys.stderr)
        sys.exit(1)

    # Optional: subsample for test
    if args.test_n and args.test_n > 0:
        df = df.sample(args.test_n, random_state=args.seed).reset_index(drop=True)
        print(f"[INFO] Test mode: sampled {len(df)} rows.")

    # Parse type ids and convert to 26-d binary labels
    if "paraphrase_type_ids" in df.columns:
        raw_ids = df["paraphrase_type_ids"].apply(to_list)
    elif "paraphrase_type" in df.columns:
        raw_ids = df["paraphrase_type"].apply(to_list)
    else:
        print("[WARN] paraphrase_type_ids not found; using empty label lists.")
        raw_ids = pd.Series([[] for _ in range(len(df))])

    labels_bin = raw_ids.apply(etpc_ids_to_binary)
    label_size = len(labels_bin.iloc[0]) if len(labels_bin) > 0 else 26

    # Prepare back-translator (optional)
    bt = None
    if args.bt_on:
        if not HAS_MARIAN:
            print("[WARN] BT requested but Marian/sentencepiece not available. Continuing without BT.")
        else:
            dev = args.bt_device if args.bt_device in {"cpu", "cuda"} else None
            print(f"[INFO] Initializing back-translator EN↔{args.bt_lang} (device={dev or ('cuda' if torch.cuda.is_available() else 'cpu')}, beams={args.bt_beams})")
            bt = BackTranslator(mid_lang=args.bt_lang, device=dev, num_beams=args.bt_beams)

    rows_out = []  # (id, s1, s2, label_vec, tag)

    # Originals
    for i, row in df.iterrows():
        rid = row["id"]
        s1 = str(row["sentence1"])
        s2 = str(row["sentence2"])
        lv = labels_bin.iloc[i]
        rows_out.append((f"{rid}", s1, s2, lv, "orig"))

    # Positives via BT (if enabled)
    it = range(len(df))
    if TQDM:
        it = tqdm(it, desc="Augmenting positives", leave=False)
    for i in it:
        if not bt:
            continue
        row = df.iloc[i]
        rid = row["id"]; s1 = str(row["sentence1"]); s2 = str(row["sentence2"])
        lv = labels_bin.iloc[i]

        do_bt_s1 = (random.random() < args.bt_frac_pos)
        do_bt_s2 = (random.random() < args.bt_frac_pos)
        if not (do_bt_s1 or do_bt_s2):
            continue

        cands = augment_positive_pair(
            s1, s2, lv, bt,
            do_bt_s1=do_bt_s1,
            do_bt_s2=do_bt_s2,
            mask_nums=(not args.no_mask_numbers),
            batch_size=1,
            max_length=args.max_length,
        )
        for k, (a_s1, a_s2, a_lv) in enumerate(cands):
            tag = f"pos_bt_{'s1' if do_bt_s1 else ''}{'s2' if do_bt_s2 else ''}".strip("_")
            rows_out.append((f"{rid}::{tag}::{k}", a_s1, a_s2, a_lv, tag))

        # Optional checkpointing
        if args.save_every and (i + 1) % args.save_every == 0 and not args.dry_run:
            tmp_df = build_output_df(rows_out, label_size)
            tmp_df["labels"] = tmp_df["labels_binary"].apply(lambda x: str(list(map(int, x))))
            tmp_df = tmp_df[["id", "sentence1", "sentence2", "labels", "source"]]
            tmp_path = args.output + ".part"
            tmp_df.to_csv(tmp_path, index=False)
            print(f"[INFO] Wrote checkpoint: {tmp_path} (rows={len(tmp_df)})")

    # Hard negatives by cross-mixing s2
    s2_pool = [str(x) for x in df["sentence2"].tolist()]
    num_negs = int(args.neg_frac * len(df))
    neg_indices = random.sample(range(len(df)), k=num_negs)
    it = neg_indices
    if TQDM:
        it = tqdm(neg_indices, desc="Creating hard negatives", leave=False)
    for i in it:
        row = df.iloc[i]
        rid = row["id"]; s1 = str(row["sentence1"])
        n_s1, n_s2, n_lv = augment_negative_mix(s1, s2_pool, label_size)

        tag = "neg_mix"
        if bt and (random.random() < args.bt_frac_neg):
            side = random.choice(["s1", "s2"])
            if side == "s1":
                n_s1 = bt([mask_numbers(n_s1) if not args.no_mask_numbers else n_s1], batch_size=1, max_length=args.max_length)[0]
                tag = "neg_mix_bt_s1"
            else:
                n_s2 = bt([mask_numbers(n_s2) if not args.no_mask_numbers else n_s2], batch_size=1, max_length=args.max_length)[0]
                tag = "neg_mix_bt_s2"

        rows_out.append((f"{rid}::{tag}", n_s1, n_s2, n_lv, tag))

    # Build output DataFrame
    out_df = build_output_df(rows_out, label_size)
    out_df["labels"] = out_df["labels_binary"].apply(lambda x: str(list(map(int, x))))
    out_df = out_df[["id", "sentence1", "sentence2", "labels", "source"]]

    # DRY RUN?
    if args.dry_run:
        print("\n[DRY RUN] Preview (first 5 rows):")
        print(out_df.head(5).to_string(index=False)[:1200])
        print(f"\n[STATS] Originals: {len(df)} | Total out: {len(out_df)} | Negatives added: ~{num_negs} | BT: {'on' if bt else 'off'}")
        return

    # Write final
    out_df.to_csv(args.output, index=False)
    print(f"[DONE] Wrote: {args.output}")
    print(f"       Originals: {len(df)} | Total (with aug): {len(out_df)} | Negatives added: ~{num_negs} | BT: {'on' if bt else 'off'}")


if __name__ == "__main__":
    main()
