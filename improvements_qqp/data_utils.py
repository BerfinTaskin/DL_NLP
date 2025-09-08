import csv
import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_multitask_data, SentencePairDataset
from .config_qqp import QQPConfig

def _ensure_four_tuples(pairs):
    """Normalize (s1, s2, id) -> (s1, s2, dummy_label, id) so SentencePairDataset can collate."""
    fixed = []
    for t in pairs:
        if len(t) == 4:
            fixed.append(t)
        elif len(t) == 3:
            s1, s2, ex_id = t
            fixed.append((s1, s2, 0, ex_id))  # dummy label
        else:
            raise ValueError(f"Unexpected tuple length {len(t)}; expected 3 or 4. Got: {t}")
    return fixed


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(use_gpu: bool):
    return torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")

def load_quora_lists(cfg: QQPConfig, split: str):
    """
    Returns raw tuples as produced by your project's loader.
    """
    # filenames for SST/STS/ETPC are required but we ignore here (pass dummy paths)
    sst, nlab, quora, sts, etpc = load_multitask_data(
        sst_filename="data/sst-sentiment-train.csv" if split!="test" else "data/sst-sentiment-test-student.csv",
        quora_filename=cfg.quora_train if split=="train" else (cfg.quora_dev if split=="dev" else cfg.quora_test),
        sts_filename="data/sts-similarity-train.csv" if split!="test" else "data/sts-similarity-test-student.csv",
        etpc_filename="data/etpc-paraphrase-train.csv",  # unused here
        split=split
    )
    return quora  # list of tuples

def build_simcse_dataset_list(quora_train_list):
    """
    Keep only positive pairs for supervised SimCSE.
    Each item: (s1, s2, label=1, id)
    """
    pos = [t for t in quora_train_list if len(t)==4 and int(t[2])==1]
    return pos

def dataloader_from_list(pairs_list, args_namespace, batch_size: int, is_regression: bool=False, shuffle=True):
    """
    Using the SentencePairDataset so tokenization & batching mirror the baseline.
    """
    pairs_list = _ensure_four_tuples(pairs_list)
    ds = SentencePairDataset(pairs_list, args_namespace, isRegression=is_regression)
    return DataLoader(ds, shuffle=shuffle, batch_size=batch_size, collate_fn=ds.collate_fn)