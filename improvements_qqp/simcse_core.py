import os
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config_qqp import QQPConfig
from .encoder import SentenceEncoder
from .losses import simcse_loss
from .data_utils import (
    set_seed, get_device, load_quora_lists, build_simcse_dataset_list,
    dataloader_from_list, write_predictions_csv
)

def _cosine(a, b):
    a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)
    return (a*b).sum(dim=1)

def find_best_threshold(dev_sims, dev_labels):
    # dense sweep in [-1,1]
    candidates = np.linspace(-1.0, 1.0, num=401)
    best_t, best_acc = 0.0, -1.0
    for t in candidates:
        pred = (dev_sims >= t).astype(np.int32)
        acc = (pred == dev_labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_t), float(best_acc)

def _build_loaders(cfg: QQPConfig):
    train_list = load_quora_lists(cfg, split="train")   # (s1,s2,label,id)
    dev_list   = load_quora_lists(cfg, split="dev")
    test_list  = load_quora_lists(cfg, split="test")

    # Supervised SimCSE uses only positives as (anchor, positive)
    simcse_list = build_simcse_dataset_list(train_list)

    ns = SimpleNamespace(local_files_only=cfg.local_files_only)
    train_loader = dataloader_from_list(simcse_list, ns, cfg.batch_size, is_regression=False, shuffle=True)
    dev_loader   = dataloader_from_list(dev_list,   ns, cfg.batch_size, is_regression=False, shuffle=False)
    test_loader  = dataloader_from_list(test_list,  ns, cfg.batch_size, is_regression=False, shuffle=False)
    return train_loader, dev_loader, test_loader, ns

def _build_model(cfg: QQPConfig, pooling: str):
    model = SentenceEncoder(local_files_only=cfg.local_files_only, finetune=True, pooling=pooling)
    opt   = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return model, opt

def train_epoch_simcse(model, opt, train_loader, temperature: float, device, symmetric: bool):
    model.train()
    pbar = tqdm(train_loader, desc="SimCSE-train")
    for batch in pbar:
        ids1 = batch["token_ids_1"].to(device)
        msk1 = batch["attention_mask_1"].to(device)
        ids2 = batch["token_ids_2"].to(device)
        msk2 = batch["attention_mask_2"].to(device)

        z1 = model(ids1, msk1)
        z2 = model(ids2, msk2)

        loss = simcse_loss(z1, z2, temperature=temperature)
        if symmetric:
            loss = loss + simcse_loss(z2, z1, temperature=temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

def eval_dev_threshold(model, dev_loader, device):
    model.eval()
    sims, labels, ids = [], [], []
    with torch.no_grad():
        for b in tqdm(dev_loader, desc="encode-dev"):
            z1 = model(b["token_ids_1"].to(device), b["attention_mask_1"].to(device))
            z2 = model(b["token_ids_2"].to(device), b["attention_mask_2"].to(device))
            sims.append(_cosine(z1, z2).cpu().numpy())
            labels.append(b["labels"].cpu().numpy())
            ids.extend(b["sent_ids"])
    sims = np.concatenate(sims)
    labels = np.concatenate(labels).astype(np.int32)
    thr, dev_acc = find_best_threshold(sims, labels)
    return thr, dev_acc, ids, sims, labels

def predict_split(model, loader, thr, device):
    model.eval()
    all_ids, all_preds = [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="predict"):
            z1 = model(b["token_ids_1"].to(device), b["attention_mask_1"].to(device))
            z2 = model(b["token_ids_2"].to(device), b["attention_mask_2"].to(device))
            s = _cosine(z1, z2).cpu().numpy()
            p = (s >= thr).astype(np.int32).tolist()
            all_preds.extend(p); all_ids.extend(b["sent_ids"])
    return all_ids, all_preds

def run_simcse(
    pooling: str = "mean",
    temperature: float = 0.05,
    lr: float = 1e-5,
    seed: int = 11711,
    epochs: int = 1,
    batch_size: int = 64,
    use_gpu: bool = True,
    local_files_only: bool = False,
    symmetric: bool = True,
    write_csvs: bool = True,
    save_ckpt_path: str | None = None,
):
    """
    Trains SimCSE, returns (dev_threshold, dev_accuracy). Optionally writes CSVs & saves checkpoint.
    """
    cfg = QQPConfig(
        batch_size=batch_size, max_epochs=epochs, lr=lr, temperature=temperature,
        seed=seed, use_gpu=use_gpu, local_files_only=local_files_only
    )
    set_seed(cfg.seed)
    device = get_device(cfg.use_gpu)

    # loaders + model/opt
    train_loader, dev_loader, test_loader, _ = _build_loaders(cfg)
    model, opt = _build_model(cfg, pooling)
    model = model.to(device)

    # train
    for _ in range(cfg.max_epochs):
        train_epoch_simcse(model, opt, train_loader, cfg.temperature, device, symmetric=symmetric)

    # dev threshold
    thr, dev_acc, dev_ids, dev_sims, dev_labels = eval_dev_threshold(model, dev_loader, device)

    # outputs
    if write_csvs:
        os.makedirs("predictions/bert", exist_ok=True)
        # dev predictions
        dev_pred = (dev_sims >= thr).astype(np.int32).tolist()
        write_predictions_csv(cfg.dev_out, dev_ids, dev_pred)
        # test predictions
        test_ids, test_pred = predict_split(model, test_loader, thr, device)
        write_predictions_csv(cfg.test_out, test_ids, test_pred)

    if save_ckpt_path:
        os.makedirs(os.path.dirname(save_ckpt_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_ckpt_path)

    return thr, float(dev_acc)
