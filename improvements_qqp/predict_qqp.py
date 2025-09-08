import argparse, numpy as np, torch, os
from types import SimpleNamespace
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoder import SentenceEncoder
from data_utils import dataloader_from_list, load_quora_lists, write_predictions_csv, get_device

def cosine(a,b):
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a*b).sum(dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", choices=["dev","test"], required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--thr", type=float, default=0.0)  # set from prior search
    args = ap.parse_args()

    device = get_device(args.use_gpu)
    model = SentenceEncoder(local_files_only=args.local_files_only, finetune=True).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    from config_qqp import QQPConfig
    cfg = QQPConfig()
    data = load_quora_lists(cfg, split=args.split)
    ns = SimpleNamespace(local_files_only=args.local_files_only)
    loader = dataloader_from_list(data, ns, cfg.batch_size, shuffle=False)

    ids, preds = [], []
    with torch.no_grad():
        for b in tqdm(loader, desc=f"predict-{args.split}"):
            z1 = model(b["token_ids_1"].to(device), b["attention_mask_1"].to(device))
            z2 = model(b["token_ids_2"].to(device), b["attention_mask_2"].to(device))
            s = cosine(z1, z2).cpu().numpy()
            p = (s >= args.thr).astype(np.int32)
            preds.extend(p.tolist()); ids.extend(b["sent_ids"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_predictions_csv(args.out, ids, preds)

if __name__ == "__main__":
    main()
