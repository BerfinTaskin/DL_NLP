import argparse, os
from .simcse_core import run_simcse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=11711)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean","cls","pooler"])
    ap.add_argument("--symmetric", dest="symmetric", action="store_true", default=True)
    ap.add_argument("--no-symmetric", dest="symmetric", action="store_false")
    ap.add_argument("--save_dir", type=str, default="models")
    args = ap.parse_args()

    ckpt = os.path.join(args.save_dir, "simcse-qqp.pt")
    thr, dev_acc = run_simcse(
        pooling=args.pooling,
        temperature=args.temperature,
        lr=args.lr,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
        local_files_only=args.local_files_only,
        symmetric=args.symmetric,
        write_csvs=True,
        save_ckpt_path=ckpt,
    )
    print(f"[SimCSE] picked threshold={thr:.4f}, dev_acc={dev_acc*100:.3f}%")

if __name__ == "__main__":
    main()
