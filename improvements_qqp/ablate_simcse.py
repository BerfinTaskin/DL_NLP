import argparse, os, csv, time, numpy as np
from .simcse_core import run_simcse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--runs", type=int, default=1, help="seeds per config")
    ap.add_argument("--out_csv", type=str, default="improvements_qqp/ablation_results_simcse.csv")
    ap.add_argument("--symmetric", dest="symmetric", action="store_true", default=True)
    ap.add_argument("--no-symmetric", dest="symmetric", action="store_false")
    args = ap.parse_args()

    poolings = ["mean", "cls"]
    temperatures = [0.05, 0.1]
    lrs = [1e-5, 3e-5]
    seeds = [11711 + k for k in range(args.runs)]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows = []
    start = time.time()
    for pooling in poolings:
        for tau in temperatures:
            for lr in lrs:
                accs, thrs = [], []
                for seed in seeds:
                    thr, dev_acc = run_simcse(
                        pooling=pooling,
                        temperature=tau,
                        lr=lr,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        use_gpu=args.use_gpu,
                        local_files_only=args.local_files_only,
                        symmetric=args.symmetric,
                        write_csvs=False,
                        save_ckpt_path=None,
                    )
                    accs.append(dev_acc); thrs.append(thr)
                    rows.append([pooling, tau, lr, seed, args.epochs, args.batch_size, thr, dev_acc])

                print(f"[GRID] pooling={pooling:>4s} tau={tau:<4} lr={lr:<7} "
                      f"-> dev_acc={np.mean(accs)*100:.2f}% (±{np.std(accs)*100:.2f}), thr≈{np.mean(thrs):.3f}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["pooling","temperature","lr","seed","epochs","batch_size","threshold","dev_acc"])
        for r in rows:
            w.writerow(r)

    dur = (time.time() - start) / 60.0
    print(f"\nWrote CSV: {args.out_csv} (took {dur:.1f} min)")

    # Print Markdown table
    agg = {}
    for (pooling, tau, lr, seed, epochs, bs, thr, acc) in rows:
        key = (pooling, tau, lr)
        agg.setdefault(key, {"accs": [], "thrs": []})
        agg[key]["accs"].append(acc)
        agg[key]["thrs"].append(thr)

    print("\nMarkdown table (avg over seeds):\n")
    print("| Pooling | Temp | LR    | Dev Acc (%) | Thr |")
    print("|:-------:|:----:|:-----:|:-----------:|:---:|")
    for (pooling, tau, lr), vals in sorted(agg.items(), key=lambda kv: -np.mean(kv[1]["accs"])):
        avg_acc = 100 * float(np.mean(vals["accs"]))
        avg_thr = float(np.mean(vals["thrs"]))
        print(f"| {pooling:6s} | {tau:<4} | {lr:<5} | {avg_acc:11.3f} | {avg_thr:.3f} |")

if __name__ == "__main__":
    main()
