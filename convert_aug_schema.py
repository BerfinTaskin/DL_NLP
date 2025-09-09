#!/usr/bin/env python3
import argparse, ast, re
import pandas as pd

# ETPC types we dropped in training; used here to map 26/27-d vectors back to original ids.
DROP_TYPES = {12, 19, 20, 23, 27}
KEPT_ORIG_IDS = [i for i in range(0, 32) if i not in DROP_TYPES]  # e.g., [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,21,22,24,25,26,28,29,30,31]

TOKEN_PAT = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\$€£]\d[\d,]*(?:\.\d+)?|[^\w\s]")
def simple_tokenize(text: str):
    return TOKEN_PAT.findall(str(text) if text is not None else "")

def parse_labels(x):
    """Accept '[0,1,...]' or python list; return list[int] of 0/1."""
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        try:
            lst = ast.literal_eval(x)
            return [int(v) for v in lst]
        except Exception:
            s = x.strip().strip("[]")
            if not s:
                return []
            return [int(t.strip()) for t in s.split(",")]
    return []

def binvec_to_orig_ids(binvec):
    """Positions with 1 -> KEPT_ORIG_IDS[pos]."""
    out = []
    for j, v in enumerate(binvec):
        if int(v) == 1:
            # guard: if vector is longer than KEPT_ORIG_IDS, ignore extras
            if j < len(KEPT_ORIG_IDS):
                out.append(KEPT_ORIG_IDS[j])
    return out

def list_to_str(lst):
    return str(list(lst))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="augmented CSV with columns id,sentence1,sentence2,labels,source")
    ap.add_argument("--output", required=True, help="path to write ETPC-format CSV")
    ap.add_argument("--segment_mode", choices=["pass", "broadcast"], default="pass",
                    help="pass: empty seg-locs for all rows; broadcast: fill per-token ids by cycling type ids.")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    needed = {"id","sentence1","sentence2","labels"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"[ERROR] Input must have columns {needed}. Got: {list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        s1 = str(r["sentence1"])
        s2 = str(r["sentence2"])
        # Convert labels -> original paraphrase_type_ids
        binvec = parse_labels(r["labels"])
        type_ids = binvec_to_orig_ids(binvec)

        tok1 = simple_tokenize(s1)
        tok2 = simple_tokenize(s2)

        if args.segment_mode == "broadcast":
            base = list(dict.fromkeys(type_ids)) or [25]
            seg1 = [base[i % len(base)] for i in range(len(tok1))]
            seg2 = [base[i % len(base)] for i in range(len(tok2))]
        else:
            seg1, seg2 = [], []

        rows.append({
            "id": r["id"],
            "sentence1": s1,
            "sentence2": s2,
            "paraphrase_type_ids": list_to_str(type_ids),
            "sentence1_segment_location": list_to_str(seg1),
            "sentence2_segment_location": list_to_str(seg2),
            "sentence1_tokenized": str(tok1),
            "sentence2_tokenized": str(tok2),
        })

    out = pd.DataFrame(rows, columns=[
        "id","sentence1","sentence2",
        "paraphrase_type_ids",
        "sentence1_segment_location","sentence2_segment_location",
        "sentence1_tokenized","sentence2_tokenized"
    ])
    out.to_csv(args.output, index=False)
    print(f"[DONE] Wrote: {args.output}  (rows={len(out)})")

if __name__ == "__main__":
    main()
