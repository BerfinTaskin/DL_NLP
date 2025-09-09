#!/usr/bin/env python3
import pandas as pd
import ast

# Path to your file
path = "data/etpc-paraphrase-train-and-dev.csv"

# Load CSV
df = pd.read_csv(path)

# Helper to parse the paraphrase_type_ids column into Python lists
def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

df["parsed_ids"] = df["paraphrase_type_ids"].apply(to_list)

# Count positives (at least one type) vs negatives (empty list)
num_pos = (df["parsed_ids"].apply(len) > 0).sum()
num_neg = (df["parsed_ids"].apply(len) == 0).sum()

print(f"File: {path}")
print(f"Total pairs: {len(df)}")
print(f"Positive pairs: {num_pos}")
print(f"Negative pairs: {num_neg}")
