import pandas as pd
import random
import ast

# Set seed for reproducibility
random.seed(42)

# File paths
input_path = "data/etpc-paraphrase-train-and-dev.csv"
train_output_path = "data/etpc-paraphrase-train.csv"
dev_output_path = "data/etpc-paraphrase-dev.csv"

# Load data
df = pd.read_csv(input_path)

# Parse the paraphrase_type_ids column
df['paraphrase_type_ids'] = df['paraphrase_type_ids'].apply(ast.literal_eval)

# Get sorted list of 7 unique labels (assert)
all_labels = sorted(set(label for row in df['paraphrase_type_ids'] for label in row))
assert len(all_labels) == 26, f"Expected 26 unique labels, got {len(all_labels)}: {all_labels}"
assert max(all_labels) == 31, f"Expected max label to be 31, got {max(all_labels)}"

# Randomly sample 400 rows for dev set
dev_df = df.sample(n=400, random_state=42)
train_df = df.drop(dev_df.index)
assert len(train_df) + len(dev_df) == len(df), "Train and dev sets do not sum to original dataset length."

# Save to new files
train_df.to_csv(train_output_path, index=False)
dev_df.to_csv(dev_output_path, index=False)

print("Done. Processed and saved to:")
print(f"- {train_output_path} ({len(train_df)} rows)")
print(f"- {dev_output_path} ({len(dev_df)} rows)")
