"""
Finetuned BART Generation model.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

# ============== Config ==============
TQDM_DISABLE = False
DEFAULT_MODEL = "facebook/bart-large"
# ====================================


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def maybe_add_control_tokens(tokenizer, model, train_df, dev_df, test_df):
    """
    If dataset contains paraphrase control columns, register special tokens and resize embeddings.
    Controls supported (optional):
      - 'paraphrase_type_id' (int/str) -> <TYPE_k>
      - 'span' (str) -> span text kept literal between <SEP> ... <SEP>
    """
    needs_sep = any(("span" in df.columns) for df in [train_df, dev_df, test_df] if df is not None)
    type_vals = set()

    for df in [train_df, dev_df, test_df]:
        if df is None or "paraphrase_type_id" not in df.columns:
            continue
        for v in df["paraphrase_type_id"].dropna().tolist():
            try:
                type_vals.add(int(v))
            except Exception:
                type_vals.add(abs(hash(str(v))) % 10)

    addl_tokens = []
    if needs_sep:
        addl_tokens.append("<SEP>")
    for k in sorted(type_vals):
        addl_tokens.append(f"<TYPE_{k}>")

    if addl_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": addl_tokens})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens: {addl_tokens}")
    return tokenizer, model


def build_control_suffix(row):
    parts = []
    if "span" in row and pd.notna(row["span"]):
        parts += ["<SEP>", str(row["span"]), "<SEP>"]
    if "paraphrase_type_id" in row and pd.notna(row["paraphrase_type_id"]):
        try:
            k = int(row["paraphrase_type_id"])
        except Exception:
            k = abs(hash(str(row["paraphrase_type_id"]))) % 10
        parts += [f"<TYPE_{k}>"]
    return (" " + " ".join(parts)) if parts else ""


def transform_data(
    dataset: pd.DataFrame,
    tokenizer,
    max_src_len=256,
    max_tgt_len=64,
    shuffle=True,
    batch_size=16,
):
    """
    Tokenize once (inputs and, if present, targets) and pack into tensors.
    For inputs, optionally append control tokens if corresponding columns exist.
    """
    assert "sentence1" in dataset.columns, "dataset must contain 'sentence1'"

    texts = []
    for _, row in dataset.iterrows():
        s1 = str(row["sentence1"])
        ctrl = build_control_suffix(row)
        texts.append(s1 + ctrl)

    enc = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_src_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_masks = enc["attention_mask"]

    has_targets = "sentence2" in dataset.columns
    if has_targets:
        targets = [str(x) for x in dataset["sentence2"].tolist()]
        tgt_enc = tokenizer(
            targets,
            add_special_tokens=True,
            max_length=max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt_enc["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        ds = TensorDataset(input_ids, attention_masks, labels)
    else:
        ds = TensorDataset(input_ids, attention_masks)

    # CUDA-friendly DataLoader defaults
    pin = torch.cuda.is_available()
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin,
        num_workers=4 if pin else 0,
        persistent_workers=True if pin else False,
    )
    return dl


@torch.no_grad()
def generate_batch(model, tokenizer, input_ids, attention_mask,
                   max_gen_len=64, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0):
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_gen_len,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        early_stopping=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def evaluate_model(model, dev_df, device, tokenizer,
                   max_gen_len=64, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0,
                   max_src_len=256, batch_size=16):
    """
    Returns penalized BLEU with correct orientation and prints in the requested style.
    """
    model.eval()
    bleu = BLEU()

    dl = transform_data(dev_df, tokenizer, max_src_len=max_src_len, shuffle=False, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in dl:
            if len(batch) == 3:
                input_ids, attention_mask, _ = batch
            else:
                input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_len,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                early_stopping=True,
            )
            preds.extend(
                tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            )

    refs = dev_df["sentence2"].tolist()
    ins  = dev_df["sentence1"].tolist()

    bleu_ref = bleu.corpus_score(preds, [refs]).score
    bleu_inp = bleu.corpus_score(preds, [ins]).score
    neg_bleu_inp = 100.0 - bleu_inp
    penalized = bleu_ref * neg_bleu_inp / 52.0

    # >>> Requested print style <<<
    print("- Finetuned BART Generation")
    print(f"  - BLEU Score: {bleu_ref:.2f}")
    print(f"  - Negative BLEU Score with input: {neg_bleu_inp:.2f}")
    print(f"  - Penalized BLEU Score: {penalized:.2f}")

    return penalized



def train_model(
    model,
    train_loader,
    dev_df,
    device,
    tokenizer,
    lr=3e-5,
    weight_decay=0.01,
    num_epochs=6,
    warmup_ratio=0.1,
    grad_accum=2,
    patience=2,
    max_gen_len=64,
    num_beams=5,
    no_repeat_ngram_size=3,
    length_penalty=1.0,
    max_src_len=256,
    batch_size=16,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) * num_epochs) // max(1, grad_accum)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # New torch.amp API
    from torch import amp
    amp_enabled = device.type == "cuda"
    scaler = amp.GradScaler('cuda') if amp_enabled else None

    best_score = -1e9
    bad_epochs = 0
    ckpt_dir = Path("checkpoints/bart-best")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=TQDM_DISABLE)
        for batch in pbar:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if amp_enabled:
                with amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_accum
                loss.backward()

            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += loss.item() * grad_accum
            n_batches += 1
            pbar.set_postfix(loss=f"{(total_loss / n_batches):.4f}")

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Validate
        val_score = evaluate_model(
            model, dev_df, device, tokenizer,
            max_gen_len=max_gen_len,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            max_src_len=max_src_len,
            batch_size=batch_size,
        )

        if val_score > best_score:
            best_score = val_score
            bad_epochs = 0
            print(f"New best penalized BLEU: {best_score:.2f} — saving checkpoint")
            model.save_pretrained(ckpt_dir.as_posix())
            tokenizer.save_pretrained(ckpt_dir.as_posix())
        else:
            bad_epochs += 1
            print(f"No improvement ({bad_epochs}/{patience}).")
            if bad_epochs > patience:
                print("Early stopping.")
                break

    # Load best checkpoint
    if ckpt_dir.exists():
        model = BartForConditionalGeneration.from_pretrained(ckpt_dir.as_posix()).to(device)
        print(f"Loaded best checkpoint from {ckpt_dir}")
    return model


def test_model(test_dl, test_ids, device, model, tokenizer,
               max_gen_len=64, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0):
    model.eval()
    generated = []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Generating Paraphrases", disable=TQDM_DISABLE):
            # Accept 2-tensor or 3-tensor batches
            if len(batch) == 3:
                input_ids, attention_mask, _ = batch
            else:
                input_ids, attention_mask = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            pred_text = generate_batch(
                model, tokenizer, input_ids, attention_mask,
                max_gen_len=max_gen_len, num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size, length_penalty=length_penalty
            )
            generated.extend(pred_text)

    return pd.DataFrame({"id": test_ids, "Generated_sentence2": generated})


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--use_gpu", action="store_true")

    # Data/paths
    p.add_argument("--train_csv", type=str, default="data/etpc-paraphrase-train.csv")
    p.add_argument("--dev_csv", type=str, default="data/etpc-paraphrase-dev.csv")
    p.add_argument("--test_csv", type=str, default="data/etpc-paraphrase-generation-test-student.csv")
    p.add_argument("--pred_path", type=str, default="finetune_predictions/etpc-paraphrase-generation-test-output.csv")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL)

    # Optimization
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)

    # Lengths
    p.add_argument("--max_src_len", type=int, default=256)
    p.add_argument("--max_tgt_len", type=int, default=64)
    p.add_argument("--max_gen_len", type=int, default=64)

    # Decoding
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--length_penalty", type=float, default=1.0)

    return p.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)

    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # Load datasets
    train_df = pd.read_csv(args.train_csv)
    dev_df = pd.read_csv(args.dev_csv)
    test_df = pd.read_csv(args.test_csv)

    # Optional control tokens
    tokenizer, model = maybe_add_control_tokens(tokenizer, model, train_df, dev_df, test_df)

    # Dataloaders
    train_dl = transform_data(
        train_df, tokenizer,
        max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
        shuffle=True, batch_size=args.batch_size,
    )
    _ = transform_data(
        dev_df, tokenizer,
        max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
        shuffle=False, batch_size=args.batch_size,
    )  # sanity

    # Train
    model = train_model(
        model, train_dl, dev_df, device, tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        grad_accum=args.grad_accum,
        patience=args.patience,
        max_gen_len=args.max_gen_len,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        max_src_len=args.max_src_len,
        batch_size=args.batch_size,
    )

    # Final dev eval on best checkpoint
    final_dev_score = evaluate_model(
        model, dev_df, device, tokenizer,
        max_gen_len=args.max_gen_len,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        max_src_len=args.max_src_len,
        batch_size=args.batch_size,
    )
    print(f"The penalized BLEU-score of the best model is: {final_dev_score:.3f}")

    # Test inference
    test_dl = transform_data(
        test_df, tokenizer,
        max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
        shuffle=False, batch_size=args.batch_size,
    )
    test_ids = test_df["id"].tolist()
    pred_df = test_model(
        test_dl, test_ids, device, model, tokenizer,
        max_gen_len=args.max_gen_len,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
    )

    out_path = Path(args.pred_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path.as_posix(), index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
