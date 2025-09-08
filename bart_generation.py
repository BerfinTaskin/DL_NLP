"""
Baseline BART Generation model.
"""


import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
import sys
sys.path.append('/home/berfintskn/DL_NLP')
from optimizer import AdamW


TQDM_DISABLE = False


def transform_data(dataset, max_length=256, shuffle=True):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase_type_ids.
    Return Data Loader.
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    input_ids = []
    attention_masks = []
    # For training/dev, we need targets (sentence2)
    has_targets = "sentence2" in dataset.columns
    targets = []
    for idx, row in dataset.iterrows():
        sentence1 = str(row["sentence1"])
        encoded = tokenizer(
            sentence1,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"].squeeze(0))
        attention_masks.append(encoded["attention_mask"].squeeze(0))
        if has_targets:
            targets.append(str(row["sentence2"]))
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    if has_targets:
        dataset = TensorDataset(input_ids, attention_masks, torch.arange(len(input_ids)))
    else:
        dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=shuffle)
    # Store targets for use in train_model
    dataloader.targets = targets if has_targets else None
    return dataloader


def train_model(model, train_data, dev_data, device, tokenizer):
    """
    Train the model. Return and save the model.
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    model.train()
    best_bleu = -float('inf')
    # Get targets for teacher forcing
    train_targets = train_data.targets
    for epoch in range(num_epochs):
        total_loss = 0
        batch_idx = 0
        for batch in tqdm(train_data, desc=f"Training Epoch {epoch+1}", disable=TQDM_DISABLE):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            # Get corresponding targets for this batch
            if train_targets is not None:
                indices = batch[2].cpu().numpy()
                target_texts = [train_targets[i] for i in indices]
                with tokenizer.as_target_tokenizer():
                    target_encodings = tokenizer(
                        target_texts,
                        max_length=64,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                labels = target_encodings["input_ids"].to(device)
                labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss
            else:
                labels = None
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_idx += 1
        avg_loss = total_loss / batch_idx
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    model.eval()
    generated_sentences = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating Paraphrases", disable=TQDM_DISABLE):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )
            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            generated_sentences.extend(pred_text)
    # test_ids is a pd.Series or list
    df = pd.DataFrame({
        "id": test_ids,
        "Generated_sentence2": generated_sentences
    })
    return df


def evaluate_model(model, test_data, device, tokenizer):
    """
    Evaluate on dev/test with SacreBLEU.
    Correct orientation:
      - bleu_ref = BLEU(preds, refs)
      - bleu_inp = BLEU(preds, inputs)  # we penalize similarity to inputs
    Penalized BLEU = bleu_ref * (100 - bleu_inp) / 52
    """
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(test_data, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            predictions.extend(
                tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            )

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    bleu_ref = bleu.corpus_score(predictions, [references]).score
    bleu_inp = bleu.corpus_score(predictions, [inputs]).score
    neg_bleu_inp = 100.0 - bleu_inp
    penalized = bleu_ref * neg_bleu_inp / 52.0

    # >>> Requested print style <<<
    print("- BART Generation")
    print(f"  - BLEU Score: {bleu_ref:.2f}")
    print(f"  - Negative BLEU Score with input: {neg_bleu_inp:.2f}")
    print(f"  - Penalized BLEU Score: {penalized:.2f}")

    model.train()
    return penalized



def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv")

    # You might do a split of the train data into train/validation set here
    # ...

    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset, shuffle=False)
    test_data = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_dataset, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_dataset, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output3.csv", index=False
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)