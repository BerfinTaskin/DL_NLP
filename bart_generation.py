import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW


TQDM_DISABLE = False


def transform_data(dataset, max_length=256):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase_type_ids.
    Return Data Loader.
    """
    ### TODO 
    raise NotImplementedError


def train_model(model, train_data, dev_data, device, tokenizer):
    """
    Train the model. Return and save the model.
    """
    ### TODO
    raise NotImplementedError


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    ### TODO
    raise NotImplementedError


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and 
    the column "sentence2" contains all target sentences
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

            # Generate paraphrases
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
            
            predictions.extend(pred_text)

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    model.train()
    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")
    

    # Penalize BLEU and rescale it to 0-100
    # If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    return penalized_bleu


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


def finetune_paraphrase_detection(args):
    import json
    print(f"Using arguments: {args}")
    model = BartWithClassifier(num_labels=26)  # Pass model_name
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    print("Loading data...")
    full_train_dataset_df = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset_df = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # --- Train/Validation Split ---
    full_train_dataset_df = full_train_dataset_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    num_dev_samples = int(len(full_train_dataset_df) * args.dev_split_ratio)

    if num_dev_samples == 0 and len(full_train_dataset_df) > 0:
        num_dev_samples = 1
    if num_dev_samples > len(full_train_dataset_df) - 1:
        num_dev_samples = len(full_train_dataset_df) - 1 if len(full_train_dataset_df) > 0 else 0

    if num_dev_samples > 0:
        dev_df = full_train_dataset_df.iloc[:num_dev_samples]
        train_df = full_train_dataset_df.iloc[num_dev_samples:]
    else:
        dev_df = full_train_dataset_df.copy()
        train_df = full_train_dataset_df.copy()

    print(f"Training with {len(train_df)} samples, validating with {len(dev_df)} samples.")

    # Transform data into DataLoaders
    print("Transforming training data...")
    train_dataloader = transform_data(train_df, tokenizer_name=args.model_name, max_length=args.max_length, batch_size=args.batch_size)
    print("Transforming development data...")
    dev_dataloader = transform_data(dev_df, tokenizer_name=args.model_name, max_length=args.max_length, batch_size=args.batch_size)
    print("Transforming test data...")
    test_dataloader = transform_data(test_dataset_df, tokenizer_name=args.model_name, max_length=args.max_length, batch_size=args.batch_size)

    print(f"Loaded {len(train_df)} training samples for DataLoader.")

    # Train
    model = train_model(
        model,
        train_dataloader,
        dev_dataloader,
        device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience
    )

    print("\nTraining finished.")
    print("Evaluating on the development set one last time...")
    val_accuracy, val_f1 = evaluate_model(model, dev_dataloader, device)
    print(f"Final Development Accuracy: {val_accuracy:.3f}")
    print(f"Final Development F1: {val_f1:.3f}")

    # Test
    test_ids = test_dataset_df["id"]
    test_results_df = test_model(model, test_dataloader, test_ids, device)

    # Save predictions
    output_dir = "predictions/bart/"
    os.makedirs(output_dir, exist_ok=True)
    test_results_df.to_csv(os.path.join(output_dir, "etpc-paraphrase-detection-test-output.csv"), index=False)
    print(f"Test predictions saved to {os.path.join(output_dir, 'etpc-paraphrase-detection-test-output.csv')}")

    # ==== Save metrics ====
    metrics = {
        "job_id": args.job_id,
        "approach": args.approach,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1
    }
    os.makedirs("metrics_logs", exist_ok=True)
    outfile = f"metrics_logs/{args.approach}_{args.job_id}.json"
    with open(outfile, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {outfile}")


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
