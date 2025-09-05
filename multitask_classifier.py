import argparse
import os
from pprint import pformat
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW

TQDM_DISABLE = True


# === helpers (top of file) ===
@torch.no_grad()
def pearsonr_torch(x, y):
    # x,y: (N,) float
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.norm() * vy.norm()).clamp(min=1e-12)
    return (vx * vy).sum() / denom



def mean_pool(last_hidden, attention_mask):
    # last_hidden: (B, T, H), attention_mask: (B, T)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B,T,1)
    summed = (last_hidden * mask).sum(dim=1)                   # (B,H)
    counts = mask.sum(dim=1).clamp(min=1e-6)                   # (B,1)
    return summed / counts


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for these tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    (- Paraphrase type detection (predict_paraphrase_types))
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True
        ### TODO
        # sentiment
        self.sentiment_linear = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)
        # paraphrase
        self.paraphrase_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_linear = nn.Linear(2*BERT_HIDDEN_SIZE, 1)
        # similarity
        self.similarity_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.similarity_linear = nn.Linear(2*BERT_HIDDEN_SIZE, 1)
        
        # replace your old STS head with a Siamese MLP head
        self.sts_hidden = 256
        # input dim = 4H: [h1, h2, |h1-h2|, h1*h2]
        self.similarity_mlp = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(4 * BERT_HIDDEN_SIZE, self.sts_hidden),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.sts_hidden, 1),
        )
        # paraphrase types
        self.paraphrase_type_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_type_linear = nn.Linear(2*BERT_HIDDEN_SIZE, 26)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)
        last_hidden = bert_output['last_hidden_state']   # (B,T,H)
        pooled = mean_pool(last_hidden, attention_mask)  # (B,H)  <-- changed
        return pooled
    # === inside MultitaskBERT (add a pooled encoder) ===
    def encode_pooled(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        last_hidden = out["last_hidden_state"]         # (B,T,H)
        return mean_pool(last_hidden, attention_mask)  # (B,H)

    def predict_sentiment(self, input_ids, attention_mask):
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        Dataset: SST
        """
        ### TODO
        cls_embedding = self.forward(input_ids, attention_mask)
        cls_embedding = self.sentiment_dropout(cls_embedding)
        output_logits = self.sentiment_linear(cls_embedding)
        return output_logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora
        """
        ### TODO
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)
        cls_embedding = torch.cat((cls_embedding_1, cls_embedding_2), dim=1)
        cls_embedding = self.paraphrase_dropout(cls_embedding)
        output_logits = self.paraphrase_linear(cls_embedding)
        return output_logits

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        h1 = self.encode_pooled(input_ids_1, attention_mask_1)    # (B,H)
        h2 = self.encode_pooled(input_ids_2, attention_mask_2)    # (B,H)
        diff = torch.abs(h1 - h2)
        prod = h1 * h2
        feats = torch.cat([h1, h2, diff, prod], dim=1)            # (B, 4H)
        score = self.similarity_mlp(feats).squeeze(-1)            # (B,)
        # raw score; we won't clamp here. We’ll train with SmoothL1 to target in [0,5].
        return score

    def predict_paraphrase_types(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs logits for detecting the paraphrase types.
        There are 26 different types of paraphrases.
        Thus, your output should contain 26 unnormalized logits for each sentence. It will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: ETPC
        """
        ### TODO
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)
        cls_embedding = torch.cat((cls_embedding_1, cls_embedding_2), dim=1)
        cls_embedding = self.paraphrase_type_dropout(cls_embedding)
        output_logits = self.paraphrase_type_linear(cls_embedding)
        return output_logits

def clone_state_dict(cpu_state):
    return {k: v.clone() for k, v in cpu_state.items()}

def average_state_dicts(state_dicts):
    if not state_dicts:
        return None
    avg = {}
    keys = state_dicts[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k] for sd in state_dicts], dim=0)
        avg[k] = stacked.mean(dim=0)
    return avg

def save_model(model, optimizer, args, config, filepath):
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")

# top of file
import json
import os


# helper
def update_metrics_log(args, task, val_accuracy):
    # create folder for this task
    os.makedirs(f"metrics_logs_{task}", exist_ok=True)
    path = os.path.join(f"metrics_logs_{task}", f"{args.approach}.json")

    data = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    # write top-level info once
    data.setdefault("approach", args.approach)
    data.setdefault("hyperparams", {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "option": args.option,
        "hidden_dropout_prob": args.hidden_dropout_prob
    })

    # write/update this task’s section
    data[task] = {
        "val_accuracy": float(val_accuracy)
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Metrics merged into {path}")


# TODO Currently only trains on SST dataset!
def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None
    
    # SST dataset
    if args.task == "sst" or args.task == "multitask":
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(
            sst_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_data.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_data.collate_fn,
        )
        
    ### TODO
    #   Load data for the other datasets
    # If you are doing the paraphrase type detection with the minBERT model as well, make sure
    # to transform the data labels into binaries (as required in the bart_detection.py script)
    # STS dataset
    if args.task == "sts" or args.task == "multitask":
        sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
        
        sts_train_dataloader = DataLoader(
            sts_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sts_train_data.collate_fn,
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sts_dev_data.collate_fn,
        )
        
    # Quora dataset
    if args.task == "qqp" or args.task == "multitask":
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)

        quora_train_dataloader = DataLoader(
            quora_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=quora_train_data.collate_fn,
        )
        quora_dev_dataloader = DataLoader(
            quora_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=quora_dev_data.collate_fn,
        )
    # ETPC dataset
    if args.task == "etpc" or args.task == "multitask":
        etpc_train_data = SentencePairDataset(etpc_train_data, args)
        etpc_dev_data = SentencePairDataset(etpc_dev_data, args)

        etpc_train_dataloader = DataLoader(
            etpc_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=etpc_train_data.collate_fn,
        )
        etpc_dev_dataloader = DataLoader(
            etpc_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=etpc_dev_data.collate_fn,
        )
    # Build dataloaders above already
    total_train_steps = 0
    if sst_train_dataloader is not None:
        total_train_steps += len(sst_train_dataloader)
    if sts_train_dataloader is not None:
        total_train_steps += len(sts_train_dataloader)
    if quora_train_dataloader is not None:
        total_train_steps += len(quora_train_dataloader)
    if etpc_train_dataloader is not None:
        total_train_steps += len(etpc_train_dataloader)
    total_train_steps *= args.epochs

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }

    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = float("-inf")
    
    scheduler = None
    if args.use_cosine_schedule and total_train_steps > 0:
        warmup_steps = int(args.warmup_ratio * total_train_steps)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # cosine from warmup_steps .. total_train_steps
            progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_dev_acc = float("-inf")
 
    best_sts_pearson = float("-inf")
    no_improve = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.task == "sst" or args.task == "multitask":
            # Train the model on the sst dataset.

            for batch in tqdm(
                sst_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids, b_mask, b_labels = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(
                    logits, b_labels.view(-1),
                    label_smoothing=getattr(args, "label_smoothing", 0.0)
                )

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "sts" or args.task == "multitask":
            # Trains the model on the sts dataset
            ### TODO
            for batch in tqdm(
                sts_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)
                optimizer.zero_grad()
                pred = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)   # (B,)
                loss = torch.nn.functional.smooth_l1_loss(pred.view(-1), b_labels.view(-1),
                                                        beta=args.sts_beta)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                train_loss += loss.item()
                num_batches += 1

        if args.task == "qqp" or args.task == "multitask":
            # Trains the model on the qqp dataset
            ### TODO
            for batch in tqdm(
                quora_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "etpc" or args.task == "multitask":
            # Trains the model on the etpc dataset
            ### TODO
            for batch in tqdm(
                etpc_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase_types(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / num_batches

        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        quora_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (0, 0),  # TODO
        }[args.task]

        print(
            f"Epoch {epoch+1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        if args.task == "sts":
            current_dev_pearson = float(sts_dev_corr)  # already computed by your evaluator
            if current_dev_pearson > best_sts_pearson:
                best_sts_pearson = current_dev_pearson
                no_improve = 0
                # keep a checkpoint that matches your save logic
                save_model(model, optimizer, args, config, args.filepath)
                # you can also log here via update_metrics_log(...)
            else:
                no_improve += 1
                if no_improve >= args.early_stop_patience:
                    print(f"[EARLY STOP] STS: no Pearson improvement for {args.early_stop_patience} epochs.")
                    break


    # if you only want to log the BEST epoch, put this inside your "if dev_acc > best_dev_acc:" block
    update_metrics_log(args, args.task, best_dev_acc)




def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    
    # in get_args()
    parser.add_argument("--sts_beta", type=float, default=1.0, help="SmoothL1 beta for STS")
    parser.add_argument("--early_stop_patience", type=int, default=2, help="patience (epochs) for STS Pearson early stop")

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="sst",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="Fraction of total steps to warm up")
    parser.add_argument("--use_cosine_schedule", action="store_true",
                        help="Enable warmup + cosine LR schedule")
    parser.add_argument("--ckpt_avg_k", type=int, default=3,
                    help="Average this many best checkpoints for the final model (SST)")


    # Model configuration
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")

    args, _ = parser.parse_known_args()

    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/sst-sentiment-test-student.csv")

    parser.add_argument("--quora_train", type=str, default="data/quora-paraphrase-train.csv")
    parser.add_argument("--quora_dev", type=str, default="data/quora-paraphrase-dev.csv")
    parser.add_argument("--quora_test", type=str, default="data/quora-paraphrase-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-similarity-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-similarity-test-student.csv")

    # TODO
    # You should split the train data into a train and dev set first and change the
    # default path of the --etpc_dev argument to your dev set.
    parser.add_argument("--etpc_train", type=str, default="data/etpc-paraphrase-train.csv")
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")
    parser.add_argument(
        "--etpc_test", type=str, default="data/etpc-paraphrase-detection-test-student.csv"
    )

    # Output paths
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )

    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-output.csv"
        ),
    )

    # Hyperparameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--approach", type=str, required=True,
                    help="Short name/description for the experiment")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
