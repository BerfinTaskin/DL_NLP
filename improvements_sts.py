import argparse
import os
from pprint import pformat
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from improvements_sts.datasets import SentencePairDataset
from datasets import load_multitask_data
from tokenizer import BertTokenizer
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW
from peft import LoraConfig, get_peft_model

TQDM_DISABLE = True
BERT_HIDDEN_SIZE = 768

###################################################################### actual modified multitawsk_classifier.py starts here
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class STSBERT(nn.Module):
    """
    This module should use BERT for the Semantic Textual Similarity (STS) task.
    """

    def __init__(self, config):
        super(STSBERT, self).__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )

        if config.use_lora:
            lora_config = LoraConfig(
                r=int(config.lora_r),
                lora_alpha=int(config.lora_alpha),
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,
            )
            self.bert = get_peft_model(self.bert, lora_config)
        else:
            for param in self.bert.parameters():
                if config.option == "pretrain":
                    param.requires_grad = False
                elif config.option == "finetune":
                    param.requires_grad = True

        if config.use_lora:
            # ensure LoRA adapter params remain trainable
            self.bert.print_trainable_parameters()

        # first pred style
        self.one_similarity_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.one_similarity_linear = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)
        
        # second pred style
        self.two_similarity_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.two_similarity_linear = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        
        self.head_style = config.head_style
        self.embedding_style = config.embedding_style

    def forward_cls(self, input_ids, attention_mask):
        """Return the [CLS] embedding."""
        bert_output = self.bert(input_ids, attention_mask)
        last_hidden = bert_output["last_hidden_state"]
        cls_embedding = last_hidden[:, 0, :]
        return cls_embedding

    def forward_mean_pooled(self, input_ids, attention_mask):
        """Return a mean-pooled embedding of all tokens (mask-aware)."""
        bert_output = self.bert(input_ids, attention_mask)
        last_hidden = bert_output["last_hidden_state"]            # [B, T, H]
        mask = attention_mask.unsqueeze(-1).float()               # [B, T, 1]
        sum_hidden = (last_hidden * mask).sum(dim=1)              # [B, H]
        lengths = mask.sum(dim=1).clamp(min=1.0)                  # [B, 1]
        mean_pooled = sum_hidden / lengths                        # [B, H]
        return mean_pooled  # [B, H]


    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of sentence pairs, output a similarity score in [0, 5].
        Dataset: STS
        """
        if self.head_style == 1:
            if self.embedding_style == "cls":
                embedding_1 = self.forward_cls(input_ids_1, attention_mask_1)
                embedding_2 = self.forward_cls(input_ids_2, attention_mask_2)
            elif self.embedding_style == "mean":
                embedding_1 = self.forward_mean_pooled(input_ids_1, attention_mask_1)
                embedding_2 = self.forward_mean_pooled(input_ids_2, attention_mask_2)
            embedding = torch.cat((embedding_1, embedding_2), dim=1)
            embedding = self.one_similarity_dropout(embedding)
            output = self.one_similarity_linear(embedding)
            output = 5 * torch.sigmoid(output)
            return output
        elif self.head_style == 2:
            if self.embedding_style == "cls":
                embedding_1 = self.forward_cls(input_ids_1, attention_mask_1)
                embedding_2 = self.forward_cls(input_ids_2, attention_mask_2)
            elif self.embedding_style == "mean":
                embedding_1 = self.forward_mean_pooled(input_ids_1, attention_mask_1)
                embedding_2 = self.forward_mean_pooled(input_ids_2, attention_mask_2)
            embedding_1 = self.two_similarity_dropout(embedding_1)
            embedding_1 = self.two_similarity_linear(embedding_1)
            embedding_2 = self.two_similarity_dropout(embedding_2)
            embedding_2 = self.two_similarity_linear(embedding_2)
            output = F.cosine_similarity(embedding_1, embedding_2, dim=1).unsqueeze(1)
            output = 2.5 * (output + 1)
            return output


def save_model(model, optimizer, args, config, filepath):
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
    print(f"Saving the model to {filepath}.", flush=True)


def train_sts(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # Load data
    _, _, _, sts_train_data_raw, _ = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train" 
    )
    _, _, _, sts_dev_data_raw, _ = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )

    sts_train_data = SentencePairDataset(sts_train_data_raw, args, isRegression=True, augment_prob=args.augment_prob)
    sts_dev_data = SentencePairDataset(sts_dev_data_raw, args, isRegression=True, augment_prob=0.0)

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

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": args.lora_target_modules,
        "lora_dropout": args.lora_dropout,
        "lora_bias": args.lora_bias,
        "head_style": args.head_style,
        "embedding_style": args.embedding_style
    }
    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator, flush=True)
    print("    BERT Model Configuration (STS)", flush=True)
    print(separator, flush=True)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}), flush=True)
    print(separator, flush=True)

    model = STSBERT(config)
    model = model.to(device)

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": args.lr_backbone},
        {"params": model.one_similarity_linear.parameters(), "lr": args.lr_head},
        {"params": model.two_similarity_linear.parameters(), "lr": args.lr_head}
    ])
    
    best_dev_acc = float("-inf")

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        print('start training epoch', epoch, flush=True)
        model.train()
        train_loss = 0
        num_batches = 0

        # Train on STS
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
            
            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            
            if args.loss_function == "mse":
                loss = F.mse_loss(logits.view(-1), b_labels.view(-1))
            elif args.loss_function == "neg_pearson":
                pred = logits.view(-1)
                gold = b_labels.view(-1)
                # Center the predictions and gold
                pred_c = pred - pred.mean()
                gold_c = gold - gold.mean()
                # Pearson correlation
                pearson = (pred_c * gold_c).sum() / (
                    pred_c.norm() * gold_c.norm() + 1e-8
                )
                # Negative Pearson (we want to maximize corr → minimize 1 - corr)
                loss = 1 - pearson.clamp(-1.0, 1.0)
                    
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        # Evaluate
        _, _, _, _, _, _, sts_train_corr, _, _, _, _, _ = model_eval_multitask(
            None, None, sts_train_dataloader, None, model=model, device=device, task="sts"
        )
        _, _, _, _, _, _, sts_dev_corr, _, _, _, _, _ = model_eval_multitask(
            None, None, sts_dev_dataloader, None, model=model, device=device, task="sts"
        )

        print(
            f"Epoch {epoch+1:02} (STS): train loss :: {train_loss:.3f}, train corr :: {sts_train_corr:.3f}, dev corr :: {sts_dev_corr:.3f}", flush=True
        )

        if sts_dev_corr > best_dev_acc:
            best_dev_acc = sts_dev_corr
            save_model(model, optimizer, args, config, args.filepath)


def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = STSBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}", flush=True)

        return test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    ################################################### MODIFICATIONS
    parser.add_argument("--filepath", type=str, default=None)  # save path
    # lora
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=float, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["query", "key", "value"])
    parser.add_argument("--lora_bias", type=str, default="none")
    # data aug
    parser.add_argument("--augment_prob", type=float, default=0.0)
    # head style
    parser.add_argument("--head_style", type=int, choices=[1, 2], default=1)
    # embedding style
    parser.add_argument("--embedding_style", type=str, choices=["cls", "mean"], default="cls")
    # loss function
    parser.add_argument("--loss_function", type=str, choices=["mse", "neg_pearson"], default="mse")
    ###################################################
    
    # Model configuration
    parser.add_argument("--task", type=str, default="sts")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: freeze BERT; finetune: update BERT",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")

    args, _ = parser.parse_known_args()

    ################################################### MODIFICATIONS
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    ###################################################

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

    parser.add_argument("--etpc_train", type=str, default="data/etpc-paraphrase-train.csv")
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")
    parser.add_argument(
        "--etpc_test", type=str, default="data/etpc-paraphrase-detection-test-student.csv"
    )

    # Output paths
    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default="predictions/bert/sts-similarity-dev-output.csv",
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default="predictions/bert/sts-similarity-test-output.csv",
    )

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.filepath is None:
        args.filepath = f"models/{args.option}-{args.epochs}-{args.lr_backbone}-{args.lr_head}-{args.head_style}-{args.augment_prob}-{args.embedding_style}-{args.loss_function}--sts.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_sts(args)
    test_model(args)
