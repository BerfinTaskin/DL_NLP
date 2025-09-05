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


BERT_HIDDEN_SIZE = 768


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
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
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

        self.similarity_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.similarity_linear = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)

    def forward(self, input_ids, attention_mask):
        """Return the [CLS] embedding."""
        bert_output = self.bert(input_ids, attention_mask)
        last_hidden = bert_output["last_hidden_state"]
        cls_embedding = last_hidden[:, 0, :]
        return cls_embedding

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of sentence pairs, output a similarity score in [0, 5].
        Dataset: STS
        """
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)
        cls_embedding = torch.cat((cls_embedding_1, cls_embedding_2), dim=1)
        cls_embedding = self.similarity_dropout(cls_embedding)
        output = self.similarity_linear(cls_embedding)
        output = 5 * torch.sigmoid(output)
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
    print(f"Saving the model to {filepath}.")


def train_sts(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # Load data
    _, _, _, sts_train_data, _ = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train" 
    )
    _, _, _, sts_dev_data, _ = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )

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

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "use_lora": args.use_lora,
    }
    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration (STS)")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = STSBERT(config)
    model = model.to(device)

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": args.lr_backbone},
        {"params": model.similarity_linear.parameters(), "lr": args.lr_head}
    ])
    
    best_dev_acc = float("-inf")

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
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
            loss = F.mse_loss(logits.view(-1), b_labels.view(-1))
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
            f"Epoch {epoch+1:02} (STS): train loss :: {train_loss:.3f}, train corr :: {sts_train_corr:.3f}, dev corr :: {sts_dev_corr:.3f}"
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
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="sts")

    ################# MODIFICATIONS
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
    # TODO
    
    # Model configuration
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
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.filepath is None:
        args.filepath = f"models/{args.option}-{args.epochs}-{args.lr_backbone}-{args.lr_head}-sts.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_sts(args)
    test_model(args)
