import torch
from torch import nn
from bert import BertModel
from tokenizer import BertTokenizer

class SentenceEncoder(nn.Module):
    """
    Thin wrapper over our minBERT implementation. Provides mean/cls/pooler pooling.
    Default = mean pooling (good for qqp task)
    """
    def __init__(self, local_files_only=False, finetune=True, pooling="mean"):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased",
                                              local_files_only=local_files_only)
        for p in self.bert.parameters():
            p.requires_grad = bool(finetune)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                       local_files_only=local_files_only)
        assert pooling in {"mean", "cls", "pooler"}
        self.pooling = pooling

    def _pool(self, last_hidden_state, attention_mask, pool_type):
        if pool_type == "pooler":
            # fall back to actual pooler if needed
            # but we'll compute it from forward() output
            raise RuntimeError("pooler needs forward() dict; use forward(..., return_dict=True)")
        if pool_type == "cls":
            return last_hidden_state[:, 0, :]  # [B, 768]
        # mean pooling (mask-aware)
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                  # [B, 768]
        denom = mask.sum(dim=1).clamp_min(1e-9)
        return summed / denom

    def forward(self, input_ids, attention_mask):
        # ask BERT to return both last_hidden_state and pooler
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last = out["last_hidden_state"]       # [B, L, 768]
        if self.pooling == "pooler":
            return out["pooler_output"]       # [B, 768]
        if self.pooling == "cls":
            return self._pool(last, attention_mask, "cls")
        return self._pool(last, attention_mask, "mean")  # default

    @torch.no_grad()
    def encode_texts(self, texts, device, max_length=128):
        self.eval()
        enc = self.tokenizer(texts, return_tensors="pt",
                             padding=True, truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        return self.forward(ids, mask)
